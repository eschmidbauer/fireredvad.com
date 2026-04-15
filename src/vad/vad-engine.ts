import type * as OrtTypes from "onnxruntime-web";
import { FbankExtractor } from "../fbank";
import { getOrt } from "./ort-loader";
import { VadStateMachine } from "./state-machine";
import type { CmvnStats, VadEvent, AedResult } from "./types";
import { AED_LABELS } from "./types";

export type { VadEvent, AedResult, CmvnStats };

export class VadEngine {
  private sess: OrtTypes.InferenceSession | null = null;
  private aedSess: OrtTypes.InferenceSession | null = null;
  private fbank = new FbankExtractor();
  private cmvn: CmvnStats;
  private caches: OrtTypes.Tensor | null = null;
  private stateMachine = new VadStateMachine();

  constructor(cmvn: CmvnStats) {
    this.cmvn = cmvn;
  }

  private async ensureCaches(): Promise<OrtTypes.Tensor> {
    if (!this.caches) {
      const ort = await getOrt();
      this.caches = new ort.Tensor("float32", new Float32Array(8 * 1 * 128 * 19), [8, 1, 128, 19]);
    }
    return this.caches;
  }

  async loadModels(vadUrl: string, aedUrl?: string): Promise<void> {
    const ort = await getOrt();
    await this.ensureCaches();
    this.sess = await ort.InferenceSession.create(vadUrl, { executionProviders: ["wasm"] });
    if (aedUrl) {
      try {
        this.aedSess = await ort.InferenceSession.create(aedUrl, { executionProviders: ["wasm"] });
      } catch (e) {
        console.warn("Failed to load AED model:", e);
      }
    }
  }

  get hasAed(): boolean {
    return this.aedSess !== null;
  }

  async reset(): Promise<void> {
    this.fbank.reset();
    const ort = await getOrt();
    this.caches = new ort.Tensor("float32", new Float32Array(8 * 1 * 128 * 19), [8, 1, 128, 19]);
    this.stateMachine.reset();
  }

  private applyMvn(frames: (number[] | Float32Array)[]): Float32Array {
    const T = frames.length;
    const featData = new Float32Array(T * 80);
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < 80; d++) {
        featData[t * 80 + d] = (frames[t][d] - this.cmvn.means[d]) * this.cmvn.inv_std[d];
      }
    }
    return featData;
  }

  async processChunk(pcmInt16: Int16Array): Promise<VadEvent[]> {
    if (!this.sess) throw new Error("Model not loaded");
    const ort = await getOrt();
    const caches = await this.ensureCaches();

    const frames = this.fbank.extract(pcmInt16);
    if (!frames || frames.length === 0) return [];

    const T = frames.length;
    const featData = this.applyMvn(frames);
    const feat = new ort.Tensor("float32", featData, [1, T, 80]);
    const results = await this.sess.run({ feat, caches_in: caches });

    this.caches = results["caches_out"];
    const probsData = results["probs"].data as Float32Array;

    const events: VadEvent[] = [];
    for (let i = 0; i < probsData.length; i++) {
      const event = this.stateMachine.processFrame(probsData[i]);
      if (event) events.push(event);
    }
    return events;
  }

  flush(): VadEvent | null {
    return this.stateMachine.flush();
  }

  async classifySegment(pcmInt16: Int16Array): Promise<AedResult | null> {
    if (!this.aedSess) return null;
    const ort = await getOrt();

    const extractor = new FbankExtractor();
    const frames = extractor.extract(pcmInt16);
    if (!frames || frames.length === 0) return null;

    const T = frames.length;
    const featData = this.applyMvn(frames);
    const feat = new ort.Tensor("float32", featData, [1, T, 80]);
    const results = await this.aedSess.run({ feat });

    const probsData = results["probs"].data as Float32Array;
    const avgProbs = [0, 0, 0];
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < 3; c++) {
        avgProbs[c] += probsData[t * 3 + c];
      }
    }
    for (let c = 0; c < 3; c++) avgProbs[c] /= T;

    const maxIdx = avgProbs.indexOf(Math.max(...avgProbs));
    const probs: Record<string, number> = {};
    for (let c = 0; c < 3; c++) {
      probs[AED_LABELS[c]] = Math.round(avgProbs[c] * 10000) / 10000;
    }
    return { label: AED_LABELS[maxIdx], probs };
  }
}
