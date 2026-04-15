import type * as OrtTypes from "onnxruntime-web";
import { getOrt } from "../vad/ort-loader";
import { ParakeetFeatureExtractor } from "./parakeet-feature-extractor";
import { decodeSentencePieces, type ParakeetVocab } from "./parakeet-tokenizer";

const ENCODER_DIM = 1024;
const DECODER_STATE_SHAPE: [number, number, number] = [2, 1, 640];
const MAX_SYMBOLS_PER_FRAME = 10;

function createInt64Tensor(value: number, ort: typeof import("onnxruntime-web")): OrtTypes.Tensor {
  return new ort.Tensor("int64", new BigInt64Array([BigInt(value)]), [1]);
}

function argmax(values: Float32Array): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

function sliceEncoderFrame(data: Float32Array, totalFrames: number, frameIndex: number): Float32Array {
  const slice = new Float32Array(ENCODER_DIM);
  for (let dim = 0; dim < ENCODER_DIM; dim++) {
    slice[dim] = data[dim * totalFrames + frameIndex];
  }
  return slice;
}

export class ParakeetAsrEngine {
  private readonly extractor = new ParakeetFeatureExtractor();
  private encoderSession: OrtTypes.InferenceSession | null = null;
  private decoderSession: OrtTypes.InferenceSession | null = null;
  private vocab: ParakeetVocab | null = null;

  async loadFromBuffers(encoderBytes: Uint8Array, decoderBytes: Uint8Array, vocab: ParakeetVocab): Promise<void> {
    const ort = await getOrt();
    const [encoderSession, decoderSession] = await Promise.all([
      ort.InferenceSession.create(encoderBytes, { executionProviders: ["wasm"] }),
      ort.InferenceSession.create(decoderBytes, { executionProviders: ["wasm"] }),
    ]);

    this.vocab = vocab;
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
  }

  unload(): void {
    this.encoderSession = null;
    this.decoderSession = null;
    this.vocab = null;
  }

  get isReady(): boolean {
    return this.encoderSession !== null && this.decoderSession !== null && this.vocab !== null;
  }

  async transcribeSegment(pcmInt16: Int16Array): Promise<string> {
    if (!this.encoderSession || !this.decoderSession || !this.vocab) {
      throw new Error("ASR model not loaded");
    }

    const features = this.extractor.extract(pcmInt16);
    if (!features || features.frameCount === 0) {
      return "";
    }

    const ort = await getOrt();
    const audioSignal = new ort.Tensor("float32", features.values, [1, 128, features.frameCount]);
    const length = createInt64Tensor(features.frameCount, ort);
    const encoderResults = await this.encoderSession.run({
      audio_signal: audioSignal,
      length,
    });

    const encoderOutputs = encoderResults["outputs"];
    const encodedLengths = encoderResults["encoded_lengths"].data as BigInt64Array;
    const totalFrames = Number(encodedLengths[0]);
    const encoderData = encoderOutputs.data as Float32Array;
    const blankId = this.vocab.blankId;

    let state1 = new ort.Tensor("float32", new Float32Array(2 * 1 * 640), DECODER_STATE_SHAPE);
    let state2 = new ort.Tensor("float32", new Float32Array(2 * 1 * 640), DECODER_STATE_SHAPE);
    let lastToken = blankId;
    const tokens: number[] = [];

    for (let frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
      const encoderFrame = new ort.Tensor(
        "float32",
        sliceEncoderFrame(encoderData, totalFrames, frameIndex),
        [1, ENCODER_DIM, 1]
      );

      for (let symbolIndex = 0; symbolIndex < MAX_SYMBOLS_PER_FRAME; symbolIndex++) {
        const decoderResults = await this.decoderSession.run({
          encoder_outputs: encoderFrame,
          targets: new ort.Tensor("int32", new Int32Array([lastToken]), [1, 1]),
          target_length: new ort.Tensor("int32", new Int32Array([1]), [1]),
          input_states_1: state1,
          input_states_2: state2,
        });

        const logits = decoderResults["outputs"].data as Float32Array;
        const tokenId = argmax(logits);
        if (tokenId === blankId) {
          break;
        }

        tokens.push(tokenId);
        lastToken = tokenId;
        state1 = decoderResults["output_states_1"];
        state2 = decoderResults["output_states_2"];
      }
    }

    return decodeSentencePieces(tokens, this.vocab.pieces);
  }
}
