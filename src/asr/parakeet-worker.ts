/// <reference lib="webworker" />

import { ParakeetAsrEngine } from "./parakeet-asr";
import {
  clearParakeetCache,
  hasParakeetCache,
  loadParakeetAssets,
  PARAKEET_VOCAB_URL,
} from "./parakeet-model-store";
import { loadParakeetVocab } from "./parakeet-tokenizer";
import type {
  ParakeetWorkerLoadState,
  ParakeetWorkerRequest,
  ParakeetWorkerResponse,
} from "./parakeet-worker-protocol";

const ctx: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;

let engine: ParakeetAsrEngine | null = null;
let cached = false;
let error: string | null = null;
let loadState: ParakeetWorkerLoadState = "checking";
let loadPromise: Promise<void> | null = null;
let loadAbortController: AbortController | null = null;
let transcriptionQueue = Promise.resolve();

function postMessage(message: ParakeetWorkerResponse): void {
  ctx.postMessage(message);
}

function reportState(): void {
  postMessage({
    type: "state",
    loadState,
    cached,
    error,
  });
}

async function refreshCacheState(): Promise<void> {
  cached = await hasParakeetCache();
}

async function bootstrap(): Promise<void> {
  loadState = "checking";
  error = null;
  reportState();
  await refreshCacheState();
  loadState = engine?.isReady ? "ready" : "idle";
  reportState();
}

async function ensureLoaded(): Promise<void> {
  if (engine?.isReady) {
    loadState = "ready";
    error = null;
    reportState();
    return;
  }

  if (loadPromise) {
    return loadPromise;
  }

  error = null;
  await refreshCacheState();
  loadState = cached ? "loading_cached" : "downloading";
  reportState();

  const abortController = new AbortController();
  loadAbortController = abortController;

  loadPromise = (async () => {
    try {
      const [assets, vocab] = await Promise.all([
        loadParakeetAssets(abortController.signal, (progress) => {
          loadState = "downloading";
          postMessage({ type: "downloadProgress", progress });
        }),
        loadParakeetVocab(PARAKEET_VOCAB_URL),
      ]);

      const nextEngine = new ParakeetAsrEngine();
      await nextEngine.loadFromBuffers(assets.encoderBytes, assets.decoderBytes, vocab);
      engine = nextEngine;
      await refreshCacheState();
      loadState = "ready";
      error = null;
      reportState();
    } catch (err) {
      if (abortController.signal.aborted) {
        loadState = "idle";
        error = null;
      } else {
        engine = null;
        loadState = "error";
        error = err instanceof Error ? err.message : "Failed to load ASR model";
      }
      await refreshCacheState();
      reportState();
    } finally {
      loadAbortController = null;
      loadPromise = null;
    }
  })();

  return loadPromise;
}

function cancelLoad(): void {
  loadAbortController?.abort();
}

async function removeCache(): Promise<void> {
  cancelLoad();
  await clearParakeetCache();
  engine?.unload();
  engine = null;
  loadState = "idle";
  error = null;
  await refreshCacheState();
  reportState();
}

function queueTranscription(jobId: number, pcm: Int16Array): void {
  transcriptionQueue = transcriptionQueue
    .then(async () => {
      if (!engine?.isReady) {
        throw new Error("ASR model is not ready");
      }

      postMessage({ type: "transcribeStarted", jobId });
      const transcript = await engine.transcribeSegment(pcm);
      postMessage({ type: "transcribeDone", jobId, transcript });
    })
    .catch((err) => {
      postMessage({
        type: "transcribeError",
        jobId,
        error: err instanceof Error ? err.message : "ASR failed",
      });
    });
}

ctx.onmessage = (event: MessageEvent<ParakeetWorkerRequest>) => {
  const message = event.data;

  switch (message.type) {
    case "bootstrap":
      void bootstrap();
      break;
    case "enable":
      void ensureLoaded();
      break;
    case "cancelLoad":
      cancelLoad();
      break;
    case "removeCache":
      void removeCache();
      break;
    case "transcribe":
      queueTranscription(message.jobId, message.pcm);
      break;
  }
};
