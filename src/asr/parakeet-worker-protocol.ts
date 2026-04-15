import type { ParakeetDownloadProgress } from "./parakeet-model-store";

export type ParakeetWorkerLoadState =
  | "checking"
  | "idle"
  | "downloading"
  | "loading_cached"
  | "ready"
  | "error";

export type ParakeetWorkerRequest =
  | { type: "bootstrap" }
  | { type: "enable" }
  | { type: "cancelLoad" }
  | { type: "removeCache" }
  | { type: "transcribe"; jobId: number; pcm: Int16Array };

export type ParakeetWorkerResponse =
  | {
      type: "state";
      loadState: ParakeetWorkerLoadState;
      cached: boolean;
      error: string | null;
    }
  | {
      type: "downloadProgress";
      progress: ParakeetDownloadProgress;
    }
  | {
      type: "transcribeStarted";
      jobId: number;
    }
  | {
      type: "transcribeDone";
      jobId: number;
      transcript: string;
    }
  | {
      type: "transcribeError";
      jobId: number;
      error: string;
    };
