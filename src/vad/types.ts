export interface CmvnStats {
  means: number[];
  inv_std: number[];
}

export interface VadEvent {
  type: "speech_start" | "speech_end";
  startFrame: number;
  endFrame: number;
}

export interface AedResult {
  label: string;
  probs: Record<string, number>;
}

export const AED_LABELS = ["speech", "music", "noise"];
