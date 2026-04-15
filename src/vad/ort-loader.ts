import type * as OrtTypes from "onnxruntime-web";

let ortModule: typeof import("onnxruntime-web") | null = null;

export async function getOrt(): Promise<typeof import("onnxruntime-web")> {
  if (!ortModule) {
    ortModule = await import("onnxruntime-web");
    ortModule.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
  }
  return ortModule;
}

export type { OrtTypes };
