export const PARAKEET_ASR_DOWNLOAD_BYTES = 654040125 + 9000000;
export const PARAKEET_MODEL_CARD_URL =
  "https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-onnx";
export const PARAKEET_ENCODER_URL =
  "https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-onnx/resolve/main/onnx_int8/encoder.int8.onnx";
export const PARAKEET_DECODER_URL =
  "https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-onnx/resolve/main/onnx_int8/decoder_joint.int8.onnx";
export const PARAKEET_VOCAB_URL = "/asr/parakeet-unified-en-0.6b-int8-vocab.json";

const PARAKEET_CACHE_NAME = "parakeet-asr-v1";

export interface ParakeetDownloadProgress {
  phase: "encoder" | "decoder";
  loadedBytes: number;
  totalBytes: number | null;
  overallLoadedBytes: number;
  overallTotalBytes: number;
}

export interface ParakeetAssets {
  encoderBytes: Uint8Array;
  decoderBytes: Uint8Array;
  fromCache: boolean;
}

async function openParakeetCache(): Promise<Cache> {
  return caches.open(PARAKEET_CACHE_NAME);
}

async function readCachedBytes(url: string): Promise<Uint8Array | null> {
  const cache = await openParakeetCache();
  const response = await cache.match(url);
  if (!response) {
    return null;
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function cacheBytes(url: string, bytes: Uint8Array): Promise<void> {
  const cache = await openParakeetCache();
  const response = new Response(bytes, {
    headers: {
      "content-length": String(bytes.byteLength),
      "content-type": "application/octet-stream",
    },
  });
  await cache.put(url, response);
}

async function downloadModel(
  url: string,
  phase: "encoder" | "decoder",
  overallOffset: number,
  signal?: AbortSignal,
  onProgress?: (progress: ParakeetDownloadProgress) => void
): Promise<Uint8Array> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Failed to download ${phase} model: ${response.status} ${response.statusText}`);
  }

  const totalHeader = response.headers.get("content-length");
  const totalBytes = totalHeader ? Number(totalHeader) : null;
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress?.({
      phase,
      loadedBytes: bytes.byteLength,
      totalBytes: bytes.byteLength,
      overallLoadedBytes: overallOffset + bytes.byteLength,
      overallTotalBytes: PARAKEET_ASR_DOWNLOAD_BYTES,
    });
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (!value) {
      continue;
    }

    chunks.push(value);
    loadedBytes += value.byteLength;
    onProgress?.({
      phase,
      loadedBytes,
      totalBytes,
      overallLoadedBytes: overallOffset + loadedBytes,
      overallTotalBytes: PARAKEET_ASR_DOWNLOAD_BYTES,
    });
  }

  const merged = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  onProgress?.({
    phase,
    loadedBytes,
    totalBytes: totalBytes ?? loadedBytes,
    overallLoadedBytes: overallOffset + loadedBytes,
    overallTotalBytes: PARAKEET_ASR_DOWNLOAD_BYTES,
  });

  return merged;
}

export async function hasParakeetCache(): Promise<boolean> {
  const cache = await openParakeetCache();
  const [encoder, decoder] = await Promise.all([
    cache.match(PARAKEET_ENCODER_URL),
    cache.match(PARAKEET_DECODER_URL),
  ]);
  return Boolean(encoder && decoder);
}

export async function clearParakeetCache(): Promise<void> {
  const cache = await openParakeetCache();
  await Promise.all([
    cache.delete(PARAKEET_ENCODER_URL),
    cache.delete(PARAKEET_DECODER_URL),
  ]);
}

export async function loadParakeetAssets(
  signal?: AbortSignal,
  onProgress?: (progress: ParakeetDownloadProgress) => void
): Promise<ParakeetAssets> {
  let fromCache = true;

  let encoderBytes = await readCachedBytes(PARAKEET_ENCODER_URL);
  if (!encoderBytes) {
    fromCache = false;
    encoderBytes = await downloadModel(PARAKEET_ENCODER_URL, "encoder", 0, signal, onProgress);
    await cacheBytes(PARAKEET_ENCODER_URL, encoderBytes);
  }

  let decoderBytes = await readCachedBytes(PARAKEET_DECODER_URL);
  if (!decoderBytes) {
    fromCache = false;
    decoderBytes = await downloadModel(
      PARAKEET_DECODER_URL,
      "decoder",
      encoderBytes.byteLength,
      signal,
      onProgress
    );
    await cacheBytes(PARAKEET_DECODER_URL, decoderBytes);
  }

  return { encoderBytes, decoderBytes, fromCache };
}
