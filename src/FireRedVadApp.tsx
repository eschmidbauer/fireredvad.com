import { useCallback, useEffect, useRef, useState } from "react";
import { FBANK_FRAME_SHIFT } from "./fbank";
import {
  PARAKEET_ASR_DOWNLOAD_BYTES,
  PARAKEET_MODEL_CARD_URL,
  type ParakeetDownloadProgress,
} from "./asr/parakeet-model-store";
import type {
  ParakeetWorkerLoadState,
  ParakeetWorkerRequest,
  ParakeetWorkerResponse,
} from "./asr/parakeet-worker-protocol";
import { VadEngine, VadEvent, AedResult } from "./vad/vad-engine";

const FRAME_PER_SECOND = 100;
const SAMPLE_RATE = 16000;
const NUM_BARS = 200;
const BAR_GAP = 1;

const CANVAS_COLORS = {
  bg: "#111318",
  waveform: "#6366f1",
  waveformSpeech: "#22c55e",
  speech: "#22c55e",
  speechDim: "rgba(34,197,94,0.15)",
  music: "#3b82f6",
  musicDim: "rgba(59,130,246,0.15)",
  noise: "#f97316",
  noiseDim: "rgba(249,115,22,0.15)",
};

type TranscriptStatus = "queued" | "transcribing" | "ready" | "error";

interface DisplayEvent {
  id: number;
  type: "speech_start" | "speech_end";
  startSec: number;
  endSec: number;
  durationSec?: number;
  aed?: AedResult;
  audioUrl?: string;
  transcript?: string;
  transcriptError?: string;
  transcriptStatus?: TranscriptStatus;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

function previewTranscript(text: string, maxLength = 88): string {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1).trimEnd()}…`;
}

function createWavUrl(pcm: Int16Array, sampleRate: number): string {
  const numChannels = 1;
  const bytesPerSample = 2;
  const dataSize = pcm.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) {
      view.setUint8(o + i, s.charCodeAt(i));
    }
  };

  w(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  w(8, "WAVE");
  w(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  w(36, "data");
  view.setUint32(40, dataSize, true);
  new Int16Array(buffer, 44).set(pcm);
  return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
}

function AedBars({ probs }: { probs: Record<string, number> }) {
  const items = [
    { key: "speech", colorClass: "bg-speech", textClass: "text-speech" },
    { key: "music", colorClass: "bg-music", textClass: "text-music" },
    { key: "noise", colorClass: "bg-noise", textClass: "text-noise" },
  ];

  return (
    <div className="flex flex-col gap-1.5 mt-3">
      {items.map(({ key, colorClass, textClass }) => {
        const pct = (probs[key] ?? 0) * 100;
        return (
          <div key={key} className="flex items-center gap-3">
            <span className="w-14 text-xs text-muted-foreground text-right font-medium capitalize">
              {key}
            </span>
            <div className="flex-1 h-2 rounded-full bg-secondary overflow-hidden">
              <div
                className={`h-full rounded-full ${colorClass} transition-all duration-300`}
                style={{ width: `${pct}%`, opacity: 0.85 }}
              />
            </div>
            <span className={`w-12 text-xs font-mono font-semibold text-right ${textClass}`}>
              {pct.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function FireRedVadApp() {
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "recording">("idle");
  const [events, setEvents] = useState<DisplayEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSpeech, setIsSpeech] = useState(false);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [copiedEventId, setCopiedEventId] = useState<number | null>(null);

  const [asrEnabled, setAsrEnabled] = useState(false);
  const [asrLoadState, setAsrLoadState] = useState<ParakeetWorkerLoadState>("checking");
  const [asrCached, setAsrCached] = useState(false);
  const [asrError, setAsrError] = useState<string | null>(null);
  const [asrDownload, setAsrDownload] = useState<ParakeetDownloadProgress | null>(null);

  const engineRef = useRef<VadEngine | null>(null);
  const asrWorkerRef = useRef<Worker | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioBufferRef = useRef<Int16Array[]>([]);
  const segmentAudioRef = useRef(new Map<number, Int16Array>());
  const totalSamplesRef = useRef(0);
  const eventIdRef = useRef(0);
  const copyTimeoutRef = useRef<number | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const barLevelsRef = useRef(new Float32Array(NUM_BARS));
  const barSpeechRef = useRef(new Uint8Array(NUM_BARS));
  const barWriteRef = useRef(0);
  const isSpeechRef = useRef(false);
  const animRef = useRef(0);
  const rmsAccRef = useRef({ sum: 0, count: 0 });
  const samplesPerBar = Math.floor(SAMPLE_RATE / 25);

  const patchEvent = useCallback((id: number, patch: Partial<DisplayEvent>) => {
    setEvents((prev) => prev.map((evt) => (evt.id === id ? { ...evt, ...patch } : evt)));
  }, []);

  const clearSessionArtifacts = useCallback(() => {
    setEvents((prev) => {
      prev.forEach((evt) => {
        if (evt.audioUrl) {
          URL.revokeObjectURL(evt.audioUrl);
        }
      });
      return [];
    });
    segmentAudioRef.current.clear();
  }, []);

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current !== null) {
        window.clearTimeout(copyTimeoutRef.current);
      }
    };
  }, []);

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, width, height);

    const barWidth = Math.max(1, (width - (NUM_BARS - 1) * BAR_GAP) / NUM_BARS);
    const readStart = barWriteRef.current;

    for (let i = 0; i < NUM_BARS; i++) {
      const idx = (readStart + i) % NUM_BARS;
      const amp = Math.min(1, barLevelsRef.current[idx] * 5);
      const hasSpeech = barSpeechRef.current[idx] === 1;
      const x = i * (barWidth + BAR_GAP);
      const barHeight = Math.max(2, amp * (height - 8));
      const y = (height - barHeight) / 2;

      if (hasSpeech) {
        ctx.fillStyle = CANVAS_COLORS.speechDim;
        ctx.beginPath();
        ctx.roundRect(x - 1, y - 2, barWidth + 2, barHeight + 4, 3);
        ctx.fill();
      }

      const color = hasSpeech ? CANVAS_COLORS.waveformSpeech : CANVAS_COLORS.waveform;
      const alpha = 0.4 + amp * 0.6;
      ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, "0");
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, Math.min(barWidth / 2, 3));
      ctx.fill();
    }

    animRef.current = requestAnimationFrame(drawWaveform);
  }, []);

  useEffect(() => {
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  const pushWaveformSamples = useCallback(
    (pcm: Int16Array) => {
      const acc = rmsAccRef.current;
      for (let i = 0; i < pcm.length; i++) {
        const normalized = pcm[i] / 32768;
        acc.sum += normalized * normalized;
        acc.count++;
        if (acc.count >= samplesPerBar) {
          const rms = Math.sqrt(acc.sum / acc.count);
          const writePointer = barWriteRef.current;
          barLevelsRef.current[writePointer] = rms;
          barSpeechRef.current[writePointer] = isSpeechRef.current ? 1 : 0;
          barWriteRef.current = (writePointer + 1) % NUM_BARS;
          acc.sum = 0;
          acc.count = 0;
        }
      }
    },
    [samplesPerBar]
  );

  useEffect(() => {
    (async () => {
      setStatus("loading");
      try {
        const cmvn = await fetch("/cmvn.json").then((r) => r.json());
        const engine = new VadEngine(cmvn);
        await engine.loadModels(
          "/onnx_models/fireredvad_stream_vad_with_cache.onnx",
          "/onnx_models/fireredvad_aed.onnx"
        );
        engineRef.current = engine;
        setStatus("ready");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load models");
        setStatus("idle");
      }
    })();
  }, []);

  useEffect(() => {
    const worker = new Worker(new URL("./asr/parakeet-worker.ts", import.meta.url), {
      type: "module",
    });
    asrWorkerRef.current = worker;

    worker.onmessage = (event: MessageEvent<ParakeetWorkerResponse>) => {
      const message = event.data;

      switch (message.type) {
        case "state":
          setAsrLoadState(message.loadState);
          setAsrCached(message.cached);
          setAsrError(message.error);
          if (message.loadState !== "downloading") {
            setAsrDownload(null);
          }
          if (message.loadState === "idle" || message.loadState === "error") {
            setAsrEnabled(false);
          }
          break;
        case "downloadProgress":
          setAsrDownload(message.progress);
          break;
        case "transcribeStarted":
          patchEvent(message.jobId, {
            transcriptStatus: "transcribing",
            transcriptError: undefined,
          });
          break;
        case "transcribeDone":
          patchEvent(message.jobId, {
            transcript: message.transcript || "No transcript generated.",
            transcriptStatus: "ready",
            transcriptError: undefined,
          });
          break;
        case "transcribeError":
          patchEvent(message.jobId, {
            transcriptStatus: "error",
            transcriptError: message.error,
          });
          break;
      }
    };

    worker.postMessage({ type: "bootstrap" } satisfies ParakeetWorkerRequest);

    return () => {
      worker.terminate();
      asrWorkerRef.current = null;
    };
  }, [patchEvent]);

  const postAsrMessage = useCallback((message: ParakeetWorkerRequest) => {
    asrWorkerRef.current?.postMessage(message);
  }, []);

  const requestAsrEnable = useCallback(() => {
    setAsrEnabled(true);
    setAsrError(null);
    if (asrLoadState !== "ready") {
      postAsrMessage({ type: "enable" });
    }
  }, [asrLoadState, postAsrMessage]);

  const disableAsr = useCallback(() => {
    setAsrEnabled(false);
  }, []);

  const cancelAsrDownload = useCallback(() => {
    setAsrEnabled(false);
    setAsrDownload(null);
    postAsrMessage({ type: "cancelLoad" });
  }, [postAsrMessage]);

  const removeAsrCache = useCallback(() => {
    setAsrEnabled(false);
    setAsrDownload(null);
    postAsrMessage({ type: "removeCache" });
  }, [postAsrMessage]);

  const requestTranscript = useCallback(
    (eventId: number) => {
      const pcm = segmentAudioRef.current.get(eventId);
      if (!pcm || asrLoadState !== "ready") {
        return;
      }

      patchEvent(eventId, {
        transcriptStatus: "queued",
        transcriptError: undefined,
      });
      postAsrMessage({
        type: "transcribe",
        jobId: eventId,
        pcm: pcm.slice(),
      });
    },
    [asrLoadState, patchEvent, postAsrMessage]
  );

  const copyTranscript = useCallback(async (eventId: number, text: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedEventId(eventId);
    if (copyTimeoutRef.current !== null) {
      window.clearTimeout(copyTimeoutRef.current);
    }
    copyTimeoutRef.current = window.setTimeout(() => {
      setCopiedEventId(null);
    }, 1200);
  }, []);

  const getSegmentAudio = useCallback((startFrame: number, endFrame: number): Int16Array | null => {
    const startSample = Math.max(0, (startFrame - 1) * FBANK_FRAME_SHIFT);
    const endSample = Math.min(totalSamplesRef.current, endFrame * FBANK_FRAME_SHIFT);
    if (endSample <= startSample) {
      return null;
    }

    const all = new Int16Array(totalSamplesRef.current);
    let offset = 0;
    for (const chunk of audioBufferRef.current) {
      all.set(chunk, offset);
      offset += chunk.length;
    }
    return all.slice(startSample, endSample);
  }, []);

  const handleVadEvents = useCallback(
    async (vadEvents: VadEvent[]) => {
      const engine = engineRef.current;

      for (const evt of vadEvents) {
        const startSec = (evt.startFrame - 1) / FRAME_PER_SECOND;
        const endSec = evt.endFrame / FRAME_PER_SECOND;

        if (evt.type === "speech_start") {
          setIsSpeech(true);
          isSpeechRef.current = true;
        } else {
          setIsSpeech(false);
          isSpeechRef.current = false;
        }

        const displayEvent: DisplayEvent = {
          id: eventIdRef.current++,
          type: evt.type,
          startSec,
          endSec,
        };

        if (evt.type === "speech_end") {
          displayEvent.durationSec = endSec - startSec;
          const segmentAudio = getSegmentAudio(evt.startFrame, evt.endFrame);

          if (segmentAudio) {
            displayEvent.audioUrl = createWavUrl(segmentAudio, SAMPLE_RATE);
            segmentAudioRef.current.set(displayEvent.id, segmentAudio);

            if (engine?.hasAed) {
              const aed = await engine.classifySegment(segmentAudio);
              if (aed) {
                displayEvent.aed = aed;
              }
            }

            if (asrEnabled && asrLoadState === "ready") {
              displayEvent.transcriptStatus = "queued";
            }

            setEvents((prev) => [displayEvent, ...prev]);

            if (displayEvent.transcriptStatus === "queued") {
              requestTranscript(displayEvent.id);
            }
            continue;
          }
        }

        setEvents((prev) => [displayEvent, ...prev]);
      }
    },
    [asrEnabled, asrLoadState, getSegmentAudio, requestTranscript]
  );

  const start = useCallback(async () => {
    const engine = engineRef.current;
    if (!engine) {
      return;
    }

    setError(null);
    await engine.reset();
    audioBufferRef.current = [];
    totalSamplesRef.current = 0;
    barLevelsRef.current.fill(0);
    barSpeechRef.current.fill(0);
    barWriteRef.current = 0;
    rmsAccRef.current = { sum: 0, count: 0 };
    clearSessionArtifacts();
    setExpandedId(null);
    setCopiedEventId(null);
    setIsSpeech(false);
    isSpeechRef.current = false;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = audioCtx;
      await audioCtx.audioWorklet.addModule("/pcm-processor.js");

      const source = audioCtx.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(audioCtx, "pcm-processor");
      workletRef.current = worklet;

      worklet.port.onmessage = async (e: MessageEvent<ArrayBuffer>) => {
        const pcm = new Int16Array(e.data);
        audioBufferRef.current.push(pcm);
        totalSamplesRef.current += pcm.length;
        pushWaveformSamples(pcm);

        try {
          const vadEvents = await engine.processChunk(pcm);
          if (vadEvents.length > 0) {
            await handleVadEvents(vadEvents);
          }
        } catch (err) {
          console.error("VAD error:", err);
        }
      };

      source.connect(worklet);
      setStatus("recording");
      animRef.current = requestAnimationFrame(drawWaveform);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Microphone error");
    }
  }, [clearSessionArtifacts, drawWaveform, handleVadEvents, pushWaveformSamples]);

  const stop = useCallback(() => {
    cancelAnimationFrame(animRef.current);
    workletRef.current?.disconnect();
    workletRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    const engine = engineRef.current;
    if (engine) {
      const evt = engine.flush();
      if (evt) {
        void handleVadEvents([evt]);
      }
    }

    setIsSpeech(false);
    isSpeechRef.current = false;
    setStatus("ready");
  }, [handleVadEvents]);

  const speechEndEvents = events.filter((event) => event.type === "speech_end");
  const transcriptReadyCount = speechEndEvents.filter((event) => event.transcriptStatus === "ready").length;
  const transcriptPendingCount = speechEndEvents.filter(
    (event) => event.transcriptStatus === "queued" || event.transcriptStatus === "transcribing"
  ).length;
  const totalSpeechDuration = speechEndEvents.reduce((sum, event) => sum + (event.durationSec ?? 0), 0);

  const overallDownloadPercent = asrDownload
    ? Math.min(100, (asrDownload.overallLoadedBytes / asrDownload.overallTotalBytes) * 100)
    : 0;
  const phaseDownloadPercent =
    asrDownload && asrDownload.totalBytes
      ? Math.min(100, (asrDownload.loadedBytes / asrDownload.totalBytes) * 100)
      : null;

  const asrPrimaryActionLabel =
    asrLoadState === "ready"
      ? asrEnabled
        ? "Disable ASR"
        : "Enable ASR"
      : asrCached
        ? "Load cached model"
        : "Download model";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/50">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-3">
            <div className="w-12 h-12 rounded-xl gradient-fire flex items-center justify-center shadow-lg glow-red">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-white">
                <path
                  d="M12 2C8 6 4 10 4 14a8 8 0 0016 0c0-4-4-8-8-12zm0 18a6 6 0 01-6-6c0-3 2.5-6.5 6-10 3.5 3.5 6 7 6 10a6 6 0 01-6 6z"
                  fill="currentColor"
                  opacity="0.9"
                />
                <path
                  d="M12 20a4 4 0 01-4-4c0-2 1.5-4 4-7 2.5 3 4 5 4 7a4 4 0 01-4 4z"
                  fill="currentColor"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-extrabold tracking-tight">
                Fire<span className="text-primary">Red</span>VAD
              </h1>
              <p className="text-sm text-muted-foreground mt-0.5">
                Real-time voice activity detection with optional on-device transcription
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        <div className="rounded-xl border border-border bg-card p-5">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              {status === "idle" && (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <div className="w-2 h-2 rounded-full bg-muted-foreground/50" />
                  <span className="text-sm">Initializing...</span>
                </div>
              )}
              {status === "loading" && (
                <div className="flex items-center gap-3">
                  <div className="w-5 h-5 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                  <span className="text-sm text-muted-foreground">Loading VAD models...</span>
                </div>
              )}
              {status === "ready" && (
                <button
                  onClick={start}
                  className="px-5 py-2.5 rounded-lg gradient-fire text-white font-semibold text-sm hover:opacity-90 transition-opacity glow-red"
                >
                  Start Recording
                </button>
              )}
              {status === "recording" && (
                <button
                  onClick={stop}
                  className="px-5 py-2.5 rounded-lg bg-destructive text-destructive-foreground font-semibold text-sm hover:opacity-90 transition-opacity"
                >
                  Stop Recording
                </button>
              )}
            </div>

            <div className="flex items-center gap-3 flex-wrap">
              {status === "ready" && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-speech" />
                  <span className="text-xs font-medium text-speech">VAD ready</span>
                </div>
              )}
              {asrLoadState === "ready" && (
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${asrEnabled ? "bg-music" : "bg-muted-foreground/50"}`} />
                  <span className={`text-xs font-medium ${asrEnabled ? "text-music" : "text-muted-foreground"}`}>
                    {asrEnabled ? "ASR enabled" : "ASR loaded"}
                  </span>
                </div>
              )}
              {status === "recording" && (
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2.5 h-2.5 rounded-full transition-colors ${
                      isSpeech ? "bg-speech glow-green" : "bg-primary"
                    }`}
                    style={{ animation: "pulse-dot 1.2s ease-in-out infinite" }}
                  />
                  <span className={`text-xs font-semibold ${isSpeech ? "text-speech" : "text-muted-foreground"}`}>
                    {isSpeech ? "Speech Detected" : "Listening..."}
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 rounded-lg border border-border/70 bg-secondary/20 px-4 py-3">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div className="max-w-2xl">
                <div className="flex items-center gap-2 flex-wrap">
                  <div className="text-sm font-semibold">Parakeet ASR</div>
                  {asrCached && (
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-secondary text-muted-foreground">
                      Cached
                    </span>
                  )}
                  {asrEnabled && asrLoadState === "ready" && (
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-music text-background">
                      Active
                    </span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Downloads about {formatBytes(PARAKEET_ASR_DOWNLOAD_BYTES)} once, then runs locally on this device for
                  completed speech segments.
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  {asrLoadState === "checking" && "Checking whether the model is already installed on this device."}
                  {asrLoadState === "idle" && !asrCached && "Model not downloaded yet. Best on desktop with plenty of memory."}
                  {asrLoadState === "idle" && asrCached && "Model is installed on this device and can be loaded without re-downloading."}
                  {asrLoadState === "loading_cached" && "Loading cached model into memory."}
                  {asrLoadState === "downloading" && "Downloading model files from Hugging Face."}
                  {asrLoadState === "ready" && !asrEnabled && "Model is ready. Enable ASR to transcribe new segments automatically."}
                  {asrLoadState === "ready" && asrEnabled && "New speech segments will be transcribed automatically."}
                  {asrLoadState === "error" && "ASR did not finish loading. You can retry or remove the cached files."}
                </p>
              </div>

              <div className="flex items-center gap-2 flex-wrap justify-end">
                {asrLoadState === "checking" && (
                  <span className="text-xs font-medium text-muted-foreground">Checking…</span>
                )}
                {asrLoadState === "ready" ? (
                  <button
                    onClick={asrEnabled ? disableAsr : () => setAsrEnabled(true)}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                      asrEnabled
                        ? "bg-secondary text-foreground hover:bg-secondary/80"
                        : "bg-primary text-primary-foreground hover:opacity-90"
                    }`}
                  >
                    {asrPrimaryActionLabel}
                  </button>
                ) : asrLoadState !== "checking" && (
                  <button
                    onClick={requestAsrEnable}
                    className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 transition-opacity"
                  >
                    {asrPrimaryActionLabel}
                  </button>
                )}
                {(asrLoadState === "downloading" || asrLoadState === "loading_cached") && (
                  <button
                    onClick={cancelAsrDownload}
                    className="px-3 py-1.5 rounded-md border border-border text-xs font-semibold text-foreground hover:bg-secondary/40 transition-colors"
                  >
                    Cancel
                  </button>
                )}
                {(asrCached || asrLoadState === "ready") && asrLoadState !== "downloading" && (
                  <button
                    onClick={removeAsrCache}
                    className="px-3 py-1.5 rounded-md border border-border text-xs font-semibold text-muted-foreground hover:text-foreground hover:bg-secondary/40 transition-colors"
                  >
                    Remove cached model
                  </button>
                )}
              </div>
            </div>

            <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground">
              <a
                href={PARAKEET_MODEL_CARD_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-2 hover:text-foreground/80 transition-colors"
              >
                Model card
              </a>
              <span>Initial load is large. After caching, enable is much faster.</span>
            </div>

            {asrLoadState === "downloading" && asrDownload && (
              <div className="mt-3">
                <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground mb-1.5">
                  <span>
                    {asrDownload.phase === "encoder" ? "Downloading encoder" : "Downloading decoder"}
                    {phaseDownloadPercent !== null ? ` ${phaseDownloadPercent.toFixed(0)}%` : ""}
                  </span>
                  <span>
                    {formatBytes(asrDownload.overallLoadedBytes)} / {formatBytes(asrDownload.overallTotalBytes)}
                  </span>
                </div>
                <div className="h-2 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full bg-primary transition-[width] duration-150"
                    style={{ width: `${overallDownloadPercent}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        {asrError && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {asrError}
          </div>
        )}

        <div className="rounded-xl border border-border bg-card overflow-hidden relative">
          <div className="absolute top-3 left-4 z-10">
            <span className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground/60">
              Waveform
            </span>
          </div>
          <canvas ref={canvasRef} className="w-full block" style={{ height: 180 }} />
        </div>

        <div>
          <div className="flex items-center gap-3 mb-4">
            <h2 className="text-lg font-bold">Segments</h2>
            {speechEndEvents.length > 0 && (
              <span className="text-xs font-mono font-medium bg-secondary text-muted-foreground px-2.5 py-0.5 rounded-full">
                {speechEndEvents.length}
              </span>
            )}
          </div>

          {speechEndEvents.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
              <div className="rounded-lg border border-border bg-card px-4 py-3">
                <div className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground/70">
                  Segments
                </div>
                <div className="mt-1 text-lg font-semibold">{speechEndEvents.length}</div>
              </div>
              <div className="rounded-lg border border-border bg-card px-4 py-3">
                <div className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground/70">
                  Speech Time
                </div>
                <div className="mt-1 text-lg font-semibold">{totalSpeechDuration.toFixed(1)}s</div>
              </div>
              <div className="rounded-lg border border-border bg-card px-4 py-3">
                <div className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground/70">
                  Transcripts
                </div>
                <div className="mt-1 text-lg font-semibold">{transcriptReadyCount}</div>
              </div>
              <div className="rounded-lg border border-border bg-card px-4 py-3">
                <div className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground/70">
                  Pending
                </div>
                <div className="mt-1 text-lg font-semibold">{transcriptPendingCount}</div>
              </div>
            </div>
          )}

          {speechEndEvents.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-secondary flex items-center justify-center">
                <svg
                  width="28"
                  height="28"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" x2="12" y1="19" y2="22" />
                </svg>
              </div>
              <p className="text-sm">
                {status === "recording" ? "Waiting for speech..." : "Start recording to detect speech segments"}
              </p>
            </div>
          )}

          <div className="space-y-2">
            {speechEndEvents.map((evt, idx) => {
              const isExpanded = expandedId !== null ? evt.id === expandedId : idx === 0;
              const labelType = evt.aed?.label ?? "speech";
              const labelColorMap: Record<string, string> = {
                speech: "bg-speech text-background",
                music: "bg-music text-background",
                noise: "bg-noise text-background",
              };
              const borderColorMap: Record<string, string> = {
                speech: "border-speech/30",
                music: "border-music/30",
                noise: "border-noise/30",
              };

              const transcriptPreview =
                evt.transcriptStatus === "ready" && evt.transcript
                  ? previewTranscript(evt.transcript)
                  : evt.transcriptStatus === "queued"
                    ? "Queued for transcription"
                    : evt.transcriptStatus === "transcribing"
                      ? "Transcribing…"
                      : evt.transcriptStatus === "error"
                        ? evt.transcriptError ?? "Transcription failed"
                        : asrEnabled && asrLoadState !== "ready"
                          ? "ASR will be available once the model finishes loading"
                          : undefined;

              return (
                <div
                  key={evt.id}
                  className={`rounded-xl bg-card border transition-all duration-200 overflow-hidden ${
                    isExpanded ? borderColorMap[labelType] || "border-border" : "border-border"
                  }`}
                  style={{ animation: "slide-up 0.3s ease-out" }}
                >
                  <div
                    onClick={() => setExpandedId(isExpanded ? null : evt.id)}
                    className={`flex items-start justify-between gap-4 cursor-pointer select-none transition-all ${
                      isExpanded ? "px-5 py-3.5" : "px-4 py-2.5"
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-3 flex-wrap">
                        <span
                          className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${labelColorMap[labelType] || "bg-muted text-muted-foreground"}`}
                        >
                          {labelType}
                        </span>
                        <span className="text-xs text-muted-foreground font-mono">
                          {evt.startSec.toFixed(1)}s – {evt.endSec.toFixed(1)}s
                        </span>
                      </div>
                      {transcriptPreview && (
                        <p
                          className={`mt-1 text-sm truncate ${
                            evt.transcriptStatus === "error"
                              ? "text-destructive"
                              : evt.transcriptStatus === "queued" || evt.transcriptStatus === "transcribing"
                                ? "text-muted-foreground"
                                : "text-foreground"
                          }`}
                        >
                          {transcriptPreview}
                        </p>
                      )}
                    </div>

                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-sm font-mono font-semibold text-foreground">
                        {evt.durationSec?.toFixed(2)}s
                      </span>
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 12 12"
                        className={`text-muted-foreground transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`}
                      >
                        <path
                          d="M2 4l4 4 4-4"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          fill="none"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="px-5 pb-4">
                      {evt.audioUrl && (
                        <div className="flex items-center gap-3 mb-3">
                          <audio
                            controls
                            src={evt.audioUrl}
                            className="h-8 flex-1 opacity-80"
                            style={{ filter: "invert(1) hue-rotate(180deg)" }}
                          />
                          <a
                            href={evt.audioUrl}
                            download={`segment_${evt.id}_${evt.startSec.toFixed(1)}s.wav`}
                            className="text-xs font-mono font-medium text-accent hover:text-accent/80 px-3 py-1.5 rounded-md border border-accent/30 hover:border-accent/50 transition-colors whitespace-nowrap"
                          >
                            ↓ .wav
                          </a>
                        </div>
                      )}

                      {evt.aed && <AedBars probs={evt.aed.probs} />}

                      {(evt.transcriptStatus || (evt.audioUrl && asrLoadState === "ready")) && (
                        <div className="mt-3 rounded-lg border border-border/70 bg-secondary/20 px-4 py-3">
                          <div className="flex items-center justify-between gap-3 mb-2 flex-wrap">
                            <div className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground/70">
                              Transcript
                            </div>
                            <div className="flex items-center gap-2 flex-wrap">
                              {evt.transcriptStatus === "ready" && evt.transcript && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    void copyTranscript(evt.id, evt.transcript ?? "");
                                  }}
                                  className="px-2.5 py-1 rounded-md border border-border text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
                                >
                                  {copiedEventId === evt.id ? "Copied" : "Copy transcript"}
                                </button>
                              )}
                              {evt.audioUrl &&
                                asrLoadState === "ready" &&
                                evt.transcriptStatus !== "queued" &&
                                evt.transcriptStatus !== "transcribing" && (
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      requestTranscript(evt.id);
                                    }}
                                    className="px-2.5 py-1 rounded-md border border-border text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
                                  >
                                    {evt.transcriptStatus === "ready" ? "Transcribe again" : "Transcribe segment"}
                                  </button>
                                )}
                            </div>
                          </div>

                          {!evt.transcriptStatus && (
                            <p className="text-sm text-muted-foreground">
                              Load or enable ASR to transcribe this segment.
                            </p>
                          )}
                          {evt.transcriptStatus === "queued" && (
                            <p className="text-sm text-muted-foreground">Queued for transcription.</p>
                          )}
                          {evt.transcriptStatus === "transcribing" && (
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <div className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                              <span>Transcribing…</span>
                            </div>
                          )}
                          {evt.transcriptStatus === "ready" && (
                            <p className="text-sm leading-6 text-foreground">{evt.transcript}</p>
                          )}
                          {evt.transcriptStatus === "error" && (
                            <p className="text-sm text-destructive">{evt.transcriptError}</p>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <footer className="pt-8 pb-4 border-t border-border/30 text-center">
          <p className="text-xs text-muted-foreground/50">
            Powered by ONNX Runtime Web · Inference runs locally in your browser after model download ·{" "}
            <a
              href="https://github.com/eschmidbauer/fireredvad.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground/70 hover:text-foreground/80 transition-colors underline underline-offset-2"
            >
              Source code
            </a>{" "}
            ·{" "}
            <a
              href="https://github.com/FireRedTeam/FireRedVAD/tree/main/pretrained_models/onnx_models"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground/70 hover:text-foreground/80 transition-colors underline underline-offset-2"
            >
              ONNX Models
            </a>
          </p>
        </footer>
      </main>
    </div>
  );
}
