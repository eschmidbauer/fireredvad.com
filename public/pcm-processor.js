/**
 * AudioWorklet processor that captures raw PCM and posts int16 chunks.
 * Handles downsampling if the AudioContext sample rate differs from 16kHz.
 */
class PcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = Math.round(sampleRate / this.targetRate);
    if (this.ratio < 1) this.ratio = 1;
    this.buffer = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const float32 = input[0];

    if (this.ratio <= 1) {
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
    } else {
      // Accumulate and downsample
      const combined = new Float32Array(this.buffer.length + float32.length);
      combined.set(this.buffer);
      combined.set(float32, this.buffer.length);

      const outLen = Math.floor(combined.length / this.ratio);
      if (outLen === 0) {
        this.buffer = combined;
        return true;
      }

      const int16 = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const s = Math.max(-1, Math.min(1, combined[i * this.ratio]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }

      this.buffer = combined.slice(outLen * this.ratio);
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true;
  }
}

registerProcessor("pcm-processor", PcmProcessor);
