/**
 * Kaldi-compatible 80-dim log-mel filterbank feature extraction.
 *
 * Settings: 16kHz, 25ms frame, 10ms shift, 80 mel bins,
 * Povey window, pre-emphasis 0.97, snip_edges=true.
 */

const SAMPLE_RATE = 16000;
const FRAME_LENGTH = 400; // 25ms
const FRAME_SHIFT = 160; // 10ms
const FFT_SIZE = 512;
const NUM_MEL_BINS = 80;
const LOW_FREQ = 20;
const HIGH_FREQ = SAMPLE_RATE / 2; // 8000
const PRE_EMPHASIS = 0.97;

function hertzToMel(f: number): number {
  return 1127.0 * Math.log(1.0 + f / 700.0);
}

function melToHertz(m: number): number {
  return 700.0 * (Math.exp(m / 1127.0) - 1.0);
}

function createPoveyWindow(): Float32Array {
  const win = new Float32Array(FRAME_LENGTH);
  for (let i = 0; i < FRAME_LENGTH; i++) {
    const hamming = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (FRAME_LENGTH - 1));
    win[i] = Math.pow(hamming, 0.85);
  }
  return win;
}

function createMelFilterbank(): Float32Array[] {
  const numFftBins = FFT_SIZE / 2 + 1;
  const melLow = hertzToMel(LOW_FREQ);
  const melHigh = hertzToMel(HIGH_FREQ);

  const melPoints = new Float32Array(NUM_MEL_BINS + 2);
  for (let i = 0; i < NUM_MEL_BINS + 2; i++) {
    melPoints[i] = melLow + ((melHigh - melLow) * i) / (NUM_MEL_BINS + 1);
  }

  const binPoints = new Float32Array(NUM_MEL_BINS + 2);
  for (let i = 0; i < NUM_MEL_BINS + 2; i++) {
    binPoints[i] = (melToHertz(melPoints[i]) * FFT_SIZE) / SAMPLE_RATE;
  }

  const filters: Float32Array[] = [];
  for (let m = 0; m < NUM_MEL_BINS; m++) {
    const filter = new Float32Array(numFftBins);
    const left = binPoints[m];
    const center = binPoints[m + 1];
    const right = binPoints[m + 2];

    for (let k = 0; k < numFftBins; k++) {
      if (k >= left && k <= center && center > left) {
        filter[k] = (k - left) / (center - left);
      } else if (k > center && k <= right && right > center) {
        filter[k] = (right - k) / (right - center);
      }
    }
    filters.push(filter);
  }

  return filters;
}

/** In-place radix-2 FFT */
function fft(re: Float32Array, im: Float32Array, N: number): void {
  let j = 0;
  for (let i = 0; i < N - 1; i++) {
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
    let m = N >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let step = 1; step < N; step <<= 1) {
    const angle = -Math.PI / step;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let group = 0; group < N; group += step << 1) {
      let curRe = 1,
        curIm = 0;
      for (let pair = 0; pair < step; pair++) {
        const a = group + pair;
        const b = a + step;
        const tRe = curRe * re[b] - curIm * im[b];
        const tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
}

function powerSpectrum(frame: Float32Array): Float32Array {
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);
  re.set(frame);

  fft(re, im, FFT_SIZE);

  const numBins = FFT_SIZE / 2 + 1;
  const power = new Float32Array(numBins);
  for (let k = 0; k < numBins; k++) {
    power[k] = re[k] * re[k] + im[k] * im[k];
  }
  return power;
}

export const FBANK_FRAME_SHIFT = FRAME_SHIFT;

export class FbankExtractor {
  private window: Float32Array;
  private melFilterbank: Float32Array[];
  private remainder: Int16Array;

  constructor() {
    this.window = createPoveyWindow();
    this.melFilterbank = createMelFilterbank();
    this.remainder = new Int16Array(0);
  }

  reset(): void {
    this.remainder = new Int16Array(0);
  }

  /** Extract fbank features from streaming int16 PCM. Handles partial frames. */
  extract(pcmInt16: Int16Array): Float32Array[] | null {
    const combined = new Int16Array(this.remainder.length + pcmInt16.length);
    combined.set(this.remainder);
    combined.set(pcmInt16, this.remainder.length);

    const numSamples = combined.length;
    const numFrames = Math.max(
      0,
      Math.floor((numSamples - FRAME_LENGTH) / FRAME_SHIFT) + 1
    );

    if (numFrames === 0) {
      this.remainder = combined;
      return null;
    }

    const frames: Float32Array[] = [];

    for (let i = 0; i < numFrames; i++) {
      const start = i * FRAME_SHIFT;

      // Pre-emphasis + windowing
      const frame = new Float32Array(FRAME_LENGTH);
      frame[0] =
        combined[start] -
        PRE_EMPHASIS * (start > 0 ? combined[start - 1] : combined[start]);
      for (let j = 1; j < FRAME_LENGTH; j++) {
        frame[j] = combined[start + j] - PRE_EMPHASIS * combined[start + j - 1];
      }
      for (let j = 0; j < FRAME_LENGTH; j++) {
        frame[j] *= this.window[j];
      }

      // Power spectrum -> mel filterbank -> log
      const power = powerSpectrum(frame);
      const fbank = new Float32Array(NUM_MEL_BINS);
      for (let m = 0; m < NUM_MEL_BINS; m++) {
        let sum = 0;
        const filter = this.melFilterbank[m];
        for (let k = 0; k < power.length; k++) {
          sum += filter[k] * power[k];
        }
        fbank[m] = Math.log(Math.max(sum, 1e-10));
      }

      frames.push(fbank);
    }

    const consumedSamples = numFrames * FRAME_SHIFT;
    this.remainder = combined.slice(consumedSamples);

    return frames;
  }

  /** Extract features from a complete audio segment (for AED). */
  extractSegment(pcmInt16: Int16Array): Float32Array[] | null {
    const extractor = new FbankExtractor();
    return extractor.extract(pcmInt16);
  }
}
