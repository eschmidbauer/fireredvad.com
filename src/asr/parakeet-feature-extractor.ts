const SAMPLE_RATE = 16000;
const N_FFT = 512;
const WIN_LENGTH = 400;
const HOP_LENGTH = 160;
const NUM_MELS = 128;
const PAD = N_FFT / 2;
const LOG_EPSILON = 2 ** -24;

function hzToMelSlaney(hz: number): number {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;

  if (hz < minLogHz) {
    return hz / fSp;
  }

  return minLogMel + Math.log(hz / minLogHz) / logStep;
}

function melToHzSlaney(mel: number): number {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;

  if (mel < minLogMel) {
    return mel * fSp;
  }

  return minLogHz * Math.exp(logStep * (mel - minLogMel));
}

function createCenteredHannWindow(): Float32Array {
  const window = new Float32Array(N_FFT);
  const offset = (N_FFT - WIN_LENGTH) / 2;
  for (let i = 0; i < WIN_LENGTH; i++) {
    window[offset + i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN_LENGTH - 1));
  }
  return window;
}

function createMelFilterbank(): Float32Array[] {
  const numBins = N_FFT / 2 + 1;
  const fftFrequencies = new Float32Array(numBins);
  for (let i = 0; i < numBins; i++) {
    fftFrequencies[i] = (SAMPLE_RATE * i) / N_FFT;
  }

  const minMel = hzToMelSlaney(0);
  const maxMel = hzToMelSlaney(SAMPLE_RATE / 2);
  const melPoints = new Float32Array(NUM_MELS + 2);
  const hzPoints = new Float32Array(NUM_MELS + 2);

  for (let i = 0; i < NUM_MELS + 2; i++) {
    melPoints[i] = minMel + ((maxMel - minMel) * i) / (NUM_MELS + 1);
    hzPoints[i] = melToHzSlaney(melPoints[i]);
  }

  const filters: Float32Array[] = [];
  for (let m = 0; m < NUM_MELS; m++) {
    const filter = new Float32Array(numBins);
    const left = hzPoints[m];
    const center = hzPoints[m + 1];
    const right = hzPoints[m + 2];
    const enorm = 2 / Math.max(right - left, 1e-12);

    for (let k = 0; k < numBins; k++) {
      const freq = fftFrequencies[k];
      let weight = 0;

      if (freq >= left && freq <= center && center > left) {
        weight = (freq - left) / (center - left);
      } else if (freq > center && freq <= right && right > center) {
        weight = (right - freq) / (right - center);
      }

      filter[k] = weight * enorm;
    }

    filters.push(filter);
  }

  return filters;
}

function reflectIndex(index: number, length: number): number {
  if (length <= 1) {
    return 0;
  }

  let reflected = index;
  while (reflected < 0 || reflected >= length) {
    if (reflected < 0) {
      reflected = -reflected;
    } else {
      reflected = 2 * length - 2 - reflected;
    }
  }
  return reflected;
}

function reflectPad(signal: Float32Array, pad: number): Float32Array {
  const padded = new Float32Array(signal.length + pad * 2);
  for (let i = 0; i < padded.length; i++) {
    padded[i] = signal[reflectIndex(i - pad, signal.length)];
  }
  return padded;
}

function fft(re: Float32Array, im: Float32Array, size: number): void {
  let j = 0;
  for (let i = 0; i < size - 1; i++) {
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
    let m = size >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let step = 1; step < size; step <<= 1) {
    const angle = -Math.PI / step;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let group = 0; group < size; group += step << 1) {
      let curRe = 1;
      let curIm = 0;
      for (let pair = 0; pair < step; pair++) {
        const a = group + pair;
        const b = a + step;
        const tRe = curRe * re[b] - curIm * im[b];
        const tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

function powerSpectrum(frame: Float32Array): Float32Array {
  const re = new Float32Array(N_FFT);
  const im = new Float32Array(N_FFT);
  re.set(frame);
  fft(re, im, N_FFT);

  const bins = new Float32Array(N_FFT / 2 + 1);
  for (let i = 0; i < bins.length; i++) {
    bins[i] = re[i] * re[i] + im[i] * im[i];
  }
  return bins;
}

export interface ParakeetFeatures {
  frameCount: number;
  values: Float32Array;
}

export class ParakeetFeatureExtractor {
  private readonly window = createCenteredHannWindow();
  private readonly melFilterbank = createMelFilterbank();

  extract(pcmInt16: Int16Array): ParakeetFeatures | null {
    if (pcmInt16.length === 0) {
      return null;
    }

    const signal = new Float32Array(pcmInt16.length);
    for (let i = 0; i < pcmInt16.length; i++) {
      signal[i] = pcmInt16[i] / 32768;
    }

    const padded = reflectPad(signal, PAD);
    const frameCount = Math.floor((padded.length - N_FFT) / HOP_LENGTH) + 1;
    if (frameCount <= 0) {
      return null;
    }

    const frameFeatures: Float32Array[] = [];
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
      const start = frameIndex * HOP_LENGTH;
      const frame = new Float32Array(N_FFT);
      for (let i = 0; i < N_FFT; i++) {
        frame[i] = padded[start + i] * this.window[i];
      }

      const spectrum = powerSpectrum(frame);
      const mel = new Float32Array(NUM_MELS);
      for (let melIndex = 0; melIndex < NUM_MELS; melIndex++) {
        let energy = 0;
        const filter = this.melFilterbank[melIndex];
        for (let binIndex = 0; binIndex < spectrum.length; binIndex++) {
          energy += filter[binIndex] * spectrum[binIndex];
        }
        mel[melIndex] = Math.log(energy + LOG_EPSILON);
      }
      frameFeatures.push(mel);
    }

    const means = new Float32Array(NUM_MELS);
    const variances = new Float32Array(NUM_MELS);
    for (const frame of frameFeatures) {
      for (let melIndex = 0; melIndex < NUM_MELS; melIndex++) {
        means[melIndex] += frame[melIndex];
      }
    }
    for (let melIndex = 0; melIndex < NUM_MELS; melIndex++) {
      means[melIndex] /= frameCount;
    }
    for (const frame of frameFeatures) {
      for (let melIndex = 0; melIndex < NUM_MELS; melIndex++) {
        const diff = frame[melIndex] - means[melIndex];
        variances[melIndex] += diff * diff;
      }
    }

    const values = new Float32Array(NUM_MELS * frameCount);
    for (let melIndex = 0; melIndex < NUM_MELS; melIndex++) {
      const std = Math.sqrt(variances[melIndex] / frameCount) + 1e-5;
      for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        values[melIndex * frameCount + frameIndex] =
          (frameFeatures[frameIndex][melIndex] - means[melIndex]) / std;
      }
    }

    return { frameCount, values };
  }
}
