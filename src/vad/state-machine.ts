import type { VadEvent } from "./types";

export class VadStateMachine {
  private frameCnt = 0;
  private smoothWindow: number[] = [];
  private smoothWindowSum = 0;
  private state: "SILENCE" | "POSSIBLE_SPEECH" | "SPEECH" | "POSSIBLE_SILENCE" = "SILENCE";
  private speechCnt = 0;
  private silenceCnt = 0;
  private hitMaxSpeech = false;
  private lastSpeechStartFrame = -1;
  private lastSpeechEndFrame = -1;

  // Config
  private smoothWindowSize = 5;
  private speechThreshold = 0.4;
  private padStartFrame = 5;
  private minSpeechFrame = 8;
  private maxSpeechFrame = 2000;
  private minSilenceFrame = 20;

  constructor() {
    this.padStartFrame = Math.max(this.smoothWindowSize, this.padStartFrame);
  }

  reset(): void {
    this.frameCnt = 0;
    this.smoothWindow = [];
    this.smoothWindowSum = 0;
    this.state = "SILENCE";
    this.speechCnt = 0;
    this.silenceCnt = 0;
    this.hitMaxSpeech = false;
    this.lastSpeechStartFrame = -1;
    this.lastSpeechEndFrame = -1;
  }

  processFrame(rawProb: number): VadEvent | null {
    this.frameCnt++;

    this.smoothWindow.push(rawProb);
    this.smoothWindowSum += rawProb;
    if (this.smoothWindow.length > this.smoothWindowSize) {
      this.smoothWindowSum -= this.smoothWindow.shift()!;
    }
    const smoothed = this.smoothWindowSum / this.smoothWindow.length;
    const isSpeech = smoothed >= this.speechThreshold;

    let event: VadEvent | null = null;

    if (this.hitMaxSpeech) {
      event = { type: "speech_start", startFrame: this.frameCnt, endFrame: this.frameCnt };
      this.lastSpeechStartFrame = this.frameCnt;
      this.hitMaxSpeech = false;
    }

    if (this.state === "SILENCE") {
      if (isSpeech) {
        this.state = "POSSIBLE_SPEECH";
        this.speechCnt = 1;
      } else {
        this.silenceCnt++;
        this.speechCnt = 0;
      }
    } else if (this.state === "POSSIBLE_SPEECH") {
      if (isSpeech) {
        this.speechCnt++;
        if (this.speechCnt >= this.minSpeechFrame) {
          this.state = "SPEECH";
          const start = Math.max(1, this.frameCnt - this.speechCnt + 1 - this.padStartFrame, this.lastSpeechEndFrame + 1);
          this.lastSpeechStartFrame = start;
          this.silenceCnt = 0;
          event = { type: "speech_start", startFrame: start, endFrame: this.frameCnt };
        }
      } else {
        this.state = "SILENCE";
        this.silenceCnt = 1;
        this.speechCnt = 0;
      }
    } else if (this.state === "SPEECH") {
      this.speechCnt++;
      if (isSpeech) {
        this.silenceCnt = 0;
        if (this.speechCnt >= this.maxSpeechFrame) {
          this.hitMaxSpeech = true;
          this.speechCnt = 0;
          event = { type: "speech_end", startFrame: this.lastSpeechStartFrame, endFrame: this.frameCnt };
          this.lastSpeechEndFrame = this.frameCnt;
          this.lastSpeechStartFrame = -1;
        }
      } else {
        this.state = "POSSIBLE_SILENCE";
        this.silenceCnt = 1;
      }
    } else if (this.state === "POSSIBLE_SILENCE") {
      this.speechCnt++;
      if (isSpeech) {
        this.state = "SPEECH";
        this.silenceCnt = 0;
        if (this.speechCnt >= this.maxSpeechFrame) {
          this.hitMaxSpeech = true;
          this.speechCnt = 0;
          event = { type: "speech_end", startFrame: this.lastSpeechStartFrame, endFrame: this.frameCnt };
          this.lastSpeechEndFrame = this.frameCnt;
          this.lastSpeechStartFrame = -1;
        }
      } else {
        this.silenceCnt++;
        if (this.silenceCnt >= this.minSilenceFrame) {
          this.state = "SILENCE";
          event = { type: "speech_end", startFrame: this.lastSpeechStartFrame, endFrame: this.frameCnt };
          this.lastSpeechEndFrame = this.frameCnt;
          this.lastSpeechStartFrame = -1;
          this.speechCnt = 0;
        }
      }
    }

    return event;
  }

  flush(): VadEvent | null {
    if (this.state === "SPEECH" || this.state === "POSSIBLE_SILENCE") {
      const event: VadEvent = {
        type: "speech_end",
        startFrame: this.lastSpeechStartFrame,
        endFrame: this.frameCnt,
      };
      this.state = "SILENCE";
      this.lastSpeechEndFrame = this.frameCnt;
      this.lastSpeechStartFrame = -1;
      this.speechCnt = 0;
      return event;
    }
    return null;
  }
}
