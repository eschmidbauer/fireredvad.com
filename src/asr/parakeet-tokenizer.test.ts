import { describe, expect, it } from "vitest";
import { decodeSentencePieces } from "./parakeet-tokenizer";

describe("decodeSentencePieces", () => {
  it("reconstructs word boundaries from sentencepiece markers", () => {
    const pieces = ["<unk>", "\u2581t", "\u2581th", "\u2581a", "in", "\u2581the"];
    expect(decodeSentencePieces([1, 2, 3, 4, 5], pieces)).toBe("t th ain the");
  });

  it("drops special tokens and normalizes repeated spaces", () => {
    const pieces = ["<unk>", "\u2581hello", "\u2581", "world", "<s>"];
    expect(decodeSentencePieces([4, 1, 2, 3, 0], pieces)).toBe("hello world");
  });
});
