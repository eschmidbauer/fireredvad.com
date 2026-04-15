export interface ParakeetVocab {
  blankId: number;
  pieces: string[];
}

export async function loadParakeetVocab(url: string): Promise<ParakeetVocab> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ASR vocab: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<ParakeetVocab>;
}

export function decodeSentencePieces(tokenIds: number[], pieces: string[]): string {
  let combined = "";
  for (const tokenId of tokenIds) {
    const piece = pieces[tokenId];
    if (!piece || piece.startsWith("<")) {
      continue;
    }
    combined += piece;
  }

  return combined.replaceAll("\u2581", " ").replace(/\s+/g, " ").trim();
}
