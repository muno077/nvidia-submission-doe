"""Tiny search engine over the challenge docs, for agentic RAG.

The agent decides WHEN to look something up and WHAT to ask; this module just ranks doc chunks
with BM25 (the classic keyword ranking behind most search engines). No embeddings or vector DB:
the docs are small, BM25 is instant, deterministic and needs no extra model or API.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


def chunk_markdown(path: Path, max_chars: int = 1200) -> list[Chunk]:
    """Split on headings, then on paragraphs so each chunk stays small."""
    chunks, heading, buf = [], path.stem, []
    def flush():
        text = "\n".join(buf).strip()
        while text:
            cut = text.rfind("\n\n", 0, max_chars) if len(text) > max_chars else len(text)
            cut = cut if cut > 0 else max_chars
            chunks.append(Chunk(path.name, heading, text[:cut].strip()))
            text = text[cut:].strip()
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            flush()
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    flush()
    return [c for c in chunks if c.text]


class DocIndex:
    def __init__(self, folder: str | Path, k1: float = 1.5, b: float = 0.75):
        self.chunks = [c for p in sorted(Path(folder).glob("*.md")) for c in chunk_markdown(p)]
        self.toks = [_tokens(c.heading + " " + c.text) for c in self.chunks]
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(t for toks in self.toks for t in set(toks))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        q = _tokens(query)
        scores = []
        for i, toks in enumerate(self.toks):
            tf = Counter(toks)
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avg))
            scores.append((s, i))
        return [self.chunks[i] for s, i in sorted(scores, reverse=True)[:k] if s > 0]
