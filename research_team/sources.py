"""Where the Researcher looks things up: a local folder of documents, or the web."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOP = set("""a an the and or of to in on for with by at from as is are was were be been it its this that these those
which who what when where how why does do did can could should would will may might than then there their they
you your we our not no yes if into about over under after before between per vs also has have had more most""".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


@dataclass
class Passage:
    source_id: str        # stable id, e.g. "local:pricing.md#2" or a URL
    title: str
    url: str
    text: str
    score: float = 0.0


class LocalCorpus:
    """BM25 search over markdown/text files split into overlapping chunks."""

    def __init__(self, folder: str | Path, chunk_words: int = 90, overlap: int = 25):
        self.folder = Path(folder)
        self.chunks: list[Passage] = []
        self.full_text: dict[str, str] = {}
        for f in sorted(self.folder.glob("*.md")) + sorted(self.folder.glob("*.txt")):
            text = f.read_text(encoding="utf-8")
            title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.strip()), f.stem)
            # pack whole lines into chunks so sentences and list items are never cut in half;
            # the last line of a chunk is repeated at the start of the next for context
            lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
            chunk, n = [], 0
            for line in lines:
                chunk.append(line)
                if sum(len(c.split()) for c in chunk) >= chunk_words:
                    self._add(f, title, n, chunk); n += 1
                    chunk = chunk[-1:] if overlap else []
            if chunk and (n == 0 or len(chunk) > 1):
                self._add(f, title, n, chunk)
        if not self.chunks:
            raise ValueError(f"No .md or .txt files in {self.folder}")
        self._tf = [Counter(tokens(c.text)) for c in self.chunks]
        self._len = [sum(t.values()) for t in self._tf]
        self._avg = sum(self._len) / len(self._len)
        df = Counter(w for t in self._tf for w in t)
        n = len(self.chunks)
        self._idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def _add(self, f: Path, title: str, n: int, lines: list[str]):
        sid = f"local:{f.name}#{n}"
        text = "\n".join(lines)
        self.chunks.append(Passage(sid, title, f.name, text))
        self.full_text[sid] = text

    def search(self, query: str, k: int = 4) -> list[Passage]:
        q = tokens(query)
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self._idf[w] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self._len[i] / self._avg))
            scores.append(s)
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [Passage(**{**self.chunks[i].__dict__, "score": round(scores[i], 2)}) for i in best if scores[i] > 0]


class WebSearch:
    """DuckDuckGo search (no key) plus a light page fetch. Results depend on the live web."""

    def __init__(self, max_chars: int = 3000):
        self.max_chars = max_chars
        self.full_text: dict[str, str] = {}

    def _fetch(self, url: str) -> str:
        import httpx
        from bs4 import BeautifulSoup
        try:
            r = httpx.get(url, timeout=12, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 research-team"})
            if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                return ""
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "h1", "h2", "h3"]))
            return re.sub(r"\s+", " ", text)[: self.max_chars]
        except Exception:                            # noqa: BLE001 - a bad page should not stop the run
            return ""

    def search(self, query: str, k: int = 4) -> list[Passage]:
        from ddgs import DDGS
        try:
            hits = DDGS().text(query, max_results=k + 2)
        except Exception as e:                       # noqa: BLE001
            raise RuntimeError(f"Web search failed: {e}") from e
        out = []
        for h in hits:
            url = h.get("href") or h.get("url") or ""
            text = self._fetch(url) or h.get("body", "")
            if len(text) < 80:
                continue
            self.full_text[url] = text
            out.append(Passage(url, h.get("title", url), url, text))
            if len(out) == k:
                break
        return out
