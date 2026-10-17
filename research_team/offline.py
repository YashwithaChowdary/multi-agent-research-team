"""Offline mode: no AI. Simple extractive rules stand in for each agent so the
app, the demo and the tests run without a key or internet. Its answers are
copied sentences, not real reasoning; use a real provider for actual research."""
from __future__ import annotations

import re

from .llm import LLM
from .sources import tokens


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+|(?<=:)\s+(?=- )|\s+-\s+(?=[A-Z])", text)
    return [p.strip(" -") for p in parts if len(p.strip()) > 25]


class OfflineLLM(LLM):
    name = "offline"

    def _call(self, role, system, prompt, schema, payload=None):
        p = payload or {}
        return getattr(self, "_" + role)(p), 0, 0

    def _planner(self, p):
        q = p["question"].rstrip("?")
        parts = [x.strip() for x in re.split(r",\s*and\s+|\s+and\s+(?=(?:what|how|why|who|which|when|where|is|are|were|was|did|does)\b)|;\s*", q, flags=re.I) if x.strip()]
        if len(parts) < 2:
            parts = [q]
        return {"sub_questions": [{"question": x + "?", "search_query": " ".join(tokens(x))} for x in parts]}

    def _researcher(self, p):
        # ignore words found in most passages (like the company name): they do not discriminate
        df: dict[str, int] = {}
        for ps in p["passages"]:
            for w in set(tokens(ps["text"])):
                df[w] = df.get(w, 0) + 1
        common = {w for w, c in df.items() if c > max(1, len(p["passages"]) // 2)}
        q = set(tokens(p["sub"]["question"])) - common
        scored = []
        for ps in p["passages"]:
            for s in _sentences(ps["text"]):
                overlap = len(q & set(tokens(s)))
                if overlap >= 2 or (overlap and len(q) <= 2):
                    scored.append((overlap, s, ps["source_id"]))
        scored.sort(key=lambda x: -x[0])
        seen, notes = set(), []
        for _, s, sid in scored:
            if s not in seen:
                seen.add(s)
                notes.append({"claim": s, "quote": s, "source_id": sid})
            if len(notes) == 3:
                break
        gaps = [] if notes else [f"Nothing found for: {p['sub']['question']}"]
        return {"notes": notes, "gaps": gaps}

    def _writer(self, p):
        by_sub: dict[str, list] = {}
        for n in p["notes"]:
            by_sub.setdefault(n.get("sub", "S"), []).append(n)
        sections = [{"heading": f"Finding {i}", "sentences": [{"text": n["claim"], "note_ids": [n["id"]]} for n in ns]}
                    for i, ns in enumerate(by_sub.values(), 1)]
        return {"title": p["question"], "sections": sections, "unanswered": list(p.get("gaps", []))}

    _baseline = _writer

    def _critic(self, p):
        verdicts, i = [], 0
        for sec in p["draft"].get("sections", []):
            for s in sec.get("sentences", []):
                ev = " ".join(p["evidence"].get(n, "") for n in s.get("note_ids", []))
                ok = bool(ev) and set(tokens(s["text"])) <= set(tokens(ev))
                verdicts.append({"n": i, "supported": "yes" if ok else "no", "problem": "" if ok else "not in evidence"})
                i += 1
        return {"verdicts": verdicts, "missing": []}
