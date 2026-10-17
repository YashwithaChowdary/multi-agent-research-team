"""Checks done by code, not by the model.

A note is only kept if its quote really appears in the source it cites.
A report sentence is flagged if it cites nothing, or cites a note that does not exist.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^\w%.,'₹ ]+", " ", s)
    s = re.sub(r"(\d)\s+%", r"\1%", s)
    return re.sub(r"\s+", " ", s).strip(" .,'\"")


_NUM = re.compile(r"\d+(?:[.,]\d+)*%?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def quote_in_source(quote: str, source: str, min_ratio: float = 0.9) -> bool:
    q, s = norm(quote), norm(source)
    if len(q) < 12:
        return False                       # too short to prove anything
    if q in s:
        return True
    # fuzzy matching may forgive a changed word, but never a changed number
    if not _numbers(q) <= _numbers(s):
        return False
    m = SequenceMatcher(None, s, q, autojunk=False).find_longest_match(0, len(s), 0, len(q))
    if m.size >= min_ratio * len(q):
        return True
    # allow small edits (a dropped word, changed punctuation): compare with the aligned window of the source
    start = max(0, m.a - m.b)
    window = s[start:start + len(q) + 10]
    return SequenceMatcher(None, window, q, autojunk=False).ratio() >= min_ratio

def check_draft(draft: dict, note_ids: set[str]) -> list[dict]:
    issues = []
    for si, sec in enumerate(draft.get("sections", [])):
        for ti, sent in enumerate(sec.get("sentences", [])):
            ids = sent.get("note_ids") or []
            bad = [i for i in ids if i not in note_ids]
            if not ids:
                issues.append({"section": si, "sentence": ti, "problem": "no citation"})
            elif bad:
                issues.append({"section": si, "sentence": ti, "problem": f"cites unknown notes {bad}"})
    return issues
