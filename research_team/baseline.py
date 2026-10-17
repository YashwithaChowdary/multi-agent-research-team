"""Single-call baseline: retrieve passages once, ask the model once. Used to show
what the team adds (and what it costs) in the evaluation."""
from __future__ import annotations

import time

from . import agents
from .llm import LLM

BASE_SYSTEM = """Answer the question using ONLY the numbered passages. Every sentence must list the passage
ids it relies on in note_ids (for example ["P2"]). If the passages do not answer something, put it in unanswered."""


def run_baseline(question: str, llm: LLM, search, k: int = 8) -> dict:
    t0 = time.perf_counter()
    passages = search.search(question, k=k)
    notes = [{"id": f"P{i}", "claim": p.text, "quote": p.text, "title": p.title, "url": p.url,
              "source_id": p.source_id} for i, p in enumerate(passages, 1)]
    block = "\n\n".join(f"{n['id']} ({n['title']}): {n['quote']}" for n in notes)
    draft = llm.json("baseline", BASE_SYSTEM, f"Question: {question}\n\nPassages:\n{block}", agents.REPORT_SCHEMA,
                     {"question": question, "notes": notes, "gaps": []})
    return {"question": question, "notes": notes, "draft": draft, "usage": llm.usage.as_dict(),
            "seconds": round(time.perf_counter() - t0, 1)}
