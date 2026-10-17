"""Team vs single-call baseline on the fictional Lumora corpus.

Metrics per system (averaged over questions):
  fact_recall     share of hand-written key facts found in the answer (code, deterministic)
  cited           share of sentences that cite at least one valid note/passage (code)
  supported       share of sentences the Critic model judges fully supported by their evidence
                  (LLM judge: the same model judges both systems, which may favour itself)
  abstained       on unanswerable questions: said it could not answer, without inventing a number
  calls / tokens / seconds  cost
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import agents
from .baseline import run_baseline
from .llm import get_llm
from .orchestrator import run_to_end
from .sources import LocalCorpus

ROOT = Path(__file__).resolve().parent.parent


def _text(draft: dict) -> str:
    return " ".join(s["text"] for s in agents.flatten(draft))


def _norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", s.lower())


def fact_recall(draft: dict, facts: list[list[str]]) -> float | None:
    if not facts:
        return None
    t = _norm(_text(draft))
    return sum(any(_norm(a) in t for a in alts) for alts in facts) / len(facts)


def abstained(draft: dict) -> bool:
    """Passes if the system said it could not answer and did not state a market-share or pay figure."""
    sents = agents.flatten(draft)
    invented = any(re.search(r"\d", s["text"]) and re.search(r"share|salary|paid|pay|compensation", s["text"], re.I)
                   for s in sents)
    return bool(draft.get("unanswered")) and not invented


def score(draft: dict, review: dict, note_ids: set[str], q: dict) -> dict:
    sents = agents.flatten(draft)
    n = len(sents) or 1
    cited = sum(1 for s in sents if s.get("note_ids") and all(i in note_ids for i in s["note_ids"]))
    if not sents:
        cited = None
    sup = sum(1 for v in review["verdicts"] if v["supported"] == "yes")
    return {"fact_recall": fact_recall(draft, q["facts"]), "cited": None if cited is None else cited / n,
            "supported": sup / n if sents else None, "sentences": len(sents),
            "abstained": abstained(draft) if not q["answerable"] else None}


def evaluate(provider: str = "auto", model: str | None = None, only: list[str] | None = None,
             pause: float = 0.0, log=print) -> dict:
    corpus = LocalCorpus(ROOT / "corpus")
    qs = json.loads((ROOT / "eval" / "questions.json").read_text())["questions"]
    if only:
        qs = [q for q in qs if q["id"] in only]
    rows = []
    for q in qs:
        log(f"[{q['id']}] team ...")
        team_llm = get_llm(provider, model)
        t = run_to_end(q["question"], team_llm, corpus)
        ts = score(t["draft"], t["review"], {n["id"] for n in t["notes"]}, q)
        ts.update(calls=t["usage"]["calls"], tokens=t["usage"]["input_tokens"] + t["usage"]["output_tokens"], seconds=t["seconds"])
        time.sleep(pause)
        log(f"[{q['id']}] baseline ...")
        base_llm = get_llm(provider, model)
        b = run_baseline(q["question"], base_llm, corpus)
        calls_before = base_llm.usage.calls
        review = agents.critique(base_llm, q["question"], b["draft"], {n["id"]: n["quote"] for n in b["notes"]})
        bs = score(b["draft"], review, {n["id"] for n in b["notes"]}, q)
        bs.update(calls=calls_before, tokens=b["usage"]["input_tokens"] + b["usage"]["output_tokens"], seconds=b["seconds"])
        rows.append({"id": q["id"], "type": q["type"], "team": ts, "baseline": bs,
                     "team_answer": t["draft"], "baseline_answer": b["draft"]})
        time.sleep(pause)
    return {"provider": team_llm.name, "model": getattr(team_llm, "model", ""), "rows": rows, "summary": summarise(rows)}


def summarise(rows: list[dict]) -> dict:
    out = {}
    for sysname in ("team", "baseline"):
        def avg(k):
            v = [r[sysname][k] for r in rows if r[sysname][k] is not None]
            return round(sum(v) / len(v), 3) if v else None
        out[sysname] = {k: avg(k) for k in ("fact_recall", "cited", "supported", "calls", "tokens", "seconds")}
        ab = [r[sysname]["abstained"] for r in rows if r[sysname]["abstained"] is not None]
        out[sysname]["abstained"] = f"{sum(ab)}/{len(ab)}" if ab else None
    return out


def to_markdown(ev: dict) -> str:
    s = ev["summary"]
    L = [f"# Evaluation: team vs single call", "", f"Provider: {ev['provider']} {ev['model']}", "",
         "| Metric | Team (multi-agent) | Single call |", "|---|---|---|"]
    names = {"fact_recall": "Key-fact recall", "cited": "Sentences with valid citations", "supported": "Sentences judged supported",
             "abstained": "Unanswerable questions handled honestly", "calls": "LLM calls per question",
             "tokens": "Tokens per question", "seconds": "Seconds per question"}
    for k, label in names.items():
        L.append(f"| {label} | {s['team'][k]} | {s['baseline'][k]} |")
    L += ["", "## Per question", "", "| Question | Type | Team recall | Single recall | Team supported | Single supported |", "|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    for r in ev["rows"]:
        tr = r["team"]["fact_recall"] if r["team"]["fact_recall"] is not None else r["team"]["abstained"]
        br = r["baseline"]["fact_recall"] if r["baseline"]["fact_recall"] is not None else r["baseline"]["abstained"]
        L.append(f"| {r['id']} | {r['type']} | {f(tr)} | {f(br)} | {f(r['team']['supported'])} | {f(r['baseline']['supported'])} |")
    L += ["", "For unanswerable questions the recall columns show True if the system honestly said it could not answer.",
          "The 'supported' metric is judged by the same model that wrote the answers, so treat it as indicative."]
    return "\n".join(L) + "\n"
