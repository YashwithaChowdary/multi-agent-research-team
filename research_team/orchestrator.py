"""Runs the team and reports every step as an event, so the UI can draw it live.

  Planner -> Researchers (one per sub-question, in parallel) -> Writer -> Critic
                                                     ^-- revise once --'
Code checks sit between the agents: notes with quotes that are not in the source
are dropped, and sentences without valid citations are sent back to the Writer.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from . import agents
from .llm import LLM
from .verify import check_draft


@dataclass
class Event:
    agent: str            # planner | researcher:S1 | writer | critic | team
    status: str           # start | done | info | warn | error
    text: str
    data: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)


def run_team(question: str, llm: LLM, search, max_sub: int = 4, parallel: int = 2, revise: bool = True,
             passages_per_query: int = 4):
    """Generator of Events. The last event has agent='team', status='done' and the full result in data."""
    t0 = time.perf_counter()
    yield Event("planner", "start", "Breaking the question into sub-questions")
    try:
        subs = agents.plan(llm, question, max_sub)
    except Exception as e:                                          # noqa: BLE001
        yield Event("planner", "error", str(e)); yield Event("team", "error", f"Planner failed: {e}"); return
    yield Event("planner", "done", f"{len(subs)} sub-questions", {"subs": subs})

    results: dict[str, agents.ResearchResult] = {}
    for s in subs:
        yield Event(f"researcher:{s['id']}", "start", s["question"], {"sub": s})
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futs = {pool.submit(agents.research, llm, s, search, passages_per_query): s for s in subs}
        for f in as_completed(futs):
            s = futs[f]
            try:
                r = f.result()
            except Exception as e:                                  # noqa: BLE001
                yield Event(f"researcher:{s['id']}", "error", str(e))
                results[s["id"]] = agents.ResearchResult(s, [], [], [], [f"Research failed: {e}"])
                continue
            results[s["id"]] = r
            for d in r.dropped:
                yield Event(f"researcher:{s['id']}", "warn", f"Dropped a note: {d['reason']}", {"note": d})
            yield Event(f"researcher:{s['id']}", "done",
                        f"{len(r.notes)} verified notes from {len({p.source_id for p in r.passages})} sources",
                        {"notes": r.notes, "gaps": r.gaps})

    notes, gaps = [], []
    for s in subs:
        r = results[s["id"]]
        for n in r.notes:
            notes.append({**n, "id": f"N{len(notes) + 1}", "sub": s["id"]})
        gaps += r.gaps
    evidence = {n["id"]: n["quote"] for n in notes}

    yield Event("writer", "start", f"Writing from {len(notes)} notes")
    draft = agents.write(llm, question, notes, gaps)
    issues = check_draft(draft, set(evidence))
    yield Event("writer", "done", f"Draft with {len(agents.flatten(draft))} sentences", {"draft": draft, "code_issues": issues})

    yield Event("critic", "start", "Fact-checking every sentence against its evidence")
    review = agents.critique(llm, question, draft, evidence)
    bad = _problems(draft, review, issues)
    yield Event("critic", "done", f"{len(bad)} problems found", {"review": review, "problems": bad})

    rounds = 0
    if bad and revise:
        rounds = 1
        yield Event("writer", "start", f"Revising: {len(bad)} problems to fix", {"feedback": bad})
        draft = agents.write(llm, question, notes, gaps, feedback=bad, previous=draft)
        issues = check_draft(draft, set(evidence))
        yield Event("writer", "done", f"Revised draft with {len(agents.flatten(draft))} sentences", {"draft": draft})
        yield Event("critic", "start", "Checking the revision")
        review = agents.critique(llm, question, draft, evidence)
        bad = _problems(draft, review, issues)
        yield Event("critic", "done", f"{len(bad)} problems remain", {"review": review, "problems": bad})

    result = {"question": question, "subs": subs, "notes": notes, "gaps": gaps, "draft": draft,
              "review": review, "code_issues": issues, "remaining_problems": bad, "revisions": rounds,
              "dropped_notes": [d for r in results.values() for d in r.dropped],
              "sources_searched": sorted({p.source_id for r in results.values() for p in r.passages}),
              "usage": llm.usage.as_dict(), "seconds": round(time.perf_counter() - t0, 1)}
    yield Event("team", "done", "Report ready", {"result": result})


def _problems(draft: dict, review: dict, code_issues: list[dict]) -> list[str]:
    sents = agents.flatten(draft)
    out = []
    for v in review["verdicts"]:
        if v["supported"] in ("no", "partly"):
            out.append(f"Sentence {v['n']} ('{sents[v['n']]['text'][:90]}') is {v['supported']} supported: {v['problem']}")
    flat_index = {}
    k = 0
    for si, sec in enumerate(draft.get("sections", [])):
        for ti, _ in enumerate(sec.get("sentences", [])):
            flat_index[(si, ti)] = k; k += 1
    for i in code_issues:
        out.append(f"Sentence {flat_index.get((i['section'], i['sentence']))}: {i['problem']}")
    out += [f"Not answered but could be: {m}" for m in review.get("missing", [])]
    return out


def run_to_end(question: str, llm: LLM, search, **kw) -> dict:
    events = list(run_team(question, llm, search, **kw))
    last = events[-1]
    if last.status != "done":
        raise RuntimeError(last.text)
    return {**last.data["result"], "events": [(e.agent, e.status, e.text) for e in events]}
