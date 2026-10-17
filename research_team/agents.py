"""The four agents. Each one is a prompt, a JSON schema, and code that checks the result."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLM
from .sources import Passage
from .verify import quote_in_source

# ---------------------------------------------------------------- Planner
PLAN_SCHEMA = {"type": "object", "properties": {"sub_questions": {"type": "array", "items": {
    "type": "object", "properties": {"question": {"type": "string"}, "search_query": {"type": "string"}},
    "required": ["question", "search_query"]}}}, "required": ["sub_questions"]}

PLAN_SYSTEM = """You are the Planner in a research team. Break the user's question into 2 to 4 focused
sub-questions that together answer it completely. Each needs a short keyword search query.
Do not answer the question. Do not add sub-questions the user did not ask about."""


def plan(llm: LLM, question: str, max_sub: int = 4) -> list[dict]:
    out = llm.json("planner", PLAN_SYSTEM, f"Question: {question}", PLAN_SCHEMA, {"question": question})
    subs = [s for s in out.get("sub_questions", []) if s.get("question", "").strip()][:max_sub]
    if not subs:                                  # never leave the team with nothing to do
        subs = [{"question": question, "search_query": question}]
    for n, s in enumerate(subs, 1):
        s["id"] = f"S{n}"
        s["search_query"] = s.get("search_query") or s["question"]
    return subs


# ---------------------------------------------------------------- Researcher
NOTES_SCHEMA = {"type": "object", "properties": {
    "notes": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "quote": {"type": "string"}, "source_id": {"type": "string"}},
        "required": ["claim", "quote", "source_id"]}},
    "gaps": {"type": "array", "items": {"type": "string"}}}, "required": ["notes", "gaps"]}

RESEARCH_SYSTEM = """You are a Researcher. Read the sources and write notes that answer the sub-question.
Write 2 to 6 notes. Prefer specific facts: numbers, percentages, dates, names, and root causes
(why something happened, not only what happened). Include related context that a reader would need.
For every note give: a short claim in your own words, the exact sentence or list item from the source that
supports it (copy it character for character as the quote, one sentence only), and the source_id.
Use only the sources given. If the sources do not answer part of the sub-question, list that under gaps."""


@dataclass
class ResearchResult:
    sub: dict
    passages: list[Passage]
    notes: list[dict]
    dropped: list[dict]
    gaps: list[str]


def research(llm: LLM, sub: dict, search, k: int = 4) -> ResearchResult:
    passages = search.search(sub["search_query"], k=k)
    if not passages:
        return ResearchResult(sub, [], [], [], [f"No sources found for: {sub['question']}"])
    src = "\n\n".join(f"[source_id: {p.source_id}] {p.title}\n{p.text}" for p in passages)
    payload = {"sub": sub, "passages": [p.__dict__ for p in passages]}
    out = llm.json("researcher", RESEARCH_SYSTEM, f"Sub-question: {sub['question']}\n\nSources:\n{src}",
                   NOTES_SCHEMA, payload)
    by_id = {p.source_id: p for p in passages}
    kept, dropped = [], []
    for n in out.get("notes", []):
        p = by_id.get(n.get("source_id", ""))
        if p is None:
            dropped.append({**n, "reason": "cites a source that was not provided"})
        elif not quote_in_source(n.get("quote", ""), p.text):
            dropped.append({**n, "reason": "quote not found in the source"})
        else:
            kept.append({**n, "title": p.title, "url": p.url})
    return ResearchResult(sub, passages, kept, dropped, list(out.get("gaps", [])))


# ---------------------------------------------------------------- Writer
REPORT_SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"},
    "sections": {"type": "array", "items": {"type": "object", "properties": {
        "heading": {"type": "string"},
        "sentences": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "note_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "note_ids"]}}}, "required": ["heading", "sentences"]}},
    "unanswered": {"type": "array", "items": {"type": "string"}}},
    "required": ["title", "sections", "unanswered"]}

WRITE_SYSTEM = """You are the Writer. Write a clear research brief that answers the question using ONLY the notes.
- Start with a direct answer that includes the key figures.
- Use the specific numbers, dates and names from the notes, and explain root causes when the notes give them.
- Every sentence must list the note_ids that support it, and must say no more than those notes say.
- If notes disagree (for example a figure that was later corrected), say so and give both values.
- unanswered: list only parts of the USER'S question that the notes do not answer. Leave it empty if the
  question is fully answered. Do not list follow-up questions. Never fill gaps from your own knowledge."""


def _notes_block(notes: list[dict]) -> str:
    return "\n".join(f"{n['id']}: {n['claim']} (quote: \"{n['quote']}\"; source: {n['title']})" for n in notes)


def write(llm: LLM, question: str, notes: list[dict], gaps: list[str], feedback: list[str] | None = None,
          previous: dict | None = None) -> dict:
    prompt = f"Question: {question}\n\nNotes:\n{_notes_block(notes)}\n\nResearcher gaps: {gaps or 'none'}"
    if feedback:
        prompt += ("\n\nYour previous draft:\n" + json.dumps(previous, indent=1) +
                   "\n\nThe Critic found these problems. Fix only these: correct or remove the flagged sentences, and add"
                   " anything the Critic says is missing if the notes support it. Keep every other sentence as it is:\n- "
                   + "\n- ".join(feedback))
    return llm.json("writer", WRITE_SYSTEM, prompt, REPORT_SCHEMA,
                    {"question": question, "notes": notes, "gaps": gaps, "feedback": feedback, "previous": previous})


# ---------------------------------------------------------------- Critic
CRITIC_SCHEMA = {"type": "object", "properties": {
    "verdicts": {"type": "array", "items": {"type": "object", "properties": {
        "n": {"type": "integer"}, "supported": {"type": "string", "enum": ["yes", "partly", "no"]},
        "problem": {"type": "string"}}, "required": ["n", "supported", "problem"]}},
    "missing": {"type": "array", "items": {"type": "string"}}},
    "required": ["verdicts", "missing"]}

CRITIC_SYSTEM = """You are the Critic, a strict fact-checker. For each numbered sentence, decide if the evidence
listed under it fully supports it: yes, partly, or no. A number, name or date that differs from the evidence
means no. Explain any problem in one line. Then list parts of the question the brief does not answer
(only real omissions where the brief could have answered, not things it honestly says are unknown).
Do not list follow-up questions the user did not ask."""


def flatten(draft: dict) -> list[dict]:
    return [s for sec in draft.get("sections", []) for s in sec.get("sentences", [])]


def critique(llm: LLM, question: str, draft: dict, evidence: dict[str, str]) -> dict:
    lines = []
    for i, s in enumerate(flatten(draft)):
        ev = "; ".join(f'"{evidence[n]}"' for n in s.get("note_ids", []) if n in evidence) or "(no evidence)"
        lines.append(f"{i}. {s['text']}\n   evidence: {ev}")
    prompt = f"Question: {question}\n\nSentences:\n" + "\n".join(lines)
    out = llm.json("critic", CRITIC_SYSTEM, prompt, CRITIC_SCHEMA,
                   {"question": question, "draft": draft, "evidence": evidence})
    n = len(lines)
    verdicts = {v["n"]: v for v in out.get("verdicts", []) if isinstance(v.get("n"), int) and 0 <= v["n"] < n}
    for i in range(n):                       # a sentence the critic skipped is not counted as checked
        verdicts.setdefault(i, {"n": i, "supported": "unchecked", "problem": "critic gave no verdict"})
    return {"verdicts": [verdicts[i] for i in range(n)], "missing": list(out.get("missing", []))}
