"""Turn a team result into a readable brief with numbered sources."""
from __future__ import annotations

import html

from .agents import flatten


def number_sources(result: dict) -> tuple[dict[str, int], list[dict]]:
    """Number sources in the order the report first cites them."""
    notes = {n["id"]: n for n in result["notes"]}
    order: dict[str, int] = {}
    srcs: list[dict] = []
    for s in flatten(result["draft"]):
        for nid in s.get("note_ids", []):
            n = notes.get(nid)
            if n and n["source_id"] not in order:
                order[n["source_id"]] = len(order) + 1
                srcs.append({"n": order[n["source_id"]], "title": n["title"], "url": n["url"],
                             "source_id": n["source_id"], "quotes": []})
            if n:
                q = srcs[order[n["source_id"]] - 1]["quotes"]
                if n["quote"] not in q:
                    q.append(n["quote"])
    return order, srcs


def _cites(sentence: dict, notes: dict, order: dict) -> list[int]:
    return sorted({order[notes[i]["source_id"]] for i in sentence.get("note_ids", []) if i in notes})


def verification(result: dict) -> dict:
    v = result["review"]["verdicts"]
    return {"sentences": len(v), "supported": sum(x["supported"] == "yes" for x in v),
            "partly": sum(x["supported"] == "partly" for x in v), "unsupported": sum(x["supported"] == "no" for x in v),
            "dropped_notes": len(result.get("dropped_notes", [])), "revisions": result.get("revisions", 0)}


def to_markdown(result: dict) -> str:
    notes = {n["id"]: n for n in result["notes"]}
    order, srcs = number_sources(result)
    flags = {x["n"]: x for x in result["review"]["verdicts"]}
    d = result["draft"]
    L = [f"# {d.get('title') or result['question']}", "", f"_Question: {result['question']}_", ""]
    k = 0
    for sec in d.get("sections", []):
        L += [f"## {sec['heading']}", ""]
        para = []
        for s in sec["sentences"]:
            c = "".join(f"[{n}]" for n in _cites(s, notes, order))
            warn = " ⚠" if flags.get(k, {}).get("supported") in ("no", "partly") else ""
            para.append(f"{s['text']} {c}{warn}".strip())
            k += 1
        L += [" ".join(para), ""]
    if d.get("unanswered"):
        L += ["## What the sources do not say", ""] + [f"- {u}" for u in d["unanswered"]] + [""]
    v = verification(result)
    L += ["## Verification", "",
          f"{v['supported']} of {v['sentences']} sentences judged fully supported by the Critic"
          f" ({v['partly']} partly, {v['unsupported']} not; marked ⚠). {v['dropped_notes']} research notes were dropped"
          f" because their quotes were not in the source. Revisions: {v['revisions']}.", ""]
    missing = [p for p in result.get("remaining_problems", []) if p.startswith("Not answered")]
    if missing:
        L += ["Critic notes still open:"] + [f"- {m}" for m in missing] + [""]
    L += ["## Sources", ""]
    for s in srcs:
        L.append(f"{s['n']}. **{s['title']}** ({s['url']})")
        L += [f"   > {q}" for q in s["quotes"][:3]]
    u = result["usage"]
    L += ["", f"_{u['calls']} model calls, {u['input_tokens'] + u['output_tokens']} tokens, {result['seconds']} s._"]
    return "\n".join(L) + "\n"


def to_html(result: dict) -> str:
    """Self-contained page: hover a citation to see the exact quote."""
    e = html.escape
    notes = {n["id"]: n for n in result["notes"]}
    order, srcs = number_sources(result)
    flags = {x["n"]: x for x in result["review"]["verdicts"]}
    d = result["draft"]
    body, k = [], 0
    for sec in d.get("sections", []):
        body.append(f"<h2>{e(sec['heading'])}</h2><p>")
        for s in sec["sentences"]:
            cites = "".join(
                f'<a class="c" href="#s{order[notes[i]["source_id"]]}" title="{e(notes[i]["quote"])}">[{order[notes[i]["source_id"]]}]</a>'
                for i in s.get("note_ids", []) if i in notes)
            bad = flags.get(k, {}).get("supported") in ("no", "partly")
            tip = e(flags.get(k, {}).get("problem", ""))
            body.append(f'<span class="{"bad" if bad else ""}" title="{tip}">{e(s["text"])}</span>{cites} ')
            k += 1
        body.append("</p>")
    if d.get("unanswered"):
        body.append("<h2>What the sources do not say</h2><ul>" + "".join(f"<li>{e(u)}</li>" for u in d["unanswered"]) + "</ul>")
    v = verification(result)
    src = "".join(f'<li id="s{s["n"]}"><b>{e(s["title"])}</b> <span class="m">{e(s["url"])}</span>'
                  + "".join(f"<blockquote>{e(q)}</blockquote>" for q in s["quotes"][:3]) + "</li>" for s in srcs)
    css = """:root{--bg:#fbfaf7;--fg:#1f2328;--m:#5b6370;--line:#e3e0d8;--a:#3b5bdb;--bad:#c2410c}
@media(prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e6e6e6;--m:#9aa3ad;--line:#2d3137;--a:#8ea4ff;--bad:#fb923c}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 Georgia,serif}main{max-width:760px;margin:0 auto;padding:32px 16px}
h1{font:600 26px/1.3 system-ui,sans-serif}h2{font:600 18px system-ui,sans-serif;margin-top:28px}
.c{color:var(--a);text-decoration:none;font:600 12px system-ui;vertical-align:super;margin-left:1px}
.bad{text-decoration:underline wavy var(--bad)}.m{color:var(--m);font:13px system-ui}
blockquote{margin:4px 0 8px;padding-left:10px;border-left:3px solid var(--line);color:var(--m);font-size:14px}
.v{font:14px system-ui;color:var(--m);border-top:1px solid var(--line);padding-top:12px;margin-top:28px}"""
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Research brief</title><style>{css}</style></head><body><main><h1>{e(d.get('title') or result['question'])}</h1>"
            f'<div class="m">Question: {e(result["question"])}</div>{"".join(body)}'
            f'<div class="v">Verification: {v["supported"]}/{v["sentences"]} sentences fully supported; '
            f'{v["dropped_notes"]} notes dropped for quotes not found in sources; {v["revisions"]} revision(s).</div>'
            f"<h2>Sources</h2><ol>{src}</ol></main></body></html>")
