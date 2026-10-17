"""Streamlit app with a live view of the agents:  streamlit run app.py"""
import os
import tempfile
from pathlib import Path

import streamlit as st

from research_team.__main__ import _load_env
from research_team.llm import get_llm
from research_team.orchestrator import run_team
from research_team.report import number_sources, to_html, to_markdown, verification
from research_team.sources import LocalCorpus, WebSearch

_load_env()
ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Multi-Agent Research Team", layout="wide")

SAMPLES = ["Which Lumora model had the highest return rate in 2025, and what caused it?",
           "How did Lumora's FY2025 revenue and units sold turn out, and were any figures later corrected?",
           "Why are customers unhappy with service, and what is Lumora doing about it?",
           "What is Lumora's market share in the Indian e-bike market?"]

COL = {"wait": ("#e9ecef", "#868e96"), "start": ("#fff3bf", "#e67700"), "done": ("#d3f9d8", "#2b8a3e"),
       "warn": ("#ffe8cc", "#d9480f"), "error": ("#ffe3e3", "#c92a2a")}


def graph(state: dict, subs: list) -> str:
    def node(key, label):
        fill, line = COL[state.get(key, "wait")]
        return f'"{key}" [label="{label}", style="filled,rounded", shape=box, fillcolor="{fill}", color="{line}", fontname="Helvetica", penwidth=2];'
    L = ["digraph G { rankdir=LR; bgcolor=transparent; nodesep=0.35; ranksep=0.45; node [fontsize=18, margin=\"0.25,0.12\"]; edge [color=\"#868e96\", penwidth=1.5];",
         node("question", "Question"), node("planner", "Planner"), node("writer", "Writer"),
         node("critic", "Critic"), node("report", "Report")]
    for s in subs or [{"id": "S?", "question": "sub-questions"}]:
        k = f"researcher:{s['id']}"
        L.append(node(k, f"Researcher {s['id']}\\n{s['question'][:34]}{'...' if len(s['question']) > 34 else ''}"))
        L += [f'"planner" -> "{k}";', f'"{k}" -> "writer";']
    L += ['"question" -> "planner";', '"writer" -> "critic";', '"critic" -> "writer" [style=dashed, label="revise", fontsize=10];',
          '"critic" -> "report";', "}"]
    return "\n".join(L)


with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Model provider", ["gemini", "claude", "ollama", "offline"],
                            index=0 if (os.environ.get("GEMINI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY")) else 1,
                            help="offline = no AI, copies sentences; only for trying the interface")
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        key = st.text_input("Gemini API key", type="password", help="Free at aistudio.google.com/apikey. Kept only in this session.")
        if key:
            os.environ["GEMINI_API_KEY"] = key.strip()
    source = st.radio("Sources", ["Sample documents (fictional company)", "Upload my documents", "Web search"])
    uploads = st.file_uploader("Documents (.md, .txt, .pdf)", type=["md", "txt", "pdf"], accept_multiple_files=True) \
        if source == "Upload my documents" else None
    revise = st.checkbox("Let the Critic send the draft back once", value=True)

st.title("Multi-Agent Research Team")
st.caption("A Planner splits the question, Researchers gather quotes, a Writer drafts with citations, "
           "and a Critic fact-checks. Code drops any quote that is not really in its source.")
pick = st.selectbox("Example questions", ["(type your own below)"] + SAMPLES)
question = st.text_input("Question", value="" if pick.startswith("(") else pick)
go = st.button("Run the team", type="primary")

if go:
    if not question.strip():
        st.info("Type a question or pick an example."); st.stop()
    try:
        llm = get_llm(provider)
    except Exception as e:                                         # noqa: BLE001
        st.error(str(e)); st.stop()
    if source.startswith("Web"):
        search = WebSearch()
    elif source.startswith("Upload"):
        if not uploads:
            st.info("Upload at least one document."); st.stop()
        tmp = Path(tempfile.mkdtemp())
        for f in uploads:
            if f.name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                import io
                text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(f.getvalue())).pages)
                (tmp / (Path(f.name).stem + ".txt")).write_text(f"# {Path(f.name).stem}\n{text}", encoding="utf-8")
            else:
                (tmp / f.name).write_bytes(f.getvalue())
        try:
            search = LocalCorpus(tmp)
        except ValueError as e:
            st.error(str(e)); st.stop()
    else:
        search = LocalCorpus(ROOT / "corpus")

    left, right = st.columns([3, 2])
    graph_box = left.empty()
    feed_box = right.container(height=360)
    state, subs, result = {"question": "done"}, [], None
    graph_box.graphviz_chart(graph(state, subs))
    for ev in run_team(question, llm, search, revise=revise):
        if ev.agent == "planner" and ev.status == "done":
            subs = ev.data["subs"]
        if ev.agent == "team":
            if ev.status == "error":
                st.error(ev.text); st.stop()
            result = ev.data["result"]
            state["report"] = "done"
        else:
            if not (ev.status == "warn" and state.get(ev.agent) == "done"):
                state[ev.agent] = "warn" if ev.status == "warn" else ev.status
        icon = {"start": "⏳", "done": "✅", "warn": "⚠️", "error": "❌", "info": "•"}[ev.status]
        feed_box.write(f"{icon} **{ev.agent}**: {ev.text}")
        graph_box.graphviz_chart(graph(state, subs))
    st.session_state["result"] = result

result = st.session_state.get("result")
if result:
    v = verification(result)
    u = result["usage"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentences fully supported", f"{v['supported']}/{v['sentences']}")
    c2.metric("Notes dropped by code checks", v["dropped_notes"])
    c3.metric("Model calls", u["calls"])
    c4.metric("Time", f"{result['seconds']} s")
    t1, t2, t3 = st.tabs(["Report", "Sources and quotes", "Agent trace"])
    with t1:
        st.markdown(to_markdown(result).split("## Verification")[0].replace("# ", "### ", 1))
        st.download_button("Download HTML report", to_html(result), "research_brief.html", "text/html")
        st.download_button("Download Markdown", to_markdown(result), "research_brief.md")
    with t2:
        _, srcs = number_sources(result)
        for s in srcs:
            st.markdown(f"**[{s['n']}] {s['title']}**  \n`{s['url']}`")
            for q in s["quotes"]:
                st.caption(f"“{q}”")
        if result["dropped_notes"]:
            st.subheader("Dropped notes (quote not found in source)")
            st.json(result["dropped_notes"])
    with t3:
        st.write("**Plan**"); st.json(result["subs"])
        st.write("**Notes**"); st.json(result["notes"])
        st.write("**Critic review**"); st.json(result["review"])
