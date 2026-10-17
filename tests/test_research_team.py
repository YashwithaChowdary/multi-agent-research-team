import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_team import agents, llm as llm_mod
from research_team.evaluate import abstained, evaluate, fact_recall
from research_team.llm import LLM, get_llm, parse_json
from research_team.orchestrator import run_team, run_to_end
from research_team.report import to_html, to_markdown
from research_team.sources import LocalCorpus
from research_team.verify import check_draft, quote_in_source

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


@pytest.fixture(scope="module")
def corpus():
    return LocalCorpus(CORPUS)


class ScriptedLLM(LLM):
    """A fake model that makes the mistakes real models make, to test the guard rails."""
    name = "scripted"

    def __init__(self):
        super().__init__()
        self.writer_calls = 0

    def _call(self, role, system, prompt, schema, payload=None):
        if role == "planner":
            return {"sub_questions": [{"question": "Trail return rate?", "search_query": "Trail return rate"}]}, 1, 1
        if role == "researcher":
            sid = payload["passages"][0]["source_id"]
            good = next(s for p in payload["passages"] for s in p["text"].split("\n") if "4.8%" in s)
            good_sid = next(p["source_id"] for p in payload["passages"] if good in p["text"])
            return {"notes": [
                {"claim": "Trail returns were 4.8%", "quote": good, "source_id": good_sid},
                {"claim": "invented", "quote": "The Trail won Bike of the Year 2025 in Germany.", "source_id": sid},
                {"claim": "wrong source", "quote": good, "source_id": "local:nowhere.md#0"}], "gaps": []}, 1, 1
        if role == "writer":
            self.writer_calls += 1
            if self.writer_calls == 1:
                return {"title": "t", "unanswered": [], "sections": [{"heading": "h", "sentences": [
                    {"text": "The Trail return rate was 4.8%.", "note_ids": ["N1"]},
                    {"text": "It is the best bike in Europe.", "note_ids": []}]}]}, 1, 1
            return {"title": "t", "unanswered": [], "sections": [{"heading": "h", "sentences": [
                {"text": "The Trail return rate was 4.8%.", "note_ids": ["N1"]}]}]}, 1, 1
        if role == "critic":
            n = len(agents.flatten(payload["draft"]))
            return {"verdicts": [{"n": i, "supported": "yes" if i == 0 else "no", "problem": "" if i == 0 else "no evidence"}
                                 for i in range(n)], "missing": []}, 1, 1
        raise AssertionError(role)


def test_quote_matching():
    src = "Water getting into the rear hub motor caused 61% of Trail returns. Most were built in 2024."
    assert quote_in_source("Water getting into the rear hub motor caused 61% of Trail returns.", src)
    assert quote_in_source("water getting into the rear hub motor caused 61 % of Trail returns", src)
    assert not quote_in_source("Water getting into the rear hub motor caused 91% of Trail returns.", src)
    assert not quote_in_source("The Trail won Bike of the Year 2025 in Germany.", src)
    assert not quote_in_source("Trail", src)


def test_changed_number_is_rejected():
    src = "Revenue for FY2025 was 412 crore rupees, up 38% from FY2024."
    assert not quote_in_source("Revenue for FY2025 was 512 crore rupees, up 88% from FY2024.", src)


def test_bm25_finds_right_document(corpus):
    top = corpus.search("hub motor seal supplier water", k=2)
    assert top and top[0].url == "postmortem_trail_hub_motor.md"
    assert corpus.search("zzzz qqqq") == []


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": 2} thanks') == {"a": 2}


def test_guardrails_drop_fake_quotes_and_force_revision(corpus):
    events = list(run_team("What was the Trail return rate?", ScriptedLLM(), corpus))
    result = events[-1].data["result"]
    reasons = sorted(d["reason"] for d in result["dropped_notes"])
    assert reasons == ["cites a source that was not provided", "quote not found in the source"]
    assert len(result["notes"]) == 1
    assert result["revisions"] == 1
    assert [s["text"] for s in agents.flatten(result["draft"])] == ["The Trail return rate was 4.8%."]
    assert result["remaining_problems"] == []
    assert any(e.status == "warn" for e in events)


def test_check_draft_flags_uncited_and_unknown():
    d = {"sections": [{"heading": "h", "sentences": [{"text": "a", "note_ids": []}, {"text": "b", "note_ids": ["N9"]}]}]}
    assert [i["problem"][:8] for i in check_draft(d, {"N1"})] == ["no citat", "cites un"]


def test_offline_end_to_end_and_reports(corpus):
    r = run_to_end("Which Lumora model had the highest return rate in 2025, and what caused it?", get_llm("offline"), corpus)
    md, page = to_markdown(r), to_html(r)
    assert "## Sources" in md and "[1]" in md and "<blockquote>" in page
    assert r["usage"]["calls"] >= 4


def test_retry_on_rate_limit(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            e = Exception("429 RESOURCE_EXHAUSTED retry in 1s"); e.code = 429; raise e
        return "ok"
    assert llm_mod._retry(flaky) == "ok" and calls["n"] == 3


def test_gemini_falls_back_when_model_retired(monkeypatch):
    g = llm_mod.GeminiLLM.__new__(llm_mod.GeminiLLM)
    LLM.__init__(g)
    g.models, g.model = ["old-model", "new-model"], "old-model"

    def gen(model, contents, config):
        if model == "old-model":
            e = Exception("404 model not found"); e.code = 404; raise e
        return SimpleNamespace(text='{"ok": true}', usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=2))
    g.client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
    assert g.json("planner", "s", "p", {"type": "object"}) == {"ok": True}
    assert g.model == "new-model" and g.usage.input_tokens == 3


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(llm_mod.LLMError, match="aistudio"):
        get_llm("gemini")


def test_auto_without_keys_is_offline(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert get_llm("auto").name == "offline"


def test_eval_metrics():
    d = {"sections": [{"heading": "h", "sentences": [{"text": "Trail had 4.8% returns from water", "note_ids": ["N1"]}]}], "unanswered": []}
    assert fact_recall(d, [["trail"], ["4.8%"], ["seal"]]) == pytest.approx(2 / 3)
    invented = {"sections": [{"heading": "h", "sentences": [{"text": "Lumora has 12% market share.", "note_ids": []}]}], "unanswered": ["x"]}
    assert not abstained(invented)
    assert abstained({"sections": [], "unanswered": ["market share not in sources"]})


def test_evaluation_runs_offline():
    ev = evaluate("offline", only=["q1_returns", "q9_share"], log=lambda m: None)
    assert len(ev["rows"]) == 2 and set(ev["summary"]) == {"team", "baseline"}


def test_questions_file_is_valid():
    qs = json.loads((CORPUS.parent / "eval" / "questions.json").read_text())["questions"]
    assert len(qs) == 10 and sum(not q["answerable"] for q in qs) == 2
    text = " ".join(p.read_text(encoding="utf-8").lower().replace(",", "") for p in CORPUS.glob("*.md"))
    for q in qs:
        for alts in q["facts"]:
            assert any(a.lower().replace(",", "") in text for a in alts), (q["id"], alts)


def test_env_file_overrides_old_variables(tmp_path, monkeypatch):
    from research_team import __main__ as m
    monkeypatch.setenv("GEMINI_MODEL", "gemini-old-lite")
    env = tmp_path / ".env"
    env.write_text("# comment\nGEMINI_MODEL=gemini-3.8-flash\n")
    monkeypatch.setattr(m, "Path", lambda *a: type("P", (), {"resolve": lambda self: type("Q", (), {"parent": type("R", (), {"parent": tmp_path})()})()})())
    m._load_env()
    import os
    assert os.environ["GEMINI_MODEL"] == "gemini-3.8-flash"
    assert llm_mod.gemini_models()[0] == "gemini-3.8-flash"
