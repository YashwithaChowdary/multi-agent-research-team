"""python -m research_team ask "question" [--web] [--provider gemini|claude|ollama|offline]
python -m research_team evaluate [--provider gemini] [--only q1_returns q9_share]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm import get_llm


def _load_env():
    """Read KEY=value lines from the project's .env file. Values in .env win over older
    Windows/user environment variables, because .env is the file you edit for this project."""
    import os
    f = Path(__file__).resolve().parent.parent / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"')


def main(argv=None) -> int:
    _load_env()
    ap = argparse.ArgumentParser(prog="research_team")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("question")
    a.add_argument("--web", action="store_true", help="search the web instead of the local corpus folder")
    a.add_argument("--corpus", default=str(Path(__file__).resolve().parent.parent / "corpus"))
    a.add_argument("--provider", default="auto"); a.add_argument("--model")
    a.add_argument("--out", help="folder for report.md / report.html / result.json")
    e = sub.add_parser("evaluate"); e.add_argument("--provider", default="auto"); e.add_argument("--model")
    e.add_argument("--only", nargs="*"); e.add_argument("--pause", type=float, default=0.0)
    sub.add_parser("models", help="list the Gemini models your key can use")
    args = ap.parse_args(argv)

    if args.cmd == "models":
        from .llm import list_gemini_models
        try:
            names = list_gemini_models()
        except Exception as ex:                                     # noqa: BLE001
            print(f"Error: {ex}", file=sys.stderr); return 1
        print("Gemini models available to your key:")
        for n in names:
            print("  " + n)
        import re as _re
        plain = [n for n in names if _re.fullmatch(r"gemini-\d+(\.\d+)?-flash", n)]   # e.g. gemini-3.8-flash
        best = max(plain, key=lambda n: [int(x) for x in _re.findall(r"\d+", n)], default=None)
        if best:
            print(f"\nSuggested: add the line  GEMINI_MODEL={best}  to .env")
        return 0

    if args.cmd == "ask":
        from .orchestrator import run_team
        from .report import to_html, to_markdown
        from .sources import LocalCorpus, WebSearch
        try:
            llm = get_llm(args.provider, args.model)
        except Exception as ex:                                     # noqa: BLE001
            print(f"Error: {ex}", file=sys.stderr); return 1
        search = WebSearch() if args.web else LocalCorpus(args.corpus)
        print(f"Using {llm.name} {getattr(llm, 'model', '')} | sources: {'web' if args.web else args.corpus}")
        if llm.name == "offline":
            print("NOTE: offline mode, no AI. It only copies matching sentences. Set GEMINI_API_KEY in .env for real answers.")
        print()
        result = None
        for ev in run_team(args.question, llm, search):
            mark = {"start": "..", "done": "OK", "warn": "!!", "error": "XX", "info": "  "}[ev.status]
            print(f"[{mark}] {ev.agent:16} {ev.text}")
            if ev.agent == "team" and ev.status == "done":
                result = ev.data["result"]
            if ev.agent == "team" and ev.status == "error":
                return 1
        print("\n" + to_markdown(result))
        if args.out:
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            (out / "report.md").write_text(to_markdown(result), encoding="utf-8")
            (out / "report.html").write_text(to_html(result), encoding="utf-8")
            (out / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
            print(f"Saved to {out}/")
    else:
        from .evaluate import evaluate, to_markdown as md
        ev = evaluate(args.provider, args.model, args.only, args.pause)
        text = md(ev)
        print("\n" + text)
        Path("reports").mkdir(exist_ok=True)
        Path("reports/evaluation.md").write_text(text, encoding="utf-8")
        Path("reports/evaluation.json").write_text(json.dumps(ev, indent=1, default=str), encoding="utf-8")
        print("Saved reports/evaluation.md and reports/evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
