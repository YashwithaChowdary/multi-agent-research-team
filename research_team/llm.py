"""One small interface over several LLM providers.

Every agent calls `llm.json(role, system, prompt, schema)` and gets a dict back.
Providers:
  gemini   Google Gemini (default, free key from aistudio.google.com)  GEMINI_API_KEY
  claude   Anthropic Claude                                           ANTHROPIC_API_KEY
  ollama   a local model through Ollama (no key)                      OLLAMA_MODEL
  offline  no AI at all: simple extractive rules, so the app and tests run anywhere
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

GEMINI_FALLBACKS = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash-lite"]


def gemini_models() -> list[str]:
    """Read GEMINI_MODEL when the client is created (after .env is loaded), not at import time."""
    first = os.environ.get("GEMINI_MODEL")
    return [first] + [m for m in GEMINI_FALLBACKS if m != first] if first else list(GEMINI_FALLBACKS)


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    by_role: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, role: str, t: float, tin: int = 0, tout: int = 0):
        with self._lock:
            self.calls += 1
            self.seconds += t
            self.input_tokens += tin
            self.output_tokens += tout
            self.by_role[role] = self.by_role.get(role, 0) + 1

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "llm_seconds": round(self.seconds, 1), "by_role": dict(self.by_role)}


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in ``` fences or add a sentence; take the outermost object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise LLMError(f"Model did not return JSON: {text[:200]}")


class LLM:
    name = "base"

    def __init__(self):
        self.usage = Usage()

    def json(self, role: str, system: str, prompt: str, schema: dict, payload: dict | None = None) -> dict:
        t = time.perf_counter()
        out, tin, tout = self._call(role, system, prompt, schema, payload)
        self.usage.add(role, time.perf_counter() - t, tin, tout)
        return out

    def _call(self, role, system, prompt, schema, payload=None):  # -> (dict, input_tokens, output_tokens)
        raise NotImplementedError


def _retry(fn, tries: int = 5, base: float = 4.0):
    """Free tiers allow only a few requests per minute: wait and retry on 429 / 5xx."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001 - provider SDKs raise different types
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            msg = str(e)
            retryable = code in (429, 500, 502, 503, 504) or "RESOURCE_EXHAUSTED" in msg or "overloaded" in msg.lower()
            if not retryable or k == tries - 1:
                raise
            wait = base * (2 ** k)
            m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay['\"]?: ?['\"]?(\d+)", msg)
            if m:
                wait = max(wait, float(m.group(1)) + 1)
            time.sleep(min(wait, 60))


class GeminiLLM(LLM):
    name = "gemini"

    def __init__(self, model: str | None = None):
        super().__init__()
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey")
        self.client = genai.Client(api_key=key)
        self.models = [model] if model else gemini_models()
        self.model = self.models[0]

    def _call(self, role, system, prompt, schema, payload=None):
        from google.genai import types
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.2,
                                          response_mime_type="application/json", response_json_schema=schema)
        last = None
        for model in [self.model] + [m for m in self.models if m != self.model]:
            try:
                r = _retry(lambda: self.client.models.generate_content(model=model, contents=prompt, config=cfg))
                if model != self.model:
                    print(f"[llm] {self.model} is not available to this key; using {model}. "
                          f"Run 'python -m research_team models' to see the options.")
                self.model = model                   # remember the one that works
                u = getattr(r, "usage_metadata", None)
                return parse_json(r.text or ""), getattr(u, "prompt_token_count", 0) or 0, getattr(u, "candidates_token_count", 0) or 0
            except Exception as e:                   # noqa: BLE001
                last = e
                if getattr(e, "code", None) == 404 or "not found" in str(e).lower():
                    continue                         # model retired or renamed: try the next one
                raise
        raise LLMError(f"No Gemini model worked ({self.models}). Set GEMINI_MODEL to a name from AI Studio. Last error: {last}")


def list_gemini_models() -> list[str]:
    """Model names this key can call for text generation, newest first."""
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set.")
    client = genai.Client(api_key=key)          # keep a reference: the list is fetched lazily page by page
    out = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        name = (m.name or "").removeprefix("models/")
        if "gemini" in name and (not actions or "generateContent" in actions):
            out.append(name)
    return sorted(set(out), reverse=True)


class ClaudeLLM(LLM):
    name = "claude"

    def __init__(self, model: str | None = None):
        super().__init__()
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

    def _call(self, role, system, prompt, schema, payload=None):
        tool = {"name": "answer", "description": "Return the answer.", "input_schema": schema}
        r = _retry(lambda: self.client.messages.create(
            model=self.model, max_tokens=4000, system=system, tools=[tool],
            tool_choice={"type": "tool", "name": "answer"}, messages=[{"role": "user", "content": prompt}]))
        for b in r.content:
            if b.type == "tool_use":
                return dict(b.input), r.usage.input_tokens, r.usage.output_tokens
        raise LLMError("Claude returned no tool call")


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, model: str | None = None):
        super().__init__()
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    def _call(self, role, system, prompt, schema, payload=None):
        import httpx
        try:
            r = httpx.post(f"{self.url}/api/chat", timeout=300, json={
                "model": self.model, "stream": False, "format": schema, "options": {"temperature": 0.2},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama not reachable at {self.url} ({e}). Is 'ollama serve' running?") from e
        d = r.json()
        return parse_json(d["message"]["content"]), d.get("prompt_eval_count", 0), d.get("eval_count", 0)


def get_llm(provider: str = "auto", model: str | None = None) -> LLM:
    if provider == "auto":
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "claude"
        else:
            provider = "offline"
    if provider == "offline":
        from .offline import OfflineLLM
        return OfflineLLM()
    return {"gemini": GeminiLLM, "claude": ClaudeLLM, "ollama": OllamaLLM}[provider](model)
