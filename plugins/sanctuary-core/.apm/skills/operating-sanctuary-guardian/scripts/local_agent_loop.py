#!/usr/bin/env python3
"""Autonomous multi-turn local agent loop on the sovereign MLX server (MISSION-022).

Loop: scratchpad plan -> one read-only tool call -> observation -> ... -> <final_answer>.
Everything runs offline against the local OpenAI-compatible MLX endpoint.

Guards (all counted in the result so callers can tell model behaviour from scaffolding):
  * hard turn ceiling (HARD_MAX_TURNS = 5)
  * repetition: an identical (tool, args) call is NOT re-executed (counted as an
    intervention); the run is falsified when one call is issued more than
    REPEAT_FALSIFY_LIMIT (4) times
  * final-turn nudge: the last allowed turn is announced as final (counted)
  * context budget: run aborts if the prompt exceeds ctx_budget tokens

Usage (via forge_llm.py):
    python3 forge_llm.py loop "Which files are in context/?" --max-turns 5
"""
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from forge_llm import parse_tool_calls_from_text, DEFAULT_MLX_HOST, DEFAULT_MLX_MODEL  # noqa: E402

HARD_MAX_TURNS = 5
REPEAT_FALSIFY_LIMIT = 4
MAX_OBS_CHARS = 6000
DEFAULT_CTX_BUDGET = 8192
IGNORED_NAMES = {".git", "__pycache__", ".DS_Store"}
GLOB_CHARS = set("*?[]")

SYSTEM_PROMPT = """You are an offline repository inspection agent. Work in turns. Each turn, write a one-line plan in <scratchpad>...</scratchpad>, then EITHER call exactly one tool OR give the final answer.

Tools (read-only; paths are relative to the repository root, '.' is the root):
- list_dir(path): entries of ONE directory; directories end with '/'. Takes a plain path, never a pattern.
- read_file(path): text content of one file.
- find_files(pattern): glob pattern relative to the root, e.g. 'LEARNING/**/*.json'. Case-sensitive.
- search_text(query, path): case-insensitive text search; returns 'file:line: text'. path defaults to '.'.

Example turn that calls a tool (the call MUST be inside <tool_call> tags with JSON):
<scratchpad>I need the entries of context/, so I list that directory.</scratchpad>
<tool_call>{"name": "list_dir", "arguments": {"path": "context"}}</tool_call>

Example turn that finishes:
<scratchpad>The observation contains the answer.</scratchpad>
<final_answer>soul.md, user.md</final_answer>

Rules: one tool call per turn; never repeat a call whose result you already have; as soon as the observations contain the answer, give the final answer containing only the requested values."""


class ReadOnlySandbox:
    """Read-only tools confined to ``root`` with an optional deny list of relative prefixes."""

    def __init__(self, root: Path, deny: Optional[List[str]] = None, max_chars: int = MAX_OBS_CHARS):
        self.root = Path(root).resolve()
        self.deny = [(self.root / d).resolve() for d in (deny or [])]
        self.max_chars = max_chars

    # -- path handling ------------------------------------------------------------
    def _denied(self, p: Path) -> bool:
        return any(p == d or d in p.parents for d in self.deny)

    def resolve(self, arg) -> Path:
        if not isinstance(arg, str):
            raise ValueError("path must be a string")
        arg = arg.strip()
        if arg in ("", ".", "/", "./"):
            return self.root
        cand = Path(arg)
        if cand.is_absolute():
            try:  # absolute paths are fine only if already inside the root
                cand.resolve().relative_to(self.root)
            except ValueError:
                cand = Path(arg.lstrip("/"))  # treat "/foo" as repo-relative "foo"
        p = (cand if cand.is_absolute() else self.root / cand).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError("path escapes repository root")
        if self._denied(p):
            raise PermissionError("path is not accessible")
        return p

    def _rel(self, p: Path) -> str:
        return "." if p == self.root else str(p.relative_to(self.root)) + ("/" if p.is_dir() else "")

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + f"\n[truncated: showing {self.max_chars} of {len(text)} chars]"

    # -- tools --------------------------------------------------------------------
    def list_dir(self, path: str = ".") -> str:
        if isinstance(path, str) and GLOB_CHARS & set(path):
            return "ERROR: list_dir takes a plain directory path, not a pattern. Use find_files for patterns."
        p = self.resolve(path)
        if not p.is_dir():
            return f"ERROR: not a directory: {path}"
        entries = [e for e in sorted(p.iterdir(), key=lambda e: e.name)
                   if e.name not in IGNORED_NAMES and not self._denied(e.resolve())]
        lines = [e.name + ("/" if e.is_dir() else "") for e in entries[:200]]
        if len(entries) > 200:
            lines.append(f"... ({len(entries) - 200} more)")
        return "\n".join(lines) if lines else "(empty directory)"

    def read_file(self, path: str) -> str:
        p = self.resolve(path)
        if not p.is_file():
            return f"ERROR: not a file: {path}"
        return self._clip(p.read_text(errors="replace"))

    def find_files(self, pattern: str) -> str:
        if not isinstance(pattern, str) or not pattern.strip():
            return "ERROR: pattern must be a non-empty glob"
        pattern = pattern.strip().lstrip("/")
        try:
            hits = [h for h in self.root.glob(pattern)
                    if not (IGNORED_NAMES & set(h.relative_to(self.root).parts)) and not self._denied(h.resolve())]
        except (ValueError, NotImplementedError) as e:
            return f"ERROR: bad pattern: {e}"
        hits = sorted(hits)
        out = [self._rel(h) for h in hits[:100]]
        if len(hits) > 100:
            out.append(f"... ({len(hits) - 100} more)")
        return "\n".join(out) if out else "(no matches)"

    def search_text(self, query: str, path: str = ".") -> str:
        if not isinstance(query, str) or not query.strip():
            return "ERROR: query must be non-empty"
        if isinstance(path, str) and GLOB_CHARS & set(path):
            return "ERROR: path must be a plain file or directory, not a pattern. Use find_files for patterns."
        base = self.resolve(path or ".")
        if not base.exists():
            return f"ERROR: path not found: {path}"
        needle = query.lower()
        files = [base] if base.is_file() else sorted(f for f in base.rglob("*") if f.is_file())
        out, scanned = [], 0
        for f in files:
            if IGNORED_NAMES & set(f.relative_to(self.root).parts) or self._denied(f.resolve()):
                continue
            scanned += 1
            if scanned > 5000 or len(out) >= 30:
                break
            try:
                if f.stat().st_size > 512_000:
                    continue
                for n, line in enumerate(f.read_text(errors="strict").splitlines(), 1):
                    if needle in line.lower():
                        out.append(f"{self._rel(f)}:{n}: {line.strip()[:200]}")
                        if len(out) >= 30:
                            break
            except (UnicodeDecodeError, OSError):
                continue
        return self._clip("\n".join(out)) if out else "(no matches)"

    def run(self, name: str, args: dict) -> str:
        fn = {"list_dir": self.list_dir, "read_file": self.read_file,
              "find_files": self.find_files, "search_text": self.search_text}.get(name)
        if fn is None:
            return f"ERROR: unknown tool '{name}'. Available: list_dir, read_file, find_files, search_text"
        try:
            return fn(**args)
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except (ValueError, PermissionError, OSError) as e:
            return f"ERROR: {e}"


def mlx_chat(messages: List[dict], host: str = DEFAULT_MLX_HOST, model: str = DEFAULT_MLX_MODEL,
             temperature: float = 0.0, max_tokens: int = 500, timeout: float = 180.0) -> dict:
    """One chat completion against the local MLX server. Returns {content, prompt_tokens}."""
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(f"{host.rstrip('/')}/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json", "User-Agent": "Sanctuary-Guardian"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return {"content": body["choices"][0]["message"]["content"],
            "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens")}


FINAL_RE = re.compile(r"<final_answer>([\s\S]*?)(?:</final_answer>|$)", re.IGNORECASE)
SCRATCH_RE = re.compile(r"<scratchpad>([\s\S]*?)</scratchpad>", re.IGNORECASE)


TAG_TOOL_ARG = {"list_dir": "path", "read_file": "path", "find_files": "pattern", "search_text": "query"}
TAG_CALL_RE = re.compile(r"<(list_dir|read_file|find_files|search_text)>\s*([\s\S]*?)\s*</\1>", re.IGNORECASE)


def _tag_call(text: str) -> List[dict]:
    """Accept the tag-per-tool form some models prefer: <search_text>query</search_text> (or a JSON object of arguments inside the tag)."""
    m = TAG_CALL_RE.search(text)
    if not m:
        return []
    name, body = m.group(1).lower(), m.group(2).strip()
    if body.startswith("{"):
        try:
            args = json.loads(body)
        except json.JSONDecodeError:
            return []
        return [{"name": name, "arguments": args}] if isinstance(args, dict) else []
    return [{"name": name, "arguments": {TAG_TOOL_ARG[name]: body}}] if body else []


def extract_calls(text: str) -> List[dict]:
    """Tool calls from a reply. Falls back to scanning for bare JSON objects because the MLX
    server strips the special ``<tool_call>`` token, leaving e.g. '<{"name": ..., "arguments": {...}}';
    then to the tag-per-tool form `<search_text>query</search_text>`."""
    calls = parse_tool_calls_from_text(text)
    if calls:
        return calls
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments", {}), dict):
            return [{"name": obj["name"], "arguments": obj.get("arguments", {})}]
    return _tag_call(text)


@dataclass
class AgentResult:
    status: str                      # final | turn_limit | repetition_falsified | context_exceeded | error
    final_answer: Optional[str]
    turns: int
    tool_calls: int = 0
    repeat_interventions: int = 0    # identical call suppressed by the harness
    malformed_replies: int = 0       # no parsable tool call / final answer
    final_turn_nudged: bool = False  # last turn was announced as final
    final_on_nudged_turn: bool = False
    max_identical_calls: int = 0
    max_prompt_tokens: int = 0
    seconds: float = 0.0
    trace: List[dict] = field(default_factory=list)

    @property
    def unassisted(self) -> bool:
        """Finished with no harness intervention at all."""
        return (self.status == "final" and self.repeat_interventions == 0
                and self.malformed_replies == 0 and not self.final_on_nudged_turn)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unassisted"] = self.unassisted
        return d


def run_agent(task: str, sandbox: ReadOnlySandbox, chat_fn: Optional[Callable[[List[dict]], dict]] = None,
              max_turns: int = HARD_MAX_TURNS, ctx_budget: int = DEFAULT_CTX_BUDGET,
              system_prompt: str = SYSTEM_PROMPT) -> AgentResult:
    chat_fn = chat_fn or mlx_chat
    max_turns = max(1, min(int(max_turns), HARD_MAX_TURNS))
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
    counts: Counter = Counter()
    res = AgentResult(status="turn_limit", final_answer=None, turns=0)
    t0 = time.time()

    for turn in range(1, max_turns + 1):
        res.turns = turn
        try:
            reply = chat_fn(messages)
        except Exception as e:  # server down, timeout, ...
            res.status, res.trace = "error", res.trace + [{"turn": turn, "error": str(e)}]
            break
        text = reply["content"]
        res.max_prompt_tokens = max(res.max_prompt_tokens, reply.get("prompt_tokens") or 0)
        step = {"turn": turn, "reply": text[:600]}
        res.trace.append(step)
        sp = SCRATCH_RE.search(text)
        if sp:
            step["scratchpad"] = sp.group(1).strip()[:300]

        if res.max_prompt_tokens > ctx_budget:
            res.status = "context_exceeded"
            break

        fin = FINAL_RE.search(text)
        calls = extract_calls(text)
        if fin and not calls:
            res.status, res.final_answer = "final", fin.group(1).strip()
            res.final_on_nudged_turn = res.final_turn_nudged and turn == max_turns
            break

        messages.append({"role": "assistant", "content": text})
        if not calls:
            res.malformed_replies += 1
            obs = ("ERROR: reply with exactly one <tool_call>{...}</tool_call> or a <final_answer>...</final_answer>. Example: "
                   '<tool_call>{"name": "search_text", "arguments": {"query": "MISSION-021", "path": "LEARNING/research/missions"}}</tool_call>')
        else:
            name, args = calls[0]["name"], calls[0]["arguments"]
            key = (name, json.dumps(args, sort_keys=True))
            counts[key] += 1
            res.max_identical_calls = max(res.max_identical_calls, counts[key])
            step["call"] = {"name": name, "arguments": args}
            if counts[key] > REPEAT_FALSIFY_LIMIT:
                res.status = "repetition_falsified"
                break
            res.tool_calls += 1
            if counts[key] > 1:
                res.repeat_interventions += 1
                obs = "NOTICE: you already made this exact call; its result is above. Do not repeat it."
            else:
                obs = sandbox.run(name, args)
            step["observation"] = obs[:300]
        nxt = turn + 1
        tail = ""
        if nxt == max_turns and turn < max_turns:
            res.final_turn_nudged = True
            tail = f"\n(Turn {nxt} of {max_turns} is your LAST: reply with <final_answer>.)"
        elif turn < max_turns:
            tail = f"\n(Turn {nxt} of {max_turns} next.)"
        messages.append({"role": "user", "content": f"<observation>\n{obs}\n</observation>{tail}"})

    res.seconds = round(time.time() - t0, 1)
    return res


def run_cli(args) -> int:
    root = Path(args.root).resolve()
    sandbox = ReadOnlySandbox(root, deny=args.deny or [])
    chat = lambda m: mlx_chat(m, host=args.host, model=args.model, temperature=args.temp)  # noqa: E731
    result = run_agent(args.prompt, sandbox, chat_fn=chat, max_turns=args.max_turns)
    print(json.dumps(result.to_dict() if args.json else
                     {"status": result.status, "final_answer": result.final_answer, "turns": result.turns,
                      "repeat_interventions": result.repeat_interventions, "seconds": result.seconds}, indent=2))
    return 0 if result.status == "final" else 1
