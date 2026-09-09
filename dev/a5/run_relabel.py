#!/usr/bin/env python
"""A5 step 3 — run the Synovitis prompt over the reports against an
OpenAI-compatible endpoint (llama-server / vLLM / the mock in `mock_server.py`).

Properties that matter for a 4,407-row job on a box that is also training:
  * resumable   — every answer is appended to a JSONL immediately; a re-run skips
                  the ids already in it (so Ctrl-C / a crash costs one request).
  * concurrent  — `--parallel N` threads over one pooled httpx client, matched to
                  the server's `--parallel N` slots.
  * robust      — fenced/prosey/`<think>`-wrapped replies are salvaged; one retry
                  with an explicit "JSON only" nudge; then a regex fallback; a row
                  that still fails is recorded with `ok=false` (never silently 0).
  * observable  — rolling rate + ETA + parse-failure count.
  * CPU-only    — no torch, no GPU. Safe to run while a fold trains.

Examples
--------
  # smoke against the mock (no model needed)
  .venv/bin/python dev/a5/mock_server.py --port 8099 &
  .venv/bin/python dev/a5/run_relabel.py --base-url http://127.0.0.1:8099/v1 \
      --model mock --limit 60 --parallel 4 --out-prefix data/a5/synovitis_llm_mock

  # the real run
  .venv/bin/python dev/a5/run_relabel.py --base-url http://127.0.0.1:8081/v1 \
      --model gemma-3-27b-it --parallel 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import build_messages, build_retry_messages  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS = REPO / "data" / "a5" / "reports.parquet"
DEFAULT_TRAIN = REPO / "data" / "comp_csv" / "train.csv"
DEFAULT_PREFIX = REPO / "data" / "a5" / "synovitis_llm"

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_PROB_RE = re.compile(r'"?synovitis"?\s*[:=]\s*([01](?:\.\d+)?|\.\d+)', re.I)
_EVID_RE = re.compile(r'"evidence"\s*:\s*"((?:[^"\\]|\\.)*)"', re.I)
_LANG_RE = re.compile(r'"language"\s*:\s*"([^"]{0,12})"', re.I)
_YESNO_RE = re.compile(r'"?synovitis"?\s*[:=]\s*"?(yes|true|no|false)"?', re.I)


# --------------------------------------------------------------- parsing
def _balanced_json(text: str) -> str | None:
    """Return the first balanced {...} block (quote/escape aware)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_reply(raw: str) -> tuple[float, str, str, str] | None:
    """-> (prob, evidence, language, how) or None if nothing usable."""
    if not raw:
        return None
    text = _THINK_RE.sub(" ", raw).strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    for how, candidate in (("json", text), ("json_block", _balanced_json(text))):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "synovitis" not in obj:
            continue
        p = _coerce_prob(obj.get("synovitis"))
        if p is None:
            continue
        ev = obj.get("evidence") or ""
        lg = obj.get("language") or ""
        return p, str(ev)[:300], str(lg)[:12], how

    # regex fallback over the raw text
    m = _PROB_RE.search(text)
    if m:
        p = _coerce_prob(m.group(1))
        if p is not None:
            ev = _EVID_RE.search(text)
            lg = _LANG_RE.search(text)
            return (
                p,
                (ev.group(1)[:300] if ev else ""),
                (lg.group(1) if lg else ""),
                "regex",
            )
    m = _YESNO_RE.search(text)
    if m:
        p = 0.9 if m.group(1).lower() in ("yes", "true") else 0.05
        return p, "", "", "regex_yesno"
    return None


def _coerce_prob(v) -> float | None:
    try:
        p = float(v)
    except (TypeError, ValueError):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("yes", "true"):
                return 0.9
            if s in ("no", "false"):
                return 0.05
        return None
    if p != p:  # NaN
        return None
    # Some models answer with a percentage. Only rescale when the value is far
    # enough above 1 to really be one — 1.0 < p < 2 is an over-shoot, not "1.7%".
    if 2.0 <= p <= 100.0:
        p = p / 100.0
    return min(max(p, 0.0), 1.0)


# --------------------------------------------------------------- io
def load_reports(path: Path, limit: int | None, order: str) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)[["StudyInstanceUID", "Report"]]
    elif DEFAULT_TRAIN.exists():
        print(f"[warn] {path} missing; falling back to train.csv", file=sys.stderr)
        df = pd.read_csv(DEFAULT_TRAIN)[["StudyInstanceUID", "Report"]]
    else:
        raise SystemExit(f"no reports at {path} and no {DEFAULT_TRAIN}")
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["Report"] = df["Report"].fillna("").astype(str)

    if order == "gold-first":
        train = pd.read_csv(DEFAULT_TRAIN)[["StudyInstanceUID", "Synovitis"]]
        train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
        gold = set(train.loc[train["Synovitis"].notna(), "StudyInstanceUID"])
        df["_g"] = [0 if u in gold else 1 for u in df["StudyInstanceUID"]]
        df = df.sort_values("_g", kind="stable").drop(columns="_g")
    elif order == "random":
        df = df.sample(frac=1.0, random_state=42)

    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def load_done(jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl.exists():
        return done
    bad = 0
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                continue
            if rec.get("ok") and rec.get("StudyInstanceUID"):
                done.add(str(rec["StudyInstanceUID"]))
    if bad:
        print(f"[warn] {bad} unparseable lines in {jsonl.name} (ignored)")
    return done


def jsonl_to_csv(jsonl: Path, csv_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("ok"):
                continue
            rows.append(
                {
                    "StudyInstanceUID": str(rec["StudyInstanceUID"]),
                    "synovitis_llm": float(rec["synovitis"]),
                    "evidence": rec.get("evidence", "") or "",
                    "llm_language": rec.get("language", "") or "",
                }
            )
    df = pd.DataFrame(rows, columns=["StudyInstanceUID", "synovitis_llm", "evidence", "llm_language"])
    if len(df):
        df = df.drop_duplicates("StudyInstanceUID", keep="last").reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


# --------------------------------------------------------------- runner
class Runner:
    def __init__(self, args, jsonl: Path):
        import httpx

        self.args = args
        self.use_system = not args.no_system
        self.jsonl = jsonl
        self.url = args.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = args.api_key or os.environ.get("OPENAI_API_KEY") or "sk-no-key-required"
        headers["Authorization"] = f"Bearer {key}"
        self.client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(args.timeout, connect=15.0),
            limits=httpx.Limits(
                max_connections=args.parallel + 2,
                max_keepalive_connections=args.parallel + 2,
            ),
        )
        self.lock = threading.Lock()
        self.fh = jsonl.open("a", encoding="utf-8")
        self.n_ok = 0
        self.n_fail = 0
        self.n_retry = 0
        self.n_regex = 0
        self.tok_out = 0
        self.tok_in = 0
        self.t0 = time.time()
        self.total = 0

    def close(self):
        try:
            self.fh.close()
        finally:
            self.client.close()

    def _chat(self, messages: list[dict]) -> tuple[str, dict]:
        payload = {
            "model": self.args.model,
            "messages": messages,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_tokens": self.args.max_tokens,
            "stream": False,
        }
        if self.args.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_err = None
        for attempt in range(self.args.http_retries + 1):
            try:
                r = self.client.post(self.url, json=payload)
                if r.status_code >= 500 or r.status_code == 429:
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(min(2 ** attempt, 8))
                    continue
                r.raise_for_status()
                data = r.json()
                content = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage") or {}
                return content, usage
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(last_err or "request failed")

    def work(self, uid: str, report: str) -> None:
        rec = {"StudyInstanceUID": uid, "ok": False}
        raw = ""
        try:
            raw, usage = self._chat(build_messages(report, self.use_system))
            self.tok_in += int(usage.get("prompt_tokens") or 0)
            self.tok_out += int(usage.get("completion_tokens") or 0)
            parsed = parse_reply(raw)
            if parsed is None:
                with self.lock:
                    self.n_retry += 1
                raw2, usage2 = self._chat(build_retry_messages(report, raw, self.use_system))
                self.tok_in += int(usage2.get("prompt_tokens") or 0)
                self.tok_out += int(usage2.get("completion_tokens") or 0)
                parsed = parse_reply(raw2)
                if parsed is not None:
                    raw = raw2
            if parsed is not None:
                p, ev, lg, how = parsed
                rec.update(
                    ok=True, synovitis=p, evidence=ev, language=lg, parse=how
                )
                if how.startswith("regex"):
                    with self.lock:
                        self.n_regex += 1
            else:
                rec["error"] = "unparseable"
                rec["raw"] = raw[:500]
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["raw"] = raw[:300]

        with self.lock:
            self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.fh.flush()
            if rec["ok"]:
                self.n_ok += 1
            else:
                self.n_fail += 1
            done = self.n_ok + self.n_fail
            if done % self.args.log_every == 0 or done == self.total:
                el = time.time() - self.t0
                rate = done / max(el, 1e-6)
                eta = (self.total - done) / max(rate, 1e-9)
                print(
                    f"[{done}/{self.total}] ok={self.n_ok} fail={self.n_fail} "
                    f"retry={self.n_retry} regex={self.n_regex} | "
                    f"{rate:.2f} rep/s | out {self.tok_out/max(el,1e-6):.0f} tok/s | "
                    f"elapsed {el/60:.1f}m ETA {eta/60:.1f}m",
                    flush=True,
                )


# --------------------------------------------------------------- self-test
_SELFTEST = [
    ('{"synovitis": 0.95, "evidence": "Sinovitis", "language": "es"}', 0.95, "json"),
    ('```json\n{"synovitis":0.1,"evidence":"","language":"en"}\n```', 0.10, "json"),
    ('Sure! Here is the JSON:\n{"synovitis": 0.8, "evidence": "synovial thickening", "language": "en"}\nHope that helps.',
     0.80, "json_block"),
    ('<think>weighing effusion vs synovitis</think>\n{"synovitis": 0.03, "evidence": "geen synovitis", "language": "nl"}',
     0.03, "json"),
    ('{"synovitis": 92, "evidence": "Synovialitis", "language": "de"}', 0.92, "json"),
    ('{"synovitis": "yes", "evidence": "sinovit", "language": "tr"}', 0.90, "json"),
    ('The answer is "synovitis": 0.65 with "evidence": "pannus" and "language": "en".',
     0.65, "regex"),
    ('{"synovitis": 0.4, "evidence": "he said {nested} \\" quote", "language": "hr"}', 0.4, "json"),
    ('synovitis: yes', 0.90, "regex_yesno"),
    ('{"synovitis": 1.7}', 1.0, "json"),
    ('{"synovitis": -0.2}', 0.0, "json"),
    ('I cannot determine this from the report provided.', None, None),
    ('', None, None),
]


def selftest() -> int:
    bad = 0
    for raw, want_p, want_how in _SELFTEST:
        got = parse_reply(raw)
        if want_p is None:
            ok = got is None
            got_s = "None" if got is None else str(got)
        else:
            ok = got is not None and abs(got[0] - want_p) < 1e-6 and got[3] == want_how
            got_s = "None" if got is None else f"p={got[0]} how={got[3]}"
        print(f"  {'ok ' if ok else 'FAIL'} {raw[:56]!r:<60} -> {got_s}")
        bad += 0 if ok else 1
    print(f"\nparse_reply self-test: {len(_SELFTEST) - bad}/{len(_SELFTEST)} passed")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="exercise the reply parser on adversarial strings and exit")
    ap.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N reports")
    ap.add_argument("--reports", default=str(DEFAULT_REPORTS))
    ap.add_argument("--out-prefix", default=str(DEFAULT_PREFIX),
                    help="writes <prefix>.jsonl (append/resume) and <prefix>.csv")
    ap.add_argument("--order", choices=["natural", "random", "gold-first"], default="natural")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--http-retries", type=int, default=2)
    ap.add_argument("--no-system", action="store_true",
                    help="fold the instructions into the user turn (chat templates "
                         "that reject a system role, e.g. some gemma-3 GGUFs)")
    ap.add_argument("--json-mode", action="store_true",
                    help="send response_format=json_object (llama-server supports it)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--no-resume", action="store_true", help="ignore existing jsonl")
    ap.add_argument("--csv-only", action="store_true",
                    help="skip inference, just rebuild <prefix>.csv from the jsonl")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(1 if selftest() else 0)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl = prefix.with_suffix(".jsonl")
    csv_path = prefix.with_suffix(".csv")

    if args.csv_only:
        df = jsonl_to_csv(jsonl, csv_path)
        print(f"wrote {csv_path} ({len(df)} rows)")
        return

    reports = load_reports(Path(args.reports), args.limit, args.order)
    done = set() if args.no_resume else load_done(jsonl)
    if args.no_resume and jsonl.exists():
        jsonl.unlink()
    todo = reports[~reports["StudyInstanceUID"].isin(done)].reset_index(drop=True)

    print(
        f"endpoint {args.base_url}  model={args.model}  parallel={args.parallel}\n"
        f"reports {len(reports)}  already done {len(done)}  to do {len(todo)}"
    )
    if not len(todo):
        df = jsonl_to_csv(jsonl, csv_path)
        print(f"nothing to do; wrote {csv_path} ({len(df)} rows)")
        return

    runner = Runner(args, jsonl)
    runner.total = len(todo)
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            list(
                pool.map(
                    lambda t: runner.work(t[0], t[1]),
                    zip(todo["StudyInstanceUID"], todo["Report"]),
                )
            )
    except KeyboardInterrupt:
        print("\n[interrupted] progress is on disk — re-run to resume", flush=True)
    finally:
        el = time.time() - runner.t0
        runner.close()
        print(
            f"\ndone: ok={runner.n_ok} fail={runner.n_fail} retry={runner.n_retry} "
            f"regex-fallback={runner.n_regex} in {el/60:.1f} min "
            f"({runner.n_ok/max(el,1e-6):.2f} rep/s; "
            f"prompt {runner.tok_in} tok, completion {runner.tok_out} tok)"
        )

    df = jsonl_to_csv(jsonl, csv_path)
    print(f"wrote {csv_path} ({len(df)} rows)")
    if len(df):
        print(
            f"synovitis_llm: mean {df.synovitis_llm.mean():.3f}  "
            f">0.5 rate {(df.synovitis_llm > 0.5).mean():.3f}  "
            f"with-evidence {(df.evidence.str.len() > 0).mean():.3f}"
        )
    if runner.n_fail:
        print(f"[note] {runner.n_fail} rows failed — re-run the same command to retry them")


if __name__ == "__main__":
    main()
