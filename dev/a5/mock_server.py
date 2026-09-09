#!/usr/bin/env python
"""A5 step 5 — a tiny OpenAI-compatible MOCK endpoint.

Purpose: exercise `run_relabel.py` + `score_synovitis.py` end-to-end on CPU today,
while the GPU is busy training and before any model weights are downloaded.
It answers /v1/chat/completions with the same strict-JSON shape the real model is
asked for, decided by the same keyword rules the prompt describes.

It deliberately emits a configurable fraction of MESSY replies (```json fences,
a prose preamble, `<think>` blocks, a bare percentage) and occasional HTTP 500s,
so the runner's fence-stripper / balanced-JSON scan / regex fallback / retry path
are all actually covered by the smoke run rather than assumed.

Stdlib only. No model, no GPU, no network egress.

    .venv/bin/python dev/a5/mock_server.py --port 8099
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RE_SYN = re.compile(
    r"sinovi|synovi|sinovy|synovy|pannus|синовит|синовиал|"
    r"υμενιτ|υμενα|υμενο|maziow",
    re.I,
)
RE_EFF = re.compile(
    r"effusion|derrame|erguss|vocht|hydrops|epanchement|efuzyon|sıvı|"
    r"versamento|izljev|wysiek|излив|συλλογη|joint fluid|gelenkerguss",
    re.I,
)
RE_NEG = re.compile(
    r"(?:\bno\b|\bnot\b|\bsin\b|\bgeen\b|\bkein\w*|\bnicht\b|\bwithout\b|"
    r"\bausencia\b|\bizlenmemis\w*|\bpas de\b|\bnegative for\b|\bfree of\b)"
    r"[^.;\n]{0,40}?(?:sinovi|synovi|sinovy|synovy)",
    re.I,
)
# The traps the prompt's NO list names: a "synovial" hit inside one of these
# windows is anatomy/fluid/a cyst, not synovitis. ~14% of all synovial-term hits
# in this corpus (plica 30, fluid/Synovia 28, cyst 23, chondromatosis 13 of 656).
# noun traps: the synovial word belongs to one of these, so it is not synovitis
RE_TRAP_NOUN = re.compile(
    r"plic|plik|cyst|kist|quist|cyste|ganglion|fluid|liquido|liquide|vocht|"
    r"sivi|sıvı|chondromatos|condromatos|hondromatoz|osteochondromat",
    re.I,
)
# "normal"-style qualifiers only count when they FOLLOW the synovial word closely
# ("Synovia unauffaellig", "sinovija uredna") — the bare word "normal" elsewhere
# in a report is far too common to treat as a trap.
RE_TRAP_NORMAL = re.compile(
    r"^.{0,30}?(unauffallig|unauffällig|normal|uredn|izlenmemis|regelrecht|"
    r"physiologisch|intact|unremarkable)",
    re.I,
)
RE_STRONG = re.compile(
    r"thicken|proliferat|hypertroph|engrosamiento|verdikking|verdickung|"
    r"kalinlasma|epaississement|zadebljanje|paxynsi|παχυνση|udebelen",
    re.I,
)
LANGS = (
    (re.compile(r"[Ѐ-ӿ]"), "bg"),
    (re.compile(r"[Ͱ-Ͽ]"), "el"),
    (re.compile(r"rodilla|menisco|hallazgos|derrame", re.I), "es"),
    (re.compile(r"izlenmi|meniskus|bulgular|diz\b", re.I), "tr"),
    (re.compile(r"kniegelenk|keine|befund|erguss", re.I), "de"),
    (re.compile(r"knie\b|bevindingen|geen|gewricht", re.I), "nl"),
    (re.compile(r"koljen|hrskavic|meniska", re.I), "hr"),
    (re.compile(r"genou|menisque|epanchement", re.I), "fr"),
)


def _strip(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _jitter(seed_text: str, span: float) -> float:
    h = hashlib.md5(seed_text.encode("utf-8")).digest()
    return (int.from_bytes(h[:4], "big") / 2**32 - 0.5) * 2 * span


def decide(report: str) -> tuple[float, str, str]:
    """Keyword stand-in for the real model. -> (prob, evidence, language)"""
    flat = _strip(report)
    lang = "en"
    for pat, code in LANGS:
        if pat.search(report):
            lang = code
            break

    if RE_NEG.search(flat):
        m = RE_NEG.search(flat)
        return round(max(0.01, 0.04 + _jitter(report, 0.02)), 3), m.group(0)[:80], lang
    # a synovial term counts only if at least one mention is NOT inside a trap window
    real_hit = None
    n_trapped = 0
    for m in RE_SYN.finditer(flat):
        s, e = m.span()
        # Judge each mention inside ITS OWN sentence. Crossing a sentence boundary
        # mis-reads "...derrame articular con sinovitis difusa. Quiste popliteo..."
        # as a synovial cyst when it is a plain synovitis assertion.
        ls = max((flat.rfind(c, 0, s) for c in ".;\n"), default=-1)
        re_end = min(
            (p for p in (flat.find(c, e) for c in ".;\n") if p != -1), default=len(flat)
        )
        sent = flat[ls + 1 : re_end]
        after = flat[e:re_end][:40]
        if RE_TRAP_NOUN.search(sent) or RE_TRAP_NORMAL.search(after):
            n_trapped += 1
            continue
        real_hit = m
        break
    if real_hit is not None:
        base = 0.94 if not RE_STRONG.search(flat) else 0.90
        s, e = real_hit.span()
        ev = report[max(0, s - 25) : min(len(report), e + 35)].strip()
        return round(min(0.99, base + _jitter(report, 0.04)), 3), ev, lang
    if n_trapped:
        # plica / cyst / synovial fluid / normal synovium only -> NO
        return round(max(0.02, 0.07 + _jitter(report, 0.03)), 3), "", lang
    if RE_STRONG.search(flat) and RE_EFF.search(flat):
        m2 = RE_STRONG.search(flat)
        return round(0.62 + _jitter(report, 0.08), 3), report[max(0, m2.start() - 20) : m2.end() + 30].strip(), lang
    if RE_EFF.search(flat):
        return round(max(0.02, 0.14 + _jitter(report, 0.05)), 3), "", lang
    return round(max(0.01, 0.08 + _jitter(report, 0.04)), 3), "", lang


def render(prob: float, evidence: str, lang: str, style: str) -> str:
    obj = {
        "synovitis": prob,
        "evidence": " ".join(evidence.replace('"', "'").split())[:120],
        "language": lang,
    }
    clean = json.dumps(obj, ensure_ascii=False)
    if style == "clean":
        return clean
    if style == "fenced":
        return f"```json\n{clean}\n```"
    if style == "prose":
        return f"Here is the assessment for this report.\n{clean}\nLet me know if you need more."
    if style == "think":
        return f"<think>\nThe report mentions the synovium or not; weighing effusion vs synovitis.\n</think>\n{clean}"
    if style == "percent":
        return json.dumps(
            {"synovitis": round(prob * 100, 1), "evidence": obj["evidence"], "language": lang},
            ensure_ascii=False,
        )
    if style == "garbage":
        return "I cannot determine this from the report provided."
    return clean


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cfg: argparse.Namespace

    def log_message(self, fmt, *a):  # quiet
        if self.cfg.verbose:
            sys.stderr.write("mock: " + fmt % a + "\n")

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "unknown route"}})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, {"error": {"message": "bad json"}})
            return

        if self.cfg.fail_rate > 0 and random.random() < self.cfg.fail_rate:
            # transient 5xx — the runner must retry it with backoff, not lose the row
            self._send(500, {"error": {"message": "mock injected failure"}})
            return

        msgs = req.get("messages") or []
        user = ""
        is_retry = False
        for m in msgs:
            if m.get("role") == "user":
                c = m.get("content") or ""
                if c.startswith("report:"):
                    user = c
                elif "not valid JSON" in c:
                    is_retry = True
        report = user[len("report:") :].strip().strip('"')

        prob, evidence, lang = decide(report)

        # style selection is deterministic in the report text, so a re-run of the
        # smoke test reproduces exactly the same mix of messy replies.
        style = "clean"
        if not is_retry and self.cfg.messy_rate > 0:
            h = hashlib.md5(report.encode("utf-8")).digest()
            r = int.from_bytes(h[:4], "big") / 2**32          # uniform [0,1)
            pick = h[4] % 5                                    # independent of r
            if r < self.cfg.messy_rate:
                style = ["fenced", "prose", "think", "percent", "garbage"][pick]
        content = render(prob, evidence, lang, style)

        if self.cfg.latency_ms:
            time.sleep(self.cfg.latency_ms / 1000.0)

        p_tok = max(1, len(report) // 4 + 1000)
        c_tok = max(1, len(content) // 4)
        self._send(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model") or "mock",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": p_tok + c_tok,
                },
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--messy-rate", type=float, default=0.15,
                    help="fraction of replies that are fenced/prose/think/percent/garbage")
    ap.add_argument("--fail-rate", type=float, default=0.0, help="fraction of HTTP 500s")
    ap.add_argument("--latency-ms", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    Handler.cfg = args
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(
        f"mock OpenAI endpoint on http://{args.host}:{args.port}/v1 "
        f"(messy_rate={args.messy_rate}, fail_rate={args.fail_rate})",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
