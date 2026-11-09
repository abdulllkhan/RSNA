#!/usr/bin/env bash
# Exercise dev/sub7.sh's READER_N + EXPECT_LOG_RE gate lines (extracted verbatim, /tmp/koutm -> a scratch dir) on a
# Kaggle-format JSON log synthesised from a local run's stdout (Kaggle escapes < > & as < > &).
# Usage: dev/sub13/test_gate_regex.sh <local_run_stdout.log> <scratch_dir> [sed substitutions file]
set -uo pipefail
LOG="$1"; D="$2"; SUBS="${3:-}"
mkdir -p "$D/kout"; rm -f "$D/kout/"*.log
/home/abdulllkhan/Documents/RSNA/.venv/bin/python -P - "$LOG" "$D/kout/fake.log" "$SUBS" <<'PY'
import json, re, sys
src, dst, subs = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src).read().split("\n")
pairs = [l.split("\t", 1) for l in open(subs).read().split("\n") if "\t" in l] if subs else []
out, t = [], 1.0
for l in lines:
    for a, b in pairs:
        l = re.sub(a, b, l)
    d = json.dumps({"stream_name": "stdout", "time": t, "data": l + "\n"}, ensure_ascii=True)
    d = d.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    out.append(("[" if not out else ",") + d)
    t += 0.1
open(dst, "w").write("\n".join(out) + "]\n")
PY
log(){ echo "[gate-test] $*"; }; fail(){ log "FAILED: $*"; exit 1; }
eval "$(grep -E '^if \[ "\$\{READER_N:-0\}"|^if \[ -n "\$\{EXPECT_LOG_RE:-\}" \]' /home/abdulllkhan/Documents/RSNA/dev/sub7.sh | sed "s|/tmp/koutm|$D/kout|g")"
log "ALL GATES PASSED"
