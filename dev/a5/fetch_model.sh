#!/usr/bin/env bash
#
# A5 step 6 — GUARDED download of the multilingual instruct model used to
# re-read the 4,407 knee-MRI reports for the Synovitis column.
#
#   *** THIS SCRIPT REFUSES TO RUN WHILE A FOLD IS TRAINING. ***
#
# A ~17 GB pull hammers page cache + disk and has already OOM'd this box once
# (see PLAN.md §9: "the page cache + workers OOM'd the box"). So:
#   guard 1  no `src.rsna.train` / `train_hr` process may exist
#   guard 2  MemAvailable must be >= 80 GB
#   guard 3  the destination filesystem must have >= 2x the model size free
#   guard 4  nothing may already be listening on the target port
# Guards 1 and 2 CANNOT be overridden. Run this only in a GPU-idle window.
#
# It downloads and then PRINTS the llama-server launch line. It never starts the
# server, and never touches the old `qwen-server.service` on :8080.
#
# Usage:
#   dev/a5/fetch_model.sh --check              # run the guards only, download nothing
#   dev/a5/fetch_model.sh --launch-line        # print the launch line only
#   dev/a5/fetch_model.sh                      # guards, then download
#   dev/a5/fetch_model.sh --model qwen3-32b    # the alternate / second-opinion model
#
set -euo pipefail

# ------------------------------------------------------------------ config
MODEL_KEY="${A5_MODEL:-gemma3-27b}"
MODE="download"
for arg in "$@"; do
  case "$arg" in
    --check)        MODE="check" ;;
    --launch-line)  MODE="launch" ;;
    --model)        MODE="$MODE" ;;                 # value consumed below
    --model=*)      MODEL_KEY="${arg#--model=}" ;;
    -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
    *)              if [[ "${PREV:-}" == "--model" ]]; then MODEL_KEY="$arg"; fi ;;
  esac
  PREV="$arg"
done

MODELS_DIR="${A5_MODELS_DIR:-/home/abdulllkhan/Documents/qwen-models/models}"
LLAMA_SERVER="/home/abdulllkhan/Documents/qwen-models/llama.cpp/build/bin/llama-server"
PORT="${A5_PORT:-8081}"           # NEVER 8080 — that is the old qwen-server service
MIN_AVAIL_GB=80
PARALLEL=4
CTX_PER_SLOT=4096                 # max observed prompt 2,336 tok + 160 out

case "$MODEL_KEY" in
  gemma3-27b)
    HF_REPO="bartowski/google_gemma-3-27b-it-GGUF"
    GGUF_FILE="google_gemma-3-27b-it-Q4_K_M.gguf"
    ALIAS="gemma-3-27b-it"
    SIZE_GB=17
    EXTRA_ARGS=""
    ;;
  qwen3-32b)
    HF_REPO="bartowski/Qwen_Qwen3-32B-GGUF"
    GGUF_FILE="Qwen_Qwen3-32B-Q4_K_M.gguf"
    ALIAS="qwen3-32b"
    SIZE_GB=20
    # Qwen3 reasons by default; a 4,407-report job cannot afford the thinking tokens
    EXTRA_ARGS="--chat-template-kwargs '{\"enable_thinking\":false}'"
    ;;
  mistral-small-24b)
    HF_REPO="bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF"
    GGUF_FILE="mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"
    ALIAS="mistral-small-3.2-24b"
    SIZE_GB=15
    EXTRA_ARGS=""
    ;;
  qwen3-14b-q8)
    HF_REPO="bartowski/Qwen_Qwen3-14B-GGUF"
    GGUF_FILE="Qwen_Qwen3-14B-Q8_0.gguf"
    ALIAS="qwen3-14b"
    SIZE_GB=16
    EXTRA_ARGS="--chat-template-kwargs '{\"enable_thinking\":false}'"
    ;;
  *)
    echo "unknown --model '$MODEL_KEY'. one of: gemma3-27b qwen3-32b mistral-small-24b qwen3-14b-q8" >&2
    exit 2
    ;;
esac

DEST_DIR="$MODELS_DIR/$MODEL_KEY"
DEST="$DEST_DIR/$GGUF_FILE"
CTX_TOTAL=$(( CTX_PER_SLOT * PARALLEL ))

launch_line() {
  cat <<EOF

--- llama-server launch line (port $PORT — NOT the old :8080 service) ---

  $LLAMA_SERVER \\
    --model $DEST \\
    --alias $ALIAS \\
    --host 127.0.0.1 --port $PORT \\
    --n-gpu-layers 99 \\
    --ctx-size $CTX_TOTAL --parallel $PARALLEL \\
    -fa on \\
    --batch-size 2048 --ubatch-size 512 \\
    --cache-reuse 256 \\
    --threads 4 --threads-http 8 \\
    --no-warmup ${EXTRA_ARGS:+\\
    $EXTRA_ARGS}

  notes
    --ctx-size $CTX_TOTAL / --parallel $PARALLEL  -> ${CTX_PER_SLOT} tok per slot; the longest
        report prompts to 2,336 tok and the reply is capped at 160, so it fits.
    --cache-reuse 256    -> the ~1,000-token instruction prefix is identical on every
        request, so each slot prefills it once and reuses the KV afterwards.
    --threads 4          -> leave CPU cores for the training dataloader. Raise only
        when nothing else is running.
    --n-gpu-layers 99    -> all layers on the GB10; ~${SIZE_GB} GB weights + ~2 GB KV,
        comfortably inside the ~35 GB budget beside a training job.
    do NOT add --port 8080; that is the old qwen-server.service.

  then, from $(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):

  .venv/bin/python dev/a5/run_relabel.py \\
      --base-url http://127.0.0.1:$PORT/v1 --model $ALIAS --parallel $PARALLEL

  (add --no-system if the server rejects a system role with this chat template)

EOF
}

if [[ "$MODE" == "launch" ]]; then
  launch_line
  exit 0
fi

# ------------------------------------------------------------------ guards
echo "=== A5 fetch_model guards ==="
FAIL=0

# guard 1 — training must not be running
TRAIN_PIDS="$(pgrep -f 'src[.]rsna[.]train|src[.]rsna[.]train_hr|train_folds[.]sh' || true)"
if [[ -n "$TRAIN_PIDS" ]]; then
  echo "  [REFUSE] training is live (pids: $(echo "$TRAIN_PIDS" | tr '\n' ' '))"
  echo "           a ${SIZE_GB} GB download alongside training has OOM'd this box before."
  echo "           wait for the folds to exit, then re-run."
  FAIL=1
else
  echo "  [ok]     no src.rsna.train process"
fi

# guard 2 — free memory
AVAIL_KB="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
if (( AVAIL_GB < MIN_AVAIL_GB )); then
  echo "  [REFUSE] MemAvailable ${AVAIL_GB} GB < required ${MIN_AVAIL_GB} GB"
  FAIL=1
else
  echo "  [ok]     MemAvailable ${AVAIL_GB} GB >= ${MIN_AVAIL_GB} GB"
fi

# guard 3 — disk headroom (2x the model, for the HF blob + the final copy).
# Check the nearest EXISTING ancestor so --check leaves no directory behind.
DF_TARGET="$DEST_DIR"
while [[ ! -d "$DF_TARGET" && "$DF_TARGET" != "/" ]]; do DF_TARGET="$(dirname "$DF_TARGET")"; done
FREE_GB="$(df -BG --output=avail "$DF_TARGET" | tail -1 | tr -dc '0-9')"
NEED_GB=$(( SIZE_GB * 2 ))
if (( FREE_GB < NEED_GB )); then
  echo "  [REFUSE] only ${FREE_GB} GB free on $DF_TARGET, need ${NEED_GB} GB"
  FAIL=1
else
  echo "  [ok]     ${FREE_GB} GB free on $DF_TARGET (need ${NEED_GB} GB)"
fi

# guard 4 — the target port must be free
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "  [REFUSE] something is already listening on :$PORT"
  FAIL=1
else
  echo "  [ok]     port $PORT is free"
fi

# guard 5 — the GPU should be idle-ish (advisory only; nvidia-smi is flaky on GB10)
if command -v nvidia-smi >/dev/null 2>&1; then
  UTIL="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '')"
  if [[ "$UTIL" =~ ^[0-9]+$ ]] && (( UTIL > 50 )); then
    echo "  [warn]   GPU utilisation ${UTIL}% — something is using the GPU"
  else
    echo "  [ok]     GPU utilisation ${UTIL:-unknown}%"
  fi
fi

# informational — a HF token helps with rate limits; never printed
if [[ -f "$HOME/.cache/huggingface/token" ]]; then
  echo "  [ok]     HF token present at ~/.cache/huggingface/token (not shown)"
else
  echo "  [note]   no HF token; these repos are public so it is only a rate-limit risk"
fi

echo
echo "model  : $HF_REPO"
echo "file   : $GGUF_FILE  (~${SIZE_GB} GB)"
echo "dest   : $DEST"

if (( FAIL )); then
  echo
  echo "GUARDS FAILED — nothing downloaded."
  exit 1
fi

if [[ "$MODE" == "check" ]]; then
  echo
  echo "guards pass; --check requested, so nothing was downloaded."
  launch_line
  exit 0
fi

if [[ -f "$DEST" ]]; then
  echo
  echo "already present, skipping download."
  launch_line
  exit 0
fi

# ---------------------------------------------------------------- download
HF_BIN=""
for c in "$(dirname "${BASH_SOURCE[0]}")/../../.venv/bin/hf" \
         "$(dirname "${BASH_SOURCE[0]}")/../../.venv/bin/huggingface-cli" \
         "$(command -v hf || true)" "$(command -v huggingface-cli || true)"; do
  [[ -x "$c" ]] && { HF_BIN="$c"; break; }
done
if [[ -z "$HF_BIN" ]]; then
  echo "no hf / huggingface-cli found; install with: .venv/bin/pip install -U huggingface_hub" >&2
  exit 1
fi

echo
mkdir -p "$DEST_DIR"
echo "downloading with $HF_BIN ..."
# --max-workers 4 keeps the download from saturating IO on a shared box
HF_HUB_ENABLE_HF_TRANSFER=0 "$HF_BIN" download "$HF_REPO" "$GGUF_FILE" \
  --local-dir "$DEST_DIR" --max-workers 4

if [[ ! -f "$DEST" ]]; then
  echo "download finished but $DEST is missing — check the repo/filename" >&2
  exit 1
fi
echo "ok: $(du -h "$DEST" | cut -f1) at $DEST"
launch_line
