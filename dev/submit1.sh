#!/usr/bin/env bash
# Detached submission driver for baseline v1 (kernel v3, T4 + cpu-fallback guard).
# Polls the kernel (curl, light); on COMPLETE, fetches output and submits ONLY if the
# predictions are real (not the all-0.5 fallback). Safe to run unattended.
cd /home/abdulllkhan/Documents/RSNA || exit 1
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
KG="../multi_step_tool_attack/.venv/bin/kaggle"
U=abdullllkhan; K=rsna-knee-baseline-infer; COMP=rsna-knee-abnormality-detection
KEY=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")
echo "=== submit1 start $(date) ==="
st=""
for i in $(seq 1 100); do
  st=$(curl -s -u "$U:$KEY" "https://www.kaggle.com/api/v1/kernels/status?userName=$U&kernelSlug=$K" 2>/dev/null)
  echo "$(date +%H:%M:%S) $st"
  echo "$st" | grep -qiE 'complete|error|cancel' && break
  sleep 30
done
echo "terminal status: $st"
if echo "$st" | grep -qi complete; then
  rm -rf /tmp/kout3; mkdir -p /tmp/kout3
  $KG kernels output "$U/$K" -p /tmp/kout3 2>&1 | tail -2
  echo "--- submission head ---"; head -5 /tmp/kout3/submission.csv 2>/dev/null
  ndist=$(cut -d, -f2- /tmp/kout3/submission.csv 2>/dev/null | tail -n +2 | sort -u | wc -l)
  echo "distinct pred rows = $ndist"
  if [ "${ndist:-0}" -gt 1 ]; then
    echo ">>> REAL predictions -> SUBMITTING #1"
    # Code-comp submits REQUIRE the kernel version (-v). Record it at push time.
    VER="${KERNEL_VERSION:-$(cat dev/kaggle/last_kernel_version 2>/dev/null)}"
    if [ -z "$VER" ]; then
      echo ">>> ERROR: kernel version unknown (set KERNEL_VERSION or dev/kaggle/last_kernel_version); NOT submitting"
    else
      $KG competitions submit "$COMP" -k "$U/$K" -v "$VER" -f submission.csv \
          -m "${SUBMIT_MSG:-$(cat dev/kaggle/last_submit_msg 2>/dev/null || echo submission)} (kernel v$VER)" 2>&1 | tail -5
    fi
    echo ">>> submitted; recent submissions:"
    $KG competitions submissions "$COMP" 2>&1 | head -4
  else
    echo ">>> STILL 0.5 fallback -> NOT submitting. Device log:"
    grep -aoE 'no kernel image|Traceback|\[device\][^\\"]*' /tmp/kout3/*.log 2>/dev/null | head
  fi
else
  echo ">>> kernel did not complete cleanly; not submitting."
fi
echo "=== SUBMIT1 DONE $(date) ==="
