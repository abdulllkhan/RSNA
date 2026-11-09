"""Run a generated standalone notebook bundle locally, CPU only, under the box rules: refuses to start below
--min-avail-gb MemAvailable, caps torch at --threads intra-op threads (and cv2 at 1), and aborts if MemAvailable
falls below --floor-gb during the run. Everything else comes from the RSNA_* env vars the notebook reads.

Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 RSNA_DEVICE=cpu RSNA_WORKERS=2 ... \
       nice -n 19 .venv/bin/python -I dev/sub13/run_bundle.py <submission_multi_standalone.py>
"""
import argparse
import os
import runpy
import sys
import threading
import time


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-avail-gb", type=float, default=30.0)
    ap.add_argument("--floor-gb", type=float, default=15.0)
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "run with CUDA_VISIBLE_DEVICES=\"\""
    assert os.environ.get("RSNA_DEVICE") == "cpu", "run with RSNA_DEVICE=cpu"
    assert int(os.environ.get("RSNA_WORKERS", "99")) <= 2, "run with RSNA_WORKERS <= 2"
    av = mem_avail_gb()
    print(f"[harness] MemAvailable {av:.1f} GB (need >= {a.min_avail_gb})", flush=True)
    if av < a.min_avail_gb:
        raise SystemExit(f"MemAvailable {av:.1f} GB < {a.min_avail_gb} GB: refusing to start")
    import cv2
    import torch
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(1)
    print(f"[harness] torch threads {torch.get_num_threads()} cv2 threads {cv2.getNumThreads()} "
          f"nice {os.nice(0)} pid {os.getpid()}", flush=True)
    stop = threading.Event()

    def watchdog():
        strikes = 0
        while not stop.wait(10):
            strikes = strikes + 1 if mem_avail_gb() < a.floor_gb else 0
            if strikes >= 2:
                print(f"[harness] WATCHDOG: MemAvailable < {a.floor_gb} GB twice -> aborting", flush=True)
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    t0 = time.time()
    sys.argv = [a.bundle]
    runpy.run_path(a.bundle, run_name="__main__")
    stop.set()
    print(f"[harness] done in {time.time() - t0:.1f}s; MemAvailable {mem_avail_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
