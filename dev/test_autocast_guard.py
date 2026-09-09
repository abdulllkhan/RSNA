#!/usr/bin/env python
"""Unit test for the two inference runtime fixes (reports/runtime_fixes.md).

Runs on CPU, needs no GPU and no checkpoints:

  1. `_autocast` picks the right dtype per compute capability. Kaggle scores on a T4
     (sm_75) which we cannot reach locally (this box is a GB10, sm_121), so the capability
     is faked by monkeypatching `torch.cuda.get_device_capability`. This is the only test
     of the fp16 rung that exists before a Kaggle run.
  2. `_forward_with_guard` recomputes a batch in fp32 when the autocast forward returns
     non-finite values or raises RuntimeError, logs exactly once, and is a pure
     pass-through otherwise.

  3. The three inlined copies (infer / infer_hr / infer_plane) are textually identical --
     they collapse into ONE namespace in the Kaggle bundle, so the last one wins and a
     divergence would silently apply to every arm.

    PYTHONPATH=. .venv/bin/python dev/test_autocast_guard.py
"""
import contextlib
import inspect
import io
import sys

import torch

from src.rsna import infer, infer_hr, infer_plane

FAIL = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAIL.append(msg)


print("[1] _autocast dtype by compute capability")
_orig = torch.cuda.get_device_capability
try:
    for cap, want in [((9, 0), torch.bfloat16), ((8, 0), torch.bfloat16), ((8, 6), torch.bfloat16),
                      ((7, 5), torch.float16), ((7, 0), torch.float16),
                      ((6, 0), None), ((0, 0), None)]:
        torch.cuda.get_device_capability = lambda *a, _c=cap, **k: _c
        ctx = infer._autocast("cuda")
        got = getattr(ctx, "fast_dtype", None)
        check(got == want, f"sm_{cap[0]}{cap[1]} -> {got} (want {want})")
finally:
    torch.cuda.get_device_capability = _orig
check(getattr(infer._autocast("cpu"), "fast_dtype", None) is None, "cpu -> no autocast")


print("[2] _forward_with_guard fp32 backstop")


class Inf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 0

    def forward(self, x, **kw):
        self.n += 1
        return (torch.full((x.shape[0], 12), float("inf")) if self.n == 1
                else x.new_zeros(x.shape[0], 12) + 7.0)


class Boom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 0

    def forward(self, x, **kw):
        self.n += 1
        if self.n == 1:
            raise RuntimeError("GET was unable to find an engine to execute this computation")
        return x.new_zeros(x.shape[0], 12) + 3.0


class Ok(torch.nn.Module):
    def forward(self, x, **kw):
        return x.new_ones(x.shape[0], 12)


infer._FP32_FALLBACK["logged"] = False
x = torch.randn(2, 3, 4)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    a = infer._forward_with_guard(Inf(), x, "cpu")
    b = infer._forward_with_guard(Boom(), x, "cpu")
    c = infer._forward_with_guard(Ok(), x, "cpu")
log = buf.getvalue()
check(bool(torch.isfinite(a).all()) and float(a[0, 0]) == 7.0, "non-finite output -> fp32 recompute")
check(bool(torch.isfinite(b).all()) and float(b[0, 0]) == 3.0, "RuntimeError -> fp32 recompute")
check(float(c[0, 0]) == 1.0, "clean forward passes through unchanged")
check(log.count("fp32 fallback") == 1, f"logged exactly once (got {log.count('fp32 fallback')})")


print("[3] the three inlined copies are textually identical")
srcs = {m.__name__: inspect.getsource(m._autocast) + inspect.getsource(m._forward_with_guard)
        for m in (infer, infer_hr, infer_plane)}
check(len(set(srcs.values())) == 1, f"identical across {list(srcs)}")

print("FAILED" if FAIL else "ALL OK")
sys.exit(1 if FAIL else 0)
