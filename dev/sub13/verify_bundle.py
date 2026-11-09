"""Prove the sub #13 bundle inlines every PRE-EXISTING module exactly as git HEAD does.

For each module section of the NEW standalone bundle:
  * pre-existing module  -> identical to the same section of the OLD bundle (HEAD builder + HEAD modules) AND to
                            the builder's strip/rename of `git show <rev>:src/rsna/<m>`;
  * reader_gpjw.py       -> the embedded literal equals src/rsna/reader_gpjw.py (working tree) byte-for-byte;
  * combine_partial.py   -> the builder's strip of src/rsna/combine_partial.py (working tree).
Also checks the working-tree src/rsna files that DIFFER from HEAD are not what the bundle carries.

Usage: .venv/bin/python dev/sub13/verify_bundle.py <new_standalone.py> <old_standalone.py> [--rev HEAD]
"""
import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARK = re.compile(r"^# ===================== (inlined|isolated): src/rsna/(\S+)")


def sections(path):
    out, cur, buf = {}, None, []
    for line in open(path).read().split("\n"):
        m = MARK.match(line)
        if m or line.startswith("# ========================= MULTI-ARM SUBMISSION DRIVER"):
            if cur:
                out[cur] = "\n".join(buf).strip("\n")
            cur, buf = (m.group(2) if m else "__driver__"), []
            continue
        if cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip("\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("old")
    ap.add_argument("--rev", default="HEAD")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("bnm", os.path.join(ROOT, "notebook/build_notebook_multi.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    new, old = sections(a.new), sections(a.old)
    ok = True
    for m in B.MODULES:
        head = subprocess.run(["git", "-C", ROOT, "show", f"{a.rev}:src/rsna/{m}"], capture_output=True)
        if head.returncode == 0:
            ref = B._strip_module(B._apply_renames(head.stdout.decode(), m))
            same_old = new.get(m) == old.get(m)
            same_head = new.get(m) == ref
            wt = open(os.path.join(ROOT, "src/rsna", m)).read()
            wt_differs = wt != head.stdout.decode()
            ok &= same_old and same_head
            print(f"[pre-existing] {m:20s} == old-bundle section: {same_old}  == strip(git {a.rev}): {same_head}"
                  + ("   (working tree differs from HEAD -> NOT used)" if wt_differs else ""))
        elif m in B.ISOLATED_MODULES:
            sec = new.get(m, "")
            lit = re.search(r"= r'''(.*?)'''\n", sec, re.S)
            src = open(os.path.join(ROOT, "src/rsna", m)).read()
            same = lit is not None and ast.literal_eval("r'''" + lit.group(1) + "'''") == src
            ok &= same
            print(f"[new, isolated] {m:20s} embedded literal == src/rsna/{m}: {same}")
        else:
            src = open(os.path.join(ROOT, "src/rsna", m)).read()
            same = new.get(m) == B._strip_module(B._apply_renames(src, m))
            ok &= same
            print(f"[new, flat]     {m:20s} == strip(src/rsna/{m}): {same}")
    extra = sorted(set(new) - set(B.MODULES) - {"__driver__"})
    if extra:
        ok = False
        print("unexpected sections:", extra)
    print("BUNDLE VERIFIED" if ok else "BUNDLE MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
