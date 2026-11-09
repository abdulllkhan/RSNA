"""Export the sub #13 bundle package: every module the builder inlines, pre-existing ones from git HEAD, new ones
(not in HEAD) from the working tree. The working tree's src/rsna carries another agent's uncommitted edits
(model_v2.py), which must NOT reach the bundle.

Usage: .venv/bin/python dev/sub13/export_head_pkg.py <dest_dir> [--rev HEAD]
Then:  RSNA_BUNDLE_PKG=<dest_dir> .venv/bin/python notebook/build_notebook_multi.py   (or BUNDLE_PKG=... dev/sub7.sh)
Writes <dest_dir>/MANIFEST.txt (module, source, sha256) and verifies every HEAD module byte-for-byte.
"""
import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def builder_modules():
    tree = ast.parse(open(os.path.join(ROOT, "notebook/build_notebook_multi.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MODULES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("MODULES not found in the builder")


def git(*a, check=True):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, check=check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--rev", default="HEAD")
    a = ap.parse_args()
    rev = git("rev-parse", a.rev).stdout.decode().strip()
    if os.path.exists(a.dest):
        shutil.rmtree(a.dest)
    os.makedirs(a.dest)
    rows = []
    for m in builder_modules():
        path = f"src/rsna/{m}"
        r = git("show", f"{rev}:{path}", check=False)
        if r.returncode == 0:
            data, src = r.stdout, f"git {rev[:7]}"
        else:
            data, src = open(os.path.join(ROOT, path), "rb").read(), "working tree (new module)"
        with open(os.path.join(a.dest, m), "wb") as f:
            f.write(data)
        rows.append((m, src, hashlib.sha256(data).hexdigest()))
    bad = []
    for m, src, _ in rows:
        if src.startswith("git"):
            if open(os.path.join(a.dest, m), "rb").read() != git("show", f"{rev}:src/rsna/{m}").stdout:
                bad.append(m)
    with open(os.path.join(a.dest, "MANIFEST.txt"), "w") as f:
        f.write(f"# sub #13 bundle package; pre-existing modules from {rev}\n")
        for m, src, h in rows:
            f.write(f"{m:22s} {src:28s} {h}\n")
    print(open(os.path.join(a.dest, "MANIFEST.txt")).read(), end="")
    if bad:
        raise SystemExit(f"MISMATCH vs {rev}: {bad}")
    n_head = sum(s.startswith("git") for _, s, _ in rows)
    print(f"exported {len(rows)} modules to {a.dest}: {n_head} byte-identical to {rev[:7]}, "
          f"{len(rows) - n_head} new from the working tree")


if __name__ == "__main__":
    sys.exit(main())
