"""Kaggle-side run of dev/sub13/test_combine_guard.py against the sub #13 bundle (pandas 2.3.3 on Kaggle vs 3.0.1
locally). Private kernel abdullllkhan/rsna-knee-combine-check, T4 image (the submission's image), no internet, no data.
  cell 1 = the bundle cell of the generated sub #13 notebook, verbatim (every module + the isolated reader);
  cell 2 = the test module (its __main__ guard stripped), exec'd in its OWN namespace, then run_all(...) on the bundle's
           objects and on the driver cell's own parse/combine functions (the driver cell text is embedded, not run).
Usage: .venv/bin/python -P dev/sub13/build_combine_check.py <submission_multi.ipynb> <push_dir>
"""
import importlib.util
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main(nb_path, out):
    spec = importlib.util.spec_from_file_location("bnm", os.path.join(ROOT, "notebook/build_notebook_multi.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    nb = json.load(open(nb_path))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    bundle, driver = "".join(code[0]["source"]), "".join(code[1]["source"])
    assert "MULTI-ARM SUBMISSION DRIVER" in driver and "isolated: src/rsna/reader_gpjw.py" in bundle
    test_src = B._strip_module(open(os.path.join(ROOT, "dev/sub13/test_combine_guard.py")).read())
    cell2 = ("import json as _json\n"
             f"_TEST_SRC = {json.dumps(test_src)}\n"
             f"_DRIVER_TEXT = {json.dumps(driver)}\n"
             "_tns = {'__name__': 'test_combine_guard'}\n"
             "exec(compile(_TEST_SRC, 'test_combine_guard.py', 'exec'), _tns)\n"
             "_rc = _tns['run_all'](ID_COL, TARGETS, rank_average_frames, rank_average_frames_partial, "
             "write_submission, reader_gpjw, _DRIVER_TEXT, '/kaggle/working')\n"
             "print('COMBINE CHECK EXIT', _rc)\n")

    def cell(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.splitlines(keepends=True)}

    out_nb = {"cells": [cell(bundle), cell(cell2)],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(out, exist_ok=True)
    json.dump(out_nb, open(os.path.join(out, "combine_check.ipynb"), "w"), indent=1)
    meta = {"id": "abdullllkhan/rsna-knee-combine-check", "title": "RSNA Knee Combine Check",
            "code_file": "combine_check.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": False,
            "competition_sources": [], "dataset_sources": [], "kernel_sources": []}
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    print(f"wrote {out}/combine_check.ipynb (bundle cell {len(bundle)} chars, verbatim from {nb_path})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
