# Agent instructions

**User-facing overview and architecture** are in [`README.md`](./README.md). **Project context, layout, how to run, and conventions** are in [`CLAUDE.md`](./CLAUDE.md). Read `CLAUDE.md` first when working in this repository; it is the source of truth for what this repo is, which files to edit, and dependency constraints.

**Summary**: `microgpt` is a no-dependencies Python demo of a tiny GPT (scalar autograd + transformer). The main entry points are `microgpt_updated.py` (imports `mgpt/` for the model and `run_report/` for saved report text; writes an `output_*.txt` run report whose name includes a local timestamp so runs do not clobber each other) and `microgpt.py` (compact; console output only). Run with `python microgpt_updated.py` or `python microgpt.py`. To add the standard narrative to older report files, use `python annotate_run_reports.py`. To diff two reports (config, final loss, samples), use `python compare_run_reports.py path/A.txt path/B.txt` (exit `0`/`1`/`2`). Details are in `README.md` and `CLAUDE.md`.

For anything not covered in `CLAUDE.md`, follow normal Python style and the existing patterns in the file you are editing.
