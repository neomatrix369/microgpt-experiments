# microgpt

A minimal, dependency-free **character-level GPT** in pure Python: scalar autograd (`Value`), a tiny transformer (embeddings, multi-head self-attention, MLP, RMSNorm), Adam, and name generation. Based on the [microGPT / makemore](https://github.com/karpathy/makemore) style exercises ([write-up](https://karpathy.github.io/2026/02/12/microgpt/)).

There is **no** `requirements.txt` or `pyproject.toml` by design: only the Python 3 standard library.

## Repository layout

| Path | Role |
|------|------|
| `microgpt.py` | Compact “single story” version: one script, global state, matches the original blog-style walkthrough. |
| `microgpt_updated.py` | Refactored entry: hyperparameters (module constants, overridable via **`argparse`** CLI), `train()` / `generate()` / `main()`, `save_run_report()`, and richer comments. Imports the **`mgpt`** package (autograd, transformer forward, data) and **`run_report`** (on-disk report format). Prefer this for changes that need structure or tests. |
| `mgpt/` | Package: `Value` (scalar autograd), ops (`linear`, `softmax`, `rmsnorm`, `make_matrix`), transformer step `gpt()`, dataset + `Tokeniser` (`load_dataset`, `build_tokeniser`), sample metrics (`evaluation.py`: character similarity + three-tier semantic heuristics; `compute_sample_quality_metrics` / `format_sample_quality_console_lines` feed the refactored entry and reports). Stdlib only. |
| `run_report/` | Package: parse/compare fields of `output_*.txt` (`parse.py`), narrative (`narrative.py`), **`paths.py`** (`run_reports_dir`, `DEFAULT_RUN_REPORT_DIR`), full report assembly (`builder.py`), text loss comparison grids (`text_loss_plot.py`). Shared by `microgpt_updated.py`, `annotate_run_reports.py`, `compare_run_reports.py`, and `experiments/report_generator.py`. |
| `experiments/report_generator.py` | `argparse` CLI: reads one or more `output_*.txt` files (default: glob under `outputs/` at repo root), writes HTML (`-o`, default `outputs/comparison_report.html`): shared/varying **primary** config (no `HEAD_DIM` row; footnote for derived per-head width), quality table, aligned samples (2+ runs), loss ASCII (`--loss-bins`, `--loss-height`), tier bars. |
| `tests/` | `pytest` tests for evaluation, report builder/parse round-trip, and loss-plot helpers. |
| `annotate_run_reports.py` | Inserts the same `--- What this run is ---` narrative into **existing** `output_*.txt` reports (stdlib-only backfill for past experiments). |
| `compare_run_reports.py` | Compares two `output_*.txt` files: parsed config keys ( **`HEAD_DIM` omitted** from the printed diff — derived from `N_EMBD` and `N_HEAD` ), final loss, ordered inference samples. Stdlib-only; exit `0` / `1` / `2` (match / diff / error). |
| `input.txt` | One document per line (e.g. names). If missing, `microgpt.py` / `microgpt_updated.py` can download the classic names dataset from the makemore repo. |
| `outputs/` | Default directory for `output_*.txt` run reports and `comparison_report.html` (`run_report.paths.DEFAULT_RUN_REPORT_DIR`); gitignored. |
| `output_*.txt` | Optional artifacts written by `microgpt_updated.py` (not by `microgpt.py`), default path **`outputs/`** + stem: hyperparameters, narrative, optional experiment-suite labels, samples, glossary. Filename encodes `L/E/H/B/S/T/seed` (no `D` token; per-head width is not a filename field) plus `_YYYYMMDD_HHMMSS` (see `format_run_output_path()` / `format_run_output_path_for_params()`). Config text lists **`N_EMBD` / `N_HEAD`**; **`HEAD_DIM`** is a `#` comment + parser-filled field, not a separate experiment knob. |

## How to run

```bash
# Refactored entry (recommended)
python microgpt_updated.py

# Refactored entry: override hyperparameters for this run (see --help)
python microgpt_updated.py --help

# Original compact script
python microgpt.py
```

**How to use microgpt_updated.py** (baseline, `--help`, `--input`, source-only vs CLI, sweep reproduction, report tools): **`README.md` → [How to use microgpt_updated.py](README.md#how-to-use-microgpt_updatedpy)**.

**Run experiments examples** (distinct `N_HEAD` × `NUM_STEPS` from saved `output_*.txt`): **`README.md` → [Run experiments examples](README.md#run-experiments-examples)** and **`docs/M2-semantic-quality.md`** (Commands → *Run experiments examples*).

Expect stdout: dataset size, vocab size, parameter count, training loss per step, then 20 sampled “hallucinated” names. **`microgpt_updated.py` also writes** a run report under **`run_reports_dir(<repo root>)`** / `outputs/` (default from `format_run_output_path()`; **`--output-dir`** overrides; directory created if missing). **`microgpt.py` does not** write that file. **`microgpt_updated.py`** is the only script with a CLI; flags map to the same symbols documented in `README.md` (e.g. `--num-steps` → `NUM_STEPS`). Experiment-suite fields (`--suite-index`, `--suite-total`, `--suite-note`) use `argparse.SUPPRESS` so omitting them leaves any values you set in the source file unchanged for that run.

**Backfill narrative on old reports** (same text as new runs, parsed from the config block):

```bash
python annotate_run_reports.py
python annotate_run_reports.py path/to/output_L1_....txt
```

**Compare two saved reports** (config + final loss + inference samples; **`HEAD_DIM` excluded from the printed config diff**; ignores narrative, glossary, quality blocks, and loss-history CSV for *equality*, but prints **text loss graphs** when both files include `--- Loss history (CSV: step,loss) ---`):

```bash
python compare_run_reports.py outputs/output_A.txt outputs/output_B.txt
python compare_run_reports.py outputs/output_A.txt outputs/output_B.txt --loss-bins 96 --loss-height 14
```

**HTML comparison** of one or more reports (defaults to all `outputs/output_*.txt`):

```bash
python experiments/report_generator.py
python experiments/report_generator.py outputs/a.txt outputs/b.txt -o outputs/comparison_report.html
```

Exit codes (`compare_run_reports.py`): `0` all match, `1` some field or sample differs, `2` bad args or parse failure. `report_generator.py` exits `2` if no input files or render error.

**Runtime**: Training is 1000 steps by default and is **slow** (scalar autograd in Python). That is expected.

## Conventions and internals

- **Autograd**: `Value` in `mgpt/value.py` implements forward ops with local gradients; `loss.backward()` walks the graph. No PyTorch.
- **Model**: GPT-2–like stack with **RMSNorm** (not LayerNorm), **no biases**, **ReLU** in the MLP (not GELU). KV caches for attention are part of the live graph during training (not detached). Forward step: `mgpt/model.py` (`gpt()`).
- **Data**: Character-level tokens; a special **BOS** (beginning-of-sequence) token wraps each line. `BLOCK_SIZE` limits context; sampling stops at BOS or max length.
- **Hyperparameters**: In `microgpt.py` they are module-level names (`n_layer`, `n_embd`, …). In `microgpt_updated.py` they are `N_LAYER`, `N_EMBD`, `BLOCK_SIZE`, … at the top of the file, with optional **CLI overrides** parsed in `main()` (`_build_arg_parser()`, `_apply_parsed_args()`, `_validate_hyperparameters()`).
- **Run reports** (`microgpt_updated.py` only): After training and `generate()`, `compute_sample_quality_metrics()` supplies character and semantic-tier stats; `save_run_report()` writes a text report including `--- What this run is ---` (input file + training summary + what loss and samples mean), then optional `--- Experiment suite ---` if any of `EXPERIMENT_SUITE_INDEX`, `EXPERIMENT_SUITE_TOTAL`, or `EXPERIMENT_SUITE_NOTE` is set, then **config** (`N_EMBD` / `N_HEAD` and other `KEY=` lines, plus a **`#` line** for derived per-head width — not a `HEAD_DIM=` row), final loss, optional `--- Sample quality (character-level) ---` and `--- Semantic quality (three-tier) ---`, optional `--- Loss history (CSV: step,loss) ---` (from the in-memory `loss_history` list), inference samples, and a parameter glossary. The narrative is implemented in `run_report/narrative.py` (`format_run_narrative_lines`) and shared with `annotate_run_reports.py`. **`compare_run_reports.py`** uses `run_report/parse.py` and diffs config (omitting **`HEAD_DIM`** via `DERIVED_EXPERIMENT_CFG_KEYS`), final loss, and sample lines; **`experiments/report_generator.py`** parses the same format into an HTML sweep summary (same omission + derived footnote).

When adding features, keep the **no third-party dependencies** rule unless the project owner explicitly changes that.

## Testing

Run **`pytest`** from the repo root: `python -m pytest tests/ -q`. See `docs/M2-semantic-quality.md` for the semantic-quality / run-report slice log.

## Git and docs

- **`README.md`**: User-facing overview, architecture diagram, configuration tables, and pointers to this file.
- **`CLAUDE.md`** (this file): project context for Claude Code and compatible assistants—read it when onboarding or before substantive edits.
- **`AGENT.md`**: short pointer for agent harnesses; it defers to `CLAUDE.md` for full detail.

## Notes for assistants

- Prefer **`microgpt_updated.py`** plus **`mgpt/`** for readability and for edits that need clear function boundaries; use **`microgpt.py`** when the user wants a minimal diff against the “canonical” one-file narrative. Report format changes belong in **`run_report/`**.
- Do not strip educational comments in `microgpt_updated.py` without the user asking; they are part of the artifact.
- Large `input.txt` may be user-specific data; do not assume it is only the default names file.
