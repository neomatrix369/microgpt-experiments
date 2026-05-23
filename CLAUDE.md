# microgpt

A minimal, dependency-free **character-level GPT** in pure Python: scalar autograd (`Value`), a tiny transformer (embeddings, multi-head self-attention, MLP, RMSNorm), Adam, and name generation. Based on the [microGPT / makemore](https://github.com/karpathy/makemore) style exercises ([write-up](https://karpathy.github.io/2026/02/12/microgpt/)).

**Runtime** is stdlib-only (Python 3). **`requirements.txt`** lists optional **test** deps (`pytest`, `pytest-cov`); no `pyproject.toml`.

## Repository layout

| Path | Role |
|------|------|
| `QUICKSTART.md` | **Fastest first run** — Python check, one command, links to experiment and learning docs. |
| `docs/README.md` | **Documentation index** — guides organized by persona (learner, workshop, explorer, experimenter, contributor). |
| `microgpt.py` | Compact “single story” version: one script, global state, matches the original blog-style walkthrough. |
| `microgpt_updated.py` | Refactored entry: hyperparameters (module constants, overridable via **`argparse`** CLI), `main()` → **`mgpt.experiment.run_experiment()`**, and richer comments. Imports **`mgpt`** (autograd, transformer forward, data, experiment API) and **`run_report`** (on-disk report format). Prefer this for changes that need structure or tests. |
| `mgpt/` | Package: `Value` (scalar autograd), ops (`linear`, `softmax`, `rmsnorm`, `make_matrix`), transformer step `gpt()`, dataset + `Tokeniser` (`load_dataset`, `build_tokeniser`), **`experiment.py`** (`RunConfig`, `run_experiment()`), **`evaluation.py`** (tier/heuristic sample scoring), **`quality.py`** (overall score formula, baseline compare, `SWEEP_RANKING_METRIC` for grid sweep). Stdlib only. |
| `run_report/` | Package: parse/compare fields of `output_*.txt` (`parse.py`), narrative (`narrative.py`), **`paths.py`** (`run_reports_dir`, `DEFAULT_RUN_REPORT_DIR`), full report assembly (`builder.py`), text loss comparison grids (`text_loss_plot.py`), **`timing.py`** (UTC + local ISO, duration, live elapsed/ETA helpers, sweep timing file). Shared by `microgpt_updated.py`, `mgpt/experiment.py`, `annotate_run_reports.py`, `compare_run_reports.py`, and `experiments/report_generator.py`. |
| `experiments/sweep.py` | Grid-search CLI: numbered JSON configs in **`experiments/configs/`** (`0_sweep-smoke-test.json` … `4_sweep-full.json`); calls **`mgpt.experiment.run_experiment()`**, writes **`sweep_summary.csv`** (per-run timing columns) and **`sweep_timing.txt`** (whole-grid wall clock) under **`outputs/sweeps/<order>-<name>/`**, ranks by **`OVERALL_QUALITY_SCORE`**. Console shows sweep elapsed/ETA. **`--list-configs`**, **`--dry-run`**, **`--summarize-only`**, **`--html`**. Grid expansion helpers in **`experiments/sweep_grid.py`**. |
| `experiments/configs/` | Numbered sweep JSON files + **`README.md`** (recommended run order). |
| `experiments/report_generator.py` | `argparse` CLI: reads one or more `output_*.txt` files (default: glob under `outputs/` at repo root), writes HTML (`-o`, default `outputs/comparison_report.html`): shared/varying training config (same key order as `compare_run_reports.py`: **`N_EMBD` → `N_HEAD` → `HEAD_DIM`**, then the rest; `HEAD_DIM` annotated as calculated), quality table, **run timing** (duration, start/end UTC and local when present), aligned samples (2+ runs), loss ASCII (`--loss-bins`, `--loss-height`), tier bars. |
| `requirements.txt` | Optional test-only deps (`pytest`, `pytest-cov`). Training/generation need no pip install. |
| `pytest.ini` / `.coveragerc` | Test runner + terminal coverage summary after `python -m pytest`. |
| `tests/` | `pytest` tests for evaluation, quality, experiment API, sweep grid, run timing, paths, report builder/parse round-trip, loss-plot helpers, and `report_generator` HTML output. |
| `annotate_run_reports.py` | Inserts the same `--- What this run is ---` narrative into **existing** `output_*.txt` reports (stdlib-only backfill for past experiments). |
| `compare_run_reports.py` | Compares two `output_*.txt` files: parsed config keys in **`run_report.parse._CFG_DISPLAY_ORDER`** ( **`HEAD_DIM` immediately after `N_EMBD` and `N_HEAD`**, with a short “calculated from …” note), final loss, ordered inference samples, and an informational **`--- Run timing ---`** section (UTC + local start/end, duration, timezone; filename timestamp as fallback for legacy reports). Stdlib-only; exit `0` / `1` / `2` (match / diff / error; timing does not affect exit code). |
| `input.txt` | One document per line (e.g. names). If missing, `microgpt.py` / `microgpt_updated.py` can download the classic names dataset from the makemore repo. |
| `example-experiments/` | **Tracked** demo set: two full **`output_*.txt`** run reports (H4 vs H1 @ 1000 steps), a saved **`compare_run_reports.py`** transcript, and **`comparison_report.html`**. See README [Example artifacts (preview)](#example-artifacts-preview). Local sweeps still use gitignored **`outputs/`**. |
| `outputs/` | Default directory for `output_*.txt` run reports and `comparison_report.html` (`run_report.paths.DEFAULT_RUN_REPORT_DIR`); gitignored. |
| `output_*.txt` | Optional artifacts written by `microgpt_updated.py` (not by `microgpt.py`), default path **`outputs/`** + stem: hyperparameters, narrative, optional experiment-suite labels, samples, glossary. Filename encodes `L/E/H/B/S/T/seed` (no `D` token; per-head width is not a filename field) plus `_YYYYMMDD_HHMMSS` (see `format_run_output_path()` / `format_run_output_path_for_params()`). **Config block order:** `N_EMBD`, `N_HEAD`, then **`HEAD_DIM=`** (`N_EMBD // N_HEAD`) and a **`#`** note — not an independent sweep knob. |

## How to run

```bash
# Refactored entry (recommended)
python microgpt_updated.py

# Refactored entry: override hyperparameters for this run (see --help)
python microgpt_updated.py --help

# Original compact script
python microgpt.py
```

**How to use microgpt_updated.py** (baseline, `--help`, `--input`, source-only vs CLI, sweep reproduction, report tools): **`README.md` → [How to use microgpt_updated.py](README.md#how-to-use-microgpt_updatedpy)**. **First run:** **`QUICKSTART.md`**. **Doc index:** **`docs/README.md`**. **Concepts:** **`docs/learn-before-you-code.md`**. **Autograd:** **`docs/autograd-deep-dive.md`** → `mgpt/value.py`. **Experiments:** **`docs/experiment-workflow.md`**.

**Run experiments examples** (distinct `N_HEAD` × `NUM_STEPS` from saved `output_*.txt`): **`README.md` → [Run experiments examples](README.md#run-experiments-examples)** and **`docs/M2-semantic-quality.md`** (Commands → *Run experiments examples*).

Expect stdout: dataset size, vocab size, parameter count, training loss per step with **live `elapsed … | ETA …`** on the same carriage-return line (padded so ETA is not clipped), a **`Run wall clock:`** summary after training, then 20 sampled “hallucinated” names. **`microgpt_updated.py` also writes** a run report under **`run_reports_dir(<repo root>)`** / `outputs/` (default from `format_run_output_path()`; **`--output-dir`** overrides; directory created if missing). **`microgpt.py` does not** write that file. **`microgpt_updated.py`** is the only script with a CLI; flags map to the same symbols documented in `README.md` (e.g. `--num-steps` → `NUM_STEPS`). Experiment-suite fields (`--suite-index`, `--suite-total`, `--suite-note`) use `argparse.SUPPRESS` so omitting them leaves any values you set in the source file unchanged for that run.

**Backfill narrative on old reports** (same text as new runs, parsed from the config block):

```bash
python annotate_run_reports.py
python annotate_run_reports.py path/to/output_L1_....txt
```

**Compare two saved reports** (config + final loss + inference samples + informational run timing; config keys follow **`_CFG_DISPLAY_ORDER`** so **`HEAD_DIM` is always after `N_EMBD` and `N_HEAD`**; ignores narrative, glossary, quality blocks, loss-history CSV, and timing for *equality*, but prints **text loss graphs** when both files include `--- Loss history (CSV: step,loss) ---`):

```bash
python compare_run_reports.py outputs/output_A.txt outputs/output_B.txt
python compare_run_reports.py outputs/output_A.txt outputs/output_B.txt --loss-bins 96 --loss-height 14
```

**HTML comparison** of one or more reports (defaults to all `outputs/output_*.txt`):

```bash
python experiments/report_generator.py
python experiments/report_generator.py outputs/a.txt outputs/b.txt -o outputs/comparison_report.html
```

**Grid sweep** (numbered configs under `experiments/configs/`):

```bash
python experiments/sweep.py --list-configs
python experiments/sweep.py --config experiments/configs/1_sweep-minimal.json --dry-run
python experiments/sweep.py --config experiments/configs/1_sweep-minimal.json
```

Exit codes (`compare_run_reports.py`): `0` all match, `1` some field or sample differs, `2` bad args or parse failure. `report_generator.py` exits `2` if no input files or render error. `sweep.py` exits `2` on config/parse errors.

**Runtime**: Training is 1000 steps by default and is **slow** (scalar autograd in Python). That is expected.

## Conventions and internals

- **Autograd**: `Value` in `mgpt/value.py` implements forward ops with local gradients; `loss.backward()` walks the graph. No PyTorch.
- **Model**: GPT-2–like stack with **RMSNorm** (not LayerNorm), **no biases**, **ReLU** in the MLP (not GELU). KV caches for attention are part of the live graph during training (not detached). Forward step: `mgpt/model.py` (`gpt()`).
- **Data**: Character-level tokens; a special **BOS** (beginning-of-sequence) token wraps each line. `BLOCK_SIZE` limits context; sampling stops at BOS or max length.
- **Hyperparameters**: In `microgpt.py` they are module-level names (`n_layer`, `n_embd`, `n_head`, `head_dim`, `block_size`, …) with **`head_dim` after `n_embd` and `n_head`**. In `microgpt_updated.py` they are `N_LAYER`, `N_EMBD`, `N_HEAD`, `HEAD_DIM`, `BLOCK_SIZE`, … at the top of the file (same relative order for the width/attention trio), with optional **CLI overrides** parsed in `main()` (`_build_arg_parser()`, `_apply_parsed_args()`, `_validate_hyperparameters()`).
- **Run reports** (`microgpt_updated.py` via **`mgpt.experiment.run_experiment()`**): After training and generation, `compute_sample_quality_metrics()` supplies character and semantic-tier stats; the report is assembled with `run_report/builder.py` including `--- What this run is ---` (input file + training summary + what loss and samples mean), then optional `--- Experiment suite ---` if any of `EXPERIMENT_SUITE_INDEX`, `EXPERIMENT_SUITE_TOTAL`, or `EXPERIMENT_SUITE_NOTE` is set, then optional **`--- Run timing ---`** (UTC + local ISO start/end, `DURATION_SECONDS`, `TIMEZONE`; authoritative wall clock — distinct from the approximate local timestamp embedded in the filename), then **config** with **`N_EMBD` → `N_HEAD` → `HEAD_DIM=`** (then `BLOCK_SIZE` and the rest), a **`#`** line noting `HEAD_DIM` is derived, final loss, optional `--- Sample quality (character-level) ---` and `--- Semantic quality (three-tier) ---`, optional `--- Loss history (CSV: step,loss) ---` (from the in-memory `loss_history` list), inference samples, and a parameter glossary. The narrative is implemented in `run_report/narrative.py` (`format_run_narrative_lines`) and shared with `annotate_run_reports.py`. **`compare_run_reports.py`** and **`experiments/report_generator.py`** use `run_report/parse.py` (`cfg_keys_for_experiment_table` / `_CFG_DISPLAY_ORDER`) so **`HEAD_DIM` never sorts ahead of `N_EMBD` or `N_HEAD`**; `experiment_cfg_calculated_caption` labels it as calculated in CLI/HTML output.

When adding features, keep **runtime** code **stdlib-only** unless maintainers explicitly add third-party packages to the training path; test-only deps stay in `requirements.txt`.

## Testing

Run **`pytest`** from the repo root: `pip install -r requirements.txt` then `python -m pytest` (`pytest.ini` + `.coveragerc` print a terminal **coverage summary** for `mgpt/`, `run_report/`, `experiments/` — informational, no fail threshold). Covers evaluation, quality hub, experiment API, sweep grid, **run timing**, paths, text loss plots, HTML report generator. See `docs/M2-semantic-quality.md` for the semantic-quality slice log.

## Git and docs

- **`README.md`**: User-facing overview, **Who is this for?** personas, architecture, configuration, run reports.
- **`QUICKSTART.md`**: Minimal first run (workshop / playgroup entry).
- **`docs/README.md`**: Documentation index by persona and task.
- **`docs/learn-before-you-code.md`**: Beginner-friendly concepts and examples—read before diving into code.
- **`docs/autograd-deep-dive.md`**: Autograd learning guide (`Value`, graph, `backward()`, worked examples, diagrams); read before `mgpt/value.py`.
- **`docs/experiment-workflow.md`**: Train runs, save reports, compare with CLI or HTML; grid sweep and run timing.
- **`docs/M2-semantic-quality.md`**: Sample quality guide (semantic tier metrics; human-readable + implementation notes).
- **`experiments/configs/README.md`**: Numbered JSON grid sweep configs (`0_sweep-smoke-test.json` … `4_sweep-full.json`).
- **`CLAUDE.md`** (this file): project context for Claude Code and compatible assistants—read it when onboarding or before substantive edits.
- **`AGENTS.md`**: short pointer for agent harnesses; it defers to `CLAUDE.md` for full detail.

## Notes for assistants

- Prefer **`microgpt_updated.py`** plus **`mgpt/`** for readability and for edits that need clear function boundaries; use **`microgpt.py`** when the user wants a minimal diff against the “canonical” one-file narrative. Report format changes belong in **`run_report/`**.
- Do not strip educational comments in `microgpt_updated.py` without the user asking; they are part of the artifact.
- Large `input.txt` may be user-specific data; do not assume it is only the default names file.
