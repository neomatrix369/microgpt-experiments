# microgpt

A minimal, dependency-free **character-level GPT** in pure Python: scalar autograd (`Value`), a tiny transformer (embeddings, multi-head self-attention, MLP, RMSNorm), Adam, and name generation. Based on the [microGPT / makemore](https://github.com/karpathy/makemore) style exercises ([write-up](https://karpathy.github.io/2026/02/12/microgpt/)).

There is **no** `requirements.txt` or `pyproject.toml` by design: only the Python 3 standard library.

## Repository layout

| Path | Role |
|------|------|
| `microgpt.py` | Compact “single story” version: one script, global state, matches the original blog-style walkthrough. |
| `microgpt_updated.py` | Refactored entry: hyperparameters, `train()` / `generate()` / `main()`, `save_run_report()`, and richer comments. Imports the **`mgpt`** package (autograd, transformer forward, data) and **`run_report`** (on-disk report format). Prefer this for changes that need structure or tests. |
| `mgpt/` | Package: `Value` (scalar autograd), ops (`linear`, `softmax`, `rmsnorm`, `make_matrix`), transformer step `gpt()`, dataset + `Tokeniser` (`load_dataset`, `build_tokeniser`). Stdlib only. |
| `run_report/` | Package: parse/compare fields of `output_*.txt`, narrative (`format_run_narrative_lines` in `narrative.py`), path encoding (`format_run_output_path_for_params` in `paths.py`), full report assembly (`build_run_report_lines` in `builder.py`). Shared by `microgpt_updated.py`, `annotate_run_reports.py`, and `compare_run_reports.py`. |
| `annotate_run_reports.py` | Inserts the same `--- What this run is ---` narrative into **existing** `output_*.txt` reports (stdlib-only backfill for past experiments). |
| `compare_run_reports.py` | Compares two `output_*.txt` files: parsed config keys, final loss, ordered inference samples. Stdlib-only; exit `0` / `1` / `2` (match / diff / error). |
| `input.txt` | One document per line (e.g. names). If missing, `microgpt.py` / `microgpt_updated.py` can download the classic names dataset from the makemore repo. |
| `output_*.txt` | Optional artifacts written by `microgpt_updated.py` (not by `microgpt.py`): hyperparameters, a short narrative of input/training/output, optional experiment-suite labels, samples, and a parameter glossary. Filename encodes `L/E/H/D/B/S/T/seed` plus `_YYYYMMDD_HHMMSS` (local time when the path is built; see `format_run_output_path()` in this entry script and `format_run_output_path_for_params()` in `run_report/paths.py`). |

## How to run

```bash
# Refactored entry (recommended)
python microgpt_updated.py

# Original compact script
python microgpt.py
```

Expect stdout: dataset size, vocab size, parameter count, training loss per step, then 20 sampled “hallucinated” names. **`microgpt_updated.py` also writes** a run report file under the current directory (default name from `format_run_output_path()`). **`microgpt.py` does not** write that file.

**Backfill narrative on old reports** (same text as new runs, parsed from the config block):

```bash
python annotate_run_reports.py
python annotate_run_reports.py path/to/output_L1_....txt
```

**Compare two saved reports** (config + final loss + inference samples; ignores narrative and glossary):

```bash
python compare_run_reports.py path/to/output_A.txt path/to/output_B.txt
```

Exit codes: `0` all match, `1` some field or sample differs, `2` bad args or parse failure.

**Runtime**: Training is 1000 steps by default and is **slow** (scalar autograd in Python). That is expected.

## Conventions and internals

- **Autograd**: `Value` in `mgpt/value.py` implements forward ops with local gradients; `loss.backward()` walks the graph. No PyTorch.
- **Model**: GPT-2–like stack with **RMSNorm** (not LayerNorm), **no biases**, **ReLU** in the MLP (not GELU). KV caches for attention are part of the live graph during training (not detached). Forward step: `mgpt/model.py` (`gpt()`).
- **Data**: Character-level tokens; a special **BOS** (beginning-of-sequence) token wraps each line. `BLOCK_SIZE` limits context; sampling stops at BOS or max length.
- **Hyperparameters**: In `microgpt.py` they are module-level names (`n_layer`, `n_embd`, …). In `microgpt_updated.py` they are `N_LAYER`, `N_EMBD`, `BLOCK_SIZE`, … at the top of the file.
- **Run reports** (`microgpt_updated.py` only): After training, `save_run_report()` writes a text report including `--- What this run is ---` (input file + training summary + what loss and samples mean), then optional `--- Experiment suite ---` if any of `EXPERIMENT_SUITE_INDEX`, `EXPERIMENT_SUITE_TOTAL`, or `EXPERIMENT_SUITE_NOTE` is set, then config, final loss, inference samples, and a parameter glossary. The narrative is implemented in `run_report/narrative.py` (`format_run_narrative_lines`) and shared with `annotate_run_reports.py`. **`compare_run_reports.py`** uses `run_report/parse.py` and diffs structured fields plus sample lines for quick regression checks.

When adding features, keep the **no third-party dependencies** rule unless the project owner explicitly changes that.

## Testing

There is no automated test suite in-repo yet. If you add one, `pytest` is a reasonable default; document the command in `README.md` and, for assistants, here.

## Git and docs

- **`README.md`**: User-facing overview, architecture diagram, configuration tables, and pointers to this file.
- **`CLAUDE.md`** (this file): project context for Claude Code and compatible assistants—read it when onboarding or before substantive edits.
- **`AGENT.md`**: short pointer for agent harnesses; it defers to `CLAUDE.md` for full detail.

## Notes for assistants

- Prefer **`microgpt_updated.py`** plus **`mgpt/`** for readability and for edits that need clear function boundaries; use **`microgpt.py`** when the user wants a minimal diff against the “canonical” one-file narrative. Report format changes belong in **`run_report/`**.
- Do not strip educational comments in `microgpt_updated.py` without the user asking; they are part of the artifact.
- Large `input.txt` may be user-specific data; do not assume it is only the default names file.
