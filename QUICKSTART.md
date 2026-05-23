# Quickstart

Get microgpt running in a few minutes. **Train and generate:** Python 3 only. **Tests:** optional — `pip install -r requirements.txt`.

> Once this works, head to [README.md](README.md) for architecture, experiments, and run reports — or pick a path from [Who is this for?](README.md#who-is-this-for) / [docs/README.md](docs/README.md).

---

## 1. Python

```bash
python --version   # Python 3.x required
```

**Tests** (optional): `pip install -r requirements.txt` then `python -m pytest`.

---

## 2. Run training + generation

From the repo root:

```bash
python microgpt_updated.py
```

**Expect (this takes a while — scalar autograd is slow on purpose):**

1. Dataset and vocabulary sizes printed once.
2. Training loss updating on one line (~1000 steps by default) with live **`elapsed … | ETA …`**, then **`Run wall clock:`** after training.
3. **20 generated name-like lines** and a sample-quality summary.
4. A run report saved under **`outputs/output_*.txt`** (includes **`--- Run timing ---`** when using current code).

If `input.txt` is missing, the script downloads the classic names dataset automatically.

**Compact one-file version** (console only, no saved report):

```bash
python microgpt.py
```

---

## 3. Smoke check without waiting for training

Browse checked-in artifacts — no GPU, no long run:

| Open this | What it is |
|-----------|------------|
| [`example-experiments/output_L1_E16_H4_B16_S1000_….txt`](example-experiments/output_L1_E16_H4_B16_S1000_T0p5_seed42_20260424_152649.txt) | Full run report (4 heads; pre-**`--- Run timing ---`** era) |
| [`example-experiments/comparison_report.html`](example-experiments/comparison_report.html) | HTML comparison (4-head vs 1-head) |
| [`example-experiments/sweep-1-minimal/sweep_summary.csv`](example-experiments/sweep-1-minimal/sweep_summary.csv) | Finished **`1_sweep-minimal.json`** ranked CSV (4 runs) |
| [docs/experiment-workflow.md → sweep console excerpt](docs/experiment-workflow.md#example-finished-1_sweep-minimaljson-run) | Head/tail of a real grid sweep run |

---

## 4. Override one setting (optional)

```bash
python microgpt_updated.py --help
python microgpt_updated.py --n-head 1 --num-steps 50   # quick ablation
python microgpt_updated.py --temperature 0.8
```

---

## 5. Grid sweep (optional)

Try the experimentation platform **without** a long training run:

```bash
# List numbered configs (0 = smoke test, 1–4 = real sweeps)
python experiments/sweep.py --list-configs

# Preview combinations only
python experiments/sweep.py --config experiments/configs/1_sweep-minimal.json --dry-run

# Quick pipeline check: 2 runs × 5 steps (~seconds)
python experiments/sweep.py --config experiments/configs/0_sweep-smoke-test.json
```

Outputs land in **`outputs/sweeps/0-smoke-test/`** (`sweep_summary.csv` with per-run timing columns, **`sweep_timing.txt`** for whole-grid wall clock, plus ranked table vs the H4 @ 1000 baseline). A finished **`1_sweep-minimal.json`** example (console excerpt + checked-in CSV) is in **[docs/experiment-workflow.md](docs/experiment-workflow.md#example-finished-1_sweep-minimaljson-run)** and **[example-experiments/sweep-1-minimal/](example-experiments/sweep-1-minimal/)**.

---

## Next steps

| If you want to… | Go to |
|-----------------|-------|
| **Learn concepts** before reading code | [docs/learn-before-you-code.md](docs/learn-before-you-code.md) |
| **Run and compare experiments** | [docs/experiment-workflow.md](docs/experiment-workflow.md) |
| **Grid sweep (JSON configs)** | [docs/experiment-workflow.md#grid-sweep-automated-search](docs/experiment-workflow.md#grid-sweep-automated-search) · [experiments/configs/README.md](experiments/configs/README.md) |
| **Understand autograd** | [docs/autograd-deep-dive.md](docs/autograd-deep-dive.md) |
| **Full reference** | [README.md](README.md) |
| **All docs by persona** | [docs/README.md](docs/README.md) |

Typical experiment pattern:

```bash
python microgpt_updated.py
python microgpt_updated.py --n-head 1
python compare_run_reports.py outputs/output_….txt outputs/output_….txt
```

See [docs/experiment-workflow.md](docs/experiment-workflow.md) for the full recipe (including HTML).
