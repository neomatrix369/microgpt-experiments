# Grid sweep configs

Numbered JSON files define **recommended run order** (minimal → full). Each file is passed to `experiments/sweep.py` with `--config`.

| Order | File | Grid axes | Valid runs (approx.) |
|------:|------|-----------|----------------------|
| 0 | [`0_sweep-smoke-test.json`](./0_sweep-smoke-test.json) | `n_head` × 5 fixed steps | 2 (**pipeline check only**) |
| 1 | [`1_sweep-minimal.json`](./1_sweep-minimal.json) | `n_head` × `num_steps` | 4 |
| 2 | [`2_sweep-arch.json`](./2_sweep-arch.json) | `n_layer` × `n_embd` × `n_head` | 12 |
| 3 | [`3_sweep-arch-steps.json`](./3_sweep-arch-steps.json) | arch + `num_steps` | 6 |
| 4 | [`4_sweep-full.json`](./4_sweep-full.json) | steps + `temperature` + `learning_rate` | 18 |

Run in order when learning the sweep tool; start with **`0_sweep-smoke-test.json`** to verify the pipeline, then **`1_`** when ready for real training runs.

```bash
# Pipeline check (~seconds)
python experiments/sweep.py --config experiments/configs/0_sweep-smoke-test.json

# Step 1 — production minimal sweep (slow)
python experiments/sweep.py --config experiments/configs/1_sweep-minimal.json --dry-run
python experiments/sweep.py --config experiments/configs/1_sweep-minimal.json

# Step 2 — when ready for architecture search
python experiments/sweep.py --config experiments/configs/2_sweep-arch.json --dry-run
```

Outputs land under `outputs/sweeps/<order>-<name>/` (e.g. `outputs/sweeps/1-minimal/`):

| Artifact | Contents |
|----------|----------|
| **`output_*.txt`** | One run report per grid point (config, quality, samples, **`--- Run timing ---`**) |
| **`sweep_summary.csv`** | Ranked rows with quality metrics, baseline deltas, and per-run timing columns (`started_utc`, `started_local`, `ended_utc`, `ended_local`, `duration_seconds`, `timezone`) |
| **`sweep_timing.txt`** | Whole-grid wall clock (`SWEEP_*` fields: UTC + local start/end, duration, timezone) |
| **`comparison_report.html`** | Optional — pass **`--html`** to **`--summarize-only`** or build manually with **`report_generator.py`** |

See [`docs/experiment-workflow.md`](../../docs/experiment-workflow.md#grid-sweep-automated-search), **[Run timing](../../docs/experiment-workflow.md#run-timing-and-progress)**, [`docs/README.md`](../../docs/README.md), and [`docs/M2-semantic-quality.md`](../../docs/M2-semantic-quality.md).

**Quality scoring:** sweeps rank by `overall_quality_score` (simple heuristics). All compare/rank logic is centralized in **`mgpt/quality.py`**; tier rules are in **`mgpt/evaluation.py`**. To use your own objective, edit those files — see **`docs/M2-semantic-quality.md`**.

The optional `"order"` field inside each JSON mirrors the filename prefix; loaders ignore it (metadata only).
