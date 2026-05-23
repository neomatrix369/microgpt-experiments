# Documentation index

All guides for **microgpt**, organized by **who you are** and **what you want to do**.

**Repo entry:** [README.md](../README.md) · **Fastest run:** [QUICKSTART.md](../QUICKSTART.md)

**Train and generate:** Python 3 only. **Tests:** optional — `pip install -r requirements.txt`.

> **Who is this for?** Same personas as [README → Who is this for?](../README.md#who-is-this-for) — this page is the **doc map**; the README is the full reference.

---

## Who is this for?

| Persona | Start here | Then |
|---------|------------|------|
| **Learner / student** | [learn-before-you-code.md](./learn-before-you-code.md) | [autograd-deep-dive.md](./autograd-deep-dive.md) → `mgpt/value.py` → `mgpt/experiment.py` |
| **Workshop / playgroup attendee** | [QUICKSTART.md](../QUICKSTART.md) | [experiment-workflow.md](./experiment-workflow.md) → smoke sweep `0_sweep-smoke-test.json` |
| **Curious explorer** (no training) | [example-experiments/](../example-experiments/) | Open `comparison_report.html` in a browser |
| **Experimenter** (sweeps & compare) | [experiment-workflow.md](./experiment-workflow.md) | [configs/README.md](../experiments/configs/README.md) · [M2-semantic-quality.md](./M2-semantic-quality.md) · [Run timing](./experiment-workflow.md#run-timing-and-progress) |
| **Autograd-focused reader** | [autograd-deep-dive.md](./autograd-deep-dive.md) | `mgpt/value.py` → `ops.py` → `mgpt/experiment.py` `train()` |
| **Contributor / extender** | [CLAUDE.md](../CLAUDE.md) | [README repository layout](../README.md#repository-layout) |

---

## Guides (by topic)

| Doc | What it covers |
|-----|----------------|
| [learn-before-you-code.md](./learn-before-you-code.md) | Tokens, BOS, loss, temperature, transformer map, sample tiers — **read before code** |
| [autograd-deep-dive.md](./autograd-deep-dive.md) | `Value`, computation graph, `backward()`, worked examples, diagrams |
| [experiment-workflow.md](./experiment-workflow.md) | Train → `outputs/` → CLI diff, HTML, or grid sweep — **linear experiment recipe** (timing + sweep artifacts) |
| [M2-semantic-quality.md](./M2-semantic-quality.md) | Sample quality guide — tiers, `mgpt/quality.py`, extending scoring, sweep ranking |
| [experiments/configs/README.md](../experiments/configs/README.md) | Numbered JSON grid configs (`0_sweep-smoke-test.json` … `4_sweep-full.json`) |

---

## Common tasks → command

| Task | Command / path |
|------|----------------|
| Default training run | `python microgpt_updated.py` |
| Head-count sweep (manual CLI) | [experiment-workflow.md → Canonical sweep commands](./experiment-workflow.md#canonical-sweep-commands) |
| Grid sweep (JSON configs) | `python experiments/sweep.py --list-configs` · smoke: `0_sweep-smoke-test.json` |
| Diff two runs (config, loss, samples; timing display) | `python compare_run_reports.py outputs/a.txt outputs/b.txt` |
| HTML table (2+ runs; timing columns when present) | `python experiments/report_generator.py` |
| Wall-clock in a saved report | `--- Run timing ---` block in `outputs/output_*.txt` |
| Sweep timing summary | `outputs/sweeps/<order>-<name>/sweep_timing.txt` |
| Browse demo results | [example-experiments/comparison_report.html](../example-experiments/comparison_report.html) |
| Run tests (optional) | `pip install -r requirements.txt` then `python -m pytest` |
| All CLI flags | `python microgpt_updated.py --help` |

**Compare tools:** terminal diff = `compare_run_reports.py` (timing shown; not part of exit code) · browser table = `experiments/report_generator.py` · grid search = `experiments/sweep.py`

---

## Suggested journeys

### Learn the system (≈1–2 hours reading)

```text
learn-before-you-code.md → autograd-deep-dive.md → mgpt/value.py → mgpt/experiment.py
```

### Run a workshop experiment (≈hours of CPU time)

```text
QUICKSTART.md → experiment-workflow.md → compare or HTML → M2-semantic-quality.md
```

Optional grid path:

```text
QUICKSTART §5 → 0_sweep-smoke-test.json → 1_sweep-minimal.json → sweep_summary.csv
```

### Skim without running

```text
example-experiments/output_….txt → comparison_report.html → README architecture section
```

---

## Maintainer / agent context

- [CLAUDE.md](../CLAUDE.md) — conventions, file layout, which script to edit
- [AGENTS.md](../AGENTS.md) — short pointer for agent harnesses
