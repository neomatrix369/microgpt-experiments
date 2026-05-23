# Documentation index

All guides for **microgpt**, organized by **who you are** and **what you want to do**.

**Repo entry:** [README.md](../README.md) · **Fastest run:** [QUICKSTART.md](../QUICKSTART.md)

---

## Who is this for?

| Persona | Start here | Then |
|---------|------------|------|
| **Learner / student** | [learn-before-you-code.md](./learn-before-you-code.md) | [autograd-deep-dive.md](./autograd-deep-dive.md) → `mgpt/value.py` → `microgpt.py` |
| **Workshop / playgroup attendee** | [QUICKSTART.md](../QUICKSTART.md) | [experiment-workflow.md](./experiment-workflow.md) → [README Run experiments](../README.md#run-experiments-examples) |
| **Curious explorer** (no training) | [example-experiments/](../example-experiments/) | Open `comparison_report.html` in a browser |
| **Experimenter** (sweeps & compare) | [experiment-workflow.md](./experiment-workflow.md) | Grid: `0_sweep-smoke-test.json` … `4_sweep-full.json` · tiers: [M2-semantic-quality.md](./M2-semantic-quality.md) |
| **Contributor / extender** | [CLAUDE.md](../CLAUDE.md) | [README repository layout](../README.md#repository-layout) |

---

## Guides (by topic)

| Doc | What it covers |
|-----|----------------|
| [learn-before-you-code.md](./learn-before-you-code.md) | Tokens, BOS, loss, temperature, transformer map, sample tiers — **read before code** |
| [autograd-deep-dive.md](./autograd-deep-dive.md) | `Value`, computation graph, `backward()`, worked examples, diagrams |
| [experiment-workflow.md](./experiment-workflow.md) | Train → `outputs/` → CLI diff or HTML — **linear experiment recipe** |
| [M2-semantic-quality.md](./M2-semantic-quality.md) | Sample quality guide — real / plausible / nonsense tiers, metrics, commands |

---

## Common tasks → command

| Task | Command / path |
|------|----------------|
| Default training run | `python microgpt_updated.py` |
| Head-count sweep (manual CLI) | See [experiment-workflow.md](./experiment-workflow.md#canonical-sweep-commands) |
| Grid sweep (JSON configs) | `python experiments/sweep.py --list-configs` · smoke: `0_sweep-smoke-test.json` |
| Diff two runs | `python compare_run_reports.py outputs/a.txt outputs/b.txt` |
| HTML table (2+ runs) | `python experiments/report_generator.py` |
| Browse demo results | [example-experiments/comparison_report.html](../example-experiments/comparison_report.html) |
| All CLI flags | `python microgpt_updated.py --help` |

**Compare tools:** terminal diff = `compare_run_reports.py` · browser table = `experiments/report_generator.py`

---

## Suggested journeys

### Learn the system (≈1–2 hours reading)

```text
learn-before-you-code.md → autograd-deep-dive.md → mgpt/value.py → microgpt_updated.py train()
```

### Run a workshop experiment (≈hours of CPU time)

```text
QUICKSTART.md → experiment-workflow.md → compare or HTML → M2-semantic-quality.md (read scores)
```

### Skim without running

```text
example-experiments/output_….txt → comparison_report.html → README architecture section
```

---

## Maintainer / agent context

- [CLAUDE.md](../CLAUDE.md) — conventions, file layout, which script to edit
- [AGENTS.md](../AGENTS.md) — short pointer for agent harnesses
