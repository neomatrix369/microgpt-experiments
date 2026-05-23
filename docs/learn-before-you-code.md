# Learn before you code

This guide is for **reading first**, before you open `microgpt.py` or `microgpt_updated.py`. It explains the ideas in everyday language, with small examples you can picture in your head.

**Technical reference** (commands, file layout, config tables) stays in [`README.md`](../README.md).  
**Fastest first run:** [`QUICKSTART.md`](../QUICKSTART.md). **All docs by persona:** [`docs/README.md`](./README.md).  
**Run and compare experiments:** [`experiment-workflow.md`](./experiment-workflow.md).  
**Autograd** (gradients, `Value`, `backward()`): [`autograd-deep-dive.md`](./autograd-deep-dive.md).  
**Sample-quality details** (how tiers are scored): [`M2-semantic-quality.md`](./M2-semantic-quality.md).

---

## What this project does (one sentence)

It trains a tiny model to **guess the next character** in a line of text, then **rolls the dice** to write new lines that look like the training data.

The default demo uses **first names** (one name per line in `input.txt`), so the model learns patterns like “names often start with a vowel” or “`th` and `an` show up a lot”—not English grammar in the abstract, but **statistics of your file**.

---

## A concrete example: one name

Suppose your file contains:

```text
emma
liam
noah
```

**Character-level** means the model sees individual letters, not whole words as single tokens.

Each line is wrapped with a special **BOS** (beginning-of-sequence) marker so the model learns **where a line starts and when to stop**:

```text
<BOS> e m m a <BOS>
<BOS> l i a m <BOS>
...
```

At training time, the model is asked: “Given `<BOS> e m m`, what letter comes next?” Answer: `a`. Then: “Given `<BOS> e m m a`, what comes next?” Answer: `<BOS>` (end of line).

That is **next-token prediction**. A GPT is exactly that, repeated at every position, with a neural network choosing the probabilities.

---

## Training vs generation

| Phase | What happens | Simple analogy |
|-------|----------------|----------------|
| **Training** | Pick a random line, show the model prefixes, compare its guesses to the true next character, nudge weights to reduce error. Repeat for many **steps**. | A student doing flashcards: “After `em`, what letter?” — wrong answers get corrected. |
| **Generation** | Start from `<BOS>`, sample one letter from the model’s probabilities, append it, repeat until `<BOS>` again or max length. | Rolling a loaded die whose faces are letters; the weights came from training. |

After training, **`microgpt_updated.py`** prints **20 generated lines** and saves a **run report** under `outputs/`. Those lines are **not copied** from the file—they are **new** strings drawn from what the model learned.

---

## Loss: did the guesses get better?

**Loss** is a single number summarizing “how surprised the model was” by the correct next characters during training.

- **High loss early** → guesses are poor (expected at step 0).
- **Loss going down** → the model fits the training patterns better.
- **Lower is better** on this objective—but very low loss can mean **memorizing** short lines, not generalizing.

Example from a real 1000-step run (4 attention heads):

```text
Final loss (last training step): 2.649694
```

You do not need the exact formula yet. Treat loss as a **speedometer** while training: watch it trend downward, then compare runs with the same seed and data.

---

## Temperature: how adventurous is sampling?

When generating, the model outputs **logits** (raw scores per character). **Temperature** scales them before turning them into probabilities:

- **Low temperature (e.g. 0.5, the default)** → sharper choices, more “typical” letters for the context.
- **High temperature (e.g. 1.0+)** → flatter probabilities, more random, weirder strings.

Try it without editing code:

```bash
python microgpt_updated.py --temperature 0.3   # safer, repetitive
python microgpt_updated.py --temperature 1.0   # more variety, more junk
```

Same trained weights, different **creativity knob**.

---

## Autograd: how the model learns (no PyTorch)

Most deep-learning tutorials hide gradients behind `tensor.backward()`. Here, every scalar is a **`Value`** object that remembers **how it was computed** (its parents in a small graph).

When you call `loss.backward()`, the code walks that graph backward and applies the **chain rule** from calculus—locally, at each operation—to fill in `.grad` on each weight.

**Tiny mental model:**

```text
y = a * b + c
```

If you change `a` a little, `y` changes in a predictable way; `backward()` accumulates those contributions for every parameter. The optimizer (Adam) then moves weights in the direction that **reduces loss**.

Why scalar autograd in Python? **You can read every step.** It is slow on purpose—that is the tradeoff for clarity.

**Go deeper:** For diagrams, hand-traced examples (`y = a*b`, chain rule, `y = a*a`), the full op table, and how softmax + cross-entropy connect to training, see **[`autograd-deep-dive.md`](./autograd-deep-dive.md)**. Then read [`mgpt/value.py`](../mgpt/value.py).

---

## Transformer pieces (what the network actually is)

You do not need to derive attention to use this repo. Here is a **map** of the pieces; the README architecture diagram shows how they connect.

| Piece | Plain English |
|-------|----------------|
| **Token embedding (`wte`)** | Each character id gets a learned vector (a list of numbers). |
| **Position embedding (`wpe`)** | “You are the 3rd character in the line” also gets a vector; added to the token vector. |
| **RMSNorm** | Rescale vectors so training stays stable (like LayerNorm’s simpler cousin). |
| **Multi-head attention** | Several parallel “views” ask: *which earlier characters should I look at for the next prediction?* **Causal** = only look left, not future letters. |
| **KV cache** | During generation, store past keys/values so we do not recompute the whole line every step. |
| **MLP block** | Two linear layers with **ReLU** in the middle—extra capacity after attention. |
| **LM head** | Final linear map: hidden vector → score per character in the vocabulary. |

**Head count intuition:** `N_EMBD=16` and `N_HEAD=4` means **four heads**, each working in a **4-dimensional** subspace (`HEAD_DIM = 16 // 4`). One head (`N_HEAD=1`) uses the full 16 dimensions in a single attention mix. Same total width, different **splitting strategy**—a common experiment in this repo.

---

## Hyperparameters you will actually touch

Defaults live at the top of `microgpt_updated.py`. Override from the shell with `--help` flags.

**Units** (same labels appear in source comments and run-report glossaries):

| Unit | Meaning |
|------|---------|
| **layers** | Count of stacked transformer blocks (`N_LAYER`) |
| **dimensions** | Length of a hidden / embedding vector (`N_EMBD`, `HEAD_DIM`) |
| **heads** | Count of parallel attention mixes (`N_HEAD`) |
| **tokens** | Context positions the model can see at once (`BLOCK_SIZE`) |
| **steps** | Optimizer updates during training (`NUM_STEPS`) |
| **dimensionless** | Pure numbers with no physical unit (learning rate, Adam betas, temperature, seed) |

| Knob | Default | Unit | Think of it as… |
|------|---------|------|------------------|
| `NUM_STEPS` | 1000 | steps | How many flashcard rounds (more → better fit, slower). |
| `N_EMBD` | 16 | dimensions | How wide each character’s hidden vector is (bigger → slower, more capacity). |
| `N_HEAD` | 4 | heads | How many parallel attention mixes (see above). |
| `HEAD_DIM` | `N_EMBD // N_HEAD` | dimensions | Per-head subspace width (derived, not swept separately). |
| `BLOCK_SIZE` | 16 | tokens | Longest line the model can see at once (including BOS). |
| `LEARNING_RATE` | 0.01 | dimensionless | Step size for weight updates (with decay over the run). |
| `TEMPERATURE` | 0.5 | dimensionless | Randomness when sampling names after training. |
| `SEED` | 42 | dimensionless | Reproducible randomness for init and sampling. |

For `N_LAYER`, Adam betas (`BETA1`, `BETA2`), and `EPS_ADAM`, see the [README configuration table](../README.md#microgpt_updatedpy-recommended-reference) (same unit labels).

**Rule of thumb:** if names in your file are longer than `BLOCK_SIZE`, increase `BLOCK_SIZE` or shorten lines.

Illustrative one-offs:

```bash
python microgpt_updated.py                          # defaults: 4 heads, 1000 steps
python microgpt_updated.py --n-head 1             # single-head comparison
python microgpt_updated.py --num-steps 50         # quick “did anything learn?” smoke test
python microgpt_updated.py --input my_lines.txt   # your own one-line-per-row file
```

Full sweep recipes: [`README.md` → Run experiments examples](../README.md#run-experiments-examples).

---

## Sample quality: how good are the generated names?

After generation, the refactored script scores the 20 lines. This is **not** a rigorous NLP benchmark—it is a **sanity check** with simple rules in `mgpt/evaluation.py`.

### Character-level

- **Char distribution similarity** — Do generated letters use roughly the same mix of `a`, `e`, `n`, … as the training file? Closer to **1.0** is better.
- **Length similarity** — Are generated names about as long as real ones on average?

### Three semantic tiers (heuristic)

| Tier | Meaning | Example (names corpus) |
|------|---------|-------------------------|
| **Tier 1 — Real** | Exact match (case-insensitive) to a line in training data | `ann`, `kamon` if those names were in `input.txt` |
| **Tier 2 — Plausible** | Not in the file, but “name-like”: pronounceable, reasonable length, letter pairs seen in training | `vialan`, `karia` |
| **Tier 3 — Nonsense** | Fails simple checks: no vowels, impossible consonant runs, repeated junk | `zxqx`, `aaaa` |

From the checked-in **4-head @ 1000 steps** report in `example-experiments/`:

```text
TIER1_REAL_RATIO=0.550000      # 11/20 exact training names
TIER2_PLAUSIBLE_RATIO=0.450000 # 9/20 new but plausible
TIER3_NONSENSE_RATIO=0.000000  # 0/20 obvious garbage
OVERALL_QUALITY_SCORE=0.582500
```

Tiers can overlap in edge cases; treat **overall score** and **tier ratios** as **comparative** signals when you change `N_HEAD` or `NUM_STEPS`, not absolute truth. Details: [`M2-semantic-quality.md`](./M2-semantic-quality.md).

---

## Run timing: how long did training take?

Scalar autograd in Python is slow on purpose. While training runs, the terminal shows **live progress**: loss, step count, **`elapsed … | ETA …`**, and after training a **`Run wall clock:`** summary.

Saved run reports include **`--- Run timing ---`**: start and end in **UTC** and **local time with offset**, plus **`DURATION_SECONDS`** and **`TIMEZONE`**. That block is **authoritative**. The `_YYYYMMDD_HHMMSS` suffix in the filename is only for **uniqueness** (approximate local time when the file path was built).

Grid sweeps also write **`sweep_summary.csv`** (per-run timing columns) and **`sweep_timing.txt`** (whole-grid wall clock). Compare and HTML tools show timing when the block is present. Full recipe: [`experiment-workflow.md` → Run timing](./experiment-workflow.md#run-timing-and-progress).

---

## Suggested reading order

1. **This page** — concepts and vocabulary.
2. **[`README.md`](../README.md)** — quick start, architecture diagram, config, run reports.
3. **Run once** — `python microgpt_updated.py` (takes a while; that is normal).
4. **Skim a report** — `example-experiments/output_L1_E16_H4_B16_S1000_….txt` without training first.
5. **Pick an entry script:**
   - **`microgpt.py`** — one continuous story (~300 lines); best “read like a blog post.”
   - **`microgpt_updated.py`** — CLI entry; calls **`mgpt/experiment.py`** (`run_experiment()`, `train()`, `generate()`).
6. **Autograd** — [`autograd-deep-dive.md`](./autograd-deep-dive.md) (diagrams + hand traces), then `mgpt/value.py` → `ops.py` → `model.py`.
7. **Experiments** — [`experiment-workflow.md`](./experiment-workflow.md) (train → compare), or open `example-experiments/comparison_report.html`.
8. **[Karpathy’s microGPT post](https://karpathy.github.io/2026/02/12/microgpt/)** — deeper theory when you want the original narrative.

---

## Glossary (quick lookup)

| Term | Short definition |
|------|------------------|
| **BOS** | Special token marking start/end of a line. |
| **Vocabulary** | All distinct characters in your file, plus BOS. |
| **Logits** | Raw scores before softmax; not yet probabilities. |
| **Softmax** | Turns logits into probabilities that sum to 1. |
| **Cross-entropy** | Loss for “how wrong were the predicted probabilities vs the true next character?” |
| **Adam** | Adaptive optimizer; moves each weight using running averages of gradients. |
| **Causal attention** | Each position may only attend to earlier positions (no peeking at future letters). |
| **Run report** | Text file `outputs/output_*.txt` with config, loss, samples, quality metrics. |

---

## What to ignore at first

- Filename tokens like `T0p5` (temperature 0.5 encoded for safe paths).
- `HEAD_DIM` in config—it is always `N_EMBD // N_HEAD`, not a separate dial you sweep.
- KV cache living inside the training graph (advanced; see README “Developing further” when you change attention).

When something in the README feels dense, come back here for the **idea**, then return to the README for the **exact command or file path**.
