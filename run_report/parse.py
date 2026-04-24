"""Parse saved ``output_*.txt`` run reports."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CFG_INT = frozenset(
    {"N_LAYER", "N_EMBD", "N_HEAD", "HEAD_DIM", "BLOCK_SIZE", "NUM_STEPS", "SEED"}
)
_CFG_FLOAT = frozenset(
    {"TEMPERATURE", "LEARNING_RATE", "BETA1", "BETA2", "EPS_ADAM"}
)
_CFG_STR = frozenset({"INPUT_PATH"})

_QUALITY_FLOAT_KEYS = frozenset(
    {
        "CHAR_DIST_SIMILARITY",
        "AVG_SAMPLE_LENGTH",
        "LENGTH_SIMILARITY",
        "TIER1_REAL_RATIO",
        "TIER2_PLAUSIBLE_RATIO",
        "TIER2_AVG_SCORE",
        "TIER3_NONSENSE_RATIO",
        "OVERALL_QUALITY_SCORE",
    }
)
_QUALITY_INT_KEYS = frozenset(
    {
        "TIER1_REAL_COUNT",
        "TIER2_PLAUSIBLE_COUNT",
        "TIER3_NONSENSE_COUNT",
    }
)

# Order used when printing a full parsed config (or legacy tools).
_CFG_DISPLAY_ORDER: tuple[str, ...] = (
    "N_LAYER",
    "N_EMBD",
    "N_HEAD",
    "HEAD_DIM",
    "BLOCK_SIZE",
    "NUM_STEPS",
    "TEMPERATURE",
    "SEED",
    "LEARNING_RATE",
    "BETA1",
    "BETA2",
    "EPS_ADAM",
    "INPUT_PATH",
)

# Omitted from experiment HTML / compare_run_reports tables: set by N_EMBD and N_HEAD;
# see ``ParsedRunReport.config`` (``HEAD_DIM``) and the derived comment in new reports.
DERIVED_EXPERIMENT_CFG_KEYS: frozenset[str] = frozenset({"HEAD_DIM"})


@dataclass(frozen=True)
class ParsedRunReport:
    config: dict[str, int | float | str]
    final_loss: float
    samples: list[str]
    loss_history: list[float] | None
    char_dist_score: float | None = None
    avg_sample_length: float | None = None
    length_similarity: float | None = None
    semantic_quality: dict[str, object] | None = None


def cfg_keys_in_display_order(keys: set[str]) -> list[str]:
    ordered = [k for k in _CFG_DISPLAY_ORDER if k in keys]
    rest = sorted(k for k in keys if k not in _CFG_DISPLAY_ORDER)
    return ordered + rest


def cfg_keys_for_experiment_table(keys: set[str]) -> list[str]:
    """Config keys to show in experiment diffs: primary run flags, not derived fields."""
    return cfg_keys_in_display_order(keys - DERIVED_EXPERIMENT_CFG_KEYS)


def _parse_quality_key_values(text: str) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if key in _QUALITY_INT_KEYS:
            try:
                out[key] = int(rest.strip())
            except ValueError:
                pass
        elif key in _QUALITY_FLOAT_KEYS:
            try:
                out[key] = float(rest.strip())
            except ValueError:
                pass
    return out


def _normalize_head_dim(cfg: dict[str, int | float | str]) -> None:
    """Set ``HEAD_DIM`` to canonical ``N_EMBD // N_HEAD`` when both ints are present.

    New reports may omit ``HEAD_DIM``; legacy reports may include a redundant line.
    We always store the derived value so consumers see one consistent architecture.
    """
    n_embd = cfg.get("N_EMBD")
    n_head = cfg.get("N_HEAD")
    if not isinstance(n_embd, int) or not isinstance(n_head, int) or n_head < 1:
        return
    if n_embd % n_head != 0:
        return
    cfg["HEAD_DIM"] = n_embd // n_head


def _parse_semantic_example_lines(text: str) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {
        "tier1_examples": [],
        "tier2_examples": [],
        "tier3_examples": [],
    }
    pending: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if s == "# Tier 1 Examples (Real):":
            pending = "tier1_examples"
            continue
        if s == "# Tier 2 Examples (Plausible):":
            pending = "tier2_examples"
            continue
        if s == "# Tier 3 Examples (Nonsense):":
            pending = "tier3_examples"
            continue
        if pending and line.startswith("#   "):
            examples[pending] = [
                x.strip() for x in line[3:].strip().split(",") if x.strip()
            ]
            pending = None
    return examples


def parse_run_report_text(text: str) -> ParsedRunReport:
    """Parse a saved run report: config key=value lines, final loss, inference samples."""
    loss_history: list[float] | None = None
    in_loss = False
    loss_vals: list[float] = []

    cfg: dict[str, int | float | str] = {}
    for line in text.splitlines():
        if not line or line.startswith("---"):
            continue
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if key in _CFG_INT:
            try:
                cfg[key] = int(rest.strip())
            except ValueError:
                pass
        elif key in _CFG_FLOAT:
            try:
                cfg[key] = float(rest.strip())
            except ValueError:
                pass
        elif key in _CFG_STR:
            cfg[key] = rest.strip()

    _normalize_head_dim(cfg)

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "--- Loss history (CSV: step,loss) ---":
            in_loss = True
            loss_vals = []
            continue
        if in_loss:
            if line.startswith("---"):
                loss_history = loss_vals if loss_vals else None
                in_loss = False
                continue
            if not stripped:
                continue
            if "," in stripped:
                _, _, loss_s = stripped.partition(",")
                try:
                    loss_vals.append(float(loss_s.strip()))
                except ValueError:
                    pass
            continue

    if in_loss and loss_vals:
        loss_history = loss_vals

    m_loss = re.search(
        r"^Final loss \(last training step\): ([0-9.eE+-]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not m_loss:
        raise ValueError("missing final loss line")
    final_loss = float(m_loss.group(1))

    samples: list[str] = []
    in_samples = False
    sample_re = re.compile(r"^Sample\s+\d+:\s*(.*)$")
    for line in text.splitlines():
        if line.strip() == "--- Inference samples ---":
            in_samples = True
            continue
        if in_samples:
            if line.startswith("---"):
                break
            m = sample_re.match(line.rstrip())
            if m:
                samples.append(m.group(1))

    qv = _parse_quality_key_values(text)

    def _qf(key: str) -> float | None:
        v = qv.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    char_dist_f = _qf("CHAR_DIST_SIMILARITY")
    avg_len_f = _qf("AVG_SAMPLE_LENGTH")
    len_sim_f = _qf("LENGTH_SIMILARITY")

    semantic: dict[str, object] | None = None
    if any(k.startswith("TIER") for k in qv) or "OVERALL_QUALITY_SCORE" in qv:
        ex = _parse_semantic_example_lines(text)
        semantic = {
            "tier1_real_count": int(qv.get("TIER1_REAL_COUNT", 0)),
            "tier1_real_ratio": float(qv.get("TIER1_REAL_RATIO", 0.0)),
            "tier2_plausible_count": int(qv.get("TIER2_PLAUSIBLE_COUNT", 0)),
            "tier2_plausible_ratio": float(qv.get("TIER2_PLAUSIBLE_RATIO", 0.0)),
            "tier2_avg_score": float(qv.get("TIER2_AVG_SCORE", 0.0)),
            "tier3_nonsense_count": int(qv.get("TIER3_NONSENSE_COUNT", 0)),
            "tier3_nonsense_ratio": float(qv.get("TIER3_NONSENSE_RATIO", 0.0)),
            "overall_quality_score": float(qv.get("OVERALL_QUALITY_SCORE", 0.0)),
            "tier1_examples": ex["tier1_examples"],
            "tier2_examples": ex["tier2_examples"],
            "tier3_examples": ex["tier3_examples"],
        }

    return ParsedRunReport(
        config=cfg,
        final_loss=final_loss,
        samples=samples,
        loss_history=loss_history,
        char_dist_score=char_dist_f,
        avg_sample_length=avg_len_f,
        length_similarity=len_sim_f,
        semantic_quality=semantic,
    )
