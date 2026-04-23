#!/usr/bin/env python3
"""
Insert the same ``--- What this run is ---`` narrative that :func:`save_run_report`
writes into **existing** ``output_*.txt`` files (past experiments).

Stdlib only; run from the repo root::

    python annotate_run_reports.py
    python annotate_run_reports.py path/to/output_L1_....txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Same module as training — narrative text stays single-sourced.
from microgpt_updated import NARRATIVE_SECTION_HEADER, format_run_narrative_lines

_CFG_INT = {
    "N_LAYER",
    "N_EMBD",
    "N_HEAD",
    "HEAD_DIM",
    "BLOCK_SIZE",
    "NUM_STEPS",
    "SEED",
}
_CFG_FLOAT = {"TEMPERATURE", "LEARNING_RATE", "BETA1", "BETA2"}


def _parse_report(text: str) -> dict[str, int | float | str] | None:
    """Best-effort parse of a saved run report; returns None if too incomplete."""
    cfg: dict[str, int | float | str] = {}
    for line in text.splitlines():
        if not line or line.startswith("---"):
            continue
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if key not in _CFG_INT and key not in _CFG_FLOAT and key != "INPUT_PATH":
            continue
        val_s = rest.strip()
        if key in _CFG_INT:
            try:
                cfg[key] = int(val_s)
            except ValueError:
                pass
        elif key in _CFG_FLOAT:
            try:
                cfg[key] = float(val_s)
            except ValueError:
                pass
        else:
            cfg[key] = val_s
    m_loss = re.search(
        r"^Final loss \(last training step\): ([0-9.eE+-]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not m_loss:
        return None
    cfg["_final_loss"] = float(m_loss.group(1))
    cfg["_num_samples"] = len(
        re.findall(r"^Sample\s+\d+:", text, re.MULTILINE)
    )
    need = {
        "N_LAYER",
        "N_EMBD",
        "N_HEAD",
        "HEAD_DIM",
        "BLOCK_SIZE",
        "NUM_STEPS",
        "TEMPERATURE",
        "SEED",
        "INPUT_PATH",
    }
    for k in need:
        if k not in cfg:
            return None
    return cfg


def annotate_file(path: Path) -> str:
    """Return a status string: 'skip', 'ok', or 'err: ...'."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"err: {e}"
    if NARRATIVE_SECTION_HEADER in text:
        return "skip (narrative already present)"
    cfg = _parse_report(text)
    if cfg is None:
        return "err: could not parse config / loss (unexpected format)"
    n_layer = int(cfg["N_LAYER"])
    n_embd = int(cfg["N_EMBD"])
    n_head = int(cfg["N_HEAD"])
    head_dim = int(cfg["HEAD_DIM"])
    block_size = int(cfg["BLOCK_SIZE"])
    num_steps = int(cfg["NUM_STEPS"])
    temperature = float(cfg["TEMPERATURE"])
    seed = int(cfg["SEED"])
    input_path = str(cfg["INPUT_PATH"])
    final_loss = float(cfg["_final_loss"])
    num_samples = int(cfg["_num_samples"])
    if num_samples < 1:
        return "err: no Sample N: lines found"

    sep = "==================="
    i = text.find(sep)
    if i < 0:
        return "err: missing report banner line"
    insert_at = text.find("\n", i) + 1
    narrative = (
        "\n".join(
            format_run_narrative_lines(
                n_layer=n_layer,
                n_embd=n_embd,
                n_head=n_head,
                head_dim=head_dim,
                block_size=block_size,
                num_steps=num_steps,
                temperature=temperature,
                seed=seed,
                input_path=input_path,
                final_loss=final_loss,
                num_samples=num_samples,
            )
        )
        + "\n"
    )
    new_text = text[:insert_at] + narrative + text[insert_at:]
    path.write_text(new_text, encoding="utf-8")
    return "ok"


def main() -> None:
    root = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(root.glob("output_*.txt"))
    if not paths:
        print("No output_*.txt files found (pass explicit paths as arguments).")
        sys.exit(0)
    for p in paths:
        status = annotate_file(p)
        print(f"{p.name}: {status}")


if __name__ == "__main__":
    main()
