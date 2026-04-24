#!/usr/bin/env python3
"""Build an HTML comparison table from one or more ``output_*.txt`` run reports."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_report.parse import ParsedRunReport, parse_run_report_text


def _cfg_int(cfg: dict[str, int | float | str], key: str, default: int = 0) -> int:
    v = cfg.get(key, default)
    return int(v) if isinstance(v, (int, float)) else default


def _tier_ratio(sem: dict[str, object] | None, key: str) -> float:
    if sem is None:
        return 0.0
    v = sem.get(key)
    return float(v) if isinstance(v, (int, float)) else 0.0


def _row_from_parsed(path: Path, parsed: ParsedRunReport) -> dict[str, object]:
    cfg = parsed.config
    sem = parsed.semantic_quality
    n_head = _cfg_int(cfg, "N_HEAD", 1)
    head_dim = _cfg_int(cfg, "HEAD_DIM", 0)
    if head_dim <= 0:
        n_embd = _cfg_int(cfg, "N_EMBD", 1)
        head_dim = n_embd // n_head if n_head else 0
    return {
        "filename": path.name,
        "n_head": n_head,
        "head_dim": head_dim,
        "final_loss": parsed.final_loss,
        "char_dist": float(parsed.char_dist_score or 0.0),
        "loss_history": parsed.loss_history or [],
        "samples": parsed.samples,
        "tier1_ratio": _tier_ratio(sem, "tier1_real_ratio"),
        "tier2_ratio": _tier_ratio(sem, "tier2_plausible_ratio"),
        "tier3_ratio": _tier_ratio(sem, "tier3_nonsense_ratio"),
        "overall_quality": _tier_ratio(sem, "overall_quality_score"),
    }


def _legacy_summary_line(entries: list[dict[str, object]], *, max_list: int = 12) -> str:
    """Single-line description of legacy runs (loss + filename)."""
    n = len(entries)
    parts: list[str] = []
    for e in entries[:max_list]:
        parts.append(
            f"{e['filename']} (loss {float(e['final_loss']):.4f})"
        )
    tail = ""
    if n > max_list:
        tail = f"; … and {n - max_list} more"
    listed = html.escape("; ".join(parts) + tail)
    return (
        f"<strong>Legacy experiments ({n}):</strong> {listed} — "
        "these reports predate semantic quality metrics (no tier / overall score block). "
        "Re-run with current <code>microgpt_updated.py</code> to populate them."
    )


def generate_html_report(
    output_files: list[Path],
    report_path: Path,
) -> None:
    modern: list[dict[str, object]] = []
    legacy: list[dict[str, object]] = []
    for path in output_files:
        text = path.read_text(encoding="utf-8")
        parsed = parse_run_report_text(text)
        row = _row_from_parsed(path, parsed)
        if parsed.semantic_quality is not None:
            modern.append(row)
        else:
            legacy.append(row)

    if not modern and not legacy:
        raise ValueError("no reports to render")

    best_loss = (
        min(float(r["final_loss"]) for r in modern) if modern else float("nan")
    )
    best_quality = (
        max(float(r["overall_quality"]) for r in modern) if modern else 0.0
    )

    rows_html: list[str] = []
    for r in modern:
        loss = float(r["final_loss"])
        oq = float(r["overall_quality"])
        loss_class = "best" if modern and loss == best_loss else ""
        quality_class = (
            "best" if modern and oq == best_quality and oq > 0 else ""
        )
        sample0 = r["samples"][0] if r["samples"] else "N/A"
        cfg_label = f"{r['n_head']}×{r['head_dim']}"
        rows_html.append(
            f"""            <tr>
                <td>{html.escape(cfg_label)}</td>
                <td class="metric {loss_class}">{loss:.4f}</td>
                <td class="metric {quality_class}">{oq:.3f}</td>
                <td>{float(r['tier1_ratio']):.1%}</td>
                <td>{float(r['tier2_ratio']):.1%}</td>
                <td>{float(r['tier3_ratio']):.1%}</td>
                <td class="samples">{html.escape(sample0)}</td>
                <td class="fname">{html.escape(str(r['filename']))}</td>
            </tr>"""
        )

    if legacy:
        rows_html.append(
            f"""            <tr class="legacy-summary">
                <td colspan="8">{_legacy_summary_line(legacy)}</td>
            </tr>"""
        )

    bars_html: list[str] = []
    for r in modern:
        cfg_name = f"{r['n_head']}×{r['head_dim']}"
        t1 = float(r["tier1_ratio"])
        t2 = float(r["tier2_ratio"])
        t3 = float(r["tier3_ratio"])
        f1 = max(1, int(round(t1 * 100)))
        f2 = max(1, int(round(t2 * 100)))
        f3 = max(1, int(round(t3 * 100)))
        bars_html.append(
            f"""        <h3>{html.escape(cfg_name)}</h3>
        <div class="tier-bars">
            <div class="bar real" style="flex: {f1}"><strong>Real: {t1:.1%}</strong></div>
            <div class="bar plausible" style="flex: {f2}"><strong>Plausible: {t2:.1%}</strong></div>
            <div class="bar nonsense" style="flex: {f3}"><strong>Nonsense: {t3:.1%}</strong></div>
        </div>"""
        )

    n_mod, n_leg = len(modern), len(legacy)
    summary_bits = [f"{n_mod} run(s) with full quality metrics"]
    if n_leg:
        summary_bits.append(
            f"{n_leg} legacy (collapsed below — no semantic quality section)"
        )
    summary_p = ", ".join(summary_bits) + ". "
    if n_mod:
        summary_p += (
            "Best final loss and best overall quality (when > 0) apply only to those runs."
        )
    else:
        summary_p += "Re-run or pick reports generated with current tooling for tier charts."

    quality_section = (
        "\n".join(bars_html)
        if bars_html
        else "    <p><em>No semantic quality data in the selected reports (legacy-only).</em></p>"
    )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>microGPT run comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; }}
    h1 {{ font-size: 1.25rem; }}
    table {{ border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px #0001; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f0f0f0; }}
    .metric.best {{ font-weight: bold; color: #0a0; }}
    .samples {{ font-family: ui-monospace, monospace; max-width: 14rem; }}
    .fname {{ font-size: 0.8rem; color: #555; }}
    .section {{ margin-top: 2rem; }}
    .tier-bars {{ display: flex; gap: 10px; margin-bottom: 1.5rem; }}
    .bar {{ padding: 10px; border-radius: 5px; }}
    .bar.real {{ background: #d4edda; }}
    .bar.plausible {{ background: #fff3cd; }}
    .bar.nonsense {{ background: #f8d7da; }}
    tr.legacy-summary td {{ background: #f0f0f0; font-size: 0.9rem; line-height: 1.4; }}
  </style>
</head>
<body>
  <h1>microGPT run comparison</h1>
  <p>{html.escape(summary_p, quote=False)}</p>
  <div class="section">
    <h2>Runs</h2>
    <table>
      <tr>
        <th>Config (H×D)</th>
        <th>Final loss</th>
        <th>Overall quality</th>
        <th>Real words</th>
        <th>Plausible</th>
        <th>Nonsense</th>
        <th>First sample</th>
        <th>File</th>
      </tr>
{chr(10).join(rows_html)}
    </table>
  </div>
  <div class="section">
    <h2>Quality distribution</h2>
{quality_section}
  </div>
</body>
</html>
"""
    report_path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="Run report paths (default: output_*.txt in cwd)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_REPO_ROOT / "comparison_report.html",
        help=f"Write HTML here (default: {_REPO_ROOT / 'comparison_report.html'})",
    )
    args = parser.parse_args()
    files = (
        list(args.reports)
        if args.reports
        else sorted(_REPO_ROOT.glob("output_*.txt"))
    )
    files = [p for p in files if p.is_file()]
    if not files:
        print("No report files found.", file=sys.stderr)
        sys.exit(2)
    try:
        generate_html_report(files, args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
