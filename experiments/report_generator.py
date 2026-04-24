#!/usr/bin/env python3
"""Build an HTML comparison table from one or more ``output_*.txt`` run reports.

Default inputs: ``run_reports_dir(repo_root)`` / ``output_*.txt`` where *repo_root*
is the parent of ``experiments/`` (same rule as ``microgpt_updated.py`` saves).
Shared / varying **training** config tables list the same keys as
``compare_run_reports.py`` (including ``HEAD_DIM``, after parse normalization).
``HEAD_DIM`` rows include a short note that the value follows from ``N_EMBD`` and ``N_HEAD``.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_report.parse import (
    ParsedRunReport,
    cfg_keys_for_experiment_table,
    experiment_cfg_calculated_caption,
    parse_run_report_text,
)
from run_report.paths import DEFAULT_RUN_REPORT_DIR, run_reports_dir
from run_report.text_loss_plot import loss_curve_comparison_lines, single_loss_curve_lines

_DEFAULT_OUTPUTS = run_reports_dir(_REPO_ROOT)


def _cfg_int(cfg: dict[str, int | float | str], key: str, default: int = 0) -> int:
    v = cfg.get(key, default)
    return int(v) if isinstance(v, (int, float)) else default


def _fmt_cfg_val(v: object) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _tier_ratio(sem: dict[str, object] | None, key: str) -> float:
    if sem is None:
        return 0.0
    val = sem.get(key)
    return float(val) if isinstance(val, (int, float)) else 0.0


def _short_label(name: str, max_len: int = 36) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


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
        "path": path,
        "config": cfg,
        "n_head": n_head,
        "head_dim": head_dim,
        "final_loss": parsed.final_loss,
        "char_dist": float(parsed.char_dist_score or 0.0),
        "loss_history": parsed.loss_history,
        "samples": parsed.samples,
        "tier1_ratio": _tier_ratio(sem, "tier1_real_ratio"),
        "tier2_ratio": _tier_ratio(sem, "tier2_plausible_ratio"),
        "tier3_ratio": _tier_ratio(sem, "tier3_nonsense_ratio"),
        "overall_quality": _tier_ratio(sem, "overall_quality_score"),
        "has_semantic": parsed.semantic_quality is not None,
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


def _html_config_block(records: list[dict[str, object]]) -> str:
    """Shared vs varying config (same ideas as ``compare_run_reports``)."""
    if not records:
        return ""
    keys_union: set[str] = set()
    for r in records:
        keys_union |= set(r["config"])  # type: ignore[arg-type]
    ordered = cfg_keys_for_experiment_table(keys_union)

    shared_lines: list[str] = []
    varying: list[tuple[str, list[str]]] = []

    cfg_list = [r["config"] for r in records]  # type: ignore[misc]

    for key in ordered:
        vals = [c.get(key) for c in cfg_list]
        if all(v == vals[0] for v in vals):
            line = f"{key}={_fmt_cfg_val(vals[0])}"
            cap = experiment_cfg_calculated_caption(key)
            if cap:
                line += f"  — {cap}"
            shared_lines.append(line)
        else:
            varying.append(
                (
                    key,
                    [
                        _fmt_cfg_val(v) if v is not None else "—"
                        for v in vals
                    ],
                )
            )

    parts: list[str] = ['    <div class="section">', "    <h2>Training config</h2>"]

    if shared_lines:
        parts.append(
            "    <h3>Shared across all runs</h3>"
            f"    <pre class=\"config-block\">{html.escape(chr(10).join(shared_lines))}</pre>"
        )

    if varying:
        headers = "".join(
            f"<th>{html.escape(_short_label(str(r['filename']), 28))}</th>"
            for r in records
        )
        body_rows = []
        for key, cells in varying:
            tds = "".join(f"<td><code>{html.escape(c)}</code></td>" for c in cells)
            cap = experiment_cfg_calculated_caption(key)
            if cap:
                th_key = (
                    f'<code>{html.escape(key)}</code>'
                    f'<div class="cfg-calculated-hint">{html.escape(cap)}</div>'
                )
            else:
                th_key = f"<code>{html.escape(key)}</code>"
            body_rows.append(
                f"            <tr><th scope=\"row\">{th_key}</th>{tds}</tr>"
            )
        parts.append("    <h3>Varying by run</h3>")
        parts.append("    <table class=\"config-diff\">")
        parts.append(f"      <tr><th scope=\"col\">Key</th>{headers}</tr>")
        parts.extend(body_rows)
        parts.append("    </table>")
    elif not shared_lines:
        parts.append("    <p><em>No config key=value lines parsed.</em></p>")

    parts.append("    </div>")
    return "\n".join(parts)


def _html_samples_block(records: list[dict[str, object]]) -> str:
    """Side-by-side inference samples with * when a row differs across runs."""
    if len(records) < 2:
        return ""
    samples_per = [list(r["samples"]) for r in records]  # type: ignore[misc]
    n = max((len(s) for s in samples_per), default=0)
    if n == 0:
        return ""

    headers = "".join(
        f"<th>{html.escape(_short_label(str(r['filename']), 24))}</th>"
        for r in records
    )
    rows_html: list[str] = []
    for i in range(n):
        cells = [
            s[i] if i < len(s) else "—"
            for s in samples_per
        ]
        same = len(set(cells)) <= 1
        mark = " " if same else "*"
        tds = "".join(
            f"<td class=\"samples\">{html.escape(c)}</td>" for c in cells
        )
        rows_html.append(
            f"            <tr><td class=\"samp-idx\">{html.escape(mark)} {i + 1}</td>{tds}</tr>"
        )

    return "\n".join(
        [
            '    <div class="section">',
            "    <h2>Inference samples (aligned)</h2>",
            "    <p><em>* marks rows where not all runs agree (same layout as "
            "<code>compare_run_reports.py</code>).</em></p>",
            "    <table class=\"samples-grid\">",
            f"      <tr><th></th>{headers}</tr>",
            *rows_html,
            "    </table>",
            "    </div>",
        ]
    )


def _html_loss_block(
    records: list[dict[str, object]],
    *,
    loss_bins: int,
    loss_height: int,
) -> str:
    """Loss ASCII: one run → single grid; several → baseline (best final loss) vs each other."""
    with_hist: list[tuple[dict[str, object], list[float]]] = []
    for r in records:
        h = r.get("loss_history")
        if isinstance(h, list) and h:
            with_hist.append((r, h))

    if not with_hist:
        return ""

    parts: list[str] = [
        '    <div class="section">',
        "    <h2>Loss history (text graphs)</h2>",
        "    <p>Same binned min–mean–max grids as <code>compare_run_reports.py</code> "
        f"(<code>--loss-bins</code>={loss_bins}, <code>--loss-height</code>={loss_height}). "
        "With multiple runs, the <strong>baseline</strong> is the one with the "
        "<strong>lowest final loss</strong> among reports that include loss history; "
        "each other run is compared to that baseline.</p>",
    ]

    if len(with_hist) == 1:
        r, h = with_hist[0]
        label = _short_label(str(r["filename"]))
        lines = single_loss_curve_lines(
            label=label,
            losses=h,
            width=loss_bins,
            grid_height=loss_height,
        )
        parts.append(
            f"    <pre class=\"loss-ascii\">{html.escape(chr(10).join(lines))}</pre>"
        )
    else:
        baseline_rec, baseline_h = min(
            with_hist,
            key=lambda t: float(t[0]["final_loss"]),
        )
        base_label = _short_label(str(baseline_rec["filename"]))
        others = [
            (r, h)
            for r, h in with_hist
            if r is not baseline_rec
        ]
        others.sort(key=lambda t: str(t[0]["filename"]))

        parts.append(
            "    <p><strong>Baseline:</strong> "
            f"{html.escape(base_label)} (lowest final loss among runs with history).</p>"
        )

        for r, h in others:
            other_label = _short_label(str(r["filename"]))
            lines = loss_curve_comparison_lines(
                label_a=base_label,
                label_b=other_label,
                losses_a=baseline_h,
                losses_b=h,
                width=loss_bins,
                grid_height=loss_height,
            )
            parts.append(f"    <h3>{html.escape(base_label)} vs {html.escape(other_label)}</h3>")
            parts.append(
                f"    <pre class=\"loss-ascii\">{html.escape(chr(10).join(lines))}</pre>"
            )

    parts.append("    </div>")
    return "\n".join(parts)


def generate_html_report(
    output_files: list[Path],
    report_path: Path,
    *,
    loss_bins: int = 72,
    loss_height: int = 12,
) -> None:
    records: list[dict[str, object]] = []
    for path in sorted(output_files, key=lambda p: p.name):
        text = path.read_text(encoding="utf-8")
        parsed = parse_run_report_text(text)
        records.append(_row_from_parsed(path, parsed))

    modern = [r for r in records if r["has_semantic"]]
    legacy = [r for r in records if not r["has_semantic"]]

    if not records:
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
            f"{n_leg} legacy (collapsed in table — no semantic quality section)"
        )
    summary_p = ", ".join(summary_bits) + ". "
    if n_mod:
        summary_p += (
            "Best final loss and best overall quality (when > 0) apply only to those runs. "
            "Config, samples, and loss graphs include every report passed in."
        )
    else:
        summary_p += "Re-run or pick reports generated with current tooling for tier charts."

    quality_section = (
        "\n".join(bars_html)
        if bars_html
        else "    <p><em>No semantic quality data in the selected reports (legacy-only).</em></p>"
    )

    config_section = _html_config_block(records)
    samples_section = _html_samples_block(records)
    loss_section = _html_loss_block(
        records, loss_bins=loss_bins, loss_height=loss_height
    )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>microGPT run comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; }}
    h1 {{ font-size: 1.25rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 0; }}
    h3 {{ font-size: 1rem; }}
    table {{ border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px #0001; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f0f0f0; }}
    .metric.best {{ font-weight: bold; color: #0a0; }}
    .samples {{ font-family: ui-monospace, monospace; max-width: 14rem; word-break: break-all; }}
    .fname {{ font-size: 0.8rem; color: #555; }}
    .section {{ margin-top: 2rem; }}
    .tier-bars {{ display: flex; gap: 10px; margin-bottom: 1.5rem; }}
    .bar {{ padding: 10px; border-radius: 5px; }}
    .bar.real {{ background: #d4edda; }}
    .bar.plausible {{ background: #fff3cd; }}
    .bar.nonsense {{ background: #f8d7da; }}
    tr.legacy-summary td {{ background: #f0f0f0; font-size: 0.9rem; line-height: 1.4; }}
    pre.config-block, pre.loss-ascii {{
      font-family: ui-monospace, monospace; font-size: 0.72rem;
      background: #fff; border: 1px solid #ddd; padding: 0.75rem;
      overflow-x: auto; line-height: 1.25;
    }}
    table.config-diff th[scope="row"] {{ text-align: left; vertical-align: top; }}
    .cfg-calculated-hint {{
      font-size: 0.72rem; font-weight: normal; color: #555; margin-top: 0.25rem;
      line-height: 1.3; max-width: 16rem; white-space: normal;
    }}
    table.samples-grid .samp-idx {{ font-family: ui-monospace, monospace; color: #666; width: 2.5rem; }}
  </style>
</head>
<body>
  <h1>microGPT run comparison</h1>
  <p>{html.escape(summary_p, quote=False)}</p>
{config_section}
  <div class="section">
    <h2>Runs (quality summary)</h2>
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
{samples_section}
{loss_section}
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
        help=f"Run report paths (default: {DEFAULT_RUN_REPORT_DIR}/output_*.txt under repo root)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUTS / "comparison_report.html",
        help=f"Write HTML here (default: {_DEFAULT_OUTPUTS / 'comparison_report.html'})",
    )
    parser.add_argument(
        "--loss-bins",
        type=int,
        default=72,
        help="Horizontal bins for loss text graphs (default: 72)",
    )
    parser.add_argument(
        "--loss-height",
        type=int,
        default=12,
        help="Vertical rows for loss grids (default: 12, min 3)",
    )
    args = parser.parse_args()
    if args.loss_bins < 1:
        print("--loss-bins must be at least 1", file=sys.stderr)
        sys.exit(2)
    if args.loss_height < 3:
        print("--loss-height must be at least 3", file=sys.stderr)
        sys.exit(2)
    files = (
        list(args.reports)
        if args.reports
        else sorted(_DEFAULT_OUTPUTS.glob("output_*.txt"))
    )
    files = [p for p in files if p.is_file()]
    if not files:
        print("No report files found.", file=sys.stderr)
        sys.exit(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        generate_html_report(
            files,
            args.output,
            loss_bins=args.loss_bins,
            loss_height=args.loss_height,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
