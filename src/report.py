"""
Turn a run's JSONL logs into the tables and plots that go in the report.

    python -m src.report                      # newest run
    python -m src.report --run 20261011_0230  # a specific run

Outputs, into runs/<run>/:
    ablation_table.md       the headline results table
    cost_latency_table.md   what the extra quality cost
    plot_correctness.png    EM / F1 / judge by arm
    plot_faithfulness.png   faithfulness by arm
    plot_cost_quality.png   tokens per query vs faithfulness
    plot_verdicts.png       verdict mix by arm

Because this reads the logs and never the API, you can change a chart and
regenerate everything for free.

CHART DESIGN NOTES
------------------
Colours come from a validated categorical palette: arms use slots 1-3 (blue,
orange, aqua), which pass colourblind-separation and normal-vision checks as a
three-colour set. Verdicts use the reserved STATUS palette (good / warning /
critical), never the series colours, so a verdict colour can never be mistaken
for an arm. Every bar carries a direct value label - this is required, not
decorative: aqua sits below 3:1 contrast on a light surface, and visible
labels are what keep the chart readable without relying on colour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")                      # no display needed; write files
import matplotlib.pyplot as plt

from src.config import load_config

# --- validated palette ----------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
# categorical slots 1-3: validated all-pairs for CVD and normal vision
# Slots 1-3 validated all-pairs for CVD; slot 4 (yellow) only ever appears
# beside them in the four-arm chart, where every bar is directly labelled.
ARM_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
# reserved status palette - deliberately distinct from the series colours
VERDICT_COLORS = {
    "ENTAILED": "#0ca30c",       # good
    "UNSUPPORTED": "#fab219",    # warning
    "CONTRADICTED": "#d03b3b",   # critical
}

ARM_LABEL = {
    "A": "A\nbaseline RAG",
    "B": "B\nagentic only",
    "C": "C\nfull, drop",
    "D": "D\nfull, warn-only",
}


def _style_axes(ax) -> None:
    """Recessive grid and axes, so the data is the loudest thing present."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)


def _new_fig(w: float = 6.4, h: float = 4.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


# ===========================================================================
# Plots
# ===========================================================================

def plot_correctness(summaries: dict[str, dict], out: Path) -> None:
    """
    Grouped bars: three correctness measures per arm.

    Three measures rather than one because they disagree in informative ways.
    EM is harsh on a verbose answer, token F1 penalises length, and the LLM
    judge reads for meaning. If the three tell the same story, the result is
    solid; if they diverge, that divergence is itself worth discussing.
    """
    arms = sorted(summaries)
    metrics = [("exact_match", "Exact match"),
               ("token_f1", "Token F1"),
               ("judge_correct", "LLM judge")]

    fig, ax = _new_fig(7.0, 4.2)
    n_arms = len(arms)
    group_w = 0.78
    bar_w = group_w / n_arms

    for a_i, arm in enumerate(arms):
        xs, ys = [], []
        for m_i, (key, _label) in enumerate(metrics):
            xs.append(m_i - group_w / 2 + bar_w * (a_i + 0.5))
            ys.append(summaries[arm].get(key) or 0.0)
        bars = ax.bar(xs, ys, width=bar_w * 0.9, color=ARM_COLORS[arm],
                      label=f"Arm {arm}", zorder=3,
                      # a surface-coloured edge creates the 2px gap between
                      # adjacent bars that keeps groups legible
                      edgecolor=SURFACE, linewidth=1.5)
        # Direct value labels: required relief for the low-contrast slot.
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK_SOFT)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([l for _k, l in metrics], color=INK_SOFT)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score", color=INK_SOFT, fontsize=9)
    ax.set_title("Answer correctness by arm", color=INK, fontsize=11,
                 loc="left", pad=28)
    # Legend ABOVE the plot area: inside the axes it collides with the value
    # labels whenever a bar reaches the top of the scale.
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right",
                    bbox_to_anchor=(1.0, 1.01), ncols=3)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    _save(fig, out / "plot_correctness.png")


def plot_faithfulness(summaries: dict[str, dict], out: Path) -> None:
    """
    The headline chart: faithfulness per arm.

    Only one series, so no legend box - the title names what is plotted.
    """
    arms = sorted(summaries)
    vals = [summaries[a].get("faithfulness") or 0.0 for a in arms]

    fig, ax = _new_fig(5.4, 4.0)
    bars = ax.bar([ARM_LABEL[a] for a in arms], vals, width=0.52,
                  color=[ARM_COLORS[a] for a in arms], zorder=3,
                  edgecolor=SURFACE, linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, min(1.08, max(vals) * 1.25 + 0.08) if any(vals) else 1.0)
    ax.set_ylabel("fraction of claims ENTAILED", color=INK_SOFT, fontsize=9)
    ax.set_title("Faithfulness by arm", color=INK, fontsize=11,
                 loc="left", pad=12)
    _save(fig, out / "plot_faithfulness.png")


def plot_cost_quality(summaries: dict[str, dict], out: Path) -> None:
    """
    The question the project exists to answer: is verification worth it?

    x = tokens per query (our primary cost metric - a property of the
    pipeline, unlike wall-clock latency which is partly OpenRouter's), and
    y = faithfulness. Up and to the left is better. Each point is labelled
    directly, so this works without the legend.
    """
    arms = sorted(summaries)
    fig, ax = _new_fig(5.8, 4.2)

    for arm in arms:
        x = summaries[arm].get("tokens_per_query") or 0.0
        y = summaries[arm].get("faithfulness") or 0.0
        ax.scatter([x], [y], s=150, color=ARM_COLORS[arm], zorder=3,
                   # 2px surface ring keeps overlapping marks separable
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"Arm {arm}", (x, y), textcoords="offset points",
                    xytext=(10, 6), fontsize=9, color=INK)

    ax.set_xlabel("tokens per query (cost)", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("faithfulness", color=INK_SOFT, fontsize=9)
    ax.set_title("What faithfulness costs", color=INK, fontsize=11,
                 loc="left", pad=12)
    ax.set_xlim(left=0)
    _save(fig, out / "plot_cost_quality.png")


def plot_verdicts(summaries: dict[str, dict], out: Path) -> None:
    """
    Stacked bars: the verdict mix per arm, as percentages.

    Uses the reserved status palette, and every segment over 4% is labelled,
    because warning-yellow is below 3:1 on a light surface and must not carry
    meaning by colour alone.
    """
    arms = sorted(summaries)
    order = ["ENTAILED", "UNSUPPORTED", "CONTRADICTED"]

    fig, ax = _new_fig(5.8, 4.0)
    bottoms = [0.0] * len(arms)

    for verdict in order:
        heights = []
        for arm in arms:
            counts = summaries[arm].get("verdict_counts", {}) or {}
            total = sum(counts.values()) or 1
            heights.append(100 * counts.get(verdict, 0) / total)
        ax.bar([ARM_LABEL[a] for a in arms], heights, bottom=bottoms,
               width=0.52, color=VERDICT_COLORS[verdict], label=verdict,
               zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for i, h in enumerate(heights):
            if h >= 4:
                ax.text(i, bottoms[i] + h / 2, f"{h:.0f}%", ha="center",
                        va="center", fontsize=8, color="#ffffff")
            bottoms[i] += h

    ax.set_ylim(0, 100)
    ax.set_ylabel("% of claims", color=INK_SOFT, fontsize=9)
    ax.set_title("Claim verdict mix by arm", color=INK, fontsize=11,
                 loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower center",
                    bbox_to_anchor=(0.5, -0.22), ncols=3)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    _save(fig, out / "plot_verdicts.png")


# ===========================================================================
# Tables
# ===========================================================================

def _row(label: str, summaries: dict[str, dict], key: str, fmt: str,
         scale: float = 1.0) -> str:
    cells = []
    for arm in sorted(summaries):
        v = summaries[arm].get(key)
        cells.append("-" if v is None else fmt.format(v * scale))
    return f"| {label} | " + " | ".join(cells) + " |"


def write_tables(summaries: dict[str, dict], out: Path, meta: dict) -> None:
    arms = sorted(summaries)
    head = "| metric | " + " | ".join(f"Arm {a}" for a in arms) + " |"
    sep = "|---" * (len(arms) + 1) + "|"

    # --- ablation table ---
    lines = [
        "# Ablation results",
        "",
        f"- questions: **{meta.get('n_questions', '?')}** "
        f"(HotpotQA distractor, seed {meta.get('seed', '?')})",
        f"- generator (fixed in every arm): `{meta.get('generator_model', '?')}`",
        f"- helper model: `{meta.get('helper_model', '?')}`",
        "",
        "| Arm | What it is |",
        "|---|---|",
    ]
    for a in arms:
        lines.append(f"| {a} | {summaries[a].get('arm_description', '')} |")
    lines += ["", "## Quality", "", head, sep,
              _row("Exact match", summaries, "exact_match", "{:.3f}"),
              _row("Token F1", summaries, "token_f1", "{:.3f}"),
              _row("LLM-judge correct", summaries, "judge_correct", "{:.3f}"),
              _row("Faithfulness", summaries, "faithfulness", "{:.3f}"),
              "", "## Claim-level behaviour", "", head, sep,
              _row("Claims per answer", summaries, "claims_per_answer", "{:.2f}"),
              _row("% claims flagged", summaries, "pct_claims_flagged", "{:.1f}"),
              _row("Claims corrected", summaries, "n_claims_corrected", "{:.0f}"),
              _row("Claims dropped", summaries, "n_claims_dropped", "{:.0f}"),
              _row("Re-plan rate", summaries, "replan_rate", "{:.2f}"),
              _row("Correction rate", summaries, "correction_rate", "{:.2f}"),
              "",
              "Faithfulness for arms A and B is measured by **shadow "
              "verification**: the verifier runs and records verdicts, but "
              "those arms never act on them. Without this the metric would be "
              "undefined for two of the three arms. Shadow calls are excluded "
              "from the cost figures below.",
              ""]
    (out / "ablation_table.md").write_text("\n".join(lines))
    print("  wrote ablation_table.md")

    # --- cost table ---
    provider = meta.get("provider", "")
    gen = meta.get("generator_model", "")
    free = all((summaries[a].get("usd_per_query") or 0) == 0 for a in arms)

    lines = [
        "# Cost and latency",
        "",
        f"- provider: `{provider}`  generator: `{gen}`",
        "",
        "LLM calls and tokens are the **primary** cost metric. Wall-clock "
        "latency is secondary: it depends on the serving environment "
        + ("(local inference, so it reflects this machine's throughput rather "
           "than any hosted service)" if provider == "ollama"
           else "(a hosted API, which adds queueing variance we do not control)")
        + ", whereas token counts are a property of the pipeline itself.",
        "",
        "Latency figures **exclude shadow-verification time**, matching the "
        "cost columns. Arms A and B run the verifier only to make their "
        "faithfulness measurable; they would never do so in deployment, and "
        "charging them that time made arm B appear slower than arm C even "
        "though C does strictly more work.",
        "",
        head, sep,
        _row("LLM calls / query", summaries, "llm_calls_per_query", "{:.1f}"),
        _row("Tokens / query", summaries, "tokens_per_query", "{:.0f}"),
        _row("USD / query", summaries, "usd_per_query", "${:.5f}"),
        _row("USD / 1000 queries", summaries, "usd_per_query", "${:.2f}", 1000),
        _row("Latency median (s)", summaries, "latency_median_ms", "{:.2f}", 0.001),
        _row("Latency p90 (s)", summaries, "latency_p90_ms", "{:.2f}", 0.001),
        "",
    ]
    if free:
        lines += [
            "The USD rows are zero because this run used local inference, "
            "where price per token is zero by definition. That is exactly why "
            "the project treats **tokens and call counts** as the primary "
            "cost metric throughout: they stay meaningful regardless of who "
            "is billing, and they are what a hosted deployment would be "
            "charged for.",
            "",
        ]
    # the overhead multiple, which is the number the viva will ask about
    base = summaries.get("A", {}).get("tokens_per_query") or 0
    if base:
        lines += ["## Overhead relative to baseline RAG (arm A)", "",
                  "| Arm | token multiple | latency multiple |", "|---|---|---|"]
        base_lat = summaries["A"].get("latency_median_ms") or 1
        for a in arms:
            tok = (summaries[a].get("tokens_per_query") or 0) / base
            lat = (summaries[a].get("latency_median_ms") or 0) / base_lat
            lines.append(f"| {a} | {tok:.2f}x | {lat:.2f}x |")
        lines.append("")
    (out / "cost_latency_table.md").write_text("\n".join(lines))
    print("  wrote cost_latency_table.md")


# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="run directory name (default: newest)")
    args = ap.parse_args()

    cfg = load_config()
    runs = sorted(
        (d for d in cfg.runs_dir.iterdir()
         if d.is_dir() and not d.name.startswith("ragtruth")),
        key=lambda d: d.name,
    )
    if not runs:
        raise SystemExit("No runs found. Run `python -m src.run_eval` first.")

    run_dir = cfg.runs_dir / args.run if args.run else runs[-1]
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    summaries: dict[str, dict[str, Any]] = {}
    for d in sorted(run_dir.glob("arm*")):
        f = d / "summary.json"
        if f.exists():
            summaries[d.name.replace("arm", "")] = json.loads(f.read_text())
    if not summaries:
        raise SystemExit(f"No arm summaries in {run_dir}")

    meta_f = run_dir / "config.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}

    print(f"Reporting on {run_dir.name}: arms {', '.join(sorted(summaries))}")
    write_tables(summaries, run_dir, meta)
    plot_correctness(summaries, run_dir)
    plot_faithfulness(summaries, run_dir)
    plot_cost_quality(summaries, run_dir)
    plot_verdicts(summaries, run_dir)
    print(f"\nAll outputs in {run_dir}")


if __name__ == "__main__":
    main()
