"""Draw the run's progress from llm.c's main.log: loss (train and val) and HellaSwag accuracy.

Usage: python plot_progress.py MAIN_LOG OUT_PNG [--baseline TXT] [--upto STEP]
--upto drops everything logged after STEP, so a figure named for a step shows that step's state.
--baseline is the output of llm.c's dev/data/hellaswag.py for OpenAI's GPT-2 124M; its acc_norm
is drawn as the line to beat. The PNG is written to a temporary file and renamed, so a reader never
sees a half-written figure.
"""
import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

TOKENS_PER_STEP = 524_288
TOTAL_STEPS = 56_352
SMOOTH = 50  # train loss is drawn as a 50-step running mean

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2


def read_log(path, upto=None):
    series = {"trl": {}, "tel": {}, "eval": {}}
    pattern = re.compile(r"^s:(\d+) (trl|tel|eval):([0-9.]+)")
    with open(path) as f:
        for line in f:
            m = pattern.match(line)
            if m and (upto is None or int(m.group(1)) <= upto):
                # a resumed run logs some steps twice; the later value wins
                series[m.group(2)][int(m.group(1))] = float(m.group(3))
    return {k: sorted(v.items()) for k, v in series.items()}


def read_baseline(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        hits = re.findall(r"acc_norm: \d+/\d+=([0-9.]+)", f.read())
    return float(hits[-1]) if hits else None


def running_mean(points, window):
    out, values, total = [], [], 0.0
    for step, value in points:
        values.append(value)
        total += value
        if len(values) > window:
            total -= values[-window - 1]
        out.append((step, total / min(len(values), window)))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main_log")
    ap.add_argument("out_png")
    ap.add_argument("--baseline")
    ap.add_argument("--upto", type=int)
    args = ap.parse_args()
    log_path, out_path = args.main_log, args.out_png
    baseline = read_baseline(args.baseline)
    data = read_log(log_path, args.upto)
    train, val, hella = data["trl"], data["tel"], data["eval"]
    if not train:
        sys.exit(f"no training steps in {log_path}")
    step = train[-1][0]

    fig, (ax_loss, ax_hs) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    parts = [f"step {step:,} of {TOTAL_STEPS:,}", f"{step * TOKENS_PER_STEP / 1e9:.2f}B tokens"]
    if val:
        parts.append(f"val loss {val[-1][1]:.3f}")
    if hella:
        parts.append(f"HellaSwag {hella[-1][1]:.1%}")
    fig.suptitle("GPT-2 124M on FineWeb-Edu  ·  " + "  ·  ".join(parts), color=INK, fontsize=12)

    # Loss: train (smoothed) and val share one axis and one unit.
    style(ax_loss)
    gap = 0  # vertical nudge in points when the two end labels would overlap
    smooth = [(s, v) for s, v in running_mean(train, SMOOTH) if s >= SMOOTH]
    val_shown = [(s, v) for s, v in val if s > 0]
    if smooth and val_shown and abs(smooth[-1][1] - val_shown[-1][1]) < 0.15:
        gap = 7
    if smooth:
        ax_loss.plot(*zip(*smooth), color=BLUE, linewidth=2, label=f"train ({SMOOTH}-step mean)")
        below = smooth[-1][1] <= (val_shown[-1][1] if val_shown else smooth[-1][1])
        end_label(ax_loss, *smooth[-1], f"{smooth[-1][1]:.3f}", -gap if below else gap)
    if val_shown:
        ax_loss.plot(*zip(*val_shown), color=ORANGE, linewidth=2, marker="o", markersize=6,
                     markeredgecolor=SURFACE, markeredgewidth=2, label="val")
        above = not smooth or val_shown[-1][1] > smooth[-1][1]
        end_label(ax_loss, *val_shown[-1], f"{val_shown[-1][1]:.3f}", gap if above else -gap)
    shown = [v for _, v in smooth] + [v for _, v in val_shown]
    if shown:
        ax_loss.set_ylim(min(shown) - 0.1, min(max(shown), 7.0) + 0.1)
    ax_loss.set_title("Loss", color=INK, fontsize=11, loc="left")
    ax_loss.set_xlabel("step")
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=INK)

    # HellaSwag: one series, against chance and the OpenAI GPT-2 124M score.
    style(ax_hs)
    ax_hs.axhline(0.25, color=AXIS, linewidth=1, linestyle=":")
    ax_hs.annotate("chance 25%", (0, 0.25), xytext=(2, -11), textcoords="offset points",
                   xycoords=("axes fraction", "data"), fontsize=8, color=MUTED)
    if baseline:
        ax_hs.axhline(baseline, color=MUTED, linewidth=1, linestyle="--")
        ax_hs.annotate(f"OpenAI GPT-2 124M {baseline:.1%}", (0, baseline), xytext=(2, 4),
                       textcoords="offset points", xycoords=("axes fraction", "data"),
                       fontsize=8, color=MUTED)
    if hella:
        ax_hs.plot(*zip(*hella), color=BLUE, linewidth=2, marker="o", markersize=6,
                   markeredgecolor=SURFACE, markeredgewidth=2)
        end_label(ax_hs, *hella[-1], f"{hella[-1][1]:.1%}")
    top = max([v for _, v in hella] + [baseline or 0.25]) + 0.01
    ax_hs.set_ylim(0.23, max(top, 0.31))
    ax_hs.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax_hs.set_title("HellaSwag accuracy (acc_norm)", color=INK, fontsize=11, loc="left")
    ax_hs.set_xlabel("step")

    right = max(step, 1000) * 1.08  # room for the end labels
    for ax in (ax_loss, ax_hs):
        ax.set_xlim(0, right)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, dpi=140, facecolor=SURFACE)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
