"""Capillary pressure against saturation, from the corpus' runs.

Each run's JSON (corpus.py) carries its curve: per step the draining body's
volume V and pressure p, the liquid trapped, the energy, whether the step
was reached by a fall. Here Pc = -p (the gas at 0: tension over the beads'
radius, both 1) and the saturation S = (V + trapped) / V0, V0 the liquid at
the start (the pore space up to the flat meniscus). A Haines jump is a fall:
the pressure drops at a fixed volume.

usage: python bench/topology/curves.py DIR [DIR ...] [--out OUT]

Writes OUT/NAME.csv per run, OUT/pc_<size>.png (a panel per contact angle,
the seeds overlaid) and OUT/pc_sizes.png (a panel per contact angle, the
sizes' runs overlaid). OUT defaults to the first DIR's curves/.
"""
import argparse, csv, json, os, re, sys, textwrap

import numpy as np

# (the reference palette's categorical slots, in order)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def runs(dirs):
    out = []
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.endswith(".json") and f != "summary.json":
                r = json.load(open(os.path.join(d, f)))
                if r.get("curve"):
                    out.append(r)
    return out


def table(r):
    """(step, V, trapped, S, Pc, energy, fell, changes) rows of a run."""
    v0 = float(r["start"])
    rows = []
    for k, V, p, trapped, energy, fell, changes in r["curve"]:
        rows.append((k, V, trapped, (V + trapped)/v0, -p, energy, int(fell), changes))
    return rows


def write_csv(r, out):
    with open(os.path.join(out, f"{r['name']}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "volume", "trapped", "saturation", "pc", "energy", "fell", "changes"])
        w.writerows(table(r))


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def panels(groups, title, path, label):
    """groups: {theta: [(key, run), ...]}; a panel per contact angle, the
    runs coloured by key (a fixed order of keys over all panels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thetas = sorted(groups)
    keys = sorted({k for g in groups.values() for k, _ in g})
    colour = {k: SLOTS[i % len(SLOTS)] for i, k in enumerate(keys)}
    width = max(6.4, 4.2*len(thetas))
    fig, axes = plt.subplots(1, len(thetas), figsize=(width, 4.4), sharey=True,
                             squeeze=False)
    for ax, theta in zip(axes[0], thetas):
        _style(ax)
        for k, r in sorted(groups[theta], key=lambda kr: str(kr[0])):
            t = np.array(table(r), float)
            if not len(t):
                continue
            ax.plot(t[:, 3], t[:, 4], color=colour[k], linewidth=1.5)
            end = "o" if r.get("reached") else "x"
            ax.plot(t[-1, 3], t[-1, 4], end, color=colour[k], markersize=6,
                    markeredgecolor="white" if end == "o" else colour[k], markeredgewidth=1.2)
        ax.set_title(f"contact angle {theta:g} degrees", fontsize=10, color=INK)
        ax.set_xlabel("saturation (liquid / pore space)", fontsize=9, color=INK)
        low = min([np.array(table(r), float)[:, 3].min() for _, r in groups[theta]
                   if r.get("curve")] or [0.0])
        ax.set_xlim(1.02, max(0.0, low - 0.05))
    axes[0][0].set_ylabel("capillary pressure Pc (tension / bead radius)", fontsize=9, color=INK)
    handles = [plt.Line2D([], [], color=colour[k], linewidth=1.5) for k in keys]
    # (the title, then the legend on a row of its own, the note wrapped to the width)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=11, color=INK)
    fig.legend(handles, [f"{label} {k}" for k in keys], loc="upper left",
               bbox_to_anchor=(0.01, 0.93), frameon=False, fontsize=9, ncol=len(keys))
    note = ("o: breakthrough reached; x: the run stopped short. A drop at a fixed saturation "
            "is a Haines jump.")
    fig.text(0.01, 0.01, textwrap.fill(note, int(width*17)), fontsize=8, color=MUTED,
             va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.dirs[0], "curves")
    os.makedirs(out, exist_ok=True)
    rs = runs(args.dirs)
    for r in rs:
        write_csv(r, out)
    sizes = sorted({r["beads"] for r in rs})
    for n in sizes:
        groups = {}
        for r in rs:
            if r["beads"] == n:
                groups.setdefault(r["theta"], []).append((r["seed"], r))
        kind = "periodic" if any(r["name"].startswith("p") for r in rs if r["beads"] == n) \
            else "walled"
        panels(groups, f"Drainage of {n}-bead packs ({kind}): capillary pressure against "
               "saturation", os.path.join(out, f"pc_{n}.png"), "seed")
    if len(sizes) > 1:
        groups = {}
        for r in rs:
            groups.setdefault(r["theta"], []).append((r["beads"], r))
        panels(groups, "Drainage, all pack sizes: capillary pressure against saturation",
               os.path.join(out, "pc_sizes.png"), "beads")
    print(f"{len(rs)} runs: curves in {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
