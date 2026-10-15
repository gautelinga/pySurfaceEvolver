"""Draw the picture of docs/pendant_drop.md: the largest drops on a ring at four
Bond numbers against the exact profiles, the largest volume and the tipping
onset against the Bond number, and a drop at Bond 12 pushed along its tipping
mode. Needs matplotlib and SciPy; takes about three minutes.

    OMP_NUM_THREADS=4 python tools/pendant_drop_figure.py

writes docs/_static/pendant_drop.png.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysurfaceevolver import recipes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
import pendant_drop_helpers as H  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")
R = H.RING


def largest(bond):
    """The run to the fold as on the page: the fold's volume and the vertices
    (in units of R) at the point of largest volume."""
    ev = H.hanging_drop(bond, np.pi/bond)
    rows, best = [], (0.0, None)
    for step in recipes.arclength(ev, ev.body(1), 0.01*R, 300):
        rows.append((step.value/R**3, step.pressures[0]*R, -ev.vertices[:, 2].min()/R))
        if rows[-1][0] > best[0]:
            best = (rows[-1][0], ev.vertices/R)
        if len(rows) > 3 and rows[-1][0] < rows[-2][0] < rows[-3][0]:
            break
    return H.fold(rows)[0], best[1]


def tipping(bond):
    """The run to the tipping onset as on the page: the onset volume, and the
    drop's section through the axis there and pushed along the first mode."""
    ev = H.hanging_drop(bond, 0.7*np.pi/bond)
    last = None
    for step in recipes.arclength(ev, ev.body(1), 0.004*R, 300):
        modes = ev.modes(3)
        here = (step.value/R**3, modes.values[0])
        if step.result.negative_modes:
            break
        last = here
    (v0, e0), (v1, e1) = last, here
    u = modes.vectors[0]
    d = np.array([u[:, 0].mean(), u[:, 1].mean(), 0.0])
    d /= np.linalg.norm(d)                         # the way it tips
    normal = (-d[1], d[0], 0.0)
    x = ev.vertices
    def section():
        c = ev.mesh().slice(normal)[0]
        return np.c_[c @ d, c[:, 2]]/R
    before = section()
    ev.vertices = x + 0.08*u*R/np.abs(u).max()
    after = section()
    ev.vertices = x
    return v0 + (v1 - v0)*e0/(e0 - e1), before, after


def main():
    warnings.simplefilter("ignore")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1.2, 1]))

    # the largest drops
    folds = {}
    for bond, colour in zip((1, 2, 4, 8), COLOURS):
        volume, x = largest(bond)
        folds[bond] = volume
        exact = H.largest_drop(bond)
        p = H.profile(bond, exact["c"], exact["s"], dense=True)
        s = np.linspace(p.t[0], exact["s"], 400)
        xs, zs = p.sol(s)[0], p.sol(s)[1] - exact["height"]
        ax[0].plot(np.r_[-xs[::-1], xs], np.r_[zs[::-1], zs], color=colour, lw=1.2,
                   label=f"Bond {bond}")
        r = np.hypot(x[:, 0], x[:, 1])
        side = np.where(x[:, 0] >= 0, 1, -1)
        near = np.abs(x[:, 1]) < 0.03*np.maximum(r, 0.3)   # vertices near the plane y = 0
        ax[0].plot(side[near]*r[near], x[near, 2], ".", color=colour, ms=2.5)
        print(f"Bond {bond}: fold {volume:.4f} (exact {exact['volume']:.4f})", flush=True)
    ax[0].plot([-1, 1], [0, 0], "k", lw=2.5)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("distance from the axis / R")
    ax[0].set_ylabel("height / R")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[0].set_title("the largest drops", fontsize=10)

    # the largest volume and the tipping onset against the Bond number
    critical = 10.363
    bonds = np.r_[np.geomspace(0.5, critical, 16), np.linspace(10.6, 14.6, 6)]
    vmax = [H.largest_drop(b)["volume"] for b in bonds]
    side_bonds = np.r_[critical, np.linspace(10.6, 14.0, 7), np.linspace(14.1, 14.65, 6)]
    onsets = [H.largest_drop(critical)["volume"]] + \
        [H.sideways_onset(b)["volume"] for b in side_bonds[1:]]
    stable = bonds <= critical
    ax[1].semilogy(bonds[stable], np.array(vmax)[stable], "k", lw=1.2, label="largest volume")
    ax[1].semilogy(bonds[~stable], np.array(vmax)[~stable], "k:", lw=1.2)
    ax[1].semilogy(side_bonds, onsets, color="#d62728", lw=1.2, label="tipping starts")
    ax[1].semilogy(list(folds), list(folds.values()), "ko", ms=4, label="pySE")
    for bond in (11, 12, 13):
        onset, before, after = tipping(bond)
        ax[1].semilogy(bond, onset, "o", color="#d62728", ms=4)
        print(f"Bond {bond}: tipping from {onset:.4f}", flush=True)
        if bond == 12:
            sections = before, after
    ax[1].axvline(critical, color="0.7", lw=0.8)
    ax[1].axvline(14.68, color="0.7", lw=0.8)
    ax[1].text(critical, 0.055, "10.363 ", fontsize=8, color="0.4", ha="right")
    ax[1].text(14.68, 0.055, " 14.68", fontsize=8, color="0.4")
    ax[1].set_xlabel("Bond number")
    ax[1].set_ylabel("volume / R$^3$")
    ax[1].set_xlim(0, 16)
    ax[1].set_ylim(0.05, 15)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("where the drop goes unstable", fontsize=10)

    # tipping at Bond 12
    before, after = sections
    ax[2].plot(before[:, 0], before[:, 1], color="0.6", lw=1.5, label="at the onset")
    ax[2].plot(after[:, 0], after[:, 1], color="#d62728", lw=1.2, label="pushed along its mode")
    ax[2].plot([-1, 1], [0, 0], "k", lw=2.5)
    ax[2].set_aspect("equal")
    ax[2].set_xlabel("along the tipping / R")
    ax[2].set_ylabel("height / R")
    ax[2].legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.9))
    ax[2].set_ylim(-0.7, 0.1)
    ax[2].set_title("tipping at Bond 12", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "pendant_drop.png"), dpi=150)


if __name__ == "__main__":
    main()
