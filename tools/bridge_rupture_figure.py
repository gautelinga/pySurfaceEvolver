"""Draw the figure of docs/bridge_rupture.md: a liquid bridge between two unit
spheres (gap 0.6, contact angle 40) shrunk through its fold with
recipes.arclength, against the axisymmetric references (the mirror-symmetric
bridges and those with the neck off the mid-plane). Needs matplotlib and SciPy;
takes half a minute.

    OMP_NUM_THREADS=4 python tools/bridge_rupture_figure.py   # writes docs/_static/bridge_rupture.png
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
import stress_helpers as H  # noqa: E402

OUT = os.path.join(HERE, "..", "docs", "_static", "bridge_rupture.png")
GAP, THETA = 0.6, 40.0
CENTRE = 1 + GAP/2
BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"


def evolver_branch():
    """The page's run: (volume, neck, negative modes) at each point."""
    a = 0.35
    zb = CENTRE - np.sqrt(1 - a*a)
    tube = shapes.tube(a, -zb, zb, nz=4, nt=16)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   constraints={1: C.sphere((0, 0, -CENTRE), 1.0, contact_angle=THETA,
                                            wet_poles=("north",)),
                                2: C.sphere((0, 0, CENTRE), 1.0, contact_angle=THETA,
                                            wet_poles=("south",))},
                   vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=0.15)])
    ev.relax(levels=3)
    rows = []
    for step in recipes.arclength(ev, ev.body(1), -0.0075, 60):
        ring = ev.mesh().slice((0, 0, 1))[0][:-1]
        rows.append((step.value, np.hypot(ring[:, 0], ring[:, 1]).mean(),
                     step.result.negative_modes))
        if rows[-1][1] < 0.085:
            break
    return np.array(rows)


def symmetric_family():
    """(neck, volume) of the mirror-symmetric bridges, widest neck first."""
    rows = H._symmetric_family(GAP, THETA, 1.0, np.geomspace(0.07, 0.24, 120))
    return np.array([(rn, H._symmetric(rn, p, GAP, THETA, 1.0)["volume"]) for rn, p in rows])


def offset_family(branch):
    """(neck height, neck, volume, energy, pressure) of the bridges with the neck off the
    mid-plane, out from the branch point."""
    rows, guess = [], (branch["neck"], branch["pressure"])
    for zn in np.linspace(0.002, 0.16, 80):
        b = H.bridge_spheres_offset(zn, GAP, THETA, guess=guess)
        guess = (b["neck"], b["pressure"])
        rows.append((zn, b["neck"], b["volume"], b["energy"], b["pressure"]))
    return np.array(rows)


def meridian(rn, p, zn):
    """The profile (r, z) of the bridge with neck rn at height zn, both halves."""
    halves = []
    for sign in (1, -1):
        def hit(s, y):
            return y[0]**2 + (y[1] - CENTRE)**2 - 1.0
        hit.terminal, hit.direction = True, -1
        sol = H.profile(rn, sign*zn, np.pi/2, p, events=(hit,), s_max=4*(CENTRE + 1))
        halves.append((sol.y[0], sign*sol.y[1]))
    r = np.concatenate([halves[1][0][::-1], halves[0][0]])
    z = np.concatenate([halves[1][1][::-1], halves[0][1]])
    return r, z


def main():
    warnings.simplefilter("ignore")
    fold = H.bridge_spheres_fold(GAP, THETA)
    branch = H.bridge_spheres_branch(GAP, THETA)
    sym = symmetric_family()
    off = offset_family(branch)
    ev = evolver_branch()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7.6))
    # neck against volume: the symmetric branch and Evolver's points
    stable = sym[:, 0] >= branch["neck"]
    ax1.plot(sym[stable, 1], sym[stable, 0], color=BLUE, label="symmetric, stable")
    ax1.plot(sym[~stable, 1], sym[~stable, 0], color=BLUE, ls="--",
             label="symmetric, unstable")
    for m, colour in ((0, BLUE), (1, ORANGE), (2, RED)):
        sel = ev[:, 2] == m
        ax1.plot(ev[sel, 0], ev[sel, 1], "o", ms=4, mfc="none", color=colour,
                 label=f"Evolver, {m} negative mode{'s' if m != 1 else ''}")
    ax1.plot(branch["volume"], branch["neck"], "k^", ms=7, label="branch point")
    ax1.plot(fold["volume"], fold["neck"], "kv", ms=7, label="fold (least volume)")
    ax1.set_xlabel("volume")
    ax1.set_ylabel("neck radius")
    ax1.set_xlim(0.07, 0.1)
    ax1.set_ylim(0.07, 0.175)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.set_title("Shrinking the symmetric bridge", fontsize=10)

    # the pitchfork: the neck's height against volume
    ax2.plot([branch["volume"], 0.1], [0, 0], color=BLUE, label="symmetric, stable")
    ax2.plot([fold["volume"], branch["volume"]], [0, 0], color=BLUE, ls="--",
             label="symmetric, unstable")
    for sign in (1, -1):
        ax2.plot(off[:, 2], sign*off[:, 0], color=ORANGE, ls="--",
                 label="neck off the mid-plane (unstable)" if sign == 1 else None)
    ax2.plot(branch["volume"], 0, "k^", ms=7)
    ax2.plot(fold["volume"], 0, "kv", ms=7)
    ax2.set_xlim(0.07, 0.1)
    ax2.set_ylim(-0.12, 0.12)
    ax2.set_xlabel("volume")
    ax2.set_ylabel("height of the neck")
    ax2.legend(fontsize=7.5, loc="lower left")
    ax2.set_title("A subcritical pitchfork", fontsize=10)

    # the barrier: the off-centre bridge's energy above the symmetric one's
    above = off[:, 2] <= 0.1
    diff, guess = [], (branch["neck"], branch["pressure"])
    for v, e in zip(off[above, 2], off[above, 3]):
        s = H.bridge_spheres(v, GAP, THETA, guess=guess)      # the stable one, followed
        guess = (s["neck"], s["pressure"])
        diff.append(e - s["energy"])
    ax3.plot(off[above, 2], diff, color=ORANGE)
    ax3.plot(branch["volume"], 0, "k^", ms=7)
    ax3.set_xlim(0.07, 0.1)
    ax3.set_xlabel("volume")
    ax3.set_ylabel("energy above the symmetric bridge")
    ax3.set_title("The barrier: the off-centre bridge is a saddle", fontsize=10)

    # shapes at one volume: the stable symmetric bridge and the off-centre saddle
    k = np.argmin(abs(off[:, 0] - 0.1))
    target = off[k, 2]
    s = H.bridge_spheres(target, GAP, THETA, guess=guess)
    t = np.linspace(0, 2*np.pi, 200)
    for zc in (-CENTRE, CENTRE):
        ax4.fill(np.cos(t), zc + np.sin(t), color="#dddddd", zorder=0)
    r, z = meridian(s["neck"], s["pressure"], 0.0)
    ax4.plot(r, z, color=BLUE, label="symmetric (stable)")
    ax4.plot(-r, z, color=BLUE)
    r, z = meridian(off[k, 1], off[k, 4], off[k, 0])
    ax4.plot(r, z, color=ORANGE, ls="--", label="off-centre (saddle)")
    ax4.plot(-r, z, color=ORANGE, ls="--")
    ax4.set_aspect("equal")
    ax4.set_xlim(-0.6, 1.2)
    ax4.set_ylim(-0.6, 0.6)
    ax4.set_axis_off()
    ax4.legend(fontsize=7.5, loc="center right")
    ax4.set_title(f"Two bridges of volume {target:.3f} (a section)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
