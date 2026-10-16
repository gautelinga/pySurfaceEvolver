"""Draw the pictures of docs/floating_lens.md: cross-sections of a lens without
and with gravity against the exact ones, the thickness of lenses under gravity
against their size, and two whole lenses. Needs matplotlib, PyVista (off-screen
rendering) and SciPy; takes about a minute.

    OMP_NUM_THREADS=4 python tools/floating_lens_figure.py

writes docs/_static/floating_lens.png (the curves) and
docs/_static/floating_lens_shapes.png.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))
import floating_lens_helpers as H  # noqa: E402

STATIC = os.path.join(HERE, "..", "docs", "_static")
TENSIONS = (1.0, 0.7, 0.5)
RHO, G = (0.9, 1.0), 100.0
TOP, BOTTOM, BATH = "#d62728", "#1f77b4", "#555555"


def snapshot(ev):
    """The surface (one engine per process: keep it before loading the next):
    vertices, facets, and which surface each facet is on (0 top, 1 bottom, 2 bath)."""
    m = ev.mesh()
    x, f = m.vertices.copy(), m.facets.copy()
    lens = (m.facet_bodies == 1).any(axis=1)
    a, b, c = (x[f[:, k]] for k in range(3))
    nz = np.cross(b - a, c - a)[:, 2]
    return x, f, np.where(lens & (nz > 0), 0, np.where(lens, 1, 2))


def section(snap, kind):
    """The vertices of one surface on the mirror plane y = 0, sorted by x."""
    x, f, kinds = snap
    rows = np.unique(f[kinds == kind])
    rows = rows[np.abs(x[rows, 1]) < 1e-9]
    rows = rows[np.argsort(x[rows, 0])]
    return x[rows, 0], x[rows, 2]


def gravity_run(volume):
    ref = H.axisymmetric_lens(volume, *TENSIONS, *RHO, G)
    ev = H.floating_lens(volume, *TENSIONS, gravity=G, rho_lens=RHO[0], rho_bath=RHO[1],
                         rim=ref["rim"], h=0.15, h_rim=0.03)
    return snapshot(ev), ref, H.lens_shape(ev)


def main():
    warnings.simplefilter("ignore")
    flat = snapshot(H.floating_lens(1.0, *TENSIONS, levels=2))
    ex = H.neumann(*TENSIONS, 1.0)
    runs = {volume: gravity_run(volume) for volume in (0.25, 1.0, 4.0)}
    volumes = np.geomspace(0.02, 64, 24)
    curve = [H.axisymmetric_lens(v, *TENSIONS, *RHO, G) for v in volumes]
    e, _ = H.langmuir(*TENSIONS, *RHO, G)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    # without gravity
    r = np.linspace(0, ex["rim"], 200)
    zt = np.sqrt(ex["R_top"]**2 - r**2) - (ex["R_top"] - ex["h_top"])
    zb = (ex["R_bottom"] - ex["h_bottom"]) - np.sqrt(ex["R_bottom"]**2 - r**2)
    ax[0].plot(r, zt, color=TOP, lw=1.2, label="exact")
    ax[0].plot(r, zb, color=BOTTOM, lw=1.2)
    ax[0].plot([ex["rim"], 1.6], [0, 0], color=BATH, lw=1.2)
    for kind in (0, 1, 2):
        xs, zs = section(flat, kind)
        keep = xs < 1.6
        ax[0].plot(xs[keep], zs[keep], ".", color="k", ms=2.5,
                   label="pySE (y = 0)" if kind == 0 else None)
    ax[0].set_aspect("equal")
    ax[0].set_xlim(0, 1.6)
    ax[0].set_xlabel("r")
    ax[0].set_ylabel("z")
    ax[0].set_title(f"no gravity: caps at {ex['angle_top']:.1f} and {ex['angle_bottom']:.1f} degrees",
                    fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right")
    # with gravity
    for volume, ls in ((1.0, "-"), (4.0, "--")):
        snap, ref, _ = runs[volume]
        for key, colour in (("profile_top", TOP), ("profile_bottom", BOTTOM), ("profile_bath", BATH)):
            rr, zz = ref[key]
            ax[1].plot(rr, zz, ls, color=colour, lw=1.1)
        for kind in (0, 1, 2):
            xs, zs = section(snap, kind)
            keep = xs < ref["rim"] + 0.6
            ax[1].plot(xs[keep], zs[keep], ".", color="k", ms=1.5)
        ax[1].text(ref["rim"] - 0.05, 0.04, f"V = {volume:g}", ha="right", fontsize=8)
    ax[1].set_xlim(0, 3.2)
    ax[1].set_ylim(-0.3, 0.1)
    ax[1].set_xlabel("r")
    ax[1].set_ylabel("z (stretched 5 times)")
    ax[1].set_aspect(5)
    ax[1].set_title("g = 100: axisymmetric (lines), pySE (dots)", fontsize=10)
    # thickness against size
    ax[2].plot([c["rim"] for c in curve], [c["thickness"] for c in curve], color="k", lw=1.2,
               label="axisymmetric")
    ax[2].plot([s["rim"] for _, _, s in runs.values()],
               [s["thickness"] for _, _, s in runs.values()], "o",
               mfc="none", color="#2ca02c", label="pySE")
    ax[2].axhline(e, color="0.4", ls=":", label="Langmuir")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("rim radius")
    ax[2].set_ylabel("thickness at the centre")
    ax[2].set_title("g = 100: capillary length of the bath 0.1", fontsize=10)
    ax[2].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "floating_lens.png"), dpi=110)

    import pyvista as pv
    p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1100, 440), border=False)
    colours = ("#f4a582", "#92c5de", "#e0e0e0")
    for k, (snap, limit, stretch) in enumerate(((flat, 1.4, 1.0), (runs[4.0][0], 2.9, 4.0))):
        p.subplot(0, k)
        x, f, kinds = snap
        # three quadrants, the fourth cut away to show the section
        for sx, sy in ((1, 1), (-1, 1), (-1, -1)):
            v = x*[sx, sy, stretch]
            for j, colour in enumerate(colours):
                sel = f[kinds == j]
                mesh = pv.PolyData(v, np.c_[np.full(len(sel), 3), sel].ravel())
                if j == 2:              # the water cut round at radius ``limit``
                    mesh["r"] = np.hypot(v[:, 0], v[:, 1])
                    mesh = mesh.clip_scalar("r", value=limit, invert=True)
                p.add_mesh(mesh, color=colour, smooth_shading=True, ambient=0.3, diffuse=0.7)
        p.camera_position = [(1.6*limit, -2.4*limit, 1.3*limit), (0, 0, -0.1*stretch), (0, 0, 1)]
        p.camera.zoom(0.9)
        p.add_text(("no gravity, V = 1" if k == 0 else "g = 100, V = 4 (height x4)"),
                   font_size=9, position="upper_left", color="black")
    p.set_background("white")
    p.screenshot(os.path.join(STATIC, "floating_lens_shapes.png"))


if __name__ == "__main__":
    main()
