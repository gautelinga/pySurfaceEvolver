"""Draw the pictures of docs/bridge_rupture.md: a liquid bridge between two unit
spheres (gap 0.6, contact angle 40) shrunk through its fold with
recipes.arclength, against the axisymmetric references (the mirror-symmetric
bridges and those with the neck off the mid-plane). Needs matplotlib, PyVista
(off-screen rendering) and SciPy; takes about a minute.

    OMP_NUM_THREADS=4 python tools/bridge_rupture_figure.py

writes docs/_static/bridge_rupture*.png: the plots, the bridge along the run
(_shapes), its unstable mode (_mode), and the symmetric and off-centre bridges
of one volume in 3D (_saddle).
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

STATIC = os.path.join(HERE, "..", "docs", "_static")
OUT = os.path.join(STATIC, "bridge_rupture.png")
GAP, THETA = 0.6, 40.0
CENTRE = 1 + GAP/2
BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"
LIQUID = dict(color="#a6cee3", show_edges=True, edge_color="#7aa9d0", line_width=0.2,
              smooth_shading=False, ambient=0.25, diffuse=0.75)


def evolver_branch():
    """The page's run: (volume, neck, negative modes) at each point, the mesh
    (vertices, facets) at each point, and the first unstable bridge's mesh with
    the displacement of a saddle() step from it."""
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
    rows, meshes, first_unstable = [], [], None
    for step in recipes.arclength(ev, ev.body(1), -0.003, 100):
        m = ev.mesh()
        ring = m.slice((0, 0, 1))[0][:-1]
        rows.append((step.value, np.hypot(ring[:, 0], ring[:, 1]).mean(),
                     step.result.negative_modes))
        meshes.append((m.vertices.copy(), m.facets.copy()))
        if rows[-1][2] == 1 and first_unstable is None:
            first_unstable = ev.save()
        if rows[-1][1] < 0.085:
            break
    ev.restore(first_unstable)
    m = ev.mesh()
    ev.saddle()
    mode = (m.vertices.copy(), m.facets.copy(), ev.mesh().vertices - m.vertices)
    return np.array(rows), meshes, mode


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


def polydata(vertices, facets):
    import pyvista as pv
    return pv.PolyData(np.asarray(vertices, float),
                       np.hstack([np.full((len(facets), 1), 3), facets]).ravel())


def spheres(p):
    import pyvista as pv
    for zc in (-CENTRE, CENTRE):
        p.add_mesh(pv.Sphere(radius=1.0, center=(0, 0, zc), theta_resolution=120,
                             phi_resolution=120), color="#c8c8c8", ambient=0.3, diffuse=0.7)


def side_view(p, scale=0.62):
    """Looking at the bridge from the side, a little from above."""
    v = np.array([0.0, -1.0, 0.25])
    v /= np.linalg.norm(v)
    p.camera.focal_point = (0, 0, 0)
    p.camera.position = tuple(4*v)
    p.camera.up = (0, 0, 1)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = scale


def render_shapes(rows, meshes, picks, labels):
    import pyvista as pv
    n = len(picks)
    p = pv.Plotter(shape=(1, n), off_screen=True, window_size=(330*n, 400), border=False,
                   lighting="three lights")
    p.set_background("white")
    for k, (i, label) in enumerate(zip(picks, labels)):
        p.subplot(0, k)
        spheres(p)
        p.add_mesh(polydata(*meshes[i]), **LIQUID)
        p.add_text(label, position="upper_left", font_size=9, color="black")
        side_view(p)
    p.screenshot(os.path.join(STATIC, "bridge_rupture_shapes.png"))
    p.close()


def render_mode(mode, exaggerate=6.0):
    import pyvista as pv
    x, facets, dx = mode
    phi = np.arctan2(x[:, 1], x[:, 0])
    outward = dx[:, 0]*np.cos(phi) + dx[:, 1]*np.sin(phi)
    outward /= abs(outward).max()
    p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(760, 420), border=False,
                   lighting="three lights")
    p.set_background("white")
    p.subplot(0, 0)
    spheres(p)
    mesh = polydata(x, facets)
    mesh["outward motion"] = outward
    p.add_mesh(mesh, scalars="outward motion", cmap="coolwarm", clim=(-1, 1),
               smooth_shading=True, ambient=0.3, diffuse=0.7,
               scalar_bar_args=dict(title="outward motion", color="black", vertical=False,
                                    position_x=0.2, position_y=0.04, width=0.6, height=0.07,
                                    label_font_size=10, title_font_size=11, n_labels=3,
                                    fmt="%.0f"))
    p.add_text("the unstable mode", position="upper_left", font_size=9, color="black")
    side_view(p)
    p.subplot(0, 1)
    spheres(p)
    p.add_mesh(polydata(x, facets), color="#bbbbbb", opacity=0.3, smooth_shading=True)
    p.add_mesh(polydata(x + exaggerate*dx, facets), **LIQUID)
    p.add_text(f"pushed along it (exaggerated {exaggerate:g}x;\ngrey: before)",
               position="upper_left", font_size=9, color="black")
    side_view(p)
    p.screenshot(os.path.join(STATIC, "bridge_rupture_mode.png"))
    p.close()


def revolve(r, z, n=64):
    """The surface of revolution of a meridian (r, z), as a structured grid."""
    import pyvista as pv
    t = np.linspace(0, 2*np.pi, n)
    R, T = np.meshgrid(r, t, indexing="ij")
    Z = np.repeat(z[:, None], n, axis=1)
    return pv.StructuredGrid(R*np.cos(T), R*np.sin(T), Z)


def render_saddle(sym_meridian, off_meridian, volume):
    import pyvista as pv
    p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(760, 420), border=False,
                   lighting="three lights")
    p.set_background("white")
    for k, ((r, z), label, colour) in enumerate((
            (sym_meridian, "symmetric: stable", "#a6cee3"),
            (off_meridian, "neck off the mid-plane: a saddle", "#fdbf6f"))):
        p.subplot(0, k)
        spheres(p)
        p.add_mesh(revolve(r, z), color=colour, smooth_shading=True, ambient=0.25,
                   diffuse=0.75)
        p.add_text(f"{label}\nvolume {volume:.3f}", position="upper_left", font_size=9,
                   color="black")
        side_view(p)
    p.screenshot(os.path.join(STATIC, "bridge_rupture_saddle.png"))
    p.close()


def main():
    warnings.simplefilter("ignore")
    fold = H.bridge_spheres_fold(GAP, THETA)
    branch = H.bridge_spheres_branch(GAP, THETA)
    sym = symmetric_family()
    off = offset_family(branch)
    ev, meshes, mode = evolver_branch()

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

    onset = int(np.flatnonzero(ev[:, 2] == 1)[0])
    fold_at = int(np.argmin(ev[:, 0]))
    picks = [0, onset, fold_at, len(ev) - 1]
    labels = [f"volume {ev[i, 0]:.4f}\nneck {ev[i, 1]:.3f}\n"
              f"{int(ev[i, 2])} negative mode{'' if ev[i, 2] == 1 else 's'}" for i in picks]
    render_shapes(ev, meshes, picks, labels)
    render_mode(mode)
    render_saddle(meridian(s["neck"], s["pressure"], 0.0),
                  meridian(off[k, 1], off[k, 4], off[k, 0]), target)
    print("wrote", os.path.normpath(STATIC) + "/bridge_rupture*.png")


if __name__ == "__main__":
    main()
