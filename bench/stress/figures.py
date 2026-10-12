"""Figures for docs/stress_cases.md: run the stress cases, store what they did,
and draw each case's result against its reference, its error per step, what
the solver did per step, and its shape at a few steps.

usage: python bench/stress/figures.py [--plot-only] [case numbers ...]

Runs go to bench/stress/figdata/ (not committed), figures to
docs/_static/stress/. ``--plot-only`` redraws from the stored runs. Needs
matplotlib, PyVista with off-screen rendering, and SciPy.
"""
import gzip, importlib, os, pickle, sys, time, warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "figdata")
OUT = os.path.join(HERE, "..", "..", "docs", "_static", "stress")

CASES = {1: "case01_cylinder", 2: "case02_catenoid", 3: "case03_sessile", 5: "case05_bridge",
         6: "case06_fibre", 8: "case08_band", 9: "case09_inflate", 11: "case11_puddle", 12: "case12_shrink",
         13: "case13_periodic", 14: "case14_sphere_drop", 15: "case15_double_bubble",
         16: "case16_pad_force", 17: "case17_wire_catenoid"}

BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"


# ---- running ---------------------------------------------------------------------

def run_case(n):
    mod = importlib.import_module(CASES[n])
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mod.run()
    print(f"case {n}: {'PASS' if r.passed else 'FAIL'} in {time.perf_counter() - t:.0f} s: {r.summary}")
    os.makedirs(DATA, exist_ok=True)
    with gzip.open(os.path.join(DATA, f"case{n:02d}.pkl.gz"), "wb") as fh:
        pickle.dump(dict(name=r.case, passed=r.passed, summary=r.summary, rows=r.rows,
                         calls=r.calls, extra=r.extra), fh)


def load_case(n):
    with gzip.open(os.path.join(DATA, f"case{n:02d}.pkl.gz"), "rb") as fh:
        return pickle.load(fh)


# ---- shared panels ---------------------------------------------------------------

def per_row_calls(d):
    """For each row, the relax() calls since the previous row's (inclusive of its own)."""
    out, prev = [], -1
    for row in d["rows"]:
        out.append(d["calls"][prev + 1: row["call"] + 1])
        prev = row["call"]
    return out


def solver_panel(ax, d, x, xlabel):
    """Gradient and Newton steps per row (stacked bars), time (line), and marks
    where relax() didn't converge or reported a saddle."""
    groups = per_row_calls(d)
    grad = np.array([sum(c["gradient"] for c in g) for g in groups])
    newton = np.array([sum(c["newton"] for c in g) for g in groups])
    secs = np.array([sum(c["seconds"] for c in g) for g in groups])
    pos = np.arange(len(x))
    ax.bar(pos, grad, color=BLUE, alpha=0.75, label="gradient steps")
    ax.bar(pos, newton, bottom=grad, color=ORANGE, label="Newton steps")
    ax.set_ylabel("steps")
    ax.set_xticks(pos)
    ax.set_xticklabels([fmt(v) for v in x], rotation=90, fontsize=7)
    ax.set_xlabel(xlabel)
    top = max(1, (grad + newton).max())
    for i, g in enumerate(groups):
        last = g[-1] if g else None
        if last and not last["converged"]:
            ax.plot(i, top*1.08, "x", color=RED, ms=7, mew=2)
        elif last and last["stable"] is False:
            ax.plot(i, top*1.08, "v", color=ORANGE, ms=7)
    ax.set_ylim(0, top*1.18)
    t = ax.twinx()
    t.plot(pos, secs, "o-", color="k", ms=3, lw=1, label="time")
    t.set_ylabel("seconds")
    t.set_ylim(0, max(secs.max()*1.25, 1e-3))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = t.get_legend_handles_labels()
    extra = []
    import matplotlib.lines as ml
    if any(g and not g[-1]["converged"] for g in groups):
        extra.append(ml.Line2D([], [], color=RED, marker="x", ls="", mew=2, label="not converged"))
    if any(g and g[-1]["converged"] and g[-1]["stable"] is False for g in groups):
        extra.append(ml.Line2D([], [], color=ORANGE, marker="v", ls="", label="reported unstable"))
    ax.legend(h1 + h2 + extra, l1 + l2 + [e.get_label() for e in extra], fontsize=7,
              loc="upper left", framealpha=0.9)
    ax.set_title("what relax() did", fontsize=10)


def fmt(v):
    if isinstance(v, str):
        return v
    v = float(v)
    if v != 0 and (abs(v) < 0.01 or abs(v) >= 1000):
        return f"{v:.0e}"
    return f"{v:.3g}"


def error_panel(ax, x, err, limit=None, label="relative error", xlabel="", logx=False,
                percent=True):
    ax.semilogy(x, np.maximum(err, 1e-16), "o-", color=BLUE, ms=4)
    if limit is not None:
        ax.axhline(limit, color=RED, ls="--", lw=1, label=f"pass limit {limit:.3g}"
                   + ("" if not percent else f" ({limit:.2%})"))
        ax.legend(fontsize=7, loc="best")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label)
    ax.set_title("error against the reference", fontsize=10)


def three_panels(title):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9), constrained_layout=True)
    fig.suptitle(title, fontsize=11)
    return fig, axs


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---- shapes ----------------------------------------------------------------------

def render(d, picks, name, context=None, view=(0.0, -1.0, 0.35), labels=None, zoom=1.0,
           same_scale=True):
    """Render the surface after the given rows, side by side; ``context(plotter,
    row)`` adds the walls, rings or fibre."""
    import pyvista as pv
    n = len(picks)
    p = pv.Plotter(shape=(1, n), off_screen=True, window_size=(330*n, 330), border=False,
                   lighting="three lights")
    p.set_background("white")
    meshes = []
    for i in picks:
        call = d["calls"][d["rows"][i]["call"]]
        f = call["facets"]
        faces = np.hstack([np.full((len(f), 1), 3), f]).ravel()
        meshes.append(pv.PolyData(call["vertices"], faces))
    extent = max(np.ptp(m.points, axis=0).max() for m in meshes)
    for k, (i, mesh) in enumerate(zip(picks, meshes)):
        p.subplot(0, k)
        p.add_mesh(mesh, color="#a6cee3", show_edges=True, edge_color="#4f86b5",
                   line_width=0.5, smooth_shading=False, ambient=0.25, diffuse=0.75)
        if context is not None:
            context(p, d["rows"][i])
        text = labels[k] if labels else ""
        p.add_text(text, position="upper_left", font_size=9, color="black")
        c = np.array(mesh.center)
        size = extent if same_scale else np.ptp(mesh.points, axis=0).max()
        v = np.array(view, float)
        v /= np.linalg.norm(v)
        p.camera.focal_point = c
        p.camera.position = c + v*size*2.6/zoom
        p.camera.up = (0, 0, 1)
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 0.75*size/zoom
    os.makedirs(OUT, exist_ok=True)
    p.screenshot(os.path.join(OUT, name))
    p.close()


def disc(p, z, radius, normal=(0, 0, 1)):
    import pyvista as pv
    p.add_mesh(pv.Disc(center=(0, 0, z), inner=0, outer=radius, normal=normal, c_res=48),
               color=GREY, opacity=0.25)


def nearest(values, targets):
    values = np.asarray(values, float)
    return [int(np.argmin(np.abs(values - t))) for t in targets]


# ---- the cases -------------------------------------------------------------------

def case1(d):
    rows = d["rows"]
    r = np.array([x["radius"] for x in rows])
    e = np.array([x["energy"] for x in rows])
    neg = np.array([x["negative"] for x in rows])
    rc = 1/np.pi
    fig, (a, b, c) = three_panels("1. A liquid column between plates, thinned past "
                                  "Rayleigh-Plateau (r < 1/pi)")
    rr = np.linspace(r.min(), r.max(), 200)
    a.plot(rr, 2*np.pi*rr, color=GREY, lw=2, label="cylinder, 2 pi r")
    a.plot(r, e, "o", color=BLUE, ms=4, label="relax()")
    a.axvspan(r.min(), rc, color=ORANGE, alpha=0.12, label="cylinder unstable")
    a.axvline(rc, color=ORANGE, lw=1)
    s = a.twinx()
    s.step(r, neg, where="mid", color=RED, lw=1)
    s.set_ylabel("negative eigenvalues", color=RED)
    s.set_ylim(-0.1, max(3, neg.max() + 1))
    a.set_xlabel("column radius r")
    a.set_ylabel("area")
    a.legend(fontsize=7, loc="upper left")
    a.set_title("result against the reference", fontsize=10)
    error_panel(b, r, np.abs(e/(2*np.pi*r) - 1), xlabel="column radius r")
    b.axvline(rc, color=ORANGE, lw=1)
    solver_panel(c, d, r, "column radius r")
    save(fig, "case01.png")
    picks = nearest(r, [0.5, 0.4, rc, 0.2])
    render(d, picks, "case01_shapes.png", context=lambda p, row: (disc(p, 0, 0.7), disc(p, 1, 0.7)),
           labels=[f"r = {r[i]:.3f}" for i in picks])


def catenoid_area(H):
    import case02_catenoid as c2
    return c2.catenoid(H)


def case2(d):
    import case02_catenoid as c2
    rows = d["rows"]
    H = np.array([x["H"] for x in rows])
    area = np.array([x["area"] for x in rows])
    fig, (a, b, c) = three_panels("2. A soap film between two rings, pulled apart to its fold")
    hh = np.linspace(H.min(), c2.H_FOLD*0.99999, 300)
    a.plot(hh, [catenoid_area(h) for h in hh], color=GREY, lw=2, label="stable catenoid")
    a.axvline(c2.H_FOLD, color=ORANGE, lw=1, label=f"fold, H = {c2.H_FOLD:.4f}")
    a.plot([c2.H_FOLD, H.max()], [2*np.pi, 2*np.pi], "--", color=GREY, lw=2, label="two discs")
    conv = np.array([x["converged"] for x in rows])
    a.plot(H[conv], area[conv], "o", color=BLUE, ms=4, label="relax(), converged")
    if (~conv).any():
        a.plot(H[~conv], area[~conv], "x", color=RED, ms=6, mew=2, label="relax(), not converged")
    a.set_xlabel("half separation H")
    a.set_ylabel("area")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    error_panel(b, H, np.array([x["error"] for x in rows]), limit=0.01,
                xlabel="half separation H", label="relative error (past the fold: two discs)")
    b.axvline(c2.H_FOLD, color=ORANGE, lw=1)
    solver_panel(c, d, H, "half separation H")
    save(fig, "case02.png")

    def rings(p, row):
        import pyvista as pv
        for z in (-row["H"], row["H"]):
            p.add_mesh(pv.Circle(radius=1.0, resolution=96).translate((0, 0, z)).extract_feature_edges(),
                       color="k", line_width=3)
    picks = nearest(H, [0.325, 0.5, 0.65, H.max()])
    render(d, picks, "case02_shapes.png", context=rings,
           labels=[f"H = {H[i]:.3f}" + (" (two discs)" if H[i] > c2.H_FOLD else "")
                   for i in picks])


def case3(d):
    from common import cap_reference
    rows = sorted(d["rows"], key=lambda x: x["theta"])
    th = np.array([x["theta"] for x in rows])
    volume = 2*np.pi/3
    fig, (a, b, c) = three_panels("3. A sessile drop, its contact angle stepped from 90 to 10 "
                                  "and to 170 degrees")
    tt = np.linspace(5, 175, 300)
    a.plot(tt, [cap_reference(volume, t)["radius"] for t in tt], color=GREY, lw=2,
           label="exact cap")
    # the contact radius relax() found, from the implied angle (the case stores that)
    a.plot(th, [cap_reference(volume, x["angle"])["radius"] for x in rows], "o", color=BLUE,
           ms=4, label="relax()")
    a.set_yscale("log")
    a.set_xlabel("contact angle (degrees)")
    a.set_ylabel("contact radius")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    b.plot(th, [x["angle_err"] for x in rows], "o-", color=BLUE, ms=4,
           label="implied angle error (degrees)")
    b.axhline(1.5, color=RED, ls="--", lw=1, label="pass limit 1.5 degrees")
    b.set_ylabel("degrees")
    b.set_xlabel("contact angle (degrees)")
    s = b.twinx()
    s.semilogy(th, [x["energy_err"] for x in rows], "s-", color=GREY, ms=3, lw=1,
               label="energy error (relative)")
    s.set_ylabel("relative energy error", color=GREY)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = s.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    b.set_title("error against the reference", fontsize=10)
    d2 = dict(d, rows=d["rows"])     # solver panel in run order (two sweeps)
    solver_panel(c, d2, [x["theta"] for x in d["rows"]], "contact angle (degrees), in run order")
    save(fig, "case03.png")
    run_th = [x["theta"] for x in d["rows"]]
    picks = nearest(run_th, [10, 50, 130, 170])
    render(d, picks, "case03_shapes.png", context=lambda p, row: disc(p, 0, 3.0),
           labels=[f"{run_th[i]:.0f} degrees" for i in picks], view=(0.0, -1.0, 0.25), zoom=1.5,
           same_scale=True)


def case5(d):
    rows = [x for x in d["rows"] if "error" in x]
    g = np.array([x["gap"] for x in rows])
    fig, (a, b, c) = three_panels("5. A liquid bridge between two spheres, the gap closed to 1e-3")
    a.semilogx(g, [x["energy_ref"] for x in rows], "-", color=GREY, lw=2, label="axisymmetric, energy")
    a.semilogx(g, [x["energy"] for x in rows], "o", color=BLUE, ms=4, label="relax(), energy")
    a.set_xlabel("gap")
    a.set_ylabel("energy (with the wetting constants)")
    s = a.twinx()
    s.semilogx(g, [x["neck_ref"] for x in rows], "--", color=GREY, lw=1.5, label="axisymmetric, neck")
    s.semilogx(g, [x["neck"] for x in rows], "s", color=ORANGE, ms=4, label="relax(), neck")
    s.set_ylabel("neck radius")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = s.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
    a.set_title("result against the reference", fontsize=10)
    error_panel(b, g, np.array([x["error"] for x in rows]), limit=0.01, xlabel="gap", logx=True,
                label="worst of energy, pressure, neck")
    solver_panel(c, dict(d, rows=rows), g, "gap")
    save(fig, "case05.png")

    def spheres(p, row):
        import pyvista as pv
        cz = 1 + row["gap"]/2
        for s_ in (-1, 1):
            p.add_mesh(pv.Sphere(radius=1.0, center=(0, 0, s_*cz), theta_resolution=96,
                                 phi_resolution=96), color=GREY, opacity=0.3)
    picks = [0, len(rows)//2, len(rows) - 1]
    d5 = dict(d, rows=rows)
    render(d5, picks, "case05_shapes.png", context=spheres,
           labels=[f"gap {rows[i]['gap']:g}" for i in picks], view=(0.0, -1.0, 0.2), zoom=1.0)


def case6(d):
    rows = d["rows"]
    th = np.array([x["theta"] for x in rows])
    fig, (a, b, c) = three_panels("6. A barrel drop on a fibre, contact angle 20 to 80 degrees")
    ok = np.array(["error" in x for x in rows])
    a.plot(th[ok], [x["equator_ref"] for x in rows if "error" in x], "-", color=GREY, lw=2,
           label="axisymmetric barrel")
    a.plot(th, [x["equator"] for x in rows], "o", color=BLUE, ms=4, label="relax()")
    uns = np.array([x["stable"] is False for x in rows])
    if uns.any():
        a.plot(th[uns], [x["equator"] for x, u in zip(rows, uns) if u], "v", color=ORANGE, ms=8,
               label="reported unstable")
    a.set_xlabel("contact angle (degrees)")
    a.set_ylabel("equator radius")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    error_panel(b, th[ok], np.array([x["error"] for x in rows if "error" in x]), limit=0.01,
                xlabel="contact angle (degrees)", label="worst of energy, pressure, equator")
    solver_panel(c, d, th, "contact angle (degrees)")
    save(fig, "case06.png")

    def fibre(p, row):
        import pyvista as pv
        p.add_mesh(pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.2, height=4.0,
                               resolution=48, capping=False), color=GREY, opacity=0.35)
    picks = nearest(th, [20, 50, 80])
    render(d, picks, "case06_shapes.png", context=fibre,
           labels=[f"{th[i]:.0f} degrees" for i in picks], view=(1.0, -1.0, 0.3))


def case8(d):
    rows, extra = d["rows"], d["extra"]
    V = np.array([x["volume"] for x in rows])
    order = np.argsort(V)
    fig, (a, b, c) = three_panels("8. The drainage band (ell = 1.1), remeshed each step as in "
                                  "its notebook")
    rv, rp = extra["reference"]
    a.plot(rv, rp, color=GREY, lw=2, label="finer run (level 3)")
    a.plot(V[order], np.array([x["pressure"] for x in rows])[order], ".", color=BLUE, ms=4,
           label="remesh + relax()")
    a.axvline(extra["snap"], color=RED, ls="--", lw=1, label=f"snap ({extra['snap']:.5f})")
    a.axvline(extra["snap_ref"], color=GREY, ls=":", lw=1, label=f"stored snap ({extra['snap_ref']:.5f})")
    a.set_xlabel("liquid volume per eighth cell")
    a.set_ylabel("pressure")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    err = np.array([x["error"] for x in rows])
    b.semilogy(V[order], np.maximum(err[order], 1e-16), ".", color=BLUE, ms=4)
    b.axhline(float(np.median(err)), color=ORANGE, lw=1.5,
              label=f"median {np.median(err):.3%} (limit 0.1%)")
    b.axhline(1e-3, color=RED, ls="--", lw=1)
    b.set_xscale("log")
    b.set_xlabel("liquid volume per eighth cell")
    b.set_ylabel("relative error, band pressure")
    b.legend(fontsize=7)
    b.set_title("error against the reference", fontsize=10)
    calls = [d["calls"][i] for i in extra["band_calls"]]
    pos = np.arange(len(calls))
    grad = np.array([k["gradient"] for k in calls])
    newton = np.array([k["newton"] for k in calls])
    c.bar(pos, grad, color=BLUE, alpha=0.75, width=1.0, label="gradient steps")
    c.bar(pos, newton, bottom=grad, color=ORANGE, width=1.0, label="Newton steps")
    top = max(1, (grad + newton).max())
    bad = [i for i, k in enumerate(calls) if not k["converged"]]
    c.plot(bad, [-0.05*top]*len(bad), "x", color=RED, ms=5, mew=1.5, clip_on=False,
           label=f"not converged ({len(bad)} of {len(calls)}, below the axis)")
    c.set_ylim(0, top*1.05)
    c.set_xlabel("band step")
    c.set_ylabel("steps")
    c.legend(fontsize=7, loc="upper left")
    c.set_title("what relax() did, per band step", fontsize=10)
    save(fig, "case08.png")
    idx = extra["band_calls"]
    picks = [idx[0], idx[len(idx)//2], idx[-1]]
    fake = dict(rows=[dict(call=i) for i in picks], calls=d["calls"])
    sys.path.insert(0, os.path.join(HERE, "..", "..", "docs"))
    import drainage_helpers as dh
    bead = dh.Cell(1.1).radius
    render(fake, [0, 1, 2], "case08_shapes.png",
           context=lambda p, row: sphere(p, (0, 0, 0), bead, opacity=0.5),
           labels=["first band step", "half way", "just before the snap"],
           view=(0.9, -0.6, 0.7), same_scale=True)


def case9(d):
    rows = d["rows"]
    V = np.array([x["volume"] for x in rows])
    err = np.array([x["error"] for x in rows])
    fig, (a, b, c) = three_panels("9. The cube sample inflated to 100 times its volume")
    vv = np.geomspace(V.min(), V.max(), 200)
    a.loglog(vv, (36*np.pi)**(1/3)*vv**(2/3), color=GREY, lw=2, label="sphere")
    a.loglog(V, [x["energy"] for x in rows], "o", color=BLUE, ms=4, label="relax()")
    a.set_xlabel("volume")
    a.set_ylabel("area")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    error_panel(b, V, err, limit=max(2*err[0], 1e-3), xlabel="volume", logx=True)
    solver_panel(c, d, V, "volume")
    save(fig, "case09.png")
    picks = nearest(V, [1, 10, 100])
    render(d, picks, "case09_shapes.png", labels=[f"V = {V[i]:.3g}" for i in picks],
           view=(1.0, -0.8, 0.5))


def case11(d):
    import axisym
    import matplotlib.pyplot as plt
    row = d["rows"][0]
    call = d["calls"][row["call"]]
    v = call["vertices"]
    fig, (a, b, c) = three_panels("11. A gravity puddle (volume 50, capillary length 1, "
                                  "60 degrees)")
    rp, zp = axisym.sessile_profile(50.0, 60.0, rho_g=1.0)
    r = np.hypot(v[:, 0] - v[:, 0].mean(), v[:, 1] - v[:, 1].mean())
    a.plot(r, v[:, 2], ".", color=BLUE, ms=2, label="relax(), vertices")
    a.plot(rp, zp, color=GREY, lw=1.5, label="axisymmetric profile")
    a.axhline(2*np.sin(np.radians(30)), color=ORANGE, ls=":", lw=1, label="large-puddle height")
    a.set_xlabel("distance from the axis")
    a.set_ylabel("height")
    a.legend(fontsize=7, loc="lower left")
    a.set_title("result against the reference", fontsize=10)
    keys = ["energy_err", "height_err", "radius_err", "pressure_err"]
    b.bar(range(4), [row[k] for k in keys], color=BLUE)
    b.set_xticks(range(4))
    b.set_xticklabels(["energy", "height", "contact radius", "pressure"], fontsize=8)
    b.set_yscale("log")
    b.axhline(5e-3, color=RED, ls="--", lw=1, label="pass limit 0.5%")
    b.legend(fontsize=7)
    b.set_ylabel("relative error")
    b.set_title("error against the reference", fontsize=10)
    calls = d["calls"]
    pos = np.arange(len(calls))
    c.bar(pos, [k["gradient"] for k in calls], color=BLUE, alpha=0.75, label="gradient steps")
    c.bar(pos, [k["newton"] for k in calls], bottom=[k["gradient"] for k in calls], color=ORANGE,
          label="Newton steps")
    c.set_xticks(pos)
    c.set_xticklabels(["start mesh", "refined once"][:len(calls)])
    c.set_ylabel("steps")
    t = c.twinx()
    t.plot(pos, [k["seconds"] for k in calls], "o-", color="k", ms=3, lw=1, label="time")
    t.set_ylabel("seconds")
    t.set_ylim(0, max(k["seconds"] for k in calls)*1.25)
    h1, l1 = c.get_legend_handles_labels()
    h2, l2 = t.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    c.set_title("what relax() did", fontsize=10)
    save(fig, "case11.png")
    render(d, [0], "case11_shapes.png", context=lambda p, row: disc(p, 0, 7.0),
           labels=["volume 50"], view=(0.0, -1.0, 0.6))


def case12(d):
    from common import cap_reference
    rows = d["rows"]
    V = np.array([x["volume"] for x in rows])
    err = np.array([x["error"] for x in rows])
    fig, (a, b, c) = three_panels("12. A sessile cap shrinking to 1e-4 of its volume")
    vv = np.geomspace(V.min(), V.max(), 200)
    a.loglog(vv, [cap_reference(x, 60.0)["radius"] for x in vv], color=GREY, lw=2,
             label="exact cap")
    a.loglog(V, [x["radius"] for x in rows], "o", color=BLUE, ms=4, label="relax()")
    a.set_xlabel("volume")
    a.set_ylabel("contact radius")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    error_panel(b, V, err, limit=max(2*err[0], 1e-3), xlabel="volume", logx=True,
                label="worst of energy, contact radius")
    solver_panel(c, d, V, "volume")
    save(fig, "case12.png")
    picks = nearest(np.log(V), np.log([V.max(), V.max()*1e-2, V.min()]))
    render(d, picks, "case12_shapes.png", labels=[f"V = {V[i]:.2g}" for i in picks],
           view=(0.0, -1.0, 0.5), same_scale=False)


def sphere(p, center, radius, opacity=0.3):
    import pyvista as pv
    p.add_mesh(pv.Sphere(radius=radius, center=center, theta_resolution=48, phi_resolution=48),
               color=GREY, opacity=opacity)


def ring(p, z, radius=1.0):
    import pyvista as pv
    t = np.linspace(0, 2*np.pi, 97)
    pts = np.c_[radius*np.cos(t), radius*np.sin(t), np.full_like(t, z)]
    p.add_mesh(pv.lines_from_points(pts), color="k", line_width=3)


def case13(d):
    import matplotlib.pyplot as plt
    rows = d["rows"]
    fig, (a, b, c) = three_panels("13. Schwarz's P and D surfaces, refined five times")
    for label, colour in (("P", BLUE), ("D", ORANGE)):
        rr = [x for x in rows if x["surface"] == label]
        lev = np.array([x["level"] for x in rr])
        area = np.array([x["area"] for x in rr])
        a.plot(lev, area/rr[0]["reference"], "o-", color=colour, ms=4,
               label=f"{label} (reference {rr[0]['reference']})")
        b.semilogy(lev, [x["error"] for x in rr], "o-", color=colour, ms=4, label=label)
    a.axhline(1.0, color=GREY, lw=1)
    a.set_xlabel("refinements")
    a.set_ylabel("area per cell / reference")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    lev = np.arange(1, 6)
    b.semilogy(lev, 2e-3*4.0**-(lev - 1), ":", color=GREY, label="second order (1/4 per level)")
    b.set_xlabel("refinements")
    b.set_ylabel("relative error")
    b.legend(fontsize=7)
    b.set_title("error against the reference", fontsize=10)
    solver_panel(c, d, [f"{x['surface']}{x['level']}" for x in rows], "surface, refinements")
    save(fig, "case13.png")
    last = [max(i for i, x in enumerate(rows) if x["surface"] == s) for s in ("P", "D")]
    render(d, last, "case13_shapes.png", labels=["P, fundamental region", "D, fundamental region"],
           view=(1.0, -0.8, 0.6), same_scale=False)


def case14(d):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "docs"))
    from stress_helpers import drop_on_sphere
    rows = sorted(d["rows"], key=lambda x: x["theta"])
    th = np.array([x["theta"] for x in rows])
    fig, (a, b, c) = three_panels("14. A drop of volume 0.5 on a sphere, contact angle "
                                  "20-150 degrees")
    tt = np.linspace(th.min(), th.max(), 200)
    a.plot(tt, [drop_on_sphere(0.5, x)["energy"] for x in tt], color=GREY, lw=2,
           label="exact cap")
    a.plot(th, [x["energy"] for x in rows], "o", color=BLUE, ms=4, label="relax()")
    a.set_xlabel("contact angle (degrees)")
    a.set_ylabel("energy (free area - cos theta wetted)")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    b.semilogy(th, [x["error"] for x in rows], "o-", color=BLUE, ms=4, label="energy")
    b.semilogy(th, [x["pressure_error"] for x in rows], "s-", color=ORANGE, ms=4,
               label="pressure")
    b.axhline(5e-3, color=RED, ls="--", lw=1, label="pass limit 0.5%")
    b.set_xlabel("contact angle (degrees)")
    b.set_ylabel("relative error")
    b.legend(fontsize=7)
    b.set_title("error against the reference", fontsize=10)
    solver_panel(c, d, [x["theta"] for x in d["rows"]], "contact angle, in the order run")
    save(fig, "case14.png")
    picks = [next(i for i, x in enumerate(d["rows"]) if x["theta"] == t) for t in (20, 90, 150)]
    render(d, picks, "case14_shapes.png", context=lambda p, row: sphere(p, (0, 0, 0), 1.0),
           labels=["20 degrees", "90 degrees", "150 degrees"], view=(0.0, -1.0, 0.3))


def case15(d):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "docs"))
    from stress_helpers import double_bubble
    rows = d["rows"]
    v2 = np.array([x["v2"] for x in rows])
    fig, (a, b, c) = three_panels("15. A double bubble, the second volume from 1 to 0.2")
    vv = np.linspace(v2.min(), v2.max(), 100)
    a.plot(vv, [double_bubble(1.0, x)["area"] for x in vv], color=GREY, lw=2,
           label="standard double bubble")
    a.plot(v2, [x["area"] for x in rows], "o", color=BLUE, ms=4, label="relax()")
    a.set_xlabel("second volume (first: 1)")
    a.set_ylabel("total area")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    b.semilogy(v2, [x["error"] for x in rows], "o-", color=BLUE, ms=4, label="area")
    b.semilogy(v2, [x["pressure_error"] for x in rows], "s-", color=ORANGE, ms=4,
               label="pressures (worst)")
    b.axhline(5e-3, color=RED, ls="--", lw=1, label="pass limit 0.5%")
    b.set_xlabel("second volume")
    b.set_ylabel("relative error")
    b.legend(fontsize=7)
    b.set_title("error against the reference", fontsize=10)
    solver_panel(c, d, v2, "second volume")
    save(fig, "case15.png")
    picks = nearest(v2, [1.0, 0.6, 0.2])
    render(d, picks, "case15_shapes.png", labels=[f"volumes 1 and {v2[i]:g}" for i in picks],
           view=(0.2, -1.0, 0.4))


def case16(d):
    rows = d["rows"]
    n = np.array([x["facets"] for x in rows])
    F = np.array([x["force"] for x in rows])
    ref = rows[0]["reference"]
    fig, (a, b, c) = three_panels("16. A solder joint's force on its pad: 3D (bga-3) against "
                                  "axisymmetric (bga-30)")
    a.semilogx(n, F, "o-", color=BLUE, ms=4, label="3D, derivative of the energy")
    a.axhline(ref, color=GREY, lw=2, label=f"axisymmetric, extrapolated ({ref:.4f} dyn)")
    a.set_xlabel("facets")
    a.set_ylabel("force on the upper pad (dyn)")
    a.legend(fontsize=7)
    a.set_title("result against the reference", fontsize=10)
    err = np.abs(F/ref - 1)
    b.loglog(n, err, "o-", color=BLUE, ms=4, label="3D")
    b.loglog(n, err[0]*(n/n[0])**-1.0, ":", color=GREY, label="second order in the edge length")
    b.set_xlabel("facets")
    b.set_ylabel("relative error")
    b.legend(fontsize=7)
    b.set_title("error against the reference", fontsize=10)
    solver_panel(c, d, n, "facets")
    save(fig, "case16.png")
    render(d, [0, len(rows) - 1], "case16_shapes.png",
           labels=[f"{n[0]} facets", f"{n[-1]} facets"], view=(0.0, -1.0, 0.25))


def case17(d):
    sys.path.insert(0, HERE)
    from case02_catenoid import H_FOLD, catenoid
    rows = d["rows"]
    h = np.array([x["height"] for x in rows])
    fig, (a, b, c) = three_panels("17. A catenoid between round wires, pulled to its fold")
    hh = np.linspace(h.min(), 2*H_FOLD*0.9999, 200)
    a.plot(hh, [catenoid(x/2) for x in hh], color=GREY, lw=2, label="exact catenoid")
    a.plot(h, [x["area"] for x in rows], "o", color=BLUE, ms=4, label="relax()")
    a.axvline(2*H_FOLD, color=RED, ls="--", lw=1, label="fold (no catenoid beyond)")
    a.set_xlabel("distance between the wires")
    a.set_ylabel("area")
    a.legend(fontsize=7)
    a.set_title("result against the reference (the run stops at the fold)", fontsize=10)
    error_panel(b, h, np.array([x["error"] for x in rows]), limit=1e-3,
                xlabel="distance between the wires")
    solver_panel(c, d, h, "distance between the wires")
    save(fig, "case17.png")
    picks = nearest(h, [h.min(), 1.0, h.max()])
    render(d, picks, "case17_shapes.png",
           context=lambda p, row: (ring(p, 0.0), ring(p, row["height"])),
           labels=[f"{h[i]:.2f} apart" for i in picks], view=(0.0, -1.0, 0.15))


PLOTS = {1: case1, 2: case2, 3: case3, 5: case5, 6: case6, 8: case8, 9: case9, 11: case11, 12: case12,
         13: case13, 14: case14, 15: case15, 16: case16, 17: case17}


def main(argv):
    import matplotlib
    matplotlib.use("Agg")
    wanted = [int(a) for a in argv if a.isdigit()] or list(CASES)
    for n in wanted:
        if "--plot-only" not in argv:
            run_case(n)
        PLOTS[n](load_case(n))
        print(f"case {n}: figures written")


if __name__ == "__main__":
    main(sys.argv[1:])
