"""5. A liquid bridge between two unit spheres, contact angle 40 degrees,
volume 0.05, the gap closed from 0.2 to 1e-3. Reference: the axisymmetric
Young-Laplace solution (axisym.bridge_spheres). The gap is in the spheres'
centres, which the constraint builders take as numbers, so each gap is a new
surface, started from a cylinder of radius 0.35 between the spheres (relaxing
meets the volume) and relaxed (default relax, two refinements)."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes

from common import Result, Tracker, rel
import axisym

NAME = "5 bridge, gap down to 1e-3"
THETA, VOLUME = 40.0, 0.05


def neck_radius(ev):
    ring = ev.mesh().slice((0, 0, 1))[0][:-1]        # the mid-plane z = 0, a closed curve
    return float(np.hypot(ring[:, 0], ring[:, 1]).mean())


def bridge(gap, track, a=0.35):
    c = 1 + gap/2
    zb = c - np.sqrt(1 - a*a)
    tube = shapes.tube(a, -zb, zb, nz=4, nt=16)
    s1 = C.sphere((0, 0, -c), 1.0, contact_angle=THETA, wet_poles=("north",))
    s2 = C.sphere((0, 0, c), 1.0, contact_angle=THETA, wet_poles=("south",))
    ev = pyse.Evolver()
    ev.load_arrays(
        tube.vertices, tube.faces, constraints={1: s1, 2: s2},
        vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
        bodies=[pyse.Body(volume=VOLUME)])          # volconst from the spheres
    track.relax(ev)
    for _ in range(2):
        ev.refine()
        track.relax(ev)
    return ev


def run(quick=False):
    track = Tracker()
    res = Result(NAME, False, "")
    guess = None
    errs = []
    ev = None
    for gap in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        ref = axisym.bridge_spheres(VOLUME, gap, THETA, guess=guess)
        guess = (ref["neck"], ref["pressure"])
        try:
            ev = bridge(gap, track)
        except Exception as e:
            res.rows.append(dict(gap=gap, failed=f"{type(e).__name__}: {str(e)[:120]}"))
            res.notes.append(f"gap {gap}: {type(e).__name__}: {str(e)[:120]}")
            errs.append(np.inf)
            continue
        e = ev.energy                  # with the wetted caps' constant energy
        p = ev.body(1).pressure
        err = max(rel(e, ref["energy"]), abs(p - ref["pressure"])/max(1.0, abs(ref["pressure"])),
                  rel(neck_radius(ev), ref["neck"]))
        errs.append(err)
        res.rows.append(dict(gap=gap, energy=e, energy_ref=ref["energy"], pressure=p,
                             pressure_ref=ref["pressure"], neck=neck_radius(ev), neck_ref=ref["neck"],
                             error=err, negative=ev.eigen_counts().negative, call=track.last))
    if ev is not None:
        track.finish(ev, res)
    res.error = max(errs)
    limit = 0.01          # what this mesh reaches where it works (0.7-2% at gaps <= 0.02)
    res.passed = max(errs) <= limit
    bad = [r["gap"] for r, e in zip(res.rows, errs) if e > limit]
    res.summary = (f"energy, pressure (relative to max(1, |p|)) and neck against the axisymmetric "
                   f"solution: {errs[0]:.2%} at gap 0.2, worst {max(errs):.2%}, {errs[-1]:.2%} at "
                   f"gap 1e-3; over {limit:.0%} at gaps {bad}")
    return res
