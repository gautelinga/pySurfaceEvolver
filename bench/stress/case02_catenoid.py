"""2. A soap film between two rings of radius 1 at z = +-H, H raised past the
fold (H = 0.6627: no catenoid beyond). Reference: the exact catenoid's area on
the stable branch; past the fold the film must pinch off into two flat discs
(area 2 pi, less for the polygons the rings' vertices make), not stay a film
reported converged."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import recipes, shapes

from common import Result, Tracker, rel

NAME = "2 catenoid to its fold"
X_FOLD = 1.19967864     # x tanh x = 1
H_FOLD = X_FOLD/np.cosh(X_FOLD)


def catenoid(H):
    """Area of the stable catenoid through the rings, or None past the fold."""
    if H >= H_FOLD:
        return None
    lo, hi = 1/np.cosh(X_FOLD), 1.0          # the larger root of a cosh(H/a) = 1
    for _ in range(100):
        a = (lo + hi)/2
        lo, hi = (lo, a) if a*np.cosh(H/a) > 1 else (a, hi)
    return np.pi*a*(2*H + a*np.sinh(2*H/a))


def run(quick=False):
    H0 = 0.3
    film = shapes.tube(1.0, -H0, H0, nz=8, nt=32)
    bottom, top = film.rims["bottom"], film.rims["top"]
    ev = pyse.Evolver()
    ev.load_arrays(
        film.vertices, film.faces, constraints={1: "x^2 + y^2 = 1", 2: "z = hh", 3: "z = -hh"},
        vertex_constraints={1: bottom + top, 2: top, 3: bottom}, parameters={"hh": H0})
    track = Tracker()
    track.relax(ev)
    ev.refine()
    track.relax(ev)
    res = Result(NAME, False, "")
    errors, past = [], []
    for step in recipes.continuation(ev, "hh",
                                     np.arange(0.325, 0.76, 0.025), relax=track.relax):
        ref = catenoid(step.value)
        m = ev.mesh()
        neck = float(np.hypot(m.vertices[:, 0], m.vertices[:, 1]).min())
        row = dict(H=step.value, area=ev.total_energy, reference=ref, neck=neck,
                   converged=step.result.converged, call=track.last)
        if ref is None:
            past.append(row)
        else:
            row["error"] = rel(ev.total_energy, ref)
            errors.append(row["error"])
        res.rows.append(row)
    track.finish(ev, res, eigen=False)
    res.error = max(errors)
    for r in past:
        r["error"] = rel(r["area"], 2*np.pi)
    discs = [r["H"] for r in past if r["converged"] and r["error"] < 0.01]
    res.passed = max(errors) < 0.01 and len(discs) == len(past)
    res.summary = (f"area against the catenoid up to the fold: worst {max(errors):.2%}; "
                   f"past the fold (H > {H_FOLD:.4f}) two discs at H = "
                   f"{[round(h, 3) for h in discs]} of {len(past)} steps, areas "
                   f"{[round(r['area'], 4) for r in past]} (2 pi = {2*np.pi:.4f})")
    return res
