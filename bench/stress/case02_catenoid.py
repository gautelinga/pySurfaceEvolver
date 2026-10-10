"""2. A soap film between two rings of radius 1 at z = +-H, H raised past the
fold (H = 0.6627: no catenoid beyond). Reference: the exact catenoid's area on
the stable branch; past the fold the solver must not report a converged film."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import recipes

from common import Result, Tracker, tube, rel

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
    v, f, bottom, top = tube(1.0, -H0, H0, nz=8, nt=32)
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: "x^2 + y^2 = 1", 2: "z = hh", 3: "z = -hh"},
        vertex_constraints={1: bottom + top, 2: top, 3: bottom}, parameters={"hh": H0}))
    track = Tracker()
    track.relax(ev)
    ev.refine()
    track.relax(ev)
    res = Result(NAME, False, "")
    errors, past = [], []
    for step in recipes.continuation(ev, recipes.parameter("hh"),
                                     np.arange(0.325, 0.76, 0.025), relax=track.relax):
        ref = catenoid(step.value)
        m = ev.mesh()
        neck = float(np.hypot(m.vertices[:, 0], m.vertices[:, 1]).min())
        row = dict(H=step.value, area=ev.total_energy, reference=ref, neck=neck,
                   converged=step.result.converged)
        if ref is None:
            past.append(row)
        else:
            row["error"] = rel(ev.total_energy, ref)
            errors.append(row["error"])
        res.rows.append(row)
    track.finish(ev, res, eigen=False)
    res.error = max(errors)
    false_ok = [r["H"] for r in past if r["converged"]]
    res.passed = max(errors) < 0.01 and not false_ok
    res.summary = (f"area against the catenoid up to the fold: worst {max(errors):.2%}; "
                   f"past the fold (H > {H_FOLD:.4f}) relax() said converged at H = "
                   f"{[round(h, 3) for h in false_ok]} (necks {[round(r['neck'], 3) for r in past]})")
    return res
