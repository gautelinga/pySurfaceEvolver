"""1. A liquid cylinder between plates z = 0 and z = 1, contact angle 90
degrees (contact lines free to slide), its volume lowered. The cylinder is an
equilibrium at every radius but stops being stable when the height exceeds
pi r (r < 1/pi): a robust solver must say so, not converge to it silently."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

from common import Result, Tracker, rel

NAME = "1 cylinder past Rayleigh-Plateau"


def run(quick=False):
    r0 = 0.5
    column = shapes.tube(r0, 0.0, 1.0, nz=12, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(
        column.vertices, column.faces,
        constraints={1: C.plane((0, 0, 1), 0.0), 2: C.plane((0, 0, -1), point=(0, 0, 1.0))},
        vertex_constraints={1: column.rims["bottom"], 2: column.rims["top"]},
        bodies=[pyse.Body(volume=np.pi*r0*r0)])
    track = Tracker()
    track.relax(ev)
    res = Result(NAME, False, "")
    rc = 1/np.pi
    first_unstable = None
    flagged = None
    for step in recipes.continuation(ev, ev.body(1),
                                     np.pi*np.linspace(r0, 0.2, 25)**2, relax=track.relax):
        r = np.sqrt(step.value/np.pi)
        neg = ev.eigen_counts().negative
        said = getattr(step.result, "stable", None)          # what the solver reports, if anything
        if neg and first_unstable is None:
            first_unstable = r
        if said is False and flagged is None:
            flagged = r
        res.rows.append(dict(radius=r, energy=ev.total_energy, cylinder=2*np.pi*r,
                             negative=neg, reported_stable=said, call=track.last))
    track.finish(ev, res)
    res.error = rel(first_unstable, rc) if first_unstable else float("nan")
    res.passed = flagged is not None and rel(flagged, rc) < 0.05
    res.summary = (f"instability at r = 1/pi = {rc:.4f}: eigenvalues show it at "
                   f"{first_unstable if first_unstable is None else round(first_unstable, 4)}; "
                   f"the solver reported it at {flagged}")
    return res
