"""17. A soap film between two round wires of radius 1 (Evolver boundaries:
refining keeps them round), pulled apart from 0.6 past the catenoid's fold
(wires 1.3255 apart) with recipes.continuation(stop_on="unconverged").
Reference: the exact catenoid's area. Checked: the area within 0.1% on the
stable branch, and the run stopping at the first step past the fold, with
the surface back at the last catenoid."""

import os
import sys
import warnings

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

from common import Result, Tracker, rel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case02_catenoid import H_FOLD, catenoid   # noqa: E402

NAME = "17 catenoid between round wires, to the fold"
STEP = 0.05


def run(quick=False):
    tube = shapes.tube(1.0, 0.0, 0.6, nz=4, nt=16)
    top = C.curve("cos(p1)", "sin(p1)", "ht", lambda x: np.arctan2(x[:, 1], x[:, 0]))
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces, boundaries={1: C.circle(), 2: top},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   parameters={"ht": 0.6})
    track = Tracker()
    ev.relax(levels=2)
    res = Result(NAME, False, "")
    worst, last = 0.0, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for step in recipes.continuation(ev, "ht", np.arange(0.65, 1.6, STEP),
                                         relax=track.relax, stop_on="unconverged"):
            ref = catenoid(step.value/2)
            e = rel(ev.total_area, ref) if ref else float("inf")
            worst, last = max(worst, e), step.value
            res.rows.append(dict(height=step.value, area=ev.total_area, reference=ref, error=e,
                                 call=track.last))
    stopped = any("stopped before" in str(w.message) for w in caught)
    fold = 2*H_FOLD
    track.finish(ev, res, eigen=False)
    res.error = worst
    back = last is not None and ev.parameters["ht"] == last
    res.passed = (worst < 1e-3 and stopped and back
                  and last is not None and fold - STEP < last <= fold)
    res.summary = (f"area against the exact catenoid: worst {worst:.2%}; stopped "
                   f"{'after' if stopped else 'NOT after'} {last} (fold at {fold:.4f}), "
                   f"{'back at the last catenoid' if back else 'surface not back'}")
    return res
