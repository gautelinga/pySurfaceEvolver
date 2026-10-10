"""9. The cube sample inflated to 100 times its volume. Reference: a sphere,
E = (36 pi)^(1/3) V^(2/3); the relative error must not grow with the volume
(the surface stretches tenfold in length)."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import recipes

from common import Result, Tracker, rel

NAME = "9 cube inflated 100x"


def sphere(volume):
    return (36*np.pi)**(1/3)*volume**(2/3)


def run(quick=False):
    ev = pyse.Evolver("cube.fe")
    track = Tracker()
    track.relax(ev)
    for _ in range(2):
        ev.refine()
        track.relax(ev)
    res = Result(NAME, False, "")
    errors = []
    for step in recipes.continuation(ev, recipes.body_target(1), np.geomspace(1, 100, 21),
                                     relax=track.relax):
        err = rel(ev.total_energy, sphere(step.value))
        errors.append(err)
        res.rows.append(dict(volume=step.value, error=err))
    track.finish(ev, res)
    res.error = max(errors)
    limit = max(2*errors[0], 1e-3)
    res.passed = max(errors) <= limit
    res.summary = (f"energy against a sphere: {errors[0]:.3%} at V = 1, worst {max(errors):.3%} "
                   f"(limit {limit:.3%})")
    return res
