"""12. A sessile cap at 60 degrees shrinking to 1e-4 of its volume (a bubble
or drop evaporating). Reference: the exact cap at every volume; the relative
error must not grow as the drop shrinks (the mesh shrinks with it)."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes

from common import Result, Tracker, cap, cap_reference, contact_radius, rel

NAME = "12 cap shrinking to 1e-4 volume"


def run(quick=False):
    theta = 60.0
    v, f, rim = cap(1.0, 1 - np.cos(np.radians(theta)))
    v0 = cap_reference(1.0, theta)
    volume = np.pi*(1 - np.cos(np.radians(theta)))**2*(3 - (1 - np.cos(np.radians(theta))))/3
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=theta)},
        vertex_constraints={1: rim}, bodies=[pyse.Body(faces=range(len(f)), volume=volume)]))
    track = Tracker()
    track.relax(ev)
    res = Result(NAME, False, "")
    errors = []
    for step in recipes.continuation(ev, recipes.body_target(1),
                                     volume*np.geomspace(1, 1e-4, 25), relax=track.relax):
        ref = cap_reference(step.value, theta)
        err = max(rel(ev.total_energy, ref["energy"]), rel(contact_radius(ev, 1), ref["radius"]))
        errors.append(err)
        res.rows.append(dict(volume=step.value, error=err))
    track.finish(ev, res)
    res.error = max(errors)
    limit = max(2*errors[0], 1e-3)
    res.passed = max(errors) <= limit
    res.summary = (f"error against exact caps: {errors[0]:.2%} at the start, worst {max(errors):.2%} "
                   f"(limit {limit:.2%}), {errors[-1]:.2%} at 1e-4 of the volume")
    return res
