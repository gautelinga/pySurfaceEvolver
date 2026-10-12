"""15. A double bubble from two unit boxes sharing a face, the second
bubble's volume shrunk from 1 to 0.2. Reference: exact, the standard double
bubble (Hutchings, Morgan, Ritore and Ros): three spherical caps meeting at
120 degrees (docs/stress_helpers.double_bubble). Checked: the total area and
both pressures to 0.5%, every step converged and stable; two bodies sharing
a wall, and a triple line."""

import os
import sys

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import recipes, shapes

from common import Result, Tracker, rel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs"))
from stress_helpers import double_bubble   # noqa: E402

NAME = "15 double bubble, volumes 1 to 0.2"


def two_boxes(n=2):
    """Unit boxes at x in [0, 1] and [1, 2] sharing the face x = 1: vertices,
    faces, and each body's face rows."""
    a = shapes.box((1, 1, 1), n=n)
    vb = a.vertices + [1.0, 0.0, 0.0]
    row = {tuple(np.round(p, 9)): i for i, p in enumerate(a.vertices)}
    shared = np.array([row.get(tuple(np.round(p, 9)), -1) for p in vb])
    new = len(a.vertices) + np.cumsum(shared < 0) - 1
    vertices = np.vstack([a.vertices, vb[shared < 0]])
    fb = np.where(shared >= 0, shared, new)[a.faces]
    wall = np.flatnonzero(np.isclose(a.vertices[a.faces].mean(axis=1)[:, 0], 1.0))
    fb = fb[~np.isclose(vertices[fb].mean(axis=1)[:, 0], 1.0)]
    faces = np.vstack([a.faces, fb])
    return vertices, faces, list(range(len(a.faces))), list(wall) + \
        list(range(len(a.faces), len(faces)))


def run(quick=False):
    vertices, faces, first, second = two_boxes()
    ev = pyse.Evolver()
    ev.load_arrays(vertices, faces, bodies=[pyse.Body(1.0, faces=first),
                                            pyse.Body(1.0, faces=second)])
    track = Tracker()
    ev.relax(levels=2)
    track.relax(ev)
    res = Result(NAME, False, "")
    worst = worst_p = 0.0
    unstable = 0
    for v2 in [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]:
        if v2 != 1.0:
            ev.body(2).target = v2
            r = track.relax(ev)
            unstable += r.stable is False
        ref = double_bubble(1.0, v2)
        p = ev.bodies().pressure
        e = rel(ev.total_area, ref["area"])
        ep = max(rel(p[0], ref["p1"]), rel(p[1], ref["p2"]))
        worst, worst_p = max(worst, e), max(worst_p, ep)
        res.rows.append(dict(v2=v2, area=ev.total_area, reference=ref["area"], error=e,
                             pressure_error=ep, call=track.last))
    track.finish(ev, res)
    res.error = max(worst, worst_p)
    res.passed = worst < 5e-3 and worst_p < 5e-3 and track.unconverged == 0 and not unstable
    res.summary = (f"area against the standard double bubble: worst {worst:.2%}; pressures "
                   f"{worst_p:.2%}; {track.unconverged} unconverged, {unstable} unstable")
    return res
