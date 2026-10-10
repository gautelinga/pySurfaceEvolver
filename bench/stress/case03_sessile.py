"""3. A sessile drop (no gravity), its contact angle stepped from 90 down to 10
and up to 170 degrees at fixed volume. Reference: the exact spherical cap.

Checked: the energy (to 1%), the contact angle implied by the contact radius
at this volume (to 1.5 degrees), and that every relax() converged. (The
contact radius itself is no fair measure at extreme angles: near 170 degrees
a 1-degree error moves it by 12%.)"""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes

from common import Result, Tracker, cap, cap_reference, contact_radius, rel


def implied_angle(volume, radius):
    """The contact angle of the cap with this volume and contact radius."""
    lo, hi = 1e-3, 179.9                     # the contact radius falls with the angle
    for _ in range(80):
        mid = (lo + hi)/2
        lo, hi = (mid, hi) if cap_reference(volume, mid)["radius"] > radius else (lo, mid)
    return (lo + hi)/2

NAME = "3 sessile drop, 10-170 degrees"


def load(volume):
    v, f, rim = cap(1.0, 1.0)                    # a hemisphere: 90 degrees
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: C.plane((0, 0, 1), 0.0, contact_angle="theta")},
        vertex_constraints={1: rim}, parameters={"theta": 90.0},
        bodies=[pyse.Body(faces=range(len(f)), volume=volume)]))
    return ev


def run(quick=False):
    volume = 2*np.pi/3
    res = Result(NAME, False, "")
    worst = 0.0
    for angles in (np.arange(80, 5, -10), np.arange(100, 175, 10)):
        ev = load(volume)
        track = Tracker()
        track.relax(ev)
        for step in recipes.continuation(ev, recipes.parameter("theta"), angles.astype(float),
                                         relax=track.relax):
            ref = cap_reference(volume, step.value)
            e_err = rel(ev.total_energy, ref["energy"])
            angle = implied_angle(volume, contact_radius(ev, 1))
            worst = max(worst, e_err)
            res.rows.append(dict(theta=step.value, energy_err=e_err, angle=angle,
                                 angle_err=abs(angle - step.value),
                                 converged=bool(step.result.converged)))
        track.finish(ev, res, eigen=False)
    res.error = worst
    bad = [r["theta"] for r in res.rows
           if r["energy_err"] >= 0.01 or r["angle_err"] >= 1.5 or not r["converged"]]
    res.passed = not bad
    worst_angle = max(r["angle_err"] for r in res.rows)
    res.summary = (f"energy against exact caps: worst {worst:.2%}; implied contact angle off by "
                   f"{worst_angle:.2f} degrees at worst"
                   + (f"; failing at theta {bad}" if bad else ""))
    return res
