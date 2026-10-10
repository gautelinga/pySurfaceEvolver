"""3. A sessile drop (no gravity), its contact angle stepped from 90 down to 10
and up to 170 degrees at fixed volume. Reference: the exact spherical cap."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes

from common import Result, Tracker, cap, cap_reference, contact_radius, rel

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
            a_err = rel(contact_radius(ev, 1), ref["radius"])
            worst = max(worst, e_err, a_err)
            res.rows.append(dict(theta=step.value, energy=ev.total_energy, energy_ref=ref["energy"],
                                 energy_err=e_err, radius_err=a_err))
        track.finish(ev, res, eigen=False)
    res.error = worst
    res.passed = worst < 0.02
    bad = [r["theta"] for r in res.rows if max(r["energy_err"], r["radius_err"]) >= 0.02]
    res.summary = (f"energy and contact radius against exact caps: worst {worst:.2%}"
                   + (f"; over 2% at theta {bad}" if bad else ""))
    return res
