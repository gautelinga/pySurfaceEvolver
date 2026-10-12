"""14. A drop on a solid sphere of radius 1 (no gravity), volume 0.5, its
contact angle swept from 90 down to 20 degrees and up to 150 as a parameter.
Reference: exact, the free surface is a spherical cap meeting the sphere at
the contact angle (docs/stress_helpers.drop_on_sphere): its energy (free area
minus cos(theta) times the wetted area) and pressure. Checked: Evolver.energy
(the wetted cap holds the sphere's pole, so its volume and the wetting energy
the line integrals leave out come from the constraint, with the angle a
parameter) and the pressure to 0.5%, every step converged; the contact angles
measured on the mesh within 3 degrees (first order in the edge length).
Each sweep starts from the 90 degree drop.

Fails on defaults (October 2026): as the contact line creeps over the sphere,
the gradient rounds never settle (1e-5 per round) and use up max_iter; the
first Newton step then lowers the energy but cuts the residual by less than
10%, so Newton stops and the step ends unconverged, although plain Newton
steps from there converge quadratically. Counting such an energy drop as
progress broke case 2 past its fold; giving the conjugate-gradient fallback
its own budget broke max_iter. Energies stay within 0.3% regardless."""

import os
import sys

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

from common import Result, Tracker, rel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs"))
from stress_helpers import drop_on_sphere   # noqa: E402

NAME = "14 drop on a sphere, 20-150 degrees"
VOLUME = 0.5


def start(volume=VOLUME, R=1.0):
    """The exact drop at 90 degrees, as a surface of revolution."""
    ref = drop_on_sphere(volume, 90.0, R)
    rho, a = ref["rho"], ref["radius"]
    d = np.sqrt(R*R + rho*rho)
    t = np.linspace(np.arctan2(a, np.sqrt(R*R - a*a) - d), 0.0, 9)
    return shapes.revolve(rho*np.sin(t), d + rho*np.cos(t), nt=24)


def drop_at_90():
    drop = start()
    ev = pyse.Evolver()
    ev.load_arrays(drop.vertices, drop.faces,
                   constraints={1: C.sphere((0, 0, 0), 1.0, contact_angle="theta",
                                            wet_poles=("north",))},
                   vertex_constraints={1: drop.rims["bottom"]}, parameters={"theta": 90.0},
                   bodies=[pyse.Body(volume=VOLUME)])
    ev.relax(levels=2)
    return ev


def run(quick=False):
    track = Tracker()
    res = Result(NAME, False, "")
    worst = worst_p = worst_angle = 0.0
    for sweep in ([90, 75, 60, 45, 30, 20], [105, 120, 135, 150]):
        ev = drop_at_90()                        # each way from the exact 90 degree drop
        for step in recipes.continuation(ev, "theta", sweep, relax=track.relax):
            ref = drop_on_sphere(VOLUME, step.value)
            e, p = rel(step.energy, ref["energy"]), rel(ev.body(1).pressure, ref["pressure"])
            angle = float(np.abs(ev.contact_angles(1) - step.value).max())
            worst, worst_p, worst_angle = max(worst, e), max(worst_p, p), max(worst_angle, angle)
            res.rows.append(dict(theta=step.value, energy=step.energy, reference=ref["energy"],
                                 error=e, pressure_error=p, angle_error=angle,
                                 converged=step.result.converged, call=track.last))
    track.finish(ev, res)
    res.error = max(worst, worst_p)
    res.passed = (worst < 5e-3 and worst_p < 5e-3 and worst_angle < 3.0
                  and track.unconverged == 0)
    res.summary = (f"energy against the exact cap: worst {worst:.2%}; pressure {worst_p:.2%}; "
                   f"contact angle on the mesh within {worst_angle:.2f} degrees; "
                   f"{track.unconverged} unconverged steps")
    return res
