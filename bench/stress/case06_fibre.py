"""6. A barrel drop (volume 2) on a fibre of radius 0.2, the contact angle
stepped from 20 to 80 degrees. Reference: the axisymmetric barrel
(axisym.barrel). Each step records the Hessian's negative eigenvalues: the
barrel rolls up into a clam-shell (symmetry breaking) at higher contact angles
for small drops (McHale and Newton 2002, Carroll 1986); there is no single
threshold value for this volume to check against, so the case only reports
where eigenvalues turn negative. Translation along the fibre is a zero mode."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes

from common import Result, Tracker, revolve, rel
import axisym

NAME = "6 barrel on a fibre, 20-80 degrees"
B, VOLUME = 0.2, 2.0


def run(quick=False):
    # a crude start: an ellipse-like meridian from fibre to fibre
    L, rm = 2.4, 0.75
    z = np.linspace(-L/2, L/2, 13)
    r = np.sqrt(B*B + (rm*rm - B*B)*(1 - (2*z/L)**2))
    v, f, bottom, top = revolve(r, z, nt=24)
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: C.cylinder((0, 0, 0), (0, 0, 1), B, contact_angle="theta")},
        vertex_constraints={1: bottom + top}, parameters={"theta": 20.0},
        bodies=[pyse.Body(faces=range(len(f)), volume=VOLUME)]))
    track = Tracker()
    track.relax(ev)
    ev.refine()
    track.relax(ev)
    res = Result(NAME, False, "")
    errs, guess = [], (0.78, 2.4)
    for step in recipes.continuation(ev, recipes.parameter("theta"), np.arange(20.0, 81, 10),
                                     relax=track.relax):
        try:
            ref = axisym.barrel(VOLUME, B, step.value, guess=guess)
            guess = (ref["equator"], ref["pressure"])
        except Exception as e:
            ref = None
        m = ev.mesh()
        eq = float(np.hypot(m.vertices[:, 0], m.vertices[:, 1]).max())
        counts = ev.eigen_counts()
        row = dict(theta=step.value, energy=ev.total_energy, pressure=ev.body(1).pressure,
                   equator=eq, negative=counts.negative, zero=counts.zero,
                   stable=step.result.stable, unstable_modes=step.result.negative_modes)
        if ref is not None:
            err = max(rel(ev.total_energy, ref["energy"]), rel(ev.body(1).pressure, ref["pressure"]),
                      rel(eq, ref["equator"]))
            row.update(energy_ref=ref["energy"], pressure_ref=ref["pressure"],
                       equator_ref=ref["equator"], error=err)
            errs.append(err)
        res.rows.append(row)
    track.finish(ev, res)
    res.error = max(errs) if errs else float("nan")
    res.passed = bool(errs) and max(errs) < 0.01
    unstable = [r["theta"] for r in res.rows if r["stable"] is False]
    res.summary = (f"energy, pressure and equator radius against the barrel: worst "
                   f"{res.error:.2%} over {len(errs)} angles; relax() reports the barrel unstable "
                   f"(roll-up toward a clam shell) at {unstable}")
    return res
