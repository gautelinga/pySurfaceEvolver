"""11. A gravity puddle: volume 50 on z = 0 with gravity (capillary length 1),
contact angle 60 degrees, started as a spherical cap. Reference: the
axisymmetric sessile drop with gravity (axisym.sessile); sanity check: the
large-puddle height 2 sin(theta/2) l_c."""

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C

from common import Result, Tracker, cap, cap_reference, contact_radius, rel
import axisym

NAME = "11 gravity puddle, 60 degrees"
THETA, VOLUME = 60.0, 50.0


def run(quick=False):
    c0 = cap_reference(VOLUME, THETA)
    v, f, rim = cap(c0["R"], c0["height"], nr=10, nt=48)
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=THETA)},
        vertex_constraints={1: rim}, header="gravity_constant 1\n",
        bodies=[pyse.Body(faces=range(len(f)), volume=VOLUME, density=1.0)]))
    track = Tracker()
    track.relax(ev)
    ev.refine()
    track.relax(ev)
    ref = axisym.sessile(VOLUME, THETA, rho_g=1.0)
    height = float(ev.mesh().vertices[:, 2].max())
    radius = contact_radius(ev, 1)
    res = Result(NAME, False, "")
    errs = dict(energy_err=rel(ev.total_energy, ref["energy"]), height_err=rel(height, ref["height"]),
                radius_err=rel(radius, ref["radius"]),
                pressure_err=abs(ev.body(1).pressure - ref["pressure"])/max(1.0, abs(ref["pressure"])))
    res.rows.append(dict(energy=ev.total_energy, energy_ref=ref["energy"], height=height,
                         height_ref=ref["height"], radius=radius, radius_ref=ref["radius"],
                         pressure=ev.body(1).pressure, pressure_ref=ref["pressure"], **errs))
    track.finish(ev, res)
    res.error = max(errs.values())
    res.passed = res.error < 5e-3
    res.summary = (f"against the axisymmetric drop: " + ", ".join(f"{k} {v:.2%}" for k, v in errs.items())
                   + f"; height {height:.4f} (reference {ref['height']:.4f}, "
                   f"large-puddle limit {2*np.sin(np.radians(THETA/2)):.4f})")
    return res
