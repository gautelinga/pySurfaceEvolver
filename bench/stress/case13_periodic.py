"""13. Triply periodic minimal surfaces: Schwarz's P and D, from Brakke's
online collection (pcell.fe: a fundamental tetrahedron, 1/48 of the cube;
dcell.fe: 1/24), refined five times, a default relax() after each
refinement. Reference: the surfaces' dimensionless areas A/V^(2/3) per
cubic unit cell, P 2.3451 and D 3.8377 (standard values), which are 12
times these fundamental regions' areas. Checked: the error falls at second
order (about fourfold per refinement), and its extrapolation (Richardson)
agrees with the reference to 1e-4; quotient spaces (a symmetry group, not a
periodic box) and Newton steps throughout."""

import numpy as np
import pysurfaceevolver as pyse

from common import Result, Tracker, rel

NAME = "13 Schwarz P and D surfaces"
SURFACES = {"pcell.fe": ("P", 2.3451), "dcell.fe": ("D", 3.8377)}
LEVELS = 5


def run(quick=False):
    track = Tracker()
    res = Result(NAME, False, "")
    worst, notes, ok = 0.0, [], True
    for name, (label, reference) in SURFACES.items():
        ev = pyse.Evolver(name)
        track.relax(ev)
        areas = [12*ev.total_energy]
        for level in range(1, LEVELS + 1):
            ev.refine()
            track.relax(ev)
            areas.append(12*ev.total_energy)
            res.rows.append(dict(surface=label, level=level, area=areas[-1], reference=reference,
                                 error=rel(areas[-1], reference), call=track.last))
        errors = [a - reference for a in areas]
        ratios = [errors[k - 1]/errors[k] for k in range(2, len(errors)) if errors[k] != 0]
        extrapolated = (4*areas[-1] - areas[-2])/3
        ex_err = rel(extrapolated, reference)
        worst = max(worst, ex_err)
        second_order = all(3.0 < q < 5.0 for q in ratios[-3:])
        ok &= second_order and ex_err < 1e-4
        notes.append(f"{label}: {areas[-1]:.6f} at level {LEVELS} ({rel(areas[-1], reference):.1e}), "
                     f"extrapolated {extrapolated:.6f} ({ex_err:.1e}), error ratios "
                     f"{', '.join(f'{q:.2f}' for q in ratios[-3:])}")
    track.finish(ev, res)
    res.error = worst
    res.passed = ok
    res.summary = "area per unit cell against the standard values; " + "; ".join(notes)
    return res
