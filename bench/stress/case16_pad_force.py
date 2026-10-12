"""16. The force of a solder joint on its upper pad (Brakke's bga-3.fe: pads
of radius 0.1 cm 0.12 cm apart, gravity, surface tension 480), refined from
432 to 25k facets, from Evolver.derivative with the joint stretched along
(Brakke's zforce4). Reference: the same joint in Brakke's axisymmetric string
model (bga-30.fe, its density set to bga-3's), refined to 515 edges and
extrapolated. Checked: both converge at second order, and the 3D forces'
extrapolation agrees with the axisymmetric one to 1e-3."""

import numpy as np
import pysurfaceevolver as pyse

from common import Result, Tracker, rel

NAME = "16 solder joint force, 3D against axisymmetric"


def _force(ev, axis):
    h = ev.parameters["height"]
    def move(x):
        out = np.zeros_like(x)
        out[:, axis] = x[:, axis]/h
        return out
    return -ev.derivative("height", method="fixed", move=move)


def run(quick=False):
    track = Tracker()
    axi = pyse.Evolver("bga-30.fe")
    axi.parameters["s_density"] = 9.0            # bga-3's SOLDER_DENSITY
    axi.relax(levels=2)
    axial = []
    for level in range(2, 10):
        if level > 2:
            axi.relax(levels=1)
        axial.append(_force(axi, 1))
    reference = (4*axial[-1] - axial[-2])/3
    ev = pyse.Evolver("bga-3.fe")
    ev.relax(levels=2)
    res = Result(NAME, False, "")
    forces = []
    for level in range(2, 6):
        if level > 2:
            track.add(ev.relax(levels=1))
        forces.append(_force(ev, 2))
        res.rows.append(dict(facets=ev.counts["facets"], force=forces[-1], reference=reference,
                             error=rel(forces[-1], reference), call=track.last))
    track.finish(ev, res)
    ratios = [(forces[k - 1] - reference)/(forces[k] - reference) for k in (2, 3)]
    axi_ratios = [(axial[k - 1] - axial[k - 2])/(axial[k] - axial[k - 1])
                  for k in (len(axial) - 2, len(axial) - 1)]
    extrapolated = (4*forces[-1] - forces[-2])/3
    res.error = rel(extrapolated, reference)
    res.passed = (res.error < 1e-3 and all(3.0 < q < 5.0 for q in ratios)
                  and all(3.5 < q < 4.5 for q in axi_ratios))
    res.summary = (f"axisymmetric {reference:.5f} dyn (ratios {axi_ratios[0]:.2f}, "
                   f"{axi_ratios[1]:.2f}); 3D {forces[-1]:.5f} at {ev.counts['facets']} facets "
                   f"({rel(forces[-1], reference):.2%}), error ratios {ratios[0]:.2f}, "
                   f"{ratios[1]:.2f}, extrapolated {extrapolated:.5f} ({res.error:.1e})")
    return res
