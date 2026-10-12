"""Generate docs/solder_joints.ipynb: Brakke's ball grid array series with pySE.

    python tools/make_solder_notebook.py [output directory, default docs/]

Written without outputs (the documentation build executes it), cell ids
numbered so a regeneration changes only what was edited. Needs nbformat.
"""
import os
import sys

import nbformat as nbf

DOCS = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = [
    md("""
# Solder joints: force, standoff and alignment

A ball grid array (BGA) joins a chip to a board with an array of solder joints: liquid
solder wetting a round pad on each side. While the solder is molten, surface tension
and gravity alone hold the chip, so the questions are mechanical: what force does a
joint exert on its pad, how high does a chip of a given weight float, and does a
misplaced chip pull itself back into line? Ken Brakke's
[BGA tutorial](https://kenbrakke.com/evolver/bga/bga.htm) answers them with a series
of datafiles, `bga-1.fe` to `bga-10.fe` and `bga-30.fe`, shipped with pySE
(`pyse.examples`) together with his command files (`zforce.cmd`, ...). Units are
cgs: lengths in cm, the solder's surface tension 480 erg/cm², forces in dynes.

This notebook follows the series with `relax()`, `derivative()` (forces),
optimizing parameters (`ev.parameters.optimize`) and `recipes.continuation`."""),
    code("""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pysurfaceevolver as pyse
from pysurfaceevolver import recipes

pv.set_jupyter_backend("static")"""),
    md("""
## The joint

`bga-3.fe`: two coaxial pads of radius 0.1 cm, 0.12 cm apart, the solder's volume 1.3
times the cylinder between them, with gravity. The pads are constraints (`z = 0` and
`z = height`, `height` a parameter) and their rims are fixed vertices. Brakke's
datafiles start from very few facets; `relax(levels=3)` refines three times."""),
    code("""
ev = pyse.Evolver("bga-3.fe")
result = ev.relax(levels=3)
print(result)
print(f"{ev.counts['facets']} facets, energy {ev.total_energy:.6f} erg, "
      f"pressure {ev.body(1).pressure:.1f} dyn/cm^2")
ev.plot(off_screen=True, color="silver")"""),
    md("""
## The force on the upper pad

The vertical force the joint exerts on the upper pad is minus the derivative of the
equilibrium energy with respect to the pad's height. Brakke's tutorial (bga-4) compares
seven ways to get it; the reliable ones move the whole joint with the pad (each vertex
by `z/height` times the change) and correct for the volume change by the pressure, or
re-converge after the move. `ev.derivative` does both:

* `method="fixed"` with `move=`: Brakke's preferred `zforce4`, no relaxing;
* the default `method="newton"`: Newton steps re-converge the moved surface (`zforce5`).

`bga-4.fe` is the same joint with Brakke's variational integral for the force (the
quantity `zforce7`), and his command file computes `zforce4`. All three agree:"""),
    code("""
def linear(ev):
    h = ev.parameters["height"]
    return lambda x: np.c_[0*x[:, :2], x[:, 2]/h]          # stretch with the pad

ev = pyse.Evolver("bga-4.fe")
ev.command('read "zforce.cmd"')
print(f"{'facets':>7} {'fixed':>12} {'newton':>12} {'variational':>12} {'zforce4':>12}")
ev.relax(levels=2)
for level in range(2, 6):
    if level > 2:
        ev.relax(levels=1)
    fixed = -ev.derivative("height", method="fixed", move=linear(ev))
    newton = -ev.derivative("height", move=linear(ev))
    variational = ev.quantity("zforce7").value
    zforce4 = float(ev.command("zforce4").split()[1])
    print(f"{ev.counts['facets']:7d} {fixed:12.6f} {newton:12.6f} {variational:12.6f} "
          f"{zforce4:12.6f}")"""),
    md("""
The methods agree to the printed digits at each refinement, but the force itself moves
from about 30 to 17 dynes as the mesh is refined: on Brakke's starting meshes it is far
from converged. Without `move`, `derivative()` would move only the pad's own row of
vertices (its rim is fixed, so the rest of the mesh doesn't follow), which differs at
first order in the edge length; give `move` whenever vertices are fixed to a moving
wall.

The axisymmetric version of the same joint, `bga-30.fe` (a string model of the
meridian, Brakke's named quantities for area, volume and gravity), converges much faster
and gives the reference the 3D joint approaches. Its pads are the lines `y = 0` and
`y = height`, so the linear move is in `y`. Its start has every vertex on a pad (nothing
free to move): `relax(levels=...)` refines before relaxing."""),
    code("""
axi = pyse.Evolver("bga-30.fe")
axi.relax(levels=2)
for level in range(2, 8):
    if level > 2:
        axi.relax(levels=1)
    h = axi.parameters["height"]
    force = -axi.derivative("height", method="fixed", move=lambda x: np.c_[0*x[:, :1], x[:, 1]/h])
    print(f"{axi.counts['edges']:4d} edges  force {force:.6f} dyn")"""),
    md("""
## Force against height

Squeezing the joint (a lower `height`) pushes the pads apart harder; pulling it up
turns the force into a pull. `recipes.continuation` steps the height, relaxing after
each step; the force comes from `derivative` and the stability from the relax result.
Below a height of about 0.09 cm the joint becomes unstable: it would rather bulge to one
side (compare Brakke's `cbga2.fe` in [Brakke's samples: capillarity](brakke_capillarity.ipynb))."""),
    code("""
def force(ev):
    return -ev.derivative("height", method="fixed", move=linear(ev))

rows = []
with warnings.catch_warnings():     # unstable joints are expected here: plotted as such
    warnings.simplefilter("ignore", pyse.UnstableEquilibriumWarning)
    for direction in (np.linspace(0.12, 0.05, 15), np.linspace(0.12, 0.20, 9)[1:]):
        ev = pyse.Evolver("bga-3.fe")
        ev.relax(levels=3)
        for step in recipes.continuation(ev, "height", direction):
            rows.append((step.value, force(ev), step.result.stable))
rows.sort()
h, F, stable = (np.array(c) for c in zip(*rows))
plt.plot(h[stable], F[stable], "o-", label="stable")
plt.plot(h[~stable], F[~stable], "x--", label="unstable")
plt.axhline(200, color="gray", lw=0.8)
plt.axhline(500, color="gray", lw=0.8, ls=":")
plt.xlabel("pad height (cm)")
plt.ylabel("force on the upper pad (dyn)")
plt.legend()
plt.show()
load_height = np.interp(200, F[::-1], h[::-1])
print(f"a load of 200 dyn is carried at height {load_height:.4f} cm")"""),
    md("""
## How high a loaded chip floats

`bga-5.fe` puts a load of 200 dynes on the upper pad (an energy `pad_weight*z` on a
vertex riding on it) and makes `height` an *optimizing parameter*: the minimization
moves it along with the vertices, so `relax()` finds the height where the joint's push
balances the load. The residual counts the force on each optimizing parameter, so
`relax()` doesn't stop until it is balanced; at the end the derivative with the load
included is close to zero (it vanishes as the mesh is refined, the move above and
Evolver's own finite difference being different discretizations).

Evolver differentiates the energy with respect to an optimizing parameter by finite
differences (with `bga-5.fe`'s step `pdelta = 1e-5`), also for the Newton steps'
second derivatives, where rounding errors then stall the residual around 1e-7: hence
`tol=1e-6` here (the default 1e-8 would end "not converged" at the same height)."""),
    code("""
ev = pyse.Evolver("bga-5.fe")
print("optimizing:", sorted(ev.parameters.optimizing))
result = ev.relax(levels=3, tol=1e-6)
print(result)
print(f"height {ev.parameters['height']:.4f} cm (from the curve: {load_height:.4f}), "
      f"dE/dheight {ev.derivative('height', method='fixed', move=linear(ev)):.3g} dyn")"""),
    md("""
`ev.parameters.optimize(name, False)` fixes an optimizing parameter at its value (and
`optimize(name)` frees any parameter): here the joint without the load's freedom, a
fixed gap again.

`bga-6.fe` loads the pad with 500 dynes. The curve says why that ends badly: the joint
carries 500 dynes only in its unstable range, so the chip crushes the joint sideways.
`relax()` reports it, and so does Brakke's own recipe (the height runs off below zero):"""),
    code("""
ev = pyse.Evolver("bga-6.fe")
result = ev.relax(levels=2)
print(result.converged, f"height {ev.parameters['height']:.3f}")
print(result.health.summary())"""),
    md("""
## A chip that pulls itself into line

`bga-7.fe` offsets the upper pad sideways (the parameter `offset`). The joint pulls it
back: the sideways force, from `derivative` with the upper part of the joint sheared
along (each vertex by `z/height` in x), grows linearly with the offset, a spring of
about 1100 dyn/cm."""),
    code("""
ev = pyse.Evolver("bga-7.fe")
ev.relax(levels=3)
shear = lambda x: np.c_[x[:, 2]/ev.parameters["height"], 0*x[:, 1:]]
for step in recipes.continuation(ev, "offset", [0.0, 0.01, 0.02, 0.04, 0.06, 0.08]):
    fx = -ev.derivative("offset", method="fixed", move=shear)
    print(f"offset {step.value:.3f} cm  sideways force {fx:8.3f} dyn  "
          f"energy {step.energy:.5f}")"""),
    md("""
`bga-9.fe` frees both the offset and the height, with the load on: started 0.03 cm off,
the chip slides back to the centre and settles at the loaded height."""),
    code("""
ev = pyse.Evolver("bga-9.fe")
print({k: ev.parameters[k] for k in sorted(ev.parameters.optimizing)})
result = ev.relax(levels=3)
print(result)
print({k: round(ev.parameters[k], 6) for k in sorted(ev.parameters.optimizing)})"""),
    md("""
`bga-10.fe` tilts the upper pad as well (a fixed tilt and 2D offset); the same tools
apply. Brakke's `bga-11.fe` and `bga-12.fe` free the tilt too, and `bga-31.fe` adds
contact angles to the axisymmetric model; in Evolver 2.70 they run away even with his
own recipes, so they are not included."""),
]

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = cells
for i, cell in enumerate(nb.cells):
    cell.id = f"solder-{i}"
nbf.write(nb, f"{DOCS}/solder_joints.ipynb")
