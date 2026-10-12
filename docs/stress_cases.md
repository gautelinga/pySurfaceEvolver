# Stress cases

Thirteen hard problems with exact or axisymmetric references, each run with `relax()`'s
defaults: contact lines that travel far, surfaces that grow or shrink a
hundredfold, wires, instabilities, gravity. They are the stress suite in
`bench/stress/` condensed into examples (`python bench/stress/run.py` runs the full
versions and checks them). The code below is not run when the documentation is built;
the outputs are from running it in October 2026 (pySE 0.7.3 and the development
version after it, 2 threads), and they vary in
the last digits with the thread count.

The starting shapes come from `pysurfaceevolver.shapes`, the reference solutions from
[`docs/stress_helpers.py`](https://github.com/gautelinga/pySurfaceEvolver/blob/main/docs/stress_helpers.py)
(they need SciPy); run the examples from the `docs` directory, or put it on
`sys.path`. `recipes.continuation` steps a parameter or a volume and relaxes after
each step; `step.result` is that relaxation's result.

Each case has two figures, from a run of the full suite version
(`python bench/stress/figures.py`, which also redraws them): the result against its
reference, the error per step with the case's pass limit, and what `relax()` did at each
step (gradient and Newton steps, seconds, and marks where it didn't converge or reported
an unstable equilibrium); and the surface at a few steps.

A fourteenth case in the suite, the drainage band of [slit_drainage](slit_drainage.ipynb),
is run the way that notebook runs it: `ev.remesh(target=h)` before each step's
`relax()`. It passes that way (median band-pressure error 0.02%, the snap within 0.4%
of a finer reference, nearly nine in ten steps converged). With `relax()` alone it
doesn't: see [Relaxing a surface](relaxing.md) for why, and why that is not automatic.

## A liquid column past Rayleigh-Plateau

A column of liquid between two plates (contact angle 90 degrees, so the contact lines
slide freely), thinned at constant height. A cylinder is an equilibrium at every radius,
but once its height exceeds its circumference (radius below 1/pi for height 1) it is a
saddle: any waist lowers the area. The solver keeps finding the cylinder; only the
stability check tells.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

# a liquid column of radius 0.5 between the plates z = 0 and z = 1, contact angle 90
column = shapes.tube(0.5, 0.0, 1.0, nz=12, nt=24)
ev = pyse.Evolver()
ev.load_arrays(column.vertices, column.faces,
               constraints={1: C.plane((0, 0, 1), 0.0), 2: C.plane((0, 0, -1), point=(0, 0, 1))},
               vertex_constraints={1: column.rims["bottom"], 2: column.rims["top"]},
               bodies=[pyse.Body(volume=np.pi*0.5**2)])
ev.relax()

# thin it at constant height: past r = 1/pi the column is a saddle (Rayleigh-Plateau)
for step in recipes.continuation(ev, ev.body(1), np.pi*np.linspace(0.5, 0.2, 25)**2):
    if step.result.stable is False:
        print(f"unstable from r = {np.sqrt(step.value/np.pi):.4f} (theory 1/pi = {1/np.pi:.4f})")
        break
```

```text
unstable from r = 0.3125 (theory 1/pi = 0.3183)
```

The check flags it one step past the onset (the steps are 0.0125 apart in radius).

```{figure} _static/stress/case01.png
:alt: Case 1: result, error and solver steps
:width: 100%

Left, the column's area against the cylinder's, with the count of negative Hessian eigenvalues (red) turning to one at r = 1/pi; relax() marks every step past the onset unstable. Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case01_shapes.png
:alt: Case 1: shapes
:width: 100%

The column at radius 0.5, 0.4, at the onset and at 0.2: still a cylinder, a saddle.
```

## A catenoid to its fold

A soap film between two wire rings of radius 1, pulled apart. Up to H = 0.6627 the
stable catenoid exists; past the fold its neck collapses and the film pinches off into
two flat discs, one in each ring. The wires are curves where two constraints meet: their vertices can only slide
along them, which Newton must not mistake for a freedom.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import recipes, shapes

def catenoid_area(H):
    """The stable catenoid between rings of radius 1 at z = +-H (None past the fold)."""
    lo, hi = 1/np.cosh(1.19967864), 1.0
    if H >= 1.19967864/np.cosh(1.19967864):
        return None
    for _ in range(100):
        a = (lo + hi)/2
        lo, hi = (lo, a) if a*np.cosh(H/a) > 1 else (a, hi)
    return np.pi*a*(2*H + a*np.sinh(2*H/a))

# a soap film between two wire rings, the rings pulled apart (parameter hh)
film = shapes.tube(1.0, -0.3, 0.3, nz=8, nt=32)
bottom, top = film.rims["bottom"], film.rims["top"]
ev = pyse.Evolver()
ev.load_arrays(film.vertices, film.faces,
               constraints={1: "x^2 + y^2 = 1", 2: "z = hh", 3: "z = -hh"},
               vertex_constraints={1: bottom + top, 2: top, 3: bottom}, parameters={"hh": 0.3})
ev.relax(levels=1)
for step in recipes.continuation(ev, "hh", np.arange(0.325, 0.76, 0.025)):
    ref = catenoid_area(step.value)
    if ref is not None:
        print(f"H = {step.value:.3f}: area {ev.total_energy:.6f}, catenoid {ref:.6f}, "
              f"error {abs(ev.total_energy/ref - 1):.2%}")
    else:
        print(f"H = {step.value:.3f}: past the fold, area {ev.total_energy:.6f} (two discs "
              f"{2*np.pi:.6f}), converged {step.result.converged}")
```

```text
H = 0.325: area 4.008181, catenoid 4.009254, error 0.03%
H = 0.350: area 4.303094, catenoid 4.304133, error 0.02%
H = 0.375: area 4.594750, catenoid 4.595759, error 0.02%
H = 0.400: area 4.882871, catenoid 4.883793, error 0.02%
H = 0.425: area 5.166914, catenoid 5.167861, error 0.02%
H = 0.450: area 5.446549, catenoid 5.447546, error 0.02%
H = 0.475: area 5.721403, catenoid 5.722374, error 0.02%
H = 0.500: area 5.990525, catenoid 5.991797, error 0.02%
H = 0.525: area 6.253760, catenoid 6.255163, error 0.02%
H = 0.550: area 6.510162, catenoid 6.511674, error 0.02%
H = 0.575: area 6.758706, catenoid 6.760303, error 0.02%
H = 0.600: area 6.997992, catenoid 6.999643, error 0.02%
H = 0.625: area 7.225880, catenoid 7.227532, error 0.02%
H = 0.650: area 7.438262, catenoid 7.439781, error 0.02%
H = 0.675: past the fold, area 6.273097 (two discs 6.283185), converged True
H = 0.700: past the fold, area 6.273097 (two discs 6.283185), converged True
H = 0.725: past the fold, area 6.273097 (two discs 6.283185), converged True
H = 0.750: past the fold, area 6.273097 (two discs 6.283185), converged True
```

Within 0.03% of the exact catenoid up to the fold. Past it, `relax()` can't converge
the film, lets the neck collapse, deletes its collapsed ring of edges and pops the
pinched vertex (`step.result.collapsed`, `.popped`): two discs, whose area is that of
the 64-sided polygons the rings' vertices make (0.16% below 2 pi).

```{figure} _static/stress/case02.png
:alt: Case 2: result, error and solver steps
:width: 100%

Left, the film's area on the exact catenoid up to the fold, and past it on two flat discs. Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case02_shapes.png
:alt: Case 2: shapes
:width: 100%

The film at H = 0.325, 0.5 and 0.65, and past the fold, pinched off into two discs.
```

## A sessile drop at high contact angles

A drop (no gravity) whose contact angle is stepped from 90 to 170 degrees at fixed
volume: the drop rolls up into nearly a sphere on a small contact disk. Reference: the
exact spherical cap. The contact circle shrinks below the size of the facets next to
it; after converging, `relax()` splits the long edges leaving the contact line and
relaxes again, while some are long (the facet count grows near the rim).

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes
from stress_helpers import cap_reference, contact_radius

# a hemispherical drop on the plane z = 0, its contact angle the parameter theta
drop = shapes.cap(1.0, 1.0)
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.plane((0, 0, 1), 0.0, contact_angle="theta")},
               vertex_constraints={1: drop.rims["rim"]}, parameters={"theta": 90.0},
               bodies=[pyse.Body(volume=2*np.pi/3)])
ev.relax()
for step in recipes.continuation(ev, "theta", np.arange(100.0, 175, 10)):
    ref = cap_reference(2*np.pi/3, step.value)
    print(f"theta {step.value:5.1f}: energy error {abs(ev.total_energy/ref['energy'] - 1):.2%}, "
          f"contact radius {contact_radius(ev, 1):.4f} (exact {ref['radius']:.4f}), "
          f"{ev.counts['facets']} facets")
```

```text
theta 100.0: energy error 0.19%, contact radius 0.9179 (exact 0.9123), 480 facets
theta 110.0: energy error 0.21%, contact radius 0.8286 (exact 0.8222), 480 facets
theta 120.0: energy error 0.24%, contact radius 0.7352 (exact 0.7274), 480 facets
theta 130.0: energy error 0.25%, contact radius 0.6351 (exact 0.6261), 608 facets
theta 140.0: energy error 0.22%, contact radius 0.5258 (exact 0.5168), 736 facets
theta 150.0: energy error 0.20%, contact radius 0.4088 (exact 0.3986), 864 facets
theta 160.0: energy error 0.18%, contact radius 0.2834 (exact 0.2717), 992 facets
theta 170.0: energy error 0.17%, contact radius 0.1550 (exact 0.1378), 1376 facets
```

Energies within 0.25%. At 170 degrees the contact disk (radius 0.14) is small next to
the facets around it: `relax()` splits the long edges leaving the contact line, up to
three times per call while some are long, and the implied contact angle is 1.3
degrees off (the radius 12% off). A finer start does better: one refinement more
brings it within 0.3 degrees.

```{figure} _static/stress/case03.png
:alt: Case 3: result, error and solver steps
:width: 100%

Left, the contact radius against the exact cap (log scale) and the error of the implied contact angle (limit 1.5 degrees) and of the energy; the first bar of each sweep includes the first relaxation. Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case03_shapes.png
:alt: Case 3: shapes
:width: 100%

The drop at 10, 50, 130 and 170 degrees, same volume.
```

## A liquid bridge as the gap closes

A bridge of liquid (volume 0.05, contact angle 40) between two unit spheres, the gap
closed from 0.2 to 1e-3. Each gap starts from a coarse cylinder (16 around, radius
0.35: relaxing meets the volume) and is relaxed with `levels=2`. Reference: the axisymmetric Young-Laplace solution. On that
coarse start the contact lines can collapse within a few steps; `relax()` notices and
undoes it.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes
from stress_helpers import bridge_spheres

def bridge(gap, theta=40.0, volume=0.05, a=0.35):
    """A liquid bridge between unit spheres, from a cylinder of radius a."""
    c = 1 + gap/2
    zb = c - np.sqrt(1 - a*a)                  # the cylinder's ends on the spheres
    tube = shapes.tube(a, -zb, zb, nz=4, nt=16)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   constraints={1: C.sphere((0, 0, -c), 1.0, contact_angle=theta, wet_poles=("north",)),
                                2: C.sphere((0, 0, c), 1.0, contact_angle=theta, wet_poles=("south",))},
                   vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=volume)])    # volconst: the spheres' caps
    ev.relax(levels=2)
    return ev

for gap in (0.2, 0.05, 0.01, 0.001):
    ev = bridge(gap)
    ref = bridge_spheres(0.05, gap, 40.0)
    # ev.energy: with the wetted caps' energy, which Evolver's total_energy leaves out
    print(f"gap {gap:5.3f}: energy {ev.energy:.5f} (reference {ref['energy']:.5f}), "
          f"pressure {ev.body(1).pressure:.4f} (reference {ref['pressure']:.4f})")
```

```text
gap 0.200: energy 0.08377 (reference 0.08390), pressure -0.1755 (reference -0.1734)
gap 0.050: energy -0.24965 (reference -0.24967), pressure -1.9178 (reference -1.9090)
gap 0.010: energy -0.37992 (reference -0.37985), pressure -1.8925 (reference -1.8867)
gap 0.001: energy -0.41160 (reference -0.41151), pressure -1.8433 (reference -1.8377)
```

Energies within 0.2% and pressures within 0.01 (0.5% where they are large) at every
gap.

```{figure} _static/stress/case05.png
:alt: Case 5: result, error and solver steps
:width: 100%

Left, energy and neck radius against the axisymmetric bridge, the gap on a log scale; each gap is a new surface relaxed with two refinements. Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case05_shapes.png
:alt: Case 5: shapes
:width: 100%

The bridge at gaps 0.2, 0.01 and 0.001.
```

## A barrel drop on a fibre

A drop of volume 2 on a fibre of radius 0.2, its contact angle stepped from 20 to 80
degrees. Reference: the axisymmetric barrel. Small drops at higher contact angles roll
up to one side (a clam shell); the symmetric barrel remains an equilibrium, a saddle,
and the stability check reports it. Sliding along the fibre is an exact zero mode,
which must not count as an instability.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes
from stress_helpers import barrel

# a barrel drop (volume 2) on a fibre of radius 0.2 (the z axis)
z = np.linspace(-1.2, 1.2, 13)
drop = shapes.revolve(np.sqrt(0.04 + (0.75**2 - 0.04)*(1 - (z/1.2)**2)), z, nt=24)
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.cylinder((0, 0, 0), (0, 0, 1), 0.2, contact_angle="theta")},
               vertex_constraints={1: drop.rims["bottom"] + drop.rims["top"]},
               parameters={"theta": 20.0}, bodies=[pyse.Body(volume=2.0)])
ev.relax(levels=1)
guess = (0.78, 2.4)
for step in recipes.continuation(ev, "theta", np.arange(20.0, 81, 10)):
    ref = barrel(2.0, 0.2, step.value, guess=guess)
    guess = (ref["equator"], ref["pressure"])
    print(f"theta {step.value:4.0f}: energy {ev.total_energy:.4f} (barrel {ref['energy']:.4f}), "
          f"stable {step.result.stable}")
```

```text
theta   20: energy 5.8215 (barrel 5.8114), stable True
theta   30: energy 6.0292 (barrel 6.0202), stable True
theta   40: energy 6.2921 (barrel 6.2842), stable True
theta   50: energy 6.5938 (barrel 6.5870), stable True
theta   60: energy 6.9187 (barrel 6.9130), stable True
theta   70: energy 7.2530 (barrel 7.2483), stable False
theta   80: energy 7.5844 (barrel 7.5806), stable False
```

Within 0.2% of the barrel; unstable (two negative modes, the roll-up in two
directions) from 70 degrees on.

```{figure} _static/stress/case06.png
:alt: Case 6: result, error and solver steps
:width: 100%

Left, the equator radius against the axisymmetric barrel; from 70 degrees relax() reports the barrel unstable (orange): it would roll up into a clam shell. Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case06_shapes.png
:alt: Case 6: shapes
:width: 100%

The barrel at 20, 50 and 80 degrees.
```

## A cube inflated a hundredfold

The sample cube relaxed to a sphere and inflated to 100 times its volume: the surface
stretches tenfold. Reference: the sphere's area. The relative error must not grow.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import recipes

ev = pyse.Evolver("cube.fe")    # the sample: a unit cube that relaxes to a sphere
ev.relax(levels=2)
for step in recipes.continuation(ev, ev.body(1), [1, 10, 100]):
    sphere = (36*np.pi)**(1/3)*step.value**(2/3)
    print(f"volume {step.value:5.0f}: area {ev.total_energy:.4f}, sphere {sphere:.4f}, "
          f"error {abs(ev.total_energy/sphere - 1):.3%}")
```

```text
volume     1: area 4.8523, sphere 4.8360, error 0.338%
volume    10: area 22.5227, sphere 22.4466, error 0.339%
volume   100: area 104.5410, sphere 104.1879, error 0.339%
```

The error stays at the coarse mesh's 0.34%.

```{figure} _static/stress/case09.png
:alt: Case 9: result, error and solver steps
:width: 100%

Left, the area against a sphere's; the error stays at the starting mesh's as the surface stretches tenfold. Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case09_shapes.png
:alt: Case 9: shapes
:width: 100%

The surface at volumes 1, 10 and 100, to scale.
```

## A gravity puddle

A drop of volume 50 with gravity (capillary length 1) and contact angle 60, started as
the cap it would be without gravity, so it has to flatten a lot. Reference: the
axisymmetric sessile drop with gravity; a large puddle's height tends to
2 sin(theta/2) = 1.

```python
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes
from stress_helpers import cap_reference, sessile

# a drop of volume 50 with gravity (capillary length 1), contact angle 60,
# started as the cap it would be without gravity
c0 = cap_reference(50.0, 60.0)
drop = shapes.cap(c0["R"], c0["height"], nr=10, nt=48)
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
               vertex_constraints={1: drop.rims["rim"]}, gravity=1,
               bodies=[pyse.Body(volume=50.0, density=1.0)])
ev.relax(levels=1)
ref = sessile(50.0, 60.0, rho_g=1.0)
print(f"height {ev.mesh().vertices[:, 2].max():.4f} (reference {ref['height']:.4f}), "
      f"energy {ev.total_energy:.4f} (reference {ref['energy']:.4f})")
```

```text
height 1.0740 (reference 1.0714), energy 62.6181 (reference 62.5967)
```

Height and energy within 0.25%.

```{figure} _static/stress/case11.png
:alt: Case 11: result, error and solver steps
:width: 100%

Left, the vertices against the axisymmetric profile with gravity, and the four errors (limit 0.5%). Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case11_shapes.png
:alt: Case 11: shapes
:width: 100%

The puddle.
```

## A drop evaporating to 1e-4 of its volume

A 60-degree cap shrinking to 1e-4 of its volume, the mesh shrinking with it.
Reference: the exact cap. The error must not grow as the drop shrinks.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes
from stress_helpers import cap_reference, contact_radius

# a 60-degree cap evaporating to 1e-4 of its volume
h = 1 - np.cos(np.radians(60.0))
drop = shapes.cap(1.0, h)
volume = np.pi*h*h*(3 - h)/3
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
               vertex_constraints={1: drop.rims["rim"]}, bodies=[pyse.Body(volume=volume)])
ev.relax()
# 25 steps, each about 70% of the last (jumps of 10x are too large to follow)
for step in recipes.continuation(ev, ev.body(1), volume*np.geomspace(1, 1e-4, 25)):
    if step.index % 6:
        continue
    ref = cap_reference(step.value, 60.0)
    print(f"volume {step.value:.2e}: energy error {abs(ev.total_energy/ref['energy'] - 1):.2%}, "
          f"radius error {abs(contact_radius(ev, 1)/ref['radius'] - 1):.2%}")
```

```text
volume 6.54e-01: energy error 0.15%, radius error 0.42%
volume 6.54e-02: energy error 0.15%, radius error 0.46%
volume 6.54e-03: energy error 0.15%, radius error 0.48%
volume 6.54e-04: energy error 0.17%, radius error 0.51%
volume 6.54e-05: energy error 0.21%, radius error 0.58%
```

The error stays near the starting mesh's (the contact radius of the coarse rim is
0.4-0.6% off).

```{figure} _static/stress/case12.png
:alt: Case 12: result, error and solver steps
:width: 100%

Left, the contact radius against the exact cap over four decades of volume; the error stays near the starting mesh's. Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case12_shapes.png
:alt: Case 12: shapes
:width: 100%

The cap at the start, at 1e-2 and at 1e-4 of its volume, each to its own scale.
```

## Schwarz's P and D surfaces

Two triply periodic minimal surfaces from Brakke's online collection (included with
pySE): a fundamental region of each in a quotient space, where a symmetry group, not
a periodic box, generates the rest. Refined five times with a default `relax()` after
each refinement; the reference is each surface's area per cubic unit cell, A/V^(2/3),
2.3451 (P) and 3.8377 (D), the standard values.

```python
import pysurfaceevolver as pyse

# fundamental regions of Schwarz's P and D surfaces, from Brakke's online
# collection; 12 times their area is the area per cubic unit cell, A/V^(2/3)
for name, label, reference in (("pcell.fe", "P", 2.3451), ("dcell.fe", "D", 3.8377)):
    ev = pyse.Evolver(name)
    ev.relax()
    areas = []
    for level in range(1, 6):
        ev.refine()
        ev.relax()
        areas.append(12*ev.total_energy)
    extrapolated = (4*areas[-1] - areas[-2])/3      # second order: error / 4 per level
    print(f"{label}: {areas[-1]:.6f} after 5 refinements, extrapolated {extrapolated:.6f}, "
          f"reference {reference}")
```

```text
P: 2.345166 after 5 refinements, extrapolated 2.345102, reference 2.3451
D: 3.838111 after 5 refinements, extrapolated 3.837785, reference 3.8377
```

The error falls fourfold per refinement (second order in the edge length), and the
extrapolated areas agree with the standard values to their last digit.

```{figure} _static/stress/case13.png
:alt: Case 13: result, error and solver steps
:width: 100%

Left, the area per unit cell against the standard value at each refinement. Middle, the error, falling fourfold per refinement. Right, what relax() did.
```

```{figure} _static/stress/case13_shapes.png
:alt: Case 13: shapes
:width: 100%

The fundamental regions after five refinements.
```

## A drop on a sphere

A drop of volume 0.5 resting on a solid sphere of radius 1, its contact angle swept as
a parameter. Without gravity the exact shape is a spherical cap meeting the sphere at
the contact angle (`drop_on_sphere` in the helpers). The wetted region holds the
sphere's north pole, where the line integrals of the wall can't reach: the
constraint's `volconst` closes the volume and `ev.energy` adds the wetting energy the
integrals leave out, both automatically, also with the angle as a parameter.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes
from stress_helpers import drop_on_sphere

V = 0.5
exact = drop_on_sphere(V, 90.0)                 # start from the exact 90 degree drop
rho, a = exact["rho"], exact["radius"]
d = np.sqrt(1 + rho**2)
t = np.linspace(np.arctan2(a, np.sqrt(1 - a**2) - d), 0, 9)
drop = shapes.revolve(rho*np.sin(t), d + rho*np.cos(t), nt=24)
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.sphere((0, 0, 0), 1.0, contact_angle="theta",
                                        wet_poles=("north",))},
               vertex_constraints={1: drop.rims["bottom"]}, parameters={"theta": 90.0},
               bodies=[pyse.Body(volume=V)])
ev.relax(levels=2)

for step in recipes.continuation(ev, "theta", [75, 60, 45, 30]):
    ref = drop_on_sphere(V, step.value)
    print(f"theta {step.value:2d}: energy {ev.energy:.5f} (exact {ref['energy']:.5f}), "
          f"pressure {ev.body(1).pressure:.4f} (exact {ref['pressure']:.4f}), "
          f"angle on the mesh {ev.contact_angles(1).mean():.1f}, "
          f"{'converged' if step.result.converged else 'NOT converged'}")
```

```text
theta 75: energy 2.46376 (exact 2.46113), pressure 3.4788 (exact 3.4772), angle on the mesh 72.9, converged
theta 60: energy 2.17851 (exact 2.17580), pressure 3.1781 (exact 3.1767), angle on the mesh 58.7, converged
theta 45: energy 1.85060 (exact 1.84797), pressure 2.8262 (exact 2.8250), angle on the mesh 43.8, converged
theta 30: energy 1.50512 (exact 1.50271), pressure 2.4525 (exact 2.4508), angle on the mesh 28.3, converged
```

The energies and pressures are within 0.2% of the exact cap's at 1,700 facets; the
contact angle measured on the mesh lags the set one by a degree or two (it is first
order in the edge length). As the contact line creeps over the sphere, gradient steps
make slow progress: once the rounds have used half of `max_iter` with the residual
falling by less than 5% a round, `relax()` tries Newton there, which converges in a
few steps. (Before that rule, the rounds used up `max_iter` and most steps ended
unconverged.)

```{figure} _static/stress/case14.png
:alt: Case 14: result, error and solver steps
:width: 100%

Left, the energy against the exact cap. Middle, the errors in energy and pressure (limit 0.5%). Right, what relax() did, in the order run.
```

```{figure} _static/stress/case14_shapes.png
:alt: Case 14: shapes
:width: 100%

The drop at 20, 90 and 150 degrees.
```

## A double bubble

Two unit boxes sharing a face relax to the standard double bubble: three spherical
caps meeting at 120 degrees along a circle, the shape proved to enclose two volumes
with the least area (Hutchings, Morgan, Ritoré and Ros). The second bubble then
shrinks; the reference is exact (`double_bubble` in the helpers). Two bodies share
the wall, so each body lists its faces.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import shapes
from stress_helpers import double_bubble

# two unit boxes side by side, sharing the face x = 1
box = shapes.box((1, 1, 1), n=2)
right = box.vertices + [1, 0, 0]
row = {tuple(np.round(p, 9)): i for i, p in enumerate(box.vertices)}
shared = np.array([row.get(tuple(np.round(p, 9)), -1) for p in right])
new = len(box.vertices) + np.cumsum(shared < 0) - 1
vertices = np.vstack([box.vertices, right[shared < 0]])
right_faces = np.where(shared >= 0, shared, new)[box.faces]
wall = np.flatnonzero(np.isclose(box.vertices[box.faces].mean(axis=1)[:, 0], 1))
right_faces = right_faces[~np.isclose(vertices[right_faces].mean(axis=1)[:, 0], 1)]
faces = np.vstack([box.faces, right_faces])
n = len(box.faces)
ev = pyse.Evolver()
ev.load_arrays(vertices, faces, bodies=[
    pyse.Body(1.0, faces=range(n)), pyse.Body(1.0, faces=list(wall) + list(range(n, len(faces))))])
ev.relax(levels=2)

for v2 in [1.0, 0.6, 0.2]:
    ev.body(2).target = v2
    ev.relax()
    ref = double_bubble(1.0, v2)
    p = ev.bodies().pressure
    print(f"volumes 1 and {v2}: area {ev.total_area:.5f} (exact {ref['area']:.5f}), "
          f"pressures {p[0]:.4f}, {p[1]:.4f} (exact {ref['p1']:.4f}, {ref['p2']:.4f})")
```

```text
volumes 1 and 1.0: area 9.15253 (exact 9.13942), pressures 3.0507, 3.0507 (exact 3.0465, 3.0465)
volumes 1 and 0.6: area 7.84328 (exact 7.83166), pressures 3.0985, 3.5491 (exact 3.0938, 3.5456)
volumes 1 and 0.2: area 6.21830 (exact 6.20805), pressures 3.1698, 4.8730 (exact 3.1640, 4.8735)
```

```{figure} _static/stress/case15.png
:alt: Case 15: result, error and solver steps
:width: 100%

Left, the total area against the standard double bubble. Middle, the errors in area and pressures. Right, what relax() did.
```

```{figure} _static/stress/case15_shapes.png
:alt: Case 15: shapes
:width: 100%

The double bubble at volume ratios 1, 0.6 and 0.2.
```

## The force of a solder joint

The force a liquid solder joint exerts on its pad (Brakke's `bga-3.fe`: pads of
radius 0.1 cm 0.12 cm apart, surface tension 480 erg/cm², gravity), from
`ev.derivative` with the joint stretched along with the pad. The reference is the same
joint in Brakke's axisymmetric string model, `bga-30.fe` (its solder density set to
`bga-3`'s 9 g/cm³; it ships with 8.9, which alone moves the force by 1.3%), refined
until it converges. See [Solder joints](solder_joints.ipynb) for the whole series.

```python
import numpy as np
import pysurfaceevolver as pyse

def force(ev, axis):                     # minus dE/dheight, the joint stretched along
    h = ev.parameters["height"]
    move = lambda x: np.where(np.arange(x.shape[1]) == axis, x/h, 0.0)
    return -ev.derivative("height", method="fixed", move=move)

axi = pyse.Evolver("bga-30.fe")
axi.parameters["s_density"] = 9.0
axi.relax(levels=2)
F = []
for level in range(8):
    if level:
        axi.relax(levels=1)
    F.append(force(axi, 1))
reference = (4*F[-1] - F[-2])/3                  # second order: error / 4 per level
print(f"axisymmetric: {F[-1]:.5f} dyn at {axi.counts['edges']} edges, extrapolated {reference:.5f}")

ev = pyse.Evolver("bga-3.fe")
ev.relax(levels=2)
forces = []
for level in range(4):
    if level:
        ev.relax(levels=1)
    forces.append(force(ev, 2))
    print(f"3D, {ev.counts['facets']:5d} facets: {forces[-1]:.5f} dyn "
          f"({forces[-1]/reference - 1:+.2%})")
print(f"3D extrapolated: {(4*forces[-1] - forces[-2])/3:.5f}")
```

```text
axisymmetric: 16.70464 dyn at 515 edges, extrapolated 16.70370
3D,   432 facets: 29.97050 dyn (+79.42%)
3D,  1632 facets: 19.87428 dyn (+18.98%)
3D,  6336 facets: 17.48835 dyn (+4.70%)
3D, 24960 facets: 16.89928 dyn (+1.17%)
3D extrapolated: 16.70292
```

Brakke's starting meshes give a force nearly twice the converged one; both models
then converge at second order, to the same force.

```{figure} _static/stress/case16.png
:alt: Case 16: result, error and solver steps
:width: 100%

Left, the 3D force against the axisymmetric reference. Middle, its error, second order in the edge length. Right, what relax() did at each refinement.
```

```{figure} _static/stress/case16_shapes.png
:alt: Case 16: shapes
:width: 100%

The joint on Brakke's starting mesh and refined three times.
```

## A catenoid between round wires, to its fold

A soap film between two round wires of radius 1, pulled apart. The wires are Evolver
boundaries (`constraints.circle` and `constraints.curve`), so refining keeps them
round, and their vertices are fixed. Past a separation of 1.3255 no catenoid spans
the wires; `continuation(stop_on="unconverged")` stops there and puts the last
catenoid back.

```python
import warnings
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

tube = shapes.tube(1.0, 0.0, 0.6, nz=4, nt=16)
top = C.curve("cos(p1)", "sin(p1)", "ht", lambda x: np.arctan2(x[:, 1], x[:, 0]))
ev = pyse.Evolver()
ev.load_arrays(tube.vertices, tube.faces, boundaries={1: C.circle(), 2: top},
               vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
               parameters={"ht": 0.6})
ev.relax(levels=2)

def catenoid(H):                       # the exact area, wires at -H and H
    lo, hi = 1/np.cosh(1.19967864), 1.0  # the stable root of a cosh(H/a) = 1
    for _ in range(100):
        a = (lo + hi)/2
        lo, hi = (lo, a) if a*np.cosh(H/a) > 1 else (a, hi)
    return np.pi*a*(2*H + a*np.sinh(2*H/a))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for step in recipes.continuation(ev, "ht", np.arange(0.8, 1.6, 0.1), stop_on="unconverged"):
        print(f"wires {step.value:.1f} apart: area {ev.total_area:.5f} "
              f"(exact {catenoid(step.value/2):.5f})")
print([str(w.message).split(";")[0] for w in caught if "stopped" in str(w.message)])
```

```text
wires 0.8 apart: area 4.88239 (exact 4.88379)
wires 0.9 apart: area 5.44628 (exact 5.44755)
wires 1.0 apart: area 5.99072 (exact 5.99180)
wires 1.1 apart: area 6.51062 (exact 6.51167)
wires 1.2 apart: area 6.99884 (exact 6.99964)
wires 1.3 apart: area 7.43941 (exact 7.43978)
['continuation() stopped before 1.4: relax() came out unconverged there']
```

```{figure} _static/stress/case17.png
:alt: Case 17: result, error and solver steps
:width: 100%

Left, the area against the exact catenoid, with the fold. Middle, the error. Right, what relax() did; the run stops at the first step past the fold.
```

```{figure} _static/stress/case17_shapes.png
:alt: Case 17: shapes
:width: 100%

The film between the wires at three separations.
```
