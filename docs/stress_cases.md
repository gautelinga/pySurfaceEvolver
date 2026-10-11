# Stress cases

Eight hard problems with exact or axisymmetric references, each run with `relax()`'s
defaults: contact lines that travel far, surfaces that grow or shrink a
hundredfold, wires, instabilities, gravity. They are the stress suite in
`bench/stress/` condensed into examples (`python bench/stress/run.py` runs the full
versions and checks them). The code below is not run when the documentation is built;
the outputs are from running it in October 2026 (the development version after pySE
0.7.0, 2 threads), and they vary in
the last digits with the thread count.

The examples import starting shapes and reference solutions from
[`docs/stress_helpers.py`](https://github.com/gautelinga/pySurfaceEvolver/blob/main/docs/stress_helpers.py)
(the references need SciPy); run them from the `docs` directory, or put it on
`sys.path`. `recipes.continuation` steps a parameter or a volume and relaxes after
each step; `step.result` is that relaxation's result.

Each case has two figures, from a run of the full suite version
(`python bench/stress/figures.py`, which also redraws them): the result against its
reference, the error per step with the case's pass limit, and what `relax()` did at each
step (gradient and Newton steps, seconds, and marks where it didn't converge or reported
an unstable equilibrium); and the surface at a few steps.

A ninth case, the drainage band of [slit_drainage](slit_drainage.ipynb), needs the mesh
remeshed explicitly at each step (`ev.remesh(target=h)`, as in that notebook); with
`relax()` alone its band pressures are off by up to several percent. See
[Relaxing a surface](relaxing.md) for why that is not automatic.

## A liquid column past Rayleigh-Plateau

A column of liquid between two plates (contact angle 90 degrees, so the contact lines
slide freely), thinned at constant height. A cylinder is an equilibrium at every radius,
but once its height exceeds its circumference (radius below 1/pi for height 1) it is a
saddle: any waist lowers the area. The solver keeps finding the cylinder; only the
stability check tells.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes
from stress_helpers import tube

# a liquid column of radius 0.5 between the plates z = 0 and z = 1, contact angle 90
v, f, bottom, top = tube(0.5, 0.0, 1.0, nz=12, nt=24)
ev = pyse.Evolver()
ev.load_string(pyse.make_datafile(
    v, f, constraints={1: C.plane((0, 0, 1), 0.0), 2: C.plane((0, 0, -1), point=(0, 0, 1))},
    vertex_constraints={1: bottom, 2: top},
    bodies=[pyse.Body(faces=range(len(f)), volume=np.pi*0.5**2)]))
ev.relax()

# thin it at constant height: past r = 1/pi the column is a saddle (Rayleigh-Plateau)
for step in recipes.continuation(ev, recipes.body_target(1), np.pi*np.linspace(0.5, 0.2, 25)**2):
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
from pysurfaceevolver import recipes
from stress_helpers import tube

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
v, f, bottom, top = tube(1.0, -0.3, 0.3, nz=8, nt=32)
ev = pyse.Evolver()
ev.load_string(pyse.make_datafile(
    v, f, constraints={1: "x^2 + y^2 = 1", 2: "z = hh", 3: "z = -hh"},
    vertex_constraints={1: bottom + top, 2: top, 3: bottom}, parameters={"hh": 0.3}))
ev.relax(levels=1)
for step in recipes.continuation(ev, recipes.parameter("hh"), np.arange(0.325, 0.76, 0.025)):
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
H = 0.500: area 5.990531, catenoid 5.991797, error 0.02%
H = 0.525: area 6.253761, catenoid 6.255163, error 0.02%
H = 0.550: area 6.510162, catenoid 6.511674, error 0.02%
H = 0.575: area 6.758706, catenoid 6.760303, error 0.02%
H = 0.600: area 6.997991, catenoid 6.999643, error 0.02%
H = 0.625: area 7.225879, catenoid 7.227532, error 0.02%
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
relaxes again (the facet count grows near the rim).

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes
from stress_helpers import cap, cap_reference, contact_radius

# a hemispherical drop on the plane z = 0, its contact angle the parameter theta
v, f, rim = cap(1.0, 1.0)
ev = pyse.Evolver()
ev.load_string(pyse.make_datafile(
    v, f, constraints={1: C.plane((0, 0, 1), 0.0, contact_angle="theta")},
    vertex_constraints={1: rim}, parameters={"theta": 90.0},
    bodies=[pyse.Body(faces=range(len(f)), volume=2*np.pi/3)]))
ev.relax()
for step in recipes.continuation(ev, recipes.parameter("theta"), np.arange(100.0, 175, 10)):
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
theta 170.0: energy error 0.17%, contact radius 0.1609 (exact 0.1378), 1120 facets
```

Energies within 0.25%. At 170 degrees the contact disk (radius 0.14) is resolved by a
coarse ring and the radius is 17% off: the implied contact angle is 1.7 degrees off.
Start from a finer mesh for such angles.

```{figure} _static/stress/case03.png
:alt: Case 3: result, error and solver steps
:width: 100%

Left, the contact radius against the exact cap (log scale) and the error of the implied contact angle (limit 1.5 degrees: only 170 fails) and of the energy; the first bar of each sweep includes the first relaxation. Middle, the error against the reference. Right, what relax() did at each step.
```

```{figure} _static/stress/case03_shapes.png
:alt: Case 3: shapes
:width: 100%

The drop at 10, 50, 130 and 170 degrees, same volume.
```

## A liquid bridge as the gap closes

A bridge of liquid (volume 0.05, contact angle 40) between two unit spheres, the gap
closed from 0.2 to 1e-3. Each gap starts from a coarse cylinder (16 around) and is
relaxed with `levels=2`. Reference: the axisymmetric Young-Laplace solution. On that
coarse start the contact lines can collapse within a few steps; `relax()` notices and
undoes it.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C
from stress_helpers import tube, bridge_spheres

def bridge(gap, theta=40.0, volume=0.05):
    """A liquid bridge between unit spheres, from a cylinder holding the volume."""
    c = 1 + gap/2
    lo, hi = 0.0, 0.999              # the cylinder radius that holds the volume
    for _ in range(60):
        a = (lo + hi)/2
        lo, hi = (a, hi) if 2*np.pi*(c*a*a - 2/3*(1 - (1 - a*a)**1.5)) < volume else (lo, a)
    zb = c - np.sqrt(1 - a*a)
    v, f, bottom, top = tube(a, -zb, zb, nz=4, nt=16)
    s1 = C.sphere((0, 0, -c), 1.0, contact_angle=theta, wet_poles=("north",))
    s2 = C.sphere((0, 0, c), 1.0, contact_angle=theta, wet_poles=("south",))
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: s1, 2: s2}, vertex_constraints={1: bottom, 2: top},
        bodies=[pyse.Body(faces=range(len(f)), volume=volume, volconst=s1.volconst + s2.volconst)]))
    ev.relax(levels=2)
    return ev, s1.energy_constant + s2.energy_constant

for gap in (0.2, 0.05, 0.01, 0.001):
    ev, const = bridge(gap)
    ref = bridge_spheres(0.05, gap, 40.0)
    print(f"gap {gap:5.3f}: energy {ev.total_energy + const:.5f} (reference {ref['energy']:.5f}), "
          f"pressure {ev.body(1).pressure:.4f} (reference {ref['pressure']:.4f})")
```

```text
gap 0.200: energy 0.08377 (reference 0.08390), pressure -0.1753 (reference -0.1734)
gap 0.050: energy -0.24965 (reference -0.24967), pressure -1.9179 (reference -1.9090)
gap 0.010: energy -0.37990 (reference -0.37985), pressure -1.8928 (reference -1.8867)
gap 0.001: energy -0.41157 (reference -0.41151), pressure -1.8437 (reference -1.8377)
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
from pysurfaceevolver import constraints as C, recipes
from stress_helpers import revolve, barrel

# a barrel drop (volume 2) on a fibre of radius 0.2 (the z axis)
z = np.linspace(-1.2, 1.2, 13)
v, f, bottom, top = revolve(np.sqrt(0.04 + (0.75**2 - 0.04)*(1 - (z/1.2)**2)), z, nt=24)
ev = pyse.Evolver()
ev.load_string(pyse.make_datafile(
    v, f, constraints={1: C.cylinder((0, 0, 0), (0, 0, 1), 0.2, contact_angle="theta")},
    vertex_constraints={1: bottom + top}, parameters={"theta": 20.0},
    bodies=[pyse.Body(faces=range(len(f)), volume=2.0)]))
ev.relax(levels=1)
guess = (0.78, 2.4)
for step in recipes.continuation(ev, recipes.parameter("theta"), np.arange(20.0, 81, 10)):
    ref = barrel(2.0, 0.2, step.value, guess=guess)
    guess = (ref["equator"], ref["pressure"])
    print(f"theta {step.value:4.0f}: energy {ev.total_energy:.4f} (barrel {ref['energy']:.4f}), "
          f"stable {step.result.stable}")
```

```text
theta   20: energy 5.8216 (barrel 5.8114), stable True
theta   30: energy 6.0292 (barrel 6.0202), stable True
theta   40: energy 6.2922 (barrel 6.2842), stable True
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
from pysurfaceevolver import examples, recipes

ev = pyse.Evolver(examples.path("cube.fe"))    # the sample: a unit cube that relaxes to a sphere
ev.relax(levels=2)
for step in recipes.continuation(ev, recipes.body_target(1), [1, 10, 100]):
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
from pysurfaceevolver import constraints as C
from stress_helpers import cap, cap_reference, sessile

# a drop of volume 50 with gravity (capillary length 1), contact angle 60,
# started as the cap it would be without gravity
c0 = cap_reference(50.0, 60.0)
v, f, rim = cap(c0["R"], c0["height"], nr=10, nt=48)
ev = pyse.Evolver()
ev.load_string(pyse.make_datafile(
    v, f, constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
    vertex_constraints={1: rim}, header="gravity_constant 1\n",
    bodies=[pyse.Body(faces=range(len(f)), volume=50.0, density=1.0)]))
ev.relax(levels=1)
ref = sessile(50.0, 60.0, rho_g=1.0)
print(f"height {ev.mesh().vertices[:, 2].max():.4f} (reference {ref['height']:.4f}), "
      f"energy {ev.total_energy:.4f} (reference {ref['energy']:.4f})")
```

```text
height 1.0739 (reference 1.0714), energy 62.6164 (reference 62.5967)
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
from pysurfaceevolver import constraints as C, recipes
from stress_helpers import cap, cap_reference, contact_radius

# a 60-degree cap evaporating to 1e-4 of its volume
h = 1 - np.cos(np.radians(60.0))
v, f, rim = cap(1.0, h)
volume = np.pi*h*h*(3 - h)/3
ev = pyse.Evolver()
ev.load_string(pyse.make_datafile(
    v, f, constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
    vertex_constraints={1: rim}, bodies=[pyse.Body(faces=range(len(f)), volume=volume)]))
ev.relax()
# 25 steps, each about 70% of the last (jumps of 10x are too large to follow)
for step in recipes.continuation(ev, recipes.body_target(1), volume*np.geomspace(1, 1e-4, 25)):
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
