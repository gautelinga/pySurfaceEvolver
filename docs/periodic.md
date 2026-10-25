# Periodic slabs

A surface can live in a slab that repeats in x and y: a bed of beads with no side walls,
a film over a textured floor, liquid in a pack that is a cell of a larger one. Give
{meth}`~pysurfaceevolver.Evolver.load_arrays` (or {func}`~pysurfaceevolver.make_datafile`)
the cell's sides, `periods=(Lx, Ly)`. It is Evolver's torus model with a z period far
beyond the surface, so nothing ever wraps in z.

The vertices may sit anywhere: each edge runs to the nearest copy of its other end, so
every edge must be shorter than half a period (the mesh is checked). The walls may be a
floor (a plane facing up) and spheres, with any gauge. A sphere's vertices are kept at
the copy of it whose centre is given; everything that moves vertices onto it (contact,
the changes of topology) moves them there first. {attr}`~pysurfaceevolver.Evolver.periods`, and
{attr}`~pysurfaceevolver.Mesh.periods` with {meth}`~pysurfaceevolver.Mesh.delta` (differences at the nearest copy), say
where a surface is periodic. Plotting and export draw a piece that doesn't wrap whole,
and a sheet that wraps as one cell's worth.

Under the periods, the walls' integrals must not see where the cell's sides are: the
volume is Evolver's z dx dy as a facet integral, the floor's wetted area the flux of
z-hat through the free surface (less that through the spheres' wetted caps, a line
integral on each sphere's contact line), and a sphere's own integrals are local to it.
With spheres of `gauge="flux"`, each sphere's field is the sum over the lattice of its
source and of its mirror image in the floor, taken to about 1e-10 of the field (near
sources directly, the rest by local harmonic expansions, the lattice's far tail by
Ewald's sum).

## A drop across the cell's corner

The same drop with and without periods, centred on the corner of the cell: in the cell,
its vertices are wrapped into it, and its edges cross all four sides.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes

def drop(periods):
    """A drop on a floor at 60 degrees, centred on the cell's corner (0, 0)."""
    cap = shapes.cap(0.5, 0.3, nr=6, nt=24)
    v, rim = cap.vertices.copy(), np.asarray(cap.rims["rim"])
    v[rim, 2] = 0.0
    if periods is not None:
        v[:, :2] %= periods                 # every vertex into the cell
    ev = pyse.Evolver()
    ev.load_arrays(v, cap.faces, constraints={"bottom": C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
                   vertex_constraints={"bottom": rim}, bodies=[pyse.Body(volume=None)],
                   periods=periods)
    ev.body(1).target = ev.body(1).volume
    ev.command("g 1; g 5; r; g 10; hessian; hessian; hessian; hessian")
    return ev.energy, ev.body(1).volume, ev.residual()

plain, periodic = drop(None), drop((1.9, 2.3))
print(f"without periods: energy {plain[0]:.12f}, volume {plain[1]:.12f}")
print(f"in the cell:     energy {periodic[0]:.12f}, volume {periodic[1]:.12f}")
print(f"apart by {abs(plain[0] - periodic[0]):.0e} in energy, residual {periodic[2]:.0e}")
```

```text
without periods: energy 0.599893035411, volume 0.110365020610
in the cell:     energy 0.599893035411, volume 0.110365020610
apart by 1e-15 in energy, residual 4e-13
```

## Two beads in a periodic bed, drained

{doc}`drainage_topology`'s ring between two stacked beads, with the box's side walls
replaced by the periods: a bed of bead pairs, 3 apart, under a layer of liquid, drained
through the floor (`outlet="bottom"`). The beads use the flux gauge.

```python
from pysurfaceevolver import recipes
from periodic_helpers import two_beads, L

ev = two_beads(theta=20.0)          # the liquid over two stacked beads, the floor "bottom"
print(f"periods {ev.periods}; the liquid starts at volume {ev.body(1).volume:.2f} "
      f"(the cell to 4.05, {L*L*4.05:.2f}, less the beads)")
steps = []
for s in recipes.arclength(ev, ev.body(1), -0.5, 200, topology=True, outlet="bottom"):
    steps.append(s)
    if ev.on_constraint("bottom").any():
        break
cut = next(i for i, s in enumerate(steps) if s.trapped > 0)
ring = steps[cut].trapped
held = max(abs(s.trapped/ring - 1) for s in steps[cut:])
print("the ring between the beads is cut off "
      + ("before the liquid reaches the floor" if cut < len(steps) - 1 else "at the floor"))
print("from there the ring holds its volume to round-off" if held < 1e-12
      else f"the ring's volume drifts by {held:.0e}")
```

```text
periods (3.0, 3.0); the liquid starts at volume 28.07 (the cell to 4.05, 36.45, less the beads)
the ring between the beads is cut off before the liquid reaches the floor
from there the ring holds its volume to round-off
```

Where the ring is cut, and so what it holds, depends on the way down: the last
digits of the steps before the cut move it, and they differ from machine to machine
and with the NumPy version. On three, the ring was cut at 9.68 to 9.89, holding 0.265
to 0.318 (about 3% of the liquid), and the liquid reached the floor at 2.43 to 2.48;
each ring held its volume to 1e-15 or better.

The run, the cell tiled two by two (`tools/drainage_topology_figure.py periodic`): the
draining liquid blue, the ring cut off orange. The layer comes down onto the upper
beads, wets them, and pulls back between them until the liquid at the waist between
the two beads of a pair is a ring of its own.

```{image} _static/periodic_ring_shapes.png
:alt: The liquid in a periodic cell of two stacked beads, tiled two by two, before the ring is cut off, as it is, and at the floor.
```

<video controls width="640" src="_static/periodic_ring.mp4"></video>

## Limits

* Walls: one floor (a plane z = const facing up) and spheres. A lid, side walls and
  cylinders aren't periodic-invariant walls here; spheres of `gauge="flux"` can't be
  mixed with other spheres.
* The cell can't be smaller than about two edges: every edge shorter than half a
  period.
* Evolver's threaded loops don't run in torus models: a slab's steps are serial.
* The flux gauge's lattice sums cost about five times its direct sums at one bead
  a Newton step (ten times a gradient step); they don't grow with the number of
  beads, and the direct sums do.
