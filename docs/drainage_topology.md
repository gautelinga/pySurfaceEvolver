# Draining through changes of topology

Liquid that drains slowly out of a porous material changes its connectivity as it
goes: films over the grains thin away and dry patches open, menisci jump past the
narrow places, and what was one body of liquid can rearrange at once. With
`topology=True`, {func}`~pysurfaceevolver.recipes.arclength` follows the liquid's
volume through such changes. At each point it looks for the changes that are coming
({meth}`~pysurfaceevolver.Evolver.events`). It makes each one in the mesh
({meth}`~pysurfaceevolver.Evolver.contact`, {meth}`~pysurfaceevolver.Evolver.merge`
and {meth}`~pysurfaceevolver.Evolver.pinch_off`) and lets the surface settle at the
same volume. Then it goes on along the new branch. ({doc}`bridge_rupture` shows the
pinch-off on its own.)

This page drains two shapes this way. The first is a drop with a solid sphere inside
it; the second is one pore of a bead pack. Both use spheres whose wetting integrals
have a single singular point, put where the liquid never goes or always is
(`constraints.sphere(axis=...)`). The default azimuth about z is singular at both
poles, and a contact line that crossed one would carry the integrals with it. Where
a contact line still comes near a sphere's singular point, the run moves the point
to the far side of the sphere from the lines, keeping the volume and energy.

## A drop draining over a sphere

A drop of radius 3 sits on a plate at 90 degrees (a mirror: half of a drop of radius
3), with a sphere of radius 0.8 inside it at 30 degrees. As the drop drains, its
surface comes down onto the sphere. The film over the sphere thins away, a dry patch
opens there, and the patch grows as the drop drains on.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

# the sphere's axis points down into the liquid, where its integrals' one
# singular point stays wet
drop = shapes.cap(3.0, 3.0, nr=8, nt=32)
ball = C.sphere((1.3, 0, 1.0), 0.8, contact_angle=30.0, axis=(-1, 0, -1),
                wet_poles=("north",))
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={"plate": C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                            "ball": ball},
               vertex_constraints={"plate": drop.rims["rim"]},
               bodies=[pyse.Body(volume=None)])
ev.body(1).target = ev.body(1).volume
ev.refine(1)
ev.relax()
steps = []
for s in recipes.arclength(ev, ev.body(1), -1.0, 60, topology=True):
    steps.append(s)
    if s.value < 10:
        break
changed = [s for s in steps if s.fell]
first = changed[0]
print(f"first dry patch at volume {first.value:.1f} (the drop full at {steps[0].value:.1f}), "
      f"the last change at {changed[-1].value:.0f}")
held = max(abs(s.volumes[0]/s.value - 1) for s in steps)
print(f"the volume held at every point {'to round-off' if held < 1e-12 else f'to {held:.0e}'}")
angles = ev.contact_angles("ball")
print(f"at volume {steps[-1].value:.0f} the contact angle along the ball's line is "
      f"{np.min(angles):.0f} to {np.max(angles):.0f} degrees (set: 30)")
```

```text
first dry patch at volume 31.5 (the drop full at 53.5), the last change at 14
the volume held at every point to round-off
at volume 10 the contact angle along the ball's line is 25 to 30 degrees (set: 30)
```

The first patch opens where the drop's surface comes within half an edge of the
sphere. The patch then grows in steps: each time the surface beside its contact line
comes to rest on the sphere, the patch takes that part in, and the line moves on.
Every point is an equilibrium at its volume. At the end, the contact angle along the
dry patch's edge is within the mesh's error of the 30 degrees set.

## One pore of a bead pack

The four beads here are a pore of a random pack: unit spheres from 0.03 to 0.2 apart,
at a contact angle of 20 degrees. The liquid fills the pore, its four menisci in the
four throats. As it drains, the menisci go back into the pore. Each bead's axis points
away from the pore, where the pore's liquid never goes.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes
from packing_helpers import _pore_surface, _on

centres = np.array([[-0.6123, 1.1001, 0.3757], [1.2879, 0.1557, 0.1317],
                    [-0.4315, -1.0784, 0.651], [-0.2441, -0.1774, -1.1584]])
v, f = _pore_surface(centres, 0.2, 0.1)     # the pore's faces, pushed out 0.2
beads = {f"bead{k + 1}": C.sphere(c, 1.0, contact_angle=20.0, axis=c)
         for k, c in enumerate(centres)}
ev = pyse.Evolver()
ev.load_arrays(v, f, constraints=beads,
               vertex_constraints={k: _on(v, c) for k, c in zip(beads, centres)},
               bodies=[pyse.Body(volume=1.0)])
ev.body(1).target = ev.body(1).volume
ev.relax()
steps = []
for s in recipes.arclength(ev, ev.body(1), -0.01, 150, topology=True):
    steps.append(s)
    if s.value < 0.2:
        break
k = np.array([-s.pressures[0] for s in steps])
V = np.array([s.value for s in steps])
jump = max((i for i, s in enumerate(steps) if s.fell), key=lambda i: k[i - 1] - k[i])
before = [s for s in steps[:jump] if s.fell]
dried = sorted({e.constraint for e in steps[jump].changes if e.kind == "solid"})
print(f"full: volume {V[0]:.2f}, curvature {k[0]:.1f}")
print(f"before the jump: {', '.join(sorted({e.constraint for s in before for e in s.changes}))} "
      f"drying from {before[0].value:.2f}, a patch at a time; curvature up to {k[:jump].max():.1f}")
print(f"at {V[jump]:.2f} the liquid jumps: curvature from {k[jump - 1]:.1f} to {k[jump]:.1f} "
      f"at the same volume, {', '.join(dried)} drying")
print(f"then on to {V[-1]:.2f}, curvature {k[-1]:.1f}")
```

```text
full: volume 0.68, curvature 4.6
before the jump: bead4 drying from 0.54, a patch at a time; curvature up to 4.8
at 0.43 the liquid jumps: curvature from 4.5 to 3.1 at the same volume, bead2, bead4 drying
then on to 0.19, curvature 5.7
```

```{figure} _static/drainage_pore.png
:alt: The liquid's curvature against its volume as the pore drains, with a drop
  from 4.5 to 3.1 at volume 0.43.

The curvature of the pore's liquid as it drains (right). Each red point is an
equilibrium reached after a change of topology. Drawn by
`tools/drainage_topology_figure.py`, as is the next picture.
```

First the menisci come down onto bead 4, and dry patches open on it one after the
other while the curvature stays near 4.5. At a volume of 0.43, the last liquid across
the pore's middle can no longer hold its place. With the volume fixed, the liquid
rearranges at once into the wedges between the beads, at a curvature of 3.1, drying
two beads where the strip was. This is the analogue of a Haines jump at a fixed
volume: the same rearrangement under a fixed pressure would be a sudden loss of
liquid. Past it, the curvature rises again as the wedges drain.

```{figure} _static/drainage_pore_shapes.png
:alt: The pore's liquid just before and just after the jump.

The liquid just before the jump (left) and just after it (right), at the same volume.
```

## What it means, and where it stops

- The liquid goes through the changes as a sequence of equilibria, each at the volume
  where the change came. The volume is held to round-off throughout, and the dry
  patches' contact angles come out as set, to the mesh's error.
- A change is made where the surface comes within half an edge of a solid (a film
  thinner than that is taken as gone), so the volumes where patches open depend on
  the mesh. Which patches open first, and how the liquid splits where it
  rearranges, follow the descent of the energy, not the liquid's flow.
- Drained further, this pore stops at a volume of about 0.16. Newton's steps
  converge only slowly in a strongly curved part of the remaining liquid, and the run
  can't step on. A finer mesh and a run down to the pendular rings are open.
- Sixteen beads in a box, its floor a wall, drain to breakthrough (the gas reaching
  the floor) through a few hundred changes: six packings at contact angles of 15, 20
  and 25 degrees, all eighteen at one thread and at two, in 2 to 16 minutes each at
  one thread (`bench/topology/corpus.py --set wide`). Their beads use the flux gauge
  (`constraints.sphere(gauge="flux")`), which has no singular point to move. Where a
  change or a descent fails on the way, the run backs up a few points and goes on
  from there another way (a shorter step, a fresh mesh).
- A run's work is bounded, in passes over the surface (an evaluation of the energy,
  forces or volumes on every facet is one; deterministic, unlike its time): past
  `budget=` (1.5 million by default; the packs' runs take 0.14 to 0.6 million) it
  ends with a warning, "the run's budget spent". A longer run, or a finer mesh's,
  passes a larger `budget` (or `None`). A descent that goes on past 100000 passes
  counts as failed, and the run backs up.
- **Liquid cut off is not yet held at its own volume.** A piece a cut leaves can be
  folded back into the body it shares the most solids with (in a pack, nearly always
  the draining one); a dry patch that cuts a region off doesn't make it a body of
  its own, so it drains with the rest; and a merge keeps the measured volume
  continuous by changing the body's constant, which hides what the mesh lost. So
  the liquid left behind (the residual saturation after breakthrough) and the
  volumes after the first cut-off are not to be trusted yet; the steps up to it,
  each an equilibrium at its volume, are.
