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
first dry patch at volume 31.5 (the drop full at 53.5), the last change at 32
the volume held at every point to round-off
at volume 10 the contact angle along the ball's line is 27 to 29 degrees (set: 30)
```

The patch opens where the drop's surface comes within half an edge of the sphere: the
one change of topology in the run. From then on the patch grows as its contact line
moves back over the sphere, with no further change. Every point is an equilibrium at
its volume. At the end, the contact angle along the
dry patch's edge is within the mesh's error of the 30 degrees set.

<video src="_static/drainage_drop.mp4" controls loop muted width="100%" title="The drop draining over the sphere, a dry patch opening and growing"></video>

The run as a movie, one frame a point (`tools/drainage_topology_figure.py drop`): the
dry patch (the sphere showing through) opens at volume 31.5 and grows as the drop
drains.

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
print(f"past a fold at {V[:jump].min():.2f} the liquid jumps: curvature from {k[jump - 1]:.1f} "
      f"to {k[jump]:.1f}, {', '.join(dried)} drying, {steps[jump].trapped:.2f} of it cut off")
print(f"the rest drains on from {V[jump]:.2f} to {V[-1]:.2f}, curvature {k[-1]:.1f}")
```

```text
full: volume 0.68, curvature 4.6
before the jump: bead4 drying from 0.54, a patch at a time; curvature up to 4.8
past a fold at 0.43 the liquid jumps: curvature from 4.4 to 3.1, bead4 drying, 0.07 of it cut off
the rest drains on from 0.37 to 0.20, curvature 5.1
```

```{figure} _static/drainage_pore.png
:alt: The draining liquid's curvature against its volume as the pore drains: a
  fold at volume 0.43, then a drop to 3.1 at 0.37, where a pocket is cut off.

The curvature of the pore's draining liquid as it drains (right). Each red point is
an equilibrium reached after a change of topology. At the jump the line runs from
0.44 to 0.37: the volume is the same, but 0.07 of it is cut off and no longer counts
as draining. Drawn by
`tools/drainage_topology_figure.py`, as is the next picture.
```

First the menisci come down onto bead 4, and dry patches open on it one after the
other while the curvature stays near 4.5. At a volume of 0.43 the branch reaches a
fold: the last liquid across the pore's middle can no longer hold its place. The run
falls from one point past the fold, at the same volume. The liquid rearranges at
once into the wedges between the beads, at a curvature of 3.1, drying bead 4 where
the strip was, and a pocket of 0.07 is cut off from the rest. It keeps its volume
from then on (see the next section), and the rest drains on. This is the analogue of
a Haines jump at a fixed volume: the same rearrangement under a fixed pressure would
be a sudden loss of liquid. Past it, the curvature rises again as the wedges drain.

```{figure} _static/drainage_pore_shapes.png
:alt: The pore's liquid just before and just after the jump.

The liquid just before the jump (left) and just after it (right), at the same volume,
the pocket cut off orange.
```

<video src="_static/drainage_pore.mp4" controls loop muted width="100%" title="The pore's liquid draining, bead 4 drying patch by patch, then the jump"></video>

The whole run, one frame a point: the patches opening on bead 4, the jump into the
wedges with the pocket cut off, and the wedges draining.

## Liquid left behind: a ring cut off

As liquid drains out of a pack, some of it is left behind: rings round the points
where beads nearly touch, and pockets the gas has gone round. That liquid is cut off
from the liquid still draining, and it keeps its volume. With an outlet,
`arclength(..., topology=True, outlet="bottom")` drains only the liquid that reaches
that wall (the floor here). After every change it finds the liquid's clusters and
makes each one cut off a body of its own, held at the volume it had. `Step.trapped`
is their volume.

Two unit beads stand one on the other, 0.03 apart, in a box of side 3, under a flat
layer of liquid, at a contact angle of 20 degrees on the beads. The liquid drains
through the floor. Its surface comes down past the upper bead and leaves a ring
between the two.

```python
from pysurfaceevolver import recipes
from ring_helpers import two_beads, at_floor

ev = two_beads(theta=20.0)          # the liquid over two stacked beads, the floor "bottom"
steps = []
for s in recipes.arclength(ev, ev.body(1), -0.5, 200, topology=True, outlet="bottom"):
    steps.append(s)
    if at_floor(ev, s):
        break
cut = next(i for i, s in enumerate(steps) if s.trapped > 0)
ring = steps[cut].trapped
held = max(abs(s.trapped/ring - 1) for s in steps[cut:])
print(f"the liquid starts at volume {steps[0].value:.2f}; at {steps[cut].value:.2f} "
      f"the ring between the beads is cut off, holding {ring:.4f}")
print(f"over the next {len(steps) - cut - 1} steps the draining liquid goes down to "
      f"{steps[-1].value:.2f}; the ring holds its volume to {held:.0e}")
end = steps[-1]
print(f"at the end the curvature of the draining liquid is {-end.pressures[0]:.2f}, "
      f"the ring's {-end.pressures[1]:.2f}")
```

```text
the liquid starts at volume 28.07; at 9.56 the ring between the beads is cut off, holding 0.2828
over the next 14 steps the draining liquid goes down to 2.60; the ring holds its volume to 3e-15
at the end the curvature of the draining liquid is 1.00, the ring's 0.12
```

```{figure} _static/drainage_ring_shapes.png
:alt: Two stacked beads in a box: the liquid's surface coming down past the upper
  bead, the ring between the beads cut off (orange), and the liquid round the lower
  bead drained down to the floor.

Just before the cut (left), just after it (middle, the ring orange) and at the end
(right): the ring stays as it was while the liquid below drains to the floor.
```

```{figure} _static/drainage_ring.png
:alt: The volumes and curvatures of the draining liquid and the ring against the
  draining liquid's volume.

The draining liquid's volume and curvature (blue) and the ring's (orange), the cut
dotted. The ring's volume is fixed; its curvature settles from 0.15 to 0.12 in the
first steps after the cut and then stays, while the draining liquid's goes on
changing.
```

<video src="_static/drainage_ring.mp4" controls loop muted width="100%" title="The liquid draining past two stacked beads, the ring between them cut off and held"></video>

The run, one frame a point (`tools/drainage_topology_figure.py ring`).

The ring has its own pressure. Relaxed on its own, without the rest of the liquid,
it comes back to the same curvature (`tests/test_clusters.py` checks this to 1e-5).
Without `outlet=`, the run drains its largest cluster, here the same liquid.

## Sixteen beads to breakthrough

Sixteen unit beads settled in a box of side 5 (`bench/topology/pack.py`), at a
contact angle of 20 degrees, the floor the outlet, drain from full until the gas
reaches the floor (breakthrough). On the way there are a few hundred changes of
topology: menisci jumping through throats, dry patches opening on the beads, sheets
of liquid against the side walls rupturing, and rings and pockets cut off.

<video src="_static/drainage_pack.mp4" controls loop muted width="100%" title="A 16-bead pack draining to breakthrough, trapped clusters in orange"></video>

One run, at 150 equally spaced volumes of the liquid (`bench/topology/series.py
--seed 2 --theta 20 --frames 150`, about ten minutes at one thread; drawn by
`tools/drainage_topology_figure.py pack`): the draining liquid blue, each cluster cut
off orange. `series.py` writes the same frames as an XDMF time series for ParaView,
each facet's cluster with it.

```{figure} _static/drainage_pack_shapes.png
:alt: The 16-bead pack full, half drained and at breakthrough, with the trapped
  clusters in orange.

Full, 36% drained, and at breakthrough with 72% drained: 14 clusters cut off by
then (orange), rings where beads nearly touch and pockets between them.
```

```{figure} _static/drainage_packs.png
:alt: For 18 runs, the share of the liquid trapped against the share drained, and
  each run's shares at breakthrough.

The 18 runs of the wide set (six packings, three contact angles;
`bench/topology/corpus.py --set wide`): the share of the liquid cut off as each
drains (left), and where each ends at breakthrough (right). Up to 1% of the liquid
is trapped by then, in up to 17 clusters, after 44 to 78% of it has drained.

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
  and 25 degrees, all eighteen, in 1.5 to 17 minutes each at one thread
  (`bench/topology/corpus.py --set wide`; at two threads too, measured before cut-off
  liquid was held). Their beads use the flux gauge
  (`constraints.sphere(gauge="flux")`), which has no singular point to move. Where a
  change or a descent fails on the way, the run backs up a few points and goes on
  from there another way (a shorter step, a fresh mesh).
- A run's work is bounded, in passes over the surface (an evaluation of the energy,
  forces or volumes on every facet is one; deterministic, unlike its time): past
  `budget=` (1.5 million by default; the packs' runs take 0.09 to 0.74 million) it
  ends with a warning, "the run's budget spent". A longer run, or a finer mesh's,
  passes a larger `budget` (or `None`). A descent that goes on past 100000 passes
  counts as failed, and the run backs up.
- **Liquid cut off keeps its volume.** After every change the liquid's clusters
  (its surface's pieces, joined where the liquid between the walls joins them, found
  on a grid of the pore space at half the edge length) are made bodies. The liquid
  reaching `outlet=` (here `"bottom"`) is the run's body and drains; every other
  cluster is held at the volume it had when it was cut off and relaxed with the rest
  at each step, rejoining the run's body if it merges with it. `Step.trapped` is
  their volume. The run ends, with a warning, when no liquid reaches the outlet. A
  layer of liquid thinner than the grid doesn't join pieces, and where the surface
  touches walls other than planes, spheres and cylinders, the clusters are its
  surface pieces (with a warning).
