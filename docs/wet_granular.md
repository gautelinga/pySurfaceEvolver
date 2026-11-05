# Wet beads: bridges that grow and merge

A little liquid holds sand together. It sits in **pendular bridges** at the contacts
between the grains, and the bridges' tension and low pressure pull the grains to each
other. Add liquid and the bridges grow until their contact lines meet on a grain: they
merge, first into small clusters around the narrowest places (a **trimer** fills the
throat between three grains), then into filled pores and clusters spanning many grains
(the **funicular** regime). Scheel et al. imaged these shapes in wet glass bead packs and
computed them with Surface Evolver.

Here the liquid grows from a bridge at each contact, with
{func}`~pysurfaceevolver.recipes.arclength` and `topology=True` making the merges as
they come (as on the {doc}`drainage_topology` page, the other way round). All the
liquid is one body at one pressure, `trap=False`: separate bridges exchange liquid, as
they do through thin films on the grains or through the vapour when the liquid is
given time, and the run steps their total volume. The force each grain feels is
{meth}`~pysurfaceevolver.Evolver.wall_force`. Beads have radius 1 and the liquid's
surface tension is 1, so a pressure is minus a mean curvature and a force is in units
of tension times radius. The starting bridges come from `bridges` in
`docs/packing_helpers.py`.

## The force of a bridge

The liquid pulls on a wall along its contact lines, with its surface tension in the
direction Young's law gives the surface there, and pushes on the part it wets with
its pressure. Only the contact lines enter: the wetted part's area vector is half the
loop integral of x × dl round them. Through any cross-section of a bridge between two
beads the same force passes, the tension round the section and the pressure over it,
so at the neck it is 2πr − πr²p for a neck of radius r:

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes

c, a = 1.1, 0.35                              # spheres at z = -c and c, 0.2 apart
zb = c - np.sqrt(1 - a*a)
for theta in (20.0, 60.0):
    tube = shapes.tube(a, -zb, zb, nz=4, nt=16)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   constraints={"lower": C.sphere((0, 0, -c), 1.0, contact_angle=theta,
                                                  wet_poles=("north",)),
                                "upper": C.sphere((0, 0, c), 1.0, contact_angle=theta,
                                                  wet_poles=("south",))},
                   vertex_constraints={"lower": tube.rims["bottom"], "upper": tube.rims["top"]},
                   bodies=[pyse.Body(volume=0.3)])
    ev.relax(levels=3)
    ring = ev.mesh().slice((0, 0, 1))[0][:-1]          # the neck, at z = 0
    r = np.hypot(ring[:, 0], ring[:, 1]).mean()
    p = ev.body(1).pressure
    F = ev.wall_force("lower")
    print(f"{theta:g} degrees: force on the lower bead {F[2]:+.4f} along z, "
          f"through the neck {2*np.pi*r - np.pi*r*r*p:+.4f}")
```

```text
20 degrees: force on the lower bead +3.2060 along z, through the neck +3.2058
60 degrees: force on the lower bead +1.5566 along z, through the neck +1.5562
```

The lower bead is pulled up, toward the other. Taking the surface's direction at the
line from the facets next to it instead of from Young's law is first order in the
edge: 2.6% off on a bridge of 2000 facets, where this is 0.06%.

## Three beads: two bridges become a trimer

Beads 1 and 2 lie 0.17 apart, each 0.03 from bead 3, with a bridge at each of those two
contacts, one body:

```python
import numpy as np
from pysurfaceevolver import recipes
from packing_helpers import bridges

gap, contact = 0.17, 0.03                     # beads 1 and 2 apart, both near bead 3
a = 1 + gap/2
centres = np.array([[-a, 0, 0], [a, 0, 0], [0, np.sqrt((2 + contact)**2 - a*a), 0]])
ev = bridges(centres, [(0, 2), (1, 2)], theta=20.0)
print(f"two bridges, volume {ev.body(1).volume:.3f}, pressure {ev.body(1).pressure:.2f}")
last = None
for step in recipes.arclength(ev, ev.body(1), 0.004, 400, topology=True, trap=False,
                              remesh=0.06):
    if step.fell and last is not None:
        made = ", ".join(e.kind for e in step.changes) or "no change of topology"
        print(f"volume {step.value:.3f}: fell, pressure {last:.2f} -> {step.pressures[0]:.2f} "
              f"({made})")
    last = step.pressures[0]
    if step.value > 0.6:
        break
F = [ev.wall_force(f"bead{k}") for k in (1, 2, 3)]
print(f"volume {step.value:.3f}: pressure {step.pressures[0]:.2f}, beads 1 and 2 pulled "
      f"together by {F[0][0] - F[1][0]:.2f}, bead 3 toward them by {-F[2][1]:.2f}")
```

```text
two bridges, volume 0.040, pressure -8.12
volume 0.224: fell, pressure -1.81 -> -1.82 (contact lines)
volume 0.224: fell, pressure -1.82 -> -1.82 (no change of topology)
volume 0.224: fell, pressure -1.82 -> -2.42 (no change of topology)
volume 0.233: fell, pressure -3.39 -> -4.06 (no change of topology)
volume 0.604: pressure -1.69, beads 1 and 2 pulled together by 11.43, bead 3 toward them by 7.58
```

The two bridges grow together at one pressure until their contact lines meet on bead
3, at a volume of 0.224, where they join round it (`contact lines`): a **dimer**, the
liquid wrapped round bead 3 and touching beads 1 and 2. Its meniscus across the gap
between beads 1 and 2 then moves into the gap, and at 0.233 it snaps across: the throat
between the three beads fills, a trimer, its pressure dropping from -3.39 to -4.06. From
there it grows smoothly. Its pressures are those of the trimer the
[packing page](packing.md) builds directly in that throat (`trimer`, with these gaps):
-3.646 against -3.656 at a volume of 0.3, -2.799 against -2.804 at 0.4, -1.700 against
-1.703 at 0.6, on about 1100 facets against 28000. The run takes half a minute.

Beads 1 and 2, which no bridge joined at the start, are now pulled together by 11.4,
about twice what a single bridge between touching beads pulls with (2π cos 20° = 5.9,
the small-bridge limit): the trimer holds three beads at once.

## Four beads: a pore fills

Four beads at the corners of a tetrahedron, each 0.03 from the others, enclose a pore
with six contacts round it. With a bridge at each, all one body, the liquid grows from
a volume of 0.12 to 1.6 in steps of 0.004 (`tools/wet_granular_figure.py pore`, ten
minutes):

```{figure} _static/wet_pore.png
:alt: Two panels against the liquid's volume, from 0.1 to 1.6. Left, the pressure rises from -8 to -2.4 at a volume of 0.53, drops to -3.3, rises a little, drops to -5.9 at 0.57 and then rises smoothly to -1. Right, the pull on each bead into the pore rises from 8.9 to 10.2, jumps to 11 and then to 13.9 at the same volumes, and falls to 8.4 as the pore overfills.

The pressure and the pull on each bead toward the pore's centre against the volume;
the points are states after a fall.
```

```{figure} _static/wet_pore_shapes.png
:alt: Four views into the pore, through the three nearer beads at the fourth behind them. First, six thin rings, the bridges at the contacts. Second, the three nearer bridges grown into a Y-shaped sheet across the near face, the three behind still loops round the far bead. Third, the same volume after the fall, the loops drawn in round the far bead, the pore filled. Last, a filled pore bulging out between the beads.

Six bridges (volume 0.12); joined, before the pore fills (0.57); filled, after the fall
at the same volume; overfilled (1.6). Seen from outside the face of three beads, the
fourth behind them.
```

The six bridges grow at one pressure until, at a volume of 0.53, their contact lines
meet on the beads and join (eight changes in one fall): the liquid is one body round
the pore, and its pressure drops from -2.4 to -3.3. At 0.571 the liquid closes over the
pore and fills it (eighteen changes in one fall), the pressure dropping to -5.9 and the pull on each bead from 11
to 13.9: a filled pore holds its beads together harder than the bridges did, at the
same volume. Then it grows smoothly, overfilling the pore, the pressure rising to -1
and the pull falling to 8.4 as the liquid bulges out between the beads. Its pressures
are those of the pore the [packing page](packing.md) fills directly (`filled_pore`,
0.03 apart): -4.490 against -4.498 at a volume of 0.7, -2.578 against -2.582 at 1.0.
The four beads' forces sum to under 0.6% of one bead's throughout: nothing else holds
the liquid.

## Notes and limits

* **One pressure.** `trap=False` keeps every piece of the liquid in the run's body,
  so separate bridges exchange liquid freely and only the total is stepped. That is
  the slow limit, where thin films or the vapour have time to even out the pressure.
  With the default, a piece cut off keeps the volume it holds, as in drainage.
* **Gaps, not contacts.** Touching beads would pinch a bridge's neck to a point, so the
  beads here stay 0.03 apart, as the packs on the {doc}`drainage_topology` page do.
  Where liquid wraps a near contact, the mesh can't resolve the meniscus in the gap,
  and a contact line leaves one bead for the other along a single edge;
  `wall_force` closes the line along the chord between where it leaves and where it
  comes back. Where a line runs along another wall instead (a drop in a corner), only
  the line's pull along the wall acts there.
* **Starting bridges.** `bridges` starts each one as a tube about the line of centres.
  In a pack, start them only at the pack's contacts: at the low pressure small bridges
  have, one across a wider gap holds next to nothing, and it pinched off as the start
  relaxed. Wider gaps fill when the liquid reaches them.
* **Forces.** `wall_force` is the force on a wall from its contact lines and the
  pressure on its wetted part, the gas at pressure 0. It takes the surface's direction
  at the line from Young's law, so it holds at equilibrium; off it (in a fall), use the
  energy's derivatives instead.

## References

* M. Scheel, R. Seemann, M. Brinkmann, M. Di Michiel, A. Sheppard, B. Breidenbach and
  S. Herminghaus, Morphological clues to wet granular pile stability, *Nature Materials*
  7 (2008) 189–193, [doi:10.1038/nmat2117](https://doi.org/10.1038/nmat2117).
  Bridges, trimers and filled pores in tomographies of wet bead packs.
* C. Semprebon, M. Scheel, S. Herminghaus, R. Seemann and M. Brinkmann, Liquid
  morphologies and capillary forces between three spherical beads, *Phys. Rev. E* 94
  (2016) 012907, [doi:10.1103/PhysRevE.94.012907](https://doi.org/10.1103/PhysRevE.94.012907).
* S. Herminghaus, Dynamics of wet granular matter, *Advances in Physics* 54 (2005)
  221–261, [doi:10.1080/00018730500167855](https://doi.org/10.1080/00018730500167855).
* C. D. Willett, M. J. Adams, S. A. Johnson and J. P. K. Seville, Capillary bridges
  between two spherical bodies, *Langmuir* 16 (2000) 9396–9405,
  [doi:10.1021/la000657y](https://doi.org/10.1021/la000657y).
