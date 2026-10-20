# Liquid in a sphere packing

In a packing of spheres, the space between them is a network of **pores**, each
enclosed by four neighbours, joined through **throats**, the windows between three.
A wetting liquid that drains from the packing or soaks into it moves from pore to pore
in jumps, and what decides each jump is the shape of a meniscus in a throat or a pore.
Pack-scale models (pore networks, cluster models) replace those shapes by simple
estimates. This page computes three of them in full with pySE, on spheres of radius 1,
and finds where each stops being an equilibrium:

* the curvature at which gas passes a throat,
* the liquid between three beads, where it bursts, and at a small contact angle,
  where it snaps into another shape and back,
* a pore full of liquid, where it empties, and through which throat.

All three follow the liquid with {func}`~pysurfaceevolver.recipes.arclength` in its
volume, as on the [bridge page](bridge_rupture.md). The starting surfaces are built
from flat patches between the spheres in `docs/packing_helpers.py`. Each example takes
a minute or two.

## Gas passing a throat

Take a planar layer of touching spheres, hexagonally packed, with liquid below it and
gas above. Three neighbouring spheres leave a throat between them, and the prism over
their centres repeats across the layer: its sides are mirror planes, and each sphere
is a 60-degree wedge of it (`C.sphere(..., span=pi/3)`). `throat_layer` starts from a
flat meniscus above the throat and relaxes it. Taking liquid away pulls the meniscus
down into the throat and curves it more, until it passes the narrowest section; past
that, less curvature holds it. The largest curvature on the way is the **entry
curvature**: at a higher pressure difference the gas goes through (a Haines jump),
at a lower one it doesn't.

```python
from pysurfaceevolver import recipes
from packing_helpers import msp_throat, throat_layer

for theta in (20.0, 40.0, 60.0):
    ev = throat_layer(theta)
    entry = 0.0
    for step in recipes.arclength(ev, ev.body(1), -0.004, 200):
        entry = max(entry, -step.pressures[0])     # the curvature (surface tension 1)
        if ev.vertices[:, 2].min() < -0.6:         # well past the throat
            break
    msp = msp_throat(theta)
    print(f"contact angle {theta:.0f}: entry curvature {entry:.3f}, "
          f"Mayer-Stowe-Princen {msp:.3f} ({entry/msp - 1:+.1%})")
```

```text
contact angle 20: entry curvature 10.651, Mayer-Stowe-Princen 10.898 (-2.3%)
contact angle 40: entry curvature 9.392, Mayer-Stowe-Princen 9.464 (-0.8%)
contact angle 60: entry curvature 7.269, Mayer-Stowe-Princen 6.869 (+5.8%)
```

The reference is the Mayer-Stowe-Princen (MSP) method that pore-network models use. It
treats the throat as a channel of constant cross-section, its narrowest one, and
balances the energy of a meniscus advancing along it. The full meniscus needs a little
less pressure than that at small contact angles and more at large ones: its curvature
depends less on the contact angle than MSP's does. Prodanović and Bryant found 3D
throats from a random packing mostly below MSP, for a perfectly wetting liquid.
Hilden and Trumble computed this layer with Surface Evolver.

## Three beads: where a trimer bursts

Two beads both touching a third, a gap apart from each other, can hold liquid in
several ways, among them separate bridges at the contacts, two bridges merged around
the third bead (a **dimer**), or all three merged into one body filling the throat
between them (a **trimer**). Semprebon et al. mapped where each is stable. `trimer` builds the third
kind from a flat slab around the throat. Draining it with `arclength`, the two menisci
that close the throat from above and below come closer, until they meet: the trimer
**bursts** open, most likely into separate bridges. `throat_thickness` measures the
liquid between them.

```python
from pysurfaceevolver import recipes
from packing_helpers import throat_thickness, trimer

for gap in (0.09, 0.17, 0.25):
    ev, centres = trimer(gap, theta=20.0, volume=0.45)
    for step in recipes.arclength(ev, ev.body(1), -0.005, 300):
        if throat_thickness(ev, centres) < 0.02 or step.result.negative_modes:
            break
    print(f"gap {gap}: the menisci in the throat meet at volume {step.value:.3f}, "
          f"pressure {step.pressures[0]:.2f}, {step.result.negative_modes} unstable modes")
```

```text
gap 0.09: the menisci in the throat meet at volume 0.085, pressure -8.06, 0 unstable modes
gap 0.17: the menisci in the throat meet at volume 0.190, pressure -4.72, 0 unstable modes
gap 0.25: the menisci in the throat meet at volume 0.323, pressure -2.83, 0 unstable modes
```

The trimer stays stable to the end, and bursts with more liquid the wider the gap. At
20 degrees the burst comes first at every gap followed, from 0.09 to 0.25. At the 5
degrees of Semprebon et al. the picture differs (next section).

## Five degrees: snap-in, pop-out and the dimer

At a contact angle of 5 degrees and gaps from about 0.125 to 0.22, Semprebon et al.
found that the trimer ends differently. Its outer meniscus, the one spanning the gap
between beads 1 and 2, jumps in toward bead 3 (a **snap-in**), and the liquid is left
as a dimer. Filled again, the dimer jumps back out into a trimer (a **pop-out**), at a
larger volume. In between, both hold the same volume. They tell the two apart by the
**meniscus distance** d, from bead 3 along the symmetry axis to that outer meniscus
(`meniscus_distance`): large for a trimer, small for a dimer.

`trimer` builds the start at 5 degrees directly. Relaxed at that angle outright, the
crude start would sink into the beads, so {meth}`~pysurfaceevolver.Evolver.relax`
starts again with the angle raised to 20 degrees and lowers it in stages. `arclength`
then follows the branch through both folds. With `remesh=` it keeps the edges in range
as the outer meniscus sweeps across the gap, merging contact-line edges too where its
contact lines close in.

```python
import numpy as np
from pysurfaceevolver import recipes
from packing_helpers import meniscus_distance, trimer

ev, centres = trimer(0.17, theta=5.0, volume=0.45)
h = float(np.median(ev.mesh().edge_lengths()))
rows = []
for step in recipes.arclength(ev, ev.body(1), -0.005, 400,
                              remesh=dict(target=h, keep_boundary=False)):
    rows.append((step.value, meniscus_distance(ev, centres), step.result.negative_modes))
    if rows[-1][1] < 0.06:
        break
V, d, unstable = np.array(rows).T
snap, pop = [k for k in range(1, len(V) - 1) if (V[k] - V[k - 1])*(V[k + 1] - V[k]) < 0]
print(f"trimer: stable down to volume {V[snap]:.3f} (d = {d[snap]:.2f}), the snap-in")
print(f"between: up to {unstable[snap:pop].max():.0f} negative mode, back up to volume "
      f"{V[pop]:.3f} (d = {d[pop]:.2f}), the pop-out")
print(f"dimer: from there down to volume {V[-1]:.3f} (d = {d[-1]:.2f}), "
      f"{unstable[pop + 1:].max():.0f} negative modes")
```

```text
trimer: stable down to volume 0.185 (d = 0.65), the snap-in
between: up to 1 negative mode, back up to volume 0.224 (d = 0.27), the pop-out
dimer: from there down to volume 0.206 (d = 0.11), 0 negative modes
```

Along the way the run passes two folds. The trimer is stable down to volume 0.185;
there its branch turns, and the surface the run follows back up is unstable, with one
negative mode. At volume 0.224 it turns again into stable dimers, which go on down to
volume 0.206. There `arclength` stops by itself: the outer meniscus has come down onto
bead 3, the film of liquid under it 6e-5 thick (an event on the step, see
{meth}`~pysurfaceevolver.Evolver.events`), where the two bridges come apart. Followed on
regardless, the meniscus goes into the bead (below volume 0.203; before events stopped
the run it went on to 0.196 that way).
Between 0.185 and 0.224, a trimer and a dimer hold the same liquid. A draining trimer
snaps in at 0.185, and a dimer being filled pops out at 0.224.

The same run at other gaps (`tools/packing_figure.py cusp`, below) gives the volumes of
the two folds:

| gap | snap-in | pop-out |
|---|---|---|
| 0.13 | 0.109 | 0.200 |
| 0.14 | 0.130 | 0.206 |
| 0.15 | 0.150 | 0.212 |
| 0.17 | 0.185 | 0.224 |
| 0.19 | 0.218 | 0.238 |
| 0.20 | 0.233 | 0.245 |
| 0.21 | 0.248 | 0.253 |
| 0.22 | 0.261 | 0.262 |
| 0.225 | 0.2667 | 0.2668 |

The window narrows as the gap widens and closes between gaps 0.225 and 0.23, at volume
0.267. Above that a trimer turns into a dimer smoothly, without a jump (at gap 0.25, d
falls from 0.82 to 0.06 with no fold). Semprebon et al. put this cusp at volume 0.26 and
gap 0.22.

Toward smaller gaps the window widens, and the liquid in the throat thins on the way
down to the snap-in; the runs at gaps 0.13 to 0.15 in the table start from a finer mesh
(`trimer(..., h=0.035)`), with about two edges across the throat at the snap-in
otherwise. At a gap of 0.11 the trimer bursts instead: the menisci in its throat meet
at volume 0.093, before any fold (d still 0.84). So the burst gives way to the snap-in
between gaps 0.11 and 0.13, near volume 0.1; Semprebon et al. put this kink at gap
0.125 and volume 0.09. (`throat_thickness` means the liquid between the throat's two menisci
only while the outer meniscus is outside the triangle of the bead centres; past the
snap-in it isn't, and its own points near z = 0 would count.)

```{figure} _static/packing_cusp.png
:alt: The meniscus distance against the liquid volume at four gaps, with the unstable parts dashed, and the snap-in and pop-out volumes against the gap meeting near Semprebon et al.'s cusp
:width: 100%

Left, the meniscus distance against the volume at 5 degrees: S-shaped below the cusp
(dashed, the unstable part between the folds), smooth above it. Right, the volumes at
which the trimer snaps in and the dimer pops out, and below the kink bursts, by gap,
with Semprebon et al.'s cusp and kink.
```

```{figure} _static/packing_dimer.png
:alt: A trimer and a dimer between three beads at gap 0.17, holding about the same volume
:width: 80%

A trimer and a dimer at gap 0.17 holding about the same liquid (volume 0.204 and
0.205), from the run above. The trimer spans the gap between the two lower beads; the
dimer has left it.
```

## A pore full of liquid

Four spheres, 0.02 apart at the corners of a tetrahedron, enclose a pore with four
throats. `filled_pore` builds the liquid filling it, from the tetrahedron of their
centres with its faces pushed out. Its four menisci sit in the throats, and between
the spheres six bridges join them. Draining it, all four menisci are pulled into their
throats together. The pore empties when one of them passes: the curvature there should
be the throats' entry curvature, so the same throat is computed in a layer too.

```python
import numpy as np
from pysurfaceevolver import recipes
from packing_helpers import filled_pore, msp_throat, throat_layer

ev, centres = filled_pore(gap=0.02, theta=20.0)
rows = []
for step in recipes.arclength(ev, ev.body(1), -0.01, 100):
    rows.append((step.value, -step.pressures[0], *ev.modes(4).values))
    if step.result.negative_modes:
        break
V, curvature, *lowest = np.array(rows).T
k = int(np.argmax(curvature))
print(f"full at volume {V[0]:.3f}; the curvature peaks at {curvature[k]:.3f} (volume {V[k]:.3f})")
for i in (0, k):
    print(f"  four lowest eigenvalues at volume {V[i]:.3f}:",
          " ".join(f"{e[i]:.1f}" for e in lowest))

ev = throat_layer(20.0, gap=0.02)
entry = 0.0
for step in recipes.arclength(ev, ev.body(1), -0.004, 200):
    entry = max(entry, -step.pressures[0])
    if ev.vertices[:, 2].min() < -0.6:
        break
print(f"the same throat in a layer: entry curvature {entry:.3f} "
      f"(Mayer-Stowe-Princen {msp_throat(20.0, 0.02):.3f})")
```

```text
full at volume 0.604; the curvature peaks at 9.886 (volume 0.249)
  four lowest eigenvalues at volume 0.604: 24.8 24.8 24.8 46.0
  four lowest eigenvalues at volume 0.249: 2.2 2.2 2.3 48.0
the same throat in a layer: entry curvature 9.871 (Mayer-Stowe-Princen 10.131)
```

The pore gives up its liquid at the curvature at which its throats let gas in, to
0.2%. Each meniscus alone, past its throat's narrowest section, holds with less
curvature, so as the pressure peaks the four menisci compete: liquid moving from one of
them into the others would lower the energy. The lowest three eigenvalues are equal, by
the tetrahedron's symmetry, and fall toward zero together as the pressure peaks (on a
mesh refined once more, the first turns negative just past the peak, at volume 0.240);
the fourth stays stiff. Which throat goes first is chance in a symmetric pore, so the
three modes come out as any combination of the three (see
{meth}`~pysurfaceevolver.Evolver.modes`). Past the peak `arclength` takes ever shorter
steps and stops just before the first of them turns negative. So close to zero, Newton
converges only from very near, nearer than tidying the mesh moves the vertices.

### Which throat goes first

The mesh spoils the symmetry a little, and splits the three modes (0.17, 0.77 and 1.08
just before the run stops), so it can't show which way the pore goes. A symmetry the
mesh keeps exactly does. Take one throat, the bottom one, against the other three: the
pore then has a three-fold axis through it and the sphere opposite, with three mirror
planes along it. `pore_wedge` relaxes the full pore with that axis along z and cuts it
down to the sixth between two of the mirror planes. That holds a 60-degree piece of the
top sphere and half of one lower sphere; the mirrors stand in for the rest. `base` sets
the gaps between the three lower spheres, so the bottom throat can be wider or narrower
than the others. `bottom_meniscus` gives the height at which its meniscus crosses the
axis (the pore's centre is near 0).

```python
import numpy as np
from pysurfaceevolver import recipes
from packing_helpers import bottom_meniscus, pore_wedge

for base in (0.03, 0.01):
    ev, centres = pore_wedge(0.02, theta=20.0, base=base)
    rows = []
    for step in recipes.arclength(ev, ev.body(1), -0.002, 300):
        rows.append((6*step.value, -step.pressures[0], bottom_meniscus(ev)))
        if step.result.negative_modes or 6*step.value < 0.15:
            break
    V, curvature, z = np.array(rows).T
    k = int(np.argmax(curvature))
    print(f"bottom gaps {base}: the curvature peaks at {curvature[k]:.3f} (volume {V[k]:.3f}); "
          f"the bottom meniscus at {z[k]:+.3f} there, {z[-1]:+.3f} at the end")
    turns = np.flatnonzero(np.diff(V[k:]) > 0)       # a fold: the volume turns back up
    print("   " + (f"the branch turns at volume {V[k + turns[0]]:.4f}" if len(turns)
                   else "the branch doesn't turn")
          + "; " + ("a negative mode" if step.result.negative_modes else
                    f"the surface about to touch {step.events[0].constraint}")
          + f" at volume {V[-1]:.3f}")
```

```text
bottom gaps 0.03: the curvature peaks at 9.532 (volume 0.292); the bottom meniscus at -0.184 there, -0.091 at the end
   the branch turns at volume 0.2878; a negative mode at volume 0.291
bottom gaps 0.01: the curvature peaks at 10.002 (volume 0.253); the bottom meniscus at -0.294 there, -0.346 at the end
   the branch doesn't turn; the surface about to touch mirror_0 at volume 0.204
```

With the bottom throat the widest (gaps 0.03 against 0.02), the curvature peaks at
9.533. That is the entry curvature of a throat with gaps of 0.03: the layer of the first
section gives 9.520 for it. Past the peak, the bottom meniscus goes on into the pore
while the other three pull back, and the branch turns at a fold just below the peak's
volume. Draining further, there is no equilibrium nearby to follow: gas breaks through
the widest throat and the pore empties, a Haines jump. With the bottom throat the
narrowest (gaps 0.01), the other three go first, at a curvature of 10.002, and the
bottom meniscus pulls back out. The run stops at volume 0.204, where the surface comes
within 0.006 of one of the mirror planes with liquid between: in the whole pore, the
liquid there is about to pinch off against its mirror image (followed on regardless, the
run reaches a negative mode at 0.190). In the wedge those three can only move together;
whether one of them would go alone is what the mirrors leave out. Either way the pore
empties through its widest throat, at that throat's entry curvature. With all four
throats alike, the wedge's run goes on past the peak (9.877) onto the second kind of
branch without a fold: which way it turns is then up to the mesh.

## Pictures

```{figure} _static/packing.png
:alt: Curvature against volume for a throat at three contact angles, the throat thickness of a trimer at three gaps, and a full pore's curvature and lowest eigenvalues
:width: 100%

Left, the throat: the curvature as the meniscus is pulled through it, against the liquid
volume above the throat plane (dotted, MSP). Middle, the trimer: the liquid's thickness
in the throat until the menisci meet (x). Right, the full pore: the curvature (black;
dotted, the throat's entry curvature from the layer) and the four lowest eigenvalues
(red; the three lowest coincide, dashed the fourth). Drawn by `tools/packing_figure.py`.
```

```{figure} _static/packing_shapes.png
:alt: The meniscus at the throat's entry, the trimer as it bursts, and the full pore at the peak of its curvature
:width: 100%

Where each run stops. Left, the meniscus in one prism of the layer at the entry
curvature, seen from below. Middle, the trimer as its menisci in the throat meet.
Right, the pore at the peak: the six bridges between the spheres (the ribbons), joined
in the middle, with the four menisci drawn deep into the throats.
```

## Notes and limits

* **Small contact angles.** At 5 degrees, relaxing these crude starts outright slides
  the vertices next to the contact line through the beads, which the liquid meets
  almost tangentially; `relax()` then starts again with the angle raised to 20 degrees
  and lowers it in stages. Newton steps don't slide vertices that way, so `arclength`
  follows a surface once it has relaxed.
* **Topology changes are where the runs stop.** A burst, a Haines jump or a pore
  emptying changes how the liquid is connected, which the mesh can't follow. The runs
  find where each happens, and `arclength` stops by itself where the surface is about
  to touch a bead, a mirror or itself (`Step.events`); the state after it needs a new
  surface.
* **Mirror planes.** The throat is computed in a prism with mirror planes, so modes that
  break the layer's symmetry don't show; likewise the pore's wedge, which keeps only
  shapes with its three-fold symmetry. The trimer and the full pore are whole surfaces.
* **The pore's start** pushes the tetrahedron's faces out by 0.3, so a face crosses the
  gap between two spheres along a strip about 0.1 wide. With touching spheres the
  strip vanishes and the start pinches: a gap is needed.

## References

* C. Semprebon, M. Scheel, S. Herminghaus, R. Seemann and M. Brinkmann, Liquid
  morphologies and capillary forces between three spherical beads, *Phys. Rev. E* 94
  (2016) 012907, [doi:10.1103/PhysRevE.94.012907](https://doi.org/10.1103/PhysRevE.94.012907).
  Trimers, dimers and their stability limits at 5 degrees, with Surface Evolver.
* M. Scheel, R. Seemann, M. Brinkmann, M. Di Michiel, A. Sheppard, B. Breidenbach and
  S. Herminghaus, Morphological clues to wet granular pile stability, *Nature Materials*
  7 (2008) 189–193, [doi:10.1038/nmat2117](https://doi.org/10.1038/nmat2117). Bridges,
  trimers and filled tetrahedra in tomographies of wet bead packs, computed with
  Surface Evolver.
* K. Melnikov, R. Mani, F. K. Wittel, M. Thielmann and H. J. Herrmann, Grain-scale
  modeling of arbitrary fluid saturation in random packings, *Phys. Rev. E* 92 (2015)
  022206, [doi:10.1103/PhysRevE.92.022206](https://doi.org/10.1103/PhysRevE.92.022206).
  A packing-scale model of bridges, clusters and filled pores with simplified menisci.
* J. L. Hilden and K. P. Trumble, Numerical analysis of capillarity in packed spheres:
  planar hexagonal-packed spheres, *J. Colloid Interface Sci.* 267 (2003) 463–474,
  [doi:10.1016/S0021-9797(03)00683-0](https://doi.org/10.1016/S0021-9797(03)00683-0).
* M. Prodanović and S. Bryant, A level set method for determining critical
  curvatures for drainage and imbibition, *J. Colloid Interface Sci.* 304 (2006)
  442–458, [doi:10.1016/j.jcis.2006.08.048](https://doi.org/10.1016/j.jcis.2006.08.048).
  Throats of a random packing against MSP.
* R. Mayer and R. Stowe, Mercury porosimetry: breakthrough pressure for
  penetration between packed spheres, *J. Colloid Sci.* 20 (1965) 893–911,
  [doi:10.1016/0095-8522(65)90061-9](https://doi.org/10.1016/0095-8522(65)90061-9),
  and H. M. Princen, Capillary phenomena in assemblies of parallel cylinders,
  *J. Colloid Interface Sci.* 30 (1969) 359–371,
  [doi:10.1016/0021-9797(69)90403-2](https://doi.org/10.1016/0021-9797(69)90403-2).
  The MSP method.
* G. Mason and N. Morrow, Meniscus displacement curvatures of a perfectly wetting
  liquid in capillary pore throats formed by spheres, *J. Colloid Interface Sci.* 109
  (1986) 46–56, [doi:10.1016/0021-9797(86)90280-8](https://doi.org/10.1016/0021-9797(86)90280-8).
  Measured entry curvatures of throats between touching balls.
