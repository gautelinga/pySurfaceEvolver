# Liquid in a sphere packing

In a packing of spheres, the space between them is a network of **pores**, each
enclosed by four neighbours, joined through **throats**, the windows between three.
A wetting liquid that drains from the packing or soaks into it moves from pore to pore
in jumps, and what decides each jump is the shape of a meniscus in a throat or a pore.
Pack-scale models (pore networks, cluster models) replace those shapes by simple
estimates. This page computes three of them in full with pySE, on spheres of radius 1,
and finds where each stops being an equilibrium:

* the curvature at which gas passes a throat,
* the liquid between three beads, and where it bursts,
* a pore full of liquid, and where it empties.

All three follow the liquid with {func}`~pysurfaceevolver.recipes.arclength` in its
volume, as on the [bridge page](bridge_rupture.md). The starting surfaces are built
from flat patches between the spheres in `docs/packing_helpers.py`. Each example takes
about a minute, and the contact angle is 20 degrees or more (see
[notes](#notes-and-limits)).

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
the 5 degrees of Semprebon et al. the picture differs: for gaps from about 0.125 up to
0.22, the trimer's outer meniscus in the gap snaps in first, at a fold of the branch,
and below 0.125 it bursts as here. Followed at 5 degrees (outside this page; see
[notes](#notes-and-limits)), the trimer branch turns at a fold at volume 0.186 for a gap
of 0.17 and 0.218 for 0.19, on the way to the paper's cusp, where the fold vanishes, at
volume 0.26 and gap 0.22. At 20 degrees the burst comes first at every gap followed,
from 0.09 to 0.25.

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
          " ".join(f"{e[i]:.2f}" for e in lowest))

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
  four lowest eigenvalues at volume 0.604: 24.82 24.83 24.84 46.00
  four lowest eigenvalues at volume 0.249: 2.18 2.23 2.29 48.01
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
{meth}`~pysurfaceevolver.Evolver.modes`). `arclength` stops just past the peak, where
its steps no longer settle.

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

* **Small contact angles.** At 5 degrees, relaxing these crude starts fails: gradient
  steps slide the vertices next to the contact line through the sphere, which the
  liquid meets almost tangentially. Newton steps don't, so `arclength` can follow a
  surface once it has relaxed. The 5-degree trimers quoted above started from slabs
  that happened to relax (gaps 0.17 and 0.19); at gaps of 0.21 and 0.25 they didn't.
* **Topology changes are where the runs stop.** A burst, a Haines jump or a pore
  emptying changes how the liquid is connected, which the mesh can't follow. The runs
  find where each happens; the state after it needs a new surface.
* **Mirror planes.** The throat is computed in a prism with mirror planes, so modes that
  break the layer's symmetry don't show. The trimer and the pore are whole surfaces.
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
