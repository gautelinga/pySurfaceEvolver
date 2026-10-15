# A liquid bridge between two disks

Liquid held between two equal, coaxial disks, its surface pinned at their sharp edges,
is the classic liquid bridge: the floating zone of crystal growth, and an experiment
flown many times in microgravity. Which bridges are stable depends on two numbers, the
**slenderness** Λ = H/(2R) (H the distance between the disks, R their radius) and the
**reduced volume** V/(πR²H), the liquid's volume over that of the cylinder between the
disks. A cylinder is stable up to Λ = π, Plateau's and Rayleigh's limit. Taking liquid
away from a bridge, it eventually pinches; adding liquid, it eventually bulges out to
one side. Gravity along the axis shrinks the stable range.

This page finds these limits with pySE for a few slendernesses, with and without
gravity. Each run starts from a cylinder's worth of liquid and follows the bridges with
{func}`~pysurfaceevolver.recipes.arclength` in the volume, as on the
[bridge page](bridge_rupture.md), reading the lowest eigenvalues on the way. An exact
calculation of the axisymmetric bridges and their stability checks every limit, and
the limits with gravity are compared with the table of Slobozhanin and Perales.

## The bridge

The disks have radius 1 and the surface tension is 1, the lower disk at z = 0 and the
upper one at z = 2Λ. `disk_bridge` in `docs/disk_bridge_helpers.py` builds a cylinder
between the disks' edges, which are round wires ({func}`~pysurfaceevolver.constraints.circle`
boundaries, so refining keeps them round; their vertices are fixed). The disks
themselves aren't part of the surface: the volume Evolver computes misses the slab of
liquid over the upper disk, and a content integral on that wire adds it back. With
gravity (along -z) the body weighs `bond` per unit volume, the Bond number
ρgR²/σ.

`to_limit(ev, slenderness, ds)` follows the bridge with `arclength` in its volume, in
steps of `ds` times the cylinder's volume (negative: taking liquid away), until
`result.negative_modes` says the bridge has turned unstable. At each point
{meth}`~pysurfaceevolver.Evolver.modes` gives the lowest eigenvalue, with the volume
held. Where the run turned back at the same time, the limit is a fold, the least (or
greatest) volume a bridge of this kind can hold; otherwise it is where the lowest
eigenvalue crossed zero, interpolated between the points. The unstable mode says how
the bridge goes: *pinching* (a neck narrows, the same all round and mirror-symmetric
about the mid-plane), *to one disk* (liquid moves from one half to the other),
*axisymmetric* (the same all round but neither of those, as with gravity) or
*sideways* (the bridge moves off the axis).

The surface is refined once and switched to the quadratic model, where facets are
curved: with 1000 to 4000 facets the limits come out within a few tenths of a percent.
`limits(slenderness, bond)` is the exact reference: it follows the axisymmetric bridges
of the Young-Laplace equation from the cylinder's volume, and computes for each the
lowest eigenvalues of the second variation of the energy, with the edges pinned, for
axisymmetric displacements (the volume held) and for those that go once around the
axis (cos θ).

## Taking liquid away

```python
import warnings
import pysurfaceevolver as pyse
from disk_bridge_helpers import disk_bridge, limits, to_limit

warnings.simplefilter("ignore", pyse.EvolverWarning)   # Evolver's notes past the limits
for slenderness in (0.5, 1.5, 2.5):
    ev = disk_bridge(slenderness)          # a cylinder's volume of liquid
    ev.relax(levels=1)
    ev.command("quadratic")
    volume, how = to_limit(ev, slenderness, -0.04)
    exact, exact_how = limits(slenderness, which=["less"])["less"]
    print(f"slenderness {slenderness}: stable down to volume {volume:.4f} ({how}), "
          f"exact {exact:.4f} ({exact_how}, {volume/exact - 1:+.2%})")
```

```text
slenderness 0.5: stable down to volume 0.2538 (pinching), exact 0.2544 (pinching, -0.25%)
slenderness 1.5: stable down to volume 0.4751 (pinching), exact 0.4751 (pinching, -0.00%)
slenderness 2.5: stable down to volume 0.6894 (to one disk), exact 0.6895 (to one disk, -0.01%)
```

Short and middling bridges hold their shape down to a fold: a neck forms in the middle
and, below the least volume, no bridge of that shape exists. This is the *minimum
volume stability limit*, which Gillette and Dyson computed for axisymmetric
disturbances. A long bridge goes another way: at
Λ = 2.5 it turns unstable at 0.6895, while the mirror-symmetric bridges carry on to a
fold at 0.6638. The mode moves liquid toward one disk, as the
[bridge between spheres](bridge_rupture.md) does: the bridge leaves the mirror-symmetric
family at a branch point. The switch from one kind to the other is at Λ ≈ 2.13
(Myshkis et al., quoted by Slobozhanin and Perales), and at Λ = π both limits reach the
cylinder.

## Adding liquid

```python
import warnings
import pysurfaceevolver as pyse
from disk_bridge_helpers import disk_bridge, limits, to_limit

warnings.simplefilter("ignore", pyse.EvolverWarning)
for slenderness in (0.5, 1.5, 2.5):
    ev = disk_bridge(slenderness)
    ev.relax(levels=1)
    ev.command("quadratic")
    volume, how = to_limit(ev, slenderness, 0.15)
    exact, exact_how = limits(slenderness, which=["more"])["more"]
    print(f"slenderness {slenderness}: stable up to volume {volume:.4f} ({how}), "
          f"exact {exact:.4f} ({exact_how}, {volume/exact - 1:+.2%})")
```

```text
slenderness 0.5: stable up to volume 1.8661 (sideways), exact 1.8603 (sideways, +0.31%)
slenderness 1.5: stable up to volume 4.2223 (sideways), exact 4.2217 (sideways, +0.02%)
slenderness 2.5: stable up to volume 7.6283 (sideways), exact 7.6281 (sideways, +0.00%)
```

A bulging bridge goes sideways: its mode shifts the bulge off the axis, the same in
any direction, so its eigenvalue is a double one (`negative_modes` goes from 0 to 2).
Russo and Steen showed that this happens where the surface meets the disks' edges
tangent to the disks' planes, and measured it (the exact bridges at these three limits
meet the edges at 90.000 degrees to the axis). The axisymmetric family carries on past
it, unstable. A long bridge holds a lot more liquid than a short one before it goes:
at Λ = 2.5 nearly eight times a cylinder's.

## With gravity

With gravity along the axis the bridges lose their mirror symmetry: the liquid sags
toward the lower disk. Slobozhanin and Perales tabulate, for each Bond number, the
largest volume any stable bridge can hold and the slenderness at which it does; the
runs below take that slenderness, add liquid until the bridge goes, and compare with
their table and with the exact reference.

```python
import warnings
import pysurfaceevolver as pyse
from disk_bridge_helpers import disk_bridge, limits, to_limit

warnings.simplefilter("ignore", pyse.EvolverWarning)
table = {0.2: (1.96, 3.865), 0.5: (1.29, 2.663), 1.0: (0.93, 2.124)}  # Bond: slenderness, volume
for bond, (slenderness, theirs) in table.items():
    ev = disk_bridge(slenderness, bond=bond)
    ev.relax(levels=1)
    ev.command("quadratic")
    volume, how = to_limit(ev, slenderness, 0.15)
    exact, exact_how = limits(slenderness, bond, which=["more"])["more"]
    print(f"Bond {bond}, slenderness {slenderness}: up to volume {volume:.4f} ({how}), "
          f"exact {exact:.4f} ({volume/exact - 1:+.2%}), Slobozhanin and Perales {theirs}")
```

```text
Bond 0.2, slenderness 1.96: up to volume 3.8650 (sideways), exact 3.8648 (+0.00%), Slobozhanin and Perales 3.865
Bond 0.5, slenderness 1.29: up to volume 2.6629 (sideways), exact 2.6625 (+0.02%), Slobozhanin and Perales 2.663
Bond 1.0, slenderness 0.93: up to volume 2.1247 (sideways), exact 2.1237 (+0.05%), Slobozhanin and Perales 2.124
```

The reference agrees with the table to its last digit, and pySE with both within 0.1%.
Gravity cuts the largest volume sharply: from nearly 4 to about 2 cylinders' worth
between Bond numbers 0.2 and 1. Taking liquid away from the same bridges:

```python
import warnings
import pysurfaceevolver as pyse
from disk_bridge_helpers import disk_bridge, limits, to_limit

warnings.simplefilter("ignore", pyse.EvolverWarning)
for bond, slenderness in ((0.2, 1.96), (0.5, 1.29), (1.0, 0.93)):
    ev = disk_bridge(slenderness, bond=bond)
    ev.relax(levels=1)
    ev.command("quadratic")
    volume, how = to_limit(ev, slenderness, -0.04)
    exact, exact_how = limits(slenderness, bond, which=["less"])["less"]
    weightless, _ = limits(slenderness, which=["less"])["less"]
    print(f"Bond {bond}, slenderness {slenderness}: down to volume {volume:.4f} ({how}), "
          f"exact {exact:.4f} ({volume/exact - 1:+.2%}); without gravity {weightless:.4f}")
```

```text
Bond 0.2, slenderness 1.96: down to volume 0.7730 (axisymmetric), exact 0.7730 (-0.00%); without gravity 0.5588
Bond 0.5, slenderness 1.29: down to volume 0.4943 (axisymmetric), exact 0.4943 (-0.01%); without gravity 0.4384
Bond 1.0, slenderness 0.93: down to volume 0.4059 (axisymmetric), exact 0.4060 (-0.03%); without gravity 0.3737
```

Gravity raises the least volume: the liquid gathers in the lower half, and the neck
forms above the middle (at 0.64 to 0.72 of the height in the exact bridges). The limit
is still a fold, but without mirror symmetry its mode is neither even nor odd about the
mid-plane: the neck narrows while liquid moves from one half to the other.

## Pictures

```{figure} _static/disk_bridge.png
:alt: The stable region of slenderness and reduced volume, without gravity and for three Bond numbers, with pySE's limits on it
:width: 100%

The stable bridges lie between the curves of each colour: exact limits without
gravity (black) and at Bond numbers 0.2, 0.5 and 1, the circles pySE's runs on this
page, the crosses the largest volumes of Slobozhanin and Perales. Below each region
the bridges pinch (or, at the longest without gravity, move to one disk), above it
they go sideways. Each region narrows toward its greatest slenderness; the dotted
lines just close it at the last slenderness computed.
Drawn by `tools/disk_bridge_figure.py`.
```

```{figure} _static/disk_bridge_shapes.png
:alt: Four bridges at their limits, coloured by their unstable modes
:width: 100%

Bridges just past their limits, each at its own scale, coloured by the unstable mode's
displacement normal to the surface (red and blue opposite ways; the sign is arbitrary). From
the left: Λ = 1.5 pinching at its least volume; Λ = 1.5 going sideways
at its largest, the surface meeting the edges flat; Λ = 2.5 moving liquid to one disk;
Bond 0.5, Λ = 1.29 at its least volume, the neck above the middle.
```

## Notes and limits

* **Pinned edges.** The model holds the contact lines on the edges whatever the angle
  the surface makes there. Real edges hold them only while that angle stays within
  what the solid allows (Gibbs' condition); otherwise the liquid spreads over a disk's
  face or down its side. For short bridges with little liquid this decides the lower
  limit before the fold does (Slobozhanin and Perales). The wires here can't show the
  spreading, but the model marks the place: at Λ = 0.4, say, taking liquid away makes
  the surface meet the edges flat, turned in over the disks, before the fold, and there
  the exact bridge goes sideways. For a perfectly wetting liquid that is where it
  starts to spread over the disks; Slobozhanin and Perales note that the two limits
  coincide. The figure's curves start at Λ = 0.45 for that reason.
* **How the runs are set up.** `to_limit` follows the bridges without spreading the
  vertices or flipping edges between points (`arclength(tidy=False)`): in the
  quadratic model the tidying stalled the runs short of the fold. It lets a step be cut
  in half six times (`halvings=6`; the default 3 wasn't enough at some folds of the
  linear model). And where a point lands right at a fold (the lowest eigenvalue near
  zero), `arclength` can find no way on from it; `to_limit` then starts again two points
  back with shorter steps.
* **Accuracy.** In the linear model the same meshes give the limits within about 1%
  (the folds of short bridges, where the neck is sharp, least well); the quadratic
  model does better than refining once more.
* **Large Bond numbers.** Above a Bond number of about 3, Slobozhanin and Perales find
  that taking liquid away also ends sideways. The runs here stay below that.

## References

* R. D. Gillette and D. C. Dyson, Stability of fluid interfaces of revolution between
  equal solid circular plates, *Chem. Eng. J.* 2 (1971) 44-54,
  [doi:10.1016/0300-9467(71)87006-5](https://doi.org/10.1016/0300-9467(71)87006-5).
  The stable region for axisymmetric disturbances without gravity.
* M. J. Russo and P. H. Steen, Instability of rotund capillary bridges to general
  disturbances: experiment and theory, *J. Colloid Interface Sci.* 113 (1986) 154-163,
  [doi:10.1016/0021-9797(86)90215-8](https://doi.org/10.1016/0021-9797(86)90215-8).
  The largest volume, lost to sideways disturbances, computed and measured.
* S. R. Coriell, S. C. Hardy and M. R. Cordes, Stability of liquid zones, *J. Colloid
  Interface Sci.* 60 (1977) 126-136,
  [doi:10.1016/0021-9797(77)90263-6](https://doi.org/10.1016/0021-9797(77)90263-6).
  Bridges of a cylinder's volume with gravity along the axis.
* L. A. Slobozhanin and J. M. Perales, Stability of liquid bridges between equal disks
  in an axial gravity field, *Phys. Fluids A* 5 (1993) 1305-1314,
  [doi:10.1063/1.858567](https://doi.org/10.1063/1.858567). The whole stable region
  for Bond numbers from 0.005 to 7, with the tables used above.
* A. D. Myshkis, V. G. Babskii, N. D. Kopachevskii, L. A. Slobozhanin and
  A. D. Tyuptsov, *Low-Gravity Fluid Mechanics*, Springer, Berlin (1987). The theory
  behind the limits, including the switch at Λ ≈ 2.13.
