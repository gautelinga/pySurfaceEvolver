# The largest drop a ring can hold

A drop hanging from the end of a tube, or from a hole in a plate, has its contact
line pinned on the rim. Feed it liquid and it grows, hangs lower and forms a neck,
until no equilibrium holds it and it falls. Tate's law puts the largest weight at
the rim's tension, 2πRγ for a rim of radius R. The static limit is more subtle,
and it was worked out in the 1970s from the axisymmetric Young-Laplace equation.
With the volume controlled, a drop loses stability in one of two ways:

* at its **largest volume**, a fold of the family of drops, where it stretches and
  falls straight down (Padday and Pitt, Pitts);
* on a wide ring, earlier, by **tipping sideways**: once the ring is wider than
  3.219 capillary lengths, the drop goes unstable before its largest volume, at the
  first drop whose surface meets the ring horizontally (Michael and Williams).

This page finds both with pySE from a general mesh, by following the drops with
{func}`~pysurfaceevolver.recipes.arclength` in their volume and counting their
unstable modes, and compares them with the axisymmetric references in
`docs/pendant_drop_helpers.py`. The drops are pinned on a circular wire of radius R,
with surface tension and density 1 and gravity along -z. The **Bond number**
ρgR²/γ is the one parameter. Results are in units of the ring's radius: volumes in
R³, lengths in R, pressures in γ/R.

## Growing a drop to its largest

`hanging_drop` starts the drop as the spherical cap of the given volume below the
ring, the shape it would have without gravity, then refines it twice and relaxes it.
Each run starts at half the volume of Tate's law (2π/Bond) and adds liquid along the
branch until the volume has fallen for three points: past the fold. `fold` puts a
parabola through the points around the largest volume, in the pressure, which is
smooth through a fold where the volume turns.

```python
import numpy as np
from pysurfaceevolver import recipes
from pendant_drop_helpers import RING, fold, hanging_drop, largest_drop, sideways

R = RING                                           # the ring's radius on the mesh
for bond in (1, 2, 4, 8):
    ev = hanging_drop(bond, np.pi/bond)            # half the volume of Tate's law
    rows = []
    for step in recipes.arclength(ev, ev.body(1), 0.01*R, 300):
        # in units of R: the volume, the pressure at the ring, the drop's height
        rows.append((step.value/R**3, step.pressures[0]*R, -ev.vertices[:, 2].min()/R))
        if len(rows) > 3 and rows[-1][0] < rows[-2][0] < rows[-3][0]:   # past the fold
            break
    volume, pressure, height = fold(rows)
    exact = largest_drop(bond)
    mode = ev.modes(1).vectors[0]                  # the lowest mode, past the fold
    print(f"Bond {bond}: largest volume {volume:.4f} (exact {exact['volume']:.4f}, "
          f"{volume/exact['volume'] - 1:+.2%}), height {height:.3f} ({exact['height']:.3f}), "
          f"pressure {pressure:.3f} ({exact['pressure']:.3f})")
    print(f"   weight {bond*volume/(2*np.pi):.3f} times 2 pi R gamma; past the fold "
          f"{step.result.negative_modes} unstable mode, "
          + ("sideways" if sideways(mode) > 0.05 else "round the axis"))
```

```text
Bond 1: largest volume 5.2719 (exact 5.2618, +0.19%), height 2.520 (2.512), pressure 0.131 (0.129)
   weight 0.839 times 2 pi R gamma; past the fold 1 unstable mode, round the axis
Bond 2: largest volume 2.9130 (exact 2.9065, +0.22%), height 1.861 (1.854), pressure -0.305 (-0.306)
   weight 0.927 times 2 pi R gamma; past the fold 1 unstable mode, round the axis
Bond 4: largest volume 1.6094 (exact 1.6058, +0.23%), height 1.313 (1.307), pressure -0.984 (-0.985)
   weight 1.025 times 2 pi R gamma; past the fold 1 unstable mode, round the axis
Bond 8: largest volume 0.8043 (exact 0.8018, +0.31%), height 0.832 (0.835), pressure -1.717 (-1.730)
   weight 1.024 times 2 pi R gamma; past the fold 1 unstable mode, round the axis
```

The largest volumes are 0.2 to 0.3% above the exact ones, on meshes of 5632 facets,
and the error falls three- to fourfold with each refinement (at Bond 2: 0.68%,
0.22% and 0.06% on 1408, 5632 and 22528 facets). Every drop is stable up to its
fold and has one unstable mode past it, which moves nothing sideways (`sideways`: the vertices'
average horizontal displacement, for a mode of root mean square 1). That is the
axisymmetric limit point: the drop stretches downward, and with the volume held
there is no equilibrium to stretch to. The four runs take about 45 seconds on two
threads.

The weight of the largest drop is not 2πRγ. Its vertical force balance is

$$\rho g V = 2\pi R\gamma \sin\varphi - p\,\pi R^2,$$

with φ the angle at which the surface meets the ring and p the pressure in the
liquid at the ring's plane (above the outside's). On a narrow ring, the largest
drop pulls on the ring at an angle and pushes on the plane of the ring with a
positive pressure, so it weighs less than 2πRγ (0.84 of it at Bond 1). On a wide
ring the pressure at the ring is negative: the liquid above holds the drop up by
suction, and the drop weighs more than the rim's tension can carry (at Bond 8 the
surface meets the ring only 9 degrees from the horizontal, and the suction carries
85% of the weight). A drop that grows from a tube is fed at some pressure, which is
part of why Tate's law needs correction factors (Lecacheux et al. measure this).

## Wide rings: tipping sideways

On a wide ring a drop is broad and shallow, and its surface can come back to
meet the ring horizontally. Michael and Williams showed that this is where the drop
turns unstable to tipping: without the ring, a horizontal shift of the drop is a
symmetry (gravity is vertical), so it costs no energy, and the shift moves the
surface along its normal by an amount that vanishes on the ring exactly when the
surface meets the ring horizontally. There it is an allowed motion of the pinned
drop at zero cost, and beyond it a motion of this kind (one wave around the ring)
lowers the energy. `sideways_onset` finds the first such drop on the exact family.

The runs here track the three lowest eigenvalues with
{meth}`~pysurfaceevolver.Evolver.modes` at each point, from 35% of Tate's volume,
and stop at the first unstable point. The onset is where the lowest eigenvalue
crosses zero, between the last two points.

```python
import numpy as np
from pysurfaceevolver import recipes
from pendant_drop_helpers import RING, hanging_drop, largest_drop, sideways, sideways_onset

R = RING
for bond in (11, 12, 13):
    ev = hanging_drop(bond, 0.7*np.pi/bond)
    last = None
    for step in recipes.arclength(ev, ev.body(1), 0.004*R, 300):
        modes = ev.modes(3)
        here = (step.value/R**3, modes.values[0])
        if step.result.negative_modes:
            break
        last = here
    (v0, e0), (v1, e1) = last, here
    onset = v0 + (v1 - v0)*e0/(e0 - e1)           # where the lowest eigenvalue crosses 0
    exact = sideways_onset(bond)
    print(f"Bond {bond}: {step.result.negative_modes} unstable modes from volume {onset:.4f} "
          f"(exact {exact['volume']:.4f}, {onset/exact['volume'] - 1:+.2%}), "
          f"largest drop {largest_drop(bond)['volume']:.4f}")
    print("   the lowest three modes: "
          + ", ".join("sideways" if sideways(u) > 0.05 else "round the axis" for u in modes.vectors))
```

```text
Bond 11: 2 unstable modes from volume 0.5175 (exact 0.5153, +0.43%), largest drop 0.5182
   the lowest three modes: sideways, sideways, round the axis
Bond 12: 2 unstable modes from volume 0.4298 (exact 0.4275, +0.55%), largest drop 0.4472
   the lowest three modes: sideways, sideways, round the axis
Bond 13: 2 unstable modes from volume 0.3318 (exact 0.3293, +0.76%), largest drop 0.3841
   the lowest three modes: sideways, sideways, round the axis
```

Two modes go unstable together, a pair with the same eigenvalue: tipping toward x
and toward y. Both move the drop sideways, while the third, the axisymmetric mode of
the fold, is still stable. The onsets are within 0.4 to 0.8% of the exact ones, and
come before the largest drop: only just at Bond 11, well before it at Bond 13. The
runs take about 35 seconds on two threads.

Where tipping starts to come first follows from the exact family alone: it is the
Bond number at which the largest drop meets the ring horizontally.

```python
from scipy.optimize import brentq
from pendant_drop_helpers import largest_drop

bond = brentq(lambda b: largest_drop(b)["angle"], 9.0, 12.0, xtol=1e-6)
print(f"tipping comes first above Bond {bond:.3f}: a ring {bond**0.5:.3f} capillary lengths wide")
```

```text
tipping comes first above Bond 10.363: a ring 3.219 capillary lengths wide
```

This is Michael and Williams's 3.219; at Bond 8, in the previous section, the fold
still came first. Above a ring radius of 3.832 capillary lengths (Bond
14.68; 3.832 is the first zero of the Bessel function J₁) even a flat meniscus
tips: liquid held up in a hole that wide runs out on one side, the instability of
a heavy liquid on top of a light one that Plateau and Maxwell described.

```{figure} _static/pendant_drop.png
:alt: Profiles of the largest drops at four Bond numbers against the exact profiles; the largest volume and the tipping onset against the Bond number; a drop at the tipping onset and pushed along its mode
:width: 100%

Left, the largest drops at Bond 1, 2, 4 and 8: pySE's vertices (dots, distance from
the axis against height) on the exact profiles (lines). Middle, the largest volume
against the Bond number (black) and, past Bond 10.363, the volume where tipping
starts (red; dotted, the largest volume there); dots, pySE. Right, the drop at Bond
12 just past the tipping onset (grey) and pushed along its first unstable mode
(red, exaggerated): their sections through the axis, in the plane the drop tips in. Drawn by `tools/pendant_drop_figure.py`.
```

## Notes and limits

* **Why the mesh's ring has radius 1/4.** `arclength` measures a step by the change
  of the value (here the volume) and the vertices' root mean square displacement
  together, in the surface's own units. On a ring of radius 1 the volumes are a few
  units while the shape changes by a few hundredths a step, so in that measure the
  branch turns almost at a right angle at the fold. After a failed step `arclength`
  predicts along the branch's tangent, which turns it there with steps up to about
  2% of Tate's volume; with 3% it still stopped at Bond 1. On a ring of radius 1/4 a
  volume is 64 times smaller and a displacement 4 times, and the fold is round
  enough to turn with the steps used here. Results are scaled back to the ring's
  radius; the Bond number is unchanged (gravity Bond/R²).
* **Pinned on a wire.** The ring is a circular wire (`C.circle`): its vertices are
  fixed, and refining puts the new ones on the circle, so the ring stays round.
* **The pressure** that pySE reports for a body under gravity is the one at z = 0,
  the ring's plane, so it is the curvature of the surface at the ring.
* **The volume of a drop on a wire** is the volume between the surface and the
  plane of the ring, which is what Evolver's facet integrals give with the ring in
  the plane z = 0.
* **Toward Bond 14.68** the onset volume falls fast with the Bond number, and the
  coarse mesh's error in the tipping eigenvalue matters more: at Bond 14 the onset
  came out 0.6 to 1.8% high, depending on the run, against 0.4 to 0.8% at
  Bond 11 to 13.
* **What happens after** is beyond a mesh of fixed topology: a drop past its fold
  pinches off, and a tipping one runs off to one side of the ring.

## References

* J. F. Padday and A. R. Pitt, The stability of axisymmetric menisci, *Phil. Trans.
  R. Soc. Lond. A* 275 (1973) 489-528,
  [doi:10.1098/rsta.1973.0113](https://doi.org/10.1098/rsta.1973.0113). Stability
  limits of axisymmetric menisci, pendant drops among them.
* E. Pitts, The stability of a drop hanging from a tube, *IMA J. Appl. Math.* 17
  (1976) 387-397, [doi:10.1093/imamat/17.3.387](https://doi.org/10.1093/imamat/17.3.387).
  For axisymmetric disturbances, a drop held at constant volume loses stability at
  its largest volume (at constant pressure head, at its largest pressure).
* D. H. Michael and P. G. Williams, The equilibrium and stability of axisymmetric
  pendent drops, *Proc. R. Soc. Lond. A* 351 (1976) 117-127,
  [doi:10.1098/rspa.1976.0132](https://doi.org/10.1098/rspa.1976.0132). The tipping
  mode (one wave around the ring) sets in where the profile meets the ring
  horizontally; it comes first on rings wider than 3.219 capillary lengths, and
  higher waves never come first. (For the flat interface the abstract gives 3.812;
  the first zero of J₁, which the criterion gives there, is 3.832.)
* D. H. Michael, Meniscus stability, *Annu. Rev. Fluid Mech.* 13 (1981) 189-216,
  [doi:10.1146/annurev.fl.13.010181.001201](https://doi.org/10.1146/annurev.fl.13.010181.001201).
  A review of these results.
* X. Lin, L. E. Johns and R. Narayanan, Static stability of pendent drops pinned to
  arbitrary closed curves, *Phys. Rev. Fluids* 2 (2017) 113605,
  [doi:10.1103/PhysRevFluids.2.113605](https://doi.org/10.1103/PhysRevFluids.2.113605).
  Volume and pressure control compared, for rims of any shape.
* L. Lecacheux, A. Sadoudi, A. Duri, V. Planchot and T. Ruiz, The role of Laplace
  pressure in the maximal weight of pendant drops, *J. Colloid Interface Sci.* 606
  (2022) 920-928, [doi:10.1016/j.jcis.2021.08.047](https://doi.org/10.1016/j.jcis.2021.08.047).
  Measured largest drops weigh less than 2πRγ; the pressure at the needle makes up
  the force balance.
