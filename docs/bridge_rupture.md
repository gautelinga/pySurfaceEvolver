# Where a shrinking bridge breaks

A liquid bridge between two solid spheres that slowly loses liquid, say by
evaporation, breaks at some point. The usual answer to "at what volume?" comes from
an axisymmetric calculation: solve the Young-Laplace equation for bridges that are
round about the axis and mirror-symmetric about the plane halfway between the
spheres, and find the least volume such a bridge can hold (a fold: below it, no
symmetric bridge exists).

This page shows that the bridge becomes unstable before that, about 8% above the fold,
with the liquid moving toward one of the spheres. pySE finds this from a general
mesh by following the bridges with {func}`~pysurfaceevolver.recipes.arclength` and
counting their unstable modes. An axisymmetric calculation that drops the mirror
symmetry checks it. The mechanism is known: it is the "small volume instability"
of T. I. Vogel (see [References](#references)).

The bridge here is between unit spheres with a gap of 0.6 and a contact angle of 40
degrees. The stress page's [liquid bridge](stress_cases.md#a-liquid-bridge-as-the-gap-closes)
is the same one at a fixed volume of 0.05, with the gap closing.

## Shrinking through the fold

{func}`~pysurfaceevolver.recipes.continuation` can't do this. It relaxes the bridge at
each volume, which stops at the fold (below it there is nothing to relax to), and
relaxing aims for stable bridges anyway. `arclength` takes steps of a
fixed length along the branch of equilibria, in the volume and the vertices' positions
together, and finds each point by Newton steps, so it turns at the fold. At each point
`result.negative_modes` counts the motions that lower the energy at that volume (with
the constraints held): 0 for a stable bridge.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

gap, theta = 0.6, 40.0
c = 1 + gap/2                                 # the spheres' centres at z = -c and c
a = 0.35
zb = c - np.sqrt(1 - a*a)                     # a cylinder of radius a, its ends on the spheres
tube = shapes.tube(a, -zb, zb, nz=4, nt=16)
ev = pyse.Evolver()
ev.load_arrays(tube.vertices, tube.faces,
               constraints={1: C.sphere((0, 0, -c), 1.0, contact_angle=theta, wet_poles=("north",)),
                            2: C.sphere((0, 0, c), 1.0, contact_angle=theta, wet_poles=("south",))},
               vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
               bodies=[pyse.Body(volume=0.15)])
ev.relax(levels=3)

def neck(ev):
    """The radius of the bridge at the mid-plane."""
    ring = ev.mesh().slice((0, 0, 1))[0][:-1]
    return float(np.hypot(ring[:, 0], ring[:, 1]).mean())

# shrink the bridge along its branch, through the fold, counting unstable modes
rows, first_unstable = [], None
for step in recipes.arclength(ev, ev.body(1), -0.0075, 40):
    modes = step.result.negative_modes
    rows.append((step.value, neck(ev), modes))
    if modes == 1 and first_unstable is None:
        first_unstable = ev.save()
    if modes == 2:
        break
V, necks, modes = np.array(rows).T
k, f = np.flatnonzero(modes == 1)[0], np.argmin(V)
print(f"stable down to volume {V[k-1]:.4f} (neck {necks[k-1]:.4f}), "
      f"one negative mode from {V[k]:.4f} (neck {necks[k]:.4f})")
print(f"least volume {V[f]:.4f} (neck {necks[f]:.3f}), then two negative modes")

# the unstable mode: step along it from the first unstable bridge
ev.restore(first_unstable)
x = ev.mesh().vertices
ev.saddle()
dx = ev.mesh().vertices - x
phi = np.arctan2(x[:, 1], x[:, 0])
outward = dx[:, 0]*np.cos(phi) + dx[:, 1]*np.sin(phi)       # the radial displacement
heights = (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25)
moved = [outward[abs(x[:, 2] - z) < 0.05].mean() for z in heights]
print("radial displacement at z =", " ".join(f"{z:+.2f}" for z in heights))
print("  (scaled to 1 at the top): ", " ".join(f"{m/moved[-1]:+.2f}" for m in moved))
```

```text
stable down to volume 0.0795 (neck 0.1325), one negative mode from 0.0773 (neck 0.1261)
least volume 0.0730 (neck 0.101), then two negative modes
radial displacement at z = -0.25 -0.15 -0.05 +0.05 +0.15 +0.25
  (scaled to 1 at the top):  -1.06 -0.99 -0.52 +0.45 +0.93 +1.00
```

The bridge turns unstable between volumes 0.0795 and 0.0773, while it still has a
volume to lose before the fold at 0.0730. Past the fold a second negative mode
appears, the fold's own. {meth}`~pysurfaceevolver.Evolver.saddle` steps along the
first unstable mode, and that mode is odd about the mid-plane: the bridge narrows on
one side of it and widens on the other, so liquid moves toward one sphere. Which
sphere is chance (the mode's sign is arbitrary). The run takes about half a minute on
two threads.

## The reference

The axisymmetric reference for bridges that aren't mirror-symmetric puts the neck,
the narrowest circle, at a height `h` above the mid-plane. It integrates the profile
from there out to each sphere, and solves for the neck's radius and the pressure that
give the contact angle on both. At `h = 0` that is the symmetric bridge. The helpers
are in `docs/stress_helpers.py`:

* `bridge_spheres_fold` finds the least volume of the symmetric bridges.
* `bridge_spheres_branch` finds the symmetric bridge where moving the neck along the
  axis keeps both contact angles to first order. That is where the off-centre bridges
  branch off.
* `bridge_spheres_offset` finds the off-centre bridges.

```python
from stress_helpers import (bridge_spheres, bridge_spheres_branch, bridge_spheres_fold,
                            bridge_spheres_offset)

gap, theta = 0.6, 40.0
fold = bridge_spheres_fold(gap, theta)
branch = bridge_spheres_branch(gap, theta)
print(f"symmetric bridges hold at least volume {fold['volume']:.5f} (neck {fold['neck']:.4f})")
print(f"off-centre bridges branch off at volume {branch['volume']:.5f} (neck {branch['neck']:.4f})")

guess = (branch["neck"], branch["pressure"])
for height in (0.02, 0.05, 0.1):
    off = bridge_spheres_offset(height, gap, theta, guess=guess)
    guess = (off["neck"], off["pressure"])
    # the stable symmetric bridge holding the same volume
    sym = bridge_spheres(off["volume"], gap, theta, guess=(branch["neck"], branch["pressure"]))
    print(f"neck {height:.2f} above the mid-plane: volume {off['volume']:.4f}, "
          f"energy {off['energy']:.5f}, {off['energy'] - sym['energy']:.2e} above the symmetric bridge's")
```

```text
symmetric bridges hold at least volume 0.07307 (neck 0.1018)
off-centre bridges branch off at volume 0.07895 (neck 0.1312)
neck 0.02 above the mid-plane: volume 0.0796, energy 0.54605, 7.57e-06 above the symmetric bridge's
neck 0.05 above the mid-plane: volume 0.0831, energy 0.55780, 2.92e-04 above the symmetric bridge's
neck 0.10 above the mid-plane: volume 0.0978, energy 0.60638, 4.81e-03 above the symmetric bridge's
```

The off-centre bridges branch off at volume 0.07895, between the last stable point of
the run above (0.0795) and the first unstable one (0.0773). The fold is at 0.07307,
where the run's least volume is 0.0730. On a mesh refined once more (32768 facets),
the onset lies between 0.0808 and 0.0785 and the least volume is 0.07306.

```{figure} _static/bridge_rupture.png
:alt: The symmetric bridge's branch with Evolver's points, the pitchfork, the energy barrier and two bridge shapes
:width: 100%

Top left, the neck against the volume: the symmetric bridges (stable down to the
branch point, then unstable through the fold and back), and the run above coloured by
its number of negative modes. Top right, the height of the neck: the off-centre bridges
branch off at the branch point and exist only at larger volumes. Bottom left, how much
more energy an off-centre bridge has than the symmetric one of the same volume. Bottom
right, a section through two bridges of volume 0.098: the stable symmetric one and the
off-centre one. Drawn by `tools/bridge_rupture_figure.py`.
```

## What it means

* **The off-centre bridges are saddles.** They exist only above the branch volume,
  each with more energy than the symmetric bridge of the same volume. Between that
  stable bridge and breaking, they are the lowest pass. As the volume falls toward the
  branch point, the pass comes down to the symmetric bridge's energy, and at the branch
  point the symmetric bridge stops being a minimum. This shape (unstable branches
  bending back toward the stable side) is a *subcritical* pitchfork.
* **A slowly shrinking bridge breaks at the branch point** (volume 0.0790 here), not
  at the fold (0.0731), and it breaks lopsided. No equilibrium is nearby to settle
  into, so the bridge doesn't drift to an off-centre shape first. Evolver can't follow
  the break itself (the surface would change topology). `relax()` from the stepped-off
  bridge above makes slow progress, because the mode's eigenvalue is barely below zero
  that close to the onset.
* **Real bridges are likely to break earlier still.** With a subcritical pitchfork,
  a small asymmetry (unequal spheres or contact angles, gravity along the axis)
  typically makes the bridge break before the symmetric branch point. A supercritical
  one would make the neck drift steadily toward one sphere instead.
* **Which comes first depends on the geometry.** Vogel's examples include one where
  the instability comes first, as here (contact angle 45 degrees), and one where the
  fold does (9 degrees).

To find instabilities like this in a surface of your own, follow the branch with
`arclength` and watch `negative_modes` change. A symmetric reference alone can't see
an asymmetric mode.

## References

* T. I. Vogel, Liquid bridges between balls: the small volume instability,
  *J. Math. Fluid Mech.* 15 (2013) 397-413,
  [doi:10.1007/s00021-012-0117-y](https://doi.org/10.1007/s00021-012-0117-y). Bridges
  between equal balls with shrinking volume lose stability either at a pitchfork to
  rotationally symmetric bridges that aren't mirror-symmetric, or at the least volume,
  depending on the geometry.
* T. I. Vogel, Convex, rotationally symmetric liquid bridges between spheres,
  *Pacific J. Math.* 224 (2006) 367-377. The stability of convex bridges, by the same
  eigenvalue problem.
* B. Y. Rubinstein and L. G. Fel, Stability of unduloidal and nodoidal menisci between
  two solid spheres, *J. Geom. Symmetry Phys.* 39 (2015) 77-98. Stability regions
  under axisymmetric perturbations.
