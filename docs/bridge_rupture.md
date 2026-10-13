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
for step in recipes.arclength(ev, ev.body(1), -0.003, 60):
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
print(f"least volume {V[f]:.4f}")

# the unstable mode of the first unstable bridge
ev.restore(first_unstable)
mode = ev.modes(1)
x, d = ev.vertices, mode.vectors[0]
phi = np.arctan2(x[:, 1], x[:, 0])
outward = d[:, 0]*np.cos(phi) + d[:, 1]*np.sin(phi)       # the radial displacement
heights = (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25)
moved = [outward[abs(x[:, 2] - z) < 0.05].mean() for z in heights]
print(f"its eigenvalue {mode.values[0]:.2f}; radial displacement at z =",
      " ".join(f"{z:+.2f}" for z in heights))
print(" " * 36, " ".join(f"{m:+.2f}" for m in moved))
```

```text
stable down to volume 0.0792 (neck 0.1315), one negative mode from 0.0775 (neck 0.1266)
least volume 0.0730
its eigenvalue -0.74; radial displacement at z = -0.25 -0.15 -0.05 +0.05 +0.15 +0.25
                                     +0.91 +0.81 +0.38 -0.39 -0.81 -0.91
```

The bridge turns unstable between volumes 0.0792 and 0.0775, while it still has a
volume to lose before the fold at 0.0730. Past the fold a second negative mode
appears, the fold's own. The run takes about 20 seconds on two threads.

```{figure} _static/bridge_rupture_shapes.png
:alt: The bridge at four points of the run: at the start, at the first unstable point, at the fold, and past the fold
:width: 100%

The bridge along the run: at the start (volume 0.15), at its first unstable point,
at the fold (the least volume), and past it on the thinner-necked branch, where the
fold's own mode has made a second one.
```

{meth}`~pysurfaceevolver.Evolver.modes` gives the lowest eigenvalues and their modes,
the motions of the vertices with the volume held. The unstable mode is odd about the
mid-plane: the bridge widens on one side of it and narrows on the other, so liquid
moves toward one sphere. Which sphere is chance: the mode's sign is arbitrary
(`modes()` fixes one, so the output above is reproducible).

```{figure} _static/bridge_rupture_mode.png
:alt: The first unstable bridge coloured by its unstable mode, and the bridge pushed along the mode
:width: 80%

Left, the first unstable bridge coloured by how its surface moves along the mode
(scaled to 1 at the most): in at one end, out at the other, nothing at the
mid-plane. Right, the bridge pushed along the mode (a root mean square displacement
of 0.03), over the bridge before (grey).
```

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
the run above (0.0792) and the first unstable one (0.0775). The fold is at 0.07307,
where the run's least volume is 0.0730. On a mesh refined once more (32768 facets),
the onset lies between 0.0790 and 0.0773 and the least volume is 0.07308.

```{figure} _static/bridge_rupture.png
:alt: The symmetric bridge's branch with Evolver's points, the pitchfork, the energy barrier and two bridge shapes
:width: 100%

Top left, the neck against the volume: the symmetric bridges (stable down to the
branch point, then unstable through the fold and back), and the run above coloured by
its number of negative modes. Top right, the height of the neck: the off-centre bridges
branch off at the branch point and exist only at larger volumes. Bottom left, how much
more energy an off-centre bridge has than the symmetric one of the same volume. Bottom
right, a section through two bridges of volume 0.098: the stable symmetric one and the
off-centre one. Drawn by `tools/bridge_rupture_figure.py`, as are the other
pictures on this page.
```

```{figure} _static/bridge_rupture_saddle.png
:alt: The stable symmetric bridge and the off-centre saddle of the same volume, in 3D
:width: 80%

The two bridges of the bottom-right section, as surfaces: the stable symmetric one,
and the off-centre one with its neck 0.1 above the mid-plane, a saddle between the
symmetric bridge and breaking.
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
  the break itself (the surface would change topology). `relax()` from the bridge pushed
  along its mode makes slow progress, because the mode's eigenvalue is barely below
  zero that close to the onset.
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
