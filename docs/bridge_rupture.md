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
    if modes == 0:
        last_stable = ev.save()
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

## Along the off-centre bridges

`arclength` can also follow the off-centre bridges, starting from the run's first
unstable bridge. With `along=`, its first step goes along the unstable mode at the same
volume, onto the branch that crosses the symmetric one there; from there it follows that
branch.

```python
# continuing from the two blocks above: back to the first unstable bridge,
# and off along its unstable mode
ev.restore(first_unstable)
rows = []
for step in recipes.arclength(ev, ev.body(1), 0.01, 12, along=mode.vectors[0]):
    rows.append((step.value, step.pressures[0], step.energy, step.result.negative_modes))

# the exact off-centre bridges, by the height of the neck
guess, exact = (branch["neck"], branch["pressure"]), []
for height in np.linspace(0.005, 0.2, 40):
    off = bridge_spheres_offset(height, gap, theta, guess=guess)
    guess = (off["neck"], off["pressure"])
    exact.append((off["volume"], off["pressure"], off["energy"]))
exact = np.array(exact)
for V, p, E, n in rows[1::2]:
    print(f"volume {V:.4f}: pressure {p:.4f} (exact {np.interp(V, exact[:, 0], exact[:, 1]):.4f}), "
          f"energy {E:.5f} ({np.interp(V, exact[:, 0], exact[:, 2]):.5f}), {n:.0f} negative mode")
```

```text
volume 0.0796: pressure 3.3414 (exact 3.3416), energy 0.54574 (0.54597), 1 negative mode
volume 0.0848: pressure 3.3209 (exact 3.3210), energy 0.56320 (0.56344), 1 negative mode
volume 0.0943: pressure 3.2857 (exact 3.2856), energy 0.59452 (0.59476), 1 negative mode
volume 0.1067: pressure 3.2430 (exact 3.2429), energy 0.63486 (0.63509), 1 negative mode
volume 0.1209: pressure 3.1978 (exact 3.1979), energy 0.68069 (0.68092), 1 negative mode
volume 0.1364: pressure 3.1527 (exact 3.1531), energy 0.72982 (0.73007), 1 negative mode
```

The run climbs the off-centre branch to larger volumes, as the reference says it
should, with one negative mode throughout: saddles. Its pressures agree with the exact
ones to 1e-4, and its energies are 0.04% low, which is the mesh's error. Taking the
first step the other way (`ds` of -0.01) gives the mirror image, with the neck below
the mid-plane. Start such a run just past the crossing, as here: the off-centre
bridges are then close by.

## Through the break

With `topology=True`, `arclength` goes on where the branch loses its stability, as the
bridge would if it lost its liquid slowly. At the first point with a negative mode it
falls, at that volume, by steps along the unstable mode, each followed by a few
gradient steps. When the neck comes (an {class}`~pysurfaceevolver.Event` of kind
`"neck"`), {meth}`~pysurfaceevolver.Evolver.pinch_off` cuts it: the two rows of
vertices round the neck collapse to a point, which is split into one vertex per side,
and `rebody` makes two bodies. The pieces then relax: two drops, one on each sphere, holding what
the bridge held. Each drop is a spherical cap meeting its sphere at 40 degrees, which
gives an exact pressure for its volume. The run then goes on draining the larger drop,
which keeps the body's number; the smaller keeps its volume.

```python
# continuing from the first block: back to the last stable bridge, and on
# through the break
def cap_pressure(volume, theta):
    """The pressure of a drop on a unit sphere: a sphere of radius r cutting it
    at contact angle theta (degrees)."""
    t = np.radians(theta)
    def held(r):
        d = np.sqrt(1 + r*r - 2*r*np.cos(t))
        lens = np.pi*(1 + r - d)**2*(d*d + 2*d*r - 3*r*r + 2*d + 6*r - 3)/(12*d)
        return 4*np.pi*r**3/3 - lens
    lo, hi = 1e-3, 10.0
    for _ in range(100):
        lo, hi = ((lo + hi)/2, hi) if held((lo + hi)/2) < volume else (lo, (lo + hi)/2)
    return 2/lo

ev.restore(last_stable)
steps = list(recipes.arclength(ev, ev.body(1), -0.0015, 8, topology=True))
k = [i for i, s in enumerate(steps) if s.fell][0]
lost, fell = steps[k - 1], steps[k]
total = fell.volumes.sum()
print(f"from {steps[0].value:.4f}: unstable at {lost.value:.4f}, where it fell; "
      f"the neck cut {abs(fell.changes[0].where[2]):.2f} from the mid-plane")
for i, (V, p) in enumerate(zip(fell.volumes, fell.pressures)):
    close = abs(p/cap_pressure(V, theta) - 1) < 0.005
    print(f"drop {i + 1}: {V/total:.1f} of the liquid, pressure {p:.2g}, "
          f"{'within' if close else 'NOT within'} 0.5% of the exact cap's")
kept = abs(steps[-1].volumes[1]/fell.volumes[1] - 1) < 1e-8
print(f"together {total:.6f}; then drop 1 drained to {steps[-1].volumes[0]:.2f}, "
      f"drop 2 {'kept' if kept else 'NOT kept'} at its volume")
```

```text
from 0.0792: unstable at 0.0777, where it fell; the neck cut 0.11 from the mid-plane
drop 1: 0.8 of the liquid, pressure 4.2, within 0.5% of the exact cap's
drop 2: 0.2 of the liquid, pressure 6.2, within 0.5% of the exact cap's
together 0.077708; then drop 1 drained to 0.06, drop 2 kept at its volume
```

The run restarts from the last stable bridge (0.0792), and its first point, at
0.0777, is already unstable: it falls from there. The drops' pressures are within 0.2%
of the exact caps', and together they hold what the bridge held, to round-off. The cut
takes the two rows of vertices round the neck to a point, which takes a few percent of
the volume there away; each piece is given back what it lost on its side (the
tetrahedra from the point to the collapsed facets), so no volume is lost or moved from
one drop to the other.

How the liquid splits between the drops is the one result here that depends on more
than equilibrium: two drops hold any split, and which one forms depends on the way
down. The fall is a descent of the energy, steepest along the unstable mode, not the
liquid's flow, and the split moves with how the fall is taken. Here the smaller drop
holds 17% to 19% of the liquid from run to run (the threads' round-off in the last
step before the fall, which the fall carries a long way); falling from 0.0762 (a step
of 0.003 from the last stable bridge) it held 25%, and on the mesh refined once less
(2048 facets) 18% to 22% as the steps along the mode and the gradient steps between
them were varied. Read the split as a rough figure; the drops' pressures are exact
for the split they have.

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
  the break itself with gradient steps alone: `relax()` from the bridge pushed along
  its mode makes slow progress, because the mode's eigenvalue is barely below zero
  that close to the onset. `arclength(topology=True)` steps along the mode instead
  (see [Through the break](#through-the-break)).
* **Real bridges are likely to break earlier still.** With a subcritical pitchfork,
  a small asymmetry (unequal spheres or contact angles, gravity along the axis)
  typically makes the bridge break before the symmetric branch point. A supercritical
  one would make the neck drift steadily toward one sphere instead.
* **Which comes first depends on the geometry.** Vogel's examples include one where
  the instability comes first, as here (contact angle 45 degrees), and one where the
  fold does (9 degrees).

To find instabilities like this in a surface of your own, follow the branch with
`arclength` and watch `negative_modes` change. A symmetric reference alone can't see
an asymmetric mode. Where a mode turns negative without a fold, another branch crosses,
and `along=` with that mode takes the run onto it.

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
* C. Semprebon, M. Scheel, S. Herminghaus, R. Seemann and M. Brinkmann, Liquid
  morphologies and capillary forces between three spherical beads, *Phys. Rev. E* 94
  (2016) 012907, [doi:10.1103/PhysRevE.94.012907](https://doi.org/10.1103/PhysRevE.94.012907).
  Liquid between three beads, computed with Surface Evolver, including the limits
  where its shapes lose stability.
* J. Wang, E. Gallo, B. François, F. Gabrieli and P. Lambert, Capillary force and
  rupture of funicular liquid bridges between three spherical bodies, *Powder
  Technology* 305 (2017) 89-98,
  [doi:10.1016/j.powtec.2016.09.060](https://doi.org/10.1016/j.powtec.2016.09.060).
  Where liquid between three spheres breaks as they are pulled apart, computed with
  Surface Evolver.
