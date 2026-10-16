# A lens floating on a liquid

A drop of oil on water, if the oil doesn't spread, floats as a **lens**. Three
interfaces meet along its rim: the water's free surface (tension $\gamma_w$), the
lens's top against the air ($\gamma_t$) and its bottom against the water ($\gamma_b$).
At the rim the three tensions balance as vectors: they close a triangle, **Neumann's
triangle**, which fixes the angles between the three surfaces. Without gravity the
water's surface stays flat, and the lens is two spherical caps meeting it at the
angles of the triangle. With gravity a large lens flattens to a puddle whose thickness
Langmuir found in 1933.

This page computes both with pySE and compares them with exact answers: the closed
form without gravity, and the axisymmetric equations, solved as a boundary value
problem, with it. It is the docs' example of facets with different tensions, and of
three films meeting along a line that moves.

The lens is computed in a quarter of space between the mirror planes x = 0 and y = 0
(which also holds it in place: a floating lens can drift sideways freely). The water's
surface is held at z = 0 on a quarter circle well away from the lens. The surfaces are
built in `docs/floating_lens_helpers.py`. Each example takes a few seconds.

## Three tensions

Give each facet its tension after loading, by face row; refined facets inherit it. The
helper's three surfaces are the lens's top and bottom (the body: a lens of fixed
volume) and the water's surface around it:

```
ev.load_arrays(vertices, faces, bodies=[pyse.Body(volume=volume/4, faces=lens_rows, ...)], ...)
tension = np.empty(len(faces))
tension[top_rows], tension[bottom_rows], tension[water_rows] = gamma_top, gamma_bottom, gamma_water
ev.set_values("facet", "tension", tension)
```

## Without gravity: two caps

With the water's surface flat, the triangle of tensions gives the angles $\theta_t$ and
$\theta_b$ of the top and bottom caps with it,

$$\cos\theta_t = \frac{\gamma_w^2 + \gamma_t^2 - \gamma_b^2}{2\gamma_w\gamma_t}, \qquad
  \cos\theta_b = \frac{\gamma_w^2 + \gamma_b^2 - \gamma_t^2}{2\gamma_w\gamma_b},$$

a lens exists only if the three tensions close a triangle: each less than the sum of
the other two. Two caps on a rim of radius $r$ have heights $r\tan(\theta/2)$; the
volume sets $r$, and the pressure inside is $2\gamma_t\sin\theta_t/r$ (or the same
with the bottom's values: the vertical balance at the rim makes them equal). `neumann`
has these. The start, `floating_lens`, puts two caps of the same angle (the mean of the
exact ones) on the exact rim, so both angles must change by several degrees. After
relaxing, `lens_shape` measures the rim, the thickness and the pressure, and fits a
sphere to each cap's vertices for its angle; the energy is counted from the water's
surface without the lens.

```python
from floating_lens_helpers import floating_lens, lens_shape, neumann

for tensions in [(1.0, 0.7, 0.5), (1.0, 0.6, 0.6), (1.0, 0.9, 0.8), (1.0, 0.55, 0.47)]:
    ev = floating_lens(1.0, *tensions, levels=2)        # volume 1, refined twice
    got, exact = lens_shape(ev, gamma_bath=tensions[0]), neumann(*tensions, 1.0)
    print(f"tensions {tensions}: angles {got['angle_top']:.2f} and {got['angle_bottom']:.2f} "
          f"(exact {exact['angle_top']:.2f} and {exact['angle_bottom']:.2f}), {ev.counts['facets']} facets")
    print("    relative errors: " + ", ".join(f"{k} {got[k]/exact[k] - 1:+.1e}"
                                         for k in ("rim", "thickness", "pressure", "energy")))
```

```text
tensions (1.0, 0.7, 0.5): angles 27.67 and 40.55 (exact 27.66 and 40.54), 18560 facets
    relative errors: rim +8.8e-05, thickness +5.1e-04, pressure +8.4e-05, energy +6.3e-05
tensions (1.0, 0.6, 0.6): angles 33.57 and 33.57 (exact 33.56 and 33.56), 18656 facets
    relative errors: rim +8.3e-05, thickness +5.3e-04, pressure +9.9e-05, energy +6.3e-05
tensions (1.0, 0.9, 0.8): angles 49.49 and 58.79 (exact 49.46 and 58.75), 12608 facets
    relative errors: rim +1.4e-04, thickness +1.0e-03, pressure +1.7e-04, energy +1.2e-04
tensions (1.0, 0.55, 0.47): angles 10.50 and 12.31 (exact 10.49 and 12.31), 39520 facets
    relative errors: rim +4.1e-05, thickness +2.5e-04, pressure +4.1e-05, energy +2.8e-05
```

The last lens is close to spreading: its spreading coefficient
$\gamma_w - \gamma_t - \gamma_b$ is $-0.02$, the caps meet the water at 10 and 12
degrees, and the lens is ten times as wide as it is thick. All four agree with the
exact lenses: the angles to a few hundredths of a degree, rim, pressure and energy to
about 1e-4, the thickness (from the highest and lowest vertices) to 1e-3. The errors
are of second order in the edge length, as refining shows for the first lens:

```python
from floating_lens_helpers import floating_lens, lens_shape, neumann

exact = neumann(1.0, 0.7, 0.5, 1.0)
for levels in (1, 2, 3):
    ev = floating_lens(1.0, 1.0, 0.7, 0.5, levels=levels)
    got = lens_shape(ev, gamma_bath=1.0)
    print(f"{ev.counts['facets']:6d} facets: " + ", ".join(
        f"{k} {got[k]/exact[k] - 1:+.1e}" for k in ("angle_top", "angle_bottom", "rim", "energy")))
```

```text
  4640 facets: angle_top +1.7e-03, angle_bottom +1.8e-03, rim +3.4e-04, energy +2.5e-04
 18560 facets: angle_top +4.1e-04, angle_bottom +4.3e-04, rim +8.8e-05, energy +6.3e-05
 74240 facets: angle_top +9.9e-05, angle_bottom +1.0e-04, rim +2.3e-05, energy +1.6e-05
```

## With gravity: a puddle

Under gravity the lens weighs $\rho_\ell g$ per volume and the water $\rho_w g$. Far
from the lens the water's level is z = 0, so the water is a reservoir at that level:
in the helper it is a body with no fixed volume (`Body(volume=None)`), whose faces are
its free surface and the lens's bottom. Its gravitational energy counts the water
lifted above z = 0 next to the lens and the water the lens pushes below it (the
buoyancy). The mirror planes and the held ring are vertical, so the faces missing there
add nothing.

A large lens is flat in the middle. Langmuir (1933) balanced the forces on a straight
edge of such a puddle and found its thickness $e$,

$$e^2 = \frac{-2S\,\rho_w}{g\,\rho_\ell(\rho_w - \rho_\ell)}, \qquad S = \gamma_w - \gamma_t - \gamma_b,$$

with a part $e\rho_\ell/\rho_w$ of it below the water's level. Here the tensions are
those of the first lens above, the densities 0.9 and 1, and g = 100: the capillary
length of the water is 0.1, a tenth of the lens's size at volume 1. The reference is
`axisymmetric_lens`: the Young-Laplace equations of the three surfaces, round about the
axis, solved together with SciPy's `solve_bvp` (unknowns the lens's pressure and the
rim's radius; at the rim the profiles meet with their tensions balanced, and far out
the water's surface decays as a Bessel function $K_0$).

The runs start from the reference's rim radius, with caps of the same angle. The mesh is
finer near the rim (`h_rim`), where the top and bottom bend over a length of 0.1 to 0.2,
and is refined once:

```python
from floating_lens_helpers import axisymmetric_lens, floating_lens, langmuir, lens_shape

tensions, rho_lens, rho_water, g = (1.0, 0.7, 0.5), 0.9, 1.0, 100.0
for volume in (0.25, 1.0, 4.0):
    ref = axisymmetric_lens(volume, *tensions, rho_lens, rho_water, g)
    ev = floating_lens(volume, *tensions, gravity=g, rho_lens=rho_lens, rho_bath=rho_water,
                       rim=ref["rim"], h=0.15, h_rim=0.03)
    got = lens_shape(ev)
    print(f"volume {volume}: rim {got['rim']:.4f} (axisymmetric {ref['rim']:.4f}), "
          f"thickness {got['thickness']:.4f} ({ref['thickness']:.4f}), "
          f"pressure {got['pressure']:.4f} ({ref['pressure']:.4f})")
e, below = langmuir(*tensions, rho_lens, rho_water, g)
print(f"Langmuir: thickness {e:.4f}, {below:.4f} of it below the water's level")
```

```text
volume 0.25: rim 0.7510 (axisymmetric 0.7517), thickness 0.2179 (0.2167), pressure 2.2158 (2.2084)
volume 1.0: rim 1.3669 (axisymmetric 1.3681), thickness 0.2259 (0.2253), pressure 2.0537 (2.0484)
volume 4.0: rim 2.5933 (axisymmetric 2.5966), thickness 0.2198 (0.2192), pressure 1.9784 (1.9725)
Langmuir: thickness 0.2108, 0.1897 of it below the water's level
```

pySE agrees with the axisymmetric lenses: the rim to 0.1%, the pressure to 0.3%, the
thickness to 0.6% (0.3% in the two larger ones). The lenses are thicker than
Langmuir's puddle, and the thickest is the middle one. The reference shows how the
thickness comes down to Langmuir's in larger lenses:

```python
from floating_lens_helpers import axisymmetric_lens, langmuir

e, below = langmuir(1.0, 0.7, 0.5, 0.9, 1.0, 100.0)
for volume in (1, 4, 16, 64):
    ref = axisymmetric_lens(volume, 1.0, 0.7, 0.5, 0.9, 1.0, 100.0)
    excess = ref["thickness"] - e
    print(f"volume {volume:2d}: rim {ref['rim']:.3f}, thickness {ref['thickness']:.5f}, "
          f"above Langmuir's by {excess:.5f} (times the rim: {excess*ref['rim']:.4f})")
```

```text
volume  1: rim 1.368, thickness 0.22531, above Langmuir's by 0.01449 (times the rim: 0.0198)
volume  4: rim 2.597, thickness 0.21916, above Langmuir's by 0.00834 (times the rim: 0.0217)
volume 16: rim 5.054, thickness 0.21499, above Langmuir's by 0.00417 (times the rim: 0.0211)
volume 64: rim 9.969, thickness 0.21290, above Langmuir's by 0.00209 (times the rim: 0.0208)
```

The excess falls off as the inverse of the rim's radius. That fits the one thing a
round puddle has and Langmuir's straight edge doesn't: the rim's curvature seen from
above, which goes as one over the radius. Small lenses are thinner than the
limit (spherical caps, as without gravity), so on the way the thickness passes through
a maximum, near a rim radius of 1.2 here (figure below). Langmuir's abstract says the
thickness "slowly approaches" the limit as the volume grows.

## Pictures

```{figure} _static/floating_lens.png
:alt: Cross-sections of a floating lens without gravity and of two lenses under gravity against exact profiles, and the thickness of lenses under gravity against their rim radius with Langmuir's limit
:width: 100%

Left, the first lens without gravity: pySE's vertices on the mirror plane y = 0 (dots)
on the exact caps. Middle, lenses of volume 1 and 4 under gravity (heights stretched
five times) on the axisymmetric profiles. Right, the thickness at the centre against the
rim radius: axisymmetric (line), pySE (circles), and Langmuir's limit (dotted). Drawn by
`tools/floating_lens_figure.py`.
```

```{figure} _static/floating_lens_shapes.png
:alt: Two floating lenses, a quarter cut away: a lens of two spherical caps without gravity and a flat puddle under gravity
:width: 100%

The lens without gravity and the puddle of volume 4 (heights stretched four times),
mirrored into three quadrants: the top (orange), the bottom (blue), and the water
around them (grey).
```

## Notes and limits

* **The rim moving over the mesh.** A Newton step moves each vertex along its normal
  only, and the rim along the surfaces beside it, so a rim that moves further than an
  edge or two would squash the facets ahead of it; pySE's Newton steps carry the
  vertices beside the rim along with it (`PYSE_NO_FOLLOW=1` turns that off). The lens
  near spreading started from a rim 5% too small or too large converges to the exact
  rim (1.4703 against 1.4702, refined twice). The runs above start at the exact rim
  (without gravity) or the reference's (with it) all the same, with the caps' angles
  wrong instead.
* **Measuring angles.** `ev.contact_angles()` is for walls. Here a sphere is fitted to
  each cap's vertices, which is accurate to second order, as the energy is; with gravity
  the caps aren't spheres, and `lens_shape`'s angles mean nothing.
* **The pressure** with gravity is Evolver's body pressure, the lens's pressure at
  z = 0, as the reference's.
* **Mirror planes** keep only shapes with the lens's mirror symmetries, which a lens
  has. They also stop it drifting: a whole lens can slide along the water at no cost,
  a direction in which the Newton steps' matrix is singular.
* **Refining further** under gravity, a second time (56 thousand facets at volume 4),
  gives the rim 2.5960 and the thickness 0.2193 (axisymmetric 2.5966 and 0.2192) in
  12 seconds; once is what the runs above use.

## References

* F. Neumann, *Vorlesungen über die Theorie der Capillarität*, ed. A. Wangerin
  (Teubner, Leipzig, 1894). The triangle of tensions at the rim.
* I. Langmuir, Oil lenses on water and the nature of monomolecular expanded films,
  *J. Chem. Phys.* 1 (1933) 756–776,
  [doi:10.1063/1.1749243](https://doi.org/10.1063/1.1749243). The limiting thickness
  of large lenses.
* P. R. Pujado and L. E. Scriven, Sessile lenticular configurations: Translationally
  and rotationally symmetric lenses, *J. Colloid Interface Sci.* 40 (1972) 82–98,
  [doi:10.1016/0021-9797(72)90175-0](https://doi.org/10.1016/0021-9797(72)90175-0).
  The statics of floating lenses, straight and round.
* J. C. Burton, F. M. Huisman, P. Alison, D. Rogerson and P. Taborek, Experimental and
  numerical investigation of the equilibrium geometry of liquid lenses, *Langmuir* 26
  (2010) 15316–15324, [doi:10.1021/la102268n](https://doi.org/10.1021/la102268n).
  Computed axisymmetric lenses against alkane lenses on water.
* P. D. Ravazzoli, A. G. González, J. A. Diez and H. A. Stone, Buoyancy and capillary
  effects on floating liquid lenses, *Phys. Rev. Fluids* 5 (2020) 073604,
  [doi:10.1103/PhysRevFluids.5.073604](https://doi.org/10.1103/PhysRevFluids.5.073604).
  The lens without gravity in closed form; with gravity, two families of solutions for
  the same parameters, one of lower energy.
