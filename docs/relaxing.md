# Relaxing a surface

`ev.relax()` takes a surface to an equilibrium with defaults meant to work as they
are, and says how it went:

```python
r = ev.relax()
print(r)
# IterationResult(converged after 130 gradient steps and 2 Newton, residual 2.8e-09,
#                 stable, health ok)
```

`r.converged` (the residual is below `tol`), `r.stable` (no direction lowers the
energy) and `r.health` (facets, walls, the surface itself; see below) are the three
things to check. `ev.relax(levels=3)` refines and relaxes three times more.

This page explains what `relax()` does, how to read its result, what to do about mesh
resolution, and where the defaults stop working. The numbers come from the stress
suite in `bench/stress/` (nine hard cases with exact or axisymmetric references, all
run with default settings) and are from October 2026.

## What `relax()` does

1. **Gradient steps in rounds.** Evolver's `g` steps, ten per round. Each round ends
   with equiangulation and vertex averaging (`u; V`; only `V` in the Lagrange model,
   which can't be equiangulated), which keep facets from degenerating where the
   surface stretches or vertices drift. The steps stop when the energy settles (a
   relative change below 1e-5 over three rounds).
2. **A watch on the residual.** At the end of each round `relax()` measures the
   residual (how far the surface is from equilibrium, `ev.residual()`). If a round
   more than doubles it from its lowest, the round is undone and Newton takes over: on
   a coarse mesh a contact line can collapse within a few steps (its vertices slide
   together), and the energy keeps falling while it does.
3. **Safeguarded Newton steps.** Up to 20, each undone if it makes things non-finite
   or raises both the energy and the residual, until the residual is below `tol`
   (1e-8). Newton converges quadratically near an equilibrium; the gradient steps are
   there to get it near.
4. **A second try.** If the surface is still not in equilibrium, conjugate gradients
   (at most 200 steps) and Newton again, kept only if that converges. Plain gradients
   crawl on flat shapes, such as a drop spread to a 10 degree contact angle.
5. **Finer next to contact lines.** Once converged (linear model), edges leaving a
   contact line that are longer than 1.5 times the contact-line edges there are split
   and the surface relaxed again, kept only if that converges: a contact circle that
   shrinks (a drop at a high contact angle) would otherwise end up smaller than the
   facets next to it.
6. **A stability check.** Once converged, the Hessian's eigenvalues are counted (one
   factorization). A saddle, such as a liquid column past its Rayleigh-Plateau limit,
   is an equilibrium too, and Newton finds it as readily as a minimum; `r.stable` is
   then `False` and an `UnstableEquilibriumWarning` is raised. Exact zero modes of
   symmetries (a drop that could slide on a plane) don't count.

In the quadratic and Lagrange models (a final stage, from a mesh settled in the linear
model) `relax()` first tries up to five Newton steps alone and keeps them if they
converge. Switching a relaxed 24k-facet cube to `lagrange 2` and relaxing takes about
a second.

## Reading the result

```python
r.health
# Health(ok: residual 2.8e-09, smallest angle 45.1 degrees, no wall nearby,
#        self gap inf)
r.health.issues          # what looks wrong, in words; empty when all is well
```

`ev.health()` gives the same check-up for any surface. Its fields:

| field | meaning |
|---|---|
| `residual` | distance from equilibrium (dimensionless; 0 at an equilibrium) |
| `stable`, `negative_modes` | from the stability check (`None` if not checked) |
| `angle_min`, `skinny` | smallest facet angle; facets with an angle below 5 degrees |
| `wall_gap`, `wall` | nearest approach of a vertex to a constraint it isn't on (in median edge lengths), leaving out vertices within two edges of the constraint's own vertices |
| `crossed` | vertices on the far side of a constraint |
| `self_gap` | nearest approach of the surface to itself, between vertices more than two edges apart |

It raises no warnings; a broken surface shows in `issues` (`"degenerate facets
(smallest angle 0.21 degrees, 6 below 5)"`, `"3 vertices on the far side of
constraint bead"`, ...). The check costs about 0.2 s at 100k facets.

## Mesh resolution

Apart from `levels` and the splitting next to contact lines, `relax()` doesn't change
the number of facets. Three tools do:

* `ev.refine()` or `relax(levels=n)`: uniform refinement, four times the facets each.
* `ev.adapt()` or `relax(adapt=True)`: refines where the surface turns by more than
  15 degrees along an edge and merges short edges where it has flattened (Evolver's
  edge deletion refuses merges that would tilt facets or make long edges). Fewer facets
  for the same accuracy where the curvature is uneven: a drop at 170 degrees came out
  3.5 degrees off with 734 facets, against 3.9 with 1920 refined uniformly. It is
  opt-in: with it on for every relaxation the suite passes, but runs several times
  longer at 15 degrees and loses accuracy on some cases at 25 to 30.
* `ev.remesh(target=h)`: splits edges longer than 1.6 h and deletes those shorter than
  0.5 h, then equiangulates. Call it each step where a moving contact line squeezes the
  mesh into a corner, as the drainage example does (`docs/drainage_helpers.py`).

That last one is not automatic, and that is deliberate. The drainage band needs
squeezed edges merged in the corner where a contact line meets a mirror plane; edges
that look the same by every measure tried (length, flatness, facet shape, how much
they shrank) must be kept on a barrel's rim around a thin fibre, at a catenoid's
narrowing neck, and in meshes made fine on purpose. Every automatic rule that fixed the
first damaged one of the others. The one who knows the geometry picks `h`.

## Where the defaults stop working

Of the nine stress cases, seven pass with defaults alone: a liquid column past
Rayleigh-Plateau (the instability shows at 1.8% past its onset), a catenoid up to its
fold, a liquid bridge with the gap closing to 1e-3, a barrel drop on a fibre (its
roll-up reported as an instability), a cube inflated a hundredfold, a gravity puddle,
and a drop shrinking to 1e-4 of its volume. Two don't:

* **A drop at 170 degrees** on a coarse starting mesh (480 facets): the implied contact
  angle is 1.7 degrees off (the case allows 1.5; up to 160 degrees it passes). More
  resolution helps.
* **The drainage band** (`docs/slit_drainage.ipynb`) on plain defaults: band pressures
  up to several percent off (median 0.3%). With `ev.remesh(target=h)` each step, as in the notebook, they
  agree with a finer reference to 0.02% (median).

Other things to know:

* **Saddles.** `relax()` converges to saddles as readily as to minima. Check
  `r.stable`, or watch for `UnstableEquilibriumWarning`.
* **Unconverged results.** `r.converged` is `False` when the residual didn't reach
  `tol`; the surface is the best `relax()` found. `r.health.issues` tells whether it
  is merely unfinished or broken.
* **Large surfaces.** At 400k facets a default `relax()` near equilibrium takes about
  4 s with 8 threads (a raw `g 5` and three Newton steps: 2.8 s); the stability check
  is one factorization of the Hessian.
