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
resolution, and where the defaults stop working. It assumes the drop of
{doc}`getting_started`; the {doc}`tutorial` shows `relax()` at work on a double bubble.
The numbers come from the stress suite in `bench/stress/` (ten hard cases
with exact or axisymmetric references, all run with default settings;
{doc}`stress_cases` has condensed versions) and are from October 2026.

## What `relax()` does

1. **Gradient steps in rounds.** Evolver's `g` steps, ten per round. Each round ends
   with equiangulation and vertex averaging (`u; V`; only `V` in the Lagrange model,
   which can't be equiangulated), which keep facets from degenerating where the
   surface stretches or vertices drift. The steps stop when the energy settles (a
   relative change below 1e-5 over three rounds). After the first step the surface is
   also equiangulated once: a coarse start triangulated badly (Brakke's `cbga1.fe`,
   a prism of quads split at their centres) would otherwise settle into what its
   mesh allows before the first round ends.
2. **A watch on the residual.** At the end of each round `relax()` measures the
   residual (how far the surface is from equilibrium, `ev.residual()`). If a round
   more than doubles it from its lowest, the round is undone and Newton takes over: on
   a coarse mesh a contact line can collapse within a few steps (its vertices slide
   together), and the energy keeps falling while it does.
3. **Safeguarded Newton steps.** Up to 20, each undone if it makes things non-finite,
   raises both the energy and the residual (unless it brings a missed volume or
   quantity target at least halfway: the residual doesn't see those), or raises the
   energy by more than 10% plus what meeting the targets costs, until the residual is
   below `tol` (1e-8) with the targets met. Newton converges quadratically near an
   equilibrium; the gradient steps are there to get it near.
4. **A second try.** If the surface is still not in equilibrium, conjugate gradients
   (at most 200 steps) and Newton again, kept only if that converges. Plain gradients
   crawl on flat or stiff shapes.
5. **Finer next to contact lines.** Once converged (linear model), edges leaving a
   contact line that are longer than 1.5 times the contact-line edges there are split
   and the surface relaxed again, kept only if that converges: a contact circle that
   shrinks (a drop at a high contact angle) would otherwise end up smaller than the
   facets next to it.
6. **A stability check.** Once converged (or within 100 times `tol`, as when a step
   of a continuation stops just short), the Hessian's eigenvalues are counted (one
   factorization; none when the last Newton step began within a residual of 1e-2
   and its own factorization counted no negative eigenvalue). A saddle, such as a liquid column past its Rayleigh-Plateau limit,
   is an equilibrium too, and Newton finds it as readily as a minimum; `r.stable` is
   then `False` and an `UnstableEquilibriumWarning` is raised. Exact zero modes of
   symmetries (a drop that could slide on a plane) don't count. `ev.saddle()` steps
   off a saddle along its unstable mode, and `ev.relax()` from there finds the
   minimum it leads to (Brakke's `cbga2.fe`: a solder bulge leaning over).

Other models change some of this. Strings are only vertex-averaged, the simplex
representation only equiangulated. Surfaces without a Hessian get plain gradient
rounds (and an explicit `newton` or `stability=True` warns that it can't be had), settled to 1e-9, in place of Newton steps, while no facet is degenerate (below
1 degree; on degenerate facets a long gradient run slides into what the mesh allows):
Evolver has no Hessian for gap energies on curved walls, crystalline integrands, or
bodies in the simplex representation outside the Lagrange model. When such a run ends
with the energy settled but the residual above `tol`, `r.settled` is `True` and the
result reads `not converged (energy settled)`. With an elastic energy (Evolver's
`linear_elastic`, `relaxed_elastic`, `neo_hookean`, `SVK_elastic` and the like) the
in-plane forces are physical, so `relax()` doesn't tidy the mesh, counts every force
direction in the residual, and takes Newton steps by line search (`hessian_seek`).
The residual is then in units of the load (a pressure, say), not the modulus: with
a stiffer membrane the same residual means a smaller strain error.
Crystalline (Wulff) energies get no tidying either: they depend on each facet's
orientation, and their residual stays above zero even at the minimum (the energy has
a corner there), so `relax()` counts them converged once the energy settles, and
`health()` leaves their residual out.
A Newton step that raises the energy by more than 10%, plus what meeting volume or
quantity targets costs, is always undone. A gradient round that pushes a met volume
off its target (Evolver's projection can fail on a degenerate mesh) is undone and
ends the rounds; `health()` reports any volume or fixed quantity more than 1e-4 off
its target. Where
Evolver changes the topology during gradient steps (`autopop`, `autochop`), a round
can't be undone, and `relax()` goes on from where it is.

In the quadratic and Lagrange models (a final stage, from a mesh settled in the linear
model) `relax()` first does what the manual recommends there, five gradient steps and
then Newton steps (`g 5; hessian`) until the residual is below `tol`, and keeps the
result if it converges. Switching a relaxed 98k-facet cube to `lagrange 2` and
relaxing takes 3 to 4 seconds on 2 threads.

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
| `junctions` | junctions that break Plateau's laws: edges where four or more films meet, vertices whose tangent cone Evolver's `o` would pop (more than four triple edges, a cube's cone, ...), or in a string network four or more strings meeting |
| `target_gap` | largest relative gap between a fixed body volume or quantity and its target |

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
  mesh into a corner, as the {doc}`drainage example <slit_drainage>` does.

That last one is not automatic, and that is deliberate. The drainage band needs
squeezed edges merged in the corner where a contact line meets a mirror plane; edges
that look the same by every measure tried (length, flatness, facet shape, how much
they shrank) must be kept on a barrel's rim around a thin fibre, at a catenoid's
narrowing neck, and in meshes made fine on purpose. Every automatic rule that fixed the
first damaged one of the others. The one who knows the geometry picks `h`.

## Where the defaults stop working

Of the fourteen stress cases, thirteen pass with defaults alone: a liquid column past
Rayleigh-Plateau (the instability shows at 1.8% past its onset), a catenoid up to its
fold, a sessile drop from 10 to 170 degrees, a liquid bridge with the gap closing to 1e-3, a barrel drop on a fibre (its
roll-up reported as an instability), a cube inflated a hundredfold, a gravity puddle,
a drop shrinking to 1e-4 of its volume, Schwarz's P and D surfaces in quotient
spaces (to 1e-6 and 2e-5 when extrapolated), a drop on a sphere from 20 to 150
degrees, a double bubble, a solder joint's force (to 5e-5 against an axisymmetric
model, extrapolated) and a catenoid between round wires stopped at its fold. One
doesn't:
* **The drainage band** ({doc}`slit_drainage`) on plain defaults: band pressures
  up to several percent off (median 0.3%). With `ev.remesh(target=h)` each step, as in the notebook, they
  agree with a finer reference to 0.02% (median). Without it the mesh breaks down at
  the corner where a bead's contact line meets the wall: in a typical band state 95%
  of the residual (squared) sits in ten vertices there, with facets under a degree
  and vertices pushed into the bead, so nearly every state ends unconverged although
  the pressure is within a few percent.

Other things to know:

* **Relax before you refine.** Every refinement multiplies the gradient steps a shape
  change takes: `cube.fe` refined five times and then relaxed takes 10 s and doesn't
  converge, while `relax(levels=5)` (a relax after each refinement) converges in 0.4 s
  to the same energy. `relax()` warns when the surface was refined four or more times
  since it last converged.

* **Saddles.** `relax()` converges to saddles as readily as to minima. Check
  `r.stable`, or watch for `UnstableEquilibriumWarning`.
* **Topology.** Plateau's laws say how soap films meet in a stable equilibrium:
  three films along an edge at 120 degrees, four such edges at a point (strings:
  three at a point). Other junctions can always lower the energy by splitting, so
  `relax()` pops them (Evolver's `o`) and relaxes again; `r.popped` counts them, and
  Brakke's `octa.fe` and `popstr.fe` reach the energies of his own recipes. If the
  surface still won't converge, a last resort: gradient steps without tidying (which
  can hold a shrinking neck open as a tube of slivers), then edges that have
  collapsed to a 20th of the median are deleted, junctions popped, and the surface
  relaxed, kept only if that converges, or lowers the energy on a sound mesh without
  raising the residual (`r.collapsed`). That is how a small film in a
  foam shrinks away, and how a catenoid past its fold pinches off into two discs
  (`cat.fe`, stress case 2). A converged surface never loses edges. The last resort
  costs up to a few hundred gradient steps and a save and restore; after k failed
  attempts in a row the next 2^k - 1 calls skip it, so a continuation that stays
  unconverged pays for a few attempts, not one per step.
* **No Hessian.** Gap energies and simplex bodies get gradient steps only, and
  gradient steps converge slowly: such results often end unconverged by the residual,
  with the energy settled (`r.settled`). Crystalline energies count as converged once
  their energy settles.
* **Elastic energies and other units.** The residual is a force balance in the
  surface's own units (a tension: load times length), sized for loads of about 1. A
  stiffer membrane under the same load needs no other `tol` (the same residual means
  a smaller strain error); a load a thousand times larger needs a `tol` about that
  much larger.
* **Unconverged results.** `r.converged` is `False` when the residual didn't reach
  `tol`; the surface is the best `relax()` found. `r.health.issues` tells whether it
  is merely unfinished or broken. Converged results agree across thread counts to
  about the tolerance; unconverged ones need not: threads add the forces in a
  different order, and a long run that never settles (no Hessian, a coarse mesh)
  follows a different path (`tankex.fe` from 8 facets ends at energy 1.60 with 16
  threads and 1.67 with 4). Compare unconverged results only at the same thread count.
* **Large surfaces.** At 400k facets a default `relax()` near equilibrium takes about
  4 s with 8 threads (a raw `g 5` and three Newton steps: 2.8 s); the stability check
  is one factorization of the Hessian.

## Next

The worked examples use `relax()` throughout, starting with {doc}`liquid_bridge`.
{doc}`stress_cases` runs the hard cases above against their references, and
{doc}`performance` has timings of `relax()` on large surfaces.
