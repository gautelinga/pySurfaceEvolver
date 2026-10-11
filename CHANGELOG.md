# Changelog

## Unreleased

### relax() on all of Brakke's samples

Each of the 25 sample datafiles was relaxed with defaults and compared with its own
`gogo` recipe; what that showed is fixed:

- Fixed: `relax()` crashed on strings ("Cannot equiangulate string model") and in the
  simplex representation ("V not implemented"): its tidying uses only the commands the
  model has. Bodies in the simplex representation (outside the Lagrange model) get no
  Newton steps: Evolver has no Hessian for them, and a failed attempt left the
  surface unusable.
- `relax()` on surfaces without a Hessian (gap energies on curved walls, crystalline
  integrands): plain gradient rounds settled to 1e-9 instead of stopping when the
  residual watch fired (tankex: energy 1.75 -> 1.50).
- Newton steps that raise the energy by more than 10% are undone, whatever the residual
  does: on a 100-cell foam a step ran off to a tangled surface (energy 19 -> 8e17) with
  a smaller residual, and `relax()` reported it converged.
- Fixed: undoing a gradient round or Newton step by the vertex coordinates when Evolver
  had changed the topology meanwhile (`autopop`, `autochop`) wrecked the surface. Such
  undos are now skipped (new binding: the topology stamp).
- Elastic energies (Evolver's elastic methods, such as `relaxed_elastic_A` in
  `mylarcube.fe`): no mesh tidying (it corrupted the facets' reference shapes), a
  residual over all force directions, Newton by `hessian_seek`.
- Elastic and crystalline energies keep relax()'s rounds of 10 steps for measuring
  convergence (only the tidying is skipped): `tidy=0` measured step to step and stopped
  early (`mylarcube.fe` along its recipe's modulus ramp: -1.2518 -> -1.2563, recipe
  -1.2577).
- Facet angles in quotient spaces from Evolver's edge lengths too (`symtest.fe`).
  `symtest.fe` reads as unstable and `twointor.fe`, the same surface, doesn't: rightly,
  since `symtest.fe`'s volumes are only comments, so its cells can trade volume
  (Evolver's `saddle` lowers the energy).
- Crystalline (Wulff) energies: no mesh tidying in `relax()` (flipping and averaging undid
  the facets' alignment with the crystal: `crystal.fe` 3.608 -> 3.3256, as plain gradient
  steps reach). Their residual stays above zero at the minimum (the energy has a corner
  there), so `relax()` takes gradient rounds only and counts them converged once the
  energy settles; `health()` leaves their residual out.
- `ev.residual()`: plain forces under `area_normalization` (it divided each by its
  vertex's area), and dimensionless for strings and simplices too.
- `ev.health()`: `junctions` counts junctions that break Plateau's laws (`octa.fe`: 7,
  popped by `o`): edges where four or more films meet, and vertices whose tangent cone
  `o` would pop, found as `o` finds them (new binding). No more false alarms from fixed vertices drawing a wall (mound's table
  top) or from contact lines on a constraint drawn twice (slidestr); facet angles in a
  torus from Evolver's edge lengths (wrapped coordinates made the foams' facets look
  degenerate); planar and simplex surfaces handled.
- `relax()` pops junctions that break Plateau's laws (Evolver's `o`) and relaxes again;
  if the surface still won't converge, edges collapsed below a 20th of the median are
  deleted (`pinch=True`, the default; a converged surface never loses edges).
  `octa.fe` and `popstr.fe` now reach the energies of Brakke's recipes on defaults.
  `IterationResult.popped` and `.collapsed` count them. The deletion runs as a last
  resort: untidied gradient steps let a shrinking neck or film collapse (the tidying
  held it open as slivers), and the outcome is kept only if it converges or lowers the
  energy on a sound mesh. `cat.fe` and the stress catenoid past its fold pinch off into
  two discs (stress case 2 now checks the discs). After k failed attempts in a row,
  the next 2^k - 1 calls skip the last resort (without that it made the drainage
  stress case on plain defaults, with its ~120 unconverged steps, about 3x slower).
- Edge deletion refuses only merges that would make two facets one (the case that lost
  surfaces in Evolver 2.70a). The check from 0.7.0 also refused shared neighbours and
  parallel edges without that, and loop edges, which is how a collapsing neck's ring
  of edges closes: `cat.fe`'s recipe (`t .05; o`) now pinches the catenoid into two
  discs as in 2.70a (13.71288; was stuck at 15.53).
- Fixed: stress case 3 counted only its second sweep's time and steps.

### Found afterwards

- Fixed: `ev.newton()` couldn't meet a new volume target (`ev.body(1).target = 2.0`
  then `ev.newton()` took no step; also in 0.7.0). The residual doesn't see an unmet
  volume, so the safeguard read the step restoring it as making things worse. A step
  that brings a missed volume or fixed-quantity target (more than 1e-4 off) at least
  halfway is now accepted, and the energy rise allowed is 10% plus what meeting the
  targets costs (pressure times the change). (With a threshold of 1e-6 instead, the
  2e-5 that vertex averaging leaves let nearly every step skip the safeguard: the
  drainage stress case's worst band pressure went from 6% to 89% off.) This hit `relax()` too: the last vertex averaging
  leaves a volume slightly off, and the conjugate-gradient pass was what rescued a
  drop spread to 10 degrees; Newton now converges it directly.
- Fixed: on a degenerate mesh, `relax()`'s last vertex averaging could lose a body's
  volume without a word (`tankex.fe` from 8 facets: 0.354 against 0.6), and plain
  gradient rounds slid into what the degenerate mesh allows (energy 1.6 -> 0.03). A
  gradient step now follows tidying that leaves a volume off; a round that pushes a
  met volume off its target is undone and ends the rounds; the gradient rounds that
  replace Newton (no Hessian) run only while no facet is below 1 degree. `ev.health()`
  reports volumes and fixed quantities more than 1e-4 off their targets (`target_gap`).
- Fixed: the last resort crashed when edge deletion left a facet without its three
  edges (it now counts as unsound and the surface goes back).
- The last resort keeps an unconverged lower energy only if the residual didn't rise
  (`tankex.fe` from 8 facets, at 2 threads: energy 1.77 -> 0.24 by collapsing facets,
  residual 0.94 -> 1.38, and that was kept).
- `relax()` warns when `newton` or `stability=True` is asked for a surface without a
  Hessian (crystalline energies, simplex bodies outside the Lagrange model); the
  defaults are now `None` (20 Newton steps, stability checked).
- `settled` describes the gradient rounds, not an undone conjugate-gradient pass.
- The wheels workflow runs on main only when what goes into a wheel changes.
- `health()` skips the cone analysis on films without junctions (no edge where three
  films meet): 55 ms at 98k facets, on every `relax()`.
- `relax()` checks stability when it ends within 100 times `tol` too (a continuation
  step stopping just short skipped the check).
- `IterationResult.settled`: the energy settled with the residual above `tol`
  (surfaces without a Hessian; shown as `not converged (energy settled)`).
- `ev.restore()` keeps `relax()`'s memory of failed attempts (it reloads a dump, which
  counted as a new surface).
- Drainage stress case on plain defaults, after all of the above: about 2.2x faster
  than before the last-resort backoff (at 2 threads, 4 runs each: 500 s against
  790-1210 s). Median band pressure 0.2% off as before; the worst state varies from run
  to run (the case is path-sensitive, also at a fixed thread count): mostly 3-8%, but
  over 10% in about a third of 11 runs before (once 147%) and a quarter of some 30
  runs now, so no change in the tail.
- The wheels workflow also runs on pushes to main and weekly (the aarch64 wheels broke
  unnoticed until the release tag built them).

### Docs

- Five notebooks on Brakke's 25 sample datafiles (films and foams, drops and capillarity,
  crystals and membranes, strings and other dimensions, quadratic and Lagrange): each
  sample's own recipe against a default `relax()` with as many refinements, with a
  table and pictures (`docs/brakke_helpers.py`). Among them Kelvin's foam against
  Weaire-Phelan's (0.35% less area per cell), and the Lagrange ladders of `cube.fe` and
  `catman.fe` (`relax()` agrees with the recipes to 1e-14; `catman.fe`'s own `gogo3`
  compares with the wrong catenoid).
- `tools/make_brakke_notebooks.py` generates the Brakke notebooks;
  `tools/check_doc_snippets.py` runs the code on `stress_cases.md` (or another page)
  and compares the outputs shown there (`--update` refreshes them).
- Stress cases: figures for each case (result against its reference, error per step,
  what `relax()` did, and the surface at a few steps), drawn by
  `bench/stress/figures.py`.

## 0.7.0 (2026-10-10)

`ev.relax()` now takes a surface to a verified equilibrium with its defaults: it
reports convergence, stability and the mesh's health, and the examples use it as is.
Of the nine hard cases in `bench/stress/`, seven pass on plain defaults. A friendlier
API (phase F) comes with it: remeshing, live body handles, constraint helpers,
continuations and diagnostics as values.

### Robust by default (phase G)

- Fixed: `relax()` failed in the Lagrange model ("Cannot equiangulate LAGRANGE model"):
  its tidying rounds now only average vertices there.
- `ev.stability()` (and so `relax()`'s check) needs `ritz` only when some eigenvalue is
  below zero; a stable surface costs one factorization. `relax()` at 98k facets:
  80 -> 1.0 s, at 393k: 275 -> 4.2 s.
- `relax()` in the quadratic and Lagrange models first tries five gradient steps and then
  safeguarded Newton steps while each cuts the residual by 10% (the manual's
  `g 5; hessian`), and keeps them if they converge (volumes at their targets
  included); otherwise it goes back and relaxes as before, and skips the attempt on its
  next call. The Lagrange stage of a 24k-facet cube: order 2 2.7 -> 1.2 s, order 4
  6.1 -> 5.2 s; the tutorial's double bubble at 45k facets reaches the manual ladder's
  energies at every order (without the settling steps, Newton diverged there at order 2). Not in the linear model, where continuations with travelling contact
  lines lost accuracy without the gradient rounds' mesh upkeep.
- `relax()` (linear model), once converged, splits the edges leaving a contact line that
  are longer than 1.5 times the contact-line edges there and relaxes again, kept only
  if that converges. A drop stepped to 160 degrees now comes out within 1.5 degrees of
  its contact angle (was 3.8 off; 170: 1.7, was 7.6), the barrel on a fibre gets
  better-shaped facets (smallest angle 49 vs 13 degrees). A cap shrinking evenly is
  left alone.
- Faster safeguards: a safeguarded Newton step undoes itself by the vertex
  coordinates instead of a dump (`ev.newton(1)` at 393k facets 3.85 -> 1.77 s, raw
  `hessian` 1.2 s), `relax()` no longer dumps the surface at its start, and
  `ev.residual()` finds the shape directions in parallel (393k facets, 8 threads:
  0.39 -> 0.14 s).
- `relax()` reorganized: the experimental `remesh=True` and `IterationResult.remeshes` are
  gone (`ev.adapt()` and `ev.remesh()` remain); a result prints as a summary
  (`IterationResult(converged after 130 gradient steps and 2 Newton, residual 2.8e-09,
  stable, health ok)`), and so does `Health`. Health messages name named constraints.
- `relax()` hands over to Newton sooner: `energy_tol` now defaults to 1e-5 when Newton
  steps follow (1e-9 with `newton=0`). The tidying in every round keeps the energy from
  settling to 1e-9, so the gradient phase used to run its full 1000 steps, long enough
  for slow mesh drift to wreck a surface (contact-line vertices sliding together on a
  coarse liquid bridge). Stress suite: 7 of 9 (the bridge passes), far fewer gradient
  steps (cylinder 26000 -> 1060, shrinking cap 25880 -> 3040).
- `relax()` undoes a gradient round that more than doubles the residual (when Newton
  follows) and hands over to Newton: on a coarse start a contact line can collapse
  within a few steps. The liquid bridge relaxes on plain defaults (`relax(levels=3)`)
  at every gap and contact angle of its notebook, which no longer needs its own
  relaxation recipe.
- `relax(cg="auto")` (new default): when the first pass ends unconverged, a second pass
  of conjugate gradient steps (at most 200, averaging vertices) and Newton, kept only if
  it converges. A sessile drop spread to 10 degrees now converges. `cg=True` still uses
  conjugate gradients throughout; `cg=False` never.
- `ev.adapt(max_turn=15)`: splits the edges along which the surface turns by more than
  `max_turn` degrees (between the vertex normals at their ends), one level per call,
  never coarsening; `relax(adapt=True)` (opt-in) alternates it with relaxation, up to
  three passes or 4x the facets. A sessile drop at 170 degrees: implied contact angle
  off by 3.5 degrees with 734 facets, against 3.9 with 1920 refined uniformly.
- `ev.adapt()` also coarsens (`coarsen=True`): short interior edges along which the
  surface turns by less than a third of `max_turn` are merged, unless that would tilt a
  facet by more than `max_turn/2`, flip one, or make an edge longer than 4/3 of the bulk
  length. Engine: Evolver's edge deletion checks this when the new variables
  `collapse_max_tilt` (degrees) and `collapse_max_edge` are set (both 0 by default:
  no change for Evolver scripts).
- `ev.health()` and `relax()`'s `result.health` (`pyse.Health`): residual, stability,
  smallest facet angle and skinny facets, the nearest approach to a wall or mirror a vertex
  isn't on (beyond two edges of its contact line) and of the surface to itself (in edge
  lengths), vertices on the far side of a constraint, and `issues` in words (empty: `ok`).
  No warnings. About 0.2 s at 98k facets. New C binding: per-vertex signed distances to
  the constraints.
- The drainage helper (docs) relaxes with plain `ev.relax(stability=False)` after its
  remeshing; its Newton rollback (`undo_if` on the mirror gap) is no longer needed
  (band pressures within 0.02% median of the stored level-3 run, snap 0.4% off).
- Newton and eigenvalue counts no longer see spurious modes from vertices on wires
  (curves where two constraints meet): their only freedom, sliding along the wire, is
  left to the gradient steps (`hessian_slant_cutoff` now 0.05, and its test uses the
  raw surface normal and the freedom's magnitude). A stable catenoid counts 0 negative
  eigenvalues (was about one per wire vertex).
- `ev.residual()` balances the pressures (volume and quantity multipliers) in the shape
  directions, as Newton does; Evolver's own fit over the full velocities, tangential
  mesh forces included, left a uniform remainder at a Newton solution (the double
  bubble stalled at 5.9e-7; now 4e-11).
- The examples (tutorial, liquid bridge, slit droplet, drainage) relax with plain
  `ev.relax()` / `ev.relax(levels=3)`.
- `ev.stability()` and, once `relax()` converges, `result.stable` /
  `result.negative_modes` with an `UnstableEquilibriumWarning`: unstable equilibria
  (a liquid column past Rayleigh-Plateau, a barrel drop rolling up) are reported
  instead of silently returned; symmetry zero modes are not mistaken for them.
- `ev.residual()`: how far the surface is from equilibrium (the normal part of the
  projected vertex forces, dimensionless; 0 at an equilibrium). New C binding.
- `relax()` defaults: rounds of 10 gradient steps with equiangulation and vertex
  averaging, then up to 20 safeguarded Newton steps, converged when the residual is
  below `tol` (now a residual tolerance, default 1e-8; `energy_tol` ends the gradient
  phase). A NaN surface is put back to the start of the gradient round where it
  happened and raised, not reported as converged.
- Newton steps (`relax`, `ev.newton()`) are safeguarded: plain Newton first, a line
  search if that is rejected; a step that makes things non-finite, or raises both the
  energy and the residual, is undone.
- `bench/stress/`: nine hard cases with exact or axisymmetric references, run with
  default settings.

### API (phase F: a friendlier API)

- `ev.relax(...)` does the whole recipe: `levels=n` (refine and relax again),
  `tidy=k` (equiangulation and vertex averaging every k steps, convergence measured
  round to round),
  `newton=n` with `seek=` and `undo_if=` (a Newton step is undone when
  `undo_if(ev)` is true). Replaces `hessian=`/`max_hessian=`, and
  `IterationResult.hessian_steps` is now `newton_steps` (breaking); the result has a
  per-iteration `level`. New `ev.newton(steps, seek=, tol=, undo_if=)`.
- `ev.remesh(target=h)` (or `max_edge=`, `min_edge=`): deletes short edges, splits
  long ones and equiangulates; `protect=` keeps an edge mask whole. Splitting honours
  `no_refine` (Evolver's `l` ignores it).
- Diagnostics as values: `ev.eigen_counts(shift)` (negative/zero/positive Hessian
  eigenvalues: an unstable equilibrium shows as `negative > 0`), `ev.check()` (topology
  problems, empty when sound), `Mesh.quality()` / `ev.mesh_quality()` (edge lengths,
  smallest angles, skinny and degenerate facets).
- Mirror images for symmetric pieces: `ev.plot(mirror=["z", "y", ("x", c)])`,
  `ev.live_view(mirror=...)` (the images share the data, so updates stay fast), and
  `Mesh.mirrored(planes)` for one combined linear mesh.
- `pyse.constraints`: planes, mirrors, spheres and cylinders with contact angles, for
  `make_datafile(constraints=)`: they write the wetting-energy and volume line
  integrals (and give the sphere's `volconst` and energy constant for wet poles);
  `Body(volconst=)`.
- `pyse.recipes.continuation(ev, set_value, values, relax=, checkpoint=, resume=)`:
  follows a family of equilibria (a body's volume, a parameter), with checkpoints that
  a stopped run resumes from; setters `recipes.body_target(i)`, `recipes.parameter(name)`.
- `ev.body(i)`: a live handle on one body (`.volume`, `.target` and `.volconst`
  settable, `.pressure`, `.fixed`); `target = None` frees the volume.
- `Bodies.target_volume` is now `Bodies.target` (breaking).
- `ev.on_constraint(k, element="vertex")`, `Mesh.edges_touching(mask, how=)` and
  `ev.set_flag(element, flag, where=, on=)` select elements with numpy masks instead
  of Evolver `where` clauses.
- `EvolverError.command` holds the failing command; errors hint at numpy reprs
  (`np.float64(...)`) and non-finite numbers in the command text.

### Fixes

- The engine is compiled without fused multiply-adds (`-ffp-contract=off`): GCC fused
  them on aarch64, which shifted unconverged runs by up to 1e-5 against x86 and
  Evolver's reference values. No change on x86-64 builds.
- Tiny-edge deletion (`t`, also used by `w` and `delete`) no longer merges the ends of an
  edge when they share neighbours other than the third vertices of its facets (the link
  condition). Evolver 2.70a merged them, making parallel and loop edges; later deletions
  then freed elements twice, hung in `edge_valence()`, or lost the surface.

### Docs

- New pages: "Relaxing a surface" (what `relax()` does, reading its result, mesh
  resolution, limits) and "Stress cases" (eight hard problems against exact references,
  with defaults; helpers in `docs/stress_helpers.py`). `docs/performance.md` compares
  `relax()` with the hand-written commands at scale (1.3-1.9x their time, for the
  convergence, stability and health checks).
- New example, `slit_drainage`: liquid drained from a bead chain in a slit, from immersed
  beads through bead emergence and the band to the snap. Runs a coarse case live and shows
  stored full runs and movies for three bead spacings.

### Speed

- Quantity mode: the value and gradient passes of the fast element loop sum into
  per-thread buffers merged in thread order, instead of recording every entry for a
  serial replay (`g 10` in quantity mode at 98k facets, 8 threads: 0.83 -> 0.61 s;
  results unchanged and deterministic).

## 0.6.0 (2026-10-08)

### Speed

Cube sample, 8 threads, against 0.5.0 (`bench/benchmark.py`): `g 1` at 98k facets
75 -> 10 ms, at 1.6M 1.34 -> 0.40 s; a Newton step at 98k 0.49 -> 0.10 s, at 1.6M
25.5 -> 2.8 s. Against the original Evolver 2.70a see `docs/performance.md`
(Lagrange 6 Newton step at 24k facets: 22.9 s there, 1.42 s here).

- Newton steps: MUMPS factoring (instead of Evolver's own), with its analysis kept
  across steps; the matrix pattern kept across steps and filled in parallel; vertex
  normals in parallel; all volume and quantity constraint columns solved at once.
- Lagrange models: the facet setup as a packed, vectorized kernel; the area and
  volume element Hessians as BLAS matrix products; each element's terms projected
  and entered once.
- Gradient steps: parallel named quantities (facet, edge and vertex integrals);
  per-thread sums kept on separate cache lines; attribute selections kept until the
  surface changes.
- `u` and `V`: the swap tests and vertex-average search in parallel, identical
  results (393k facets: `u` 0.30 -> 0.022 s, `V` 0.60 -> 0.12 s).
- Builds prefer the system's OpenMP OpenBLAS and link it statically with private
  symbols (no clash with NumPy's OpenBLAS); wheels bundle OpenBLAS 0.3.34 (OpenMP).

### Threads

- Default thread count: physical cores; `pyse.set_threads()`, `pyse.threads_limit()`
  and `threads=` per call; `pyse.map` workers run single-threaded.
- Safe Ctrl-C with threads: aborts wait until the parallel region or MUMPS call ends.
- Results agree with Evolver's serial loops to round-off and are reproducible to
  round-off from run to run.

### Library

- `ev.body_surfaces(cap=True)`: closed per-body surfaces, openings capped on the
  constraint their rim lies on (points projected by Evolver); `BodySurface.cap_ids`
  and `cap_constraints`; `BodySurface.volume_mesh()` for tetrahedral meshes with
  Gmsh (physical groups for the surface, caps and body).
- Notebook display: HTML summaries of `Evolver`, `Mesh`, bodies and parameters
  (`jupyter` extra); faster live view; a warning before very large tessellations.
- `busy_timeout` / `EvolverBusyError`: optionally wait for a running call from
  another thread instead of failing at once.
- `pyse.map` fails fast with `WorkerStartError` when workers can't start.

### Breaking changes

- `Mesh` arrays renamed to `facets`, `facet_ids`, `facet_bodies`; unused low-level
  bindings (`edges`, `facets`, `element_nodes`) removed.

### Fixes

- Edges with a density in a soapfilm model with a metric crashed (stack overflow,
  also in the original Evolver); they now use the metric edge length.
- Switching `bezier_basis` in a Lagrange model gave wrong energies (stale cached
  basis tables).
- Double free after a failed Newton step; signed overflow in the sparse Hessian
  hash; `metis_factor` falling through to the MKL setup; METIS 5 API.
- The torus display mode no longer carries over to the next loaded datafile.

### Documentation and tooling

- Sphinx documentation (API reference, double-bubble tutorial, liquid-bridge and
  slit-droplet examples with mesh export), published to GitHub Pages by CI.
- Performance comparison with the original Evolver 2.70a.
- MIT license for pySE's own code; third-party license notices shipped in wheels.
- Sanitizer runs include an OpenMP build with Newton steps, `u` and `V`;
  `tools/valgrind.supp`.
