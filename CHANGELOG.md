# Changelog

## Unreleased

### Stress cases

- Four new cases (`bench/stress/`, and sections in the stress cases page), each on
  `relax()`'s defaults against an exact or converged reference:
  - 14, a drop on a sphere, its contact angle swept from 20 to 150 degrees as a
    parameter (exact spherical caps): energies within 0.2%. It found the defect
    fixed below.
  - 15, a double bubble from two boxes, the second volume shrunk to 0.2 (the standard
    double bubble): area and pressures within 0.2%.
  - 16, a solder joint's force on its pad (Brakke's bga-3) against his axisymmetric
    model (bga-30): both converge at second order, the extrapolations agree to 5e-5.
  - 17, a catenoid between round wires pulled past its fold with
    `continuation(stop_on="unconverged")`: areas within 0.04%, stopped at the first
    step past the fold.
- `docs/stress_helpers.py` has the exact references (`drop_on_sphere`,
  `double_bubble`).

### relax()

- Gradient rounds that creep without settling (a contact line moving slowly over a
  curved wall) used up `max_iter`, and Newton then gave up after one slow step:
  stress case 14 ended most steps unconverged. Now, once the rounds have used half of
  `max_iter` with the residual falling by less than 5% a round (three rounds
  running), Newton is tried there; kept if it converges or cuts the residual a
  hundredfold, otherwise the surface goes back and the rounds go on (a neck
  shrinking toward a pinch-off creeps the same way, and must go on: case 2). Case 14
  now converges at every step, in half the time; every other case gives the results
  it gave before. Tried and dropped on the way: counting an energy drop as Newton
  progress (case 2 failed past its fold), a budget of its own for the
  conjugate-gradient fallback (it broke `max_iter`), and the trial without the
  half-budget condition (fewer tidying rounds cost accuracy: the gravity puddle
  went from 0.22% to 0.54%).
- The split of long edges leaving a contact line (after convergence) repeats, up to
  three times per call, while some are long: a contact circle that shrinks a lot in
  one step (a drop from 160 to 170 degrees) needed more than one. Stress case 3 now
  passes at 170 degrees (the implied contact angle 1.3 degrees off, was 1.7; the
  limit is 1.5), so 13 of the 14 cases pass on defaults; only the drainage band
  (case 8) doesn't.
- `relax()` warns when the surface was refined four or more times since it last
  converged: relaxing that fine is slow and may not converge (`cube.fe` refined five
  times: 10 s unconverged, against 0.4 s converged with `relax(levels=5)`).
- Stress case 8's unconverged states traced: in a typical band state 95% of the
  residual (squared) sits in ten vertices at the corner where a bead's contact line
  meets the wall, with facets under a degree and vertices pushed into the bead. A
  local mesh breakdown, which only remeshing there fixes (not done by `relax()`).

### Documentation

- Figures for stress cases 13 to 17.

- The site reads as one path: grouped navigation (getting started, guides,
  examples, Brakke's samples, reference), every page saying what it adds and
  pointing to the next, conventions in one place (Getting started), the examples
  on the current API, and README's threading notes moved to the performance page.

## 0.7.3 (2026-10-12)

Replaces 0.7.2, whose x86 wheels and source distribution did not build. Faster
Newton steps and Lagrange stages, fewer silent mistakes when building surfaces, Brakke's
solder-joint series, and fixes to three bugs in Evolver itself.

### Fixes

- The 0.7.2 test of `relax()` on `tankex.fe` at one refinement level asserted a
  stopping energy that depends on the thread count (0.7.1 failed it too, on one
  thread); it now asserts what is guaranteed: no slide, volume met, failure
  reported. The 0.7.2 wheels for x86 and the source distribution did not build
  because of it.

### Breaking changes

- `ev["name"] = value` assigns only parameters and variables that exist (a typo
  raised nothing and created a new variable); `ev.define(name, value)` makes a new
  one. `continuation()` with a misspelt name raises `KeyError` with suggestions.
- `Body(volume=..., faces=None, ...)`: `volume` comes first, and `faces` defaults to
  all faces. `volconst` defaults to the sum of `Constraint.volconst` over the walls
  from `pysurfaceevolver.constraints` that the body touches, and with a single body
  over untouched ones too (a solid immersed in the liquid); explicit values win.
- Vertices on a `boundaries=` curve are fixed (free, they bunch up along the wire).
- `continuation()`'s `Step.energy` is `Evolver.energy` (with the walls' constants).

### Found in review

- Two more Evolver bugs fixed. Popping a fixed edge between two free facets divided
  it and retried without end, so `relax()` (which pops by default) hung; it was so in
  0.7.1 too. Trimming the warning-message buffer could read past its text.
- Vertex averaging's new ring cache moved vertices fixed after it was built;
  `energies()` counted an instance's modulus twice; the fused Lagrange energy pass
  ran before the pressures were recomputed, which mattered for an energy reading a
  pressure (now skipped there).
- Bulk writes on a second element type failed (Evolver's attribute names are
  global): the scratch attributes are now per type (`pyse_vertex_mark`, ...); they
  are written into dumps.
- `make_datafile`: a closing wall that several bodies touch (or, with several bodies,
  none) is no longer added to each body's `volconst`, with a warning; several bodies
  must each give their faces; `optimizing_parameter` angles and exponents in angle
  expressions pass the parameter check; per-vertex constraints take numpy integers
  (0: none); single-letter names are refused; string-model bodies are oriented
  counterclockwise.
- `health()` checks walls no vertex is on again (only constraints that only fixed
  vertices are on are skipped).
- `continuation(stop_on=...)` also stops where `relax()` pinches the surface off.
- `restore()` brings back the snapshot's walls and name; relax()'s memory of failed
  attempts only for the same load.
- Warnings from `relax()` callbacks keep their location, and `-W error` no longer
  replaces an error from `relax()`.

### Solder joints (Brakke's ball grid array series)

- Vendored `bga-2.fe` to `bga-10.fe`, `bga-30.fe` and Brakke's force command files
  (`zforce.cmd`, ...); a *Solder joints* notebook: the force on a pad (three ways,
  against Brakke's variational integral), force against height with the unstable
  range, the height a loaded chip floats at, and a chip pulling itself into line.
  (`bga-11.fe`, `bga-12.fe` and `bga-31.fe` run away even with Brakke's own
  recipes in Evolver 2.70, so they are left out.)
- `residual()` counts the force on optimizing parameters, so `relax()` balances
  them; `Parameters.optimize(name, on=True)` frees or fixes one.
- An Evolver bug fixed: reading a datafile reset the pressures for every named
  quantity without a Lagrange multiplier, energies included, so after a restore
  Newton refused to run ("Pressures invalid") on surfaces with an energy quantity
  and fixed volumes, inside `relax()` too.
- `relax()` undoing a step puts optimizing parameters back with the coordinates.
- `derivative(name, move=..., method="newton" | "fixed" | "relax")`: with `move`
  (how the vertices follow the parameter) Brakke's `zforce5` (Newton re-converged,
  the default) or `zforce4` (no relaxing, pressure-corrected); they agree with his
  variational integral to 8 digits. The surface, multipliers included, comes back.
- `relax(levels=n)` refines past a start with nothing free to move.
- `health()` no longer reports pad rims or display solids (equality constraints only
  fixed vertices are on) as walls, nor fixed vertices as crossing them.
- `relax()` passes on each kind of Evolver warning once, with how often it came up
  (19,430 "doesn't converge to constraint" lines became one).

### Faster

- Against 0.7.2 (2 threads, energies identical to 15 digits): a Newton step on a
  linear cube of 98k facets 313 -> 197 ms (393k: 1.46 -> 1.14 s), `relax()` 1.14 ->
  0.96 s; Lagrange 2, 98k facets: gradient step 373 -> 313 ms, Newton step 1.44 ->
  1.19 s, residual 255 -> 205 ms, `relax()` 5.25 -> 4.09 s. From: no serial normal
  in the Hessian setup for vertices on no constraint; Lagrange gradient and value
  kernels with their per-point factors computed once; Lagrange vertex normals from
  the per-topology normal cache; vertex averaging from per-topology rings (bit-identical).
- Gradient steps with named quantities only (Lagrange model, `convert_to_quantities`):
  9 passes over the elements instead of 14 (energy gradients with the volume
  gradients, the energy with the last volume projection pass; bit-identical).
  Lagrange 2, 98k facets: `g 1` 330 -> 292 ms; Lagrange 4, 24k facets: 218 -> 196 ms.

### Fewer silent mistakes

- A body whose faces point into it (negative starting volume against a positive
  target) warns when it loads; it used to turn inside out and report "converged,
  stable, health ok". So does a wetted plane whose normal points into the solid
  (the drop came out at 180 degrees minus the contact angle).
- `SharedEngineWarning`: an `Evolver` handle used after another handle loaded a
  surface (every handle is a view of the one engine).
- `make_datafile` errors name the mistake: faces given 1-based, a vertex put on an
  undefined constraint, `z == 0` for `z = 0`, a contact angle naming an undeclared
  parameter. `refine(-1)` raises; `parameters["typo"]` suggests names;
  `load_string` surfaces are named `surface.fe`, not a temporary path.
- `relax()` keeps the "Hessian not positive definite" warnings of its trial Newton
  steps to itself (it checks stability at the end); `restore()` those of loading its
  own dump.

### Simpler building and measuring

- `pyse.shapes`: `cap`, `disc`, `tube`, `revolve` and `box` starting shapes with
  named rims. The start needn't have the target volume.
- `Evolver.energy`: the total energy plus the wetting energy the walls' line integrals
  leave out (`Constraint.energy_constant`, also when the contact angle is a
  parameter); `make_datafile` records it.
- Constraints may be named (`{"table": C.plane(...)}`); `vertex_constraints` may give
  each vertex's constraints; `boundaries=` with `constraints.circle()` or `curve()`
  for films on curved wires (refining keeps the wire round).
- `load_arrays` remembers its walls: `plot(walls=True)`, `live_view(walls=True)`.
- `contact_lines(k)`: where the surface meets constraint k, as ordered polylines.
  `BodySurface.volume`, `.area`, `.areas()`; `Mesh.facet_areas()`, `edge_lengths()`,
  `edge_valence()`.
- `derivative(name)`: the derivative of the equilibrium energy with respect to a
  parameter (minus it: the force on a wall it moves; Brakke's "reconverge" method).
- `energies()`: the energy term by term, Brakke's datafile check.
- `continuation(stop_on="unconverged" | "unstable", halvings=k)`: stops at a fold
  with the surface back at the last good step, after retrying in smaller steps.
- `stability(eigenvalues=True)` reports the lowest eigenvalues when stable too.

### Simpler code

- `_evolver.py` (2,750 lines) is split into modules: errors, results, views,
  geometry, engine, analysis, relax and export. `relax()` keeps its settings and
  stages in one object per call instead of passing ten arguments around; magic
  numbers are named constants.
- Bulk writes (`set_flag`, `set_constraint`, `set_values` of other attributes,
  `remesh(protect=)`) are one `set ... where` command over scratch attributes
  instead of a statement per element (fixing half of 12k vertices: 4 ms).

### Tried and dropped (Brakke's tips, measured)

- Storage reordering along a space-filling curve (`reorder_storage`): Newton steps
  10-18% and vertex averaging 40% faster, but gradient steps and the residual 1.6-2.3x
  slower, on cubes of 98k and 393k facets.
- Conjugate gradients in the gradient rounds, history cleared after each tidying:
  fewer steps in several stress cases, but the catenoid case fails (degenerate
  facets) and the bridge case is slower.
- `sobolev_mode` Newton steps (positive definite) before the gradient rounds: the
  catenoid case fails and most cases are slower.
- `immediate_autopop`: coarsening of `100grain.fe` slows down a lot (1185 vertices left
  after 20,000 steps against 712) with no sign the default does harm; not switched on.
- Lowering `target_tolerance` for tiny volumes (`cbga1.fe` in cgs units): volumes
  were met exactly either way.

### Documentation and examples

- A *Getting started* page: a drop on a table end to end, and the conventions
  (normals out of the liquid, plane normals into it, spheres and cylinders solid;
  `ev.energy` against `ev.total_energy`; a surface tension other than 1; contact
  angles first order in the edge length; 1-based Evolver numbers against 0-based array
  rows). The API reference lists `pyse.shapes`, `constraints.circle`/`curve`/`Boundary`,
  `Stability` and `SharedEngineWarning`.
- The notebooks, their helpers, the stress cases page and the stress suite use the
  new API: starting shapes from `pyse.shapes` (the copies of `cap`, `tube` and
  `revolve` in `docs/stress_helpers.py` and `bench/stress/common.py` are gone),
  `Body(volume=...)` without face lists or hand-summed `volconst`, `ev.energy` instead
  of adding wall energy constants by hand, `load_arrays` with `walls=True` in plots,
  per-vertex constraint sets, `BodySurface.volume`/`areas()`. Starts are no longer
  fitted to the target volume by bisection (the liquid bridge, the slit droplet, the
  drainage, stress case 5): relaxing meets it. Results agree with before to the
  discretization error (the bridge's forces within 0.3%).
- The Brakke notebooks no longer silence `EvolverWarning` globally; `brakke_helpers`
  silences only the instability warning of `relax()` (its table reports stability)
  and Evolver's messages during a sample's own recipe.

## 0.7.2 (2026-10-12)

Faster `relax()`, a simpler API (with removals, listed first) and Brakke's online
examples.

### Breaking changes

- Removed `ev(...)` (use `ev.command(...)`) and `ev.hessian()` (an unguarded Newton
  step; use `ev.newton()`, or `ev.command("hessian")`).
- `relax()` no longer takes `cg`, `window`, `energy_tol` or `pinch`: the defaults
  were the only settings worth having (the conjugate-gradient retry stays, see below).
- `recipes.continuation(ev, what, values)`: `what` is a parameter or variable name,
  `ev.body(i)` (its target volume), `ev.quantity(name)` (its target) or a callable;
  `recipes.body_target()` and `recipes.parameter()` are gone.
- `stability()` reports `lowest` and `threshold` with Evolver's `linear_metric`
  (eigenvalues of the continuous surface, which don't shrink as the mesh is refined;
  the counts are unchanged).

### Faster

- `relax()` against 0.7.1 (2 threads, same energies to 12 digits): linear cube,
  98k facets 1.60 -> 1.09 s, 393k facets 9.7 -> 7.0 s; Lagrange 2, 98k facets
  8.3 -> 5.4 s; Lagrange 4, 24k facets about 7 -> 4.2 s. From:
  - the Lagrange stage's Newton steps stop once the residual is below `tol` (two or
    three of four or five were in round-off);
  - the stability check takes the negative-eigenvalue count of the last Newton step's
    factorization when that step began within a residual of 1e-2 and the surface is
    as it left it (it matched the full check in all of some 230 relaxes: stress cases,
    tests, Brakke's samples); otherwise the full check as before;
  - vertex normals in the plain linear film (the residual, the Hessian setup) from
    facet normals computed once per facet and the facets around each vertex kept per
    topology: the residual 3x faster (checked against Evolver's own normals under
    `PYSE_CHECK_FACET_CACHE`);
  - no second residual after Newton; `quantities()` skips its recalculation when
    there are none; `health()` reads flags and valences from the mesh.
- `restore()` (reading datafiles) about 20% faster: the lexer reads with
  `getc_unlocked`.

### More robust

- `relax()` equiangulates once after the first gradient step of each phase: a coarse
  start triangulated badly (Brakke's `cbga1.fe`, a prism of quads split at their
  centres, whose recipe swaps the edges by hand) used to settle at an energy of 11
  instead of 5.3, and `relax()` ended degenerate with the volume 49% off; it now
  converges at every level. Stress suite: same results, fewer steps (case 6: 60 -> 44
  Newton steps).
- Tried and dropped: a watch on the scale factor ("a scale factor going to 0 does not
  mean convergence", Brakke's hints) stopped `relax()` from recognising settled
  energies (the scale legitimately falls more than 1000-fold within a phase); fewer
  vertex averagings inside the conjugate-gradient retry rescued no more steps. The
  retry stays: without it the drainage case is ~30% faster but its median error
  doubles.
- Two topology tests (octa's popping, cat's pinch-off) run with one thread: their
  path varies at round-off with several threads (also in 0.7.1), and so, now and
  then, the final state.

### New

- `ev.quantity(name)`: a live handle on a named quantity (`value`, `target` settable,
  `None` frees it, `modulus`, `pressure`, `kind`). `ev.gravity`, `BodyView.density`,
  `make_datafile(gravity=)`.
- `ev.saddle()`: steps off a saddle along its unstable mode (Evolver's `saddle`, with
  the gradient step it needs first); `ev.relax()` from there finds the minimum it
  leads to. `ev.pop()`: Evolver's `o`, public.
- `ev.contact_angles(constraint)`: the contact angle at each contact-line vertex, from
  the facet normals and the constraint's gradient (first order in the edge length).
- `Mesh.slice(normal, point)`: a cross-section as ordered polylines.
- Constraints from `pysurfaceevolver.constraints` keep their geometry:
  `Constraint.to_pyvista()`, and `ev.plot(walls=[...])` / `ev.live_view(walls=[...])`
  draw them with the surface.
- Torus models plot each facet whole, unwrapped (they were stretched across the cell).
- `ev.write()` writes glTF (`.gltf`, `.glb`) and VRML (`.wrl`) scenes through PyVista;
  PLY files no longer trigger meshio's 64-bit integer warning.
- `make_datafile` orients a body's faces itself when they close a surface (consistent
  and outward); otherwise all +1 as before.
- Eight of Brakke's online examples ship with the samples: `cbga1.fe`, `cbga2.fe` and
  `bga-1.fe` (solder joints), `pcell.fe`, `dcell.fe`, `IWP.fe` (periodic minimal
  surfaces), `TwoSpheres.fe` and `TwoSpheres-explicit.fe`, with tests and notebook
  sections. Stress case 13: Schwarz's P and D surfaces against their standard areas
  (extrapolated to 1e-6 and 2e-5).

### Docs

- The notebooks use the new API instead of hand-written helpers (contact angles,
  cross-sections, walls) and raw commands; the tutorial's bodies need no orientation.

## 0.7.1 (2026-10-11)

The v0.7.0 tag's aarch64 wheels failed to build; 0.7.1 is the first release with
wheels for all platforms, and carries the fixes below.

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
