# Changelog

## Unreleased

### API

- `constraints.sphere()`: a coordinate of the centre may be a datafile parameter's name
  or an expression (`(0, 0, "1 + gap/2")`), so setting the parameter moves the sphere and
  `recipes.arclength` can follow it (a bridge pulled apart at a fixed volume, say). The
  volume over a wet pole is then in the line integral (the pole's value taken off inside)
  instead of `volconst`, which can't follow a parameter; the same bridge comes out the
  same to 2e-6 in energy, converging alike under refinement. Both poles wet needs a
  centre of numbers. `plot(walls=True)` draws such a sphere where it is.

- `pyse.Evolver(process=True)`: a surface in a worker process of its own, so several
  surfaces at once. The handle works like any `Evolver`: attribute reads, assignments
  and method calls go to the worker (a round trip is about 0.04 ms plus copying),
  views such as `ev.body(1)` and `ev.parameters` are handles too (`recipes.continuation`
  works on them), and results come back as copies. Prints, warnings and errors from the
  worker show in the caller, Ctrl-C interrupts the worker's call, callables given to
  methods run in the worker, and `plot()` draws in the caller. `threads=` sets the
  worker's threads; `close()`, `with`, or the last handle going away ends it, and a
  dead worker raises `WorkerCrashed`. The worker is a new interpreter
  (`python -m pysurfaceevolver._worker`), so scripts need no `__main__` guard.
  POSIX only. Errors keep `errnum` and `output` when pickled.

- `ev.newton()` returns an `IterationResult`, as `relax()` does, instead of the number
  of steps kept: `newton_steps` (the steps kept), `residual`, and `converged` (the
  residual below `tol`, or 1e-8 without one, with the volume and quantity targets met).
  Breaking: code using the number reads `ev.newton(...).newton_steps`.

- `ev.modes(n=1)`: the `n` lowest Hessian eigenvalues with the constraints held
  (`linear_metric`, as `stability()` reports them) and their modes as displacements
  per vertex, `Modes(values, vectors)` with `vectors` shaped (n, vertices, 3), each
  scaled to a root mean square of 1 and signed so its largest component is positive:
  a simple eigenvalue's mode comes out the same on any thread count. Evolver's `ritz`
  below the lowest eigenvalue, each vector turned into vertex motion through Evolver's
  own move (boundaries, constraint projection), undone. Unlike `saddle()`, the surface
  is left as it is and the mode carries no line search's even part.

### Continuation

- `recipes.arclength(ev, what, ds, steps, bounds=)`: pseudo-arclength continuation,
  following a branch of equilibria through a fold (where `continuation()` stops) and on
  along the unstable branch. Each point is `ds` along the branch from the last (the
  value's change and the vertices' RMS displacement together), predicted along the
  secant and corrected by Newton's method on the equilibrium plus the arclength
  condition, built from two of Evolver's Newton steps per iteration (bordering with
  its own factorization; no gradient steps, which would leave an unstable branch).
  Between points the vertices are averaged along the surface and edges flipped (Newton
  steps move vertices only along the normal; without this the error grew to 0.5% and
  the run broke down before the fold); the vertices themselves stay the same, and the
  step is measured along the surface's normals (the tidying's sliding isn't a step;
  counting it, a run turned back at the bridge's fold). Steps yield
  `result.negative_modes`, counted with the constraints held (`eigen_counts`). On the catenoid between round wires (stress case 17)
  it turns at the fold and follows the unstable catenoid within 0.03% of the exact
  areas. On a liquid bridge between two spheres (gap 0.6, 40 degrees) shrunk at fixed
  volume it passes the rupture fold (V 0.07306 against the axisymmetric 0.07307,
  refined four times) onto the thinner-necked branch within 5e-4 of the reference
  energies, and finds one negative mode from a neck of about 0.13 (V 0.08), before the
  fold, steady under refinement: an axisymmetric mode odd about the mid-plane (Vogel's
  small volume instability; see "Where a shrinking bridge breaks" below); following it
  with `saddle()` lowers the energy. Stops with a warning when a step fails after
  halving or the mesh degenerates.
  The two Newton steps of each iteration start from the same Lagrange multipliers
  (pressures), and the corrector carries them along with the vertices: Evolver's Newton
  step uses them in its Hessian and updates them, so the second step had started from
  the first one's, and away from an equilibrium the change with the value came out
  several times too large. The corrector had then converged slowly or not at all, and
  the bridge above (8192 facets) stopped at its fold in about half the runs; now each
  point takes 3-5 iterations at the full `ds` and 8 of 8 runs pass the fold, in 7 s
  instead of about 30. Undoing a step (in `relax()` too) puts the multipliers back with
  the vertices: the stress cases and 40 of the 42 sample datafiles that finish
  `relax(levels=2)` come out the same (cat.fe varies run to run either way); bga-6, an
  overloaded pad with no equilibrium, undoes 20 steps instead of 12 before giving up
  (5 s instead of 1).

- `arclength(..., along=mode)`: the first step goes along a displacement per vertex
  (such as `ev.modes(1).vectors[0]`) at the starting value instead of in the value,
  onto a branch that crosses there, and the run follows that branch. From the first
  unstable point of a shrinking bridge (a mode turning negative without a fold), it
  follows the off-centre bridges, with pressures within 1e-4 of the exact ones and one
  negative mode throughout; the two signs of `ds` give mirror images (the rupture page
  has the example). Not found for the packing page's pore: its symmetric branch stops
  before its triply degenerate mode turns negative.
- `arclength()` past folds and soft modes: a corrected point farther from its
  prediction than the step itself is a failed step (halved), not a point (a pore's
  liquid, stepped 0.01 in volume, had landed at -0.49); a point that doesn't settle
  again after tidying is tried with the averaging undone, and failing that it is a
  failed step too: the run goes back to the last point by the corrector along the same
  tangent and halves the step, where it used to stop. A catenoid on a coarse mesh in
  steps of 0.3 now turns at its fold (it stopped at wires 1.1 apart); a trimer between
  three beads at 5 degrees goes through its snap-in fold and on (it stopped there).
  A remeshed surface that doesn't settle is put back and the run goes on.
- `arclength(..., remesh=h)`: keeps the edges in range on a branch that stretches or
  shrinks the surface, with `ev.remesh(target=h)` between points when an edge has left
  0.5h-1.6h (or a dict of `remesh()`'s arguments, or a function of the surface called
  between points). The last step is carried over to the new vertices (by vertex id, and
  for a split edge's midpoint the mean of its neighbours') and the remeshed surface is
  brought back to the branch by the corrector. A catenoid between round wires, through
  its fold and down the unstable branch: without remeshing its facets degrade and at
  wires 0.85 apart the run is back to no negative mode with areas 30% off; with
  `remesh=h` it follows the unstable catenoid to wires 0.43 apart within 0.4% of the
  exact areas.

- `continuation()` predicts each step: from the third step on, relax starts from every
  vertex extrapolated from the last two steps (a secant predictor), which saves a fifth
  to a half of the gradient steps. Over the stress cases that step a value, the gradient
  steps went from 4820 to 2970 (case 2), 3010 to 1690 (case 6), 3116 to 1606 (case 12),
  3410 to 2520 (case 17) and 5870 to 5110 (case 14), with the same results. It applies
  only while the vertices stay the same (not after a refinement, remeshing or pop, in
  relax or in the loop), and not to a step more than twice the last (a jump such as
  1, 10, 100, where a straight line would overshoot). A predicted step that comes out unconverged is relaxed again
  from the last step's positions before `stop_on` or halving sees it. `Step.predicted`
  says whether a step kept the prediction; `predictor=False` turns it off.
- Tried and not adopted for stress case 8 (the drainage band on defaults, without
  remeshing): extrapolating the band pushes the vertices squeezed by the receding
  contact line further, and over 3 paired runs the worst errors went from 4-9% to
  170-1550%, with the snap 15-100% off (from 3-4%).

### Speed

- Newton steps: a sparsity pattern factored six times is analysed again with MUMPS's
  PORD ordering instead of AMD (slower to analyse, less fill). At 4 threads the steps
  after it are 10-15% faster in Lagrange 2 at 98k facets, Lagrange 4 at 24k and the
  linear model at 1.6M, and about the same at 393k facets and below (its extra
  analysis is 0.1 s at 98k, 1.25 s at 1.6M). `relax()` takes 2-3 steps on a pattern and
  is unchanged; `arclength()` takes 6-9 a point: six points in Lagrange 2 at 98k
  facets take 37-40 s instead of 41-42. Each datafile load starts again with AMD, so a
  run's results don't depend on what ran before it.
- The quadratic model's facet tangents and Gauss points, and the Lagrange model's edge
  setup, multiply their small matrices inline instead of through `mat_mult` (the same
  sums in the same order: bit-identical). Quadratic model at 98k facets, one thread:
  `g 1` 0.64 -> 0.47 s, a Newton step 2.44 -> 2.35 s.
- Not changed: `r` stays serial (refining 393k facets to 1.6M takes 1.4 s, less than
  half a Newton step there; about 40% of it is zeroing new elements in fresh memory,
  the rest walking edge loops: no single hotspot).

### Documentation

- "Liquid in a sphere packing" (`docs/packing.md`): the curvature at which gas passes
  the throat between three spheres of a layer (contact angles 20-60 degrees, against
  the Mayer-Stowe-Princen estimate: 2% below to 6% above), a trimer between three
  beads drained until the menisci in its throat meet (Semprebon et al.'s burst; at 20
  degrees it comes before any fold at gaps 0.09-0.25), and a pore full of liquid
  between four spheres, whose curvature peaks at its throats' entry curvature (9.886
  against 9.871) as a triply degenerate mode goes soft. Starting surfaces from flat
  patches between the spheres in `docs/packing_helpers.py`; figures by
  `tools/packing_figure.py`; checked by `tools/check_doc_snippets.py` (now a default
  page). At 5 degrees, the paper's angle, relaxing these starts slides vertices through
  the spheres (in Notes and limits). The `docs` extra now includes SciPy (the notebooks' reference
  helpers use it).

- "Liquid in a sphere packing": which throat a pore empties through. `pore_wedge` relaxes
  the full pore with a three-fold axis through one sphere and the opposite (bottom)
  throat, and cuts it down to the sixth between two mirror planes, which holds the
  bottom throat against the other three exactly (the full pore's mesh had split the
  triply degenerate mode). With the bottom throat the widest (gaps 0.03 against 0.02),
  the curvature peaks at 9.533, that throat's entry curvature (9.520 in the layer); its
  meniscus then goes on into the pore while the others pull back, and the branch turns
  at a fold: the pore empties through it. With it the narrowest, the other three go
  first (at 10.002). `bottom_meniscus` and the cut in the helpers (`_pore_surface`
  shared with `filled_pore`, unchanged).
- The 5-degree trimers near Semprebon et al.'s kink, on a finer mesh: the snap-in at
  gaps 0.13-0.15 with the throat's menisci meeting just past it, and at 0.11 a burst
  with no fold before it, so the change lies between gaps 0.11 and 0.13 near volume 0.1
  (the paper: 0.125 and 0.09); the figure shows both lines.
- Docs CI: the notebooks run side by side (`sphinx-build -j auto`, one thread each: 3
  minutes locally instead of 9 of notebooks), and the snippet pages in two jobs, each
  running its blocks four at a time (`tools/check_doc_snippets.py --jobs n`, which
  queues every page's blocks at once and reports them in order: all pages in 6 minutes
  locally instead of about 15).
- "Liquid in a sphere packing": a section on trimers at 5 degrees, Semprebon et al.'s
  angle: `arclength` with `remesh=` follows the trimer through its snap-in fold, along
  the unstable branch, through the pop-out fold and down the dimer branch (gap 0.17:
  bistable between volumes 0.185 and 0.224), and the two folds, mapped over gaps,
  meet in a cusp at about volume 0.267 and gap 0.226 (the paper: 0.26 and 0.22).
  `meniscus_distance` in the helpers; figures by `tools/packing_figure.py cusp`. The
  notes no longer say 5-degree starts fail.
- "Where a shrinking bridge breaks": a section following the off-centre bridges with
  `arclength(along=...)` against the exact ones. `tools/check_doc_snippets.py` runs a
  block starting with `# continuing` after the page's blocks before it.
- `bridge_spheres_branch` (docs helpers) weighs the neck's sliding by the neck's
  curvature: unweighted, the slope it solves for also vanishes at the cylindrical
  bridge (neck R cos(theta), in range above about 53 degrees), where sliding the neck
  moves nothing. That was the unexplained second zero at 60 degrees, where Evolver
  found no instability. The branch points it returns are the same.

- Liquid bridge example: "How far a bridge stretches". Bridges of the volumes that break
  at gaps 0.4 and 0.7 (20 and 60 degrees, from the exact axisymmetric bridges) pulled
  apart with `arclength` in the gap, the beads following the parameter: at 20 degrees
  they end at the fold (the neck pinches), at 60 degrees at the branch point well before
  it (the liquid goes to one bead); Evolver's widest stable bridge within 0.01 of the
  exact gap, Lian et al.'s rupture distance within 5%. The notebook's beads follow
  `gap` throughout. `bridge_spheres_branch` in `docs/stress_helpers.py` returns the branch
  point where shrinking bridges lose stability: at 60 degrees there is a second, at
  large volumes, where Evolver finds none, and the helper had taken that one. References
  to Surface Evolver work on liquid in bead packs on the bridge, slit drainage and
  bridge rupture pages.

- "Several surfaces at once" (`docs/several_surfaces.md`): `save()`/`restore()`,
  `pyse.map` and `Evolver(process=True)` compared, with an example of four drops relaxed
  and shrunk side by side from threads (twice as fast as one after another), what
  crosses between the processes, and the limits. Its example is checked by
  `tools/check_doc_snippets.py` with the stress page's.
- The stress page's catenoid between round wires goes past the fold with
  `recipes.arclength` (a runnable section).
- "Where a shrinking bridge breaks" (`docs/bridge_rupture.md`): a liquid bridge between
  two spheres (gap 0.6, 40 degrees) shrunk with `recipes.arclength` turns unstable at
  volume 0.079, 8% above the fold at 0.0731 where an axisymmetric calculation puts its
  end, and the unstable mode moves the liquid toward one sphere. An axisymmetric
  reference without the mirror symmetry checks it: bridges with the neck off the
  mid-plane branch off at volume 0.07895, exist only at larger volumes and have more
  energy than the symmetric one (a subcritical pitchfork; T. I. Vogel's small volume
  instability, 2013). New helpers in `docs/stress_helpers.py`
  (`bridge_spheres_fold`, `bridge_spheres_branch`, `bridge_spheres_offset`), a figure
  drawn by `tools/bridge_rupture_figure.py` along with pictures of the bridge along
  the run, its unstable mode and the off-centre bridge in 3D, and the page's code
  checked by `tools/check_doc_snippets.py`. The unstable mode comes from `ev.modes()`.

- A shorter README (92 lines, was 261): what pySE is, a picture of four relaxed
  surfaces, install, one example with its output, and where to read on, with badges.
  Its API tour moved to the docs as "A tour of the API" (`docs/tour.md`), and building,
  testing and the code layout to "Development" (`docs/development.md`).
  `tools/readme_figure.py` redraws the picture.

### Fixes

- `arclength()` in the Lagrange model and for strings: tidying between points
  equiangulated, which Evolver refuses there ("Cannot equiangulate LAGRANGE model"), so
  the first point failed. It now tidies as `relax()` does (vertex averaging only). And
  on fine Lagrange meshes the corrector waited for steps below `tol` that round-off
  doesn't allow: in Lagrange 2 at 98k facets the vertices slide 2e-8 to 1e-7 a step
  along the surface with the residual at 1e-9, so points failed, steps halved and runs
  stopped after one or two points, at different points from run to run. A point is now
  also converged when it is in equilibrium (residual below `tol`) and the steps have
  stopped shrinking; a run of six points takes 38 s there.
- The edge force of a string with a conformal metric (`CONFORMAL_METRIC`), fixed on
  2026-10-08 (the metric's derivative term lacked a 1/4) but thought unreachable, is
  reached by the linear string model outside quantities mode, and now tested against
  the energy's gradient (`metric.fe` made conformal; the old formula was off by order
  1 near the throat).

- `arclength()` stopping at a point that doesn't settle after tidying leaves the surface
  at the last point (its vertices on the tidied mesh, Newton there), not at the
  unsettled one: a catenoid stepped past its fold was left with its wires 1.45 apart.
  `_core.multipliers()` is typed (a float64 array) in the stubs.

- `arclength()` halves its first step too, as it did the later ones (up to `halvings`
  times), instead of stopping with "try a smaller ds"; the attempts are undone from a
  copy once tidying has flipped edges. Finer meshes and shallow angles had needed a
  smaller `ds` by hand (the packing page's pore refined once, a throat at 20 degrees).

- Small contact angles: a surface that `relax()` leaves unconverged with vertices sunk
  into a wall from `constraints` whose contact angle is below 15 degrees is relaxed
  again from the call's start with those angles raised to 20 degrees, then lowered in
  stages (by 0.7 each) to their own, kept only if that converges. At a small angle the
  liquid meets the solid almost tangentially, gradient steps from a crude start slide
  the vertices next to the contact line through it, and the wall's integrals reward
  that (the energy fell without end). Trimers between three beads at 5 degrees (gaps
  0.25 and 0.30) and a bridge from a fat tube, which sank, now converge, within 2e-4
  of the same surfaces lowered by hand (the bridge within 7e-4 of the exact one).
  Surfaces that relaxed before come out the same, with the same speed. For this, a
  wall's contact angle given as a parameter, or as a number below 20 degrees, enters its
  integrals as `maximum(angle, pyse_angle_floor)`, a datafile parameter declared at 0
  (`Constraint.floored` says which); such a wall's `energy_constant` is NaN, as for a
  parameter angle, and `Evolver.energy` adds the left-out part.
- `relax(levels=n)` starts again one level finer when its last level ends
  unconverged: from the call's start refined once, then the levels as before (kept only
  if that converges). A coarse start can hold equilibria the shape doesn't have, and
  which one the gradient rounds found was chance: a 20-degree bridge from a tube 12
  facets around, with its volume moved by a few units in the last place, failed 6 times
  in 13 (the "different builds" of the bridge notebook: different SciPy versions
  computed its volume 1 ulp apart); now all 13 converge. Stress cases unchanged; the
  sample datafiles at levels 1 and 2 converge as before, to the same energies (those
  that don't, try the finer start too: up to 1.7 s more, tankex.fe).
- `relax()` leaves a surface that is already in equilibrium as it is (with `levels=0`:
  residual and target gaps below `tol`, no degenerate facet, also after Evolver's
  `recalc`, which puts the vertices on wires moved by a parameter): only the stability
  check runs. Its gradient rounds were mesh motion only, and next to a shallow contact
  line they slid vertices through the wall: a bridge between beads at 20 degrees,
  relaxed a second time, ended unconverged with 14 vertices inside a bead (175 after
  300 plain gradient steps, the energy falling throughout), and `arclength()`, which
  relaxes first, failed from it. Tests, the 43 sample datafiles and the stress errors
  are as before; stress case 6's continuation, whose first step re-relaxed the
  converged barrel, now needs a retry at 60 degrees (2020 gradient steps instead of
  1690).

- `relax()` with a body volume or fixed quantity far from its target (more than 25%):
  if the first gradient step leaves degenerate facets (below 1 degree) the surface
  didn't have, it is undone and the target approached in stages of at most 10%, each
  with 20 tidied gradient steps. Evolver's projection onto a far target moves each
  vertex along its volume gradient, several edge lengths and unequally where the mesh
  is irregular, and folded facets: drops at 30-120 degrees taken from volume 1 to
  0.25-4 in one relax() failed 15 of 72 times (3 perturbed starts each), now 9 (the
  rest mostly a contact line receding far, as stress case 8: a continuation or
  remeshing). The trial step is undone otherwise (coordinates and step size), so
  other surfaces relax exactly as before (same stress results and gradient steps).
  Staging every far target instead lost coarse starts (stress case 5's bridge, 2.6
  times its volume: its contact lines drift during the stages); not with optimizing
  parameters (bga-5's loaded pad has no equilibrium at the smaller volumes on the way).

- With optimizing parameters, constraint projection takes one more step for a vertex
  that ends within `constraint_tolerance` of its constraints but not well inside it
  (above 1e-3 times the tolerance): projection converges quadratically, so the vertex
  lands on them to rounding. Left up to the tolerance off, the vertices stalled
  Newton's residual on fine meshes: `bga-5.fe` refined four times ended "not converged"
  at a residual of 2.6e-7 after 11 s; it now converges (6.5e-10) in under 1 s. The
  limit was blamed on Evolver's finite-difference parameter Hessian; varying its step
  (`pdelta`) didn't move it. Surfaces without optimizing parameters project as before:
  applied to all, the extra step left stress case 8 (the remeshed drainage band)
  unconverged at two threads. The solder notebook drops its `tol=1e-6`.

- `continuation()`'s warning when it stops names the step as a plain number; with
  NumPy 2 and NumPy steps it said `np.float64(1.4)`.

## 0.7.4 (2026-10-12)

Four new stress cases, all fourteen passing; `relax()` converges where gradient rounds
used to creep (a contact line moving slowly over a curved wall) and keeps a shrinking
contact circle resolved; a warning when a surface is refined too far before relaxing.

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
- Case 8, the drainage band, is now run as its notebook runs it, with
  `ev.remesh(target=h)` before each band step, and checked on what is stable from run
  to run: over three runs the median band-pressure error was 0.017-0.022%, the snap
  0.1-0.4% off, 87-89% of steps converged. On defaults alone its worst state scattered
  from 3% to 52% between identical runs, too noisy to catch regressions. Why it fails
  on defaults is in the guide: the contact line receding over the bead squeezes the
  vertices behind it into slivers where it meets the mid-plane. It now has figures and
  a section with a runnable snippet on the stress page, like every other case. All 14
  stress cases pass.

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
  limit is 1.5).
- `relax()` warns when the surface was refined four or more times since it last
  converged: relaxing that fine is slow and may not converge (`cube.fe` refined five
  times: 10 s unconverged, against 0.4 s converged with `relax(levels=5)`).
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
