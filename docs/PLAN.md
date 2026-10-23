# pySurfaceEvolver development plan

Living document: decisions, status, how we work, and what is open. Condensed at the
0.7.0 release (2026-10-10); the full history of every phase, with its measurements and
dropped experiments, is in git (`git show e8f7434:docs/PLAN.md`, or
`git log -p docs/PLAN.md`). What changed for users is in CHANGELOG.md.

## Resume here (2026-10-23)

**Topology runs, 16-bead packs to breakthrough (stage-f-pack, unpushed):** the wide set
(6 seeds x 15/20/25 degrees) reaches breakthrough 18/18 at 1 thread with no back-ups
(median 5 min, max 16 min, s1_t15's 329 steps) and 18/18 at 2 threads (7 back-ups, all
recovered; max 13 min), idle machine, 4 runs at a time. Runs and descents are bounded in
work (`_core.work()`, passes over the surface; deterministic): a descent at 100000 passes
(a stop the run backs up from), a run at `arclength(budget=1.5e6)`. The runs take
0.14-0.6 million passes. Last session's 27-43 minute runs were mostly the machine's load
(12-17): time runs on an idle machine, or compare passes. A laptop suspend shows as the
same ~4100 s gap in every log.

**Trapped liquid done (2026-10-23, section below):** `outlet=`, `Step.trapped`,
`_clusters.py`. Wide set (`~/pyse_scratch/wide_trap`): 18/18 to breakthrough at 1 thread,
3 back-ups; 0-1.12 of liquid trapped in 0-17 bodies, held to ~1e-15; no reconcile changed
the targets' total; passes 5.22M vs wide_w3's 4.97M (+5%); wall time not comparable
(loaded machine), an idle rerun of s4_t25 125 s vs 165 s; clusters cost ~2.5 s a run.
Small set (`~/pyse_scratch/small_trap`): 18/18, books to 1.6e-11, nothing trapped.
Breakthrough volumes moved in a few runs (s0_t20 49.2 vs 53.6).
The +5% passes is path change, not overhead: runs that trap nothing follow identical
paths (-1%); four runs took 20-35 more steps. Freezing trapped bodies while the branch
steps (vertices/edges/facets fixed, target lifted: a fully fixed body with a target
makes Evolver's constraint matrix singular) was tried and DROPPED: on 6 wide runs A/B
+13% time, +14% passes, 7 back-ups vs 3 (scratch `~/pyse_scratch/freeze`). The cost is
in the draining liquid's path, not in relaxing trapped rings (a small part of the mesh).

**Next:** merge/push/release (the user).

**Open (2026-10-24):** the Newton step on the cube was 16% (98k) to 21% (393k) slower than
0.7.5. Cause found by callgrind on unstripped builds (`pip install --target DIR -C
cmake.define.PYSE_NOSTRIP=ON -C build-dir=BD`: the unstripped module is in BD, the install
strips it): fl_follow_end (f367957) passed over all vertices and facets after every step.
Fixed in eefbb2a (only the line facets, cached): instructions back to 0.7.5's. Left:
separate processes still time the 393k step ~5% above 0.7.5 with the follow on (vs off),
with identical instructions and results and no difference toggled in one process
(`~/pyse_scratch/perf`: warm.py, toggle.py). Rerun relax_bench.py (0.7.5 venv
`~/pyse_scratch/v075`) after. Doc outputs can differ between numpy
1.26 (local .venv) and 2.5 (CI): bridge_rupture block 4's neck 0.06 vs 0.11.

**Open:** s1_t15 just over the 15-minute target; branch steps are ~60% of a run's time,
and failed correctors about half their Newton work (a stall exit made them cheaper but
failed more often and changed the runs' paths: 7 back-ups and slower in all; dropped).
Corrector work per point (~16 Newton pairs) is the lever. Pre-existing mypy errors in
`_surgery.py` (`np.ix_` with lists x2, a tuple assignment ~line 2948) under local numpy
1.26 stubs: check under CI's numpy before releasing. Persistent scratch:
`~/pyse_scratch` (not /tmp, which a reboot wiped with the failure library).

## Resume here (2026-10-15)

**Released:** 0.7.5 (2026-10-15, tag v0.7.5) and 0.7.4 (2026-10-12, tag v0.7.4); wheels as
workflow artifacts, not on PyPI.
What each release changed is in CHANGELOG.md; how we got there (measurements, dropped
experiments) in git: `git log -p docs/PLAN.md`, and the 0.7.0 condensation at
`git show e8f7434:docs/PLAN.md`.

**In 0.7.5** (CHANGELOG 0.7.5; besides Python 3.11-3.14, the parallel quadratic model and
the Newton fix for free surfaces below): `continuation()`'s stop warning prints
plain numbers (NumPy 2 printed `np.float64(1.4)` and failed the snippet check on the
v0.7.4 tag; main is green); a short README with a picture (`tools/readme_figure.py`),
its API tour moved to `docs/tour.md` and the development notes to `docs/development.md`;
`continuation()` predicts each step from the last two (secant predictor, on by default,
13-49% fewer gradient steps over the continuation stress cases); `newton()` returns an
IterationResult; with optimizing parameters, constraint projection polishes vertices
within the tolerance (bga-5 refined four times converges; the "parameter Hessian
limit" was the projection; for all surfaces it broke case 8 at 2 threads);
`Evolver(process=True)` (a surface per worker process, same API, `_remote.py`,
`_worker.py`; docs page several_surfaces.md); `recipes.arclength` (pseudo-arclength
through folds; Newton-only, bordered via two Evolver Newton steps; V + u between
points; catenoid's unstable branch within 0.03%); `relax()` approaches a far target
in stages when its first step folds facets; docs page bridge_rupture.md (a bridge
between spheres turns unstable 8% above its fold, a subcritical pitchfork to off-centre
bridges, Vogel's small volume instability; reference helpers in stress_helpers.py,
figure by `tools/bridge_rupture_figure.py`); `arclength`'s corrector holds the Lagrange
multipliers fixed for its two Newton steps and carries them with the vertices (Evolver's
`hessian` uses the stored pressures in its Hessian and updates them, so they are Newton
state; the second step had started from the first one's pressures and the corrector
crawled or diverged, stopping at the bridge's fold in half the runs). Undoing a step
(`_put_back`, in `relax()` too) restores them; `_core.multipliers()`/`set_multipliers()`.

**Since 0.7.5** (CHANGELOG "Unreleased", 2026-10-16): four literature pages, each by an
agent in a worktree, against exact references in their helpers: `disk_bridge.md`
(Slobozhanin and Perales' table to the last digit), `pendant_drop.md` (0.2-0.3%, Michael
and Williams' 3.219), `fibres.md` (barrel onset exact within 1%, Princen, crossed fibres at 0 degrees
against Sauret's model; no symmetry breaking before the ends close round the fibres at
30 degrees crossing, volume 70-80 against the model's 88), `floating_lens.md` (1e-4, second order). Found and fixed on the
way: `cylinder()`'s azimuthal content sign on turned fibres; `arclength` at sharp folds
(tangent after a failed step, sliding scaled to the shape change; still stops at Bond 1
on a ring of radius 1 with steps of 3% of Tate's volume); `Boundary(content=...)`.
Then (user: "fix the Evolver code, the straightforward and the fixable"; two agents in
worktrees plus me): Newton steps carry the mesh along a moving contact or triple line
(fasthess.c follow, hook in hessian_move; the real cause was the normal-only step, not
V or u; lens near spreading and the gravity lens refined twice converge; arclength's
corrector takes plain steps via `_core.set_follow`); `arclength` doesn't spread vertices
outside the linear model (the quadratic model settled on a different equilibrium each
time), halvings=6 by default with a floor at ds/4**halvings; the deflation reuses the
last step's null directions (as per-vertex displacements: the normal coordinates are
rebuilt each step; `sp_hessian_mult` adds into its output); crossed fibres at 0 degrees
by continuation in the contact angle; the clam shell's crossover extrapolated (order
1.6, volume about 39); `check_doc_snippets --update` rewrites only differing blocks.
Tried and dropped: weighting the value in `arclength`'s step measure (halvings=6 alone
turned the drop's fold; the weighting cost the catenoid 10-60% more points).
Still open:
* A long finite column's length (fibres.md uses a periodic slice): discretization
  error, converged lengths differ by 1e-4 in energy; neither the follow fix nor
  full-coordinate Newton helps.
* Seen while timing, not investigated: the 98k cube switched to quadratic from an
  unconverged linear state (`g 10` after each refinement, no Newton steps) and
  relaxed ends unconverged after 28 s (residual 3e-7), and plain `hessian` steps
  from there raise the residual to 5e-4. From a Newton-converged linear cube it
  converges in 5-6 s.
* OpenMP waiting under oversubscription (documented in performance.md, default
  unchanged): with 16 busy processes a 4-thread test run took 185-348 s spinning,
  82-98 s with OMP_WAIT_POLICY=passive; GOMP_SPINCOUNT 1000/10000 didn't help reliably
  (153/338 s). Idle, passive costs 9% (tests) to 30% (gradient steps). Switching
  automatically would need the load at OpenMP's start (libgomp reads it once).

**State:** all 14 stress cases pass, 13 on `relax()`'s defaults and case 8 (the drainage
band) with `ev.remesh(target=h)` each step, as its notebook runs it. 491 tests.

Open, for the user to pick (proposed 2026-10-13, updated after 0.7.5):

* **PyPI** (0.7.5 shipped without it): a publish job in `wheels.yml` on `v*` tags (trusted publishing,
  environment `pypi` with the user as reviewer; the user sets up the pending publisher
  on PyPI); Python 3.11-3.14 since 2026-10-15 (3.9/3.10 dropped, cp314 added). The name
  `pysurfaceevolver` was free on 2026-10-12.
* **macOS wheels** (OpenMP, MUMPS with gfortran, Intel and ARM); Windows later.
* **Candidates:** `volconst="auto"` as calibration from a given starting volume was
  tried and dropped 2026-10-13 (the starting mesh's enclosed volume is as unknown as
  volconst; a constant can't follow a contact line on a wall without content
  integrals); a geometric closure of a body's open loops would be the next idea.
* **Known limits:** on defaults alone case 8's mesh breaks down where the bead's
  receding contact line meets the mid-plane (slivers; docs/relaxing.md); mylarcube ends
  0.1% above its recipe; topology tests are path-sensitive at several threads (run
  with threads=1).
* **Do not re-propose:** remeshing inside `relax()` for case 8 (researched 2026-10-10,
  and sliver merging/moving again 2026-10-12: worse over repeated runs). Curvature-sized
  remeshing (PMP-style h from the principal curvatures, 2026-10-13): equilibrium
  curvature varies too little where meshes must be fine (case 8's slivers, a catenoid
  near its fold, a thin bridge), so it sizes like uniform remeshing. The secant
  predictor for case 8 on defaults (2026-10-13): extrapolating pushes the squeezed
  vertices further, the worst errors went from 4-9% to 170-1550% over 3 paired runs.
  The slivers build up within each step; tangential motion in Newton steps would be
  remeshing by another name.
* **Long-term:** isotropic remeshing in C toward a size field; phase E (not now);
  topology changes (planned below, "Future: topology changes").
* **After the multiplier fix (2026-10-13):** restoring the pressures on undo leaves the
  stress cases and 40/42 sample datafiles as they were; bga-6 (no equilibrium) takes
  20 undone steps instead of 12 to give up. `relax()`'s inertia cache keyed by the
  vertices is fine (reset per call, only a count of 0 reused, and vertices back now
  means pressures back). `ev.modes(n)` added (ritz + Evolver's own move, unit RMS,
  sign fixed; repeated eigenvalues' modes are any combination).
* **Packing examples (2026-10-14):** references to Surface Evolver work on liquid in
  bead packs added; liquid bridge notebook "how far a bridge stretches" (arclength in
  volume to the fold or branch point, against the exact bridges and Lian et al.);
  docs/packing.md (a throat in a layer against MSP, a trimer drained to its burst, a
  full pore whose curvature peaks at its throats' entry curvature, with a triply
  degenerate mode), helpers in docs/packing_helpers.py, figures by
  `tools/packing_figure.py`. `relax()` leaves a converged surface alone (8d1cd56).
  Open from it: nothing from (9) or (8b) (see the fourth batch below). The pore's
  one-of-three branch (one side throat alone) would need a whole surface with a
  mesh of the tetrahedron's symmetry; the wedge shows the three side throats only
  together.
  Done since: sphere centres from parameters, `arclength`'s first step halved
  (dcf7196), its unsettled stop put back, the bridge notebook pulling beads apart,
  multipliers typed; CI replicated in fresh venvs (Python 3.12 only), which caught
  SciPy missing from the `docs` extra. Then (2026-10-14, second batch): `arclength`
  `remesh=` (catenoid down the unstable branch to wires 0.43 apart) and `along=` (the
  rupture page follows the off-centre bridges, pressures within 1e-4 of exact);
  `relax(levels=n)` starts again one level finer when the last level fails (the
  "build-dependent" 20-degree bridge was SciPy's last digit of its volume tipping a
  chaotic coarse relax: 6 of 13 ulp-perturbed volumes failed, now 13 of 13; samples
  and stress cases unchanged); `bridge_spheres_branch`'s second zero at 60 degrees
  was the cylindrical bridge (slope now weighted by the neck's curvature). Then
  (third batch): `arclength` cuts a step whose point lands farther than the step from
  its prediction, and halves (instead of stopping) when a point doesn't settle after
  tidying, going back by the corrector; `relax()` stages small contact angles (walls
  below 15 degrees that a crude start sinks into: raised to 20 via the datafile
  parameter `pyse_angle_floor`, lowered by 0.7 a stage); the packing page's 5-degree
  section follows trimers through snap-in and pop-out onto the dimers and maps the
  folds to a cusp at (V, s) = (0.267, 0.226), the paper's (0.26, 0.22). Then
  (fourth batch, 2026-10-15): the pore in a C3v wedge (`pore_wedge`: the full pore
  with a three-fold axis along z, relaxed, cut to a sixth between two mirrors): the
  widest throat goes first at its own entry curvature (bottom gaps 0.03: 9.533 against
  the layer's 9.520, then a fold); 5-degree trimers near the kink on a finer mesh
  (snap-in and pop-out at gaps 0.13-0.15, burst at 0.11; `throat_thickness` only
  means the lamella while the outer meniscus is outside the triangle of centres);
  docs CI parallel (`sphinx-build -j auto`, snippets `--jobs 4` in two jobs).
* **Small loose ends, closed 2026-10-15:** PORD after six factorizations on one
  pattern (4 threads: 10-15% faster steps in Lagrange 2/4 and linear 1.6M, same at
  393k and below; `arclength` gains, `relax` with 2-3 a pattern unchanged; reset on
  every load for reproducibility); the quadratic/Lagrange-edge setups' small products
  inline (bit-identical; quadratic `g 1` -27% at 98k); `r` left serial (1.4 s
  393k->1.6M, 40% first-touch zeroing of new elements, no hotspot; a non-atomic
  attribute stamp in new_element was no faster); the conformal 1/4 is reachable
  (linear strings) and tested. Found on the way: `arclength` failed at once in the
  Lagrange model and for strings (equiangulate), and on fine Lagrange meshes its
  corrector couldn't meet the step criterion at round-off: both fixed.
* **Quadratic model in parallel, 2026-10-15:** a 6-control-point facet cache; Evolver's
  facet energy/force/volume/volgrad loops, the named-quantity loops and Hessian
  assembly (area, volume, vertex integrals; fasthess.c), vertex normals; volume as
  pair sums (round-off). Cube 98k, 4 threads: `g 1` ~0.75 -> 0.18 s, Newton step
  ~2.0 -> 0.95 s, `relax()` ~69 -> 36 s (unconverged; see the next item). Quadratic
  edge and facet integral methods stay serial.
* **Rigid motions out of Newton steps, 2026-10-15:** a free surface (cube.fe) makes the
  Newton system singular along its rigid motions (eigenvalue ~1e-13 of a mean diagonal
  4.7 at 98k quadratic facets, falling 50-80x per refinement); round-off moved each step
  4e-6 along them and held the residual at 2e-6, so `relax()` ended unconverged.
  `fl_hessian_deflate()` (fasthess.c, after the solve in `sp_Hessian_solver`): the rigid
  motions projected onto the step's freedoms, orthonormalized (dropping those that only
  slide the surface along itself), Rayleigh-Ritz on their span (6 products with the
  matrix; rotations about any axis); a Ritz value below 1e-10 x mean diagonal with no
  constraint gradient along it -> one solve (inverse iteration), taken out of the step.
  Deflating along the projected motion itself (cosine 1 - 1e-9) left 2e-8. `relax()`
  there: 5 gradient + 2 Newton, residual 1e-9, 6 s; the solves add ~0.25 s to a ~1.1 s
  step. Held surfaces (mound sliding on its plane: quotient 1e-3 x diagonal)
  bit-identical. Possible: reuse the null vectors between steps (they barely move).

Ask the user before each step; run scripts with OMP_NUM_THREADS=4 or less.

## Done: trapped liquid keeps its volume (planned and done 2026-10-23)

Before the next release (the user: fix this first). Today a piece a cut leaves can be
folded back into the draining body (`_rejoin_strays`), dry patches/contacts/sheet
holes never split bodies, and crumbs are deleted with their volume only logged.
(Merges' constant shifts are already limited to whole pole quanta, `_kept_constant`;
the conservation check confirms or reopens it.)

**Decisions (user, 2026-10-23):** the liquid connected to an outlet wall drains
(`arclength(..., outlet="bottom")`, a name or list; without it the largest cluster
drains, as today); trapped clusters keep fixed volumes, relaxed every step, and rejoin
the draining body if they merge with it (two trapped merging stay trapped, targets
added); connectivity from the liquid region itself (a grid of the pore space), not
facet connectivity; a crumb's liquid goes to the cluster it came from (or the
nearest); merges conserve physically (no constant shift); no liquid connected to the
outlet: stop with a warning; outputs: trapped volume per step, cluster ids in the
XDMF series; validation: conservation on the packs, an exact trapped case. Pacing:
all phases, a commit per item, report at the end.

**Invariant:** liquid is only split (each piece its measured volume as target, the
draining piece the rest), joined (targets added) or stepped by the run; so the targets'
total changes only by the run's steps.

1. **Clusters** (`_clusters.py`, unit tests). Surface pieces (edge-connected facets
   per body side, as Evolver's `rebody`) are always within one liquid region; joins
   between them through the bulk found on a grid of the pore space at h/2: cells
   inside a solid (beads, outside the box, from the walls' plane/sphere shapes)
   dropped; neighbouring cells linked unless the segment crosses a facet or a solid;
   each linked group labelled liquid of body b or gas by a vote of nearby facets'
   sides; pieces whose liquid sides share a group are one cluster. A cluster touches
   the outlet where its cells border the outlet wall (at a pack's start no vertex is
   on the floor). Other wall shapes: surface pieces only, with a warning. numpy first,
   C if the profile asks; run only after changes.
2. **Bookkeeping**: one `_reconcile(ev, keep, outlet)` after every change (descents,
   `_change`) and after crumbs go. Clusters touching the outlet are one body `keep`
   (several pieces allowed: one outlet pressure, their total stepped); every other
   cluster its own body at a fixed target. Replaces `_rejoin_strays` and the
   largest-piece rule when an outlet is given.
3. **API/outputs**: `outlet=`; "all the remaining liquid is trapped" stop;
   `Step.trapped` (the trapped clusters' volume; draining = `Step.value`);
   `series.py` per-facet cluster id and trapped flag; corpus passes
   `outlet="bottom"` and records the trapped volume.
4. **Validation**: small and wide sets, at every step draining + trapped = start minus
   the run's steps to 1e-6, still 18/18 to breakthrough at 1 thread, times against
   wide_w3; a test where a pendular bridge between two beads is cut off while liquid
   connected to a floor outlet drains: its volume to round-off, its pressure the same
   bridge's relaxed alone at that volume; unit tests: a throat with two menisci (one
   cluster), a bubble inside the liquid (no false cluster), two drops (two).

**Risks:** liquid layers thinner than the grid could split one cluster (a false
trapping): conservative grid, the packs' clusters checked once against a finer grid,
clusters logged at each split. Re-splitting after dry patches changes the runs' paths:
robustness re-measured in phase 4. Then the limitation notes (drainage page,
CHANGELOG, arclength docstring) go.

## Future: topology changes (planned 2026-10-16, not started)

Why: every example stops where the liquid's connectivity changes (a bridge
ruptures, a trimer bursts, menisci meet in a pore, a clam shell closes round its
fibre, a column's ends close round crossed fibres), and drainage through a bead pack
is a chain of such events. Detection is cheap; handling them needs mesh surgery, done
under volume control so that each event happens on a branch `arclength` follows. The
level-set method (Prodanovic and Bryant)
gets topology for free; pySE's case for doing it on meshes is exact menisci, contact
angles, stability and saddles. Near term, bead packs go the hybrid way instead: exact
throat thresholds from `arclength` runs, a pore network (invasion percolation) for
where the gas goes (procedure agreed with the user, not run yet; pilot in the session
scratchpad: random 40-bead cell, MSP generalized to irregular throats, pores turned so
contact lines keep off the sphere constraints' poles, one pore 27% below MSP).

Events (each needs detection, surgery, and a test):
1. **Pinch-off**: a neck thins (bridge rupture, snap-off, a column breaking up): cut
   at the narrowest ring, cap both sides, two bodies (the volume split by the cut; a
   part cut off from the rest keeps its volume, stage B). Easiest; start here.
2. **Meniscus merge**: two free sheets meet (menisci from two throats in a pore, drops
   coalescing, a clam shell closing round its fibre): delete the facets where they
   touch and stitch the two openings (Evolver's `vertex_merge`, `edge_merge`,
   `facet_merge`, `dissolve` are the primitives; volumes add).
3. **Sheet meets a solid**: part of a free surface touches a bead or wall and becomes a
   new contact line on that constraint (a meniscus advancing into the next pore).
4. **Contact lines meet on a solid**: wetted patches on one constraint merge or split
   (the crossed-fibre ends, the trimer's contact lines closing in).

Stages:
* **A. Detect and stop well** (done 2026-10-17, the pilot bead pack as the case):
  `Evolver.events(factor=0.5)` (`_events.py`, NumPy) returns `Event`s of four kinds:
  "neck" and "touch" (the surface within factor*h of a facet across a body / the
  outside, facing it; a neck within 3*factor*h/order: a ring of five edges in the
  linear model), "solid" (within factor*h of a constraint it isn't on, from Evolver's
  f/|grad f|; kept when the stretch doesn't reach a contact line on it, or is closer
  than 0.7 of the contact line's wedge, s*sin(angle) at a distance s from the line's
  edges, angle through what lies between; gone through: negative gap) and "contact
  lines" (on one constraint, within factor*h, far apart along the line). h: each
  vertex's mean edge, from edges between free vertices where it has some (a pad
  drawn in large fixed facets made bga-3's joint a "neck"). Left out: vertices in a
  fold of the mesh, facets within two edges, self events covering half their piece
  (a mesh too coarse for it: qmound.fe unrefined), tori. Candidates: vertex pairs
  within reach with facing normals, then the facets round the far one. `arclength` and `continuation`
  (`events=True` by default) put them on `Step.events` and stop, with a warning, at
  the first point where one is coming (there at the last point too and closer by 0.01
  of an edge); a point gone through a solid is a failed step (halved). `relax` puts
  them on `result.events`. The pilot pores (1 and 2, 20 and 40 degrees, 661 facets)
  stop at the meniscus coming down on the bead opposite the draining throat (a film
  0.004-0.016 thick), before the surface goes through it (before: through it, then
  stalled); one level finer the same. Found on the way: packing.md's 5-degree dimer
  ended inside bead 3 (its "stops at 0.196" open item: the meniscus reaching the bead,
  now stopped at 0.206), and the pore wedge with the narrow bottom throat stops at
  0.204 where the liquid is about to pinch at a mirror plane (before: on to a negative
  mode at 0.190). Cost: 11 ms per look at 661 facets, 90 ms at 17k, 360 ms at 68k
  (health(): 89 ms there). Default `events=True` checked against the tests, the
  stress suite, every docs snippet and the executed notebooks: only packing.md's two
  runs above changed (an early version stopped a barrel on a fibre, a 30-degree drop
  and bga-3, all fixed). Limits: a
  surface that crosses itself between two looks isn't found; at small contact angles
  next to a contact line, a curved meniscus can come within the wedge rule (a 5-degree
  meniscus did, truly); a barrel on a thin fibre stays above 0.9 of its wedge.
  Not done: the Hessian eigenvalue at zero (arclength's negative_modes already says
  it); relax doesn't stop (its pinch-offs go through `collapsed`).
* **B. Bodies that split off** (days): when a pinch-off or a closure cuts a blob of
  liquid off from the rest, it becomes a body of its own with its volume fixed (trapped
  liquid); the rest goes on under the run's control. Optional, later: `Body(pressure=p)`
  (Evolver's body pressure) and relaxing through a lost equilibrium at fixed pressure,
  only to compare directly with pressure-controlled pore-network curves.
* **C. Pinch-off surgery** (done 2026-10-17, the bridge between spheres as the case;
  design agreed: fall at the instability, `pinch_off` plus an arclength option, the
  run going on with the largest piece). `Evolver.pinch_off(event)` (`_surgery.py`):
  the two rows of vertices at the ends of the edges crossing the plane through the
  neck, across its axis (the direction the waist's normals are all across), collapse
  to their centroid; Evolver deletes the collapsed edges (`delete edge where length <
  tiny`), `pop vertex[v]` splits the point, `rebody`; the largest piece gets the
  body's number (facets swapped, `rebody` again for the quantities); each piece's
  volconst from the walls it reaches (`surface.walls`; rebody copies the old body's
  to each, two spheres' constants each); fixed targets: the smaller pieces what they
  hold plus what the collapse took from their side (the tetrahedra from the point to
  the facets round the ring, by side: exact, they add up to what was lost), the
  largest the rest. Then the edges under h/2 within 3h of the cut go (Evolver's
  deletion; the fall squeezed the neck's ring to 64 edges of 0.005, the cones' facets
  met at the tips), and the pieces settle: at the volumes held, `g 5` and Newton with
  line search (relax() from the cones slid a 40-degree drop's vertices through its
  sphere in one of four cuts: gradient steps next to a shallow contact line), then
  Newton steps to the targets in up to 16 parts (relax() to them at once failed in
  most). `arclength(topology=True)`: at the first point with more negative modes, the
  fall (from that point after a branch point, where the mode is across the branch's
  shape change, |cos| < 0.5; at a fold, where it is along it (0.99), from the value
  one step past the last stable point): steps along the mode of 0.3 median edges at
  most, the way five gradient steps go along it (the first step, where that is
  round-off, modes()'s sign: else the bridge fell to either sphere from run to run),
  until a neck event (cut, settle) or a positive lowest eigenvalue and a residual
  under 1e-3 (Newton finishes; relax() there slid vertices through the sphere on 8192
  facets); other events coming stop it, solids only once gone through, and vertices
  through a solid within two edges of its contact line go back out along its normal
  (the gradient steps took one through the top sphere in 1 of 5 page runs; 0 of 10
  after), and a neck pinch_off refuses (reaching a contact line) is passed over
  (1 in 12 runs stopped there before). Plain gradient steps fell too slowly (eigenvalue -0.74: 5e-7 of energy per
  10 steps), `saddle` overshot (the neck from 0.05 to 0.001 in one line search),
  Newton with the mode projected out diverged, global remeshing during the fall
  wrecked the contact lines. Tests: the sphere bridge falling at its branch point
  into two drops (exact caps to 0.3%), the disk bridge pinching at its fold
  (0.4687 on 1024 facets, exact 0.4751) into two equal caps, a catenoid past its fold
  into two flat disks (a parameter run, no bodies), pinch_off either way (the swap).
  Costs: the fall 1 s on 2048 facets, 13-16 s on 8192 (modes() each step); the settle
  1-3 s. Limits: the split depends on the way down (smaller drop 18-22% on 2048
  facets as the steps vary, 17-19% on 8192 (with the thread count), 25% falling from
  a step further), a model,
  not the flow; a neck meshed 16 edges round (512 facets) cut, but the drops didn't
  relax; necks at a constraint (a drop leaving a solid) are a contact-line event
  (stage E); one neck at a time (Rayleigh-Plateau columns, the trimer: not tried);
  linear model only. Not done: continuation(), relax() (still `collapsed`).
* **D. Merge surgery** (done 2026-10-17; design agreed: as C, `Evolver.merge` plus
  arclength's topology=; the merged body the run's number, else the lower; cases: two
  free drops, drops on a plane). `_surgery.merge`/`_glue`: `facet_merge` on the facet
  nearest the touch and the nearest one facing it, unset bodies, dissolve, the other
  body's facets moved, `rebody`, dissolve body; volconst and targets added. The
  descent after a change (`_surgery.descend`, which C's fall became): `relax()` in
  runs of 50 gradient steps (gradient steps alone took >1000 to widen a merged neck at
  5704 facets), edges kept within 0.4-2 h0 (`_keep_edges`: Evolver's deletion only
  where the link condition holds, its ends sharing no neighbour but the facets'
  corners, else a doubled edge and a facet with a corner twice; never a chord between
  two contact-line vertices, which pinched two lines together; splitting with
  `protect`, whose mask `remesh()` applied after the deletion had renumbered the rows,
  fixed), events between the runs: coming necks cut (a merge's own neck widens: not
  cut), coming touches merged, contacts made, a change refused (ValueError) passed
  over; a converged state with a squashed facet tidied and relaxed once more; Newton
  or relax() finishes; an unstable end falls. arclength after a change: the next
  points remesh the same way unless `remesh=` was given, and a branch that stalls
  after a change goes on from a relaxed state (quiet; three such in a row stop).
  Two free drops at 1368 facets: the sphere to 0.11% (0.03% at 6842). Two 120-degree
  drops on a plane: merge, then the contact lines meet twice (D brings E), one cap
  to 0.2%.
* **E. Contact-line events** (done 2026-10-17 as far as below; design agreed: wetting,
  drying and contact lines meeting; cases: drops on a plane, a drop growing onto a
  sphere, the pilot pore). `_surgery.contact`: a solid event's vertices (off
  other constraints) put on the solid (`x -= d n`), the facets among them go, their
  border on the constraint (`_onto`: each new edge stored the way its facet goes
  round it, orientation 1: `reverse_orientation` then `orientation 1` for one stored
  the other way; `orientation -1` alone gave the right energy integral but not the
  content, on a sphere a dry patch's volume 0.03 off and the pressure -0.8 for -4.45);
  a contact next to an existing contact line grows that patch (the strip between
  goes; a patch opened a facet's width beside one broke the mesh later); the
  volumes continuous by volconst (it also takes the poles' content constants); a new
  loop winding round a sphere's axis turns its pole wet or dry, and the wall's
  `pole_energy` (new Constraint field) goes on `_SurfaceMemory.energy_shift`, added by
  `Evolver.energy` and kept in snapshots. Contact lines meeting: wet between (the
  facet along a line has the wet side to its left, seen from outside the solid) cut
  as a dry strip; dry between: the facets lying on the solid there (a dry channel
  roofed over) taken away, else the facing contact-line edges glued (`vertex_merge`
  twice, `edge_merge`, off the constraint; refused where the vertices share
  neighbours, which doubled edges with facets going round them the same way, and
  where the lines meet at an angle). On a solid beside its contact line, a "solid"
  event in a descent is the gradient steps' (vertices put back out if through).
  Results: a drop growing onto a sphere at 60 degrees, then relaxed at volume 2.5:
  the exact plane-sphere bridge's pressure 0.15%, energy 0.35% (with the pole's
  share: without it 2 pi off); 60-degree drops whose contact lines meet: one cap to
  0.11%; a dry patch on a sphere's side inside a draining drop. The pilot pore 1 at
  20 degrees (661 facets) drained from 0.69 to 0.05: bead 4 dried in eight steps as
  the meniscus slid along it, the meniscus left it at 0.433 (the pressure jumping
  from -4.5 to -3.1), beads 2 and 4 dried again at 0.169; one level finer it got to
  0.41. Limits: drying at a sphere's pole fails (the integrals take the azimuth about
  its axis: two spurious negative modes, the fall drags the line across the axis):
  keep contact lines and contacts off the poles, as the pilot pores are turned to;
  long runs after drying are fragile (the descent's gradient steps take the surface
  through the solid in new places, contact after contact, until Evolver's pop fails,
  ERROR 1303 cone_analyze: the side-drying case after its second patch, the finer pore
  at 0.41; free vertices aren't kept out of a solid); the split at a pinch and which
  dry patches form depend on the way down. Not done: a contact line leaving a solid
  altogether (a drop detaching), merges across a body (films, foams), quadratic and
  Lagrange models.
* **F. Drainage through a bead pack in full** (in progress 2026-10-17; design agreed:
  a cluster of ~16 beads settled in a box with hard walls, so that the 90-degree
  mirror walls cut no bead; gas on top; each bead's `sphere(axis=)` pointing at its
  nearest contact; robustness first; a run and a docs page). Done: the Evolver
  fixes bfe1131 (equiangulation's doubled edge, a border spike under `t`: these
  made the ERROR 1303 that ended stage E's long drying runs); `sphere(axis=)`
  (three constructions: two from the z frame left points where singular terms
  cancelled only on the sphere, and dry patches opened there ran away along the
  chords; the third uses the area form about the axis with a denominator that
  vanishes inside the sphere only on the axis at the north pole); guards (one-sided
  copies `pyse_guard_<wall>` of the walls on free vertices, the two nearest within
  5 edges, only in the descent after a merge or contact; off around surgery and
  tidying, as Evolver won't delete an edge between differently constrained
  vertices; three hits at once is Evolver's ERROR 5883); a surface resting on a
  solid and contact lines met are changes (they never come closer); a descent may
  end near a wall; a run of the descent that went wrong is undone. A fall keeps the
  old rules (contacts made from its mode steps' surface spoiled a bridge's cut).
  Results: the pilot pore drains to 0.045 at both mesh levels (the finer died at
  0.41); the 16-bead pack (scratchpad f/cluster.py) from 87.5 to about 68 through
  ~12 dry patches. The descent after a change: a relax() run that goes wrong is
  undone for 10 plain gradient steps (one, where those go wrong too), relax() once
  near equilibrium; the dry patch on a sphere inside a draining drop now goes the
  whole run (from 53.5 to 6.3 through 9 changes; plain gradient runs alone took it
  to 2.6 but were too slow for merges and the pack). The pack: a contact line's
  step across a bead's singular north pole (the integrals jump by the whole
  sphere, 4 pi cos 20 = 11.8, and the volume constraint then drives the surface
  off) ended it in 1-12 changes. Tried: one-sided balls round the poles (0.15 R:
  chords still crossed; 0.25-0.5 R: lines stalled at the balls); wetted facets on
  the beads (probe: free bead vertices degenerate under pressure, fixed ones break
  the rotational symmetry, negative sliding modes). Done: moving poles
  (`_surgery.move_poles`): a pole within 0.5 R of a line goes to the point
  farthest from the lines, a new runtime constraint (the old one retired:
  `_SurfaceMemory.retired`, kept out of `_constraint_gaps`; a contact made on a
  retired one drifted the energy), the volumes and energy shifted by the whole
  spheres between the poles' integrals as the lines' solid angles from the
  antipodes count them (`_fans`; the integrals' own difference was off by half a
  sphere for a line 0.05 R from the old pole or with 1.6-long edges); runs of the
  descent that bring a line within 0.25 R or across (solid angle jumps by 2 pi;
  one relax() run swept a line from 0.83 to 0.32 beyond) are redone with the pole
  moved; contact-line edges stored against their facet are turned round
  (`orient_lines`; two of 35 seen once, the energy below what the beads can give).
  Then (branch stage-f-pack): shrunk contact-line loops covered (`close_patches`:
  a three-edge loop 0.003 across ended a third of the runs at the first change);
  falls split line edges past 2 h0 (`_split_lines`; chord integrals 1.5 off at
  edges 1.6); where a change or fall ends the mesh is evened (`recipes._smooth`:
  vertex averaging, flips, Newton between, five times; Newton's first step from a
  7-degree mesh converged x0.7 a step and failed, x20 once evened); a branch
  that can't take one step from where a fall ended makes the nearest change
  (`_nearest`); unforced pole moves need 1.5 times the distance (a bead's pole
  moved 19 times). Pack runs (6 per variant, 2 threads): final volumes 57-83
  run to run, medians 64-71 between batches of the same code: six runs don't
  resolve a variant. Remaining stops: no stable state within 200 steps, the
  descent running away, Evolver internal errors (1303 cone_analyze, 1374 detach
  edge) in the surgery. Then, from single failures at 1 thread (deterministic
  there; six configurations: theta 15/20/25, h 0.18/0.2/0.22, ds -0.4/-0.5):
  joins of lines running the same way refused (a facet turned over: the later
  ERROR 1303); flat facets on a solid dropped (`drop_flat`; also the flaky
  merge test); Evolver's use-after-free in set_e/v_constraint_map fixed (ASan;
  a segfault at h 0.18); a NumPy 2 repr in move_poles' shift; refused changes
  passed over; descent budget per change; the pinch split by hand
  (`_split_pinch`; Evolver's pop made a three-facet junction); close_patches
  within 0.2 h (0.5 h churned on walls); equiangulation in falls only after
  line splits (in every fall step a catenoid never settled). Tools: overlays
  of the package with trace patches (scratchpad mp/build_traced.py), dumps
  before each change, asan_venv/dbg_venv builds. Remaining stops: descents
  that don't settle on a degenerate mesh (coincident vertices), the descent
  running away, falls without a change.
  Original aim (after A-E): volume-controlled, as a
  slow syringe-pump drainage (the experiments of Berg et al. 2013 and Moebius and Or
  come closer to that than to a fixed pressure): `arclength` in the liquid's volume, a
  Finney cut, every event handled. Passing a throat is then continuous (the pressure
  drops, the other menisci pull back; the pilot's pore stayed stable past its widest
  throat), the pressure trace shows each Haines event as a drop, and the
  pressure-controlled answer follows from the same run (from each pressure peak, the
  next state at that pressure). Against the hybrid network run on the same pack.

Design points to settle before C: surgery in C (fast, near Evolver's element
structures) or in Python on `mesh()` with a reload (simpler, slower, loses element
attributes unless carried); how bodies split and merge in the datafile model (body
ids, volconst, pressures); keeping the mesh good after surgery (remesh around the
seam; the line-following Newton fix must accept new lines); whether that remeshing
needs C (start C with `ev.remesh()` limited to the seam's neighbourhood and measure,
on a small test mesh of a few hundred facets so each event runs in seconds, the
Newton steps right after each event and the time per event; build the C remesher
toward a size field only if seam quality or speed limits it, and then together with
the surgery if that is in C; `remesh()` is linear-only, so events in a quadratic run
go through linear around the event); thread safety of the fast
loops' caches (facet cache, pattern reuse: invalidate on surgery; the deflation cache
is keyed by top_timestamp already). Reference for robust collision handling and
topology changes on explicit meshes: El Topo (Brochu and Bridson, SIAM J. Sci. Comput.
2009). Risks: merges and contact-line events are the hard part; under volume control
the events happen on a branch `arclength` follows, so the surgery must leave a state
it can go on from (a point near equilibrium on the new mesh).

## Proposal: robust surgery and remeshing in C (2026-10-18, agreed; phases 0-1 started)

**Why.** Stage F's 16-bead pack runs end at 60-73 of 87.5 (six deterministic
1-thread configurations on d0e51c2). Each change of topology is a Python-driven
surgery plus a descent, failing about 1 in 30; a 64+ bead pack needs thousands
of changes, so under 1 in 10,000. The user chose (2026-10-18): research first,
everything in C inside Evolver's element structures, a failure-rate corpus as
the measure, and an invariant checker plus in-memory snapshot first.

**What the research found** (El Topo, LosTopos/MultiTracker, PMP, DSC, IPC;
clones and paper texts in the session scratchpad, repos/):
- El Topo (BSD-2 since 2018, predicates public domain) and LosTopos (BSD-2)
  are unmaintained C++ on their own arrays; El Topo leaves border (contact-line)
  edges out of every operation; neither knows level-set constraints, line
  integrals or fixed volumes. Adopting either means rebuilding Evolver's
  elements at each call (today's reload problem in C++). DSC is GPL-3 and
  needs a tet mesh of the pore space. Nothing published tracks moving contact
  lines on curved solids with topology changes robustly: that part is ours.
- What to take: LosTopos's design (every edit atomic: checked, then completed
  or cancelled; topology changes from snap, degenerate-facet removal and
  vertex separation; the mesh valid between edits); El Topo's collapse checks
  (link condition, no normal turned over, volume-change bound, no new
  duplicate facet, intersection checks); PMP/Botsch-Kobbelt remeshing (split
  above 4/3 h, collapse below 4/5 h, flip toward valence 6/4 on the boundary,
  tangential smoothing, projection) with its boundary and feature rules, which
  are contact lines' rules; Jiao's null-space smoothing (restricted to a
  constraint's tangent plane, or a contact line's tangent); exact predicates
  (Shewchuk's predicates.c or Bridson's tunicate, public domain, vendored).
- What Evolver has: radial-edge facetedges (non-manifold native), element
  blocks addressed by ids (a block copy snapshot is workable), edge_refine
  with projection, delete_edge (with our link condition), edgeswap without
  geometric checks and never on border edges, merge/dissolve/rebody, film pops,
  run_checks (list, facetedge, body orientation, duplicates). Missing: vertex
  star and contact-line invariants, geometric checks (turned-over facets,
  intersections, chords through solids), an in-memory topology snapshot, vertex
  separation, snap merging, constrained smoothing.

**Plan, phase by phase** (each measured on the corpus before the next):
0. Failure-rate corpus: deterministic 1-thread drainage runs over pack seeds,
   sizes (4, 16, 64 beads), contact angles (15-25) and h; a runner that logs
   every change, every refusal and every stop with its class; the measure is
   changes per failure and the volume reached. Today's code is the baseline.
1. C invariant checker (global, and local around an edit): run_checks plus
   single-fan vertex stars, 1-2 facets per edge, consistent orientation,
   contact-line invariants (each border edge on one solid with both ends,
   oriented with its facet, methods present), no degenerate or turned-over
   facet, no vertex or chord inside a solid, exact edge-facet intersection
   in the patch, body volumes continuous. A debug flag runs it after every
   edit (tests and the corpus); errors carry the edit that broke it.
   C snapshot: block copy of elements, skel, bodies, quantities and the
   pySE-side constraint state, for ev.save()/restore() inside the descent
   (the text dump and reparse go; a reload also changed memory layout).
2. C local edits, each atomic with a local undo journal and checks before
   and after: collapse (delete_edge plus no turnover, a volume bound,
   PMP boundary rules as "same constraint set"), flip (edgeswap plus normal
   agreement, a volume bound, no new edge through a solid; contact-line-aware),
   split (exists), constrained tangential smoothing (2D on free surface, 1D
   along contact lines, projected), vertex snap, vertex separation (one copy
   per fan: the pinch split, now by hand in Python), degenerate facet-pair
   removal.
3. C remesher from those edits: PMP-style isotropic sweeps with contact lines
   as features, sizes adapted near thin films and lines, run locally after
   every change and in sweeps during descents (in the topology pipeline only:
   relax()'s default stays as decided, see remeshing-in-relax).
4. Events rebuilt in C from the edits, each all-or-nothing with fallbacks
   (refine and retry, another pair, a smaller step): meniscus merge (split,
   snap, separate), pinch (collapse the ring, separate), contact-line join and
   split (snap or separate border vertices), dry patch (remove facets, new
   line with constraint and methods), sheet onto solid.
5. The event detection and descent pipeline stays in Python (the user,
   2026-10-18, after the baseline: its 16-bead stops are decisions and
   meshes going wrong, not speed), rebuilt on phases 1-4's C edits and
   checks: each failure the checker names, fixed in the policy; ported to C
   only if profiling ever shows it costs time.
6. If the corpus still shows them: intersection-safe descent steps (a line
   search filtered by static intersection and solid checks, IPC-style).

**Progress.**
- Phase 0 (2026-10-18): bench/topology/corpus.py (and pack.py), the quick set
  of 18 runs (4- and 16-bead packs, seeds 0-2, 15/20/25 degrees, h 0.2) and a
  full set (mesh sizes, 64 beads); the `pysurfaceevolver.topology` logger.
  Baseline (d0e51c2): all nine 4-bead runs reach half the liquid, none of the
  nine 16-bead runs; 495 changes, 9 failures (55 changes per failure); stops:
  no stable state 3, ran away 3, relaxed but unsound 2, three falls 1; the
  refusals mostly "the place is on other constraints or fixed" (64) and "the
  contact lines run the same way" (28).
- Phase 1 (2026-10-18): the snapshot in memory (bindings/pyse_api.c,
  Evolver._save/_Saved): element blocks copied whole, attribute layouts put
  back (Evolver's expand_attribute, then attribute by attribute where padding
  moved them), quantities, parameters; bit for bit; only once the surface is
  in quantities (Evolver can't convert back), dumped before a text restore's
  reload. 26x faster at 98k facets. The check (bindings/pyse_check.c,
  _surgery.check_topology, PYSE_CHECK_TOPOLOGY=1). It found at once that
  contact() on a plane can leave a pinched vertex (the new patch's border
  touching a contact line at a vertex), in two plane-drop tests.
  The corpus with the snapshot: the outcomes of single runs moved (element
  order no longer renumbered by reloads: the runs are chaotic), and a
  wet-and-close churn loop with no bound but the 2000-step cap appeared in
  two 16-bead runs: a corpus of 18 runs can't judge a change by its runs, only
  by its classes of failure.
  The corpus with the check on (PYSE_CHECK_TOPOLOGY=1): 12 of the 18 runs
  stop at a pinched vertex right after a change (226 changes: one in 19 at
  least), 11 of them drying contacts ("the liquid between thinning": the dry
  patch's border touching a contact line at a vertex), one wetting; the
  4-bead runs, which reach half the liquid without the check, pinch too (the
  descent gets over it there). Logged on the way (records summed over the
  checks, not distinct): edges through a solid 905, folds 291, edges through
  a facet 127 (the surface through itself in descents: nothing stops that
  now). First targets for phase 2: contact() leaving no pinch (separate the
  vertex as it is made), and descent steps that can't pass the surface
  through itself.

- Phase 2 (2026-10-18; the user: undo by snapshot, all edits first then
  wired in, self-crossing in phase 2): bindings/pyse_edit.c and _edits.py.
  Seven edits (collapse, flip, split, smooth, snap, separate, degenerate),
  each with rules first and the patch's problems counted before and after
  (a grid of facets and edges made once per batch); a batch takes one
  snapshot and an edit that comes out worse puts it back and makes the
  batch's earlier edits again. Wired in: _keep_edges, the tidy and fall
  flips, recipes._smooth, contact()'s pinch separation (in place of the
  Python _split_pinch), and a crossing guard on runs, fall steps, mode steps
  and changes. The crossings the guard met in the 16-bead runs were folds an
  edge or two across beside a change or a contact line, not two sheets: the
  mesh is now evened out after each change, and before a run refused for
  crossings is tried again; the guard counts only crossings more than a ring
  of edges apart (folds one ring across, beside contact lines at small
  angles, came back run after run and the runs had got past them before:
  the mesh's to mend). recipes._smooth keeps Evolver's vertex averaging (the
  checked smoothing there changed which dry patch the docs' pore opened, and
  its run went past the jump's fold onto another branch), with checked flips.
  Corpus (quick set, final code): 16-bead runs drain further (median 0.83 of
  the way to half the liquid, mean 0.71; baseline 0.54 and 0.48), 46 changes
  per failure on 16 beads (baseline 28); 4-bead runs 6 of 9 reach half (2
  run away early, 1 stops at 0.998; baseline 9 of 9); all: 6 of 18, 46.5
  changes per failure (baseline 9, 55). Stops: descent ran away 8, three
  falls 3, relaxed but unsound 1. With the check on, one structural problem
  in 558 changes: a merge (Python merge(), Evolver's facet_merge) left border
  edges on a constraint their vertices aren't on: phase 4's.

- Phases 3 and 4 (2026-10-19; the user: uniform h0, remesh after changes and
  falls, new events first with the old ones as fallbacks): the remesher
  (PYSE_OP_REMESH, _edits.remesh: Botsch-Kobbelt sweeps of checked edits),
  merge by a tube (PYSE_OP_MERGE), contact-line joins by a bridge
  (PYSE_OP_BRIDGE), pinch_off's vertex split by the checked separation.
  Remeshing at the end of falls was measured and left out: it changed the
  drainage page's pore (its jump dried one bead, not two), coarsened a
  pinched-off drop (pressure 2% off the exact cap's, a test failing), and in
  the wide set (16 beads, seeds 0-5, 15/20/25 degrees) did worse. Wide set:
  remesh after changes only 3 of 18 runs to half the liquid, median 0.83 of
  the way, 46 changes per failure; with falls too 1, 0.71, 32; phase 2's
  edge keeping with the new events 0, 0.76, 36. The first 16-bead runs reach
  half the liquid. Fallbacks used twice in 688 changes (contact-line joins).
  Stops: descent ran away 7, three falls 5, relaxed but unsound 3.

- Robustness to breakthrough (2026-10-18; the user: all wide runs to
  breakthrough, h 0.2, the floor a wall at 90 degrees, the C patch edit with the
  Python patch as fallback; speed: analyse and propose only). Breakthrough:
  the floor's "liquid between thinning" event in a step or a descent, or a dry
  patch on it. Fixed on the way: Evolver's bk_constraint_setup counted
  constraint entries from each row's end and missed a row's first entry (a row
  without its diagonal), writing one past CJA/CA (the "double free" crashes of
  4-bead runs; found by printing the counts in the deterministic build, which
  ASan's and Valgrind's allocators didn't reproduce: their heap addresses take
  other paths); MUMPS's PORD exited the process on small systems (knotty.fe);
  Evolver.body() by id. The policy additions are listed in the CHANGELOG
  (nearest change when held back, step-checked changes and tidying, undo after
  Evolver errors, cycle breaker, fold push, crushed facets in falls, remesh
  when stuck, relative volume and per-step energy plausibility, dry-patch
  hysteresis, poles clear of the swept path and shortened runs, no tangent-wall
  corners, corner vertices out of solid events). 4-bead set: 10 of 18 to
  breakthrough at the start, 12-16 of 18 since (paths differ with any change).
  16-bead set: runs got from 0.1-0.25 of the way (stops in the first minutes)
  to 0.2-0.6, none to breakthrough yet. What stops them: (1) narrow contacts
  (beads 0.03 apart, 0.015 from the walls and floor, under a tenth of an edge):
  the gas front reaching a bead-wall or bead-bead contact needs the pendular
  ring there pinched off from the bulk, contact lines on both solids joined at
  once, which no change does (pinch_off refuses necks reaching a contact line);
  (2) pole churn on heavily wetted top beads (bead 13: 79 forced pole moves in
  one descent; lines all over the bead, no point clear of them): one pole per
  sphere can't serve many lines; a pole per contact-line loop (a copy of the
  sphere constraint per loop) would, but then lines on two copies of one bead
  must still meet as one solid's. Both are design questions for the user.
  Speed (profiles of a 4-bead run, 190 s, and a 16-bead run's first 775 s):
  Evolver's gradient and Newton steps 41-51%, the wall-gap table rebuilt per
  wall 7-19%, checked edits 13-17% (the remesher's extra lists scanned per
  query: quadratic in a batch), _unguard 6%, crossings 4-5%; in slow steps the
  volume projection iterating (77%) over interpreted wall integrands (22%).

**Risks.** Every edit must keep all the bookkeeping (facetedge loops, body
facet lists, vertex edge loops, method and quantity instances, content
orientation); state outside the element blocks escapes a block snapshot
(runtime constraints, quantities, facet and fast-loop caches, the Hessian
pattern keyed on top_timestamp); delete_edge and edge_refine are linear model
only; refusals need fallbacks to reach the target; size about 3-5k lines of C
for phases 1-4 (estimate).

## Decisions (agreed with the user)

- **Bindings:** nanobind extension around a headless Evolver; the C glue in
  `bindings/pyse_api.c` contains Evolver's setjmp/longjmp error handling.
- **`src/` is our fork** of Surface Evolver 2.70a (the first commit is the untouched
  original). Bug fixes apply to every build; only library hooks sit behind `PYSE`.
  Small, separate commits. Engine variables we add default to the old behaviour
  (`collapse_max_tilt`, `collapse_max_edge`: 0).
- **Numerics:** speed first, physics correct. Results may differ from serial Evolver
  at round-off, and with threads from run to run (MUMPS). **Bit-identity is not a
  goal**; verify with tolerances.
- **Robustness:** robust defaults over per-case knobs; examples and the stress suite
  run on plain defaults. Newton steps stay exact (no chord Newton or factorization
  reuse: the user's problems are often non-smooth, such as foams and triple junctions).
- **Engine model:** one engine per process; `Evolver` objects are handles to it.
  `save()`/`restore()` for snapshots, `pyse.map` (worker processes) for sweeps.
- **Threads:** OpenMP, default the number of physical cores; `pyse.set_threads`,
  `OMP_NUM_THREADS`, `PYSE_THREADS`; `pyse.map` workers use one thread each. Evolver's
  pthread mode is compiled out.
- **Factorization:** MUMPS (AMD ordering, analysis kept across steps); Evolver's own as
  the fallback. Wheels bundle an OpenMP OpenBLAS 0.3.34.
- **User's workload:** linear soapfilm and a Lagrange final stage
  (`lagrange n; g 5; hessian`), Newton at 100k+ facets, surfaces of 100k-1M facets,
  sweeps, FEM meshing.
- **API:** pre-1.0, may break without deprecation shims. pySE's own code is MIT.
- **Process:** ask design questions before feature-sized work; one commit per item;
  report measurements, negative results included; push only when the user asks.

## What exists (by phase)

| Phase | What | Headline |
|---|---|---|
| A | memory/UB fixes, exact dumps, `save`/`restore`, handle engine | -23% instructions per step |
| B | benchmark, direct element writes, `mesh()`, `pyse.map` | |
| C, C2 | parallel facet/quantity loops, compact caches, MUMPS, parallel Hessian assembly, Lagrange kernels, parallel `u`/`V` | against 2.70a at 8 threads: `g 1` 8-11x, linear Newton 5-10x, Lagrange 6 Newton 3.8-5.1x |
| D | live view, tessellation limit, HTML reprs, busy timeout, Sphinx docs + tutorial | 0.6.0 |
| F | friendlier API: `relax(levels, tidy, newton)`, `ev.newton`, `ev.remesh`, `ev.body(i)`, mask selections, constraint builders, continuations, diagnostics as values | |
| G, G2 | stress suite, `ev.residual()`, safeguarded Newton, stability check, conjugate-gradient fallback, contact-line spoke pass, Newton-first in Lagrange, `ev.adapt()` with the guarded collapse in C, `ev.health()` | 0.7.0; suite 1 -> 7 of 9 on defaults |
| H | Brakke's samples and BGA series as notebooks, module split of `_evolver.py`, `shapes`, named walls, curved wires, stress cases 13-17, slow-rounds Newton trial, repeated spoke splits | 0.7.1-0.7.4; 14 of 14 stress cases |

Numbers: docs/performance.md (engine and `relax()` at scale). What `relax()` does and
where it stops working: docs/relaxing.md. The stress cases as examples:
docs/stress_cases.md.

**Stress suite** (`.venv/bin/python bench/stress/run.py [case numbers]`, about 6 min,
results in bench/stress/results.json; `bench/stress/figures.py` redraws the docs'
figures): 14 cases, all passing, 13 on defaults and case 8 with explicit remeshing.
Case 8 varies a lot between identical runs: judge changes on it with three or more
paired runs and the median. Liquid-bridge notebook grid: 25 of 25 on `relax(levels=3)`.

**Lessons from G/G2 that still constrain design:**
- Thresholds layered on Evolver's mesh operations (`V`, `l`, `t`, `K`, `conj_grad`)
  tend to fix one case and break another; judge every change on the whole suite.
- `ev.residual()` sees only the shape-direction force balance, not volume constraint
  violations: check volumes separately before declaring convergence (`_volumes_met`).
- Newton in the linear model has no tangential mesh motion: continuations with
  travelling contact lines need the gradient rounds' `u; V`. Hence Newton-first
  only in quadratic/Lagrange, and after `lagrange n` it needs `g 5` first (Newton
  diverged at 45k facets without it).
- Evolver's `l` ignores `no_refine`; boundary-wire vertices give spurious Newton
  freedoms (handled by `hessian_slant_cutoff` 0.05); measure drop contact radii
  about the rim's centre (drops slide).
- `stability()`: one factorization; `ritz` only when something is negative.

## How we work (lessons)

**Measuring**
- Profile before optimizing; drop the item if the profile doesn't support it.
- Check the machine first (`uptime`, `ps --sort=-pcpu`); report the load with timings.
- A/B against a reference build (`git worktree` + venv), interleaved, best of N. If a
  few-% change shows mixed results, drop it and move on.
- Compare at 1 and 4 threads (8 is noisy here); time loops themselves for loop-level
  questions. Instruction counts miss memory effects: confirm with wall clock.
- Parallel experiment runs: set `OMP_NUM_THREADS` (2-4) per run, or they oversubscribe.
- `perf`: `/usr/lib/linux-tools-6.8.0-146/perf`; `-D -1 --control fifo:<f>` profiles just
  the iterations; `--sort pid` shows whether the main thread is the bottleneck.

**Where the time goes**
- Once the main thread's serial code dominates, faster parallel loops stop paying.
- Evolver's cost is memory traffic (linked lists, facet-edge walks); compact caches keyed
  on `top_timestamp` paid, arithmetic didn't. Every cache is checked under
  `PYSE_CHECK_FACET_CACHE=1`, and its invalidation test must fail when the bump is
  removed.

**Correctness bar** (every change to `src/`)
- Single evaluations agree with Evolver's loops (`PYSE_NO_FAST_LOOPS=1`) to ~1e-12;
  relaxed equilibria to ~1e-9 at 1, 4 and 8 threads (1e-8 against stock Evolver).
- Full tests with `PYSE_CHECK_FACET_CACHE=1`; `tools/run_sanitizers.sh` (includes the
  OpenMP run on refined surfaces); check compiler warnings.

## Tools and conventions

- The local `.venv` install is not editable: `pip install .` after every edit, or the
  tests run the old package.
- The local `.venv` has numpy 1.26 and old PyVista, CI the latest (numpy floats
  formatted with `!r` break Evolver commands and change printed text). A CI-like venv:
  `python -m venv <dir>; <dir>/bin/pip install ".[test]" mypy scipy`.
- CI runs `python -m mypy -p pysurfaceevolver`; run it before pushing.
- Docs: `sphinx-build -W docs <out>`; notebooks execute in the build (600 s each,
  drainage 900 s). docs/stress_cases.md is not executed in the build:
  `OMP_NUM_THREADS=1 python tools/check_doc_snippets.py --jobs 4` runs and compares its snippets
  (also a CI job; `--update` rewrites the outputs).
- Tests: `pytest`. Benchmark: `python bench/benchmark.py --levels 6 8 --threads 1 4`.
- Profiling build: `pip install . -C build-dir=<dir> -C cmake.define.PYSE_NOSTRIP=ON
  -C install.strip=false -C cmake.define.CMAKE_C_FLAGS="-g -fno-omit-frame-pointer"`.
  Memcheck: `PYTHONMALLOC=malloc valgrind --suppressions=tools/valgrind.supp python <script>`.
- Evolver sources use CRLF: edit with a script that keeps CRLF. Our own files
  (`fastloops.c/h`, `fasthess.c`, bindings, Python) are LF.
- Don't `pkill -f` a pattern that also appears in the same shell command (it kills the
  shell); kill by PID in a separate command.
- Release: first replicate every CI job in a fresh CI-like venv (tests with
  `PYSE_CHECK_FACET_CACHE=1`, mypy from the repo root, the docs build, the snippet
  check). Then bump `__version__` in `python/pysurfaceevolver/__init__.py`, date the
  CHANGELOG section, annotated tag `vX.Y.Z` ("pySurfaceEvolver X.Y.Z (see
  CHANGELOG.md)"); the tag builds wheels as workflow artifacts.
