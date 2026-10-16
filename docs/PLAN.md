# pySurfaceEvolver development plan

Living document: decisions, status, how we work, and what is open. Condensed at the
0.7.0 release (2026-10-10); the full history of every phase, with its measurements and
dropped experiments, is in git (`git show e8f7434:docs/PLAN.md`, or
`git log -p docs/PLAN.md`). What changed for users is in CHANGELOG.md.

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
Found, not fixed:
* `arclength(tidy=True)` stalls in the quadratic model near a bridge's fold (steps
  down to 7e-7, disk_bridge.md runs `tidy=False`); `halvings=3` too few at some folds
  with gravity (`halvings=6` there).
* A contact line (or a lens's rim) can't travel more than an edge or two: equiangulation
  doesn't swap boundary edges and vertex averaging doesn't move boundary vertices
  (floating_lens.md starts at the exact rim). Under gravity a second refinement of the
  lens didn't converge (residual 0.04, 56k facets).
* `relax()` doesn't settle a long finite column's length (tangential motion; residual
  and Newton steps are normal only): fibres.md uses a periodic slice.

**State:** all 14 stress cases pass, 13 on `relax()`'s defaults and case 8 (the drainage
band) with `ev.remesh(target=h)` each step, as its notebook runs it. 485 tests.

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
* **Long-term:** isotropic remeshing in C toward a size field; phase E (not now).
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
