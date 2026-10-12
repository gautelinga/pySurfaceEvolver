# pySurfaceEvolver development plan

Living document: decisions, status, how we work, and what is open. Condensed at the
0.7.0 release (2026-10-10); the full history of every phase, with its measurements and
dropped experiments, is in git (`git show e8f7434:docs/PLAN.md`, or
`git log -p docs/PLAN.md`). What changed for users is in CHANGELOG.md.

## Resume here (2026-10-12)

0.7.3 released 2026-10-12 (tag v0.7.3): everything below since 0.7.2, after a three-pillar
review (speed, simplicity, friendliness) whose findings were all fixed.
0.7.2 released 2026-10-12 (tag v0.7.2; the user chose 0.7.2 despite the API removals).
Its CI failed on one test (tankex at one level asserted a thread-dependent energy; 0.7.1
failed it too on one thread), so its x86 wheels and sdist didn't build; fixed in c6068eb.

After 0.7.2 (2026-10-12, CHANGELOG "Unreleased"; research by four agents, then the user
asked to fix every correctness/friendliness, speed and simplicity finding and try
Brakke's tips): `_evolver.py` split into `_errors`, `_results`, `_views`, `_geometry`,
`_engine` (`surface` memory: relax's failed attempts, walls, load generation),
`_analysis`, `_relax` (`_Run` holds one relax() call's settings and stages) and
`_export`; Evolver = _Relaxing(_Analysis(_Engine)) + _Export. Bulk writes through
scratch attributes (`pyse_mark`, `pyse_value`) and one `set ... where`. New: `shapes`,
orientation/plane warnings at load, typo-proof `ev[...] =` (`define()`),
`SharedEngineWarning`, `Body(volume, faces=None)` with automatic volconst,
`Evolver.energy` (a datafile function `pyse_energy_constant()`), named constraints,
per-vertex constraint sets, curved wires (`boundaries=`; wire vertices fixed: free
ones bunch up, real negative modes), `contact_lines`, `energies` (energy_terms
binding), `derivative`, `continuation(stop_on, halvings)`, `stability(eigenvalues=)`.
C speed: Hessian setup slant only on constrained vertices, Lagrange kernels, Lagrange
normals in the vn cache, V from per-topology rings. Tried and dropped (CHANGELOG):
reorder_storage, CG in the rounds, sobolev_mode steps, immediate_autopop,
target_tolerance; fewer Lagrange line-search passes, PORD.
Then (same day, user: "do 3 and 4"): Brakke's BGA series vendored (bga-2..10, 30; 11, 12,
31 diverge with his own recipes) with docs/solder_joints.ipynb (tools/make_solder_notebook.py);
residual counts optimizing-parameter forces; Evolver fixes in lexinit2.c (pressures reset
by energy quantities on load -> Newton refused after restore) and userio.c (warning
buffer memcpy overlap); _Coords carries optimizing parameters; derivative(move, method);
relax refines past a start without freedom; health ignores placement-only constraints;
relax groups Evolver warnings. Lagrange g step: 9 element passes instead of 14 (agent;
bit-identical, g 1 10-20% faster, relax 3-5%). Case 8 on old and new code
alike: worst 3-6%, ~120 unconverged steps of ~120 states, 240-280 s.
0.7.1 released 2026-10-11 (tag v0.7.1; wheels as workflow artifacts). 0.7.0 is tagged
at fd9783f, whose wheel build failed (fixed in b5216d4: no fused multiply-adds on
aarch64, render probe in a subprocess); its tag stays.

In 0.7.1 (CHANGELOG): stress-case figures
(`bench/stress/figures.py`); `relax()` and `health()` on all 25 Brakke samples (strings,
simplex, no-Hessian fallback, 10% Newton energy cap, topology stamp for undo, elastic
and crystalline energies keep the mesh, Plateau junctions popped by default, a guarded
last resort that lets collapsing necks/films pinch off); edge deletion refuses only
merges that duplicate facets (studied on an instrumented 2.70a: cat.fe pinches as in
the original); five notebooks `docs/brakke_*.ipynb` (recipe vs relax per sample).

Also done (2026-10-11, "Found afterwards" in CHANGELOG): newton() meets new volume
targets (safeguard counts a missed target > 1e-4); last-resort backoff (2^k - 1 calls
skipped after k failures); `health().junctions` via a cone_analyze binding (matches `o`
on all 25 samples); volume guards for degenerate meshes and `health().target_gap`;
`IterationResult.settled`; crystalline converged by energy settling; stability checked
within 100*tol (barrel roll-up now reported at 70 and 80); wheels CI on main + weekly;
`tools/make_brakke_notebooks.py`; `tools/check_doc_snippets.py` (+ docs CI job). The
"spurious quotient-space negative mode" was wrong: symtest.fe's volumes are comments,
so the instability is real. 341 tests. Case 8 on defaults (2 threads): about 500 s,
median 0.2%; the worst state is run-dependent (over 10% in about 1/4-1/3 of runs,
the same before and after this work: 11 old-code runs against ~30 new).

In 0.7.2 (2026-10-11/12, CHANGELOG; the user asked for speed,
simplicity and user-friendliness from Brakke's resources, chose all groups, "remove
outright" for API breaks and "run all phases, report at end"): Lagrange-stage Newton
stops at tol; stability from the last Newton step's inertia (cutoff 1e-2, ~230
comparisons, none disagreed); cached vertex normals (vn_* in fastloops.c);
getc_unlocked lexer; linear_metric in stability(); one early `u` after each phase's
first step (cbga1.fe; with `V` too it cost case 8 accuracy); API: quantity(), gravity,
density, pop(), saddle(), contact_angles(), Mesh.slice(), walls in plots, torus
unwrapping (facet_corners binding), glTF/VRML export, inferred body orientation;
removed ev(), hessian(), relax's cg/window/energy_tol/pinch, recipes setters. Eight of
Brakke's online examples vendored (fe/, attributed in THIRD_PARTY_LICENSES.txt); stress
case 13 (Schwarz P, D). Dropped after measuring: scale-factor watch (breaks settling
detection), fewer averagings in the CG retry (no gain), removing the CG retry (case 8
median error doubles), pull-based contact angles (worse than normals), a C speed-up of
save() (not profiled; only rescue paths use it). Candidates not done: splitting
`_evolver.py` (2700 lines) into modules; `volconst="auto"`; a BGA notebook on
parameter optimization (bga-2..12 need their .cmd files).

Open, for the user to pick:

* **Candidates:** `newton()` returning a result object; Evolver's finite-difference
  parameter Hessian limits relax to ~1e-7 with optimizing parameters (bga-5).

* **Known limits:** case 8 needs explicit remeshing (a mesh breakdown at the bead-wall corner) for
  full accuracy; mylarcube ends 0.1% above its recipe; the conjugate-gradient retry
  rescues ~1 step in 130 in case 8 but keeps its median error down; topology tests
  are path-sensitive at several threads (run with threads=1).
* **Do not re-propose:** remeshing inside `relax()` for case 8 (researched 2026-10-10).
* **Long-term:** isotropic remeshing in C toward a size field; phase E (not now).
* **Small loose ends:** PORD ordering for long Newton runs at 1.6M; quadratic/edge
  setups on generic `mat_mult`; serial `r`; metric.c conformal 1/4 (unverified).
* **Scripts:** run benchmarks and tests with OMP_NUM_THREADS=4 or less.

Ask the user before each step.

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

Numbers: docs/performance.md (engine and `relax()` at scale). What `relax()` does and
where it stops working: docs/relaxing.md. The stress cases as examples:
docs/stress_cases.md.

**Stress suite** (`.venv/bin/python bench/stress/run.py [case numbers]`, about 3 min,
results in bench/stress/results.json): 7 of 9 pass on defaults (1 column past
Rayleigh-Plateau, 2 catenoid to its fold, 5 bridge, 6 barrel, 9 inflated cube, 11
puddle, 12 shrinking cap). Case 3 (sessile drop) fails only at 170 degrees; case 8
(drainage band) needs `ev.remesh(target=h)` each step. Liquid-bridge notebook grid:
25 of 25 on `relax(levels=3)`.

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
- The local `.venv` has numpy 1.26, CI numpy 2 (numpy floats formatted with `!r` break
  Evolver commands). A CI-like venv: `python -m venv <dir>; <dir>/bin/pip install ".[docs]"`.
- CI runs `python -m mypy -p pysurfaceevolver`; run it before pushing.
- Docs: `sphinx-build -W docs <out>`; notebooks execute in the build (600 s each,
  drainage 900 s). docs/stress_cases.md is not executed: rerun its snippets by hand
  when relax() changes.
- Tests: `pytest`. Benchmark: `python bench/benchmark.py --levels 6 8 --threads 1 4`.
- Profiling build: `pip install . -C build-dir=<dir> -C cmake.define.PYSE_NOSTRIP=ON
  -C install.strip=false -C cmake.define.CMAKE_C_FLAGS="-g -fno-omit-frame-pointer"`.
  Memcheck: `PYTHONMALLOC=malloc valgrind --suppressions=tools/valgrind.supp python <script>`.
- Evolver sources use CRLF: edit with a script that keeps CRLF. Our own files
  (`fastloops.c/h`, `fasthess.c`, bindings, Python) are LF.
- Don't `pkill -f` a pattern that also appears in the same shell command (it kills the
  shell); kill by PID in a separate command.
- Release: bump `__version__` in `python/pysurfaceevolver/__init__.py`, date the
  CHANGELOG section, annotated tag `vX.Y.Z` ("pySurfaceEvolver X.Y.Z (see
  CHANGELOG.md)"); the tag builds wheels as workflow artifacts.
