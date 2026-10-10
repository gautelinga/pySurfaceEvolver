# pySurfaceEvolver development plan

Living document: decisions, status, how we work, and the remaining work. Update
it when a step is finished or a decision changes.

## Decisions (agreed with the user)

- **Bindings:** nanobind extension around a headless Evolver; C glue in
  `bindings/pyse_api.c` contains Evolver's setjmp/longjmp error handling.
- **`src/` is our fork** of Surface Evolver 2.70a. The first git commit is the
  untouched original. Bug fixes apply to every build (not behind `#ifdef PYSE`);
  only library hooks are guarded. Keep fixes in small, separate commits.
- **Numerics:** speed first, as long as the physics is correct. Results may differ
  from serial Evolver at round-off level, and with several threads also from run
  to run at round-off level (MUMPS's threaded factorization; making it bit-exact
  costs 5-30% of a Newton step, not worth it). One thread is bit-reproducible. **Bit-identity is not a goal**: choose the fastest design
  and verify with tolerances (below). Identical results are only a free check when
  a change keeps the summation order anyway.
- **Engine model:** one engine per process; every `Evolver` object is a handle to
  it. `save()`/`restore()` for exact snapshots; `pyse.map` (worker processes,
  crash-isolated) for sweeps. Long term: move Evolver's globals into one state
  struct (phase E).
- **Threads:** OpenMP; default = physical cores; `pyse.set_threads(n)`,
  `OMP_NUM_THREADS`, `PYSE_THREADS`; `pyse.map` workers 1 thread each. Evolver's
  old pthread mode (`-p`, `thread_launch`, `THREADS` blocks) stays in the source
  but is compiled out and unsupported (user's choice: keep `src/` close to
  upstream); per-thread data (`GET_THREAD_DATA`) is per OpenMP thread. Scaling beyond 4 threads is modest on the
  development machine (4 fast + 4 compact cores, 2-way SMT, laptop memory
  bandwidth); after the false-sharing fix (2026-10-08) 8 threads beat 4 by 10-40%
  at 393k+ facets, less for small surfaces. The default thread count is decided in C2
  step 6 from measurements.
- **User's workload:** linear soapfilm *and* quadratic/Lagrange (as a final
  high-precision stage: `lagrange n; g 5; hessian` ladders), Newton steps at
  100k+ facets, surfaces of 100k-1M facets, parameter sweeps, FEM meshing.
- **Process:** ask design questions before implementing feature-sized work; one
  commit per item; pause after each phase and at marked decision points to report
  measurements, including negative results.

## Status

Version 0.5.0 + C2 work; 231 tests; mypy clean; manylinux wheel builds and passes.

| Phase | What | Result |
|---|---|---|
| A | LTO; memory/UB fixes (path_open, kb_error strncat, delete_facet, string facet area, MAXINT, sdrv offsets, body-x crash); C23 fixes; sanitizer sweep; exact dumps; `save`/`restore`; handle engine; `relax()`; bundled samples | -23% instructions per iteration |
| B | `bench/benchmark.py`; direct C element writes; `mesh()`; vectorized tessellation and watertight checks; `pyse.map` | |
| C | facet corner cache; parallel facet volume/energy/force loops (`src/fastloops.c`); `pyse.set_threads` | 1.6M facets: 11.1 -> 2.3 s per iteration |
| C2.1 | per-thread accumulation in the facet loops (compensated sums, per-thread force arrays) | loops faster, iteration unchanged: serial code dominated |
| C2.5 | parallel volume gradients (`fl_film_grad`); per-facet body table; cached vertex list for per-vertex work (forces, move, save/restore, volume restoration, DV^T DV); `FL_FOR_SELECTED` for constraint/boundary vertices and edge integrals; `sp_hash` overflow fix | see below |

One linear iteration, 1.6M-facet cube, quiet machine, best of N:

| threads | v0.5.0 | now |
|---|---|---|
| 1 | 1.97 s | 1.30 s |
| 4 | 1.45 s | 0.65 s |
| 8 | 1.43 s | 0.74 s |

98k facets: 62 -> 32 ms at 1 thread. The main thread is now about as busy as the
workers; the serial remainder is ~1-2% items.

**End of C2 vs v0.5.0** (`bench/benchmark.py --threads 8`, same machine, same
energies; Lagrange stage = `lagrange n; g 5; hessian` x3 after a converged
linear surface):

| | v0.5.0 | C2 |
|---|---|---|
| linear 98k: `g 1` / Newton step | 75 ms / 0.49 s | 40 ms / 0.24 s |
| linear 1.6M: `g 1` / Newton step | 1.34 s / 25.5 s | 0.58 s / 6.4 s |
| Lagrange 2 stage, 24k facets | 2.5 s | 0.9 s |
| Lagrange 4 stage, 24k facets | 17.7 s | 5.4 s |
| Lagrange 6 stage, 24k facets | 88.9 s | 22.5 s |
| Lagrange 6 stage, 6k facets | 17.5 s | 5.0 s |

**Where a typical run spends its time now** (cube at 98k facets, 4 threads:
`g 20; hessian; hessian; lagrange 2; g 5; hessian; lagrange 4; g 5; hessian`,
61 s in total):

| stage | time | share |
|---|---|---|
| linear: `g 20` + 2 Newton steps | 1.5 s | 2% |
| Lagrange 2: `g 5` + Newton | 8.0 s | 13% |
| Lagrange 4: `g 5` | 18.4 s | 30% |
| Lagrange 4: Newton step | 32.2 s | 53% |

- A Lagrange-4 gradient step (3.7 s, ~120 linear iterations) is 89% serial.
  Two-thirds of it is `mat_mult` in `q_facet_setup`: control points to quadrature
  points and tangents, through the generic `REAL**` multiply, redone for every
  quantity (area, each body volume) on the same facet. By caller: volume 45%,
  energy 20%, gradients 18%; the method code itself is a few percent.
- A Lagrange-4 Newton step is 99.96% serial: factorization 65% (`factor_recur` in
  `xmd_factor`), Hessian assembly 28% (`hessian_fill`: tension and volume
  Hessians), the rest 7%.
- That ladder starts far from equilibrium at 98k facets, so its Lagrange Newton
  steps raise the energy (v0.5.0 does the same; at 6k facets every build matches
  and converges). Benchmarks must converge the linear stage first.

## How we work (lessons)

**Measuring**

- Profile before optimizing, set a target, drop the item if the profile doesn't
  support it (one-call `mesh()` and the live-view fast path gave nothing).
- Before timing, check the machine: `uptime` and `ps --sort=-pcpu`. An orphaned
  `pyse.map` sweep once ran 4 h on 3-4 cores and skewed a whole session.
- A/B against a reference build (`git worktree` + venv), interleaved, best of N.
  Take one careful measurement; if a small change (a few %) shows mixed or
  unexplained results, drop it or note it and move on (the parallel edge-content
  scan: +5% at 4 threads, -17% at 8, dropped then; the same idea later went in as
  part of `FL_FOR_SELECTED`).
- Compare at 1 and 4 threads; 8-thread whole-iteration timings are noisy here. For
  loop-level questions, time the loop itself (temporary `omp_get_wtime` timers
  around the `fl_*` functions) rather than the iteration.
- Instruction counts (callgrind) miss memory effects (the corner cache cut
  instructions 6% and wall time 38%); confirm with wall clock.
- `perf`: use `/usr/lib/linux-tools-6.8.0-146/perf` (the 6.17 kernel's tools
  package has no perf). `-D -1 --control fifo:<f>`, with the script writing
  `enable`/`disable` to `<f>`, profiles just the iterations. `--sort pid` shows
  whether the main thread is the bottleneck; `--call-graph fp` with a
  `-fno-omit-frame-pointer` build gives callers.

**Where the time goes**

- Speeding up parallel loops stops paying once the main thread's serial code
  dominates (C2.1 changed nothing at the iteration level). Look at the main
  thread's share first.
- Evolver's cost is mostly memory traffic: linked-list traversals
  (`FOR_ALL_*`), facet-edge walks, and large element records. The fixes that paid
  were compact caches keyed on `top_timestamp` (corners, facet list, vertex list,
  facet bodies), not arithmetic.
- `set_facet_body()` bumps `top_timestamp`; NONCONTENT changes and body deletion
  bump `fl_body_stamp`. Every cache is verified under `PYSE_CHECK_FACET_CACHE=1`,
  and an invalidation test must fail when its bump is removed (mutation check).
- Code that evaluates expressions or sums in order stays serial but runs only over
  the elements that need it (`FL_FOR_SELECTED`); everything else per-vertex or
  per-facet runs in parallel.

**Correctness bar** (every change to `src/`)

- Single evaluations (energy, area, volumes, forces) agree with Evolver's original
  loops (`PYSE_NO_FAST_LOOPS=1`) to ~1e-12 relative; relaxed equilibria (energy,
  volumes, pressures, Hessian index) to ~1e-9 at 1, 4 and 8 threads. Runs that
  stop short of equilibrium amplify round-off (cube `g 5; r; g 10; hessian`:
  1.6e-9), so regression values against stock Evolver use 1e-8.
- Full tests with `PYSE_CHECK_FACET_CACHE=1`; `tools/run_sanitizers.sh`; compiler
  warnings (the `-Wdangling-else` ones from hooks before `FOR_ALL_*` macros are
  expected).
- Also run ASan/UBSan with OpenMP on refined surfaces (above the 4096-facet
  threshold, with `V`, `u`, `w`, `hessian`, `set facet noncontent`): this found the
  `sp_hash` overflow that the small-sample sweep never reaches.

## Phase C2 (remaining, in this order)

1. **Done.** Packed, vectorized setup kernel (`fl_lagrange_facet_setup`), same
   results: Lagrange-4 gradient step at 98k facets 3.71 -> ~1.4 s, Lagrange-2
   `g 5` 4.85 -> 2.64 s; the typical-run ladder 65.7 -> 47.6 s. Setup is now ~25%
   of a step (kernel), `get_facet_verts` ~10% (gathering control points); both go
   parallel in step 4. Sharing setup across passes was not done: setup is already
   shared across quantities within a pass, and coordinates change between passes.
   The quadratic-model and edge setups still use `mat_mult` (not in the user's
   ladder). Original item: **Lagrange facet setup** (small to medium, quick win
   for every curved-element iteration). In `q_facet_setup_lagrange` (and the quadratic variant):
   1. Replace the generic `mat_mult` calls with a fixed-shape kernel on contiguous
      arrays (quadrature points x control points x 3).
   2. Compute each facet's quadrature points and tangents once per evaluation and
      share them between quantities (area, every body volume), instead of redoing
      the setup per quantity. Invalidate when coordinates move.
   3. Measure on the Lagrange-2/4 gradient steps; target: setup from two-thirds of
      a step to a small fraction. Serial gains only; threads come in step 4.
2. **Done.** Benchmark: `bench/benchmark.py` (extended, not replaced). Linear
   section: bulk operations plus `g 1` and `hessian` on a relaxed surface
   (`--levels`). Lagrange section (`--lagrange-levels`, default 6k and 24k
   facets; `--orders`, default 2 4 6): the manual's pattern, linear relaxation
   then per order `lagrange n; g 5; hessian; hessian; hessian`, each timed.
   `--threads 1 4`, JSON output, load warning.
   The Evolver manual (5.3, 16.11) is explicit that the Lagrange model is a
   limited, short final stage: no refining or other triangulation changes, a
   few `g` steps and a few Newton steps, on a mesh settled in the linear model;
   higher order on a coarse mesh beats refinement for accuracy (refinement 2 at
   order 6: 3e-10; linear at refinement 5: 8e-4) and cost climbs steeply with
   order. So **Newton steps at high order are the Lagrange cost**, not gradient
   steps; the facet counts at which the user converts are still to be confirmed
   (benchmark defaults 6k and 24k).
3. **Faster Newton steps** (medium; the main cost of both the Lagrange stage and
   large linear runs).
   1. METIS experiment: port `src/metis.c` to the METIS 5 API, build METIS from
      source via CMake (Apache-2.0, shippable in wheels), compare fill-in and
      factorization time with the current ordering (linear and Lagrange 2/4/6,
      benchmark sizes). Default only if it wins. (System has libmetis.so.5 but no
      headers.)
   2. Parallel Hessian assembly (`hessian_fill`, `calc_quant_hess`): per-facet
      blocks computed in parallel, merged into the sparse matrix (the hash in
      `matrix.c`).
   3. **Decided (2026-10-07): MUMPS** replaces Evolver's factorization as the
      default for Newton steps; Evolver's own (mindeg) stays as an option and
      as the fallback in builds without MUMPS. Measured on exported Newton
      matrices (`PYSE_DUMP_HESSIAN`), factor incl. ordering, 4 threads:
      linear 393k facets 1.39 s (Evolver) -> MUMPS 0.31 s / PARDISO 0.21 s;
      linear 1.6M 18.9 s -> 2.2 / 1.7 s (1 thread: 3.0 / 3.1 s); Lagrange 4
      24k 1.79 -> 0.43 / 0.30 s; Lagrange 6 24k 7.9 -> 1.03 / 0.74 s. CHOLMOD
      (simplicial LDL^T, the only indefinite option) no faster than Evolver.
      MUMPS: open source (CeCILL-C), inertia incl. null pivots, sequential
      build without MPI, any platform; PARDISO: ~1.3x faster but closed MKL,
      ~200 MB, x86 only. The ordering can be reused while the topology is
      unchanged. Expected Newton step: linear 1.6M 26 -> ~9 s; Lagrange 6 24k
      (4 threads) 16 -> ~9 s.
      Done before the decision: parallel Hessian assembly for Lagrange facet
      quantities (`src/fasthess.c`: order 6 at 24k, 4 threads, 22 -> 14.7 s);
      METIS 5 port of `metis.c` and the `metis_factor` fall-through fix
      (METIS ordering alone: -3% to -16%, slower at 98k).
   4. **Done.** MUMPS integrated (`src/mumpsfactor.c`, optional FetchContent
      build, OpenBLAS; manylinux wheel 15.8 MB bundles OpenBLAS + gfortran
      runtime). Matches Evolver's factoring (energies 1e-16, same index).
   5. **Done.** Parallel assembly for the linear Newton path (`area_hessian`,
      `body_hessian` via the runner in `fasthess.c`).
      One Newton step, 4 threads, now vs v0.5.0-era: linear 393k facets
      3.1 -> 1.2 s, 1.6M 25.4 -> 6.0 s; Lagrange 6 at 24k 22.6 -> 8.2 s.
      Wheel BLAS: AlmaLinux 8's serial OpenBLAS 0.3.15. Threaded BLAS inside
      MUMPS's threads oversubscribes (pthreads OpenBLAS is made serial at run
      time), so serial is right; the wheel is still 0-19% slower than a dev
      build with OpenBLAS 0.3.26 (likely no Zen 4/5 kernels in 0.3.15).
      Follow-up option: build a recent OpenBLAS from source for the wheels.
4. **Done.** Thread defaults (user's choices, 2026-10-07): physical cores (at most
   the available processors; OMP_NUM_THREADS / PYSE_THREADS / set_threads
   override; also given to OpenMP so MUMPS follows); `pyse.map` workers 1 thread
   each. A pthreads OpenBLAS is made single-threaded at run time (its threads
   inside MUMPS's oversubscribe: 393k-facet Newton step at 8 threads 3.0 -> 1.2 s).
   Measured 1/4/8 threads, Newton step 393k: 1.65 / 1.23 / 1.21 s; Lagrange 4
   at 24k: 3.01 / 1.82 / 1.60 s. Also: Ctrl-C is safe with threads (aborts are
   deferred out of parallel regions and MUMPS; signals on worker threads are
   forwarded to the engine thread).
5. **Done.** Parallel named-quantity loops (user chose the full scope). Done:
   facet loops of `calc_quants`/`calc_quant_grads` for the area and volume
   methods (linear, Lagrange), record-and-replay, bit-identical at 1 thread;
   `g 5` at 24k facets: order 2 0.51 -> 0.17 s, 4 1.84 -> 0.75 s, 6 4.66 ->
   1.82 s. Then: per-OpenMP-thread `thread_data` (eval stacks); error traps
   (`kb_error` in a parallel loop jumps back; the element is redone serially,
   so messages are unchanged); facet/edge/vertex integral methods with
   read-only integrands (expression node whitelist); values, gradients and
   Hessians. Dead pthread code kept (see Decisions).
   Original text: (large; demoted: Lagrange runs take few
   gradient steps, so this matters mainly for named-quantity-heavy models).
   Parallelize the per-facet quantity value/gradient loops (`calc_quants`,
   `calc_quant_grads` and the methods) with OpenMP, reusing the caches and
   `FL_FOR_SELECTED`; per-thread accumulation. Retire Evolver's old threading
   code in the same step: drop its scheduling (worker threads, `thread_launch`,
   `-p`), turn its per-thread data (`thread_data`: eval stacks, `q_info`) into
   OpenMP thread-local storage. Re-profile before starting.

Done earlier in C2: per-thread accumulation in the facet loops (was step 1) and
the remaining linear hot spots (was step 5).

Pause after step 3 (solver decision) and at the end of C2.

## Current state and next steps

### Resume here (2026-10-10, end of session)

**Where things stand.** Pushed to GitHub (main, 272a31f; CI tests and docs green):
phase F (friendlier API: `ev.body(i)`, mask selections, `relax(levels, tidy,
newton)`, `ev.newton`, `ev.remesh`, diagnostics, mirrored plots,
`pyse.constraints` builders with contact angles, `pyse.recipes.continuation`),
phase G (stress suite in `bench/stress/`, `ev.residual()`, robust `relax()`
defaults) and G2.2-G2.3 (no spurious Newton freedoms on wires, stability reports).
Local only (not pushed): 4809c2d, the revised G2 plan. Version still 0.6.0;
everything since is under "Unreleased" in CHANGELOG.md.

**relax() today** (default): rounds of 10 gradient steps with `u; V`, then up to 20
safeguarded Newton steps (plain first, line search if rejected; a step is undone
if anything turns non-finite or both energy and residual rise), converged when
`ev.residual()` < tol (1e-8; tol is a residual tolerance now, `energy_tol` ends
the gradient phase: 1e-5 by default when Newton follows, 1e-9 with newton=0), then `ev.stability()` (`result.stable`,
`UnstableEquilibriumWarning`). Opt-in, not robust yet: `cg=True` (conjugate
gradients), `remesh=True` (graded remeshing).

**Stress suite** (`.venv/bin/python bench/stress/run.py [case numbers]`, defaults
only, ~3 min for all; results in bench/stress/results.json): 7 of 9 pass (1
cylinder past Rayleigh-Plateau, 2 catenoid to its fold, 5 bridge, 6 barrel, 9
inflated cube, 11 puddle, 12 shrinking cap). Failing: 3 sessile drop at 150-170
degrees (one level more fixes 150-160), 8 drainage band (worst 8-15% at level
2, ~20 states over 1%; level 3 with the old tolerance: 2.4%, snap exact).

**Next** (Phase G2 below, revised): G2.4, G2.4b, G2.5 and G2.6 done
(2026-10-10, see there; the liquid-bridge notebook is on `relax(levels=3)`;
relax() falls back to conjugate gradients), G2.7 in part (`ev.adapt()`, opt-in
`relax(adapt=...)`), G2.8 coarsening (guarded collapse in C; adapt still
opt-in: with it on everywhere the suite passes but runs several times longer),
G2.9 health report. Deferred by the user: adapt by default, case 8, case 3 at
160-170. Next: G2.10 examples on defaults (the drainage helper without its
crutches). Ask the user before each step.

**Engine changes this round** (src/, CRLF): edge deletion keeps the link condition
(trirevis.c, upstream bug); hessian_normal's slant measured against the raw normal
by magnitude, `hessian_slant_cutoff` 0.05 (hessian.c, lexinit.c); bindings:
`pyse_residual` (force balance in the shape directions, multipliers fitted by
least squares there; bindings/pyse_api.c, module.cpp).

**Practical** (see also "How we work"):
* The local .venv has numpy 1.26, CI numpy 2: numpy floats formatted with `!r`
  become `np.float64(...)` and break Evolver commands. CI's docs build catches it;
  a CI-like venv is scratchpad/venv_ci (rebuild: `pip install ".[docs]"`).
* CI runs `python -m mypy -p pysurfaceevolver` in the tests job; run it before
  pushing (local errors about cloudpickle/meshio/gmsh stubs are local-only).
* Docs notebooks execute in the build (600 s each; the drainage notebook sets 900 s).
* The user wants quick turnaround: run checks in parallel, keep reports short,
  don't chase last digits.

2026-10-08: C2 speed items done (MUMPS analysis reuse 4-8%; parallel Newton-step
normals 17-25% at 393k-1.6M); phase D complete (all nine items, see below).
User decisions (2026-10-08): pySE's own code is MIT (LICENSE); docs published to
GitHub Pages from main (workflow ready; the repo has no remote yet, Pages must be
enabled with "GitHub Actions" as source); next speed item: opt-in factorization
reuse over Newton steps; hash-free assembly not chosen; phase E not now.
Then measured before starting it: the factorization is only 11% (Lagrange 6 at
24k), 24% (393k) and 39% (1.6M) of a step, so reuse would gain little; serial
hash insertion was ~35% of a Lagrange 6 step. The user switched to hash-free
assembly: **done** (kept CSR pattern across steps, entries added in parallel
with atomics, misses merged through the hash; `PYSE_NO_PATTERN=1` off). Newton
step, 8 threads: 393k 0.88 -> 0.60 s, 1.6M 4.75 -> 3.6 s, Lagrange 6 at 24k
5.7 -> 4.15 s; 1 thread 11-14% faster. Energies identical in these runs.
Factorization reuse: dropped for now (small gain, algorithm change).
After the kept pattern the factor share grew (393k ~35%, 1.6M ~53%, Lagrange 6
~17%); the user asked for mixed precision first. Measured with a harness
(MUMPS single + double, AMD, 8 threads, saved Newton matrices): single factor only
1.2-1.5x faster (393k 0.19 -> 0.12 s, 1.6M 1.29 -> 0.90 s, Lagrange 6 0.65 -> 0.53 s),
and plain iterative refinement stalls (residual 8e-4 at 393k, 2e-2 at 1.6M after
30 solves: too ill-conditioned for single). **Dropped.** Useful number: a double
solve with existing factors is ~10x cheaper than a factor (393k 0.03 s, 1.6M
0.15 s, Lagrange 6 0.07 s), which makes "old factors as preconditioner" plausible.
Experiment (scratch build, not committed), 8 threads:
- Chord Newton (old factors reused, PYSE_CHORD prototype): a reused step costs
  393k 0.59 -> 0.40 s, 1.6M 3.7 -> 1.7 s, Lagrange 6 4.2 -> 3.5 s. Converges like
  full Newton for the cube (also from a rough start) and mound.fe; on the double
  bubble (triple junctions, two volume constraints) it does not settle: energy
  stays 1e-11..4e-8 above full Newton's and one step goes up. Needs a safeguard.
- Old factors as GMRES preconditioner (consistent right sides, steps 1->2, 2->3,
  1->3): cube 393k 3-6 iterations to 1e-12, bubble 10-17. A solve with old
  factors is only ~6-10x cheaper than a factor, and each step needs 1 + (number
  of constraints) solves: no gain at 393k, perhaps ~20% at 1.6M; much work
  (inertia, null pivots, several right sides).
Not built. User decision (2026-10-08): their problems are often non-smooth
(foams, triple junctions) and need robustness, so no chord Newton or other
factorization reuse; Newton steps stay exact.
Comparison with Brakke's original 2.70a (2026-10-08): table in README and
docs/performance.md (8 threads: `g 1` 10-13x, linear Newton 5-8x, Lagrange 6
Newton 4.5-5.6x; 1 thread: 4-8x / 2-4x / 1.4-1.7x). Re-measured with the OpenMP
OpenBLAS (static): 8 threads `g 1` 8-11x, linear Newton 5-10x, Lagrange 6 3.8-5.1x.
OpenBLAS: source builds link the system's OpenMP OpenBLAS statically with private
symbols (3c1dd35); a pthreads one loaded first by the system NumPy had replaced it.
Wheel re-checked after these CMake changes (cibuildwheel 4.3.0, podman, cp312
manylinux_2_28): shared OpenMP OpenBLAS 0.3.34 bundled under a mangled name,
16.4 MB, 232 tests pass; 393k Newton step 0.55 s (factor 0.16 s) from a plain venv.
Liquid-bridge example (docs/liquid_bridge.ipynb). Then, at the user's request, caps on
curved constraints and volume meshing in pySE: `ev.body_surfaces(cap=True)` caps each
opening on the constraint its rim lies on (ring caps, points projected by Evolver via
temporary vertices), `BodySurface.cap_ids`/`cap_constraints`, `BodySurface.volume_mesh()`
(Gmsh, discrete surfaces kept, physical groups).
Slit example (docs/slit_droplet.ipynb): bead between walls z = +-0.5, mirrors x = 0, y = 0,
x = ell; bead integrals over the dry part (poles inside the wetted region) + volconst.
A touching bead (R = 0.5) fails (zero-thickness wedge); R = 0.48 works. Droplets pinch
on the y = 0 mirror when the wall meniscus sag ~0.5(1 - sin t)/cos t reaches the
half-width, and for non-wetting beads; the notebook checks and explains.
MUMPS ordering re-checked now that the analysis is reused (2026-10-08, 8 threads,
analysis + factor): 393k AMD 0.06+0.157 s, AMF 0.05+0.159, PORD 0.45+0.124;
Lagrange 6 24k AMD 0.23+0.546, AMF 0.24+0.613, PORD 0.39+0.535; 1.6M AMD
0.43+1.31, AMF 0.47+1.07, PORD 2.07+0.86 (PORD = MUMPS's automatic choice; QAMD
worse). PORD's serial analysis needs ~4 steps per pattern to pay off at 1.6M
(~12 at 393k); typical ladders do 2-4. AMF: -18% factor at 1.6M linear only,
+12% for Lagrange 6. **No change** (AMD stays); possible later: PORD after the
2nd step on one pattern, for long Newton runs on big fixed meshes.
Speed items the user picked next (2026-10-08, "do 2, then 1"):
- Batched constraint solves (a35335f): sp_CHinvC / BK_hess_project_setup solve all
  constraint columns in one multi-right-side MUMPS call (blocks of 32). 9-body foam,
  8 threads: 46k 0.057 -> 0.050 s, 184k 0.288 -> 0.245 s per step; 2-49 constraints
  only (50+ go into the matrix).
- Lagrange element Hessians as BLAS products (fasthess.c `fl_lagrange_tension_hess`,
  dsyrk; `fl_lagrange_volume_hess`, dgemm; 2D facets; `PYSE_NO_FAST_LAGRANGE=1` off).
  Newton step, 8 threads, cube: Lagrange 6 at 24k 4.3 -> 1.7 s, at 6k 1.2 -> 0.38 s,
  Lagrange 4 0.24 -> 0.14 s; assembled Hessians match the loops to ~1e-15 relative.
  Found on the way: the Lagrange basis tables cached in fastloops.c went stale when
  `bezier_basis` rebuilt them in place (wrong energies); fixed with a version counter
  bumped by gauss_lagrange_setup().
  Valgrind's 85 "uninitialised value" reports (also with the new kernels off) were
  traced to MUMPS's mumps_build_sort_index: at -O3 gfortran tests an absent optional
  argument's stride before present(); harmless, none at -O2. Suppressed in
  tools/valgrind.supp; nothing in our code.
Profile after these (2026-10-08, cube): Lagrange 6 at 24k, 8 threads: fill 0.72 s
(calc_quant_hess 0.69), factor 0.45, 3 solves 0.11, move 0.14, init 0.07 (1.55 s);
1 thread 4.5 s, of which the BLAS kernel 15%, its scatter 9%, mixed_entry (normal
projection of every node-pair block, generic mat_mult, once per quantity) 12.6%.
Mesh commands at 393k, 8 threads: g 10 1.46 s, r 1.28, V 0.60, u 0.28, all three
serial. Done: V's search and u's swap tests in parallel (fastloops.c
fl_vertex_averages / fl_calc_edges / fl_edge_marks; PYSE_NO_FAST_MESH=1 off): u
0.30 -> 0.022 s (1 thread 0.11: lengths no longer recomputed five times), V 0.60 ->
0.11 s, `g 10; u; V` 2.2 -> 1.3-1.5 s; meshes and swap counts bit-identical to the
serial code (test, also with perturbed vertices). Swaps stay serial in the original
order; after a swap, edges in facets around its four vertices are tested again.
Done: mixed_entry's projection as direct loops (same arithmetic) and the element's
methods summed before projecting, so each block is entered once. Lagrange 6 at 24k:
1 thread 4.35-4.78 -> 4.09-4.13 s, 8 threads 1.53-1.71 -> 1.44-1.56 s; linear
unchanged (bit-identical Hessians); Lagrange Hessians within 2e-16. Most of what is
left in the fill is recording entries (put_entry) and the BLAS kernel.
Left: r has no single hotspot (left alone).
`g 1` threading (2026-10-08; this laptop: Ryzen AI 7 350, 4 Zen 5 + 4 Zen 5c cores):
quiet machine, 98k 1/2/4/8 threads 27.5/20.0/21.3/28.5 ms, 393k 252/156/133/146 ms.
Causes: false sharing (fl_facet_volumes' per-thread body sums and fl_calc_leftside's
per-thread matrices were adjacent, one cache line for all threads with few bodies;
now padded), and fl_sel_begin rescanning all edges/vertices for attribute bits at
every selected loop (12-20% of g). Selections are now kept per site until the
topology, the element counts or fl_attr_stamp change (bumped by set_attr/unset_attr,
element allocation/freeing, modify.c's NEGBOUNDARY flips; PYSE_CHECK_FACET_CACHE=1
verifies every reuse; stress over all samples with fix/density/constraint/tension
changes between g steps: no stale selection). After (load ~5): 98k 21.2/15.2/12.0/
12.7 ms, 393k 219/127/88/68 ms.
Found on the way and fixed (stringl.c): quadm.fe + `set edge density 2` overflowed a
stack buffer, also in the stock program. Edges with a density in a soapfilm model with
a metric took their length from simplex_energy_metric()/simplex_force_metric(), which
use the model's simplices (facets) and read a third vertex. Now edge_metric_length():
metric at the midpoint (as edge_energy_l_metric()), with its exact gradient; forces
match finite differences (full and conformal metric, tests in test_numerics.py).
Unverified, not changed: metric.c edge_force_l_metric()'s conformal branch lacks the
1/4 on the metric-derivative term (fp = gg_partial*|v|^2, the non-conformal branch has
/4); it is installed only by quad_to_linear() and no sample reached it.

### 2026-10-07, before a session restart

Phase C2 is complete (see the table in Status). After it, at the user's request:

- Done: thread control from Python (`pyse.threads_limit(n)`, `threads=` on
  `relax()`/`hessian()`; commit 4af2a9d). Wheels build a pinned OpenBLAS 0.3.34
  (OpenMP build, DYNAMIC_ARCH; `tools/build_openblas.sh`) and ship
  `THIRD_PARTY_LICENSES.txt` (66420bd). Decisions: wheels for Linux x86-64 only;
  pySE's own license decided later (notices shipped now).
- Lesson: a USE_THREAD=0 OpenBLAS without USE_LOCKING=1 gives wrong Newton steps
  (MUMPS calls BLAS from several threads). The OpenMP build is both safe and 4-13%
  faster than serial+locking.
- Speed-up items the user asked for ("do 1 and 2"):
  1. Reuse MUMPS's analysis across Newton steps: **done (8701b7c)**. Measured
     (8 threads, best of 3-4 consecutive steps, two interleaved rounds, load ~7,
     against 66420bd): 393k 1.22 -> 1.16 s; 1.6M 6.45 -> 5.93 s; Lagrange 6 at 24k
     6.78 -> 6.49 s. About 4-8%, at the noise level; energies agree.
  2. Parallelize the remaining serial Newton-step setup: **done**. The per-vertex
     normals of `hessian_init` (`new_calc_vertex_normal`) run in parallel
     (`fl_vertex_normals`, fastloops.c) for vertices without constraints or
     boundaries; the rest stays serial. Newton step, 8 threads, against 8701b7c:
     393k 1.18 -> 0.88 s; 1.6M 5.9 -> 4.9 s; Lagrange 6 at 24k unchanged (~6.3 s,
     factorization-bound). Tests, stock sanitizers and an OpenMP sanitizer run
     on refined samples (cube, mound, catbody, column, sphere, twointor,
     phelanc) pass.
- Future speed options (noted 2026-10-07; the user chose phase D first; effort
  and gain are estimates, not measured):
  - Hash-free direct assembly: build the sparsity pattern once per topology and
    add entries straight into CSR slots instead of `sp_hash_search`; hash path
    kept as fallback. Effort 1-2 days (fasthess.c, linsys setup). Gain 5-10% of
    a linear Newton step (older noisy profile), little for Lagrange. Low risk.
  - Factorization reuse over several Newton steps (chord Newton), opt-in;
    refactor when convergence slows. Effort small-medium, mostly convergence
    testing. Gain up to ~2x on a `hessian` ladder (factorization dominates
    Lagrange 6 and 1.6M). Changes the algorithm (linear convergence, can fail
    far from equilibrium): user decision.
  - Phase E compact layouts / state struct: weeks, high risk; speed gain
    uncertain (perhaps 10-30% in memory-bound loops); main value is fresh state
    per load and multiple engines.
- Scratch tools (session scratchpad, lost on restart): newton_t.py, prof_newton2.py,
  solvers.c harness, matrix dumps (`PYSE_DUMP_HESSIAN=path` recreates them).

## Phase D: polish and usability

Decided 2026-10-08: all items, in the order 9, 8, 2, 1, 4, 5, 3, 7, 6 (docs last),
pausing after each. The Python API may break (pre-1.0, no deprecation shims).

1. Live view that only moves points (no PolyData rebuild) when topology is unchanged.
   **Done**: the actor was already kept; now an update with unchanged facets
   skips the PolyData rebuild and VTK comparison too: it compares the cached
   `ev.mesh()` connectivity arrays and samples points from a cached
   tessellation (`_Tessellation`: per lattice point, the vertex rows with
   nonzero weight). Off-screen update: 393k linear 0.235 -> 0.059 s (rest is
   `ev.mesh()`), Lagrange 3 at 24k 0.223 -> 0.031 s. One-off `tessellate()`
   unchanged (393k Lagrange 3: 1.5 s, 0.9 GB peak).
2. Size warnings for curved tessellation and export (Lagrange-3 at 1.6M facets,
   6x6 tessellation = 57M triangles).
   **Done** (user chose warn only, module setting): `pyse.tessellation_limit`
   (default 10M triangles, None: off) gives a `LargeTessellationWarning` with the
   count and an estimate of the memory (393k Lagrange 3, n=6: 14M triangles,
   estimated 1.2 GB, measured peak 0.94 GB, 1.4 s).
3. Generic wrappers in `bindings/module.cpp`; a more uniform `Mesh`.
   **Done** (user chose A+B): removed the unused `_core.edges/facets/element_nodes`
   bindings and their C wrappers (`mesh()` returns all of it); after that little
   repetition was left, so no generic helper. `Mesh.faces/face_ids/face_bodies`
   renamed to `facets/facet_ids/facet_bodies` (breaking). Inputs keep `faces=`
   (polygons, like the datafile's `faces` section). Not done: per-element groups,
   `Bodies.target_volume` -> `target`.
4. Notebook: `_repr_html_`; the user checks the live view in Jupyter.
   `_repr_html_` **done** (user's choice: summary table, no image): `Evolver`
   (datafile, model, counts, energy/area, threads/solver, bodies), `Mesh`,
   `Bodies`, `Parameters`. Live view in Jupyter: checked by the user, works
   (`[jupyter]` extra). The check notebook was minimal: the item 6 tutorial should
   show a realistic case (larger surface, Lagrange stage, scalars, export).
5. Optional wait-with-timeout instead of the immediate "busy" error.
   **Done** (user's choices): `pyse.busy_timeout` (default None: fail at once;
   seconds; inf). The C++ lock (`std::timed_mutex`) reads it only when the lock
   is taken, waits with the GIL released and checks Ctrl-C every 100 ms; a
   re-entrant call (same thread) always fails. New `EvolverBusyError`
   (RuntimeError subclass, defined in the bindings).
6. Docs: API reference and a tutorial notebook (load/build -> relax -> plot ->
   export for FEM).
   **Done** (user's choices: Sphinx + autodoc, all four topics, built in CI without
   publishing): `docs/conf.py`, `index.md`, `api.md`, `tutorial.ipynb` (double
   bubble of volumes 1 and 2 built with make_datafile: relax with refinement and
   Newton, Young-Laplace check via pressures and mean curvature, Lagrange 2-4
   vs a finer linear mesh, plots, watertight per-body STL and native .msh).
   Executed at build time (~8 s); `docs` extra; `.github/workflows/docs.yml`.
7. Optional stop-gap for global settings that leak between datafiles (the
   "clipped" display mode from the torus sample; the real fix is phase E).
   **Done**, measured first: every sample loaded after every other (25x25, plain
   and after a batch of toggles such as conj_grad, runge_kutta, gravity off,
   autorecalc off) gives the fresh-process energy, so `reset_web()` covers the
   physics. Only dumps differed: "clipped on" after 100grain.fe (torus display
   mode is sticky by design; now reset in pySE's load), and the view matrix of
   100grain/metric/slidestr after some samples (display only; left alone).
8. `pyse.map` fails fast with a clear error when workers die before taking a job
   (e.g. a script read from stdin under the `spawn` start method), instead of
   failing every job.
   **Done**: workers report ready; a death before that raises `WorkerStartError`
   (with the cause: stdin/`-c` scripts, missing `__main__` guard). README example
   now has the guard.
9. Add the OpenMP, refined-surface sanitizer run to `tools/run_sanitizers.sh`.
   **Done** (7 refined samples, Newton in linear and Lagrange 2; ~9 min locally).

## Phase E: state refactor (long term)

1. Feasibility probe: generate a state struct from the ~1,240 extern globals and
   ~280 file statics; count name clashes (`el_list` is one we already hit).
2. Globals into one struct behind macros (`#define web (evolver_state->web)`),
   fresh state on every load (fixes the settings leak); measure the indirection.
3. Optional multiple engines per process, after making Evolver's libc state
   per-engine (`strtok`, `rand`/`drand48`, time formatting, `chdir`).
4. Explicit state passing only in hot paths where profiling shows a gain.

Probe done 2026-10-08 (nm on the stock build's plain objects, current sources):
2051 globals (~210 KB), 293 file statics + 17 function statics (~970 KB, mostly
fasthess.c's per-thread qinfo array and the parser tables); 9 static names in more
than one file, 10 statics named like a global; 24 global names also used as struct
members (list, view, metric, filename, hashtable, line_no, ...), which a `#define
name (state->name)` scheme would break, so those globals would be renamed. Besides:
libc state (strtok, drand48, chdir), pySE's 108 statics (fastloops/fasthess caches,
the shared MUMPS instance). Feasible with a generated header and a few dozen renames.
Expected gain: no speed (one extra pointer load per global access, likely 0-3% cost);
the value is several independent surfaces per process. Today's alternative:
save/restore at 25k/98k/393k facets 0.21/0.51/1.93 s save, 0.15/0.43/2.08 s
restore; pyse.map for sweeps. Estimate: engines used one at a time 3-5 days,
concurrent engines 2-3 weeks; risk of silent state sharing through a missed static.
**User decision (2026-10-08): not now** (they don't alternate between large
surfaces in one session); if done later: engines one at a time, libc state
(random, cwd) per engine.

## Phase F: a friendlier API (design for review, 2026-10-10)

**Done 2026-10-10** (ce50fc4 F4/F3/F9, 8d004de F1, 336eeda F2, 855cdc4 F6,
ef6c100 F7, 5c953b8 + 01b5e50 F5, cf9243b F8, 06b0d50 the examples). Not pushed
at the time of writing. The suite also passes under numpy 2.5 (CI's), which
the local venv (1.26) can't show: run it in a fresh venv before pushing.

Goal: the examples and the typical workflows without Evolver command strings.
Today all four notebooks and the drainage helper use `ev.command(...)` for
relaxation recipes (`"g 10; u; V"`, `"g 5; hessian; hessian; hessian"`), element
selection (`foreach ... where ... do set ...`), body settings (`set body[1] target
...`) and diagnostics (`eigenprobe`). The Python API may break (pre-1.0, no shims).

Checked against the code: `relax()` exists (gradient steps until the energy
settles, optional `hessian=True` Newton steps) but has no mesh tidying, refinement
or rollback, so nobody uses it. `set_values("body", "target", v)` works already,
and it formats `float(v)!r`, which is numpy-safe. `ev.bodies()` is a snapshot
dataclass (23 uses in tests); `ev.parameters` is a live view. `Mesh` has no
constraint membership (it is built in C); `ev.values("vertex", "on_constraint 3")`
gives it.

F1. **One relaxation call.**
    `ev.relax(tol=1e-10, max_iter=1000, *, window=5, tidy=0, levels=0, newton=0,
    seek=False, undo_if=None, callback=None, every=1, threads=None)`.
    `tidy=k`: `u; V` every k gradient steps, with convergence measured round to
    round (the energy after a round against the one before). Found while
    building: a step-by-step test never settles (each `V` bumps the energy), and
    tidying only between converged relaxations let contact-line facets
    degenerate (slit droplet: smallest angle 2.4 degrees against 31.7);
    round-to-round with `tol=1e-7, window=1` reproduces the notebooks' old
    `"g 10; u; V"` loops exactly. `levels=n`: relax, refine, relax, ... n refinements. `newton=n`:
    up to n Newton steps after the last level (`seek=True`: `hessian_seek`), until
    one changes the energy by less than `tol`. `undo_if(ev) -> bool`: a snapshot
    before each Newton step, restored (and Newton stopped) when it returns True;
    the drainage needed this for steps that crossed a mirror. Replaces
    `hessian=`/`max_hessian=` (breaking; decided). Also `ev.newton(steps=1, *, seek=False,
    tol=None, undo_if=None) -> int` (steps taken); `ev.hessian()` stays as the
    single step. The result's history gains a per-iteration `level`.
F2. **Remeshing by edge length.** (built: `t` first (merging lengthens edges), then `refine edge where length > max and not no_refine`: Evolver's `l` ignores `no_refine`, so the drainage helper's protection never worked; requires max_edge >= 2 min_edge)
    `ev.remesh(target=None, *, max_edge=None, min_edge=None, equiangulate=True,
    average=False, protect=None) -> dict` (edges split, deleted). `target=h` means
    `max_edge=1.6h, min_edge=0.5h`, the ratios that fixed the drainage band. Runs
    `l`, `t`, `u` (and `V`). `protect`: an edge mask not to split; sets
    `no_refine` for the `l` and restores each edge's previous flag after. Raises
    for Lagrange/quadratic models (Evolver's `t` doesn't support them).
F3. **Selections as masks.** `ev.on_constraint(k, element="vertex") -> bool array`
    aligned with `mesh()` rows; `Mesh.edges_touching(vertex_mask, how="any"|"all"|
    "one")` (`"one"`: exactly one end, the contact-line spokes of F2);
    `ev.set_flag(element, flag, where=None, on=True)` for boolean attributes
    (`fixed`, `no_refine`, ...; `fix`/`unfix` become one-liners on it). The API docs
    explain the two Evolver quirks the masks avoid (a bare element type in an
    aggregate means all elements; `vertex[2]` in a `where` is global vertex 2).
F4. **Bodies.** A live handle `ev.body(i)` with `.volume`, `.target` (settable),
    `.volconst` (settable), `.pressure`, `.fixed`; `ev.bodies()` stays the snapshot
    for arrays and plots. One internal number formatter (`repr(float(x))`) for every
    command the API builds. Also the pending rename `Bodies.target_volume` ->
    `target`. **Decided:** the live handle `ev.body(i)`; `ev.bodies()` stays.
F5. **Constraint builders with contact angles.** A `pyse.constraints` module:
    `plane(normal, offset, contact_angle=None)`, `mirror(axis, at=0)` (no
    wetting energy), `sphere(center, radius, contact_angle=None, axis="z")`.
    Each renders the constraint text with the energy and content integrals and
    tells `make_datafile` the body's `volconst` correction; `make_datafile(constraints=)`
    takes them besides strings. Risks found while doing it by hand for the slit
    bead: signs depend on the facet orientation relative to the body and on which
    side the liquid is (an explicit `liquid=` side argument), and the sphere's
    dphi line integrals need the axis through the centre to stay in the wetted
    region (document, and check at load). Acceptance: exact cases to 1e-6 relative
    on refined meshes: sessile drops (spherical caps) on planes at several angles,
    the immersed slit meniscus, the pendular ring, and the slit bead (against the
    hand-written integrals in `docs/drainage_helpers.py`, verified against exact
    sheets and rings). **Decided:** planes, mirrors, spheres and cylinders
    (`cylinder(axis_point, direction, radius, contact_angle=None)`, exact checks:
    a liquid ring on a fibre, a drop between two parallel fibres if feasible).
    **Built** (`pyse.constraints`), with what testing taught: the integrands must
    vanish wherever the wetted region's boundary runs along a mirror, so each
    builder has a gauge: planes `t ds` in a frame set by `ref`/`origin` (an
    arbitrary frame made the drainage wall's x = ell/2 mirror count), spheres
    `h dphi` about z (mirrors through the axis and the equator plane; wet poles
    give `volconst`/`energy_constant` for a `span`), cylinders `l dphi`
    ("azimuthal") or `phi dl` ("axial", for fibres cut by mirrors across them).
    Faces no contact line touches (a floor under the film) and tilted mirror
    faces can't be closed by line integrals: documented (keep mirrors vertical or
    at z = 0; add the floor's z dx dy to `volconst`). Tests: plane caps exact to
    1e-12 (table, ceiling, tilted), the immersed slit meniscus exact (walls,
    mirrors, a fully wetted bead), the sphere zone converging at second order
    and identical to the drainage helper's hand-written bead integrals, fibres
    (horizontal exact, also moved; vertical converging).
F6. **Diagnostics as values.** `ev.eigen_counts(shift=0.0) -> (negative, zero,
    positive)` from `eigenprobe` (costs one factorization); `ev.check() -> list of
    problems` (Evolver's `check`; empty when the topology is sound);
    `ev.mesh_quality()` -> edge lengths (min, median, max), smallest facet angles,
    skinny-facet count (< 15 degrees), zero-area facets. "Vertices inside solids"
    stays the user's (constraints are formulas); the docs show it with F3 masks.
F7. **Mirrored display.** `Mesh.mirrored(planes)` and `plot(..., mirror=...)` /
    `live_view(..., mirror=...)`: each plane is `"x"` (x = 0), `("x", c)`, or
    `(normal, point)`; reflections apply in order, each doubling what is there (the
    slit's eighth cell: `["z", "y", "x", ("x", ell/2)]`).
F8. **Continuation.** `pyse.continuation(ev, set_value, values, *, relax=None,
    checkpoint=None, resume=None)`: a generator; per value it calls `set_value(ev,
    v)` (helpers for a body's target and for a parameter), relaxes (`relax`: kwargs
    for F1 or a callable), checkpoints (dump + state, written atomically), and
    yields a record (value, energy, pressures, `ev`). Events stay in user code: the
    caller can rebuild and `send()` a new `Evolver` (the drainage rebuilt at the
    bead emergence). **Decided:** in a `pyse.recipes` module.
F9. **Better command errors.** `EvolverError` carries the command text
    (`.command`) and adds hints for common mistakes: `np.`/`np.float64(` in the
    text (a numpy value formatted with repr: use `float()`), `nan`/`inf`.

Order: F9, F4, F3, F1, F2, F6, F7, F5, F8; then convert the four notebooks and the
drainage helper to the new API (they are the integration test: same numbers as
now), update the tutorial and API docs. Each item: tests first where the behaviour
is exact, full suite, quick sanitizers for C changes (none planned), one commit.

## Phase G: robust by default (planned 2026-10-10)

The user: "The solvers should be robust, not needing tailored fixes for every
case" (the drainage example needed remeshing calls, a tidy cadence, Newton with
rollback, tuned tolerances and per-layout gauges). And: "Probably needs more
tough examples to challenge the approach". So first a stress suite, run with
default settings only, as the baseline; then robustness inside the solvers
(residual-based convergence, safeguarded Newton, quality-aware relaxation, a
health report, builders that take the mirrors), judged on the whole suite.
After success, the cases are condensed into notebook examples.

Stress suite (`bench/stress/`, run on demand, not in CI); each case runs with
default `relax()` (and `recipes.continuation`), no per-case knobs, and reports
pass/fail against a reference, steps, wall time, mesh quality and stability:

1. Liquid cylinder between plates at 90 degrees, volume lowered: unstable when
   L > pi r (sliding contact lines). Must be reported, not silently converged.
2. Catenoid between rings, separation raised to the fold (h/R = 1.3255): exact
   area on the stable branch; no false "converged" past the fold.
3. Sessile drop, contact angle 10 to 170 degrees (a parameter): exact caps.
5. Bridge between spheres, gap down to 1e-3: axisymmetric Young-Laplace reference.
6. Barrel drop on a fibre: axisymmetric reference; the clam-shell transition.
8. The drainage band (ell = 1.1): the stored level-3 curve and break.
9. Cube inflated 100x: E = (36 pi)^(1/3) V^(2/3) throughout.
11. Gravity puddle (large Bond number): axisymmetric reference with gravity.
12. Sessile cap shrinking to 1e-4 of its volume: exact caps all the way.

(4 edge pinning, 7 foam, 10 Lagrange-3 at 100k+: skipped, the user's choice.)
A shared axisymmetric Young-Laplace solver (shooting, with gravity) gives the
references for 5, 6 and 11.

**Status 2026-10-10.** Suite built (nine cases; `bench/stress/run.py`). Baseline
with the old defaults: 1 of 9. In the package now: `ev.residual()` (normal part
of the projected vertex forces, dimensionless; 1e-14 at a Newton solution, any
scale), and default `relax()` = rounds of 10 gradient steps with `u; V`, then up
to 20 safeguarded Newton steps (plain first, line search if rejected; reject only
non-finite, or energy and residual both up: with a volume constraint, restoring
the volume can cost energy), converged on the residual; NaN restores and raises.
Result: 5 of 9 (2 catenoid 0.03%, 6 barrel, 9, 11 puddle, 12 shrinking cap).

Tried and left opt-in (each fixed some cases and broke others, so not robust):
* conjugate gradients (`cg=True`): the 10-degree drop converges (residual 7e-2
  -> 2e-11), but the catenoid and the bridge diverge (finite blow-ups the guards
  don't catch).
* graded automatic remeshing (`remesh=True`): helps the high-angle drops, breaks
  the barrel (rim on a thin fibre gets refined, then crushed) and, with other
  thresholds, the band and the puddle; results swing with two thresholds.
Learned: Evolver's `l` ignores `no_refine`; spurious negative eigenvalues come
from boundary-wire vertices (even spacing is an energy maximum for sliding) and
from soft contact-line modes of flat drops; contact radii of drops must be
measured about the rim's centre (drops slide).

Open (failing): 1 (no stability signal yet), 3 (150-170 and 10 degrees), 5
(the bridge's start at gap 0.2; reference unconfirmed), 8 (the band: mesh at the
moving contact line). Next ideas: a stability check that leaves out boundary
sliding modes; a divergence guard for the gradient phase (energy blow-up per
round -> restore and fall back), which would make cg safe; remeshing driven by
the residual's distribution or by contact-angle resolution rather than by edge
lengths.

## Phase G2: robustness in the core (replanned 2026-10-10)

The user: changes in the core are permitted; robustness is key. Phase G showed
that thresholds layered on Evolver's operations (V, l, t, K, conj_grad) fix one
case and break another. The failures trace to four root causes, each better
fixed in the engine than worked around in Python:

1. **Tangential drift.** Gradient steps move vertices along the surface too;
   vertices pile up (wire rings, contact lines, thin fibres) and facets collapse.
   `V` fights it by moving vertices to neighbour averages, which also moves them
   off the surface: it bumps the energy every round and keeps convergence from
   settling.
2. **Spurious Hessian modes.** Vertices on a boundary curve (two constraints in
   3D) keep a tangential degree of freedom in `hessian_normal`: sliding along
   the wire, a mesh mode with negative curvature. It makes Newton indefinite
   and eigenvalue counts meaningless (the catenoid: ~120 "negative"
   eigenvalues = its wire vertices).
3. **Resolution that doesn't follow the geometry.** Fixed meshes go too coarse
   where features shrink (a small contact disk, a thin film over a bead, the
   neck of a bridge) and stay needlessly fine elsewhere.
4. **Unguarded descent.** Conjugate gradients can blow up to finite but huge
   energies; no guard notices.

Steps, each measured on the whole stress suite (all nine cases, defaults only),
the test suite, sanitizers (C changes), and the benchmark (no slowdown beyond
noise on the user's workload sizes):

(Done 2026-10-10, G2.1/G2.2: see the notes under each.)
G2.1 **Tangential relaxation in C** -- *not needed*: Evolver's `V` (VOLKEEP, its
     default) already removes the normal part of the averaging motion and, for
     constrained vertices, averages only with neighbours on the same constraints
     (even spacing along wires and contact lines); relax() runs it every 10 steps.
     Its energy bump is second order (1.2e-5 relative on the refined cube);
     projecting back onto the old facets is worse (-8.5e-4: the flat facets lie
     inside the curved surface), and exact shape preservation would need a smooth
     reconstruction, not worth it at that size. Original idea: (a new command, used by relax() instead of
     `V`): move each vertex toward the area-weighted centroid of its
     neighbours, but only within its tangent plane (interior vertices), along
     its boundary curve (wire vertices: even spacing) or within the constraint
     and along the contact line (contact-line vertices: even spacing along the
     line). Shape-neutral to first order, so no energy bumps. Target: the
     catenoid, barrel and drainage band keep sound facets with plain gradients,
     and rounds settle (the gradient phase stops hitting max_iter).
G2.2 *Done*: hessian_normal's slant test is now measured against the raw surface
     normal (it used the normal already projected onto the constraints, so on a
     wire it read ~1, or not, depending on which code path had stored it) and by
     magnitude (the freedom's sign is arbitrary); default hessian_slant_cutoff
     0.05. The stable catenoid: 0 negative eigenvalues (was ~1 per wire vertex,
     erratic); 1 just past the fold. residual() (now in C) measures exactly the
     freedoms Newton controls, so relax() converges with wires present.
     Original plan: **Hessian degrees of freedom on boundary curves** (hessian.c): a vertex
     whose free directions are only tangent to its boundary curve gets no
     Hessian freedom (it is placed by G2.1, not by Newton). Contact-line vertices
     keep their one physical direction (normal projected into the wall).
     Target: the catenoid shows 0 negative eigenvalues while stable; the
     cylinder's count turns negative at r = 1/pi.
G2.3 *Done*: `ev.stability()` counts eigenvalues below -1% of the scale of the
     lowest positive ones (from `ritz`): symmetry zero modes (a drop sliding on a
     plane, a barrel along its fibre: +-1e-5..1e-4) don't read as instabilities,
     real ones (the barrel's roll-up pair, -1e-3) do. relax() checks once
     converged (`stable`, `negative_modes`, UnstableEquilibriumWarning). Suite:
     6 of 9 (the cylinder: reported at r = 0.3125, 1.8% past 1/pi). Plan was:
     **Stability in every relax() result**: an eigenvalue count after
     convergence (one factorization; skippable), `stable` in the result and a
     warning when not. Target: case 1 passes; case 2 reports the fold.
**Revised 2026-10-10 (after G2.2, G2.3 and two failed remeshing attempts).**
Both attempts failed on the remeshing details, not the idea: a size field that
chased its own mesh, and edge collapses that lost volume (20% on a cap). Most of
the remaining failures may need refinement only, and collapses caused all the
damage, so the order is now: find out what each failure needs, fix the cheap
things, then refine only, and collapse only if mesh growth calls for it. Each
step judged as before (stress suite with defaults, tests, sanitizers for C).

G2.4 **Resolution check** (about 15 minutes, first). Rerun the failing cases
     (3, 5, 8) with defaults and one uniform refinement more. If they pass,
     resolution is the cause and adaptive refinement only has to be cheaper; if
     not, the cause is elsewhere and gets its own step. Case 5's reference is
     confirmed (a converged fine run matches it to 0.5%).
     *Done* (2026-10-10). One level more, defaults otherwise: case 3 passes at
     150 and 160 degrees (worst energy 0.23%), still fails at 170 (implied angle
     2.05 degrees off: a small contact circle, local refinement) and 10 (relax()
     unconverged: G2.6). Case 5 still fails at gap 0.2 (all other gaps improve
     to 0.08%): not resolution. Case 8 at level 3: pressure 2.4% (was 168%),
     snap exact, 93 steps unconverged, 32 min. Cause of case 5: on the coarse
     16 x 4 start the contact-line vertices slide together in pairs and the
     discrete energy keeps falling along that drift (to -13.8, the volume lost);
     plain `g` does it too, `u; V` only shifts when. The gradient phase never
     ended (each `u; V` moves the energy ~3e-4, `energy_tol` was 1e-9), so it ran
     1000 steps and the drift won; Newton (normal motion only) can't drift so.
G2.4b *Done*: `energy_tol` default 1e-5 when Newton follows (1e-9 with
     newton=0). A warning when relax() ends unconverged with a grown residual was
     tried and dropped: the drainage helper's unconverged steps grow it up to 13x
     with correct results, the liquid bridge's gradient-only rounds 1.02x; no
     threshold tells the broken bridge (3.7x) apart. Detection belongs to G2.9.
     Suite with 1e-5: 7 of 9 (case 5 passes, 0.49%
     worst), far fewer gradient steps (cylinder 26000 -> 1060, cap 25880 -> 3040,
     catenoid 18460 -> 5290); case 8 at level 2: 8.3% (was 168%), snap 4% off,
     120 unconverged, smallest facet angle 0; case 3 unchanged.
     *Done* (G2.5 as built, 2026-10-10): what broke the liquid bridge on defaults
     was not Newton but the coarse start: on the 8-around cylinder the contact
     line collapses within one 10-step round (residual 0.03 -> 3-7, smallest
     angle -> 0), and refinement only spreads the damage. With Newton following,
     relax() now measures the residual at the end of each round (before `u; V`,
     which lifts it to its own floor) and, if it more than doubled from its
     lowest, puts the round's vertices back (`ev.vertices`; a dump would cost 2x
     a round at 393k facets) and hands over. Bridge grid (gaps 0-0.6 x theta 20,
     60, 100 and theta 20-80 at gap 0.2): 25 of 25 converge, energies within
     0.007% and forces within 0.1% of the notebook's tailored recipe (was 18 of
     25 on defaults); the notebook now uses `relax(levels=3)`. Tried and dropped:
     also ending the gradient phase when the residual set no new low for 3
     rounds (case 8: median error 2.3% vs 0.3%; where Newton can't finish, the
     gradient steps do the work). Not needed so far: alternating Newton and
     tidied rounds. Suite 7 of 9 unchanged (case 8 8%, case 3 as before).
G2.6 *Done* (2026-10-10), differently from the plan. Case 3 at 10 degrees:
     plain gradients crawl (residual 0.14 after 650 steps), Newton can't take
     over; with cg it converges in 70 steps + 4 Newton. Conjugate gradients on
     every relax: cases 2 and 5 no longer diverge (the new tolerance and the
     round undo), but the bridge grid loses 3 of 25 (theta 20 at gaps 0.4-0.6:
     folded contact lines, angles +-4 degrees) and case 8 doubles its states
     over 1% (36-39 vs 17-22): cg wears the mesh where contact lines travel, no
     energy jump for a guard to catch. So instead: `cg=None` (default) adds a
     cg + Newton pass when the first pass ends unconverged, kept only if it
     converges (kept when the residual merely fell, case 8 got worse again:
     34 states over 1%, median 0.5%). Suite: case 3 now fails only at 150-170
     (resolution, G2.7); bridge grid unchanged (the pass never runs); case 8
     back to baseline (20 states over 1%, median 0.35%) but ~60% slower, the
     pass tried and reverted on its ~125 unconverged steps.
G2.7 *Done in part* (2026-10-10; user's choices: `ev.adapt()` plus opt-in
     `relax(adapt=...)`, 15 degrees, case 8 parked). What case 3 at 170 needs is
     not in the plan's criteria: its first ring of edges out from the contact
     line is longer (0.35) than the contact radius (0.24), and the turning runs
     along those edges, so facet angles there are small (3 degrees) and a 32-gon
     turns 11.25 degrees per vertex at any size. Criterion built instead: the
     angle between the vertex normals at an edge's ends (sign-aligned per
     vertex; edges at triple lines not judged). 170 degree drop, per
     continuation step: none 7.6 degrees off (480 facets), 15 degrees 3.5 (734),
     10 degrees 1.6 (2022), uniform level 3.9 (1920). Static shapes: one pass,
     then nothing left to split (cube 384 -> 1248, drop 480 -> 1312, hemisphere
     480 -> 544). Suite with adapt on everywhere: all pass as before, more
     accurate (bridge 0.09%, inflated cube 0.02%), case 3 fails only at 160-170
     (2.97 worst); but refinement accumulates over continuations: inflated cube
     92 s (2 s), catenoid past the fold 129 s (3-8 s), case 8 not done in 25 min.
     So opt-in until coarsening (G2.8) exists. Not built: boundary-turning and
     wall-gap criteria.
G2.8 *Done* (2026-10-10; user's choices: goal adapt safe by default, the guard
     in C inside Evolver's edge loop). trirevis.c collapse_keeps_shape(), called
     in delete_edge() after the link condition: refuses a merge that tilts any
     facet around either end by more than `collapse_max_tilt` degrees (or flips
     it), or makes an edge from the merged vertex longer than
     `collapse_max_edge` (new internal variables, 0 = off by default; linear
     soapfilm 3D, no torus/symmetry). adapt() merges interior edges shorter than
     0.75 of the bulk (75th percentile) with turning < max_turn/3, guard at
     max_turn/2 and 4/3 bulk. Static shapes settle in 2-3 passes (a merge pass
     moves the volume ~1e-4 before the constraint restores it). Suite with adapt
     everywhere: all pass as before; catenoid 46 s (129 refine-only, 3-8 off),
     inflated cube 66 s (92; 2 off), bridge 66 s (33 off), cylinder 20 s (3.5).
     Not a pile-up any more: turning per edge is scale-invariant, so 15 degrees
     simply asks for more facets than the suite's coarse meshes (and gains
     accuracy: cube 0.02% vs 0.34%). The target "adapt on at today's times" is
     not met; adapt stays opt-in. Case 3 still fails at 160-170 with it.
G2.9 *Done* (2026-10-10; user's choices: everything incl. gaps, fields only
     and no warnings, cheap parts on every relax). `ev.health()` /
     `result.health` (`pyse.Health`): residual, stable/negative_modes (from
     relax), angle_min, skinny (< 5 deg), wall_gap (vertices beyond two edges of
     a constraint's own vertices, in median edge lengths) and wall, crossed
     (one-sided: the forbidden side; equality: at most 5% against the rest),
     self_gap (pairs more than two edges apart, searched within h/2 by a cell
     grid), issues in words. C binding `constraint_gaps` (f/|grad f| per vertex
     and constraint in use, NaN on it). 0.18 s at 98k facets (residual 0.08 of
     it). Suite: no issue on any converged and stable relax except one in case 8
     at the snap (a vertex past the y = 0 mirror, plausibly real); unconverged
     steps of cases 1, 2 and 8 all flagged (degenerate facets, crossings,
     near-contacts).
G2.5 **Tidying around Newton** (was G2.3b; small). relax()'s Newton phase moves
     a contact line far without tidying, and its facets degenerate (the liquid
     bridge: smallest angle 3e-5 degrees). Alternate tidied gradient rounds and
     a few Newton steps until both the residual and the mesh are sound (a quality
     check between them). Target: the liquid-bridge example on plain defaults;
     maybe case 5.
G2.6 **Guarded descent** (a few hours). Per gradient round, an energy jump beyond
     what volume corrections explain is undone, falling back from conjugate to
     plain gradients. Then conjugate gradients by default. Target: case 3 at 10
     degrees (the flat drop converges only with them), no regression in 2 and 5
     (where they diverged).
G2.7 **Refine-only adaptivity** (if G2.4 says resolution). Split edges where the
     geometry needs it, never collapse: a boundary curve turning more than
     ~11 degrees per vertex (at least ~32 edges around a small contact circle),
     facets meeting at more than ~20 degrees, and (as a C helper) thin gaps to
     nearby walls. At most one level per pass, from a size field smoothed over
     the one-ring so it can't chase mesh noise; relaxation in between. Validate
     on static shapes first (a pass keeps the volume to 1e-4, a second pass after
     relaxation changes nothing, the facet count stays bounded), then on the
     suite. Target: cases 3 and 8.
G2.8 **Guarded collapsing** (only if mesh growth becomes a problem). A collapse is
     rejected if it would flip a facet, tilt neighbouring normals by more than
     ~15 degrees, change the local enclosed volume beyond a tolerance, or make
     edges longer than 4/3 of the target; in C, inside the edge loop.
G2.9 **Health report** (was G2.6): residual, stability, facet quality,
     near-contacts with walls, mirrors and the surface itself, in every relax()
     result.
G2.10 **Examples on defaults** (was G2.7): the liquid bridge and the drainage
     helper without their crutches (remesh() each step, undo_if, stability=False);
     then the stress cases condensed into notebook examples.

The earlier G2.4 design (full isotropic remeshing in C toward a geometric size
field) stays the long-term target that G2.7 and G2.8 build toward.

## Tools and conventions

- Tests: `pytest`; with `PYSE_CHECK_FACET_CACHE=1` for cache verification.
- Sanitizers: `tools/run_sanitizers.sh [build-dir]` (stock program, ASan+UBSan,
  all samples; then an OpenMP build on refined samples with Newton steps).
- Benchmark: `python bench/benchmark.py --levels 6 8 --threads 1 4` (linear at 98k
  and 1.6M facets; Lagrange 2/4/6 at 6k and 24k; `--json` to compare runs).
- Profiling build: `pip install . -C build-dir=<dir> -C cmake.define.PYSE_NOSTRIP=ON
  -C install.strip=false -C cmake.define.CMAKE_C_FLAGS="-g -fno-omit-frame-pointer"`;
  then `perf` (above) or `valgrind --tool=callgrind --toggle-collect=iterate`
  (LTO may inline/rename functions; `--threshold=100` shows small entries).
- Memcheck: same build, `PYTHONMALLOC=malloc valgrind --suppressions=tools/valgrind.supp
  python <script>` (the file explains its one entry, an -O3 false positive in MUMPS).
- Reference build for A/B timing: `git worktree add <dir> <commit>` + separate venv.
- Evolver source files use CRLF line endings: edit them with a script that keeps
  CRLF (never rewrite them with LF). Our own files (`fastloops.c/h`, bindings,
  Python) are LF.
- Don't `pkill -f` with a pattern that also appears in the same shell command;
  kill stray processes by PID after checking what they are.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
