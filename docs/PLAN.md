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
  it. `save()`/`restore()` for exact snapshots; `pse.map` (worker processes,
  crash-isolated) for sweeps. Long term: move Evolver's globals into one state
  struct (phase E).
- **Threads:** OpenMP; default = physical cores; `pse.set_threads(n)`,
  `OMP_NUM_THREADS`, `PYSE_THREADS`; `pse.map` workers 1 thread each. Evolver's
  old pthread mode (`-p`, `thread_launch`, `THREADS` blocks) stays in the source
  but is compiled out and unsupported (user's choice: keep `src/` close to
  upstream); per-thread data (`GET_THREAD_DATA`) is per OpenMP thread. **Don't expect
  scaling above about 4 threads** on the development machine (4 fast + 4 compact
  cores, 2-way SMT, laptop memory bandwidth): 8 threads gain 0-10% over 4 or lose,
  and the facet loops are memory-bound. The default thread count is decided in C2
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
| B | `bench/benchmark.py`; direct C element writes; `mesh()`; vectorized tessellation and watertight checks; `pse.map` | |
| C | facet corner cache; parallel facet volume/energy/force loops (`src/fastloops.c`); `pse.set_threads` | 1.6M facets: 11.1 -> 2.3 s per iteration |
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
  `pse.map` sweep once ran 4 h on 3-4 cores and skewed a whole session.
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
   override; also given to OpenMP so MUMPS follows); `pse.map` workers 1 thread
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
MUMPS ordering re-checked now that the analysis is reused (2026-10-08, 8 threads,
analysis + factor): 393k AMD 0.06+0.157 s, AMF 0.05+0.159, PORD 0.45+0.124;
Lagrange 6 24k AMD 0.23+0.546, AMF 0.24+0.613, PORD 0.39+0.535; 1.6M AMD
0.43+1.31, AMF 0.47+1.07, PORD 2.07+0.86 (PORD = MUMPS's automatic choice; QAMD
worse). PORD's serial analysis needs ~4 steps per pattern to pay off at 1.6M
(~12 at 393k); typical ladders do 2-4. AMF: -18% factor at 1.6M linear only,
+12% for Lagrange 6. **No change** (AMD stays); possible later: PORD after the
2nd step on one pattern, for long Newton runs on big fixed meshes.

### 2026-10-07, before a session restart

Phase C2 is complete (see the table in Status). After it, at the user's request:

- Done: thread control from Python (`pse.threads_limit(n)`, `threads=` on
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
   **Done** (user chose warn only, module setting): `pse.tessellation_limit`
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
   **Done** (user's choices): `pse.busy_timeout` (default None: fail at once;
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
8. `pse.map` fails fast with a clear error when workers die before taking a job
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
- Reference build for A/B timing: `git worktree add <dir> <commit>` + separate venv.
- Evolver source files use CRLF line endings: edit them with a script that keeps
  CRLF (never rewrite them with LF). Our own files (`fastloops.c/h`, bindings,
  Python) are LF.
- Don't `pkill -f` with a pattern that also appears in the same shell command;
  kill stray processes by PID after checking what they are.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
