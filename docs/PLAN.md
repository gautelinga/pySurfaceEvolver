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
  from serial Evolver at round-off level; they must be reproducible run to run for
  a given thread count. **Bit-identity is not a goal**: choose the fastest design
  and verify with tolerances (below). Identical results are only a free check when
  a change keeps the summation order anyway.
- **Engine model:** one engine per process; every `Evolver` object is a handle to
  it. `save()`/`restore()` for exact snapshots; `pse.map` (worker processes,
  crash-isolated) for sweeps. Long term: move Evolver's globals into one state
  struct (phase E).
- **Threads:** OpenMP, on by default; `pse.set_threads(n)`, `OMP_NUM_THREADS`,
  `PYSE_THREADS` (set by `pse.map` workers to CPUs/processes). **Don't expect
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
   3. Decision point: if factorization still dominates, bring measurements and a
      recommendation on a modern sparse LDL^T (must still report the Hessian
      index/inertia). Expect parallel speedups to flatten at ~4 threads here.
      **Pause here and report.**
4. **Thread defaults** (small, partly by the user on a quiet machine):
   `bench/benchmark.py --threads 1 2 4 8`; set the default thread count (likely
   capped at physical cores, possibly 4 on hybrid laptops), `FL_PARALLEL_MIN`, and
   the `pse.map` split (more processes with fewer threads each may beat threads).
5. **Parallel named-quantity loops** (large; demoted: Lagrange runs take few
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

## Phase D: polish and usability

1. Live view that only moves points (no PolyData rebuild) when topology is unchanged.
2. Size warnings for curved tessellation and export (Lagrange-3 at 1.6M facets,
   6x6 tessellation = 57M triangles).
3. Generic wrappers in `bindings/module.cpp`; a more uniform `Mesh`.
4. Notebook: `_repr_html_`; the user checks the live view in Jupyter.
5. Optional wait-with-timeout instead of the immediate "busy" error.
6. Docs: API reference and a tutorial notebook (load/build -> relax -> plot ->
   export for FEM).
7. Optional stop-gap for global settings that leak between datafiles (the
   "clipped" display mode from the torus sample; the real fix is phase E).
8. `pse.map` fails fast with a clear error when workers die before taking a job
   (e.g. a script read from stdin under the `spawn` start method), instead of
   failing every job.
9. Add the OpenMP, refined-surface sanitizer run to `tools/run_sanitizers.sh`.

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
  all samples). OpenMP variant: build the stock program with `-fopenmp` added to
  the sanitizer flags and run refined samples with `OMP_NUM_THREADS=8`.
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
