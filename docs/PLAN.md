# pySurfaceEvolver development plan

Living document: decisions, status, lessons, and the remaining work. Update it
when a step is finished or a decision changes.

## Decisions (agreed with the user)

- **Bindings:** nanobind extension around a headless Evolver; C glue in
  `bindings/pyse_api.c` contains Evolver's setjmp/longjmp error handling.
- **`src/` is our fork** of Surface Evolver 2.70a. The first git commit is the
  untouched original. Bug fixes apply to every build (not behind `#ifdef PYSE`);
  only library hooks are guarded. Keep fixes in small, separate commits.
- **Numerics:** speed first, as long as the physics is correct. Results may differ
  from serial Evolver at round-off level (tolerance suffices); they should be
  reproducible run to run for a given thread count.
- **Engine model:** one engine per process; every `Evolver` object is a handle to
  it. `save()`/`restore()` for exact snapshots; `pse.map` (worker processes,
  crash-isolated) for sweeps. Long term: move Evolver's globals into one state
  struct (phase E).
- **OpenMP** on by default with all cores; `pse.set_threads(n)`, `OMP_NUM_THREADS`,
  `PYSE_THREADS` (set by `pse.map` workers to CPUs/processes).
- **User's workload:** linear soapfilm *and* quadratic/Lagrange (as a final
  high-precision stage: `lagrange n; g 5; hessian` ladders), Newton steps at
  100k+ facets, surfaces of 100k-1M facets, parameter sweeps, FEM meshing.
- **Process:** ask design questions before implementing feature-sized work;
  pause after each phase and report measurements; one commit per item.

## Status

Done (see `git log`):

- **Phase A**: LTO (-23% instructions/iteration), memory/UB fixes (path_open,
  kb_error strncat, delete_facet overflow, string-model facet-area overflow,
  MAXINT, sdrv null offsets, body-x crash), C23 fixes, sanitizer CI
  (`tools/run_sanitizers.sh`), exact 17-digit dumps, `save`/`restore`, handle-based
  engine, `relax()`, bundled samples (`pysurfaceevolver.examples`, EVOLVERPATH).
- **Phase B**: `bench/benchmark.py`; direct C element writes; one-call `mesh()`;
  vectorized tessellation numbering and watertight checks; live view in-place path;
  `pse.map`.
- **Phase C (round 1)**: facet corner cache; two-pass facet volume/energy/force
  loops in `src/fastloops.c` (parallel compute, ordered serial accumulation,
  bit-identical); `pse.set_threads`; cache-backed `mesh()`. One iteration at 1.6M
  facets: 11.1 s -> 2.3 s; 98k: 0.50 -> 0.17 s.

- **Phase C2 step 1**: per-thread accumulation (compensated sums for volumes and
  energies, per-thread force arrays, merged in thread order). Single evaluations
  agree with the original loops to ~2e-15; equilibria to <1e-9 with the same
  Hessian index; reproducible per thread count. In-loop time at 1.6M facets,
  8 threads (whole run): energy 0.64 -> 0.50 s, force 0.49 -> 0.27 s, volume
  unchanged (~1.8 s, memory-bound, ~11 calls per iteration). A vertex-gather
  force variant (CSR, bit-identical) was measured and was slower (0.32 s).
  **Iteration wall time unchanged** (~1.8 s at 8 threads): see lesson 9.

- **Phase C2 step 5 (done)**: parallel `film_grad_l` (`fl_film_grad`, per-vertex
  gather from the corner cache); compact per-facet body table for the volume loop;
  a cached vertex list for parallel per-vertex work (zero forces, move, save/restore
  coordinates, volume restoration, DV^T DV with per-thread matrices, dense and
  sparse); `FL_FOR_SELECTED` (parallel scan, serial in-order body) for constraint
  and boundary vertices and edge energy/force/content integrals. Fixed a signed
  overflow in the sparse Hessian hash (`sp_hash`). One iteration at 1.6M facets vs
  v0.5.0 (quiet machine, best of N): 1.97 -> 1.30 s at 1 thread, 1.45 -> 0.65 s at
  4, 1.43 -> 0.74 s at 8; 98k facets: 62 -> 32 ms (1 thread). The main thread is
  now about as busy as the workers; what is left serial is ~1-2% items.

Version 0.5.0. 231 tests; mypy clean; manylinux wheel builds and passes.

## Lessons

1. Profile before optimizing, set a target, drop the item if the profile doesn't
   support it (one-call `mesh()` and the live-view fast path gave nothing; the cost
   was elsewhere).
2. This machine is heavily loaded (load 11-17 on 16 cores): use callgrind
   instruction counts and interleaved best-of-N timings against a reference build.
   But instruction counts miss memory latency (the corner cache cut instructions
   6% and wall time 38%), so confirm with wall clock too.
3. After the linear speedups, recalculation is cheap; serial ordered accumulation
   is now the limit for threads (~35% of an iteration in parallel regions,
   Amdahl ceiling ~1.4x at 8 threads).
4. Compiler warnings (LTO) and sanitizers find real bugs; run both on every change
   to `src/`.
5. Global Evolver settings leak between datafiles (the "clipped" display mode from
   the torus sample causes the remaining test warnings) -> phase E.
6. Curved-element output can explode (Lagrange-3 at 1.6M facets, 6x6
   tessellation = 57M triangles): needs size warnings (phase D).
7. Quadratic/Lagrange are used as a short final stage, but at Lagrange 4 one
   gradient step costs ~13 linear ones and time splits about evenly between the
   `g 5` steps and the Newton step; Newton's share grows with size.
8. Newton profile (24k facets): factorization (`mindeg.c` minimum degree) ~53%,
   Hessian assembly ~30%. A Newton step costs 5 linear iterations at 98k facets,
   11 at 393k.

9. Profile of a 1.6M-facet iteration at 8 threads (after C2 step 1): the main
   thread spends ~78% of the wall time in serial Evolver code. `film_grad_l` and
   the `get_edge_side` walks it makes via `get_fe_side` are ~40% of wall time;
   `local_calc_content`, `get_bv_new_vgrad`, `volume_restore`, `calc_leftside`
   ~17%. The three parallel loops are ~22%. So C2 step 5 is where linear
   iterations gain now; parallel loops alone have hit Amdahl's limit.
11. This machine is a hybrid laptop CPU (Ryzen AI 7 PRO 350: 4 fast + 4 compact
    cores, SMT): 8 threads are barely faster than 4 and 8-thread timings are
    noisy; compare loop-level timers rather than whole iterations. Check for
    stray background jobs first (an orphaned `pse.map` sweep from a stdin script
    ran 4 h, every job dying at worker startup; `map` should fail fast then).
10. `perf` works via `/usr/lib/linux-tools-6.8.0-146/perf` (the 6.17 kernel's
    tools package ships no perf). Use `-D -1 --control fifo:...` and have the
    script write `enable`/`disable` to profile just the iterations; per-thread
    breakdown with `--sort pid` and `--tid`.

## Phase C2 (next)

Correctness bar for every step: each step agrees with Evolver's original loops to
~1e-12 relative (`PYSE_NO_FAST_LOOPS=1` gives the original loops); relaxed
equilibria (energy, volumes, pressures, Hessian index) agree to ~1e-9 across 1, 4
and 8 threads; `PYSE_CHECK_FACET_CACHE=1` test suite, sanitizer sweep and full
tests pass.

1. **Done.** Per-thread accumulation in the linear loops (small). Per-thread partial sums
   (volumes, energies, forces), merged in a fixed order, replacing the serial
   ordered pass. Turn the bit-identity tests (`tests/test_fastloops.py`) into the
   tolerance and physics tests above.
2. **Benchmark that matches the workload** (small). Extend `bench/benchmark.py`:
   linear iterations, the high-order ladder (Lagrange 2/4/6: convert, `g 5`,
   `hessian`) and linear Newton steps at 98k / 393k / 1.6M facets, with
   `--threads 1 4 8 16`, JSON output for comparing runs.
3. **Faster Newton steps** (medium, highest value).
   1. METIS experiment: port `src/metis.c` to the METIS 5 API, build METIS from
      source via CMake (Apache-2.0, shippable in wheels), compare fill-in and
      factorization time with `mindeg.c` (linear and Lagrange, three sizes).
      Default only if it wins. (System has libmetis.so.5 but no headers.)
   2. Parallel Hessian assembly (`hessian_fill` and the named-quantity Hessian
      path): per-facet blocks computed in parallel, merged into the sparse matrix.
   3. Decision point: if factorization still dominates at 1.6M, bring measurements
      and a recommendation on a modern sparse LDL^T (must still report the Hessian
      index/inertia). **Pause here and report.**
4. **Parallel named-quantity loops** (large). Profile a Lagrange-4 iteration
   first; parallelize the per-facet quantity value/gradient loops (`calc_quants`
   and methods) with OpenMP. Retire Evolver's old threading code in the same step:
   drop its scheduling (worker threads, `thread_launch`, `-p`), turn its
   per-thread data (`thread_data`: eval stacks, `q_info`) into OpenMP thread-local
   storage.
5. **Done.** Remaining linear hot spots (medium): volume gradients (`film_grad_l`,
   ~16% of an iteration) and the volume-restoring projection (`volume_restore`,
   ~12%), with per-thread accumulation.
6. **Scaling run on a quiet machine** (later, by the user): `bench/benchmark.py
   --threads 1 4 8 16`; tune thread defaults (`FL_PARALLEL_MIN`, `pse.map` split).

Pause after step 3 (METIS/solver decision) and at the end of C2.

## Phase D: polish and usability

1. Live view that only moves points (no PolyData rebuild) when topology is unchanged.
2. Size warnings for curved tessellation and export.
3. Generic wrappers in `bindings/module.cpp`; a more uniform `Mesh`.
4. Notebook: `_repr_html_`; the user checks the live view in Jupyter.
5. Optional wait-with-timeout instead of the immediate "busy" error.
6. Docs: API reference and a tutorial notebook (load/build -> relax -> plot ->
   export for FEM).
7. Optional stop-gap for the display-mode leak.

## Phase E: state refactor (long term)

1. Feasibility probe: generate a state struct from the ~1,240 extern globals and
   ~280 file statics; count name clashes.
2. Globals into one struct behind macros (`#define web (evolver_state->web)`),
   fresh state on every load (fixes the settings leak); measure the indirection.
3. Optional multiple engines per process, after making Evolver's libc state
   per-engine (`strtok`, `rand`/`drand48`, time formatting, `chdir`).
4. Explicit state passing only in hot paths where profiling shows a gain.

## Tools and conventions

- Tests: `pytest`; with `PYSE_CHECK_FACET_CACHE=1` for cache verification.
- Sanitizers: `tools/run_sanitizers.sh [build-dir]` (stock program, ASan+UBSan,
  all samples).
- Benchmark: `python bench/benchmark.py --levels 6 8` (98k and 1.6M facets).
- Profiling build: `pip install . -C build-dir=<dir> -C cmake.define.PYSE_NOSTRIP=ON
  -C install.strip=false -C cmake.define.CMAKE_C_FLAGS=-g`, then
  `valgrind --tool=callgrind --toggle-collect=iterate python script.py`
  (LTO may inline/rename functions; use `--threshold=100` to see small entries).
- Reference build for A/B timing: `git worktree add <dir> <commit>` + separate venv.
- Evolver source files use CRLF line endings: edit them with a script that keeps
  CRLF (never rewrite them with LF).
- Don't `pkill -f` with a pattern that also appears in the same shell command.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
