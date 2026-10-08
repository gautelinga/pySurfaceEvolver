# Performance

A rough comparison with Brakke's original Surface Evolver 2.70a, measured on
2026-10-08. The original was built from the unmodified sources with its own Linux
settings (`gcc -O3 -DLINUX`, no graphics; one empty stub function had to be added
for it to link) and uses its own sparse factoring for Newton steps. pySE used
MUMPS with the OpenMP build of OpenBLAS (0.3.26 from Ubuntu's
`libopenblas-openmp-dev`, linked statically; the wheels bundle 0.3.34). With a
pthreads OpenBLAS, large Newton steps are up to 2x slower (1.6M facets, 8 threads:
3.8 s).

**Test surface:** the sample `cube.fe` (a unit volume relaxing to a sphere),
refined repeatedly with `r; g 5` after each refinement, then two Newton steps.
`g 1` is one gradient iteration; a Newton step is one `hessian` (the better of
two consecutive steps). The Lagrange cases continue with `lagrange 2`, `4`, `6`,
each followed by `g 5` and a Newton step, before the timed steps.

**Timing:** the original is single-threaded and was timed with Evolver's `clock`
(process CPU time; a wall-clock check of 20 iterations at 98k facets agreed);
pySE was timed with `time.perf_counter()` around `ev.command(...)`, with
`OMP_NUM_THREADS` = 1, 4 and 8, one run each, starting on a quiet machine (load
below 2). Repeated runs vary by about 10-15%: for example the Lagrange 6 steps at
8 threads took 0.95 s and 4.1 s in an earlier run. At these sizes 8 threads are not
always faster than 4 for `g 1`, which is limited by memory bandwidth.

Seconds, with the speed-up over the original in parentheses:

| | original | pySE, 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| `g 1`, 98k facets | 0.32 | 0.035 (9×) | 0.035 (9×) | 0.039 (8×) |
| Newton step, 98k facets | 0.58 | 0.26 (2.2×) | 0.13 (4.5×) | 0.11 (5.2×) |
| `g 1`, 393k facets | 1.29 | 0.28 (4.5×) | 0.105 (12×) | 0.13 (10×) |
| Newton step, 393k facets | 3.9 | 1.42 (2.8×) | 0.62 (6.3×) | 0.55 (7.2×) |
| `g 1`, 1.6M facets | 7.1 | 1.14 (6×) | 0.49 (15×) | 0.65 (11×) |
| Newton step, 1.6M facets | 28.7 | 7.2 (4.0×) | 3.3 (8.8×) | 2.8 (10×) |
| Newton step, Lagrange 6, 6k facets | 4.3 | 3.4 (1.3×) | 1.34 (3.2×) | 1.15 (3.8×) |
| Newton step, Lagrange 6, 24k facets | 22.9 | 13.8 (1.7×) | 6.2 (3.7×) | 4.5 (5.1×) |

The energies agree: to 15-16 digits for the linear cases, and to about
$10^{-12}$ relative for Lagrange 6 at 24k facets.

**Where the gains come from:**

- Iterations (`g 1`): a cached table of facet corners instead of walking Evolver's
  facet-edge structures; facet energies, forces, volumes and volume gradients in
  parallel; per-vertex work in parallel; link-time optimization.
- Newton steps: MUMPS instead of Evolver's own factoring, with its analysis kept
  from step to step; the matrix pattern kept from step to step and filled in
  parallel; vertex normals and the Lagrange element Hessians in parallel.
- What is left: the Lagrange element Hessian kernels (about 40% of a Lagrange 6
  step) and the MUMPS factorization (about a third to a half of a large linear
  step).

`bench/benchmark.py` in the repository measures pySE alone (`--threads`,
`--levels`, `--json` to compare runs).
