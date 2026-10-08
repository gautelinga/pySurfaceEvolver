# Performance

A rough comparison with Brakke's original Surface Evolver 2.70a, measured on
2026-10-08. The original was built from the unmodified sources with its own Linux
settings (`gcc -O3 -DLINUX`, no graphics; one empty stub function had to be added
for it to link) and uses its own sparse factoring for Newton steps. pySE used
MUMPS with OpenBLAS (OpenMP build).

**Test surface:** the sample `cube.fe` (a unit volume relaxing to a sphere),
refined repeatedly with `r; g 5` after each refinement, then two Newton steps.
`g 1` is one gradient iteration; a Newton step is one `hessian` (the better of
two consecutive steps). The Lagrange cases continue with `lagrange 2`, `4`, `6`,
each followed by `g 5` and a Newton step, before the timed steps.

**Timing:** the original is single-threaded and was timed with Evolver's `clock`
(process CPU time; a wall-clock check of 20 iterations at 98k facets agreed);
pySE was timed with `time.perf_counter()` around `ev.command(...)`, with
`OMP_NUM_THREADS` = 1, 4 and 8. One run each on a 16-core laptop under light load
(load average about 1.5-2); the 4-thread runs overlapped with short single-core
runs, which probably explains the slow 4-thread `g 1` at 1.6M facets (0.65 s on a
quiet machine earlier).

Seconds, with the speed-up over the original in parentheses:

| | original | pySE, 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| `g 1`, 98k facets | 0.32 | 0.039 (8×) | 0.026 (12×) | 0.029 (11×) |
| Newton step, 98k facets | 0.58 | 0.28 (2.0×) | 0.13 (4.4×) | 0.12 (4.8×) |
| `g 1`, 393k facets | 1.29 | 0.30 (4.3×) | 0.16 (8×) | 0.13 (10×) |
| Newton step, 393k facets | 3.9 | 1.43 (2.7×) | 0.84 (4.7×) | 0.59 (6.6×) |
| `g 1`, 1.6M facets | 7.1 | 1.17 (6×) | 0.91 (8×) | 0.54 (13×) |
| Newton step, 1.6M facets | 28.7 | 7.3 (3.9×) | 4.9 (5.9×) | 3.6 (7.9×) |
| Newton step, Lagrange 6, 6k facets | 4.3 | 3.2 (1.4×) | 1.7 (2.5×) | 0.95 (4.5×) |
| Newton step, Lagrange 6, 24k facets | 22.9 | 13.5 (1.7×) | 6.9 (3.3×) | 4.1 (5.6×) |

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
