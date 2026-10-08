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
`g 1` is one gradient iteration; a Newton step is one `hessian`; `u` is
equiangulation and `V` vertex averaging. The Lagrange cases continue with
`lagrange 2`, `4`, `6`, each followed by `g 5` and a Newton step, before the timed
steps.

**Timing:** the original is single-threaded and was timed with Evolver's `clock`
(process CPU time; a wall-clock check of 20 iterations at 98k facets agreed);
pySE was timed with `time.perf_counter()` around `ev.command(...)`, the better of
two consecutive runs of each command, with `OMP_NUM_THREADS` = 1, 4 and 8 on a
quiet machine (load below 2 for the linear cases, about 4 for the Lagrange ones).
The original's numbers are from 2026-10-08 (the `u` and `V` rows from the same
day's evening), pySE's (version 0.6.0) from the evening of 2026-10-08. Repeated runs vary by about 10-15%. The machine is a
laptop with 4 full and 4 compact cores (Ryzen AI 7 350), so 8 threads gain less
over 4 than the core count suggests.

Seconds, with the speed-up over the original in parentheses:

| | original | pySE, 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| `g 1`, 98k facets | 0.32 | 0.026 (12×) | 0.012 (25×) | 0.010 (31×) |
| Newton step, 98k facets | 0.58 | 0.26 (2.2×) | 0.12 (4.7×) | 0.10 (5.7×) |
| `u`, 98k facets | 0.061 | 0.022 (2.8×) | 0.0067 (9×) | 0.0044 (14×) |
| `V`, 98k facets | 0.19 | 0.13 (1.5×) | 0.036 (5.3×) | 0.025 (7.6×) |
| `g 1`, 393k facets | 1.29 | 0.22 (5.8×) | 0.084 (15×) | 0.054 (24×) |
| Newton step, 393k facets | 3.9 | 1.42 (2.8×) | 0.70 (5.6×) | 0.46 (8.5×) |
| `u`, 393k facets | 0.32 | 0.115 (2.8×) | 0.038 (8×) | 0.022 (14×) |
| `V`, 393k facets | 0.91 | 0.62 (1.5×) | 0.20 (4.5×) | 0.12 (7.6×) |
| `g 1`, 1.6M facets | 7.1 | 0.94 (7.6×) | 0.46 (16×) | 0.40 (18×) |
| Newton step, 1.6M facets | 28.7 | 7.1 (4.0×) | 3.2 (9×) | 2.8 (10×) |
| `u`, 1.6M facets | 1.38 | 0.55 (2.5×) | 0.16 (9×) | 0.099 (14×) |
| `V`, 1.6M facets | 3.9 | 2.55 (1.5×) | 0.80 (4.9×) | 0.50 (7.7×) |
| Newton step, Lagrange 6, 6k facets | 4.3 | 0.89 (4.9×) | 0.39 (11×) | 0.31 (14×) |
| Newton step, Lagrange 6, 24k facets | 22.9 | 4.2 (5.5×) | 1.8 (13×) | 1.42 (16×) |

The energies agree with the original's (checked with the same command sequence):
to 15-16 digits for the linear cases, and to about $10^{-12}$ relative for
Lagrange 6 at 24k facets.

**Where the gains come from:**

- Iterations (`g 1`): a cached table of facet corners instead of walking Evolver's
  facet-edge structures; facet energies, forces, volumes and volume gradients in
  parallel, with per-thread sums on separate cache lines; per-vertex work in
  parallel; selections of edges and vertices by attribute kept until the surface
  changes; link-time optimization.
- Newton steps: MUMPS instead of Evolver's own factoring, with its analysis kept
  from step to step and all constraint columns solved at once; the matrix pattern
  kept from step to step and filled in parallel; vertex normals in parallel; the
  Lagrange element Hessians as BLAS matrix products, with each element's terms
  projected and entered once.
- `u` and `V`: the edge-swap tests and the vertex-average search in parallel (the
  swaps themselves stay serial, in Evolver's order; results are identical), and
  `u` no longer recomputes every edge length five times.
- What is left: the MUMPS factorization (about a third of a Lagrange 6 step and a
  third to a half of a large linear one), recording the matrix entries, and `r`
  (refinement), which is still serial (about 1.3 s at 393k facets).

`bench/benchmark.py` in the repository measures pySE alone (`--threads`,
`--levels`, `--json` to compare runs).
