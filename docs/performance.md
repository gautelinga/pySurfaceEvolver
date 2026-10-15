# Performance

How fast pySE is against Brakke's original program, what a default `relax()` costs on
large surfaces, and how to control threads and the Newton solver. For setting up and
relaxing surfaces, start with {doc}`getting_started` and {doc}`relaxing`.

## Against the original Evolver

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
  from step to step and all constraint columns solved at once; a pattern factored six
  times analysed again with PORD, slower to analyse than AMD but with less fill (at 4
  threads, steps 10-15% faster in Lagrange 2 at 98k facets and Lagrange 4 at 24k, and in
  the linear model at 1.6M; about the same at 393k and below). `arclength()` takes 6-9
  steps a point on one pattern, `relax()` 2-3: a Lagrange 2 run of six points at 98k
  facets took 37-40 s instead of 41-42. The matrix pattern
  kept from step to step and filled in parallel; vertex normals in parallel; the
  Lagrange element Hessians as BLAS matrix products, with each element's terms
  projected and entered once.
- `u` and `V`: the edge-swap tests and the vertex-average search in parallel (the
  swaps themselves stay serial, in Evolver's order; results are identical), and
  `u` no longer recomputes every edge length five times.
- The quadratic model's facet loops in parallel, from a cached table of each facet's
  six control points: Evolver's own energies, forces, volumes and volume gradients
  (used until the first Newton step), the named-quantity loops and Hessian assembly
  that `hessian` switches to, and the vertex normals (as in the Lagrange model). The
  volume and its gradient are summed over pairs of control points, 90 products
  instead of Evolver's 216 a point (equal to round-off). Cube at 98k facets, 4
  threads: `g 1` 0.65-0.84 -> 0.17-0.19 s before the first Newton step and
  0.48-0.57 -> 0.21-0.27 s after it, a Newton step 1.8-2.1 -> 0.95 s, a default
  `relax()` 67-71 -> 34-38 s. The small matrix products of the quadratic tangents and
  Gauss points (and the Lagrange edges') are inline, in `mat_mult`'s order.
- What is left: the MUMPS factorization (about a third of a Lagrange 6 step and a
  third to a half of a large linear one), recording the matrix entries, and `r`
  (refinement), which is still serial: refining 393k facets to 1.6M takes 1.4 s,
  less than half a Newton step there at 4 threads. Its profile has no hotspot to
  take out: about 40% is zeroing new elements in fresh memory (the atomic counter
  that the profile puts it on isn't the cost: a plain one was no faster), the rest
  walking edge loops and setting up the new edges and facets.

## `relax()` at scale

What a default `ev.relax()` costs against the commands a script would run by hand, on
a surface near its equilibrium (8 threads, best of two, measured 2026-10-10 with the
desktop busy: load 3 to 8, so expect 10-20% noise). By hand: `g 5; hessian; hessian;
hessian` (in the Lagrange model after `lagrange n`, the manual's final stage). `relax()`
does more: it takes Newton steps until the residual is below 1e-8 and verifies it,
checks stability (one factorization) and the mesh's health. In the linear model it also
runs its tidied gradient rounds (at least three of ten steps); in the quadratic and
Lagrange models it does what the hand-written stage does, `g 5` and then Newton.
Seconds:

| surface | by hand | `relax()` |
|---|---|---|
| cube, linear, 98k facets | 0.58 | 0.95 |
| cube, linear, 393k facets | 3.05 | 5.15 |
| cube, Lagrange 2, 24k facets | 0.79 | 1.04 |
| cube, Lagrange 4, 24k facets | 2.96 | 5.40 |
| double bubble (tutorial), Lagrange 2, 45k facets | 1.43 | 2.71 |
| double bubble (tutorial), Lagrange 4, 45k facets | 5.93 | 10.92 |

All converged and were reported stable with a healthy mesh. `relax(stability=False)`
saves the factorization when the stability isn't needed.

Since then (pySE after 0.7.1) `relax()` stops the Lagrange stage's Newton steps once
the residual is below `tol` (two or three of four or five steps were in round-off),
takes the stability check from the last Newton step's own factorization when that
step began close to equilibrium, and computes vertex normals (the residual, the
Hessian setup) from facet normals kept per topology. Against 0.7.1, the same
`relax()` on the same surfaces (2 threads, best of two, the versions alternated,
identical energies to 12 digits), seconds:

| surface | 0.7.1 | now |
|---|---|---|
| cube, linear, 98k facets | 1.60 | 1.09 |
| cube, linear, 393k facets | 9.7 | 7.0 |
| cube, Lagrange 2, 98k facets | 8.3 | 5.4 |
| cube, Lagrange 4, 24k facets | 6.5-7.6 | 4.0-4.4 |

After 0.7.2 the Hessian setup skips Evolver's serial per-vertex normal for
vertices that are on no constraint (their factor is 1), the Lagrange gradient and
value kernels compute their per-Gauss-point factors once, Lagrange vertex normals
come from the per-topology normal cache too, and vertex averaging (`V`) uses
per-topology vertex rings built in parallel (bit-identical to Evolver's). Against
0.7.2 (2 threads, best of three interleaved rounds, energies identical to 15
digits), milliseconds per operation and seconds per `relax()`:

| surface | operation | 0.7.2 | now |
|---|---|---|---|
| cube, linear, 98k facets | Newton step | 313 | 197 |
| | `V` | 41 | 26 |
| | `relax()` | 1.14 | 0.96 |
| cube, linear, 393k facets | Newton step | 1460 | 1140 |
| | `relax()` | 7.0 | 6.5 |
| cube, Lagrange 2, 98k facets | `g 1` | 373 | 313 |
| | Newton step | 1440 | 1190 |
| | residual | 255 | 205 |
| | `relax()` | 5.25 | 4.09 |
| cube, Lagrange 4, 24k facets | residual | 195 | 129 |
| | `relax()` | 5.1 | 4.8 |

Then the gradient step with named quantities only (the Lagrange model,
`convert_to_quantities`) makes 9 passes over the elements instead of 14: the energy
gradients come with the volume gradients, and each line-search move evaluates the
energy in its last volume projection pass (bit-identical results). Against the
above (2 threads, best of three interleaved rounds, loaded machine): Lagrange 2,
98k facets, `g 1` 330 -> 292 ms, `g 5` 1.53 -> 1.38 s; Lagrange 4, 24k facets,
`g 1` 218 -> 196 ms, `g 5` 0.98 -> 0.92 s (instructions 17% fewer; `relax()`, with
its five gradient steps, 3-5% faster). Merging the linear model's facet volume and
energy loops the same way was slower (the fused loop cost more than the two) and
was dropped.

Tried and not kept: MUMPS's PORD ordering for every analysis (it costs more than it
saves on the 2-3 steps `relax()` takes on a pattern; it is used after six, above;
METIS isn't built), storage reordering along a space-filling curve (gradient steps
2x slower).

## Threads and the Newton solver

For linear soapfilm surfaces (no torus, symmetry, metric, Wulff or curvature
energies), the facet volume, energy and force loops and the body volume
gradients use a cached facet topology and run in parallel (OpenMP): each thread
sums its share of the facets, and the partial sums are merged in thread order. Results agree with Evolver's original
loops to round-off (about 1e-15 per evaluation; runs that stop short of
equilibrium can drift further apart, equilibria agree) and are reproducible for a
given thread count.

Named quantities (the Lagrange model, `convert_to_quantities`, quantity integrals
on facets, edges and vertices) run in parallel too, for methods known to be
thread-safe; integrands that use only arithmetic, math functions, parameters and
coordinates qualify, others (user procedures, assignments, ...) run serially.

Threads default to the number of physical cores. `pyse.set_threads(n)` /
`pyse.threads()` control them (also `OMP_NUM_THREADS`, or `PYSE_THREADS`);
`with pyse.threads_limit(n):` sets them for a block, and `ev.relax(...,
threads=n)` / `ev.newton(threads=n)` for one call; `pyse.map`
workers run single-threaded by default, since for sweeps more processes beat more
threads. With several threads, results can differ from run to run at round-off
level; with one thread they are reproducible bit for bit. Newton steps keep the
matrix's sparsity pattern from one step to the next while the topology is unchanged,
and add entries into it in parallel (`PYSE_NO_PATTERN=1` turns that off).
`PYSE_NO_FAST_LOOPS=1` switches back to the original loops; `PYSE_CHECK_FACET_CACHE=1` verifies every cached
facet corner against Evolver's own topology (CI runs with it).

Newton steps (`hessian`) factor the Hessian with [MUMPS](https://mumps-solver.org/)
(sequential, LDL^T with pivoting, inertia for the Hessian index) when pySE is built
with it: wheels are; a source build downloads and builds MUMPS if a Fortran compiler
and LAPACK/BLAS are found (use an optimized BLAS such as OpenBLAS: the reference BLAS
makes MUMPS several times slower), and otherwise uses Evolver's own factoring.
Use the OpenMP build of OpenBLAS: a pthreads OpenBLAS runs single-threaded inside
MUMPS's threads, and large Newton steps are up to 2x slower (1.6M facets: 3.8 s against
2.7 s). The build looks for one (Debian/Ubuntu: `libopenblas-openmp-dev`; Fedora:
`openblas-openmp`, with `openblas-static`) and links its static library into pySE with
private symbols, so it is used whatever BLAS NumPy loads; without one it warns at build
time and at the first Newton step. Wheels bundle their own.
`pyse.set_solver("evolver")`
(or `PYSE_SOLVER=evolver`) switches back to Evolver's minimal-degree factoring;
`pyse.solver()` tells which is in use. For the Lagrange model, the Hessian is also
assembled in parallel.

`bench/benchmark.py` in the repository measures pySE alone (`--threads`,
`--levels`, `--json` to compare runs).

See also: {doc}`relaxing` (what `relax()` does at each stage, and its cost at 400k
facets) and the {doc}`api` for {func}`~pysurfaceevolver.set_threads` and
{func}`~pysurfaceevolver.set_solver`.
