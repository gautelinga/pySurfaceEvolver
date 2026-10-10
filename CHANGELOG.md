# Changelog

## Unreleased

### Robust by default (phase G, in progress)

- Newton and eigenvalue counts no longer see spurious modes from vertices on wires
  (curves where two constraints meet): their only freedom, sliding along the wire, is
  left to the gradient steps (`hessian_slant_cutoff` now 0.05, and its test uses the
  raw surface normal and the freedom's magnitude). A stable catenoid counts 0 negative
  eigenvalues (was about one per wire vertex).
- `ev.residual()`: how far the surface is from equilibrium (the normal part of the
  projected vertex forces, dimensionless; 0 at an equilibrium). New C binding.
- `relax()` defaults: rounds of 10 gradient steps with equiangulation and vertex
  averaging, then up to 20 safeguarded Newton steps, converged when the residual is
  below `tol` (now a residual tolerance, default 1e-8; `energy_tol` ends the gradient
  phase). A NaN surface is restored and raised, not reported as converged.
- Newton steps (`relax`, `ev.newton()`) are safeguarded: plain Newton first, a line
  search if that is rejected; a step that makes things non-finite, or raises both the
  energy and the residual, is undone.
- `bench/stress/`: nine hard cases with exact or axisymmetric references, run with
  default settings.

### API (phase F: a friendlier API)

- `ev.relax(...)` does the whole recipe: `levels=n` (refine and relax again),
  `tidy=k` (equiangulation and vertex averaging every k steps, convergence measured
  round to round),
  `newton=n` with `seek=` and `undo_if=` (a Newton step is undone when
  `undo_if(ev)` is true). Replaces `hessian=`/`max_hessian=`, and
  `IterationResult.hessian_steps` is now `newton_steps` (breaking); the result has a
  per-iteration `level`. New `ev.newton(steps, seek=, tol=, undo_if=)`.
- `ev.remesh(target=h)` (or `max_edge=`, `min_edge=`): deletes short edges, splits
  long ones and equiangulates; `protect=` keeps an edge mask whole. Splitting honours
  `no_refine` (Evolver's `l` ignores it).
- Diagnostics as values: `ev.eigen_counts(shift)` (negative/zero/positive Hessian
  eigenvalues: an unstable equilibrium shows as `negative > 0`), `ev.check()` (topology
  problems, empty when sound), `Mesh.quality()` / `ev.mesh_quality()` (edge lengths,
  smallest angles, skinny and degenerate facets).
- Mirror images for symmetric pieces: `ev.plot(mirror=["z", "y", ("x", c)])`,
  `ev.live_view(mirror=...)` (the images share the data, so updates stay fast), and
  `Mesh.mirrored(planes)` for one combined linear mesh.
- `pyse.constraints`: planes, mirrors, spheres and cylinders with contact angles, for
  `make_datafile(constraints=)`: they write the wetting-energy and volume line
  integrals (and give the sphere's `volconst` and energy constant for wet poles);
  `Body(volconst=)`.
- `pyse.recipes.continuation(ev, set_value, values, relax=, checkpoint=, resume=)`:
  follows a family of equilibria (a body's volume, a parameter), with checkpoints that
  a stopped run resumes from; setters `recipes.body_target(i)`, `recipes.parameter(name)`.
- `ev.body(i)`: a live handle on one body (`.volume`, `.target` and `.volconst`
  settable, `.pressure`, `.fixed`); `target = None` frees the volume.
- `Bodies.target_volume` is now `Bodies.target` (breaking).
- `ev.on_constraint(k, element="vertex")`, `Mesh.edges_touching(mask, how=)` and
  `ev.set_flag(element, flag, where=, on=)` select elements with numpy masks instead
  of Evolver `where` clauses.
- `EvolverError.command` holds the failing command; errors hint at numpy reprs
  (`np.float64(...)`) and non-finite numbers in the command text.

### Fixes

- Tiny-edge deletion (`t`, also used by `w` and `delete`) no longer merges the ends of an
  edge when they share neighbours other than the third vertices of its facets (the link
  condition). Evolver 2.70a merged them, making parallel and loop edges; later deletions
  then freed elements twice, hung in `edge_valence()`, or lost the surface.

### Docs

- New example, `slit_drainage`: liquid drained from a bead chain in a slit, from immersed
  beads through bead emergence and the band to the snap. Runs a coarse case live and shows
  stored full runs and movies for three bead spacings.

### Speed

- Quantity mode: the value and gradient passes of the fast element loop sum into
  per-thread buffers merged in thread order, instead of recording every entry for a
  serial replay (`g 10` in quantity mode at 98k facets, 8 threads: 0.83 -> 0.61 s;
  results unchanged and deterministic).

## 0.6.0 (2026-10-08)

### Speed

Cube sample, 8 threads, against 0.5.0 (`bench/benchmark.py`): `g 1` at 98k facets
75 -> 10 ms, at 1.6M 1.34 -> 0.40 s; a Newton step at 98k 0.49 -> 0.10 s, at 1.6M
25.5 -> 2.8 s. Against the original Evolver 2.70a see `docs/performance.md`
(Lagrange 6 Newton step at 24k facets: 22.9 s there, 1.42 s here).

- Newton steps: MUMPS factoring (instead of Evolver's own), with its analysis kept
  across steps; the matrix pattern kept across steps and filled in parallel; vertex
  normals in parallel; all volume and quantity constraint columns solved at once.
- Lagrange models: the facet setup as a packed, vectorized kernel; the area and
  volume element Hessians as BLAS matrix products; each element's terms projected
  and entered once.
- Gradient steps: parallel named quantities (facet, edge and vertex integrals);
  per-thread sums kept on separate cache lines; attribute selections kept until the
  surface changes.
- `u` and `V`: the swap tests and vertex-average search in parallel, identical
  results (393k facets: `u` 0.30 -> 0.022 s, `V` 0.60 -> 0.12 s).
- Builds prefer the system's OpenMP OpenBLAS and link it statically with private
  symbols (no clash with NumPy's OpenBLAS); wheels bundle OpenBLAS 0.3.34 (OpenMP).

### Threads

- Default thread count: physical cores; `pyse.set_threads()`, `pyse.threads_limit()`
  and `threads=` per call; `pyse.map` workers run single-threaded.
- Safe Ctrl-C with threads: aborts wait until the parallel region or MUMPS call ends.
- Results agree with Evolver's serial loops to round-off and are reproducible to
  round-off from run to run.

### Library

- `ev.body_surfaces(cap=True)`: closed per-body surfaces, openings capped on the
  constraint their rim lies on (points projected by Evolver); `BodySurface.cap_ids`
  and `cap_constraints`; `BodySurface.volume_mesh()` for tetrahedral meshes with
  Gmsh (physical groups for the surface, caps and body).
- Notebook display: HTML summaries of `Evolver`, `Mesh`, bodies and parameters
  (`jupyter` extra); faster live view; a warning before very large tessellations.
- `busy_timeout` / `EvolverBusyError`: optionally wait for a running call from
  another thread instead of failing at once.
- `pyse.map` fails fast with `WorkerStartError` when workers can't start.

### Breaking changes

- `Mesh` arrays renamed to `facets`, `facet_ids`, `facet_bodies`; unused low-level
  bindings (`edges`, `facets`, `element_nodes`) removed.

### Fixes

- Edges with a density in a soapfilm model with a metric crashed (stack overflow,
  also in the original Evolver); they now use the metric edge length.
- Switching `bezier_basis` in a Lagrange model gave wrong energies (stale cached
  basis tables).
- Double free after a failed Newton step; signed overflow in the sparse Hessian
  hash; `metis_factor` falling through to the MKL setup; METIS 5 API.
- The torus display mode no longer carries over to the next loaded datafile.

### Documentation and tooling

- Sphinx documentation (API reference, double-bubble tutorial, liquid-bridge and
  slit-droplet examples with mesh export), published to GitHub Pages by CI.
- Performance comparison with the original Evolver 2.70a.
- MIT license for pySE's own code; third-party license notices shipped in wheels.
- Sanitizer runs include an OpenMP build with Newton steps, `u` and `V`;
  `tools/valgrind.supp`.
