# Changelog

## Unreleased

### Robust by default (phase G, in progress)

- Fixed: `relax()` failed in the Lagrange model ("Cannot equiangulate LAGRANGE model"):
  its tidying rounds now only average vertices there.
- `ev.stability()` (and so `relax()`'s check) needs `ritz` only when some eigenvalue is
  below zero; a stable surface costs one factorization. `relax()` at 98k facets:
  80 -> 1.0 s, at 393k: 275 -> 4.2 s.
- Faster safeguards: a safeguarded Newton step undoes itself by the vertex
  coordinates instead of a dump (`ev.newton(1)` at 393k facets 3.85 -> 1.77 s, raw
  `hessian` 1.2 s), `relax()` no longer dumps the surface at its start, and
  `ev.residual()` finds the shape directions in parallel (393k facets, 8 threads:
  0.39 -> 0.14 s).
- `relax()` reorganized: the experimental `remesh=True` and `IterationResult.remeshes` are
  gone (`ev.adapt()` and `ev.remesh()` remain); a result prints as a summary
  (`IterationResult(converged after 130 gradient steps and 2 Newton, residual 2.8e-09,
  stable, health ok)`), and so does `Health`. Health messages name named constraints.
- `relax()` hands over to Newton sooner: `energy_tol` now defaults to 1e-5 when Newton
  steps follow (1e-9 with `newton=0`). The tidying in every round keeps the energy from
  settling to 1e-9, so the gradient phase used to run its full 1000 steps, long enough
  for slow mesh drift to wreck a surface (contact-line vertices sliding together on a
  coarse liquid bridge). Stress suite: 7 of 9 (the bridge passes), far fewer gradient
  steps (cylinder 26000 -> 1060, shrinking cap 25880 -> 3040).
- `relax()` undoes a gradient round that more than doubles the residual (when Newton
  follows) and hands over to Newton: on a coarse start a contact line can collapse
  within a few steps. The liquid bridge relaxes on plain defaults (`relax(levels=3)`)
  at every gap and contact angle of its notebook, which no longer needs its own
  relaxation recipe.
- `relax(cg="auto")` (new default): when the first pass ends unconverged, a second pass
  of conjugate gradient steps (at most 200, averaging vertices) and Newton, kept only if
  it converges. A sessile drop spread to 10 degrees now converges. `cg=True` still uses
  conjugate gradients throughout; `cg=False` never.
- `ev.adapt(max_turn=15)`: splits the edges along which the surface turns by more than
  `max_turn` degrees (between the vertex normals at their ends), one level per call,
  never coarsening; `relax(adapt=True)` (opt-in) alternates it with relaxation, up to
  three passes or 4x the facets. A sessile drop at 170 degrees: implied contact angle
  off by 3.5 degrees with 734 facets, against 3.9 with 1920 refined uniformly.
- `ev.adapt()` also coarsens (`coarsen=True`): short interior edges along which the
  surface turns by less than a third of `max_turn` are merged, unless that would tilt a
  facet by more than `max_turn/2`, flip one, or make an edge longer than 4/3 of the bulk
  length. Engine: Evolver's edge deletion checks this when the new variables
  `collapse_max_tilt` (degrees) and `collapse_max_edge` are set (both 0 by default:
  no change for Evolver scripts).
- `ev.health()` and `relax()`'s `result.health` (`pyse.Health`): residual, stability,
  smallest facet angle and skinny facets, the nearest approach to a wall or mirror a vertex
  isn't on (beyond two edges of its contact line) and of the surface to itself (in edge
  lengths), vertices on the far side of a constraint, and `issues` in words (empty: `ok`).
  No warnings. About 0.2 s at 98k facets. New C binding: per-vertex signed distances to
  the constraints.
- The drainage helper (docs) relaxes with plain `ev.relax(stability=False)` after its
  remeshing; its Newton rollback (`undo_if` on the mirror gap) is no longer needed
  (band pressures within 0.02% median of the stored level-3 run, snap 0.4% off).
- Newton and eigenvalue counts no longer see spurious modes from vertices on wires
  (curves where two constraints meet): their only freedom, sliding along the wire, is
  left to the gradient steps (`hessian_slant_cutoff` now 0.05, and its test uses the
  raw surface normal and the freedom's magnitude). A stable catenoid counts 0 negative
  eigenvalues (was about one per wire vertex).
- `ev.residual()` balances the pressures (volume and quantity multipliers) in the shape
  directions, as Newton does; Evolver's own fit over the full velocities, tangential
  mesh forces included, left a uniform remainder at a Newton solution (the double
  bubble stalled at 5.9e-7; now 4e-11).
- The examples (tutorial, liquid bridge, slit droplet, drainage) relax with plain
  `ev.relax()` / `ev.relax(levels=3)`.
- `ev.stability()` and, once `relax()` converges, `result.stable` /
  `result.negative_modes` with an `UnstableEquilibriumWarning`: unstable equilibria
  (a liquid column past Rayleigh-Plateau, a barrel drop rolling up) are reported
  instead of silently returned; symmetry zero modes are not mistaken for them.
- `ev.residual()`: how far the surface is from equilibrium (the normal part of the
  projected vertex forces, dimensionless; 0 at an equilibrium). New C binding.
- `relax()` defaults: rounds of 10 gradient steps with equiangulation and vertex
  averaging, then up to 20 safeguarded Newton steps, converged when the residual is
  below `tol` (now a residual tolerance, default 1e-8; `energy_tol` ends the gradient
  phase). A NaN surface is put back to the start of the gradient round where it
  happened and raised, not reported as converged.
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
