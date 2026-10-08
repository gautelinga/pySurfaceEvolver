# pySurfaceEvolver

A Python frontend to Ken Brakke's [Surface Evolver](http://www.susqu.edu/brakke/evolver)
(version 2.70a). Evolver runs in-process as a compiled extension module built with
[nanobind](https://github.com/wjakob/nanobind); the mesh comes back as NumPy arrays.

```bash
pip install .            # core
pip install ".[all]"     # + PyVista (visualization, also in Jupyter) and meshio (mesh files)
pip install ".[jupyter]" # PyVista with its notebook backend only
```

Documentation: an API reference and a tutorial (a double bubble, from arrays to a
FEM-ready mesh) are built from `docs/` with Sphinx (see Development).

## Scripting

```python
from pysurfaceevolver import Evolver

ev = Evolver("cube.fe")              # bundled samples are found from anywhere
result = ev.iterate(10)              # energy, area, scale per iteration
ev.refine()
ev.relax(tol=1e-10, hessian=True)    # iterate until the energy settles, then Newton
ev.set_model("lagrange", 3)

snapshot = ev.save()                 # exact snapshot of the surface
ev.refine(2); ev.relax()
ev.restore(snapshot)                 # back to it, bit for bit

ev.eval("body[1].volume")            # numeric expressions -> float
ev.values("vertex", "x^2 + y^2")     # one value per vertex, aligned with ev.mesh()
ev.values("facet", "area")

ev.define_attribute("vertex", "temperature")
ev.set_values("vertex", "temperature", temps)      # write per-element data (fast in C
                                                   # for coordinates and attributes)
ev.fix("vertex", where=ev.values("vertex", "z") > 0.9)
ev.set_constraint("vertex", 1, where=mask)
ev.parameters["angle"] = 60          # parameters declared in the datafile
ev.quantities()                      # named quantities: value, target, modulus, ...

ev.command("g 5; r; u")              # anything else: plain Evolver commands
```

There is one Evolver engine per process: every `Evolver` object is a handle to
it, so all handles see the same surface. Use `save()`/`restore()` to keep surfaces
around, and separate processes to work on several at once. `ev.mesh()` is cached
until the surface changes, so its arrays are read-only. The sample datafiles and
command scripts are in `pysurfaceevolver.examples`.

## Parameter sweeps

```python
import pysurfaceevolver as pse

def run(volume):
    ev = pse.Evolver("cube.fe")
    ev.set_values("body", "target", volume)
    ev.relax(tol=1e-10)
    return ev.eval("total_area")

if __name__ == "__main__":   # needed in scripts: workers re-import them
    areas = pse.map(run, [0.5, 1, 2, 4], processes=4)   # one engine per worker process
```

Each worker process runs jobs one after another. A job that raises, or even crashes
its worker, comes back as a `JobError` (`errors="raise"` or `"return"`) without
affecting the other jobs. With `cloudpickle` installed, functions defined in a
notebook work too. Workers are started with `spawn`, which re-imports the main
script: run scripts from a file, with the guard above; if workers die while
starting, `pse.map` raises `WorkerStartError` right away.

## Building surfaces in Python

```python
from pysurfaceevolver import Body

ev = Evolver()
ev.load_arrays(vertices, faces,                  # triangles or polygons
               bodies=[Body(faces=range(len(faces)), volume=1.0)],
               constraints={1: "z = 0"}, vertex_constraints={1: on_floor},
               fixed=wire_vertices, parameters={"angle": 90})

ev.load_mesh_file("part.stl", volume="current")  # any meshio format
```

`make_datafile(...)` returns the same thing as `.fe` text, to save or edit.

## Visualization (PyVista)

```python
ev.plot("area")                                  # color by any expression
ev.plot("x^2 + y^2", element="vertex")
view = ev.live_view("area")                      # window / notebook widget
ev.iterate(200, callback=view.update, every=10)  # redraws while it runs (moves the
                                                 # points in place until topology changes)
poly = ev.mesh().to_pyvista()                    # or work with the data
```

## Mesh files

```python
ev.write("surface.vtu")                    # STL, OBJ, PLY, VTU, Gmsh .msh, XDMF, ...
ev.write("surface.msh", curved="native")   # high-order cells (Gmsh, VTU; XDMF order 2)
ev.write_bodies("body_{id}.stl")           # one closed surface per body
ev.write_bodies("drop_{id}.stl", cap=True) # close openings on constraint planes
```

Curved (quadratic/Lagrange) elements are written either as flat triangles sampled
on the curved surface (`curved="tessellate"`, the default, works everywhere) or as
native high-order cells. Per-body surfaces have outward normals; films between
two bodies appear in both. `.msh` means Gmsh format 2.2.

## Performance

For linear soapfilm surfaces (no torus, symmetry, metric, Wulff or curvature
energies), the facet volume, energy and force loops and the body volume
gradients use a cached facet topology and run in parallel (OpenMP): each thread
sums its share of the facets, and the partial sums are merged in thread order. Results agree with Evolver's original
loops to round-off (about 1e-15 per evaluation; runs that stop short of
equilibrium can drift further apart, equilibria agree) and are reproducible for a
given thread count. One iteration on a 1.6M-facet surface went from 11 s to 2.3 s.

Named quantities (the Lagrange model, `convert_to_quantities`, quantity integrals
on facets, edges and vertices) run in parallel too, for methods known to be
thread-safe; integrands that use only arithmetic, math functions, parameters and
coordinates qualify, others (user procedures, assignments, ...) run serially.

Threads default to the number of physical cores. `pse.set_threads(n)` /
`pse.threads()` control them (also `OMP_NUM_THREADS`, or `PYSE_THREADS`);
`with pse.threads_limit(n):` sets them for a block, and `ev.relax(...,
threads=n)` / `ev.hessian(threads=n)` for one call; `pse.map`
workers run single-threaded by default, since for sweeps more processes beat more
threads. With several threads, results can differ from run to run at round-off
level; with one thread they are reproducible bit for bit. `PYSE_NO_FAST_LOOPS=1`
switches back to the original loops; `PYSE_CHECK_FACET_CACHE=1` verifies every cached
facet corner against Evolver's own topology (CI runs with it).

Newton steps (`hessian`) factor the Hessian with [MUMPS](https://mumps-solver.org/)
(sequential, LDL^T with pivoting, inertia for the Hessian index) when pySE is built
with it: wheels are; a source build downloads and builds MUMPS if a Fortran compiler
and LAPACK/BLAS are found (use an optimized BLAS such as OpenBLAS: the reference BLAS
makes MUMPS several times slower), and otherwise uses Evolver's own factoring. A
Newton step on a 1.6M-facet surface takes 9 s instead of 25 s. `pse.set_solver("evolver")`
(or `PYSE_SOLVER=evolver`) switches back to Evolver's minimal-degree factoring;
`pse.solver()` tells which is in use. For the Lagrange model, the Hessian is also
assembled in parallel.

## Errors and interrupts

Errors raise `EvolverError` (with `InvalidSurfaceError` after a failed load, until a
datafile loads), warnings are issued as `EvolverWarning`, and Ctrl-C stops a long run
at the next iteration; pressing it again aborts the operation (`KeyboardInterrupt`).
Pass `echo=True` to see output live, and `input=` to answer interactive prompts.

There is one engine per process, so calls from several threads are serialized: a call
made while another thread's call runs raises `EvolverBusyError` (a `RuntimeError`),
or with `pse.busy_timeout = seconds` (`float("inf")`: no limit) waits for it first.
A call from inside a running call, such as from a callback, always raises.

Limitations: loading a datafile resets Evolver's settings (tested: results of every
sample are the same whatever was loaded before), but the view matrix in dumps can
still depend on earlier datafiles. Builds and CI cover Linux only. Torus
models aren't unwrapped for plotting or export.

## Development

```bash
pip install ".[test]"
pytest
python bench/benchmark.py --levels 6 8 --threads 1 4   # linear at 98k and 1.6M facets,
                                                       # Lagrange 2/4/6 at 6k and 24k
pip install ".[docs]"
sphinx-build -W docs docs/_build/html   # API reference + tutorial (docs/tutorial.ipynb,
                                       # executed during the build)
```

CI (`.github/workflows/`) runs the tests and mypy on several Python versions, runs
every sample datafile through the stock program under AddressSanitizer and UBSan,
plus refined samples with Newton steps through an OpenMP build (`tools/run_sanitizers.sh`), builds the docs, and builds manylinux wheels (x86_64, aarch64) plus an
sdist; the docs, wheels and sdist are uploaded as workflow artifacts. The build uses link-time optimization (`PYSE_LTO`);
`-C cmake.define.PYSE_NOSTRIP=ON -C install.strip=false` keeps symbols for profiling.
The version lives in `python/pysurfaceevolver/__init__.py` only.

## Layout

- `src/` – the Surface Evolver C sources, maintained as a fork of 2.70a (the first git
  commit is the untouched original). Bug fixes apply to every build, and the stock
  `Makefile` still builds the standalone program (`GRAPH=nulgraph.o` for headless);
  hooks for the Python library are guarded by `#ifdef PYSE`.
- `bindings/` – C glue (`pyse_api.c`) that contains Evolver's `setjmp`/`longjmp`
  error handling, and the nanobind module (`module.cpp`).
- `python/pysurfaceevolver/` – the Python API; type stubs for the extension are
  generated at build time.
- `tests/` – the pytest suite.
- `fe/`, `doc/`, `manual270.pdf` – Evolver's sample datafiles and manual.

## License

pySurfaceEvolver's own code is MIT-licensed (`LICENSE`). Surface Evolver itself (most
of `src/`) is by Kenneth A. Brakke and freely available under his terms; the software
bundled in the wheels keeps its own licenses (`THIRD_PARTY_LICENSES.txt`).
