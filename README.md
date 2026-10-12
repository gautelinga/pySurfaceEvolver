# pySurfaceEvolver

A Python frontend to Ken Brakke's [Surface Evolver](http://www.susqu.edu/brakke/evolver)
(version 2.70a). Evolver runs in-process as a compiled extension module built with
[nanobind](https://github.com/wjakob/nanobind); the mesh comes back as NumPy arrays.

```bash
pip install .            # core
pip install ".[all]"     # + PyVista (visualization, also in Jupyter) and meshio (mesh files)
pip install ".[jupyter]" # PyVista with its notebook backend only
```

Documentation: <https://gautelinga.github.io/pySurfaceEvolver/>, with an API
reference, a tutorial (a double bubble, from arrays to a FEM-ready mesh), worked
examples and performance numbers; built from `docs/` with Sphinx (see Development).

## A drop on a table, end to end

```python
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes

drop = shapes.cap(radius=1, height=0.6)          # any start: relaxing meets the volume
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60)},  # normal into the liquid
               vertex_constraints={1: drop.rims["rim"]},
               bodies=[pyse.Body(volume=1.0)])   # faces: all, oriented out of the liquid
result = ev.relax(levels=2)                      # converged? stable? health?
print(result, ev.energy, ev.body(1).pressure)
rim = ev.contact_lines(1)[0]                     # the contact line, in order
ev.plot(walls=True)                              # with the table drawn
```

Conventions: facets point out of the liquid, a plane's normal into it (spheres and
cylinders are solid); body and element numbers in Evolver (`ev.body(1)`, `vertex[5]`)
are 1-based, array rows (`ev.mesh()`, `ev.bodies().pressure[0]`) 0-based. A body
whose faces point into it, or a plane facing the solid, gets a warning when it loads.

## Scripting

```python
from pysurfaceevolver import Evolver

ev = Evolver("cube.fe")              # bundled samples are found from anywhere
result = ev.iterate(10)              # energy, area, scale per iteration
ev.refine()
ev.relax(tol=1e-10, newton=10)       # iterate until the energy settles, then Newton
ev.relax(levels=2, tidy=10, newton=3)   # refine twice, equiangulate + average every 10
ev.remesh(target=0.05)               # even out edge lengths (split long, delete short)
ev.body(1).target = 1.2              # a body's volume; ev.body(1).pressure
ev.eigen_counts().negative           # > 0: an unstable equilibrium
ev.saddle(); ev.relax()              # step off a saddle into the minimum it leads to
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
ev.set_flag("edge", "no_refine", where=ev.mesh().edges_touching(ev.on_constraint(1), "one"))
ev.parameters["angle"] = 60          # parameters declared in the datafile
ev["scale_limit"] = 0.01             # any existing variable (a typo raises KeyError)
ev.define("h", 0.1)                  # a new one, for commands
ev.gravity = 2                       # gravity_constant
ev.energy                            # total energy + what wall integrals leave out
ev.derivative("height")              # dE/dparameter at equilibrium (minus: the force)
ev.quantities()                      # named quantities: value, target, modulus, ...
ev.quantity("vol").target = 1.5      # a live handle (target None frees it)
ev.contact_angles(1)                 # degrees, at each vertex on constraint 1
ev.contact_lines(1)                  # where the surface meets constraint 1: polylines
ev.mesh().slice((0, 0, 1))           # cross-section z = 0: ordered polylines

ev.command("g 5; r; u")              # anything else: plain Evolver commands
```

Walls with contact angles, written as constraint integrals for you:

```python
from pysurfaceevolver import constraints as C

bead = C.sphere((0, 0, 0), 0.48, contact_angle=40, wet_poles=("north",), span=np.pi/2)
datafile = pyse.make_datafile(vertices, faces, constraints={
    "ceiling": C.plane((0, 0, -1), point=(0, 0, 0.5), contact_angle="theta"),  # a parameter
    "bead": bead, "mx": C.mirror("x"), "my": C.mirror("y")},
    bodies=[pyse.Body(volume=0.1)],          # volconst from the bead, as it touches it
    parameters={"theta": 60}, vertex_constraints={"bead": on_bead, ...})
# films on curved wires: boundaries={1: C.circle(radius=1)}, vertex_boundaries={1: rim}
```

`pyse.recipes.continuation(ev, ev.body(1), volumes)` (or a parameter's name, or
`ev.quantity(name)`) steps through a family of equilibria, with checkpoints a stopped
run resumes from; `stop_on="unconverged"` ends it at a fold, back at the last good
step. `ev.plot(walls=True)` draws the constraints' solids with the surface.

There is one Evolver engine per process: every `Evolver` object is a handle to
it, so all handles see the same surface (a handle used after another one loaded a
different surface warns). Use `save()`/`restore()` to keep surfaces around, and
separate processes to work on several at once. `ev.mesh()` is cached
until the surface changes, so its arrays are read-only. The sample datafiles and
command scripts are in `pysurfaceevolver.examples`.

## Parameter sweeps

```python
import pysurfaceevolver as pyse

def run(volume):
    ev = pyse.Evolver("cube.fe")
    ev.body(1).target = volume
    ev.relax(tol=1e-10)
    return ev.total_area

if __name__ == "__main__":   # needed in scripts: workers re-import them
    areas = pyse.map(run, [0.5, 1, 2, 4], processes=4)   # one engine per worker process
```

Each worker process runs jobs one after another. A job that raises, or even crashes
its worker, comes back as a `JobError` (`errors="raise"` or `"return"`) without
affecting the other jobs. With `cloudpickle` installed, functions defined in a
notebook work too. Workers are started with `spawn`, which re-imports the main
script: run scripts from a file, with the guard above; if workers die while
starting, `pyse.map` raises `WorkerStartError` right away.

## Building surfaces in Python

```python
from pysurfaceevolver import Body

ev = Evolver()
ev.load_arrays(vertices, faces,                  # triangles or polygons
               bodies=[Body(volume=1.0)],        # all faces unless faces= is given
               constraints={1: "z = 0"}, vertex_constraints={1: on_floor},
               fixed=wire_vertices, parameters={"angle": 90})
box = pyse.shapes.box(open_bottom=True, n=4)     # also cap, disc, tube, revolve

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
ev.write("surface.glb")                    # glTF or VRML scenes (through PyVista)
ev.write("surface.msh", curved="native")   # high-order cells (Gmsh, VTU; XDMF order 2)
ev.write_bodies("body_{id}.stl")           # one closed surface per body
ev.write_bodies("drop_{id}.stl", cap=True) # close openings where a body meets a constraint
surface = ev.body_surfaces(cap=True)[1]    # closed surface of body 1, caps tagged (cap_ids)
mesh = surface.volume_mesh(size=0.05, path="drop.msh")   # tetrahedra with Gmsh, groups kept
```

Curved (quadratic/Lagrange) elements are written either as flat triangles sampled
on the curved surface (`curved="tessellate"`, the default, works everywhere) or as
native high-order cells. Per-body surfaces have outward normals; films between
two bodies appear in both. A body that ends on a constraint (a drop on a table or a
bead) has openings; `cap=True` closes each with a cap on that constraint, Evolver
projecting its points (any constraint formula). `volume_mesh` needs `pip install gmsh`. `.msh` means Gmsh format 2.2.

## Performance

Against Brakke's original Evolver 2.70a (its own `gcc -O3` build and sparse solver),
on the sample cube refined to N facets, seconds (rough: best of two runs, quiet
laptop with 4 full and 4 compact cores; pySE with the OpenMP OpenBLAS; details and
more rows in the docs' performance page):

| | original | pySE, 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| `g 1`, 98k facets | 0.32 | 0.026 | 0.012 | 0.010 |
| `g 1`, 1.6M facets | 7.1 | 0.94 | 0.46 | 0.40 |
| Newton step, 393k facets | 3.9 | 1.42 | 0.70 | 0.46 |
| Newton step, 1.6M facets | 28.7 | 7.1 | 3.2 | 2.8 |
| Newton step, Lagrange 6, 24k facets | 22.9 | 4.2 | 1.8 | 1.42 |
| `u` + `V`, 393k facets | 1.23 | 0.73 | 0.24 | 0.14 |

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

## Errors and interrupts

Errors raise `EvolverError` (with `InvalidSurfaceError` after a failed load, until a
datafile loads), warnings are issued as `EvolverWarning`, and Ctrl-C stops a long run
at the next iteration; pressing it again aborts the operation (`KeyboardInterrupt`).
Pass `echo=True` to see output live, and `input=` to answer interactive prompts.

There is one engine per process, so calls from several threads are serialized: a call
made while another thread's call runs raises `EvolverBusyError` (a `RuntimeError`),
or with `pyse.busy_timeout = seconds` (`float("inf")`: no limit) waits for it first.
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
plus refined samples with Newton steps through an OpenMP build (`tools/run_sanitizers.sh`), builds the docs (published to GitHub Pages from `main`), and builds manylinux wheels
(x86_64, aarch64) plus an sdist as workflow artifacts. The build uses link-time optimization (`PYSE_LTO`);
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
