# pySurfaceEvolver

A Python frontend to Ken Brakke's [Surface Evolver](http://www.susqu.edu/brakke/evolver)
(version 2.70a). Evolver runs in-process as a compiled extension module built with
[nanobind](https://github.com/wjakob/nanobind); the mesh comes back as NumPy arrays.

```bash
pip install .            # core
pip install ".[all]"     # + PyVista (visualization) and meshio (mesh files)
```

## Scripting

```python
from pysurfaceevolver import Evolver

ev = Evolver("fe/cube.fe")
result = ev.iterate(10)              # energy, area, scale per iteration
ev.refine(); ev.iterate(10)
ev.hessian()
ev.set_model("lagrange", 3)

ev.eval("body[1].volume")            # numeric expressions -> float
ev.values("vertex", "x^2 + y^2")     # one value per vertex, aligned with ev.mesh()
ev.values("facet", "area")

ev.define_attribute("vertex", "temperature")
ev.set_values("vertex", "temperature", temps)      # write per-element data
ev.fix("vertex", where=ev.values("vertex", "z") > 0.9)
ev.set_constraint("vertex", 1, where=mask)
ev.parameters["angle"] = 60          # parameters declared in the datafile
ev.quantities()                      # named quantities: value, target, modulus, ...

ev.command("g 5; r; u")              # anything else: plain Evolver commands
```

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
ev.iterate(200, callback=view.update, every=10)  # redraws while it runs
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

## Errors and interrupts

Errors raise `EvolverError` (with `InvalidSurfaceError` after a failed load, until a
datafile loads), warnings are issued as `EvolverWarning`, and Ctrl-C stops a long run
at the next iteration; pressing it again aborts the operation (`KeyboardInterrupt`).
Pass `echo=True` to see output live, and `input=` to answer interactive prompts.

Limitations: Evolver keeps its state in C globals, so there is one engine per process
(use `multiprocessing` for parallel runs). Calls from several threads are serialized:
a call made while another is running raises `RuntimeError`. Builds and CI cover
Linux only. Torus models aren't unwrapped for plotting or export.

## Development

```bash
pip install ".[test]"
pytest
```

CI (`.github/workflows/`) runs the tests and mypy on several Python versions, and
builds manylinux wheels (x86_64, aarch64) plus an sdist as workflow artifacts.
The version lives in `python/pysurfaceevolver/__init__.py` only.

## Layout

- `src/` – the Surface Evolver C sources. The stock `Makefile` still builds the
  standalone program; edits for the library build are guarded by `#ifdef PYSE`.
- `bindings/` – C glue (`pyse_api.c`) that contains Evolver's `setjmp`/`longjmp`
  error handling, and the nanobind module (`module.cpp`).
- `python/pysurfaceevolver/` – the Python API; type stubs for the extension are
  generated at build time.
- `tests/` – the pytest suite.
- `fe/`, `doc/`, `manual270.pdf` – Evolver's sample datafiles and manual.
