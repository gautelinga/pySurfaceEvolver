# A tour of the API

The main calls on one page, each with a line on what it does; the
[API reference](api.md) has the details, and [Getting started](getting_started.md) the
conventions. The code here is not run when the documentation is built: it shows calls,
not a whole problem (`vertices`, `temps`, `mask` and the like stand for your own arrays).

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
import numpy as np
import pysurfaceevolver as pyse
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
import pysurfaceevolver as pyse
from pysurfaceevolver import Body

ev = pyse.Evolver()
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

## Threads

Iterations, Newton steps, equiangulation and vertex averaging run in parallel
(OpenMP), and Newton steps factor the Hessian with [MUMPS](https://mumps-solver.org/).
Threads default to the number of physical cores; `pyse.set_threads(n)`,
`with pyse.threads_limit(n):` or `OMP_NUM_THREADS` change that. With one thread
results are reproducible bit for bit, with several they can differ at round-off. The
[performance page](https://gautelinga.github.io/pySurfaceEvolver/performance.html)
says what runs in parallel, how to control threads and the solver, and how to build
with an optimized BLAS.

## Errors and interrupts

Errors raise `EvolverError` (with `InvalidSurfaceError` after a failed load, until a
datafile loads), warnings are issued as `EvolverWarning`, and Ctrl-C stops a long run
at the next iteration; pressing it again aborts the operation (`KeyboardInterrupt`).
Pass `echo=True` to see output live, and `input=` to answer interactive prompts.

There is one engine per process, so calls from several threads are serialized: a call
made while another thread's call runs raises `EvolverBusyError` (a `RuntimeError`),
or with `pyse.busy_timeout = seconds` (`float("inf")`: no limit) waits for it first.
A call from inside a running call, such as from a callback, always raises.

## Limitations

loading a datafile resets Evolver's settings (tested: results of every
sample are the same whatever was loaded before), but the view matrix in dumps can
still depend on earlier datafiles. Builds and CI cover Linux only. Torus
models aren't unwrapped for plotting or export.
