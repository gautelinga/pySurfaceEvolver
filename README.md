# pySurfaceEvolver

[![tests](https://github.com/gautelinga/pySurfaceEvolver/actions/workflows/tests.yml/badge.svg)](https://github.com/gautelinga/pySurfaceEvolver/actions/workflows/tests.yml)
[![docs](https://github.com/gautelinga/pySurfaceEvolver/actions/workflows/docs.yml/badge.svg)](https://gautelinga.github.io/pySurfaceEvolver/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python 3.9-3.13](https://img.shields.io/badge/python-3.9--3.13-blue.svg)

Soap films, bubbles, drops on walls, liquid bridges and solder joints from Python.
pySurfaceEvolver runs Ken Brakke's
[Surface Evolver](https://kenbrakke.com/evolver/evolver.html), the standard program for
surfaces of least energy, inside Python: surfaces go in and come back as NumPy arrays.

![A drop on a table, a double bubble, a drop on a fibre and a soap film between rings](https://raw.githubusercontent.com/gautelinga/pySurfaceEvolver/main/docs/_static/readme.png)

* **Evolver itself, not a reimplementation.** Its datafiles and commands work as they
  are; Brakke's sample files are included.
* **`relax()` takes a surface to equilibrium** and says how it went: converged,
  stable, mesh health. It looks after the mesh and switches to Newton steps on its
  own.
* **Contact angles, walls, mirrors and wires** in a line each, with no constraint
  integrals to work out by hand.
* **Out to plots and meshes:** PyVista views (also in Jupyter), STL, VTU, Gmsh and
  more, closed bodies and tetrahedral meshes for finite-element codes.
* **Fast:** parallel (OpenMP) with a sparse direct solver (MUMPS): iterations up to
  30 times faster than the original program, Newton steps 2 to 10 times.

## Install

```bash
git clone https://github.com/gautelinga/pySurfaceEvolver
cd pySurfaceEvolver
pip install ".[all]"      # with PyVista for plots and meshio for mesh files
```

This builds the engine from source with a C and C++ compiler (and, for the fast Newton
solver, gfortran with LAPACK/BLAS); builds are tested on Linux. See
[Install](https://gautelinga.github.io/pySurfaceEvolver/getting_started.html#install)
for the details.

## A drop on a table

```python
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes

drop = shapes.cap(radius=1, height=0.6)        # a rough start: relaxing meets the volume
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60)},
               vertex_constraints={1: drop.rims["rim"]},
               bodies=[pyse.Body(volume=1.0)])
result = ev.relax(levels=2)                    # refine twice, relax to equilibrium
print(result)
print(f"pressure {ev.body(1).pressure:.4f}, contact angles {ev.contact_angles(1).mean():.1f} degrees")
ev.plot(walls=True)                            # the drop, with the table drawn
```

```text
IterationResult(converged after 291 gradient steps and 3 Newton, residual 7.5e-13, stable, health ok)
pressure 1.7366, contact angles 59.2 degrees
```

## Documentation

<https://gautelinga.github.io/pySurfaceEvolver/>

* [Getting started](https://gautelinga.github.io/pySurfaceEvolver/getting_started.html):
  install, this drop step by step, and the conventions.
* [Tutorial](https://gautelinga.github.io/pySurfaceEvolver/tutorial.html): a double
  bubble from arrays to a finite-element mesh.
* [A tour of the API](https://gautelinga.github.io/pySurfaceEvolver/tour.html): the
  main calls on one page.
* [Relaxing a surface](https://gautelinga.github.io/pySurfaceEvolver/relaxing.html):
  what `relax()` does and where its defaults stop working.
* Worked examples (liquid bridges, solder joints, drops between beads), Brakke's
  sample files with his own recipes, the
  [API reference](https://gautelinga.github.io/pySurfaceEvolver/api.html),
  [stress tests](https://gautelinga.github.io/pySurfaceEvolver/stress_cases.html) and
  [performance](https://gautelinga.github.io/pySurfaceEvolver/performance.html).

## Contributing

Bug reports and questions are welcome as
[issues](https://github.com/gautelinga/pySurfaceEvolver/issues). Building, testing
and the layout of the code are described in
[Development](https://gautelinga.github.io/pySurfaceEvolver/development.html).

## License

pySurfaceEvolver's own code is MIT-licensed (`LICENSE`). Surface Evolver itself (most
of `src/`) is by Kenneth A. Brakke and freely available under his terms; the software
bundled in the wheels keeps its own licenses (`THIRD_PARTY_LICENSES.txt`).
