# pySurfaceEvolver

A Python frontend to Ken Brakke's [Surface Evolver](https://kenbrakke.com/evolver/evolver.html):
the engine runs in-process, surfaces come back as NumPy arrays, and they can be
plotted with PyVista and written for finite-element meshing.

```bash
pip install .            # core
pip install ".[all]"     # + PyVista (also in Jupyter) and meshio
```

```{toctree}
:maxdepth: 2

getting_started
tutorial
liquid_bridge
solder_joints
slit_droplet
slit_drainage
relaxing
stress_cases
brakke_films
brakke_capillarity
brakke_crystals_membranes
brakke_strings
brakke_lagrange
api
performance
```

The README has a
compact overview of the API; *Getting started* puts a drop on a table and sets out the
conventions (orientation, energies, surface tension, contact-angle accuracy, numbering);
the tutorial works through one problem from datafile to
FEM mesh; the liquid-bridge example adds contact angles on curved solids, the slit example walls and mirror planes; *Relaxing a surface* explains what `relax()` does, how to read its result and where the defaults stop working, and *Stress cases* runs nine hard problems against exact references; the API reference documents every public class and function.
