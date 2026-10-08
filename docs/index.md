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

tutorial
api
```

The README has a
compact overview of the API; the tutorial works through one problem from datafile to
FEM mesh; the API reference documents every public class and function.
