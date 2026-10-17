# pySurfaceEvolver

A Python frontend to Ken Brakke's [Surface Evolver](https://kenbrakke.com/evolver/evolver.html).
Surface Evolver finds surfaces of least energy: soap films, bubbles, drops on walls,
liquid bridges, solder joints. pySE runs the engine in-process. Surfaces are built
from NumPy arrays and come back as NumPy arrays, `ev.relax()` takes them to an
equilibrium and says how it went, and the results can be plotted with PyVista and
written out for finite-element meshing.

## Where to start

* New here? {doc}`getting_started` installs pySE, puts a drop on a table and sets out
  the conventions every other page uses. The {doc}`tutorial` then takes one problem,
  a double bubble, from arrays to a finite-element mesh, and {doc}`tour` lists the
  main calls on one page.
* {doc}`relaxing` explains what `relax()` does, how to read its result, and where its
  defaults stop working. Read it before the examples, or when a result looks wrong.
* The examples add one thing at a time: contact angles on curved solids
  ({doc}`liquid_bridge`), forces and optimizing parameters from Brakke's datafiles
  ({doc}`solder_joints`), several walls and mirror planes ({doc}`slit_droplet`), and a
  long quasi-static run ({doc}`slit_drainage`), following a bridge through its
  instability and fold ({doc}`bridge_rupture`), liquid in throats and pores of a
  sphere packing ({doc}`packing`), and draining through changes of topology: dry
  patches opening and a pore's liquid jumping ({doc}`drainage_topology`).
* Classic results from the literature, each against an exact reference: the
  stability limits of a bridge between two disks ({doc}`disk_bridge`), the largest
  drop a ring can hold ({doc}`pendant_drop`), drops on one, two and crossed fibres
  ({doc}`fibres`), and a lens floating on a liquid ({doc}`floating_lens`).
* Brakke's sample datafiles run as they come, each with his own recipe and with
  `relax()` side by side: a gallery of what Evolver can model.
* The reference has the API, fourteen stress cases with exact references, performance
  numbers, and how to build and test pySE ({doc}`development`).

```{toctree}
:caption: Getting started
:maxdepth: 2

getting_started
tutorial
tour
```

```{toctree}
:caption: Guides
:maxdepth: 2

relaxing
several_surfaces
```

```{toctree}
:caption: Examples
:maxdepth: 2

liquid_bridge
solder_joints
slit_droplet
slit_drainage
bridge_rupture
packing
drainage_topology
disk_bridge
pendant_drop
fibres
floating_lens
```

```{toctree}
:caption: Brakke's samples
:maxdepth: 1

brakke_films
brakke_capillarity
brakke_crystals_membranes
brakke_strings
brakke_lagrange
```

```{toctree}
:caption: Reference
:maxdepth: 1

api
stress_cases
performance
development
```
