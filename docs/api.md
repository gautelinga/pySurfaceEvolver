# API reference

Every public class and function, from the docstrings. For how the pieces fit
together, start with {doc}`getting_started` (the conventions) and {doc}`relaxing`.

```{eval-rst}
.. currentmodule:: pysurfaceevolver
```

## The engine

There is one Surface Evolver engine per process; every {class}`Evolver` is a
handle to it.

```{eval-rst}
.. autoclass:: Evolver
   :inherited-members:
.. autoclass:: IterationResult
.. autoclass:: Health
   :members: ok
.. autoclass:: BodyView
.. autoclass:: QuantityView
.. autoclass:: EigenCounts
   :no-members:
.. autoclass:: Stability
.. autoclass:: Snapshot
.. autoclass:: Parameters
   :members: optimizing
```

## Surface data

```{eval-rst}
.. autoclass:: Mesh
.. autoclass:: MeshQuality
   :no-members:
.. autoclass:: Bodies
   :no-members:
.. autoclass:: Quantity
   :no-members:
.. autoclass:: BodySurface
.. autofunction:: is_watertight
```

## Building surfaces

```{eval-rst}
.. autofunction:: make_datafile
.. autoclass:: Body
   :no-members:
```

### Starting shapes

```{eval-rst}
.. automodule:: pysurfaceevolver.shapes
   :members: cap, disc, tube, revolve, box, Shape
.. currentmodule:: pysurfaceevolver
```

### Walls, mirrors and wires

```{eval-rst}
.. automodule:: pysurfaceevolver.constraints
   :members: plane, mirror, sphere, cylinder, Constraint, circle, curve, Boundary
.. currentmodule:: pysurfaceevolver
```

## Visualization

```{eval-rst}
.. autoclass:: LiveView
```

## Recipes

```{eval-rst}
.. automodule:: pysurfaceevolver.recipes
   :members: continuation, arclength, Step, load_steps
.. currentmodule:: pysurfaceevolver
```

## Parameter sweeps

```{eval-rst}
.. autofunction:: map
.. autoclass:: JobError
   :no-members:
.. autoexception:: WorkerCrashed
.. autoexception:: WorkerStartError
```

## Settings

```{eval-rst}
.. autofunction:: set_threads
.. autofunction:: threads
.. autofunction:: threads_limit
.. autofunction:: set_solver
.. autofunction:: solver
.. autodata:: tessellation_limit
.. autodata:: busy_timeout
```

## Errors and warnings

```{eval-rst}
.. autoexception:: EvolverError
.. autoexception:: InvalidSurfaceError
.. autoexception:: EvolverFatalError
.. autoexception:: EvolverExit
.. autoexception:: EvolverBusyError
.. autoexception:: EvolverWarning
.. autoexception:: LargeTessellationWarning
.. autoexception:: UnstableEquilibriumWarning
.. autoexception:: SharedEngineWarning
```

## Sample files

```{eval-rst}
.. automodule:: pysurfaceevolver.examples
```
