# API reference

```{eval-rst}
.. currentmodule:: pysurfaceevolver
```

## The engine

There is one Surface Evolver engine per process; every {class}`Evolver` is a
handle to it.

```{eval-rst}
.. autoclass:: Evolver
.. autoclass:: IterationResult
.. autoclass:: Health
   :members: ok
.. autoclass:: BodyView
.. autoclass:: EigenCounts
   :no-members:
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

### Walls and mirrors

```{eval-rst}
.. automodule:: pysurfaceevolver.constraints
   :members: plane, mirror, sphere, cylinder, Constraint
.. currentmodule:: pysurfaceevolver
```

## Visualization

```{eval-rst}
.. autoclass:: LiveView
```

## Recipes

```{eval-rst}
.. automodule:: pysurfaceevolver.recipes
   :members: continuation, Step, body_target, parameter, load_steps
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
```

## Sample files

```{eval-rst}
.. automodule:: pysurfaceevolver.examples
```
