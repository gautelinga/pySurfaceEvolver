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
.. autoclass:: Snapshot
.. autoclass:: Parameters
   :members: optimizing
```

## Surface data

```{eval-rst}
.. autoclass:: Mesh
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

## Visualization

```{eval-rst}
.. autoclass:: LiveView
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
```

## Sample files

```{eval-rst}
.. automodule:: pysurfaceevolver.examples
```
