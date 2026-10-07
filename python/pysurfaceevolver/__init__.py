"""Python frontend to Ken Brakke's Surface Evolver.

>>> from pysurfaceevolver import Evolver
>>> ev = Evolver("cube.fe")          # sample datafiles are found anywhere
>>> ev.iterate(5); ev.refine(); ev.iterate(5)
>>> ev.eval("body[1].volume")
>>> ev.values("vertex", "x^2 + y^2")   # one value per vertex
>>> mesh = ev.mesh()                   # NumPy arrays: mesh.vertices, mesh.faces, ...
>>> ev.write("surface.vtu")            # any meshio format
>>> ev.plot("area")                    # PyVista
"""

from ._build import Body, make_datafile
from ._evolver import (
    Evolver,
    EvolverError,
    EvolverExit,
    EvolverFatalError,
    EvolverWarning,
    InvalidSurfaceError,
    IterationResult,
    Parameters,
    Snapshot,
)
from ._mesh import Bodies, BodySurface, Mesh, Quantity, is_watertight
from ._viz import LiveView
from . import examples
from ._parallel import JobError, WorkerCrashed, map  # noqa: A004 (pse.map)


def set_threads(n: int) -> None:
    """Threads for Evolver's parallel facet loops; n <= 0 restores the
    default (all cores, or OMP_NUM_THREADS / PYSE_THREADS)."""
    from . import _core
    _core.set_threads(int(n))


def threads() -> int:
    """Threads the parallel facet loops use (1 if built without OpenMP)."""
    from . import _core
    return _core.threads()

def _start_engine() -> None:
    """Start the engine if it isn't running (it applies PYSE_SOLVER then)."""
    from . import _core
    if not _core.is_initialized():
        Evolver()


def set_solver(name: str) -> None:
    """Sparse factoring for Newton steps (``hessian``): ``"mumps"`` (the
    default when pySE is built with MUMPS) or ``"evolver"`` (Evolver's own
    minimal-degree factoring). Also set by the ``PYSE_SOLVER`` environment
    variable at start-up."""
    from . import _core
    _start_engine()
    if not _core.set_solver(str(name)):
        if name == "mumps":
            raise ValueError("this build of pySurfaceEvolver has no MUMPS")
        raise ValueError(f"unknown solver {name!r}; use 'mumps' or 'evolver'")


def solver() -> str:
    """The sparse factoring used for Newton steps (see :func:`set_solver`)."""
    from . import _core
    _start_engine()
    return _core.solver()


examples._add_to_evolverpath()

__all__ = [
    "Bodies",
    "Body",
    "BodySurface",
    "Evolver",
    "EvolverError",
    "EvolverExit",
    "EvolverFatalError",
    "EvolverWarning",
    "InvalidSurfaceError",
    "IterationResult",
    "JobError",
    "LiveView",
    "Mesh",
    "Parameters",
    "Quantity",
    "Snapshot",
    "WorkerCrashed",
    "is_watertight",
    "examples",
    "make_datafile",
    "set_solver",
    "set_threads",
    "solver",
    "threads",
]
__version__ = "0.5.0"
