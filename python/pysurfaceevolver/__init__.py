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

import contextlib
from typing import Generator, Optional

from ._build import Body, make_datafile
from ._evolver import (
    Evolver,
    EvolverBusyError,
    EvolverError,
    EvolverExit,
    EvolverFatalError,
    EvolverWarning,
    InvalidSurfaceError,
    IterationResult,
    Parameters,
    Snapshot,
)
from ._mesh import Bodies, BodySurface, LargeTessellationWarning, Mesh, Quantity, is_watertight
from ._viz import LiveView
from . import examples
from ._parallel import JobError, WorkerCrashed, WorkerStartError, map  # noqa: A004 (pse.map)

#: Tessellations (plots, live views, export of curved surfaces) with more
#: triangles than this give a LargeTessellationWarning; None: no check.
tessellation_limit: Optional[int] = 10_000_000

#: Seconds a call waits while Evolver runs a call from another thread, then
#: EvolverBusyError; None: fail at once; float("inf"): wait as long as needed.
busy_timeout: Optional[float] = None


def set_threads(n: int) -> None:
    """Threads for Evolver's parallel loops and the Newton-step solver;
    n <= 0 restores the default: the physical cores (or OMP_NUM_THREADS /
    PYSE_THREADS when set)."""
    from . import _core
    _core.set_threads(int(n))


def threads() -> int:
    """Threads the parallel loops and Newton steps use (1 without OpenMP)."""
    from . import _core
    return _core.threads()


@contextlib.contextmanager
def threads_limit(n: int) -> Generator[None, None, None]:
    """Use ``n`` threads inside the ``with`` block, then restore the previous
    setting (including "the default")::

        with pse.threads_limit(4):
            ev.relax(hessian=True)
    """
    from . import _core
    previous = _core.thread_setting()
    _core.set_threads(int(n))
    try:
        yield
    finally:
        _core.set_threads(previous)

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
    "EvolverBusyError",
    "EvolverError",
    "EvolverExit",
    "EvolverFatalError",
    "EvolverWarning",
    "InvalidSurfaceError",
    "IterationResult",
    "JobError",
    "LargeTessellationWarning",
    "LiveView",
    "Mesh",
    "Parameters",
    "Quantity",
    "Snapshot",
    "WorkerCrashed",
    "WorkerStartError",
    "is_watertight",
    "examples",
    "make_datafile",
    "set_solver",
    "set_threads",
    "solver",
    "threads",
    "threads_limit",
]
__version__ = "0.5.0"
