"""High-level Python interface to Surface Evolver."""

from __future__ import annotations

from typing import Any

from ._errors import (EvolverBusyError, EvolverError, EvolverExit, EvolverFatalError,
                      EvolverWarning, InvalidSurfaceError, SharedEngineWarning,
                      UnstableEquilibriumWarning)
from ._export import _Export
from ._relax import _Relaxing
from ._results import EigenCounts, Health, IterationResult, Snapshot, Stability
from ._views import BodyView, Parameters, QuantityView

__all__ = [
    "BodyView", "EigenCounts", "Health", "QuantityView", "Stability",
    "SharedEngineWarning", "UnstableEquilibriumWarning",
    "Evolver",
    "IterationResult",
    "Parameters",
    "Snapshot",
    "EvolverBusyError",
    "EvolverError",
    "EvolverExit",
    "EvolverFatalError",
    "InvalidSurfaceError",
    "EvolverWarning",
]


class Evolver(_Relaxing, _Export):
    """A handle to the Surface Evolver engine of this process.

    Parameters
    ----------
    datafile:
        Optional path of a ``.fe`` datafile to load right away.
    echo:
        If true, print Evolver's output live while commands run. It's always
        returned from :meth:`command` either way.
    input:
        Optional callable ``input(prompt) -> str | None`` that answers
        Evolver's interactive prompts, for example in ``hessian_menu``.
        ``None`` (the default) answers every prompt with end-of-file.
    process:
        If true, the surface lives in a worker process of its own, with an
        engine of its own: several surfaces at once (see Notes).
    threads:
        With ``process=True``: the worker's threads (default: as
        :func:`set_threads` would choose; several busy workers share the
        cores).

    Notes
    -----
    Surface Evolver keeps its state in C globals, so there is one engine per
    process, and every ``Evolver`` object is a handle to it: they all see the
    same surface, and loading through one changes it for all. ``echo`` and
    ``input`` belong to each handle. Use :meth:`save` and :meth:`restore` to
    keep a surface around, and ``process=True`` to work on several at
    once.

    ``Evolver(process=True)`` starts a worker process (a new Python
    interpreter) with an engine of its own and returns a handle on it that
    works like any other: attribute reads, assignments and method calls
    go to the worker, views (``ev.body(1)``, ``ev.parameters``) are handles
    too, and results (meshes, IterationResults, arrays, snapshots) come back
    as copies. Each call costs a round trip (a fraction of a millisecond)
    plus copying its arguments and result. Callables given to methods
    (``relax(callback=...)``, ``undo_if=``) are copied to the worker and run
    there (sending functions defined in a script or notebook needs
    cloudpickle); ``input=`` isn't available. ``plot()`` and
    ``live_view()`` draw in this process (a live view updates when you call
    ``update()``; a callback running in the worker can't reach it).
    Prints and warnings from the worker show here, and Ctrl-C interrupts
    the worker's call. ``ev.close()`` (or ``with``, or the last handle going
    away) ends the worker; ``ev.pid`` is its process id. If it dies, the
    call raises :class:`WorkerCrashed`. POSIX systems only.

    Calls are serialized: a call made while another thread is running one
    raises ``RuntimeError`` instead of waiting.

    Ctrl-C during a call (from the main thread) stops Evolver at the next
    iteration or statement; pressing it again aborts the operation. Both
    raise ``KeyboardInterrupt``.
    """

    def __new__(cls, *args: Any, process: bool = False, **kwargs: Any) -> Any:
        if process:
            from ._remote import RemoteEvolver
            return RemoteEvolver(*args, **kwargs)
        return super().__new__(cls)

