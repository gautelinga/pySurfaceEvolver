"""High-level Python interface to Surface Evolver."""

from __future__ import annotations

from ._errors import (EvolverBusyError, EvolverError, EvolverExit, EvolverFatalError,
                      EvolverWarning, InvalidSurfaceError, UnstableEquilibriumWarning)
from ._export import _Export
from ._relax import _Relaxing
from ._results import EigenCounts, Health, IterationResult, Snapshot, Stability
from ._views import BodyView, Parameters, QuantityView

__all__ = [
    "BodyView", "EigenCounts", "Health", "QuantityView", "Stability",
    "UnstableEquilibriumWarning",
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

    Notes
    -----
    Surface Evolver keeps its state in C globals, so there is one engine per
    process, and every ``Evolver`` object is a handle to it: they all see the
    same surface, and loading through one changes it for all. ``echo`` and
    ``input`` belong to each handle. Use :meth:`save` and :meth:`restore` to
    keep a surface around, and separate processes to work on several at
    once.

    Calls are serialized: a call made while another thread is running one
    raises ``RuntimeError`` instead of waiting.

    Ctrl-C during a call (from the main thread) stops Evolver at the next
    iteration or statement; pressing it again aborts the operation. Both
    raise ``KeyboardInterrupt``.
    """
