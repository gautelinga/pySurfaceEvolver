"""Workflow recipes built on the core API.

:func:`continuation` follows a family of equilibria: it steps a value (a body's
volume, a parameter, ...), relaxes after each step, can checkpoint to disk, and
yields a record per step::

    from pysurfaceevolver import recipes

    for step in recipes.continuation(ev, ev.body(1), np.linspace(1.0, 0.2, 50),
                                     checkpoint="runs/drain"):
        print(step.value, step.pressures[0])
        if something_happened(ev):
            ev.load_string(rebuilt_surface)   # same Evolver: the loop goes on with it

A run stopped for any reason resumes from its last finished step with
``resume="runs/drain"``.
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union

import numpy as np

__all__ = ["Step", "continuation", "load_steps"]


@dataclass
class Step:
    """One step of :func:`continuation`, after relaxing."""

    index: int                 # position in ``values``
    value: Any                 # the value set
    energy: float
    volumes: np.ndarray        # per body
    pressures: np.ndarray      # per body
    result: Any = None         # what the relax call returned (IterationResult, ...)
    extra: dict = field(default_factory=dict)   # yours: kept in checkpoints
    substeps: int = 1          # relaxes it took (more after halving, see continuation)


def _setter(what: Any) -> Callable[[Any, Any], None]:
    """continuation()'s setter for what it steps."""
    from ._views import BodyView, QuantityView
    if isinstance(what, (BodyView, QuantityView)):
        def set_target(ev, value) -> None:
            what.target = value
        return set_target
    if isinstance(what, str):
        def set_named(ev, value) -> None:
            if what in ev.parameters:
                ev.parameters[what] = value
            else:
                ev[what] = value
        return set_named
    if callable(what):
        return what
    raise TypeError("continuation() steps a parameter or variable name, ev.body(i), "
                    "ev.quantity(name), or a callable(ev, value)")


def _relax_with(relax: Union[None, dict, Callable[[Any], Any]]) -> Callable[[Any], Any]:
    if relax is None:
        return lambda ev: ev.relax()
    if isinstance(relax, dict):
        options = dict(relax)
        return lambda ev: ev.relax(**options)
    if callable(relax):
        return relax
    raise TypeError("relax must be None, a dict of relax() options, or a callable(ev)")


def _save(prefix: str, ev, steps: List[Step]) -> None:
    folder = os.path.dirname(prefix)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ev.dump(prefix + ".dmp.tmp")
    os.replace(prefix + ".dmp.tmp", prefix + ".dmp")
    kept = [Step(s.index, s.value, s.energy, s.volumes, s.pressures, None, s.extra, s.substeps)
            for s in steps]
    with open(prefix + ".steps.tmp", "wb") as f:
        pickle.dump(kept, f)
    os.replace(prefix + ".steps.tmp", prefix + ".steps")


def load_steps(prefix: str) -> List[Step]:
    """The steps a checkpointed :func:`continuation` has finished."""
    with open(prefix + ".steps", "rb") as f:
        return pickle.load(f)


def _failed(result: Any, stop_on: Optional[str]) -> bool:
    if stop_on is None:
        return False
    if getattr(result, "converged", True) is False:
        return True
    return stop_on == "unstable" and getattr(result, "stable", None) is False


def continuation(ev, what: Any, values: Iterable[Any], *,
                 relax: Union[None, dict, Callable[[Any], Any]] = None,
                 checkpoint: Optional[str] = None,
                 resume: Optional[str] = None,
                 stop_on: Optional[str] = None, halvings: int = 0) -> Iterator[Step]:
    """Step through ``values``: for each, set ``what`` to it, relax, and
    yield a :class:`Step`.

    ``what``: a parameter or variable name (``"theta"``), a body's target
    volume (``ev.body(1)``), a fixed quantity's target
    (``ev.quantity("vol")``), or a callable ``(ev, value)`` for anything
    else. ``relax``: options for :meth:`Evolver.relax` (a dict),
    a callable ``relax(ev)``, or None for ``ev.relax()``.

    ``checkpoint``: a path prefix; after every step the surface goes to
    ``<prefix>.dmp`` and the steps so far to ``<prefix>.steps`` (both written
    atomically). ``resume``: such a prefix; the surface is loaded from it and
    the values it finished are skipped (``values`` must be the same sequence).
    A step's ``extra`` dict is yours to fill in the loop; it is checkpointed
    with the next step.

    Changing ``ev`` inside the loop (refining, loading a rebuilt surface) is
    fine: the next step continues from whatever ``ev`` holds.

    ``stop_on="unconverged"`` (or ``"unstable"``: also a converged but
    unstable equilibrium) ends the run at the first step whose relax result
    says so, with the surface put back to the last good step and a warning:
    past a fold of the branch (a bridge stretched until it breaks, say)
    every later step would start from a wreck. ``halvings=k`` first retries
    such a step from the last good one in 2, 4, ... 2**k smaller steps
    (numeric values only); ``Step.substeps`` says how many it took.
    """
    if stop_on not in (None, "unconverged", "unstable"):
        raise ValueError("stop_on must be None, 'unconverged' or 'unstable'")
    if halvings < 0:
        raise ValueError("halvings must be non-negative")
    if halvings and stop_on is None:
        raise ValueError("halvings retries the steps stop_on rejects: give stop_on too")
    values = list(values)
    set_value = _setter(what)
    relax_fn = _relax_with(relax)
    done: List[Step] = []
    if resume is not None:
        ev.load(resume + ".dmp")
        done = load_steps(resume)
    previous = values[len(done) - 1] if done else None
    for index in range(len(done), len(values)):
        value = values[index]
        good = ev.save() if stop_on is not None else None
        set_value(ev, value)
        result = relax_fn(ev)
        substeps = 1
        numeric = isinstance(value, (int, float, np.number)) and \
            isinstance(previous, (int, float, np.number))
        k = 0
        while _failed(result, stop_on) and k < halvings and numeric:
            k += 1
            ev.restore(good)
            n = 2**k
            for j in range(1, n + 1):
                set_value(ev, previous + (value - previous)*j/n)
                result = relax_fn(ev)
                if _failed(result, stop_on):
                    break
            substeps = n
        if _failed(result, stop_on):
            ev.restore(good)
            warnings.warn(f"continuation() stopped before {value!r}: relax() came out "
                          f"{'unstable' if getattr(result, 'converged', True) else 'unconverged'}"
                          " there; the surface is back at the last good step",
                          UserWarning, stacklevel=2)
            return
        previous = value
        bodies = ev.bodies()
        step = Step(index, value, ev.energy, np.array(bodies.volume),
                    np.array(bodies.pressure), result, substeps=substeps)
        done.append(step)
        yield step
        if checkpoint is not None:
            _save(checkpoint, ev, done)
