"""Workflow recipes built on the core API.

:func:`continuation` follows a family of equilibria: it steps a value (a body's
volume, a parameter, ...), relaxes after each step, can checkpoint to disk, and
yields a record per step::

    from pysurfaceevolver import recipes

    for step in recipes.continuation(ev, recipes.body_target(1),
                                     np.linspace(1.0, 0.2, 50),
                                     relax=dict(tidy=3, newton=3),
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
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union

import numpy as np

__all__ = ["Step", "body_target", "parameter", "continuation", "load_steps"]


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


def body_target(number: int) -> Callable[[Any, float], None]:
    """A setter for :func:`continuation`: the prescribed volume of a body."""
    def set_value(ev, value: float) -> None:
        ev.body(number).target = value
    set_value.__name__ = f"body_target({number})"
    return set_value


def parameter(name: str) -> Callable[[Any, float], None]:
    """A setter for :func:`continuation`: a datafile parameter."""
    def set_value(ev, value: float) -> None:
        ev.parameters[name] = value
    set_value.__name__ = f"parameter({name!r})"
    return set_value


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
    kept = [Step(s.index, s.value, s.energy, s.volumes, s.pressures, None, s.extra)
            for s in steps]
    with open(prefix + ".steps.tmp", "wb") as f:
        pickle.dump(kept, f)
    os.replace(prefix + ".steps.tmp", prefix + ".steps")


def load_steps(prefix: str) -> List[Step]:
    """The steps a checkpointed :func:`continuation` has finished."""
    with open(prefix + ".steps", "rb") as f:
        return pickle.load(f)


def continuation(ev, set_value: Callable[[Any, Any], None], values: Iterable[Any], *,
                 relax: Union[None, dict, Callable[[Any], Any]] = None,
                 checkpoint: Optional[str] = None,
                 resume: Optional[str] = None) -> Iterator[Step]:
    """Step through ``values``: for each, ``set_value(ev, value)``, relax, and
    yield a :class:`Step`.

    ``set_value``: a callable ``(ev, value)``, e.g. :func:`body_target` or
    :func:`parameter`. ``relax``: options for :meth:`Evolver.relax` (a dict),
    a callable ``relax(ev)``, or None for ``ev.relax()``.

    ``checkpoint``: a path prefix; after every step the surface goes to
    ``<prefix>.dmp`` and the steps so far to ``<prefix>.steps`` (both written
    atomically). ``resume``: such a prefix; the surface is loaded from it and
    the values it finished are skipped (``values`` must be the same sequence).
    A step's ``extra`` dict is yours to fill in the loop; it is checkpointed
    with the next step.

    Changing ``ev`` inside the loop (refining, loading a rebuilt surface) is
    fine: the next step continues from whatever ``ev`` holds.
    """
    values = list(values)
    relax_fn = _relax_with(relax)
    done: List[Step] = []
    if resume is not None:
        ev.load(resume + ".dmp")
        done = load_steps(resume)
    for index in range(len(done), len(values)):
        value = values[index]
        set_value(ev, value)
        result = relax_fn(ev)
        bodies = ev.bodies()
        step = Step(index, value, ev.total_energy, np.array(bodies.volume),
                    np.array(bodies.pressure), result)
        done.append(step)
        yield step
        if checkpoint is not None:
            _save(checkpoint, ev, done)
