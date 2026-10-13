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
    predicted: bool = False    # relaxed from the secant prediction (see continuation)


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
    kept = [Step(s.index, s.value, s.energy, s.volumes, s.pressures, None, s.extra, s.substeps,
                 s.predicted) for s in steps]
    with open(prefix + ".steps.tmp", "wb") as f:
        pickle.dump(kept, f)
    os.replace(prefix + ".steps.tmp", prefix + ".steps")


def load_steps(prefix: str) -> List[Step]:
    """The steps a checkpointed :func:`continuation` has finished."""
    with open(prefix + ".steps", "rb") as f:
        return pickle.load(f)


def _current(ev, what: Any) -> Any:
    """What ``what`` is now, as the start of a halved first step (None if
    unknown)."""
    from ._views import BodyView, QuantityView
    try:
        if isinstance(what, (BodyView, QuantityView)):
            return what.target
        if isinstance(what, str):
            return ev.parameters[what] if what in ev.parameters else ev[what]
    except Exception:
        return None
    return None


def _numeric(*values: Any) -> bool:
    return all(isinstance(v, (int, float, np.number)) for v in values)


def _snapshot(ev, value: Any) -> tuple:
    """An equilibrium for the predictor: the value, vertex positions and ids."""
    m = ev.mesh()
    return value, m.vertices.copy(), m.vertex_ids.copy()


def _same_vertices(ev, snapshot: tuple) -> bool:
    return np.array_equal(ev.mesh().vertex_ids, snapshot[2])


def _at(ev, snapshot: tuple) -> bool:
    """ev is still the surface of ``snapshot``: same vertices, not moved."""
    m = ev.mesh()
    return np.array_equal(m.vertex_ids, snapshot[2]) and np.array_equal(m.vertices, snapshot[1])


def _predict(ev, history: list, value: Any) -> bool:
    """Move the vertices to the secant prediction for ``value`` from the last
    two equilibria in ``history``, if they have the same vertices and ``ev``
    is still at the last (no refinement, remeshing, pop or other surface in
    between). Whether it did."""
    if len(history) < 2:
        return False
    (v0, x0, ids0), (v1, x1, ids1) = history[-2], history[-1]
    if not _numeric(v0, v1, value) or v1 == v0:
        return False
    ratio = (value - v1)/(v1 - v0)
    if abs(ratio) > 2:      # a jump: a straight line would overshoot (1, 10, 100)
        return False
    if not (np.array_equal(ids0, ids1) and _at(ev, history[-1])):
        return False
    ev.vertices = x1 + ratio*(x1 - x0)
    return True


def _unconverged(result: Any) -> bool:
    return getattr(result, "converged", True) is False or bool(getattr(result, "collapsed", 0))


def _failed(result: Any, stop_on: Optional[str]) -> bool:
    if stop_on is None:
        return False
    if getattr(result, "converged", True) is False:
        return True
    # past a fold relax() can pinch the surface off (deleting collapsed edges)
    # and converge on the pieces: another branch, not the next step of this one
    if getattr(result, "collapsed", 0):
        return True
    return stop_on == "unstable" and getattr(result, "stable", None) is False


def continuation(ev, what: Any, values: Iterable[Any], *,
                 relax: Union[None, dict, Callable[[Any], Any]] = None,
                 checkpoint: Optional[str] = None,
                 resume: Optional[str] = None,
                 stop_on: Optional[str] = None, halvings: int = 0,
                 predictor: bool = True) -> Iterator[Step]:
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
    says so, or whose relax pinched the surface off (``collapsed`` edges: a
    film or bridge breaking past a fold, then converged on as pieces), with
    the surface put back to the last good step and a warning. ``halvings=k`` first retries
    such a step from the last good one in 2, 4, ... 2**k smaller steps
    (numeric values only); ``Step.substeps`` says how many it took.

    ``predictor``: start each relax from the secant prediction, every vertex
    extrapolated from the last two steps (from the third step on, for numeric
    values and steps at most twice the last, and only while the vertices stay
    the same: after a refinement, remeshing or pop a step starts from the last
    equilibrium). It saves a fifth to a half of the gradient steps on smooth
    continuations. A predicted step that comes out unconverged is relaxed
    again from the last equilibrium's positions before anything else;
    ``Step.predicted`` says whether the prediction was kept.
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
    previous = values[len(done) - 1] if done else _current(ev, what)
    history: list = []      # the equilibria relaxed here (the start may not be one)

    def relax_to(v: Any) -> tuple:
        """Set v and relax, from the prediction when there is one."""
        set_value(ev, v)
        predicted = predictor and _predict(ev, history, v)
        result = relax_fn(ev)
        if predicted and _unconverged(result) and _same_vertices(ev, history[-1]):
            # back to the last equilibrium's positions (cheaper than a snapshot;
            # a relax that refined or popped is left to stop_on)
            ev.vertices = history[-1][1]
            result, predicted = relax_fn(ev), False
        if predictor and not _failed(result, stop_on):
            history[:] = (history + [_snapshot(ev, v)])[-2:]
        return result, predicted

    for index in range(len(done), len(values)):
        value = values[index]
        good = ev.save() if stop_on is not None else None
        good_history = list(history)
        result, predicted = relax_to(value)
        substeps = 1
        numeric = _numeric(value, previous)
        k = 0
        while _failed(result, stop_on) and k < halvings and numeric:
            k += 1
            ev.restore(good)
            history[:] = good_history
            n = 2**k
            for j in range(1, n + 1):
                result, predicted = relax_to(previous + (value - previous)*j/n)
                if _failed(result, stop_on):
                    break
            substeps = n
        if _failed(result, stop_on):
            ev.restore(good)
            why = ("unconverged" if getattr(result, "converged", True) is False else
                   "pinched off" if getattr(result, "collapsed", 0) else "unstable")
            at = np.asarray(value).tolist()     # plain numbers, not np.float64(...)
            warnings.warn(f"continuation() stopped before {at}: relax() came out {why} "
                          "there; the surface is back at the last good step",
                          UserWarning, stacklevel=2)
            return
        previous = value
        bodies = ev.bodies()
        step = Step(index, value, ev.energy, np.array(bodies.volume),
                    np.array(bodies.pressure), result, substeps=substeps,
                    predicted=predicted)
        done.append(step)
        yield step
        if checkpoint is not None:
            _save(checkpoint, ev, done)
