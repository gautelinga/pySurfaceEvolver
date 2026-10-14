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

:func:`arclength` follows a branch of equilibria through a fold (where
:func:`continuation` stops) and on along the unstable branch past it.
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union

import numpy as np

from . import _core

__all__ = ["Step", "arclength", "continuation", "load_steps"]


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


# ---- through folds: pseudo-arclength continuation --------------------------------

def _hessian(ev) -> None:
    """A plain Newton step (Evolver's ``hessian``), without the warning that
    the Hessian isn't positive definite: on an unstable branch it isn't."""
    from ._errors import EvolverWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "WARNING 1825", EvolverWarning)
        ev.command("hessian")


def _newton_target(ev, put: Callable[[Any], None],
                   value: float) -> "tuple[np.ndarray, np.ndarray]":
    """Where one plain Newton step at ``value`` takes the vertices and the
    Lagrange multipliers (undone, both: the step's Hessian uses the
    multipliers, so two steps from the same point need the same ones)."""
    put(value)
    start = ev._coords()
    _hessian(ev)
    x, multipliers = ev.vertices, _core.multipliers()
    ev._put_back(start)
    return x, multipliers


def _normal_part(ev, dx: np.ndarray) -> np.ndarray:
    """The part of a displacement along the surface's normals (area-weighted
    facet normals at each vertex): the shape's change, without the vertices'
    sliding along the surface (the mesh tidying between points)."""
    m = ev.mesh()
    if m.facets is None or m.vertices.shape[1] != 3 or m.order != 1:
        return dx
    x, f = m.vertices, m.facets
    cross = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    normals = np.zeros_like(x)
    for k in range(3):
        np.add.at(normals, f[:, k], cross)
    length = np.linalg.norm(normals, axis=1)
    unit = np.divide(normals, length[:, None], out=np.zeros_like(normals),
                     where=length[:, None] > 0)
    return unit*(dx*unit).sum(axis=1)[:, None]


def _arc_norm2(dx: np.ndarray, dvalue: float) -> float:
    """arclength()'s measure of a step: the value's change and the vertices'
    root mean square displacement."""
    return dvalue*dvalue + float((dx*dx).sum(axis=1).mean())


def _arc_correct(ev, put, x_pred: np.ndarray, v_pred: float, tx: np.ndarray, tv: float,
                 tol: float, max_iter: int) -> "tuple[bool, int]":
    """Newton's method on the equilibrium and the plane through the
    prediction normal to the tangent: from one Newton step at the value and
    one at a nearby value, the step that is both (Keller's bordering, with
    Evolver's factorization), for the vertices and the Lagrange multipliers
    together. Whether it converged, and the iterations."""
    ev.vertices = x_pred
    value = v_pred
    n = len(x_pred)
    for it in range(1, max_iter + 1):
        x = ev.vertices
        delta = 1e-6*max(1.0, abs(value))
        n0, m0 = _newton_target(ev, put, value)
        n1, m1 = _newton_target(ev, put, value + delta)
        dn = (n1 - n0)/delta
        # the new point n0 + dv*dn on the plane: tx.(x - x_pred)/n + tv*(v - v_pred) = 0
        gap = float((tx*(n0 - x_pred)).sum())/n + tv*(value - v_pred)
        slope = float((tx*dn).sum())/n + tv
        dv = -gap/slope
        x_new = n0 + dv*dn
        value += dv
        put(value)
        ev.vertices = x_new
        _core.set_multipliers(m0 + dv*(m1 - m0)/delta)
        if _arc_norm2(x_new - x, dv) < tol*tol and ev.residual() < tol:
            return True, it
    return False, max_iter


def _arc_settle(ev, put, value: float, tol: float, tidy: bool,
                tangent: "Optional[tuple[np.ndarray, float]]" = None) -> "tuple[bool, int, float]":
    """After a converged step: the vertices spread along the surface (Newton
    moves them along its normal only) and edges flipped, keeping the same
    vertices; then back to the branch, by the corrector along ``tangent``
    (near a fold the equilibrium at a fixed value is ill-conditioned) or by
    plain Newton steps without one. Whether it is there, the negative
    eigenvalues (with the constraints held), the residual."""
    if tidy:
        for _ in range(2):
            ev.vertex_average()
            ev.equiangulate()
    if tangent is not None:
        ok, _ = _arc_correct(ev, put, ev.vertices, value, tangent[0], tangent[1], tol, 12)
        residual = ev.residual()
    else:
        ok, residual = False, ev.residual()
        for _ in range(8):
            _hessian(ev)
            residual = ev.residual()
            if residual < tol:
                ok = True
                break
    # counted with the constraints held (a Newton step's own count,
    # eigen_neg, isn't: it counts as if the volume were free)
    return ok, (ev.eigen_counts(0.0).negative if ok else 0), residual


def _back_to(ev, put, snapshot, value: float, tol: float) -> bool:
    """Back at an earlier point after tidying has flipped edges, which
    _put_back refuses: that point's vertices and multipliers on the mesh as it
    is now (flips keep the vertices), then Newton steps at its value. Whether
    that converged."""
    if not ev._put_back(snapshot):
        if len(snapshot.coords) != len(ev.vertices):
            return False
        ev.vertices = snapshot.coords
        if snapshot.multipliers is not None \
                and len(snapshot.multipliers) == _core.multiplier_count():
            _core.set_multipliers(snapshot.multipliers)
    put(value)
    return _arc_settle(ev, put, value, tol, False)[0]


def arclength(ev, what: Any, ds: float, steps: int, *,
              bounds: Optional["tuple[float, float]"] = None, tol: float = 1e-8,
              tidy: bool = True, halvings: int = 3) -> Iterator[Step]:
    """Follow a branch of equilibria through folds: pseudo-arclength
    continuation in ``what`` (a parameter or variable name, ``ev.body(i)``'s
    target volume or ``ev.quantity(name)``'s target).

    Where :func:`continuation` steps the value and stops at a fold (no
    equilibrium past it), this steps along the branch itself: each point is
    a distance ``ds`` from the last, measured as the change of the value and
    the root mean square displacement of the vertices together. So it turns
    at a fold and goes on along the other branch, typically unstable (a
    catenoid past its widest span, a drop or bridge past its largest
    volume). Yields a :class:`Step` per point, starting with the first two;
    ``result.negative_modes`` counts the Hessian's negative eigenvalues there
    with the constraints (volumes, ...) held, from
    :meth:`Evolver.eigen_counts` (0 on a stable branch, 1 past a simple
    fold; a symmetry's zero modes can come out slightly negative, which
    :meth:`Evolver.stability` sorts out, at the cost of Evolver's ``ritz``).

    It starts by relaxing the surface at the current value, then takes Newton
    steps to the value plus ``ds`` (the sign sets the direction). From there
    each point is a prediction along the secant of the last two, corrected by
    Newton's method on the equilibrium together with the condition of being
    ``ds`` along the branch (each iteration: two Newton steps of Evolver's).
    No gradient steps: those would leave an unstable branch. The vertices
    stay the same: no refinement or remeshing (refine before), but between
    points (``tidy=True``) they are spread along the surface and edges
    flipped, which Newton steps alone don't do.

    Stops after ``steps`` points, at the first point outside ``bounds``
    (``(low, high)`` on the value; the surface is put back at the last one
    inside), or, with a warning, when a step (the first one too) doesn't
    converge even at ``ds/2**halvings`` or doesn't settle again after tidying
    (put back at the last point either way) or the facets degenerate
    (below 1 degree: refine and start again from there). ``tol`` bounds the
    residual of each point, as in :meth:`Evolver.relax`.
    """
    if ds == 0 or steps < 0 or halvings < 0:
        raise ValueError("arclength() needs ds != 0, steps >= 0 and halvings >= 0")
    set_value = _setter(what)
    value = _current(ev, what)
    if not _numeric(value):
        raise TypeError("arclength() follows a value it can read: a parameter or variable "
                        "name, ev.body(i) or ev.quantity(name)")

    def put(v: float) -> None:
        set_value(ev, v)
        ev.command("recalc")        # boundaries follow a parameter (assigning one doesn't)

    def step(index: int, v: float, iterations: int, negative: int, residual: float) -> Step:
        bodies = ev.bodies()
        none = np.empty(0)
        result = IterationResult(none, none, none, "", converged=True, newton_steps=iterations,
                                 residual=residual, negative_modes=negative,
                                 stable=negative == 0)
        return Step(index, v, ev.energy, np.array(bodies.volume), np.array(bodies.pressure),
                    result)

    def outside(v: float) -> bool:
        return bounds is not None and not (min(bounds) <= v <= max(bounds))

    from ._results import IterationResult
    r = ev.relax(stability=False, tol=tol)
    if not r.converged:
        warnings.warn("arclength(): the surface didn't relax at the start", UserWarning,
                      stacklevel=2)
        return
    ok, negative, residual = _arc_settle(ev, put, float(value), tol, False)
    points = [(float(value), ev.vertices)]
    yield step(0, float(value), 0, negative, residual)
    if steps < 2:
        return
    # the first step in the value alone, halved like the others when Newton
    # doesn't get there
    # (back by the coordinates, or by a copy once tidying has flipped edges)
    start, copy, first = ev._coords(), ev.save(), ds
    for _ in range(halvings + 1):
        put(value + first)
        kept = ev.newton(20, tol=tol)
        ok = False
        if kept.converged:
            ok, negative, residual = _arc_settle(ev, put, float(value + first), tol, tidy)
        if ok:
            break
        if not ev._put_back(start):
            ev.restore(copy)
        put(value)
        first /= 2
    if not ok:
        warnings.warn(f"arclength(): no equilibrium found at {value + 2*first:.6g} by Newton "
                      f"steps (even with steps of {abs(2*first):.3g}); the surface is back at "
                      "the start", UserWarning, stacklevel=2)
        return
    points.append((float(value + first), ev.vertices))
    yield step(1, float(value + first), kept.newton_steps, negative, residual)
    h = abs(first)              # (and back up to ds below)
    for index in range(2, steps):
        (v0, x0), (v1, x1) = points[-2], points[-1]
        if len(x0) != len(x1) or len(x1) != len(ev.vertices):
            raise RuntimeError("arclength(): the vertices changed between points (it "
                               "needs the same mesh throughout)")
        # the step's length and the plane it ends on from the shape's change
        # (along the normals: the tidying's sliding isn't a step; counting it
        # turned a bridge back at its fold); the prediction along the whole
        # secant (the vertices keep spreading as the surface stretches)
        dx = _normal_part(ev, x1 - x0)
        norm = np.sqrt(_arc_norm2(dx, v1 - v0))
        tx, tv = dx/norm, (v1 - v0)/norm
        sx = (x1 - x0)/norm
        last = ev._coords()
        for k in range(halvings + 1):
            ok, iterations = _arc_correct(ev, put, x1 + h*sx, v1 + h*tv, tx, tv, tol, 12)
            if ok:
                break
            ev._put_back(last)
            put(v1)
            h /= 2
        if not ok:
            warnings.warn(f"arclength(): no equilibrium found past {v1:.6g} (even with "
                          f"steps of {h*2:.3g}); the surface is back at the last point",
                          UserWarning, stacklevel=2)
            return
        v = float(_current(ev, what))
        if outside(v):
            ev._put_back(last)
            put(v1)
            return
        ok, negative, residual = _arc_settle(ev, put, v, tol, tidy, (tx, tv))
        v = float(_current(ev, what))       # (settling back can move it a little)
        if not ok:
            where = ("the surface is back at the last point" if _back_to(ev, put, last, v1, tol)
                     else "the surface is left there, out of equilibrium")
            warnings.warn(f"arclength(): the surface didn't settle again at {v:.6g} after "
                          f"tidying its mesh (too coarse there? refine); stopping, and {where}",
                          UserWarning, stacklevel=2)
            return
        points = [points[-1], (v, ev.vertices)]
        yield step(index, v, iterations, negative, residual)
        if ev.mesh_quality().angle_min < 1.0:
            warnings.warn(f"arclength(): facets below 1 degree at {v:.6g}: refine and start "
                          "again from here", UserWarning, stacklevel=2)
            return
        h = min(abs(ds), 2*h)       # back up to ds after halving
