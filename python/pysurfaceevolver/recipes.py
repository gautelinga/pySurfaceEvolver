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

import functools
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union

import numpy as np

from . import _core
from ._relax import _MERGE_UNDER, _SPLIT_ABOVE

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
    events: list = field(default_factory=list)  # Evolver.events() there (see events=)
    fell: bool = False         # reached by a descent after a change or a fall (topology=)
    changes: list = field(default_factory=list)  # the events whose changes were made


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
                 s.predicted, s.events, s.fell, s.changes) for s in steps]
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


def _event_factor(events: Union[bool, float, None]) -> Optional[float]:
    """events= as a factor for Evolver.events(), or None (off)."""
    if events is None or events is False:
        return None
    if events is True:
        from ._analysis import EVENT_FACTOR
        return EVENT_FACTOR
    factor = float(events)
    if not factor > 0:
        raise ValueError("events must be True, False or a factor > 0")
    return factor


def _through(found: list) -> list:
    """The events of a surface gone through a solid."""
    return [e for e in found if e.kind == "solid" and e.gap < 0]


def _coming(found: list, before: list) -> list:
    """The events at this point that were at the last one too (the same kind,
    on the same constraint, within a few edge lengths) and are closer now, by
    a hundredth of an edge at least: a neck that widens after a merge, or a
    gap that stays as it is while its flag comes and goes (a vertex next to
    a contact line, about as close as the line's wedge), isn't coming."""
    out = []
    for e in found:
        same = [b.gap - 0.01*max(b.scale, e.scale) for b in before
                if b.kind == e.kind and b.constraint == e.constraint
                and np.linalg.norm(np.asarray(b.where) - np.asarray(e.where))
                < 4*max(b.scale, e.scale)]
        if same and e.gap < min(same):
            out.append(e)
    return out


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
                 predictor: bool = True,
                 events: Union[bool, float] = True) -> Iterator[Step]:
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

    ``events`` (on by default; a number sets :meth:`Evolver.events`'s
    ``factor``, False turns it off): each step's :meth:`Evolver.events` go
    on its ``Step.events``, and the run ends, with a warning, after the
    first step with an event that is coming (there at the last step too, and
    closer now): a neck about to pinch off, the surface about to touch itself, a
    solid or contact lines about to meet. A step whose surface has gone
    through a solid ends the run before it, the surface back at the last
    good step.
    """
    if stop_on not in (None, "unconverged", "unstable"):
        raise ValueError("stop_on must be None, 'unconverged' or 'unstable'")
    if halvings < 0:
        raise ValueError("halvings must be non-negative")
    if halvings and stop_on is None:
        raise ValueError("halvings retries the steps stop_on rejects: give stop_on too")
    factor = _event_factor(events)
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

    before = ev.events(factor) if factor is not None else []
    for index in range(len(done), len(values)):
        value = values[index]
        good = ev.save() if stop_on is not None or factor is not None else None
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
        found = [] if factor is None else \
            result.events if factor == _event_factor(True) and \
            isinstance(getattr(result, "events", None), list) else ev.events(factor)
        if _through(found):
            ev.restore(good)
            at = np.asarray(value).tolist()
            warnings.warn(f"continuation() stopped before {at}: {_through(found)[0]}; the "
                          "surface is back at the last good step", UserWarning, stacklevel=2)
            return
        previous = value
        bodies = ev.bodies()
        step = Step(index, value, ev.energy, np.array(bodies.volume),
                    np.array(bodies.pressure), result, substeps=substeps,
                    predicted=predicted, events=found)
        done.append(step)
        coming = _coming(found, before)
        before = found
        yield step
        if checkpoint is not None:
            _save(checkpoint, ev, done)
        if coming:
            at = np.asarray(value).tolist()
            warnings.warn(f"continuation() stopped at {at}: {coming[0]} (see Step.events)",
                          UserWarning, stacklevel=2)
            return


# ---- through folds: pseudo-arclength continuation --------------------------------

def _hessian(ev) -> None:
    """A plain Newton step (Evolver's ``hessian``), without the warning that
    the Hessian isn't positive definite: on an unstable branch it isn't.
    (Evolver needs pressures from a gradient step first: one is taken if
    none was, as for a flat meniscus that relax() found in equilibrium
    without iterating. A body none of whose vertices can move, held all
    round by solids, as a sliver a pack's patches left in a corner: its
    volume's target let go, which the solids hold anyway, and again.)"""
    from ._errors import EvolverError, EvolverWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "WARNING 1825", EvolverWarning)
        for _ in range(4):
            try:
                ev.command("hessian")
                return
            except EvolverError as e:
                held = re.search(r"(?:body_|[Bb]ody )(\d+)(?:_vol)? is not connected to any "
                                 r"mobile", str(e))
                if e.errnum == 1606:        # "Pressures invalid": not iterated yet
                    ev.command("g 1")
                elif e.errnum == 3362 and held:
                    ev.body(int(held.group(1))).target = None
                else:
                    raise
        ev.command("hessian")


def _newton_target(ev, put: Callable[[Any], None],
                   value: float) -> "tuple[np.ndarray, np.ndarray]":
    """Where one plain Newton step at ``value`` takes the vertices and the
    Lagrange multipliers (undone, both: the step's Hessian uses the
    multipliers, so two steps from the same point need the same ones).
    Plain: without the mesh following a line the step moves far, which
    isn't linear in the step, and the corrector's two steps must be."""
    put(value)
    start = ev._coords()
    was = _core.set_follow(False)
    try:
        _hessian(ev)
    finally:
        _core.set_follow(was)
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
                 tol: float, max_iter: int,
                 reach: Optional[float] = None) -> "tuple[bool, int]":
    """Newton's method on the equilibrium and the plane through the
    prediction normal to the tangent: from one Newton step at the value and
    one at a nearby value, the step that is both (Keller's bordering, with
    Evolver's factorization), for the vertices and the Lagrange multipliers
    together. Whether it converged, and the iterations. With ``reach``, it
    fails as soon as the point is farther than that from the prediction: on
    the plane, but somewhere else (a pore's liquid at a volume 0.7 below a
    step of 0.01, where Newton's basin was a fifth of the step)."""
    ev.vertices = x_pred
    value = v_pred
    n = len(x_pred)
    last = float("inf")
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
        if reach is not None and _arc_norm2(x_new - x_pred, value - v_pred) > reach*reach:
            return False, it
        # converged: in equilibrium, with steps below tol or no longer
        # shrinking, which is round-off (Lagrange 2 at 98k facets: the
        # vertices slide 2e-8 to 1e-7 a step with the residual at 1e-9)
        step = _arc_norm2(x_new - x, dv)
        if (step < tol*tol or step > 0.25*last) and ev.residual() < tol:
            return True, it
        last = step
    return False, max_iter


def _arc_settle(ev, put, value: float, tol: float, tidy: bool,
                tangent: "Optional[tuple[np.ndarray, float]]" = None,
                reach: Optional[float] = None) -> "tuple[bool, int, float]":
    """After a converged step: the vertices spread along the surface (Newton
    moves them along its normal only) and edges flipped, keeping the same
    vertices; then back to the branch, by the corrector along ``tangent``
    (near a fold the equilibrium at a fixed value is ill-conditioned, and
    ``reach`` bounds how far it may go) or by plain Newton steps without one.
    If that fails, the spreading is undone (the point's vertices on the
    flipped mesh) and it tries again; failing that, the point counts as not
    there (the caller goes back and halves the step). Near a fold or where a
    mode goes soft, Newton's basin can be smaller than what tidying moves (a
    pore's liquid near pinching, moved 2e-4 RMS), and a coarse mesh can hold
    an equilibrium the shape doesn't have, which tidying undoes (a catenoid
    past its fold). Whether it is there, the negative eigenvalues (with the
    constraints held), the residual."""
    ok, residual = False, float("inf")
    flip, spread = ev._tidy_kinds() if tidy else (False, False)
    # spreading only in the linear model: in the quadratic and Lagrange models
    # the energy depends on where the nodes sit along the surface, which
    # Newton steps (along the normals) don't move back, so each spreading left
    # a slightly worse equilibrium (a bridge between disks ended 2.3% short of
    # its fold, the quadratic model's 0.3% without it)
    spread = spread and ev.model == "linear"
    if flip or spread:
        start = ev._coords()
        for _ in range(2):
            if spread:
                ev.vertex_average()
            if flip:
                ev.equiangulate()
        ok, residual = _arc_back(ev, put, value, tol, tangent, reach)
        if not ok:
            _put_coords(ev, start)
            put(value)
            ok, residual = _arc_back(ev, put, value, tol, tangent, reach)
    else:
        ok, residual = _arc_back(ev, put, value, tol, tangent, reach)
    # counted with the constraints held (a Newton step's own count,
    # eigen_neg, isn't: it counts as if the volume were free)
    return ok, (ev.eigen_counts(0.0).negative if ok else 0), residual


def _arc_back(ev, put, value: float, tol: float,
              tangent: "Optional[tuple[np.ndarray, float]]",
              reach: Optional[float]) -> "tuple[bool, float]":
    """Back to the branch from near it (see _arc_settle): whether it got
    there, and the residual."""
    if tangent is not None:
        ok, _ = _arc_correct(ev, put, ev.vertices, value, tangent[0], tangent[1], tol, 12,
                             reach)
        return ok, ev.residual()
    residual = ev.residual()
    for k in range(12):
        _hessian(ev)
        residual = ev.residual()
        if residual < tol:
            return True, residual
        # (a few more where it is nearly there: a free drop's rigid motions,
        # nearly singular, slow the steps down; after tidying a merged drop
        # ended 8 steps at 1.2e-8)
        if k == 7 and residual > 10*tol:
            break
    return False, residual


def _put_coords(ev, snapshot) -> bool:
    """A point's vertices and multipliers (from _coords) on the mesh as it is
    now, which _put_back refuses after edges were flipped (flips keep the
    vertices). Whether the vertices still match."""
    if ev._put_back(snapshot):
        return True
    if len(snapshot.coords) != len(ev.vertices):
        return False
    ev.vertices = snapshot.coords
    if snapshot.multipliers is not None \
            and len(snapshot.multipliers) == _core.multiplier_count():
        _core.set_multipliers(snapshot.multipliers)
    return True


def _carry(ids: np.ndarray, values: np.ndarray, mesh) -> np.ndarray:
    """A field per vertex (``values``, rows aligned with Evolver vertex ``ids``) on a
    remeshed surface: by id where a vertex was kept, and where it is new (an
    edge split) the mean over its neighbours that have one, layer by layer
    inward."""
    new_ids = mesh.vertex_ids
    out = np.zeros((len(new_ids),) + values.shape[1:])
    order = np.argsort(ids)
    at = np.minimum(np.searchsorted(ids, new_ids, sorter=order), len(ids) - 1)
    known = ids[order[at]] == new_ids
    out[known] = values[order[at[known]]]
    a, b = mesh.edges[:, 0], mesh.edges[:, 1]
    while not known.all():
        sums, count = np.zeros_like(out), np.zeros(len(out))
        for u, w in ((a, b), (b, a)):
            take = known[w] & ~known[u]
            np.add.at(sums, u[take], out[w[take]])
            np.add.at(count, u[take], 1)
        newly = count > 0
        if not newly.any():
            break                   # (a piece without a kept vertex: left at 0)
        out[newly] = sums[newly]/count[newly].reshape((-1,) + (1,)*(out.ndim - 1))
        known |= newly
    return out


def _remesher(ev, remesh: Any) -> Optional[Callable[[Callable[[], None]], bool]]:
    """arclength()'s ``remesh``, as a call that remeshes when the edges have
    left their range (calling ``ready()`` first) and says whether it changed
    the mesh."""
    if remesh is None:
        return None
    if callable(remesh):
        def by_function(ready: Callable[[], None]) -> bool:
            ready()
            before = ev.mesh()
            remesh(ev)
            after = ev.mesh()
            return not (np.array_equal(before.vertex_ids, after.vertex_ids)
                        and np.array_equal(before.facets, after.facets)
                        and np.array_equal(before.vertices, after.vertices))
        return by_function
    kwargs = dict(remesh) if isinstance(remesh, dict) else {"target": float(remesh)}
    if ev.model != "linear":
        raise ValueError("arclength(remesh=...) needs the linear model (see Evolver.remesh)")
    target = kwargs.get("target")
    longest = kwargs.get("max_edge", _SPLIT_ABOVE*target if target is not None else None)
    shortest = kwargs.get("min_edge", _MERGE_UNDER*target if target is not None else None)

    def by_length(ready: Callable[[], None]) -> bool:
        m = ev.mesh()
        length = m.edge_lengths()
        due = longest is not None and bool((length > longest).any())
        if shortest is not None:
            inner = m.edge_valence() != 1 if kwargs.get("keep_boundary", True) else True
            due = due or bool(((length < shortest) & inner).any())
        if not due:
            return False
        ready()
        return any(ev.remesh(**kwargs).values())
    return by_length


_QUIET_FALLS = 20       # arclength(topology=True): falls in a row without a change, at most

def arclength(ev, what: Any, ds: float, steps: int, *,
              bounds: Optional["tuple[float, float]"] = None, tol: float = 1e-8,
              tidy: bool = True, halvings: int = 6, along: Optional[np.ndarray] = None,
              remesh: Union[None, float, dict, Callable[[Any], Any]] = None,
              events: Union[bool, float] = True, topology: bool = False) -> Iterator[Step]:
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
    each point is a prediction along the secant of the last two (where that
    fails, along the branch's tangent at the last one), corrected by
    Newton's method on the equilibrium together with the condition of being
    ``ds`` along the branch (each iteration: two Newton steps of Evolver's).
    No gradient steps: those would leave an unstable branch. Between points
    (``tidy=True``) the vertices are spread along the surface and edges
    flipped, which Newton steps alone don't do (as :meth:`Evolver.relax`
    tidies: no flips in the Lagrange model or for strings; the vertices are
    spread in the linear model only, since in the others the energy depends
    on where the nodes sit along the surface and Newton steps don't move
    them back).

    ``along`` (a displacement per vertex, such as ``ev.modes(1).vectors[0]``)
    starts on another branch: the first step goes a distance ``ds`` along it
    at the starting value instead of in the value, onto a branch that crosses
    this one there, such as a less symmetric one past a symmetric
    bifurcation (start just past it, where the mode's eigenvalue has turned
    negative: from the point a first run stopped at). The sign of ``ds``
    picks the direction along it; from the second step on the run follows the
    new branch, either way in the value.

    ``remesh`` keeps the edges' lengths in range on a branch that stretches
    or shrinks the surface: ``h`` (:meth:`Evolver.remesh`'s ``target``) or a
    dict of its arguments, called between points when an edge has left the
    range they set (``target=h``: 0.5h to 1.6h; edges on the boundary are
    never deleted), or a function of the surface, called between points. The
    remeshed surface is brought back to the branch at the same point, and the
    last step carried over to its new vertices (a split edge's midpoint takes
    the mean of its neighbours'); if it doesn't get back, the surface is put
    back as it was and the run goes on. Without it the vertices stay the same
    (refine before). Next to a contact line that shrinks, let it merge
    contact-line edges too (``remesh=dict(target=h, keep_boundary=False)``).

    Stops after ``steps`` points, at the first point outside ``bounds``
    (``(low, high)`` on the value; the surface is put back at the last one
    inside), or, with a warning, when a step (the first one too) doesn't
    converge, or doesn't settle again after tidying, even at ``ds/2**halvings``
    (put back at the last point) or the facets degenerate (below 1 degree:
    refine and start again from there). A step is also cut when its point
    lands farther from the prediction than the step itself: on the branch's
    plane, but on some other branch or none. ``tol``
    bounds the residual of each point, as in :meth:`Evolver.relax`.

    ``events`` (on by default; a number sets :meth:`Evolver.events`'s
    ``factor``, False turns it off): each point's :meth:`Evolver.events` go
    on its ``Step.events``, and the run ends, with a warning, at the first
    point with an event that is coming (there at the last point too, and
    closer now), the surface there: a neck about to pinch off, the surface about
    to touch itself, a solid or contact lines about to meet. A point whose
    surface has gone through a solid counts as a failed step (halved).

    ``topology=True`` goes on through changes of topology, as the surface
    would under the run's control (a syringe pump draining liquid slowly).
    Where an event is coming, the change is made instead of stopping: a neck
    cut (:meth:`Evolver.pinch_off`), a touch glued (:meth:`Evolver.merge`),
    a solid wetted or dried or contact lines joined (:meth:`Evolver.contact`).
    Where the branch loses its stability (a point with more negative modes
    than the last) the surface falls from there (at a fold, where the branch
    turned back between the points along the unstable mode, from the value
    one step past the last stable point, past the fold), by steps along the
    unstable mode (each moving a vertex by at most 0.3 of the median edge,
    the way the gradient goes along it) with five gradient steps after each.
    After a change or in a fall, the surface goes down to a stable state,
    making the changes that come on the way (after a change by
    :meth:`Evolver.relax` in short runs, the edges kept in range; next to a
    contact line, vertices the gradient steps take through its solid are put
    back out). Bodies cut off keep their volumes; a body the run steps keeps
    its number for the piece holding most of the liquid, and for a merged
    body, whose target is the bodies' added. The state each such descent ends
    in is a :class:`Step` with ``fell`` set and the events whose changes were
    made in ``changes``; the run goes on from there as from a new start, its
    edges kept in range (``remesh`` if given, else within 0.4 and 4/3 of
    their targets, the median edge or finer near gaps, contact lines too;
    within 0.4 and 2 of them right after a change). A branch that stops short after a change (a
    step that doesn't settle) goes on the same way from a relaxed state there.
    The run ends, with a warning, where a change or a descent fails, and
    after three falls in a row that change neither the topology nor the value
    (or twenty without a change of topology). How liquid
    splits at a pinch depends on the way down, here a guided descent of the
    energy, not the liquid's flow.
    """
    if not topology:
        yield from _branch(ev, what, ds, steps, bounds=bounds, tol=tol, tidy=tidy,
                           halvings=halvings, along=along, remesh=remesh, events=events)
        return
    from ._analysis import EVENT_FACTOR
    factor = _event_factor(events) or EVENT_FACTOR
    set_value = _setter(what)
    from ._views import BodyView
    from ._surgery import _resting, move_poles, orient_lines
    keep = what.number if isinstance(what, BodyView) else None
    index, quiet, stuck, start = 0, 0, 0, True
    run_remesh = remesh
    last: List[Step] = []
    moves: List[float] = []             # (the value's steps along the branch)
    while index < steps:
        last = last[-1:]                # (the state a change ended in: events coming from it)
        seen: List[np.ndarray] = []
        lost, event, moved = False, None, False
        if not start and callable(run_remesh):
            # (the edges in range from where a change or fall ended, not only
            # from a branch's third point on: falling every few steps, a
            # pack's surface went on with the edges a descent leaves, up to
            # twice their targets, 0.41 for 0.2; the branch relaxes it first)
            run_remesh(ev)
        for step in _branch(ev, what, ds, steps - index + (0 if start else 1), bounds=bounds,
                            tol=tol, tidy=tidy, halvings=halvings,
                            along=along if start else None, remesh=run_remesh, events=events,
                            report=start):
            if not start and step.index == 0:
                continue                # where the fall ended, yielded already
            moved = moved or step.index > 0     # (a step past where the branch began)
            if last and not step.fell:
                moves.append(step.value - last[-1].value)
            step.index = index
            index += 1
            yield step
            orient_lines(ev)
            move_poles(ev)      # (sphere(axis=): its pole kept off its contact lines)
            last, seen = (last + [step])[-3:], (seen + [ev.vertices])[-2:]
            if len(last) > 1 and _negative(last[-1]) > _negative(last[-2]):
                lost = True
                break
            coming = _coming(step.events, last[-2].events) if len(last) > 1 else []
            coming += [e for e in step.events if _resting(e) and not any(e is c for c in coming)]
            if coming:
                event = coming[0]       # (the run would stop here: made instead)
                break
        if index >= steps:
            return
        if moved:
            quiet = stuck = 0           # (falls without a change between steps aren't a stall)
        stalled = not (lost or event is not None)
        if stalled and not start and not moved:
            # not a step from where a change or fall ended: the change
            # nearest to coming there is made (the descent from there alone
            # came back to it, a pack's film over a bead beside its contact
            # line too thick to be resting, and the run stalled again and
            # again)
            event = _nearest(ev.events(factor))
            stalled = event is None
        if stalled:
            value = _current(ev, what)
            # (the first branch ends the run only where it took no step: one
            # that went on to a fold goes on past it as a later branch does,
            # a pack's run resumed short of its fold stopped there)
            if (start and not moved) or (bounds is not None
                                         and not (min(bounds) <= value <= max(bounds))):
                return
        start = False
        if remesh is None:
            # after a change the mesh is uneven where it was made (a joined
            # contact line, a merged neck): the run goes on keeping its edges
            # in range, as the change's relaxing did
            from ._surgery import _keep_edges, _edge_scale, _ON_THE_WAY
            h0 = _edge_scale(ev)
            run_remesh = functools.partial(_keep_edges, h0=h0, hi=_ON_THE_WAY)
        if stalled:
            # the branch stopped short (a step that didn't settle after
            # tidying: a pore's contact line receding over dry patches): from
            # a relaxed state there, as after a change
            from ._surgery import descend
            if quiet and moves:
                # (and it came back to the same state last time: past a fold
                # where the value has its least, as a pack's liquid ran out
                # of room above the floor, the branch steps on less and
                # less; the value set past it, a typical step on, twice
                # that the next time, and the liquid falls from there)
                typical = float(np.median(np.abs(moves)))
                way = float(np.sign(sum(moves))) or -1.0
                # (at most eight typical steps, nor a tenth of the value:
                # doubled without end, a pack's liquid was set to -56)
                push = min(typical*2**min(quiet - 1, 3), 0.1*abs(value) or typical)
                set_value(ev, value + way*push)
                ev.command("recalc")
            result, cut, why = descend(ev, factor, tol, falling=False, keep=keep)
        elif event is not None:
            result, cut, why = _change(ev, event, factor, tol, keep)
        elif len(last) == 3 and last[-2].value != last[-3].value and _at_fold(ev, *seen):
            # past a fold, the stable state is still there at the unstable
            # point's value (a catenoid's at 1.3241, its fold at 1.3255): the
            # fall at the value one step past the last stable point, the way
            # the value went there
            stable, previous = last[-2].value, last[-3].value
            way = np.sign(stable - previous)
            far = max(stable*way, last[-1].value*way)*way
            set_value(ev, far + (stable - previous))
            ev.command("recalc")
            result, cut, why = _fall(ev, factor, tol, keep)
        else:
            result, cut, why = _fall(ev, factor, tol, keep)
        if result is None:
            what_ = (f"{event}" if event is not None else
                     f"relaxing where the branch stopped short ({_current(ev, what):.6g})"
                     if stalled else f"the fall from {last[-1].value:.6g}")
            warnings.warn(f"arclength(): {what_} stopped: {why}", UserWarning, stacklevel=2)
            return
        _smooth(ev, factor, tol)
        bodies = ev.bodies()
        settled = Step(index, _current(ev, what), ev.energy, np.array(bodies.volume),
                       np.array(bodies.pressure), result, events=ev.events(factor),
                       fell=True, changes=cut)
        before = last[-1].value if last else None
        yield settled
        last = [settled]
        index += 1
        quiet = 0 if cut else quiet + 1
        # (stuck: falls without a change that didn't move the value either;
        # falls pushed on past a fold each to a stable state at less liquid
        # are drainage, if the branch can't step from them: a pack's run
        # stopped after three, 72% drained, each draining more)
        moved_on = before is not None and abs(settled.value - before) > 1e-9*max(1.0, abs(before))
        stuck = 0 if (cut or moved_on) else stuck + 1
        if stuck == 3 or quiet == _QUIET_FALLS:
            reason = ("three falls in a row ended without a change of topology or of the "
                      "value" if stuck == 3 else
                      f"{quiet} falls in a row ended without a change of topology")
            warnings.warn(f"arclength(): {reason}: stopping", UserWarning, stacklevel=2)
            return


def _smooth(ev, factor: float, tol: float) -> None:
    """The mesh evened where a change or fall ended: Evolver's vertex
    averaging (it keeps the volumes; the checked smoothing of _edits here
    changed which dry patch a pore opened before its jump, and the run went
    on past the jump's fold onto another branch) and edges flipped by the
    checked edits, Newton steps back to the equilibrium after each, five
    times. (Not remeshed here: remeshing at the end of falls toward the
    median edge coarsened a pinched-off drop, its pressure 2% off the exact
    cap's, and in 18 16-bead runs drained less far, 1 of them to half the
    liquid against 3 without it.)
    Newton's steps from there converged only linearly (a pack's contact line
    beside facets of 7 degrees: x0.7 a step, and the run's first step
    failed), x20 once evened (18 degrees). Undone where Newton doesn't get
    back, the surface went through a solid or through itself."""
    from ._surgery import _crossings, _even_out
    start, crossed = ev._save(), _crossings(ev)
    for _ in range(5):
        ev.vertex_average()
        _even_out(ev, smooth=False)
        ev.newton(3, tol=tol)
    if not ev.newton(20, tol=tol).converged or _through(ev.events(factor)) \
            or _crossings(ev) > crossed:
        ev.restore(start)
    from ._surgery import _checked
    _checked(ev, "evening the mesh")


def _nearest(found: list) -> Any:
    """Of the events, the solid, contact-line or touch one with the smallest
    gap for its edges, or None."""
    return min((e for e in found if e.kind in ("solid", "contact lines", "touch")),
               key=lambda e: e.gap/float(e.scale), default=None)


def _change(ev, event, factor: float, tol: float, keep) -> "tuple[Any, list, str]":
    """Make the change a coming event announces, then down to a stable state
    (as _fall). The cut of a neck settles as pinch_off's does; a merge or a
    contact relaxes from there. A change refused (ValueError: the surface is
    left as it was) is passed over, as the descent passes over one: down
    from here, the changes coming on the way made."""
    from ._errors import EvolverError
    from ._surgery import _crossings, _settle, apply, descend, guard
    crossed, start = _crossings(ev), ev._save()
    try:
        apply(ev, event, keep)
        if _crossings(ev) > crossed:
            ev.restore(start)
            raise ValueError("the change passed the surface through itself")
    except ValueError:
        return descend(ev, factor, tol, falling=False, keep=keep)
    except EvolverError as error:
        return None, [], str(error)
    if event.kind == "neck":
        guard(ev)
        r = _settle(ev, tol)
        if r is None:
            # (down from the cut in the descent's runs instead, as the
            # descent does: a 16-bead pack's pieces, other changes coming)
            result, more, why = descend(ev, factor, tol, falling=False, keep=keep)
            return result, [event] + more, why
        if r.stable is not False:
            return r, [event], ""
        result, more, why = descend(ev, factor, tol, falling=True, keep=keep)
    else:
        result, more, why = descend(ev, factor, tol, falling=False, keep=keep)
    return result, [event] + more, why


def _at_fold(ev, before: np.ndarray, now: np.ndarray) -> bool:
    """Whether the stability went at a fold rather than a branch point: the
    unstable mode along the branch's shape change between the last two points
    (at a fold the branch turns along it; where branches cross, the mode
    leads off the branch, across it). A changed mesh counts as a fold."""
    if before.shape != now.shape:
        return True
    modes = ev.modes(1)
    if not len(modes.values):
        return True
    change = _normal_part(ev, now - before)
    u = modes.vectors[0]
    size = float(np.linalg.norm(change)*np.linalg.norm(u))
    return size == 0 or abs(float((change*u).sum()))/size > 0.5


def _negative(step: Step) -> int:
    return int(getattr(step.result, "negative_modes", 0) or 0)


def _fall(ev, factor: float, tol: float = 1e-8, keep=None) -> "tuple[Any, list, str]":
    """From a point that has lost its stability, down to a stable state at
    the same value, making the changes of topology that come on the way
    (_surgery.descend)."""
    from ._surgery import descend
    return descend(ev, factor, tol, falling=True, keep=keep)


def _branch(ev, what: Any, ds: float, steps: int, *,
            bounds: Optional["tuple[float, float]"] = None, tol: float = 1e-8,
            tidy: bool = True, halvings: int = 6, along: Optional[np.ndarray] = None,
            remesh: Union[None, float, dict, Callable[[Any], Any]] = None,
            events: Union[bool, float] = True, report: bool = True) -> Iterator[Step]:
    """arclength() along one branch (see there), without falling; without
    ``report``, its stops aren't warned of (the caller goes on from them)."""
    def say(message: str) -> None:
        if report:
            warnings.warn(message, UserWarning, stacklevel=4)

    if ds == 0 or steps < 0 or halvings < 0:
        raise ValueError("arclength() needs ds != 0, steps >= 0 and halvings >= 0")
    set_value = _setter(what)
    value = _current(ev, what)
    if not _numeric(value):
        raise TypeError("arclength() follows a value it can read: a parameter or variable "
                        "name, ev.body(i) or ev.quantity(name)")
    if along is not None:
        along = np.array(along, dtype=float)
        rms = float(np.sqrt((along*along).sum(axis=1).mean())) if along.ndim == 2 else 0.0
        if along.shape != ev.vertices.shape or not rms > 0:
            raise ValueError(f"arclength(along=...) needs a nonzero displacement per vertex, "
                             f"shape {ev.vertices.shape}")
        along /= rms
    remesh_now = _remesher(ev, remesh)
    factor = _event_factor(events)

    def look() -> list:
        return ev.events(factor) if factor is not None else []

    def crossings() -> int:
        from ._surgery import _crossings
        return _crossings(ev) if factor is not None else 0

    def passed(found: list, crossed: int) -> "Any":
        """What the step went through: a solid (its event), itself (the
        surface's crossings of itself up from the last point's: a 16-bead
        pack's step put two sheets through each other, 34 crossings, before
        their touch was seen, and the branch went on crossed three steps),
        else None."""
        through = _through(found)
        if through:
            return through[0]
        if factor is not None and crossings() > crossed:
            return "the surface through itself"
        return None

    def put(v: float) -> None:
        set_value(ev, v)
        ev.command("recalc")        # boundaries follow a parameter (assigning one doesn't)

    def step(index: int, v: float, iterations: int, negative: int, residual: float,
             found: list) -> Step:
        bodies = ev.bodies()
        none = np.empty(0)
        result = IterationResult(none, none, none, "", converged=True, newton_steps=iterations,
                                 residual=residual, negative_modes=negative,
                                 stable=negative == 0)
        return Step(index, v, ev.energy, np.array(bodies.volume), np.array(bodies.pressure),
                    result, events=found)

    def stop(v: float, coming: list) -> None:
        say(f"arclength(): stopped at {v:.6g}: {coming[0]} (see Step.events)")

    def outside(v: float) -> bool:
        return bounds is not None and not (min(bounds) <= v <= max(bounds))

    from ._results import IterationResult
    r = ev.relax(stability=False, tol=tol)
    if not r.converged:
        say("arclength(): the surface didn't relax at the start")
        return
    ok, negative, residual = _arc_settle(ev, put, float(value), tol, False)
    points = [(float(value), ev.vertices)]
    before = look()
    crossed = crossings()
    yield step(0, float(value), 0, negative, residual, before)
    if steps < 2:
        return
    # the first step in the value alone (or along ``along`` at the value),
    # halved like the others when Newton doesn't get there
    # (back by the coordinates, or by a copy once tidying has flipped edges)
    start, copy, first, x_start = ev._coords(), ev.save(), ds, ev.vertices
    for _ in range(halvings + 1):
        if along is None:
            put(value + first)
            kept = ev.newton(20, tol=tol)
            iterations, ok = kept.newton_steps, False
            if kept.converged:
                ok, negative, residual = _arc_settle(ev, put, float(value + first), tol, tidy)
                found = look() if ok else []
                ok = ok and passed(found, crossed) is None
        else:
            tangent = (np.sign(first)*along, 0.0)
            ok, iterations = _arc_correct(ev, put, x_start + first*along, float(value),
                                          *tangent, tol, 12, abs(first))
            if ok:
                ok, negative, residual = _arc_settle(ev, put, float(_current(ev, what)), tol,
                                                     tidy, tangent, abs(first))
                found = look() if ok else []
                ok = ok and passed(found, crossed) is None
        if ok:
            break
        if not ev._put_back(start):
            ev.restore(copy)
        put(value)
        first /= 2
    if not ok:
        where = f"at {value + 2*first:.6g}" if along is None else "along the given mode"
        say(f"arclength(): no equilibrium found {where} by Newton steps (even with "
                      f"steps of {abs(2*first):.3g}); the surface is back at the start")
        return
    v = float(_current(ev, what))
    points.append((v, ev.vertices))
    crossed = crossings()
    coming = _coming(found, before)
    before = found
    yield step(1, v, iterations, negative, residual, found)
    if coming:
        stop(v, coming)
        return
    h = abs(first)              # (and back up to ds below)
    for index in range(2, steps):
        if remesh_now is not None:
            # (a remeshed surface that doesn't settle is put back as it was:
            # the run goes on, and remeshing is tried again after the next point)
            points, _ = _arc_remesh(ev, put, points, remesh_now, tol, tidy,
                                    lambda: float(_current(ev, what)))
            crossed = crossings()
        if ev.mesh_quality().angle_min < 1.0:
            say(f"arclength(): facets below 1 degree at {points[-1][0]:.6g}: refine "
                          "and start again from here")
            return
        (v0, x0), (v1, x1) = points[-2], points[-1]
        if len(x0) != len(x1) or len(x1) != len(ev.vertices):
            raise RuntimeError("arclength(): the vertices changed between points (it "
                               "needs the same mesh throughout, or remesh=...)")
        # the step's length and the plane it ends on from the shape's change
        # (along the normals: the tidying's sliding isn't a step; counting it
        # turned a bridge back at its fold); the prediction along the whole
        # secant (the vertices keep spreading as the surface stretches)
        tx, tv, sx = _arc_secant(ev, x0, v0, x1, v1)
        last = ev._coords()
        corrected = exact = ok = False
        through = None
        # up to ``halvings`` halvings of the step as it stands (a run near a
        # fold carries short steps on), but not below ds/4**halvings, where it
        # only crept on (a dimer between beads: six points within 1e-13)
        smallest = max(h/2**halvings, abs(ds)/4**halvings)
        while True:
            ok, iterations = _arc_correct(ev, put, x1 + h*sx, v1 + h*tv, tx, tv, tol, 12, h)
            if ok:
                corrected = True
                v = float(_current(ev, what))
                if outside(v):
                    ev._put_back(last)
                    put(v1)
                    return
                # then tidied and settled again, or a failed step too (put back
                # untidied, so the last point's mesh)
                ok, negative, residual = _arc_settle(ev, put, v, tol, tidy, (tx, tv), h)
                found = look() if ok else []
                if ok and passed(found, crossed) is not None:
                    ok, through = False, passed(found, crossed)
                if ok:
                    break
            # back at the last point: on the mesh as tidying left it (flips
            # keep the vertices), then back to the branch there by the
            # corrector along the same tangent
            if not ev._put_back(last):
                _put_coords(ev, last)
                put(v1)
                if not _arc_back(ev, put, v1, tol, (tx, tv), h)[0]:
                    _put_coords(ev, last)
                    put(v1)
                    say(f"arclength(): no equilibrium found past {v1:.6g} that "
                                  "settles again after tidying its mesh (too coarse there? "
                                  "refine), nor back at it: stopping, with the last point's "
                                  "vertices on the tidied mesh (out of equilibrium)")
                    return
            put(v1)
            if h/2 < smallest*(1 - 1e-9):
                break
            h /= 2
            if not exact:
                # the shorter steps along the branch's tangent at the last
                # point: where the branch turns sharply between points (a drop
                # hanging from a ring at its largest volume: the value's change
                # dwarfs the shape's), the secant points off it by an angle
                # that halving the step doesn't shrink, and the corrector's
                # point lands farther than the step. The vertices' sliding
                # along the surface comes from the secant, per unit of its
                # shape change (per unit of the step, it is mostly the value's
                # there, the sliding came out too large and the drop still
                # stopped; without it a catenoid's coarse mesh doesn't settle
                # on its unstable branch). Not for the full step: there the
                # tangent put a catenoid on its stable branch. (Trying the
                # secant too at each shorter step got a dimer between beads
                # further but left a bridge between disks creeping on with
                # short steps short of its largest volume.)
                ex, ev_ = _arc_tangent(ev, put, v1, tx, tv)
                shape = _rms(tx)
                slide = (sx - _normal_part(ev, sx))*(_rms(ex)/shape if shape > 0 else 0.0)
                tx, tv, sx, exact = ex, ev_, ex + slide, True
        if not ok:
            why = (f" where the surface goes through nothing ({through})" if through else
                   " that settles again after tidying its mesh (too coarse there? refine)"
                   if corrected else "")
            say(f"arclength(): no equilibrium found past {v1:.6g}{why} (even with "
                          f"steps of {h:.3g}); the surface is back at the last point")
            return
        v = float(_current(ev, what))       # (settling back can move it a little)
        points = [points[-1], (v, ev.vertices)]
        crossed = crossings()
        coming = _coming(found, before)
        before = found
        yield step(index, v, iterations, negative, residual, found)
        if coming:
            stop(v, coming)
            return
        h = min(abs(ds), 2*h)       # back up to ds after halving


def _arc_secant(ev, x0: np.ndarray, v0: float, x1: np.ndarray,
                v1: float) -> "tuple[np.ndarray, float, np.ndarray]":
    """The unit tangent (its vertices' part along the normals, and the
    value's) and the whole secant scaled alike, from two points."""
    dx = _normal_part(ev, x1 - x0)
    norm = np.sqrt(_arc_norm2(dx, v1 - v0))
    return dx/norm, (v1 - v0)/norm, (x1 - x0)/norm


def _rms(dx: np.ndarray) -> float:
    """Root mean square length of a displacement per vertex."""
    return float(np.sqrt((dx*dx).sum(axis=1).mean()))


def _arc_tangent(ev, put, value: float, tx: np.ndarray,
                 tv: float) -> "tuple[np.ndarray, float]":
    """The branch's unit tangent at the current point (the vertices' part,
    along the normals, and the value's), from where Newton steps at the
    value and at a nearby one take the vertices, turned to point as the
    secant (tx, tv) does."""
    start = ev._coords()
    delta = 1e-6*max(1.0, abs(value))
    n0, _ = _newton_target(ev, put, value)
    n1, _ = _newton_target(ev, put, value + delta)
    ev._put_back(start)
    put(value)
    dn = _normal_part(ev, (n1 - n0)/delta)
    norm = np.sqrt(_arc_norm2(dn, 1.0))
    ex, ev_ = dn/norm, 1.0/norm
    if float((ex*tx).sum())/len(tx) + ev_*tv < 0:
        ex, ev_ = -ex, -ev_
    return ex, ev_


def _arc_remesh(ev, put, points: list, remesh_now: Callable[[Callable[[], None]], bool],
                tol: float, tidy: bool, current: Callable[[], float]) -> "tuple[list, bool]":
    """arclength()'s remeshing between points: remesh, carry the last step
    over to the new vertices, and back to the branch at the same point by the
    corrector along it. The points as they are then, and whether it settled
    (if not, the surface is restored as it was)."""
    (v0, x0), (v1, x1) = points
    before: dict = {}

    def ready() -> None:
        before.update(ids=ev.mesh().vertex_ids, copy=ev._save())
    if not remesh_now(ready):
        return points, True
    ids, copy = before["ids"], before["copy"]
    d = _carry(ids, x1 - x0, ev.mesh())
    tx, tv, _ = _arc_secant(ev, ev.vertices - d, v0, ev.vertices, v1)
    ok, _, _ = _arc_settle(ev, put, v1, tol, tidy, (tx, tv))
    if not ok:
        ev.restore(copy)
        put(v1)
        return points, False
    x = ev.vertices             # (tidying flips edges: the same vertices)
    return [(v0, x - d), (current(), x)], True
