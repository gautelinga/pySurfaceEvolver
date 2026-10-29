"""Local descents: a fall made where it happens.

A Haines jump moves one pore's surface (in a 32-bead column 1.5-5% of the
vertices moved more than an edge, within two bead radii); the rest of the
surface follows the pressure by a fraction of an edge. Here the fall's
descent (_surgery.descend) runs on a region round the lowest mode, the
rest fixed and frozen (_core.freeze: its facets left out of the quantity
loops), the draining body's pressure following its volume as the rest of the
surface would have given it back (its stiffness dp/dV measured with the
region held); then the whole surface back to the body's target by Newton
steps. A jump at fixed pressure ran away (49 changes where the global
descent made 5); coupled, a 32-bead column's jump took 48 s against 191 s,
the same 5 changes, the energy within 0.002.

After a change of topology (recipes._change) the region is round the
change's place instead, and the descent is not a fall (falling=False).

descend_local() returns descend()'s (result, changes, why), or None where it
doesn't apply (no mode found, a region of most of the surface, vertices
fixed already, no body's target to follow): then the caller descends as
before.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import _core

LOCAL_REACH = 8.0       # the region: within this many edges of the unstable mode's core
LOCAL_CORE = 0.1        # ... its core: the vertices it moves by this share of its most
LOCAL_MOST = 0.6        # not where the region would be more than this share of the vertices


def _bodies_held(ev, mobile: np.ndarray) -> dict:
    """The bodies with a target none of whose facets has a mobile vertex
    (Evolver can't hold their volumes: ERROR 3005, 3362): their targets
    released, returned to be put back."""
    m = ev.mesh()
    if m.facet_bodies is None:
        return {}
    f, fb = np.asarray(m.facets), np.asarray(m.facet_bodies)
    b = ev.bodies()
    held = {}
    for bid, fixed, target in zip(b.ids, b.fixed, b.target):
        rows = f[(fb == int(bid)).any(axis=1)]
        if fixed and len(rows) and not mobile[rows].any():
            held[int(bid)] = float(target)
            ev.body(int(bid)).target = None
    return held


def _put_back_targets(ev, held: dict) -> None:
    for bid, target in held.items():
        ev.body(bid).target = target


def descend_local(ev, body: Optional[int], factor: float, tol: float = 1e-8, keep=None,
                  history: Optional[list] = None, centre=None,
                  falling: bool = True) -> "Optional[tuple[Any, list, str]]":
    """See the module's docstring; ``centre``, a point the region is round
    (else the lowest mode's core)."""
    from . import _surgery
    from ._errors import EvolverError
    if body is None or not hasattr(_core, "freeze"):
        return None
    view = ev.body(body)
    target = view.target
    m = ev.mesh()
    if target is None or np.asarray(m.fixed, bool).any():
        return None
    x = np.asarray(m.vertices, float)
    if centre is not None:
        core = np.asarray(centre, float).reshape(1, -1)
        where = "the change"
    else:
        # (the lowest mode, of either sign: where the surface is about to move; a
        # fall's point can be stable, the descent then driven by a change coming)
        modes = ev.modes(1)
        if not len(modes.values):
            return None
        u = np.asarray(modes.vectors[0], float)[:len(x)].reshape(len(x), -1)
        size = np.linalg.norm(u, axis=1)
        core = x[size > LOCAL_CORE*size.max()]
        where = "the lowest mode"
    h = _surgery._edge_scale(ev)
    L = _surgery._slab(ev)
    reach = np.full(len(x), np.inf)
    for c in core:
        reach = np.minimum(reach, np.linalg.norm(_surgery._delta(c, x, L), axis=1))
    start = ev._save()
    p0 = view.pressure
    r: Any = None
    changes: list = []
    why = ""
    # (the region, and where its descent fails, twice as far: a jump past the
    # region's edge, held there by the frozen rest, ran its budget out)
    for k, grown in enumerate((1.0, 2.0)):
        region = reach < grown*LOCAL_REACH*h
        if region.mean() > LOCAL_MOST:
            break
        if k:
            ev.restore(start)
        _surgery._log.debug("local descent: %d of %d vertices round %s",
                            int(region.sum()), len(x), where)
        try:
            r, changes, why = _attempt(ev, int(body), region, p0, float(target), factor, tol,
                                       keep, history, falling)
        except EvolverError as error:
            _surgery._log.debug("local descent failed: %s", str(error).splitlines()[0])
            ev.restore(start)
            return None
        if r is not None:
            break
    if r is None and not changes:
        ev.restore(start)
        return None
    # the whole surface back to the target: Newton, then relax() to see it stable;
    # where that doesn't settle (or the region's descent got nowhere), the global
    # descent on from here, the region's changes kept
    if r is not None and ev.newton(20, tol=tol).converged:
        settled = ev.relax(tol=tol)
        if settled.converged and settled.stable is not False:
            return settled, changes, why
    _surgery._log.debug("local descent: not settled, on globally")
    r, more, why = _surgery.descend(ev, factor, tol, falling=False, keep=keep, history=history)
    return r, changes + more, why


def _attempt(ev, body: int, region: np.ndarray, p0: float, target: float, factor: float,
             tol: float, keep, history, falling: bool = True) -> "tuple[Any, list, str]":
    """The region's descent: the rest's stiffness measured (one Newton pair
    with the region held), then the rest fixed and frozen, the body's pressure
    coupled to its volume; the body's target and the rest put back. descend()'s
    result, with a quarter of its budget (frozen elements aren't counted as
    work: the same budget let a region's descent go on four times as long)."""
    from . import _surgery
    view = ev.body(body)
    ev.fix("vertex", region)
    held = _bodies_held(ev, ~region)
    delta = 1e-3*max(1.0, abs(target))
    try:
        _, result = ev._call(_core.newton_pair, body, float(delta))
    finally:
        ev.unfix("vertex")
        _put_back_targets(ev, held)
    if result.data is None:
        return None, [], "the stiffness: a step thrown far"
    _, m0, _, m1 = result.data
    stiff = (float(m1[body - 1]) - float(m0[body - 1]))/delta
    ev.fix("vertex", ~region)
    held = _bodies_held(ev, region)
    view.target = None
    ev.command(f"set body[{body}] pressure {p0!r}")
    _surgery._COUPLING[:] = [(body, p0, target, stiff)]
    was = _core.freeze(True)
    budget = _surgery._BUDGET
    _surgery._BUDGET = budget//4
    try:
        return _surgery.descend(ev, factor, tol, falling=falling, keep=keep, history=history)
    finally:
        _surgery._BUDGET = budget
        _core.freeze(was)
        _surgery._COUPLING.clear()
        ev.unfix("vertex")
        ev.command(f"unset body[{body}] pressure")
        view.target = target
        _put_back_targets(ev, held)
