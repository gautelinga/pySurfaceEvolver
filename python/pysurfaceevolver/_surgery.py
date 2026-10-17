"""Changes of topology made with Evolver's own commands (see
:meth:`Evolver.pinch_off`)."""

from __future__ import annotations

import warnings
from typing import Any, List

import numpy as np

from . import _core
from ._engine import surface
from ._errors import _num

_SEAM = 3.0     # the cut's tips are tidied within this many median edges


def pinch_off(ev, event, relax: bool = True) -> List[int]:
    """Cut the surface at a neck: see :meth:`Evolver.pinch_off`."""
    if ev.model != "linear":
        raise ValueError("pinch_off() needs the linear model (Evolver deletes edges only there)")
    if event is None:
        necks = [e for e in ev.events() if e.kind == "neck"]
        if not necks:
            raise ValueError("pinch_off(): no neck (Evolver.events() finds none)")
        event = necks[0]
    elif getattr(event, "kind", None) != "neck":
        raise ValueError("pinch_off() cuts at a neck: an Event of kind 'neck'")
    m = ev.mesh()
    x, f, edges = np.asarray(m.vertices, float), np.asarray(m.facets), np.asarray(m.edges)
    ring, centre, side = _ring(x, f, edges, event)
    if len(ring) < 3:
        raise ValueError("pinch_off(): no ring of edges round the neck found")
    border = np.zeros(len(x), dtype=bool)
    border[edges[m.edge_valence() == 1].ravel()] = True
    _, _, gaps, _ = ev._constraint_gaps()
    held = m.fixed | border | (np.isnan(gaps).any(axis=1) if gaps.size else False)
    if held[ring].any():
        raise ValueError("pinch_off(): the neck reaches a fixed vertex, a constraint or the "
                         "surface's border (a contact line meeting itself, not a neck)")
    touching = np.isin(f, ring).any(axis=1)
    bodies = np.unique(np.asarray(m.facet_bodies)[touching])
    bodies = bodies[bodies > 0]
    if len(bodies) > 1:
        raise ValueError("pinch_off(): the neck is between two bodies (a film), not round one")
    body = int(bodies[0]) if len(bodies) else 0
    # what the collapse takes from each side: the tetrahedra from the centre to
    # the facets round the ring, which collapse onto it (signed as the body's
    # volume counts them: facet normals point out of the front body)
    a, b, c = (x[f[touching, k]] - centre for k in range(3))
    vol = np.einsum("ij,ij->i", a, np.cross(b, c))/6
    if body:
        vol = np.where(np.asarray(m.facet_bodies)[touching, 0] == body, vol, -vol)
    corners = side[f[touching]]
    inner = np.isin(f[touching], ring)
    # by the corners off the ring (facets with all three on it: by all three)
    facet_side = np.sign(np.where(inner.all(axis=1), corners.mean(axis=1),
                                  np.where(inner, 0.0, corners).sum(axis=1)))
    lost = {s: float(vol[facet_side == s].sum()) for s in (-1.0, 1.0)}
    before = _body_state(ev, body) if body else {}
    facet_ids = np.asarray(m.facet_ids)
    near = {s: facet_ids[touching][facet_side == s] for s in (-1.0, 1.0)}
    lengths = m.edge_lengths()[m.edge_valence() != 1]
    h = float(np.median(lengths)) if len(lengths) else float(event.scale)
    snapshot = ev.save()
    try:
        _cut(ev, x, ring, centre, event.scale)
        made = _split_body(ev, body, before, near, lost) if body else []
        _tidy_seam(ev, centre, h)
    except Exception:
        ev.restore(snapshot)
        raise
    if relax and _settle(ev) is None:
        warnings.warn("pinch_off(): the pieces didn't relax after the cut (the surface is "
                      "left cut, out of equilibrium)", UserWarning, stacklevel=3)
    return made


def _ring(x, f, edges, event):
    """The vertices to collapse at a neck: the ends of the edges crossing the
    plane through the neck's narrowest point, across its axis, there (two
    rows; collapsing one left slivers that didn't relax); the point they
    collapse to (their centroid); and each vertex's signed distance from the
    plane."""
    p = np.asarray(event.where, float)
    reach = max(3*event.gap, 3*event.scale)
    cross = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    normal = np.zeros_like(x)
    for k in range(3):
        np.add.at(normal, f[:, k], cross)
    length = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, length[:, None], out=np.zeros_like(normal),
                       where=length[:, None] > 0)
    # the axis: the direction the normals round the waist are all across
    waist = np.asarray(event.vertices, int)
    if len(waist) < 3:
        waist = np.flatnonzero(np.linalg.norm(x - p, axis=1) < max(event.gap, 2*event.scale))
    _, vectors = np.linalg.eigh(normal[waist].T @ normal[waist])
    axis = vectors[:, 0]
    s = (x - p) @ axis
    radial = np.linalg.norm((x - p) - np.outer(s, axis), axis=1)
    # the edges crossing the plane near the neck, in the piece of the mesh
    # round it that the waist's vertices are in
    inside = np.linalg.norm(x - p, axis=1) < reach
    local = edges[inside[edges].all(axis=1)]
    reached = np.zeros(len(x), dtype=bool)
    reached[waist[inside[waist]]] = True
    while True:
        grow = reached[local].any(axis=1) & ~reached[local].all(axis=1)
        if not grow.any():
            break
        reached[local[grow].ravel()] = True
    local = local[reached[local].all(axis=1)]
    crossing = local[(np.sign(s[local[:, 0]]) != np.sign(s[local[:, 1]]))
                     & (radial[local].max(axis=1) < reach)]
    if len(crossing) == 0:
        return np.zeros(0, dtype=int), p, s
    plus = np.unique(np.where(s[crossing[:, 0]] > 0, crossing[:, 0], crossing[:, 1]))
    minus = np.unique(np.where(s[crossing[:, 0]] > 0, crossing[:, 1], crossing[:, 0]))
    ring = np.union1d(plus, minus)
    return ring, x[ring].mean(axis=0), s


def _cut(ev, x, ring, centre, scale) -> None:
    """Collapse the ring to its centre, delete the edges that collapsed and
    pop the vertex where the two sides now meet."""
    moved = x.copy()
    moved[ring] = centre
    ev.vertices = moved
    tiny = 1e-9*scale
    count = ev.counts["edges"]
    for _ in range(len(ring)):
        ev.command(f"delete edge where length < {_num(tiny, 'length')}")
        now = ev.counts["edges"]
        if now == count:
            break
        count = now
    m = ev.mesh()
    at = np.flatnonzero(np.linalg.norm(np.asarray(m.vertices) - centre, axis=1) < 1e3*tiny)
    if len(at) != 1:
        raise ValueError(f"pinch_off(): the ring didn't collapse to one vertex ({len(at)} left)")
    v = int(np.asarray(m.vertex_ids)[at[0]])
    f = np.asarray(m.facets)
    if _sides_at(f, int(at[0])) < 2:
        raise ValueError("pinch_off(): collapsing the ring didn't pinch the surface")
    ev.command(f"pop vertex[{v}]")


def _tidy_seam(ev, centre, h: float) -> None:
    """Coarsen round the two tips the cut left: the ring's edges were short
    (a neck squeezed to an edge or two across keeps its edges round it),
    and the cones' facets all meet at the tips. Deletes the edges there
    shorter than h/2 (not on the border; Evolver refuses unsafe merges),
    equiangulating after each pass, then averages and equiangulates there."""
    def near_edges() -> str:
        m = ev.mesh()
        x, e = np.asarray(m.vertices), np.asarray(m.edges)
        return ev._mark(_core.EDGE, np.linalg.norm(x[e].mean(axis=1) - centre, axis=1) < _SEAM*h)

    for _ in range(10):
        count = ev.counts["edges"]
        ev.command(f"delete edge{near_edges()} and length < {_num(h/2, 'length')} "
                   "and valence != 1")
        ev.command(f"equiangulate edge{near_edges()}")
        if ev.counts["edges"] == count:
            break
    x = ev.vertices
    ev.command("vertex_average vertex"
               + ev._mark(_core.VERTEX, np.linalg.norm(x - centre, axis=1) < _SEAM*h))
    ev.command(f"equiangulate edge{near_edges()}")


def _sides_at(f, v) -> int:
    """How many fans of facets meet at vertex row v (2 where two cones touch)."""
    fan = np.flatnonzero((f == v).any(axis=1))
    label = np.arange(len(fan))
    rims = [frozenset(int(u) for u in f[k] if u != v) for k in fan]
    changed = True
    while changed:
        changed = False
        for i in range(len(fan)):
            for j in range(i + 1, len(fan)):
                if label[i] != label[j] and rims[i] & rims[j]:
                    label[label == max(label[i], label[j])] = min(label[i], label[j])
                    changed = True
    return len(np.unique(label))


def _body_state(ev, body: int) -> dict:
    b = ev.body(body)
    return dict(volume=b.volume, fixed=b.fixed, target=b.target, volconst=b.volconst,
                walls=_walls_touched(ev, body))


def _walls_touched(ev, body: int) -> list:
    """The walls (constraints from pysurfaceevolver.constraints with a volume
    closure) that a body's facets reach."""
    closing = [k for k, w in surface.walls.items() if getattr(w, "volconst", 0.0)]
    if not closing:
        return []
    m = ev.mesh()
    f = np.asarray(m.facets)
    verts = np.unique(f[(np.asarray(m.facet_bodies) == body).any(axis=1)])
    return [k for k in closing if ev.on_constraint(k)[verts].any()]


def _split_body(ev, body: int, before: dict, near: dict, lost: dict) -> List[int]:
    """After the cut: the body's pieces made bodies (Evolver's ``rebody``),
    the largest keeping its number; each piece's volconst from the walls it
    reaches; and their targets, if the volume was fixed: each smaller piece
    what it holds plus what the collapse took from its side, the largest the
    rest of the old target, so they add up to it."""
    old = set(int(i) for i in ev.bodies().ids)
    _rebody(ev)
    new = sorted(set(int(i) for i in ev.bodies().ids) - old)
    if not new:
        raise ValueError("pinch_off(): the body didn't split (the pieces are still joined)")
    pieces = [body] + new
    walls = {k: float(getattr(surface.walls[k], "volconst", 0.0)) for k in before["walls"]}
    extra = before["volconst"] - sum(walls.values())

    def volconst_of(i: int) -> float:
        return float(sum(walls[k] for k in _walls_touched(ev, i)))

    for i in pieces:
        ev.body(i).volconst = volconst_of(i)
    volumes = {i: ev.body(i).volume for i in pieces}
    largest = max(pieces, key=lambda i: volumes[i])
    if largest != body:
        _swap(ev, body, largest)
        volumes[body], volumes[largest] = volumes[largest], volumes[body]
        for i in (body, largest):
            ev.body(i).volconst = volconst_of(i)
    if abs(extra) > 1e-12*max(1.0, abs(before["volconst"])):
        ev.body(body).volconst = ev.body(body).volconst + extra
        volumes[body] += extra
        warnings.warn(f"pinch_off(): body {body}'s volconst ({before['volconst']:.6g}) "
                      f"isn't its walls' ({sum(walls.values()):.6g}); the rest stays with "
                      "the largest piece", UserWarning, stacklevel=4)
    if not before["fixed"]:
        for i in pieces:
            ev.body(i).target = None
        return new
    # which side of the cut each piece is on, by the facets next to it
    facet_ids = np.asarray(ev.mesh().facet_ids)
    fb = np.asarray(ev.mesh().facet_bodies)
    targets = {}
    for i in pieces:
        if i == body:
            continue
        mine = facet_ids[(fb == i).any(axis=1)]
        share = sum(lost[s] for s in (-1.0, 1.0) if np.isin(near[s], mine).any())
        targets[i] = volumes[i] + share
    targets[body] = before["target"] - sum(targets.values())
    for i in pieces:
        ev.body(i).target = targets[i]
    return new


def _swap(ev, a: int, b: int) -> None:
    """Swap two bodies' facets (and so their numbers), then rebuild their
    quantities (``rebody``, finding nothing to split now)."""
    m = ev.mesh()
    fb = np.asarray(m.facet_bodies)
    on_a, on_b = (fb == a).any(axis=1), (fb == b).any(axis=1)
    for mask, old, new in ((on_b, b, a), (on_a, a, b)):
        clause = ev._mark(_core.FACET, mask)
        for side in ("frontbody", "backbody"):
            ev.command(f"set facet {side} {new}{clause} and {side} == {old}")
    _rebody(ev)


def _rebody(ev) -> None:
    """Evolver's rebody, without its warning that a volconst may need
    adjusting (pinch_off sets them)."""
    from ._errors import EvolverWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "WARNING 2171", EvolverWarning)
        ev.command("rebody")


def _settle(ev, tol: float = 1e-8) -> Any:
    """Relax after a cut (see Evolver.pinch_off): first at the volumes the
    pieces hold, where the cones the cut left round off, by Newton steps with
    a line search (Evolver's hessian_seek) after a few gradient steps; then to
    their targets (what the cut took from them back) by plain Newton steps, as
    arclength's first step goes, in up to 16 parts. relax() from the cut (its
    long gradient phase) slid the vertices next to a drop's contact line on a
    sphere at 40 degrees through the sphere, in one of four cuts at the
    volumes held, and in most going to the targets at once. relax()'s result
    at the end, None if it failed (the targets are left as they were)."""
    bodies = ev.bodies()
    want = {int(i): ev.body(int(i)).target for i, fixed in zip(bodies.ids, bodies.fixed)
            if fixed}
    held = {i: ev.body(i).volume for i in want}

    def failed() -> None:
        for i in want:
            ev.body(i).target = want[i]

    for i in want:
        ev.body(i).target = held[i]
    ev.command("g 5")
    ev.newton(30, seek=True, tol=tol)
    r = ev.relax(tol=tol)
    if not r.converged:
        return failed()
    done, part = 0.0, 1.0
    while done < 1.0:
        start, now = ev._coords(), min(1.0, done + part)
        for i in want:
            ev.body(i).target = held[i] + now*(want[i] - held[i])
        if ev.newton(20, tol=tol).converged:
            done = now
            continue
        if not ev._put_back(start) or part < 1/16:
            return failed()
        part /= 2
    r = ev.relax(tol=tol)
    return r if r.converged else failed()
