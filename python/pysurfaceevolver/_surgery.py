"""Changes of topology made with Evolver's own commands (see
:meth:`Evolver.pinch_off`)."""

from __future__ import annotations

import warnings
from typing import Any, List

import numpy as np

from . import _core
from ._engine import GUARD_PREFIX, surface
from ._errors import _num

_SEAM = 3.0     # the cut's tips are tidied within this many median edges
_TOUCH = 0.05   # a solid this close (in its event's edges): the surface resting on it
_GUARD_REACH = 5.0  # free vertices are guarded from the walls within this many median edges
_POLE_MARGIN = 0.5  # a sphere(axis=)'s pole moved when a contact line comes this near (radii)
_POLE_TRIES = 3     # a run that came within half that done again this often, the poles moved


def guard(ev) -> None:
    """Keep free vertices out of the solids in the descent after a change
    (descend(), not falling: gradient steps move the vertices): each
    wall with a contact angle gets a one-sided copy of its constraint
    (no integrals, named GUARD_PREFIX + its key), put on the vertices on
    no wall and not fixed, for the two walls nearest each within
    _GUARD_REACH median edges (a contact line's vertex in the 0.03 gap
    between two beads hit the next bead's guard and a third, and a free
    vertex three guards: Evolver's ERROR 5883, more constraints than
    dimensions); called again as the vertices move. Evolver stops a vertex there; the energy and the
    volumes don't change, and the surface resting on the solid is a
    contact to make (a film thinned away). Gradient steps took vertices
    through a sphere beside its contact line, contact after contact, until
    the mesh broke; Newton steps there followed integrals the tangled
    line made wrong (an energy of -76 for 29). Not where vertices are put
    by steps (a fall along a mode, arclength's predictor: one put inside
    stayed on the solid, and the drops a bridge fell into came out on a
    squashed mesh, 0.9% from exact caps; there a vertex through a solid is
    a failed step, or put back out)."""
    walls = {k: w for k, w in surface.walls.items()
             if getattr(w, "contact_angle", None) is not None and getattr(w, "formula", None)}
    if not walls:
        return
    nums, _, dist, names = ev._constraint_gaps(guards=True)
    names = [str(n) for n in names]
    defined = []
    for key, wall in walls.items():
        name = f"{GUARD_PREFIX}{key}"
        if name not in names:
            ev.command(f"define constraint {name} nonnegative formula: {getattr(wall, 'formula')}")
            defined.append(name)
    if defined:
        nums, _, dist, names = ev._constraint_gaps(guards=True)
        names = [str(n) for n in names]
    _unguard(ev, names)
    m = ev.mesh()
    free = ~np.asarray(m.fixed, dtype=bool)
    real = [i for i, n in enumerate(names) if not n.startswith(GUARD_PREFIX)]
    if real:
        free &= ~np.isnan(dist[:, real]).any(axis=1)
    cols = [_column(nums, names, key) for key in walls]
    gap = np.abs(dist[:, cols])
    gap[~free] = np.inf
    gap[gap > _GUARD_REACH*_median_edge(m)] = np.inf
    # the two nearest walls only (three hits: Evolver's ERROR 5883)
    order = np.argsort(gap, axis=1)[:, :2]
    chosen = np.zeros_like(gap, dtype=bool)
    rows = np.arange(len(gap))[:, None]
    chosen[rows, order] = np.isfinite(gap[rows, order])
    for j, key in enumerate(walls):
        if chosen[:, j].any():
            ev.set_constraint("vertex", f"{GUARD_PREFIX}{key}", where=chosen[:, j])


def _pole_lines(ev):
    """For each sphere(axis=) with a contact line: (key, wall, the
    constraint's name or number, centre, radius, the line's vertices on the
    unit sphere about the centre, the least distance of its edges from the
    pole, the solid angle its edges span from the antipode, the way the
    facets go round them, and its edges: their ends on the unit sphere and
    their facets' bodies). (The edges' distance, not the vertices': Evolver
    integrates along the chords, and a pack's line whose vertices were 0.47
    from a pole had a vertex at 0.08 once its edges were split. The solid
    angle, the sum of the triangles from the antipode to the edges, jumps by
    4 pi where the line crosses the pole, as the integrals do.)"""
    nums, _, dist, names = ev._constraint_gaps()
    x = ev.vertices
    f = np.asarray(ev.mesh().facets)
    turns = np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
    _, inverse, counts = np.unique(np.sort(turns, axis=1), axis=0, return_inverse=True,
                                   return_counts=True)
    border = counts[inverse.ravel()] == 1
    edges = turns[border]                               # the border, as its facets go
    bodies = np.asarray(ev.mesh().facet_bodies)[np.flatnonzero(border) % len(f)]
    out = []
    for key, wall in surface.walls.items():
        shape = getattr(wall, "shape", None)
        if not getattr(wall, "south_regular", False) or shape is None or shape[0] != "sphere":
            continue
        try:
            col = _column(nums, names, key)
        except ValueError:
            continue
        on = np.isnan(dist[:, col])
        if not on.any():
            continue                # no contact line: no integral to keep off the pole
        centre, R = np.asarray(shape[1], float), float(shape[2])
        line = (x[on] - centre)/R
        mine = on[edges[:, 0]] & on[edges[:, 1]]
        e = edges[mine]
        a, b = (x[e[:, 0]] - centre)/R, (x[e[:, 1]] - centre)/R
        pole = np.asarray(getattr(wall, "pole_axis"), float)
        t = np.clip(np.einsum("ij,ij->i", pole - a, b - a)
                    / np.maximum(np.einsum("ij,ij->i", b - a, b - a), 1e-300), 0.0, 1.0)
        near = float(np.linalg.norm(a + t[:, None]*(b - a) - pole, axis=1).min(initial=np.inf))
        out.append((key, wall, names[col] or int(nums[col]), centre, R, line, near,
                    float(_fans(a, b, pole).sum()), a, b, bodies[mine]))
    return out


def _fans(a, b, pole) -> np.ndarray:
    """The solid angles of the triangles from the pole's antipode to the
    edges a->b on the unit sphere: summed round a line, they jump by 4 pi
    where it crosses the pole, as the integrals of a sphere(axis=) do (and
    from a point, not its antipode, where it crosses the antipode)."""
    return 2*np.arctan2(-np.cross(a, b) @ pole,
                        1 - a @ pole - b @ pole + np.einsum("ij,ij->i", a, b))


def _near_poles(ev, margin: float) -> list:
    """The sphere(axis=)s with a contact line nearer than margin (radii) to
    the pole."""
    return [p[0] for p in _pole_lines(ev) if p[6] < margin]


def move_poles(ev, force=()) -> list:
    """For each sphere(axis=) with a contact line nearer than _POLE_MARGIN of
    its radius to its north pole (where its integrals are singular), and
    those whose keys are in force, the pole
    moved to the point of the sphere farthest from its contact lines: a new
    constraint with that axis (Evolver doesn't redefine one), the sphere's
    vertices and edges moved onto it, and the volumes (volconst) and the
    energy (_SurfaceMemory.energy_shift) shifted by the whole spheres the
    integrals about the two poles differ by. A contact line stepping across
    the pole moved the wetted area by the whole sphere (4 pi cos theta), and
    the volume constraint then drove the surface off. Returns the walls
    moved."""
    from . import constraints as C
    moved = []
    n = 400
    k = np.arange(n) + 0.5
    polar, turn = np.arccos(1 - 2*k/n), np.pi*(1 + 5**0.5)*k
    cand = np.c_[np.cos(turn)*np.sin(polar), np.sin(turn)*np.sin(polar), np.cos(polar)]
    for key, wall, old, centre, R, line, near, _, a, b, sides in _pole_lines(ev):
        if near >= _POLE_MARGIN and key not in force:
            continue
        reach = np.linalg.norm(cand[:, None, :] - line[None, :, :], axis=2).min(axis=1)
        axis = cand[int(np.argmax(reach))]
        new = C.sphere(tuple(centre), R, contact_angle=getattr(wall, "contact_angle"),
                       axis=tuple(axis))
        base = str(key).split("_pole")[0] if isinstance(key, str) else f"c{key}"
        count = 1 + max([int(str(k2).rsplit("_pole", 1)[1]) for k2 in surface.walls
                         if str(k2).startswith(f"{base}_pole")] or [0])
        name = f"{base}_pole{count}"
        ev.command('exec "define constraint ' + name + " " + " ".join(new.text().split()) + '"')
        ev.command(f"set vertex constraint {name} where on_constraint {old}; "
                   f"set edge constraint {name} where on_constraint {old}; "
                   f"unset vertex constraint {old} where on_constraint {name}; "
                   f"unset edge constraint {old} where on_constraint {name}")
        del surface.walls[key]
        surface.walls[name] = new
        surface.retired.add(old)
        # the integrals about the two poles differ by whole spheres (the
        # area's 4 pi R^2 cos theta, the volume's 4 pi R^3/3) where the line
        # goes round one pole and not the other: that much is kept, as the
        # lines' solid angles from the antipodes give it. (Not what the
        # integrals give: off by their error, more than half a sphere for a
        # line near the old pole, 0.05 of a radius, or with edges 1.6 long)
        spheres = (_fans(a, b, axis) - _fans(a, b, getattr(wall, "pole_axis")))/(4*np.pi)
        for i in [int(i) for i in ev.bodies().ids]:
            side = np.where(sides[:, 0] == i, 1, np.where(sides[:, 1] == i, -1, 0))
            if side.any():
                body = ev.body(i)
                body.volconst = body.volconst + 4*np.pi*R**3/3*round(float(side @ spheres))
        cos = np.cos(np.radians(getattr(wall, "contact_angle")))
        surface.energy_shift.append((1, repr(4*np.pi*R*R*cos*round(float(spheres.sum())))))
        moved.append(name)
    return moved


def _away_from_poles(ev, run) -> tuple:
    """run() from here; where that brought a contact line within half
    _POLE_MARGIN of a sphere(axis=)'s pole, or took it across (its solid
    angle (_fans) jumped by over 2 pi: a pack's line swept across in
    one relax() run, from 0.83 to 0.32 beyond), run again from here with
    those poles moved (up to _POLE_TRIES times: a line draining toward a pole
    was refused run after run). Returns run's result, and the state it
    started from with the energy there."""
    start, energy = ev.save(), ev.energy
    fans = {p[0]: p[7] for p in _pole_lines(ev)}
    result = run()
    for _ in range(_POLE_TRIES):
        bad = [p[0] for p in _pole_lines(ev) if p[6] < _POLE_MARGIN/2
               or (p[0] in fans and abs(p[7] - fans[p[0]]) > 2*np.pi)]
        if not bad:
            break
        ev.restore(start)
        move_poles(ev, force=bad)
        start, energy = ev.save(), ev.energy
        fans = {p[0]: p[7] for p in _pole_lines(ev)}
        result = run()
    return result, start, energy


def _guarded(ev) -> bool:
    """Whether a vertex is on a guard."""
    _, _, dist, names = ev._constraint_gaps(guards=True)
    cols = [i for i, n in enumerate(names) if str(n).startswith(GUARD_PREFIX)]
    return bool(cols) and bool(np.isnan(dist[:, cols]).any())


def _unguard(ev, names=None) -> None:
    """Take the guards off every vertex (guard() puts them back): Evolver
    deletes an edge only if one end's constraints are among the other's or
    the edge's, and a guarded vertex next to a contact line's (a guard, a
    wall) is neither, so the edges there didn't go and their facets
    degenerated (0.3 degrees)."""
    if names is None:
        _, _, _, names = ev._constraint_gaps(guards=True)
    for name in names:
        if str(name).startswith(GUARD_PREFIX):
            ev.command(f"unset vertex constraint {name}")


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


def merge(ev, event, keep=None) -> int:
    """Glue the surface where it is about to touch itself across the outside:
    see :meth:`Evolver.merge`. Returns the merged body's number (0 without
    bodies)."""
    if getattr(event, "kind", None) != "touch":
        raise ValueError("merge() glues at a touch: an Event of kind 'touch'")
    m = ev.mesh()
    x, f = np.asarray(m.vertices, float), np.asarray(m.facets)
    a, b = _facing_facets(x, f, np.asarray(event.where, float))
    return _glue(ev, m, a, b, keep)


def _glue(ev, m, a: int, b: int, keep) -> int:
    """Merge facet rows a and b (Evolver's facet_merge), open the merged
    facet, and make the bodies on either side one: the one numbered keep if
    it is one of them, else the lower; its volconst theirs added, its target
    their targets added (free if either was). The merged body's number."""
    fb = np.asarray(m.facet_bodies)
    bodies = sorted({int(v) for v in fb[[a, b]].ravel() if v > 0})
    if len(bodies) > 2:
        raise ValueError("merge(): the facets touch more than two bodies")
    states = {i: _body_state(ev, i) for i in bodies}
    target = keep if keep in bodies else (bodies[0] if bodies else 0)
    other = [i for i in bodies if i != target]
    fa, fbid = int(m.facet_ids[a]), int(m.facet_ids[b])
    snapshot = ev.save()
    try:
        ev.command(f"facet_merge({fa},{fbid})")
        ev.command(f"unset facet[{fa}] frontbody; unset facet[{fa}] backbody")
        ev.command(f"dissolve facet[{fa}]")
        ev.command("dissolve edges; dissolve vertices")
        for i in other:
            for side in ("frontbody", "backbody"):
                ev.command(f"set facet {side} {target} where {side} == {i}")
        if bodies:
            _rebody(ev)
        for i in other:
            ev.command(f"dissolve body[{i}]")
    except Exception:
        ev.restore(snapshot)
        raise
    if bodies:
        body = ev.body(target)
        body.volconst = sum(s["volconst"] for s in states.values())
        if all(s["fixed"] for s in states.values()):
            body.target = sum(s["target"] for s in states.values())
        else:
            body.target = None
    return target


def _facing_facets(x, f, p):
    """The two facets either side of a touch: the facet nearest its middle,
    and on the side its normal points to, the nearest facet facing back."""
    centre = x[f].mean(axis=1)
    normal = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
    dist = np.linalg.norm(centre - p, axis=1)
    a = int(np.argmin(dist))
    across = (centre - centre[a]) @ normal[a]
    facing = (normal @ normal[a] < -0.5) & (across > 0)
    if not facing.any():
        # the nearest one may face the other way round: from the far side
        facing = (normal @ normal[a] < -0.5) & (across < 0)
        if not facing.any():
            raise ValueError("merge(): no facet facing the touch's nearest one")
    b = int(np.flatnonzero(facing)[np.argmin(dist[facing])])
    return a, b


def contact(ev, event, keep=None) -> None:
    """Make the contact a "solid" or "contact lines" event announces: see
    :meth:`Evolver.contact`."""
    kind = getattr(event, "kind", None)
    if kind not in ("solid", "contact lines"):
        raise ValueError("contact() makes a contact: an Event of kind 'solid' or "
                         "'contact lines'")
    nums, _, dist, names = ev._constraint_gaps()
    c = _column(nums, names, event.constraint)
    num = int(nums[c])
    m = ev.mesh()
    x, f, edges = np.asarray(m.vertices, float), np.asarray(m.facets), np.asarray(m.edges)
    on = np.isnan(dist[:, c])
    rows = np.asarray(event.vertices, int)
    flat = None
    if kind == "contact lines":
        pinched = _pinched(m, edges, rows[on[rows]])
        if pinched is not None:
            # the lines already meet at a vertex (the border through it
            # twice): Evolver's pop splits it, and the lines are then two
            # (a pore's contact lines on one bead after its liquid jumped)
            ev.command(f"pop vertex[{int(m.vertex_ids[pinched])}]")
            return
        a, b = _closest_pair(x, rows, on)
        if _wet_between(ev, m, x, f, on, num, a, b):
            core = _strip(x, a, b, on, float(event.scale))
        else:
            # facets lying on the solid between the lines there (all three
            # corners on it: a dry channel roofed over), else a point contact
            near = np.linalg.norm(x[f].mean(axis=1) - (x[a] + x[b])/2, axis=1) \
                < 2*float(event.scale)
            flat = on[f].all(axis=1) & near
            if not flat.any():
                _join_lines(ev, m, x, f, a, b, keep)
                return
            core = np.zeros(0, dtype=int)
    else:
        core = rows[~on[rows]]
    # (not vertices on other constraints, or fixed: the patch stays clear of them)
    held = (np.isnan(dist).any(axis=1) & ~on) | m.fixed
    core = core[~held[core]]
    if flat is not None:
        gone = flat
    elif not len(core):
        raise ValueError("contact(): the place is on other constraints or fixed")
    else:
        # next to a contact line on that solid (a vertex of the place one
        # edge from it), the patch there grows, the strip of facets between
        # going too (a patch opened beside it, a facet's width away, broke
        # the mesh later): else a patch of its own
        beside = on[edges[np.isin(edges, core).any(axis=1)].ravel()].any()
        gone = (np.isin(f, core) | on[f]).all(axis=1) & np.isin(f, core).any(axis=1) \
            if kind == "contact lines" or beside else np.isin(f, core).all(axis=1)
    if not gone.any():
        # a single vertex close: the facets round the closest one
        k = core[np.argmin(np.abs(dist[core, c]))]
        ring = np.unique(f[(f == k).any(axis=1)])
        core = np.union1d(core, ring[~on[ring] & ~held[ring]])
        gone = (f == k).any(axis=1)
    # onto the solid first (so taking the facets away moves no liquid), then
    # the volumes the bodies hold, which the new contact line keeps
    _, grad = ev._call(_core.constraint_normals, num)
    normal = np.asarray(grad.data)
    moved = x.copy()
    put = np.unique(f[gone])
    put = put[~on[put]]
    moved[put] -= (dist[put, c])[:, None]*normal[put]
    ev.vertices = moved
    bodies = [int(i) for i in ev.bodies().ids]
    volume = {i: ev.body(i).volume for i in bodies}
    snapshot = ev.save()
    try:
        was_border = set(np.asarray(m.edge_ids)[m.edge_valence() == 1].tolist())
        ids = np.asarray(m.facet_ids)[gone]
        ev.command("; ".join(f"dissolve facet[{int(i)}]" for i in ids))
        ev.command("dissolve edges; dissolve vertices")
        added = _onto(ev, num, was_border)
    except Exception:
        ev.restore(snapshot)
        raise
    _poles(ev, event.constraint, added)
    for i in bodies:
        if i in [int(j) for j in ev.bodies().ids]:
            b = ev.body(i)
            b.volconst = b.volconst + volume[i] - b.volume


def _pinched(m, edges, rows) -> "int | None":
    """A vertex among rows that four or more border edges meet at."""
    border = edges[m.edge_valence() == 1]
    for v in rows:
        if int((border == v).any(axis=1).sum()) >= 4:
            return int(v)
    return None


def _column(nums, names, key) -> int:
    if isinstance(key, str):
        return names.index(key)
    return [int(n) for n in nums].index(int(key))


def _onto(ev, num: int, was_border: set) -> np.ndarray:
    """Put the surface's new border edges (not among the ids was_border) and
    their vertices on constraint num, each edge stored the way its facet goes
    round it, with orientation 1, as the contact lines from make_datafile
    are: an edge stored the other way is turned round (Evolver's
    reverse_orientation, then orientation 1) before it goes on. (Leaving it
    stored the other way with orientation -1 gave the right energy integral
    but not the content: on a sphere a dry patch's volume came out 0.03 off
    and the pressure -0.8 for -4.45.)"""
    m = ev.mesh()
    f, edges = np.asarray(m.facets), np.asarray(m.edges)
    on = ev.on_constraint(num)
    edge_on = ev.on_constraint(num, "edge")
    along = {(int(r[i]), int(r[(i + 1) % 3])) for r in f for i in range(3)}
    new = ~np.isin(np.asarray(m.edge_ids), list(was_border))
    border = np.flatnonzero((m.edge_valence() == 1) & ~edge_on & new)
    commands = []
    for v in np.unique(edges[border]):
        if not on[v]:
            commands.append(f"set vertex[{int(m.vertex_ids[v])}] constraint {num}")
    for i in border:
        e = int(m.edge_ids[i])
        if (int(edges[i, 0]), int(edges[i, 1])) not in along:
            commands.append(f"reverse_orientation edge[{e}]")
            commands.append(f"set edge[{e}] orientation 1")
        commands.append(f"set edge[{e}] constraint {num}")
    if commands:
        ev.command("; ".join(commands))
    return np.asarray(m.edge_ids)[border]


def _on_walls(ev) -> np.ndarray:
    """Which edges (mesh() rows) are on a wall."""
    names = {n or int(k) for k, n in zip(*ev._constraint_gaps()[::3])}
    out = np.zeros(ev.counts["edges"], dtype=bool)
    for key in surface.walls:
        if key in names:
            out |= ev.on_constraint(key, "edge")
    return out


def orient_lines(ev) -> int:
    """Every contact line's edges (border edges on a wall) stored the way
    their facet goes round them, with orientation 1, as make_datafile and
    _onto put them: where an edge isn't, its integrals come out with the
    wrong sign (a pack's line with 2 of its 35 edges turned had its energy
    integral at -1.26 for 0.40, and the energy drifted below what the beads
    can give). Returns how many edges were turned."""
    m = ev.mesh()
    if not surface.walls or not len(m.edges):
        return 0
    f, edges = np.asarray(m.facets), np.asarray(m.edges)
    n = len(m.vertices)
    turns = np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
    along = np.isin(edges[:, 0]*n + edges[:, 1], turns[:, 0]*n + turns[:, 1])
    lines = _on_walls(ev) & (m.edge_valence() == 1)
    wrong = np.flatnonzero(lines & (~along | (ev.values("edge", "orientation") < 0)))
    commands = []
    for i in wrong:
        e = int(m.edge_ids[i])
        if not along[i]:
            commands.append(f"reverse_orientation edge[{e}]")
        commands.append(f"set edge[{e}] orientation 1")
    if commands:
        ev.command("; ".join(commands))
    return len(wrong)


def close_patches(ev, h: float) -> int:
    """Contact-line loops within 0.5 h of their first vertex (a patch shrunk
    to nothing: none of its edges can be deleted, the two left would be
    doubled, and relax() ended with a pack's three-edge loop 0.003 across as
    degenerate facets) covered: Evolver's new_facet over the loop, with the
    bodies of the facet beside it, the loop's edges and vertices taken off
    the wall (the tidying then deletes them). Not where that moved the
    energy (a loop round a pole). The lines oriented (orient_lines). Returns
    how many were closed."""
    m = ev.mesh()
    f, edges = np.asarray(m.facets), np.asarray(m.edges)
    x = np.asarray(m.vertices)
    names = {n or int(k): (n or int(k)) for k, n in zip(*ev._constraint_gaps()[::3])}
    valence = m.edge_valence()
    closed = 0
    for key in [k for k in surface.walls if k in names]:
        line = np.flatnonzero(ev.on_constraint(key, "edge") & (valence == 1))
        degree = np.bincount(edges[line].ravel(), minlength=len(x))
        after = {int(edges[i, 0]): i for i in line if degree[edges[i]].max() == 2}
        seen: set = set()
        for start in list(after):
            if start in seen:
                continue
            loop, v = [], start
            while v in after and v not in seen:
                seen.add(v)
                loop.append(after[v])
                v = int(edges[after[v], 1])
            if v != start or not 3 <= len(loop) <= 6:
                continue
            corners = edges[loop, 0]
            if np.linalg.norm(x[corners] - x[corners[0]], axis=1).max() > 0.5*h:
                continue
            a, b = edges[loop[0]]
            beside = np.flatnonzero(((f == a) & (np.roll(f, -1, axis=1) == b)).any(axis=1))
            if not len(beside):
                continue
            front, back = (int(i) for i in np.asarray(m.facet_bodies)[beside[0]])
            ids = [int(m.edge_ids[i]) for i in loop]
            energy, saved = ev.energy, ev.save()
            body = "" if len(ids) == 3 else " where original == pyse_patch"
            commands = [f"pyse_patch := new_facet({', '.join(str(-e) for e in reversed(ids))})"]
            for side, b_id in (("frontbody", front), ("backbody", back)):
                if b_id:
                    commands.append(f"set facet{body or '[pyse_patch]'} {side} {b_id}")
            commands += [f"unset edge[{e}] constraint {names[key]}" for e in ids]
            commands += [f"unset vertex[{int(m.vertex_ids[c])}] constraint {names[key]}"
                         for c in corners]
            ev.command("; ".join(commands))
            if abs(ev.energy - energy) > 0.05:
                ev.restore(saved)
                continue
            closed += 1
            m = ev.mesh()
            f, edges, x = np.asarray(m.facets), np.asarray(m.edges), np.asarray(m.vertices)
            valence = m.edge_valence()
            break
    return closed


def _poles(ev, key, added) -> None:
    """If the new contact line (edge ids added, each with the wet side to
    its left seen from outside the solid) winds round a sphere's axis, the
    pole it encloses has turned wet or dry, and the energy the wall's
    integrals leave out changes by that pole's share
    (_SurfaceMemory.energy_shift; the volume's constant is taken care of by
    contact()'s volconst)."""
    wall = surface.walls.get(key)
    shape = getattr(wall, "shape", None)
    if not len(added) or shape is None or shape[0] != "sphere" \
            or not getattr(wall, "pole_energy", None):
        return
    centre = np.asarray(shape[1], float) if all(not isinstance(c, str) for c in shape[1]) \
        else None
    if centre is None:
        return                      # a centre that follows a parameter: no constant to keep
    from .constraints import _frame
    e3 = np.asarray(getattr(wall, "pole_axis", None) or (0.0, 0.0, 1.0), float)
    e1, e2 = _frame(e3) if e3[2] != 1.0 else (np.eye(3)[0], np.eye(3)[1])
    m = ev.mesh()
    rows = np.flatnonzero(np.isin(np.asarray(m.edge_ids), added))
    x = np.asarray(m.vertices, float) - centre
    edges = np.asarray(m.edges)[rows]
    phi = np.arctan2(x @ e2, x @ e1)
    turn = np.angle(np.exp(1j*(phi[edges[:, 1]] - phi[edges[:, 0]]))).sum()
    winding = int(round(turn/(2*np.pi)))
    if winding == 0:
        return
    north = (x[edges].reshape(-1, 3) @ e3).mean() > 0
    if not north and getattr(wall, "south_regular", False):
        return                      # (sphere(axis=): the integrals are regular there)
    sign = winding*(1 if north else -1)     # +1: the pole now wet (anticlockwise round it)
    surface.energy_shift.append((int(np.sign(sign)), str(getattr(wall, "pole_energy"))))


def _closest_pair(x, rows, on):
    """The two vertices on the constraint among an event's, closest together
    and not neighbours along the line (the event's two stretches)."""
    pts = rows[on[rows]]
    if len(pts) < 2:
        raise ValueError("contact(): the contact lines' vertices weren't found")
    d = np.linalg.norm(x[pts][:, None, :] - x[pts][None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return int(pts[i]), int(pts[j])


def _wet_between(ev, m, x, f, on, num, a, b) -> bool:
    """Whether the solid between two contact-line vertices is wet: b on the
    side of a's contact line where the liquid is. A facet goes round its
    contact-line edge with the wet side to its left, seen from outside the
    solid, its normal pointing out of the liquid (a drop's rim facet at any
    contact angle)."""
    _, grad = ev._call(_core.constraint_normals, num)
    n = np.asarray(grad.data)[a]
    fb = np.asarray(m.facet_bodies)
    for r, row in enumerate(f):
        for i in range(3):
            p, q = int(row[i]), int(row[(i + 1) % 3])
            if (p == a and on[q]) or (q == a and on[p]):
                t = x[q] - x[p]
                if fb.size and fb[r, 0] == 0 and fb[r, 1] > 0:
                    t = -t                  # the liquid behind the facet
                return float(np.cross(n, t) @ (x[b] - x[a])) > 0
    raise ValueError("contact(): no facet along the contact line at the event")


def _strip(x, a, b, on, h):
    """The vertices over a wet strip between two contact-line vertices: within
    an edge of the segment between them."""
    p, q = x[a], x[b]
    t = np.clip(((x - p) @ (q - p))/max(float((q - p) @ (q - p)), 1e-300), 0, 1)
    near = np.linalg.norm(x - (p + t[:, None]*(q - p)), axis=1) < h + 0.5*np.linalg.norm(q - p)
    return np.flatnonzero(near & ~on)


def _join_lines(ev, m, x, f, a, b, keep) -> None:
    """Two contact lines about to meet across dry ground (vertex rows a and b
    on them): a's and b's contact-line edges that face each other glued into
    one edge (Evolver's vertex_merge on their ends, then edge_merge), which
    the two facets along them now share, off the constraint: the lines
    reconnect there (two patches join; one patch meeting itself encloses a
    dry island). The bodies either side become one as merge() makes them, and
    the volumes stay as they were (volconst)."""
    edges = np.asarray(m.edges)
    valence = m.edge_valence()
    num = None
    nums, _, dist, names = ev._constraint_gaps()
    on = np.isnan(dist)
    for c in range(on.shape[1]):
        if on[a, c] and on[b, c]:
            num = c
    if num is None:
        raise ValueError("contact(): the two vertices aren't on one constraint")
    line = (valence == 1) & on[edges[:, 0], num] & on[edges[:, 1], num]

    def along(v):
        rows = np.flatnonzero(line & (edges == v).any(axis=1))
        return [(int(r), int(edges[r, 1] if edges[r, 0] == v else edges[r, 0])) for r in rows]

    # a's line edge and b's whose other ends are closest (the lines run
    # opposite ways where they face each other)
    pairs = [(float(np.linalg.norm(x[p] - x[q])), ra, p, rb, q)
             for ra, p in along(a) for rb, q in along(b) if p != q and p != b and q != a]
    if not pairs:
        raise ValueError("contact(): no facing contact-line edges at the meeting")
    _, ra, a2, rb, b2 = min(pairs)
    # the lines side by side there (their next vertices about as close as
    # a and b); where they cross at an angle the merge would drag vertices
    # far
    step = 0.5*(np.linalg.norm(x[a2] - x[a]) + np.linalg.norm(x[b2] - x[b]))
    if np.linalg.norm(x[a2] - x[b2]) > np.linalg.norm(x[a] - x[b]) + 0.5*step:
        raise ValueError("contact(): the contact lines meet at an angle, not side by side: "
                         "not joined there")
    # merging vertices with a neighbour in common doubles their edges to it,
    # with facets going round them the same way (Evolver's pop fails on them)
    ring = {v: set(edges[(edges == v).any(axis=1)].ravel().tolist()) - {v}
            for v in (a, b, a2, b2)}
    if ring[a] & ring[b] or ring[a2] & ring[b2]:
        raise ValueError("contact(): the contact lines meet where the mesh already joins "
                         "them (shared neighbours): not joined there")
    fb = np.asarray(m.facet_bodies)
    bodies = sorted({int(v) for r in range(len(f))
                     if {a, a2} <= set(f[r].tolist()) or {b, b2} <= set(f[r].tolist())
                     for v in fb[r] if v > 0})
    states = {i: _body_state(ev, i) for i in bodies}
    target = keep if keep in bodies else (bodies[0] if bodies else 0)
    volume = {i: ev.body(i).volume for i in bodies}
    ids = np.asarray(m.vertex_ids)
    eids = np.asarray(m.edge_ids)
    snapshot = ev.save()
    try:
        ev.command(f"vertex_merge({int(ids[a])},{int(ids[b])})")
        ev.command(f"vertex_merge({int(ids[a2])},{int(ids[b2])})")
        ev.command(f"edge_merge({int(eids[ra])},{int(eids[rb])})")
        ev.command(f"unset edge[{int(eids[ra])}] constraint {int(nums[num])}")
        for i in bodies:
            if i == target:
                continue
            for side in ("frontbody", "backbody"):
                ev.command(f"set facet {side} {target} where {side} == {i}")
        if len(bodies) > 1:
            _rebody(ev)
            for i in bodies:
                if i != target:
                    ev.command(f"dissolve body[{i}]")
    except Exception:
        ev.restore(snapshot)
        raise
    if bodies:
        body = ev.body(target)
        body.volconst = sum(s["volconst"] for s in states.values())
        body.volconst = body.volconst + sum(volume.values()) - body.volume
        if all(s["fixed"] for s in states.values()):
            body.target = sum(s["target"] for s in states.values())
        else:
            body.target = None


_FALL_STEP = 0.3        # a fall's step along the unstable mode: at most this many median edges
_FALL_GRADIENT = 5      # gradient steps after each
_FALL_ITERATIONS = 200
_FALL_NEAR = 0.05       # the residual below which Newton steps try to finish a descent
_FALL_TRY = 10          # iterations between such tries
_FALL_CLEAR = 0.1       # the gradient steps' cosine with the mode that sets the way
_REMESH = (0.4, 2.0)    # after a merge or contact: edges kept within these many median edges
_CHUNK = 50             # after a change: relax() runs of this many gradient steps at most
_GRADIENT_RUN = 10      # ... or, where one went wrong, this many plain gradient steps


def _median_edge(m) -> float:
    lengths = m.edge_lengths()[m.edge_valence() != 1]
    return float(np.median(lengths)) if len(lengths) else float(np.median(m.edge_lengths()))


def apply(ev, event, keep=None) -> None:
    """Make the change an event announces (without relaxing; the guards
    off, see _unguard)."""
    _unguard(ev)
    if event.kind == "neck":
        pinch_off(ev, event, relax=False)
    elif event.kind == "touch":
        merge(ev, event, keep)
    else:
        contact(ev, event, keep)


def descend(ev, factor: float, tol: float = 1e-8, *, falling: bool = True, keep=None,
            h0=None) -> "tuple[Any, list, str]":
    """See _descend; an EvolverError on the way (a mesh the changes left
    broken) ends it, the reason the error's."""
    from ._errors import EvolverError
    changes: list = []
    try:
        return _descend(ev, factor, tol, falling=falling, keep=keep, h0=h0, changes=changes)
    except EvolverError as error:
        return None, changes, f"Evolver: {error}"
    finally:
        _unguard(ev)


def _descend(ev, factor: float, tol: float, *, falling: bool, keep, h0,
             changes: list) -> "tuple[Any, list, str]":
    """Down to a stable state at the same value (see arclength's topology=),
    making the changes of topology that come on the way. ``falling``: from a
    point that has lost its stability (steps along its unstable mode); else
    from a change just made (gradient steps, the edges kept in range, until
    near an equilibrium). Returns relax's result there (None if it got
    nowhere), the events whose changes were made, and why it stopped if it
    did."""
    way: Any = None
    before: list = []
    refused = ""
    tried = _FALL_TRY
    remeshing = not falling
    if h0 is None:
        h0 = _median_edge(ev.mesh())
    if not falling:
        guard(ev)
    for _ in range(_FALL_ITERATIONS):
        orient_lines(ev)
        close_patches(ev, h0)
        move_poles(ev)
        x = ev.vertices
        chunk = None
        if falling:
            ev.command(f"g {_FALL_GRADIENT}")
            # (a fall doesn't remesh, but its contact lines are kept fine: a
            # pack's stretched to edges 1.6 long, and their integrals along
            # the chords went 1.5 off)
            _split_lines(ev, h0)
        else:
            # after a change, relax() in short runs (a merged neck widening
            # takes thousands of gradient steps on a fine mesh), looking at
            # the events between them; a run that went wrong (relax()'s took
            # a pack's liquid from 80.3 to 83.5) is undone for a run of plain
            # gradient steps
            chunk, start, energy = _away_from_poles(ev, lambda: ev.relax(tol, max_iter=_CHUNK))
            if not _plausible(ev, energy):
                ev.restore(start)
                chunk = _gradient_run(ev, tol)
                if chunk is None:
                    return None, changes, "the descent ran away (volumes or energy)"
            guard(ev)
        # (vertices a split added come last: the rows before are the same)
        moved = ev.vertices[:len(x)] - x if len(x) <= len(ev.vertices) else None
        if remeshing and not (chunk is not None and chunk.converged):
            _keep_edges(ev, h0)
            moved = None if moved is None or len(moved) != len(ev.vertices) else moved
        found = ev.events(factor)
        m = ev.mesh()
        h = _median_edge(m)
        # a neck coming (thinning: not a merge's, widening) that can be cut
        # (not one reaching a contact line: the neck that pinches comes later)
        neck = None
        for e in _coming(found, before):
            if e.kind == "neck":
                try:
                    _unguard(ev)
                    pinch_off(ev, e, relax=False)
                    neck = e
                    break
                except ValueError as error:
                    refused = f"{e}: {error}"
        if neck is not None:
            changes.append(neck)
            way, before = None, []
            r = _settle(ev, tol)
            if r is None:
                return None, changes, "the pieces didn't relax after the cut"
            if r.stable is not False:
                return r, changes, ""
            falling, remeshing = True, False
            continue
        # the surface resting on a solid (held there by its guard) or gone
        # through it: a contact (beside a contact line, the patch grows: the
        # line moves on past a vertex at the solid, which held the gradient
        # steps up); coming elsewhere, a contact too, as are touches and
        # contact lines coming (merges, wetted or dry patches). Coming beside
        # a contact line is the line's own (it moves there).
        resting = [e for e in found if _resting(e)]
        if falling:
            # (a fall's path is the mode's, not the liquid's: only a surface
            # gone through a solid counts, put back out beside a contact line,
            # a contact elsewhere. Made contacts, a bridge's surface lying on
            # the sphere it drained from grew the lines there, and the pieces
            # cut after didn't settle)
            resting = [e for e in resting if e.kind == "solid" and e.gap < 0]
            held = [e for e in resting if _beside_line(ev, e, h)]
            resting = [e for e in resting if not any(e is r for r in held)]
            _push_out(ev, held, h)
        beside = [e for e in found if e.kind == "solid" and (falling or not _resting(e))
                  and _beside_line(ev, e, h)]
        coming = resting + [e for e in _coming(found, before)
                            if not any(e is r for r in resting)
                            and (e.kind in ("touch", "contact lines")
                                 or (e.kind == "solid" and not any(e is b for b in beside)))]
        made = None
        for e in coming:
            try:
                apply(ev, e, keep)
                made = e
                break
            except ValueError as error:
                refused = f"{e}: {error}"       # (passed over: it may come again, or go)
        if made is not None:
            e = made
            changes.append(e)
            guard(ev)
            way, before, tried = None, [], 0
            falling, remeshing = False, True
            continue
        through = [e for e in resting if e.kind == "solid" and e.gap < 0
                   and _beside_line(ev, e, h)]
        if through:
            _push_out(ev, through, h)
        before = [e for e in found if not (e.kind == "solid" and e.gap < 0)]
        residual = ev.residual()
        tried += 1
        if chunk is not None:
            if not chunk.converged:
                continue
            if chunk.stable is not False and not _sound(chunk.health):
                # (a facet squashed in the change, a patch shrunk to nothing:
                # tidied, and once more)
                close_patches(ev, h0)
                ev.command("u")
                _keep_edges(ev, h0)
                chunk = ev.relax(tol)
            if chunk.converged and chunk.stable is not False and _sound(chunk.health):
                return chunk, changes, ""
            if chunk.stable is not False:
                return None, changes, f"relaxed, but {chunk.health}"
            remeshing = False           # an unstable equilibrium: down along its mode
        modes = ev.modes(1)
        if not len(modes.values) or not modes.values[0] < 0:
            # no way down along a mode: gradient steps until near an
            # equilibrium, which Newton steps then finish (relax()'s long
            # gradient phase slid a drop's vertices through its sphere), every
            # few steps
            if residual > _FALL_NEAR or tried < _FALL_TRY:
                continue
            # (or relax(), where Newton steps alone don't get there: a drop
            # free to move as a whole, which relax() deflates)
            tried = 0
            start = ev.save()
            for finish in ("newton", "relax"):
                if finish == "newton" and not ev.newton(20, tol=tol).converged:
                    continue
                r = ev.relax(tol=tol)
                if r.converged and r.stable is not False and _sound(r.health):
                    return r, changes, ""
                ev.restore(start)
            continue                    # a saddle, or not there yet: on down
        if not falling:
            _unguard(ev)
        falling = True
        # the way the gradient steps went along the mode; the first step,
        # where that is round-off (at a symmetric start: a bridge fell to
        # either sphere from run to run), by modes()'s sign, fixed by the
        # mode itself (modes() can turn it from one step to the next, so not
        # after that)
        u = modes.vectors[0]
        along = _common(moved, u) if moved is not None else 0.0
        clear = _FALL_CLEAR*float(np.linalg.norm(moved)*np.linalg.norm(u[:len(moved)])) \
            if moved is not None else 0.0
        if way is None:
            sign = float(np.sign(along)) if clear > 0 and abs(along) > clear else 1.0
        elif along != 0.0:
            sign = float(np.sign(along))
        else:
            sign = float(np.sign(_common(way, u))) or 1.0
        way = sign*u
        ev.vertices = ev.vertices + _FALL_STEP*h*way/np.linalg.norm(u, axis=1).max()
    return None, changes, (f"no stable state within {_FALL_ITERATIONS} steps"
                           + (f" (the last change not made: {refused})" if refused else ""))


def _common(a, b) -> float:
    """The dot product of two per-vertex arrays over the rows both have."""
    n = min(len(a), len(b))
    return float((a[:n]*b[:n]).sum())


def _split_lines(ev, h0: float) -> None:
    """Contact-line edges (border edges on a wall) longer than _REMESH[1] h0
    split till none is (Evolver's edge refinement: the midpoint goes onto
    the wall)."""
    for _ in range(6):
        m = ev.mesh()
        long = (m.edge_valence() == 1) & (m.edge_lengths() > _REMESH[1]*h0)
        if not long.any():
            return
        long &= _on_walls(ev)
        if not long.any():
            return
        ev.command(f"refine edge{ev._mark(_core.EDGE, long)}")


def _gradient_run(ev, tol: float) -> Any:
    """A run of the descent after a change where relax()'s went wrong:
    _GRADIENT_RUN plain gradient steps (one, where those went wrong too;
    None where that did), then relax() if the residual is under _FALL_NEAR
    (kept if it converged and went right). relax()'s result, else an
    unconverged one."""
    _, start, energy = _away_from_poles(ev, lambda: ev.command(f"g {_GRADIENT_RUN}"))
    if not _plausible(ev, energy):
        ev.restore(start)
        _, start, energy = _away_from_poles(ev, lambda: ev.command("g 1"))
        if not _plausible(ev, energy):
            ev.restore(start)
            return None
    if ev.residual() < _FALL_NEAR:
        r, here, energy = _away_from_poles(ev, lambda: ev.relax(tol))
        if r.converged and _plausible(ev, energy):
            return r
        ev.restore(here)
    return _unconverged()


def _unconverged() -> Any:
    from ._results import IterationResult
    none = np.empty(0)
    return IterationResult(none, none, none, "", converged=False)


def _plausible(ev, energy: float) -> bool:
    """Whether a run of the descent kept its volumes (within 1e-3 of their
    targets), lowered the energy by less than a quarter of the area, and
    kept the contact lines half _POLE_MARGIN from the poles of sphere(axis=)s
    (moved at the next iteration): a run past that is one the integrals got
    wrong (relax() runs after a contact took a pack's liquid from 80.3 to
    83.5, conjugate gradients a drop's from 31.5 to 44.5 and its energy to
    -12136; a pack's contact line came from half a radius to 0.07 of a pole
    in one run, the integrals off by up to 4 pi cos theta)."""
    if ev._target_gap()[0] > 1e-3 or _near_poles(ev, _POLE_MARGIN/2):
        return False
    return energy - ev.energy < 0.25*_core.total_area()


def _sound(health) -> bool:
    """Whether a descent may end in this state: healthy but for coming
    near a wall or itself (a contact or merge to come: after a drop's top
    dried on one bead the film over the next was 0.015 thick)."""
    return health is not None and all(i.startswith("the surface nearly touches")
                                      for i in health.issues)


def _resting(e) -> bool:
    """Whether a solid event is the surface resting on the solid (or
    through it), or contact lines have met (they stay met, so aren't
    coming: the facets between them squashed flat while they waited)."""
    return e.kind in ("solid", "contact lines") and e.gap < _TOUCH*float(e.scale)


def _coming(found: list, before: list) -> list:
    from .recipes import _coming as coming
    return coming(found, before)


def _keep_edges(ev, h0: float) -> None:
    """Edges back in range after a change: those shorter than 0.4 h0 deleted
    (Evolver's edge deletion), contact-line edges too (a joined contact line
    kept its short ones, and its facets degenerated as it moved), but not
    chords between two contact-line vertices (deleting one pinched two
    contact lines together at a vertex: they meet by contact() instead);
    those longer than 2 h0 split, but not edges into the surface from a
    contact line on a constraint (their midpoints can land inside a solid);
    then equiangulated."""
    lo, hi = _REMESH
    m = ev.mesh()
    lengths = m.edge_lengths()
    if not ((lengths < lo*h0).any() or (lengths > hi*h0).any()):
        return
    guarded = _guarded(ev)
    _unguard(ev)
    try:
        _keep_edges_unguarded(ev, h0, m)
    finally:
        if guarded:
            guard(ev)


def _keep_edges_unguarded(ev, h0: float, m) -> None:
    lo, hi = _REMESH
    _, _, dist, _ = ev._constraint_gaps()
    on = np.isnan(dist).any(axis=1) if dist.size else np.zeros(len(m.vertices), dtype=bool)
    for _ in range(5):
        edges = np.asarray(m.edges)
        short = _collapsible(m, edges, on, m.edge_lengths() < lo*h0)
        if not short.any():
            break
        ev.command(f"delete edge{ev._mark(_core.EDGE, short)}")
        m = ev.mesh()
        _, _, dist, _ = ev._constraint_gaps()
        on = np.isnan(dist).any(axis=1) if dist.size else np.zeros(len(m.vertices), dtype=bool)
    edges = np.asarray(m.edges)
    _, _, dist, _ = ev._constraint_gaps()
    on = np.isnan(dist).any(axis=1) if dist.size else np.zeros(len(m.vertices), dtype=bool)
    protect = on[edges].any(axis=1) & ~on[edges].all(axis=1)
    ev.remesh(max_edge=hi*h0, min_edge=None, protect=protect)


def _collapsible(m, edges, on, short) -> np.ndarray:
    """Of the short edges, those that can be collapsed (Evolver's edge
    deletion) without making the mesh non-manifold, no two at one vertex:
    not fixed, not a chord between two contact-line vertices, and their ends
    with no neighbour in common but the facets' third corners (the link
    condition; Evolver doesn't check it, and a contact line's edge whose ends
    had another neighbour in common came out as a doubled edge and a facet
    with a corner twice)."""
    valence = m.edge_valence()
    f = np.asarray(m.facets)
    nbrs: list = [set() for _ in range(len(m.vertices))]
    for a, b in edges.tolist():
        nbrs[a].add(b)
        nbrs[b].add(a)
    third: dict = {}
    for r in f.tolist():
        for i in range(3):
            key = (min(r[i], r[(i + 1) % 3]), max(r[i], r[(i + 1) % 3]))
            third.setdefault(key, set()).add(r[(i + 2) % 3])
    out = np.zeros(len(edges), dtype=bool)
    used: set = set()
    lengths = np.linalg.norm(m.vertices[edges[:, 0]] - m.vertices[edges[:, 1]], axis=1)
    for k in np.flatnonzero(short)[np.argsort(lengths[short])]:
        a, b = int(edges[k, 0]), int(edges[k, 1])
        if a in used or b in used or m.fixed[a] or m.fixed[b]:
            continue
        if on[a] and on[b] and valence[k] != 1:
            continue                    # a chord across: the lines would pinch together
        if nbrs[a] & nbrs[b] != third.get((min(a, b), max(a, b)), set()):
            continue
        out[k] = True
        used |= {a, b} | nbrs[a] | nbrs[b]
    return out


def _beside_line(ev, event, h: float) -> bool:
    """Whether a solid event's vertices are within an edge and a half (its
    edges there) of a vertex on that constraint: next to its contact line,
    where the gradient steps bring vertices to the solid (a merged neck
    sinking onto a plane between the drops' old contact lines is further)."""
    nums, _, dist, names = ev._constraint_gaps()
    c = _column(nums, names, event.constraint)
    on = np.isnan(dist[:, c])
    if not on.any():
        return False
    x = ev.vertices
    rows = np.asarray(event.vertices, int)
    reach = np.linalg.norm(x[rows][:, None, :] - x[on][None, :, :], axis=2).min(axis=1)
    return bool((reach < 1.5*float(event.scale)).all())


def _push_out(ev, through: list, h: float) -> None:
    """Put the vertices that have gone through a solid back on the surface's
    side of it, along its normal, a thousandth of an edge out."""
    nums, _, dist, names = ev._constraint_gaps()
    moved = ev.vertices
    for e in through:
        c = _column(nums, names, e.constraint)
        d = dist[:, c]
        off = ~np.isnan(d)
        side = 1.0 if (d[off] > 0).sum() >= (d[off] < 0).sum() else -1.0
        rows = np.asarray(e.vertices, int)
        rows = rows[off[rows] & (side*d[rows] < 0)]
        if not len(rows):
            continue
        _, grad = ev._call(_core.constraint_normals, int(nums[c]))
        normal = np.asarray(grad.data)[rows]
        moved[rows] += side*normal*(-side*d[rows] + 1e-3*h)[:, None]
    ev.vertices = moved


def relax_after(ev, keep=None) -> Any:
    """merge()'s and contact()'s relaxing: descend() from the change, with
    Evolver.events()'s default factor; a warning if it gets nowhere."""
    from ._analysis import EVENT_FACTOR
    result, _, why = descend(ev, EVENT_FACTOR, falling=False, keep=keep)
    if result is None:
        warnings.warn(f"the surface didn't relax after the change: {why}", UserWarning,
                      stacklevel=3)
    return result
