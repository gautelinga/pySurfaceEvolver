"""Changes of topology made with Evolver's own commands (see
:meth:`Evolver.pinch_off`)."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, List, NamedTuple, Optional

import numpy as np

from . import _core
from ._engine import GUARD_PREFIX, surface
from ._errors import _num

_SEAM = 3.0     # the cut's tips are tidied within this many median edges
_TOUCH = 0.05   # a solid this close (in its event's edges): the surface resting on it
_GUARD_REACH = 5.0  # free vertices are guarded from the walls within this many median edges
_GUARDS = 3         # ... the nearest this many (a free vertex hits at most that many at once)
_POLE_MARGIN = 0.5  # a sphere(axis=)'s pole moved when a contact line comes this near (radii)
_POLE_TRIES = 3     # a run that came within half that done again this often, the poles moved
_POLE_CHUNK = 10    # runs in chunks of this many steps, the poles moved ahead of the lines between

# what the topology runs do, change by change (debug records: changes made and
# refused, descents undone, poles moved, patches closed, why a descent stopped)
_log = logging.getLogger("pysurfaceevolver.topology")

# the topology check (pyse_check.c): problems no change may leave (the first
# group), and shapes a descent can pass through (the second)
_STRUCTURAL = {1: "an edge on no facet", 2: "an edge on more than two facets",
               3: "facets turned against each other across an edge", 4: "a pinched vertex",
               5: "a border edge on no constraint",
               6: "a border edge on a constraint its vertex isn't on",
               7: "a contact-line edge stored against its facet"}
_GEOMETRIC = {8: "a degenerate facet", 9: "a fold", 10: "a vertex inside a solid",
              11: "an edge through a solid", 12: "an edge through a facet"}
_TYPES = {0: "vertex", 1: "edge", 2: "facet"}


class TopologyCheckError(RuntimeError):
    """A change left the surface broken (PYSE_CHECK_TOPOLOGY=1): ``what``
    the change was, ``problems`` what check_topology() found."""

    def __init__(self, what: str, problems: list):
        self.what, self.problems = what, problems
        super().__init__(f"after {what}: " + "; ".join(p[1] for p in problems[:5])
                         + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""))


def check_topology(ev, tol: float = 0.01) -> list:
    """What's wrong with the surface beyond Evolver's own check: a list of
    (code, text, element type, number, other, value); codes 1-7 structural
    (_STRUCTURAL), 8-12 geometric (_GEOMETRIC). The solids are the walls
    (their solid side as events() takes it), a depth over tol mean edges
    counting; edges on three or more facets count with one body or none."""
    from ._edits import walls as _walls
    walls, sides = _walls(ev)
    manifold = _core.count(_core.BODY) <= 1
    _, result = ev._call(_core.check_topology, walls, sides, float(tol), manifold)
    count, rows = result.data
    found = []
    for code, kind, number, other, value in rows:
        text = _STRUCTURAL.get(code) or _GEOMETRIC.get(code, f"problem {code}")
        found.append((code, f"{text} ({_TYPES.get(kind, kind)} {number})", kind, number,
                      other, value))
    if count > len(rows):
        found.append((0, f"... {count - len(rows)} more", -1, 0, 0, 0.0))
    return found


def _checked(ev, what: str) -> None:
    """With PYSE_CHECK_TOPOLOGY=1 (tests, the corpus): the topology check
    after ``what``, raising TopologyCheckError on a structural problem,
    logging geometric ones."""
    if os.environ.get("PYSE_CHECK_TOPOLOGY", "") in ("", "0"):
        return
    found = check_topology(ev)
    structural = [p for p in found if p[0] in _STRUCTURAL]
    geometric = [p for p in found if p[0] in _GEOMETRIC]
    if geometric:
        _log.debug("check after %s: %s", what, "; ".join(p[1] for p in geometric[:5]))
    if structural:
        _log.debug("check failed after %s: %s", what, "; ".join(p[1] for p in structural[:5]))
        raise TopologyCheckError(what, structural)


def _guard_order(gap: np.ndarray) -> np.ndarray:
    """Per vertex, the columns of the walls guard() puts it on: the three
    nearest (those within reach, finite gap, count), nearest first."""
    return np.argsort(gap, axis=1, kind="stable")[:, :_GUARDS]


def guard(ev) -> None:
    """Keep the vertices out of the solids in the descent after a change
    (descend(), not falling: gradient steps move the vertices): each
    wall with a contact angle gets a one-sided copy of its constraint
    (no integrals, named GUARD_PREFIX + its key), put on the vertices not
    fixed, for the walls nearest each within _GUARD_REACH median edges, as
    many as make _GUARDS with the walls it is on (Evolver stops, ERROR 5883,
    where a vertex hits more constraints than the space has dimensions: a
    contact line's vertex, on its wall, in the 0.03 gap between two beads
    hit the next bead's guard and a third); called again as the vertices
    move. Evolver stops a vertex there; the energy and the
    volumes don't change, and the surface resting on the solid is a
    contact to make (a film thinned away). Gradient steps took vertices
    through a sphere beside its contact line, contact after contact, until
    the mesh broke; Newton steps there followed integrals the tangled
    line made wrong (an energy of -76 for 29). Not where vertices are put
    by steps (a fall along a mode, arclength's predictor: one put inside
    stayed on the solid, and the drops a bridge fell into came out on a
    squashed mesh, 0.9% from exact caps; there a vertex through a solid is
    a failed step, or put back out)."""
    walls = {}
    seen = set()
    for k, w in surface.walls.items():
        # (one per solid: move_poles' copies of a sphere are one surface)
        if getattr(w, "contact_angle", None) is not None and getattr(w, "formula", None) \
                and _base(k) not in seen:
            walls[k] = w
            seen.add(_base(k))
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
        ev._register_native_formulas()      # (theirs in closed form too)
        nums, _, dist, names = ev._constraint_gaps(guards=True)
        names = [str(n) for n in names]
    _unguard(ev, names)
    m = ev.mesh()
    real = [i for i, n in enumerate(names) if not n.startswith(GUARD_PREFIX)]
    on = np.isnan(dist[:, real]) if real else np.zeros((len(m.vertices), 0), dtype=bool)
    # (the guards a vertex can take: _GUARDS less the walls it is on, so it
    # hits no more constraints than that at once: unguarded, a contact
    # line's vertex on a side wall went 0.17 out through the side wall it
    # meets at the box's edge, run after run undone)
    room = np.where(np.asarray(m.fixed, dtype=bool), 0, _GUARDS - on.sum(axis=1))
    cols = [_column(nums, names, key) for key in walls]
    h = _edge_scale(ev)
    # (not a vertex through the solid by more than a tenth of an edge: the
    # surface gone through it is a contact to make, and put on the guard,
    # Evolver projected it out: a pack's vertex 0.47 inside a bead, on
    # another bead's guard too, went 3.9 onto both, out of the box)
    signed = dist[:, cols]
    gap = np.abs(np.nan_to_num(signed, nan=np.inf))
    gap[room <= 0] = np.inf
    gap[np.nan_to_num(signed, nan=0.0) < -0.1*h] = np.inf
    gap[gap > _GUARD_REACH*h] = np.inf
    # the three nearest (not two, planes first, as they were: a pack's free
    # vertices in a box's corner, 0.4-0.6 from both side walls and on their
    # guards, went 0.02-0.27 into the bead in the corner, bead8 or bead11, 151
    # runs undone so in one descent, which then ran away; nor the two nearest,
    # as before that: a free vertex by a bead in the corner, guarded against
    # the bead and one side, went 0.23 out through the other)
    order = _guard_order(gap)
    chosen = np.zeros_like(gap, dtype=bool)
    rows = np.arange(len(gap))[:, None]
    chosen[rows, order] = np.isfinite(gap[rows, order]) \
        & (np.arange(order.shape[1])[None, :] < room[:, None])
    # (not two whose normals there are within 10 degrees of each other, nor
    # one within 10 degrees of a wall the vertex is on: a free vertex in the
    # 0.015 gap where a bead touches a side wall, on both guards, left
    # Evolver's constraint projection singular, ERROR 3005; the nearer one
    # only, the wall it is on first)
    normals: dict = {}
    # (only at the vertices that can have two: on a wall and guarded, or
    # guarded twice)
    need = np.ascontiguousarray(np.flatnonzero(
        (chosen.any(axis=1) & on.any(axis=1)) | (chosen.sum(axis=1) >= 2)), dtype=np.int64)

    def normal(num: int) -> np.ndarray:
        if num not in normals:
            normals[num] = np.zeros((len(gap), 3))
            if len(need):
                _, grad = ev._call(_core.constraint_normals_at, num, need)
                normals[num][need] = np.asarray(grad.data)
        return normals[num]

    def tangent(rows_, a, b) -> np.ndarray:
        cos = np.abs(np.einsum("ij,ij->i", a, b)) / np.maximum(
            np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-300)
        return rows_[cos > 0.985]

    guarded = chosen.any(axis=1)
    for c, col in enumerate(real):
        for far in range(order.shape[1]):
            pair = np.flatnonzero(on[:, c] & guarded)
            pair = pair[chosen[pair, order[pair, far]]]
            if len(pair):
                b = np.array([normal(int(nums[cols[int(order[i, far])]]))[i] for i in pair])
                drop = tangent(pair, normal(int(nums[col]))[pair], b)
                chosen[drop, order[drop, far]] = False
    several = np.flatnonzero(chosen.sum(axis=1) >= 2)
    for near in range(order.shape[1]):
        for far in range(near + 1, order.shape[1]):
            pair = several[chosen[several, order[several, near]]
                           & chosen[several, order[several, far]]]
            if not len(pair):
                continue
            a = np.array([normal(int(nums[cols[int(order[i, near])]]))[i] for i in pair])
            b = np.array([normal(int(nums[cols[int(order[i, far])]]))[i] for i in pair])
            drop = tangent(pair, a, b)
            chosen[drop, order[drop, far]] = False
    for j, key in enumerate(walls):
        if chosen[:, j].any():
            ev.set_constraint("vertex", f"{GUARD_PREFIX}{key}", where=chosen[:, j])


def _pole_lines(ev):
    """For each sphere(axis=) with a contact line: (key, wall, the
    constraint's name or number, centre, radius, the line's vertices on the
    unit sphere about the centre, the least distance of its edges from the
    pole, the solid angle its edges span from the antipode, the way the
    facets go round them, and its edges: their ends on the unit sphere and
    their facets' bodies; which vertices are on it; and the edges as rows of
    vertices, as the facets go). (The edges'
    distance, not the vertices': Evolver integrates along the chords, and a
    pack's line whose vertices were 0.47
    from a pole had a vertex at 0.08 once its edges were split. The solid
    angle, the sum of the triangles from the antipode to the edges, jumps by
    4 pi where the line crosses the pole, as the integrals do.)"""
    if not any(getattr(w, "south_regular", False) for w in surface.walls.values()):
        return []       # (no poles: sphere(gauge="flux") and planes; 6% of a pack's descent)
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
                    float(_fans(a, b, pole).sum()), a, b, bodies[mine], on, e))
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


def move_poles(ev, force=(), ahead=None) -> list:
    """For each sphere(axis=) with a contact line nearer than _POLE_MARGIN of
    its radius to its north pole (where its integrals are singular), and
    those whose keys are in force, the pole moved away: to the point of the
    sphere farthest from the line (and, for those in force, from the way its
    vertices go to ``ahead``, the vertices where a run took them: a pack's
    line swept across a bead in one relax() run, and the pole moved to the
    point farthest from the line where it began was the same point each time
    the run was redone). A new constraint with that axis (Evolver doesn't
    redefine one), and the volumes (volconst) and the energy
    (_SurfaceMemory.energy_shift) shifted by the whole spheres the integrals
    about the two poles differ by (a contact line stepping across the pole
    moved the wetted area by the whole sphere, 4 pi cos theta, and the
    volume constraint then drove the surface off). Per contact-line loop: a
    sphere's loop that needs it moves alone onto a copy of the sphere's
    constraint with its own pole (_pole_copy), the others stay; where all
    of them do, the sphere's constraint is replaced (a heavily wetted bead,
    lines all over it, had no point clear of them all, and its one pole
    moved 96 times in 12 steps). Returns the walls made."""
    moved = []
    x = ev.vertices
    # (where the run took each vertex, by id: ``ahead`` is _places' of its end)
    there_all = None
    if ahead is not None:
        ids, y = ahead
        by_id = {int(i): k for k, i in enumerate(ids)}
        rows_ahead = np.array([by_id.get(int(i), -1) for i in ev.mesh().vertex_ids])
        there_all = np.where((rows_ahead >= 0)[:, None], y[np.maximum(rows_ahead, 0)], x)
    for key, wall, old, centre, R, line, near, _, a, b, sides, on, e in _pole_lines(ev):
        if near >= _POLE_MARGIN and key not in force:
            continue
        pole = np.asarray(getattr(wall, "pole_axis"), float)
        unit = lambda p: (p - centre)/R     # noqa: E731
        swept = key in force and there_all is not None
        loops = []
        for rows in _loop_rows(e):
            vs = np.unique(e[rows])
            points = unit(x[vs])
            if swept and there_all is not None:
                there = unit(there_all[vs])
                way = [(1 - t)*points + t*there for t in (0.25, 0.5, 0.75, 1.0)]
                points = np.concatenate([points] + [w/np.linalg.norm(w, axis=1)[:, None]
                                                    for w in way])
            reach = float(np.linalg.norm(points - pole, axis=1).min())
            loops.append((rows, vs, points, _chord_reach(a[rows], b[rows], pole), reach))
        needs = [lp for lp in loops if lp[3] < _POLE_MARGIN or (swept and lp[4] < _POLE_MARGIN)]
        if not needs:
            needs = loops if key in force else []
        if not needs:
            continue
        if len(needs) == len(loops):
            # all of them: the sphere's constraint replaced, as one
            points = np.concatenate([lp[2] for lp in loops])
            axis, clear = _farthest(points)
            if key not in force and clear < 1.5*near:
                continue        # (no farther place: a bead's pole moved 19 times)
            name = _pole_copy(ev, key, wall, centre, R, axis)
            ev.command(f"set vertex constraint {name} where on_constraint {old}; "
                       f"set edge constraint {name} where on_constraint {old}; "
                       f"unset vertex constraint {old} where on_constraint {name}; "
                       f"unset edge constraint {old} where on_constraint {name}")
            del surface.walls[key]
            surface.retired.add(old)
            _shift_spheres(ev, wall, R, a, b, sides, pole, axis)
            moved.append(name)
            _log.debug("pole moved: %s to %s (line %.3g radii from the pole%s)", old, name, near,
                       ", forced" if key in force else "")
            continue
        m = ev.mesh()
        index = {tuple(sorted(map(int, pair))): k for k, pair in enumerate(np.asarray(m.edges))}
        for rows, vs, points, chord, _ in needs:
            axis, clear = _farthest(points)
            if key not in force and clear < 1.5*chord:
                continue
            name = _pole_copy(ev, key, wall, centre, R, axis)
            _move_loop(ev, m, index, old, name, vs, e[rows])
            _shift_spheres(ev, wall, R, a[rows], b[rows], sides[rows], pole, axis)
            moved.append(name)
            _log.debug("pole of a loop moved: %s to %s (line %.3g radii from the pole%s)", old,
                       name, chord, ", forced" if key in force else "")
    return moved


def _move_loop(ev, m, index, src, dst, vs, pairs) -> None:
    """A contact line's vertices (rows vs of m) and edges (pairs of vertex
    rows; index: m's edge rows by their ends) from constraint src onto dst."""
    v_mask = np.zeros(len(m.vertices), dtype=bool)
    v_mask[vs] = True
    e_mask = np.zeros(len(m.edges), dtype=bool)
    e_mask[[index[tuple(sorted(map(int, pair)))] for pair in pairs]] = True
    ev.set_constraint("vertex", dst, where=v_mask)
    ev.set_constraint("edge", dst, where=e_mask)
    ev.set_constraint("vertex", src, where=v_mask, on=False)
    ev.set_constraint("edge", src, where=e_mask, on=False)


def _onto_copy(ev, event) -> None:
    """Before a contact on a copy of a solid (move_poles' per-loop poles):
    the contact lines on its other copies near the event (within three
    edges, or among its vertices) moved onto its copy, so the change makes
    its lines on one constraint (lines on two copies of a bead meeting are
    one solid's); a copy left without any taken out of sight (retired)."""
    key = event.constraint
    sibs = [k for k in surface.walls if k != key and _base(k) == _base(key)]
    if not sibs or key not in surface.walls:
        return
    lines = {p[0]: p for p in _pole_lines(ev)}
    x = ev.vertices
    where = np.asarray(event.where, float)
    radius = 3*max(float(event.scale), float(event.gap))
    mine = set(int(v) for v in np.asarray(event.vertices, int))
    target = surface.walls[key]
    m, index = None, None
    for k in sibs:
        if k not in lines:
            continue
        _, wall, old, centre, R, _, _, _, a, b, sides, on, e = lines[k]
        for rows in _loop_rows(e):
            vs = np.unique(e[rows])
            if not ((np.linalg.norm(x[vs] - where, axis=1) < radius).any()
                    or mine & set(int(v) for v in vs)):
                continue
            if m is None:
                m = ev.mesh()
                index = {tuple(sorted(map(int, pair))): j
                         for j, pair in enumerate(np.asarray(m.edges))}
            _move_loop(ev, m, index, old, key, vs, e[rows])
            _shift_spheres(ev, wall, R, a[rows], b[rows], sides[rows],
                           np.asarray(getattr(wall, "pole_axis"), float),
                           np.asarray(getattr(target, "pole_axis"), float))
            _log.debug("contact line moved onto %s from %s (one solid)", key, k)
        if m is not None and not ev.on_constraint(k).any():
            del surface.walls[k]
            surface.retired.add(old)


def _loop_rows(e) -> list:
    """The edges (rows of e, pairs of vertex rows) of each contact-line loop
    (each connected stretch of line)."""
    if not len(e):
        return []
    parent: dict = {}

    def root(v):
        while parent.setdefault(v, v) != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for i, j in e:
        parent[root(int(i))] = root(int(j))
    groups: dict = {}
    for k, (i, _) in enumerate(e):
        groups.setdefault(root(int(i)), []).append(k)
    return [np.asarray(g) for g in groups.values()]


def _chord_reach(a, b, pole) -> float:
    """The least distance of the chords a->b (on the unit sphere) from the
    pole."""
    if not len(a):
        return float("inf")
    t = np.clip(np.einsum("ij,ij->i", pole - a, b - a)
                / np.maximum(np.einsum("ij,ij->i", b - a, b - a), 1e-300), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None]*(b - a) - pole, axis=1).min())


def _farthest(points) -> "tuple[np.ndarray, float]":
    """The point of the unit sphere farthest from the points, and how far."""
    n = 400
    k = np.arange(n) + 0.5
    polar, turn = np.arccos(1 - 2*k/n), np.pi*(1 + 5**0.5)*k
    cand = np.c_[np.cos(turn)*np.sin(polar), np.sin(turn)*np.sin(polar), np.cos(polar)]
    reach = np.linalg.norm(cand[:, None, :] - points[None, :, :], axis=2).min(axis=1)
    best = int(np.argmax(reach))
    return cand[best], float(reach[best])


def _pole_copy(ev, key, wall, centre, R, axis) -> str:
    """A new constraint, the sphere of wall key with its pole along axis
    (named <base>_pole<n>), defined and made a wall (its old one stays)."""
    from . import constraints as C
    new = C.sphere(tuple(centre), R, contact_angle=getattr(wall, "contact_angle"),
                   axis=tuple(axis))
    base = _base(key)
    count = 1 + max([int(str(k2).rsplit("_pole", 1)[1]) for k2 in surface.walls
                     if str(k2).startswith(f"{base}_pole")] or [0])
    # (past names Evolver still defines too: those of moves undone since)
    while ev.eval(f'is_defined("{base}_pole{count}")') != 0:
        count += 1
    name = f"{base}_pole{count}"
    ev.command('exec "define constraint ' + name + " " + " ".join(new.text().split()) + '"')
    surface.walls[name] = new
    ev._register_native()
    return name


def _base(key) -> str:
    """The solid a wall is a copy of (copies differ only in their poles)."""
    return str(key).split("_pole")[0] if isinstance(key, str) else f"c{key}"


def _shift_spheres(ev, wall, R, a, b, sides, old_axis, new_axis) -> None:
    """The volumes (volconst) and energy (energy_shift) shifted by the whole
    spheres the integrals of the edges a->b about the two poles differ by
    (where a loop goes round one pole and not the other), as the edges'
    solid angles from the antipodes count them. (Not what the integrals
    give: off by their error, more than half a sphere for a line near the
    old pole, 0.05 of a radius, or with edges 1.6 long.)"""
    spheres = (_fans(a, b, new_axis) - _fans(a, b, old_axis))/(4*np.pi)
    for i in [int(i) for i in ev.bodies().ids]:
        side = np.where(sides[:, 0] == i, 1, np.where(sides[:, 1] == i, -1, 0))
        if side.any():
            body = ev.body(i)
            body.volconst = body.volconst + 4*np.pi*R**3/3*round(float(side @ spheres))
    cos = np.cos(np.radians(getattr(wall, "contact_angle")))
    surface.energy_shift.append((1, _num(4*np.pi*R*R*cos*round(float(spheres.sum())))))


def _away_from_poles(ev, run, n: int) -> tuple:
    """run(n) from here (a run of n steps), in chunks of _POLE_CHUNK: after
    each, the poles of sphere(axis=)s whose contact lines have come within
    _POLE_MARGIN moved (move_poles), away from the lines and from where they
    are going (as far again as the chunk took them, twice), so a line
    sweeping across a bead keeps clear of its pole as it goes (a pack's
    falls swept lines across whole beads in a few runs; undone and redone
    whole, a bead's pole moved 72 times in one descent and the descent ran
    away). A chunk that still brought a line within half _POLE_MARGIN of a
    pole, or took it across (_crossed), is done again from its start with
    those poles moved (away from where it took the lines), shorter where it
    does so again (a quarter as many steps each time, down to one). Returns
    run's result (its last chunk's), and the state the run started from
    with the energy, the surface's crossings of itself, the volumes' gap
    from their targets and the vertices through a wall there (for
    _plausible)."""
    from ._errors import EvolverError
    start, energy = ev._save(), (ev.energy, _crossings(ev), ev._target_gap()[0], _outside(ev))
    _RUN_FAILED.clear()
    result: Any = None
    left, k, tries = n, min(n, _POLE_CHUNK), 0
    while left > 0:
        here, before = ev._save(), _places(ev)
        fans = {p[0]: _line_state(ev, p) for p in _pole_lines(ev)}
        step = min(k, left)
        try:
            result = run(step)
        except EvolverError as error:
            # (Evolver failing in a run, an internal error on a mesh gone bad:
            # the run judged gone wrong, back where it began)
            _log.debug("run failed: %s", str(error).splitlines()[0])
            ev.restore(start)
            _RUN_FAILED.append(str(error).splitlines()[0])
            return _unconverged(), start, energy
        bad = [p[0] for p in _pole_lines(ev) if p[6] < _POLE_MARGIN/2
               or (p[0] in fans and _crossed(ev, fans[p[0]], p))]
        if bad and tries < _POLE_TRIES + 3:
            tries += 1
            _log.debug("run undone: a contact line came near or across a pole (%s)", bad)
            ahead = _places(ev)
            ev.restore(here)
            move_poles(ev, force=bad, ahead=ahead)
            if tries > 1 and k > 1:
                k = max(1, k//4)
                _log.debug("runs shortened to %d steps (lines sweeping across poles)", k)
            continue
        left -= step
        if left <= 0 or getattr(result, "converged", False):
            break
        coming = [p[0] for p in _pole_lines(ev) if p[6] < _POLE_MARGIN]
        if coming:
            ids, x = _places(ev)
            move_poles(ev, force=coming, ahead=(ids, x + 2*_moved_since(ev, before)))
    return result, start, energy


def _line_state(ev, p) -> tuple:
    """What _crossed compares a run's end with: a sphere's lines' solid
    angle (p[7]), their least distance from the pole, and their vertices'
    ids and places."""
    rows = np.flatnonzero(p[11])
    m = ev.mesh()
    return p[7], p[6], np.asarray(m.vertex_ids)[rows], np.asarray(m.vertices)[rows], float(p[4])


def _crossed(ev, before: tuple, p) -> bool:
    """Whether a run may have taken a sphere's line across its pole: the
    solid angle the lines bound, fanned from the pole's antipode, changed by
    over 2 pi (it jumps by 4 pi at a crossing), and the lines' vertices moved
    as far as the pole was from them (a line drying half a bead in one run
    changes the angle as much: a pack's bead's pole moved 45 times on that,
    in runs of 12 steps, its line 1.1 radii away)."""
    fans, near, ids, x, R = before
    if abs(p[7] - fans) <= 2*np.pi:
        return False
    # (by the vertices' ids: Evolver gives a split's new vertex a freed id,
    # and the rows after it shift; compared by row, a line's vertices seemed
    # to cross the bead, and a pack's bead's pole moved 338 times)
    m = ev.mesh()
    now = {int(i): k for k, i in enumerate(m.vertex_ids)}
    kept = [(k, now[int(i)]) for k, i in enumerate(ids) if int(i) in now]
    if not kept:
        return True
    a, b = np.asarray(kept).T
    moved = float(np.linalg.norm(np.asarray(m.vertices)[b] - x[a], axis=1).max())/R
    return bool(moved >= near)


def _places(ev) -> tuple:
    """The vertices' ids and places, to compare a later state with
    (_moved_since)."""
    return np.asarray(ev.mesh().vertex_ids), ev.vertices


def _moved_since(ev, before: tuple) -> np.ndarray:
    """How far each vertex (as rows now) has moved since ``before``
    (_places), by id: Evolver gives a split's new vertex a freed id, so the
    rows after it shift and rows don't match. 0 for vertices new since."""
    ids, x = before
    m = ev.mesh()
    now, here = np.asarray(m.vertex_ids), np.asarray(m.vertices, float)
    old = {int(i): k for k, i in enumerate(ids)}
    rows = np.array([old.get(int(i), -1) for i in now])
    out = np.zeros_like(here)
    seen = rows >= 0
    out[seen] = here[seen] - x[rows[seen]]
    return out


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
    degenerated (0.3 degrees). (Only the guards some vertex is on, in one
    command: a command per guard, 21 a call, took 6% of a 16-bead run.)"""
    _, _, dist, all_names = ev._constraint_gaps(guards=True)
    used = [str(n) for c, n in enumerate(all_names)
            if str(n).startswith(GUARD_PREFIX) and np.isnan(dist[:, c]).any()
            and (names is None or n in names)]
    if used:
        ev.command("; ".join(f"unset vertex constraint {n}" for n in used))


def pinch_off(ev, event, relax: bool = True, at_wall: bool = True) -> List[int]:
    """Cut the surface at a neck: see :meth:`Evolver.pinch_off`. Without
    ``at_wall``, a sheet thinnest at a wall ruptures by a hole beside its
    contact line, not by its footprint dried (_descend: where a footprint
    was dried at the same place already)."""
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
    if _sheet(x, f, event):
        # (a sheet of liquid thinning, gas either side, not a tube: it
        # ruptures, a hole opening through it at its thinnest, the liquid
        # one body round it. Cut as a tube, along an axis its normals don't
        # fix, a pack's sheet between two beads was cut in a band across
        # it from bead to bead, its footprint on one dried, and the sheet,
        # as thin, was cut so again ten times)
        p = np.asarray(event.where, float)
        reach = 2*float(event.gap) + float(event.scale)
        start = ev._save()
        try:
            at_wall = at_wall and _sheet_at_wall(ev, m, x, event)
        except ValueError as error:
            ev.restore(start)
            at_wall = False
            _log.debug("the sheet's footprint on the wall not dried: %s", error)
        if at_wall:
            if relax and _settle(ev) is None:
                warnings.warn("pinch_off(): the surface didn't relax after the sheet ruptured "
                              "at the wall (it is left out of equilibrium)", UserWarning,
                              stacklevel=3)
            return []
        m = ev.mesh()
        x, f = np.asarray(m.vertices, float), np.asarray(m.facets)
        if not _sheet_hole(ev, m, x, f, p, reach):
            # (the sheet thinnest where it meets a wall, its footprint there
            # not dried, the facets there on its contact line, which the
            # merge doesn't open: a pack's sheet against a side wall,
            # refused run after run. Then the nearest
            # pair with no edge on the border nor on a constraint: the hole
            # opens beside the line and grows to it)
            inner = _inner_facets(ev, m)
            if not inner.any() or not _sheet_hole(ev, m, x, f, p, reach + 2*float(event.scale),
                                                  allowed=inner):
                raise ValueError("pinch_off(): the sheet's hole refused (the checked merge)")
            _log.debug("the sheet's hole opened beside its contact line")
        _log.debug("a sheet, not a neck: a hole opened through it")
        if relax and _settle(ev) is None:
            warnings.warn("pinch_off(): the surface didn't relax after the hole opened (it is "
                          "left out of equilibrium)", UserWarning, stacklevel=3)
        return []
    ring, centre, side = _ring(x, f, edges, event)
    if len(ring) < 3:
        raise ValueError("pinch_off(): no ring of edges round the neck found")
    border = np.zeros(len(x), dtype=bool)
    border[edges[m.edge_valence() == 1].ravel()] = True
    _, _, gaps, _ = ev._constraint_gaps()
    on = np.isnan(gaps) if gaps.size else np.zeros((len(x), 0), dtype=bool)
    held = m.fixed | border | on.any(axis=1)
    walls = np.flatnonzero(on[ring].any(axis=0))
    if len(walls) == 1:
        # (against one wall, its axis along the wall, as the neck mirrored
        # in it is whole: its waist's vertices, those facing each other
        # across it, are the two flanks beside the wall, their normals along
        # the wall, and the axis came out across it: the ring, cut by a plane
        # along the wall, ran beside the wall on both flanks, not over the
        # neck, and the cut squashed them together, leaving the neck: a
        # 16-bead pack's neck on a side wall was cut six times in a descent,
        # the body one piece still each time, the neck back where it was
        # cut, until the descent ran away)
        nums, _, _, _ = ev._constraint_gaps()
        at_wall = ring[on[ring, walls[0]]]
        k = at_wall[np.argmin(np.linalg.norm(x[at_wall] - centre, axis=1))]
        _, grad = ev._call(_core.constraint_normals, int(nums[walls[0]]))
        normal = np.asarray(grad.data, float)[k]
        if np.linalg.norm(normal) > 0:
            ring, centre, side = _ring(x, f, edges, event, normal/np.linalg.norm(normal))
            if len(ring) < 3:
                raise ValueError("pinch_off(): no ring of edges round the neck found")
            walls = np.flatnonzero(on[ring].any(axis=0))
    if held[ring].any():
        # a neck against one wall (half a neck, its ring from the wall's
        # contact line round to it again: liquid against a side wall or a
        # bead, 86 and 75 of the 161 necks a round of 16-bead runs refused):
        # the ring collapses to its vertex on the wall nearest its middle, the
        # contact line then goes through that vertex twice, and the
        # separation splits it, a line per side
        lone = (len(walls) == 1 and not m.fixed[ring].any()
                and (on[ring].sum(axis=1) <= 1).all()
                and not (border[ring] & ~on[ring].any(axis=1)).any())
        if not lone:
            two = (len(walls) == 2 and not m.fixed[ring].any()
                   and (on[ring].sum(axis=1) <= 1).all()
                   and not (border[ring] & ~on[ring].any(axis=1)).any())
            if two:
                # a neck against two solids (a bead and a side wall, liquid in
                # the wedge between them): its footprint on one of them dried
                # across (the two strands of contact line either side of it
                # joined over the wet ground: contact()); the liquid then
                # joins only over the other, whose neck comes next
                _dry_footprint(ev, m, x, ring, walls, gaps, event)
                return []
            raise ValueError("pinch_off(): the neck reaches a fixed vertex, two constraints "
                             "or the surface's free border")
        at_wall = ring[on[ring, walls[0]]]
        centre = x[at_wall[np.argmin(np.linalg.norm(x[at_wall] - centre, axis=1))]]
    made = _pinch_ring(ev, m, x, f, ring, centre, side, float(event.scale))
    if relax and _settle(ev) is None:
        warnings.warn("pinch_off(): the pieces didn't relax after the cut (the surface is "
                      "left cut, out of equilibrium)", UserWarning, stacklevel=3)
    return made


def _pinch_ring(ev, m, x, f, ring, centre, side, scale: float) -> List[int]:
    """pinch_off's cut: the ring (rows of m's vertices) collapsed to centre,
    the body split where that makes it two pieces (side: each vertex's side
    of the cut, by sign). The bodies made."""
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
    h = float(np.median(lengths)) if len(lengths) else float(scale)
    snapshot = ev._save()
    try:
        _cut(ev, x, ring, centre, scale)
        made = _split_body(ev, body, before, near, lost) if body else []
        _tidy_seam(ev, centre, h)
    except Exception:
        ev.restore(snapshot)
        raise
    return made


def _dry_footprint(ev, m, x, ring, walls, gaps, event, what="neck against two solids",
                   make=None) -> None:
    """A neck's footprint on one of the two solids its ring reaches dried
    across: on each, the closest pair of the ring's vertices on that solid
    from the two strands of its contact line (far apart along the line, or
    on different lines); the narrower footprint's joined over the wet ground
    between them (contact()'s wet strip: the neck's surface over it laid
    onto the solid), else the other's. (A sheet's footprint on the wall it
    is thinnest at likewise, made by ``make``, the "contact lines" event
    given: what names the case.)"""
    from ._events import Event
    from ._geometry import _chains
    nums, _, _, names = ev._constraint_gaps()
    edges, valence = np.asarray(m.edges), m.edge_valence()
    on = np.isnan(gaps)
    pairs = []
    for c in walls:
        line = edges[(valence == 1) & on[edges[:, 0], c] & on[edges[:, 1], c]]
        pos: dict = {}
        for k, path in enumerate(_chains(line)):
            p = np.asarray(path)
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(x[p], axis=0), axis=1))]
            for v, t in zip(p, s):
                pos.setdefault(int(v), (k, float(t)))
        mine = [int(v) for v in ring[on[ring, c]] if int(v) in pos]
        best = None
        for i in mine:
            for j in mine:
                if i < j:
                    (ki, si), (kj, sj) = pos[i], pos[j]
                    d = float(np.linalg.norm(x[i] - x[j]))
                    if (ki != kj or abs(si - sj) > 3*float(event.scale)) \
                            and (best is None or d < best[0]):
                        best = (d, i, j)
        if best is not None:
            pairs.append((best, names[c] or int(nums[c])))
    why = "no strands either side of it"
    for (d, i, j), key in sorted(pairs, key=lambda p: p[0][0]):
        join = Event("contact lines", d, float(event.scale), (x[i] + x[j])/2,
                     np.array([i, j]), key)
        start = ev._save()
        try:
            if make is not None:
                made = make
            else:
                c = next(c for c in walls if (names[c] or int(nums[c])) == key)
                made = contact if _domed(ev, m, x, i, j, int(nums[c]), key,
                                         float(event.scale)) else _join_across
            made(ev, join)
            _log.debug("%s: its footprint on %s dried across", what, key)
            return
        except ValueError as error:
            ev.restore(start)
            why = str(error)
    raise ValueError(f"pinch_off(): the {what}: no footprint dried ({why})")


def _domed(ev, m, x, a: int, b: int, num: int, key, h: float) -> bool:
    """Whether the liquid over the strip of a solid between contact-line
    vertices a and b (rows; num, key: the solid's constraint) is closed off
    close to it, as under a neck lying against it (the surface arching over
    the strip, or the neck's other solid across a wedge): a ray from the
    strip's middle along the solid's normal meets a facet or another solid
    within three times the strip's width and two edges. Else they are a
    sheet's two faces rising off the solid, the liquid between them running
    out far: its footprint is cut by joining the lines (_join_across), not
    by laying the surface over it onto the solid (contact(), which wetted
    more of the solid there: a pack's sheet in the corner of two side walls,
    a neck by its normals, the surface then through itself). (A path along
    the surface from a to b over the strip, the first test, missed a neck in
    the wedge between a side wall and a bead: its ring is cut by both solids,
    the lines were joined, and the descent made 60 contacts and never
    settled.)"""
    xa, xb = x[a], x[b]
    width = float(np.linalg.norm(xa - xb))
    reach = 3*width + 2*h
    p = (xa + xb)/2
    _, grad = ev._call(_core.constraint_normals, num)
    g = np.asarray(grad.data, float)
    n = g[a] + g[b]
    if not np.linalg.norm(n):
        return True
    n = n/np.linalg.norm(n)
    f = np.asarray(m.facets)
    at = np.isin(f, [a, b]).any(axis=1)
    if at.any() and float((x[f[at]].mean(axis=(0, 1)) - p) @ n) < 0:
        n = -n                                  # (towards the surface's side)
    # the facets: Moller-Trumbore, for those within reach
    near = np.linalg.norm(x[f].mean(axis=1) - p, axis=1) < reach + 2*h
    if near.any():
        t0, t1, t2 = x[f[near, 0]], x[f[near, 1]], x[f[near, 2]]
        e1, e2 = t1 - t0, t2 - t0
        q = np.cross(n, e2)
        det = np.einsum("ij,ij->i", e1, q)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0/np.where(ok, det, 1.0), 0.0)
        s0 = p - t0
        u = np.einsum("ij,ij->i", s0, q)*inv
        r = np.cross(s0, e1)
        v = (r @ n)*inv
        t = np.einsum("ij,ij->i", e2, r)*inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9*reach) & (t <= reach)
        if hit.any():
            return True
    # the other solids: spheres and planes (constraints' shapes)
    for k, w in surface.walls.items():
        shape = getattr(w, "shape", None)
        if k == key or shape is None or isinstance(shape[1], str):
            continue
        if shape[0] == "sphere" and not any(isinstance(c, str) for c in shape[1]):
            c, R = np.asarray(shape[1], float), float(shape[2])
            dc = p - c
            bq = float(dc @ n)
            disc = bq*bq - (float(dc @ dc) - R*R)
            if disc >= 0:
                t = -bq - np.sqrt(disc)
                if 1e-9*reach < t <= reach:
                    return True
        elif shape[0] == "plane":
            nq = np.asarray(shape[1], float)
            along = float(nq @ n)
            if abs(along) > 0.1:
                t = (float(shape[2]) - float(nq @ p))/along
                if 1e-9*reach < t <= reach:
                    return True
    return False


def _sheet_hole(ev, m, x, f, p, reach: float, allowed=None) -> bool:
    """A hole opened through a sheet at p by the checked merge (_tube): of
    the facets within reach of p, nearest first, each with the facet facing
    it across the sheet, up to _HOLE_TRIES pairs (the nearest pair refused,
    the mesh folded there: a pack's sheet 0.00075 thin refused four times
    so, the surface then through itself in every run, and the descent gave
    up). Whether it opened."""
    centre = x[f].mean(axis=1)
    near = np.linalg.norm(centre - p, axis=1) <= reach
    ok = near if allowed is None else (near & allowed)
    for _ in range(_HOLE_TRIES):
        if not ok.any():
            return False
        try:
            a, b = _facing_facets(x, f, p, behind=True, reach=reach, allowed=ok)
        except ValueError:
            return False
        if _tube(ev, m, a, b, None, sheet=True) is not None:
            return True
        ok[a] = False
    return False


def _sheet_at_wall(ev, m, x, event) -> bool:
    """A sheet thinnest where it meets a wall ruptured there: its footprint
    on the wall dried across between the strands of contact line its two
    faces end in (_dry_footprint). At a side wall at 90 degrees (a mirror)
    the sheet is half of one twice as wide, whose hole opens in its middle,
    on the mirror: half that hole is this. Whether the sheet is at a wall
    (False: no wall within its gap and an edge of its thinnest point);
    ValueError where no footprint could be dried."""
    _, _, gaps, _ = ev._constraint_gaps()
    if not gaps.size:
        return False
    on = np.isnan(gaps)
    p = np.asarray(event.where, float)
    d = np.linalg.norm(x - p, axis=1)
    reach = float(event.gap) + float(event.scale)
    walls = [c for c in range(on.shape[1]) if (on[:, c] & (d < reach)).any()]
    if not walls:
        return False
    near = np.flatnonzero(d < 2*reach)
    _dry_footprint(ev, m, x, near, walls, gaps, event, "sheet against a wall", _join_across)
    return True


def _join_across(ev, event) -> None:
    """A "contact lines" event's two strands joined across the wet strip
    between them, as two lines across dry ground are (_bridge_lines, else
    _join_lines; then the vertex the line goes through twice separated):
    the footprint of a sheet at the wall it is thinnest at cut in two, the
    gas either side of the sheet meeting over the wall there. (contact()
    lays the surface over a wet strip onto the solid, where the liquid is
    under it, a neck's dome; a sheet's faces rise off the wall with gas
    under them, and laid onto it they wetted the wall more.)"""
    ids = np.asarray(ev.mesh().vertex_ids)[np.asarray(event.vertices, int)]
    _onto_copy(ev, event)
    nums, _, _, names = ev._constraint_gaps()
    num = int(nums[_column(nums, names, event.constraint)])
    m = ev.mesh()
    row = {int(v): k for k, v in enumerate(m.vertex_ids)}
    a, b = (row[int(v)] for v in ids)
    x, f = np.asarray(m.vertices, float), np.asarray(m.facets)
    if not _bridge_lines(ev, m, num, a, b, None):
        _join_lines(ev, m, x, f, a, b, None)
    _separate_pinches(ev, num)


def _waist_normals(x, f, event) -> np.ndarray:
    """The unit vertex normals at a neck's waist: its vertices within its
    gap and two edges of its narrowest point (_events.waist), else those
    near it."""
    p = np.asarray(event.where, float)
    cross = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    normal = np.zeros_like(x)
    for k in range(3):
        np.add.at(normal, f[:, k], cross)
    length = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, length[:, None], out=np.zeros_like(normal),
                       where=length[:, None] > 0)
    from ._events import waist as near_waist
    waist = near_waist(x, np.asarray(event.vertices, int), p, float(event.gap),
                       float(event.scale))
    if len(waist) < 3:
        waist = np.flatnonzero(np.linalg.norm(x - p, axis=1) < max(event.gap, 2*event.scale))
    return normal[waist]


def _sheet(x, f, event) -> bool:
    """Whether a neck is a sheet thinning rather than a tube (_events.sheet
    of its waist's normals)."""
    from ._events import sheet
    return sheet(_waist_normals(x, f, event))


def _ring(x, f, edges, event, mirror=None):
    """The vertices to collapse at a neck: the ends of the edges crossing the
    plane through the neck's narrowest point, across its axis, there (two
    rows; collapsing one left slivers that didn't relax); the point they
    collapse to (their centroid); and each vertex's signed distance from the
    plane. ``mirror``: the unit normal of a wall the neck lies against (the
    axis then along the wall)."""
    p = np.asarray(event.where, float)
    reach = max(3*event.gap, 3*event.scale)
    # the axis: the direction the normals round the waist are all across
    waist = np.asarray(event.vertices, int)
    if len(waist) < 3:
        waist = np.flatnonzero(np.linalg.norm(x - p, axis=1) < max(event.gap, 2*event.scale))
    normals = _waist_normals(x, f, event)
    moment = normals.T @ normals
    if mirror is not None:
        # (along the wall: the direction across it never the axis)
        along = np.eye(3) - np.outer(mirror, mirror)
        moment = along @ moment @ along + (np.trace(moment) + 1.0)*np.outer(mirror, mirror)
    _, vectors = np.linalg.eigh(moment)
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
    f = np.asarray(m.facets)
    # (a neck against a wall: its ring's two ends on the wall's contact line
    # stay two vertices, Evolver deleting no edge between two of a border's
    # vertices that would pinch it; that edge, between the sides, made two
    # contact-line edges, one per side, and the two vertices separated, one
    # per side (the checked unzip, _edits): left, it held the sides
    # together, the cut one piece, its two vertices pinched)
    border = np.zeros(len(m.vertices), dtype=bool)
    edges = np.asarray(m.edges)
    border[edges[m.edge_valence() == 1].ravel()] = True
    if len(at) != 1 and not (len(at) == 2 and border[at].all()):
        raise ValueError(f"pinch_off(): the ring didn't collapse to one vertex ({len(at)} left)")
    if len(at) == 2:
        from ._edits import REASONS, edit
        joint = np.flatnonzero(np.isin(edges, at).all(axis=1))
        if len(joint) > 1:
            raise ValueError("pinch_off(): the neck's ends on the wall are joined by "
                             f"{len(joint)} edges")
        if len(joint):
            r = edit(ev, "unzip", [int(np.asarray(m.edge_ids)[joint[0]])])
            if r[0] != 0:
                raise ValueError("pinch_off(): the neck's ends on the wall can't be parted "
                                 f"({REASONS[int(r[0])]})")
            return
    sides = [_sides_at(f, int(k)) for k in at]
    if len(at) == 1 and sides[0] < 2:
        raise ValueError("pinch_off(): collapsing the ring didn't pinch the surface")
    # one vertex per cone (the checked separation, _edits); else Evolver's pop
    # (a wall's two vertices: those with two cones; one with one is already
    # its piece's alone)
    from ._edits import REASONS, edit
    ids = [int(i) for i, n in zip(np.asarray(m.vertex_ids)[at], sides) if n >= 2]
    if not ids:
        return
    r = edit(ev, "separate", ids)
    for v, why in zip(ids, r):
        if why != 0:
            if len(ids) > 1:
                raise ValueError("pinch_off(): the neck's ends on the wall can't be separated "
                                 f"({REASONS[int(why)]})")
            _log.debug("fallback: pinch split by pop (the separation refused: %s)",
                       REASONS[int(why)])
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


def _rejoin_strays(ev, body: int, old: set, near: dict) -> None:
    """After a cut's rebody: the pieces of the body it made that no facet
    by the cut is in (pieces the liquid joins through the walls before it,
    not split by the cut) joined again to the piece by the cut they share
    the most walls with (else the largest), under body's number where
    either had it."""
    pieces = [body] + sorted(set(int(i) for i in ev.bodies().ids) - old)
    if len(pieces) < 2:
        return
    m = ev.mesh()
    ids, fb = np.asarray(m.facet_ids), np.asarray(m.facet_bodies)
    by_cut = np.concatenate([np.asarray(v) for v in near.values()]) if near else np.zeros(0)
    cut = [i for i in pieces if np.isin(ids[(fb == i).any(axis=1)], by_cut).any()]
    strays = [i for i in pieces if i not in cut]
    if not cut or not strays:
        return
    walls = {i: set(_walls_touched(ev, i)) for i in pieces}
    size = {i: int((fb == i).any(axis=1).sum()) for i in pieces}
    for i in strays:
        into = max(cut, key=lambda j: (len(walls[i] & walls[j]), size[j]))
        if i == body:
            _fold(ev, into, i)
            cut = [i if j == into else j for j in cut]
            walls[i] |= walls[into]
        else:
            _fold(ev, i, into)
            walls[into] |= walls[i]


def _split_body(ev, body: int, before: dict, near: dict, lost: dict) -> List[int]:
    """After the cut: the body's pieces made bodies (Evolver's ``rebody``;
    none where the cut took a handle off it), the largest keeping its number; each piece's volconst from the walls it
    reaches; and their targets, if the volume was fixed: each smaller piece
    what it holds plus what the collapse took from its side, the largest the
    rest of the old target, so they add up to it."""
    old = set(int(i) for i in ev.bodies().ids)
    _rebody(ev, split=True)
    _rejoin_strays(ev, body, old, near)
    new = sorted(set(int(i) for i in ev.bodies().ids) - old)
    if not new:
        # (the liquid joined elsewhere, round a bead or along a wall: the cut
        # took a handle off it, one body still; the volume constraint takes
        # back what the collapse moved)
        _log.debug("neck cut, the body one piece still (a handle taken off)")
        return []
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


def _rebody(ev, split: bool = False) -> None:
    """Evolver's rebody (the bodies' quantities rebuilt), without its
    warning that a volconst may need adjusting (pinch_off sets them); unless
    ``split``, the bodies it splits made whole again. (A body's surface can
    be in pieces the liquid joins through the walls, as round a bead's
    contact with a wall or another bead: a pack's liquid was in pieces of
    4934 and 543 facets, and rebody after a merge made the small one body 1
    with the whole volume target, the rest body 2.)"""
    from ._errors import EvolverWarning
    m = ev.mesh() if not split else None
    states = {int(b): _body_state(ev, int(b)) for b in ev.bodies().ids} if m is not None else {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "WARNING 2171", EvolverWarning)
        ev.command("rebody")
    if m is None or m.facet_bodies is None:
        return
    was = {int(i): int(b) for i, b in zip(m.facet_ids, np.asarray(m.facet_bodies)[:, 0])}
    old = {int(b) for b in np.asarray(m.facet_bodies).ravel() if b > 0}
    now = ev.mesh()
    fb = np.asarray(now.facet_bodies)
    folded = set()
    for b in sorted({int(b) for b in ev.bodies().ids} - old):
        mine = np.asarray(now.facet_ids)[fb[:, 0] == b]
        before = [was[int(i)] for i in mine if int(i) in was and was[int(i)] > 0]
        if before:
            into = max(set(before), key=before.count)
            _fold(ev, b, into)
            folded.add(into)
    # (rebody set the pieces' targets to what each held)
    for b in folded:
        if b in states:
            body = ev.body(b)
            body.volconst = states[b]["volconst"]
            body.target = states[b]["target"] if states[b]["fixed"] else None


def _fold(ev, b: int, into: int) -> None:
    """Body b's facets made body into's, b dissolved."""
    for side in ("frontbody", "backbody"):
        ev.command(f"set facet {side} {into} where {side} == {b}")
    ev.command(f"dissolve body[{b}]")
    ev.command("recalc")


def _settle(ev, tol: float = 1e-8) -> Any:
    """Relax after a cut (see Evolver.pinch_off): first at the volumes the
    pieces hold, where the cones the cut left round off, by Newton steps with
    a line search (Evolver's hessian_seek) after a few gradient steps; then to
    their targets (what the cut took from them back) by plain Newton steps, as
    arclength's first step goes, in up to 16 parts. relax() from the cut (its
    long gradient phase) slid the vertices next to a drop's contact line on a
    sphere at 40 degrees through the sphere, in one of four cuts at the
    volumes held, and in most going to the targets at once. relax()'s result
    at the end, None if it failed (the surface put back as the cut left it,
    the targets as they were: a 16-bead pack's failed settling had taken
    351 vertices up to 0.84 radii into a bead, and the descent went on down
    from there, its energy and volume wrong by the bead's). Failed too where
    it passed the surface through itself or a wall (a pack's, after a sheet
    ruptured, through itself 16 times in a corner neck 0.005 across)."""
    bodies = ev.bodies()
    want = {int(i): ev.body(int(i)).target for i, fixed in zip(bodies.ids, bodies.fixed)
            if fixed}
    held = {i: ev.body(i).volume for i in want}
    cut, crossed, out = ev._save(), _crossings(ev), _outside(ev)

    def failed() -> None:
        ev.restore(cut)
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
    if not r.converged or _crossings(ev) > crossed or _outside(ev) > out:
        return failed()
    return r


def merge(ev, event, keep=None) -> int:
    """Glue the surface where it is about to touch itself across the outside:
    see :meth:`Evolver.merge`. Returns the merged body's number (0 without
    bodies)."""
    if getattr(event, "kind", None) != "touch":
        raise ValueError("merge() glues at a touch: an Event of kind 'touch'")
    m = ev.mesh()
    x, f = np.asarray(m.vertices, float), np.asarray(m.facets)
    a, b = _facing_facets(x, f, np.asarray(event.where, float))
    made = _tube(ev, m, a, b, keep)
    if made is not None:
        return made
    return _glue(ev, m, a, b, keep)


def _tube(ev, m, a: int, b: int, keep, sheet: bool = False) -> "int | None":
    """merge() by the checked edits (_edits): facet rows a and b taken out
    and a short tube of facets joining their edges; the bodies either side
    made one as _glue makes them. None where the edit is refused (then
    _glue's Evolver facet_merge is tried). ``sheet``: a sheet's hole, folds
    allowed (the edit's mode 1: between faces closer than a fifth of the
    facets the tube folds back on them whatever the pair)."""
    from ._edits import REASONS, edit
    fb = np.asarray(m.facet_bodies)
    bodies = sorted({int(v) for v in fb[[a, b]].ravel() if v > 0})
    if len(bodies) > 2:
        return None
    states = {i: _body_state(ev, i) for i in bodies}
    r = edit(ev, "merge", [int(m.facet_ids[a]), int(m.facet_ids[b])], mode=1 if sheet else 0)
    if r[0] != 0:
        _log.debug("fallback: merge by facet_merge (the tube refused: %s)", REASONS[int(r[0])])
        return None
    return _make_one(ev, bodies, keep, states)


def _make_one(ev, bodies: list, keep, states: dict, volume=None, wall=None) -> int:
    """The bodies made one: the one numbered keep if it is one of them, else
    the lower; its volconst theirs added (and with volume, the volumes before
    a change, shifted by what wall's pole adds: _kept_constant), its target
    their targets added (free if either was). The merged body's number (0
    without bodies)."""
    target = keep if keep in bodies else (bodies[0] if bodies else 0)
    other = [i for i in bodies if i != target]
    for i in other:
        for side in ("frontbody", "backbody"):
            ev.command(f"set facet {side} {target} where {side} == {i}")
    if bodies:
        _rebody(ev)
    for i in other:
        ev.command(f"dissolve body[{i}]")
    if bodies:
        body = ev.body(target)
        body.volconst = sum(s["volconst"] for s in states.values())
        if volume is not None:
            body.volconst = body.volconst + _kept_constant(
                ev, sum(volume.values()) - body.volume, wall)
        if all(s["fixed"] for s in states.values()):
            body.target = sum(s["target"] for s in states.values())
        else:
            body.target = None
    return target


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
    snapshot = ev._save()
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


def _inner_facets(ev, m) -> np.ndarray:
    """The facets with no corner on the surface's border nor on a wall
    (what the checked merge opens)."""
    f = np.asarray(m.facets)
    edges = np.asarray(m.edges)
    border = np.zeros(len(m.vertices), dtype=bool)
    border[edges[m.edge_valence() == 1].ravel()] = True
    _, _, gaps, _ = ev._constraint_gaps()
    on = np.isnan(gaps).any(axis=1) if gaps.size else np.zeros(len(m.vertices), dtype=bool)
    return ~(border | on)[f].any(axis=1)


def _facing_facets(x, f, p, behind: bool = False, reach: float = np.inf, allowed=None):
    """The two facets either side of a touch: the facet nearest its middle,
    and on the side its normal points to, the nearest facet facing back.
    With ``behind`` (a sheet's two faces, across the body: the normals point
    out of it), on the side it points from, within ``reach``: the one most
    nearly straight across (a sheet 0.002 thin on facets of 0.05, the facet
    across nearest the middle was offset along the sheet, the tube joining
    the two lay all but flat and folded the mesh, refused at every pair)."""
    centre = x[f].mean(axis=1)
    normal = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
    dist = np.linalg.norm(centre - p, axis=1)
    if allowed is not None:
        dist = np.where(allowed, dist, np.inf)
    a = int(np.argmin(dist))
    across = (centre - centre[a]) @ normal[a]
    if behind:
        facing = (normal @ normal[a] < -0.5) & (across < 0) \
            & (np.linalg.norm(centre - centre[a], axis=1) < reach)
        if allowed is not None:
            facing &= allowed
        if not facing.any():
            raise ValueError("pinch_off(): no facet across the sheet from its nearest one")
        offset = centre - centre[a]
        aside = np.linalg.norm(offset - np.outer(offset @ normal[a], normal[a]), axis=1)
        b = int(np.flatnonzero(facing)[np.argmin(aside[facing])])
        return a, b
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
    :meth:`Evolver.contact`. A vertex the new contact line goes through
    twice (two fans of facets there: a dry patch's border touching a
    contact line, 11 of the 12 pinches the topology check found in the
    corpus) is separated, one vertex per fan (_edits); where that can't be,
    the patch is made again grown over the facets round it (its two lobes
    one: 116 refusals in a round of the corpus's runs), and where that can't
    be either, the contact is refused."""
    start = ev._save()
    _onto_copy(ev, event)
    _contact(ev, event, keep)
    nums, _, _, names = ev._constraint_gaps()
    num = int(nums[_column(nums, names, event.constraint)])
    try:
        _separate_pinches(ev, num)
    except ValueError:
        m = ev.mesh()
        pinched = np.asarray(m.vertex_ids)[_pinch_rows(ev, m, num)]
        ev.restore(start)
        if not len(pinched):
            raise
        _onto_copy(ev, event)
        _contact(ev, event, keep, grow=[int(i) for i in pinched])
        _separate_pinches(ev, num)
        _log.debug("patch grown over the vertices its line went through twice")


def _pinch_rows(ev, m, num: int) -> np.ndarray:
    """Rows of m's vertices on constraint num where four or more border
    edges meet."""
    border = np.asarray(m.edges)[m.edge_valence() == 1]
    count = np.bincount(border.ravel(), minlength=len(m.vertices))
    return np.flatnonzero((count >= 4) & ev.on_constraint(num))


def _separate_pinches(ev, num: int) -> None:
    """Vertices on constraint num where four or more border edges meet (a
    contact line through them twice) separated, one per fan of facets."""
    from ._edits import REASONS, edit
    m = ev.mesh()
    rows = _pinch_rows(ev, m, num)
    if not len(rows):
        return
    r = edit(ev, "separate", np.asarray(m.vertex_ids)[rows])
    if (r != 0).any():
        raise ValueError("contact(): a vertex the contact line goes through twice can't be "
                         f"separated ({REASONS[int(r[r != 0][0])]})")


def _contact(ev, event, keep=None, grow=()) -> None:
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
            # twice): separated there (contact() does that), and the lines
            # are then two (a pore's contact lines on one bead after its
            # liquid jumped)
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
                if not _bridge_lines(ev, m, num, a, b, keep):
                    _join_lines(ev, m, x, f, a, b, keep)
                return
            core = np.zeros(0, dtype=int)
    else:
        core = rows[~on[rows]]
    # (not fixed vertices, nor ones on two other constraints already: the
    # patch stays clear of them; one on one other wall becomes a corner, as
    # where the gas reaches the floor along a side wall)
    held = ((np.isnan(dist) & ~on[:, None]).sum(axis=1) >= 2) | m.fixed
    core = core[~held[core]]
    if flat is None and not len(core) and kind == "contact lines":
        # (the strip between two lines wet but with no free vertex over it,
        # its facets' corners all on the lines: as across dry ground, the
        # facets lying on the solid taken away, else the lines joined; a
        # pack's lines on a bead 0.0014 apart were refused as "on other
        # constraints", and the descent ran away round them)
        a, b = _closest_pair(x, rows, on)
        near = np.linalg.norm(x[f].mean(axis=1) - (x[a] + x[b])/2, axis=1) < 2*float(event.scale)
        flat = on[f].all(axis=1) & near
        if not flat.any():
            if not _bridge_lines(ev, m, num, a, b, keep):
                _join_lines(ev, m, x, f, a, b, keep)
            return
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
    if len(grow):
        # (the facets round the vertices the line went through twice, where
        # their corners lie within half an edge of the solid)
        at = np.flatnonzero(np.isin(np.asarray(m.vertex_ids), grow))
        ring = np.isin(f, at).any(axis=1) & ~gone
        corners = np.unique(f[ring])
        near = on[corners] | (np.abs(np.nan_to_num(dist[corners, c], nan=0.0))
                              < 0.5*float(event.scale))
        if not ring.any() or not near.all() or held[corners[~on[corners]]].any():
            raise ValueError("contact(): a vertex the contact line goes through twice "
                             "can't be separated, nor the patch grown over it")
        gone = gone | ring
    gone = _clear_of_touching(ev, num, f, gone, on, dist, nums, c, float(event.scale))
    if not gone.any():
        raise ValueError("contact(): the place is where two walls touch")
    _dissolve_onto(ev, m, gone, event.constraint)


def _clear_of_touching(ev, num: int, f, gone, on, dist, nums, col: int, h: float):
    """The patch's facets less those at a vertex on another wall that doesn't
    meet this one there: put onto this wall, it would be farther from the
    other than 0.05 h (to first order, the distance it moves times the
    cosine between the walls' normals; a corner where a side wall meets the
    floor moves along both), or the normals are within 10 degrees. Where a
    bead nearly touches a side wall (0.015 apart, so nowhere on both), a
    patch reaching the bead's contact line put a vertex on both, Evolver's
    constraint projection singular (ERROR 3005); kept a facet off it, the
    strip between the two lines is the pendular ring's meniscus round the
    contact. (Refusing the patch left the gas front stuck there: 3 of 5
    stops in a round of 16-bead runs.) Then no vertex where the patch's
    facets meet in two fans (its border through it twice)."""
    rows = np.unique(f[gone])
    others = np.isnan(dist[rows]) & ~on[rows][:, None]
    if not others.any():
        return gone
    _, grad = ev._call(_core.constraint_normals, num)
    mine = np.asarray(grad.data)
    bad = np.zeros(len(on), dtype=bool)
    for other in np.flatnonzero(others.any(axis=0)):
        at = rows[others[:, other]]
        _, g2 = ev._call(_core.constraint_normals, int(nums[other]))
        a, b = mine[at], np.asarray(g2.data)[at]
        cos = np.abs(np.einsum("ij,ij->i", a, b)) / np.maximum(
            np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-300)
        moves = np.abs(np.nan_to_num(dist[at, col], nan=0.0))
        bad[at[(cos > 0.985) | (moves*cos > 0.05*h)]] = True
    if not bad.any():
        return gone
    _log.debug("patch kept off %d vertices where two walls don't meet", int(bad.sum()))
    gone = gone & ~bad[f].any(axis=1)
    return _one_fan(f, gone)


def _one_fan(f, gone):
    """The patch's facets less those that make a vertex where they meet in
    two fans or more (all but the largest fan there), until none does."""
    for _ in range(20):
        changed = False
        rows = np.flatnonzero(gone)
        for v in np.unique(f[rows]):
            mine = rows[(f[rows] == v).any(axis=1)]
            if len(mine) < 2:
                continue
            # fans: the patch's facets round v joined across their edges at v
            parent = {int(k): int(k) for k in mine}

            def root(k):
                while parent[k] != k:
                    parent[k] = parent[parent[k]]
                    k = parent[k]
                return k
            for i in range(len(mine)):
                for j in range(i + 1, len(mine)):
                    if len(set(f[mine[i]]) & set(f[mine[j]])) == 2:
                        parent[root(int(mine[i]))] = root(int(mine[j]))
            fans: dict = {}
            for k in mine:
                fans.setdefault(root(int(k)), []).append(int(k))
            if len(fans) > 1:
                keep = max(fans.values(), key=len)
                for group in fans.values():
                    if group is not keep:
                        gone[group] = False
                changed = True
        if not changed:
            break
    return gone


def _pole_quantum(ev, wall_key) -> float:
    """The volume a sphere(axis=)'s integrals jump by where a contact line
    comes to wind round, or leaves, its pole (its whole sphere, 4 pi R^3/3);
    0 for other walls (a plane, a sphere(gauge="flux"): no constants)."""
    wall = surface.walls.get(wall_key)
    if wall is None and isinstance(wall_key, (int, np.integer)):
        nums, _, _, names = ev._constraint_gaps()
        for c, name in zip(nums, names):
            if int(c) == int(wall_key):
                wall = surface.walls.get(name or int(c))
    shape = getattr(wall, "shape", None)
    if shape is None or shape[0] != "sphere" or getattr(wall, "pole_axis", None) is None:
        return 0.0
    return 4*np.pi*float(shape[2])**3/3


def _kept_constant(ev, change: float, wall_key) -> float:
    """What a change's volconst shift may be: whole multiples of the wall's
    pole quantum (_pole_quantum), the jump its integrals made, nearest the
    volume the change moved; never the rest. The volume is what the surface
    holds: a merge's or patch's own error, hidden in the constant, left a
    pack's mesh holding 0.129 less liquid than its target; the body's fixed
    volume brings the surface back to it instead."""
    q = _pole_quantum(ev, wall_key)
    return q*round(change/q) if q > 0 else 0.0


def _dissolve_onto(ev, m, gone, key) -> None:
    """The facets gone (mask over m's rows) taken away, their vertices put
    onto the wall key first, the new border edges on it as contact lines;
    volconst shifted only by what a sphere's pole adds (_kept_constant)."""
    nums, _, dist, names = ev._constraint_gaps()
    c = _column(nums, names, key)
    num = int(nums[c])
    x, f = np.asarray(m.vertices, float), np.asarray(m.facets)
    on = np.isnan(dist[:, c])
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
    was_border = set(np.asarray(m.edge_ids)[m.edge_valence() == 1].tolist())
    ids = np.asarray(m.facet_ids)[gone]
    # the checked patch edit (_edits): a vertex on another wall too becomes
    # a corner (the gas reaching the floor along a side wall), a vertex the
    # new line goes through twice is separated; else Evolver's dissolving
    from ._edits import REASONS, edit
    r = edit(ev, "patch", [num] + [int(i) for i in ids])
    if r[0] == 0:
        m2 = ev.mesh()
        edge_on = ev.on_constraint(num, "edge")
        new = (m2.edge_valence() == 1) & edge_on \
            & ~np.isin(np.asarray(m2.edge_ids), list(was_border))
        added = np.asarray(m2.edge_ids)[new]
    else:
        _log.debug("fallback: patch by dissolving (the edit refused: %s)", REASONS[int(r[0])])
        # (never a vertex on three constraints: fixed in place, and a box
        # corner's broke Evolver's Hessian setup)
        nums2, _, dist2, _ = ev._constraint_gaps()
        mine = ev.on_constraint(num)
        on_other = np.isnan(dist2) & ~mine[:, None]
        rows = np.unique(np.asarray(m.facets)[gone])
        if (on_other[rows].sum(axis=1) >= 2).any():
            raise ValueError("contact(): the patch would put a vertex on three constraints")
        # (nor on two walls touching there, their normals within 10 degrees:
        # Evolver's constraint projection singular, as the C edit refuses)
        corner = rows[on_other[rows].any(axis=1) & ~mine[rows]]
        for col in np.flatnonzero(on_other[corner].any(axis=0)):
            at = corner[on_other[corner, col]]
            _, g2 = ev._call(_core.constraint_normals, int(nums2[col]))
            a, b = normal[at], np.asarray(g2.data)[at]
            cos = np.abs(np.einsum("ij,ij->i", a, b)) / np.maximum(
                np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-300)
            if (cos > 0.985).any():
                raise ValueError("contact(): the patch would put a vertex on two walls "
                                 "touching there")
        snapshot = ev._save()
        try:
            ev.command("; ".join(f"dissolve facet[{int(i)}]" for i in ids))
            ev.command("dissolve edges; dissolve vertices")
            added = _onto(ev, num, was_border)
        except Exception:
            ev.restore(snapshot)
            raise
    _poles(ev, key, added)
    for i in bodies:
        if i in [int(j) for j in ev.bodies().ids]:
            b = ev.body(i)
            b.volconst = b.volconst + _kept_constant(ev, volume[i] - b.volume, key)


def drop_flat(ev, h: float) -> int:
    """Facets with their corners on one wall and next to no area (under a
    hundredth of h squared: the surface lying on the solid, as a merged
    neck's sank onto the plane between two drops, its contact lines a chord
    0.011 apart that edge deletion can't take, the lines would pinch) taken
    away as contact() takes a roofed channel's. Returns how many."""
    m = ev.mesh()
    f = np.asarray(m.facets)
    if not len(f):
        return 0
    x = np.asarray(m.vertices, float)
    area = 0.5*np.linalg.norm(np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]]), axis=1)
    small = area < 0.01*h*h
    if not small.any():
        return 0
    nums, _, dist, names = ev._constraint_gaps()
    for key in surface.walls:
        try:
            c = _column(nums, names, key)
        except ValueError:
            continue
        gone = small & np.isnan(dist[:, c])[f].all(axis=1)
        if gone.any():
            _dissolve_onto(ev, m, gone, key)
            _log.debug("flat facets dropped on %s (%d)", key, int(gone.sum()))
            return int(gone.sum())
    return 0


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


def _edges_on(ev, keys) -> dict:
    """For each of the walls keys (the defined ones), which edges (mesh()
    rows) are on it, from one query (asked a wall at a time, 0.85 ms each,
    the tidying's 27k queries were 4.5% of a 16-bead pack's descent)."""
    nums, _, _, names = ev._constraint_gaps()
    number = {(n or int(k)): int(k) for k, n in zip(nums, names)}
    keys = [k for k in keys if k in number]
    if not keys:
        return {}
    if not hasattr(_core, "edges_on_constraints"):
        return {k: ev.on_constraint(k, "edge") for k in keys}
    _, result = ev._call(_core.edges_on_constraints, [number[k] for k in keys])
    on = np.asarray(result.data) > 0
    return {k: on[:, j] for j, k in enumerate(keys)}


def _on_walls(ev) -> np.ndarray:
    """Which edges (mesh() rows) are on a wall."""
    out = np.zeros(ev.counts["edges"], dtype=bool)
    for on in _edges_on(ev, list(surface.walls)).values():
        out |= on
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


def close_patches(ev, h: float, out: Optional[list] = None) -> int:
    """Contact-line loops within 0.2 h of their first vertex (a patch shrunk
    to nothing: none of its edges can be deleted, the two left would be
    doubled, and relax() ended with a pack's three-edge loop 0.003 across as
    degenerate facets) covered: Evolver's new_facet over the loop, with the
    bodies of the facet beside it, the loop's edges and vertices taken off
    the wall (the tidying then deletes them). Not where that moved the
    energy (a loop round a pole), nor where Evolver can't take a gradient
    step from it. (Within half an edge, the one-facet holes
    contacts open on a wall were closed and opened again over and over,
    piling vertices up; and within 0.2 of the edges round the loop where
    the mesh is finer than h: a contact where it was refined, edges 0.046
    for h 0.165, opened a patch of one facet that the next tidying closed,
    the surface lay on the wall again, its contact refused as reopening a
    closed patch, and the descent crept 400 iterations without settling.)
    The lines oriented (orient_lines). Returns how many were closed; each
    one's wall and middle appended to ``out``."""
    m = ev.mesh()
    f, edges = np.asarray(m.facets), np.asarray(m.edges)
    x = np.asarray(m.vertices)
    valence = m.edge_valence()
    lengths = m.edge_lengths()
    closed = 0
    ons = _edges_on(ev, list(surface.walls))
    for key in list(ons):
        on_wall = ons[key]
        line = np.flatnonzero(on_wall & (valence == 1))
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
            spokes = np.isin(edges, corners).any(axis=1)
            spokes[loop] = False
            local = min(h, float(np.median(lengths[spokes]))) if spokes.any() else h
            if np.linalg.norm(x[corners] - x[corners[0]], axis=1).max() > 0.2*local:
                continue
            a, b = edges[loop[0]]
            beside = np.flatnonzero(((f == a) & (np.roll(f, -1, axis=1) == b)).any(axis=1))
            if not len(beside):
                continue
            front, back = (int(i) for i in np.asarray(m.facet_bodies)[beside[0]])
            ids = [int(m.edge_ids[i]) for i in loop]
            energy, saved = ev.energy, ev._save()
            body = "" if len(ids) == 3 else " where original == pyse_patch"
            commands = [f"pyse_patch := new_facet({', '.join(str(-e) for e in reversed(ids))})"]
            for side, b_id in (("frontbody", front), ("backbody", back)):
                if b_id:
                    commands.append(f"set facet{body or '[pyse_patch]'} {side} {b_id}")
            commands += [f"unset edge[{e}] constraint {key}" for e in ids]
            commands += [f"unset vertex[{int(m.vertex_ids[c])}] constraint {key}"
                         for c in corners]
            ev.command("; ".join(commands))
            # (nor where Evolver can't step from it: a pack's closed loop
            # left a constraint matrix singular, and the run stopped there)
            if abs(ev.energy - energy) > 0.05 or _unsteppable(ev):
                ev.restore(saved)
                continue
            closed += 1
            if out is not None:
                out.append((key, x[corners].mean(axis=0)))
            _log.debug("patch closed on %s (%d edges)", key, len(loop))
            # (and the walls' edges again: a loop of 4 to 6 edges is
            # covered by facets round a new vertex, new edges, and the next
            # wall's rows, from before, no longer matched the mesh's: a
            # 16-bead pack's replay stopped there)
            m = ev.mesh()
            f, edges, x = np.asarray(m.facets), np.asarray(m.edges), np.asarray(m.vertices)
            valence, lengths = m.edge_valence(), m.edge_lengths()
            ons = _edges_on(ev, list(surface.walls))
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


def _bridge_lines(ev, m, num: int, a: int, b: int, keep) -> bool:
    """Two contact lines about to meet across dry ground on constraint num
    (vertex rows a and b on them) joined by the checked edits (_edits): a
    facing pair of their edges bridged by two facets, whose outer edges are
    the contact line now; the bodies either side made one, the volumes kept
    (volconst). Whether it was made (else _join_lines is tried)."""
    from ._edits import REASONS, edit
    edges, valence = np.asarray(m.edges), m.edge_valence()
    on = ev.on_constraint(num)
    line = (valence == 1) & on[edges[:, 0]] & on[edges[:, 1]]
    at_a = np.flatnonzero(line & (edges == a).any(axis=1))
    at_b = np.flatnonzero(line & (edges == b).any(axis=1))
    near = np.isin(np.asarray(m.facets), np.r_[edges[at_a].ravel(), edges[at_b].ravel()]).any(axis=1)
    fb = np.asarray(m.facet_bodies)
    bodies = sorted({int(v) for v in fb[near].ravel() if v > 0})
    states = {i: _body_state(ev, i) for i in bodies}
    volume = {i: ev.body(i).volume for i in bodies}
    ids = np.asarray(m.edge_ids)
    why = "no facing edges"
    for ra in at_a:
        for rb in at_b:
            r = edit(ev, "bridge", [int(ids[ra]), int(ids[rb])])
            if r[0] == 0:
                _make_one(ev, bodies, keep, states, volume, num)
                return True
            why = REASONS[int(r[0])]
    _log.debug("fallback: contact lines joined by vertex_merge (the bridge refused: %s)", why)
    return False


def _join_lines(ev, m, x, f, a, b, keep) -> None:
    """Two contact lines about to meet across dry ground (vertex rows a and b
    on them): a's and b's contact-line edges that face each other glued into
    one edge (Evolver's vertex_merge on their ends, then edge_merge), which
    the two facets along them now share, off the constraint: the lines
    reconnect there (two patches join; one patch meeting itself encloses a
    dry island). The bodies either side become one as merge() makes them;
    volconst shifted only by what a sphere's pole adds (_kept_constant)."""
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
    # the glued edge's two facets must go round it opposite ways (lines facing
    # across dry ground run opposite ways; a pack's two lines running the same
    # way, glued, left a facet turned over, and Evolver's pop then failed)
    turns = {(int(r[i]), int(r[(i + 1) % 3])) for r in f.tolist() for i in range(3)}
    if ((a, a2) in turns) == ((b, b2) in turns):
        raise ValueError("contact(): the contact lines run the same way there: not joined")
    fb = np.asarray(m.facet_bodies)
    bodies = sorted({int(v) for r in range(len(f))
                     if {a, a2} <= set(f[r].tolist()) or {b, b2} <= set(f[r].tolist())
                     for v in fb[r] if v > 0})
    states = {i: _body_state(ev, i) for i in bodies}
    target = keep if keep in bodies else (bodies[0] if bodies else 0)
    volume = {i: ev.body(i).volume for i in bodies}
    ids = np.asarray(m.vertex_ids)
    eids = np.asarray(m.edge_ids)
    snapshot = ev._save()
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
        body.volconst = body.volconst + _kept_constant(
            ev, sum(volume.values()) - body.volume, names[num] or int(nums[num]))
        if all(s["fixed"] for s in states.values()):
            body.target = sum(s["target"] for s in states.values())
        else:
            body.target = None


_FALL_STEP = 0.3        # a fall's step along the unstable mode: at most this many median edges
_FALL_GRADIENT = 5      # gradient steps after each
_FALL_ITERATIONS = 200     # iterations of a descent since its last change, at most
_FALL_TOTAL = 10           # ... and this many times that in all
_FALL_NEAR = 0.05       # the residual below which Newton steps try to finish a descent
_NEWTON_NEAR = 1.0      # ... and below which its relax() runs take Newton steps (gradient runs
                        # alone level off at 0.1-0.5, the tidying between rounds disturbing it)
_FALL_TRY = 10          # iterations between such tries
_FLAT = 0.05            # a facet crushed flat: its area under this many h^2 (h the mean edge)
_AGAIN = 3              # a change made this often at one place in a descent: not again there
_FALL_CLEAR = 0.1       # the gradient steps' cosine with the mode that sets the way
_REMESH = (0.4, 2.0)    # after a change: edges kept within these many times their targets (_sizes)
_ON_THE_WAY = 4/3       # ... and between a run's points, edges split past this many times theirs
                        # (twice: a pack's surface, stretching as it drained, went on with
                        # edges of 0.39 for 0.2, angles of 8 degrees)
_CHUNK = 50             # after a change: relax() runs of this many gradient steps at most
_LINE_CHOP = 4/3        # in a descent: contact-line edges split past this many h0 each step
_SHORTER = (12, 3, 1)   # a run that crossed twice: again in runs of these many steps
_FINE = 0.25            # _sizes: the finest target edge, in h0
_GRADE = 0.5            # _sizes: how fast the targets grow away from a gap (per length)
_COARSEN = 1.5          # _sizes: a vertex's target grows at most this much a remeshing
_LOCAL = 4.0            # after a change: remeshed within this many h0 of it
_HOLE_TRIES = 8         # a sheet's hole: this many facet pairs tried, nearest its thinnest point first
_PINCHED = 0.05         # a neck three edges round, each under this times the edge scale: collapsed
_GRADIENT_RUN = 10      # ... or, where one went wrong, this many plain gradient steps
_STALLED = 2            # runs in a row leaving the state as it was: the descent stalled
_PLACE = 3.0            # changes at one place: on one solid, within this many of their edges
_MARGIN = 0.5           # ... made again there (or undone) only at under this share of the gap


def _median_edge(m) -> float:
    lengths = m.edge_lengths()[m.edge_valence() != 1]
    return float(np.median(lengths)) if len(lengths) else float(np.median(m.edge_lengths()))


def _edge_scale(ev) -> float:
    """The edge length descents remesh toward: the median edge when first
    asked (surface memory), kept (taken afresh at each descent, a pack's
    local refinement lowered it run by run, and the mesh went from 1.5k
    facets to 23k). The scale of everything measured in edges too (a tenth
    of an edge through a wall, crumbs, collapsed necks): with the median a
    pack's refined sheet took the median from 0.16 to 0.05, and the
    thresholds with it."""
    if surface.edge_scale is None:
        surface.edge_scale = _median_edge(ev.mesh())
    return float(surface.edge_scale)


def apply(ev, event, keep=None, other: bool = False) -> None:
    """Make the change an event announces (without relaxing; the guards
    off, see _unguard). ``other``: a sheet's neck at a wall cut by a hole,
    not by drying its footprint (see pinch_off)."""
    _unguard(ev)
    if event.kind == "neck":
        pinch_off(ev, event, relax=False, at_wall=not other)
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
    if h0 is None:
        h0 = _edge_scale(ev)
    # (each gradient step ends with the contact-line edges longer than
    # _LINE_CHOP h0 split onto their walls: a chord on a curved wall grown
    # long cuts into it, its integrals go wrong and the steps follow the
    # error: a 16-bead pack's relax() run after a contact stretched one on a
    # bead of radius 1 from 0.36 to 1.9 in 18 steps, the energy 1.4 lower,
    # where split back onto the bead it was 0.4 higher than at the start;
    # the lines crossed and every run after was undone)
    chop = _core.line_chop(_LINE_CHOP*h0)
    _CROSSED_AT.clear()
    _PINCH_LEFT.clear()
    # (what the last descent refined that is no longer close, coarsened
    # first: its targets back toward h0, as the next steps let them)
    _coarsen(ev, h0)
    held, surface.sizes_held = surface.sizes_held, True
    # (and its runs make no changes of their own: relax()'s last resort,
    # Evolver's 't' on edges shrunk to a 20th of the median, pinched a
    # 16-bead pack's surface along edges, two edges joining the same
    # vertices, 80 of them by the end of a fall, and its constraint
    # projection then failed: ERROR 3005)
    topology, ev._topology_held = ev._topology_held, True
    # (and gradient steps with fewer energy evaluations: no move at scale 0
    # where nothing changed since the last step, the scale from a parabola
    # with the energy's slope; a pack's step 11 ms to 7; each move's volumes
    # predicted for its restoring motion, not evaluated, a 16-bead pack's
    # step at 2k facets 64 million instructions to 44. Not the stock
    # search, which relax() and Evolver.command("g") otherwise keep)
    fast = _core.set_fast_steps(True) if hasattr(_core, "set_fast_steps") else None
    try:
        result = _descend(ev, factor, tol, falling=falling, keep=keep, h0=h0, changes=changes)
        if result[2]:
            _log.debug("descent stopped: %s", result[2])
        return result
    except EvolverError as error:
        _log.debug("descent stopped: Evolver: %s", error)
        return None, changes, f"Evolver: {error}"
    finally:
        _core.line_chop(chop)
        if fast is not None:
            _core.set_fast_steps(fast)
        surface.sizes_held = held
        ev._topology_held = topology
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
    from ._errors import EvolverError
    way: Any = None
    before: list = []
    refused = ""
    tried = _FALL_TRY
    remeshing = not falling
    if h0 is None:
        h0 = _edge_scale(ev)
    if not falling:
        guard(ev)
    since, stuck = 0, False
    closing, fresh = True, None         # whether the tidying closes patches; the iteration's start
    shut: list = []                     # patches closed in this descent: (wall, middle)
    made_at: list = []                  # changes made in it: (kind, solid, place)
    pre: list = []                      # the state before the change _try_change made
    undo: list = []                     # (that state, the change) for the last one made

    refused_at: list = []               # changes refused in it: (kind, solid, place)
    history: list = []                  # changes made, undone or refused in it, by place (_Tried)
    told: list = []                     # ... and those put off, told of
    unsound: Any = None                 # the state the last "relaxed, but" round left
    last: Any = None                    # the state the last unconverged run left
    stalled = 0                         # runs in a row that left it as it was

    def again(e) -> bool:
        """Whether e's change was made _AGAIN times here already (a pack's
        forced join of contact lines where a side wall meets the floor was
        made 160 times, each run after it through itself, until the run
        timed out), or refused as often (a contact squeezed into the gap
        where a bead nearly touches a wall was refused 1215 times in a run)."""
        h = _edge_scale(ev)

        def count(at):
            return sum(k == e.kind and c == str(e.constraint)
                       and float(np.linalg.norm(np.asarray(e.where) - w)) < h
                       for k, c, w in at)
        return count(made_at) >= _AGAIN or count(refused_at) >= _AGAIN

    def note_refused(e) -> None:
        refused_at.append((e.kind, str(e.constraint), np.asarray(e.where, float)))
        history.append(_tried(ev, e, "refused"))

    def later(e) -> bool:
        """Whether e's change is put off (_put_off), told of once a place."""
        if not _put_off(ev, e, history):
            return False
        here = _tried(ev, e, "")
        if not any(_same_place(here, t) for t in told):
            told.append(here)
            _log.debug("change put off: %s (made, undone or refused at its place in this "
                       "descent, and no nearer now)", e)
        return True

    def other(e) -> bool:
        """Whether e is a neck where one was cut in this descent already: a
        sheet at a wall then ruptures by a hole beside its contact line (its
        footprint dried there, a pack's sheet thinned there again, its
        footprint dried four times and its lines joined back, until the
        descent ran away)."""
        if e.kind != "neck":
            return False
        here = _tried(ev, e, "")
        return any(t.kind == "neck" and t.what == "made" and _same_place(here, t)
                   for t in history)

    def made_change(e) -> None:
        """A change made: noted, the mesh remeshed, and down from there."""
        nonlocal way, before, tried, since, stuck, falling, remeshing, stalled
        _checked(ev, f"the change: {e}")
        made_at.append((e.kind, str(e.constraint), np.asarray(e.where, float)))
        history.append(_tried(ev, e, "made"))
        undo[:] = [(pre.pop(), e)] if pre else []
        pre.clear()
        _remesh_now(ev, h0, near=[e.where])
        _checked(ev, "remeshing after the change")
        changes.append(e)
        guard(ev)
        way, before, tried, since, stuck = None, [], 0, 0, False
        falling, remeshing, stalled = False, True, 0

    for _ in range(_FALL_TOTAL*_FALL_ITERATIONS):
        try:
            since += 1
            if since > _FALL_ITERATIONS:
                # (once, the mesh remeshed and as many again: a pack's fall,
                # held back by folds or crawling past squashed facets, went
                # 200 steps without a change or a stable state)
                if stuck:
                    break
                stuck, since = True, 0
                if stalled > _STALLED:
                    _log.debug("descent remeshed: its runs left the state as it was")
                else:
                    _log.debug("descent remeshed: no stable state within %d steps", _FALL_ITERATIONS)
                stalled = 0
                _remesh_now(ev, h0)
                _checked(ev, "remeshing a descent that got nowhere")
                way, before = None, []
                # (and a fall goes on as a descent after a change does, in
                # relax() runs: a 4-bead fall wandered 400 steps at residual
                # 0.4-1.5 along the soft modes of its unremeshed mesh, the
                # energy going up as often as down, its lowest mode -129 to
                # +0.3 from one step to the next)
                if falling:
                    falling, remeshing = False, True
                    guard(ev)
            fresh = ev._save()          # (where this iteration began)
            _tidy(ev, h0, shut, close=closing)
            _checked(ev, "tidying (lines, patches, poles)")
            x = _places(ev)
            chunk = None
            if falling:
                _drop_crushed(ev)
                if _steps_uncrossed(ev, _FALL_GRADIENT):
                    _unguard(ev)
                    forced = _change_nearest(ev, factor, keep, again, pre, note_refused, later, other)
                    if forced is not None:
                        made_change(forced)
                        continue
                # (a fall doesn't remesh, but its contact lines are kept fine: a
                # pack's stretched to edges 1.6 long, and their integrals along
                # the chords went 1.5 off; the facets the splits halve flipped,
                # which moves no vertex: a pack's fall went to facets of 0 degrees
                # in 15 steps. Not otherwise: flipping in a catenoid's fall kept
                # it from ever settling)
                if _split_lines(ev, h0):
                    _even_out(ev, smooth=False)
                _checked(ev, "a fall's gradient steps")
            else:
                # after a change, relax() in short runs (a merged neck widening
                # takes thousands of gradient steps on a fine mesh), looking at
                # the events between them; a run that went wrong (relax()'s took
                # a pack's liquid from 80.3 to 83.5) is undone for a run of plain
                # gradient steps
                chunk, start, energy = _away_from_poles(ev, lambda n: _run(ev, tol, n),
                                                        _CHUNK)
                if _implausible(ev, energy, getattr(chunk, "energy", None)) == "crossings":
                    # (the surface folded through itself near a change, its mesh
                    # tangled there: a pack's drying contact left an edge twice
                    # the others' and the next run crossed edges two apart):
                    # evened out and run again
                    ev.restore(start)
                    _remesh_now(ev, h0, near=_CROSSED_AT[0] if _CROSSED_AT and len(_CROSSED_AT[0])
                                else None)
                    chunk, start, energy = _away_from_poles(ev, lambda n: _run(ev, tol, n),
                                                        _CHUNK)
                    # (then in shorter runs, the way down followed finely: in
                    # steps of 50 a pack's surface was taken past where it
                    # came close, run after run, and the changes made there
                    # instead, 30 in one descent, were contacts 0.35 from a
                    # bead, merges across 0.17)
                    for n in _SHORTER:
                        if _implausible(ev, energy, getattr(chunk, "energy", None)) != "crossings":
                            break
                        ev.restore(start)
                        chunk, start, energy = _away_from_poles(
                            ev, lambda k: _run(ev, tol, k), n)
                    if _implausible(ev, energy, getattr(chunk, "energy", None)) == "crossings":
                        ev.restore(start)
                        _unguard(ev)
                        forced = _change_nearest(ev, factor, keep, again, pre, note_refused, later, other)
                        if forced is not None:
                            made_change(forced)
                            continue
                        guard(ev)
                        chunk, start, energy = _away_from_poles(
                            ev, lambda n: _run(ev, tol, n), _CHUNK)
                if not _plausible(ev, energy, getattr(chunk, "energy", None)):
                    _log.debug("run undone: implausible (volumes, energy, poles or crossings)")
                    ev.restore(start)
                    chunk = _gradient_run(ev, tol)
                    if chunk is None:
                        # (a pack's runs, then single gradient steps, went
                        # through itself or off their volumes after a contact:
                        # the change coming made, else the mesh evened out and
                        # once more)
                        _unguard(ev)
                        forced = _change_nearest(ev, factor, keep, again, pre, note_refused, later, other)
                        if forced is not None:
                            made_change(forced)
                            continue
                        _remesh_now(ev, h0)
                        guard(ev)
                        _checked(ev, "remeshing a descent gone wrong")
                        chunk = _gradient_run(ev, tol)
                    if chunk is None:
                        return None, changes, "the descent ran away (volumes or energy)"
                guard(ev)
                _checked(ev, "a relax run")
            if remeshing and not (chunk is not None and chunk.converged):
                _keep_edges(ev, h0)
                _checked(ev, "keeping the edges in range")
            moved = _moved_since(ev, x)
            found = ev.events(factor)
            h = _edge_scale(ev)
            # a neck coming (thinning: not a merge's, widening) that can be cut
            # (not one reaching a contact line: the neck that pinches comes later)
            neck = None
            for e in _coming(found, before):
                if e.kind == "neck":
                    # (not again where a change was made, undone or refused
                    # in this descent, unless nearer than then: _put_off)
                    if again(e) or later(e):
                        refused = f"{e}: put off (made, undone or refused at its place)"
                        continue
                    crossed, here = _crossings(ev), ev._save()
                    try:
                        _unguard(ev)
                        pinch_off(ev, e, relax=False, at_wall=not other(e))
                        if _crossings(ev) > crossed:
                            # (as every other change is checked: a pack's sheet
                            # ruptured at a wall left the surface through
                            # itself there, and the run went on so)
                            ev.restore(here)
                            raise ValueError("the cut passed the surface through itself")
                        neck = e
                        break
                    except ValueError as error:
                        refused = f"{e}: {error}"
                        _log.debug("change refused: %s", refused)
                        note_refused(e)
            if neck is not None:
                _log.debug("change made: %s", neck)
                _checked(ev, f"pinch_off: {neck}")
                made_at.append((neck.kind, str(neck.constraint), np.asarray(neck.where, float)))
                history.append(_tried(ev, neck, "made"))
                changes.append(neck)
                undo.clear()
                way, before, since = None, [], 0
                # (guarded as the descent's runs are: unguarded, the settling
                # took a pack's surface deep into a bead)
                guard(ev)
                r = _settle(ev, tol)
                if r is None:
                    # (a 16-bead pack's pieces, other changes coming: down
                    # from the cut in the descent's short runs instead)
                    _log.debug("the pieces didn't relax after the cut: on down from it")
                    falling, remeshing = False, True
                    continue
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
            # (not where a patch on that solid closed in this descent: a dry
            # patch opened on a bead's top before the liquid had drained enough
            # shrank and closed, the surface lay on the bead again, and a pack's
            # run opened and closed patches there 37 times; its guard holds it)
            resting = [e for e in found if _resting(e) and not _reopening(e, shut, h)]
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
                if held:
                    _checked(ev, "pushing the surface out of a solid")
            beside = [e for e in found if e.kind == "solid" and (falling or not _resting(e))
                      and _beside_line(ev, e, h)]
            coming = resting + [e for e in _coming(found, before)
                                if not any(e is r for r in resting)
                                and (e.kind in ("touch", "contact lines")
                                     or (e.kind == "solid" and not any(e is b for b in beside)
                                         and not _reopening(e, shut, h)))]
            made = None
            for e in coming:
                if again(e):
                    refused = f"{e}: made or refused {_AGAIN} times here already"
                    continue
                if later(e):
                    refused = f"{e}: put off (made, undone or refused at its place)"
                    continue
                why = _try_change(ev, e, keep, pre, other(e))
                if not why:
                    made = e
                    break
                refused = f"{e}: {why}"             # (passed over: it may come again, or go)
                note_refused(e)
                _log.debug("change refused: %s", refused)
            if made is not None:
                # (remeshed toward h0 before it relaxes: a contact's patch leaves
                # edges twice the others', and the next run folded the surface
                # through itself there)
                _log.debug("change made: %s", made)
                made_change(made)
                continue
            through = [e for e in resting if e.kind == "solid" and e.gap < 0
                       and _beside_line(ev, e, h)]
            if through:
                _push_out(ev, through, h)
                _checked(ev, "pushing the surface out of a solid")
            before = [e for e in found if not (e.kind == "solid" and e.gap < 0)]
            residual = ev.residual()
            tried += 1
            if chunk is not None:
                if not chunk.converged:
                    # (and where runs leave the state as it was, on to the
                    # budget's remeshing now: a 16-bead pack's runs stopped
                    # moving at residual 2, the energy the same to 1e-10,
                    # and the descent went on so 200 times, 215 s)
                    state = (tuple(ev.counts.values()), ev.energy)
                    stalled = stalled + 1 if _same_state(state, last) else 0
                    last = state
                    if stalled == _STALLED:
                        stalled += 1
                        since = max(since, _FALL_ITERATIONS)
                    continue
                if chunk.stable is not False and not _sound(chunk.health):
                    # (a facet squashed in the change, a patch shrunk to nothing,
                    # the surface lying on a solid: tidied, and once more)
                    drop_flat(ev, h0)
                    close_patches(ev, h0, shut)
                    # (facets crushed flat anywhere: a pack's relaxed state had
                    # three of 0 degrees, and the descent went on 1300 s round it)
                    _drop_crushed(ev)
                    _even_out(ev, smooth=False)
                    _keep_edges(ev, h0)
                    chunk = ev.relax(tol)
                    if chunk.health is not None and any(
                            i.startswith("degenerate facets") for i in chunk.health.issues):
                        _remesh_now(ev, h0)
                        chunk = ev.relax(tol)
                    _checked(ev, "tidying a relaxed state")
                if chunk.converged and chunk.stable is not False and _sound(chunk.health):
                    return chunk, changes, ""
                if chunk.stable is not False:
                    # (on down, within the descent's budget: a pack's relax()
                    # ended at residual 7e-6, not quite in equilibrium, and the
                    # run stopped there)
                    state = (tuple(ev.counts.values()), ev.energy)
                    if not _same_state(state, unsound):
                        unsound = state
                        _log.debug("relaxed, but %s: on down", chunk.health)
                        continue
                    # (but not where the tidying and relaxing came back to
                    # the state they began from: each round remeshed it twice
                    # more, to the end of the budget, 400 rounds. Settled
                    # there if only its mesh is wanting; else the budget's
                    # remeshing now, and where that comes back again, the end)
                    if chunk.converged and _mesh_only(chunk.health):
                        _log.debug("relaxed, but %s, as the last round left it: settled there",
                                   chunk.health)
                        return chunk, changes, ""
                    _log.debug("relaxed, but %s, as the last round left it", chunk.health)
                    since, stalled = max(since, _FALL_ITERATIONS), _STALLED + 1
                    continue
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
                start = ev._save()
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
            # (halved where it passes the surface through itself, up to three
            # times, then not taken: the touch comes as an event instead)
            x, crossed = ev.vertices, _crossings(ev)
            step = _FALL_STEP*h*way/np.linalg.norm(u, axis=1).max()
            for _ in range(4):
                ev.vertices = x + step
                if _crossings(ev) <= crossed:
                    break
                step = step/2
            else:
                ev.vertices = x
                _log.debug("step along the mode not taken: the surface through itself")
                forced = _change_nearest(ev, factor, keep, again, pre, note_refused, later, other)
                if forced is not None:
                    made_change(forced)
                    continue
            _checked(ev, "a step along the unstable mode")
        except EvolverError as error:
            # (Evolver failing after a change that took a gradient step: a
            # pack's join of contact lines on a side wall left a constraint
            # matrix singular in the remeshing after it) the change undone
            # and not made again here; once per change
            if not undo:
                # (none to undo: after a patch closed in the tidying, a
                # pack's constraint matrix went singular a few steps on)
                # back to where this iteration began, closing no more
                # patches in this descent; once
                if not closing or fresh is None:
                    raise
                ev.restore(fresh)
                closing, fresh = False, None
                _log.debug("iteration undone: Evolver failed (%s); no more patches closed",
                           str(error).splitlines()[0])
                way, before, since = None, [], 0
                continue
            saved, e = undo.pop()
            ev.restore(saved)
            for k, c in enumerate(changes):
                if c is e:
                    del changes[k]
                    break
            made_at.extend([(e.kind, str(e.constraint), np.asarray(e.where, float))]*_AGAIN)
            history.append(_tried(ev, e, "undone"))
            refused = f"{e}: Evolver failed after it ({str(error).splitlines()[0]})"
            _log.debug("change undone: %s", refused)
            way, before, since = None, [], 0
    return None, changes, ("the descent stalled (its runs left the state as it was), and again "
                           "after remeshing" if since > _FALL_ITERATIONS and stalled > _STALLED else
                           f"no stable state within {_FALL_ITERATIONS} steps of the last "
                           f"change, and as many after remeshing" if since > _FALL_ITERATIONS else
                           f"no stable state within {_FALL_TOTAL*_FALL_ITERATIONS} steps") \
        + (f" (the last change not made: {refused})" if refused else "")


def _common(a, b) -> float:
    """The dot product of two per-vertex arrays over the rows both have."""
    n = min(len(a), len(b))
    return float((a[:n]*b[:n]).sum())


def _split_lines(ev, h0: float) -> bool:
    """Contact-line edges (border edges on a wall) longer than _REMESH[1] h0
    split till none is (Evolver's edge refinement: the midpoint goes onto
    the wall). Whether any was."""
    split = False
    for _ in range(6):
        m = ev.mesh()
        long = (m.edge_valence() == 1) & (m.edge_lengths() > _REMESH[1]*h0)
        if long.any():
            long &= _on_walls(ev)
        if not long.any():
            break
        ev.command(f"refine edge{ev._mark(_core.EDGE, long)}")
        split = True
    return split


def _lift_chords(ev) -> int:
    """Edges across a curved wall between two of its contact-line vertices
    (off the wall themselves, on two facets) whose middle lies more than a
    tenth of an edge inside it, split, the new vertex put out on the wall's
    surface, free (and guarded where the surface is). A band of facets with
    every corner on a bead's contact lines has nothing to hold it off the
    bead: a 4-bead pack's fall left 18 such facets on one, its contact lines
    moved apart and the band was stretched along chords up to 1.3 long, 0.24
    deep in a bead of radius 1; the remeshing can't split them (the
    midpoint is inside the solid), and every run after crossed there, 70
    changes made instead in 900 s. How many were split."""
    h = _edge_scale(ev)
    walls = [(k, w.shape) for k, w in surface.walls.items()
             if getattr(w, "shape", None) is not None and w.shape[0] in ("sphere", "cylinder")
             and not any(isinstance(c, str) for c in w.shape[1])]
    if not walls:
        return 0
    lifted = 0
    guarded = None
    for _ in range(4):
        m = ev.mesh()
        edges = np.asarray(m.edges)
        if not len(edges):
            break
        nums, _, dist, names = ev._constraint_gaps()
        on = np.isnan(dist)
        x = np.asarray(m.vertices, float)
        free = (m.edge_valence() == 2) & ~_on_walls(ev)
        cut = np.zeros(len(edges), dtype=bool)
        mids, shapes = [], []
        for key, shape in walls:
            try:
                c = _column(nums, names, key)
            except ValueError:
                continue
            chord = free & ~cut & on[edges[:, 0], c] & on[edges[:, 1], c]
            if not chord.any():
                continue
            mid = x[edges[chord]].mean(axis=1)
            axis, R = _axis_of(shape)
            inside = R - np.linalg.norm(_radial(mid, shape, axis), axis=1) > 0.1*h
            cut[np.flatnonzero(chord)[inside]] = True
            mids += list(mid[inside])
            shapes += [shape]*int(inside.sum())
        if not cut.any():
            break
        if guarded is None:
            guarded = _guarded(ev)
            _unguard(ev)
        old = set(np.asarray(m.vertex_ids).tolist())
        ev.command(f"refine edge{ev._mark(_core.EDGE, cut)}")
        lifted += int(cut.sum())
        # the new vertices, each at the middle of its chord, out to the wall
        m = ev.mesh()
        x = np.array(m.vertices, dtype=float)
        mids = np.asarray(mids)
        for v in np.flatnonzero([i not in old for i in np.asarray(m.vertex_ids).tolist()]):
            k = int(np.argmin(np.linalg.norm(mids - x[v], axis=1)))
            axis, R = _axis_of(shapes[k])
            r = _radial(x[v][None, :], shapes[k], axis)[0]
            if np.linalg.norm(r) > 0:
                x[v] += r/np.linalg.norm(r)*(R + 1e-3*h - np.linalg.norm(r))
        ev.vertices = x
    if lifted:
        if guarded:
            guard(ev)
        _log.debug("chords through a wall lifted: %d edges split", lifted)
    return lifted


def _axis_of(shape) -> tuple:
    """A sphere's or cylinder's axis (None for a sphere) and radius."""
    if shape[0] == "sphere":
        return None, float(shape[2])
    d = np.asarray(shape[2], float)
    return d/np.linalg.norm(d), float(shape[3])


def _radial(points, shape, axis) -> np.ndarray:
    """The points' offsets from a sphere's centre or a cylinder's axis."""
    r = np.asarray(points, float) - np.asarray(shape[1], float)
    if axis is not None:
        r = r - np.outer(r @ axis, axis)
    return r


def _run(ev, tol: float, n: int) -> Any:
    """A descent's relax() run of up to n gradient steps; its Newton steps
    (and relax()'s conjugate-gradient pass) only once the residual is under
    _NEWTON_NEAR. Far from an equilibrium they took a 16-bead pack's falling
    surface 10 lower in energy in four steps, its residual 14 to 34, by
    putting contact lines onto the beads' poles, where the walls' integrals
    are singular: 0.025 radii from one where gradient steps alone left it
    0.14 away, run after run undone, the poles moved 20 times."""
    return ev.relax(tol, max_iter=n, newton=None if ev.residual() < _NEWTON_NEAR else 0)


def _gradient_run(ev, tol: float) -> Any:
    """A run of the descent after a change where relax()'s went wrong:
    _GRADIENT_RUN plain gradient steps (one, where those went wrong too;
    None where that did), then relax() if the residual is under _FALL_NEAR
    (kept if it converged and went right). relax()'s result, else an
    unconverged one."""
    def steps(n: int) -> list:
        out = []
        for _ in range(n):
            ev.command("g 1")
            out.append(_core.total_energy())
        return out
    trace, start, energy = _away_from_poles(ev, steps, _GRADIENT_RUN)
    if not _plausible(ev, energy, trace):
        ev.restore(start)
        trace, start, energy = _away_from_poles(ev, steps, 1)
        if not _plausible(ev, energy, trace):
            ev.restore(start)
            return None
    if ev.residual() < _FALL_NEAR:
        r, here, energy = _away_from_poles(ev, lambda n: _run(ev, tol, n), 1000)
        if r.converged and _plausible(ev, energy, r.energy):
            return r
        ev.restore(here)
    return _unconverged()


def _unconverged() -> Any:
    from ._results import IterationResult
    none = np.empty(0)
    return IterationResult(none, none, none, "", converged=False)


def _plausible(ev, before: tuple, trace=None) -> bool:
    """Whether a run of the descent kept its volumes (within 1e-3 of their
    targets, or nearer than where it began: a pack's patch made by
    dissolving left a volume 5.5% off, and the runs that brought it back,
    to 0.2%, were undone as gone wrong, the descent stopped), lowered the
    energy by less than a quarter of the area, kept the contact lines half
    _POLE_MARGIN from the poles of sphere(axis=)s
    (moved at the next iteration), and didn't pass the surface through
    itself: a run past that is one the integrals got wrong (relax() runs
    after a contact took a pack's liquid from 80.3 to 83.5, conjugate
    gradients a drop's from 31.5 to 44.5 and its energy to -12136; a pack's
    contact line came from half a radius to 0.07 of a pole in one run, the
    integrals off by up to 4 pi cos theta), or one that stepped past a
    merge (the surface through itself: 4 of the 18 runs of the corpus with
    the topology check, first after a relax() run), left no more free
    vertices through a wall (_outside), and Evolver didn't fail in it (an
    internal error: _away_from_poles put the surface back). ``before``: the energy, the
    crossings, the volumes' gap and the vertices through a wall where the
    run began (_away_from_poles). ``trace``: the energies of the run's gradient steps,
    if known: the energy then judged gone wrong only where it also jumped by
    a tenth of the area in one step (menisci merging beside a contact line
    gave up a quarter of a 4-bead pack's area in a relax() run, step by
    step, and every run after the merge was undone)."""
    return not _implausible(ev, before, trace)


def _implausible(ev, before: tuple, trace=None) -> str:
    """Why a run isn't plausible (see _plausible), "" if it is."""
    energy, crossed, gap = before[:3]
    out = before[3] if len(before) > 3 else 0
    area = _core.total_area()
    trace = np.asarray(trace if trace is not None else [], float)
    jumped = energy - ev.energy >= 0.25*area and (
        len(trace) < 2 or float(np.abs(np.diff(trace)).max()) >= 0.1*area)
    why = ("Evolver failed" if _RUN_FAILED else
           "volumes" if ev._target_gap()[0] > max(1e-3, gap) else
           "poles" if _near_poles(ev, _POLE_MARGIN/2) else
           "energy" if jumped else
           "crossings" if _crossings(ev) > crossed else
           "through a wall" if _outside(ev) > out else "")
    if why:
        _log.debug("implausible run: %s", why)
    if why == "crossings":
        _note_crossed(ev)
    return why


def _outside(ev) -> int:
    """How many free vertices are through a solid wall (a plane: a box's
    side, its floor; a sphere, a cylinder) by more than a tenth of an edge:
    the surface out of the box or inside a solid, which no change mends (a
    run took one 0.23 out beside a bead in a corner, and the descent ran
    away out there; one inside a bead takes a flux-gauged sphere's wetted
    area and the liquid's volume wrong by the bead's, and the descent ran
    away from that)."""
    walls = [k for k, w in surface.walls.items()
             if getattr(w, "shape", (None,))[0] in ("plane", "sphere", "cylinder")
             and getattr(w, "formula", None)]
    if not walls:
        return 0
    nums, _, dist, names = ev._constraint_gaps()
    cols = []
    for k in walls:
        try:
            cols.append(_column(nums, names, k))
        except ValueError:
            pass
    if not cols:
        return 0
    signed = np.nan_to_num(dist[:, cols], nan=0.0)
    return int((signed < -0.1*_edge_scale(ev)).any(axis=1).sum())


def _note_crossed(ev) -> None:
    """Where the surface has gone through itself here, for _change_nearest
    (which makes the change there first)."""
    from ._edits import crossing_places
    _CROSSED_AT[:] = [crossing_places(ev, far=True)]


_CROSSED_AT: list = []             # (where the last run judged crossed did)
_RUN_FAILED: list = []             # (Evolver's error in the last run, if it failed: _away_from_poles)


def _crossings(ev) -> int:
    """Edges through facets more than a ring of edges apart: the surface
    through itself (see _edits). (Not folds one ring across, beside contact
    lines at small angles: those the corpus's 16-bead runs met run after run
    after a contact, even evened out, and the runs got past them before;
    folds are the mesh's to mend.) (Kept for the same vertices and facets:
    the descents ask again of a state they asked of, 16 ms each on a 16-bead
    pack.)"""
    from ._edits import crossings
    m = ev.mesh()
    key = (hash(np.asarray(m.vertices).tobytes()), hash(np.asarray(m.facets).tobytes()))
    if _CROSSINGS[0] != key:
        _CROSSINGS[:] = [key, crossings(ev, far=True)]
    return int(_CROSSINGS[1])


_CROSSINGS: list = [None, 0]       # (the state's key, its crossings)


def _change_nearest(ev, factor: float, keep, skip=None, saved=None, refused=None, later=None,
                    other=None) -> Any:
    """Where the descent would pass the surface through itself (a run, a
    fall's steps or a step along the mode held back for it), the change
    nearest to coming made instead: of the events here, nearest first (gap
    over edges), the first apply() makes without crossings. Two pieces of a
    pack's liquid came together on a bead that way, their contact lines
    0.03 apart, and the descent, held back run after run, ran away. Where
    the last run judged crossed went through itself (_note_crossed), the
    events there come first, nearest the place first (the changes made
    instead had been elsewhere). (Within factor's gaps only: taken out to
    two edges there, a pack's descent made contacts 0.35 from beads;
    runs that cross are followed in shorter ones first, see _descend.)
    ``later``: whether an event's change is put off (_put_off), tried after
    the others there or elsewhere; ``other``: whether to make its other
    change (apply). The event made, or None."""
    found = sorted(ev.events(factor), key=lambda e: e.gap/float(e.scale))
    places = _CROSSED_AT.pop() if _CROSSED_AT else np.empty((0, 3))
    if len(places):
        def key(e) -> tuple:
            return (e.kind, e.constraint, tuple(np.round(np.asarray(e.where, float), 9)))
        there = []
        for e in ev.events(factor):
            d = float(np.linalg.norm(places - np.asarray(e.where, float), axis=1).min())
            if d < 2*float(e.scale):
                there.append((d, e))
        there.sort(key=lambda t: t[0])
        first = {key(e) for _, e in there}
        found = [e for _, e in there] + [e for e in found if key(e) not in first]
    else:
        there = []
    if later is not None:
        # (each group, there and elsewhere, the changes not put off first)
        n = len(there)
        found = sorted(found[:n], key=later) + sorted(found[n:], key=later)
    for e in found:
        if skip is not None and skip(e):
            continue
        why = _try_change(ev, e, keep, saved, other is not None and other(e))
        if why:
            _log.debug("change refused: %s: %s", e, why)
            if refused is not None:
                refused(e)
            continue
        _log.debug("change made (the nearest, the surface about to pass through itself): %s", e)
        return e
    return None


def _try_change(ev, e, keep, saved=None, other: bool = False) -> str:
    """The change for event e made (apply), unless it passes the surface
    through itself or leaves a surface Evolver can't take a gradient step
    from (a pack's merge made by facet_merge left a constraint matrix
    singular, and the run stopped at the next step): then undone. Why not
    ("" if made)."""
    crossed, start, energy = _crossings(ev), ev._save(), ev.energy
    try:
        apply(ev, e, keep, other)
        if _crossings(ev) > crossed:
            raise ValueError("the change passed the surface through itself")
        why = _unsteppable(ev, energy)
        if why:
            raise ValueError(f"Evolver can't step from it ({why})")
    except ValueError as error:
        ev.restore(start)
        return str(error)
    if saved is not None:
        saved.append(start)
    return ""


def _unsteppable(ev, energy=None) -> str:
    """Why Evolver can't take a gradient step from here (its error's first
    line; or, with the energy before a change, the step's energy that far
    from it, by a quarter of the area or more, as _plausible judges a run: a
    merge made by facet_merge beside a contact line, and every run after it
    was undone), "" if it can; the surface as it was either way."""
    from ._errors import EvolverError
    here = ev._save()
    try:
        ev.command("g 1")
        if energy is not None and abs(ev.energy - energy) >= 0.25*_core.total_area():
            return f"the energy {energy:.6g} to {ev.energy:.6g}"
    except EvolverError as error:
        return str(error).splitlines()[0] or "an error"
    finally:
        ev.restore(here)
    return ""


def _steps_uncrossed(ev, n: int) -> bool:
    """n gradient steps, or as many as don't pass the surface through itself
    nor take free vertices through a wall (_outside: a fall is unguarded,
    and a resumed pack's first steps took one 0.33 through the floor, a
    contact made there and the descent run away): where n do, the mesh
    evened out (smoothed, flipped) and n again (the crossings in the
    corpus's falls were folds an edge or two across, beside a contact line),
    then one at a time (a touch, a contact then comes as an event). Whether
    they were held back."""
    crossed, out, start = _crossings(ev), _outside(ev), ev._save()

    def wrong() -> bool:
        return _crossings(ev) > crossed or _outside(ev) > out
    ev.command(f"g {n}")
    if not wrong():
        return False
    ev.restore(start)
    _even_out(ev)
    crossed, out, start = _crossings(ev), _outside(ev), ev._save()
    ev.command(f"g {n}")
    if not wrong():
        return False
    ev.restore(start)
    for _ in range(n):
        here = ev._save()
        ev.command("g 1")
        if wrong():
            if _crossings(ev) > crossed:
                _note_crossed(ev)
            ev.restore(here)
            _log.debug("gradient steps stopped: the surface through itself or a wall")
            return True
    return False


def _drop_crushed(ev) -> None:
    """Facets crushed flat in a fall taken away (_edits' degenerate: a
    needle's short edge collapsed, a cap's long edge flipped, each checked).
    A fall doesn't remesh, and such facets set its gradient steps' scale: a
    pack's fall went 200 steps at residual 2, moving 1e-4 a step, past
    facets of 0.3 degrees; another's bunched three vertices 0.02 apart, edges
    0.13, and its steps folded the mesh there two rings across."""
    from ._edits import edit
    edit(ev, "degenerate", ev.mesh().facet_ids, flat=_FLAT)


def _remesh_now(ev, h0: float, near=None) -> None:
    """The surface remeshed toward edges of h0, finer where it comes close to
    a wall or itself (_sizes; _edits.remesh: splits,
    collapses, flips toward valence 6, smoothing along the surface, each
    checked), the guards off meanwhile (a guarded vertex is on one more
    constraint, which the edits keep). With ``near`` (points: where a change
    was made), only within _LOCAL h0 of them, and wherever an edge is twice
    its target or half it (remeshing all of a refined pack's surface after
    each of a descent's 200 changes took most of the descent)."""
    from ._edits import remesh
    guarded = _guarded(ev)
    _unguard(ev)
    try:
        sizes = _sizes(ev, h0)
        region = None
        if near is not None:
            m = ev.mesh()
            x, edges = np.asarray(m.vertices), np.asarray(m.edges)
            points = np.asarray(near, float).reshape(-1, 3)
            region = np.zeros(len(x), dtype=bool)
            for p in points:
                region |= np.linalg.norm(x - p, axis=1) < _LOCAL*h0
            if len(edges):
                ratio = m.edge_lengths()/sizes[edges].mean(axis=1)
                region[edges[(ratio > 2) | (ratio < 0.5)].ravel()] = True
            region = np.flatnonzero(region)
        done = remesh(ev, h0, sizes=sizes, region=region)
        _log.debug("remeshed: %s", done)
        _drop_crushed(ev)
    finally:
        if guarded:
            guard(ev)


def _even_out(ev, smooth: bool = True, edges=None) -> None:
    """The mesh evened by the checked edits (_edits): edges flipped where
    that makes the facets more equiangular, and with smooth, vertices moved
    toward their neighbours along the surface first."""
    from ._edits import edit
    if smooth:
        edit(ev, "smooth", ev.mesh().vertex_ids)
    m = ev.mesh()
    ids = np.asarray(m.edge_ids)[m.edge_valence() == 2] if edges is None else edges
    for _ in range(3):
        if not (edit(ev, "flip", ids, mode=1) == 0).any():
            break


def _same_state(state: tuple, last: Any) -> bool:
    """Whether a state (the counts, the energy) is the one before, the
    energy within 1e-7 of itself (a stalled pack's runs moved it by 3e-8)."""
    return last is not None and state[0] == last[0] \
        and abs(state[1] - last[1]) <= 1e-7*max(1.0, abs(last[1]))


def _mesh_only(health) -> bool:
    """Whether all a relaxed state lacks is a better mesh: its facets
    degenerate, or it comes near a wall or itself (as _sound)."""
    return health is not None and all(
        i.startswith(("the surface nearly touches", "degenerate facets")) for i in health.issues)


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


def _tidy(ev, h0: float, shut: list, close: bool = True) -> None:
    """A descent's tidying: necks collapsed below the mesh cut
    (_pinch_collapsed), closed pieces enclosing nothing taken away
    (_drop_empty), contact lines oriented, shrunk patches closed
    (close_patches), poles moved off the lines (move_poles); where that
    leaves a surface Evolver can't take a gradient step from (a pack's patch
    closed, then a pole moved, and the constraint matrix was singular at the
    next step), undone and done without the patches closed, else not done.
    Chords through a wall lifted first (_lift_chords)."""
    _lift_chords(ev)
    _pinch_collapsed(ev)
    _drop_empty(ev)
    good = ev._save()
    for close in ((True, False) if close else (False,)):
        orient_lines(ev)
        closed = close_patches(ev, h0, shut) if close else 0
        moved = move_poles(ev)
        if not (closed or moved) or not _unsteppable(ev):
            return
        _log.debug("tidying undone: Evolver can't step from it%s",
                   "" if close else " (without closing patches either)")
        ev.restore(good)
        if close and shut and closed:
            del shut[-closed:]


_PINCH_LEFT: set = set()      # _pinch_collapsed: necks (their vertices' ids) it couldn't cut, this descent


def _collapsed_rings(m, h: float) -> list:
    """Necks collapsed below the mesh: three vertices joined by edges under
    _PINCHED times the median edge h that are no facet (a tube three edges round, shrunk to a
    point). Rows of m's vertices, one triple per neck."""
    edges = np.asarray(m.edges)
    if not len(edges):
        return []
    short = edges[m.edge_lengths() < _PINCHED*h]
    if not len(short):
        return []
    facets = {tuple(sorted(t)) for t in np.asarray(m.facets).tolist()}
    joined: dict = {}
    for a, b in short.tolist():
        joined.setdefault(a, set()).add(b)
        joined.setdefault(b, set()).add(a)
    rings = set()
    for a, b in short.tolist():
        for c in joined[a] & joined[b]:
            t = tuple(sorted((a, b, c)))
            if t not in facets:
                rings.add(t)
    return [np.array(t) for t in sorted(rings)]


def _ring_sides(f, ring) -> "np.ndarray | None":
    """Each vertex's side of a collapsed neck's ring (+1, -1; 0 on the ring
    or away from it): the facets round the ring in two pieces joined across
    edges off it. None where they aren't two."""
    touching = np.flatnonzero(np.isin(f, ring).any(axis=1))
    on = set(int(v) for v in ring)
    parent = {int(k): int(k) for k in touching}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    by_edge: dict = {}
    for k in touching:
        for i in range(3):
            a, b = int(f[k, i]), int(f[k, (i + 1) % 3])
            if a in on and b in on:
                continue                # an edge of the ring: across the cut
            key = (min(a, b), max(a, b))
            if key in by_edge:
                parent[root(int(k))] = root(by_edge[key])
            else:
                by_edge[key] = int(k)
    pieces = sorted({root(int(k)) for k in touching})
    if len(pieces) != 2:
        return None
    side = np.zeros(int(f.max()) + 1)
    for k in touching:
        sign = 1.0 if root(int(k)) == pieces[0] else -1.0
        for v in f[k]:
            if int(v) in on:
                continue
            if side[v] == -sign:
                return None             # a vertex on both sides: no neck
            side[v] = sign
    return side


def _pinch_collapsed(ev) -> int:
    """Necks collapsed below the mesh (_collapsed_rings) cut as pinch_off
    cuts a neck: a 16-bead pack's gradient runs squeezed tubes of liquid and
    gas between beads 0.03 apart to three vertices 1e-4 apart, too small for
    events() to see as necks (their vertices are each other's neighbours),
    and Evolver's 't' merged them later, pinching the surface along edges
    (relax()'s last resort, now held in descents); left, they crossed run
    after run. Each cut kept only if it passes the surface through nothing
    and Evolver can step from it; not where the ring reaches a wall, a
    fixed vertex or the surface's border. How many."""
    made = 0
    for _ in range(20):
        m = ev.mesh()
        if m.facets is None or not len(m.facets):
            break
        h = _edge_scale(ev)
        x, f = np.asarray(m.vertices, float), np.asarray(m.facets)
        border = np.zeros(len(x), dtype=bool)
        border[np.asarray(m.edges)[m.edge_valence() == 1].ravel()] = True
        _, _, gaps, _ = ev._constraint_gaps()
        on = np.isnan(gaps).any(axis=1) if gaps.size else np.zeros(len(x), dtype=bool)
        held = np.asarray(m.fixed, dtype=bool) | border | on
        cut = False
        ids = np.asarray(m.vertex_ids)
        for ring in _collapsed_rings(m, h):
            key = frozenset(int(i) for i in ids[ring])
            side = None if held[ring].any() or key in _PINCH_LEFT else _ring_sides(f, ring)
            if side is None:
                continue
            centre = x[ring].mean(axis=0)
            across = float(max(np.linalg.norm(x[ring[i]] - x[ring[(i + 1) % 3]])
                               for i in range(3)))
            crossed, start, energy = _crossings(ev), ev._save(), ev.energy
            guarded = _guarded(ev)
            _unguard(ev)
            try:
                _pinch_ring(ev, m, x, f, ring, centre, side, h)
                if _crossings(ev) > crossed:
                    raise ValueError("the cut passed the surface through itself")
                why = _unsteppable(ev, energy)
                if why:
                    raise ValueError(f"Evolver can't step from it ({why})")
            except ValueError as error:
                ev.restore(start)
                _PINCH_LEFT.add(key)
                _log.debug("collapsed neck at (%s) left: %s",
                           ", ".join(f"{c:.4g}" for c in centre), error)
                continue
            finally:
                if guarded:
                    guard(ev)
            _log.debug("change made: a neck collapsed below the mesh, %.2g across at (%s), cut",
                       across, ", ".join(f"{c:.4g}" for c in centre))
            made += 1
            cut = True
            break
        if not cut:
            break
    return made


def _drop_empty(ev) -> int:
    """Closed pieces of the surface (edge-connected, no contact line nor
    other border) enclosing next to no volume (under 1e-6 of their area to
    the 3/2), or within two median edges across, taken away with the
    bodies they alone bound: a bubble with no gas in it, a crumb of liquid;
    so are bodies under half an edge across, and pieces with a border but
    next to no area (under a hundredth of an edge squared). (Two drops
    merging on a plane, a second merge beside the first left a sheet of 16
    facets, one edge on three of them, its eight vertices at one point, and
    the surface came out in two pieces.) How many."""
    m = ev.mesh()
    if m.facets is None or not len(m.facets):
        return 0
    x, f = np.asarray(m.vertices, float), np.asarray(m.facets)
    h = _edge_scale(ev)
    sides = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    pairs, edge = np.unique(sides, axis=0, return_inverse=True)
    edge = edge.ravel().reshape(3, -1).T
    count = np.bincount(edge.ravel(), minlength=len(pairs))
    label = np.arange(len(f))
    while True:
        low = np.full(len(pairs), len(f))
        np.minimum.at(low, edge.ravel(), np.repeat(label, 3))
        new = low[edge].min(axis=1)
        if np.array_equal(new, label):
            break
        label = new
    if len(np.unique(label)) < 2:
        return 0
    gone = []
    lost = 0.0
    areas = 0.5*np.linalg.norm(np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]]), axis=1)
    for c in np.unique(label):
        mine = label == c
        if (count[np.unique(edge[mine])] == 1).any():
            # a border (a contact line); a crumb where it has next to no
            # area (patches made where a box's walls meet left lone facets
            # lying along the edge or in the corner, their corners on the
            # walls, all three edges border: nothing could take them, and a
            # pack's relaxed state kept them as facets of 0 degrees, the
            # descent going on down from it 195 times in two hours)
            if float(areas[mine].sum()) < 0.01*h*h:
                gone.append(mine)
            continue
        # (about its own middle; or flat, its vertices in one plane: with an
        # edge on three facets the piece doesn't close, and its volume
        # depends on where it's measured from)
        vs = np.unique(f[mine])
        middle = x[vs].mean(axis=0)
        a, b, d = x[f[mine, 0]] - middle, x[f[mine, 1]] - middle, x[f[mine, 2]] - middle
        volume = abs(float(np.einsum("ij,ij->i", a, np.cross(b, d)).sum()))/6
        area = 0.5*float(np.linalg.norm(np.cross(b - a, d - a), axis=1).sum())
        spread = np.linalg.svd(x[vs] - middle, compute_uv=False)
        flat = len(spread) == 3 and spread[2] < 1e-6*spread[0]
        # (or no size at all: that sheet's eight vertices were at one point;
        # or within two edges across, a crumb the mesh can't hold: necks cut
        # below the mesh left drops of 4 and 10 facets, volumes 1e-5 and
        # 1e-10, their own bodies, collapsing as their volumes were held)
        size = float(np.ptp(x[vs], axis=0).max())
        if volume < 1e-6*area**1.5 or flat or spread[0] < 1e-6*h or size < 2*h:
            gone.append(mine)
            lost += volume
    # (and bodies under half an edge across, contact lines or not: a crumb
    # of liquid on a bead, volume 6e-10 in 24 facets, left by a neck cut
    # below the mesh; its volume 100 times off after any run, and every
    # run after it was undone. Not one sharing facets with another body.)
    fb = np.asarray(m.facet_bodies) if m.facet_bodies is not None else np.zeros((len(f), 2), int)
    for b in ev.bodies().ids:
        b = int(b)
        mine = (fb == b).any(axis=1)
        if not mine.any() or ((fb[mine] != b) & (fb[mine] > 0)).any():
            continue
        volume = abs(float(ev.eval(f"body[{b}].volume")))
        if volume < (h/2)**3 and not any(g[mine].all() for g in gone):
            gone.append(mine)
            lost += volume
    if not gone:
        return 0
    taken = np.any(gone, axis=0)
    ids = np.asarray(m.facet_ids)[taken]
    # (and the bodies they alone bounded: a body left without facets, its
    # volume still fixed, made Evolver's constraint projection singular,
    # ERROR 3005; a neck cut below the mesh had split one off)
    empty: list = []
    if m.facet_bodies is not None:
        fb = np.asarray(m.facet_bodies)
        kept = {int(b) for b in fb[~taken].ravel() if b > 0}
        empty = sorted({int(b) for b in fb[taken].ravel() if b > 0} - kept)
    ev.command("; ".join(f"dissolve facet[{int(i)}]" for i in ids))
    ev.command("dissolve edges; dissolve vertices")
    for b in empty:
        ev.command(f"dissolve body[{b}]")
    _log.debug("closed pieces enclosing nothing, crumbs, taken away: %d (%d facets, volume %.2g%s)",
               len(gone), len(ids), lost, f"; bodies {empty}" if empty else "")
    return len(gone)


def _reopening(e, shut: list, h: float) -> bool:
    """Whether a solid event is a dry patch (the liquid between thinning)
    within 3 h of a patch closed on its solid."""
    return e.kind == "solid" and e.between == "body" and any(
        str(key) == str(e.constraint) and np.linalg.norm(np.asarray(e.where) - middle) < 3*h
        for key, middle in shut)


class _Tried(NamedTuple):
    """A change a descent made, undid or refused (``what``): its event's
    kind, the solids it is on or against (_solids_at), where, its edges
    there and its gap in them."""
    kind: str
    solids: frozenset
    where: np.ndarray
    scale: float
    gap: float
    what: str


def _tried(ev, e, what: str) -> _Tried:
    return _Tried(e.kind, _solids_at(ev, e), np.asarray(e.where, float), float(e.scale),
                  float(e.gap)/max(float(e.scale), 1e-300), what)


def _solids_at(ev, e) -> frozenset:
    """The solids an event is on or against: a solid's or contact lines'
    own (its base, for a copy of a sphere: _base); a neck's or touch's,
    those with a vertex on them within its gap and two edges of its middle
    (a sheet thinnest at a side wall, a neck in the wedge between a bead and
    a wall)."""
    if e.kind in ("solid", "contact lines"):
        return frozenset([_base(e.constraint)])
    nums, _, dist, names = ev._constraint_gaps()
    if not dist.size:
        return frozenset()
    reach = abs(float(e.gap)) + 2*float(e.scale)
    near = np.linalg.norm(ev.vertices - np.asarray(e.where, float), axis=1) < reach
    on = np.isnan(dist[near]).any(axis=0)
    return frozenset(_base(names[c] or int(nums[c])) for c in np.flatnonzero(on))


def _same_place(a: _Tried, b: _Tried) -> bool:
    """Whether two changes are at one place: on (or against) the same solid,
    or both on none, within _PLACE of the larger of their edges."""
    return (bool(a.solids & b.solids) or not (a.solids or b.solids)) \
        and float(np.linalg.norm(a.where - b.where)) < _PLACE*max(a.scale, b.scale)


def _put_off(ev, e, history: list) -> bool:
    """Whether a descent puts e's change off: one was made or undone at its
    place in this descent (the same change, which didn't take, or the
    opposite: cut and joined, dried and wetted), or one of its kind refused
    there, and e isn't nearer than _MARGIN of that one's gap (in their
    edges; through the solid deeper than 1/_MARGIN of it, where that one
    was at it or through; a neck, touch or contact lines met). Made at the
    same gap, a change and its opposite take
    turns: a pack's sheet at a side wall had its footprint dried three
    times and its lines joined back, the runs between crossing there, until
    the descent ran away. The guards and the descent's runs hold the
    surface meanwhile; a neck that pinches, a film that thins, comes nearer
    and is made then (and where a run would cross, _change_nearest still
    makes it, after the others)."""
    if not history:
        return False
    here = _tried(ev, e, "")
    gaps = [t.gap for t in history if _same_place(here, t)
            and (t.what != "refused" or t.kind == e.kind)]
    if not gaps:
        return False
    g = min(gaps)
    # (a neck, touch or contact lines can't come nearer than 0: there, met)
    return not (here.gap < (g*_MARGIN if g > 0 else g/_MARGIN)
                or (e.kind != "solid" and here.gap <= 0))


def _coming(found: list, before: list) -> list:
    from .recipes import _coming as coming
    return coming(found, before)


def _keep_edges(ev, h0: float, hi: Optional[float] = None) -> None:
    """Edges back in range after a change, by the checked edits (_edits):
    those shorter than 0.4 of their target (h0, finer where the surface
    comes close to a wall or itself: _sizes) collapsed, contact-line edges
    too (a joined contact line kept its short ones, and its facets
    degenerated as it moved), never a contact line's vertex inward (that
    pinched two contact lines together at a vertex: they meet by contact()
    instead), no edge made longer than twice its target (``hi`` times it);
    those longer than that split (not where the midpoint lands inside a
    solid: an edge into the surface from a contact line); then flipped where
    that makes the facets more equiangular."""
    lo = _REMESH[0]
    hi = _REMESH[1] if hi is None else hi
    m = ev.mesh()
    lengths = m.edge_lengths()
    # (against each edge's own target: tested against h0, edges of 0.2
    # where a pack's contact lines met 0.008 apart were never split)
    sizes = _sizes(ev, h0)
    target = sizes[np.asarray(m.edges)].mean(axis=1) if len(m.edges) else lengths
    if not ((lengths < lo*target).any() or (lengths > hi*target).any()):
        return
    guarded = _guarded(ev)
    _unguard(ev)
    try:
        _keep_edges_unguarded(ev, h0, m, sizes, hi)
    finally:
        if guarded:
            guard(ev)


def _carried(m, ids, sizes, h0: float) -> np.ndarray:
    """The target sizes (per row of an earlier mesh whose vertex ids were
    ids) carried onto mesh m by id: a vertex an edit made takes its
    neighbours' mean, as the edits give a split's new vertex its ends' (h0
    where none is known). (Not found afresh after each pass of edits: the
    event search was a fifth of a 16-bead pack's descent, and each pass let
    the targets grow by _COARSEN again.)"""
    known = dict(zip(np.asarray(ids).tolist(), np.asarray(sizes, float).tolist()))
    out = np.array([known.get(int(i), np.nan) for i in m.vertex_ids])
    edges = np.asarray(m.edges)
    for _ in range(3):
        new = np.isnan(out)
        if not new.any() or not len(edges):
            break
        total = np.zeros(len(out))
        count = np.zeros(len(out))
        for a, b in ((0, 1), (1, 0)):
            ok = new[edges[:, a]] & ~new[edges[:, b]]
            np.add.at(total, edges[ok, a], out[edges[ok, b]])
            np.add.at(count, edges[ok, a], 1)
        fill = new & (count > 0)
        out[fill] = total[fill]/count[fill]
    return np.where(np.isnan(out), float(h0), out)


def _keep_edges_unguarded(ev, h0: float, m, sizes=None, hi: Optional[float] = None) -> None:
    from ._edits import edit
    lo = _REMESH[0]
    hi = _REMESH[1] if hi is None else hi
    if sizes is None:
        sizes = _sizes(ev, h0)
    for _ in range(5):
        target = sizes[np.asarray(m.edges)].mean(axis=1)
        lengths = m.edge_lengths()
        short = np.flatnonzero(lengths < lo*target)
        if not len(short):
            break
        order = short[np.argsort(lengths[short]/target[short])]
        if not (edit(ev, "collapse", np.asarray(m.edge_ids)[order], sizes=sizes, max_len=hi,
                     h=h0) == 0).any():
            break
        ids, m = m.vertex_ids, ev.mesh()
        sizes = _carried(m, ids, sizes, h0)
    for _ in range(5):
        target = sizes[np.asarray(m.edges)].mean(axis=1)
        lengths = m.edge_lengths()
        long = np.flatnonzero(lengths > hi*target)
        if not len(long):
            break
        order = long[np.argsort(-lengths[long]/target[long])]
        if not (edit(ev, "split", np.asarray(m.edge_ids)[order], sizes=sizes, h=h0) == 0).any():
            break
        ids, m = m.vertex_ids, ev.mesh()
        sizes = _carried(m, ids, sizes, h0)
    _even_out(ev, smooth=False)
    _drop_crushed(ev)


def _coarsen(ev, h0: float) -> None:
    """Edges shorter than half their targets (_sizes) collapsed, the whole
    surface over: at the start of each descent (not under 0.8, a remeshing's
    collapse, without its splits: a bridge's even mesh lost a third of its
    facets at its first descent, and every remeshing after refined it again)
    (the descents remesh only where their changes were, and a pack's mesh
    refined there, the targets back to h0, stayed at 0.39 of them: 7.6k
    facets, 2.7k after this)."""
    from ._edits import edit
    guarded = _guarded(ev)
    _unguard(ev)
    try:
        m = ev.mesh()
        sizes = _sizes(ev, h0)
        edges = np.asarray(m.edges)
        if not len(edges):
            return
        lengths = m.edge_lengths()
        target = sizes[edges].mean(axis=1)
        short = np.flatnonzero(lengths < 0.5*target)
        if len(short):
            order = short[np.argsort(lengths[short]/target[short])]
            done = edit(ev, "collapse", np.asarray(m.edge_ids)[order], sizes=sizes,
                        max_len=4/3, h=h0)
            _log.debug("coarsened: %d edges collapsed", int((done == 0).sum()))
            _even_out(ev, smooth=False)
            _drop_crushed(ev)
    finally:
        if guarded:
            guard(ev)


def _held_sizes(numbers: np.ndarray) -> np.ndarray:
    """The last targets (surface.sizes) of the vertices with these numbers,
    inf where none (a table indexed by number: a lookup a vertex took 5 ms
    at 30k facets)."""
    numbers = np.asarray(numbers, dtype=np.int64)
    out = np.full(len(numbers), np.inf)
    if not surface.sizes or not len(numbers):
        return out
    keys = np.fromiter(surface.sizes.keys(), dtype=np.int64, count=len(surface.sizes))
    values = np.fromiter(surface.sizes.values(), dtype=float, count=len(surface.sizes))
    top = int(max(keys.max(), numbers.max())) + 1
    if keys.min() < 0 or numbers.min() < 0 or top > 8*(len(keys) + len(numbers)) + 1024:
        return np.array([surface.sizes.get(int(i), np.inf) for i in numbers])
    table = np.full(top, np.inf)
    table[keys] = values
    return table[numbers]


def _sizes(ev, h0: float) -> np.ndarray:
    """A target edge length per vertex for a descent's remeshing: h0, but
    the gap where the surface comes within h0 of a wall it isn't on (away
    from its contact lines there) with another wall within h0 too, of
    another contact line on its wall, or of itself (the events: at a 16-bead
    pack's near-contacts contact lines
    met across 0.08 on edges of 0.2, and neither joining nor drying them
    could be made without the surface through itself), down to _FINE h0;
    graded out from there, no vertex's more than a neighbour's and _GRADE
    of the edge between. (The events taken out to h0 whatever the edges:
    looked for in local edges only, a gap the finer mesh resolved would no
    longer be found, and the next remeshing coarsened it again.)"""
    m = ev.mesh()
    size = np.full(len(m.vertices), float(h0))
    edges = np.asarray(m.edges)
    if not len(edges):
        return size
    x = np.asarray(m.vertices, float)
    # (events() measures in local edges, the finest _FINE h0: so out to h0;
    # the surface close to itself within an edge, necks NECK times that)
    from ._events import find, _base as base_of
    nums, _, dist, names = ev._constraint_gaps()
    keys = [str(n or int(k)) for k, n in zip(nums, names)]
    for e in find(ev, 1/_FINE, self_factor=1.0):
        if e.kind == "solid":
            # (a film thinning over one solid has nothing narrow to resolve:
            # refined, its contact only came at a smaller gap, a pack's
            # facets 2.2k to 4.6k on the way to one. Only its vertices within
            # h0 of another wall too, the surface between two solids, each
            # to the room it has there, its gaps to both: a pack's sheet of
            # 161 vertices coming to a bead, one of them by another, was
            # refined all over)
            if abs(e.gap) >= h0:
                continue
            mine = base_of(str(e.constraint))
            rows = np.asarray(e.vertices, int)
            rows = rows[(rows >= 0) & (rows < len(size))]
            others = [c for c, k in enumerate(keys) if base_of(k) != mine]
            col = [c for c, k in enumerate(keys) if base_of(k) == mine]
            if not others or not col or not len(rows):
                continue
            other = np.abs(np.nan_to_num(dist[np.ix_(rows, np.asarray(others))], nan=np.inf)).min(axis=1)
            here = np.abs(np.nan_to_num(dist[np.ix_(rows, np.asarray(col))], nan=np.inf)).min(axis=1)
            between = other < h0
            room = np.minimum(here, abs(float(e.gap)) + h0) + other
            size[rows[between]] = np.minimum(size[rows[between]], room[between])
            continue
        if abs(e.gap) < h0:
            rows = np.asarray(e.vertices, int)
            rows = rows[(rows >= 0) & (rows < len(size))]
            if e.kind == "neck":
                # (a neck's waist only, within h0 of its narrowest point: its
                # event's vertices were a whole sheet's, and a pack's 1423 of
                # them refined to its thickness took the mesh from 2.1k
                # facets to 6.1k, the median edge to a third. A sheet's too,
                # where its hole opens: left at edges of 0.19 over a gap of
                # 0.0014, the facing facets' merge folded the mesh, refused
                # at every pair, and the descent ran away)
                near = np.linalg.norm(x[rows] - np.asarray(e.where, float), axis=1) < h0
                rows = rows[near]
            size[rows] = np.minimum(size[rows], abs(float(e.gap)))
    # (coarsened back at most _COARSEN times a remeshing: a gap the finer mesh
    # resolved, its event found at a slightly different place or not at all,
    # coarsened, refined at the next: a pack's remeshings split and collapsed
    # a thousand edges each)
    # (within a descent not at all where anything is still within h0: it
    # remeshes many times, and a pack's refined places were coarsened and
    # refined again within one, its facets going 1.3k, 2.8k, 1.4k in four
    # steps, slivers left by the collapses; but where nothing is, as before:
    # held everywhere, a pack's long descent kept what the front had passed
    # refined, its facets 3.6k to 8.7k, a remeshing 38 s)
    numbers = np.asarray(m.vertex_ids)
    before = _held_sizes(numbers)
    grow = np.where(size >= h0, _COARSEN, 1.0 if surface.sizes_held else _COARSEN)
    size = np.minimum(size, grow*before)
    size = np.clip(size, _FINE*h0, h0)
    # (each vertex's neighbours' targets plus _GRADE of the edge, the least
    # of them, by the edges' ends sorted once: np.minimum.at, unbuffered,
    # took 3 ms a round at 30k facets)
    lengths = m.edge_lengths()
    ends = np.concatenate([edges[:, 0], edges[:, 1]])
    order = np.argsort(ends, kind="stable")
    ends = ends[order]
    other = np.concatenate([edges[:, 1], edges[:, 0]])[order]
    step = _GRADE*np.concatenate([lengths, lengths])[order]
    first = np.flatnonzero(np.r_[True, ends[1:] != ends[:-1]])
    at = ends[first]
    for _ in range(30):
        new = size.copy()
        new[at] = np.minimum(new[at], np.minimum.reduceat(size[other] + step, first))
        if np.array_equal(new, size):
            break
        size = new
    keep = size < h0
    surface.sizes = dict(zip(numbers[keep].tolist(), size[keep].tolist()))
    return size


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
        _, grad = ev._call(_core.constraint_normals_at, int(nums[c]),
                           np.ascontiguousarray(rows, dtype=np.int64))
        normal = np.asarray(grad.data)
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
