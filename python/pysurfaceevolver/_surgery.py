"""Changes of topology made with Evolver's own commands (see
:meth:`Evolver.pinch_off`)."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, List, Optional

import numpy as np

from . import _core
from ._engine import GUARD_PREFIX, surface
from ._errors import _num

_SEAM = 3.0     # the cut's tips are tidied within this many median edges
_TOUCH = 0.05   # a solid this close (in its event's edges): the surface resting on it
_GUARD_REACH = 5.0  # free vertices are guarded from the walls within this many median edges
_POLE_MARGIN = 0.5  # a sphere(axis=)'s pole moved when a contact line comes this near (radii)
_POLE_TRIES = 3     # a run that came within half that done again this often, the poles moved

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
    # (not two whose normals there are within 10 degrees of each other: a
    # free vertex in the 0.015 gap where a bead touches a side wall, on both
    # guards, left Evolver's constraint projection singular, ERROR 3005; the
    # nearer one only)
    both = np.flatnonzero(chosen.sum(axis=1) == 2)
    if len(both):
        normals = {}
        for j in np.unique(order[both]):
            _, grad = ev._call(_core.constraint_normals, int(nums[cols[j]]))
            normals[int(j)] = np.asarray(grad.data)
        a = np.array([normals[int(j)][i] for i, j in zip(both, order[both, 0])])
        b = np.array([normals[int(j)][i] for i, j in zip(both, order[both, 1])])
        cos = np.abs(np.einsum("ij,ij->i", a, b)) / np.maximum(
            np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-300)
        tangent = both[cos > 0.985]
        chosen[tangent, order[tangent, 1]] = False
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
    for key, wall, old, centre, R, line, near, _, a, b, sides, on, e in _pole_lines(ev):
        if near >= _POLE_MARGIN and key not in force:
            continue
        pole = np.asarray(getattr(wall, "pole_axis"), float)
        unit = lambda p: (p - centre)/R     # noqa: E731
        swept = key in force and ahead is not None and len(ahead) == len(x)
        loops = []
        for rows in _loop_rows(e):
            vs = np.unique(e[rows])
            points = unit(x[vs])
            if swept:
                there = unit(ahead[vs])
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
    """run(n) from here (a run of n steps); where that brought a contact line
    within half _POLE_MARGIN of a sphere(axis=)'s pole, or took it across
    (its solid angle (_fans) jumped by over 2 pi: a pack's line swept across
    in one relax() run, from 0.83 to 0.32 beyond), run again from here with
    those poles moved (move_poles, away from where the run took the lines
    too), and where it does so again, shorter runs (a quarter as many steps
    each time, down to one): the poles move on between them (a 16-bead
    pack's relax() runs moved vertices 0.8 to 2 radii after a change, the
    lines swept across most of a bead, no pole was clear of them). Returns
    run's result, and the state it started from with the energy, the
    surface's crossings of itself and the volumes' gap from their targets
    there (for _plausible)."""
    start, energy = ev._save(), (ev.energy, _crossings(ev), ev._target_gap()[0])
    fans = {p[0]: p[7] for p in _pole_lines(ev)}
    result = run(n)
    for attempt in range(_POLE_TRIES + 3):
        bad = [p[0] for p in _pole_lines(ev) if p[6] < _POLE_MARGIN/2
               or (p[0] in fans and abs(p[7] - fans[p[0]]) > 2*np.pi)]
        if not bad:
            break
        _log.debug("run undone: a contact line came near or across a pole (%s)", bad)
        ahead = ev.vertices
        ev.restore(start)
        move_poles(ev, force=bad, ahead=ahead)
        if attempt and n > 1:
            n = max(1, n//4)
            _log.debug("runs shortened to %d steps (lines sweeping across poles)", n)
        start, energy = ev._save(), (ev.energy, _crossings(ev), ev._target_gap()[0])
        fans = {p[0]: p[7] for p in _pole_lines(ev)}
        result = run(n)
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
    degenerated (0.3 degrees). (Only the guards some vertex is on, in one
    command: a command per guard, 21 a call, took 6% of a 16-bead run.)"""
    _, _, dist, all_names = ev._constraint_gaps(guards=True)
    used = [str(n) for c, n in enumerate(all_names)
            if str(n).startswith(GUARD_PREFIX) and np.isnan(dist[:, c]).any()
            and (names is None or n in names)]
    if used:
        ev.command("; ".join(f"unset vertex constraint {n}" for n in used))


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
    on = np.isnan(gaps) if gaps.size else np.zeros((len(x), 0), dtype=bool)
    held = m.fixed | border | on.any(axis=1)
    if held[ring].any():
        # a neck against one wall (half a neck, its ring from the wall's
        # contact line round to it again: liquid against a side wall or a
        # bead, 86 and 75 of the 161 necks a round of 16-bead runs refused):
        # the ring collapses to its vertex on the wall nearest its middle, the
        # contact line then goes through that vertex twice, and the
        # separation splits it, a line per side
        walls = np.flatnonzero(on[ring].any(axis=0))
        lone = (len(walls) == 1 and not m.fixed[ring].any()
                and (on[ring].sum(axis=1) <= 1).all()
                and not (border[ring] & ~on[ring].any(axis=1)).any())
        if not lone:
            raise ValueError("pinch_off(): the neck reaches a fixed vertex, two constraints "
                             "or the surface's free border")
        at_wall = ring[on[ring, walls[0]]]
        centre = x[at_wall[np.argmin(np.linalg.norm(x[at_wall] - centre, axis=1))]]
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
    snapshot = ev._save()
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
    f = np.asarray(m.facets)
    # (a neck against a wall: its ring's two ends on the wall's contact line
    # stay two vertices, Evolver deleting no edge between two of a border's
    # vertices that would pinch it; each is then where the sides meet)
    border = np.zeros(len(m.vertices), dtype=bool)
    border[np.asarray(m.edges)[m.edge_valence() == 1].ravel()] = True
    if len(at) != 1 and not (len(at) == 2 and border[at].all()):
        raise ValueError(f"pinch_off(): the ring didn't collapse to one vertex ({len(at)} left)")
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


def _split_body(ev, body: int, before: dict, near: dict, lost: dict) -> List[int]:
    """After the cut: the body's pieces made bodies (Evolver's ``rebody``;
    none where the cut took a handle off it), the largest keeping its number; each piece's volconst from the walls it
    reaches; and their targets, if the volume was fixed: each smaller piece
    what it holds plus what the collapse took from its side, the largest the
    rest of the old target, so they add up to it."""
    old = set(int(i) for i in ev.bodies().ids)
    _rebody(ev)
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
    made = _tube(ev, m, a, b, keep)
    if made is not None:
        return made
    return _glue(ev, m, a, b, keep)


def _tube(ev, m, a: int, b: int, keep) -> "int | None":
    """merge() by the checked edits (_edits): facet rows a and b taken out
    and a short tube of facets joining their edges; the bodies either side
    made one as _glue makes them. None where the edit is refused (then
    _glue's Evolver facet_merge is tried)."""
    from ._edits import REASONS, edit
    fb = np.asarray(m.facet_bodies)
    bodies = sorted({int(v) for v in fb[[a, b]].ravel() if v > 0})
    if len(bodies) > 2:
        return None
    states = {i: _body_state(ev, i) for i in bodies}
    r = edit(ev, "merge", [int(m.facet_ids[a]), int(m.facet_ids[b])])
    if r[0] != 0:
        _log.debug("fallback: merge by facet_merge (the tube refused: %s)", REASONS[int(r[0])])
        return None
    return _make_one(ev, bodies, keep, states)


def _make_one(ev, bodies: list, keep, states: dict, volume=None) -> int:
    """The bodies made one: the one numbered keep if it is one of them, else
    the lower; its volconst theirs added (and with volume, the volumes before
    a change kept), its target their targets added (free if either was).
    The merged body's number (0 without bodies)."""
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
            body.volconst = body.volconst + sum(volume.values()) - body.volume
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
    _dissolve_onto(ev, m, gone, event.constraint)


def _dissolve_onto(ev, m, gone, key) -> None:
    """The facets gone (mask over m's rows) taken away, their vertices put
    onto the wall key first, the new border edges on it as contact lines,
    the volumes kept (volconst)."""
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
            b.volconst = b.volconst + volume[i] - b.volume


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
    piling vertices up.) The lines oriented (orient_lines). Returns how many
    were closed; each one's wall and middle appended to ``out``."""
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
            if np.linalg.norm(x[corners] - x[corners[0]], axis=1).max() > 0.2*h:
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
            commands += [f"unset edge[{e}] constraint {names[key]}" for e in ids]
            commands += [f"unset vertex[{int(m.vertex_ids[c])}] constraint {names[key]}"
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
                _make_one(ev, bodies, keep, states, volume)
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
        body.volconst = body.volconst + sum(volume.values()) - body.volume
        if all(s["fixed"] for s in states.values()):
            body.target = sum(s["target"] for s in states.values())
        else:
            body.target = None


_FALL_STEP = 0.3        # a fall's step along the unstable mode: at most this many median edges
_FALL_GRADIENT = 5      # gradient steps after each
_FALL_ITERATIONS = 200     # iterations of a descent since its last change, at most
_FALL_TOTAL = 10           # ... and this many times that in all
_FALL_NEAR = 0.05       # the residual below which Newton steps try to finish a descent
_FALL_TRY = 10          # iterations between such tries
_FLAT = 0.05            # a facet crushed flat: its area under this many h^2 (h the mean edge)
_AGAIN = 3              # a change made this often at one place in a descent: not again there
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
        result = _descend(ev, factor, tol, falling=falling, keep=keep, h0=h0, changes=changes)
        if result[2]:
            _log.debug("descent stopped: %s", result[2])
        return result
    except EvolverError as error:
        _log.debug("descent stopped: Evolver: %s", error)
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
    from ._errors import EvolverError
    way: Any = None
    before: list = []
    refused = ""
    tried = _FALL_TRY
    remeshing = not falling
    if h0 is None:
        h0 = _median_edge(ev.mesh())
    if not falling:
        guard(ev)
    since, stuck = 0, False
    shut: list = []                     # patches closed in this descent: (wall, middle)
    made_at: list = []                  # changes made in it: (kind, solid, place)
    pre: list = []                      # the state before the change _try_change made
    undo: list = []                     # (that state, the change) for the last one made

    def again(e) -> bool:
        """Whether e's change was made _AGAIN times here already (a pack's
        forced join of contact lines where a side wall meets the floor was
        made 160 times, each run after it through itself, until the run
        timed out)."""
        h = _median_edge(ev.mesh())
        return sum(k == e.kind and c == str(e.constraint)
                   and float(np.linalg.norm(np.asarray(e.where) - w)) < h
                   for k, c, w in made_at) >= _AGAIN

    def made_change(e) -> None:
        """A change made: noted, the mesh remeshed, and down from there."""
        nonlocal way, before, tried, since, stuck, falling, remeshing
        _checked(ev, f"the change: {e}")
        made_at.append((e.kind, str(e.constraint), np.asarray(e.where, float)))
        undo[:] = [(pre.pop(), e)] if pre else []
        pre.clear()
        _remesh_now(ev, h0)
        _checked(ev, "remeshing after the change")
        changes.append(e)
        guard(ev)
        way, before, tried, since, stuck = None, [], 0, 0, False
        falling, remeshing = False, True

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
                _log.debug("descent remeshed: no stable state within %d steps", _FALL_ITERATIONS)
                _remesh_now(ev, h0)
                _checked(ev, "remeshing a descent that got nowhere")
                way, before = None, []
            _tidy(ev, h0, shut)
            _checked(ev, "tidying (lines, patches, poles)")
            x = ev.vertices
            chunk = None
            if falling:
                _drop_crushed(ev)
                if _steps_uncrossed(ev, _FALL_GRADIENT):
                    _unguard(ev)
                    forced = _change_nearest(ev, factor, keep, again, pre)
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
                chunk, start, energy = _away_from_poles(ev, lambda n: ev.relax(tol, max_iter=n),
                                                        _CHUNK)
                if _implausible(ev, energy, getattr(chunk, "energy", None)) == "crossings":
                    # (the surface folded through itself near a change, its mesh
                    # tangled there: a pack's drying contact left an edge twice
                    # the others' and the next run crossed edges two apart):
                    # evened out and run again
                    ev.restore(start)
                    _remesh_now(ev, h0)
                    chunk, start, energy = _away_from_poles(ev, lambda n: ev.relax(tol, max_iter=n),
                                                        _CHUNK)
                    if _implausible(ev, energy, getattr(chunk, "energy", None)) == "crossings":
                        ev.restore(start)
                        _unguard(ev)
                        forced = _change_nearest(ev, factor, keep, again, pre)
                        if forced is not None:
                            made_change(forced)
                            continue
                        guard(ev)
                        chunk, start, energy = _away_from_poles(
                            ev, lambda n: ev.relax(tol, max_iter=n), _CHUNK)
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
                        forced = _change_nearest(ev, factor, keep, again, pre)
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
            # (vertices a split added come last: the rows before are the same)
            moved = ev.vertices[:len(x)] - x if len(x) <= len(ev.vertices) else None
            if remeshing and not (chunk is not None and chunk.converged):
                _keep_edges(ev, h0)
                _checked(ev, "keeping the edges in range")
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
                        _log.debug("change refused: %s", refused)
            if neck is not None:
                _log.debug("change made: %s", neck)
                _checked(ev, f"pinch_off: {neck}")
                changes.append(neck)
                undo.clear()
                way, before, since = None, [], 0
                r = _settle(ev, tol)
                if r is None:
                    # (a 16-bead pack's pieces, other changes coming: down
                    # from the cut in the descent's short runs instead)
                    _log.debug("the pieces didn't relax after the cut: on down from it")
                    falling, remeshing = False, True
                    guard(ev)
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
                    refused = f"{e}: made {_AGAIN} times here already"
                    continue
                why = _try_change(ev, e, keep, pre)
                if not why:
                    made = e
                    break
                refused = f"{e}: {why}"             # (passed over: it may come again, or go)
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
                    continue
                if chunk.stable is not False and not _sound(chunk.health):
                    # (a facet squashed in the change, a patch shrunk to nothing,
                    # the surface lying on a solid: tidied, and once more)
                    drop_flat(ev, h0)
                    close_patches(ev, h0, shut)
                    _even_out(ev, smooth=False)
                    _keep_edges(ev, h0)
                    chunk = ev.relax(tol)
                    _checked(ev, "tidying a relaxed state")
                if chunk.converged and chunk.stable is not False and _sound(chunk.health):
                    return chunk, changes, ""
                if chunk.stable is not False:
                    # (on down, within the descent's budget: a pack's relax()
                    # ended at residual 7e-6, not quite in equilibrium, and the
                    # run stopped there)
                    _log.debug("relaxed, but %s: on down", chunk.health)
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
                forced = _change_nearest(ev, factor, keep, again, pre)
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
                raise
            saved, e = undo.pop()
            ev.restore(saved)
            changes.remove(e)
            made_at.extend([(e.kind, str(e.constraint), np.asarray(e.where, float))]*_AGAIN)
            refused = f"{e}: Evolver failed after it ({str(error).splitlines()[0]})"
            _log.debug("change undone: %s", refused)
            way, before, since = None, [], 0
    return None, changes, (f"no stable state within {_FALL_ITERATIONS} steps of the last "
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
        r, here, energy = _away_from_poles(ev, lambda n: ev.relax(tol, max_iter=n), 1000)
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
    the topology check, first after a relax() run). ``before``: the energy,
    the crossings and the volumes' gap where the run began
    (_away_from_poles). ``trace``: the energies of the run's gradient steps,
    if known: the energy then judged gone wrong only where it also jumped by
    a tenth of the area in one step (menisci merging beside a contact line
    gave up a quarter of a 4-bead pack's area in a relax() run, step by
    step, and every run after the merge was undone)."""
    return not _implausible(ev, before, trace)


def _implausible(ev, before: tuple, trace=None) -> str:
    """Why a run isn't plausible (see _plausible), "" if it is."""
    energy, crossed, gap = before
    area = _core.total_area()
    trace = np.asarray(trace if trace is not None else [], float)
    jumped = energy - ev.energy >= 0.25*area and (
        len(trace) < 2 or float(np.abs(np.diff(trace)).max()) >= 0.1*area)
    why = ("volumes" if ev._target_gap()[0] > max(1e-3, gap) else
           "poles" if _near_poles(ev, _POLE_MARGIN/2) else
           "energy" if jumped else
           "crossings" if _crossings(ev) > crossed else "")
    if why:
        _log.debug("implausible run: %s", why)
    return why


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


def _change_nearest(ev, factor: float, keep, skip=None, saved=None) -> Any:
    """Where the descent would pass the surface through itself (a run, a
    fall's steps or a step along the mode held back for it), the change
    nearest to coming made instead: of the events here, nearest first (gap
    over edges), the first apply() makes without crossings. Two pieces of a
    pack's liquid came together on a bead that way, their contact lines
    0.03 apart, and the descent, held back run after run, ran away. The
    event made, or None."""
    found = sorted(ev.events(factor), key=lambda e: e.gap/float(e.scale))
    for e in found:
        if skip is not None and skip(e):
            continue
        why = _try_change(ev, e, keep, saved)
        if why:
            _log.debug("change refused: %s: %s", e, why)
            continue
        _log.debug("change made (the nearest, the surface about to pass through itself): %s", e)
        return e
    return None


def _try_change(ev, e, keep, saved=None) -> str:
    """The change for event e made (apply), unless it passes the surface
    through itself or leaves a surface Evolver can't take a gradient step
    from (a pack's merge made by facet_merge left a constraint matrix
    singular, and the run stopped at the next step): then undone. Why not
    ("" if made)."""
    crossed, start, energy = _crossings(ev), ev._save(), ev.energy
    try:
        apply(ev, e, keep)
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
    """n gradient steps, or as many as don't pass the surface through itself:
    where n do, the mesh evened out (smoothed, flipped) and n again (the
    crossings in the corpus's falls were folds an edge or two across, beside
    a contact line), then one at a time (a touch then comes as an event).
    Whether they were held back."""
    crossed, start = _crossings(ev), ev._save()
    ev.command(f"g {n}")
    if _crossings(ev) <= crossed:
        return False
    ev.restore(start)
    _even_out(ev)
    crossed, start = _crossings(ev), ev._save()
    ev.command(f"g {n}")
    if _crossings(ev) <= crossed:
        return False
    ev.restore(start)
    for _ in range(n):
        here = ev._save()
        ev.command("g 1")
        if _crossings(ev) > crossed:
            ev.restore(here)
            _log.debug("gradient steps stopped: the surface through itself")
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


def _remesh_now(ev, h0: float) -> None:
    """The surface remeshed toward edges of h0 (_edits.remesh: splits,
    collapses, flips toward valence 6, smoothing along the surface, each
    checked), the guards off meanwhile (a guarded vertex is on one more
    constraint, which the edits keep)."""
    from ._edits import remesh
    guarded = _guarded(ev)
    _unguard(ev)
    try:
        done = remesh(ev, h0)
        _log.debug("remeshed: %s", done)
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


def _tidy(ev, h0: float, shut: list) -> None:
    """A descent's tidying: contact lines oriented, shrunk patches closed
    (close_patches), poles moved off the lines (move_poles); where that
    leaves a surface Evolver can't take a gradient step from (a pack's patch
    closed, then a pole moved, and the constraint matrix was singular at the
    next step), undone and done without the patches closed, else not done."""
    good = ev._save()
    for close in (True, False):
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


def _reopening(e, shut: list, h: float) -> bool:
    """Whether a solid event is a dry patch (the liquid between thinning)
    within 3 h of a patch closed on its solid."""
    return e.kind == "solid" and e.between == "body" and any(
        str(key) == str(e.constraint) and np.linalg.norm(np.asarray(e.where) - middle) < 3*h
        for key, middle in shut)


def _coming(found: list, before: list) -> list:
    from .recipes import _coming as coming
    return coming(found, before)


def _keep_edges(ev, h0: float) -> None:
    """Edges back in range after a change, by the checked edits (_edits):
    those shorter than 0.4 h0 collapsed, contact-line edges too (a joined
    contact line kept its short ones, and its facets degenerated as it
    moved), never a contact line's vertex inward (that pinched two contact
    lines together at a vertex: they meet by contact() instead), no edge
    made longer than 2 h0; those longer than 2 h0 split (not where the
    midpoint lands inside a solid: an edge into the surface from a contact
    line); then flipped where that makes the facets more equiangular."""
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
    from ._edits import edit
    lo, hi = _REMESH
    for _ in range(5):
        lengths = m.edge_lengths()
        short = np.flatnonzero(lengths < lo*h0)
        if not len(short):
            break
        order = short[np.argsort(lengths[short])]
        if not (edit(ev, "collapse", np.asarray(m.edge_ids)[order], max_len=hi, h=h0) == 0).any():
            break
        m = ev.mesh()
    for _ in range(5):
        lengths = m.edge_lengths()
        long = np.flatnonzero(lengths > hi*h0)
        if not len(long):
            break
        order = long[np.argsort(-lengths[long])]
        if not (edit(ev, "split", np.asarray(m.edge_ids)[order], h=h0) == 0).any():
            break
        m = ev.mesh()
    _even_out(ev, smooth=False)


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
