"""Coming changes of topology, found on the mesh in NumPy, the loops over it
in C (bindings/pyse_events.c): where the surface is about to pinch, to
touch itself or a solid it isn't on, or where contact lines on one solid
are about to meet (see :meth:`Evolver.events`)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Union

import numpy as np

from . import _core

KINDS = ("neck", "touch", "solid", "contact lines")
NECK = 4.0      # a neck counts within this many times the gap two sheets do
SHEET = 0.25    # a neck whose waist normals' second moment is under this of the first: a sheet
WEDGE = 0.7     # next to a contact line: closer than this share of its wedge's height


@dataclass(frozen=True)
class Event:
    """A change of topology that is coming (see :meth:`Evolver.events`).

    ``kind``: ``"neck"`` (the surface close to itself across a body: a
    bridge or a column about to pinch off), ``"touch"`` (close to itself
    across the outside: drops or menisci about to merge), ``"solid"`` (close
    to a constraint it isn't on, away from its contact lines there: a
    meniscus reaching a bead, a neck reaching a wall) or ``"contact lines"``
    (two stretches of contact line on one constraint close together: wetted
    patches about to join or split). ``gap``: the distance across, the
    smallest found; for ``"solid"`` negative where vertices have gone
    through. ``scale``: the edge length there, which ``gap`` was compared
    with. ``where``: the middle of the gap. ``vertices``: the vertices (rows
    of :attr:`Evolver.vertices`) close to each other or to the solid.
    ``constraint``: the solid's number or name (``"solid"``, ``"contact
    lines"``). ``between``: for ``"solid"``, what lies between the surface
    and the solid, ``"body"`` (a film of liquid thinning: the solid is about
    to dry there) or ``"outside"`` (the liquid is about to touch it); None
    without bodies.
    """

    kind: str
    gap: float
    scale: float
    where: np.ndarray
    vertices: np.ndarray = field(repr=False)
    constraint: Optional[Union[int, str]] = None
    between: Optional[str] = None

    def __str__(self) -> str:
        at = "(" + ", ".join(f"{c:.4g}" for c in self.where) + ")"
        if self.kind == "solid":
            what = (f"the surface has gone through constraint {self.constraint}"
                    if self.gap < 0 else
                    f"the surface comes within {self.gap:.3g} of constraint {self.constraint}")
            film = {"body": ", the liquid between thinning",
                    "outside": ", about to wet it"}.get(self.between or "", "")
            return f"{what} at {at}{film} (edges {self.scale:.3g})"
        what = {"neck": "a neck", "touch": "the surface close to itself across the outside",
                "contact lines": f"contact lines on constraint {self.constraint}"}[self.kind]
        return f"{what} {self.gap:.3g} across at {at} (edges {self.scale:.3g})"


def find(ev, factor: float, self_factor: Optional[float] = None) -> List[Event]:
    """The events on ``ev``'s surface, closest (relative to the edges) first;
    the surface close to itself within ``self_factor`` (default factor:
    its search grows as the cube of it, 34 ms of a 16-bead pack's 50 ms at
    4, one thread)."""
    if not factor > 0:
        raise ValueError("events() needs factor > 0")
    m = ev.mesh()
    if m.facets is None or len(m.facets) == 0 or _core.symmetry():
        return []
    x = np.ascontiguousarray(m.vertices, dtype=float)
    if x.shape[1] != 3:
        return []
    f = np.ascontiguousarray(m.facets, dtype=np.int64)
    # the facets' edges (as a unique over their rows sorts them), each
    # vertex's normal, and the vertices in a fold of the mesh (facets round
    # them facing opposite ways: their normal means nothing, and they face
    # the surface by mistake); in C, pyse_events.c: the unique alone was
    # 18 ms of a 15k-facet surface's 80 ms search
    e, normal, folded = _core.events_frame(x, f)
    n = len(x)
    length = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
    count = np.bincount(e.ravel(), minlength=n)
    used = count > 0
    # each vertex's edge length, from its edges between vertices free to move
    # where it has some (fixed facets often draw a pad or a table, coarsely)
    fixed = np.asarray(m.fixed, bool)
    free = ~fixed[e[:, 0]] & ~fixed[e[:, 1]]
    # (the weights each edge's length twice in a row, as ravel() lists its
    # two ends: listed all once and then again, 7-17% of a pack's vertices
    # had a scale off by more than a fifth, other edges' lengths)
    h = np.bincount(e.ravel(), np.repeat(length, 2), minlength=n)/np.maximum(count, 1)
    nfree = np.bincount(e[free].ravel(), minlength=n)
    hfree = np.bincount(e[free].ravel(), np.repeat(length[free], 2), minlength=n)
    h = np.where(nfree > 0, hfree/np.maximum(nfree, 1), h)
    bodies = np.zeros((len(f), 2), dtype=np.int64) if m.facet_bodies is None \
        else np.asarray(m.facet_bodies)
    # the body on each side of a vertex: the facets' normals point out of
    # their front body, so it lies behind (-normal) and the back body in front
    behind, ahead = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    # (unbuffered: an assignment through f[:, i] kept the last facet's body
    # where a vertex has several facets in that column)
    for i in range(3):
        np.maximum.at(behind, f[:, i], bodies[:, 0])
        np.maximum.at(ahead, f[:, i], bodies[:, 1])
    with_bodies = bool(bodies.any())
    moving = used & ~np.asarray(m.fixed, bool)
    events = _self(x, f, e, h, normal, behind, ahead, with_bodies, moving & ~folded,
                   folded, factor if self_factor is None else self_factor,
                   NECK/max(int(m.order), 1))
    nums, attrs, dist, names, copies = _one_per_solid(*ev._constraint_gaps())
    found = _solid(ev, m, x, e, h, normal, behind, ahead, with_bodies, moving, factor,
                   nums, attrs, dist, names)
    found += _contact_lines(m, x, e, h, used, factor, nums, dist, names)
    events += [_on_copy(event, copies) for event in found]
    return sorted(events, key=lambda ev_: ev_.gap/ev_.scale)


def _base(key) -> str:
    """The solid a wall is a copy of (_surgery.move_poles' copies of a
    sphere differ only in their poles: <base>_pole<n>)."""
    return str(key).split("_pole")[0] if isinstance(key, str) else f"c{key}"


def _one_per_solid(nums, attrs, dist, names):
    """The constraints' table with the copies of one solid as one: a vertex
    on any copy on it, the others' gaps the same (one sphere). And the
    copies of each kept column's solid, by its key."""
    keys = [n or int(k) for k, n in zip(nums, names)]
    groups: dict = {}
    for c, key in enumerate(keys):
        groups.setdefault(_base(key), []).append(c)
    if all(len(g) == 1 for g in groups.values()):
        return nums, attrs, dist, names, {}
    first = [g[0] for g in groups.values()]
    dist = np.asarray(dist, float)
    merged = dist[:, first].copy()
    for j, g in enumerate(groups.values()):
        if len(g) > 1:
            merged[np.isnan(dist[:, g]).any(axis=1), j] = np.nan
    copies = {keys[g[0]]: [keys[c] for c in g] for g in groups.values() if len(g) > 1}
    return (np.asarray(nums)[first], np.asarray(attrs)[first], merged,
            [names[c] for c in first], copies)


def _on_copy(event: Event, copies: dict) -> Event:
    """An event on a solid with copies named by the copy whose pole is
    farthest from it (a contact made there leaves its new line clear of
    the integrals' singular pole)."""
    from ._engine import surface
    keys = copies.get(event.constraint)
    if not keys:
        return event
    def clearance(key):
        wall = surface.walls.get(key)
        shape, pole = getattr(wall, "shape", None), getattr(wall, "pole_axis", None)
        if shape is None or pole is None or shape[0] != "sphere":
            return -1.0
        u = np.asarray(event.where, float) - np.asarray(shape[1], float)
        u = u/max(float(np.linalg.norm(u)), 1e-300)
        return float(np.linalg.norm(u - np.asarray(pole, float)))
    return replace(event, constraint=max(keys, key=clearance))


# ---- the surface close to itself ---------------------------------------------------

def _self(x, f, e, h, normal, behind, ahead, with_bodies, moving, folded,
          factor, neck_factor) -> List[Event]:
    """Vertices close to a facet across the surface (facing it, the gap along
    the normal; not within two edges of it, where a fold of the mesh can
    face back): a neck when a body lies across, a touch when the outside
    does. Two sheets count within factor*h, a neck within NECK*factor*h
    across: a neck is round, and a ring of fewer than about six edges no
    longer follows it (NECK*0.5 edges across: six round), where Evolver's
    tiny-edge deletion soon closes it; curved edges follow it further
    (neck_factor: NECK over the elements' order). (NECK was 3 while h, each
    vertex's edge length, was other edges' mean: at a thinning waist 1.1-1.35
    times the true one; with the true h, 3 let a bridge pinch between two
    looks, 0.43 across at the last, under 0.37 to count.) Not a sheet's
    (sheet(): liquid between two faces that are flat across it, not round
    a waist): its two faces are two sheets, and count as those do (at four
    times that a 16-bead pack's sheets against a side wall ruptured 1-1.8
    edges thick; the liquid came back over the cut, the lines joined again
    and the sheet ruptured again, four times in a descent that then ran
    away)."""
    r = factor*h
    reach = max(neck_factor, 1.0)
    if not moving.any():
        return []
    # the (vertex, facet) pairs across, in C (pyse_events.c; in NumPy, with
    # Python neighbour lists for all the vertices, 0.7-3.5 s of a 16-bead
    # pack's search at 4 edges): vertices facing each other (normals apart
    # by more than 120 degrees, which most close pairs, neighbours along the
    # surface, aren't) within reach plus an edge, the facets round the far
    # one; the facet facing the vertex by its corners' normals (an inverted
    # facet's own normal faces back into the surface), within reach of its
    # centroid, then the distance itself, facing each other across the gap,
    # not neighbours along a smooth surface
    vi, ti, gap, c, neck = _core.events_self(
        x, f, e, normal, r, moving, moving | ~folded, folded, behind, ahead, with_bodies,
        reach*float(r[moving].max()) + float(h.max()), reach, neck_factor)
    if len(vi) == 0:
        return []
    # each connected piece of the surface, to tell a local event from a mesh
    # too coarse for its piece (all of it within reach of itself)
    piece = np.maximum(_labels(np.bincount(e.ravel(), minlength=len(x)) > 0, e), 0)
    size = np.bincount(piece, minlength=len(x))
    events = []
    for kind, mask in (("neck", neck), ("touch", ~neck)):
        if not mask.any():
            continue
        a, t, g, cc = vi[mask], ti[mask], gap[mask], c[mask]
        flagged = np.zeros(len(x), dtype=bool)
        flagged[a] = True
        flagged[f[t].ravel()] = True
        label = _labels(flagged, np.concatenate([e, np.stack([np.repeat(a, 3), f[t].ravel()], 1)]))
        nearest = _first_least(label[a], g)
        for group, members in _split(label):
            k = nearest.get(group)
            if k is None or 2*len(members) > size[piece[members]].max():
                continue
            if kind == "neck" and g[k] >= r[a[k]] and sheet(normal[members]):
                continue
            events.append(Event(kind, float(g[k]), float(h[a[k]]), (x[a[k]] + cc[k])/2,
                                np.asarray(members)))
    return events


def sheet(normals: np.ndarray) -> bool:
    """Whether a neck's waist (its vertices' unit normals) is a sheet's, not
    a tube's: the normals all along one direction (round a tube they turn
    about its axis), the second of their moments under SHEET of the first."""
    if len(normals) < 3:
        return False
    lam = np.linalg.eigvalsh(normals.T @ normals)
    return bool(lam[2] > 0 and lam[1] < SHEET*lam[2])


# ---- the surface close to a solid -----------------------------------------------

def _solid(ev, m, x, e, h, normal, behind, ahead, with_bodies, moving, factor,
           nums, attrs, dist, names) -> List[Event]:
    """Per constraint, the surface within factor*h of it (or on its far
    side), except where that is a contact line's own wedge: next to a
    contact line on it the surface comes close anyway, at a distance s from
    the line about s*sin(angle) off the solid, the angle through what lies
    between (the contact angle through the liquid, 180 minus it through the
    outside), s measured to the contact line's edges. Kept: stretches that
    don't reach a contact line on it, and vertices less than WEDGE times that
    far off (the surface coming down onto the solid next to a contact line;
    a barrel drop on a thin fibre, its surface curving round the fibre, stays
    above 0.9 of it). With no known contact angle (not a wall from
    :mod:`constraints`), only the first."""
    from ._engine import surface
    events: List[Event] = []
    edges, valence = np.asarray(m.edges), m.edge_valence()
    # all the constraints' columns at once, then a loop over those with
    # vertices close (a loop over all of them, each through the whole
    # column, was a third of a 15k-facet pack's search at 0.5 edges)
    nan = np.isnan(dist)
    walls_on = nan.sum(axis=1)
    on_all = nan & moving[:, None]
    # (not vertices on two other constraints: where they meet, and no
    # contact can take them; a box's corner vertex on two walls lies on
    # the third to round-off, 1e-10 through it, and was a solid event
    # at every step of a pack's run)
    off_all = ~nan & (moving & (walls_on < 2))[:, None]
    attrs = np.asarray(attrs, dtype=np.int64).reshape(-1)
    # the side allowed: NONNEGATIVE, NONPOSITIVE, else the side most of the
    # surface is on
    most = np.where((off_all & (dist > 0)).sum(axis=0) >= (off_all & (dist < 0)).sum(axis=0),
                    1.0, -1.0)
    sides = np.where(attrs & 2, 1.0, np.where(attrs & 1, -1.0, most))
    g_all = np.where(off_all, sides*dist, np.inf)
    band_all = off_all & (g_all < (factor*h)[:, None])
    # constraints only fixed vertices are on just place them (a pad's rim)
    placing = ((attrs & 3) == 0) & (nan & ~moving[:, None]).any(axis=0) & ~on_all.any(axis=0)
    for c in np.flatnonzero(band_all.any(axis=0) & ~placing):
        on, g, band, side = on_all[:, c], g_all[:, c], band_all[:, c], float(sides[c])
        key: Union[int, str] = names[c] or int(nums[c])
        _, grad = ev._call(_core.constraint_normals, int(nums[c]))
        towards = -side*np.asarray(grad.data)          # from the surface to the solid
        # what lies between (None without bodies): the body behind or ahead
        outside = None
        if with_bodies:
            outside = ~(np.where(np.einsum("ij,ij->i", towards, normal) < 0, behind, ahead) > 0)
        # the stretches (of the band and the vertices on it) that reach no
        # contact line on it
        label = _labels(band | on, e)
        wet = np.zeros(len(x), dtype=bool)
        wet[label[on]] = True
        flag = (label >= 0) & ~wet[np.maximum(label, 0)]
        angle = _contact_angle(ev, surface.walls.get(key))
        near = np.flatnonzero(band & ~flag)
        lines = edges[on[edges[:, 0]] & on[edges[:, 1]] & (valence == 1)]
        if angle is not None and len(near) and len(lines):
            # (each vertex's distance to the nearest of the contact line's
            # edges, in C: in NumPy, in blocks of vertices, it was 4 ms of a
            # 15k-facet pack's search; a vertex at a time before that, the
            # most called function of its descent, 4.8 million calls)
            s_ = _core.events_segments(x[near], x[lines[:, 0]], x[lines[:, 1]])
            wedge = np.full(len(near), float(angle)) if outside is None else \
                np.where(outside[near], 180.0 - angle, float(angle))
            flag[near] = g[near] < WEDGE*s_*np.sin(np.radians(wedge))
        flag &= band
        for _, members in _groups(flag, e):
            k = members[np.argmin(g[members])]
            between = None if outside is None else ("outside" if outside[k] else "body")
            events.append(Event("solid", float(g[k]), float(h[k]),
                                x[k] + 0.5*float(g[k])*towards[k], members, key, between))
    return events


def _contact_angle(ev, wall) -> Optional[float]:
    """A wall's contact angle in degrees (90 without wetting energy), or None."""
    if wall is None:
        return None
    angle = getattr(wall, "contact_angle", None)
    if angle is None:
        return 90.0
    try:
        return float(ev.eval(angle)) if isinstance(angle, str) else float(angle)
    except Exception:
        return None


# ---- contact lines close to each other ---------------------------------------------

def _contact_lines(m, x, e, h, used, factor, nums, dist, names) -> List[Event]:
    """Per constraint, contact-line vertices within factor*h of each other
    that are far apart along the line (or on different lines)."""
    events: List[Event] = []
    valence = m.edge_valence()
    edges = np.asarray(m.edges)
    # each constraint's contact-line edges (all the columns at once)
    on = np.isnan(dist) & used[:, None]
    line_edges = on[edges[:, 0]] & on[edges[:, 1]] & (valence == 1)[:, None]
    for c in np.flatnonzero(line_edges.sum(axis=0) >= 3):
        mask = line_edges[:, c]
        # the lines (the edges joined into paths as _geometry._chains does),
        # each vertex's distance along its line from its start, and the
        # pairs of their vertices within factor*h of each other (pairs i <
        # j, rows of line_vertices, d apart, scale their mean h), far apart
        # along the line (more than three times that or d; a loop the
        # shorter way round) or on two lines: in C (pyse_events.c; in
        # Python and NumPy, two fifths of a 16-bead pack's search at 0.5)
        line_vertices, i, j, d, scale = _core.events_lines(
            x, h, np.ascontiguousarray(edges[mask], dtype=np.int64), float(factor))
        if len(i) == 0:
            continue
        pts = x[line_vertices]
        flagged = np.zeros(len(x), dtype=bool)
        flagged[line_vertices[i]] = flagged[line_vertices[j]] = True
        links = np.r_[edges[mask], np.c_[line_vertices[i], line_vertices[j]]]
        label = _labels(flagged, links)
        nearest = _first_least(label[line_vertices[i]], d)
        for group, members in _split(label):
            k = nearest[group]
            key: Union[int, str] = names[c] or int(nums[c])
            events.append(Event("contact lines", float(d[k]), float(scale[k]),
                                (pts[i[k]] + pts[j[k]])/2, np.asarray(members), key))
    return events


# ---- grouping ------------------------------------------------------------------------

def _labels(mask: np.ndarray, links: np.ndarray) -> np.ndarray:
    """Per vertex the smallest member of its connected group of masked
    vertices, joined by links (vertex pairs) between masked ones; -1 where
    not masked. (Union-find in C, pyse_events.c: in NumPy, labels hooked and
    jumped until they stayed, it was a tenth of a 16-bead pack's search.)"""
    return _core.events_groups(np.ascontiguousarray(mask, dtype=bool),
                               np.ascontiguousarray(links, dtype=np.int64).reshape(-1, 2))


def _split(label: np.ndarray):
    """The groups of a labelling (see _labels): (label, member rows), the
    label the smallest member, in increasing order, the members increasing."""
    idx = np.flatnonzero(label >= 0)
    if len(idx) == 0:
        return []
    order = np.argsort(label[idx], kind="stable")
    cuts = np.flatnonzero(np.diff(label[idx][order])) + 1
    return [(int(idx[g[0]]), idx[g]) for g in np.split(order, cuts)]


def _groups(mask: np.ndarray, links: np.ndarray):
    """The connected groups of masked vertices joined by links (see
    _labels, _split)."""
    return _split(_labels(mask, links))


def _first_least(label: np.ndarray, value: np.ndarray) -> dict:
    """Per label, the row with the smallest value, the first of equal ones
    (as argmin over each group's rows; a search of the rows per group, by
    isin, was 10 ms of a 15k-facet surface's at 4 edges)."""
    if len(label) == 0:
        return {}
    order = np.lexsort((value, label))
    head = order[np.r_[True, label[order][1:] != label[order][:-1]]]
    return dict(zip(label[head].tolist(), head.tolist()))
