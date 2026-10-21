"""Coming changes of topology, found on the mesh in NumPy: where the surface
is about to pinch, to touch itself or a solid it isn't on, or where contact
lines on one solid are about to meet (see :meth:`Evolver.events`)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import List, Optional, Union

import numpy as np

from . import _core
from ._geometry import _chains

KINDS = ("neck", "touch", "solid", "contact lines")
NECK = 3.0      # a neck counts within this many times the gap two sheets do
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
    its search grows as the cube of it, 0.45 s of a 16-bead pack's 1 s at
    4)."""
    if not factor > 0:
        raise ValueError("events() needs factor > 0")
    m = ev.mesh()
    if m.facets is None or len(m.facets) == 0 or _core.symmetry():
        return []
    x = np.asarray(m.vertices, float)
    if x.shape[1] != 3:
        return []
    f = np.asarray(m.facets)
    e = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1), axis=0)
    n = len(x)
    length = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
    count = np.bincount(e.ravel(), minlength=n)
    used = count > 0
    # each vertex's edge length, from its edges between vertices free to move
    # where it has some (fixed facets often draw a pad or a table, coarsely)
    fixed = np.asarray(m.fixed, bool)
    free = ~fixed[e[:, 0]] & ~fixed[e[:, 1]]
    h = np.bincount(e.ravel(), np.r_[length, length], minlength=n)/np.maximum(count, 1)
    nfree = np.bincount(e[free].ravel(), minlength=n)
    hfree = np.bincount(e[free].ravel(), np.r_[length[free], length[free]], minlength=n)
    h = np.where(nfree > 0, hfree/np.maximum(nfree, 1), h)
    area = 0.5*np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    normal = np.zeros_like(x)
    for i in range(3):
        np.add.at(normal, f[:, i], area)
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
    # vertices in a fold of the mesh (facets round them facing opposite
    # ways): their normal means nothing, and they face the surface by mistake
    unit = area/np.maximum(np.linalg.norm(area, axis=1), 1e-300)[:, None]
    agree = np.ones(n)
    for i in range(3):
        np.minimum.at(agree, f[:, i], np.einsum("ij,ij->i", unit, normal[f[:, i]]))
    folded = agree < 0
    bodies = np.zeros((len(f), 2), dtype=np.int64) if m.facet_bodies is None \
        else np.asarray(m.facet_bodies)
    # the body on each side of a vertex: the facets' normals point out of
    # their front body, so it lies behind (-normal) and the back body in front
    behind, ahead = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for i in range(3):
        behind[f[:, i]] = np.maximum(behind[f[:, i]], bodies[:, 0])
        ahead[f[:, i]] = np.maximum(ahead[f[:, i]], bodies[:, 1])
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
    across: a neck is round, and a ring of fewer than about five edges no
    longer follows it (NECK*0.5 edges across: five round), where Evolver's
    tiny-edge deletion soon closes it; curved edges follow it further
    (neck_factor: NECK over the elements' order)."""
    r = factor*h
    reach = max(neck_factor, 1.0)
    if not moving.any():
        return []
    # candidates: vertices facing each other (normals apart by more than 120
    # degrees, which most close pairs, neighbours along the surface, aren't)
    # within reach plus an edge, then the facets round the far one
    vi, vj = _facing_pairs(x, normal, moving | ~folded,
                           reach*float(r[moving].max()) + float(h.max()))
    vi, ti = _round(vi, vj, f, len(x))
    keep = moving[vi] & ~(f[ti] == vi[:, None]).any(axis=1) & ~folded[f[ti]].any(axis=1)
    vi, ti = vi[keep], ti[keep]
    # the facet's side by its vertices' normals (an inverted facet's own
    # normal faces back into the surface); facing, then within reach of the
    # facet's centroid, then the distance itself
    tn = normal[f[ti]].sum(axis=1)
    tn /= np.maximum(np.linalg.norm(tn, axis=1), 1e-300)[:, None]
    keep = np.einsum("ij,ij->i", tn, normal[vi]) < -0.5
    vi, ti, tn = vi[keep], ti[keep], tn[keep]
    centre = x[f[ti]].mean(axis=1)
    radius = np.linalg.norm(x[f[ti]] - centre[:, None, :], axis=2).max(axis=1)
    keep = np.linalg.norm(centre - x[vi], axis=1) - radius < reach*r[vi]
    vi, ti, tn = vi[keep], ti[keep], tn[keep]
    if len(vi) == 0:
        return []
    c = _closest_on_triangles(x[vi], x[f[ti, 0]], x[f[ti, 1]], x[f[ti, 2]])
    d = c - x[vi]
    gap = np.linalg.norm(d, axis=1)
    along = np.einsum("ij,ij->i", d, normal[vi])
    across = np.where(along < 0, behind[vi], ahead[vi])
    neck = across > 0 if with_bodies else along < 0
    # facing each other across the gap, not neighbours along a smooth surface
    ok = (gap < np.where(neck, neck_factor, 1.0)*r[vi]) & (np.abs(along) > 0.5*gap)
    ok[ok] = ~_near_in_mesh(vi[ok], f[ti[ok]], e, len(x))
    vi, ti, gap, c, neck = vi[ok], ti[ok], gap[ok], c[ok], neck[ok]
    if len(vi) == 0:
        return []
    # each connected piece of the surface, to tell a local event from a mesh
    # too coarse for its piece (all of it within reach of itself)
    piece = np.zeros(len(x), dtype=np.int64)
    for label, members in _groups(np.bincount(e.ravel(), minlength=len(x)) > 0, e):
        piece[members] = label
    size = np.bincount(piece, minlength=len(x))
    events = []
    for kind, mask in (("neck", neck), ("touch", ~neck)):
        if not mask.any():
            continue
        a, t, g, cc = vi[mask], ti[mask], gap[mask], c[mask]
        links = np.r_[e, np.c_[a, f[t, 0]], np.c_[a, f[t, 1]], np.c_[a, f[t, 2]]]
        flagged = np.zeros(len(x), dtype=bool)
        flagged[a] = True
        flagged[f[t].ravel()] = True
        for _, members in _groups(flagged, links):
            mine = np.flatnonzero(np.isin(a, members))
            if len(mine) == 0 or 2*len(members) > size[piece[members]].max():
                continue
            k = mine[np.argmin(g[mine])]
            events.append(Event(kind, float(g[k]), float(h[a[k]]), (x[a[k]] + cc[k])/2,
                                np.asarray(members)))
    return events


def _near_in_mesh(vi, tri, e, n):
    """Whether each facet (its vertex rows) has a vertex within two edges of
    the vertex vi along the mesh."""
    nbrs = [[] for _ in range(n)]
    for a, b in e.tolist():
        nbrs[a].append(b)
        nbrs[b].append(a)
    out = np.zeros(len(vi), dtype=bool)
    for k, (v, t) in enumerate(zip(vi.tolist(), tri.tolist())):
        ring = set(nbrs[v])
        ring.add(v)
        two = set(ring)
        for w in ring:
            two.update(nbrs[w])
        out[k] = bool(two.intersection(t))
    return out


def _facing_pairs(x, normal, mask, r):
    """Ordered pairs (i, j), both masked, closer than r, their normals facing
    (dot below -0.5)."""
    idx = np.flatnonzero(mask)
    if len(idx) < 2 or not r > 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    p, nrm = x[idx], normal[idx]
    cell = np.floor((p - p.min(axis=0))/r).astype(np.int64) + 1
    dims = cell.max(axis=0) + 2
    k = (cell[:, 0]*dims[1] + cell[:, 1])*dims[2] + cell[:, 2]
    order = np.argsort(k, kind="stable")
    sk = k[order]
    out_i, out_j = [], []
    for o in product((-1, 0, 1), repeat=3):
        if o < (0, 0, 0):
            continue
        nk = sk + (o[0]*dims[1] + o[1])*dims[2] + o[2]
        lo = np.searchsorted(sk, nk, "left")
        cnt = np.searchsorted(sk, nk, "right") - lo
        a = np.repeat(order, cnt)
        within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        b = order[np.repeat(lo, cnt) + within]
        keep = a < b if o == (0, 0, 0) else a != b
        a, b = a[keep], b[keep]
        keep = np.einsum("ij,ij->i", nrm[a], nrm[b]) < -0.5
        a, b = a[keep], b[keep]
        keep = np.einsum("ij,ij->i", p[a] - p[b], p[a] - p[b]) < r*r
        out_i.append(a[keep])
        out_j.append(b[keep])
    i, j = idx[np.concatenate(out_i)], idx[np.concatenate(out_j)]
    return np.r_[i, j], np.r_[j, i]


def _round(vi, vj, f, n):
    """(vertex vi, facet) pairs for the facets round each vj, once each."""
    if len(vi) == 0:
        return vi, vi
    order = np.argsort(f.ravel(), kind="stable")
    start = np.searchsorted(f.ravel()[order], np.arange(n), "left")
    stop = np.searchsorted(f.ravel()[order], np.arange(n), "right")
    cnt = stop[vj] - start[vj]
    a = np.repeat(vi, cnt)
    within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    t = order[np.repeat(start[vj], cnt) + within]//3
    pairs = np.unique(np.c_[a, t], axis=0)
    return pairs[:, 0], pairs[:, 1]


def _closest_on_triangles(p, a, b, c):
    """The point of each triangle (a, b, c) closest to p (rowwise)."""
    ab, ac = b - a, c - a
    nrm = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", nrm, nrm)
    ap = p - a
    # barycentric coordinates of the projection onto the plane
    w2 = np.einsum("ij,ij->i", np.cross(ab, ap), nrm)/np.maximum(nn, 1e-300)
    w1 = np.einsum("ij,ij->i", np.cross(ap, ac), nrm)/np.maximum(nn, 1e-300)
    w0 = 1 - w1 - w2
    proj = a + w1[:, None]*ab + w2[:, None]*ac
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (nn > 0)
    best = proj.copy()
    if (~inside).any():
        q = p[~inside]
        cands = [_closest_on_segments(q, s, t) for s, t in
                 ((a[~inside], b[~inside]), (b[~inside], c[~inside]), (c[~inside], a[~inside]))]
        dist = np.stack([np.einsum("ij,ij->i", q - k, q - k) for k in cands])
        pick = np.argmin(dist, axis=0)
        best[~inside] = np.stack(cands)[pick, np.arange(len(q))]
    return best


def _to_segments(p, a, b, block: int = 256) -> np.ndarray:
    """Each point's distance to the nearest of the segments (a, b), in blocks
    of points (a loop over the points, a vertex at a time, was the most
    called function of a 16-bead pack's descent: 4.8 million calls)."""
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    out = np.empty(len(p))
    for i in range(0, len(p), block):
        q = p[i:i + block, None, :] - a[None, :, :]
        t = np.clip(np.einsum("ijk,jk->ij", q, ab)/ab2, 0.0, 1.0)
        d = q - t[:, :, None]*ab[None, :, :]
        out[i:i + block] = np.sqrt(np.einsum("ijk,ijk->ij", d, d).min(axis=1))
    return out


def _closest_on_segments(p, a, b):
    ab = b - a
    t = np.einsum("ij,ij->i", p - a, ab)/np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    return a + np.clip(t, 0, 1)[:, None]*ab


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
    walls_on = np.isnan(dist).sum(axis=1)
    for c in range(len(nums)):
        d = dist[:, c]
        on = np.isnan(d) & moving
        # (not vertices on two other constraints: where they meet, and no
        # contact can take them; a box's corner vertex on two walls lies on
        # the third to round-off, 1e-10 through it, and was a solid event
        # at every step of a pack's run)
        off = ~np.isnan(d) & moving & (walls_on < 2)
        if not off.any():
            continue
        # constraints only fixed vertices are on just place them (a pad's rim)
        if not attrs[c] & 3 and (np.isnan(d) & ~moving).any() and not on.any():
            continue
        if attrs[c] & 2:            # NONNEGATIVE: the allowed side is known
            side = 1.0
        elif attrs[c] & 1:          # NONPOSITIVE
            side = -1.0
        else:                       # the side most of the surface is on
            side = 1.0 if (d[off] > 0).sum() >= (d[off] < 0).sum() else -1.0
        g = np.where(off, side*d, np.inf)
        band = off & (g < factor*h)
        if not band.any():
            continue
        key: Union[int, str] = names[c] or int(nums[c])
        _, grad = ev._call(_core.constraint_normals, int(nums[c]))
        towards = -side*np.asarray(grad.data)          # from the surface to the solid
        between = np.full(len(x), None, dtype=object)
        if with_bodies:
            body = np.where(np.einsum("ij,ij->i", towards, normal) < 0, behind, ahead)
            between = np.where(body > 0, "body", "outside").astype(object)
        flag = np.zeros(len(x), dtype=bool)
        for _, members in _groups(band | on, e):
            members = np.asarray(members)
            if not on[members].any():
                flag[members] = True
        angle = _contact_angle(ev, surface.walls.get(key))
        near = np.flatnonzero(band & ~flag)
        lines = edges[on[edges[:, 0]] & on[edges[:, 1]] & (valence == 1)]
        if angle is not None and len(near) and len(lines):
            s_ = _to_segments(x[near], x[lines[:, 0]], x[lines[:, 1]])
            wedge = np.where(between[near] == "outside", 180.0 - angle, angle).astype(float)
            flag[near] = g[near] < WEDGE*s_*np.sin(np.radians(wedge))
        flag &= band
        for _, members in _groups(flag, e):
            members = np.asarray(members)
            k = members[np.argmin(g[members])]
            events.append(Event("solid", float(g[k]), float(h[k]),
                                x[k] + 0.5*float(g[k])*towards[k], members, key,
                                between[k]))
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
    for c in range(len(nums)):
        on = np.isnan(dist[:, c]) & used
        mask = on[edges[:, 0]] & on[edges[:, 1]] & (valence == 1)
        if mask.sum() < 3:
            continue
        # each line's vertices with their distance along it (from its start)
        rows, along, lines, totals, loops = [], [], [], [], []
        for k, path in enumerate(_chains(edges[mask])):
            p = np.asarray(path)
            seg = np.linalg.norm(np.diff(x[p], axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            loop = len(p) > 2 and p[0] == p[-1]
            if loop:
                p, cum = p[:-1], cum[:-1]
            rows.append(p)
            along.append(cum)
            lines.append(np.full(len(p), k))
            totals.append(np.full(len(p), float(seg.sum())))
            loops.append(np.full(len(p), loop))
        line_vertices, s = np.concatenate(rows), np.concatenate(along)
        line, total, closed = np.concatenate(lines), np.concatenate(totals), np.concatenate(loops)
        pts = x[line_vertices]
        i, j, d = _close_pairs(pts, factor*float(h[line_vertices].max()))
        scale = 0.5*(h[line_vertices[i]] + h[line_vertices[j]])
        keep = d < factor*scale
        i, j, d, scale = i[keep], j[keep], d[keep], scale[keep]
        apart = np.abs(s[i] - s[j])
        apart = np.where(closed[i], np.minimum(apart, total[i] - apart), apart)
        far = (line[i] != line[j]) | (apart > 3*np.maximum(d, scale))
        i, j, d, scale = i[far], j[far], d[far], scale[far]
        if len(i) == 0:
            continue
        flagged = np.zeros(len(x), dtype=bool)
        flagged[line_vertices[i]] = flagged[line_vertices[j]] = True
        links = np.r_[edges[mask], np.c_[line_vertices[i], line_vertices[j]]]
        for _, members in _groups(flagged, links):
            mine = np.flatnonzero(np.isin(line_vertices[i], members))
            k = mine[np.argmin(d[mine])]
            key: Union[int, str] = names[c] or int(nums[c])
            events.append(Event("contact lines", float(d[k]), float(scale[k]),
                                (pts[i[k]] + pts[j[k]])/2, np.asarray(members), key))
    return events


def _close_pairs(p, r):
    """Pairs (i < j) of points closer than r, with their distances."""
    if len(p) < 2 or not r > 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    cell = np.floor((p - p.min(axis=0))/r).astype(np.int64) + 1
    dims = cell.max(axis=0) + 2
    k = (cell[:, 0]*dims[1] + cell[:, 1])*dims[2] + cell[:, 2]
    order = np.argsort(k, kind="stable")
    sk = k[order]
    out_i, out_j = [], []
    for o in product((-1, 0, 1), repeat=3):
        if o < (0, 0, 0):
            continue
        nk = sk + (o[0]*dims[1] + o[1])*dims[2] + o[2]
        lo = np.searchsorted(sk, nk, "left")
        cnt = np.searchsorted(sk, nk, "right") - lo
        a = np.repeat(order, cnt)
        within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        b = order[np.repeat(lo, cnt) + within]
        keep = a < b if o == (0, 0, 0) else a != b
        out_i.append(a[keep])
        out_j.append(b[keep])
    i, j = np.concatenate(out_i), np.concatenate(out_j)
    i, j = np.minimum(i, j), np.maximum(i, j)
    # (each pair comes once: the offsets are one of each +- pair, a cell with
    # itself only i < j; so the close ones, then in order of (i, j), as a
    # unique over all the candidates gave them, a fifth of the event search)
    d = np.linalg.norm(p[i] - p[j], axis=1)
    close = d < r
    i, j, d = i[close], j[close], d[close]
    order = np.lexsort((j, i))
    return i[order], j[order], d[order]


# ---- grouping ------------------------------------------------------------------------

def _groups(mask: np.ndarray, links: np.ndarray):
    """The connected groups of masked vertices, joined by links (vertex
    pairs) between masked vertices: (label, member rows), the label the
    smallest member, in increasing order, the members increasing. (Array
    union-find: links hooked to the smaller label, then labels jumped to
    their labels' until they stay; a dict-based one was 3% of a 16-bead
    pack's descent, its root() called 60 million times.)"""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    links = np.asarray(links)
    links = links[mask[links[:, 0]] & mask[links[:, 1]]] if len(links) else links
    if not len(links):
        return [(int(v), [int(v)]) for v in idx]
    where = np.full(len(mask), -1, dtype=np.int64)
    where[idx] = np.arange(len(idx))
    a, b = where[links[:, 0]], where[links[:, 1]]
    label = np.arange(len(idx))
    while True:
        la, lb = label[a], label[b]
        low = np.minimum(la, lb)
        if np.array_equal(la, lb):
            break
        np.minimum.at(label, la, low)
        np.minimum.at(label, lb, low)
        while True:
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped
    order = np.argsort(label, kind="stable")
    cuts = np.flatnonzero(np.diff(label[order])) + 1
    return [(int(idx[g[0]]), idx[g].tolist()) for g in np.split(order, cuts)]
