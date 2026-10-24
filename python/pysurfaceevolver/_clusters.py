"""Which liquid is one: clusters of a body's surface pieces joined through
the liquid between them, and the bodies made to match them in a topology
run (see recipes.arclength's ``outlet``).

A body's surface comes in pieces (edge-connected facets with the body on
one side, as Evolver's ``rebody`` finds them); each piece bounds one region
of liquid, but one region can have several pieces, joined only through the
liquid between walls (a throat between near-touching beads has a meniscus at
either end). Those joins are found on a grid of the space the solids leave:
nodes inside a solid dropped, neighbouring nodes linked unless the segment
between them crosses a facet or a solid, and each linked group of nodes
joined to the pieces whose liquid side faces it. A layer of liquid thinner
than the grid is not seen, so pieces joined only through one stay apart.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import numpy as np

from . import _periodic
from ._engine import GUARD_PREFIX, surface

_log = logging.getLogger("pysurfaceevolver.topology")

_SPACING = 0.5      # the grid's spacing, in the descents' edge scale
_NODES = 4_000_000  # the grid's nodes at most (its spacing grown to fit)
_NEAR = 1.5         # a node within this many spacings of a wall borders it
_OFFSET = 1/np.pi   # the grid's origin off the box by this share of a spacing
#                    (no node or line exactly on a wall or a facet's edge)


class Clusters(NamedTuple):
    """clusters()'s result."""

    side: np.ndarray        # (facets, 2): each facet side's cluster, -1 where no body
    bodies: List[frozenset]  # each cluster's bodies (one, unless liquid joins two)
    outlet: np.ndarray      # each cluster: whether it reaches the outlet
    nodes: np.ndarray       # each cluster: its liquid nodes on the grid (its size)
    walls: Dict[object, np.ndarray]  # a wall's key: per cluster, its nodes bordering it
    spacing: float          # the grid's (0 where there was none: pieces only)
    mixed: int              # groups of nodes facing both liquid and gas


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The connected components of n nodes linked in pairs (a[i], b[i]):
    each node's label, the smallest node of its component."""
    parent = np.arange(n)
    a, b = np.asarray(a, np.intp), np.asarray(b, np.intp)
    while len(a):
        pa, pb = parent[a], parent[b]
        differ = pa != pb
        if not differ.any():
            break
        a, b, pa, pb = a[differ], b[differ], pa[differ], pb[differ]
        np.minimum.at(parent, np.maximum(pa, pb), np.minimum(pa, pb))
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    return parent


def _edges(f: np.ndarray) -> np.ndarray:
    """Each facet row's three edges, as indices of the distinct vertex pairs."""
    sides = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    _, edge = np.unique(sides, axis=0, return_inverse=True)
    return edge.ravel().reshape(3, -1).T


def pieces(f: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """(facets, 2): each facet side's piece (edge-connected facets with its
    body on one side), numbered from 0; -1 where no body."""
    out = np.full(fb.shape, -1, np.intp)
    rows, sides = np.nonzero(fb > 0)
    if not len(rows):
        return out
    body = fb[rows, sides]
    edge = _edges(f)
    # entries (a facet side), linked where two share an edge and a body
    key = np.r_[edge[rows, 0], edge[rows, 1], edge[rows, 2]].astype(np.int64)
    key += (edge.max() + 1)*np.tile(body.astype(np.int64), 3)
    entry = np.tile(np.arange(len(rows)), 3)
    order = np.argsort(key, kind="stable")
    key, entry = key[order], entry[order]
    same = key[1:] == key[:-1]
    label = components(len(rows), entry[:-1][same], entry[1:][same])
    _, label = np.unique(label, return_inverse=True)
    out[rows, sides] = label
    return out


def _solids(ev) -> "tuple[list, list]":
    """The walls' solids as (key, kind, data) (kind "plane": normal, offset,
    the solid where normal . x < offset; "sphere": centre, radius;
    "cylinder": point, unit direction, radius; one per solid, not its pole
    copies), and the constraints the surface touches that aren't such walls."""
    from ._surgery import _base
    solids, seen = [], set()
    for key, w in surface.walls.items():
        shape = getattr(w, "shape", None)
        if shape is None:
            continue
        data: tuple
        try:
            if shape[0] == "plane":
                data = (np.asarray(shape[1], float), float(shape[2]))
            elif shape[0] == "sphere":
                data = (np.asarray(shape[1], float), float(shape[2]))
            elif shape[0] == "cylinder":
                d = np.asarray(shape[2], float)
                data = (np.asarray(shape[1], float), d/np.linalg.norm(d), float(shape[3]))
            else:
                continue
        except (TypeError, ValueError):
            continue            # (a shape following a parameter)
        base = _base(key)
        if base not in seen:
            seen.add(base)
            solids.append((key, shape[0], data))
    known = {_base(k) for k, _, _ in solids}
    nums, _, dist, names = ev._constraint_gaps()
    dist = np.asarray(dist)
    other: list = []
    on_wall = np.zeros(ev.counts["vertices"], bool)
    if dist.ndim == 2 and dist.shape[1]:
        on = np.isnan(dist)
        for i in np.flatnonzero(on.any(axis=0)):
            name = names[i] or int(nums[i])
            if str(names[i]).startswith(GUARD_PREFIX):
                continue
            if _base(name) in known:
                on_wall |= on[:, i]
            else:
                other.append(name)
    # (a border edge off the walls: the liquid closed by something else, a
    # boundary wire's content integral)
    m = ev.mesh()
    border = np.asarray(m.edges)[m.edge_valence() == 1]
    if len(border) and not on_wall[border].all():
        other.append("a border off the walls")
    return solids, other


def _gap(kind: str, data, p: np.ndarray, periods=None) -> np.ndarray:
    """Signed distance of points p from a solid's surface (negative inside;
    in a periodic slab from its nearest image)."""
    if kind == "plane":
        return p @ data[0] - data[1]
    q = p - data[0] if periods is None else _periodic.delta(data[0], p, periods)
    if kind == "cylinder":
        q = q - np.outer(q @ data[1], data[1])
    return np.sqrt(np.einsum("ij,ij->i", q, q)) - data[-1]


def _near(kind: str, data, lo: np.ndarray, s, shape, reach: float,
          periodic=(False, False, False)) -> np.ndarray:
    """The nodes (flat indices) that can be within ``reach`` of a solid's
    surface or inside it: a sphere's box (round the grid along a periodic
    axis: its images), else all."""
    if kind != "sphere":
        return np.arange(int(np.prod(shape)))
    c, R = data
    cyc = np.asarray(periodic, bool)
    a = np.floor((c - R - reach - lo)/s).astype(int)
    b = np.ceil((c + R + reach - lo)/s).astype(int)
    a = np.where(cyc, a, np.maximum(a, 0))
    b = np.where(cyc, b, np.minimum(b, np.array(shape) - 1))
    if (b < a).any():
        return np.zeros(0, np.intp)
    axes = [np.unique(np.arange(a[k], b[k] + 1) % shape[k]) if cyc[k]
            else np.arange(a[k], b[k] + 1) for k in range(3)]
    i, j, k = np.meshgrid(*axes, indexing="ij")
    return np.ravel_multi_index((i.ravel(), j.ravel(), k.ravel()), shape)


def _box(x: np.ndarray, solids: list, s: float) -> "tuple[np.ndarray, np.ndarray]":
    """The grid's box: the vertices', reaching the planes beyond it along
    an axis (a floor no vertex is on yet) and down to z = 0 (where
    Evolver's volume, z dx dy, closes a body), a spacing or two over."""
    lo, hi = x.min(axis=0), x.max(axis=0)
    lo[2] = min(lo[2], 0.0)
    for _, kind, data in solids:
        if kind != "plane":
            continue
        n, d = data
        k = int(np.argmax(np.abs(n)))
        if abs(n[k]) < 1 - 1e-12:
            continue
        at = d/n[k]
        if n[k] > 0:
            lo[k] = min(lo[k], at)
        else:
            hi[k] = max(hi[k], at)
    return lo - (2 - _OFFSET)*s, hi + 2*s


def _grid(x: np.ndarray, solids: list, s: float, periods
          ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, tuple]":
    """The grid: its box, spacing along each axis and nodes along each
    (see _box). In a periodic slab x and y cover one cell, a whole number of
    nodes per period (the node after the last the first's image)."""
    lo, hi = _box(x, solids, s)
    if periods is None:
        shape = tuple(int(n) for n in np.floor((hi - lo)/s).astype(int) + 1)
        return lo, hi, np.full(3, s), shape
    L = np.asarray(periods, float)
    n = np.maximum(np.round(L/s), 3).astype(int)
    sp = np.r_[L/n, s]
    lo[:2] = -_OFFSET*sp[:2]
    hi[:2] = lo[:2] + L - sp[:2]
    return lo, hi, sp, (int(n[0]), int(n[1]), int(np.floor((hi[2] - lo[2])/s)) + 1)


def _crossings(tri: np.ndarray, shape, periodic=(False, False, False)
               ) -> "tuple[np.ndarray, ...]":
    """Where the facets cross the grid's segments, tri the facets' corners
    in grid units (nodes at the integers; (k, 3, 3)): per crossing the axis,
    the segment's lower node (flat index), how far along it, the facet row
    and the sign of the facet's normal along the axis. Along a periodic
    axis the grid is one period, its last node's segment going to its
    first (a facet may reach past the last node)."""
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out: list = [[], [], [], [], []]
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        use = np.flatnonzero(np.abs(normal[:, k]) > 1e-12)
        p = tri[use][:, :, [i, j]]
        cyc = np.array([periodic[i], periodic[j]])
        lo = np.ceil(p.min(axis=1)).astype(np.intp)
        hi = np.floor(p.max(axis=1)).astype(np.intp)
        lo = np.where(cyc, lo, np.maximum(lo, 0))
        hi = np.where(cyc, hi, np.minimum(hi, np.array([shape[i] - 1, shape[j] - 1])))
        ni, nj = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0), np.maximum(hi[:, 1] - lo[:, 1] + 1, 0)
        count = ni*nj
        rep = np.repeat(np.arange(len(use)), count)
        if not len(rep):
            continue
        local = np.arange(len(rep)) - np.repeat(np.cumsum(count) - count, count)
        gi = lo[rep, 0] + local//nj[rep]
        gj = lo[rep, 1] + local % nj[rep]
        a, b, c = p[rep, 0], p[rep, 1], p[rep, 2]
        q = np.c_[gi, gj] - a
        ab, ac = b - a, c - a
        det = ab[:, 0]*ac[:, 1] - ab[:, 1]*ac[:, 0]
        u = (q[:, 0]*ac[:, 1] - q[:, 1]*ac[:, 0])/det
        w = (ab[:, 0]*q[:, 1] - ab[:, 1]*q[:, 0])/det
        inside = (u >= 0) & (w >= 0) & (u + w <= 1)
        rep, gi, gj, u, w = rep[inside], gi[inside], gj[inside], u[inside], w[inside]
        tk = tri[use[rep], :, k]
        z = tk[:, 0] + u*(tk[:, 1] - tk[:, 0]) + w*(tk[:, 2] - tk[:, 0])
        seg = np.floor(z).astype(np.intp)
        frac = z - seg
        if periodic[i]:
            gi = gi % shape[i]
        if periodic[j]:
            gj = gj % shape[j]
        if periodic[k]:
            seg = seg % shape[k]
            ok = np.ones(len(seg), bool)
        else:
            ok = (seg >= 0) & (seg <= shape[k] - 2)
        idx = np.zeros((int(ok.sum()), 3), np.intp)
        idx[:, i], idx[:, j], idx[:, k] = gi[ok], gj[ok], seg[ok]
        out[0].append(np.full(len(idx), k))
        out[1].append(np.ravel_multi_index(tuple(idx.T), shape))
        out[2].append(frac[ok])
        out[3].append(use[rep[ok]])
        out[4].append(np.sign(normal[use[rep[ok]], k]).astype(np.intp))
    if not out[0]:
        return tuple(np.zeros(0, t) for t in (np.intp, np.intp, float, np.intp, np.intp))
    return tuple(np.concatenate(o) for o in out)


def clusters(ev, outlet: Optional[Sequence] = None, spacing: Optional[float] = None
             ) -> Clusters:
    """The surface's liquid clusters (see the module notes): the facet
    sides' pieces joined where the liquid between them is one on the grid
    (``spacing``, default half the descents' edge scale, or the median
    edge's before there is one), and whether each
    reaches the walls named in ``outlet``: a node of its liquid within 1.5
    spacings of one, or a vertex on one. Where the surface touches a
    constraint that isn't a plane, sphere or cylinder wall, or has a border
    off the walls (a boundary wire's), pieces only, with a warning."""
    from ._surgery import _median_edge
    m = ev.mesh()
    none = Clusters(np.zeros((0, 2), np.intp), [], np.zeros(0, bool), np.zeros(0, np.intp), {},
                    0.0, 0)
    if m.facets is None or not len(m.facets) or m.facet_bodies is None:
        return none
    x, f = np.asarray(m.vertices, float), np.asarray(m.facets)
    fb = np.asarray(m.facet_bodies)
    piece = pieces(f, fb)
    npieces = int(piece.max()) + 1
    if not npieces:
        return none._replace(side=piece)
    body_of = np.zeros(npieces, np.intp)
    body_of[piece[piece >= 0]] = fb[piece >= 0]
    outlet = [outlet] if isinstance(outlet, (str, int)) else list(outlet or [])
    solids, other = _solids(ev)
    # each piece reaching the outlet by a vertex on it
    by_vertex = np.zeros(npieces, bool)
    for key in outlet:
        on = np.asarray(ev.on_constraint(key))
        hit = on[f].any(axis=1)
        for j in (0, 1):
            by_vertex[piece[hit & (piece[:, j] >= 0), j]] = True
    if other:
        warnings.warn(f"the surface touches constraints that aren't plane, sphere or "
                      f"cylinder walls, or has a border off them "
                      f"({', '.join(map(str, other[:5]))}): its liquid's clusters are its "
                      "surface pieces", UserWarning, stacklevel=3)
        alone = np.zeros(0, np.intp)
        return _gather(piece, body_of, components(npieces, alone, alone), by_vertex,
                       np.zeros(npieces, np.intp), {}, 0.0, 0)
    # (the descents' edge scale, not set here where it isn't yet: that is
    # the first descent's to set)
    h = surface.edge_scale if surface.edge_scale is not None else _median_edge(m)
    s = float(spacing) if spacing is not None else _SPACING*float(h)
    periods = m.periods
    lo, hi, sp, shape = _grid(x, solids, s, periods)
    while np.prod(shape, dtype=float) > _NODES:
        s *= 1.25
        lo, hi, sp, shape = _grid(x, solids, s, periods)
        _log.debug("clusters: the grid's spacing grown to %.3g (%s nodes)", s, shape)
    periodic = (periods is not None,)*2 + (False,)
    nodes = int(np.prod(shape))
    p = lo + sp*np.stack(np.unravel_index(np.arange(nodes), shape), axis=1)
    # each solid's nodes near it and their gaps (in a periodic slab, near
    # any of its images)
    near_solid = []
    solid = np.zeros(nodes, bool)
    for _, kind, data in solids:
        idx = _near(kind, data, lo, sp, shape, _NEAR*s, periodic)
        gap = _gap(kind, data, p[idx], periods)
        solid[idx[gap < 0]] = True
        near_solid.append((idx, gap))
    if periods is None:
        tri = ((x - lo)/s)[f]
    else:
        # each facet whole, its first corner in the grid's cell
        tri = np.array(m.facet_corners())
        L = np.asarray(periods, float)
        tri[:, :, :2] -= (L*np.floor((tri[:, 0, :2] - lo[:2])/L))[:, None, :]
        tri = (tri - lo)/sp
    axis, low, frac, row, sign = _crossings(tri, shape, periodic)
    strides = np.array([shape[1]*shape[2], shape[2], 1])

    def up(node: np.ndarray, k) -> np.ndarray:
        """The next node along axis k (round the grid along a periodic one)."""
        if periods is None:
            return node + strides[k]
        k = np.broadcast_to(k, node.shape)
        idx = np.stack(np.unravel_index(node, shape), axis=1)
        idx[np.arange(len(node)), k] += 1
        idx %= np.array(shape)          # (z's last node has no segment)
        return np.ravel_multi_index(tuple(idx.T), shape)

    # links between neighbouring nodes in the liquid or gas
    t = np.linspace(0, 1, 9)[1:-1]
    a_all, b_all = [], []
    for k in range(3):
        idx = np.arange(nodes).reshape(shape)
        a = idx[tuple(slice(0, -1) if d == k and not periodic[k] else slice(None)
                      for d in range(3))].ravel()
        b = up(a, k)
        cut = np.zeros(nodes, bool)
        cut[low[axis == k]] = True
        cut |= solid
        if periodic[k]:
            cut[a] |= solid[b]
        else:
            cut[:-strides[k]] |= solid[strides[k]:]
        for (_, kind, data), (idx, gap) in zip(solids, near_solid):
            # (a sphere or cylinder can bulge between two nodes outside it)
            if kind == "plane":
                continue
            close = idx[(gap >= 0) & (gap < s)]
            close = close[~cut[close]]
            if len(close):
                q = p[close][:, None, :] + sp[k]*t[None, :, None]*np.eye(3)[k]
                inside = (_gap(kind, data, q.reshape(-1, 3), periods) < 0).reshape(-1, len(t))
                cut[close[inside.any(axis=1)]] = True
        ok = ~cut[a]
        a_all.append(a[ok])
        b_all.append(b[ok])
    group = components(nodes, np.concatenate(a_all), np.concatenate(b_all))
    # each segment's nearest crossing to either end: that end's side
    order = np.lexsort((frac, low, axis))
    key = axis[order]*nodes + low[order]
    first = np.r_[True, key[1:] != key[:-1]]
    last = np.r_[key[1:] != key[:-1], True]
    fo, la = order[first], order[last]
    node = np.r_[low[fo], up(low[la], axis[la])]
    side = np.r_[np.where(sign[fo] > 0, 0, 1), np.where(sign[la] > 0, 1, 0)]
    frow = np.r_[row[fo], row[la]]
    keep = ~solid[node]
    node, side, frow = node[keep], side[keep], frow[keep]
    voter = piece[frow, side]
    liquid = voter >= 0
    g = group[node]
    gas_groups = set(np.unique(g[~liquid]).tolist())
    liquid_groups = np.unique(g[liquid])
    mixed = len(gas_groups & set(liquid_groups.tolist()))
    if mixed:
        _log.debug("clusters: %d groups of nodes face both liquid and gas", mixed)
    # pieces joined through the groups their liquid sides face
    gid = np.searchsorted(liquid_groups, g[liquid])
    label = components(npieces + len(liquid_groups), voter[liquid], npieces + gid)
    # each liquid group's nodes and the walls they border
    member = np.isin(group, liquid_groups) & ~solid
    size = np.zeros(npieces + len(liquid_groups), np.intp)
    in_group = np.searchsorted(liquid_groups, group[member])
    np.add.at(size, npieces + in_group, 1)
    near_walls: Dict[object, np.ndarray] = {}
    where = np.full(nodes, -1, np.intp)
    where[member] = in_group
    for (key, _, _), (idx, gap) in zip(solids, near_solid):
        close = idx[(gap >= 0) & (gap < _NEAR*s)]
        close = where[close][where[close] >= 0]
        if len(close):
            near = np.zeros(npieces + len(liquid_groups), np.intp)
            np.add.at(near, npieces + close, 1)
            near_walls[key] = near
    reached = np.zeros(npieces + len(liquid_groups), bool)
    reached[:npieces] = by_vertex
    from ._surgery import _base
    for key in outlet:
        for wall in near_walls:
            if _base(wall) == _base(key):
                reached |= near_walls[wall] > 0
    return _gather(piece, body_of, label, reached, size, near_walls,
                   s if periods is None else float(np.prod(sp))**(1/3), mixed)


def _gather(piece, body_of, label, reached, size, near_walls, s, mixed) -> Clusters:
    """clusters()'s result from the union-find's labels over the pieces (and
    the grid's liquid groups after them)."""
    npieces = len(body_of)
    roots, cluster = np.unique(label[:npieces], return_inverse=True)
    full = np.searchsorted(roots, label)
    has = np.isin(label, roots)     # (a group no piece faces: no cluster)
    n = len(roots)
    outlet = np.zeros(n, bool)
    np.logical_or.at(outlet, full[has], np.asarray(reached)[has])
    nodes = np.zeros(n, np.intp)
    np.add.at(nodes, full[has], np.asarray(size)[has])
    walls = {}
    for key, near in near_walls.items():
        w = np.zeros(n, np.intp)
        np.add.at(w, full[has], near[has])
        walls[key] = w
    bodies: List[set] = [set() for _ in range(n)]
    for i, b in zip(cluster, body_of):
        bodies[i].add(int(b))
    side = np.where(piece >= 0, cluster[np.maximum(piece, 0)], -1)
    return Clusters(side, [frozenset(b) for b in bodies], outlet, nodes, walls, s, mixed)


def reconcile(ev) -> None:
    """In a topology run (``ev._liquid``, set by recipes.arclength: the body
    it steps, its outlet walls and the bodies it started with), the bodies
    made to match the liquid's clusters: those reaching the outlet one body,
    the run's (several pieces allowed: one pressure, their total stepped);
    without an outlet, or where none reaches it, the run's body's largest
    cluster (the run then stops: ``ev._liquid["dry"]``). Every other cluster
    its own body, a cluster of two bodies one (the lower number, or the
    run's). Targets: each body split or joined gives each of its pieces what
    it holds, and the rest of its target (what the run has yet to step, a
    cut's collapse) to its largest piece, the run's body's to the run's; so
    the targets' total stays as it was. Fixed volumes only; a free body is
    left as it is."""
    run = getattr(ev, "_liquid", None)
    if run is None:
        return
    keep, outlet, original = run["keep"], run["outlet"], run["original"]
    bodies = ev.bodies()
    fixed = {int(i) for i, fx in zip(bodies.ids, bodies.fixed) if fx}
    c = clusters(ev, outlet)
    run["dry"] = False
    if not len(c.bodies):
        return
    m = ev.mesh()
    fb = np.asarray(m.facet_bodies)
    # (a cluster with no liquid on the grid, apart from the outlet, kept
    # with its body's largest: pieces enclosing nothing, which the
    # descents' tidying takes away, made bodies of no volume, some
    # negative; only a ring holding under a node's worth drains with it)
    side = c.side.copy()
    for b in fixed:
        mine = np.unique(side[fb == b])
        mine = mine[mine >= 0]
        empty = [cl for cl in mine if c.nodes[cl] == 0 and not c.outlet[cl]]
        full = [cl for cl in mine if cl not in empty]
        if empty and full:
            into = max(full, key=lambda cl: c.nodes[cl])
            side[(fb == b) & np.isin(side, empty)] = into
    on = np.isin(fb, list(fixed)) & (side >= 0)
    pairs = np.unique(np.c_[fb[on], side[on]], axis=0) if on.any() else np.zeros((0, 2), int)
    parts = [(int(b), int(cl)) for b, cl in pairs]
    if not parts:
        return
    facets = {p: int(((fb == p[0]) & (side == p[1])).any(axis=1).sum()) for p in parts}

    def size(p) -> tuple:
        return (int(c.nodes[p[1]]), facets[p])

    of_body: Dict[int, list] = {}
    in_cluster: Dict[int, list] = {}
    for p in parts:
        of_body.setdefault(p[0], []).append(p)
        in_cluster.setdefault(p[1], []).append(p)
    # the run's clusters
    drains: set = set()
    if keep in fixed:
        if outlet:
            drains = {cl for cl, ps in in_cluster.items() if c.outlet[cl]
                      and any(b == keep or b not in original for b, _ in ps)}
            run["dry"] = not drains
        if not drains and keep in of_body:
            drains = {max(of_body[keep], key=size)[1]}
    main = {b: max(ps, key=size)[1] for b, ps in of_body.items()}
    number: Dict[int, int] = {}
    for cl in in_cluster:
        if cl in drains:
            number[cl] = keep
        else:
            own = [b for b in of_body if main[b] == cl and b != keep]
            number[cl] = min(own) if own else 0
    moved = [p for p in parts if number[p[1]] != p[0]]
    if not moved:
        if run["dry"]:
            _log.debug("clusters: no liquid reaches the outlet (%s)", ", ".join(map(str, outlet)))
        return
    from ._surgery import _rebody
    total = float(np.sum(np.asarray(bodies.target)[np.asarray(bodies.fixed, bool)]))
    ev.command("recalc")                # (the volumes as the change left them)
    held = {b: ev.body(b).volume for b in of_body}
    target = {b: ev.body(b).target for b in of_body}
    volconst = {b: ev.body(b).volconst for b in of_body}
    for cl in sorted(in_cluster):
        if not number[cl]:
            ev.command("pyse_new_body := new_body")
            number[cl] = int(ev.eval("pyse_new_body"))
    involved = sorted({b for b, _ in moved} | {number[cl] for _, cl in moved})
    for b, cl in moved:
        for s, name in ((0, "frontbody"), (1, "backbody")):
            mask = (fb[:, s] == b) & (side[:, s] == cl)
            if mask.any():
                ev.command(f"set facet {name} {number[cl]}{ev._mark(2, mask)} and {name} == {b}")
    gone = sorted(set(of_body) - set(number.values()))
    for b in gone:
        ev.command(f"dissolve body[{b}]")
    _rebody(ev)
    # each changed body's constant: the walls' (sphere(axis=)'s poles) with
    # the piece on them (by vertices, else by the grid), the rest with its target
    final = {number[cl] for cl in in_cluster}
    changed = {number[cl] for b, cl in parts if b in involved}
    const = {n: 0.0 for n in changed}
    rest = {n: 0.0 for n in changed}
    walls = {k: float(getattr(w, "volconst", 0.0)) for k, w in surface.walls.items()
             if getattr(w, "volconst", 0.0)}
    for b in involved:
        if b not in of_body:
            continue
        to = number[next(iter(drains))] if b == keep and drains else number[main[b]]
        if to not in changed:
            continue
        rest[to] += target[b] - held[b] if target[b] is not None else 0.0
        mine = volconst[b]
        for k, v in walls.items():
            on_wall = np.asarray(ev.on_constraint(k))[np.asarray(m.facets)]
            counts = {p: (int((on_wall & ((fb == b) & (side == p[1])).any(axis=1)[:, None])
                              .sum()), int(c.walls.get(k, np.zeros(len(c.bodies)))[p[1]]))
                      for p in of_body[b]}
            best = max(counts, key=lambda p: counts[p])
            if counts[best] == (0, 0):
                continue
            const[number[best[1]]] += v
            mine -= v
        const[to] += mine
    for n in sorted(changed & final):
        body = ev.body(n)
        body.volconst = const[n]
    ev.command("recalc")
    for n in sorted(changed & final):
        body = ev.body(n)
        body.target = body.volume + rest[n]
    after = ev.bodies()
    now = float(np.sum(np.asarray(after.target)[np.asarray(after.fixed, bool)]))
    _log.debug("clusters: bodies %s made %s (%s)%s%s",
               involved, sorted(changed & final),
               ", ".join(f"{n}: {ev.body(n).target:.6g}" for n in sorted(changed & final)),
               "; no liquid reaches the outlet" if run["dry"] else "",
               f"; the targets' total changed by {now - total:.3g}"
               if abs(now - total) > 1e-9*max(1.0, abs(total)) else "")
