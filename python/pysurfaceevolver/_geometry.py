"""Mesh geometry in NumPy, for relax(), health() and adapt()."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ._mesh import Mesh

# a few vertices (this fraction at most) on the other side of an equality
# constraint than all the others have crossed it
_FEW_CROSSED = 0.05
_CLOSE_PAIRS = 5000             # _self_gap() looks through this many nearest pairs, at most


def _crossed(nums: np.ndarray, attrs: np.ndarray, dist: np.ndarray, used: np.ndarray,
             eps: float) -> Dict[int, int]:
    """Per constraint, the vertices on its far side: the forbidden side of a
    one-sided constraint, or a few (at most 5%) against all the others."""
    out: Dict[int, int] = {}
    for c in range(len(nums)):
        d = dist[:, c]
        off = ~np.isnan(d) & used
        if not off.any():
            continue
        if attrs[c] & 2:            # NONNEGATIVE
            bad = int((d[off] < -eps).sum())
        elif attrs[c] & 1:          # NONPOSITIVE
            bad = int((d[off] > eps).sum())
        else:
            pos, neg = int((d[off] > eps).sum()), int((d[off] < -eps).sum())
            bad = min(pos, neg) if min(pos, neg) <= _FEW_CROSSED*(pos + neg) else 0
        if bad:
            out[int(nums[c])] = bad
    return out


def _grow(mask: np.ndarray, edges: np.ndarray, rings: int) -> np.ndarray:
    """The vertices within ``rings`` edges of the masked ones."""
    out = mask.copy()
    for _ in range(rings):
        nxt = out.copy()
        nxt[edges[out[edges[:, 1]], 0]] = True
        nxt[edges[out[edges[:, 0]], 1]] = True
        out = nxt
    return out


def _self_gap(x: np.ndarray, edges: np.ndarray, used: np.ndarray, h: float) -> float:
    """The smallest distance, in units of h, between used vertices more than
    two edges apart (inf if none are closer than h/2)."""
    idx = np.flatnonzero(used)
    if len(idx) < 2 or len(edges) == 0:
        return float("inf")
    r = h/2
    p = x[idx]
    # cells on the first three coordinates (zero-padded in the plane): pairs
    # closer than r in the full space are closer in those too
    grid = np.zeros((len(p), 3))
    grid[:, :min(3, p.shape[1])] = p[:, :3]
    cell = np.floor(grid/r).astype(np.int64)
    cell -= cell.min(axis=0) - 1                 # a margin: neighbours never wrap
    dims = cell.max(axis=0) + 2
    k = (cell[:, 0]*dims[1] + cell[:, 1])*dims[2] + cell[:, 2]
    order = np.argsort(k, kind="stable")
    sk = k[order]
    found_i, found_j, found_d = [], [], []
    offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
    offsets = offsets[[tuple(o) >= (0, 0, 0) for o in offsets]]   # each pair of cells once
    for off in offsets:
        same = not off.any()
        nk = sk + (off[0]*dims[1] + off[1])*dims[2] + off[2]       # sorted, like sk
        lo = np.searchsorted(sk, nk, "left")
        cnt = np.searchsorted(sk, nk, "right") - lo
        if not cnt.any():
            continue
        i = np.repeat(order, cnt)
        within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        j = order[np.repeat(lo, cnt) + within]
        keep = i < j if same else i != j
        i, j = i[keep], j[keep]
        d = np.linalg.norm(p[i] - p[j], axis=1)
        close = d < r
        found_i.append(i[close])
        found_j.append(j[close])
        found_d.append(d[close])
    d = np.concatenate(found_d)
    if len(d) == 0:
        return float("inf")
    i = idx[np.concatenate(found_i)]
    j = idx[np.concatenate(found_j)]
    # the nearest pair more than two edges apart (close pairs are few)
    a = np.concatenate([edges[:, 0], edges[:, 1]])
    b = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    start = np.searchsorted(a, np.arange(len(x)), "left")
    stop = np.searchsorted(a, np.arange(len(x)), "right")
    for k in np.argsort(d)[:_CLOSE_PAIRS]:
        ni = b[start[i[k]]:stop[i[k]]]
        nj = b[start[j[k]]:stop[j[k]]]
        if j[k] in ni or np.intersect1d(ni, nj).size:
            continue
        return float(d[k]/h)
    return float("inf")


def _edge_turn(m: Mesh) -> "tuple[np.ndarray, np.ndarray]":
    """Per edge, the angle (degrees, 0-90) between the vertex normals at its
    ends, and whether it was judged: not for edges touching a vertex where
    three or more facets meet on an edge (their angle is 0). Vertex normals:
    area-weighted facet normals, each sign-aligned with one facet at the
    vertex (facet orientations needn't agree across a sheet)."""
    f, x = m.facets, m.vertices
    assert f is not None
    n = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    ref = np.zeros_like(x)
    for k in range(3):
        ref[f[:, k]] = n
    vn = np.zeros_like(x)
    for k in range(3):
        s = np.where(np.einsum("ij,ij->i", n, ref[f[:, k]]) < 0, -1.0, 1.0)
        np.add.at(vn, f[:, k], n*s[:, None])
    norm = np.linalg.norm(vn, axis=1)
    vn = vn/np.where(norm > 0, norm, 1.0)[:, None]
    e = m.edges
    turn = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", vn[e[:, 0]], vn[e[:, 1]])),
                                        0.0, 1.0)))
    junction = np.zeros(len(x), dtype=bool)
    hub = e[m.edge_valence() > 2]
    junction[hub.ravel()] = True
    unjudged = junction[e[:, 0]] | junction[e[:, 1]] | (norm[e[:, 0]] == 0) | (norm[e[:, 1]] == 0)
    turn[unjudged] = 0.0
    return turn, ~unjudged


def _long_spokes(m: Mesh, ratio: float) -> np.ndarray:
    """Edges with one end on a contact line (a vertex of an edge on one facet
    only) and the other off it, longer than ``ratio`` times the mean length of
    the contact-line edges at that end."""
    if m.facets is None or len(m.facets) == 0:
        return np.zeros(len(m.edges), dtype=bool)
    e, n = m.edges, len(m.vertices)
    lengths, valence = m.edge_lengths(), m.edge_valence()
    rim = valence == 1
    total, count = np.zeros(n), np.zeros(n)
    for k in (0, 1):
        np.add.at(total, e[rim, k], lengths[rim])
        np.add.at(count, e[rim, k], 1)
    on = count > 0
    mean = total/np.maximum(count, 1)
    spoke = (valence >= 2) & (on[e[:, 0]] != on[e[:, 1]])
    ref = np.where(on[e[:, 0]], mean[e[:, 0]], mean[e[:, 1]])
    return spoke & (lengths > ratio*ref)


def _chains(edges: np.ndarray) -> "list[list[int]]":
    """The edges (vertex pairs) joined into paths of vertices, longest first;
    a closed loop repeats its first vertex. Paths end at junctions."""
    links: Dict[int, list] = {}
    for a, b in edges.tolist():
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    used: set = set()
    paths = []
    for start in [k for k, v in links.items() if len(v) != 2] + list(links):
        for nxt in links[start]:
            if (min(start, nxt), max(start, nxt)) in used:
                continue
            path, prev, cur = [start], start, nxt
            used.add((min(start, nxt), max(start, nxt)))
            while True:
                path.append(cur)
                if len(links[cur]) != 2 or cur == start:
                    break
                follow = links[cur][0] if links[cur][1] == prev else links[cur][1]
                if (min(cur, follow), max(cur, follow)) in used:
                    break
                used.add((min(cur, follow), max(cur, follow)))
                prev, cur = cur, follow
            paths.append(path)
    return sorted(paths, key=len, reverse=True)
