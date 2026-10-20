"""Checked local mesh edits (bindings/pyse_edit.c): collapse, flip, split,
smooth, snap, separate, and degenerate-facet removal, each refused or
undone where it would leave the surface round it worse (the topology
check's problems: structure, pinches, degenerate facets, folds, solids,
edges through facets). For pySE's changes of topology and remeshing."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from . import _core
from ._engine import surface

OPS = {"collapse": 1, "flip": 2, "split": 3, "smooth": 4, "snap": 5, "separate": 6,
       "degenerate": 7, "merge": 8, "bridge": 9, "remesh": 10, "patch": 11}
# why an edit wasn't made (0: it was)
REASONS = {0: "done", 1: "gone (an earlier edit took it)", 2: "not applicable",
           3: "no better", 4: "fixed", 5: "constraints", 6: "a contact line's",
           7: "the link condition", 8: "orientation", 9: "a facet would turn",
           10: "an edge would be too long", 11: "the volume would change",
           12: "Evolver refused", 13: "structure", 14: "a pinch", 15: "a degenerate facet",
           16: "a fold", 17: "into a solid", 18: "an edge through a facet",
           19: "not the same in the replay"}
DONE = 0
_PARAMS = ("max_turn", "max_volume", "max_len", "tol", "mode", "smoothing", "offset",
           "flat", "h", "sweeps")
DEFAULTS = dict(max_turn=45.0, max_volume=0.05, max_len=0.0, tol=0.01, mode=0,
                smoothing=0.5, offset=0.05, flat=1e-4, h=0.0, sweeps=3)


def walls(ev) -> Tuple[List[int], List[int]]:
    """The walls (pySE's constraints with a shape, not guards or retired
    ones) as constraint numbers, and each one's solid side (+1: where its
    formula is negative) as events() takes it: a one-sided constraint's
    forbidden side, else the side the fewest vertices are on."""
    nums, attrs, dist, names = ev._constraint_gaps()
    out, sides = [], []
    for c in range(len(nums)):
        if (names[c] or int(nums[c])) not in surface.walls:
            continue
        d = np.asarray(dist)[:, c]
        off = d[~np.isnan(d)]
        if attrs[c] & 2:
            side = 1
        elif attrs[c] & 1:
            side = -1
        else:
            side = 1 if (off > 0).sum() >= (off < 0).sum() else -1
        out.append(int(nums[c]))
        sides.append(side)
    return out, sides


def edit(ev, op: str, ids: Iterable, sizes=None, **params) -> np.ndarray:
    """One batch of edits of a kind (OPS) on the elements numbered ids
    (Evolver's numbers, as Mesh.edge_ids and the like give; pairs for
    "snap" (vertices), "merge" (facing facets) and "bridge" (facing border
    edges on one wall); for "patch", the wall's constraint number then the
    facets of a wet or dry patch, one edit): per edit, DONE (0) or why not
    (REASONS). Parameters
    (DEFAULTS): max_turn (degrees a facet's normal may turn), max_volume
    (the volume an edit may sweep, in h^3), max_len (longest edge a collapse
    may make, in h; 0: any), tol (depth into a wall allowed, in h), mode
    (flips: 0 any, 1 more equiangular, 2 valences nearer 6), smoothing (the
    step toward the neighbours), offset (how far separated vertices move
    apart, in h), flat (a degenerate facet's area, in h^2), h (the length
    scale; 0: the mean edge), sweeps (remesh). ``sizes``: a target edge
    length per vertex (as ev.mesh()'s rows; 0 or NaN: h), for remesh's splits
    and collapses (each edge's, its ends' mean) and a collapse's longest
    edge; a split's new vertex takes its ends' mean."""
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"edit(): unknown parameters {sorted(unknown)}")
    values = [float(params.get(k, DEFAULTS[k])) for k in _PARAMS]
    rows = [int(i) for i in np.asarray(list(ids), dtype=np.int64).ravel()]
    if not rows and op != "remesh":
        return np.zeros(0, dtype=int)
    w, sides = walls(ev)
    manifold = _core.count(_core.BODY) <= 1
    by_number: list = []
    if sizes is not None:
        numbers = np.asarray(ev.mesh().vertex_ids, dtype=np.int64)
        sizes = np.nan_to_num(np.asarray(sizes, float), nan=0.0)
        if len(sizes) != len(numbers):
            raise ValueError("edit(): one size per vertex")
        table = np.zeros(int(numbers.max(initial=0)), float)
        table[numbers - 1] = sizes
        by_number = table.tolist()
    _, result = ev._call(_core.edit, OPS[op], rows, values, w, sides, manifold, by_number)
    return np.asarray(result.data, dtype=int)


def crossings(ev, far: bool = False) -> int:
    """Edges through facets on the whole surface (pairs sharing no vertex);
    far: only those more than a ring of edges apart (an edge's end not
    joined to the facet's corners): two sheets through each other, not a
    fold of the mesh. With pairs of contact-line edges on one wall crossing
    each other on it (lying on the wall, neither goes through a facet)."""
    _, result = ev._call(_core.crossings, bool(far))
    return int(result.data)


def crossing_places(ev, far: bool = False) -> np.ndarray:
    """Where the crossings counted by crossings() are (n, 3): each edge's
    midpoint, the middle of two crossed contact-line edges (up to 4096)."""
    _, result = ev._call(_core.crossing_places, bool(far))
    return np.asarray(result.data, float).reshape(-1, 3)


def remesh(ev, h: float, sweeps: int = 3, sizes=None, region=None, **params) -> dict:
    """Isotropic remeshing toward edges of length h (Botsch and Kobbelt):
    per sweep, edges over 4/3 h split, under 4/5 h collapsed, flips toward
    valence 6 (4 on a border), vertices smoothed along the surface, each
    edit checked (see edit); contact lines kept. Up to sweeps sweeps, fewer
    where one changes nothing but positions; with ``sizes`` (per vertex, see
    edit), toward those lengths instead; with ``region`` (a mask or rows of
    ev.mesh()'s vertices), there only (the edges with an end there, a
    split's new vertex joining it). How many of each were made."""
    ids = [] if region is None else np.asarray(ev.mesh().vertex_ids)[np.asarray(region)]
    if region is not None and not len(ids):
        return dict.fromkeys(("splits", "collapses", "flips", "moves"), 0)
    done = edit(ev, "remesh", ids, sizes=sizes, h=h, sweeps=sweeps, **params)
    return dict(zip(("splits", "collapses", "flips", "moves"), (int(k) for k in done)))
