"""Mesh snapshots: high-order elements, tessellation, per-body surfaces,
and conversion to meshio and PyVista."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from math import factorial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from . import _html

if TYPE_CHECKING:  # optional dependencies
    import meshio
    import pyvista

__all__ = ["Mesh", "Bodies", "Quantity", "BodySurface", "LargeTessellationWarning"]


class LargeTessellationWarning(UserWarning):
    """A tessellation has more triangles than ``pse.tessellation_limit``."""


def _check_tessellation_size(facets: int, n: int, sdim: int) -> None:
    """Warn when ``facets`` facets subdivided ``n`` times exceed the limit."""
    limit = getattr(sys.modules.get("pysurfaceevolver"), "tessellation_limit", None)
    triangles = facets * n * n
    if limit is None or triangles <= limit:
        return
    per_facet = (n + 1) * (n + 2) // 2
    # sampled points (and the merged copy), triangles (and their renumbered copy)
    gb = facets * (2 * per_facet * sdim * 8 + 2 * n * n * 3 * 8) / 1e9
    warnings.warn(
        f"tessellating {facets:,} facets with n={n} gives {triangles:,} triangles "
        f"(about {gb:.1f} GB while building, more for plotting or export); pass a "
        f"smaller n, or raise pse.tessellation_limit (now {limit:,}; None: no check)",
        LargeTessellationWarning, stacklevel=4)



# ---------------------------------------------------------------------------
# Reference elements

def _lattice(n: int, dim: int) -> np.ndarray:
    """Barycentric lattice points with denominator n, as integer rows.

    For dim 2, rows are (n-a-b, a, b), a outer.
    """
    if dim == 1:
        return np.array([[n - a, a] for a in range(n + 1)])
    return np.array([[n - a - b, a, b] for a in range(n + 1) for b in range(n + 1 - a)])


def _lattice_triangles(n: int) -> np.ndarray:
    """Triangles of _lattice(n, 2), oriented like corners (0, 1, 2)."""
    index = {}
    for row, (_, a, b) in enumerate(_lattice(n, 2)):
        index[a, b] = row
    tris = []
    for a in range(n):
        for b in range(n - a):
            tris.append((index[a, b], index[a + 1, b], index[a, b + 1]))
            if a + b < n - 1:
                tris.append((index[a + 1, b], index[a + 1, b + 1], index[a, b + 1]))
    return np.array(tris, dtype=np.int64)


def _basis(node_index: np.ndarray, order: int, bezier: bool, lam: np.ndarray) -> np.ndarray:
    """Shape functions of every node at barycentric points lam.

    node_index: (nodes, d+1) multi-indices summing to order.
    lam: (points, d+1) barycentric coordinates.
    Returns (points, nodes).
    """
    out = np.ones((len(lam), len(node_index)))
    for k, alpha in enumerate(node_index):
        if bezier:
            # Bernstein polynomial: order!/prod(alpha!) * prod(lam^alpha)
            coef = factorial(order) / np.prod([factorial(int(a)) for a in alpha])
            out[:, k] = coef * np.prod(lam ** alpha, axis=1)
        else:
            # Lagrange polynomial for the node at alpha/order, as in model.c
            for i, a in enumerate(alpha):
                for m in range(int(a)):
                    out[:, k] *= (order * lam[:, i] - m) / (a - m)
    return out


def _recursive_triangle_order(p: int) -> List[Tuple[int, int, int]]:
    """Node order used by both Gmsh and VTK for order-p triangles.

    Corners, then the nodes of edges 0-1, 1-2, 2-0, then the interior nodes
    recursively as an order p-3 triangle. Entries are barycentric
    multi-indices summing to p.
    """
    if p == 0:
        return [(0, 0, 0)]
    out = [(p, 0, 0), (0, p, 0), (0, 0, p)]
    out += [(p - k, k, 0) for k in range(1, p)]
    out += [(0, p - k, k) for k in range(1, p)]
    out += [(k, 0, p - k) for k in range(1, p)]
    if p >= 3:
        out += [(a + 1, b + 1, c + 1) for a, b, c in _recursive_triangle_order(p - 3)]
    return out


def _reversed_facets(faces: np.ndarray, facet_nodes: np.ndarray,
                     node_index: np.ndarray, order: int) -> np.ndarray:
    """Facets whose node layout runs opposite to their ``faces`` row."""
    corners = [int(np.flatnonzero(node_index[:, i] == order)[0]) for i in range(3)]
    c = facet_nodes[:, corners]
    same = np.zeros(len(c), dtype=bool)
    for shift in range(3):
        same |= (np.roll(c, shift, axis=1) == faces).all(axis=1)
    return ~same


def native_cell_name(kind: str, order: int, bezier: bool, flavor: str) -> str:
    """meshio cell type name for an order-``order`` triangle or line.

    ``flavor`` is the target: ``"gmsh"`` (Gmsh files), ``"vtk"`` (VTU and
    legacy VTK), ``"xdmf"``, or ``"linear"`` (formats with flat cells only).
    Raises ``ValueError`` for combinations the format can't store.
    """
    def unsupported() -> ValueError:
        what = "Bezier" if bezier else f"order-{order}"
        return ValueError(f"{flavor} files can't store {what} {kind}s; "
                          "use curved='tessellate' or another format")
    if order == 1:
        return kind
    if flavor == "gmsh":
        if bezier or order > 10:
            raise unsupported()
        return f"line{order + 1}" if kind == "line" else f"triangle{(order + 1) * (order + 2) // 2}"
    if flavor == "vtk":
        if kind == "line":
            if bezier or order > 3:
                raise unsupported()
            return f"line{order + 1}"
        if bezier:
            return "VTK_BEZIER_TRIANGLE"
        return "triangle6" if order == 2 else "VTK_LAGRANGE_TRIANGLE"
    if flavor == "xdmf":
        if bezier or order > 2:
            raise unsupported()
        return "line3" if kind == "line" else "triangle6"
    raise unsupported()


def _as_3d(points: np.ndarray) -> np.ndarray:
    """Pad 1-D/2-D coordinates with zeros; keep the first 3 of higher ones."""
    if points.shape[1] == 3:
        return points
    if points.shape[1] > 3:
        return points[:, :3]
    return np.hstack([points, np.zeros((len(points), 3 - points.shape[1]))])


def _directed_edge_keys(triangles: np.ndarray):
    """Directed edges of triangles as integer keys u*N + v, the reverse keys,
    and the endpoints."""
    t = np.asarray(triangles, dtype=np.int64)
    u = t.ravel()
    v = t[:, [1, 2, 0]].ravel()
    N = int(t.max()) + 1 if t.size else 1
    return u * N + v, v * N + u, u, v


def _has_reverse(keys: np.ndarray, rev: np.ndarray) -> "tuple[np.ndarray, bool]":
    """For each edge, whether its reverse exists; and whether keys are unique."""
    order = np.sort(keys)
    unique = not (order[1:] == order[:-1]).any()
    pos = np.searchsorted(order, rev)
    pos[pos == len(order)] = 0
    return order[pos] == rev, unique


def _boundary_loops(triangles: np.ndarray) -> List[List[int]]:
    """Open boundary loops of a triangle surface, as point index lists.

    Each loop runs along the boundary edges in the triangles' own direction.
    """
    if len(triangles) == 0:
        return []
    keys, rev, u, v = _directed_edge_keys(triangles)
    found, _ = _has_reverse(keys, rev)
    boundary = list(zip(u[~found].tolist(), v[~found].tolist()))
    following: Dict[int, List[int]] = {}
    for a, b in boundary:
        following.setdefault(a, []).append(b)
    loops = []
    used = set()
    for start_edge in boundary:
        if start_edge in used:
            continue
        loop = [start_edge[0]]
        a, b = start_edge
        while True:
            used.add((a, b))
            if b == loop[0]:
                break
            loop.append(b)
            nxt = [w for w in following.get(b, []) if (b, w) not in used]
            if not nxt:
                break  # not a simple loop; leave it open
            a, b = b, nxt[0]
        loops.append(loop)
    return loops


def _compact(cells: np.ndarray, n_points: int) -> "tuple[np.ndarray, np.ndarray]":
    """Renumber the points used by cells as 0..m-1 (in point order).

    Returns (used point indices, renumbered cells); O(n), no sorting.
    """
    used = np.zeros(n_points, dtype=bool)
    used[cells.ravel()] = True
    new = np.cumsum(used) - 1
    return np.flatnonzero(used), new[cells]


def is_watertight(triangles: np.ndarray) -> bool:
    """True if every edge is shared by exactly two triangles, with opposite
    directions (a closed, consistently oriented surface)."""
    if len(triangles) == 0:
        return False
    keys, rev, _, _ = _directed_edge_keys(triangles)
    found, unique = _has_reverse(keys, rev)
    return unique and bool(found.all())


# ---------------------------------------------------------------------------
# Data classes

@dataclass
class BodySurface:
    """The closed surface around one body, oriented with outward normals.

    ``cells`` are triangles (``cell_type == "triangle"``) or native
    high-order triangles in Gmsh/VTK node order (``"triangle6"``,
    ``"triangle10"``, ...).
    """

    body: int
    points: np.ndarray
    cells: np.ndarray
    cell_type: str
    facet_ids: np.ndarray      # Evolver facet id of each cell; 0 for caps
    watertight: bool
    open_loops: int            # boundary loops before capping
    order: int = 1
    bezier: bool = False
    cap_ids: Optional[np.ndarray] = None   # k on cells of cap k, 0 elsewhere
    cap_constraints: Dict[int, Optional[int]] = field(default_factory=dict)
    # the constraint cap k lies on (None: a flat cap)

    def to_meshio(self, flavor: str = "gmsh") -> "meshio.Mesh":
        """As a meshio mesh; ``flavor`` picks high-order cell names (see
        :func:`native_cell_name`)."""
        import meshio
        cell_type = native_cell_name("triangle", self.order, self.bezier, flavor) \
            if self.cell_type != "triangle" else "triangle"
        return meshio.Mesh(_as_3d(self.points), [(cell_type, self.cells)],
                           cell_data={"facet_id": [self.facet_ids],
                                      "gmsh:physical": [np.full(len(self.cells), self.body)],
                                      "gmsh:geometrical": [self.facet_ids]})

    def volume_mesh(self, size: Optional[float] = None, path: Optional[str] = None,
                    *, algorithm: int = 10) -> "meshio.Mesh":
        """Tetrahedra filling this closed surface, made with Gmsh
        (``pip install gmsh``); the surface triangles are kept as they are.

        Physical groups: 1 "surface" for the Evolver facets, 1 + k "cap k"
        for cap k (see ``cap_ids``), and the body id for the tetrahedra. The
        result is a meshio mesh with tetrahedra and boundary triangles, their
        groups in ``cell_data["gmsh:physical"]``. ``size`` is the largest
        element size (default: the mean edge length of the surface);
        ``path`` also writes the mesh (``.msh`` through Gmsh, with the group
        names; other formats through meshio). ``algorithm`` is Gmsh's
        ``Mesh.Algorithm3D`` (10: HXT, 1: Delaunay).
        """
        if self.cell_type != "triangle":
            raise ValueError("volume_mesh() needs flat triangles (curved='tessellate')")
        if not self.watertight:
            raise ValueError("the surface isn't closed (see body_surfaces(cap=True))")
        try:
            import gmsh
        except ImportError:
            raise ImportError("volume_mesh() needs Gmsh: pip install gmsh") from None
        import contextlib
        import io
        import os
        import tempfile
        import meshio
        caps = self.cap_ids if self.cap_ids is not None else np.zeros(len(self.cells), int)
        groups = {1: "surface"}
        for k in sorted(set(caps.tolist()) - {0}):
            con = self.cap_constraints.get(k)
            groups[1 + k] = f"cap {k}" + (f" (constraint {con})" if con else "")
        if size is None:
            a, b = self.points[self.cells[:, 0]], self.points[self.cells[:, 1]]
            size = float(np.linalg.norm(a - b, axis=1).mean())
        started = gmsh.isInitialized()
        if not started:
            gmsh.initialize()
        old_terminal = gmsh.option.getNumber("General.Terminal")
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"pysurfaceevolver body {self.body}")
        try:
            for tag in groups:
                gmsh.model.addDiscreteEntity(2, tag)
            gmsh.model.mesh.addNodes(2, 1, np.arange(1, len(self.points) + 1),
                                     np.asarray(self.points, float).ravel())
            for tag in groups:
                cells = self.cells[caps == tag - 1]
                gmsh.model.mesh.addElementsByType(tag, 2, [], (cells + 1).ravel())
                gmsh.model.addPhysicalGroup(2, [tag], tag, groups[tag])
            # a volume bounded by the discrete surfaces (a discrete volume
            # would count as already meshed)
            gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(list(groups))], 1)
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(3, [1], self.body, f"body {self.body}")
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.optimize()
            with tempfile.TemporaryDirectory() as tmp:
                msh = os.path.join(tmp, "volume.msh")
                gmsh.write(msh)
                with contextlib.redirect_stdout(io.StringIO()):   # meshio prints a blank line
                    mesh = meshio.read(msh)
                if path is not None and str(path).endswith(".msh"):
                    gmsh.write(str(path))
        finally:
            gmsh.model.remove()
            gmsh.option.setNumber("General.Terminal", old_terminal)
            if not started:
                gmsh.finalize()
        if path is not None and not str(path).endswith(".msh"):
            meshio.write(path, mesh)
        return mesh

    def to_pyvista(self) -> "pyvista.PolyData":
        if self.cell_type != "triangle":
            raise ValueError("to_pyvista() needs flat triangles (curved='tessellate')")
        import pyvista as pv
        faces = np.hstack([np.full((len(self.cells), 1), 3), self.cells]).ravel()
        poly = pv.PolyData(_as_3d(self.points), faces=faces)
        poly.cell_data["facet_id"] = self.facet_ids
        return poly


class _Tessellation:
    """Mesh.tessellate()'s structure for one connectivity and ``n``: every
    sampled point as a weighted sum of a few vertex rows, grouped by the
    point's place in the facet lattice."""

    def __init__(self, weights: np.ndarray, nodes: np.ndarray, triangles: np.ndarray,
                 ids: Optional[np.ndarray], total: int):
        """weights: (P, nodes) basis values at the P lattice points; nodes:
        (k, nodes) vertex rows of each facet; triangles: (k*n*n, 3) into the
        sampled points; ids: merged point of each (facet, lattice point), or
        None (unmerged); total: number of sampled points."""
        self.triangles = triangles
        self.total = total
        k, per_facet = len(nodes), len(weights)
        if ids is None:
            first = np.arange(total)
        else:   # one (facet, lattice point) for each merged point
            first = np.empty(total, dtype=np.int64)
            first[ids] = np.arange(len(ids))
        lattice_of = first % per_facet
        by_lattice = np.argsort(lattice_of, kind="stable")
        bounds = np.searchsorted(lattice_of[by_lattice], np.arange(per_facet + 1))
        # per lattice point: output points, their vertex rows, the weights
        self.groups = []
        for p in range(per_facet):
            dest = by_lattice[bounds[p]:bounds[p + 1]]
            if not len(dest):
                continue
            cols = np.flatnonzero(np.abs(weights[p]) > 1e-15)   # one on a node
            rows = nodes[first[dest] // per_facet][:, cols]
            self.groups.append((dest, rows, weights[p, cols]))

    def sample(self, values: np.ndarray) -> np.ndarray:
        """Per-vertex-row ``values`` (coordinates, or data) at the points."""
        values = np.asarray(values, dtype=float)
        out = np.empty((self.total,) + values.shape[1:])
        for dest, rows, w in self.groups:
            acc = values[rows[:, 0]]
            if len(w) > 1 or w[0] != 1.0:
                acc *= w[0]
                for j in range(1, len(w)):
                    acc += w[j] * values[rows[:, j]]
            out[dest] = acc
        return out


@dataclass
class Mesh:
    """A snapshot of the surface geometry.

    ``edges`` and ``facets`` hold row indices into ``vertices``. The ``*_ids``
    arrays hold Evolver's own 1-based element numbers, as used in commands
    such as ``vertex[5].x``.

    In quadratic and Lagrange models, ``vertices`` also holds the extra nodes,
    ``facets`` and ``edges`` use the corner vertices only, and ``facet_nodes``
    and ``edge_nodes`` list every node of each element. Use
    :meth:`tessellate` for flat triangles that follow the curved facets.
    """

    vertices: np.ndarray       # (n, sdim) float64
    edges: np.ndarray          # (m, 2) int64
    facets: Optional[np.ndarray]  # (k, 3) int64; None outside the soapfilm model
    vertex_ids: np.ndarray     # (n,)
    edge_ids: np.ndarray       # (m,)
    facet_ids: Optional[np.ndarray]     # (k,)
    facet_bodies: Optional[np.ndarray]  # (k, 2) front/back body id, 0 = none
    fixed: np.ndarray          # (n,) bool, vertex has the FIXED attribute
    order: int = 1             # polynomial order of the elements
    bezier: bool = False       # nodes are Bezier control points, not on the surface
    edge_nodes: Optional[np.ndarray] = None        # (m, order+1) vertex rows
    edge_node_index: Optional[np.ndarray] = None   # (order+1, 2) barycentric
    facet_nodes: Optional[np.ndarray] = None       # (k, nodes) vertex rows
    facet_node_index: Optional[np.ndarray] = None  # (nodes, 3) barycentric

    def _repr_html_(self) -> str:
        order = "linear" if self.order == 1 else f"order {self.order}"
        if self.bezier:
            order += " (Bezier)"
        corners = len(np.unique(self.facets)) if self.facets is not None else None
        rows = [("elements", " · ".join(
                    [f"{_html.number(len(self.vertices))} vertex rows"]
                    + ([f"{_html.number(corners)} corners"]
                       if corners is not None and self.order > 1 else [])
                    + [f"{_html.number(len(self.edges))} edges"]
                    + ([f"{_html.number(len(self.facets))} facets"]
                       if self.facets is not None else []))),
                ("order", order),
                ("fixed vertices", _html.number(int(self.fixed.sum())))]
        if len(self.vertices):
            lo, hi = self.vertices.min(axis=0), self.vertices.max(axis=0)
            rows.append(("bounds", " × ".join(f"[{_html.number(a)}, {_html.number(b)}]"
                                               for a, b in zip(lo, hi))))
        return _html.fields(f"Mesh ({self.vertices.shape[1]}D)", rows)

    # ---- tessellation -----------------------------------------------------

    def _default_n(self, n: Optional[int]) -> int:
        if n is None:
            n = 1 if self.order == 1 else 2 * self.order
        if n < 1:
            raise ValueError("n must be at least 1")
        return n

    def _facet_data(self):
        faces, nodes, index = self.facets, self.facet_nodes, self.facet_node_index
        if faces is None or nodes is None or index is None:
            raise ValueError("this needs facets (soapfilm representation)")
        return faces, nodes, index

    def tessellate(self, n: Optional[int] = None, *, merge: bool = True,
                   values: Optional[np.ndarray] = None):
        """Subdivide every facet into n*n flat triangles, for plotting or export.

        Points are sampled on the curved (quadratic or Lagrange) facets, so the
        triangles follow the actual surface. ``n`` defaults to 1 for linear
        elements and 2*order otherwise. Triangle ``t`` belongs to facet
        ``t // n**2``, and is oriented like that facet's ``facets`` row.
        More than ``pse.tessellation_limit`` triangles (default 10 million)
        gives a :class:`LargeTessellationWarning`.

        With ``merge=True`` (the default), points shared by neighboring
        facets appear once, so the result is a connected surface. With
        ``merge=False``, each facet gets its own ``(n+1)(n+2)/2`` points.

        ``values``, one per vertex row (shape ``(n_vertices,)`` or
        ``(n_vertices, c)``), are interpolated like the coordinates and
        returned as a third item.

        In torus models, facets that cross the periodic boundary aren't
        unwrapped.
        """
        n = self._default_n(n)
        t = self._tessellation(n, merge)
        points = t.sample(self.vertices)
        if values is not None:
            vals = np.asarray(values, dtype=float)
            if vals.shape[0] != len(self.vertices):
                raise ValueError("values needs one entry per vertex row")
            return points, t.triangles, t.sample(vals)
        return points, t.triangles

    def _tessellation(self, n: int, merge: bool = True) -> "_Tessellation":
        """The structure of :meth:`tessellate`, which depends only on the
        connectivity: reusable for new coordinates or values."""
        faces, nodes, index = self._facet_data()
        _check_tessellation_size(len(nodes), n, self.vertices.shape[1])
        lattice = _lattice(n, 2)
        weights = _basis(index, self.order, self.bezier, lattice / n)   # (P, nodes)
        k, per_facet = len(nodes), len(lattice)
        tris = _lattice_triangles(n)
        tris = np.broadcast_to(tris, (k,) + tris.shape).copy()
        flip = _reversed_facets(faces, nodes, index, self.order)
        tris[flip] = tris[flip][:, :, ::-1]
        tris += (np.arange(k) * per_facet)[:, None, None]
        tris = tris.reshape(-1, 3)
        ids, total = None, k * per_facet
        if merge:
            ids, total = self._lattice_point_ids(nodes, index, n, lattice)
            tris = ids[tris]
        return _Tessellation(weights, nodes, tris, ids, total)

    def _lattice_point_ids(self, nodes, index, n, lattice) -> "tuple[np.ndarray, int]":
        """Merged point number of every (facet, lattice point), flattened.

        Numbered from the topology: facet corners by vertex, points inside
        edges by edge and position, points inside facets by facet. Points a
        facet shares with its neighbours get the same number.
        """
        k, nv = len(nodes), len(self.vertices)
        corners = np.stack([nodes[:, int(np.flatnonzero(index[:, i] == self.order)[0])]
                            for i in range(3)], axis=1)                  # (k, 3)
        used = np.zeros(nv, dtype=bool)
        used[corners.ravel()] = True
        corner_id = np.cumsum(used) - 1
        n_corner = int(used.sum())
        sides = [(0, 1), (1, 2), (2, 0)]
        lo = np.stack([np.minimum(corners[:, i], corners[:, j]) for i, j in sides], axis=1)
        hi = np.stack([np.maximum(corners[:, i], corners[:, j]) for i, j in sides], axis=1)
        _, edge_of_side = np.unique((lo * nv + hi).ravel(), return_inverse=True)
        edge_of_side = edge_of_side.reshape(k, 3)
        n_edge = int(edge_of_side.max()) + 1 if k else 0
        interior_base = n_corner + n_edge * (n - 1)
        per_interior = (n - 1) * (n - 2) // 2
        f = np.arange(k)
        ids = np.empty((k, len(lattice)), dtype=np.int64)
        q = 0
        for p, a in enumerate(lattice.tolist()):
            nonzero = [i for i in range(3) if a[i] > 0]
            if len(nonzero) == 1:
                ids[:, p] = corner_id[corners[:, nonzero[0]]]
            elif len(nonzero) == 2:
                i, j = nonzero
                side = sides.index((i, j)) if (i, j) in sides else sides.index((j, i))
                # steps from the lower-numbered end of the edge
                t = np.where(corners[:, i] < corners[:, j], a[j], a[i])
                ids[:, p] = n_corner + edge_of_side[:, side] * (n - 1) + t - 1
            else:
                ids[:, p] = interior_base + f * per_interior + q
                q += 1
        return ids.ravel(), interior_base + k * per_interior

    def tessellate_edges(self, n: Optional[int] = None, *, merge: bool = True,
                         values: Optional[np.ndarray] = None):
        """Subdivide every edge into n straight segments, for plotting.

        Returns ``(points, segments)`` (and interpolated ``values`` if
        given), sampled on the curved edges like :meth:`tessellate`.
        Segment ``s`` belongs to edge ``s // n``. ``n`` defaults to 1 for
        linear elements and 2*order otherwise.
        """
        nodes, index = self.edge_nodes, self.edge_node_index
        if nodes is None or index is None:
            raise ValueError("this model has no edge node layout")
        n = self._default_n(n)
        lam = _lattice(n, 1)
        weights = _basis(index, self.order, self.bezier, lam / n)
        points = np.einsum("pn,knd->kpd", weights, self.vertices[nodes])
        m = points.shape[0]
        sampled = None
        if values is not None:
            vals = np.asarray(values, dtype=float)
            sampled = np.einsum("pn,kn...->kp...", weights, vals[nodes])
            sampled = sampled.reshape((-1,) + sampled.shape[2:])
        seg = np.stack([np.arange(n), np.arange(1, n + 1)], axis=1)
        seg = (seg[None, :, :] + (np.arange(m) * (n + 1))[:, None, None]).reshape(-1, 2)
        points = points.reshape(-1, points.shape[-1])
        if merge:
            # endpoints by vertex, interior points by edge
            ends = np.stack([nodes[:, 0], nodes[:, -1]], axis=1)
            used = np.zeros(len(self.vertices), dtype=bool)
            used[ends.ravel()] = True
            end_id = np.cumsum(used) - 1
            n_end = int(used.sum())
            ids = np.empty((m, n + 1), dtype=np.int64)
            ids[:, 0], ids[:, n] = end_id[ends[:, 0]], end_id[ends[:, 1]]
            for t in range(1, n):
                ids[:, t] = n_end + np.arange(m) * (n - 1) + t - 1
            ids = ids.ravel()
            total = n_end + m * (n - 1)
            merged = np.empty((total, points.shape[1]))
            merged[ids] = points
            points = merged
            if sampled is not None:
                merged_values = np.empty((total,) + sampled.shape[1:])
                merged_values[ids] = sampled
                sampled = merged_values
            seg = ids[seg]
        if values is not None:
            return points, seg, sampled
        return points, seg

    # ---- native high-order cells ----------------------------------------------

    def native_cells(self, reorient: bool = True) -> Tuple[str, np.ndarray]:
        """Facets as high-order triangles in Gmsh/VTK node order.

        Returns ``(cell_type, cells)`` with meshio cell type names
        (``"triangle"``, ``"triangle6"``, ``"triangle10"``, ...) and cells
        indexing ``vertices``. With ``reorient``, cells follow the facet
        orientation of ``facets``.
        """
        faces, nodes, index = self._facet_data()
        p = self.order
        column = {tuple(int(x) for x in row): c for c, row in enumerate(index)}
        perm = [column[m] for m in _recursive_triangle_order(p)]
        cells = nodes[:, perm]
        if reorient:
            flip = _reversed_facets(faces, nodes, index, p)
            swapped = [column[(a, c, b)] for a, b, c in _recursive_triangle_order(p)]
            cells[flip] = nodes[flip][:, swapped]
        n_nodes = len(perm)
        return ("triangle" if p == 1 else f"triangle{n_nodes}"), cells

    def native_edge_cells(self) -> Tuple[str, np.ndarray]:
        """Edges as high-order lines in Gmsh/VTK order (ends, then interior)."""
        nodes = self.edge_nodes
        if nodes is None:
            raise ValueError("this model has no edge node layout")
        p = self.order
        perm = [0, p] + list(range(1, p))
        return ("line" if p == 1 else f"line{p + 1}"), nodes[:, perm]

    # ---- per-body surfaces ----------------------------------------------------------

    def body_surfaces(self, curved: str = "tessellate", n: Optional[int] = None,
                      *, cap: bool = False, project=None) -> Dict[int, BodySurface]:
        """The surface around each body, with outward normals.

        Facets with the body in front keep their orientation; facets with it
        behind are flipped. A film between two bodies appears in both.

        Bodies that Evolver closes off with a constraint (for instance a drop
        on a plane) have open boundary loops. ``cap=True`` closes each loop
        with a cap: rings of triangles between the loop and its centre, about
        as fine as the loop (``curved="tessellate"`` only). The cap is flat,
        which is exact for planar loops; ``project(points, loop_points)``,
        returning the points moved onto the surface the loop lies on, makes
        it follow a curved one (:meth:`Evolver.body_surfaces` does this for
        loops on a constraint). ``cap_ids`` tells the caps apart. Check
        ``watertight``.
        """
        faces, nodes, index = self._facet_data()
        bodies = self.facet_bodies
        assert bodies is not None
        ids = sorted({int(b) for b in bodies.ravel() if b > 0})
        out: Dict[int, BodySurface] = {}

        if curved == "tessellate":
            n_ = self._default_n(n)
            points, tris = self.tessellate(n_)
            per = n_ * n_
            facet_of = np.arange(len(tris)) // per
        elif curved == "native":
            cell_type, cells = self.native_cells()
            _, rev = self._native_reversal()
        else:
            raise ValueError("curved must be 'tessellate' or 'native'")

        for b in ids:
            front = bodies[:, 0] == b
            back = bodies[:, 1] == b
            if curved == "tessellate":
                sel = front[facet_of] | back[facet_of]
                body_tris = tris[sel].copy()
                flip = back[facet_of][sel] & ~front[facet_of][sel]
                body_tris[flip] = body_tris[flip][:, ::-1]
                fids = self.facet_ids[facet_of[sel]] if self.facet_ids is not None else facet_of[sel]
                used, body_tris = _compact(body_tris, len(points))
                body_points = points[used]
                loops = _boundary_loops(body_tris)
                open_loops = len(loops)
                cap_ids = np.zeros(len(body_tris), dtype=np.int64)
                if cap and loops:
                    body_points, body_tris, fids, cap_ids = _cap_loops(
                        body_points, body_tris, fids, loops, project)
                out[b] = BodySurface(b, body_points, body_tris, "triangle",
                                     np.asarray(fids), is_watertight(body_tris), open_loops,
                                     cap_ids=cap_ids,
                                     cap_constraints={k: None for k in range(1, len(loops) + 1)}
                                     if cap else {})
            else:
                if cap:
                    raise ValueError("cap=True needs curved='tessellate'")
                sel = front | back
                body_cells = cells[sel].copy()
                flip = back[sel] & ~front[sel]
                body_cells[flip] = body_cells[flip][:, rev]
                used, body_cells = _compact(body_cells, len(self.vertices))
                corners = body_cells[:, :3]
                fids = self.facet_ids[sel] if self.facet_ids is not None else np.flatnonzero(sel)
                out[b] = BodySurface(b, self.vertices[used], body_cells, cell_type,
                                     np.asarray(fids), is_watertight(corners),
                                     len(_boundary_loops(corners)), self.order, self.bezier)
        return out

    def _native_reversal(self):
        """Permutation of native cell columns that reverses orientation."""
        order = _recursive_triangle_order(self.order)
        position = {m: i for i, m in enumerate(order)}
        rev = [position[(a, c, b)] for a, b, c in order]
        return order, rev

    # ---- conversions ------------------------------------------------------------------

    def to_meshio(self, curved: str = "tessellate", n: Optional[int] = None,
                  flavor: str = "gmsh") -> "meshio.Mesh":
        """The whole surface as a meshio mesh.

        Soapfilm surfaces give triangles (or native high-order triangles with
        ``curved="native"``); the string model gives lines. Cell data:
        ``facet_id`` (or ``edge_id``), ``front_body`` and ``back_body``, and
        Gmsh tags (``gmsh:physical`` = front body, ``gmsh:geometrical`` =
        Evolver element id). ``flavor`` (``"gmsh"``, ``"vtk"``, ``"xdmf"``
        or ``"linear"``) picks native cell names for the target format.
        """
        import meshio
        if self.facets is not None and self.facet_nodes is not None:
            assert self.facet_bodies is not None and self.facet_ids is not None
            if curved == "tessellate":
                n_ = self._default_n(n)
                points, tris = self.tessellate(n_)
                facet_of = np.arange(len(tris)) // (n_ * n_)
                cells = [("triangle", tris)]
            elif curved == "native":
                _, native = self.native_cells()
                cell_type = native_cell_name("triangle", self.order, self.bezier, flavor)
                used, local = _compact(native, len(self.vertices))
                points = self.vertices[used]
                cells = [(cell_type, local)]
                facet_of = np.arange(len(native))
            else:
                raise ValueError("curved must be 'tessellate' or 'native'")
            ids = self.facet_ids[facet_of]
            front, back = self.facet_bodies[facet_of, 0], self.facet_bodies[facet_of, 1]
            cell_data = {"facet_id": [ids], "front_body": [front], "back_body": [back],
                         "gmsh:physical": [front], "gmsh:geometrical": [ids]}
        else:
            if curved == "tessellate":
                n_ = self._default_n(n)
                points, segs = self.tessellate_edges(n_)
                edge_of = np.arange(len(segs)) // n_
                cells = [("line", segs)]
            elif curved == "native":
                _, native = self.native_edge_cells()
                cell_type = native_cell_name("line", self.order, self.bezier, flavor)
                used, local = _compact(native, len(self.vertices))
                points = self.vertices[used]
                cells = [(cell_type, local)]
                edge_of = np.arange(len(native))
            else:
                raise ValueError("curved must be 'tessellate' or 'native'")
            ids = self.edge_ids[edge_of]
            cell_data = {"edge_id": [ids], "gmsh:physical": [np.ones_like(ids)],
                         "gmsh:geometrical": [ids]}
        return meshio.Mesh(_as_3d(points), cells, cell_data=cell_data)

    def to_pyvista(self, n: Optional[int] = None, *,
                   point_values: Optional[Dict[str, np.ndarray]] = None,
                   cell_values: Optional[Dict[str, np.ndarray]] = None) -> "pyvista.PolyData":
        """The surface as a PyVista PolyData, tessellated for curved elements.

        Soapfilm surfaces give triangles (cell data ``facet_id``,
        ``front_body``, ``back_body``); the string model gives lines (cell
        data ``edge_id``). ``point_values`` (one value per vertex row) are
        interpolated onto the tessellation; ``cell_values`` (one per facet,
        or per edge in the string model) are repeated for its sub-cells.
        """
        import pyvista as pv
        point_values = point_values or {}
        cell_values = cell_values or {}
        if self.facets is not None and self.facet_nodes is not None:
            n_ = self._default_n(n)
            return self._facets_to_pyvista(self._tessellation(n_), n_, point_values,
                                           cell_values)
        else:
            n_ = self._default_n(n)
            names = list(point_values)
            stacked = (np.column_stack([np.asarray(point_values[k], float) for k in names])
                       if names else None)
            if stacked is not None:
                points, segs, sampled = self.tessellate_edges(n_, values=stacked)
            else:
                points, segs = self.tessellate_edges(n_)
            lines = np.hstack([np.full((len(segs), 1), 2), segs]).ravel()
            poly = pv.PolyData(_as_3d(points), lines=lines)
            element_of = np.arange(len(segs)) // n_
            poly.cell_data["edge_id"] = self.edge_ids[element_of]
        if stacked is not None:
            for i, name in enumerate(names):
                poly.point_data[name] = sampled[:, i]
        for name, vals in cell_values.items():
            poly.cell_data[name] = np.asarray(vals)[element_of]
        return poly

    def _facets_to_pyvista(self, tess: "_Tessellation", n: int,
                           point_values: Dict[str, np.ndarray],
                           cell_values: Dict[str, np.ndarray]) -> "pyvista.PolyData":
        """to_pyvista() for facets, from this mesh's tessellation ``tess``."""
        import pyvista as pv
        assert self.facet_bodies is not None and self.facet_ids is not None
        tris = tess.triangles
        faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
        poly = pv.PolyData(_as_3d(tess.sample(self.vertices)), faces=faces)
        facet_of = np.arange(len(tris)) // (n * n)
        poly.cell_data["facet_id"] = self.facet_ids[facet_of]
        poly.cell_data["front_body"] = self.facet_bodies[facet_of, 0]
        poly.cell_data["back_body"] = self.facet_bodies[facet_of, 1]
        for name, vals in point_values.items():
            poly.point_data[name] = tess.sample(np.asarray(vals, float))
        for name, vals in cell_values.items():
            poly.cell_data[name] = np.asarray(vals)[facet_of]
        return poly

    def _same_facets(self, other: "Mesh") -> bool:
        """Whether ``other`` has the same facets, nodes and bodies (so the
        same tessellation)."""
        if other is self:
            return True
        if (self.order, self.bezier, len(self.vertices)) != (other.order, other.bezier,
                                                             len(other.vertices)):
            return False
        return all(a is not None and b is not None and np.array_equal(a, b)
                   for a, b in ((self.facets, other.facets),
                                (self.facet_nodes, other.facet_nodes),
                                (self.facet_ids, other.facet_ids),
                                (self.facet_bodies, other.facet_bodies)))


def _zip_rings(outer, s_outer, inner, s_inner):
    """Triangles between two closed rings of points, given as index lists
    with their positions along the ring (fractions in [0, 1), increasing):
    a strip that advances along whichever ring has the nearer next point.
    Oriented like the fans of :func:`_cap_loops` (the cap runs against the
    loop)."""
    A, B = list(outer) + [outer[0]], list(inner) + [inner[0]]
    sa, sb = list(s_outer) + [1.0], list(s_inner) + [1.0]
    a = b = 0
    tris = []
    while a < len(outer) or b < len(inner):
        if a < len(outer) and (b >= len(inner) or sa[a + 1] <= sb[b + 1]):
            tris.append((A[a + 1], A[a], B[b]))
            a += 1
        else:
            tris.append((B[b], B[b + 1], A[a]))
            b += 1
    return tris


def _cap_loops(points, tris, fids, loops, project=None):
    """Close each boundary loop with a cap: rings of points between the loop
    and its centre, fewer points on inner rings (spacing about that of the
    loop), the centre last. With ``project(points, loop_points)``, every new
    point is projected onto the surface the loop lies on; without, the cap
    is flat for planar loops. Returns points, triangles, facet ids (0 on caps)
    and cap ids (k on cap k, 0 elsewhere)."""
    points = [np.asarray(points, float)]
    count = len(points[0])
    new_tris = [np.asarray(tris)]
    new_fids = [np.asarray(fids)]
    cap_ids = [np.zeros(len(tris), dtype=np.int64)]
    base = points[0]
    for k, loop in enumerate(loops, start=1):
        P = base[loop]
        closed = np.vstack([P, P[:1]])
        edge = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        length = np.concatenate([[0.0], np.cumsum(edge)])
        s_loop = length[:-1] / length[-1]
        centre = P.mean(axis=0)
        if project is not None:
            centre = project(centre[None], P)[0]
        rings = max(1, int(round(np.linalg.norm(P - centre, axis=1).mean() / edge.mean())))
        outer, s_outer = list(loop), s_loop
        cap = []
        for r in range(1, rings):
            f = 1 - r / rings
            m = max(3, int(round(len(loop) * f)))
            t = np.arange(m) / m
            on_loop = np.column_stack([np.interp(t * length[-1], length, closed[:, d]) for d in range(3)])
            ring = centre + f * (on_loop - centre)
            if project is not None:
                ring = project(ring, P)
            points.append(ring)
            inner = list(range(count, count + m))
            count += m
            cap += _zip_rings(outer, s_outer, inner, t)
            outer, s_outer = inner, t
        points.append(centre[None])
        c = count
        count += 1
        cap += [(outer[(i + 1) % len(outer)], outer[i], c) for i in range(len(outer))]
        cap = np.array(cap, dtype=np.int64)
        new_tris.append(cap)
        new_fids.append(np.zeros(len(cap), dtype=np.int64))
        cap_ids.append(np.full(len(cap), k, dtype=np.int64))
    return (np.vstack(points), np.vstack(new_tris), np.concatenate(new_fids),
            np.concatenate(cap_ids))


@dataclass
class Bodies:
    """Body data, one entry per body."""

    ids: np.ndarray            # (b,) 1-based body numbers
    volume: np.ndarray         # (b,) current volume
    target_volume: np.ndarray  # (b,) prescribed volume, NaN if not fixed
    pressure: np.ndarray       # (b,) Lagrange multiplier for the volume
    fixed: np.ndarray          # (b,) bool, volume constraint active

    def _repr_html_(self) -> str:
        rows = [(_html.number(i), _html.number(v), _html.number(t), _html.number(p),
                 "fixed" if f else "")
                for i, v, t, p, f in zip(self.ids[:_html.MAX_ROWS], self.volume,
                                         self.target_volume, self.pressure, self.fixed)]
        return _html.table(rows, ["body", "volume", "target", "pressure", ""],
                           title="Bodies", total=len(self.ids))


@dataclass(frozen=True)
class Quantity:
    """A named quantity declared in the datafile."""

    name: str
    value: float
    target: float    # NaN unless the quantity is fixed
    modulus: float
    pressure: float  # Lagrange multiplier, for fixed quantities
    kind: str        # "energy", "fixed", "info" or "conserved"
