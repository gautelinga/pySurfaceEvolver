"""Mesh snapshots: high-order elements, tessellation, per-body surfaces,
and conversion to meshio and PyVista."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # optional dependencies
    import meshio
    import pyvista

__all__ = ["Mesh", "Bodies", "Quantity", "BodySurface"]


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


def _directed_edges(triangles: np.ndarray) -> np.ndarray:
    return np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])


def _boundary_loops(triangles: np.ndarray) -> List[List[int]]:
    """Open boundary loops of a triangle surface, as point index lists.

    Each loop runs along the boundary edges in the triangles' own direction.
    """
    edges = _directed_edges(triangles)
    present = {tuple(e) for e in edges.tolist()}
    boundary = [tuple(e) for e in edges.tolist() if (e[1], e[0]) not in present]
    following: Dict[int, List[int]] = {}
    for u, v in boundary:
        following.setdefault(u, []).append(v)
    loops = []
    used = set()
    for start_edge in boundary:
        if start_edge in used:
            continue
        loop = [start_edge[0]]
        u, v = start_edge
        while True:
            used.add((u, v))
            if v == loop[0]:
                break
            loop.append(v)
            nxt = [w for w in following.get(v, []) if (v, w) not in used]
            if not nxt:
                break  # not a simple loop; leave it open
            u, v = v, nxt[0]
        loops.append(loop)
    return loops


def is_watertight(triangles: np.ndarray) -> bool:
    """True if every edge is shared by exactly two triangles, with opposite
    directions (a closed, consistently oriented surface)."""
    edges = _directed_edges(np.asarray(triangles))
    if len(edges) == 0:
        return False
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if (counts != 1).any():
        return False
    present = {tuple(e) for e in unique.tolist()}
    return all((v, u) in present for u, v in present)


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

    def to_pyvista(self) -> "pyvista.PolyData":
        if self.cell_type != "triangle":
            raise ValueError("to_pyvista() needs flat triangles (curved='tessellate')")
        import pyvista as pv
        faces = np.hstack([np.full((len(self.cells), 1), 3), self.cells]).ravel()
        poly = pv.PolyData(_as_3d(self.points), faces=faces)
        poly.cell_data["facet_id"] = self.facet_ids
        return poly


@dataclass
class Mesh:
    """A snapshot of the surface geometry.

    ``edges`` and ``faces`` hold row indices into ``vertices``. The ``*_ids``
    arrays hold Evolver's own 1-based element numbers, as used in commands
    such as ``vertex[5].x``.

    In quadratic and Lagrange models, ``vertices`` also holds the extra nodes,
    ``faces`` and ``edges`` use the corner vertices only, and ``facet_nodes``
    and ``edge_nodes`` list every node of each element. Use
    :meth:`tessellate` for flat triangles that follow the curved facets.
    """

    vertices: np.ndarray       # (n, sdim) float64
    edges: np.ndarray          # (m, 2) int64
    faces: Optional[np.ndarray]  # (k, 3) int64; None outside the soapfilm model
    vertex_ids: np.ndarray     # (n,)
    edge_ids: np.ndarray       # (m,)
    face_ids: Optional[np.ndarray]     # (k,)
    face_bodies: Optional[np.ndarray]  # (k, 2) front/back body id, 0 = none
    fixed: np.ndarray          # (n,) bool, vertex has the FIXED attribute
    order: int = 1             # polynomial order of the elements
    bezier: bool = False       # nodes are Bezier control points, not on the surface
    edge_nodes: Optional[np.ndarray] = None        # (m, order+1) vertex rows
    edge_node_index: Optional[np.ndarray] = None   # (order+1, 2) barycentric
    facet_nodes: Optional[np.ndarray] = None       # (k, nodes) vertex rows
    facet_node_index: Optional[np.ndarray] = None  # (nodes, 3) barycentric

    # ---- tessellation -----------------------------------------------------

    def _default_n(self, n: Optional[int]) -> int:
        if n is None:
            n = 1 if self.order == 1 else 2 * self.order
        if n < 1:
            raise ValueError("n must be at least 1")
        return n

    def _facet_data(self):
        faces, nodes, index = self.faces, self.facet_nodes, self.facet_node_index
        if faces is None or nodes is None or index is None:
            raise ValueError("this needs facets (soapfilm representation)")
        return faces, nodes, index

    def tessellate(self, n: Optional[int] = None, *, merge: bool = True,
                   values: Optional[np.ndarray] = None):
        """Subdivide every facet into n*n flat triangles, for plotting or export.

        Points are sampled on the curved (quadratic or Lagrange) facets, so the
        triangles follow the actual surface. ``n`` defaults to 1 for linear
        elements and 2*order otherwise. Triangle ``t`` belongs to facet
        ``t // n**2``, and is oriented like that facet's ``faces`` row.

        With ``merge=True`` (the default), points shared by neighboring
        facets appear once, so the result is a connected surface. With
        ``merge=False``, each facet gets its own ``(n+1)(n+2)/2`` points.

        ``values``, one per vertex row (shape ``(n_vertices,)`` or
        ``(n_vertices, c)``), are interpolated like the coordinates and
        returned as a third item.

        In torus models, facets that cross the periodic boundary aren't
        unwrapped.
        """
        faces, nodes, index = self._facet_data()
        n = self._default_n(n)
        lattice = _lattice(n, 2)
        weights = _basis(index, self.order, self.bezier, lattice / n)   # (P, nodes)
        points = np.einsum("pn,knd->kpd", weights, self.vertices[nodes])  # (k, P, sdim)
        k, per_facet = points.shape[0], points.shape[1]
        sampled = None
        if values is not None:
            vals = np.asarray(values, dtype=float)
            if vals.shape[0] != len(self.vertices):
                raise ValueError("values needs one entry per vertex row")
            v = vals[nodes] if vals.ndim == 1 else vals[nodes]           # (k, nodes[, c])
            sampled = np.einsum("pn,kn...->kp...", weights, v)

        tris = _lattice_triangles(n)
        tris = np.broadcast_to(tris, (k,) + tris.shape).copy()
        flip = _reversed_facets(faces, nodes, index, self.order)
        tris[flip] = tris[flip][:, :, ::-1]
        tris += (np.arange(k) * per_facet)[:, None, None]
        points = points.reshape(-1, points.shape[-1])
        tris = tris.reshape(-1, 3)
        if sampled is not None:
            sampled = sampled.reshape((-1,) + sampled.shape[2:])

        if merge:
            keys = self._lattice_keys(nodes, index, n, lattice)
            _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                          return_inverse=True)
            inverse = inverse.ravel()
            # renumber in order of first appearance, to keep a stable layout
            order_ = np.argsort(first)
            rank = np.empty_like(order_)
            rank[order_] = np.arange(len(order_))
            points = points[first[order_]]
            if sampled is not None:
                sampled = sampled[first[order_]]
            tris = rank[inverse][tris]

        if values is not None:
            return points, tris, sampled
        return points, tris

    def _lattice_keys(self, nodes, index, n, lattice) -> np.ndarray:
        """A key per (facet, lattice point) identifying shared points exactly:
        corners by vertex row, edge points by edge and position, interior
        points by facet."""
        k = len(nodes)
        corners = np.stack([nodes[:, int(np.flatnonzero(index[:, i] == self.order)[0])]
                            for i in range(3)], axis=1)                  # (k, 3)
        P = len(lattice)
        keys = np.zeros((k, P, 4), dtype=np.int64)
        f = np.arange(k)
        for p, (a0, a1, a2) in enumerate(lattice):
            a = (a0, a1, a2)
            nonzero = [i for i in range(3) if a[i] > 0]
            if len(nonzero) == 1:                     # a corner
                keys[:, p, 0] = 0
                keys[:, p, 1] = corners[:, nonzero[0]]
            elif len(nonzero) == 2:                   # on an edge
                i, j = nonzero
                ri, rj = corners[:, i], corners[:, j]
                lo, hi = np.minimum(ri, rj), np.maximum(ri, rj)
                # distance from the lower-numbered end, in lattice steps
                t = np.where(ri < rj, a[j], a[i])
                keys[:, p, 0] = 1
                keys[:, p, 1], keys[:, p, 2], keys[:, p, 3] = lo, hi, t
            else:                                     # interior
                keys[:, p, 0] = 2
                keys[:, p, 1], keys[:, p, 2], keys[:, p, 3] = f, a1, a2
        return keys.reshape(-1, 4)

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
            tail, head = nodes[:, 0], nodes[:, -1]
            keys = np.zeros((m, n + 1, 4), dtype=np.int64)
            for t in range(n + 1):
                if t == 0 or t == n:
                    keys[:, t, 1] = tail if t == 0 else head
                else:
                    keys[:, t, 0] = 1
                    keys[:, t, 1], keys[:, t, 2] = np.arange(m), t
            _, first, inverse = np.unique(keys.reshape(-1, 4), axis=0,
                                          return_index=True, return_inverse=True)
            inverse = inverse.ravel()
            order_ = np.argsort(first)
            rank = np.empty_like(order_)
            rank[order_] = np.arange(len(order_))
            points = points[first[order_]]
            if sampled is not None:
                sampled = sampled[first[order_]]
            seg = rank[inverse][seg]
        if values is not None:
            return points, seg, sampled
        return points, seg

    # ---- native high-order cells ----------------------------------------------

    def native_cells(self, reorient: bool = True) -> Tuple[str, np.ndarray]:
        """Facets as high-order triangles in Gmsh/VTK node order.

        Returns ``(cell_type, cells)`` with meshio cell type names
        (``"triangle"``, ``"triangle6"``, ``"triangle10"``, ...) and cells
        indexing ``vertices``. With ``reorient``, cells follow the facet
        orientation of ``faces``.
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
                      *, cap: bool = False) -> Dict[int, BodySurface]:
        """The surface around each body, with outward normals.

        Facets with the body in front keep their orientation; facets with it
        behind are flipped. A film between two bodies appears in both.

        Bodies that Evolver closes off with a constraint (for instance a drop
        on a plane) have open boundary loops. ``cap=True`` closes each loop
        with a fan of triangles around its centroid, which is exact for
        planar loops (``curved="tessellate"`` only). Check ``watertight``.
        """
        faces, nodes, index = self._facet_data()
        bodies = self.face_bodies
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
                fids = self.face_ids[facet_of[sel]] if self.face_ids is not None else facet_of[sel]
                used, local = np.unique(body_tris, return_inverse=True)
                body_points = points[used]
                body_tris = local.reshape(-1, 3)
                loops = _boundary_loops(body_tris)
                open_loops = len(loops)
                if cap and loops:
                    body_points, body_tris, fids = _cap_loops(body_points, body_tris, fids, loops)
                out[b] = BodySurface(b, body_points, body_tris, "triangle",
                                     np.asarray(fids), is_watertight(body_tris), open_loops)
            else:
                if cap:
                    raise ValueError("cap=True needs curved='tessellate'")
                sel = front | back
                body_cells = cells[sel].copy()
                flip = back[sel] & ~front[sel]
                body_cells[flip] = body_cells[flip][:, rev]
                used, local = np.unique(body_cells, return_inverse=True)
                body_cells = local.reshape(body_cells.shape)
                corners = body_cells[:, :3]
                fids = self.face_ids[sel] if self.face_ids is not None else np.flatnonzero(sel)
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
        if self.faces is not None and self.facet_nodes is not None:
            assert self.face_bodies is not None and self.face_ids is not None
            if curved == "tessellate":
                n_ = self._default_n(n)
                points, tris = self.tessellate(n_)
                facet_of = np.arange(len(tris)) // (n_ * n_)
                cells = [("triangle", tris)]
            elif curved == "native":
                _, native = self.native_cells()
                cell_type = native_cell_name("triangle", self.order, self.bezier, flavor)
                used, local = np.unique(native, return_inverse=True)
                points = self.vertices[used]
                cells = [(cell_type, local.reshape(native.shape))]
                facet_of = np.arange(len(native))
            else:
                raise ValueError("curved must be 'tessellate' or 'native'")
            ids = self.face_ids[facet_of]
            front, back = self.face_bodies[facet_of, 0], self.face_bodies[facet_of, 1]
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
                used, local = np.unique(native, return_inverse=True)
                points = self.vertices[used]
                cells = [(cell_type, local.reshape(native.shape))]
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
        if self.faces is not None and self.facet_nodes is not None:
            assert self.face_bodies is not None and self.face_ids is not None
            n_ = self._default_n(n)
            names = list(point_values)
            stacked = (np.column_stack([np.asarray(point_values[k], float) for k in names])
                       if names else None)
            if stacked is not None:
                points, tris, sampled = self.tessellate(n_, values=stacked)
            else:
                points, tris = self.tessellate(n_)
            faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
            poly = pv.PolyData(_as_3d(points), faces=faces)
            facet_of = np.arange(len(tris)) // (n_ * n_)
            poly.cell_data["facet_id"] = self.face_ids[facet_of]
            poly.cell_data["front_body"] = self.face_bodies[facet_of, 0]
            poly.cell_data["back_body"] = self.face_bodies[facet_of, 1]
            element_of = facet_of
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


def _cap_loops(points, tris, fids, loops):
    """Close each boundary loop with a fan around its centroid."""
    points = list(points)
    new_tris = [tris]
    new_fids = [np.asarray(fids)]
    for loop in loops:
        centroid = np.mean([points[i] for i in loop], axis=0)
        c = len(points)
        points.append(centroid)
        # the surface runs along the loop as u -> v; the cap uses v -> u
        fan = np.array([(loop[(i + 1) % len(loop)], loop[i], c) for i in range(len(loop))])
        new_tris.append(fan)
        new_fids.append(np.zeros(len(fan), dtype=np.int64))
    return np.array(points), np.vstack(new_tris), np.concatenate(new_fids)


@dataclass
class Bodies:
    """Body data, one entry per body."""

    ids: np.ndarray            # (b,) 1-based body numbers
    volume: np.ndarray         # (b,) current volume
    target_volume: np.ndarray  # (b,) prescribed volume, NaN if not fixed
    pressure: np.ndarray       # (b,) Lagrange multiplier for the volume
    fixed: np.ndarray          # (b,) bool, volume constraint active


@dataclass(frozen=True)
class Quantity:
    """A named quantity declared in the datafile."""

    name: str
    value: float
    target: float    # NaN unless the quantity is fixed
    modulus: float
    pressure: float  # Lagrange multiplier, for fixed quantities
    kind: str        # "energy", "fixed", "info" or "conserved"
