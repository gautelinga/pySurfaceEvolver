"""Build Surface Evolver datafiles from arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

__all__ = ["Body", "make_datafile"]


@dataclass
class Body:
    """A body for :func:`make_datafile`.

    ``faces`` are 0-based face rows. ``orientation`` (+1/-1 per face, default
    all +1) says whether each face's normal, by its vertex order, points out
    of the body (+1) or into it (-1). ``volume`` fixes the volume; ``None``
    leaves it free. ``density`` sets the body's weight density (for gravity).
    ``volconst`` is added to the computed volume (for example a solid inside
    the body; see :attr:`pysurfaceevolver.constraints.Constraint.volconst`).
    """

    faces: Sequence[int]
    volume: Optional[float] = None
    orientation: Optional[Sequence[int]] = None
    density: Optional[float] = None
    volconst: Optional[float] = None


def _num(x: float) -> str:
    """A float in a form Evolver reads back exactly."""
    return repr(float(x))


def _constraint_text(number: int, spec) -> str:
    if hasattr(spec, "text") and callable(spec.text):      # constraints.Constraint
        return f"constraint {number}\n{spec.text()}"
    spec = spec.strip()
    body = spec if ("formula" in spec.lower() or "function" in spec.lower()) \
        else f"formula: {spec}"
    return f"constraint {number}\n{body}\n"


def make_datafile(
    vertices,
    faces: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
    *,
    edges=None,
    bodies: Optional[Iterable[Union[Body, Mapping]]] = None,
    fixed=None,
    constraints: Optional[Mapping[int, Any]] = None,
    vertex_constraints: Optional[Mapping[int, Iterable]] = None,
    edge_fixed=None,
    edge_constraints: Optional[Mapping[int, Iterable]] = None,
    parameters: Optional[Mapping[str, float]] = None,
    string: bool = False,
    header: str = "",
    commands: str = "",
) -> str:
    """Write Surface Evolver datafile text for a surface given as arrays.

    Parameters
    ----------
    vertices:
        ``(n, sdim)`` coordinates. A ``space_dimension`` line is added when
        ``sdim != 3``.
    faces:
        Faces as vertex-row loops: an ``(k, 3)`` array of triangles, or a list
        of polygons. Edges and the faces' oriented edge loops are derived from
        them. Optional in the string model.
    edges:
        Extra ``(m, 2)`` edges, for example the curves of a string model.
    bodies:
        :class:`Body` objects (or dicts with the same keys).
    fixed:
        Vertex rows (or a boolean mask) to fix.
    constraints:
        ``{number: formula}``, e.g. ``{1: "z = 0"}``. A value that already
        contains ``formula:`` (or ``function:``) is copied as is, so energy
        and content integrals can be included. A value from
        :mod:`pysurfaceevolver.constraints` (planes, mirrors, spheres and
        cylinders with contact angles) writes its integrals itself.
    vertex_constraints:
        ``{number: vertex rows or mask}`` putting vertices on constraints.
    edge_fixed, edge_constraints:
        Same for edges, as rows into the derived edge list (see below). By
        default, a boundary edge (used by at most one face) is fixed, or on a
        constraint, when both of its vertices are.
    parameters:
        ``{name: value}``, written as ``parameter name = value``.
    string:
        Use the string model (curves; faces become 2D cells).
    header:
        Extra datafile lines placed before ``vertices`` (gravity, quantities,
        options, ...).
    commands:
        Commands placed after ``read`` at the end of the file.

    Edges are numbered in this order: explicit ``edges`` first, then edges
    derived from faces in order of first use.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2:
        raise ValueError("vertices must be an (n, sdim) array")
    nv, sdim = vertices.shape

    def mask_of(sel, size) -> np.ndarray:
        if sel is None:
            return np.zeros(size, dtype=bool)
        arr = np.asarray(sel)
        if arr.dtype == bool:
            if arr.shape != (size,):
                raise ValueError("boolean mask has the wrong length")
            return arr
        mask = np.zeros(size, dtype=bool)
        mask[arr.astype(int)] = True
        return mask

    # ---- edges -------------------------------------------------------------
    edge_list: List[tuple] = []
    edge_index: Dict[tuple, int] = {}

    def edge_id(u: int, v: int) -> int:
        """Signed 1-based id of edge u->v, creating it if needed."""
        if u == v:
            raise ValueError(f"degenerate edge at vertex {u}")
        key = (min(u, v), max(u, v))
        if key not in edge_index:
            edge_index[key] = len(edge_list)
            edge_list.append((u, v))
        i = edge_index[key]
        return (i + 1) if edge_list[i] == (u, v) else -(i + 1)

    if edges is not None:
        for u, v in np.asarray(edges, dtype=int).tolist():
            edge_id(u, v)

    face_loops: List[List[int]] = []
    if faces is not None:
        face_list = faces.tolist() if isinstance(faces, np.ndarray) else [list(f) for f in faces]
        for f in face_list:
            if len(f) < 3:
                raise ValueError("faces need at least 3 vertices")
            face_loops.append([edge_id(int(f[i]), int(f[(i + 1) % len(f)]))
                               for i in range(len(f))])
    if not edge_list:
        raise ValueError("no edges: give faces or edges")

    ne = len(edge_list)
    use_count = np.zeros(ne, dtype=int)
    for loop in face_loops:
        for e in loop:
            use_count[abs(e) - 1] += 1
    boundary_edge = use_count <= 1
    ev = np.array(edge_list)

    vfixed = mask_of(fixed, nv)
    vcons = {c: mask_of(sel, nv) for c, sel in (vertex_constraints or {}).items()}
    if edge_fixed is None:
        efixed = boundary_edge & vfixed[ev[:, 0]] & vfixed[ev[:, 1]]
    else:
        efixed = mask_of(edge_fixed, ne)
    if edge_constraints is None:
        econs = {c: boundary_edge & m[ev[:, 0]] & m[ev[:, 1]] for c, m in vcons.items()}
    else:
        econs = {c: mask_of(sel, ne) for c, sel in edge_constraints.items()}

    # ---- text --------------------------------------------------------------
    out: List[str] = ["// written by pysurfaceevolver.make_datafile\n"]
    if string:
        out.append("STRING\n")
    if sdim != 3:
        out.append(f"space_dimension {sdim}\n")
    for name, value in (parameters or {}).items():
        out.append(f"parameter {name} = {_num(value)}\n")
    if header:
        out.append(header.rstrip() + "\n")
    for number, spec in (constraints or {}).items():
        out.append(_constraint_text(int(number), spec))

    def attrs(i: int, fixed_mask, cons) -> str:
        parts = [f"constraint {c}" for c, m in cons.items() if m[i]]
        if fixed_mask[i]:
            parts.append("fixed")
        return (" " + " ".join(parts)) if parts else ""

    out.append("\nvertices\n")
    for i, x in enumerate(vertices):
        out.append(f"{i + 1} " + " ".join(_num(c) for c in x) + attrs(i, vfixed, vcons) + "\n")
    out.append("\nedges\n")
    for i, (u, v) in enumerate(edge_list):
        out.append(f"{i + 1} {u + 1} {v + 1}" + attrs(i, efixed, econs) + "\n")
    if face_loops:
        out.append("\nfaces\n")
        for i, loop in enumerate(face_loops):
            out.append(f"{i + 1} " + " ".join(str(e) for e in loop) + "\n")

    body_list = [b if isinstance(b, Body) else Body(**b) for b in (bodies or [])]
    if body_list:
        out.append("\nbodies\n")
        for i, b in enumerate(body_list):
            rows = [int(f) for f in b.faces]
            if any(r < 0 or r >= len(face_loops) for r in rows):
                raise ValueError(f"body {i + 1} refers to a face that doesn't exist")
            signs = [1] * len(rows) if b.orientation is None else [int(s) for s in b.orientation]
            if len(signs) != len(rows) or any(s not in (1, -1) for s in signs):
                raise ValueError(f"body {i + 1}: orientation needs one +1/-1 per face")
            line = f"{i + 1} " + " ".join(str(s * (r + 1)) for r, s in zip(rows, signs))
            if b.volume is not None:
                line += f" volume {_num(b.volume)}"
            if b.density is not None:
                line += f" density {_num(b.density)}"
            if b.volconst is not None:
                line += f" volconst {_num(b.volconst)}"
            out.append(line + "\n")

    if commands:
        out.append("\nread\n" + commands.rstrip() + "\n")
    return "".join(out)
