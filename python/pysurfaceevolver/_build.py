"""Build Surface Evolver datafiles from arrays."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

__all__ = ["Body", "make_datafile"]


@dataclass
class Body:
    """A body for :func:`make_datafile`.

    ``volume`` fixes the volume; ``None`` leaves it free. ``faces`` are
    0-based face rows; ``None`` (default) means all faces. ``orientation``
    (+1/-1 per face) says whether each face's normal, by its vertex order,
    points out of the body (+1) or into it (-1). By default it is worked out
    when the faces close a surface (each edge on two of them): consistent
    across the edges and outward; otherwise all +1, and the body's starting
    volume is checked when it loads (see :meth:`Evolver.load`). ``density``
    sets the body's weight density (for gravity). ``volconst`` is added to
    the computed volume (for example a solid inside the body); by default
    it is the sum of :attr:`pysurfaceevolver.constraints.Constraint.volconst`
    over the walls from :mod:`pysurfaceevolver.constraints` that the body's
    vertices touch, and, with a single body, over the walls no vertex
    touches too (a solid immersed in the liquid).
    """

    volume: Optional[float] = None
    faces: Optional[Sequence[int]] = None
    orientation: Optional[Sequence[int]] = None
    density: Optional[float] = None
    volconst: Optional[float] = None


def _outward(rows: List[int], face_loops: List[List[int]], edge_list: List[tuple],
             vertices: np.ndarray) -> List[int]:
    """Signs (+1/-1) that orient the faces ``rows`` consistently and outward,
    if they close a surface (each edge on exactly two of them); else all +1."""
    on: Dict[int, List[int]] = {}
    for k, r in enumerate(rows):
        for e in face_loops[r]:
            on.setdefault(abs(e), []).append(k)
    if not rows or any(len(ks) != 2 for ks in on.values()):
        return [1]*len(rows)
    # neighbours across an edge agree when they run along it in opposite directions
    direction = [{abs(e): (1 if e > 0 else -1) for e in face_loops[r]} for r in rows]
    signs = [0]*len(rows)
    for seed in range(len(rows)):
        if signs[seed]:
            continue
        signs[seed], todo = 1, [seed]
        while todo:
            k = todo.pop()
            for e in direction[k]:
                for j in on[e]:
                    if j != k and not signs[j]:
                        signs[j] = -signs[k]*direction[k][e]*direction[j][e]
                        todo.append(j)
    # outward: the enclosed volume (divergence theorem over fan triangles) positive
    x = np.asarray(vertices, dtype=float)
    if x.shape[1] != 3:
        return signs
    volume = 0.0
    for k, r in enumerate(rows):
        loop = [edge_list[abs(e) - 1][0 if e > 0 else 1] for e in face_loops[r]]
        p = x[loop]
        for i in range(1, len(p) - 1):
            volume += signs[k]*np.dot(p[0], np.cross(p[i], p[i + 1]))/6
    return signs if volume >= 0 else [-s for s in signs]


def _num(x: float) -> str:
    """A float in a form Evolver reads back exactly."""
    return repr(float(x))


def _constraint_text(number: Union[int, str], spec) -> str:
    if hasattr(spec, "text") and callable(spec.text):      # constraints.Constraint
        return f"constraint {number}\n{spec.text()}"
    spec = spec.strip()
    if re.search(r"[^<>=!]==[^=]", spec):
        raise ValueError(f"constraint {number}: {spec!r} has '==': a formula is written "
                         "with one '=' (e.g. 'z = 0')")
    body = spec if ("formula" in spec.lower() or "function" in spec.lower()) \
        else f"formula: {spec}"
    return f"constraint {number}\n{body}\n"


def _key(k: Any, what: str) -> Union[int, str]:
    """A constraint or boundary key: a number, or a name for a constraint."""
    if isinstance(k, str):
        if not re.fullmatch(r"[A-Za-z_]\w*", k):
            raise ValueError(f"{what} name {k!r} is not an identifier")
        return k
    return int(k)


def _per_vertex(sets, nv: int) -> Dict[Any, List[int]]:
    """{constraint: rows} from one collection of constraints per vertex."""
    out: Dict[Any, List[int]] = {}
    for row, cons in enumerate(sets):
        for c in ([] if cons is None else [cons] if isinstance(cons, (int, str)) else cons):
            out.setdefault(c, []).append(row)
    return out


def make_datafile(
    vertices,
    faces: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
    *,
    edges=None,
    bodies: Optional[Iterable[Union[Body, Mapping]]] = None,
    fixed=None,
    constraints: Optional[Mapping[Any, Any]] = None,
    vertex_constraints: Union[None, Mapping[Any, Iterable], Sequence[Any]] = None,
    edge_fixed=None,
    edge_constraints: Optional[Mapping[int, Iterable]] = None,
    parameters: Optional[Mapping[str, float]] = None,
    gravity: Optional[float] = None,
    boundaries: Optional[Mapping[int, Any]] = None,
    vertex_boundaries: Optional[Mapping[int, Iterable]] = None,
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
        ``{number: formula}``, e.g. ``{1: "z = 0"}``; a key may also be a
        name (``{"floor": ...}``). A value that already contains
        ``formula:`` (or ``function:``) is copied as is, so energy and
        content integrals can be included. A value from
        :mod:`pysurfaceevolver.constraints` (planes, mirrors, spheres and
        cylinders with contact angles) writes its integrals itself, and
        its constants go where they belong: ``volconst`` to the bodies that
        touch it, the left-out wetting energy to :attr:`Evolver.energy`.
    vertex_constraints:
        ``{constraint: vertex rows or mask}`` putting vertices on
        constraints, or one entry per vertex: the constraint (or a set of
        them, or None) each vertex is on.
    edge_fixed, edge_constraints:
        Same for edges, as rows into the derived edge list (see below). By
        default, a boundary edge (used by at most one face) is fixed, or on a
        constraint, when both of its vertices are.
    parameters:
        ``{name: value}``, written as ``parameter name = value``.
    gravity:
        The gravitational constant (``gravity_constant``); bodies weigh
        their ``density`` times it, along -z.
    boundaries, vertex_boundaries:
        Curves vertices slide along (Evolver's ``boundary``; a curved wire
        for a film), ``{number: Boundary}`` from
        :func:`pysurfaceevolver.constraints.circle` or ``curve``, and
        ``{number: vertex rows or mask}``. A boundary edge (on one face at
        most) whose ends are both on a boundary is on it too. Vertices on a
        boundary are fixed (free, they would bunch up along it: that lowers
        the area of the discrete film); refining puts the new ones on the
        curve, so the wire stays round.
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

    def mask_of(sel, size, what="vertex") -> np.ndarray:
        if sel is None:
            return np.zeros(size, dtype=bool)
        arr = np.asarray(sel)
        if arr.dtype == bool:
            if arr.shape != (size,):
                raise ValueError(f"boolean mask has the wrong length ({arr.shape[0]}, "
                                 f"not {size})")
            return arr
        rows = arr.astype(int).ravel()
        if len(rows) and (rows.min() < 0 or rows.max() >= size):
            raise ValueError(f"{what} row {int(rows.max() if rows.max() >= size else rows.min())} "
                             f"doesn't exist: there are {size} (rows are 0-based)")
        mask = np.zeros(size, dtype=bool)
        mask[rows] = True
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

    def check_rows(rows, what: str) -> None:
        bad = [r for r in rows if not 0 <= r < nv]
        if bad:
            raise ValueError(f"{what} refer to vertex row {bad[0]}, but there are {nv} "
                             "vertices (rows are 0-based)")

    if edges is not None:
        edge_rows = np.asarray(edges, dtype=int).tolist()
        check_rows([r for e in edge_rows for r in e], "edges")
        for u, v in edge_rows:
            edge_id(u, v)

    face_loops: List[List[int]] = []
    if faces is not None:
        face_list = faces.tolist() if isinstance(faces, np.ndarray) else [list(f) for f in faces]
        check_rows([int(r) for f in face_list for r in f], "faces")
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

    # ---- constraints and boundaries ----------------------------------------
    cons = {_key(k, "constraint"): spec for k, spec in (constraints or {}).items()}
    if vertex_constraints is not None and not isinstance(vertex_constraints, Mapping):
        if len(vertex_constraints) != nv:
            raise ValueError("vertex_constraints as a sequence needs one entry per vertex")
        vertex_constraints = _per_vertex(vertex_constraints, nv)
    vcons = {_key(c, "constraint"): mask_of(sel, nv)
             for c, sel in (vertex_constraints or {}).items()}
    defined = set(cons) | {_key(k, "constraint") for k in
                           re.findall(r"^\s*constraint\s+(\w+)", header, re.MULTILINE | re.I)
                           if not k.isdigit()} | {int(k) for k in
                           re.findall(r"^\s*constraint\s+(\d+)", header, re.MULTILINE | re.I)}
    used = set(vcons) | set((edge_constraints or {}).keys())
    for c in used:
        if c not in defined:
            hint = difflib.get_close_matches(str(c), [str(k) for k in defined], 1)
            raise ValueError(f"vertices or edges are put on constraint {c!r}, which isn't "
                             "defined (in constraints= or header=)"
                             + (f"; did you mean {hint[0]!r}?" if hint else ""))
    declared = set(parameters or {}) | set(re.findall(r"^\s*parameter\s+(\w+)", header,
                                                       re.MULTILINE | re.I))
    for c, spec in cons.items():
        missing = [n for n in getattr(spec, "parameters", ()) if n not in declared]
        if missing:
            raise ValueError(f"constraint {c}: the contact angle uses {missing[0]!r}, which "
                             "isn't a declared parameter; add parameters="
                             f"{{{missing[0]!r}: value}}")
    bounds = {int(k): b for k, b in (boundaries or {}).items()}
    vbounds = {int(k): mask_of(sel, nv) for k, sel in (vertex_boundaries or {}).items()}
    for k in vbounds:
        if k not in bounds:
            raise ValueError(f"vertices are put on boundary {k}, which isn't in boundaries=")
    on_boundary = np.zeros(nv, dtype=int)
    for k, m in vbounds.items():
        if (on_boundary[m] != 0).any():
            raise ValueError("a vertex can be on one boundary only")
        on_boundary[m] = k
    params = np.zeros(nv)
    for k, m in vbounds.items():
        if m.any():
            params[m] = np.asarray(bounds[k].parameter(_xyz(vertices[m])), dtype=float)

    # vertices on a wire are fixed (as in Brakke's cat.fe): free, they would
    # bunch up along it, which lowers the discrete area; refining still puts
    # the new ones on the curve
    vfixed = mask_of(fixed, nv) | (on_boundary != 0)
    if edge_fixed is None:
        efixed = boundary_edge & vfixed[ev[:, 0]] & vfixed[ev[:, 1]]
    else:
        efixed = mask_of(edge_fixed, ne, "edge")
    if edge_constraints is None:
        econs = {c: boundary_edge & m[ev[:, 0]] & m[ev[:, 1]] for c, m in vcons.items()}
    else:
        econs = {_key(c, "constraint"): mask_of(sel, ne, "edge")
                 for c, sel in edge_constraints.items()}
    ebound = np.where(boundary_edge & (on_boundary[ev[:, 0]] != 0)
                      & (on_boundary[ev[:, 0]] == on_boundary[ev[:, 1]]),
                      on_boundary[ev[:, 0]], 0)

    # ---- bodies ------------------------------------------------------------
    body_list = [b if isinstance(b, Body) else Body(**b) for b in (bodies or [])]
    body_lines: List[str] = []
    # walls with a volume closure that no body touches (a solid inside the
    # liquid, the meniscus off it): the one body gets it; with several, say so
    closing = {c for c, spec in cons.items() if getattr(spec, "volconst", 0.0)}
    touched_any: set = set()
    for i, b in enumerate(body_list):
        rows = list(range(len(face_loops))) if b.faces is None else [int(f) for f in b.faces]
        if any(r < 0 or r >= len(face_loops) for r in rows):
            raise ValueError(f"body {i + 1} refers to a face that doesn't exist")
        signs = ([int(s) for s in b.orientation] if b.orientation is not None
                 else _outward(rows, face_loops, edge_list, vertices))
        if len(signs) != len(rows) or any(s not in (1, -1) for s in signs):
            raise ValueError(f"body {i + 1}: orientation needs one +1/-1 per face")
        corners = np.zeros(nv, dtype=bool)
        for r in rows:
            for e in face_loops[r]:
                corners[list(edge_list[abs(e) - 1])] = True
        touched = [c for c, m in vcons.items() if (m & corners).any()]
        touched_any |= set(touched)
        _check_walls(i + 1, touched, cons, vertices[corners])
        volconst = b.volconst
        if volconst is None:
            if len(body_list) == 1:
                touched = sorted(set(touched) | closing, key=str)
            volconst = sum(float(getattr(cons.get(c), "volconst", 0.0) or 0.0) for c in touched)
        line = f"{i + 1} " + " ".join(str(s * (r + 1)) for r, s in zip(rows, signs))
        if b.volume is not None:
            line += f" volume {_num(b.volume)}"
        if b.density is not None:
            line += f" density {_num(b.density)}"
        if volconst:
            line += f" volconst {_num(volconst)}"
        body_lines.append(line + "\n")

    stray = sorted(closing - touched_any, key=str)
    if stray and len(body_list) > 1 and any(b.volconst is None for b in body_list):
        warnings.warn(f"constraints {stray} close a volume (Constraint.volconst) but no body "
                      "touches them: with several bodies, give the right body's volconst "
                      "yourself", UserWarning, stacklevel=2)

    # ---- text --------------------------------------------------------------
    out: List[str] = ["// written by pysurfaceevolver.make_datafile\n"]
    if string:
        out.append("STRING\n")
    if sdim != 3:
        out.append(f"space_dimension {sdim}\n")
    for name, value in (parameters or {}).items():
        out.append(f"parameter {name} = {_num(value)}\n")
    if gravity is not None:
        out.append(f"gravity_constant {float(gravity)!r}\n")
    if header:
        out.append(header.rstrip() + "\n")
    for number, spec in cons.items():
        out.append(_constraint_text(number, spec))
    for number, b in bounds.items():
        out.append(f"boundary {number} parameters 1\n{b.text()}")

    def attrs(i: int, fixed_mask, on) -> str:
        parts = [f"constraint {c}" for c, m in on.items() if m[i]]
        if fixed_mask[i]:
            parts.append("fixed")
        return (" " + " ".join(parts)) if parts else ""

    out.append("\nvertices\n")
    for i, x in enumerate(vertices):
        if on_boundary[i]:
            out.append(f"{i + 1} {_num(params[i])} boundary {on_boundary[i]}"
                       + attrs(i, vfixed, vcons) + "\n")
        else:
            out.append(f"{i + 1} " + " ".join(_num(c) for c in x) + attrs(i, vfixed, vcons) + "\n")
    out.append("\nedges\n")
    for i, (u, v) in enumerate(edge_list):
        extra = f" boundary {ebound[i]}" if ebound[i] else ""
        out.append(f"{i + 1} {u + 1} {v + 1}" + extra + attrs(i, efixed, econs) + "\n")
    if face_loops:
        out.append("\nfaces\n")
        for i, loop in enumerate(face_loops):
            out.append(f"{i + 1} " + " ".join(str(e) for e in loop) + "\n")
    if body_lines:
        out.append("\nbodies\n")
        out += body_lines

    # the wetting energy the walls' line integrals leave out (Evolver.energy adds it)
    constant = [_num(c.energy_constant) for c in cons.values()
                if np.isfinite(getattr(c, "energy_constant", 0.0) or 0.0)
                and getattr(c, "energy_constant", 0.0)]
    constant += [c.energy_constant_text for c in cons.values()
                 if getattr(c, "energy_constant_text", None)]
    tail = commands.rstrip()
    if constant:
        tail = (f"function real {ENERGY_CONSTANT}() {{ return {' + '.join(constant)}; }}\n"
                + tail)
    if tail.strip():
        out.append("\nread\n" + tail.rstrip() + "\n")
    return "".join(out)


# the datafile function make_datafile defines for the energy its walls leave out
ENERGY_CONSTANT = "pyse_energy_constant"


def _xyz(x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), 3))
    out[:, :min(3, x.shape[1])] = x[:, :3]
    return out


def _check_walls(body: int, touched, cons, corners: np.ndarray) -> None:
    """Warn when a body lies on the solid side of a wetted plane it touches:
    the plane's normal must point into the liquid; reversed, the contact
    angle becomes 180 degrees minus the one asked for."""
    if len(corners) == 0:
        return
    centre = _xyz(corners).mean(axis=0)
    for c in touched:
        spec = cons.get(c)
        shape = getattr(spec, "shape", None)
        if shape is None or shape[0] != "plane" or getattr(spec, "energy", None) is None:
            continue
        _, normal, offset = shape
        size = float(np.ptp(_xyz(corners), axis=0).max()) or 1.0
        if float(np.dot(normal, centre)) - offset < -1e-6*size:
            warnings.warn(f"body {body} lies on the far side of constraint {c} (a plane whose "
                          "normal should point into the liquid): its contact angle acts as "
                          "180 degrees minus the one given. Flip the plane's normal.",
                          UserWarning, stacklevel=3)
