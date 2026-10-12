"""Surfaces from arrays and mesh files; export and plotting."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from ._build import Body, make_datafile
from ._engine import _Engine
from ._engine import surface as _surface
from ._mesh import BodySurface, is_watertight

_PROJECT_BATCH = 200            # points projected per Evolver command (body_surfaces caps)

if TYPE_CHECKING:
    from ._viz import LiveView


def _import_meshio():
    try:
        import meshio
    except ImportError:
        raise ImportError(
            "mesh file input/output needs meshio: pip install 'pysurfaceevolver[io]'") from None
    return meshio


# write(): file extensions written as PyVista scenes, and the exporter
_SCENE_FORMATS = {".gltf": "export_gltf", ".glb": "export_gltf", ".wrl": "export_vrml",
                  ".vrml": "export_vrml"}


def _target_format(path: str, file_format: Optional[str]) -> "tuple[Optional[str], str]":
    """meshio file format and native-cell flavor for writing path.

    meshio would read ".msh" as ANSYS; here it means Gmsh.
    """
    meshio = _import_meshio()
    ext = os.path.splitext(path)[1].lower()
    if file_format is None:
        if ext == ".msh":
            # Gmsh 2.2: meshio's 4.1 output lacks the $Entities section that
            # Gmsh itself needs to open the file
            file_format = "gmsh22"
        elif ext not in meshio.extension_to_filetypes:
            raise ValueError(f"unknown mesh file extension {ext!r}; pass file_format")
    fmt = file_format or meshio.extension_to_filetypes[ext][0]
    if fmt.startswith("gmsh"):
        flavor = "gmsh"
    elif fmt in ("vtu", "vtk"):
        flavor = "vtk"
    elif fmt == "xdmf":
        flavor = "xdmf"
    else:
        flavor = "linear"
    return file_format, flavor

def _vertex_index(vertices: np.ndarray, tol: float = 1e-9):
    """Lookup from (rounded) position to vertex row, for :func:`_rows_of`."""
    step = tol * (float(np.abs(vertices).max()) or 1.0)
    keys = np.round(vertices / step).astype(np.int64)
    return step, {tuple(k): i for i, k in enumerate(keys.tolist())}


def _rows_of(index, points: np.ndarray) -> np.ndarray:
    """Vertex row at the position of each point, or -1."""
    step, rows = index
    keys = np.round(np.asarray(points, float) / step).astype(np.int64)
    return np.array([rows.get(tuple(k), -1) for k in keys.tolist()], dtype=np.int64)


def _merge_points(points: np.ndarray, faces: np.ndarray, tolerance: float):
    """Merge points closer than tolerance * bounding box size; drop unused ones."""
    size = float(np.ptp(points, axis=0).max()) or 1.0
    keys = np.round(points / (tolerance * size)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    faces = inverse.ravel()[faces]
    points = points[first]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                  & (faces[:, 0] != faces[:, 2])]
    used, local = np.unique(faces, return_inverse=True)
    return points[used], local.reshape(faces.shape)


def _signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = (points[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def _boundary_vertices(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                    axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1])

class _Export(_Engine):
    """Building surfaces from arrays and mesh files; writing and plotting them."""

    # ------------------------------------------------------------------
    # Building surfaces
    def load_arrays(self, vertices, faces=None, **kwargs: Any) -> str:
        """Load a surface given as arrays; see :func:`make_datafile` for options.
        The walls among ``constraints`` (from :mod:`pysurfaceevolver.constraints`)
        are remembered for ``plot(walls=True)``."""
        output = self.load_string(make_datafile(vertices, faces, **kwargs))
        _surface.walls = {k: c for k, c in (kwargs.get("constraints") or {}).items()
                         if getattr(c, "shape", None) is not None}
        return output

    def load_mesh_file(self, path: Union[str, os.PathLike], *,
                       volume: Union[None, float, str] = None,
                       fix_boundary: bool = True, merge_tolerance: float = 1e-9,
                       **kwargs: Any) -> str:
        """Load a triangle surface from a mesh file (STL, OBJ, PLY, VTK, ...).

        Duplicate points (closer than ``merge_tolerance`` times the bounding
        box size) are merged, and quads are split into triangles.

        volume:
            ``None``: no body. A number: one body enclosed by the whole
            surface, with that target volume. ``"current"``: the same, with
            the volume the surface encloses now. The surface must be closed;
            its orientation is fixed up so the volume is positive.
        fix_boundary:
            Fix the vertices and edges on open boundaries, so a film spanning
            a wire keeps its wire.

        Other keyword arguments go to :func:`make_datafile`.
        """
        meshio = _import_meshio()
        mesh = meshio.read(os.fspath(path))
        tris = []
        for block in mesh.cells:
            if block.type == "triangle":
                tris.append(block.data)
            elif block.type == "quad":
                q = block.data
                tris.append(np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
        if not tris:
            raise ValueError(f"{os.fspath(path)} has no triangles or quads")
        points, faces = _merge_points(np.asarray(mesh.points, float), np.vstack(tris),
                                      merge_tolerance)
        if volume is not None:
            if not is_watertight(faces):
                raise ValueError("volume needs a closed, consistently oriented surface")
            current = _signed_volume(points, faces)
            sign = 1 if current >= 0 else -1
            target = abs(current) if volume == "current" else float(volume)
            kwargs.setdefault("bodies", [Body(faces=range(len(faces)), volume=target,
                                              orientation=[sign] * len(faces))])
        if fix_boundary and "fixed" not in kwargs:
            boundary = _boundary_vertices(faces)
            if len(boundary):
                kwargs["fixed"] = boundary
        return self.load_arrays(points, faces, **kwargs)

    # ------------------------------------------------------------------
    # Export and visualization

    def write(self, path: Union[str, os.PathLike], *, curved: str = "tessellate",
              n: Optional[int] = None, file_format: Optional[str] = None) -> None:
        """Write the surface to a mesh file with meshio (STL, OBJ, PLY, VTU,
        Gmsh .msh, XDMF, ...); the format follows the file extension.

        ``curved="tessellate"`` writes flat triangles sampled on curved
        elements (``n`` per facet edge); ``curved="native"`` writes the
        high-order elements themselves: Gmsh up to order 10, VTU/VTK any
        order, XDMF order 2; STL, OBJ, PLY etc. only take flat triangles.
        XDMF needs h5py. ``.msh`` means Gmsh format 2.2 (meshio alone would pick ANSYS, and
        its Gmsh 4.1 output can't be opened by Gmsh). See
        :meth:`Mesh.to_meshio` for cell data.

        Scenes for viewers and 3D printing software go through PyVista:
        ``.gltf``/``.glb`` and ``.wrl`` (VRML; flat triangles, no cell data).
        """
        ext = os.path.splitext(os.fspath(path))[1].lower()
        if file_format is None and ext in _SCENE_FORMATS:
            from ._viz import _import_pyvista
            pv = _import_pyvista()
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(self.mesh().to_pyvista(n), color="lightskyblue")
            getattr(plotter, _SCENE_FORMATS[ext])(os.fspath(path))
            plotter.close()
            return
        meshio = _import_meshio()
        file_format, flavor = _target_format(os.fspath(path), file_format)
        data = self.mesh().to_meshio(curved, n, flavor=flavor)
        meshio.write(os.fspath(path), data, file_format=file_format)

    def body_surfaces(self, *, curved: str = "tessellate", n: Optional[int] = None,
                      cap: bool = False) -> Dict[int, BodySurface]:
        """The closed surface around each body, with outward normals.

        See :meth:`Mesh.body_surfaces`. With ``cap=True``, an opening whose
        boundary vertices all lie on a constraint (a drop on a curved solid,
        say) is closed by a cap on that constraint: Evolver projects its
        points, so any constraint formula works (3D models; elsewhere caps
        are flat). ``cap_constraints`` says which constraint each cap is on
        (None: not on one; a flat cap).

        Projecting creates and dissolves temporary vertices: the surface is
        left as it was, but Evolver counts it as changed (the next Newton
        step rebuilds its matrix pattern, and later new elements may get
        other numbers than without the export).
        """
        mesh = self.mesh()
        if not cap or curved != "tessellate":
            return mesh.body_surfaces(curved, n, cap=cap)
        lookup: Dict[str, Any] = {}       # built on the first loop that needs it
        constraint_of: Dict[bytes, Optional[int]] = {}

        def constraint_for(loop_points) -> Optional[int]:
            key = np.asarray(loop_points).tobytes()
            if key not in constraint_of:
                if not lookup:
                    lookup["on"] = self._vertex_constraints()
                    lookup["index"] = _vertex_index(mesh.vertices)
                constraint_of[key] = self._loop_constraint(lookup["index"], lookup["on"],
                                                           loop_points)
            return constraint_of[key]

        def project(points, loop_points):
            if self.sdim != 3:
                return points
            con = constraint_for(loop_points)
            return points if con is None else self._project_points(points, con)

        surfaces = mesh.body_surfaces(curved, n, cap=True, project=project)
        for s in surfaces.values():   # the constraint of each cap, from its rim
            assert s.cap_ids is not None
            if not s.cap_constraints:
                continue
            rim = np.zeros(len(s.points), bool)
            rim[s.cells[s.cap_ids == 0].ravel()] = True
            for k in s.cap_constraints:
                cap_points = np.zeros(len(s.points), bool)
                cap_points[s.cells[s.cap_ids == k].ravel()] = True
                rim_points = s.points[cap_points & rim]
                s.cap_constraints[k] = (constraint_for(rim_points) if self.sdim == 3 else None)
        return surfaces

    def _vertex_constraints(self) -> Dict[int, np.ndarray]:
        """{constraint: mask of vertex rows on it}"""
        nums, _, dist, _ = self._constraint_gaps()
        return {int(k): np.isnan(dist[:, c]) for c, k in enumerate(nums)}

    @staticmethod
    def _loop_constraint(index, on: Dict[int, np.ndarray], loop_points) -> Optional[int]:
        """The constraint all vertices of a boundary loop lie on, if any."""
        if not on:
            return None
        rows = _rows_of(index, np.asarray(loop_points))
        rows = rows[rows >= 0]
        if len(rows) == 0:
            return None
        common = [k for k, mask in on.items() if mask[rows].all()]
        if len(common) > 1:
            warnings.warn(f"an opening lies on constraints {common} at once; "
                          f"its cap follows constraint {common[0]}", stacklevel=4)
        return common[0] if common else None

    def _project_points(self, points, constraint: int) -> np.ndarray:
        """Points moved onto a constraint by Evolver itself: a temporary
        vertex is put on the constraint (which projects it), read and
        dissolved."""
        lines = [f"pse_tmp_v := new_vertex({x!r}, {y!r}, {z!r}); "
                 f"set vertex[pse_tmp_v] constraint {constraint}; "
                 'printf "@pse %.17g %.17g %.17g\\n", vertex[pse_tmp_v].x, '
                 "vertex[pse_tmp_v].y, vertex[pse_tmp_v].z; dissolve vertex[pse_tmp_v]"
                 for x, y, z in np.asarray(points, float)[:, :3].tolist()]
        out = []
        for i in range(0, len(lines), _PROJECT_BATCH):
            text = self.command("; ".join(lines[i:i + _PROJECT_BATCH]))
            out += [line for line in text.splitlines() if line.startswith("@pse ")]
        return np.array([[float(v) for v in line.split()[1:]] for line in out])

    def write_bodies(self, pattern: str = "body_{id}.stl", *, curved: str = "tessellate",
                     n: Optional[int] = None, cap: bool = False,
                     require_watertight: bool = True,
                     file_format: Optional[str] = None) -> Dict[int, str]:
        """Write one closed surface per body, for volume meshing.

        ``pattern`` is formatted with the body ``id``. Raises ``ValueError``
        for a body whose surface isn't watertight (for example a drop that
        Evolver closes with a constraint plane) unless ``cap=True`` closes it
        or ``require_watertight=False``. Returns ``{body id: path}``.
        """
        meshio = _import_meshio()
        surfaces = self.body_surfaces(curved=curved, n=n, cap=cap)
        if require_watertight:
            open_ = [b for b, s in surfaces.items() if not s.watertight]
            if open_:
                raise ValueError(
                    f"bodies {open_} are not closed by facets (they likely end on a "
                    "constraint); use cap=True to close planar openings, or "
                    "require_watertight=False")
        written = {}
        for b, surface in surfaces.items():
            path = pattern.format(id=b)
            fmt, flavor = _target_format(path, file_format)
            meshio.write(path, surface.to_meshio(flavor), file_format=fmt)
            written[b] = path
        return written

    def plot(self, scalars: Union[None, str, np.ndarray] = None, *, element: Optional[str] = None,
             n: Optional[int] = None, off_screen: bool = False,
             screenshot: Optional[str] = None, mirror=None, walls=None,
             **kwargs: Any) -> Any:
        """Show the surface with PyVista.

        ``scalars`` colors it: an Evolver expression evaluated per facet
        (or per vertex with ``element="vertex"``), or an array with one value
        per facet or vertex row. Curved elements are tessellated with ``n``
        subdivisions. ``mirror`` adds mirror images, for a symmetric piece of
        a surface: a list of planes, each ``"x"`` (x = 0), ``("x", c)`` (x = c)
        or ``(normal, point)``, applied in order, each doubling what is there
        (an eighth of a cell: ``["z", "y", "x"]``). ``walls`` draws solids and
        walls with it: constraints from :mod:`pysurfaceevolver.constraints`
        (``C.sphere(...)``, ``C.plane(...)``; planes as translucent patches
        around the surface) or PyVista meshes; ``walls=True`` draws those
        given to :meth:`load_arrays`. Other keyword arguments go to
        ``Plotter.add_mesh``.
        """
        from ._viz import _import_pyvista, add_images, add_walls, surface_dataset
        pv = _import_pyvista()
        dataset, name = surface_dataset(self, scalars, element, n)
        plotter = pv.Plotter(off_screen=off_screen)
        add_images(plotter, dataset, mirror, "evolver-surface", scalars=name,
                   **{"show_edges": True, **kwargs})
        add_walls(plotter, self._walls(walls))
        return plotter.show(screenshot=screenshot)

    def live_view(self, scalars: Union[None, str, np.ndarray] = None, **kwargs: Any) -> "LiveView":
        """A PyVista view that redraws on :meth:`LiveView.update`; see :class:`LiveView`.

        Typical use: ``ev.iterate(200, callback=ev.live_view().update, every=10)``.
        """
        from ._viz import LiveView
        if "walls" in kwargs:
            kwargs["walls"] = self._walls(kwargs["walls"])
        return LiveView(self, scalars, **kwargs)

    def _walls(self, walls):
        """``walls`` for drawing: True means those given to load_arrays()."""
        return list(_surface.walls.values()) if walls is True else walls
