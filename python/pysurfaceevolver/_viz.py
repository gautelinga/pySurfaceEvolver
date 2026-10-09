"""PyVista visualization: static plots and a live view that follows a run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ._mesh import _as_3d

if TYPE_CHECKING:
    import pyvista
    from ._evolver import Evolver
    from ._mesh import Mesh, _Tessellation

__all__ = ["LiveView", "surface_dataset"]

Scalars = Union[None, str, np.ndarray]


def _import_pyvista():
    try:
        import pyvista
    except ImportError:
        raise ImportError(
            "visualization needs PyVista: pip install 'pysurfaceevolver[viz]'") from None
    return pyvista


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def surface_dataset(ev: "Evolver", scalars: Scalars = None, element: Optional[str] = None,
                    n: Optional[int] = None) -> "tuple[pyvista.PolyData, Optional[str]]":
    """The current surface as PyVista data, plus the name of the scalars.

    ``scalars`` is an Evolver expression evaluated with :meth:`Evolver.values`
    (on ``element``: "vertex", or "facet"/"edge" for the default), or an
    array with one value per vertex row or per facet (edge, in the string
    model). Vertex values are interpolated over curved elements.
    """
    mesh = ev.mesh()
    point_values, cell_values, name = _scalar_values(ev, mesh, scalars, element)
    return mesh.to_pyvista(n, point_values=point_values, cell_values=cell_values), name


def _scalar_values(ev: "Evolver", mesh: "Mesh", scalars: Scalars, element: Optional[str]
                   ) -> "tuple[dict, dict, Optional[str]]":
    """surface_dataset()'s scalars: point values, cell values, name."""
    cell_element = "facet" if mesh.facets is not None else "edge"
    allowed = ("vertex", "vertices", cell_element, cell_element + "s")
    if element is not None and element not in allowed:
        raise ValueError(f"scalars can be per vertex or per {cell_element}, not {element!r}")
    point_values, cell_values, name = {}, {}, None
    if scalars is not None:
        if isinstance(scalars, str):
            element = element or cell_element
            values = ev.values(element, scalars)
            name = scalars
        else:
            values = np.asarray(scalars)
            name = "scalars"
            if element is None:
                element = "vertex" if len(values) == len(mesh.vertices) else cell_element
        if element in ("vertex", "vertices"):
            point_values[name] = values
        elif element in (cell_element, cell_element + "s"):
            cell_values[name] = values
        else:
            raise ValueError(f"scalars can be per vertex or per {cell_element}, not {element!r}")
    return point_values, cell_values, name


def add_images(plotter: "pyvista.Plotter", dataset: "pyvista.PolyData", mirror,
               name: str, **kwargs: Any) -> None:
    """Add a dataset and its mirror images (one actor each, same data)."""
    from ._mesh import mirror_matrices
    mats = mirror_matrices(mirror or [])
    for k, m in enumerate(mats):
        extra = {} if k == 0 else {"show_scalar_bar": False}
        actor = plotter.add_mesh(dataset, name=name if k == 0 else f"{name}-{k}",
                                 **{**kwargs, **extra})
        if k:
            actor.user_matrix = m


def _same_cells(a: "pyvista.PolyData", b: "pyvista.PolyData") -> bool:
    """Whether two PolyData have identical connectivity."""
    if a.n_points != b.n_points or a.n_cells != b.n_cells:
        return False
    return bool(np.array_equal(a.faces, b.faces) and np.array_equal(a.lines, b.lines))


class LiveView:
    """A PyVista window (or notebook widget) that follows the surface.

    Call :meth:`update` to redraw, for example after every few iterations
    with ``ev.iterate(100, callback=view.update, every=5)``. While the
    facets are unchanged, an update only moves the points (and refreshes
    the scalars); after refinement or other topology changes the surface
    is rebuilt. The camera is kept.

    Parameters
    ----------
    ev:
        The Evolver to show.
    scalars, element, n:
        As for :meth:`Evolver.plot`; an expression is re-evaluated on every
        update.
    plotter:
        An existing ``pyvista.Plotter``; by default a new one is made.
    off_screen:
        Render without a window (for tests and image files).
    mirror:
        Planes to show mirror images in (see :meth:`Evolver.plot`).
    **mesh_kwargs:
        Passed to ``Plotter.add_mesh`` (``cmap``, ``show_edges``, ...).
    """

    def __init__(self, ev: "Evolver", scalars: Scalars = None, element: Optional[str] = None,
                 n: Optional[int] = None, plotter: Optional["pyvista.Plotter"] = None,
                 off_screen: bool = False, mirror=None, **mesh_kwargs: Any):
        pv = _import_pyvista()
        self.ev = ev
        self.mirror = mirror
        self.scalars = scalars
        self.element = element
        self.n = n
        self.mesh_kwargs = {"show_edges": True, **mesh_kwargs}
        self.notebook = _in_notebook() and not off_screen
        self.plotter = plotter or pv.Plotter(off_screen=off_screen, notebook=self.notebook)
        self.dataset: Optional["pyvista.PolyData"] = None
        self._mesh: Optional["Mesh"] = None            # what the dataset shows
        self._tess: Optional["_Tessellation"] = None   # its tessellation (facets)
        self._n = 1
        self.updates = 0
        self.fast_updates = 0   # updates that only moved points
        self._draw(reset_camera=True)
        self._shown = False
        if not off_screen:
            self.show()

    def _draw(self, reset_camera: bool = False) -> None:
        mesh = self.ev.mesh()
        if not reset_camera and self._move_points(mesh):
            self.fast_updates += 1
            return
        if mesh.facets is not None and mesh.facet_nodes is not None:
            # facets: keep the tessellation for later updates
            point_values, cell_values, name = _scalar_values(self.ev, mesh, self.scalars,
                                                             self.element)
            self._n = mesh._default_n(self.n)
            self._tess = mesh._tessellation(self._n)
            self._mesh = mesh
            self.dataset = mesh._facets_to_pyvista(self._tess, self._n, point_values,
                                                   cell_values)
            add_images(self.plotter, self.dataset, self.mirror, "evolver-surface",
                       scalars=name, reset_camera=reset_camera, **self.mesh_kwargs)
            return
        self._mesh = self._tess = None
        dataset, name = surface_dataset(self.ev, self.scalars, self.element, self.n)
        if self.dataset is not None and not reset_camera and _same_cells(self.dataset, dataset):
            # same connectivity: move the points and refresh values in place,
            # without rebuilding the actor
            self.dataset.points = dataset.points
            for key in dataset.point_data:
                self.dataset.point_data[key] = dataset.point_data[key]
            for key in dataset.cell_data:
                self.dataset.cell_data[key] = dataset.cell_data[key]
            self._update_range(name, dataset)
            self.fast_updates += 1
            return
        self.dataset = dataset
        add_images(self.plotter, dataset, self.mirror, "evolver-surface",
                   scalars=name, reset_camera=reset_camera, **self.mesh_kwargs)

    def _move_points(self, mesh: "Mesh") -> bool:
        """Update the dataset in place if the facets are unchanged."""
        if self.dataset is None or self._mesh is None or self._tess is None:
            return False
        if not mesh._same_facets(self._mesh):
            return False
        point_values, cell_values, name = _scalar_values(self.ev, mesh, self.scalars,
                                                         self.element)
        tess = self._tess
        self.dataset.points = _as_3d(tess.sample(mesh.vertices))
        for key, vals in point_values.items():
            self.dataset.point_data[key] = tess.sample(np.asarray(vals, float))
        if cell_values:
            facet_of = np.arange(len(tess.triangles)) // (self._n * self._n)
            for key, vals in cell_values.items():
                self.dataset.cell_data[key] = np.asarray(vals)[facet_of]
        self._update_range(name, self.dataset)
        self._mesh = mesh
        return True

    def _update_range(self, name: Optional[str], dataset: "pyvista.PolyData") -> None:
        if name is None:
            return
        values = dataset.point_data.get(name, dataset.cell_data.get(name))
        if values is not None and len(values):
            self.plotter.update_scalar_bar_range([float(np.min(values)),
                                                  float(np.max(values))])

    def show(self) -> Any:
        """Open the window (or display the notebook widget) without blocking."""
        if self._shown:
            return None
        self._shown = True
        if self.notebook:
            return self.plotter.show()
        return self.plotter.show(interactive_update=True, auto_close=False)

    def update(self, *_: Any) -> None:
        """Redraw with the current surface. Accepts and ignores callback args."""
        self._draw()
        self.updates += 1
        if self._shown and not self.notebook:
            self.plotter.update()
        else:
            self.plotter.render()

    def screenshot(self, filename: Optional[str] = None, **kwargs: Any):
        """Save (or return) an image of the current view."""
        return self.plotter.screenshot(filename, **kwargs)

    def close(self) -> None:
        self.plotter.close()

    def __enter__(self) -> "LiveView":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
