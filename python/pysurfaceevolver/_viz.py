"""PyVista visualization: static plots and a live view that follows a run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import pyvista
    from ._evolver import Evolver

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
    cell_element = "facet" if mesh.faces is not None else "edge"
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
    return mesh.to_pyvista(n, point_values=point_values, cell_values=cell_values), name


class LiveView:
    """A PyVista window (or notebook widget) that follows the surface.

    Call :meth:`update` to redraw, for example after every few iterations
    with ``ev.iterate(100, callback=view.update, every=5)``. The surface is
    rebuilt on each update, so refinement and other topology changes show
    up too. The camera is kept.

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
    **mesh_kwargs:
        Passed to ``Plotter.add_mesh`` (``cmap``, ``show_edges``, ...).
    """

    def __init__(self, ev: "Evolver", scalars: Scalars = None, element: Optional[str] = None,
                 n: Optional[int] = None, plotter: Optional["pyvista.Plotter"] = None,
                 off_screen: bool = False, **mesh_kwargs: Any):
        pv = _import_pyvista()
        self.ev = ev
        self.scalars = scalars
        self.element = element
        self.n = n
        self.mesh_kwargs = {"show_edges": True, **mesh_kwargs}
        self.notebook = _in_notebook() and not off_screen
        self.plotter = plotter or pv.Plotter(off_screen=off_screen, notebook=self.notebook)
        self.dataset: Optional["pyvista.PolyData"] = None
        self.updates = 0
        self._draw(reset_camera=True)
        self._shown = False
        if not off_screen:
            self.show()

    def _draw(self, reset_camera: bool = False) -> None:
        dataset, name = surface_dataset(self.ev, self.scalars, self.element, self.n)
        self.dataset = dataset
        self.plotter.add_mesh(dataset, name="evolver-surface", scalars=name,
                              reset_camera=reset_camera, **self.mesh_kwargs)

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
