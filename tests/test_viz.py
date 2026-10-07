"""PyVista conversion, plots and the live view."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")


@pytest.fixture(scope="module")
def can_render():
    try:
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(pv.Sphere())
        plotter.screenshot(return_img=True)
        plotter.close()
    except Exception as e:  # no display / no OpenGL
        pytest.skip(f"off-screen rendering unavailable: {e}")


def test_to_pyvista(cube):
    cube.command("g 5; r; g 5")
    poly = cube.mesh().to_pyvista()
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == cube.counts["facets"]
    assert poly.area == pytest.approx(cube.eval("total_area"), rel=1e-12)
    assert {"facet_id", "front_body", "back_body"} <= set(poly.cell_data)


def test_to_pyvista_curved_with_values(cube):
    cube.command("g 5; r; g 5; lagrange 3")
    m = cube.mesh()
    poly = m.to_pyvista(4, point_values={"z": m.vertices[:, 2]},
                        cell_values={"area": cube.values("facet", "area")})
    assert poly.n_cells == cube.counts["facets"] * 16
    np.testing.assert_allclose(poly.point_data["z"], poly.points[:, 2], atol=1e-12)
    assert len(poly.cell_data["area"]) == poly.n_cells


def test_string_model_to_pyvista(load):
    ev = load("knotty.fe")
    poly = ev.mesh().to_pyvista()
    assert poly.n_lines == ev.counts["edges"]


def test_two_dimensional_points_are_padded(load):
    ev = load("100grain.fe")
    assert ev.sdim == 2
    poly = ev.mesh().to_pyvista()
    assert poly.points.shape[1] == 3 and (poly.points[:, 2] == 0).all()


def test_plot(cube, tmp_path, can_render):
    path = tmp_path / "plot.png"
    cube.plot("area", off_screen=True, screenshot=str(path))
    assert path.stat().st_size > 1000
    cube.plot("x + y", element="vertex", off_screen=True, cmap="viridis")
    cube.plot(np.arange(cube.counts["facets"]), off_screen=True)


def test_plot_bad_element(cube):
    with pytest.raises(ValueError):
        cube.plot("x", element="body", off_screen=True)


def test_live_view_follows_the_run(cube, tmp_path, can_render):
    view = cube.live_view("area", off_screen=True)
    first = view.dataset.points.copy()
    cube.iterate(6, callback=view.update, every=3)
    assert view.updates == 2
    assert not np.array_equal(view.dataset.points, first)
    cube.refine()
    view.update()
    assert view.dataset.n_cells == cube.counts["facets"] == 96
    np.testing.assert_allclose(view.dataset.cell_data["area"], cube.values("facet", "area"))
    view.screenshot(str(tmp_path / "live.png"))
    view.close()


def test_live_view_moves_points_in_place(cube, can_render):
    view = cube.live_view("area", off_screen=True)
    dataset = view.dataset
    cube.iterate(4, callback=view.update)
    assert view.fast_updates == 4 and view.dataset is dataset
    np.testing.assert_allclose(view.dataset.points, cube.mesh().vertices)
    np.testing.assert_allclose(view.dataset.cell_data["area"], cube.values("facet", "area"))
    cube.refine()
    view.update()                       # topology changed: full rebuild
    assert view.fast_updates == 4 and view.dataset is not dataset
    view.close()


def test_live_view_context_manager(cube, can_render):
    with cube.live_view(off_screen=True) as view:
        view.update()
    assert view.updates == 1
