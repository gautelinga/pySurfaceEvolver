"""PyVista conversion, plots and the live view."""

import numpy as np
import pytest

from pysurfaceevolver._viz import surface_dataset

pv = pytest.importorskip("pyvista")



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


def test_live_view_fast_update_matches_rebuild(cube, can_render):
    cube.command("lagrange 3")
    view = cube.live_view("x + y", element="vertex", off_screen=True)
    cube.iterate(2, callback=view.update)
    assert view.fast_updates == 2
    fresh, _ = surface_dataset(cube, "x + y", element="vertex")
    np.testing.assert_allclose(view.dataset.points, fresh.points, atol=1e-13)
    np.testing.assert_allclose(view.dataset.point_data["x + y"], fresh.point_data["x + y"],
                               atol=1e-13)
    np.testing.assert_array_equal(view.dataset.faces, fresh.faces)
    view.close()


def test_live_view_context_manager(cube, can_render):
    with cube.live_view(off_screen=True) as view:
        view.update()
    assert view.updates == 1


def test_plot_draws_walls(load, can_render):
    import pyvista as pv
    from pysurfaceevolver import constraints as C
    ev = load("cube.fe")
    walls = [C.plane((0, 0, 1), 0.0), C.sphere((0, 0, 2), 0.5), C.cylinder((2, 0, 0), (0, 0, 1), 0.3),
             pv.Cube(center=(-2, 0, 0))]
    ev.plot(off_screen=True, walls=walls)
    view = ev.live_view(off_screen=True, walls=walls[:2])
    assert "evolver-wall-1" in view.plotter.actors
    with pytest.raises(TypeError):
        ev.plot(off_screen=True, walls=["z = 0"])


def test_constraint_geometry_to_pyvista():
    pytest.importorskip("pyvista")
    from pysurfaceevolver import constraints as C
    s = C.sphere((0, 0, 1), 0.5).to_pyvista()
    assert np.allclose(np.linalg.norm(s.points - [0, 0, 1], axis=1), 0.5)
    p = C.plane((0, 0, 2), 1.0).to_pyvista(bounds=(-1, 1, -1, 1, -1, 1))
    assert np.allclose(p.points[:, 2], 0.5)                # 2z = 1
    assert C.Constraint("x^2 = 1").to_pyvista() is None


def test_write_scenes_and_ply(load, tmp_path, can_render):
    import warnings
    ev = load("cube.fe")
    for ext in (".gltf", ".wrl"):
        path = tmp_path / ("cube" + ext)
        ev.write(path)
        assert path.stat().st_size > 100
    pytest.importorskip("meshio")
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # meshio warned about 64-bit integers
        ev.write(tmp_path / "cube.ply")


def test_torus_facets_are_shown_whole(load):
    pytest.importorskip("pyvista")
    ev = load("twointor.fe")
    m = ev.mesh()
    a, b = m.vertices[m.facets[:, 0]], m.vertices[m.facets[:, 1]]
    longest = ev.mesh_quality().edge_max
    assert np.linalg.norm(b - a, axis=1).max() > 2*longest      # wrapped across the cell
    dataset, _ = surface_dataset(ev, "area")
    p = dataset.points.reshape(-1, 3, 3)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        assert np.linalg.norm(p[:, j] - p[:, i], axis=1).max() <= longest*(1 + 1e-9)
    assert dataset.n_cells == len(m.facets) and "area" in dataset.cell_data
