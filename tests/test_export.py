"""Mesh files (meshio), native high-order cells, and per-body surfaces."""

import numpy as np
import pytest

from pysurfaceevolver import is_watertight
from pysurfaceevolver._mesh import _recursive_triangle_order, native_cell_name

meshio = pytest.importorskip("meshio")


def triangle_area(points, triangles):
    a, b, c = (points[triangles[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()


def enclosed_volume(points, triangles):
    a, b, c = (points[triangles[:, i]] for i in range(3))
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6


@pytest.fixture
def lagrange_sphere(cube):
    cube.command("g 5; r; g 5; lagrange 3; g 2")
    return cube


# --- tessellation used for export ----------------------------------------------------

def test_merged_tessellation_is_watertight(lagrange_sphere):
    m = lagrange_sphere.mesh()
    points, triangles = m.tessellate(6)
    assert is_watertight(triangles)
    k = len(m.faces)
    # Euler: a sphere has V - E + F = 2
    assert len(points) - 3 * len(triangles) // 2 + len(triangles) == 2
    unmerged, _ = m.tessellate(6, merge=False)
    assert len(unmerged) == k * 28


def test_merged_linear_tessellation_is_the_mesh(cube):
    cube.command("r")
    m = cube.mesh()
    points, triangles = m.tessellate()
    assert len(points) == len(m.vertices)
    assert triangle_area(points, triangles) == pytest.approx(cube.eval("total_area"), rel=1e-14)


def test_interpolated_values_hit_the_nodes(lagrange_sphere):
    m = lagrange_sphere.mesh()
    values = m.vertices[:, 2] * 2 + 1
    points, _, sampled = m.tessellate(3, values=values)
    # at n == order the lattice points are the nodes themselves
    np.testing.assert_allclose(sampled, points[:, 2] * 2 + 1, atol=1e-12)


# --- whole-surface files ---------------------------------------------------------------

@pytest.mark.parametrize("ext", ["stl", "obj", "ply", "vtu", "vtk", "msh", "xdmf"])
def test_write_tessellated(lagrange_sphere, tmp_path, ext):
    if ext == "xdmf":
        pytest.importorskip("h5py")   # meshio writes XDMF data as HDF5
    path = tmp_path / f"surface.{ext}"
    lagrange_sphere.write(path, n=6)
    back = meshio.read(path)
    tris = np.vstack([c.data for c in back.cells if c.type == "triangle"])
    assert len(tris) == lagrange_sphere.counts["facets"] * 36
    area = triangle_area(back.points, tris)
    assert area == pytest.approx(lagrange_sphere.total_area, rel=5e-3)


def test_msh_means_gmsh(cube, tmp_path):
    path = tmp_path / "surface.msh"
    cube.write(path)
    assert path.read_bytes().startswith(b"$MeshFormat\n2.2")
    back = meshio.read(path, file_format="gmsh")
    assert "gmsh:physical" in back.cell_data


def test_cell_data(cube, tmp_path):
    cube.command("r")
    path = tmp_path / "surface.vtu"
    cube.write(path)
    back = meshio.read(path)
    m = cube.mesh()
    np.testing.assert_array_equal(back.cell_data["facet_id"][0], m.face_ids)
    np.testing.assert_array_equal(back.cell_data["front_body"][0], m.face_bodies[:, 0])


@pytest.mark.parametrize("model, gmsh_type, vtk_type", [
    ("quadratic", "triangle6", "triangle6"),
    ("lagrange 2", "triangle6", "triangle6"),
    ("lagrange 3", "triangle10", "VTK_LAGRANGE_TRIANGLE"),
    ("lagrange 5", "triangle21", "VTK_LAGRANGE_TRIANGLE"),
])
def test_write_native(cube, tmp_path, model, gmsh_type, vtk_type):
    cube.command("g 5; r; g 5; " + model)
    cube.write(tmp_path / "s.msh", curved="native")
    cube.write(tmp_path / "s.vtu", curved="native")
    msh = meshio.read(tmp_path / "s.msh", file_format="gmsh")
    vtu = meshio.read(tmp_path / "s.vtu")
    assert [c.type for c in msh.cells] == [gmsh_type]
    assert len(msh.cells[0].data) == cube.counts["facets"]
    assert vtu.cells[0].type == vtk_type


def test_native_unsupported_formats(lagrange_sphere, tmp_path):
    pytest.importorskip("h5py")
    with pytest.raises(ValueError, match="can't store"):
        lagrange_sphere.write(tmp_path / "s.stl", curved="native")
    with pytest.raises(ValueError, match="can't store"):
        lagrange_sphere.write(tmp_path / "s.xdmf", curved="native")
    lagrange_sphere.set_model("linear")
    lagrange_sphere.write(tmp_path / "s.stl", curved="native")   # flat cells are fine


def test_native_cells_area_by_vtk(lagrange_sphere, tmp_path):
    # VTK evaluates the Lagrange cells itself; a wrong node order would give
    # a very different area
    pv = pytest.importorskip("pyvista")
    lagrange_sphere.write(tmp_path / "s.vtu", curved="native")
    grid = pv.read(tmp_path / "s.vtu")
    area = grid.compute_cell_sizes(length=False, volume=False).cell_data["Area"].sum()
    assert area == pytest.approx(lagrange_sphere.total_area, rel=2e-2)


def test_node_order_matches_vtk():
    vtk = pytest.importorskip("vtk")
    for p in range(1, 7):
        n = (p + 1) * (p + 2) // 2
        cell = vtk.vtkLagrangeTriangle()
        cell.GetPointIds().SetNumberOfIds(n)
        cell.GetPoints().SetNumberOfPoints(n)
        cell.Initialize()
        pc = cell.GetParametricCoords()
        rs = np.array([pc[3 * i:3 * i + 2] for i in range(n)])
        ours = np.array([(a[1] / p, a[2] / p) for a in _recursive_triangle_order(p)])
        np.testing.assert_allclose(rs, ours, atol=1e-12)


def test_native_curved_area_by_gmsh(tmp_path, load):
    # Gmsh integrates the exported high-order elements exactly enough to
    # reproduce Evolver's own area
    gmsh = pytest.importorskip("gmsh")
    ev = load("cube.fe")
    ev.command("g 5; r; g 5; lagrange 5; g 2")
    ev.write(tmp_path / "s.msh", curved="native")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(tmp_path / "s.msh"))
        (etype,) = gmsh.model.mesh.getElementTypes(2)
        pts, weights = gmsh.model.mesh.getIntegrationPoints(etype, "Gauss12")
        _, det, _ = gmsh.model.mesh.getJacobians(etype, pts)
        nel = len(gmsh.model.mesh.getElementsByType(etype)[0])
        area = (np.array(det).reshape(nel, -1) * np.array(weights)).sum()
    finally:
        gmsh.finalize()
    # (a wrong node order would be off by far more than Evolver's own
    # quadrature error)
    assert area == pytest.approx(ev.total_area, rel=1e-6)


def test_native_cell_names():
    assert native_cell_name("triangle", 1, False, "linear") == "triangle"
    assert native_cell_name("triangle", 10, False, "gmsh") == "triangle66"
    assert native_cell_name("triangle", 4, True, "vtk") == "VTK_BEZIER_TRIANGLE"
    assert native_cell_name("line", 3, False, "vtk") == "line4"
    with pytest.raises(ValueError):
        native_cell_name("triangle", 11, False, "gmsh")
    with pytest.raises(ValueError):
        native_cell_name("triangle", 2, True, "gmsh")


def test_string_model_export(load, tmp_path):
    ev = load("knotty.fe")
    ev.write(tmp_path / "knot.vtu")
    back = meshio.read(tmp_path / "knot.vtu")
    assert back.cells[0].type == "line"
    assert len(back.cells[0].data) == ev.counts["edges"]


# --- per-body surfaces ---------------------------------------------------------------------

def test_body_surface_is_closed_and_outward(lagrange_sphere):
    surfaces = lagrange_sphere.body_surfaces(n=8)
    assert list(surfaces) == [1]
    s = surfaces[1]
    assert s.watertight and s.open_loops == 0
    volume = enclosed_volume(s.points, s.cells)
    assert volume == pytest.approx(lagrange_sphere.eval("body[1].volume"), rel=5e-3)
    assert volume > 0


def test_native_body_surface(lagrange_sphere, tmp_path):
    s = lagrange_sphere.body_surfaces(curved="native")[1]
    assert s.cell_type == "triangle10" and s.cells.shape[1] == 10
    assert s.watertight
    paths = lagrange_sphere.write_bodies(str(tmp_path / "body_{id}.msh"), curved="native")
    back = meshio.read(paths[1], file_format="gmsh")
    assert back.cells[0].type == "triangle10"


def test_body_on_constraint_needs_cap(load, tmp_path):
    ev = load("mound.fe")
    ev.command("g 10; r; g 10")
    s = ev.body_surfaces()[1]
    assert not s.watertight and s.open_loops == 1
    with pytest.raises(ValueError, match="cap=True"):
        ev.write_bodies(str(tmp_path / "b{id}.stl"))
    capped = ev.body_surfaces(cap=True)[1]
    assert capped.watertight
    # the base lies on the plane z = 0, so the fan cap is exact
    assert enclosed_volume(capped.points, capped.cells) == pytest.approx(
        ev.eval("body[1].volume"), rel=1e-10)
    paths = ev.write_bodies(str(tmp_path / "b{id}.stl"), cap=True)
    back = meshio.read(paths[1])
    assert is_watertight(back.cells[0].data)
    ev.write_bodies(str(tmp_path / "open{id}.stl"), require_watertight=False)


def test_two_bodies_share_a_film(load):
    ev = load("twointor.fe")   # torus model: two bodies
    ev.command("g 5")
    m = ev.mesh()
    front, back = m.face_bodies[:, 0], m.face_bodies[:, 1]
    shared = (front > 0) & (back > 0) & (front != back)
    assert shared.any()
    surfaces = m.body_surfaces("tessellate", 1)
    assert set(surfaces) == {1, 2}
    # each shared film appears once in each body, with opposite orientations
    for s in surfaces.values():
        assert len(set(s.facet_ids.tolist()) & set(m.face_ids[shared].tolist())) == shared.sum()


def test_is_watertight():
    tetra = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    assert is_watertight(tetra)
    assert not is_watertight(tetra[:3])
    flipped = tetra.copy()
    flipped[0] = flipped[0][::-1]
    assert not is_watertight(flipped)
