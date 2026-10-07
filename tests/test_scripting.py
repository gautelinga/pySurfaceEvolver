"""Scripting from Python: operations, element writes, building surfaces."""

import numpy as np
import pytest

from pysurfaceevolver import Body, Evolver, EvolverError, make_datafile

CUBE_VERTICES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                          [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
CUBE_FACES = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7], [3, 2, 1, 0]]


# --- operations ----------------------------------------------------------------

def test_iterate_matches_g(load):
    a = load("cube.fe")
    a.command("g 10")
    expected = a.total_energy
    b = load("cube.fe")
    result = b.iterate(10)
    assert result.energy.shape == result.area.shape == result.scale.shape == (10,)
    assert result.energy[-1] == expected
    assert result.output.count("area:") == 10
    assert (result.scale >= 0).all() and result.scale[0] > 0   # 0 once converged


def test_iterate_callback(cube):
    calls = []
    cube.iterate(7, callback=lambda ev, i: calls.append(i), every=3)
    assert calls == [3, 6, 7]


def test_relax_converges(cube):
    result = cube.relax(tol=1e-12)
    assert result.converged
    assert len(result.energy) < 1000
    last = result.energy[-5:]
    assert np.ptp(last) <= 1e-12 * max(1.0, abs(last[-1])) * 5
    # nothing left to gain from more gradient steps
    assert cube.iterate(5).energy[-1] == pytest.approx(result.energy[-1], rel=1e-11)


def test_relax_reports_no_convergence(cube):
    cube.refine(2)
    result = cube.relax(tol=1e-14, max_iter=3)
    assert result.converged is False and len(result.energy) == 3


def test_relax_with_hessian(cube):
    cube.iterate(5)
    cube.refine()
    gradient_only = cube.save()
    plain = cube.relax(tol=1e-9)
    cube.restore(gradient_only)
    polished = cube.relax(tol=1e-9, hessian=True)
    assert polished.converged and polished.hessian_steps >= 1
    assert cube.total_energy <= plain.energy[-1] + 1e-12


def test_relax_callback(cube):
    calls = []
    result = cube.relax(tol=1e-8, callback=lambda ev, i: calls.append(i), every=4)
    assert calls and all(i % 4 == 0 for i in calls[:-1])
    assert calls[-1] == len(result.energy)


def test_refine_and_model_operations(cube):
    cube.iterate(3)
    cube.refine()
    assert cube.counts["facets"] == 96
    cube.refine(2)
    assert cube.counts["facets"] == 96 * 16
    cube.equiangulate()
    cube.vertex_average()
    cube.iterate(2)
    cube.hessian()
    cube.hessian(seek=True)
    cube.set_model("quadratic")
    assert cube.model == "quadratic"
    cube.set_model("lagrange", 3)
    assert cube.model == "lagrange" and cube.lagrange_order == 3
    cube.set_model("linear")
    assert cube.model == "linear"
    with pytest.raises(ValueError):
        cube.set_model("lagrange")
    with pytest.raises(ValueError):
        cube.set_model("cubic")


def test_dump_round_trip(cube, tmp_path):
    cube.iterate(5)
    path = tmp_path / "dumped.fe"
    cube.dump(path)
    area = cube.eval("total_area")
    again = Evolver(str(path))
    assert again.eval("total_area") == pytest.approx(area, rel=1e-12)


# --- writing element data -----------------------------------------------------------

def test_set_values_round_trip(load):
    ev = load("mound.fe")
    n = ev.counts["vertices"]
    ev.define_attribute("vertex", "temperature")
    values = np.random.default_rng(0).random(n)
    ev.set_values("vertex", "temperature", values)
    np.testing.assert_array_equal(ev.values("vertex", "temperature"), values)


def test_set_values_scalar_and_where(load):
    ev = load("mound.fe")
    ev.set_values("facet", "tension", 2.0)
    assert set(ev.values("facet", "tension")) == {2.0}
    mask = np.zeros(ev.counts["facets"], dtype=bool)
    mask[:3] = True
    ev.set_values("facet", "tension", 0.5, where=mask)
    tension = ev.values("facet", "tension")
    assert (tension[:3] == 0.5).all() and (tension[3:] == 2.0).all()


def test_set_coordinates_through_set_values(cube):
    z = cube.values("vertex", "z") + 0.25
    cube.set_values("vertex", "z", z)
    np.testing.assert_array_equal(cube.mesh().vertices[:, 2], z)


def test_set_body_target(cube):
    cube.set_values("body", "target", 1.5)
    cube.iterate(10)
    assert cube.bodies().target_volume[0] == 1.5
    assert cube.eval("body[1].volume") == pytest.approx(1.5, rel=1e-6)


def test_fix_and_unfix(load):
    ev = load("mound.fe")
    before = ev.values("vertex", "fixed").astype(bool)
    top = ev.values("vertex", "z") > 0.3
    ev.fix("vertex", where=top)
    assert np.array_equal(ev.values("vertex", "fixed").astype(bool), before | top)
    ev.unfix("vertex", where=top)
    assert np.array_equal(ev.values("vertex", "fixed").astype(bool), before & ~top)
    # fixed vertices don't move
    ev.fix("vertex", where=top)
    z = ev.values("vertex", "z")
    ev.iterate(5)
    np.testing.assert_array_equal(ev.values("vertex", "z")[top], z[top])


def test_where_accepts_rows(load):
    ev = load("mound.fe")
    ev.set_values("facet", "tension", 0.75, where=[0, 2])
    tension = ev.values("facet", "tension")
    assert tension[0] == tension[2] == 0.75 and tension[1] == 1.0


def test_set_constraint(load):
    ev = load("mound.fe")
    on = ev.values("vertex", "on_constraint 1").astype(bool)
    target = ~on
    target[np.flatnonzero(target)[2:]] = False   # two vertices
    ev.set_constraint("vertex", 1, where=target)
    assert ev.values("vertex", "on_constraint 1").astype(bool)[target].all()
    ev.set_constraint("vertex", 1, where=target, on=False)
    assert not ev.values("vertex", "on_constraint 1").astype(bool)[target].any()


def test_bulk_writes_recalculate_once_and_keep_autorecalc(load):
    ev = load("mound.fe")
    ev.set_values("facet", "tension", 2.0)
    # energies are up to date after the batch
    assert ev.total_energy == pytest.approx(2 * ev.eval("total_area"), rel=1e-12)
    assert ev.eval("autorecalc") == 1
    ev.command("autorecalc off")
    ev.set_values("facet", "tension", 1.0)
    assert ev.eval("autorecalc") == 0       # the user's setting is kept


def test_failed_bulk_write_restores_autorecalc(cube):
    with pytest.raises(EvolverError):
        cube.set_values("vertex", "no_such_attribute", 1.0)
    assert cube.eval("autorecalc") == 1


def test_fast_coordinate_writes(load):
    ev = load("cube.fe")
    ev.refine()
    z = ev.values("vertex", "z") * 1.5
    ev.set_values("vertex", "z", z)
    np.testing.assert_array_equal(ev.values("vertex", "z"), z)
    assert ev.total_area == pytest.approx(ev.eval("sum(facet, area)"), rel=1e-12)


def test_fast_extra_attribute_writes(load):
    ev = load("mound.fe")
    ev.define_attribute("vertex", "weight")
    ev.define_attribute("facet", "label", "integer")
    n = ev.counts["vertices"]
    w = np.linspace(0, 1, n)
    ev.set_values("vertex", "weight", w)
    np.testing.assert_array_equal(ev.values("vertex", "weight"), w)
    labels = np.arange(ev.counts["facets"]) % 4
    ev.set_values("facet", "label", labels)
    np.testing.assert_array_equal(ev.values("facet", "label"), labels)
    mask = np.zeros(n, dtype=bool)
    mask[::3] = True
    ev.set_values("vertex", "weight", -1.0, where=mask)
    out = ev.values("vertex", "weight")
    assert (out[mask] == -1).all() and np.array_equal(out[~mask], w[~mask])


def test_fast_writes_match_set_command(load):
    # the same writes through Evolver's set command give the same surface
    ev = load("cube.fe")
    ev.refine()
    x = ev.values("vertex", "x") + 0.01 * ev.values("vertex", "y")
    ev.set_values("vertex", "x", x)
    fast = (ev.vertices, ev.total_energy)
    ev = load("cube.fe")
    ev.refine()
    ids = ev.values("vertex", "id").astype(int)
    ev.command("; ".join(f"set vertex[{i}] x {float(v)!r}" for i, v in zip(ids, x)))
    np.testing.assert_array_equal(ev.vertices, fast[0])
    assert ev.total_energy == fast[1]


def test_where_wrong_length(cube):
    with pytest.raises(ValueError, match="one entry per vertex"):
        cube.fix("vertex", where=np.ones(3, dtype=bool))


def test_bad_attribute_raises(cube):
    with pytest.raises(EvolverError):
        cube.set_values("vertex", "no_such_attribute", 1.0)


# --- building surfaces --------------------------------------------------------------

def test_cube_from_arrays_matches_datafile(load):
    reference = load("cube.fe")
    reference.command("g 10")
    expected = reference.total_energy
    ev = Evolver()
    ev.load_arrays(CUBE_VERTICES, CUBE_FACES, bodies=[Body(faces=range(6), volume=1)])
    assert ev.counts == {"vertices": 14, "edges": 36, "facets": 24, "bodies": 1}
    ev.command("g 10")
    assert ev.total_energy == expected


def test_body_orientation(load):
    # the same cube with every face listed backwards
    faces = [list(reversed(f)) for f in CUBE_FACES]
    ev = Evolver()
    ev.load_arrays(CUBE_VERTICES, faces,
                   bodies=[{"faces": range(6), "volume": 1, "orientation": [-1] * 6}])
    ev.iterate(5)
    assert ev.eval("body[1].volume") == pytest.approx(1.0, rel=1e-6)


def test_film_between_fixed_rings():
    k = 16
    theta = np.linspace(0, 2 * np.pi, k + 1)[:-1]
    ring = np.c_[np.cos(theta), np.sin(theta)]
    vertices = np.vstack([np.c_[ring, np.full(k, -0.3)], np.c_[ring, np.full(k, 0.3)]])
    faces = [[i, (i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    ev = Evolver()
    ev.load_arrays(vertices, faces, fixed=range(2 * k))
    # boundary edges between fixed vertices are fixed; the others aren't
    assert ev.values("edge", "fixed").sum() == 2 * k
    start = ev.total_area
    ev.refine(2)
    ev.iterate(50)
    # a catenoid-like neck forms: smaller area, narrower waist
    assert ev.total_area < start
    waist = np.hypot(*ev.mesh().vertices[:, :2].T).min()
    assert waist < 0.99


def test_constraints_and_parameters():
    # a hemispherical drop on the plane z = 0, with its contact line on the plane
    theta = np.linspace(0, 2 * np.pi, 9)[:-1]
    vertices = np.vstack([np.c_[np.cos(theta), np.sin(theta), np.zeros(8)], [[0, 0, 1]]])
    faces = [[i, (i + 1) % 8, 8] for i in range(8)]
    text = make_datafile(
        vertices, faces,
        constraints={1: "z = 0"},
        vertex_constraints={1: range(8)},
        parameters={"angle": 90},
        bodies=[Body(faces=range(8), volume=1.0)],
    )
    assert "parameter angle = 90.0" in text
    assert "constraint 1\nformula: z = 0" in text
    ev = Evolver()
    ev.load_string(text)
    assert dict(ev.parameters) == {"angle": 90.0}
    on_plane = ev.values("vertex", "on_constraint 1").astype(bool)
    assert on_plane.sum() == 8
    # boundary edges on the plane get the constraint too
    assert ev.values("edge", "on_constraint 1").sum() == 8
    ev.iterate(5)
    ev.refine()
    ev.iterate(5)
    z = ev.values("vertex", "z")
    assert np.abs(z[ev.values("vertex", "on_constraint 1").astype(bool)]).max() < 1e-12


def test_string_model_from_arrays():
    t = np.linspace(0, 2 * np.pi, 13)[:-1]
    points = np.c_[np.cos(t), np.sin(t)] * 0.5
    ev = Evolver()
    ev.load_arrays(points, [list(range(12))], string=True,
                   bodies=[{"faces": [0], "volume": 0.6}])
    assert ev.representation == "string" and ev.sdim == 2
    ev.iterate(20)
    assert ev.eval("body[1].volume") == pytest.approx(0.6, rel=1e-6)
    # close to a circle of that area
    assert ev.total_area == pytest.approx(2 * np.sqrt(np.pi * 0.6), rel=2e-2)


def test_header_and_commands():
    text = make_datafile(CUBE_VERTICES, CUBE_FACES,
                         bodies=[Body(faces=range(6), volume=1)],
                         header="gravity_constant 0", commands="my_macro := { g 2 }")
    assert "gravity_constant 0" in text and text.rstrip().endswith("my_macro := { g 2 }")
    ev = Evolver()
    ev.load_string(text)
    assert ev.command("my_macro").count("area:") == 2


def test_make_datafile_errors():
    with pytest.raises(ValueError):
        make_datafile(np.zeros(3))
    with pytest.raises(ValueError):
        make_datafile(CUBE_VERTICES)  # nothing but vertices
    with pytest.raises(ValueError):
        make_datafile(CUBE_VERTICES, CUBE_FACES, bodies=[Body(faces=[99])])
    with pytest.raises(ValueError):
        make_datafile(CUBE_VERTICES, [[0, 0, 1]])


def test_load_mesh_file(tmp_path):
    meshio = pytest.importorskip("meshio")
    # a closed box written with outward faces
    tris = []
    for f in CUBE_FACES:
        tris += [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]
    path = tmp_path / "box.stl"
    meshio.write(str(path), meshio.Mesh(CUBE_VERTICES, [("triangle", np.array(tris))]))
    ev = Evolver()
    ev.load_mesh_file(path, volume="current")
    assert ev.counts["vertices"] == 8        # STL duplicates merged
    assert ev.bodies().target_volume[0] == pytest.approx(1.0)
    ev.iterate(10)
    assert ev.eval("body[1].volume") == pytest.approx(1.0, rel=1e-6)

    # an open surface: boundary fixed by default
    open_tris = np.array(tris[:-2])         # drop the bottom face
    path = tmp_path / "open.ply"
    meshio.write(str(path), meshio.Mesh(CUBE_VERTICES, [("triangle", open_tris)]))
    ev.load_mesh_file(path)
    assert ev.values("vertex", "fixed").sum() == 4
    with pytest.raises(ValueError, match="closed"):
        ev.load_mesh_file(path, volume="current")
