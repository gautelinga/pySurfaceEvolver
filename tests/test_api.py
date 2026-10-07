import math

import numpy as np
import pytest

from pysurfaceevolver import (
    Evolver,
    EvolverError,
    EvolverExit,
    EvolverWarning,
)

TETRA_NO_VOLUME = """\
vertices
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
edges
1 1 2
2 2 3
3 3 1
4 1 4
5 2 4
6 3 4
faces
1 1 5 -4
2 2 6 -5
3 3 4 -6
4 -3 -2 -1
bodies
1 1 2 3 4
"""


# --- loading ---------------------------------------------------------------

def test_load_reports_surface(cube):
    assert cube.datafile.endswith("cube.fe")
    assert cube.representation == "soapfilm"
    assert cube.model == "linear"
    assert cube.sdim == 3
    assert not cube.torus
    # Evolver adds a center vertex to each square face when it loads the cube
    assert cube.counts == {"vertices": 14, "edges": 36, "facets": 24, "bodies": 1}
    assert "cube.fe" in repr(cube)


def test_load_missing_file_raises(cube):
    with pytest.raises(FileNotFoundError):
        cube.load("no_such_file.fe")
    # the current surface is untouched
    assert cube.datafile.endswith("cube.fe")
    assert cube.counts["vertices"] == 14


def test_load_adds_fe_extension(load):
    ev = load("cube")
    assert ev.counts["facets"] == 24


def test_load_searches_evolverpath(tmp_path, monkeypatch, fe_dir):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVOLVERPATH", str(fe_dir))
    ev = Evolver("mound.fe")
    assert ev.datafile.endswith("mound.fe")


def test_load_string():
    ev = Evolver()
    ev.load_string(TETRA_NO_VOLUME)
    assert ev.counts == {"vertices": 4, "edges": 6, "facets": 4, "bodies": 1}


def test_load_invalid_datafile_raises():
    ev = Evolver()
    with pytest.raises(EvolverError):
        ev.load_string("vertices\n1 0 0 0\nedges\n1 1 7\n")


def test_load_runs_datafile_commands(cube):
    # gogo is defined after "read" in cube.fe
    out = cube.command("gogo")
    assert "area:" in out


def test_load_command(cube):
    cube.command('load "mound.fe"')
    assert cube.datafile.endswith("mound.fe")
    assert cube.counts["vertices"] == 18


def test_read_command(cube, tmp_path):
    script = tmp_path / "steps.cmd"
    script.write_text("g 2\n")
    out = cube.command(f'read "{script}"')
    assert out.count("area:") == 2


# --- commands and expressions ------------------------------------------------

def test_command_returns_output(cube):
    out = cube.command("g 3")
    lines = [l for l in out.splitlines() if "area:" in l]
    assert len(lines) == 3


def test_call_is_command(cube):
    assert "area:" in cube("g 1")


def test_quiet_by_default(cube, capsys):
    cube.command("g 1")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_echo_prints_output(load, capsys):
    ev = load("cube.fe", echo=True)
    capsys.readouterr()
    ev.command("g 1")
    assert "area:" in capsys.readouterr().out


def test_eval(cube):
    cube.command("g 5")
    assert cube.eval("total_area") == pytest.approx(cube.total_area)
    assert cube.eval("body[1].volume") == pytest.approx(1.0, rel=1e-6)
    assert cube.eval("2*pi") == pytest.approx(2 * math.pi)


def test_getitem_and_setitem(cube):
    cube["my_parameter"] = 2.5
    assert cube["my_parameter"] == 2.5
    assert cube.eval("my_parameter * 2") == 5.0


def test_total_energy_matches_area_without_other_energies(cube):
    cube.command("g 2")
    assert cube.total_energy == pytest.approx(cube.total_area)


# --- errors -------------------------------------------------------------------

def test_syntax_error_raises(cube):
    with pytest.raises(EvolverError) as info:
        cube.command("g 5 foo bar (")
    assert "syntax error" in str(info.value)


def test_runtime_error_has_errnum(cube):
    with pytest.raises(EvolverError) as info:
        cube.command("vertex[99999].x := 1")
    assert info.value.errnum == 1200
    assert "not valid" in str(info.value)


def test_eval_error_raises(cube):
    with pytest.raises(EvolverError):
        cube.eval("1 +")


def test_surface_survives_errors(cube):
    cube.command("g 3")
    area = cube.total_area
    for bad in ["g 5 foo (", "vertex[99999].x := 1", "nonsense_command_xyz"]:
        with pytest.raises(EvolverError):
            cube.command(bad)
    assert cube.eval("total_area") == area
    cube.command("r; g 1")
    assert cube.total_area < area


@pytest.mark.parametrize("cmd, code", [("q", 0), ("quit", 0), ("quit 3", 3)])
def test_quit_raises_exit_but_engine_survives(cube, cmd, code):
    with pytest.raises(EvolverExit) as info:
        cube.command(cmd)
    assert info.value.code == code
    assert cube.eval("body[1].volume") == pytest.approx(1.0)


# --- interactive input -------------------------------------------------------

def test_prompt_without_input_callback_warns(cube):
    cube.command("g 2")
    with pytest.warns(EvolverWarning, match="interactive input"):
        cube.command("hessian_menu")


def test_warnings_are_not_repeated(cube, recwarn):
    cube.command("g 2")
    with pytest.warns(EvolverWarning):
        cube.command("hessian_menu")
    recwarn.clear()
    Evolver("cube.fe")
    cube = Evolver("cube.fe")
    cube.command("g 1")
    assert not [w for w in recwarn if issubclass(w.category, EvolverWarning)]


def test_input_callback_answers_prompts(load):
    prompts = []
    answers = iter(["1", "0"])  # hessian_menu: fill Hessian, then leave

    def answer(prompt):
        prompts.append(prompt)
        return next(answers, None)

    ev = load("cube.fe", input=answer)
    ev.command("g 2; hessian_menu")
    assert len(prompts) == 2
    assert all(p.startswith("Choice") for p in prompts)


def test_reentrant_call_is_rejected(load):
    errors = []

    def answer(prompt):
        try:
            ev.command("g 1")
        except RuntimeError as e:
            errors.append(str(e))
        return None

    ev = load("cube.fe", input=answer)
    ev.command("g 1; hessian_menu")
    assert errors and "busy" in errors[0]
    # the engine is fine afterwards
    ev.command("g 1")


# --- mesh access ---------------------------------------------------------------

def test_mesh_shapes(cube):
    cube.command("r")
    m = cube.mesh()
    n, k = len(m.vertices), len(m.faces)
    assert m.vertices.shape == (n, 3) and m.vertices.dtype == np.float64
    assert m.edges.shape == (cube.counts["edges"], 2)
    assert m.faces.shape == (cube.counts["facets"], 3)
    assert m.vertex_ids.shape == (n,)
    assert m.face_ids.shape == (k,)
    assert m.face_bodies.shape == (k, 2)
    assert m.fixed.dtype == bool


def test_mesh_indices_are_consistent(cube):
    cube.command("r; g 2")
    m = cube.mesh()
    n = len(m.vertices)
    assert m.edges.min() >= 0 and m.edges.max() < n
    assert m.faces.min() >= 0 and m.faces.max() < n
    # every triangle has three distinct corners
    assert all(len(set(f)) == 3 for f in m.faces.tolist())
    # Evolver ids are 1-based and unique
    assert len(set(m.vertex_ids.tolist())) == n
    assert m.vertex_ids.min() >= 1
    # every face of the cube bounds body 1 on exactly one side
    assert set(map(tuple, m.face_bodies.tolist())) <= {(1, 0), (0, 1)}


def test_mesh_area_matches_evolver(cube):
    cube.command("r; g 3")
    m = cube.mesh()
    a, b, c = (m.vertices[m.faces[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    assert area == pytest.approx(cube.eval("total_area"), rel=1e-12)


def test_mesh_volume_matches_evolver(cube):
    cube.command("r; g 3")
    m = cube.mesh()
    a, b, c = (m.vertices[m.faces[:, i]] for i in range(3))
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    # faces with body 1 on the back side are oriented the other way
    sign = np.where(m.face_bodies[:, 0] == 1, 1.0, -1.0)
    assert abs((sign * signed).sum()) == pytest.approx(cube.eval("body[1].volume"), rel=1e-12)


def test_vertex_ids_match_evolver_numbering(cube):
    m = cube.mesh()
    for row in [0, 5, len(m.vertices) - 1]:
        vid = int(m.vertex_ids[row])
        for axis, name in enumerate("xyz"):
            assert cube.eval(f"vertex[{vid}].{name}") == m.vertices[row, axis]


def test_fixed_vertices(load):
    ev = load("mound.fe")
    fixed = ev.mesh().fixed
    assert fixed.any() and not fixed.all()
    m = ev.mesh()
    for row in np.flatnonzero(fixed)[:3]:
        vid = int(m.vertex_ids[row])
        assert ev.eval(f"vertex[{vid}].fixed") == 1


def test_mesh_string_model_has_no_faces(load):
    ev = load("knotty.fe")
    assert ev.representation == "string"
    m = ev.mesh()
    assert m.faces is None and m.face_ids is None and m.face_bodies is None
    assert m.edges.shape == (ev.counts["edges"], 2)


@pytest.mark.parametrize("datafile, command", [
    ("cube.fe", "r"), ("cube.fe", "r; lagrange 3"), ("cube.fe", "quadratic"),
    ("knotty.fe", "g 1"), ("100grain.fe", "g 1"), ("simplex3.fe", "g 1"),
])
def test_one_call_mesh_matches_parts(load, datafile, command):
    from pysurfaceevolver import _core
    ev = load(datafile)
    ev.command(command)
    m = ev.mesh()
    kw = dict(out=None, input=None, sigint=False)
    xyz, vids, fixed = _core.vertices(**kw).data
    edges, eids = _core.edges(**kw).data
    np.testing.assert_array_equal(m.vertices, xyz)
    np.testing.assert_array_equal(m.vertex_ids, vids)
    np.testing.assert_array_equal(m.edges, edges)
    np.testing.assert_array_equal(m.edge_ids, eids)
    if ev.representation == "soapfilm":
        faces, fids, fbodies = _core.facets(**kw).data
        np.testing.assert_array_equal(m.faces, faces)
        np.testing.assert_array_equal(m.face_bodies, fbodies)
        nodes = _core.element_nodes(_core.FACET, **kw).data
        np.testing.assert_array_equal(m.facet_nodes, nodes[0])
    else:
        assert m.faces is None


def test_mesh_is_cached_until_the_surface_changes(cube):
    first = cube.mesh()
    assert cube.mesh() is first
    cube.eval("total_area")                 # evaluating doesn't change it
    cube.values("vertex", "x")
    cube.bodies()
    assert cube.mesh() is first
    cube.iterate(1)
    second = cube.mesh()
    assert second is not first
    assert not np.array_equal(second.vertices, first.vertices)
    cube.vertices = second.vertices * 2     # coordinate writes count too
    assert cube.mesh() is not second


def test_failed_command_invalidates_the_cache(cube):
    first = cube.mesh()
    with pytest.raises(EvolverError):
        cube.command("g 1; nonsense_command_xyz")
    assert cube.mesh() is not first


def test_cached_mesh_is_read_only(cube):
    m = cube.mesh()
    with pytest.raises(ValueError):
        m.vertices[0, 0] = 5.0
    m.vertices.copy()[0, 0] = 5.0           # copies are fine


def test_mesh_returns_copies(cube):
    v = cube.vertices
    v[:] = 0.0
    assert np.abs(cube.vertices).sum() > 0


# --- bodies ----------------------------------------------------------------------

def test_bodies_fixed_volume(cube):
    cube.command("g 5")
    b = cube.bodies()
    assert b.ids.tolist() == [1]
    assert b.fixed.tolist() == [True]
    assert b.target_volume[0] == 1.0
    assert b.volume[0] == pytest.approx(1.0, rel=1e-9)
    assert b.pressure[0] > 0  # surface tension pushes inward


def test_bodies_free_volume():
    ev = Evolver()
    ev.load_string(TETRA_NO_VOLUME)
    b = ev.bodies()
    assert b.fixed.tolist() == [False]
    assert math.isnan(b.target_volume[0])
    assert b.volume[0] == pytest.approx(1 / 6)


# --- setting vertex coordinates ----------------------------------------------------

def test_set_vertices_scales_area_and_volume(cube):
    cube.command("r; g 3")
    area = cube.eval("total_area")
    volume = cube.eval("body[1].volume")
    cube.vertices = cube.vertices * 2
    assert cube.total_area == pytest.approx(4 * area, rel=1e-12)
    assert cube.eval("body[1].volume") == pytest.approx(8 * volume, rel=1e-12)


def test_set_vertices_round_trip(cube):
    v = cube.vertices + 0.01
    cube.vertices = v
    np.testing.assert_array_equal(cube.vertices, v)


def test_set_vertices_wrong_count_raises(cube):
    with pytest.raises(EvolverError, match="Expected"):
        cube.vertices = np.zeros((3, 3))


def test_set_vertices_wrong_ndim_raises(cube):
    with pytest.raises(ValueError):
        cube.vertices = np.zeros(9)


# --- instances and process-level behavior -------------------------------------------

def test_handles_share_the_engine(load):
    first = load("cube.fe")
    second = load("mound.fe")
    # both handles keep working, and see the surface loaded last
    assert first.datafile.endswith("mound.fe")
    assert first.counts == second.counts
    first.iterate(2)
    assert second.eval("total_energy") == first.total_energy


def test_handles_keep_their_own_options(load, capsys):
    quiet = load("cube.fe")
    loud = Evolver(echo=True)
    capsys.readouterr()
    quiet.command("g 1")
    assert capsys.readouterr().out == ""
    loud.command("g 1")
    assert "area:" in capsys.readouterr().out


def test_save_and_restore_are_exact(load):
    ev = load("cube.fe")
    ev.command("g 5; r; g 5")
    snapshot = ev.save()
    vertices, energy = ev.vertices, ev.eval("total_energy")
    ev.refine()
    ev.iterate(5)
    ev.load("mound.fe")
    ev.restore(snapshot)
    assert ev.datafile.endswith("cube.fe")
    np.testing.assert_array_equal(ev.vertices, vertices)
    assert ev.eval("total_energy") == energy
    # and it carries on exactly like the original would have
    ev.iterate(3)
    reference = load("cube.fe")
    reference.command("g 5; r; g 5; g 3")
    np.testing.assert_array_equal(ev.vertices, reference.vertices)


def test_snapshot_keeps_lagrange_model_and_parameters(load, tmp_path):
    ev = load("mound.fe")
    ev.parameters["angle"] = 60
    ev.command("g 5; lagrange 2; g 2")
    snapshot = ev.save()
    snapshot.write(tmp_path / "saved.fe")
    ev.load("cube.fe")
    ev.restore(snapshot)
    assert ev.model == "lagrange" and ev.lagrange_order == 2
    assert ev.parameters["angle"] == 60
    assert Evolver(str(tmp_path / "saved.fe")).model == "lagrange"


def test_sigint_interrupts_long_run(run_python):
    result = run_python("""
        import os, signal, threading
        from pysurfaceevolver import Evolver
        ev = Evolver("cube.fe")
        ev.command("r; r; r")
        threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT)).start()
        try:
            ev.command("g 10000000")
        except KeyboardInterrupt:
            print("INTERRUPTED")
        print("VOLUME", ev.eval("body[1].volume"))
        print("HANDLER", signal.getsignal(signal.SIGINT) is signal.default_int_handler)
    """)
    assert "INTERRUPTED" in result.stdout, result.stderr
    assert "HANDLER True" in result.stdout
    volume = float(result.stdout.split("VOLUME")[1].split()[0])
    assert volume == pytest.approx(1.0, rel=1e-6)


def test_quit_does_not_end_process(run_python):
    result = run_python("""
        from pysurfaceevolver import Evolver, EvolverExit
        ev = Evolver("cube.fe")
        try:
            ev.command("quit 7")
        except EvolverExit as e:
            print("EXIT", e.code)
        print("ALIVE")
    """)
    assert result.returncode == 0, result.stderr
    assert "EXIT 7" in result.stdout and "ALIVE" in result.stdout
