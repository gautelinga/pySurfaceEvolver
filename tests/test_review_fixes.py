"""What the review of the post-0.7.2 batch found, each pinned down."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import EvolverError, constraints as C, recipes, shapes


# ---- engine ---------------------------------------------------------------------

def test_bulk_writes_on_several_element_types(cube):
    # Evolver's attribute names are global: one scratch name per element type
    cube.fix("vertex", [0, 1])
    cube.fix("edge", [0])
    cube.set_values("facet", "tension", 2.0, where=[0])
    cube.set_values("body", "target", 1.5, where=[0])
    cube.unfix("vertex", [0, 1])
    assert cube.values("edge", "fixed")[0] == 1 and cube.body(1).target == 1.5
    cube.relax(levels=1, stability=False)


def test_vertex_averaging_leaves_vertices_fixed_after_the_cache(cube):
    cube.refine(5)                  # enough vertices for the cached rings
    cube.command("V")
    ids = cube.values("vertex", "id")
    mask = ids % 7 == 0
    cube.fix("vertex", mask)
    before = cube.vertices[mask]
    cube.command("V")
    assert np.array_equal(cube.vertices[mask], before)


def test_energies_count_a_modulus_once(fe_dir):
    text = (fe_dir/"cube.fe").read_text().replace(
        "vertices", "method_instance zg method facet_vector_integral modulus 3 global\n"
        "vector_integrand:\nq1: 0\nq2: 0\nq3: z\nquantity zq energy method zg\n\nvertices", 1)
    ev = pyse.Evolver()
    ev.load_string(text)
    ev.command("g 3")
    terms = ev.energies()
    assert sum(terms.values()) == pytest.approx(ev.total_energy, rel=1e-12)


def test_an_energy_reading_a_pressure_is_taken_after_the_pressures(fe_dir):
    # the fused energy pass runs before the pressures are recomputed
    text = (fe_dir/"cube.fe").read_text().replace(
        "vertices", "quantity pq energy method facet_scalar_integral\n"
        "scalar_integrand: 0.001*body[1].pressure*z\n\nvertices", 1)
    ev = pyse.Evolver()
    ev.load_string(text)
    ev.command("g 5; lagrange 2; g 3")
    energy = ev.total_energy
    ev.recalc()
    assert ev.total_energy == pytest.approx(energy, rel=1e-13)


def test_restore_brings_back_the_snapshots_own_memory(fe_dir, monkeypatch):
    monkeypatch.chdir(fe_dir)
    cube = pyse.Evolver("cube.fe")
    snap = cube.save()
    drop = shapes.cap(1.0, 0.6)
    other = pyse.Evolver()
    other.load_arrays(drop.vertices, drop.faces,
                      constraints={1: C.plane((0, 0, 1), 0.0, 60)},
                      vertex_constraints={1: drop.rims["rim"]}, bodies=[pyse.Body(1.0)])
    assert len(other._walls(True)) == 1
    cube.restore(snap)                         # (this handle loads: no warning)
    assert cube._walls(True) == []            # the cube has no walls
    with pytest.warns(pyse.SharedEngineWarning):
        other.counts                           # the drop is gone: it says so


def test_shared_engine_warning_points_at_the_callers_line(load):
    first = load("cube.fe")
    load("mound.fe")
    with pytest.warns(pyse.SharedEngineWarning) as record:
        first.total_energy
    assert record[0].filename == __file__


def test_relax_keeps_callback_warnings_where_they_were_raised(cube):
    def callback(ev, i):
        warnings.warn("from the callback", RuntimeWarning)
    with pytest.warns(RuntimeWarning) as record:
        cube.relax(callback=callback, every=50, stability=False)
    assert record[0].filename == __file__


def test_relax_errors_are_not_replaced_by_warnings_under_W_error(cube):
    def callback(ev, i):
        warnings.warn("noise", RuntimeWarning)
        raise ValueError("the real error")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="the real error"):
            cube.relax(callback=callback, stability=False)


def test_unknown_constraints_and_names(cube):
    with pytest.raises(KeyError, match="no constraint"):
        cube.on_constraint(7)
    with pytest.raises(KeyError, match="scale_limit"):
        cube["sclae_limit"] = 1
    with pytest.raises(EvolverError) as info:
        pyse.Evolver().load_string("vertices\n1 0 0 0 constraint 9\n")
    assert "tmp" not in str(info.value)


# ---- building -------------------------------------------------------------------

def _two_drops_on_a_bead():
    bead = C.sphere((0, 0, 0), 0.5, contact_angle=60, wet_poles=("north", "south"))
    top, bottom = shapes.cap(0.4, 0.3), shapes.cap(0.4, 0.3)
    v = np.vstack([top.vertices, bottom.vertices*[1, 1, -1]])
    n = len(top.vertices)
    f = np.vstack([top.faces, bottom.faces[:, ::-1] + n])
    rims = list(top.rims["rim"]) + [r + n for r in bottom.rims["rim"]]
    return bead, v, f, rims, len(top.faces)


def test_a_wall_two_bodies_touch_is_left_to_the_user():
    bead, v, f, rims, k = _two_drops_on_a_bead()
    with pytest.warns(UserWarning, match="several bodies touch"):
        text = pyse.make_datafile(v, f, constraints={1: bead}, vertex_constraints={1: rims},
                                  bodies=[pyse.Body(0.1, faces=range(k)),
                                          pyse.Body(0.1, faces=range(k, len(f)))])
    assert "volconst" not in text


def test_several_bodies_need_their_faces():
    box = shapes.box()
    with pytest.raises(ValueError, match="give each its faces"):
        pyse.make_datafile(box.vertices, box.faces, bodies=[pyse.Body(1.0), pyse.Body(2.0)])


def test_body_arguments_in_the_old_order_say_so():
    with pytest.raises(TypeError, match="give faces= by name"):
        pyse.Body(list(range(6)), 1.0)


def test_per_vertex_constraints_from_numpy():
    drop = shapes.cap(1.0, 0.6)
    on = np.zeros(len(drop.vertices), dtype=np.int64)
    on[drop.rims["rim"]] = 1
    text = pyse.make_datafile(drop.vertices, drop.faces, constraints={1: "z = 0"},
                              vertex_constraints=on, bodies=[pyse.Body(1.0)])
    assert text.count("constraint 1\n") >= len(drop.rims["rim"])
    with pytest.raises(TypeError, match="for a mask"):
        pyse.make_datafile(drop.vertices, drop.faces, constraints={1: "z = 0"},
                           vertex_constraints=on.astype(bool))


def test_contact_angle_names_optimizing_parameters_and_exponents():
    drop = shapes.cap(1.0, 0.6)
    for header, angle in (("optimizing_parameter theta = 30", "theta"),
                          ("", "60 + 0*1e3")):
        pyse.make_datafile(drop.vertices, drop.faces, header=header,
                           constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=angle)},
                           vertex_constraints={1: drop.rims["rim"]})


def test_single_letter_names_are_refused():
    drop = shapes.cap(1.0, 0.6)
    with pytest.raises(ValueError, match="single letters"):
        pyse.make_datafile(drop.vertices, drop.faces, parameters={"h": 1.0})


def test_no_orientation_warning_at_ninety_degrees():
    drop = shapes.cap(1.0, 0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyse.make_datafile(drop.vertices, drop.faces,
                           constraints={1: C.plane((0, 0, -1), 0.0, contact_angle=90)},
                           vertex_constraints={1: drop.rims["rim"]}, bodies=[pyse.Body(1.0)])


def test_a_string_body_is_oriented_counterclockwise():
    square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], float)      # clockwise
    ev = pyse.Evolver()
    ev.load_arrays(square, [[0, 1, 2, 3]], string=True, bodies=[pyse.Body(1.0)])
    assert ev.body(1).volume == pytest.approx(1.0)


# ---- health and continuation ----------------------------------------------------

def test_health_sees_a_wall_no_vertex_is_on():
    drop = shapes.cap(1.0, 0.6)
    ev = pyse.Evolver()
    ev.load_arrays(drop.vertices, drop.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, 90), 2: "x = 0.95"},
                   vertex_constraints={1: drop.rims["rim"]}, bodies=[pyse.Body(2*np.pi/3)])
    ev.relax(stability=False)
    assert not ev.health().ok


def test_continuation_stops_where_a_film_pinches_off():
    tube = shapes.tube(1, 0, 1.0, nz=4, nt=16)
    top = C.curve("cos(p1)", "sin(p1)", "ht", lambda x: np.arctan2(x[:, 1], x[:, 0]))
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces, boundaries={1: C.circle(), 2: top},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   parameters={"ht": 1.0})
    ev.relax(levels=1)
    with pytest.warns(UserWarning, match="stopped before"):
        steps = list(recipes.continuation(ev, "ht", [1.1, 1.2, 1.3, 1.4, 1.5],
                                          stop_on="unconverged"))
    assert steps[-1].value <= 1.3 and ev.parameters["ht"] == steps[-1].value


# ---- shapes ---------------------------------------------------------------------

def test_box_rim_is_in_order():
    box = shapes.box((2, 1, 1), open_bottom=True, n=3)
    rim = box.vertices[box.rims["rim"]]
    steps = np.linalg.norm(np.diff(np.vstack([rim, rim[:1]]), axis=0), axis=1)
    assert np.all(np.isclose(steps, 2/3) | np.isclose(steps, 1/3))     # neighbours on it


@pytest.mark.parametrize("make", [lambda: shapes.tube(nt=2), lambda: shapes.disc(nr=0),
                                  lambda: shapes.revolve([0, 0], [0, 1])])
def test_degenerate_shapes_are_refused(make):
    with pytest.raises(ValueError):
        make()


def test_popping_a_fixed_edge_between_free_facets_ends(run_python):
    # Evolver divided such an edge and retried without end (pre-existing):
    # relax() pops by default, so it hung; run apart in case it comes back
    out = run_python("""
        import pysurfaceevolver as pyse
        ev = pyse.Evolver("cube.fe")
        ev.command("set edge[1] fixed")
        print(ev.command("o").split()[-1], ev.check())
    """, timeout=60)
    assert out.returncode == 0 and out.stdout.strip().endswith("0 []")
