"""eval(), values(), parameters and named quantities."""

import math

import numpy as np
import pytest

from pysurfaceevolver import EvolverError, Quantity


# --- eval ------------------------------------------------------------------------

def test_eval_is_exact(cube):
    cube.command("g 3")
    assert cube.eval("total_area") == cube.total_area
    assert cube.eval("2*pi") == 2 * math.pi
    assert cube.eval("1e-300") == 1e-300


def test_eval_handles_huge_and_non_finite_values(cube):
    assert cube.eval("1e301") == 1e301
    assert cube.eval("2^1023*1.5") == 2.0**1023 * 1.5
    assert cube.eval("1e-320") == 1e-320
    assert cube.eval("-1e308*10") == -math.inf
    assert math.isnan(cube.eval("1e308*10 - 1e308*10"))
    assert cube.values("vertex", "x*1e305").max() == 1e305


def test_eval_defines_no_variable(cube):
    cube.eval("total_area")
    out = cube.command("list topinfo")
    assert "pyse" not in out
    with pytest.raises(EvolverError):
        cube.eval("pyse_eval_result_")


def test_eval_error_message_shows_only_the_expression(cube):
    with pytest.raises(EvolverError) as info:
        cube.eval("1/0")
    message = str(info.value)
    assert "Divide by zero" in message
    assert "@pyse@" not in message and "printf" not in message
    assert message.rstrip().endswith("1/0")


def test_eval_does_not_print(cube, capsys):
    cube.eval("total_area")
    assert capsys.readouterr().out == ""


# --- values -----------------------------------------------------------------------

def test_vertex_values_align_with_mesh(cube):
    cube.command("r; g 2")
    m = cube.mesh()
    np.testing.assert_array_equal(cube.values("vertex", "x"), m.vertices[:, 0])
    np.testing.assert_array_equal(cube.values("vertex", "z"), m.vertices[:, 2])
    np.testing.assert_array_equal(cube.values("vertex", "id"), m.vertex_ids)
    np.testing.assert_allclose(cube.values("vertex", "x^2 + y^2"),
                               m.vertices[:, 0] ** 2 + m.vertices[:, 1] ** 2, rtol=1e-15)


def test_edge_and_facet_values_align_with_mesh(cube):
    cube.command("r; g 2")
    m = cube.mesh()
    np.testing.assert_array_equal(cube.values("edge", "id"), m.edge_ids)
    np.testing.assert_array_equal(cube.values("facet", "id"), m.facet_ids)
    lengths = np.linalg.norm(m.vertices[m.edges[:, 1]] - m.vertices[m.edges[:, 0]], axis=1)
    np.testing.assert_allclose(cube.values("edge", "length"), lengths, rtol=1e-12)
    assert cube.values("facet", "area").sum() == pytest.approx(cube.total_area, rel=1e-12)


def test_body_values(cube):
    cube.command("g 5")
    np.testing.assert_allclose(cube.values("body", "volume"), cube.bodies().volume)


def test_values_accepts_plural_names(cube):
    assert len(cube.values("vertices", "x")) == cube.counts["vertices"]


def test_values_unknown_element_type(cube):
    with pytest.raises(ValueError, match="unknown element type"):
        cube.values("tetrahedron", "x")


def test_values_bad_expression(cube):
    with pytest.raises(EvolverError) as info:
        cube.values("vertex", "x/0")
    assert "@pyse@" not in str(info.value)


def test_body_coordinates_raise_instead_of_crashing(cube):
    # Stock Evolver pushes nothing for x on a body, which misaligns its eval
    # stack and crashes printf; the PYSE build reports an error instead.
    for call in (lambda: cube.values("body", "x"), lambda: cube.eval("body[1].x"),
                 lambda: cube.eval("sum(body, x)")):
        with pytest.raises(EvolverError, match="Can't have x on body"):
            call()
    assert cube.eval("body[1].volume") == pytest.approx(1.0)


def test_values_string_model(load):
    ev = load("knotty.fe")
    lengths = ev.values("edge", "length")
    assert lengths.shape == (ev.counts["edges"],)
    assert lengths.sum() == pytest.approx(ev.eval("sum(edge, length)"), rel=1e-12)


# --- parameters -------------------------------------------------------------------

def test_parameters(load):
    ev = load("tankex.fe")
    assert dict(ev.parameters) == {"ENDT": 0.707, "WALLT": 0.707, "GY": 0.0, "GZ": -1.0}
    assert "ENDT" in ev.parameters
    assert len(ev.parameters) == 4


def test_set_parameter(load):
    ev = load("mound.fe")
    assert ev.parameters["angle"] == 90
    ev.parameters["angle"] = 60
    assert ev.parameters["angle"] == 60
    assert ev.eval("angle") == 60


def test_parameters_exclude_command_variables(load):
    ev = load("mound.fe")
    ev["my_variable"] = 3
    assert "my_variable" not in ev.parameters
    assert list(ev.parameters) == ["angle"]


def test_unknown_parameter(load):
    ev = load("mound.fe")
    with pytest.raises(KeyError):
        ev.parameters["nope"]
    with pytest.raises(KeyError):
        ev.parameters["nope"] = 1
    with pytest.raises(TypeError):
        del ev.parameters["angle"]


def test_no_optimizing_parameters(load):
    assert load("mound.fe").parameters.optimizing == frozenset()


# --- quantities -------------------------------------------------------------------

def test_fixed_quantity(load):
    ev = load("qmound.fe")
    ev.command("g 5")
    q = ev.quantities()
    assert list(q) == ["vol"]
    vol = q["vol"]
    assert isinstance(vol, Quantity)
    assert vol.kind == "fixed"
    assert vol.target == 1.0
    assert vol.value == pytest.approx(1.0, rel=1e-3)


def test_quantity_kinds(load):
    q = load("mylarcube.fe").quantities()
    assert {name: x.kind for name, x in q.items()} == {
        "cube_volume": "energy", "stretch": "energy",
        "stretch1": "info", "stretch2": "info",
    }
    assert math.isnan(q["stretch"].target)
    assert q["stretch"].modulus == 10


@pytest.mark.parametrize("datafile", ["knotty.fe", "qmound.fe", "mylarcube.fe"])
def test_quantities_match_evolver_attributes(load, datafile):
    ev = load(datafile)
    ev.command("g 2")
    for name, q in ev.quantities().items():
        assert q.value == ev.eval(f"{name}.value")
        assert q.modulus == ev.eval(f"{name}.modulus")
        assert q.pressure == ev.eval(f"{name}.pressure")


def test_no_quantities(cube):
    assert cube.quantities() == {}
