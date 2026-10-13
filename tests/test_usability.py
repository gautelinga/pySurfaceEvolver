"""Building surfaces without silent mistakes: orientation and wall checks,
constants put where they belong, clear errors, shapes, measurements."""

import types
import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes


def _drop(theta=60.0, volume=1.0, faces=None, normal=(0, 0, 1), **kwargs):
    drop = shapes.cap(1.0, 0.6, nr=6, nt=24)
    f = drop.faces if faces is None else faces(drop.faces)
    ev = pyse.Evolver()
    ev.load_arrays(drop.vertices, f, constraints={1: C.plane(normal, 0.0, contact_angle=theta)},
                   vertex_constraints={1: drop.rims["rim"]},
                   bodies=[pyse.Body(volume=volume)], **kwargs)
    return ev


def _cap_energy(volume, theta):
    t = np.radians(theta)
    R = (volume/(np.pi*(2 - 3*np.cos(t) + np.cos(t)**3)/3))**(1/3)
    h, a = R*(1 - np.cos(t)), R*np.sin(t)
    return 2*np.pi*R*h - np.cos(t)*np.pi*a*a


def test_reversed_faces_warn_at_load():
    with pytest.warns(UserWarning, match="oriented into it"):
        _drop(faces=lambda f: f[:, ::-1])


def test_a_plane_facing_into_the_solid_warns():
    with pytest.warns(UserWarning, match="far side of constraint 1"):
        pyse.make_datafile(*_cap_arrays(), constraints={1: C.plane((0, 0, -1), 0.0, 60)},
                           vertex_constraints={1: shapes.cap(1.0, 0.6).rims["rim"]},
                           bodies=[pyse.Body(volume=1)])


def _cap_arrays():
    drop = shapes.cap(1.0, 0.6)
    return drop.vertices, drop.faces


def test_a_drop_from_shapes_matches_the_exact_cap():
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # no warnings on the way
        ev = _drop()
        r = ev.relax(levels=1)
    assert r.converged and r.stable
    assert ev.energy == pytest.approx(_cap_energy(1.0, 60), rel=2e-3)
    line = ev.contact_lines(1)[0]
    assert np.allclose(line[0], line[-1])         # closed
    s = ev.body_surfaces(cap=True)[1]
    assert s.volume == pytest.approx(1.0, rel=1e-6)
    assert s.area == pytest.approx(s.areas().sum())


def test_volconst_and_energy_constant_come_from_the_walls():
    # a drop around the north pole of a sphere: the wetted cap holds the pole,
    # so the sphere's volconst goes to the body and its left-out energy to
    # ev.energy, also with the contact angle as a parameter
    ball = C.sphere((0, 0, -1.0), 1.0, contact_angle=70, wet_poles=("north",))
    ballp = C.sphere((0, 0, -1.0), 1.0, contact_angle="th", wet_poles=("north",))
    v, f = _cap_arrays()
    v = v.copy()
    rim = shapes.cap(1.0, 0.6).rims["rim"]
    # rim onto the sphere: a circle of radius 0.6 at the right height
    v[rim, :2] *= 0.6/np.hypot(v[rim, 0], v[rim, 1])[:, None]
    v[rim, 2] = -1 + np.sqrt(1 - 0.36)
    results = []
    for wall, params in ((ball, {}), (ballp, {"th": 70.0})):
        ev = pyse.Evolver()
        ev.load_arrays(v, f, constraints={1: wall}, vertex_constraints={1: rim},
                       parameters=params, bodies=[pyse.Body(volume=0.5)])
        assert ev.body(1).volconst == pytest.approx(ball.volconst)
        ev.relax(levels=1)
        results.append((ev.energy, ev.total_energy))
    (e1, t1), (e2, t2) = results
    assert e1 - t1 == pytest.approx(ball.energy_constant)
    assert e2 == pytest.approx(e1, rel=1e-9)


def test_named_constraints_and_per_vertex_sets():
    v, f = _cap_arrays()
    rim = set(shapes.cap(1.0, 0.6).rims["rim"])
    per_vertex = [{"table"} if i in rim else None for i in range(len(v))]
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints={"table": C.plane((0, 0, 1), 0.0, 60)},
                   vertex_constraints=per_vertex, bodies=[pyse.Body(volume=1)])
    assert ev.on_constraint("table").sum() == len(rim)
    assert ev.relax().converged


@pytest.mark.parametrize("kwargs, message", [
    (dict(faces_shift=1), "rows are 0-based"),
    (dict(vertex_constraints={2: [0]}), "isn't defined"),
    (dict(constraints={1: "z == 0"}), "one '='"),
    (dict(constraints={1: C.plane((0, 0, 1), 0.0, contact_angle="th")}), "declared parameter"),
])
def test_clear_errors(kwargs, message):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    faces = [[0 + kwargs.pop("faces_shift", 0), 1, 2]]
    kwargs.setdefault("constraints", {1: "z = 0"})
    kwargs.setdefault("vertex_constraints", {1: [0, 1]})
    with pytest.raises(ValueError, match=message):
        pyse.make_datafile(v, [[f + (1 if message == "rows are 0-based" else 0)
                                for f in face] for face in faces], **kwargs)


def test_refine_refuses_negative_counts(cube):
    with pytest.raises(ValueError):
        cube.refine(-1)


def test_load_string_names_the_surface():
    ev = pyse.Evolver()
    ev.load_arrays(*_cap_arrays())
    assert ev.datafile == "surface.fe" and "tmp" not in repr(ev)


def test_walls_true_draws_the_load_arrays_walls():
    ev = _drop()
    walls = ev._walls(True)
    assert len(walls) == 1 and walls[0].shape[0] == "plane"


def test_film_on_a_circular_wire_stays_round():
    film = shapes.disc(1.0, nr=1, nt=8)
    ev = pyse.Evolver()
    ev.load_arrays(film.vertices, film.faces, boundaries={1: C.circle()},
                   vertex_boundaries={1: film.rims["rim"]})
    r = ev.relax(levels=3)
    assert r.converged and r.stable
    rim = ev.mesh().vertices[ev.values("vertex", "on_boundary 1") != 0]
    assert np.allclose(np.hypot(rim[:, 0], rim[:, 1]), 1.0)
    assert ev.total_area == pytest.approx(np.pi, rel=3e-3)


def test_shapes_are_closed_and_outward():
    box = shapes.box((1, 2, 3), n=2)
    a, b, c = (box.vertices[box.faces[:, i]] for i in range(3))
    assert pyse.is_watertight(box.faces)
    assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum()/6 == pytest.approx(6.0)
    t = np.linspace(0, np.pi, 17)
    ball = shapes.revolve(np.sin(t), -np.cos(t), 48)
    assert pyse.is_watertight(ball.faces)
    tube = shapes.tube(1, 0, 1)
    assert set(tube.rims) == {"bottom", "top"}


def test_derivative_of_a_drop_energy_by_contact_angle():
    ev = _drop(theta="th", parameters={"th": 60.0})
    ev.relax(levels=1)
    before = ev.vertices
    d = ev.derivative("th", 0.5)
    exact = (_cap_energy(1.0, 60.5) - _cap_energy(1.0, 59.5))/1.0
    assert d == pytest.approx(exact, rel=0.02)
    assert np.array_equal(ev.vertices, before) and ev.parameters["th"] == 60.0


def test_continuation_stops_at_the_first_failed_step_and_halves(cube):
    cube.relax()
    cube.define("pp", 0.0)
    last = {"pp": 0.0}

    def relax(ev):               # "converges" only for steps of 0.25 or less
        ok = abs(ev["pp"] - last["pp"]) <= 0.25 + 1e-12
        last["pp"] = ev["pp"]
        return types.SimpleNamespace(converged=ok, stable=True)

    # (no predictor: its retry of a failed step would look like a step of 0 here)
    with pytest.warns(UserWarning, match="stopped before"):
        steps = list(recipes.continuation(cube, "pp", [0.2, 0.4, 1.0, 1.2], relax=relax,
                                          stop_on="unconverged", predictor=False))
    assert [s.value for s in steps] == [0.2, 0.4] and cube["pp"] == 0.4
    last["pp"] = 0.0
    cube["pp"] = 0.0
    # numpy steps: the warning names the value as a plain number (numpy 2's repr
    # would say np.float64(1.0))
    with pytest.warns(UserWarning, match=r"stopped before 1\.0:"):
        list(recipes.continuation(cube, "pp", np.array([0.2, 0.4, 1.0]), relax=relax,
                                  stop_on="unconverged", predictor=False))
    last["pp"] = 0.0
    cube["pp"] = 0.0
    steps = list(recipes.continuation(cube, "pp", [0.2, 0.4, 1.0], relax=relax,
                                      stop_on="unconverged", halvings=2, predictor=False))
    assert [s.substeps for s in steps] == [1, 1, 4] and cube["pp"] == 1.0


def test_continuation_rejects_a_misspelt_name(load):
    ev = load("mound.fe")
    with pytest.raises(KeyError, match="did you mean 'angle'"):
        next(recipes.continuation(ev, "angel", [50]))


def test_stability_can_report_eigenvalues_when_stable(cube):
    cube.relax()
    s = cube.stability(eigenvalues=True)
    assert s.stable and len(s.lowest) and s.lowest.min() > 0


def test_energies_add_up_and_leave_the_surface_alone(load):
    ev = load("tankex.fe")
    ev.relax(stability=False)
    before = ev.vertices
    terms = ev.energies()
    assert sum(terms.values()) == pytest.approx(ev.total_energy, rel=1e-12)
    assert terms["default_area_inst"] > 0 and "gap_energy" in terms
    assert np.array_equal(ev.vertices, before)
    assert ev.quantities().keys() == {"arb_grav"}     # not converted for good


def test_a_solid_inside_the_only_body_closes_its_volume():
    # a sphere around the origin inside a drop that never touches it: the
    # one body still gets its volconst (and a second body would warn)
    ball = C.sphere((0, 0, 0.3), 0.1, contact_angle=50, wet_poles=("north", "south"))
    v, f = _cap_arrays()
    rim = shapes.cap(1.0, 0.6).rims["rim"]
    text = pyse.make_datafile(v, f, constraints={1: C.plane((0, 0, 1), 0.0, 60), 2: ball},
                              vertex_constraints={1: rim}, bodies=[pyse.Body(volume=1)])
    assert f"volconst {float(ball.volconst)!r}" in text
