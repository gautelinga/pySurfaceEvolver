"""Phase F: the friendlier API (relaxation, remeshing, selections, bodies, ...)."""

import numpy as np
import pytest

import pysurfaceevolver as pyse


# ---- F9: command errors ------------------------------------------------------

def test_command_error_carries_the_command(cube):
    with pytest.raises(pyse.EvolverError) as info:
        cube.command("set body[1] target")
    assert info.value.command == "set body[1] target"


def test_command_error_hints_at_numpy_repr(cube):
    # what f"{x!r}" gives for a numpy float with numpy >= 2
    with pytest.raises(pyse.EvolverError, match="numpy number's repr"):
        cube.command("set body[1] target np.float64(0.5)")


def test_command_error_hints_at_nan(cube):
    with pytest.raises(pyse.EvolverError, match="not finite"):
        cube.command("set body[1] target nan")


def test_no_hint_for_ordinary_errors(cube):
    with pytest.raises(pyse.EvolverError) as info:
        cube.command("set body[1] targett 1")
    assert "Hint" not in str(info.value)


# ---- F4: bodies -----------------------------------------------------------------

def test_body_handle_reads_and_sets(cube):
    body = cube.body(1)
    assert body.volume == pytest.approx(cube.bodies().volume[0])
    assert body.target == pytest.approx(1.0) and body.fixed
    body.target = np.float64(1.5)          # numpy scalars are fine
    assert cube.bodies().target[0] == pytest.approx(1.5)
    cube.relax(1e-8, 300)
    assert body.volume == pytest.approx(1.5, rel=1e-6)
    assert body.pressure == pytest.approx(cube.bodies().pressure[0])


def test_body_target_none_frees_the_volume(cube):
    cube.body(1).target = None
    assert not cube.body(1).fixed and cube.body(1).target is None
    cube.body(1).target = 1.0
    assert cube.body(1).fixed


def test_body_volconst(cube):
    v = cube.body(1).volume
    cube.body(1).volconst = 0.25
    assert cube.body(1).volconst == 0.25
    assert cube.body(1).volume == pytest.approx(v + 0.25)


def test_body_handle_checks(cube):
    with pytest.raises(IndexError):
        cube.body(2)
    with pytest.raises(ValueError, match="finite"):
        cube.body(1).target = float("nan")
    assert "body 1" in repr(cube.body(1))


# ---- F3: selections ------------------------------------------------------------

def _square_on_constraints():
    # a unit square film; its left side on constraint 1 (x = 0)
    v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]]
    f = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(v, f, constraints={1: "x = 0"},
                                      vertex_constraints={1: [0, 3]}))
    return ev


def test_on_constraint_and_edges_touching():
    ev = _square_on_constraints()
    on1 = ev.on_constraint(1)
    assert on1.tolist() == [True, False, False, True, False]
    m = ev.mesh()
    one = m.edges_touching(on1, "one")
    both = m.edges_touching(on1, "all")
    rows = {tuple(sorted(e)) for e in m.edges[both]}
    assert rows == {(0, 3)}
    assert all((on1[a] != on1[b]) for a, b in m.edges[one])
    assert (m.edges_touching(on1, "any") == (one | both)).all()
    assert ev.on_constraint(1, "edge").sum() == 1
    with pytest.raises(ValueError):
        m.edges_touching(on1[:2])


def test_set_flag_no_refine_keeps_edges_whole():
    ev = _square_on_constraints()
    m = ev.mesh()
    spokes = m.edges_touching(ev.on_constraint(1), "one")
    ev.set_flag("edge", "no_refine", where=spokes)
    assert (ev.values("edge", "no_refine") != 0).tolist() == spokes.tolist()
    ev.set_flag("edge", "no_refine", where=spokes, on=False)
    assert not ev.values("edge", "no_refine").any()
    ev.refine()
    plain = ev.counts["vertices"]
    ev = _square_on_constraints()      # one engine per process: a fresh surface
    ev.set_flag("edge", "no_refine", where=spokes)
    ev.refine()
    # the protected spokes stay whole: one vertex fewer per spoke
    assert ev.counts["vertices"] == plain - spokes.sum()


def test_fix_and_unfix_use_set_flag(cube):
    cube.fix("vertex", where=[0, 1])
    assert cube.mesh().fixed[:2].all() and cube.mesh().fixed.sum() == 2
    cube.unfix("vertex")
    assert not cube.mesh().fixed.any()
    with pytest.raises(ValueError):
        cube.set_flag("vertex", "fixed; quit")


# ---- F1: relaxation -----------------------------------------------------------

def test_relax_levels_refine_and_record_the_level(cube):
    facets = cube.counts["facets"]
    result = cube.relax(tol=1e-8, max_iter=300, levels=2)
    assert cube.counts["facets"] == 16*facets
    assert result.level[0] == 0 and result.level[-1] == 2
    assert (np.diff(result.level) >= 0).all() and len(result.level) == len(result.energy)


def test_relax_tidy_reaches_the_same_energy(cube):
    start = cube.save()
    cube.refine()
    plain = cube.relax(tol=1e-9, max_iter=2000)
    cube.restore(start)
    cube.refine()
    tidy = cube.relax(tol=1e-9, max_iter=2000, tidy=5)
    assert plain.converged and tidy.converged
    assert tidy.energy[-1] == pytest.approx(plain.energy[-1], rel=1e-8)
    assert len(tidy.energy) > len(plain.energy)      # it did relax again after tidying


def test_relax_newton_with_undo(cube):
    cube.relax(tol=1e-8, max_iter=300)
    e = cube.total_energy
    result = cube.relax(tol=1e-8, max_iter=5, newton=3, undo_if=lambda ev: True)
    assert result.newton_steps == 0 and result.converged is False
    assert cube.total_energy == pytest.approx(cube.relax(tol=1e-8, max_iter=5).energy[-1])
    assert abs(cube.total_energy - e) < 1e-6


def test_newton_steps(cube):
    cube.refine()
    cube.relax(tol=1e-6, max_iter=300)
    before = cube.total_energy
    kept = cube.newton(5, tol=1e-12, seek=True)
    assert 1 <= kept <= 5 and cube.total_energy <= before + 1e-12
    calls = []
    assert cube.newton(2, undo_if=lambda ev: calls.append(1) or len(calls) == 2) == 1
    with pytest.raises(ValueError):
        cube.relax(tidy=-1)
