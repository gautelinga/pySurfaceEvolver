"""relax() and health() across Evolver's models: strings, simplices, tori,
elastic membranes, surfaces without a Hessian, automatic topology changes."""

import numpy as np
import pytest

import pysurfaceevolver as pyse


def test_relax_string_models(load):
    # strings can't be equiangulated: the tidying averages vertices only
    for name in ("knotty.fe", "slidestr.fe", "metric.fe"):
        ev = load(name)
        r = ev.relax(levels=1 if name != "knotty.fe" else 0)
        assert r.converged, name
        assert r.health.ok, (name, r.health.issues)


def test_relax_simplex_takes_no_newton(load):
    # no Hessian for simplex bodies outside the Lagrange model: gradient only
    ev = load("simplex3.fe")
    r = ev.relax(max_iter=30)
    assert r.newton_steps == 0
    assert np.isfinite(ev.total_energy)
    assert r.health is not None


def test_relax_without_hessian_runs_gradient_rounds(load):
    # tankex: gap energy on a curved wall, no Hessian ("third derivatives");
    # relax() goes on with plain gradient rounds instead of stopping early
    ev = load("tankex.fe")
    r = ev.relax(levels=1)
    assert r.newton_steps == 0
    assert len(r.energy) > 400
    assert ev.total_energy < 1.65          # the residual watch stopped at 1.75


def test_newton_step_raising_the_energy_is_undone(load):
    # far from equilibrium a Newton step on the 100-cell foam runs off to a
    # tangled surface (energy x1e4) with a smaller residual: it must be undone
    ev = load("100grain.fe")
    ev.command("g 10")
    e0 = ev.total_energy
    ev.newton(1, seek=False)
    assert ev.total_energy <= 1.1*e0


def test_put_back_refuses_a_changed_topology(cube):
    snap = cube._coords()
    cube.command("g 2")
    assert cube._put_back(snap)
    snap = cube._coords()
    cube.refine()
    assert not cube._put_back(snap)


def test_relax_with_autopop_keeps_the_surface(load):
    # autopop and autochop change the topology during gradient steps; undoing
    # by coordinates onto the new topology wrecked the foam (energy 15.8 -> 901)
    ev = load("100grain.fe")
    e0 = ev.total_energy
    ev.relax(max_iter=200)
    assert ev.total_energy < e0


def test_residual_ignores_area_normalization(load):
    ev = load("100grain.fe")
    assert ev.eval("area_normalization") == 1
    with_norm = ev.residual()
    ev.command("area_normalization off")
    assert ev.residual() == pytest.approx(with_norm, rel=1e-12)


def test_elastic_membrane_detected_and_not_retriangulated(load, cube):
    assert not cube._elastic()
    ev = load("mylarcube.fe")
    assert ev._elastic()
    edges = ev.mesh().edges.copy()
    ev.relax(max_iter=50)
    assert np.array_equal(ev.mesh().edges, edges)       # no 'u' flips


def test_health_ignores_fixed_and_contact_vertices(load):
    # mound.fe draws its table top with fixed vertices lying in the plane of
    # constraint 1; slidestr.fe draws its slide as a second constraint on the
    # same curve as the one the drop's ends slide on
    ev = load("mound.fe")
    ev.relax(levels=1)
    assert not any("touches" in i for i in ev.health().issues)
    ev = load("slidestr.fe")
    ev.relax(levels=1)
    assert not any("touches" in i for i in ev.health().issues)


def test_torus_facet_angles_from_edge_lengths(load):
    # coordinates are wrapped into the unit cell: facets across its sides
    # looked like slivers (smallest angle 0.05 degrees)
    ev = load("twointor.fe")
    ev.refine(2)
    assert ev.mesh_quality().angle_min > 5


def test_health_counts_junctions_against_plateau(load):
    ev = load("octa.fe")
    assert ev.health().junctions == 6
    ev.command("o")
    assert ev.health().junctions == 0
    ev = load("popstr.fe")
    h = ev.health()
    assert h.junctions == 1 and any("Plateau" in i for i in h.issues)
    ev = load("cube.fe")
    assert ev.health().junctions == 0


def test_health_in_the_plane(load):
    ev = load("metric.fe")             # a string in two dimensions
    h = ev.health()
    assert np.isfinite(h.residual)


def test_relax_pops_junctions_against_plateau(load):
    # octa.fe: a symmetric film with six edges where four films meet; popping
    # them (and deleting the small films that then shrink away) reaches the
    # energy of Brakke's own recipe (gogo)
    ev = load("octa.fe")
    r = ev.relax()
    assert r.converged and r.popped > 0
    assert ev.total_energy == pytest.approx(22.627417, rel=1e-6)
    assert ev.health().junctions == 0
    ev = load("popstr.fe")
    r = ev.relax()
    assert r.popped == 1
    assert ev.total_energy == pytest.approx(2.7320508, rel=1e-7)


def test_relax_pinch_false_keeps_collapsing_edges(load):
    ev = load("octa.fe")
    r = ev.relax(pinch=False)
    assert r.collapsed == 0 and r.popped > 0


def test_relax_leaves_a_sound_surface_alone(cube):
    r = cube.relax()
    assert r.popped == 0 and r.collapsed == 0


def test_relax_pinches_a_collapsing_neck(load):
    # cat.fe: a catenoid at its fold. The neck collapses during untidied
    # gradient steps, its ring of edges is deleted and the pinched vertex
    # popped: two discs (the rings are 12-gons after one refinement)
    ev = load("cat.fe")
    r = ev.relax(levels=1)
    assert r.converged and r.collapsed > 0 and r.popped > 0
    assert ev.total_energy == pytest.approx(13.6603, rel=1e-4)
    assert r.health.ok
