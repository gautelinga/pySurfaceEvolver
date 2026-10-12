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
    r = ev.relax(levels=2)
    assert r.newton_steps == 0
    assert len(r.energy) > 400
    assert ev.total_energy < 1.52          # the residual watch stopped at 1.71
    assert r.health.target_gap < 1e-4


def test_relax_without_hessian_keeps_a_degenerate_mesh_from_sliding(load):
    # from 8 facets refined once, tankex's facets degenerate; plain gradient
    # rounds then slid to energy 0.03, and the last tidying lost 40% of the
    # volume without a word. Where it stops now is path-dependent (0.7.x stop
    # anywhere in 1.29-1.78 depending on the thread count): the 24-vertex mesh
    # cannot settle against the curved wall, and relax() says so
    ev = load("tankex.fe")
    r = ev.relax(levels=1)
    assert ev.total_energy > 1.0           # not the slide (0.03); level 2 gives 1.50
    assert r.health.target_gap < 1e-4
    assert not r.converged and not r.health.ok


def test_newton_meets_a_new_volume_target(cube):
    # the residual doesn't see an unmet volume, and the safeguard undid the
    # step restoring it (energy and residual both rose)
    cube.relax(levels=1, stability=False)
    for target in (2.0, 0.5):
        cube.body(1).target = target
        assert cube.newton(10) > 0
        assert cube.body(1).volume == pytest.approx(target, rel=1e-9)
        assert cube.residual() < 1e-8


def test_health_reports_a_missed_target(cube):
    cube.body(1).target = 2.0
    h = cube.health()
    assert h.target_gap == pytest.approx(0.5)
    assert any("off its target" in i for i in h.issues)


def test_last_resort_backs_off_after_failures(cube):
    from pysurfaceevolver import _evolver
    r = cube.relax(newton=0, max_iter=5, tol=1e-14, stability=False)     # unconverged, nothing collapses
    assert not r.converged and r.collapsed == 0
    assert (_evolver._collapse_failures, _evolver._collapse_skip) == (1, 1)
    cube.relax(newton=0, max_iter=5, tol=1e-14, stability=False)         # skipped
    assert (_evolver._collapse_failures, _evolver._collapse_skip) == (1, 0)
    cube.relax(newton=0, max_iter=5, tol=1e-14, stability=False)         # tried again
    assert (_evolver._collapse_failures, _evolver._collapse_skip) == (2, 3)


def test_relax_checks_stability_just_short_of_tol(load):
    # a step that ends a little above tol still gets its stability checked
    first = load("cube.fe").relax(newton=0, max_iter=200)
    ev = load("cube.fe")
    r = ev.relax(newton=0, max_iter=200, tol=first.residual/10)
    assert not r.converged and r.stable is not None


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
    assert ev.health().junctions == 7       # six edges, and the cone at the centre
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
    # energy of Brakke's own recipe (gogo). One thread: topology changes follow
    # the path, which varies at round-off from run to run with several
    ev = load("octa.fe")
    r = ev.relax(threads=1)
    assert r.converged and r.popped > 0
    assert ev.total_energy == pytest.approx(22.627417, rel=1e-6)
    assert ev.health().junctions == 0
    ev = load("popstr.fe")
    r = ev.relax()
    assert r.popped == 1
    assert ev.total_energy == pytest.approx(2.7320508, rel=1e-7)


def test_relax_leaves_a_sound_surface_alone(cube):
    r = cube.relax()
    assert r.popped == 0 and r.collapsed == 0


def test_relax_pinches_a_collapsing_neck(load):
    # cat.fe: a catenoid at its fold. The neck collapses during untidied
    # gradient steps, its ring of edges is deleted and the pinched vertex
    # popped: two discs (the rings are 12-gons after one refinement). One
    # thread, as above
    ev = load("cat.fe")
    r = ev.relax(levels=1, threads=1)
    assert r.converged and r.collapsed > 0 and r.popped > 0
    assert ev.total_energy == pytest.approx(13.6603, rel=1e-4)
    assert r.health.ok


def test_relax_crystalline_skips_tidying(load):
    # crystal.fe: a crystalline (Wulff) energy; equiangulating and averaging
    # undo the facets' alignment, and relax() ended at 3.608 where plain
    # gradient steps reach 3.3256
    ev = load("crystal.fe")
    r = ev.relax()
    assert ev.total_energy == pytest.approx(3.3256068, rel=1e-5)
    # its residual doesn't vanish at the Wulff shape: converged = settled
    assert r.converged and r.settled and r.residual > 1
    assert r.health.ok


def test_relax_warns_when_asked_for_newton_without_a_hessian(load):
    ev = load("crystal.fe")
    with pytest.warns(pyse.EvolverWarning, match="no Newton steps"):
        ev.relax(newton=5, max_iter=20)
    with pytest.warns(pyse.EvolverWarning, match="no Newton steps"):
        load("simplex3.fe").relax(stability=True, max_iter=20)


# ---- Brakke's online examples (kenbrakke.com/evolver/examples) -----------------------

def test_relax_equiangulates_a_coarse_start_early(load):
    # cbga1: solder on a pad under a sphere, started as a hexagonal prism of
    # quads split at their centres. Its recipe first swaps the vertical edges
    # by hand; without that, gradient steps settle at an energy of 11 on the
    # badly triangulated prism (5.3 swapped). relax() equiangulates after its
    # first step, which swaps them.
    ev = load("cbga1.fe")
    r = ev.relax(levels=1)
    assert r.converged and r.health.angle_min > 10
    assert ev.total_energy < 4.6


def test_saddle_then_relax_finds_the_tilted_drop(load):
    # cbga2 (solder at a 90 degree contact angle on a sphere): the symmetric
    # bulge is a saddle, as the file's comments say; saddle() and relax()
    # find the drop leaning to one side
    ev = load("cbga2.fe")
    with pytest.warns(pyse.UnstableEquilibriumWarning):
        r = ev.relax(levels=2)
    assert r.converged and r.stable is False
    symmetric = ev.total_energy
    assert ev.saddle() > 0
    r = ev.relax()
    assert r.converged and r.stable is True and ev.total_energy < symmetric - 0.05


@pytest.mark.parametrize("name, levels, recipe", [
    ("pcell.fe", 3, 0.1955095825),     # Schwarz P, quotient space, after its gogo
    ("dcell.fe", 4, 0.3199260248),     # Schwarz D
    ("bga-1.fe", 2, 0.1547576148),     # a ball grid array joint
])
def test_relax_matches_the_online_recipes(load, name, levels, recipe):
    ev = load(name)
    r = ev.relax(levels=levels)
    assert r.converged and r.stable is True
    assert ev.total_energy == pytest.approx(recipe, rel=3e-5)
