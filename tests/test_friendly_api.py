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
    tidy = cube.relax(tol=1e-9, max_iter=2000, tidy=10)
    assert plain.converged and tidy.converged and len(tidy.energy) % 10 == 0
    # vertex averaging moves the vertices along the surface: a slightly
    # different discrete equilibrium
    assert tidy.energy[-1] == pytest.approx(plain.energy[-1], rel=1e-4)


def test_relax_newton_with_undo(cube):
    cube.refine(2)
    cube.relax(max_iter=20, newton=0)               # not converged: Newton has work
    before = cube.save()
    result = cube.relax(max_iter=1, tidy=0, newton=3, undo_if=lambda ev: True)
    assert result.newton_steps == 0 and result.converged is False
    one_step = cube.total_energy                    # the one gradient step, nothing more
    cube.restore(before)
    cube.relax(max_iter=1, tidy=0, newton=0)
    assert cube.total_energy == pytest.approx(one_step, rel=1e-12)


def test_newton_steps(cube):
    cube.refine()
    cube.relax(tol=1e-6, max_iter=300)
    before = cube.total_energy
    kept = cube.newton(5, tol=1e-12, seek=True)
    assert 1 <= kept <= 5 and cube.total_energy <= before + 1e-12
    calls = []           # undo_if is asked after every attempt (plain, then line search)
    assert cube.newton(2, undo_if=lambda ev: calls.append(1) or len(calls) >= 2) <= 1
    with pytest.raises(ValueError):
        cube.relax(tidy=-1)


# ---- F2: remeshing --------------------------------------------------------------

def _edge_lengths(ev):
    m = ev.mesh()
    return np.linalg.norm(m.vertices[m.edges[:, 0]] - m.vertices[m.edges[:, 1]], axis=1)


def test_remesh_evens_out_edges(cube):
    cube.refine(2)
    xyz = cube.vertices.copy()
    xyz[:, 0] *= 3                      # stretch: long edges along x
    cube.vertices = xyz
    lengths = _edge_lengths(cube)
    h = float(np.median(lengths))
    lo, hi = 0.5*h, 1.6*h
    counts = cube.remesh(target=h)
    assert counts["split"] > 0
    after = _edge_lengths(cube)
    assert (after > hi).sum() < (lengths > hi).sum()
    assert counts["deleted"] > 0
    assert cube.relax(tol=1e-6, max_iter=500).energy[-1] > 0


def test_remesh_protect_keeps_edges_whole_and_restores_flags():
    v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]]
    f = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(v, f))
    m = ev.mesh()
    spokes = m.edges_touching(np.arange(5) == 4, "any")      # the 4 diagonals
    ev.set_flag("edge", "no_refine", where=[0])               # one edge flagged already
    flagged_before = ev.values("edge", "no_refine") != 0
    ev.remesh(max_edge=0.6, min_edge=None, equiangulate=False, protect=spokes)
    # the sides split, except edge 0, flagged no_refine before (flags are honoured)
    assert ev.counts["vertices"] == 5 + int((~spokes & ~flagged_before).sum())
    flags = ev.values("edge", "no_refine")[:len(spokes)] != 0
    assert (flags == flagged_before).all()                     # flags as before


def test_remesh_needs_the_linear_model(cube):
    cube.set_model("quadratic")
    with pytest.raises(ValueError, match="linear"):
        cube.remesh(0.1)
    cube.set_model("linear")
    with pytest.raises(ValueError, match="2\\*min_edge"):
        cube.remesh(max_edge=0.3, min_edge=0.2)


# ---- F6: diagnostics -----------------------------------------------------------

def test_eigen_counts(cube):
    cube.relax(tol=1e-9)
    counts = cube.eigen_counts()
    assert counts.negative == 0 and counts.positive > 0
    assert cube.eigen_counts(shift=1e3).negative == sum(counts)    # all below a huge shift


def test_check_is_clean_for_a_sound_surface(cube):
    assert cube.check() == []


def test_mesh_quality():
    v = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0.05, 0]]
    f = [[0, 1, 2], [1, 3, 2]]           # a right triangle and a sliver
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(v, f))
    q = ev.mesh_quality()
    assert q.edge_min == pytest.approx(1.0) and q.edge_max == pytest.approx(np.hypot(10, 0.95))
    assert 5 < q.angle_min < 6 and q.skinny == 1 and q.degenerate == 0
    assert ev.mesh().quality(skinny_angle=1).skinny == 0


# ---- F7: mirror images ----------------------------------------------------------

def _quadrant_film():
    # a unit square in the first quadrant of the plane z = 0, normals +z
    v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(v, [[0, 1, 2], [0, 2, 3]]))
    return ev


def test_mirror_matrices():
    from pysurfaceevolver._mesh import mirror_matrices
    mats = mirror_matrices(["z", ("x", 0.5), ((0, 1, 1), (0, 0, 0))])
    assert len(mats) == 8 and np.allclose(mats[0], np.eye(4))
    m = mirror_matrices([("x", 0.5)])[1]
    assert np.allclose(m @ [0, 2, 3, 1], [1, 2, 3, 1])
    with pytest.raises(ValueError):
        mirror_matrices(["w"])


def test_mesh_mirrored_keeps_orientation():
    ev = _quadrant_film()
    full = ev.mesh().mirrored(["x", "y"])
    assert len(full.facets) == 8 and len(full.vertices) == 16
    p = [full.vertices[full.facets[:, i]] for i in range(3)]
    normals = np.cross(p[1] - p[0], p[2] - p[0])
    assert (normals[:, 2] > 0).all()                     # all still +z
    lo, hi = full.vertices.min(axis=0), full.vertices.max(axis=0)
    assert np.allclose(lo[:2], -1) and np.allclose(hi[:2], 1)


def test_plot_and_live_view_mirror(can_render):
    pv = pytest.importorskip("pyvista")
    ev = _quadrant_film()
    view = ev.live_view(off_screen=True, mirror=["x", "y"])
    actors = [a for name, a in view.plotter.actors.items() if name.startswith("evolver-surface")]
    assert len(actors) == 4
    xyz = ev.vertices.copy()
    xyz[:, 2] = 0.1*xyz[:, 0]          # move the points; the images share the data
    ev.vertices = xyz
    view.update()
    assert view.fast_updates == 1
    bounds = np.array(view.plotter.bounds)
    assert bounds[0] == pytest.approx(-1) and bounds[1] == pytest.approx(1)
    view.close()
    ev.plot(off_screen=True, mirror=["x"], screenshot=None)


@pytest.fixture(scope="module")
def can_render():
    pv = pytest.importorskip("pyvista")
    try:
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(pv.Sphere())
        plotter.screenshot(return_img=True)
        plotter.close()
    except Exception as e:  # no display / no OpenGL
        pytest.skip(f"off-screen rendering unavailable: {e}")


# ---- F8: continuation ------------------------------------------------------------

def test_continuation_steps_and_records(cube):
    from pysurfaceevolver import recipes
    values = [1.0, 0.9, 0.8]
    steps = list(recipes.continuation(cube, recipes.body_target(1), values,
                                      relax=dict(tol=1e-8, max_iter=400)))
    assert [s.value for s in steps] == values and [s.index for s in steps] == [0, 1, 2]
    for s in steps:
        assert s.volumes[0] == pytest.approx(s.value, rel=1e-6)
    # a cube relaxing to a sphere: pressure ~ 2/r rises as the volume shrinks
    assert steps[0].pressures[0] < steps[1].pressures[0] < steps[2].pressures[0]


def test_continuation_checkpoint_and_resume(cube, tmp_path):
    from pysurfaceevolver import recipes
    values = [1.0, 0.95, 0.9, 0.85]
    prefix = str(tmp_path / "run" / "cube")
    full = [s.energy for s in recipes.continuation(cube, recipes.body_target(1), values,
                                                   relax=dict(tol=1e-9, max_iter=500))]
    cube.load("cube.fe")
    for step in recipes.continuation(cube, recipes.body_target(1), values,
                                     relax=dict(tol=1e-9, max_iter=500), checkpoint=prefix):
        step.extra["note"] = step.index
        if step.index == 1:
            break                      # stopped after two steps (the second unsaved)
    assert [s.index for s in recipes.load_steps(prefix)] == [0]
    cube.load("cube.fe")               # whatever the engine holds: resume replaces it
    resumed = list(recipes.continuation(cube, recipes.body_target(1), values,
                                        relax=dict(tol=1e-9, max_iter=500), resume=prefix,
                                        checkpoint=prefix))
    assert [s.index for s in resumed] == [1, 2, 3]
    steps = recipes.load_steps(prefix)
    assert [s.index for s in steps] == [0, 1, 2, 3] and steps[0].extra == {"note": 0}
    assert [s.energy for s in steps] == pytest.approx(full, rel=1e-7)


def test_continuation_parameter_and_relax_callable(cube):
    from pysurfaceevolver import recipes
    calls = []
    steps = list(recipes.continuation(cube, lambda ev, v: calls.append(v), [1, 2],
                                      relax=lambda ev: ev.iterate(2)))
    assert calls == [1, 2] and len(steps[0].result.energy) == 2
    with pytest.raises(TypeError):
        next(recipes.continuation(cube, recipes.parameter("x"), [1], relax=5))


# ---- G2.3: stability ----------------------------------------------------------

def _column(radius, nz=12, nt=24):
    # a liquid column between plates z = 0 and z = 1 at 90 degrees: stable while
    # its height is below pi*radius (Rayleigh-Plateau with sliding contact lines)
    from pysurfaceevolver import constraints as C
    v = [[radius*np.cos(p), radius*np.sin(p), z] for z in np.linspace(0, 1, nz + 1)
         for p in 2*np.pi*np.arange(nt)/nt]
    f = []
    for i in range(nz):
        for j in range(nt):
            a, b = i*nt + j, i*nt + (j + 1) % nt
            f += [[a, b, b + nt], [a, b + nt, a + nt]]
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: C.plane((0, 0, 1), 0.0), 2: C.plane((0, 0, -1), point=(0, 0, 1))},
        vertex_constraints={1: list(range(nt)), 2: list(range(nz*nt, (nz + 1)*nt))},
        bodies=[pyse.Body(faces=range(len(f)), volume=np.pi*radius**2)]))
    return ev


def test_relax_reports_stable_equilibria(cube):
    r = cube.relax()
    assert r.converged and r.stable is True and r.negative_modes == 0
    s = cube.stability()
    assert s.stable and s.lowest[0] > s.threshold     # the translations aren't instabilities


def test_relax_warns_about_unstable_equilibria():
    ev = _column(0.45)                    # height 1 < pi*0.45: stable
    assert ev.relax().stable is True
    ev = _column(0.25)                    # height 1 > pi*0.25: unstable
    with pytest.warns(pyse.UnstableEquilibriumWarning):
        r = ev.relax()
    assert r.converged and r.stable is False and r.negative_modes >= 1
    assert ev.relax(stability=False).stable is None


def _coarse_bridge(gap=0.2, theta=40.0, volume=0.05, nz=4, nt=16):
    """A liquid bridge between unit spheres, started from a coarse tube (by
    default volume 0.05, contact angle 40, gap 0.2, 16 x 4: there, long gradient
    runs let the contact-line vertices slide together until the surface breaks)."""
    from pysurfaceevolver import constraints as C
    c = 1 + gap/2
    lo, hi = 0.0, 0.999                   # the tube radius that holds the volume
    for _ in range(60):
        a = (lo + hi)/2
        v = 2*np.pi*(c*a*a - 2/3*(1 - (1 - a*a)**1.5))
        lo, hi = (a, hi) if v < volume else (lo, a)
    zb = c - np.sqrt(1 - a*a)
    v = [[a*np.cos(p), a*np.sin(p), z]
         for z in np.linspace(-zb, zb, nz + 1) for p in 2*np.pi*np.arange(nt)/nt]
    f = []
    for i in range(nz):
        for j in range(nt):
            p, q = i*nt + j, i*nt + (j + 1) % nt
            f += [[p, q, q + nt], [p, q + nt, p + nt]]
    s1 = C.sphere((0, 0, -c), 1.0, contact_angle=theta, wet_poles=("north",))
    s2 = C.sphere((0, 0, c), 1.0, contact_angle=theta, wet_poles=("south",))
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        np.array(v), f, constraints={1: s1, 2: s2},
        vertex_constraints={1: list(range(nt)), 2: list(range(nz*nt, (nz + 1)*nt))},
        bodies=[pyse.Body(faces=range(len(f)), volume=volume, volconst=s1.volconst + s2.volconst)]))
    return ev, s1.energy_constant + s2.energy_constant


def test_relax_hands_over_to_newton_before_the_mesh_drifts():
    ev, const = _coarse_bridge()
    r = ev.relax()
    assert r.converged and len(r.energy) < 200
    assert abs(ev.total_energy + const - 0.0806) < 1e-3      # the axisymmetric 0.0839 when refined


def test_relax_undoes_a_round_that_breaks_the_mesh():
    # on this 8 x 3 start the contact line collapses within one round of 10
    # steps (the residual jumps 100-fold); relax() undoes it and Newton finishes
    ev, _ = _coarse_bridge(gap=0.4, theta=100.0, volume=0.3, nz=3, nt=8)
    r = ev.relax()
    assert r.converged and ev.mesh_quality().angle_min > 20


def _hemisphere_drop(nr=8, nt=32):
    """A hemispherical drop of volume 2 pi/3 on the plane z = 0, its contact
    angle the parameter theta (90)."""
    from pysurfaceevolver import constraints as C
    v, f = [[0, 0, 1.0]], []
    for i in range(1, nr + 1):
        t = np.pi/2*i/nr
        v += [[np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)]
              for p in 2*np.pi*np.arange(nt)/nt]
    ring = lambda i, j: 1 + (i - 1)*nt + j % nt
    f += [[0, ring(1, j), ring(1, j + 1)] for j in range(nt)]
    for i in range(1, nr):
        for j in range(nt):
            f += [[ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)],
                  [ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)]]
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        np.array(v), f, constraints={1: C.plane((0, 0, 1), 0.0, contact_angle="theta")},
        vertex_constraints={1: [ring(nr, j) for j in range(nt)]}, parameters={"theta": 90.0},
        bodies=[pyse.Body(faces=range(len(f)), volume=2*np.pi/3)]))
    return ev


def test_relax_falls_back_to_conjugate_gradients():
    # spreading to a flat 10 degree drop, plain gradients crawl and stop far
    # from where Newton can take over; the conjugate-gradient pass gets there
    ev = _hemisphere_drop()
    ev.relax()
    ev.parameters["theta"] = 10.0
    snap = ev.save()
    assert not ev.relax(cg=False).converged
    ev.restore(snap)
    r = ev.relax()
    assert r.converged and r.newton_steps > 1
