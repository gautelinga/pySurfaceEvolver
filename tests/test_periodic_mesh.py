"""A periodic slab's mesh: geometry, export, plots and the liquid's clusters
the same as for the surface without periods (its feature across the cell's
sides), and a layer across the cell drawn as one cell's worth."""

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import _clusters as K, constraints as C, shapes

from test_periodic import PERIODS, _bead, _drop, _into_cell

L = np.asarray(PERIODS)
_TOL = 1e-11      # (the runs with and without periods part at round-off)


def _moved(a, b):
    """b moved by whole periods onto a (as its first point lies)."""
    b = np.array(b, dtype=float)
    b[:, :2] += np.round((a[0, :2] - b[0, :2])/L)*L
    return b


def _pair(make, commands=(), measure=lambda ev: ev):
    """measure(ev) of the surface without periods and in the slab (each
    measured before the next is loaded: one engine)."""
    out = []
    for periods in (None, PERIODS):
        ev = make(periods)
        for c in commands:
            ev.command(c)
        out.append(measure(ev))
    return out


def _sheet(z, up=True, n=(12, 14)):
    """A flat sheet across the cell at height z, in nearly equilateral
    triangles joined round the cell's sides, facing up (or down)."""
    nx, ny = n
    dx, dy = L[0]/nx, L[1]/ny
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    v = np.c_[((i + (j % 2)/2)*dx).ravel(), (j*dy).ravel(), np.full(i.size, z)]

    def k(a, b):
        return (b % ny)*nx + (a % nx)
    f = []
    for b in range(ny):
        for a in range(nx):
            if b % 2 == 0:
                f += [(k(a, b), k(a + 1, b), k(a, b + 1)),
                      (k(a + 1, b), k(a + 1, b + 1), k(a, b + 1))]
            else:
                f += [(k(a, b), k(a + 1, b + 1), k(a, b + 1)),
                      (k(a, b), k(a + 1, b), k(a + 1, b + 1))]
    f = np.array(f)
    return v, (f if up else f[:, [0, 2, 1]])


def _layer(*heights, floor=True):
    """Liquid across the cell under sheets at the heights, the lower one
    facing down (the liquid between them), on a floor z = 0 with ``floor``."""
    vs, fs, n = [], [], 0
    for i, z in enumerate(heights):
        v, f = _sheet(z, up=(i == len(heights) - 1))
        vs.append(v)
        fs.append(f + n)
        n += len(v)
    cons = {"bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0)} if floor else {}
    ev = pyse.Evolver()
    ev.load_arrays(np.concatenate(vs), np.concatenate(fs), constraints=cons,
                   bodies=[pyse.Body(volume=None)], periods=PERIODS)
    ev.body(1).target = ev.body(1).volume
    return ev


def _pockets(periods):
    """Two drops on the floor in one body: one on the cell's corner, one in
    its middle."""
    cap = shapes.cap(0.5, 0.3, nr=6, nt=24)
    v = cap.vertices.copy()
    rim = np.asarray(cap.rims["rim"])
    v[rim, 2] = 0.0
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(np.r_[v, v + [0.95, 1.15, 0]], periods),
                   np.r_[cap.faces, cap.faces + len(v)],
                   constraints={"bottom": C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
                   vertex_constraints={"bottom": np.r_[rim, rim + len(v)]},
                   bodies=[pyse.Body(volume=None)], periods=periods)
    return ev


def _drop_measures(ev):
    m = ev.mesh()
    return dict(mesh=m, quality=m.quality(), valence=m.edge_valence(), tess=m.tessellate(),
                meshio=m.to_meshio(), capped=ev.body_surfaces(cap=True)[1],
                volume=ev.body(1).volume,
                slices=[m.slice(nrm, (0, 0, 0.1)) for nrm in ((0, 0, 1), (0, 1, 0), (1, 1, 0))],
                lines=ev.contact_lines("bottom"), angles=ev.contact_angles("bottom"))


def test_a_drop_across_the_corner_measures_as_without_periods():
    a, b = _pair(_drop, ["g 5"], _drop_measures)
    assert a["mesh"].periods is None and b["mesh"].periods == PERIODS
    assert np.ptp(b["mesh"].vertices[:, 0]) > 1.5           # (wrapped into the cell)
    assert b["mesh"].edge_lengths().max() < 0.2
    qa, qb = a["quality"], b["quality"]
    for name in ("edge_min", "edge_max", "angle_min", "angle_median"):
        assert getattr(qb, name) == pytest.approx(getattr(qa, name), rel=1e-9)
    assert (qa.skinny, qa.degenerate) == (qb.skinny, qb.degenerate)
    assert np.array_equal(a["valence"], b["valence"])
    # drawn whole, as without periods
    (pa, ta), (pb, tb) = a["tess"], b["tess"]
    assert np.array_equal(ta, tb)
    assert np.abs(pa - _moved(pa, pb)).max() < _TOL
    ma, mb = a["meshio"], b["meshio"]
    assert np.abs(ma.points - _moved(ma.points, mb.points)).max() < _TOL
    # the body capped on the floor: closed, its volume Evolver's
    sa, sb = a["capped"], b["capped"]
    assert sb.watertight and sb.open_loops == sa.open_loops == 1
    assert sb.cap_constraints == sa.cap_constraints == {1: 1}
    assert sb.volume == pytest.approx(b["volume"], rel=1e-12)
    assert sb.volume == pytest.approx(sa.volume, rel=1e-12)
    assert sb.area == pytest.approx(sa.area, rel=1e-12)
    # sections round the corner
    for ca, cb in zip(a["slices"], b["slices"]):
        assert len(ca) and [len(c) for c in ca] == [len(c) for c in cb]
        assert max(np.abs(x - _moved(x, y)).max() for x, y in zip(ca, cb)) < _TOL
    # the contact line in one piece, the angles
    la, lb = a["lines"], b["lines"]
    assert len(la) == len(lb) == 1 and np.abs(la[0] - _moved(la[0], lb[0])).max() < _TOL
    assert np.abs(lb[0][-1] - lb[0][0]).max() < _TOL           # closed
    np.testing.assert_allclose(b["angles"], a["angles"], atol=1e-10)


@pytest.mark.parametrize("model", ["quadratic", "lagrange 3"])
def test_curved_facets_across_the_corner_are_drawn_whole(model):
    def measure(ev):
        m = ev.mesh()
        return (m.tessellate(), m.to_meshio("native", flavor="vtk"),
                m.body_surfaces("native")[1], ev.body_surfaces(cap=True)[1],
                m.slice((0, 0, 1), (0, 0, 0.1)))
    a, b = _pair(_drop, ["g 3", model], measure)
    (pa, ta), (pb, tb) = a[0], b[0]
    assert np.array_equal(ta, tb) and np.abs(pa - _moved(pa, pb)).max() < _TOL
    na, nb = a[1], b[1]
    assert np.array_equal(na.cells[0].data, nb.cells[0].data)
    assert np.abs(na.points - _moved(na.points, nb.points)).max() < _TOL
    assert (b[2].open_loops, b[2].watertight) == (a[2].open_loops, a[2].watertight) == (1, False)
    assert b[3].watertight and b[3].volume == pytest.approx(a[3].volume, rel=1e-12)
    ca, cb = a[4], b[4]
    assert len(ca) == len(cb) == 1 and np.abs(ca[0] - _moved(ca[0], cb[0])).max() < _TOL


def test_a_bridge_to_a_bead_across_the_corner_caps_on_the_bead():
    # (the bead's cap projected onto it at the image its vertices are at)
    def measure(ev):
        return (ev.body_surfaces(cap=True)[1],
                {k: (ev.contact_lines(k), ev.contact_angles(k)) for k in ("bead", "bottom")})
    (sa, lines_a), (sb, lines_b) = _pair(_bead, ["g 10"], measure)
    assert sb.watertight and sb.open_loops == 2
    assert sb.cap_constraints == sa.cap_constraints and set(sb.cap_constraints.values()) == {1, 2}
    # (the caps' points the same; their strips may zip differently on
    # round-off, the bead's curved)
    assert np.abs(sa.points - _moved(sa.points, sb.points)).max() < _TOL
    np.testing.assert_allclose(sb.areas(), sa.areas(), rtol=1e-5)
    assert sb.volume == pytest.approx(sa.volume, rel=1e-3)
    for name in ("bead", "bottom"):
        (la, angle_a), (lb, angle_b) = lines_a[name], lines_b[name]
        assert len(la) == len(lb) == 1 and np.abs(la[0] - _moved(la[0], lb[0])).max() < _TOL
        np.testing.assert_allclose(angle_b, angle_a, atol=1e-10)


def test_a_drop_whose_vertices_lie_at_other_images_is_drawn_whole():
    # (vertices moved by whole periods: the drop still drawn in one piece,
    # capped where its vertices are)
    from pysurfaceevolver import _periodic
    ev = _bead(PERIODS)
    before = ev.body_surfaces(cap=True)[1]
    m = ev.mesh()
    free = np.flatnonzero(~ev.on_constraint("bead") & ~ev.on_constraint("bottom"))[::3]
    _periodic.move_near(ev, m.vertex_ids[free], m.vertices[free] + [3*L[0], -2*L[1], 0])
    m = ev.mesh()
    assert np.ptp(m.vertices[:, 0]) > 3*L[0]
    points, _ = m.tessellate()
    assert np.ptp(points[:, 0]) < 1.2 and np.ptp(points[:, 1]) < 1.2
    s = ev.body_surfaces(cap=True)[1]
    assert s.watertight and s.cap_constraints == {1: 1, 2: 2}
    assert s.volume == pytest.approx(before.volume, rel=1e-12)


def test_a_layer_across_the_cell_is_one_cells_worth():
    ev = _layer(0.4)
    m = ev.mesh()
    points, tris = m.tessellate()
    assert len(tris) == len(m.facets) and len(points) > len(m.vertices)     # seams twice
    middle = points[tris].mean(axis=1)[:, :2]
    assert (middle >= 0).all() and (middle < L).all()
    corners = points[tris]
    area = 0.5*np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                       corners[:, 2] - corners[:, 0]), axis=1)
    assert area.sum() == pytest.approx(L.prod(), rel=1e-12)
    assert m.facet_areas().sum() == pytest.approx(L.prod(), rel=1e-12)
    # on a floor: no region of its own (open to the floor), no loops to cap
    s = m.body_surfaces(cap=True)[1]
    assert not s.watertight and s.open_loops == 0 and np.isnan(s.volume)
    # a section across it, around the point, one cell long
    (curve,) = m.slice((1, 0, 0), (0.3, 0.2, 0))
    assert np.ptp(curve[:, 1]) == pytest.approx(L[1], rel=1e-12)
    assert np.allclose(curve[:, 0], 0.3) and np.allclose(curve[:, 2], 0.4)
    # between two films: a region, z dx dy
    s = _layer(0.2, 0.8, floor=False).body_surfaces()[1]
    assert s.watertight and s.open_loops == 0
    assert s.volume == pytest.approx(0.6*L.prod(), rel=1e-12)
    with pytest.raises(ValueError, match="round the periodic cell"):
        s.volume_mesh()


def test_the_liquid_layer_is_one_cluster_reaching_the_floor():
    c = K.clusters(_layer(0.4), "bottom")
    assert len(c.bodies) == 1 and c.outlet[0] and c.mixed == 0
    assert c.nodes[0]*c.spacing**3 == pytest.approx(0.4*L.prod(), rel=0.25)
    assert c.walls["bottom"][0] > 0
    # two films: one liquid between them, gas below
    c = K.clusters(_layer(0.2, 0.8), "bottom")
    assert len(c.bodies) == 1 and not c.outlet[0] and c.mixed == 0
    assert c.nodes[0]*c.spacing**3 == pytest.approx(0.6*L.prod(), rel=0.25)


def test_pockets_across_the_corner_are_one_cluster_each():
    plain, slab = _pair(_pockets, measure=lambda ev: K.clusters(ev, "bottom"))
    assert len(slab.bodies) == len(plain.bodies) == 2
    assert slab.outlet.all() and slab.mixed == 0
    np.testing.assert_allclose(slab.nodes*slab.spacing**3, plain.nodes*plain.spacing**3,
                               rtol=0.05)


def test_a_ring_round_a_bead_on_the_corner_is_one_cluster():
    # (the bead's images in the cell solid, round the grid)
    (a, _), (b, volume) = _pair(_bead, measure=lambda ev: (K.clusters(ev, "bottom"),
                                                           ev.body(1).volume))
    assert len(b.bodies) == 1 and b.outlet[0] and b.mixed == 0
    assert set(b.walls) == set(a.walls) == {"bead", "bottom"}
    assert b.nodes[0]*b.spacing**3 == pytest.approx(volume, rel=0.25)
    assert b.nodes[0]*b.spacing**3 == pytest.approx(a.nodes[0]*a.spacing**3, rel=0.05)


def test_what_a_slab_cant_do():
    m = _drop(PERIODS).mesh()
    with pytest.raises(ValueError, match="periodic slab"):
        m.mirrored(["x"])
    assert "periods" in m._repr_html_()
    with pytest.raises(ValueError, match="periods"):
        pyse.Evolver().load_mesh_file("nowhere.stl", volume="current", periods=PERIODS)


def test_plotting_a_slab_draws_it_whole(can_render, tmp_path):
    from pysurfaceevolver._viz import surface_dataset
    da, db = _pair(_drop, ["g 3"], lambda ev: surface_dataset(ev, "area")[0])
    assert np.abs(da.points - _moved(da.points, db.points)).max() < _TOL
    slab = _drop(PERIODS)
    slab.plot(off_screen=True, walls=True, screenshot=str(tmp_path / "slab.png"))
    slab.command("quadratic")
    view = slab.live_view("x", element="vertex", off_screen=True, walls=True)
    slab.iterate(2, callback=view.update)
    assert view.fast_updates == 2
    fresh, _ = surface_dataset(slab, "x", element="vertex")
    np.testing.assert_allclose(view.dataset.points, fresh.points, atol=1e-13)
    view.close()
