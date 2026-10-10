"""Mesh maintenance: tiny-edge deletion keeps the surface a manifold."""

import numpy as np
import pysurfaceevolver as pyse


def _film(vertices, faces):
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(np.asarray(vertices, float), faces))
    return ev


def test_tiny_edge_kept_when_its_ends_share_other_neighbours():
    # A thin triangle (v1, v2, x) holding two vertices p, q, and a triangle
    # (v2, v1, y) below it. The facets on the short edge v1-v2 have third
    # vertices p and y, but v1 and v2 also share q and x: merging them would
    # make parallel edges. Evolver 2.70a merged them anyway and lost the surface.
    v1, v2, p, q, x, y = range(6)
    ev = _film([[0, 0, 0], [0.01, 0, 0], [0.005, 0.3, 0], [0.005, 0.6, 0],
                [0.005, 1, 0], [0.005, -1, 0]],
               [[v1, v2, p], [v1, p, q], [v2, q, p], [v1, q, x], [v2, x, q], [v2, v1, y]])
    ev.command("t 0.02")
    assert ev.counts == {"vertices": 6, "edges": 11, "facets": 6, "bodies": 0}
    assert ev.command("check").strip() == ""


def test_tiny_edge_deleted_when_safe():
    # A hexagon fan whose centre is split into two close vertices c, d: their
    # only common neighbours are the third vertices of the two facets on c-d.
    ring = [[np.cos(a), np.sin(a), 0] for a in np.radians(np.arange(0, 360, 60))]
    c, d = 6, 7
    faces = [[d, 0, 1], [c, d, 1], [c, 1, 2], [c, 2, 3], [c, 3, 4], [c, 4, 5],
             [c, 5, d], [d, 5, 0]]
    ev = _film(ring + [[0, 0, 0], [0.005, 0, 0]], faces)
    ev.command("t 0.02")
    assert ev.counts["vertices"] == 7 and ev.counts["facets"] == 6
    assert ev.command("check").strip() == ""



def _split_fan(radius=1.0, lift=0.0):
    # the hexagon fan above, its centre split into c and d (d lifted by `lift`)
    ring = [[radius*np.cos(a), radius*np.sin(a), 0] for a in np.radians(np.arange(0, 360, 60))]
    c, d = 6, 7
    faces = [[d, 0, 1], [c, d, 1], [c, 1, 2], [c, 2, 3], [c, 3, 4], [c, 4, 5],
             [c, 5, d], [d, 5, 0]]
    return _film(ring + [[0, 0, 0], [0.01, 0, lift]], faces)


def test_collapse_guard_refuses_merges_that_tilt_facets():
    # merging c and d at their midpoint tilts the facets around them by 7.5
    # to 14.3 degrees (d starts lifted by 0.008 over a fan of radius 0.03)
    ev = _split_fan(radius=0.03, lift=0.008)
    ev.command("collapse_max_tilt := 12")
    ev.command("t 0.015")
    assert ev.counts["vertices"] == 8
    ev.command("collapse_max_tilt := 15")
    ev.command("t 0.015")
    assert ev.counts["vertices"] == 7 and ev.command("check").strip() == ""


def test_collapse_guard_refuses_merges_that_make_long_edges():
    ev = _split_fan()
    ev.command("collapse_max_edge := 0.5")       # the ring is at distance 1
    ev.command("t 0.02")
    assert ev.counts["vertices"] == 8
    ev.command("collapse_max_edge := 0")
    ev.command("t 0.02")
    assert ev.counts["vertices"] == 7

def test_wire_vertices_add_no_spurious_hessian_modes():
    # A catenoid between two wire rings (radius 1, z = +-0.3) is stable. Its wire
    # vertices can only slide along the wires: a mesh mode, and an energy
    # maximum for even spacing, which used to show as one negative eigenvalue
    # per wire vertex. They take no Newton freedom now (hessian_slant_cutoff).
    nt, nz, H = 24, 6, 0.3
    v = [[np.cos(p), np.sin(p), z] for z in np.linspace(-H, H, nz + 1)
         for p in 2*np.pi*np.arange(nt)/nt]
    f = []
    for i in range(nz):
        for j in range(nt):
            a, b = i*nt + j, i*nt + (j + 1) % nt
            f += [[a, b, b + nt], [a, b + nt, a + nt]]
    bottom, top = list(range(nt)), list(range(nz*nt, (nz + 1)*nt))
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: "x^2 + y^2 = 1", 2: f"z = {H}", 3: f"z = {-H}"},
        vertex_constraints={1: bottom + top, 2: top, 3: bottom}))
    assert ev.relax().converged
    counts = ev.eigen_counts()
    assert counts.negative == 0
    assert sum(counts) == (nz - 1)*nt           # one (normal) freedom per inner vertex
