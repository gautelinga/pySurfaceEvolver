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
