"""Topology changes done with Evolver's own commands (see docs/PLAN.md,
"Future: topology changes")."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes


def _pieces(ev):
    """The surface's edge-connected pieces, each as (the volume it encloses
    with the plane z = z0 of its rim, z0 = 0 or 3, the body of its facets)."""
    m = ev.mesh()
    f, x = np.asarray(m.facets), ev.vertices
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    _, edge = np.unique(e, axis=0, return_inverse=True)
    edge = edge.ravel().reshape(3, -1).T
    label = np.arange(len(f))                 # facets sharing an edge, by relabelling
    while True:
        low = np.full(edge.max() + 1, len(f))
        np.minimum.at(low, edge.ravel(), np.repeat(label, 3))
        new = low[edge].min(axis=1)
        if np.array_equal(new, label):
            break
        label = new
    out = []
    for c in np.unique(label):
        ff = f[label == c]
        z0 = 0.0 if x[ff, 2].min() < 0.5 else 3.0
        a, b, d = (x[ff[:, i]] - [0, 0, z0] for i in range(3))
        body = np.unique(m.facet_bodies[label == c])
        out.append((np.einsum("ij,ij->i", a, np.cross(b, d)).sum()/6, z0, body[body > 0]))
    return out


def test_a_bridge_that_pinches_off_splits_into_two_drops_with_their_volumes():
    # a bridge between disks 3 apart, pinned at their rims (the top one closes
    # the body through its content integral), loses liquid until it pinches:
    # Evolver's tiny-edge deletion shrinks the neck to a vertex, "pop vertex"
    # splits it, and "rebody" makes the two pieces bodies, each holding its
    # volume. The top rim's content belongs to the upper drop after that, also
    # when the rim is refined
    H, nt = 3.0, 16
    tube = shapes.tube(1.0, 0.0, H, nz=8, nt=nt)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   boundaries={1: C.circle((0, 0, 0)),
                               2: C.circle((0, 0, H), content=(f"-{H}*y/2", f"{H}*x/2", "0"))},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=np.pi*H)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ev.relax().converged
        full = ev.body(1).volume
        v = 1.0
        for _ in range(100):                      # 2% of the liquid at a time
            if len(_pieces(ev)) == 2:
                break
            x = ev.vertices
            middle = np.abs(x[:, 2] - H/2) < 0.9
            if np.hypot(x[middle, 0], x[middle, 1]).min() > 0.3:
                v -= 0.02
                ev.body(1).target = v*full
            ev.command("g 10")
            ev.remesh(target=0.25)
        assert len(_pieces(ev)) == 2 and 0.3 < v < 0.6
        before = ev.body(1).volume
        ev.command("pop vertex")
        ev.command("rebody")
        assert len(ev.bodies().ids) == 2
        assert ev.bodies().volume.sum() == pytest.approx(before, rel=1e-12)
        for volume, z0, body in _pieces(ev):
            assert len(body) == 1
            assert ev.body(int(body[0])).volume == pytest.approx(volume, rel=1e-10)
            assert ev.body(int(body[0])).target == pytest.approx(volume, rel=1e-10)
        assert ev.relax(levels=1).converged
    for volume, z0, body in _pieces(ev):
        assert ev.body(int(body[0])).volume == pytest.approx(volume, rel=1e-8)
        # the drop is a cap on the rim: its pressure is 4h/(1 + h^2)
        h = np.roots([np.pi/6, 0, np.pi/2, -volume]).real.max()
        assert ev.body(int(body[0])).pressure == pytest.approx(4*h/(1 + h*h), rel=0.02)
