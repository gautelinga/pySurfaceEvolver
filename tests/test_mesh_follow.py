"""Newton steps carry the mesh along a moving line: a contact or triple line
that travels drags the vertices beside it along their surface, instead of
inverting the facets between them (PYSE_NO_FOLLOW=1 turns it off)."""

import os
import subprocess
import sys

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C


def _cap_factor(t):
    """A spherical cap's volume over pi r^3, r its rim's radius, t its angle."""
    return (2 - 3*np.cos(t) + np.cos(t)**3)/(3*np.sin(t)**3)


def neumann_rim(volume, gw, gt, gb):
    """The exact rim of a weightless lens of this volume (tensions: bath, top,
    bottom), and the angles of its caps with the bath."""
    tt = np.arccos((gw*gw + gt*gt - gb*gb)/(2*gw*gt))
    tb = np.arccos((gw*gw + gb*gb - gt*gt)/(2*gw*gb))
    return (volume/(np.pi*(_cap_factor(tt) + _cap_factor(tb))))**(1/3), tt, tb


def quarter_lens(volume, gw, gt, gb, rim, nr=12, nt=16, nb=20):
    """A lens on a bath held flat on a ring at three times the exact rim, in the
    quarter x, y >= 0 between mirror planes; started as two caps of the same
    angle (the mean of the exact ones) on a rim of radius ``rim``."""
    exact, tt, tb = neumann_rim(volume, gw, gt, gb)
    lo, hi = 1e-3, np.pi/2
    for _ in range(100):                       # the caps' angle that holds the volume
        a = (lo + hi)/2
        lo, hi = (a, hi) if 2*np.pi*rim**3*_cap_factor(a) < volume else (lo, a)
    R = rim/np.sin(a)
    phi = np.linspace(0, np.pi/2, nt + 1)
    radii_in = rim*np.arange(1, nr + 1)/nr
    radii_out = rim + (3*exact - rim)*(np.arange(1, nb + 1)/nb)**1.5
    pts, index = [], {}

    def point(key, xyz):
        index[key] = len(pts)
        pts.append(xyz)
    point(("top", 0, 0), [0, 0, R - R*np.cos(a)])
    point(("bottom", 0, 0), [0, 0, -(R - R*np.cos(a))])
    for i, r in enumerate(radii_in, 1):
        z = np.sqrt(R*R - r*r) - R*np.cos(a)
        for j, p in enumerate(phi):
            xy = [r*np.cos(p), r*np.sin(p)]
            if i == nr:                        # the rim: shared
                point(("rim", 0, j), xy + [0.0])
            else:
                point(("top", i, j), xy + [z])
                point(("bottom", i, j), xy + [-z])
    for i, r in enumerate(radii_out, 1):
        for j, p in enumerate(phi):
            point(("bath", i, j), [r*np.cos(p), r*np.sin(p), 0.0])

    def ring(side, i, j):
        if side in ("top", "bottom") and i == nr:
            return index[("rim", 0, j)]
        if side == "bath" and i == 0:
            return index[("rim", 0, j)]
        return index[(side, i, j)] if i else index[(side, 0, 0)]
    faces, kinds = [], []
    for side, n in (("top", nr), ("bottom", nr), ("bath", nb)):
        start = 0 if side != "bath" else 0
        for i in range(start, n):
            for j in range(nt):
                if side != "bath" and i == 0:
                    tri = [[ring(side, 0, 0), ring(side, 1, j), ring(side, 1, j + 1)]]
                else:
                    a0, a1 = ring(side, i, j), ring(side, i, j + 1)
                    b0, b1 = ring(side, i + 1, j), ring(side, i + 1, j + 1)
                    tri = [[a0, b0, b1], [a0, b1, a1]]
                for t in tri:
                    faces.append(t[::-1] if side == "bottom" else t)   # normals out of the lens
                    kinds.append(side)
    v, f, kinds = np.array(pts, float), np.array(faces), np.array(kinds)
    lens = np.flatnonzero(kinds != "bath")
    on_x = np.flatnonzero(np.abs(v[:, 0]) < 1e-12)
    on_y = np.flatnonzero(np.abs(v[:, 1]) < 1e-12)
    fixed = np.flatnonzero(np.abs(np.hypot(v[:, 0], v[:, 1]) - radii_out[-1]) < 1e-9)
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints={"mirror_x": C.mirror("x"), "mirror_y": C.mirror("y")},
                   vertex_constraints={"mirror_x": on_x, "mirror_y": on_y}, fixed=fixed,
                   bodies=[pyse.Body(volume=volume/4, faces=lens, volconst=0.0)])
    ev.set_values("facet", "tension", np.where(kinds == "top", gt, np.where(kinds == "bottom", gb, gw)))
    return ev, exact


def rim_radius(ev):
    """The mean distance from the axis of the vertices the lens and bath share."""
    m = ev.mesh()
    lens = (m.facet_bodies == 1).any(axis=1)
    shared = np.intersect1d(np.unique(m.facets[lens]), np.unique(m.facets[~lens]))
    return float(np.hypot(*ev.vertices[shared, :2].T).mean())


@pytest.mark.parametrize("start", [0.9, 1.05])
def test_a_lens_rim_travels_to_its_neumann_angles(start):
    # near spreading (S = -0.02) the energy hardly changes as the rim moves:
    # gradient rounds settle with it short of its place, and Newton's steps
    # moved it by more than the facets beside it, which inverted
    T = (1.0, 0.55, 0.47)
    ev, exact = quarter_lens(1.0, *T, rim=start*neumann_rim(1.0, *T)[0])
    r = ev.relax(levels=1)
    assert r.converged and ev.mesh_quality().angle_min > 5
    assert rim_radius(ev) == pytest.approx(exact, rel=2e-3)


FOLLOW_OFF = r"""
import os, sys, json, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
from test_mesh_follow import quarter_lens, neumann_rim, rim_radius
T = (1.0, 0.55, 0.47)
ev, exact = quarter_lens(1.0, *T, rim=0.9*neumann_rim(1.0, *T)[0])
r = ev.relax(levels=1)
print(json.dumps([bool(r.converged), rim_radius(ev)/exact]))
"""


def test_without_following_the_lens_rim_stops_short():
    # the same lens with PYSE_NO_FOLLOW=1 (read once per process): the case
    # the test above guards
    env = dict(os.environ, PYSE_NO_FOLLOW="1", OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", FOLLOW_OFF, os.path.dirname(__file__)],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr
    converged, ratio = __import__("json").loads(out.stdout.strip().splitlines()[-1])
    assert not converged or abs(ratio - 1) > 2e-3


def test_converged_newton_steps_are_unchanged(load):
    # near equilibrium the steps don't slip past the mesh: following leaves
    # them alone, bit for bit (one thread), so Newton converges as it did
    runs = []
    with pyse.threads_limit(1):
        for off in (False, True):
            code = (r"import os, sys, json; " + ("os.environ['PYSE_NO_FOLLOW'] = '1'; " if off else "")
                    + r"""
import warnings; warnings.simplefilter("ignore")
import pysurfaceevolver as pyse
ev = pyse.Evolver("mound.fe")
ev.command("g 10; r; g 10; r; g 40")
out = []
for _ in range(3):
    ev.command("hessian")
    out.append(ev.vertices.tolist())
print(json.dumps(out))
""")
            res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 cwd=os.path.join(os.path.dirname(__file__), "..", "fe"),
                                 env=dict(os.environ, OMP_NUM_THREADS="1"), timeout=600)
            assert res.returncode == 0, res.stderr
            runs.append(np.array(__import__("json").loads(res.stdout.strip().splitlines()[-1])))
    assert np.array_equal(runs[0], runs[1])
