"""Starting shapes for :func:`pysurfaceevolver.make_datafile`.

Each builder returns a :class:`Shape`: vertices, triangles (normals outward,
away from the axis or up), and its rims by name (rows of the vertices along
each open edge, in order around it), ready for ``vertex_constraints``::

    from pysurfaceevolver import shapes, constraints as C
    drop = shapes.cap(radius=1, height=0.6)
    ev.load_arrays(drop.vertices, drop.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60)},
                   vertex_constraints={1: drop.rims["rim"]},
                   bodies=[pyse.Body(volume=0.5)])

The start needn't have the target volume: relaxing meets it (an open cube of
volume 1 relaxes to targets from 0.2 to 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

__all__ = ["Shape", "cap", "disc", "tube", "revolve", "box"]


@dataclass
class Shape:
    """A triangulated starting shape: ``vertices`` (n, 3), ``faces`` (k, 3)
    vertex rows, ``rims``: name -> vertex rows along an open edge."""

    vertices: np.ndarray
    faces: np.ndarray
    rims: Dict[str, List[int]] = field(default_factory=dict)


def _bands(rings: int, nt: int, offset: int = 0) -> List[List[int]]:
    """Triangles between consecutive rings of nt vertices each."""
    f = []
    for i in range(rings - 1):
        for j in range(nt):
            a, b = offset + i*nt + j, offset + i*nt + (j + 1) % nt
            f += [[a, b, b + nt], [a, b + nt, a + nt]]
    return f


def revolve(r, z, nt: int = 24) -> Shape:
    """A surface of revolution about the z axis through the meridian points
    (r_i, z_i), from the bottom ring to the top ring; normals away from the
    axis. Rims ``"bottom"`` and ``"top"``. A point with r = 0 at either end
    closes it there (a pole) instead of a ring."""
    r, z = np.asarray(r, float), np.asarray(z, float)
    if r.shape != z.shape or r.ndim != 1 or len(r) < 2:
        raise ValueError("r and z must be 1-D arrays of the same length (at least 2)")
    if nt < 3:
        raise ValueError("nt must be at least 3")
    phis = 2*np.pi*np.arange(nt)/nt
    v: List[List[float]] = []
    f: List[List[int]] = []
    rims: Dict[str, List[int]] = {}
    pole = np.abs(r) <= 1e-12*np.abs(r).max(initial=1.0)
    lo = 1 if pole[0] else 0
    hi = len(r) - 1 if pole[-1] else len(r)
    if hi - lo < 1 or pole[lo:hi].any():
        raise ValueError("r may be 0 only at the ends (poles), with at least one ring between")
    if lo:
        v.append([0.0, 0.0, z[0]])
    first = len(v)
    v += [[ri*np.cos(p), ri*np.sin(p), zi] for ri, zi in zip(r[lo:hi], z[lo:hi]) for p in phis]
    f += _bands(hi - lo, nt, first)
    last = first + (hi - lo - 1)*nt
    if lo:
        f += [[0, first + (j + 1) % nt, first + j] for j in range(nt)]
    else:
        rims["bottom"] = list(range(first, first + nt))
    if hi < len(r):
        v.append([0.0, 0.0, z[-1]])
        top = len(v) - 1
        f += [[last + j, last + (j + 1) % nt, top] for j in range(nt)]
    else:
        rims["top"] = list(range(last, last + nt))
    return Shape(np.array(v), np.array(f, dtype=np.int64), rims)


def cap(radius: float = 1.0, height: float = 1.0, *, nr: int = 8, nt: int = 32,
        z0: float = 0.0) -> Shape:
    """A spherical cap of sphere ``radius`` and ``height`` standing on the
    plane z = ``z0`` (a drop's start); rim ``"rim"`` on that plane."""
    R, h = float(radius), float(height)
    if not (R > 0 and 0 < h < 2*R):
        raise ValueError("need radius > 0 and 0 < height < 2*radius")
    t_max = np.arccos(1 - h/R)
    t = t_max*np.arange(nr, -1, -1)/nr            # from the rim up to the pole
    shape = revolve(R*np.sin(t), z0 + h - R + R*np.cos(t), nt)
    shape.rims = {"rim": shape.rims["bottom"]}
    return shape


def disc(radius: float = 1.0, *, nr: int = 4, nt: int = 24, z: float = 0.0) -> Shape:
    """A flat disc in the plane z = ``z`` (a film's start), normal +z; rim
    ``"rim"``."""
    if nr < 1:
        raise ValueError("nr must be at least 1")
    rr = float(radius)*np.arange(nr + 1)[::-1]/nr
    shape = revolve(rr, np.full(nr + 1, float(z)), nt)     # (normal up)
    shape.rims = {"rim": shape.rims["bottom"]}
    return shape


def tube(radius: float = 1.0, z0: float = 0.0, z1: float = 1.0, *, nz: int = 8,
         nt: int = 24) -> Shape:
    """A cylinder r = ``radius`` from z0 to z1 (a liquid bridge's start),
    normals outward; rims ``"bottom"`` and ``"top"``."""
    return revolve(np.full(nz + 1, float(radius)), np.linspace(z0, z1, nz + 1), nt)


def box(size=(1.0, 1.0, 1.0), *, open_bottom: bool = False, n: int = 1) -> Shape:
    """An axis-aligned box from the origin to ``size``, each face split into
    n x n squares of two triangles, normals outward. With ``open_bottom``,
    no face at z = 0, and the rim there is ``"rim"`` (a drop or a
    film on a table)."""
    sx, sy, sz = (float(s) for s in size)
    if n < 1:
        raise ValueError("n must be at least 1")
    index: Dict[tuple, int] = {}
    v: List[List[float]] = []

    def vid(p) -> int:
        key = tuple(int(round(c)) for c in p)
        if key not in index:
            index[key] = len(v)
            v.append([key[0]*sx/n, key[1]*sy/n, key[2]*sz/n])
        return index[key]

    f: List[List[int]] = []
    # each face: origin corner, two in-face directions (counterclockwise seen from outside)
    faces = [((0, 0, n), (1, 0, 0), (0, 1, 0)),     # top, +z
             ((0, 0, 0), (0, 1, 0), (1, 0, 0)),     # bottom, -z
             ((0, 0, 0), (1, 0, 0), (0, 0, 1)),     # y = 0, -y
             ((0, n, 0), (0, 0, 1), (1, 0, 0)),     # y = n, +y
             ((0, 0, 0), (0, 0, 1), (0, 1, 0)),     # x = 0, -x
             ((n, 0, 0), (0, 1, 0), (0, 0, 1))]     # x = n, +x
    for k, (o, a, b) in enumerate(faces):
        if open_bottom and k == 1:
            continue
        o_, a_, b_ = (np.array(x) for x in (o, a, b))
        for i in range(n):
            for j in range(n):
                p00, p10 = o_ + i*a_ + j*b_, o_ + (i + 1)*a_ + j*b_
                p01, p11 = o_ + i*a_ + (j + 1)*b_, o_ + (i + 1)*a_ + (j + 1)*b_
                q = [vid(p00), vid(p10), vid(p11), vid(p01)]
                f += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    rims = {}
    if open_bottom:              # in order around the rim
        rim = [i for i, p in enumerate(v) if p[2] == 0]
        x = np.array(v)[rim]
        angle = np.arctan2(x[:, 1] - sy/2, x[:, 0] - sx/2)
        rims["rim"] = [rim[k] for k in np.argsort(angle, kind="stable")]
    return Shape(np.array(v), np.array(f, dtype=np.int64), rims)
