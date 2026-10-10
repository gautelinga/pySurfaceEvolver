"""Shared parts of the stress suite: results, shapes, helpers.

Every case runs with pySE's default settings: ``ev.relax()`` without options,
stepped with ``recipes.continuation``. No case may tune the solver; anything a
case needs beyond the defaults is a defect the suite is there to show.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import pysurfaceevolver as pyse


@dataclass
class Result:
    case: str
    passed: bool
    summary: str                         # one line: what was checked, how it went
    error: float = float("nan")          # the worst relative error against the reference
    seconds: float = 0.0
    gradient_steps: int = 0
    newton_steps: int = 0
    unconverged_steps: int = 0           # continuation steps whose relax() didn't converge
    min_angle: float = float("nan")      # smallest facet angle at the end, degrees
    negative_eigenvalues: Optional[int] = None   # at the end (None: not checked)
    notes: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)   # per step, for plots


class Tracker:
    """Collects what the default relaxation did over a run."""

    def __init__(self):
        self.start = time.perf_counter()
        self.gradient = 0
        self.newton = 0
        self.unconverged = 0

    def relax(self, ev) -> Any:
        r = ev.relax()
        self.add(r)
        return r

    def add(self, r) -> None:
        self.gradient += len(r.energy)
        self.newton += r.newton_steps
        if not r.converged:
            self.unconverged += 1

    def finish(self, ev, result: Result, eigen: bool = True) -> Result:
        result.seconds = time.perf_counter() - self.start
        result.gradient_steps = self.gradient
        result.newton_steps = self.newton
        result.unconverged_steps = self.unconverged
        try:
            result.min_angle = ev.mesh_quality().angle_min
        except Exception as e:      # an invalid surface at the end
            result.notes.append(f"mesh quality unavailable: {e}")
        if eigen:
            try:
                result.negative_eigenvalues = ev.eigen_counts().negative
            except Exception as e:
                result.notes.append(f"eigen_counts failed: {e}")
        return result


# ---- shapes --------------------------------------------------------------------

def cap(r: float = 1.0, h: float = 1.0, nr: int = 8, nt: int = 32, z0: float = 0.0):
    """A spherical cap of sphere radius r and height h on the plane z = z0,
    normals outward; returns vertices, faces, rim rows."""
    a = np.sqrt(h*(2*r - h))
    t_max = np.arcsin(a/r) if h <= r else np.pi - np.arcsin(a/r)
    v, f = [[0, 0, z0 + h]], []
    for i in range(1, nr + 1):
        t = t_max*i/nr
        v += [[r*np.sin(t)*np.cos(p), r*np.sin(t)*np.sin(p), z0 + h - r + r*np.cos(t)]
              for p in 2*np.pi*np.arange(nt)/nt]
    ring = lambda i, j: 1 + (i - 1)*nt + j % nt
    f += [[0, ring(1, j), ring(1, j + 1)] for j in range(nt)]
    for i in range(1, nr):
        for j in range(nt):
            f += [[ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)],
                  [ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)]]
    return np.array(v), f, [ring(nr, j) for j in range(nt)]


def tube(radius: float, z0: float, z1: float, nz: int = 8, nt: int = 24):
    """A cylinder r = radius from z0 to z1, normals outward; returns vertices,
    faces, bottom-ring rows, top-ring rows."""
    v = [[radius*np.cos(p), radius*np.sin(p), z]
         for z in np.linspace(z0, z1, nz + 1) for p in 2*np.pi*np.arange(nt)/nt]
    f = []
    for i in range(nz):
        for j in range(nt):
            a, b = i*nt + j, i*nt + (j + 1) % nt
            f += [[a, b, b + nt], [a, b + nt, a + nt]]
    return np.array(v), f, list(range(nt)), list(range(nz*nt, (nz + 1)*nt))


def cap_reference(volume: float, theta_deg: float) -> Dict[str, float]:
    """A sessile spherical cap (no gravity) of given volume and contact angle:
    energy (area - cos(theta) * wetted area), contact radius, height."""
    t = np.radians(theta_deg)
    shape = np.pi*(2 - 3*np.cos(t) + np.cos(t)**3)/3      # V = shape * R^3
    R = (volume/shape)**(1/3)
    h, a = R*(1 - np.cos(t)), R*np.sin(t)
    return dict(energy=2*np.pi*R*h - np.cos(t)*np.pi*a*a, radius=a, height=h, R=R)


def contact_radius(ev, constraint: int) -> float:
    """Mean distance from the z axis of the vertices on a constraint."""
    m = ev.mesh()
    on = ev.on_constraint(constraint)
    return float(np.hypot(m.vertices[on, 0], m.vertices[on, 1]).mean())


def rel(a: float, b: float) -> float:
    return abs(a/b - 1)


def revolve(r, z, nt: int = 24):
    """A surface of revolution about the z axis through the meridian points
    (r_i, z_i), i from the bottom ring to the top ring; normals point away from
    the axis. Returns vertices, faces, bottom-ring rows, top-ring rows."""
    r, z = np.asarray(r, float), np.asarray(z, float)
    phis = 2*np.pi*np.arange(nt)/nt
    v = [[ri*np.cos(p), ri*np.sin(p), zi] for ri, zi in zip(r, z) for p in phis]
    f = []
    for i in range(len(r) - 1):
        for j in range(nt):
            a, b = i*nt + j, i*nt + (j + 1) % nt
            f += [[a, b, b + nt], [a, b + nt, a + nt]]
    return np.array(v), f, list(range(nt)), list(range((len(r) - 1)*nt, len(r)*nt))
