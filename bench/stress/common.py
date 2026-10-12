"""Shared parts of the stress suite: results, exact caps, helpers (starting
shapes come from ``pysurfaceevolver.shapes``).

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
    calls: List[Dict[str, Any]] = field(default_factory=list)  # Tracker.calls
    extra: Dict[str, Any] = field(default_factory=dict)        # case-specific, for figures


class Tracker:
    """Collects what the default relaxation did over a run."""

    def __init__(self):
        self.start = time.perf_counter()
        self.gradient = 0
        self.newton = 0
        self.unconverged = 0
        self.calls: List[Dict[str, Any]] = []   # per relax(), with the mesh after it

    def relax(self, ev, **options) -> Any:
        """ev.relax(**options) (no options: the defaults), recorded."""
        t = time.perf_counter()
        r = ev.relax(**options)
        seconds = time.perf_counter() - t
        self.add(r)
        m = ev.mesh()
        self.calls.append(dict(seconds=seconds, gradient=len(r.energy), newton=r.newton_steps,
                               converged=bool(r.converged), stable=r.stable,
                               negative_modes=r.negative_modes, residual=float(r.residual),
                               vertices=m.vertices.copy(),
                               facets=None if m.facets is None else m.facets.copy()))
        return r

    @property
    def last(self) -> int:
        """Index of the latest relax() call in ``calls``, for a row to refer to."""
        return len(self.calls) - 1

    def add(self, r) -> None:
        self.gradient += len(r.energy)
        self.newton += r.newton_steps
        if not r.converged:
            self.unconverged += 1

    def finish(self, ev, result: Result, eigen: bool = True) -> Result:
        result.seconds = time.perf_counter() - self.start
        result.calls = self.calls
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


# ---- references ----------------------------------------------------------------

def cap_reference(volume: float, theta_deg: float) -> Dict[str, float]:
    """A sessile spherical cap (no gravity) of given volume and contact angle:
    energy (area - cos(theta) * wetted area), contact radius, height."""
    t = np.radians(theta_deg)
    shape = np.pi*(2 - 3*np.cos(t) + np.cos(t)**3)/3      # V = shape * R^3
    R = (volume/shape)**(1/3)
    h, a = R*(1 - np.cos(t)), R*np.sin(t)
    return dict(energy=2*np.pi*R*h - np.cos(t)*np.pi*a*a, radius=a, height=h, R=R)


def contact_radius(ev, constraint: int) -> float:
    """Mean distance of the vertices on a constraint from their centre in x, y
    (a drop on a plane may slide sideways: that costs nothing)."""
    m = ev.mesh()
    on = ev.on_constraint(constraint)
    p = m.vertices[on, :2]
    return float(np.linalg.norm(p - p.mean(axis=0), axis=1).mean())


def rel(a: float, b: float) -> float:
    return abs(a/b - 1)
