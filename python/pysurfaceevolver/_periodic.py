"""Slabs periodic in x and y: Evolver's torus model, with a z period far
beyond any surface (Z_PERIOD), so nothing wraps in z.

Coordinates aren't wrapped into the cell: a vertex may sit at any image, and
each edge's wrap (Evolver's per-period +, *, -) says which image of its head
it runs to. Every edge is shorter than half a period, so the wrap is the
minimum image of the difference of its ends' coordinates (``wraps``), and
geometry from differences (``delta``) needs no wraps at all.

The walls must not see the periods: their integrands are evaluated at a
vertex's coordinates, or along an edge from its tail's. So:

* the volume is Evolver's z dx dy as a facet integral of (0, 0, z - z_f)
  (z_f the floor's height): invariant under the periods, and zero on the
  floor; walls' contents as without periods (a sphere's measured from its
  centre, at the image its vertices are kept at), with z_f's part on each
  sphere, z_f (0, 0, 1)'s flux through its wetted part;
* a floor's (a plane z = z_f facing up) own wetted area isn't its line
  integral (t ds changes by a period's length times ds across the cell's
  side): it is the flux of z-hat out of the liquid through its free surface
  (a facet integral), less that through the other walls' wetted parts, on
  each sphere the line integral of B = (x - c) x z-hat/2 along its contact
  lines (curl B = -z-hat: Evolver's orientation of a constraint's line
  integral, as _flux's monopoles, whose curl is minus their source's field);
* spheres keep their vertices at the image near their centre (``home``), so
  their own integrands, local to them, need nothing else.

Planes other than one floor (walls x = const, a lid) and cylinders aren't
periodic-invariant walls here.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

Z_PERIOD = 1.0e6          # (nothing wraps in z: far beyond any surface)

VOLUME_METHOD = "pyse_slab_volume"
WETTING_QUANTITY = "pyse_slab_wetting"


def periods_of(periods) -> np.ndarray:
    """(Lx, Ly) as floats, checked."""
    p = np.asarray(periods, dtype=float).ravel()
    if p.shape != (2,) or not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("periods must be (Lx, Ly), two positive numbers: the slab's cell")
    return p


def delta(a, b, periods) -> np.ndarray:
    """b - a at the nearest images in x and y (rows of points, or one)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if periods is None:
        return d
    L = np.asarray(periods, dtype=float)
    d = np.array(d, dtype=float)
    d[..., :2] -= L*np.round(d[..., :2]/L)
    return d


def wraps(tails, heads, x: np.ndarray, periods) -> np.ndarray:
    """Each edge's wrap, (m, 2) integers: the periods to add to its head's
    coordinates for its true end, x[head] + w L - x[tail] the minimum image.
    Raises where an edge is too long for that to be clear."""
    L = np.asarray(periods, dtype=float)
    d = x[np.asarray(heads), :2] - x[np.asarray(tails), :2]
    w = -np.round(d/L).astype(int)
    rest = np.abs(d + w*L)
    if np.any(rest > 0.45*L):
        raise ValueError("an edge is nearly half a period long: the periodic cell is too "
                         "small for the mesh (refine it, or make the cell larger)")
    return w


def unwrap_loop(rows: Sequence[int], x: np.ndarray, periods) -> np.ndarray:
    """A face's corners at one image: the first's coordinates, the others
    by minimum-image steps along the loop."""
    p = [np.asarray(x[rows[0]], dtype=float)]
    for a, b in zip(rows[:-1], rows[1:]):
        p.append(p[-1] + delta(x[a], x[b], periods))
    return np.array(p)


def home(x: np.ndarray, centre, periods) -> np.ndarray:
    """Points moved to the image nearest centre."""
    return np.asarray(centre, dtype=float) + delta(centre, x, periods)


def header(periods) -> str:
    L = periods_of(periods)
    return (f"torus\nperiods\n{L[0]!r} 0 0\n0 {L[1]!r} 0\n0 0 {Z_PERIOD!r}\n")


def wrap_text(w) -> str:
    """Evolver's wrap signs for one edge (z never wraps)."""
    sign = {1: "+", 0: "*", -1: "-"}
    if any(abs(int(k)) > 1 for k in w):
        raise ValueError("an edge wraps more than once: the cell is too small for the mesh")
    return " ".join(sign[int(k)] for k in w) + " *"


def _n(x: float) -> str:
    x = float(x)
    if x == 0:
        return "0"
    return f"({x!r})" if x < 0 else repr(x)


def assemble(cons: Dict[Any, Any], periods) -> Tuple[Dict[Any, Any], str]:
    """The walls made periodic-invariant (see the module notes), and the
    datafile lines for the slab's facet integrals (before the vertices).
    Walls made so before (``slab``) are kept as they are."""
    from .constraints import _Coefficient
    periods_of(periods)
    floor_key, z_f, k_f = None, 0.0, None
    for key, spec in cons.items():
        shape = getattr(spec, "shape", None)
        if getattr(spec, "flux", None) is not None:
            raise ValueError(f"constraint {key}: spheres with gauge='flux' aren't periodic yet")
        if shape is None:
            raise ValueError(f"constraint {key}: with periods, walls must come from "
                             "pysurfaceevolver.constraints (a floor plane and spheres)")
        if shape[0] == "plane":
            n = np.asarray(shape[1], dtype=float)
            if isinstance(shape[1], str) or abs(n[0]) > 1e-12 or abs(n[1]) > 1e-12 or n[2] <= 0:
                raise ValueError(f"constraint {key}: with periods, the only plane may be a "
                                 "floor z = const facing up (normal (0, 0, 1))")
            if floor_key is not None:
                raise ValueError("with periods, one floor only (a lid isn't supported yet)")
            floor_key, z_f = key, float(shape[2])
            k_f = _Coefficient(spec.contact_angle, spec.frame[4] if spec.frame else 1.0)
        elif shape[0] != "sphere" or any(isinstance(v, str) for v in shape[1]):
            raise ValueError(f"constraint {key}: with periods, walls may be a floor plane and "
                             "spheres with fixed centres")
    ke = None
    if k_f:
        ke = f"(-{k_f.times(1.0)})" if k_f.expr is not None else _n(-float(k_f.value or 0.0))
    out = dict(cons)
    for key, spec in cons.items():
        if getattr(spec, "slab", False):
            continue
        if key == floor_key:
            # its own line integrals go: the wetting is the facets' and the
            # spheres', the content zero in the volume's gauge
            out[key] = dataclasses.replace(spec, energy=None, content=None, slab=True,
                                           native=None)
            continue
        c = np.asarray(spec.shape[1], dtype=float)
        X, Y = f"(x - {_n(c[0])})", f"(y - {_n(c[1])})"
        B = [f"({Y}/2)", f"(-{X}/2)", "0"]       # (x - c) x z-hat/2: curl -z-hat, as _flux's
        energy = list(spec.energy) if spec.energy is not None else None
        content = list(spec.content) if spec.content is not None else None
        changed = False
        if ke is not None:
            energy = energy or ["0", "0", "0"]
            energy = [e if b == "0" else f"{e} + {ke}*{b}" for e, b in zip(energy, B)]
            changed = True
        if z_f != 0:
            # the volume's gauge z - z_f: -z_f z-hat's flux through the wetted part
            content = content or ["0", "0", "0"]
            content = [e if b == "0" else f"{e} + {_n(-z_f)}*{b}" for e, b in zip(content, B)]
            changed = True
        out[key] = dataclasses.replace(spec, energy=energy, content=content, slab=True,
                                       native=None if changed else spec.native)
    lines = [f"method_instance {VOLUME_METHOD} method facet_vector_integral\n",
             "vector_integrand:\nq1: 0\nq2: 0\n",
             f"q3: z - {_n(z_f)}\n" if z_f else "q3: z\n",
             f'volume_method_name "{VOLUME_METHOD}"\n']
    if ke is not None:
        lines += [f"quantity {WETTING_QUANTITY} energy method facet_vector_integral global\n",
                  "vector_integrand:\nq1: 0\nq2: 0\n", f"q3: {ke}\n"]
    return out, "".join(lines)


def wrap_code(shift) -> int:
    """Evolver's wrap (group element) for a shift by whole periods (kx, ky):
    TWRAPBITS = 6 bits a period, two's complement in 5."""
    kx, ky = (int(k) for k in shift)
    return (kx & 0o37) | ((ky & 0o37) << 6)


def move_near(ev, vertex_ids, points) -> None:
    """Move vertices to their images nearest points (rows), their edges'
    wraps kept right (Evolver's wrap_vertex): before putting a vertex on a
    sphere, whose vertices are kept near its centre."""
    periods = ev.periods
    if periods is None:
        return
    m = ev.mesh()
    rows = np.searchsorted(m.vertex_ids, np.asarray(vertex_ids))
    L = np.asarray(periods)
    shift = np.round((np.atleast_2d(points)[:, :2] - m.vertices[rows, :2])/L).astype(int)
    cmds = [f"wrap_vertex({int(v)},{wrap_code(s)})"
            for v, s in zip(np.asarray(vertex_ids).ravel(), shift) if s.any()]
    if cmds:
        ev.command("; ".join(cmds))
