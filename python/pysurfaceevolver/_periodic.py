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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    return (f"torus\nperiods\n{float(L[0])!r} 0 0\n0 {float(L[1])!r} 0\n0 0 {Z_PERIOD!r}\n")


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
        if shape is None:
            raise ValueError(f"constraint {key}: with periods, walls must come from "
                             "pysurfaceevolver.constraints (a floor plane and spheres)")
        if shape[0] == "plane":
            n = (None if isinstance(shape[1], str) or any(isinstance(v, str) for v in shape[1])
                 else np.asarray(shape[1], dtype=float))
            if n is None or abs(n[0]) > 1e-12 or abs(n[1]) > 1e-12 or n[2] <= 0:
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
    flux = [key for key, spec in cons.items()
            if getattr(spec, "flux", None) is not None or getattr(spec, "slabflux", None)]
    if flux:
        return _assemble_flux(cons, flux, floor_key, z_f, k_f)
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
    ids = np.asarray(m.vertex_ids)
    order = np.argsort(ids, kind="stable")      # (ids aren't sorted after changes)
    rows = order[np.searchsorted(ids, np.asarray(vertex_ids).ravel(), sorter=order)]
    L = np.asarray(periods)
    shift = np.round((np.atleast_2d(points)[:, :2] - m.vertices[rows, :2])/L).astype(int)
    cmds = [f"wrap_vertex({int(v)},{wrap_code(s)})"
            for v, s in zip(np.asarray(vertex_ids).ravel(), shift) if s.any()]
    if cmds:
        ev.command("; ".join(cmds))


def _assemble_flux(cons, flux, floor_key, z_f, k_f) -> Tuple[Dict[Any, Any], str]:
    """assemble() with spheres of gauge="flux": their fields periodic (the
    lattice sums, mirrored in the floor), all integrands but the floor's
    evaluated in C (pyse_slabflux.c, registered at load: slab_flux); the
    datafile's are placeholders."""
    from .constraints import ANGLE_FLOOR, ANGLE_FLOOR_TOP
    spheres = [k for k, spec in cons.items() if k != floor_key]
    if set(spheres) != set(flux):
        raise ValueError("with periods, spheres with gauge='flux' can't be mixed with other "
                         "spheres: mixed gauges (give them all gauge='flux')")
    small = False
    wet = bool(k_f)
    out = dict(cons)
    for key in spheres:
        spec = cons[key]
        data = spec.slabflux or spec.flux
        c, R, angle, tension = data
        if isinstance(angle, str):
            raise ValueError(f"constraint {key}: in a periodic slab, a flux sphere's contact "
                             "angle must be a number")
        if angle is not None and np.cos(np.radians(float(angle))) != 0:
            wet = True
            small |= 0 < float(angle) < ANGLE_FLOOR_TOP
        out[key] = dataclasses.replace(spec, flux=None, slabflux=tuple(data), slab=True,
                                       native=None, content=["0", "0", "0"])
    if k_f and k_f.expr is not None:
        if isinstance(cons[floor_key].contact_angle, str):
            raise ValueError("in a periodic slab with flux spheres, the floor's contact angle "
                             "must be a number")
        small = True
    for key in spheres:
        if wet:
            out[key] = dataclasses.replace(out[key], energy=["0", "0", "0"])
    if floor_key is not None:
        out[floor_key] = dataclasses.replace(cons[floor_key], energy=None, content=None,
                                             slab=True, native=None)
    # (three Gauss points a facet, as _flux's)
    lines = ["integral_order_2d 2\n",
             f"method_instance {VOLUME_METHOD} method facet_vector_integral\n",
             "vector_integrand:\nq1: 0\nq2: 0\n",
             f"q3: z - {_n(z_f)}\n" if z_f else "q3: z\n",
             f'volume_method_name "{VOLUME_METHOD}"\n']
    if wet:
        lines += [f"quantity {WETTING_QUANTITY} energy method facet_vector_integral global\n",
                  "vector_integrand:\nq1: 0\nq2: 0\nq3: 0\n"]
    if small:
        lines.append(f"// (C raises small angles to {ANGLE_FLOOR})\n")
    return out, "".join(lines)


# the fast sum's settings (pyse_slabflux.c): boxes of side ~2h (h a quarter of
# the smallest radius), sources within kappa box radii summed directly, the
# rest to degree p; a sphere's within kappa_w radii, to degree p_w; explicit
# images |n| <= M, the tail per layer to degree p_T. Errors ~1e-11 of the
# field (scratch tests against Ewald's sum, 2026-10-24).
SLAB_FLUX = dict(kappa=6.0, p=10, kappa_w=5.0, p_w=14, M=3, p_T=16)

_FIT: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _monomials(p: int):
    return [(a, b, n - a - b) for n in range(p + 1) for a in range(n, -1, -1)
            for b in range(n - a, -1, -1)]


def _fit(p: int):
    """The tail's fit at degree p: sample directions (on the unit sphere),
    harmonic polynomials H (monomials x basis; no constant) and the
    pseudo-inverse taking the samples of grad (3 a point) to H's
    coefficients."""
    if p in _FIT:
        return _FIT[p]
    mono = _monomials(p)
    index = {m: i for i, m in enumerate(mono)}
    cols = []
    for n in range(1, p + 1):
        ms = [m for m in mono if sum(m) == n]
        lo = [m for m in mono if sum(m) == n - 2]
        if n < 2:
            ns = np.eye(len(ms))
        else:
            lap = np.zeros((len(lo), len(ms)))
            for j, m in enumerate(ms):
                for q in range(3):
                    if m[q] >= 2:
                        t = list(m)
                        t[q] -= 2
                        lap[lo.index(tuple(t)), j] += m[q]*(m[q] - 1)
            _, sv, vt = np.linalg.svd(lap)
            ns = vt[int((sv > 1e-10*sv[0]).sum()):].T
        for v in ns.T:
            col = np.zeros(len(mono))
            for j, m in enumerate(ms):
                col[index[m]] = v[j]
            cols.append(col)
    H = np.array(cols).T
    nh = H.shape[1]
    k = np.arange(2*nh) + 0.5
    phi, th = np.arccos(1 - 2*k/(2*nh)), np.pi*(1 + 5**0.5)*k
    u = np.c_[np.cos(th)*np.sin(phi), np.sin(th)*np.sin(phi), np.cos(phi)]
    G = np.zeros((len(u), 3, len(mono)))
    for j, m in enumerate(mono):
        for q in range(3):
            if m[q]:
                e = list(m)
                e[q] -= 1
                G[:, q, j] = m[q]*u[:, 0]**e[0]*u[:, 1]**e[1]*u[:, 2]**e[2]
    A = (G @ H).reshape(-1, nh)
    _FIT[p] = (u, H, np.linalg.pinv(A))
    return _FIT[p]


def slab_flux(walls: Dict[Any, Any], number: Dict[Any, int], periods, bodies) -> bool:
    """Set up the periodic flux gauge's fields in C and register the
    integrands (after a load; number: wall key -> constraint number). False
    where there is nothing to set up."""
    from . import _core
    beads = [(k, w) for k, w in walls.items() if getattr(w, "slabflux", None)]
    if not beads:
        return False
    floor = [w for w in walls.values() if getattr(w, "shape", (None,))[0] == "plane"
             and not isinstance(w.shape[1], str) and float(w.shape[1][2]) > 0.5]
    zf, fa, ft = float("nan"), float("nan"), 1.0
    if floor:
        zf = float(floor[0].shape[2])
        if floor[0].contact_angle is not None:
            fa = float(floor[0].contact_angle)
        ft = float(floor[0].frame[4]) if floor[0].frame else 1.0
    rows = []
    for _, w in beads:
        c, R, angle, tension = w.slabflux
        rows += [*map(float, c), float(R), float("nan") if angle is None else float(angle),
                 float(tension)]
    radii = [float(w.slabflux[1]) for _, w in beads]
    h = min(radii)/4
    s = SLAB_FLUX
    u, H, pinv = _fit(int(s["p_T"]))
    sources: List[float] = []
    if periods is not None:
        lattice, grid = list(map(float, periods)), [0.0, 0.0, *map(float, periods)]
    else:
        # without periods (_flux.assemble_fast): each sphere's source and its
        # images in the floor and, in a box, in its walls (their fields' flux
        # through the walls 0: a lattice of twice the box), the grid over the
        # box; else the spheres' bounds, every source outside it
        from ._flux import _fast_walls
        layout = _fast_walls(walls)
        if layout is None:
            raise RuntimeError("the fast flux sum: walls it can't take")
        box = layout[1]
        c = np.array([w.slabflux[0] for _, w in beads], float)
        for i, (_, w) in enumerate(beads):
            xs, ys = [c[i, 0]], [c[i, 1]]
            if box is not None:
                xs.append(2*box[0] - c[i, 0])
                ys.append(2*box[1] - c[i, 1])
            zs = [c[i, 2]] + ([2*zf - c[i, 2]] if floor else [])
            for a, x in enumerate(xs):
                for b, y in enumerate(ys):
                    for k, z in enumerate(zs):
                        sources += [x, y, z, float(i), float(a == b == k == 0)]
        if box is not None:
            lattice = [2*(box[2] - box[0]), 2*(box[3] - box[1])]
            # (the whole lattice cell: no point falls outside it in x, y)
            grid = [box[0], box[1], *lattice]
        else:
            lattice = [0.0, 0.0]
            lo, hi = (c - np.array(radii)[:, None]).min(axis=0), \
                (c + np.array(radii)[:, None]).max(axis=0)
            m = 3*max(radii)
            grid = [lo[0] - m, lo[1] - m, hi[0] - lo[0] + 2*m, hi[1] - lo[1] + 2*m]
    # every source but a sphere's own must lie outside it: its field's flux
    # through the sphere's wetted part is a line integral only then (a sphere
    # crossing the floor or a box's wall has its mirror inside it)
    srcs = (np.array(sources, float).reshape(-1, 5) if sources else np.array(
        [[*w.slabflux[0], i, 1.0] for i, (_, w) in enumerate(beads)]
        + ([[w.slabflux[0][0], w.slabflux[0][1], 2*zf - w.slabflux[0][2], i, 0.0]
            for i, (_, w) in enumerate(beads)] if floor else []), float))
    for j, (key, w) in enumerate(beads):
        d = srcs[:, :3] - np.asarray(w.slabflux[0], float)
        if lattice[0] > 0:
            d[:, :2] -= np.asarray(lattice)*np.round(d[:, :2]/np.asarray(lattice))
        inside = (np.linalg.norm(d, axis=1) < radii[j]) & ~((srcs[:, 3] == j) & (srcs[:, 4] != 0))
        if inside.any():
            raise ValueError(f"constraint {key}: a sphere with gauge='flux' must not cross the "
                             "floor, a box's wall or another such sphere (another source, or "
                             "its own image, inside it)")
    ok = _core.slabflux_setup(lattice, zf, rows, fa, ft,
                              [h, s["kappa"], s["p"], s["kappa_w"], s["p_w"], s["M"], s["p_T"]],
                              list(u.ravel()), int(H.shape[1]), list(H.ravel()),
                              list(pinv.ravel()), sources, grid)
    if not ok:
        raise RuntimeError("the periodic flux gauge's fields couldn't be set up (memory?)")
    keys = [(VOLUME_METHOD, 0, 0, 0, 0)] + [(f"body_{int(b)}_vol_meth", 0, 0, 0, 0)
                                             for b in bodies]
    done = [_core.slabflux_key(*k) for k in keys]
    if not done[0]:
        raise RuntimeError("the periodic flux gauge's volume couldn't be registered")
    for name in (f"{WETTING_QUANTITY}1_facet_vector_integral", f"{WETTING_QUANTITY}_inst",
                 WETTING_QUANTITY):
        if _core.slabflux_key(name, 0, 0, 1, 0):
            break
    for j, (key, w) in enumerate(beads):
        if key not in number:
            continue
        if w.energy is not None and not _core.slabflux_key("", number[key], 0, 2, j):
            raise RuntimeError(f"the periodic flux gauge: wall {key}'s energy not registered")
        if not _core.slabflux_key("", number[key], 1, 3, j):
            raise RuntimeError(f"the periodic flux gauge: wall {key}'s content not registered")
    return True
