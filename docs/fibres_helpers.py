"""Starting surfaces and references for docs/fibres.md: liquid on fibres of
radius 1 (surface tension 1, so a pressure is a curvature).

* ``barrel(theta, volume)``: the axisymmetric drop on a fibre (Carroll), from
  its profile; ``barrel_onset(theta)``: where it turns unstable to shifting
  sideways as it shrinks (exact); ``barrel_drop`` and ``clamshell``: the two
  shapes relaxed; ``shift(ev, axes)``: the eigenvalue of the mode that moves
  the liquid as a whole.
* ``princen(d, theta)``: the cross-section of a long column between two
  parallel fibres a gap 2d apart (Princen), and ``princen_limit(theta)`` the
  widest d with one; ``column_slice``, ``column_area`` and ``column_excess``:
  a slice of a long column's middle, to find it with pySE.
* ``crossed(delta, theta, volume)``: liquid at two fibres crossing at delta
  degrees; ``crossed_column``: the column model of Sauret et al.;
  ``wrap_angle``: how far round the fibres the liquid reaches.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes


# ---- one fibre: the barrel ----------------------------------------------------
#
# The fibre is the z axis, radius b = 1. The profile r(s) runs from the equator
# (s = 0, r = rm, psi = 0) to the contact line (r = 1), psi the angle of the
# tangent from the z axis: dz/ds = cos psi, dr/ds = sin psi. With the normal out
# of the liquid the curvatures are k_m = -dpsi/ds (meridian) and
# k_p = cos(psi)/r (parallel), their sum the pressure p. The first integral
# r cos(psi) - p r^2/2 = const gives p from the equator and the contact angle
# (psi = -theta at the fibre).

def _barrel_pressure(rm, theta):
    return 2*(rm - np.cos(np.radians(theta)))/(rm*rm - 1)


def _barrel_profile(rm, theta):
    """Integrate the half profile from the equator to the fibre: r, z, psi,
    free area, volume (outside the fibre), as functions of s."""
    p = _barrel_pressure(rm, theta)

    def rhs(s, y):
        r, z, psi = y[:3]
        return [np.sin(psi), np.cos(psi), np.cos(psi)/r - p,
                2*np.pi*r, np.pi*(r*r - 1)*np.cos(psi)]

    def hit(s, y):
        return y[0] - 1.0
    hit.terminal, hit.direction = True, -1
    sol = solve_ivp(rhs, (0, 40*rm + 40), [rm, 0.0, 0.0, 0.0, 0.0], events=hit,
                    rtol=1e-12, atol=1e-12)
    if sol.status != 1:
        raise ValueError("the profile never reaches the fibre")
    return sol, p


def barrel(theta, volume):
    """The barrel of this volume at contact angle ``theta`` (degrees) on a fibre
    of radius 1: dict with equator radius, length (between the contact lines),
    pressure and energy (free area - cos(theta) wetted area)."""
    def half(rm):
        sol, p = _barrel_profile(rm, theta)
        return sol.y[:, -1], p

    rm = brentq(lambda x: 2*half(x)[0][4] - volume, 1.0 + 1e-6, 1e3, xtol=1e-14)
    (r, z, psi, area, vol), p = half(rm)
    wet = 2*np.pi*2*z
    return dict(equator=rm, length=2*z, pressure=p,
                energy=2*area - np.cos(np.radians(theta))*wet)


def barrel_onset(theta):
    """Where the barrel at contact angle ``theta`` turns unstable as its volume
    shrinks (the drop shifting to one side): dict with volume, equator
    radius, length and pressure there.

    Shifting the drop sideways, u = n_x (its normal motion, cos(psi) cos(phi)
    on the profile), solves the Jacobi equation of the second variation: it is
    a rigid motion of the free surface. The contact line's condition for a
    motion that slides it along the fibre at a fixed angle is u' = q u with
    q = k_m cot(theta) (k_m the meridian's curvature there; the fibre's own
    curvature along its axis is zero; sliding along the fibre, u = n_z, meets
    it). The shift meets it only where k_m = 0 at the contact line: there it
    is a neutral mode, the lowest of its kind (it has no zero on the profile),
    and below that volume the shift lowers the energy. With the first
    integral, k_m = 0 at the fibre means the pressure equals cos(theta), the
    equator radius (1 + sin(theta))/cos(theta): the barrel turns unstable as
    the inflexion of its profile reaches the contact line.
    """
    t = np.radians(theta)
    rm = (1 + np.sin(t))/np.cos(t)
    sol, p = _barrel_profile(rm, theta)
    return dict(volume=2*sol.y[4, -1], equator=rm, length=2*sol.y[1, -1], pressure=p)


def barrel_drop(theta, volume, levels=2):
    """The barrel on a fibre along z, from a rough profile, relaxed."""
    ref = barrel(theta, volume)
    L, rm = ref["length"], ref["equator"]
    z = np.linspace(-L/2, L/2, 13)
    drop = shapes.revolve(np.sqrt(1 + (rm*rm - 1)*(1 - (2*z/L)**2)), z, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(drop.vertices, drop.faces,
                   constraints={"fibre": C.cylinder((0, 0, 0), (0, 0, 1), 1.0,
                                                    contact_angle=theta)},
                   vertex_constraints={"fibre": drop.rims["bottom"] + drop.rims["top"]},
                   bodies=[pyse.Body(volume=volume)])
    ev.relax(levels=levels)
    return ev


def shift(ev, axes=(0, 1), n=3):
    """The Hessian eigenvalue, among the ``n`` lowest, whose mode moves the
    liquid furthest as a whole along ``axes`` (by default sideways off a fibre
    along z)."""
    m = ev.modes(n)
    moved = [np.linalg.norm(v[:, list(axes)].mean(axis=0))/np.sqrt((v*v).sum(axis=1).mean())
             for v in m.vectors]
    return float(m.values[int(np.argmax(moved))])


def clamshell(theta, volume, levels=1):
    """A drop sitting on one side of a fibre along z (a clam shell), from a
    ball touching the fibre on a small patch, relaxed."""
    R = (3*volume/(4*np.pi))**(1/3)
    a = 0.7                                      # the patch's radius
    cap = shapes.cap(radius=R, height=R + np.sqrt(R*R - a*a), nr=8, nt=24)
    v = np.c_[cap.vertices[:, 2] + 1.0, cap.vertices[:, 0], cap.vertices[:, 1]]  # axis along x
    rim = cap.rims["rim"]
    v[rim, :2] /= np.hypot(v[rim, 0], v[rim, 1])[:, None]        # onto the fibre
    ev = pyse.Evolver()
    ev.load_arrays(v, cap.faces,
                   constraints={"fibre": C.cylinder((0, 0, 0), (0, 0, 1), 1.0,
                                                    contact_angle=theta)},
                   vertex_constraints={"fibre": rim}, bodies=[pyse.Body(volume=volume)])
    ev.relax(levels=levels)
    return ev



# ---- two parallel fibres: the column -------------------------------------------
#
# Fibres of radius 1 along z with their axes at x = +-c, c = 1 + d (a gap 2d).
# A cross-section of a column between them: on each fibre the liquid wets the
# arc within alpha of the point facing the other fibre, and two circular arcs
# (the menisci, curvature k: the pressure, positive when convex) close it above
# and below, meeting the fibres at theta.

def _section(alpha, c, theta, points=False):
    """A column's cross-section (alpha may be an array): pressure, area, free
    and wetted perimeter; with ``points``, the upper meniscus from the right
    fibre to the middle (x, y) instead."""
    th = np.radians(theta)
    Px, Py = c - np.cos(alpha), np.sin(alpha)
    # the meniscus leaves P along cos(theta) (along the fibre, to the wetted
    # side) + sin(theta) (out of the fibre)
    phi0 = np.arctan2(-np.cos(th)*np.cos(alpha) + np.sin(th)*np.sin(alpha),
                      -np.cos(th)*np.sin(alpha) - np.sin(th)*np.cos(alpha))
    k = np.sin(phi0)/Px                                    # it reaches x = 0 level
    turn = np.where(k > 0, np.pi - phi0, -np.pi - phi0)
    # the quarter's area, half the integral of x dy - y dx around it: along the
    # fibre from its inner point to P, then the meniscus (centre O, signed
    # radius 1/k, from the angle t1) to x = 0; nothing along the axes
    R = 1/k
    t1 = phi0 - np.pi/2
    t2 = t1 + turn
    Ox, Oy = Px - R*np.cos(t1), Py - R*np.sin(t1)
    if points:
        s = np.linspace(t1, t2, 200)
        return Ox + R*np.cos(s), Oy + R*np.sin(s)
    arc = R*R*turn + R*(Ox*(np.sin(t2) - np.sin(t1)) - Oy*(np.cos(t2) - np.cos(t1)))
    quarter = 0.5*(c*np.sin(alpha) - alpha + arc)
    return k, 4*quarter, 4*turn/k, 4*alpha


def princen(d, theta):
    """The cross-section in the middle of a long column between two fibres of
    radius 1 a gap 2d apart (Princen): where a column can end, its energy per
    length equals its pressure times its area (the force along the fibres
    through a cross-section is the same everywhere, and zero past the ends).
    Dict with pressure, area, alpha (degrees) and energy per length; None if
    there is no such column (the gap is too wide)."""
    c = 1.0 + d
    t = np.cos(np.radians(theta))

    def balance(alpha):
        k, A, free, wet = _section(alpha, c, theta)
        return free - t*wet - k*A
    alphas = np.r_[np.geomspace(1e-6, 0.05, 60), np.linspace(0.05, np.pi - 0.05, 300)[1:],
                   np.pi - np.geomspace(0.05, 1e-9, 100)[1:]]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sign(balance(alphas))
    i = np.flatnonzero(s[:-1]*s[1:] < 0)
    if len(i) == 0:
        return None
    alpha = brentq(balance, alphas[i[0]], alphas[i[0] + 1], xtol=1e-14)
    k, A, free, wet = _section(alpha, c, theta)
    return dict(pressure=float(k), area=float(A), alpha=float(np.degrees(alpha)),
                energy=float(free - t*wet))


def princen_limit(theta):
    """The widest half gap d at which a long column between parallel fibres of
    radius 1 exists (Princen; sqrt(2) at theta = 0, where the liquid wraps
    the fibres; a fold of the balance below that at larger angles)."""
    if princen(np.sqrt(2) - 1e-9, theta) is not None:
        return np.sqrt(2)
    return brentq(lambda d: 1.0 if princen(d, theta) is not None else -1.0,
                  0.05, np.sqrt(2) - 1e-9, xtol=1e-10)


def column_slice(d, theta, length=0.5, h=0.05):
    """A slice of a long uniform column between two parallel fibres of radius 1
    along x, their axes at y = +-(1 + d) in the plane z = 0, contact angle
    ``theta``: a quarter of the cross-section (mirror planes y = 0 and z = 0)
    between the mirror planes x = 0 and x = ``length``, so that it repeats
    along the fibres. Starts flat, its top at height 0.7, facets about h
    across, and relaxes it. Set its cross-section with :func:`column_excess`."""
    c = 1.0 + d
    z0 = 0.7
    Y = c - np.sqrt(1 - z0*z0)                 # the fibre's near side at that height
    nu, nx = int(np.ceil(Y/h)), int(np.ceil(length/h))
    v = np.array([[x, u*Y, z0] for u in np.linspace(0, 1, nu + 1)
                  for x in np.linspace(0, length, nx + 1)])
    f = _grid(nu, nx)[:, [0, 2, 1]]                         # normal +z, out of the liquid
    walls = {"fibre": C.cylinder((0, c, 0), (1, 0, 0), 1.0, contact_angle=theta,
                                 gauge="axial", ref=(0, -1, 0)),
             "mirror": C.mirror("y"), "end_0": C.mirror("x"), "end_1": C.mirror("x", length)}
    rows = {"fibre": np.flatnonzero(np.abs(v[:, 1] - Y) < 1e-12).tolist(),
            "mirror": np.flatnonzero(np.abs(v[:, 1]) < 1e-12).tolist(),
            "end_0": np.flatnonzero(np.abs(v[:, 0]) < 1e-12).tolist(),
            "end_1": np.flatnonzero(np.abs(v[:, 0] - length) < 1e-12).tolist()}
    u = -np.sqrt(1 - z0*z0)                    # the fibre's part of the flat start's quarter
    quarter = Y*z0 - (u*np.sqrt(1 - u*u) + np.arcsin(u) + np.pi/2)/2
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=walls, vertex_constraints=rows,
                   parameters={"slice_length": length},
                   bodies=[pyse.Body(volume=quarter*length)])
    ev.relax()
    return ev


def column_area(ev):
    """The cross-section's area of a :func:`column_slice`."""
    return 4*ev.bodies().volume[0]/ev.parameters["slice_length"]


def column_excess(ev, area):
    """Give a :func:`column_slice` this cross-section's area, relax it, and
    return its pressure minus its energy per volume: zero for the
    cross-section in the middle of a long column (Princen's balance)."""
    ev.body(1).target = area*ev.parameters["slice_length"]/4
    ev.relax()
    return ev.bodies().pressure[0] - ev.energy/ev.bodies().volume[0]


def _grid(nu, nv):
    """Triangles of an (nu+1) x (nv+1) grid of points numbered row by row,
    each normal along (next in i) x (next in j)."""
    idx = np.arange((nu + 1)*(nv + 1)).reshape(nu + 1, nv + 1)
    f = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            f += [[a, b, c], [a, c, d]]
    return np.array(f)


def _merge_close(v, f, tol=1e-9):
    """Merge coincident points."""
    keep, index = [], np.empty(len(v), dtype=int)
    for i, p in enumerate(v):
        for k in keep:
            if np.abs(v[k] - p).max() < tol:
                index[i] = index[k]
                break
        else:
            index[i] = len(keep)
            keep.append(i)
    return v[keep], index[f]


# ---- crossed fibres ------------------------------------------------------------
#
# Two fibres of radius 1 crossing at an angle delta, touching at the origin: the
# upper one along (cos(delta/2), sin(delta/2), 0) through (0, 0, 1), the lower
# one along (cos(delta/2), -sin(delta/2), 0) through (0, 0, -1). In the plane
# x = X they are ellipses centred at (X tan(delta/2), 1) and
# (-X tan(delta/2), -1), 1/cos(delta/2) wide, their surfaces
# 2 (sqrt(1 + (X tan(delta/2))^2) - 1) apart (Sauret et al.).

def _crossed_point(X, phi, upper, delta):
    """The point at angle phi around a fibre (0: the side facing the other)
    in the plane x = X."""
    h = np.radians(delta)/2
    s = 1.0 if upper else -1.0
    return np.array([X, s*X*np.tan(h) + np.sin(phi)/np.cos(h), s*(1 - np.cos(phi))])


def crossed_column(delta, theta, volume=None):
    """The column model of Sauret et al. for crossed fibres: each cross-section
    is a long column's (:func:`princen`) at the local gap. Returns the
    largest half length (where the gap reaches :func:`princen_limit`) and the
    volume the column holds then; with ``volume``, also the half length of a
    column holding that much."""
    t = np.tan(np.radians(delta)/2)
    dc = princen_limit(theta)*(1 - 1e-7)
    L_max = np.sqrt((1 + dc)**2 - 1)/t
    xs = L_max*np.sqrt(np.linspace(0, 1, 400))            # finer toward the ends
    A = np.array([princen(np.sqrt(1 + (x*t)**2) - 1, theta)["area"] if x > 0 else 0.0
                  for x in xs])
    V = 2*np.concatenate([[0.0], np.cumsum(np.diff(xs)*(A[1:] + A[:-1])/2)])
    out = dict(half_length_max=L_max, volume_max=V[-1])
    if volume is not None:
        out["half_length"] = float(np.interp(volume, V, xs)) if volume <= V[-1] else None
    return out


def crossed_fibres(delta, theta):
    """The two crossing fibres as constraints ``upper`` and ``lower``."""
    h = np.radians(delta)/2
    # angles around each fibre from the side facing the other ("axial" integrals)
    return {"upper": C.cylinder((0, 0, 1), (np.cos(h), np.sin(h), 0), 1.0, contact_angle=theta,
                                gauge="axial", ref=(0, 0, -1)),
            "lower": C.cylinder((0, 0, -1), (np.cos(h), -np.sin(h), 0), 1.0, contact_angle=theta,
                                gauge="axial", ref=(0, 0, 1))}


def wrap_angle(ev, delta):
    """How far round the crossed fibres the liquid reaches: the largest angle
    of a contact line from the side of its fibre facing the other (180: the
    far side)."""
    h = np.radians(delta)/2
    x = ev.vertices
    largest = 0.0
    for z, u in ((1.0, (np.cos(h), np.sin(h), 0.0)), (-1.0, (np.cos(h), -np.sin(h), 0.0))):
        q = x - [0.0, 0.0, z]
        radial = q - np.outer(q @ np.array(u), u)
        on = np.abs(np.linalg.norm(radial, axis=1) - 1) < 1e-6
        largest = max(largest, np.degrees(np.arccos(np.clip(-z*radial[on, 2], -1, 1))).max())
    return largest


def crossed(delta, theta, volume, half_length=2.0, h=0.25):
    """Liquid at the crossing of two fibres (see above), contact angle
    ``theta``: starts from a box between them, ``half_length`` each way
    along the x axis, its sides the planes y = +-w as wide apart as the
    fibres allow at its ends, and relaxes it to ``volume``."""
    th = np.radians(delta)/2
    w = 0.9*np.cos(th) - half_length*np.tan(th)
    if w < 0.1:
        raise ValueError("the box is too long for this angle: shorten half_length")
    Xs = np.linspace(-half_length, half_length, max(2, int(np.ceil(2*half_length/h))) + 1)

    def phi(X, y, upper):                # the angle at which a fibre reaches y
        return np.arcsin(np.cos(th)*(y - (1 if upper else -1)*X*np.tan(th)))

    def side(X, y, t):                   # straight up from the lower fibre to the upper one
        return ((1 - t)*_crossed_point(X, phi(X, y, False), False, delta)
                + t*_crossed_point(X, phi(X, y, True), True, delta))
    tall = max(np.linalg.norm(side(X, w, 1) - side(X, w, 0)) for X in Xs)
    taus = np.linspace(0, 1, max(2, int(np.ceil(tall/h))) + 1)
    pts, faces, count = [], [], 0

    def add(p, tri):
        nonlocal count
        pts.append(p)
        faces.append(tri + count)
        count += len(p)
    for y in (-w, w):                    # the two sides along the column
        p = np.array([[side(X, y, t) for t in taus] for X in Xs])
        add(p.reshape(-1, 3), _grid(len(Xs) - 1, len(taus) - 1))
    for X in (-half_length, half_length):   # the two ends, triangulated in their plane
        n = max(2, int(np.ceil(2*w/h)))
        lo = np.linspace(phi(X, -w, False), phi(X, w, False), n + 1)
        up = np.linspace(phi(X, w, True), phi(X, -w, True), n + 1)
        rim = np.array([_crossed_point(X, p, False, delta) for p in lo]
                       + [side(X, w, t) for t in taus[1:]]
                       + [_crossed_point(X, p, True, delta) for p in up[1:]]
                       + [side(X, -w, t) for t in taus[::-1][1:-1]])
        add(*_planar_patch(rim, h))
    v, f = _merge_close(np.vstack(pts), np.vstack(faces))
    # each facet's normal out of the liquid: away from the x axis, or out at an end
    for k, tri in enumerate(f):
        a, b2, c = v[tri]
        m = (a + b2 + c)/3
        end = abs(abs(m[0]) - half_length) < 1e-9
        out = np.array([np.sign(m[0]), 0.0, 0.0]) if end else np.array([0.0, m[1], 0.0])
        if np.cross(b2 - a, c - a) @ out < 0:
            f[k] = tri[[0, 2, 1]]
    walls = crossed_fibres(delta, theta)
    rows = {}
    for name, z in (("upper", 1.0), ("lower", -1.0)):
        axis = np.array(walls[name].shape[2])
        r = np.linalg.norm(np.cross(v - [0.0, 0.0, z], axis), axis=1)
        rows[name] = np.flatnonzero(np.abs(r - 1) < 1e-9).tolist()
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=walls, vertex_constraints=rows,
                   bodies=[pyse.Body(volume=volume)])
    ev.relax()
    return ev


def _planar_patch(rim, h):
    """Triangles filling the planar polygon ``rim`` (closed, in order), with
    points about h apart inside: its points and the triangles."""
    from matplotlib.path import Path
    from scipy.spatial import Delaunay
    o = rim.mean(axis=0)
    n = np.cross(rim[1] - rim[0], rim[len(rim)//3] - rim[0])
    n /= np.linalg.norm(n)
    e1 = rim[0] - o - ((rim[0] - o) @ n)*n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uv = np.c_[(rim - o) @ e1, (rim - o) @ e2]
    path = Path(uv)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    g = np.array([[x, y] for x in np.arange(lo[0], hi[0], h) for y in np.arange(lo[1], hi[1], h)])
    edge_dist = np.min(np.linalg.norm(g[:, None, :] - uv[None, :, :], axis=2), axis=1)
    inner = g[path.contains_points(g) & (edge_dist > 0.6*h)]
    allp = np.vstack([uv, inner])
    tri = Delaunay(allp).simplices
    keep = path.contains_points(allp[tri].mean(axis=1))
    return o + allp[:, :1]*e1 + allp[:, 1:]*e2, tri[keep]
