"""Starting surfaces and references for docs/floating_lens.md: a lens of one
liquid floating on another, under air.

Three interfaces meet along the lens's rim: the bath's free surface (tension
``gamma_bath``), the lens's top (``gamma_top``, lens against air) and its
bottom (``gamma_bottom``, lens against bath). The lens is computed in a quarter
of space, between the mirror planes x = 0 and y = 0; the bath's surface is held
at z = 0 on a quarter circle of radius ``reach``.

* ``floating_lens(volume, gamma_bath, gamma_top, gamma_bottom, ...)``: the
  surface, from two spherical caps on a flat ring, with the facets' tensions set
  and (with ``gravity``) the weights of lens and bath; relaxed.
* ``lens_shape(ev)``: what a relaxed lens measures: the rim, the heights and
  thickness, the pressure, and spheres fitted to the caps.
* ``neumann(gamma_bath, gamma_top, gamma_bottom, volume)``: the exact lens
  without gravity: the caps' angles from the Neumann triangle, and the size
  that holds the volume.
* ``axisymmetric_lens(volume, ..., g)``: the lens under gravity, from the
  axisymmetric equations (a boundary value problem).
* ``langmuir(gamma_bath, gamma_top, gamma_bottom, rho_lens, rho_bath, g)``:
  the thickness of a large lens under gravity, the puddle limit.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.optimize import brentq
from scipy.spatial import Delaunay
from scipy.special import k0, k1

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C


# ---- the exact references --------------------------------------------------------

def neumann_angles(gamma_bath, gamma_top, gamma_bottom):
    """The angles (radians) of the top and bottom caps with the bath's flat surface
    at the rim, from the closed triangle of the three tensions."""
    a, t, b = gamma_bath, gamma_top, gamma_bottom
    if not (a < t + b and t < a + b and b < a + t):
        raise ValueError("the tensions don't close a triangle: the lens spreads or sinks")
    top = np.arccos((a*a + t*t - b*b)/(2*a*t))
    bottom = np.arccos((a*a + b*b - t*t)/(2*a*b))
    return top, bottom


def neumann(gamma_bath, gamma_top, gamma_bottom, volume):
    """The lens without gravity: two spherical caps on a circle of radius ``rim``,
    meeting the flat bath at the Neumann angles. Returns a dict with the angles
    (degrees), the rim radius, the caps' heights and radii, the lens's pressure
    and its energy counted from a bath without the lens."""
    top, bottom = neumann_angles(gamma_bath, gamma_top, gamma_bottom)
    # a cap of angle a on a circle of radius r: height r tan(a/2), volume pi h (3 r^2 + h^2)/6
    u, w = np.tan(top/2), np.tan(bottom/2)
    shape = np.pi*(u*(3 + u*u) + w*(3 + w*w))/6
    r = (volume/shape)**(1/3)
    h_top, h_bottom = r*u, r*w
    R_top, R_bottom = r/np.sin(top), r/np.sin(bottom)
    area_top, area_bottom = np.pi*(r*r + h_top**2), np.pi*(r*r + h_bottom**2)
    return {"angle_top": np.degrees(top), "angle_bottom": np.degrees(bottom), "rim": r,
            "h_top": h_top, "h_bottom": h_bottom, "thickness": h_top + h_bottom,
            "R_top": R_top, "R_bottom": R_bottom,
            "pressure": 2*gamma_top/R_top,
            "energy": gamma_top*area_top + gamma_bottom*area_bottom - gamma_bath*np.pi*r*r}


def langmuir(gamma_bath, gamma_top, gamma_bottom, rho_lens, rho_bath, g):
    """Langmuir's thickness of a large floating lens (a puddle): the spreading
    coefficient S = gamma_bath - gamma_top - gamma_bottom (negative) balances
    the lens's weight less its buoyancy, rho_lens (1 - rho_lens/rho_bath) g e^2/2 = -S.
    Returns (thickness, the part below the bath's level)."""
    S = gamma_bath - gamma_top - gamma_bottom
    e = np.sqrt(-2*S/(rho_lens*(1 - rho_lens/rho_bath)*g))
    return e, e*rho_lens/rho_bath


def axisymmetric_lens(volume, gamma_bath, gamma_top, gamma_bottom, rho_lens, rho_bath, g,
                      far=12.0, tol=1e-8):
    """The floating lens under gravity from the axisymmetric equations, solved as a
    boundary value problem (``scipy.integrate.solve_bvp``): the top z_t(r) and
    bottom z_b(r) out to the rim, and the bath's surface z_a(r) from the rim out to
    ``far`` of its capillary lengths, where it decays as K0(r/l). Unknowns besides
    the profiles: the lens's pressure P (at z = 0) and the rim's radius.
    Conditions: the profiles flat on the axis, all three meeting at the rim with
    their tensions balanced there, and the volume. Followed from a small gravity
    up to ``g``, from the exact lens without it. Returns the pressure, the rim's
    radius and height ``z_rim``, the heights of the top and bottom (``top``,
    ``bottom``, from z = 0), the thickness, and the profiles as (r, z) pairs."""
    a, t, b = gamma_bath, gamma_top, gamma_bottom

    def zpp(kappa, p, r):
        # z'' of a profile z(r) whose two principal curvatures add up to kappa
        w = 1 + p*p
        safe = np.where(r > 0, r, 1.0)
        return np.where(r > 0, kappa*w**1.5 - p*w/safe, kappa/2)

    def solve(g, guess):
        ell = np.sqrt(a/(rho_bath*g))
        L = far*ell

        def f(x, y, q):
            P, rc = q
            zt, pt, zb, pb, Q, za, pa = y
            r, ra = rc*x, rc + L*x
            kt = -(P - rho_lens*g*zt)/t                        # air above
            kb = (P - rho_lens*g*zb + rho_bath*g*zb)/b          # bath below
            ka = rho_bath*g*za/a
            return np.vstack([rc*pt, rc*zpp(kt, pt, r), rc*pb, rc*zpp(kb, pb, r),
                              rc*2*np.pi*r*(zt - zb), L*pa, L*zpp(ka, pa, ra)])

        def bc(y0, y1, q):
            P, rc = q
            ft, fb, fa = np.arctan(y1[1]), np.arctan(y1[3]), np.arctan(y0[6])
            rf = rc + L
            return np.array([y0[1], y0[3], y0[4], y1[0] - y1[2], y0[5] - y1[0],
                             a*np.cos(fa) - t*np.cos(ft) - b*np.cos(fb),
                             a*np.sin(fa) - t*np.sin(ft) - b*np.sin(fb),
                             y1[4] - volume,
                             y1[6] + k1(rf/ell)/(ell*k0(rf/ell))*y1[5]])

        sol = solve_bvp(f, bc, *guess, tol=tol, max_nodes=300000)
        if not sol.success:
            raise RuntimeError(f"no axisymmetric lens at g = {g}: {sol.message}")
        return sol, L

    ex = neumann(a, t, b, volume)
    x = np.linspace(0, 1, 41)
    r = ex["rim"]*x
    Rt, Rb = ex["R_top"], ex["R_bottom"]
    zt = np.sqrt(Rt**2 - r**2) - (Rt - ex["h_top"])
    zb = (Rb - ex["h_bottom"]) - np.sqrt(Rb**2 - r**2)
    y = np.vstack([zt, -r/np.sqrt(Rt**2 - r**2), zb, r/np.sqrt(Rb**2 - r**2),
                   cumulative_trapezoid(2*np.pi*r*(zt - zb), r, initial=0), 0*x, 0*x])
    guess = (x, y, np.array([ex["pressure"], ex["rim"]]))
    g0 = min(0.3*a/(rho_bath*ex["rim"]**2), g)          # a Bond number of 0.3
    for gk in np.geomspace(g0, g, 2 + int(np.log(g/g0)/np.log(1.5))):
        sol, L = solve(gk, guess)
        guess = (sol.x, sol.y, sol.p)
    P, rc = sol.p
    xs = np.linspace(0, 1, 400)
    y = sol.sol(xs)
    return {"pressure": P, "rim": rc, "z_rim": y[0, -1], "top": y[0, 0], "bottom": y[2, 0],
            "thickness": y[0, 0] - y[2, 0],
            "profile_top": (rc*xs, y[0]), "profile_bottom": (rc*xs, y[2]),
            "profile_bath": (rc + L*xs, y[5])}


# ---- the starting surface ---------------------------------------------------------

def _rings(r0, r1, h, centre=False):
    """Points on quarter circles (x, y >= 0) of radii from r0 to r1, about h(rho)
    apart along and across; with ``centre``, r0 = 0 and the origin is the first
    point."""
    radii = [r0]
    while radii[-1] < r1:
        radii.append(radii[-1] + h(radii[-1]))
    radii = np.asarray(radii)
    radii = r0 + (radii - r0)*(r1 - r0)/(radii[-1] - r0)   # the last one on r1
    pts = [np.zeros((1, 2))] if centre else []
    for rho in radii[1 if centre else 0:]:
        m = max(1, int(np.ceil(rho*np.pi/2/h(rho))))
        t = np.linspace(0, np.pi/2, m + 1)
        pts.append(rho*np.c_[np.cos(t), np.sin(t)])
    return np.vstack(pts)


def _triangulate(pts, keep):
    """Delaunay triangles of 2D points whose centroids pass ``keep``, oriented
    counterclockwise (normal +z), slivers along the straight sides dropped."""
    tri = Delaunay(pts).simplices
    tri = tri[keep(pts[tri].mean(axis=1))]
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0])*(c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1])*(c[:, 0] - a[:, 0])
    keep_area = np.abs(cross) > 1e-9*np.abs(cross).max()
    tri, cross = tri[keep_area], cross[keep_area]
    tri[cross < 0] = tri[cross < 0][:, [0, 2, 1]]
    return tri


def _cap(rho, rim, height):
    """Heights of a spherical cap of this height over a circle of radius rim."""
    R = (rim*rim + height*height)/(2*height)
    return np.sqrt(R*R - rho*rho) - (R - height)


def floating_lens(volume, gamma_bath, gamma_top, gamma_bottom, *, reach=None, h=0.12,
                  h_rim=None, gravity=None, rho_lens=0.9, rho_bath=1.0, rim=None, levels=1,
                  relax=True):
    """A lens of this volume floating on a bath, in the quarter x, y >= 0 between
    two mirror planes (``volume`` is the whole lens's). The bath's surface is held
    at z = 0 on the quarter circle of radius ``reach``. The start: two spherical
    caps of the same angle on a circle of radius ``rim``, holding the volume, on a
    flat ring out to ``reach``, edges about ``h`` long (``h_rim`` at the rim,
    growing by 0.3 per unit distance from it); relaxed with
    ``relax(levels=levels)``. By default the rim is the exact one without gravity,
    and with gravity at least that of a puddle of Langmuir's thickness; ``reach`` is
    three times the rim without gravity, and ten capillary lengths of the bath
    beyond the rim with it.

    With ``gravity`` (the gravitational constant g), the lens weighs ``rho_lens``
    per volume and the bath ``rho_bath``: the bath is an open body (no volume
    fixed: a reservoir at the level z = 0) whose faces are its free surface and the
    lens's bottom, so its weight counts the water lifted above z = 0 and the
    water the lens displaces below it.
    """
    exact = neumann(gamma_bath, gamma_top, gamma_bottom, volume)
    if rim is None:
        rim = exact["rim"]
        if gravity:
            e = langmuir(gamma_bath, gamma_top, gamma_bottom, rho_lens, rho_bath, gravity)[0]
            rim = max(rim, np.sqrt(volume/(np.pi*e)))
    # two caps of the same angle (the mean of the exact ones), holding the volume
    u = np.tan(np.radians(exact["angle_top"] + exact["angle_bottom"])/4)
    k = brentq(lambda k: np.pi*rim**3*k*u*(3 + (k*u)**2)/3 - volume, 1e-6, 1e3)
    u = w = k*u
    if reach is None:
        reach = 1.15*rim + 10*np.sqrt(gamma_bath/(rho_bath*gravity)) if gravity else 3*rim
    if h_rim is None:
        h_rim = h

    def size(rho):
        return min(h, h_rim + 0.3*abs(rho - rim))
    disc = _rings(0.0, rim, size, centre=True)
    ring = _rings(rim, reach, size)
    on_rim = np.abs(np.hypot(*disc.T) - rim) < 1e-12
    cap_tri = _triangulate(disc, lambda c: np.ones(len(c), bool))
    ring_tri = _triangulate(ring, lambda c: np.hypot(*c.T) > rim)
    # vertices: the rim (shared), the top's and bottom's insides, the ring outside the rim
    z_top = _cap(np.hypot(*disc.T), rim, rim*u)
    z_bottom = _cap(np.hypot(*disc.T), rim, rim*w)
    inside = np.flatnonzero(~on_rim)
    rim_rows = np.flatnonzero(on_rim)
    outer = np.flatnonzero(np.abs(np.hypot(*ring.T) - rim) > 1e-12)
    v = np.vstack([np.c_[disc[rim_rows], np.zeros(len(rim_rows))],
                   np.c_[disc[inside], z_top[inside]],
                   np.c_[disc[inside], -z_bottom[inside]],
                   np.c_[ring[outer], np.zeros(len(outer))]])
    n_rim, n_in = len(rim_rows), len(inside)
    disc_map = np.empty(len(disc), int)
    disc_map[rim_rows] = np.arange(n_rim)
    top_map, bottom_map = disc_map.copy(), disc_map.copy()
    top_map[inside] = n_rim + np.arange(n_in)
    bottom_map[inside] = n_rim + n_in + np.arange(n_in)
    ring_map = np.empty(len(ring), int)
    ring_inner = np.flatnonzero(np.abs(np.hypot(*ring.T) - rim) <= 1e-12)
    # the ring's inner points are the rim's: match them by angle
    rim_angle = np.arctan2(disc[rim_rows, 1], disc[rim_rows, 0])
    ring_angle = np.arctan2(ring[ring_inner, 1], ring[ring_inner, 0])
    if len(ring_inner) != n_rim or np.abs(np.sort(rim_angle) - np.sort(ring_angle)).max() > 1e-12:
        raise RuntimeError("the ring's inner circle and the rim don't match")
    ring_map[ring_inner] = np.argsort(rim_angle)[np.argsort(np.argsort(ring_angle))]
    ring_map[outer] = n_rim + 2*n_in + np.arange(len(outer))
    top = top_map[cap_tri]                      # normal +z: out of the lens
    bottom = bottom_map[cap_tri][:, [0, 2, 1]]  # normal -z: out of the lens
    bath = ring_map[ring_tri]                   # normal +z: out of the bath
    faces = np.vstack([top, bottom, bath])
    nt, nb = len(top), len(bottom)
    rows_top = np.arange(nt)
    rows_bottom = nt + np.arange(nb)
    rows_bath = nt + nb + np.arange(len(bath))

    bodies = [pyse.Body(volume=volume/4, faces=np.r_[rows_top, rows_bottom],
                        orientation=[1]*(nt + nb), volconst=0.0,
                        density=rho_lens if gravity else None)]
    if gravity:
        # the bath: its free surface and the lens's bottom, both seen from below
        bodies.append(pyse.Body(faces=np.r_[rows_bath, rows_bottom],
                                orientation=[1]*len(bath) + [-1]*nb,
                                volconst=0.0, density=rho_bath))
    walls = {"mirror_x": C.mirror("x"), "mirror_y": C.mirror("y")}
    rows = {"mirror_x": np.flatnonzero(np.abs(v[:, 0]) < 1e-12),
            "mirror_y": np.flatnonzero(np.abs(v[:, 1]) < 1e-12)}
    fixed = np.flatnonzero(np.abs(np.hypot(v[:, 0], v[:, 1]) - reach) < 1e-9)
    ev = pyse.Evolver()
    ev.load_arrays(v, faces, constraints=walls, vertex_constraints=rows, fixed=fixed,
                   bodies=bodies, gravity=gravity)
    tension = np.empty(len(faces))
    tension[rows_top], tension[rows_bottom], tension[rows_bath] = gamma_top, gamma_bottom, gamma_bath
    ev.set_values("facet", "tension", tension)
    if relax:
        ev.relax(levels=levels)
    return ev


# ---- measuring a lens --------------------------------------------------------------

def _sphere_fit(p):
    """Least-squares sphere through points: centre, radius."""
    A = np.c_[2*p, np.ones(len(p))]
    sol = np.linalg.lstsq(A, (p*p).sum(axis=1), rcond=None)[0]
    c = sol[:3]
    return c, np.sqrt(sol[3] + c @ c)


def lens_shape(ev, gamma_bath=None):
    """What a relaxed lens measures: the rim's radius and height ``z_rim``, the
    heights of the top and bottom (``top``, ``bottom``, from z = 0; ``h_top`` and
    ``h_bottom`` from the rim), the thickness, the lens's pressure (at z = 0), and
    the radii of spheres fitted to each cap's vertices with the angles at which
    they meet the rim's plane (degrees). With ``gamma_bath`` (no gravity), also the
    whole lens's energy counted from the bath without it."""
    m = ev.mesh()
    lens = (m.facet_bodies == 1).any(axis=1)
    x = m.vertices
    a, b, c = (x[m.facets[:, k]] for k in range(3))
    nz = np.cross(b - a, c - a)[:, 2]
    top_f, bottom_f = lens & (nz > 0), lens & (nz < 0)
    top_v, bottom_v = np.unique(m.facets[top_f]), np.unique(m.facets[bottom_f])
    rim_v = np.intersect1d(top_v, bottom_v)
    rim = np.hypot(x[rim_v, 0], x[rim_v, 1]).mean()
    z_rim = x[rim_v, 2].mean()
    top, bottom = x[top_v, 2].max(), x[bottom_v, 2].min()
    out = {"rim": rim, "z_rim": z_rim, "top": top, "bottom": bottom,
           "h_top": top - z_rim, "h_bottom": z_rim - bottom, "thickness": top - bottom,
           "pressure": ev.body(1).pressure}
    if gamma_bath is not None:
        # the bath's surface without the lens: the polygon of the held ring
        # (refining puts the new vertices on its straight edges)
        ring = x[m.fixed]
        ring = ring[np.argsort(np.arctan2(ring[:, 1], ring[:, 0]))]
        area = 0.5*np.sum(ring[:-1, 0]*ring[1:, 1] - ring[1:, 0]*ring[:-1, 1])
        out["energy"] = 4*(ev.energy - gamma_bath*area)
    for name, rows in (("top", top_v), ("bottom", bottom_v)):
        centre, R = _sphere_fit(x[rows])
        out["R_" + name] = R
        out["angle_" + name] = np.degrees(np.arcsin(min(rim/R, 1.0)))
    return out
