"""Helpers for the stress-case examples (docs/stress_cases.md) and the
shrinking bridge (docs/bridge_rupture.md): exact
spherical caps and axisymmetric Young-Laplace references (the starting shapes
come from ``pysurfaceevolver.shapes``).

The references solve the axisymmetric Young-Laplace equation along the
meridian (r(s), z(s)) by arc length s, psi the tangent's angle to the r axis:
dr/ds = cos psi, dz/ds = sin psi, dpsi/ds + sin(psi)/r = p - rho_g*z (surface
tension 1; p the pressure jump at z = 0). They need SciPy.

* ``sessile(volume, theta, rho_g=0)``: a drop on the plane z = 0.
* ``bridge_spheres(volume, gap, theta, R=1)``: a bridge between two spheres of
  radius R centred at z = +-(R + gap/2).
* ``bridge_spheres_fold(gap, theta)``, ``bridge_spheres_branch(gap, theta)`` and
  ``bridge_spheres_offset(neck_height, gap, theta)``: where the mirror-symmetric
  bridges end, where bridges with the neck off the mid-plane branch off them,
  and those bridges.
* ``barrel(volume, b, theta)``: a barrel drop on a fibre of radius b (the z axis).

The same code drives the stress suite in bench/stress/ (all but the
off-centre bridges).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---- exact caps -------------------------------------------------------------

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


# ---- axisymmetric references ---------------------------------------------------

RTOL, ATOL = 1e-11, 1e-13


def _rhs(p, rho_g):
    def rhs(s, y):
        r, z, psi = y[0], y[1], y[2]
        k = p - rho_g*z
        curv = k - (np.sin(psi)/r if r > 1e-12 else k/2)
        return [np.cos(psi), np.sin(psi), curv,
                2*np.pi*r, np.pi*r*r*np.sin(psi), np.pi*r*r*z*np.sin(psi)]
    return rhs


def _curled(s, y):
    """Stops profiles that curl over (a trial pressure far off): psi out of range."""
    return 1.5*np.pi - abs(y[2] - np.pi/2)
_curled.terminal = True


def profile(r0, z0, psi0, p, rho_g=0.0, s_max=50.0, events=()):
    """Integrate from (r0, z0) at angle psi0; y = r, z, psi, area, volume, z-moment.
    Events first (status 1 with t_events[0] set means the first event hit)."""
    return solve_ivp(_rhs(p, rho_g), (0, s_max), [r0, z0, psi0, 0, 0, 0],
                     events=list(events) + [_curled], rtol=RTOL, atol=ATOL)


def _root_in_scan(f, xs):
    vals = []
    for x in xs:
        try:
            vals.append(f(x))
        except Exception:
            vals.append(np.nan)
    vals = np.array(vals, float)
    for i in range(len(xs) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i]*vals[i + 1] < 0:
            return brentq(f, xs[i], xs[i + 1], xtol=1e-14, rtol=1e-14)
    raise ValueError("no sign change in the scan")


# ---- sessile drop --------------------------------------------------------------

def _sessile(b, theta, rho_g):
    """From the apex (curvature 1/b, at height H unknown) down to tangent angle
    theta. Uses depth y below the apex: dphi/ds = 2/b + rho_g y - sin(phi)/x."""
    t = np.radians(theta)

    def rhs(s, u):
        x, y, phi = u[0], u[1], u[2]
        k = 2/b + rho_g*y
        return [np.cos(phi), np.sin(phi), k - (np.sin(phi)/x if x > 1e-12 else k/2),
                2*np.pi*x, np.pi*x*x*np.sin(phi), np.pi*x*x*y*np.sin(phi)]

    def hit(s, u):
        return u[2] - t
    hit.terminal, hit.direction = True, 1
    sol = solve_ivp(rhs, (0, 400*max(b, 1)), [0.0, 0.0, 0.0, 0, 0, 0], events=[hit],
                    rtol=RTOL, atol=ATOL)
    if sol.status != 1:
        raise ValueError("contact angle not reached")
    return sol.y[:, -1]


def sessile(volume, theta, rho_g=0.0):
    t = np.radians(theta)
    b = brentq(lambda b: _sessile(b, theta, rho_g)[4] - volume, 1e-3, 1e3, xtol=1e-14, rtol=1e-14)
    x, h, phi, area, vol, ymom = _sessile(b, theta, rho_g)
    grav = rho_g*(h*vol - ymom)                     # int rho_g z dV, z = h - y
    energy = area - np.cos(t)*np.pi*x*x + grav
    return dict(radius=x, height=h, area=area, volume=vol, gravity=grav, energy=energy,
                apex_radius=b, pressure=2/b + rho_g*h)


# ---- bridge between spheres ---------------------------------------------------

def _bridge_half(rn, p, c, R, theta, z0=0.0):
    """From the neck, radius rn at height z0, up to the sphere centred at z = c."""
    def hit(s, y):
        return y[0]**2 + (y[1] - c)**2 - R*R
    hit.terminal, hit.direction = True, -1

    def lost(s, y):
        return y[0] - 1e-6
    lost.terminal = True
    sol = profile(rn, z0, np.pi/2, p, events=(hit, lost), s_max=4*(c + R))
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise ValueError("missed the sphere")
    r, z, psi, area, vol, _ = sol.y[:, -1]
    n_l = np.array([np.sin(psi), -np.cos(psi)])          # out of the liquid
    n_s = np.array([r, z - c])/R                          # out of the sphere
    angle = np.degrees(np.arccos(np.clip(n_l @ n_s, -1, 1)))
    return angle - theta, r, z, area, vol


def bridge_spheres(volume, gap, theta, R=1.0, guess=None):
    """``guess``: (neck radius, pressure), e.g. a previous solution's. Without
    one, of two bridges holding the volume (a stable one and, with a thinner
    neck, an unstable one) this finds the unstable one."""
    c = R + gap/2
    t = np.radians(theta)
    if guess is not None:
        def f(x):
            try:
                ang, r, z, area, vol = _bridge_half(x[0], x[1], c, R, theta)
            except ValueError:
                return [1e3, 1e3]
            hc = z - (c - R)
            return [ang, 2*(vol - np.pi*hc*hc*(3*R - hc)/3) - volume]
        x = _solve2(f, guess)
        if x is not None:
            _, r, z, area, vol = _bridge_half(x[0], x[1], c, R, theta)
            hc = z - (c - R)
            wet = 2*np.pi*R*hc
            return dict(neck=x[0], pressure=x[1], contact_z=z, contact_r=r, area=2*area,
                        wetted=2*wet, energy=2*(area - np.cos(t)*wet),
                        volume=2*(vol - np.pi*hc*hc*(3*R - hc)/3))

    def half(rn):
        p = _root_in_scan(lambda p: _bridge_half(rn, p, c, R, theta)[0],
                          np.linspace(-3/rn - 3, 3/rn + 3, 61))
        _, r, z, area, vol = _bridge_half(rn, p, c, R, theta)
        hc = z - (c - R)                                  # sphere cap height below z
        cap = np.pi*hc*hc*(3*R - hc)/3
        return p, r, z, area, vol - cap, 2*np.pi*R*hc

    rn = _root_in_scan(lambda x: 2*half(x)[4] - volume, np.geomspace(1e-3, 0.95*R, 25))
    p, r, z, area, v, wet = half(rn)
    return dict(neck=rn, pressure=p, contact_z=z, contact_r=r, area=2*area, wetted=2*wet,
                energy=2*(area - np.cos(t)*wet), volume=2*v)


def _symmetric_pressure(rn, c, R, theta, guess=None):
    """The pressure of the mirror-symmetric bridge with neck radius rn."""
    def f(p):
        return _bridge_half(rn, p, c, R, theta)[0]
    if guess is not None:
        from scipy.optimize import newton
        try:
            p = newton(f, guess, tol=1e-13, maxiter=50)
            if abs(f(p)) < 1e-9:
                return p
        except (RuntimeError, ValueError):
            pass
    return _root_in_scan(f, np.linspace(-3/rn - 3, 3/rn + 3, 61))


def _offset_bridge(rn, p, zn, c, R, theta):
    """The bridge with neck radius rn at height zn: the contact-angle errors at
    the upper and lower spheres, the energy and the volume. The part below the
    neck is the mirror image of a profile from height -zn up to the upper sphere."""
    t = np.radians(theta)
    angles, energy, volume = [], 0.0, 0.0
    for z0 in (zn, -zn):
        ang, r, z, area, vol = _bridge_half(rn, p, c, R, theta, z0)
        hc = z - (c - R)
        angles.append(ang)
        energy += area - np.cos(t)*2*np.pi*R*hc
        volume += vol - np.pi*hc*hc*(3*R - hc)/3
    return angles, energy, volume


def _symmetric_family(gap, theta, R, necks):
    """The mirror-symmetric bridges at these neck radii (largest first):
    (neck, pressure) rows; necks where none is found are skipped."""
    c = R + gap/2
    rows, p = [], None
    for rn in sorted(necks, reverse=True):
        try:
            p = _symmetric_pressure(rn, c, R, theta, p)
        except ValueError:
            p = None
            continue
        rows.append((rn, p))
    return rows


def _symmetric(rn, p, gap, theta, R):
    c = R + gap/2
    _, energy, volume = _offset_bridge(rn, p, 0.0, c, R, theta)
    return dict(neck=rn, pressure=p, volume=volume, energy=energy)


def bridge_spheres_fold(gap, theta, R=1.0, necks=(0.02, 0.6)):
    """The least volume a mirror-symmetric bridge holds, for neck radii in the
    range ``necks`` (times R): where its stable branch (wider neck) meets the
    unstable one. Below it no such bridge exists. A dict: volume, neck,
    pressure, energy."""
    from scipy.optimize import minimize_scalar
    c = R + gap/2
    rows = _symmetric_family(gap, theta, R, np.geomspace(necks[0]*R, necks[1]*R, 60))
    vols = [_symmetric(rn, p, gap, theta, R)["volume"] for rn, p in rows]
    k = int(np.argmin(vols))
    if k in (0, len(rows) - 1):
        raise ValueError("no least volume inside the range of necks")
    pressure = {"p": rows[k][1]}

    def volume(rn):
        pressure["p"] = _symmetric_pressure(rn, c, R, theta, pressure["p"])
        return _symmetric(rn, pressure["p"], gap, theta, R)["volume"]
    best = minimize_scalar(volume, bounds=(rows[k + 1][0], rows[k - 1][0]), method="bounded",
                           options={"xatol": 1e-10})
    return _symmetric(best.x, _symmetric_pressure(best.x, c, R, theta, pressure["p"]),
                      gap, theta, R)


def bridge_spheres_branch(gap, theta, R=1.0, necks=(0.02, 0.6), eps=1e-5):
    """Where bridges whose neck sits off the mid-plane branch off the
    mirror-symmetric ones: the symmetric bridge at which sliding the neck
    along the axis keeps both contact angles to first order. The slope is taken
    per unit of the neck's curvature p - 1/rn: unweighted, it also vanishes at
    the cylindrical bridge (rn = R cos(theta), in the range ``necks`` (times R)
    above about 53 degrees), where sliding moves nothing; that is no branch
    point, and Evolver sees no instability there. The weighted slope changes
    sign once, from negative to positive as the neck narrows: there a
    shrinking bridge loses stability (as Evolver's eigenvalues show). The result
    may lie past the fold (a thinner neck than the fold's): then the fold comes
    first. A dict: volume, neck, pressure, energy."""
    c = R + gap/2

    def slope(rn, p):            # d(contact angle at the upper sphere)/d(neck height),
        return (_bridge_half(rn, p, c, R, theta, eps)[0]     # per unit of the neck's curvature
                - _bridge_half(rn, p, c, R, theta, -eps)[0])/(2*eps*(p - 1/rn))
    rows = _symmetric_family(gap, theta, R, np.geomspace(necks[0]*R, necks[1]*R, 60))
    s = [slope(rn, p) for rn, p in rows]
    for k in range(len(rows) - 1):           # widest first
        if s[k] < 0 < s[k + 1]:
            pressure = {"p": rows[k][1]}

            def f(rn):
                pressure["p"] = _symmetric_pressure(rn, c, R, theta, pressure["p"])
                return slope(rn, pressure["p"])
            rn = brentq(f, rows[k + 1][0], rows[k][0], xtol=1e-12)
            return _symmetric(rn, _symmetric_pressure(rn, c, R, theta, pressure["p"]),
                              gap, theta, R)
    raise ValueError("no branch point in the range of necks")


def bridge_spheres_offset(neck_height, gap, theta, R=1.0, guess=None):
    """The axisymmetric bridge with its neck at ``neck_height`` above the
    mid-plane (not mirror-symmetric), on the family through
    :func:`bridge_spheres_branch`; its mirror image has the neck below.
    ``guess``: (neck radius, pressure), by default the branch point's (fine for
    small heights; step out from there for larger ones). A dict: volume, neck,
    neck_height, pressure, energy."""
    c = R + gap/2
    if guess is None:
        b = bridge_spheres_branch(gap, theta, R)
        guess = (b["neck"], b["pressure"])

    def f(x):
        try:
            return _offset_bridge(x[0], x[1], neck_height, c, R, theta)[0]
        except ValueError:
            return [1e3, 1e3]
    x = _solve2(f, guess)
    if x is None:
        raise ValueError(f"no bridge with the neck at {neck_height} near {guess}")
    _, energy, volume = _offset_bridge(x[0], x[1], neck_height, c, R, theta)
    return dict(volume=volume, neck=x[0], neck_height=neck_height, pressure=x[1],
                energy=energy)


# ---- barrel on a fibre ---------------------------------------------------------

def _barrel_half(rm, p, b, theta):
    def hit(s, y):
        return y[0] - b
    hit.terminal, hit.direction = True, -1
    sol = profile(rm, 0.0, np.pi/2, p, events=(hit,), s_max=20*rm + 20)
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise ValueError("never reached the fibre")
    r, z, psi, area, vol, _ = sol.y[:, -1]
    angle = np.degrees(np.arccos(np.clip(np.sin(psi), -1, 1)))     # n_l . (1, 0)
    return angle - theta, z, area, vol


def _solve2(f, guess):
    """A 2D root from a guess (None if it fails)."""
    from scipy.optimize import root
    try:
        sol = root(f, guess, method="hybr", options={"xtol": 1e-13})
    except Exception:
        return None
    if not sol.success or np.max(np.abs(f(sol.x))) > 1e-9:
        return None
    return sol.x


def barrel(volume, b, theta, guess=None):
    """``guess``: (equator radius, pressure), e.g. a previous solution's; falls
    back to scanning when it fails."""
    t = np.radians(theta)
    if guess is not None:
        def f(x):
            try:
                ang, z, area, vol = _barrel_half(x[0], x[1], b, theta)
            except ValueError:
                return [1e3, 1e3]
            return [ang, 2*(vol - np.pi*b*b*z) - volume]
        x = _solve2(f, guess)
        if x is not None:
            _, z, area, vol = _barrel_half(x[0], x[1], b, theta)
            wet = 2*np.pi*b*z
            return dict(equator=x[0], pressure=x[1], length=2*z, area=2*area, wetted=2*wet,
                        energy=2*(area - np.cos(t)*wet), volume=2*(vol - np.pi*b*b*z))

    def half(rm):
        p = _root_in_scan(lambda p: _barrel_half(rm, p, b, theta)[0], np.linspace(1e-3, 2/b, 401))
        _, z, area, vol = _barrel_half(rm, p, b, theta)
        return p, z, area, vol - np.pi*b*b*z

    rm = _root_in_scan(lambda x: 2*half(x)[3] - volume, np.geomspace(1.05*b, 60*b, 30))
    p, z, area, v = half(rm)
    wet = 2*np.pi*b*z
    return dict(equator=rm, pressure=p, length=2*z, area=2*area, wetted=2*wet,
                energy=2*(area - np.cos(t)*wet), volume=2*v)


# ---- exact references from spheres ------------------------------------------------

def _cap(r: float, h: float) -> "tuple[float, float]":
    """Area and volume of a spherical cap of sphere radius r and height h."""
    return 2*np.pi*r*h, np.pi*h*h*(3*r - h)/3


def drop_on_sphere(volume: float, theta_deg: float, R: float = 1.0) -> Dict[str, float]:
    """A drop (no gravity) on a solid sphere of radius R at the origin, its
    contact angle theta through the liquid: a spherical cap of radius rho centred
    at height d on the axis, cos(theta) = (R^2 + rho^2 - d^2)/(2 R rho) at the
    contact circle. Energy (free area - cos(theta) wetted area), free and wetted
    areas, pressure 2/rho, contact radius, and the drop's height above the sphere."""
    c = np.cos(np.radians(theta_deg))

    def parts(rho):
        d = np.sqrt(R*R + rho*rho - 2*R*rho*c)
        z0 = (d*d + R*R - rho*rho)/(2*d)              # the contact circle's plane
        a_in, v_in = _cap(rho, z0 - (d - rho))         # the liquid ball inside the solid
        a_wet, v_wet = _cap(R, R - z0)                 # the solid inside the liquid ball
        return d, z0, 4*np.pi*rho**2 - a_in, a_wet, 4/3*np.pi*rho**3 - v_in - v_wet

    rho = brentq(lambda r: parts(r)[4] - volume, 1e-6*R, 1e3*R)
    d, z0, free, wet, _ = parts(rho)
    return dict(energy=free - c*wet, free=free, wetted=wet, pressure=2/rho, rho=rho,
                radius=float(np.sqrt(R*R - z0*z0)), height=d + rho - R)


def double_bubble(v1: float, v2: float) -> Dict[str, float]:
    """The standard double bubble enclosing v1 >= v2 (Hutchings, Morgan, Ritore
    and Ros): three spherical caps meeting at 120 degrees along a circle, radii
    r1, r2 and r3 with 1/r3 = 1/r2 - 1/r1, centres a distance D apart with
    D^2 = r1^2 + r2^2 - r1 r2. Total area, pressures 2/r1 and 2/r2."""
    from scipy.optimize import fsolve

    def geometry(r1, r2):
        D = np.sqrt(r1*r1 + r2*r2 - r1*r2)
        x0 = (D*D + r1*r1 - r2*r2)/(2*D)               # the triple circle's plane
        a = np.sqrt(max(r1*r1 - x0*x0, 0.0))
        a1, c1 = _cap(r1, r1 - x0)                     # sphere 1 beyond the plane: cut off
        a2, c2 = _cap(r2, r2 - (D - x0))
        if abs(r1 - r2) < 1e-14*r1:
            a3, v3 = np.pi*a*a, 0.0                    # a flat wall
        else:
            r3 = r1*r2/(r1 - r2)
            h3 = r3 - np.sqrt(r3*r3 - a*a)
            a3, v3 = _cap(r3, h3)                      # bulging into the larger bubble
        V1 = 4/3*np.pi*r1**3 - c1 - v3
        V2 = 4/3*np.pi*r2**3 - c2 + v3
        return V1, V2, 4*np.pi*r1**2 - a1 + 4*np.pi*r2**2 - a2 + a3

    if v2 > v1:
        raise ValueError("give the larger volume first")
    if v1 == v2:                    # a flat wall: each bubble 27 pi r^3/24
        r = (24*v1/(27*np.pi))**(1/3)
        return dict(area=6.75*np.pi*r*r, r1=r, r2=r, p1=2/r, p2=2/r)
    # (started off r1 = r2, where the wall turns flat and fsolve stalls)
    g = 1.001*(3*v1/(4*np.pi))**(1/3), (3*v2/(4*np.pi))**(1/3)
    (r1, r2), info, ier, msg = fsolve(lambda x: np.array(geometry(*x)[:2]) - [v1, v2], g,
                                      xtol=1e-14, full_output=True)
    V1, V2, area = geometry(r1, r2)
    if abs(V1 - v1) > 1e-10*v1 or abs(V2 - v2) > 1e-10*v1:
        raise RuntimeError(f"double_bubble({v1}, {v2}) did not converge: {msg}")
    return dict(area=area, r1=r1, r2=r2, p1=2/r1, p2=2/r2)


# ---- validation ----------------------------------------------------------------
