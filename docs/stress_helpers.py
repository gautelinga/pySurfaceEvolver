"""Helpers for the stress-case examples (docs/stress_cases.md): starting
shapes, exact spherical caps, and axisymmetric Young-Laplace references.

The references solve the axisymmetric Young-Laplace equation along the
meridian (r(s), z(s)) by arc length s, psi the tangent's angle to the r axis:
dr/ds = cos psi, dz/ds = sin psi, dpsi/ds + sin(psi)/r = p - rho_g*z (surface
tension 1; p the pressure jump at z = 0). They need SciPy.

* ``sessile(volume, theta, rho_g=0)``: a drop on the plane z = 0.
* ``bridge_spheres(volume, gap, theta, R=1)``: a bridge between two spheres of
  radius R centred at z = +-(R + gap/2).
* ``barrel(volume, b, theta)``: a barrel drop on a fibre of radius b (the z axis).

The same code drives the stress suite in bench/stress/.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---- starting shapes and exact caps --------------------------------------------

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

def _bridge_half(rn, p, c, R, theta):
    def hit(s, y):
        return y[0]**2 + (y[1] - c)**2 - R*R
    hit.terminal, hit.direction = True, -1

    def lost(s, y):
        return y[0] - 1e-6
    lost.terminal = True
    sol = profile(rn, 0.0, np.pi/2, p, events=(hit, lost), s_max=4*(c + R))
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise ValueError("missed the sphere")
    r, z, psi, area, vol, _ = sol.y[:, -1]
    n_l = np.array([np.sin(psi), -np.cos(psi)])          # out of the liquid
    n_s = np.array([r, z - c])/R                          # out of the sphere
    angle = np.degrees(np.arccos(np.clip(n_l @ n_s, -1, 1)))
    return angle - theta, r, z, area, vol


def bridge_spheres(volume, gap, theta, R=1.0, guess=None):
    """``guess``: (neck radius, pressure), e.g. a previous solution's."""
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


# ---- validation ----------------------------------------------------------------
