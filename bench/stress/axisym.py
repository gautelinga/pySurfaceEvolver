"""Axisymmetric Young-Laplace references for the stress suite.

A meridian (r(s), z(s)) by arc length s, with psi the tangent's angle to the r
axis: dr/ds = cos psi, dz/ds = sin psi, and the mean curvature balancing the
pressure, dpsi/ds + sin(psi)/r = p - rho_g*z (surface tension 1; p is the
pressure jump at z = 0, the liquid on the axis side of the profile). The area
(2 pi r ds), the volume (pi r^2 dz) and the gravity moment (pi r^2 z dz) are
integrated along with the shape, so the adaptive step stays accurate.

Solvers (each returns a dict):

* ``sessile(volume, theta, rho_g=0)``: a drop on the plane z = 0, contact angle
  theta (degrees, through the liquid), with gravity rho_g.
* ``bridge_spheres(volume, gap, theta, R=1)``: a bridge between two spheres of
  radius R centred at z = +-(R + gap/2).
* ``barrel(volume, b, theta)``: a barrel drop on a fibre of radius b (the z axis).

``python axisym.py`` validates against exact shapes and prints the errors.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

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

def validate():
    out = {}
    for th in (20.0, 90.0, 150.0):
        d = sessile(1.0, th)
        t = np.radians(th)
        R = (1.0/(np.pi*(2 - 3*np.cos(t) + np.cos(t)**3)/3))**(1/3)
        e = 2*np.pi*R*R*(1 - np.cos(t)) - np.cos(t)*np.pi*(R*np.sin(t))**2
        out[f"cap {th:g} deg: energy"] = abs(d["energy"]/e - 1)
        out[f"cap {th:g} deg: pressure"] = abs(d["pressure"]*R/2 - 1)
    sol = profile(0.7, 0.0, np.pi/2, 1/0.7, s_max=3.0)
    out["cylinder: radius drift"] = float(np.max(np.abs(sol.y[0] - 0.7)))
    a = 0.6
    sol = profile(a, 0.0, np.pi/2, 0.0, s_max=1.0)
    r, z = sol.y[0], sol.y[1]
    out["catenoid: r - a cosh(z/a)"] = float(np.max(np.abs(r - a*np.cosh(z/a))))
    H = z[-1]
    out["catenoid: area"] = abs(sol.y[3, -1]/(np.pi*a*(H + a*np.sinh(H/a)*np.cosh(H/a))) - 1)
    return out


if __name__ == "__main__":
    for k, v in validate().items():
        print(f"{k:32s} {v:.2e}")
    print("bridge gap 0.2:", {k: round(float(v), 6) for k, v in bridge_spheres(0.05, 0.2, 40.0).items()})
    print("barrel:", {k: round(float(v), 6) for k, v in barrel(2.0, 0.2, 30.0).items()})
    d = sessile(50.0, 60.0, rho_g=1.0)
    print("puddle: height", round(d["height"], 6), "2 sin(theta/2) =", round(2*np.sin(np.radians(30)), 6),
          "radius", round(d["radius"], 4), "energy", round(d["energy"], 6))
