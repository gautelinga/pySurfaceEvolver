"""Starting surfaces and references for docs/pendant_drop.md: a drop hanging from
a ring in the plane z = 0, with surface tension 1 and liquid density 1.

The references are in units of the ring's radius R, with the Bond number
``bond`` = rho g R^2 / gamma: volumes in R^3, lengths in R, pressures in
gamma/R. The meshes are built on a ring of radius ``RING`` (gravity
``bond``/RING^2), see the page for why.

* ``hanging_drop(bond, volume)``: the drop (``volume`` in R^3), started as the
  spherical cap of that volume below the ring with its rim on the ring (a
  circular wire: fixed vertices, new ones put on the circle when refining),
  refined twice (``levels``) and relaxed.
* ``profile(bond, c, s_end)``: the axisymmetric Young-Laplace profile from the
  apex with total curvature ``c`` there, up to arclength ``s_end``.
* ``pendant_family(bond)``: the axisymmetric drops whose profile meets the ring,
  from a flat meniscus through the largest volume and past it.
* ``largest_drop(bond)``: the drop of largest volume.
* ``sideways_onset(bond)``: the first drop whose surface meets the ring
  horizontally, where it turns unstable to tipping sideways.
* ``fold(rows)``: the largest volume along a run, from its points around it.
* ``sideways(u)``: how far a mode moves the drop sideways on average.

The profile is integrated in its arclength s from the apex, with x the
distance from the axis, z the height above the apex and phi the angle of the
tangent from the horizontal:

    x' = cos(phi),  z' = sin(phi),  phi' = c - bond z - sin(phi)/x

(the total curvature falls with height by the hydrostatic pressure). A drop
is the profile up to a point where x = 1; its pressure, the curvature at the
ring, is c - bond h with h the drop's height, and phi there is the angle at
which its surface meets the ring.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes

RING = 0.25


# ---- the exact axisymmetric drops ---------------------------------------------

def _rhs(bond, c):
    def f(s, y):
        x, z, phi, v, xc, zc, phic, vc = y        # and their derivatives in c
        sn, cs = np.sin(phi), np.cos(phi)
        return [cs, sn, c - bond*z - sn/x, np.pi*x*x*sn,
                -sn*phic, cs*phic, 1 - bond*zc - (cs*phic*x - sn*xc)/(x*x),
                np.pi*(2*x*xc*sn + x*x*cs*phic)]
    return f


def profile(bond, c, s_end, dense=False):
    """Integrate the profile from the apex (total curvature c) to arclength
    s_end: solve_ivp's result, y = (x, z, phi, volume below z) and their
    derivatives in c."""
    s0 = 1e-6                                      # series start next to the apex
    y0 = [s0, c*s0*s0/4, c*s0/2, np.pi*c*s0**4/8, 0.0, s0*s0/4, s0/2, np.pi*s0**4/8]
    return solve_ivp(_rhs(bond, c), (s0, s_end), y0, method="DOP853",
                     rtol=1e-12, atol=1e-13, dense_output=dense)


def _on_ring(bond, c, s, move="s"):
    """Newton steps in s (or in c, ``move="c"``) to the point where x = 1:
    c, s and the profile's values there."""
    for _ in range(50):
        y = profile(bond, c, s).y[:, -1]
        f = y[0] - 1.0
        if abs(f) < 1e-13:
            break
        if move == "c":
            c -= f/y[4]
        else:
            s -= f/np.cos(y[2])
    return c, s, y


def pendant_family(bond, ds=0.05, past=0.5):
    """The drops hanging from the ring, from a flat meniscus (c = 0) along the
    curve x(c, s) = 1 in the plane of (c, s), through the largest volume and on
    for a share ``past`` of the way to it. Rows: c, s, volume, height,
    pressure, and the curve's unit tangent (dc, ds)."""
    c, s = 1e-3, 1.0
    rows, tangent, vmax_at = [], np.array([1.0, 0.0]), None
    while True:
        for _ in range(50):                        # Newton back onto x = 1
            y = profile(bond, c, s).y[:, -1]
            g = np.array([y[4], np.cos(y[2])])     # the gradient of x in (c, s)
            f = y[0] - 1.0
            c, s = c - f*g[0]/(g @ g), s - f*g[1]/(g @ g)
            if abs(f) < 1e-13:
                break
        t = np.array([-g[1], g[0]])/np.linalg.norm(g)
        tangent = t if t @ tangent > 0 else -t
        rows.append((c, s, y[3], y[1], c - bond*y[1], *tangent))
        if vmax_at is None and len(rows) > 2 and rows[-1][2] < rows[-2][2]:
            vmax_at = len(rows)
        if vmax_at is not None and len(rows) > vmax_at*(1 + past):
            break
        c, s = c + ds*tangent[0], s + ds*tangent[1]
    return np.array(rows)


def largest_drop(bond):
    """The drop of largest volume: dict of volume, height, pressure (the
    curvature at the ring), apex (the curvature at the apex), angle (at which
    its surface meets the ring, degrees from the horizontal), and c, s."""
    fam = pendant_family(bond, past=0.1)
    k = int(np.argmax(fam[:, 2]))
    move = "s" if abs(fam[k, 5]) > abs(fam[k, 6]) else "c"   # along the curve in c, or in s
    j = 0 if move == "s" else 1
    def drop(t):
        return _on_ring(bond, t, fam[k, 1], "s") if move == "s" else \
            _on_ring(bond, fam[k, 0], t, "c")
    a, b = sorted((fam[k - 1, j], fam[k + 1, j]))
    best = minimize_scalar(lambda t: -drop(t)[2][3], bounds=(a, b), method="bounded",
                           options=dict(xatol=1e-10))
    c, s, y = drop(best.x)
    return dict(volume=y[3], height=y[1], pressure=c - bond*y[1], apex=c,
                angle=np.degrees(y[2]), c=c, s=s)


def sideways_onset(bond):
    """The first drop past the flat meniscus whose surface meets the ring
    horizontally, or None if the largest drop comes first: dict of volume,
    height, pressure, c, s."""
    fam = pendant_family(bond, past=0.1)
    k = int(np.argmax(fam[:, 2]))
    sine = [np.sin(profile(bond, c, s).y[2, -1]) for c, s in fam[:k + 1, :2]]
    down = [i for i in range(2, k + 1) if sine[i] <= 0 < sine[i - 1]]
    if not down:
        return None
    a, b = fam[down[0] - 1, :2], fam[down[0], :2]
    def drop(t):                                   # along the chord, then onto x = 1
        return _on_ring(bond, *(a + t*(b - a)))
    t = brentq(lambda t: np.sin(drop(t)[2][2]), 0.0, 1.0, xtol=1e-12)
    c, s, y = drop(t)
    return dict(volume=y[3], height=y[1], pressure=c - bond*y[1], c=c, s=s)


# ---- reading a run ----------------------------------------------------------------

def fold(rows):
    """The largest volume along a run past its fold, from rows of (volume,
    pressure, height): parabolas in the pressure through the five points
    around the largest volume (the volume is smooth in the pressure there,
    while the pressure turns in the volume). Returns volume, pressure,
    height."""
    rows = np.asarray(rows, float)
    k = int(np.argmax(rows[:, 0]))
    near = rows[max(k - 2, 0):k + 3]
    a, b, c = np.polyfit(near[:, 1], near[:, 0], 2)
    p = -b/(2*a)
    return np.polyval([a, b, c], p), p, np.polyval(np.polyfit(near[:, 1], near[:, 2], 2), p)


def sideways(u):
    """How far a mode (a displacement per vertex, unit root mean square)
    moves the vertices sideways on average: 0 for a motion round the axis."""
    return float(np.hypot(u[:, 0].mean(), u[:, 1].mean()))


# ---- the mesh -----------------------------------------------------------------

def hanging_drop(bond, volume, levels=2, nr=6, nt=32):
    """The drop below a ring of radius RING, with ``volume`` in units of
    RING^3 and gravity bond/RING^2: started as the spherical cap of that volume
    (the drop without gravity), refined ``levels`` times and relaxed."""
    height = np.real(np.roots([1.0, 0.0, 3.0, -6*volume/np.pi])).max()  # pi h (3 + h^2)/6 = V
    radius = (1 + height**2)/(2*height)            # the sphere through the ring
    cap = shapes.cap(radius, height, nr=nr, nt=nt)
    vertices = RING*cap.vertices*np.array([1.0, 1.0, -1.0])   # hang it below z = 0
    faces = cap.faces[:, ::-1]                     # normals out of the liquid again
    ev = pyse.Evolver()
    ev.load_arrays(vertices, faces,
                   boundaries={1: C.circle((0, 0, 0), RING)},
                   vertex_boundaries={1: cap.rims["rim"]}, gravity=bond/RING**2,
                   bodies=[pyse.Body(volume=volume*RING**3, density=1.0)])
    ev.relax(levels=levels)
    return ev
