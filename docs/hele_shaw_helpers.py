"""The Hele-Shaw cell of docs/hele_shaw_defects.md: two plates 3.6 mm apart,
a flat cover and a printed plate with defects where the gap narrows, after Holtzman et al.
(Phys. Rev. Fluids 9, 064001, 2024, Appendix D): the capillary pressure
2 gamma/b rises 60 Pa along a 10 mm ramp and stays there for a 5 mm plateau,
then the gap opens again. Silicone oil (20 mN/m, 0.93 g/mL) against air,
the cell tilted 38 degrees, lying on the printed plate. z runs from the
cover (z = 0) to the printed plate (z = b), so gravity across the gap
pulls toward +z. Units: mm and the oil's surface tension, so a
pressure of 1 is 20 Pa.

* ``height(centres)``: the printed plate z = b(x, y) as a formula, defects
  10 mm wide at x = centres (and their mirror images across x = 0; None: one
  defect across the whole width).
* ``cell(centres, y0)``: a flat meniscus across the cell at y = y0, the oil
  below it, between mirror planes x = 0 and x = X; ``contact_line(ev, x)``:
  where the meniscus meets the printed plate near x.
* ``meniscus(b, theta)``: the pressure of a meniscus between parallel plates a
  gap b apart and the area per width it leaves dry between them.
* ``envelope(V, P, up)``: pressure control along a traced P(V) curve, its
  jumps and the energy each dissipates.
* ``half_gap(points, x, centres)``: where a meniscus crosses half the gap near
  x, which is what a camera looking through the plates sees of it.
* ``line_model(centres, width, slope)``: the paper's model, the meniscus as a
  line in the plane of the cell, through the same cycle.
"""

import numpy as np

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C

B0 = 3.6                     # the gap
W, LS, LP = 10.0, 10.0, 5.0  # a defect's width, ramp and plateau
P0 = 2/B0                    # the design's capillary pressure 2 gamma/b (perfect wetting)
PI = 0.3                     # its rise along the ramp (6 kPa/m)
S = 0.25                     # the printed edges smoothed over this
THETA = 20.0                 # the oil's contact angle
TILT = 38.0
RHO_G = 930*9.81/0.020*1e-6  # rho g/gamma, per mm^2
GY, GZ = RHO_G*np.sin(np.radians(TILT)), -RHO_G*np.cos(np.radians(TILT))  # (toward +z)
X = 20.0                     # the cell's half-width modelled (mirror planes x = 0, X)


def height(centres=None, rounding=S) -> str:
    """The printed plate z = b(x, y): the ramp from y = 0 to LS, the plateau to
    LS + LP, at x = centres (and -centres; None: all x)."""
    sp = lambda z: f"(maximum({z}, 0) + log(1 + exp(-abs({z}))))"  # noqa: E731
    S_ = float(rounding)
    ramp = f"({S_}*({sp(f'y/{S_}')} - {sp(f'(y - {LS})/{S_}')}))"
    gap = f"(2/({P0!r} + {PI}*{ramp}))"
    end = f"(0.5*(1 - tanh((y - {LS + LP})/{S_})))"
    if centres is None:
        return f"{B0} - ({B0} - {gap})*{end}"
    xs = [f"0.5*(tanh((x - {c - W/2})/{S_}) - tanh((x - {c + W/2})/{S_}))"
          for c0 in centres for c in sorted({c0, -c0})]
    return f"{B0} - ({B0} - {gap})*({' + '.join(xs)})*{end}"


def gap(y, x=0.0, centres=None, rounding=S):
    """b(x, y) evaluated, for the references."""
    y = np.asarray(y, float)
    return C._height_values(height(centres, rounding), np.full_like(y, x), y)


def profile() -> str:
    """A defect's top as a function of y alone, the ramp from y = 0 (flush
    with the open plate) and the plateau after LS, the 2.7 degree turn
    between them smoothed over S: a sharp block's top (constraints.plate)."""
    sp = lambda z: f"(maximum({z}, 0) + log(1 + exp(-abs({z}))))"  # noqa: E731
    return f"2/({P0!r} + {PI}*(y - {S}*{sp(f'(y - {LS})/{S}')}))"


def cell(centres=None, y0=-4.0, *, width=X, nx=41, nz=7, gravity=True,
         spacing=None, ylo=-12.0, yhi=30.0, sharp=False, rounding=S):
    """A flat meniscus across the cell at y = y0, the oil below it (to y =
    ylo, where the walls' integrals start), between mirror planes x = 0 and
    x = width; nx by nz vertices. The volume is held at the start's. The
    defects' edges are rounded over ``rounding`` (its tables then a fifth of
    that apart, unless ``spacing`` is given). With
    ``sharp``, the defects are blocks with sharp edges (constraints.plate:
    its faces ``printed_open``, ``printed_top1``, ...; y0 before them)."""
    h = height(centres, rounding)
    if spacing is None:            # (the tables a fifth of the rounding apart)
        spacing = min(0.05, rounding/5)
    if sharp and (centres is None or y0 >= 0):
        raise ValueError("cell(sharp=True) needs the defects' centres and y0 < 0")
    xs, zs = np.linspace(0, width, nx), np.linspace(0, 1, nz)
    V = np.array([[x, y0, z*gap(y0, x, centres, rounding)] for x in xs for z in zs])
    F = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            a, b, c, d = i*nz + j, (i + 1)*nz + j, (i + 1)*nz + j + 1, i*nz + j + 1
            F += [(a, c, d), (a, b, c)]
    F = np.array(F)
    if np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[:, 1].mean() < 0:
        F = F[:, ::-1]                   # (the normal out of the oil, along +y)
    gy, gz = (GY, GZ) if gravity else (0.0, 0.0)
    # (the weight of the oil: its free surface's share here, the printed
    # plate's in the heightfield, the cover's is 0)
    header = ("quantity oil_weight energy method facet_vector_integral global\n"
              f"vector_integrand:\nq1: 0\nq2: 0\nq3: {gy!r}*y*z + {gz!r}*z^2/2\n"
              if gravity else "")
    if sharp:
        blocks = [C.Block((c - W/2, c + W/2) if c > 0 else (-W/2, W/2), (0.0, LS + LP),
                          profile()) for c in centres]
        printed = C.plate(B0, blocks, THETA, x_range=(0, width), y_from=ylo,
                          gravity=(gy, gz), spacing=spacing/5, name="printed")
    else:
        printed = {"printed": C.heightfield(h, (-0.5, width + 0.5), (ylo, yhi), THETA,
                                            spacing=spacing, gravity=(gy, gz), refine=4)}
    ev = pyse.Evolver()
    ev.load_arrays(V, F, header=header,
                   constraints={"cover": C.plane((0, 0, 1), 0.0, contact_angle=THETA,
                                                 origin=(0, ylo, 0)),
                                **printed,
                                "middle": C.mirror("x", 0.0), "side": C.mirror("x", width)},
                   vertex_constraints={"cover": np.flatnonzero(np.isclose(V[:, 2], 0)),
                                       "printed_open" if sharp else "printed":
                                       np.arange(nz - 1, len(V), nz),
                                       "middle": np.flatnonzero(np.isclose(V[:, 0], 0)),
                                       "side": np.flatnonzero(np.isclose(V[:, 0], width))},
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    return ev


def contact_line(ev, x=None, half=0.5):
    """The median y of the meniscus's vertices on the printed plate within half
    of x (None: all of them)."""
    v = ev.vertices
    on = np.zeros(len(v), bool)
    for name in _printed(ev):
        on |= np.asarray(ev.on_constraint(name), bool)
    if x is not None:
        on &= np.abs(v[:, 0] - x) < half
    return float(np.median(v[on, 1]))


def _printed(ev):
    """The printed plate's constraints: the heightfield, or a sharp plate's faces."""
    from pysurfaceevolver._engine import surface
    return [k for k in surface.walls if str(k) == "printed" or str(k).startswith("printed_")]


def meniscus(b, theta=THETA):
    """Between parallel plates a gap b apart, no gravity: the meniscus is a
    circular arc of radius b/(2 cos theta). Its pressure, and the area per
    width between it and the line joining its contact lines (dry, for
    theta < 90)."""
    t = np.radians(theta)
    r, a = b/(2*np.cos(t)), np.pi/2 - t
    return -1/r, r*r*(2*a - np.sin(2*a))/2


def envelope(V, P, up=True):
    """Pressure control along a traced P(V) curve (V in its order): the
    running maximum of P (imbibition, up) or minimum (drainage), and its
    jumps as (first row, last row, pressure, V before, V after, energy
    dissipated): the area between the curve and the jump's pressure, the
    work the jump turns into heat."""
    P = np.asarray(P, float)
    env = np.maximum.accumulate(P) if up else np.minimum.accumulate(P)
    off = (P < env - 1e-9) if up else (P > env + 1e-9)
    jumps, k = [], 0
    while k < len(P):
        if not off[k]:
            k += 1
            continue
        j = k
        while j < len(P) and off[j]:
            j += 1
        a, b = k - 1, min(j, len(P) - 1)
        seg = slice(a, b + 1)
        d, v = P[seg] - env[seg], np.asarray(V, float)[seg]
        area = abs(np.sum((d[1:] + d[:-1])/2*np.diff(v)))
        jumps.append((a, b, env[a], V[a], V[b], area))
        k = j
    return env, jumps


def half_gap(points, x, centres=None, rounding=S, band=0.3):
    """Where the meniscus (its vertices, ``points``) crosses half the gap within
    band of x: the median y of the vertices there between 0.38 and 0.62 of the
    local gap. Outside the defects gravity across the gap makes the meniscus
    lean, its line on the printed plate ahead of the rest; this is the meniscus
    as seen through the plates. NaN where there are none."""
    p = np.asarray(points, float)
    p = p[np.abs(p[:, 0] - x) < band]
    if not len(p):
        return float("nan")
    ys = np.linspace(p[:, 1].min() - 1, p[:, 1].max() + 1, 2001)
    s = p[:, 2]/np.interp(p[:, 1], ys, gap(ys, x, centres, rounding))
    on = np.abs(s - 0.5) < 0.12
    return float(np.median(p[on, 1])) if on.any() else float("nan")


def line_model(centres=None, width=X, slope=PI*np.cos(np.radians(THETA)), *,
               lo=-3.0, hi=16.0, dh=0.02, dx=0.1, rounding=S):
    """Holtzman et al.'s model: the meniscus a line y = h(x) in the plane of the
    cell, its energy (per gamma) the integral of h'^2/2 + (h - h0)^2/(2 lc^2) less
    the capillary pressure's work, slope*y along a defect's ramp (y 0 to LS) and
    slope*LS on its plateau (to LS + LP), the defects' edges rounded over
    ``rounding`` as here. lc^2 = gamma/(rho g sin TILT). Mirror planes x = 0 and
    width, defects 10 wide at x = centres (None: one at x = 0). The undisturbed
    line h0 goes from lo up to hi and back in steps dh, each state relaxed from
    the last (so a jump happens where the state it was on stops being a minimum).
    ``slope``: per mm^2 in gamma; the default, the design's 6 kPa/m times
    cos(THETA). Returns rows (imbibition, h_f, eta_m): h_f where the line meets
    the far mirror plane, eta_m the line at the first defect's middle less h_f."""
    from scipy.optimize import minimize
    centres = (0.0,) if centres is None else tuple(centres)
    inv_lc2 = RHO_G*np.sin(np.radians(TILT))
    xs = np.arange(0, width + dx/2, dx)
    sp = lambda z: np.logaddexp(0, z)  # noqa: E731
    inside = np.minimum(1, sum(0.5*(np.tanh((xs - c + W/2)/rounding) - np.tanh((xs - c - W/2)/rounding))
                               for c0 in centres for c in {c0, -c0}))
    yt = np.linspace(lo - 3, hi + 15, 20001)
    qt = slope*rounding*(sp(yt/rounding) - sp((yt - LS)/rounding))*0.5*(1 - np.tanh((yt - LS - LP)/rounding))
    Qt = np.concatenate([[0.0], np.cumsum(0.5*(qt[1:] + qt[:-1])*np.diff(yt))])
    Qt -= np.interp(0.0, yt, Qt)
    wts = np.full(len(xs), dx)
    wts[[0, -1]] = dx/2

    def energy(h, h0):
        d = np.diff(h)/dx
        e = 0.5*np.sum(d*d)*dx + np.sum(wts*(0.5*inv_lc2*(h - h0)**2 - inside*np.interp(h, yt, Qt)))
        g = wts*(inv_lc2*(h - h0) - inside*np.interp(h, yt, qt))
        g[:-1] -= d
        g[1:] += d
        return e, g

    h, rows = np.full(len(xs), lo), []
    mid = int(np.argmin(np.abs(xs - centres[0])))
    for up, h0s in ((True, np.arange(lo, hi + 1e-9, dh)), (False, np.arange(hi, lo - 1e-9, -dh))):
        for h0 in h0s:
            h = minimize(energy, h, args=(h0,), jac=True, method="L-BFGS-B",
                         options=dict(maxiter=50000, gtol=1e-11, ftol=1e-15, maxcor=30)).x
            rows.append((up, h[-1], h[mid] - h[-1]))
    return np.array(rows, float)
