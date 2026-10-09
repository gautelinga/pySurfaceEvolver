"""Constraint builders (pyse.constraints): wetting energies and volumes against
exact values. Plane integrands are polynomials along straight contact lines, so
plane cases agree with the discrete geometry to round-off; curved walls are
compared with the analytic values and checked to converge."""

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C

THETA = 50.0
K = np.cos(np.radians(THETA))


def _load(vertices, faces, cons, on, volconst=None):
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        np.asarray(vertices, float), faces, constraints=cons,
        vertex_constraints={k: [i for i, c in enumerate(on) if k in c] for k in cons},
        bodies=[pyse.Body(faces=range(len(faces)), volconst=volconst)]))
    ev.recalc()
    return ev


def _area(v, faces):
    v = np.asarray(v, float)
    f = np.asarray(faces)
    return 0.5*np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()


# ---- planes ----------------------------------------------------------------------

def _cap(r=1.0, h=0.6, nr=8, nt=24):
    """A spherical cap on z = 0 (outward normals), and its rim rows."""
    a = np.sqrt(h*(2*r - h))
    t_max = np.arcsin(a/r)
    v, f = [[0, 0, h]], []
    for i in range(1, nr + 1):
        t = t_max*i/nr
        v += [[r*np.sin(t)*np.cos(p), r*np.sin(t)*np.sin(p), h - r + r*np.cos(t)]
              for p in 2*np.pi*np.arange(nt)/nt]
    ring = lambda i, j: 1 + (i - 1)*nt + j % nt
    f += [[0, ring(1, j), ring(1, j + 1)] for j in range(nt)]
    for i in range(1, nr):
        for j in range(nt):
            f += [[ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)],
                  [ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)]]
    return np.array(v), f, [ring(nr, j) for j in range(nt)]


def _closed_volume(v, faces, rim):
    """Volume enclosed by the facets and the flat polygon through the rim (its
    triangles oriented against the facets' rim edges, to close the surface)."""
    v = np.asarray(v, float)
    c = v[rim].mean(axis=0)
    directed = {(f[i], f[(i + 1) % 3]) for f in faces for i in range(3)}
    tri = [v[f] for f in faces]
    for j in range(len(rim)):
        p, q = rim[j], rim[(j + 1) % len(rim)]
        a, b = (q, p) if (p, q) in directed else (p, q)
        tri.append(np.array([c, v[a], v[b]]))
    return sum(np.dot(t[0], np.cross(t[1], t[2])) for t in tri)/6


def _rim_area(v, rim):
    p = np.asarray(v, float)[rim]
    return 0.5*np.linalg.norm(sum(np.cross(p[j], p[(j + 1) % len(p)]) for j in range(len(p))))


@pytest.mark.parametrize("case", ["table", "ceiling", "tilted"])
def test_plane_wetting_and_volume_are_exact(case):
    v, f, rim = _cap()
    rot, shift = np.eye(3), np.array([0.0, 0.0, 0.3])
    if case == "ceiling":          # the cap hanging from z = 0.3, liquid below
        v = v*[1, 1, -1]
        f = [face[::-1] for face in f]
    if case == "tilted":
        a, b = 0.4, 0.7
        rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
        rot = rz @ rx
    v = v @ rot.T + shift
    normal = rot @ [0, 0, -1 if case == "ceiling" else 1]
    con = C.plane(normal, point=shift, contact_angle=THETA)
    ev = _load(v, f, {1: con}, [{1} if i in rim else set() for i in range(len(v))])
    assert ev.total_energy == pytest.approx(_area(v, f) - K*_rim_area(v, rim), rel=1e-12)
    assert abs(ev.bodies().volume[0]) == pytest.approx(abs(_closed_volume(v, f, rim)), rel=1e-12)
    assert ev.bodies().volume[0] > 0


# ---- a film around a quarter sphere / a vertical fibre ---------------------------

def _film_around_quarter_disk(a, L, h0, n):
    """The plane z = h0 over [0, L]^2 minus the quarter disk of radius a;
    normals +z (the liquid below). Constraint tags: 3 the inner arc, 4 x = 0,
    6 y = 0, 5 x = L, 7 y = L."""
    phis = np.linspace(0, np.pi/2, 2*n + 1)
    ss = np.linspace(0, 1, n + 1)
    v, on = [], []
    for k, p in enumerate(phis):
        outer = (L, L*np.tan(p)) if p <= np.pi/4 else (L/np.tan(p), L)
        for s in ss:
            xy = (1 - s)*a*np.array([np.cos(p), np.sin(p)]) + s*np.array(outer)
            v.append([xy[0], xy[1], h0])
            tags = set()
            if s == 0: tags.add(3)
            if k == 0: tags.add(6)
            if k == len(phis) - 1: tags.add(4)
            if s == 1: tags |= ({5} if p < np.pi/4 else {7} if p > np.pi/4 else {5, 7})
            on.append(tags)
    m = len(ss)
    f = []
    for k in range(len(phis) - 1):
        for j in range(n):
            q = [k*m + j, k*m + j + 1, (k + 1)*m + j + 1, (k + 1)*m + j]
            f += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return np.array(v), f, on


def _mirrors(L):
    return {4: C.mirror("x"), 6: C.mirror("y"), 5: C.mirror("x", L), 7: C.mirror("y", L)}


def _sphere_film(n, R=0.5, h0=0.3, L=1.0, wall=None):
    a = np.sqrt(R*R - h0*h0)
    v, f, on = _film_around_quarter_disk(a, L, h0, n)
    wall = wall or C.sphere((0, 0, 0), R, contact_angle=THETA, span=np.pi/2)
    return _load(v, f, {3: wall, **_mirrors(L)}, on,
                 volconst=getattr(wall, "volconst", 0.0) or None), v, f


def test_sphere_zone_converges_to_exact():
    R, h0, L = 0.5, 0.3, 1.0
    a = np.sqrt(R*R - h0*h0)
    errors = []
    for n in (8, 16, 32):
        ev, v, f = _sphere_film(n, R, h0, L)
        wetted = 2*np.pi*R*h0/4                              # the zone 0 < z < h0
        energy = _area(v, f) - K*wetted                      # the film is the polygon
        volume = L*L*h0 - np.pi/4*(R*R*h0 - h0**3/3)         # box minus the sphere part
        errors.append((abs(ev.total_energy - energy), abs(ev.bodies().volume[0] - volume)))
    (e1, v1), (e2, v2), (e3, v3) = errors
    assert e3 < 2e-4 and v3 < 2e-4
    assert e3 < e2/3 and v3 < v2/3                           # second order in the chords


def test_sphere_matches_the_hand_written_bead_integrals():
    # the drainage example's bead, verified there against exact sheets and rings
    R, k = 0.5, K
    dphi = lambda g, n: f"{n}1: (-y*({g}))/(x^2 + y^2)\n{n}2: (x*({g}))/(x^2 + y^2)\n{n}3: 0\n"
    hand = (f"formula: x^2 + y^2 + z^2 = {R*R!r}\nenergy:\n" + dphi(f"{k!r}*{R!r}*z", "e")
            + "content:\n" + dphi("z^3/3", "c"))
    ev, _, _ = _sphere_film(12, R, wall=None)
    built = (ev.total_energy, ev.bodies().volume[0])
    ev, _, _ = _sphere_film(12, R, wall=hand)
    assert built == pytest.approx((ev.total_energy, ev.bodies().volume[0]), rel=1e-13)
    wet = C.sphere((0, 0, 0), R, contact_angle=THETA, span=np.pi/2, wet_poles=("north",))
    assert wet.volconst == pytest.approx(-np.pi*R**3/6)
    assert wet.energy_constant == pytest.approx(-K*np.pi*R*R/2)


def test_vertical_fibre_converges_to_exact():
    R, h0, L = 0.4, 0.3, 1.0
    errors = []
    for n in (8, 16, 32):
        v, f, on = _film_around_quarter_disk(R, L, h0, n)
        wall = C.cylinder((0, 0, 0), (0, 0, 1), R, contact_angle=THETA)
        ev = _load(v, f, {3: wall, **_mirrors(L)}, on)
        energy = _area(v, f) - K*R*np.pi/2*h0
        volume = (L*L - np.pi*R*R/4)*h0
        errors.append((abs(ev.total_energy - energy), abs(ev.bodies().volume[0] - volume)))
    assert errors[-1][0] < 2e-4 and errors[-1][1] < 2e-4
    assert errors[-1][0] < errors[-2][0]/3


def _rigid(case):
    """A rigid motion x -> Q x + t."""
    if case == "identity":
        return np.eye(3), np.zeros(3)
    a, b, c = 0.5, -0.3, 0.8
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    t = np.array([0.3, -0.2, 0.7])
    if case == "translated":
        return np.eye(3), t
    if case == "turned about z":
        return rz, t
    return rz @ ry @ rx, t


def _moved_mirror(q, t, normal, point):
    return C.mirror((q @ normal, q @ point + t))


@pytest.mark.parametrize("case", ["identity", "translated", "turned about z"])
def test_horizontal_fibre_is_exact(case):
    # a fibre along x through the mid-plane z = 0; the film z = h0 meets it in
    # the straight line y = b0: the contact line is exact, so is everything.
    # Moved rigidly (mirrors and all), energy and volume must stay the same.
    R, h0, L, n = 0.4, 0.25, 1.0, 6
    b0 = np.sqrt(R*R - h0*h0)
    xs, ys = np.linspace(0, L, n + 1), np.linspace(b0, L, n + 1)
    v, on = [], []
    for y in ys:
        for x in xs:
            v.append([x, y, h0])
            on.append({k for k, c in ((3, y == b0), (7, y == L), (4, x == 0), (5, x == L)) if c})
    m = len(xs)
    f = []
    for i in range(n):
        for j in range(n):
            q = [i*m + j, i*m + j + 1, (i + 1)*m + j + 1, (i + 1)*m + j]
            f += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    Q, t = _rigid(case)
    v = np.array(v) @ Q.T + t
    wall = C.cylinder(t, Q @ [1, 0, 0], R, contact_angle=THETA, gauge="axial", ref=Q @ [0, 1, 0])
    mirrors = {4: _moved_mirror(Q, t, [1, 0, 0], [0, 0, 0]),
               5: _moved_mirror(Q, t, [1, 0, 0], [L, 0, 0]),
               7: _moved_mirror(Q, t, [0, 1, 0], [0, L, 0])}
    ev = _load(v, f, {3: wall, **mirrors}, on)
    wetted = R*np.arcsin(h0/R)*L
    segment = (h0*np.sqrt(R*R - h0*h0) + R*R*np.arcsin(h0/R))/2   # integral of sqrt(R^2-z^2)
    assert ev.total_energy == pytest.approx(L*(L - b0) - K*wetted, rel=1e-12)
    # the floor (on the mid-plane, L x (L - R)) touches no contact line, so no
    # line integral closes it; Evolver's volume misses its z dx dy
    centre = Q @ [L/2, (L + R)/2, 0] + t
    floor = (Q @ [0, 0, -1])[2]*L*(L - R)*centre[2]
    assert ev.bodies().volume[0] == pytest.approx(L*(L*h0 - segment) - floor, rel=1e-11)


@pytest.mark.parametrize("shift", [(0, 0, 0.6), (0.2, -0.4, -0.3)])
def test_sphere_moved_gives_the_same(shift):
    # the sphere's axis stays parallel to z: translate the whole setup
    R, h0, L, n = 0.5, 0.3, 1.0, 12
    a = np.sqrt(R*R - h0*h0)
    v, f, on = _film_around_quarter_disk(a, L, h0, n)
    results = []
    for t in (np.zeros(3), np.array(shift, float)):
        wall = C.sphere(t, R, contact_angle=THETA, span=np.pi/2)
        mirrors = {4: C.mirror(((1, 0, 0), t)), 6: C.mirror(((0, 1, 0), t)),
                   5: C.mirror(((1, 0, 0), t + [L, 0, 0])), 7: C.mirror(((0, 1, 0), t + [0, L, 0]))}
        ev = _load(v + t, f, {3: wall, **mirrors}, on)
        results.append((ev.total_energy, ev.bodies().volume[0]))
    # the floor on the mid-plane (the box minus the quarter disk) touches no
    # contact line: its z dx dy is the only difference
    floor = -(L*L - np.pi*R*R/4)*shift[2]
    # equal up to the contact line's chords (the floor term is exact)
    scale = 1e-3*(abs(results[0][1]) + abs(floor))      # chords against the exact circle
    assert results[1][0] == pytest.approx(results[0][0], rel=1e-12)
    assert results[1][1] == pytest.approx(results[0][1] - floor, abs=scale)


def test_immersed_slit_meniscus_is_exact():
    # the drainage example's start: a bead (R = 0.48) fully under a meniscus that
    # is straight along x and a circular arc across the slit; an eighth cell:
    # mirrors x = 0, x = ell/2, y = 0 (unused) and z = 0, the wall z = 0.5
    # (contact angle 60), the bead wetted all over (no contact line: only the
    # constants act). Exact against the discrete geometry.
    R, half, y0, tw, tb = 0.48, 0.55, 0.6, 60.0, 40.0
    r = 0.5/np.cos(np.radians(tw))
    zs, xs = np.linspace(0, 0.5, 9), np.linspace(0, half, 5)
    v, on = [], []
    for z in zs:
        for x in xs:
            v.append([x, y0 + r - np.sqrt(r*r - z*z), z])
            on.append({k for k, c in ((4, x == 0), (5, x == half), (1, z == 0), (2, z == 0.5)) if c})
    m = len(xs)
    f = []
    for i in range(len(zs) - 1):
        for j in range(m - 1):
            q = [i*m + j, (i + 1)*m + j, (i + 1)*m + j + 1, i*m + j + 1]
            f += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    bead = C.sphere((0, 0, 0), R, contact_angle=tb, span=np.pi/2, wet_poles=("north",))
    cons = {1: C.mirror("z"), 2: C.plane((0, 0, -1), point=(0, 0, 0.5), contact_angle=tw),
            3: bead, 4: C.mirror("x"), 5: C.mirror("x", half)}
    ev = _load(v, f, cons, on, volconst=bead.volconst)
    prof = np.array(v)[:len(zs)*m:m][:, 1:]          # the profile at x = 0: (y, z) per row
    y_top = prof[-1, 0]
    wetted_wall = half*y_top
    area_yz = np.sum(np.diff(prof[:, 1])*(prof[1:, 0] + prof[:-1, 0])/2)   # under the profile
    energy = _area(v, f) - np.cos(np.radians(tw))*wetted_wall \
        - np.cos(np.radians(tb))*np.pi*R*R/2
    volume = half*area_yz - np.pi*R**3/6
    assert ev.total_energy + bead.energy_constant == pytest.approx(energy, rel=1e-12)
    assert ev.bodies().volume[0] == pytest.approx(volume, rel=1e-12)


def test_builders_check_their_input():
    with pytest.raises(ValueError):
        C.plane((0, 0, 0), 1.0)
    with pytest.raises(ValueError):
        C.sphere((0, 0, 0), 1.0, wet_poles=("east",))
    with pytest.raises(ValueError):
        C.mirror("w")
    assert "formula" in str(C.mirror("x", 0.5))


def test_contact_angle_as_a_parameter():
    # the same cap with the angle as a datafile parameter: same energy, and
    # changing the parameter changes the wetting energy
    v, f, rim = _cap()
    on = [{1} if i in rim else set() for i in range(len(v))]
    numeric = _load(v, f, {1: C.plane((0, 0, 1), contact_angle=THETA)}, on).total_energy
    ev = pyse.Evolver()
    ev.load_string(pyse.make_datafile(
        v, f, constraints={1: C.plane((0, 0, 1), contact_angle="theta")},
        vertex_constraints={1: rim}, bodies=[pyse.Body(faces=range(len(f)))],
        parameters={"theta": THETA}))
    assert ev.total_energy == pytest.approx(numeric, rel=1e-13)
    ev.parameters["theta"] = 90.0
    assert ev.total_energy == pytest.approx(_area(v, f), rel=1e-12)
    s = C.sphere((0, 0, 0), 1.0, contact_angle="theta", wet_poles=("north",), span=np.pi)
    assert np.isnan(s.energy_constant) and s.area_constant == pytest.approx(np.pi)
