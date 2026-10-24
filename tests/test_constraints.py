"""Constraint builders (pyse.constraints): wetting energies and volumes against
exact values. Plane integrands are polynomials along straight contact lines, so
plane cases agree with the discrete geometry to round-off; curved walls are
compared with the analytic values and checked to converge."""

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C

THETA = 50.0
K = float(np.cos(np.radians(THETA)))


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


@pytest.mark.parametrize("case", ["along x", "tilted", "turned"])
def test_drop_round_a_fibre_has_its_volume_however_the_fibre_points(case):
    # a band of liquid round a fibre: on a vertical fibre the wetted face
    # adds nothing to z dx dy, turned it does, with the sign that keeps the
    # volume (it was added instead of taken off)
    from pysurfaceevolver import shapes
    z = np.linspace(-1, 1, 9)
    drop = shapes.revolve(np.sqrt(0.16 + 0.5*(1 - z*z)), z, nt=96)
    rims = drop.rims["bottom"] + drop.rims["top"]
    q, t = {"along x": (np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), np.zeros(3)),
            "tilted": (_rigid("turned")[0], np.zeros(3)),
            "turned": _rigid("turned")}[case]
    volumes = []
    for Q, shift in ((np.eye(3), np.zeros(3)), (q, t)):
        ev = pyse.Evolver()
        ev.load_arrays(drop.vertices @ Q.T + shift, drop.faces,
                       constraints={"fibre": C.cylinder(shift, Q @ [0, 0, 1], 0.4,
                                                        contact_angle=THETA)},
                       vertex_constraints={"fibre": rims}, bodies=[pyse.Body(volume=None)])
        volumes.append(ev.bodies().volume[0])
    assert volumes[1] == pytest.approx(volumes[0], rel=1e-3)


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


def _bridge(gap, centres_from_parameter, levels=2, theta=40.0, volume=0.1, a=0.35):
    """A liquid bridge between unit spheres gap apart, wetted around the poles
    facing each other."""
    c = 1 + gap/2
    zb = c - np.sqrt(1 - a*a)
    tube = pyse.shapes.tube(a, -zb, zb, nz=4, nt=16)
    lo, hi = ((0, 0, "-(1 + gap/2)"), (0, 0, "1 + gap/2")) if centres_from_parameter \
        else ((0, 0, -c), (0, 0, c))
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   constraints={1: C.sphere(lo, 1.0, contact_angle=theta, wet_poles=("north",)),
                                2: C.sphere(hi, 1.0, contact_angle=theta, wet_poles=("south",))},
                   vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=volume)],
                   parameters={"gap": gap} if centres_from_parameter else None)
    ev.relax(levels=levels)
    return ev


def test_sphere_centre_from_a_parameter():
    # the volume over the wet pole moves into the line integral: the same
    # bridge (to the integrals' quadrature along the contact line)
    fixed = _bridge(0.4, False)
    energy, pressure = fixed.energy, fixed.body(1).pressure    # (one engine: read first)
    moving = _bridge(0.4, True)
    assert moving.energy == pytest.approx(energy, rel=2e-5)
    assert moving.body(1).pressure == pytest.approx(pressure, rel=1e-4)
    s = C.sphere((0, 0, "1 + gap/2"), 1.0, contact_angle="theta", wet_poles=("south",))
    assert s.parameters == ("gap", "theta") and s.volconst == 0.0
    with pytest.raises(ValueError):
        C.sphere((0, 0, "c"), 1.0, wet_poles=("north", "south"))


def test_arclength_moves_a_sphere_by_its_parameter():
    from pysurfaceevolver import recipes
    ev = _bridge(0.4, True)
    steps = list(recipes.arclength(ev, "gap", 0.05, 4))
    gaps = [s.value for s in steps]
    assert np.all(np.diff(gaps) > 0) and gaps[-1] > 0.5
    # the contact line on the moved sphere, and the bridge there as built afresh
    top = np.flatnonzero(ev.vertices[:, 2] > 0)
    on = np.abs(np.linalg.norm(ev.vertices[top] - [0, 0, 1 + gaps[-1]/2], axis=1) - 1) < 1e-9
    assert on.sum() >= 16
    # drawn where it is now
    wall = ev._walls(True)[1]
    assert wall.shape[1] == pytest.approx((0, 0, 1 + gaps[-1]/2))
    energy = ev.energy
    assert _bridge(gaps[-1], False).energy == pytest.approx(energy, rel=5e-3)


def test_texts_hold_plain_numbers():
    # numpy >= 2 formats numpy floats as np.float64(...): none may reach a datafile,
    # whichever numpy runs the tests
    f64 = np.float64
    for con in (C.plane(np.array([0, 0, -1.0]), point=np.array([0, 0, 0.5]), contact_angle=f64(60)),
                C.mirror("z"), C.mirror("x", f64(0.55)),
                C.sphere(np.zeros(3), f64(0.48), contact_angle=f64(40), wet_poles=("north",),
                         span=f64(np.pi/2)),
                C.cylinder(np.zeros(3), np.array([1.0, 0, 0]), f64(0.4), contact_angle=f64(50),
                           gauge="axial", ref=np.array([0, 1.0, 0])),
                C.cylinder(np.zeros(3), np.array([0, 0, 1.0]), f64(0.4), contact_angle=f64(50))):
        assert "np." not in con.text() and "float64" not in con.text()


def test_arclength_along_a_mode_onto_the_off_centre_bridges():
    # a shrinking bridge loses stability where bridges with the neck off the
    # middle branch off (no fold: the volume goes on down); along its mode,
    # either way, the run follows them (mirror images of each other)
    from pysurfaceevolver import recipes
    ev = _bridge(0.4, False, levels=1, volume=0.035)
    for s in recipes.arclength(ev, ev.body(1), -0.004, 30):
        if s.result.negative_modes:
            break
    assert s.result.negative_modes == 1 and 0.022 < s.value < 0.026    # (exact 0.0248)
    mode = ev.modes(1).vectors[0]
    start = ev.save()
    runs = []
    for sign in (1, -1):
        steps = list(recipes.arclength(ev, ev.body(1), sign*0.02, 4, along=mode))
        x = ev.vertices
        r = [np.hypot(*x[ev.on_constraint(k)][:, :2].T).mean() for k in (1, 2)]
        runs.append((steps, r[1] - r[0]))      # the contact lines' radii: one wider
        ev.restore(start)
    (up, wider_up), (down, wider_down) = runs
    assert len(up) == 4 and np.all(np.diff([s.value for s in up]) > 0)
    assert all(s.result.negative_modes == 1 and s.result.residual < 1e-8 for s in up)
    assert abs(wider_up) > 0.05 and wider_down == pytest.approx(-wider_up, rel=1e-3)
    assert [s.energy for s in down] == pytest.approx([s.energy for s in up], rel=1e-6)
    with pytest.raises(ValueError):
        next(recipes.arclength(ev, ev.body(1), 0.02, 4, along=mode[:-1]))


def test_relax_levels_from_a_crude_low_angle_start():
    # a 20-degree bridge from a tube 12 facets around: relaxed at that
    # coarseness first, which equilibrium it found was chance (its volume
    # moved by a few units in the last place: 6 of 13 failed, at the last
    # level); relax(levels=2) now starts again one level finer when the last
    # level fails, and these all converge, to about the same energy
    v0 = 1.4*0.18484601870300293
    energies = []
    for k in (-6, -4, 0, 2):
        c, a = 1.35, 0.35
        zb = c - np.sqrt(1 - a*a)
        tube = pyse.shapes.tube(a, -zb, zb, nz=4, nt=12)
        ev = pyse.Evolver()
        ev.load_arrays(tube.vertices, tube.faces,
                       constraints={1: C.sphere((0, 0, -c), 1.0, contact_angle=20.0,
                                                wet_poles=("north",)),
                                    2: C.sphere((0, 0, c), 1.0, contact_angle=20.0,
                                                wet_poles=("south",))},
                       vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                       bodies=[pyse.Body(volume=v0 + k*np.spacing(v0))])
        r = ev.relax(levels=2)
        assert r.converged and ev.mesh_quality().angle_min > 10
        energies.append(ev.energy)
    assert max(energies) == pytest.approx(min(energies), rel=5e-3)     # (the meshes differ)


def test_relax_lowers_a_small_contact_angle_in_stages():
    # a bridge at 5 degrees from a fat tube (far more liquid than its volume):
    # relaxed at 5 degrees outright, the vertices next to the contact lines
    # sank into the spheres, the energy falling without end (-1.6 against
    # -0.35); relax() now starts again with the angle raised to 20 degrees
    # and lowers it in stages
    gap, a = 0.2, 0.7
    c = 1 + gap/2
    zb = c - np.sqrt(1 - a*a)
    tube = pyse.shapes.tube(a, -zb, zb, nz=4, nt=12)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   constraints={1: C.sphere((0, 0, -c), 1.0, contact_angle=5.0, wet_poles=("north",)),
                                2: C.sphere((0, 0, c), 1.0, contact_angle=5.0, wet_poles=("south",))},
                   vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=0.3)])
    r = ev.relax(levels=3)
    assert r.converged and ev.parameters["pyse_angle_floor"] == 0
    x = ev.vertices
    depth = 1 - np.minimum(np.linalg.norm(x - [0, 0, -c], axis=1),
                           np.linalg.norm(x - [0, 0, c], axis=1))
    assert depth.max() < 1e-9
    # the exact axisymmetric bridge (docs/stress_helpers.py): -0.352478
    assert ev.energy == pytest.approx(-0.352478, rel=2e-3)
    # the floor is in the datafile only for small angles and parameters
    assert C.sphere((0, 0, 0), 1.0, contact_angle=5.0).floored
    assert C.sphere((0, 0, 0), 1.0, contact_angle="theta").floored
    assert not C.sphere((0, 0, 0), 1.0, contact_angle=40.0).floored


@pytest.mark.parametrize("angles", [(30, 20, 10), (90, 0, 0), (150, -40, 70)])
def test_sphere_axis_gives_the_same_bridge_turned(angles):
    # the bridge between two spheres turned as a whole, the spheres' axes
    # (their integrals' one singular pole: inside the lower one's wetted patch,
    # outside the upper one's) turned with it: Evolver's volume (z dx dy)
    # isn't turned with it, the forms are. The same as unturned (and as the
    # default z-axis spheres, to the mesh's error: 0.26% here, 0.017% two
    # levels finer)
    from pysurfaceevolver import shapes
    c, a = 1.3, 0.35
    zb = c - np.sqrt(1 - a*a)
    tube = shapes.tube(a, -zb, zb, nz=4, nt=16)

    def bridge(rot, **kw):
        ev = pyse.Evolver()
        ev.load_arrays(tube.vertices @ rot.T, tube.faces,
                       constraints={1: C.sphere(rot @ [0, 0, -c], 1.0, 40.0, wet_poles=("north",),
                                                **kw),
                                    2: C.sphere(rot @ [0, 0, c], 1.0, 40.0,
                                                **(kw if kw else {"wet_poles": ("south",)}))},
                       vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                       bodies=[pyse.Body(volume=0.09)])
        assert ev.relax(levels=2).converged
        return ev.body(1).pressure, ev.energy
    p0, e0 = bridge(np.eye(3))
    p1, e1 = bridge(np.eye(3), axis=(0, 0, 1))
    assert p1 == pytest.approx(p0, rel=4e-3) and e1 == pytest.approx(e0, rel=5e-3)
    def turn(k, t):                     # by t degrees about axis k
        c, s = np.cos(np.radians(t)), np.sin(np.radians(t))
        i, j = (k + 1) % 3, (k + 2) % 3
        r = np.eye(3)
        r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
        return r
    rot = turn(2, angles[2]) @ turn(1, angles[1]) @ turn(0, angles[0])   # about x, then y, then z
    p, e = bridge(rot, axis=rot @ [0, 0, 1.0])
    assert p == pytest.approx(p1, rel=1e-4) and e == pytest.approx(e1, rel=1e-4)


@pytest.mark.parametrize("tilt", [10.0, 40.0])
def test_moving_a_sphere_axis_pole_off_a_contact_line_keeps_the_bridge(tilt):
    # the lower sphere's pole tilted near the contact line (20 degrees round
    # its top): inside the wetted patch, 0.18 from the line, or outside it,
    # 0.34 from it. Moved to the point farthest from the line, the sphere's
    # bottom (from inside the patch, the integrals differ by the whole
    # sphere: kept), the energy and volume carried over (from 0.18, to the
    # old integrals' error), and relaxed, the same bridge as with the pole at
    # the bottom from the start
    from pysurfaceevolver import _surgery, shapes
    from pysurfaceevolver._engine import surface
    c, a = 1.3, 0.35
    zb = c - np.sqrt(1 - a*a)
    tube = shapes.tube(a, -zb, zb, nz=4, nt=16)

    def bridge(axis, wet):
        ev = pyse.Evolver()
        ev.load_arrays(tube.vertices, tube.faces,
                       constraints={1: C.sphere((0, 0, -c), 1.0, 40.0, wet_poles=wet, axis=axis),
                                    2: C.sphere((0, 0, c), 1.0, 40.0, axis=(0, 0, 1))},
                       vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                       bodies=[pyse.Body(volume=0.09)])
        return ev
    ev = bridge((0, 0, -1), ())
    ev.refine(1)
    assert ev.relax().converged and _surgery.move_poles(ev) == []
    far = ev.energy, ev.body(1).pressure
    t = np.radians(tilt)
    ev = bridge((np.sin(t), 0, np.cos(t)), ("north",) if tilt < 20 else ())
    ev.refine(1)
    energy, volume = ev.energy, ev.body(1).volume
    assert _surgery.move_poles(ev) == ["c1_pole1"]
    assert np.allclose(surface.walls["c1_pole1"].pole_axis, (0, 0, -1), atol=0.1)
    # the old constraint (Evolver keeps it) out of sight, of events and
    # contacts too, and so after a snapshot
    ev.restore(ev.save())
    nums, _, _, names = ev._constraint_gaps()
    assert 1 not in [int(n) for n in nums] and "c1_pole1" in names
    assert ev.energy == pytest.approx(energy, abs=0.02 if tilt < 20 else 1e-3)
    assert ev.body(1).volume == pytest.approx(volume, abs=0.01 if tilt < 20 else 1e-3)
    assert ev.relax().converged
    assert ev.energy == pytest.approx(far[0], rel=2e-3)
    assert ev.body(1).pressure == pytest.approx(far[1], rel=3e-3)


def test_a_pole_near_one_of_two_contact_lines_moves_for_that_line_alone():
    # two drops on one sphere, on its top and its side; the sphere's pole
    # tilted to 3 degrees outside the top drop's line. move_poles gives that
    # line a copy of the sphere's constraint with its own pole, the side
    # drop's line staying (one pole can't serve lines all round a bead):
    # relaxed at the same volumes, the drops are as with the pole far from
    # both from the start
    from pysurfaceevolver import _surgery, shapes
    from pysurfaceevolver._engine import surface
    cap = shapes.cap(0.5, 0.25, nr=6, nt=24)
    v = np.asarray(cap.vertices, float) + [0, 0, 1.0]
    rim = np.asarray(cap.rims["rim"])
    v[rim] /= np.linalg.norm(v[rim], axis=1)[:, None]
    f = np.asarray(cap.faces)
    turn = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]])        # the top to the side

    def drops(axis):
        ev = pyse.Evolver()
        ev.load_arrays(np.r_[v, v @ turn.T], np.r_[f, f + len(v)],
                       constraints={"ball": C.sphere((0, 0, 0), 1.0, 60.0, axis=axis)},
                       vertex_constraints={"ball": np.r_[rim, rim + len(v)]},
                       bodies=[pyse.Body(faces=range(len(f))),
                               pyse.Body(faces=range(len(f), 2*len(f)))])
        return ev
    ev = drops((0, 0, -1))
    for i in (1, 2):
        ev.body(i).target = ev.body(i).volume
    targets = [ev.body(i).target for i in (1, 2)]
    assert _surgery.move_poles(ev) == [] and ev.relax().converged
    far = ev.energy
    t = np.radians(27)
    ev = drops((-np.sin(t), 0, np.cos(t)))
    assert [round(p[6], 2) for p in _surgery._pole_lines(ev)] == [0.06]
    assert _surgery.move_poles(ev) == ["ball_pole1"]
    assert "ball" in surface.walls and ev.on_constraint("ball").sum() == len(rim)
    assert ev.on_constraint("ball_pole1").sum() == len(rim)
    # (the top drop's volume as its integrals about the far pole give it)
    assert ev.body(1).volume == pytest.approx(targets[0], rel=1e-3)
    assert ev.events() == []
    for i, target in zip((1, 2), targets):
        ev.body(i).target = target
    assert ev.relax().converged
    assert ev.energy == pytest.approx(far, rel=1e-4)


def test_sphere_axis_integrands_in_c_match_evolvers_expressions():
    # (bindings/pyse_native.c) the bridge's energy, volume and a Newton
    # step the same to round-off with the integrands in C and through
    # Evolver's expressions
    from pysurfaceevolver import _core, shapes
    c, a = 1.3, 0.35
    zb = c - np.sqrt(1 - a*a)
    tube = shapes.tube(a, -zb, zb, nz=4, nt=16)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   constraints={1: C.sphere((0, 0, -c), 1.0, 40.0, axis=(0.3, 0, -0.95)),
                                2: C.sphere((0, 0, c), 1.0, 40.0, axis=(0, 0.2, 1))},
                   vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=0.09)])
    ev.refine(1)
    ev.command("convert_to_quantities")
    ev.command("g 5")
    start = ev._save()
    out = []
    try:
        for on in (True, False):
            _core.native_enable(on)
            ev.restore(start)
            ev.command("recalc")
            e, v = ev.energy, ev.body(1).volume
            ev.command("hessian")
            out.append((e, v, ev.vertices.copy()))
    finally:
        _core.native_enable(True)
    assert out[0][0] == pytest.approx(out[1][0], rel=1e-12)
    assert out[0][1] == pytest.approx(out[1][1], rel=1e-12)
    assert np.abs(out[0][2] - out[1][2]).max() < 1e-12
    # (and the C path is the one taken: registered with another energy
    # coefficient, the energy changes)
    from pysurfaceevolver._engine import surface
    centre, R, e1, e2, e3, k = surface.walls[1].native
    ev.restore(start)
    assert _core.native_sphere(1, list(centre), R, list(e1), list(e2), list(e3), 0.0)
    ev.command("recalc")
    assert abs(ev.energy - out[0][0]) > 1e-3
    ev._register_native()
    ev.command("recalc")
    assert ev.energy == pytest.approx(out[0][0], rel=1e-12)
