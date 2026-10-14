"""Starting surfaces and references for docs/packing.md: liquid between spheres
of radius 1 (surface tension 1, so a pressure is minus a curvature).

* ``throat_layer(theta, gap)``: liquid below a planar hexagonal layer of spheres,
  in the prism over three of them (its sides are mirror planes), from a flat
  meniscus above the throat between them.
* ``trimer(gap, theta, volume)``: liquid between three beads, two of them
  touching the third and ``gap`` apart, from a flat slab around the throat;
  ``throat_thickness`` measures the liquid in the throat.
* ``filled_pore(gap, theta)``: liquid filling the pore between four spheres
  ``gap`` apart, from the tetrahedron of their centres with its faces pushed out.
* ``pore_wedge(gap, theta, base)``: the same pore with a three-fold axis along z,
  cut down to a sixth between two mirror planes; ``base`` sets the gaps between
  the three lower spheres, and so the bottom throat's width against the others'.
* ``msp_throat(theta, gap)``: the Mayer-Stowe-Princen estimate of the curvature
  at which gas passes the throat between three spheres.

The free surfaces are built from flat patches between the spheres, meeting
vertex to vertex; each sphere's contact line is where the patches meet it.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import Delaunay

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C


# ---- flat patches -------------------------------------------------------------

def _segment(a, b, h):
    """Points from a to b (both included), about h apart."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a)/h)))
    return a + np.linspace(0, 1, n + 1)[:, None]*(b - a)


def _arc(centre, radius, normal, a, b, h):
    """Points on the circle (centre, radius) in the plane with this normal,
    from a to b (both on it), the shorter way."""
    c, n = np.asarray(centre, float), np.asarray(normal, float)
    n = n/np.linalg.norm(n)
    u = (np.asarray(a, float) - c)/radius
    v = np.cross(n, u)
    w = np.asarray(b, float) - c
    phi = np.arctan2(w @ v, w @ u)
    m = max(2, int(np.ceil(abs(phi)*radius/h)))
    t = np.linspace(0, phi, m + 1)
    return c + radius*(np.cos(t)[:, None]*u + np.sin(t)[:, None]*v)


def _closed(*curves):
    """A loop from curves laid end to end (each one's last point dropped)."""
    return np.vstack([c[:-1] for c in curves])


def _inside(polygon, points):
    """Whether each point (m, 2) lies inside the polygon (n, 2): crossings of
    a ray along +x with its edges."""
    x, y = points[:, :1], points[:, 1:]
    a, b = polygon, np.roll(polygon, -1, axis=0)
    spans = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = a[:, 0] + (y - a[:, 1])*(b[:, 0] - a[:, 0])/(b[:, 1] - a[:, 1])
    return np.count_nonzero(spans & (x < cross), axis=1) % 2 == 1


def _patch(loop, normal, h):
    """Triangulate the flat region inside a closed loop of points: the loop's
    points as given, inside them a triangular lattice of spacing h. Returns
    points and triangles oriented along ``normal``."""
    n = np.asarray(normal, float)
    n = n/np.linalg.norm(n)
    o = loop.mean(axis=0)
    e1 = loop[0] - o
    e1 = e1 - (e1 @ n)*n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    xy = np.c_[(loop - o) @ e1, (loop - o) @ e2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    xs = np.arange(lo[0], hi[0] + h, h)
    ys = np.arange(lo[1], hi[1] + h, h*np.sqrt(3)/2)
    grid = np.array([(x + (j % 2)*h/2, y) for j, y in enumerate(ys) for x in xs])
    near = np.min(np.linalg.norm(grid[:, None, :] - xy[None, :, :], axis=2), axis=1)
    grid = grid[_inside(xy, grid) & (near > 0.6*h)]
    pts = np.vstack([xy, grid])
    tri = Delaunay(pts).simplices
    tri = tri[_inside(xy, pts[tri].mean(axis=1))]
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0])*(c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1])*(c[:, 0] - a[:, 0])
    keep = np.abs(cross) > 1e-9*h*h            # (three points along a straight side)
    tri, cross = tri[keep], cross[keep]
    tri[cross < 0] = tri[cross < 0][:, [0, 2, 1]]
    return np.vstack([loop, o + grid[:, :1]*e1 + grid[:, 1:]*e2]), tri


def _merge(pieces, tol=1e-9):
    """One surface from several (points, triangles); shared points (the same
    loop points in both patches) merged."""
    pts = np.vstack([p for p, _ in pieces])
    offsets = np.cumsum([0] + [len(p) for p, _ in pieces[:-1]])
    tris = np.vstack([t + o for (_, t), o in zip(pieces, offsets)])
    _, first, inverse = np.unique(np.round(pts/tol).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    return pts[first], inverse.ravel()[tris]


def _on(points, centre):
    """Rows of the points on the unit sphere about centre."""
    return np.flatnonzero(np.abs(np.linalg.norm(points - np.asarray(centre), axis=1) - 1) < 1e-7)


def _line_circle(p, d, c, r):
    """The two points p + t d (d a unit vector) on the circle |x - c| = r."""
    f = p - c
    b = f @ d
    t = -b + np.array([-1, 1])*np.sqrt(b*b - (f @ f - r*r))
    return p + t[:, None]*d


# ---- a throat in a layer of spheres ---------------------------------------------

def throat_layer(theta, gap=0.0, z0=0.5, h=0.06, levels=1):
    """Liquid below a planar hexagonal layer of spheres (centres 2 + gap apart in
    the plane z = 0, contact angle theta), in the prism over three of them: its
    sides are mirror planes, and each sphere is a 60-degree wedge about its own
    vertical axis, wetted from the contact line down. Starts from a flat meniscus
    at z = z0 and relaxes it; the body's volume is the liquid above the throat
    plane z = 0 (negative once the meniscus has passed below it)."""
    L = 2 + gap
    centres = {"a": np.array([0.0, 0.0, 0.0]), "b": np.array([L, 0.0, 0.0]),
               "c": np.array([L/2, L*np.sqrt(3)/2, 0.0])}
    r0 = np.sqrt(1 - z0*z0)                   # the spheres' circles at z0
    up = np.array([0.0, 0.0, 1.0])
    corners = {}                               # where the sides leave and reach the circles
    order = (("a", "b"), ("b", "c"), ("c", "a"))
    for p, q in order:
        d = (centres[q] - centres[p])/L
        corners[p + q] = (centres[p] + r0*d + z0*up, centres[q] - r0*d + z0*up)
    curves = []
    for k, (p, q) in enumerate(order):
        nxt = order[(k + 1) % 3]
        side = _segment(*corners[p + q], h)
        curves += [side, _arc(centres[q] + z0*up, r0, up, side[-1], corners[nxt[0] + nxt[1]][0], h)]
    v, f = _merge([_patch(_closed(*curves), up, h)])
    walls = {f"sphere_{k}": C.sphere(c, 1.0, contact_angle=theta, wet_poles=("south",),
                                     span=np.pi/3) for k, c in centres.items()}
    rows = {f"sphere_{k}": _on(v, c) for k, c in centres.items()}
    for p, q in order:
        d = (centres[q] - centres[p])/L
        normal = np.array([-d[1], d[0], 0.0])
        walls[f"mirror_{p}{q}"] = C.mirror((normal, centres[p]))
        rows[f"mirror_{p}{q}"] = np.flatnonzero(np.abs((v - centres[p]) @ normal) < 1e-9)
    ev = pyse.Evolver()
    # volconst 0: the volume counted from the throat plane up
    ev.load_arrays(v, f, constraints=walls, vertex_constraints=rows,
                   bodies=[pyse.Body(volume=1.0, volconst=0.0)])
    ev.body(1).target = ev.body(1).volume
    ev.relax(levels=levels)
    return ev


def msp_throat(theta, gap=0.0):
    """Mayer-Stowe-Princen: the curvature (times the sphere radius) at which gas
    passes the cross-section between three circles of radius 1 (centres 2 + gap
    apart), the liquid wetting them at theta and left in the three corners."""
    t = np.radians(theta)
    L = 2 + gap
    m = L/2
    pore = np.sqrt(3)/4*L*L - np.pi/2

    def corner(r):
        # the meniscus arc (radius r) centred on the corner's bisector at height yc
        yc = np.sqrt(max(r*r + 2*r*np.cos(t) + 1 - m*m, 0.0))
        d = np.hypot(m, yc)
        phi = np.arctan2(yc, m) - np.arccos(np.clip((1 + d*d - r*r)/(2*d), -1, 1))
        a0 = np.arctan2(np.sin(phi) - yc, np.cos(phi) - m)
        a1 = -np.pi - a0
        s = np.linspace(0, phi, 200)
        u = np.linspace(a0, a1, 400)
        poly = np.vstack([np.c_[np.cos(s), np.sin(s)], np.c_[m + r*np.cos(u), yc + r*np.sin(u)],
                          np.c_[L - np.cos(s[::-1]), np.sin(s[::-1])]])
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5*abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))
        return area, 2*phi, r*abs(a0 - a1)

    def balance(r):
        area, wet, meniscus = corner(r)
        return (pore - 3*area)/r - (3*meniscus + (np.pi - 3*wet)*np.cos(t))
    rs = np.geomspace(1e-3, 0.5, 400)
    f = np.array([balance(r) for r in rs])
    k = np.flatnonzero(np.diff(np.sign(f)) != 0)[0]
    return 1/brentq(balance, rs[k], rs[k + 1])


# ---- three beads ------------------------------------------------------------------

def three_beads(gap):
    """Bead centres: beads 1 and 2 gap apart, both touching bead 3, in z = 0."""
    a = 1 + gap/2
    return np.array([[-a, 0, 0], [a, 0, 0], [0, np.sqrt(4 - a*a), 0]])


def trimer(gap, theta, volume, z0=0.35, delta=0.3, h=0.05, levels=1):
    """Liquid between three beads (see three_beads) wetting all three: from a
    slab |z| < z0 over the triangle of their centres, its sides pushed out by
    delta (clear of the beads' contacts), relaxed at ``volume``. Returns the
    Evolver and the centres."""
    Cs = three_beads(gap)
    g = Cs.mean(axis=0)
    r0, rw = np.sqrt(1 - z0*z0), np.sqrt(1 - delta*delta)
    up = np.array([0.0, 0.0, 1.0])
    sides = [(0, 1), (1, 2), (2, 0)]
    walls = []
    for i, j in sides:
        d = (Cs[j] - Cs[i])/np.linalg.norm(Cs[j] - Cs[i])
        n = np.array([d[1], -d[0], 0.0])
        walls.append((i, j, d, n if n @ (g - Cs[i]) < 0 else -n))
    top, bottom = {}, {}                        # where each wall meets the beads at +-z0
    for w, (i, j, d, n) in enumerate(walls):
        for z, store in ((z0, top), (-z0, bottom)):
            p = Cs[i] + delta*n + z*up
            store[w] = (_line_circle(p, d, Cs[i] + z*up, r0)[1],
                        _line_circle(p, d, Cs[j] + z*up, r0)[0])
    top_side = [_segment(*top[w], h) for w in range(3)]
    bottom_side = [_segment(*bottom[w], h) for w in range(3)]
    top_loop, bottom_loop = [], []
    for w in range(3):
        k, nxt = sides[w][1], (w + 1) % 3
        top_loop += [top_side[w], _arc(Cs[k] + z0*up, r0, up, top[w][1], top[nxt][0], h)]
        bottom_loop += [bottom_side[w], _arc(Cs[k] - z0*up, r0, up, bottom[w][1],
                                             bottom[nxt][0], h)]
    pieces = [_patch(_closed(*top_loop), up, h), _patch(_closed(*bottom_loop), -up, h)]
    for w, (i, j, d, n) in enumerate(walls):
        down_j = _arc(Cs[j] + delta*n, rw, n, top[w][1], bottom[w][1], h)
        up_i = _arc(Cs[i] + delta*n, rw, n, bottom[w][0], top[w][0], h)
        pieces.append(_patch(_closed(top_side[w], down_j, bottom_side[w][::-1], up_i), n, h))
    v, f = _merge(pieces)
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints={f"bead{k + 1}": C.sphere(c, 1.0, contact_angle=theta)
                                      for k, c in enumerate(Cs)},
                   vertex_constraints={f"bead{k + 1}": _on(v, c) for k, c in enumerate(Cs)},
                   bodies=[pyse.Body(volume=volume)])
    ev.relax(levels=levels)
    return ev, Cs


def throat_thickness(ev, centres):
    """The liquid's thickness across z = 0 in the throat: twice the smallest
    |z| of the free surface's vertices over the triangle of the centres. (Only
    while the outer meniscus in the gap lies outside that triangle, as for a
    trimer: d of meniscus_distance above bead 3's distance to the line through
    the other two centres.)"""
    x = ev.vertices
    free = np.all(np.abs(np.linalg.norm(x[:, None, :] - centres[None], axis=2) - 1) > 1e-6,
                  axis=1)
    inside = _inside(centres[:, :2], x[:, :2])
    return 2*np.abs(x[free & inside, 2]).min()



def meniscus_distance(ev, centres):
    """Semprebon et al.'s d: along the symmetry axis (x = 0, z = 0) from bead
    3's surface toward the gap, the distance to the first liquid surface (the
    outer meniscus in the gap). Large for a trimer, which fills the throat;
    small for a dimer, whose liquid has left it."""
    m = ev.mesh()
    x, f = m.vertices, m.facets
    p0 = x[f[:, 0]]
    a, b = x[f[:, 1]] - p0, x[f[:, 2]] - p0
    # where each facet's plane meets the axis: p0 + u a + v b with x = z = 0
    det = a[:, 0]*b[:, 2] - a[:, 2]*b[:, 0]
    ok = np.abs(det) > 1e-14
    det = np.where(ok, det, 1.0)
    u = (b[:, 0]*p0[:, 2] - b[:, 2]*p0[:, 0])/det
    v = (a[:, 2]*p0[:, 0] - a[:, 0]*p0[:, 2])/det
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
    y = (p0[:, 1] + u*a[:, 1] + v*b[:, 1])[hit]
    surface = centres[2, 1] - 1                     # bead 3's surface on the axis
    below = y[y < surface - 1e-9]
    return float(surface - below.max()) if len(below) else float("nan")

# ---- the pore between four spheres ---------------------------------------------------

def pore_centres(gap):
    """Four spheres at alternate corners of a cube, 2 + gap apart. (Each sphere's
    contact line then keeps about 18 degrees from its poles about z, where the
    wetting integrals are singular.)"""
    a = (2 + gap)/np.sqrt(2)/2
    return a*np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)


def _pore_surface(Cs, delta, h):
    """The faces of the tetrahedron of the centres Cs pushed out by delta, minus
    the spheres: (vertices, triangles)."""
    g = Cs.mean(axis=0)
    rw = np.sqrt(1 - delta*delta)
    faces = []
    for skip in range(4):
        tri = [k for k in range(4) if k != skip]
        n = np.cross(Cs[tri[1]] - Cs[tri[0]], Cs[tri[2]] - Cs[tri[0]])
        n /= np.linalg.norm(n)
        faces.append((tri, n if n @ (Cs[tri[0]] - g) > 0 else -n))
    edge = {}                                   # along each edge, from sphere i to sphere j
    for i in range(4):
        for j in range(i + 1, 4):
            (n1, _), (n2, _) = [(n, t) for t, n in faces if i in t and j in t]
            u = np.cross(n1, n2)
            u /= np.linalg.norm(u)
            u = u if u @ (Cs[j] - Cs[i]) > 0 else -u
            p0 = np.linalg.solve(np.array([n1, n2, u]),
                                 [n1 @ Cs[i] + delta, n2 @ Cs[i] + delta, u @ Cs[i]])
            pts = _segment(_line_circle(p0, u, Cs[i], 1.0)[1], _line_circle(p0, u, Cs[j], 1.0)[0], h)
            edge[i, j], edge[j, i] = pts, pts[::-1]
    pieces = []
    for tri, n in faces:
        a, b, c = tri
        if np.cross(Cs[b] - Cs[a], Cs[c] - Cs[a]) @ n < 0:
            b, c = c, b
        loop = []
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            loop += [edge[p, q], _arc(Cs[q] + delta*n, rw, n, edge[p, q][-1], edge[q, r][0], h)]
        pieces.append(_patch(_closed(*loop), n, h))
    return _merge(pieces)


def filled_pore(gap, theta, delta=0.3, h=0.05, levels=1):
    """Liquid filling the pore between four spheres (see pore_centres): the
    tetrahedron of their centres with its faces pushed out by delta, minus the
    spheres, relaxed at the volume it holds. Returns the Evolver and the centres."""
    Cs = pore_centres(gap)
    v, f = _pore_surface(Cs, delta, h)
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints={f"sphere{k + 1}": C.sphere(c, 1.0, contact_angle=theta)
                                      for k, c in enumerate(Cs)},
                   vertex_constraints={f"sphere{k + 1}": _on(v, c) for k, c in enumerate(Cs)},
                   bodies=[pyse.Body(volume=1.0)])
    ev.body(1).target = ev.body(1).volume
    ev.relax(levels=levels)
    return ev, Cs


# ---- the pore in a wedge: one throat against the other three -----------------------

# the wedge's mirror planes, through the z axis: y = 0 and the azimuth 60 degrees
WEDGE = (np.array([0.0, 1.0, 0.0]), np.array([np.sin(np.pi/3), -np.cos(np.pi/3), 0.0]))


def axis_centres(gap, base=None):
    """The pore's spheres with a three-fold axis along z: A on it above, B, C, D
    below at azimuths 0, 120 and 240 degrees, ``gap`` from A and ``base``
    (default ``gap``) from each other. Their three-fold symmetry about z (with
    mirror planes through it) holds the bottom throat, BCD, against the other three."""
    base = gap if base is None else base
    R = (2 + base)/np.sqrt(3)
    H = np.sqrt((2 + gap)**2 - R*R)
    return np.array([[0, 0, 3*H/4]] + [[R*np.cos(t), R*np.sin(t), -H/4]
                                       for t in np.radians([0, 120, 240])])


def _cut(v, f, n, Cs, snap, kept=()):
    """The part of a surface with n.x >= 0. Vertices within ``snap`` of the plane
    are moved onto it (along the planes ``kept``, for those on them; not those on
    a sphere whose centre is off the plane, which can't touch it), and back onto
    their sphere; triangles across it are split."""
    v = v.copy()
    d = v @ n
    on_sphere = [np.abs(np.linalg.norm(v - c, axis=1) - 1) < 1e-7 for c in Cs]
    near = np.abs(d) < snap
    for c, on in zip(Cs, on_sphere):
        if abs(c @ n) > 1e-9:
            near &= ~on
    move = np.outer(d, n)
    for m in kept:
        on = np.abs(v @ m) < 1e-12
        t = n - (n @ m)*m
        move[on] = np.outer(d[on]/(n @ t), t)
    v[near] -= move[near]
    for c, on in zip(Cs, on_sphere):
        sel = near & on
        v[sel] = c + (v[sel] - c)/np.linalg.norm(v[sel] - c, axis=1)[:, None]
    d = np.where(near, 0.0, v @ n)
    pts, new, tris = list(v), {}, []

    def cross(i, j):
        key = (min(i, j), max(i, j))
        if key not in new:
            x = v[i] + d[i]/(d[i] - d[j])*(v[j] - v[i])
            for c, on in zip(Cs, on_sphere):     # a contact-line edge: back onto its sphere
                if on[i] and on[j]:
                    x = c + (x - c)/np.linalg.norm(x - c)
            new[key] = len(pts)
            pts.append(x)
        return new[key]
    for t in f:
        s = d[t]
        if (s >= 0).all():
            tris.append(t)
        elif not (s <= 0).all():
            poly = []
            for k in range(3):
                i, j = t[k], t[(k + 1) % 3]
                if d[i] >= 0:
                    poly.append(i)
                if d[i]*d[j] < 0:
                    poly.append(cross(i, j))
            tris += [[poly[0], poly[k], poly[k + 1]] for k in range(1, len(poly) - 1)]
    pts, tris = np.array(pts), np.array(tris)
    used = np.unique(tris)
    index = np.full(len(pts), -1)
    index[used] = np.arange(len(used))
    return pts[used], index[tris]


def pore_wedge(gap, theta, base=None, delta=0.3, h=0.05, levels=1):
    """The full pore of axis_centres(gap, base), relaxed, cut down to the wedge
    between the mirror planes WEDGE: a sixth of it, with a 60-degree piece of
    sphere A (its south pole wet) and half of sphere B; C and D lie outside.
    Returns the Evolver and the centres."""
    Cs = axis_centres(gap, base)
    v, f = _pore_surface(Cs, delta, h)
    names = ("sphere_a", "sphere_b", "sphere_c", "sphere_d")
    full = pyse.Evolver()
    full.load_arrays(v, f, constraints={nm: C.sphere(c, 1.0, contact_angle=theta,
                                                     wet_poles=("south",) if k == 0 else ())
                                        for k, (nm, c) in enumerate(zip(names, Cs))},
                     vertex_constraints={nm: _on(v, c) for nm, c in zip(names, Cs)},
                     bodies=[pyse.Body(volume=1.0)])
    full.body(1).target = full.body(1).volume
    full.relax(levels=levels)
    volume, m = full.body(1).volume/6, full.mesh()
    snap = 0.3*float(np.median(m.edge_lengths()))
    v, f = _cut(m.vertices, m.facets, WEDGE[0], Cs, snap)
    v, f = _cut(v, f, WEDGE[1], Cs, snap, kept=(WEDGE[0],))
    f = f[np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) > 1e-12]
    walls = {"sphere_a": C.sphere(Cs[0], 1.0, contact_angle=theta, wet_poles=("south",),
                                  span=np.pi/3),
             "sphere_b": C.sphere(Cs[1], 1.0, contact_angle=theta),
             "mirror_0": C.mirror((WEDGE[0], np.zeros(3))),
             "mirror_60": C.mirror((WEDGE[1], np.zeros(3)))}
    rows = {"sphere_a": _on(v, Cs[0]), "sphere_b": _on(v, Cs[1]),
            "mirror_0": np.flatnonzero(np.abs(v @ WEDGE[0]) < 1e-9),
            "mirror_60": np.flatnonzero(np.abs(v @ WEDGE[1]) < 1e-9)}
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=walls, vertex_constraints=rows,
                   bodies=[pyse.Body(volume=volume)])
    ev.relax()
    return ev, Cs


def bottom_meniscus(ev):
    """Where the meniscus in the bottom throat crosses the z axis (the wedge's
    corner): its height, the pore's centre at about 0."""
    x = ev.vertices
    on = (np.abs(x @ WEDGE[0]) < 1e-9) & (np.abs(x @ WEDGE[1]) < 1e-9)
    return float(x[on, 2].min())
