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
* ``msp_throat(theta, gap)``: the Mayer-Stowe-Princen estimate of the curvature
  at which gas passes the throat between three spheres.

The free surfaces are built from flat patches between the spheres, meeting
vertex to vertex; each sphere's contact line is where the patches meet it.
"""

import numpy as np
from matplotlib.path import Path
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
    path = Path(np.vstack([xy, xy[:1]]))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    xs = np.arange(lo[0], hi[0] + h, h)
    ys = np.arange(lo[1], hi[1] + h, h*np.sqrt(3)/2)
    grid = np.array([(x + (j % 2)*h/2, y) for j, y in enumerate(ys) for x in xs])
    near = np.min(np.linalg.norm(grid[:, None, :] - xy[None, :, :], axis=2), axis=1)
    grid = grid[path.contains_points(grid) & (near > 0.6*h)]
    pts = np.vstack([xy, grid])
    tri = Delaunay(pts).simplices
    tri = tri[path.contains_points(pts[tri].mean(axis=1))]
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
    |z| of the free surface's vertices over the triangle of the centres."""
    x = ev.vertices
    free = np.all(np.abs(np.linalg.norm(x[:, None, :] - centres[None], axis=2) - 1) > 1e-6,
                  axis=1)
    inside = Path(centres[:, :2]).contains_points(x[:, :2])
    return 2*np.abs(x[free & inside, 2]).min()


# ---- the pore between four spheres ---------------------------------------------------

def pore_centres(gap):
    """Four spheres at alternate corners of a cube, 2 + gap apart. (Each sphere's
    contact line then keeps about 18 degrees from its poles about z, where the
    wetting integrals are singular.)"""
    a = (2 + gap)/np.sqrt(2)/2
    return a*np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)


def filled_pore(gap, theta, delta=0.3, h=0.05, levels=1):
    """Liquid filling the pore between four spheres (see pore_centres): the
    tetrahedron of their centres with its faces pushed out by delta, minus the
    spheres, relaxed at the volume it holds. Returns the Evolver and the centres."""
    Cs = pore_centres(gap)
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
    v, f = _merge(pieces)
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints={f"sphere{k + 1}": C.sphere(c, 1.0, contact_angle=theta)
                                      for k, c in enumerate(Cs)},
                   vertex_constraints={f"sphere{k + 1}": _on(v, c) for k, c in enumerate(Cs)},
                   bodies=[pyse.Body(volume=1.0)])
    ev.body(1).target = ev.body(1).volume
    ev.relax(levels=levels)
    return ev, Cs
