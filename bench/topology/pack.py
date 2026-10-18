"""Bead packs for the topology corpus: drainage of unit beads in a box.

Beads (unit spheres) settled into a box [0, L] x [0, L] x [0, ...] with hard
walls; the liquid fills the box up to a flat meniscus above them. The side
walls are planes at 90 degrees (mirrors for the liquid), the floor z = 0
needs no constraint (Evolver's volume, z dx dy, has nothing there). Each bead
is a sphere at the contact angle, its axis (the integrals' singular pole) at
its nearest contact, where a ring of liquid stays.
"""
import numpy as np
from scipy.optimize import minimize


def settle(n, Lx, Ly, gap=0.03, seed=0, g=1e-3):
    """n unit spheres settled under a weak gravity into the box, at least
    2 + gap apart and 1 + gap/2 from the walls and floor."""
    rng = np.random.default_rng(seed)
    D, W = 2 + gap, 1 + gap/2
    iu = np.triu_indices(n, 1)

    def energy(flat, g):
        x = flat.reshape(n, 3)
        dx = x[:, None, :] - x[None, :, :]
        r = np.linalg.norm(dx, axis=2)
        np.fill_diagonal(r, np.inf)
        o = np.maximum(0.0, D - r)
        e = 0.5*(o*o).sum()/2
        grad = -(o/np.where(np.isfinite(r), r, 1))[:, :, None]*dx
        grad = grad.sum(axis=1)
        for k, (lo, hi) in enumerate([(W, Lx - W), (W, Ly - W), (W, np.inf)]):
            below = np.maximum(0.0, lo - x[:, k])
            above = np.maximum(0.0, x[:, k] - hi)
            e += 0.5*(below**2 + above**2).sum()
            grad[:, k] += -below + above
        e += g*x[:, 2].sum()
        grad[:, 2] += g
        return e, grad.ravel()

    x = np.c_[rng.uniform(W, Lx - W, n), rng.uniform(W, Ly - W, n),
              W + np.arange(n)*1.2 + rng.uniform(0, 0.5, n)]
    for gk in (g, g/10, g/100, 0.0):
        res = minimize(energy, x.ravel(), args=(gk,), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=50000, gtol=1e-14, ftol=1e-18))
        x = res.x.reshape(n, 3)
    r = np.linalg.norm(x[:, None] - x[None], axis=2)[iu]
    walls = np.r_[x[:, 0] - 1, Lx - 1 - x[:, 0], x[:, 1] - 1, Ly - 1 - x[:, 1], x[:, 2] - 1]
    return x, float(r.min() - 2), float(walls.min())


def axes(x, Lx, Ly):
    """Each bead's axis: toward its nearest contact (another bead, a side
    wall or the floor)."""
    out = []
    for i, c in enumerate(x):
        best, d = np.inf, None
        for j, o in enumerate(x):
            if j != i:
                gap = np.linalg.norm(o - c) - 2
                if gap < best:
                    best, d = gap, o - c
        for k, (lo, hi) in enumerate([(0, Lx), (0, Ly), (0, None)]):
            e = np.eye(3)[k]
            if c[k] - 1 - lo < best:
                best, d = c[k] - 1 - lo, -e
            if hi is not None and hi - c[k] - 1 < best:
                best, d = hi - c[k] - 1, e
        out.append(d/np.linalg.norm(d))
    return np.array(out)


def square(Lx, Ly, z, h):
    """A flat triangulated rectangle [0, Lx] x [0, Ly] at height z, nearly
    equilateral triangles of edge about h (a grid of right triangles has
    vertices whose edges all face right angles: no stiffness, and Newton
    crawled), facing up; and the rows on each side."""
    from scipy.spatial import Delaunay
    dy = h*np.sqrt(3)/2
    ny = max(2, int(round(Ly/dy)))
    ys = np.linspace(0, Ly, ny + 1)
    pts = []
    for j, y in enumerate(ys):
        if j in (0, ny):
            nx = max(2, int(round(Lx/h)))
            xs = np.linspace(0, Lx, nx + 1)
        else:
            off = h/2 if j % 2 else 0.0
            xs = np.arange(off, Lx - h/3, h)
            xs = xs[xs > h/3]
            xs = np.r_[0.0, xs, Lx]
        pts += [(x, y) for x in xs]
    p = np.array(pts)
    tri = Delaunay(p).simplices
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    flip = ((b - a)[:, 0]*(c - a)[:, 1] - (b - a)[:, 1]*(c - a)[:, 0]) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    v = np.c_[p, np.full(len(p), z)]
    tol = 1e-9
    sides = {"wallxlo": np.flatnonzero(p[:, 0] < tol), "wallxhi": np.flatnonzero(p[:, 0] > Lx - tol),
             "wallylo": np.flatnonzero(p[:, 1] < tol), "wallyhi": np.flatnonzero(p[:, 1] > Ly - tol)}
    return v, tri, sides


def load(x, Lx, Ly, theta=20.0, h=0.2, above=0.005, floor=False):
    """The liquid filling the box to a flat meniscus `above` over the top
    bead: an Evolver with the beads as constraints "bead<i>" and the side
    walls "wallxlo", "wallxhi", "wallylo", "wallyhi" (planes at 90 degrees);
    with floor, the floor z = 0 a wall "bottom" too (at 90 degrees: the gas
    reaching it is breakthrough; "floor" is an Evolver word), else only the
    volume's lower bound."""
    import pysurfaceevolver as pyse
    from pysurfaceevolver import constraints as C
    H = float(x[:, 2].max() + 1 + above)
    v, f, sides = square(Lx, Ly, H, h)
    ax = axes(x, Lx, Ly)
    cons = {"wallxlo": C.plane((1, 0, 0), 0.0, contact_angle=90.0),
            "wallxhi": C.plane((-1, 0, 0), -Lx, contact_angle=90.0),
            "wallylo": C.plane((0, 1, 0), 0.0, contact_angle=90.0),
            "wallyhi": C.plane((0, -1, 0), -Ly, contact_angle=90.0)}
    if floor:
        cons["bottom"] = C.plane((0, 0, 1), 0.0, contact_angle=90.0)
    for i, c in enumerate(x):
        cons[f"bead{i + 1}"] = C.sphere(tuple(c), 1.0, contact_angle=theta, axis=tuple(ax[i]),
                                        wet_poles=("north",))
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=cons, vertex_constraints={k: r for k, r in sides.items()},
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    return ev, H
