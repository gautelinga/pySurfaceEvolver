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


def load(x, Lx, Ly, theta=20.0, h=0.2, above=0.005, floor=False, gauge="pole"):
    """The liquid filling the box to a flat meniscus `above` over the top
    bead: an Evolver with the beads as constraints "bead<i>" and the side
    walls "wallxlo", "wallxhi", "wallylo", "wallyhi" (planes at 90 degrees);
    with floor, the floor z = 0 a wall "bottom" too (at 90 degrees: the gas
    reaching it is breakthrough; "floor" is an Evolver word), else only the
    volume's lower bound. gauge: the beads' (constraints.sphere): "pole"
    (an axis each) or "flux" (no poles; needs floor)."""
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
        if gauge == "flux":
            # (no poles: constraints.sphere's flux gauge)
            cons[f"bead{i + 1}"] = C.sphere(tuple(c), 1.0, contact_angle=theta, gauge="flux")
        else:
            cons[f"bead{i + 1}"] = C.sphere(tuple(c), 1.0, contact_angle=theta, axis=tuple(ax[i]),
                                            wet_poles=("north",))
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=cons, vertex_constraints={k: r for k, r in sides.items()},
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    return ev, H


def settle_periodic(n, L, gap=0.03, seed=0, g=1e-3):
    """n unit spheres settled under a weak gravity onto the floor of a cell
    [0, L)^2 periodic in x and y, at least 2 + gap apart (at the nearest
    copy) and 1 + gap/2 above the floor; centres wrapped into the cell."""
    rng = np.random.default_rng(seed)
    D, W = 2 + gap, 1 + gap/2

    def energy(flat, g):
        x = flat.reshape(n, 3)
        dx = x[:, None, :] - x[None, :, :]
        dx[..., :2] -= L*np.round(dx[..., :2]/L)
        r = np.linalg.norm(dx, axis=2)
        np.fill_diagonal(r, np.inf)
        o = np.maximum(0.0, D - r)
        e = 0.5*(o*o).sum()/2
        grad = -((o/np.where(np.isfinite(r), r, 1))[:, :, None]*dx).sum(axis=1)
        below = np.maximum(0.0, W - x[:, 2])
        e += 0.5*(below**2).sum() + g*x[:, 2].sum()
        grad[:, 2] += -below + g
        return e, grad.ravel()

    x = np.c_[rng.uniform(0, L, n), rng.uniform(0, L, n),
              W + np.arange(n)*1.2*16/L**2 + rng.uniform(0, 0.5, n)]
    for gk in (g, g/10, g/100, 0.0):
        res = minimize(energy, x.ravel(), args=(gk,), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=50000, gtol=1e-14, ftol=1e-18))
        x = res.x.reshape(n, 3)
    x[:, :2] %= L
    dx = x[:, None] - x[None]
    dx[..., :2] -= L*np.round(dx[..., :2]/L)
    r = np.linalg.norm(dx, axis=2)[np.triu_indices(n, 1)]
    return x, float(r.min() - 2), float((x[:, 2] - 1).min())


def sheet_periodic(L, z, h):
    """A flat sheet over the cell [0, L)^2 at height z, facing up, joined
    across the cell's sides: rows about h*sqrt(3)/2 apart (an even number),
    every other row offset by half a step (nearly equilateral triangles)."""
    nx = max(4, int(round(L/h)))
    ny = max(4, 2*int(round(L/(h*np.sqrt(3)))))
    xs, ys = np.arange(nx)*L/nx, np.arange(ny)*L/ny
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X = X + 0.5*(np.arange(ny)[None, :] % 2)*L/nx
    v = np.c_[X.ravel() % L, Y.ravel(), np.full(nx*ny, z)]
    at = lambda i, j: (i % nx)*ny + (j % ny)
    f = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)
            f += [(a, b, c), (b, d, c)] if j % 2 == 0 else [(a, b, d), (a, d, c)]
    return v, np.array(f)


def load_periodic(x, L, theta=20.0, h=0.2, above=0.005):
    """The liquid filling a cell periodic in x and y up to a flat meniscus
    `above` over the top bead: the beads constraints "bead<i>" (flux gauge),
    the floor z = 0 a wall "bottom" at 90 degrees (the outlet)."""
    import pysurfaceevolver as pyse
    from pysurfaceevolver import constraints as C
    H = float(x[:, 2].max() + 1 + above)
    v, f = sheet_periodic(L, H, h)
    cons = {"bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0)}
    for i, c in enumerate(x):
        cons[f"bead{i + 1}"] = C.sphere(tuple(float(t) for t in c), 1.0, contact_angle=theta,
                                        gauge="flux")
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=cons, bodies=[pyse.Body(volume=None)], periods=(L, L))
    ev.body(1).target = ev.body(1).volume
    return ev, H


def contacts_periodic(x, L, reach=0.1):
    """The pairs (i, j), i < j, of beads in a cell periodic in x and y less
    than 2 + reach apart (at the nearest copy), and the vector from i to
    that copy of j."""
    dx = x[None, :, :] - x[:, None, :]
    dx[..., :2] -= L*np.round(dx[..., :2]/L)
    r = np.linalg.norm(dx, axis=2)
    i, j = np.nonzero(np.triu(r < 2 + reach, 1))
    return list(zip(i.tolist(), j.tolist())), dx[i, j]


def bridges_periodic(x, L, theta=20.0, a=0.25, reach=0.1, nz=2, nt=12):
    """Pendular bridges at the contacts of beads in a cell periodic in x and
    y: a tube of radius a about the line of centres of each pair closer
    than 2 + reach (contacts_periodic; across the cell's sides at the
    nearest copy), its rims on the two beads, all the tubes one body
    (several pieces, one pressure). Beads "bead<i>" (flux gauge), the floor
    z = 0 a wall "bottom" at 90 degrees (the integrals' strings leave
    through it; lift the beads off it). Returns the Evolver and the pairs."""
    import pysurfaceevolver as pyse
    from pysurfaceevolver import constraints as C, shapes
    pairs, sep = contacts_periodic(x, L, reach)
    vs, fs, rims, n = [], [], {}, 0
    for (i, j), d in zip(pairs, sep):
        D = float(np.linalg.norm(d))
        zb = D/2 - np.sqrt(1 - a*a)
        t = shapes.tube(a, -zb, zb, nz=nz, nt=nt)
        R = _rotation(d/D)
        vs.append(np.asarray(t.vertices) @ R.T + x[i] + d/2)
        fs.append(np.asarray(t.faces) + n)
        rims.setdefault(f"bead{i + 1}", []).extend((np.asarray(t.rims["bottom"]) + n).tolist())
        rims.setdefault(f"bead{j + 1}", []).extend((np.asarray(t.rims["top"]) + n).tolist())
        n += len(t.vertices)
    v = np.concatenate(vs)
    v[:, :2] %= L
    cons = {"bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0)}
    for k, c in enumerate(x):
        cons[f"bead{k + 1}"] = C.sphere(tuple(float(s) for s in c), 1.0, contact_angle=theta,
                                        gauge="flux")
    ev = pyse.Evolver()
    ev.load_arrays(v, np.concatenate(fs), constraints=cons,
                   vertex_constraints={k: np.array(r) for k, r in rims.items()},
                   bodies=[pyse.Body(volume=None)], periods=(L, L))
    ev.body(1).target = ev.body(1).volume
    return ev, pairs


def _rotation(u):
    """A rotation taking z to the unit vector u."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, u)
    s, c = np.linalg.norm(v), float(z @ u)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = v/s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s*K + (1 - c)*K @ K


def pore_volume(x, L):
    """The space between the beads (unit spheres) of a cell periodic in x and
    y, between the planes through the lowest and the highest centre."""
    lo, hi = float(x[:, 2].min()), float(x[:, 2].max())
    cut = lambda h: np.pi*(np.clip(h, 0, 2)**2)*(3 - np.clip(h, 0, 2))/3   # cap of height h
    beads = sum(cut(hi - (z - 1)) - cut(lo - (z - 1)) for z in x[:, 2])
    return L*L*(hi - lo) - beads
