"""Starting surfaces and references for docs/disk_bridge.md: a liquid bridge
between two equal coaxial disks of radius 1, its contact lines pinned at the
disk edges, the disks 2*slenderness apart (surface tension 1; gravity along -z,
the Bond number B = density * g * R^2 / tension).

* ``disk_bridge(slenderness, volume, bond)``: the bridge from a cylinder, the
  lower disk at z = 0 and the upper one at z = 2*slenderness.
* ``to_limit(ev, slenderness, ds)``: follows it with ``recipes.arclength`` in
  its volume to where it turns unstable, and says how (``kind``).
* ``limits(slenderness, bond)``: the exact stability limits of the
  axisymmetric bridges, from the Young-Laplace equation and the second
  variation of the energy on the profile (see ``Profile``).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, eigh_tridiagonal, null_space
from scipy.optimize import brentq

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes


# ---- the bridge in pySE --------------------------------------------------------

def _rim(z):
    """The disk's rim at height ``z``, a circular wire with the volume Evolver
    leaves out over the disk (its z dx dy) as a content integral."""
    return C.circle((0, 0, z), 1.0, content=(f"-{z!r}*y/2", f"{z!r}*x/2", "0"))


def disk_bridge(slenderness, volume=1.0, bond=0.0, nt=24):
    """The bridge between disks z = 0 and z = 2*slenderness, from a cylinder
    of radius 1 with ``nt`` facets around and about square ones along, holding
    ``volume`` times the cylinder's (not relaxed)."""
    H = 2*slenderness
    tube = shapes.tube(1.0, 0.0, H, nz=max(2, round(H*nt/(2*np.pi))), nt=nt)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   boundaries={1: _rim(0.0), 2: _rim(H)},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=volume*np.pi*H, density=1.0 if bond else None)],
                   gravity=bond if bond else None)
    return ev


def _axisymmetric(odd):
    """The name of an axisymmetric mode, from its odd share about the mid-plane."""
    return "to one disk" if odd > 0.95 else "pinching" if odd < 0.05 else "axisymmetric"


def kind(ev, mode, slenderness):
    """What a mode does: "sideways" (its radial displacement goes as cos or
    sin of the angle about the axis), else, the same all round, "pinching"
    (even about the mid-plane: a neck narrows or widens), "to one disk" (odd
    about it: liquid moves from one half to the other) or "axisymmetric"
    (neither, as with gravity)."""
    x, d = np.asarray(ev.vertices), np.asarray(mode)
    theta = np.arctan2(x[:, 1], x[:, 0])
    u = d[:, 0]*np.cos(theta) + d[:, 1]*np.sin(theta)        # radial displacement
    H = 2*slenderness
    edges = np.linspace(0, H, 21)
    band = np.clip(np.digitize(x[:, 2], edges) - 1, 0, 19)
    m0 = m1 = 0.0
    mean = np.zeros(20)
    for b in range(20):
        k = band == b
        A = np.column_stack([np.ones(k.sum()), np.cos(theta[k]), np.sin(theta[k])])
        c = np.linalg.lstsq(A, u[k], rcond=None)[0]
        m0, m1 = m0 + c[0]**2, m1 + (c[1]**2 + c[2]**2)/2
        mean[b] = c[0]
    if m1 > m0:
        return "sideways"
    odd = np.sum((mean - mean[::-1])**2)/(2*np.sum(mean**2 + mean[::-1]**2))
    return _axisymmetric(odd)


def to_limit(ev, slenderness, ds, steps=200, tidy=False):
    """Follow the bridge with recipes.arclength in its volume, in steps of ds
    times the cylinder's volume (ds < 0: taking liquid away), to the first
    point with a negative eigenvalue (the volume held), and up to three
    points on. Returns the reduced volume V/(pi R^2 H) at the limit, and
    ``kind`` of the unstable mode there.

    If the volume turns back there, the limit is a fold: the least (or
    greatest) volume of a parabola through the three points around the turn,
    against the distance travelled. Otherwise it is where the lowest
    eigenvalue crosses zero: from a parabola through the three points up to
    the first unstable one, the volume against the eigenvalue.

    Between points the surface isn't tidied by default (in the quadratic model
    that stalls the run near the fold here), and a failed step is halved up to
    six times (three weren't enough at some folds)."""
    cylinder = 2*np.pi*slenderness
    sign = np.sign(ds)
    rows, how, first, turned = [], None, None, False   # rows: distance, volume, eigenvalue, vertices
    for step in recipes.arclength(ev, ev.body(1), ds*cylinder, steps, halvings=6, tidy=tidy):
        x = ev.vertices
        dist = 0.0 if not rows else rows[-1][0] + np.hypot(
            step.value - rows[-1][1], np.sqrt(((x - rows[-1][3])**2).sum(axis=1).mean()))
        lowest = ev.modes(1)
        rows.append((dist, step.value, lowest.values[0], x))
        if first is None and step.result.negative_modes and len(rows) >= 3:
            first, how = len(rows) - 1, kind(ev, lowest.vectors[0], slenderness)
        if first is not None:
            volumes = np.array([r[1] for r in rows])
            turned = volumes[-1] > volumes.min() if sign < 0 else volumes[-1] < volumes.max()
            if turned or len(rows) - 1 - first == 3:
                break
    if first is None:
        raise RuntimeError("no limit found")
    dist, volumes, values = np.array([r[:3] for r in rows]).T
    j = int(np.argmin(volumes) if sign < 0 else np.argmax(volumes))
    if turned and j <= first + 1:                    # a fold where it turned unstable
        a, b, c = np.polyfit(dist[j - 1:j + 2], volumes[j - 1:j + 2], 2)
        limit = c - b*b/(4*a)
    else:
        limit = np.polyval(np.polyfit(values[first - 2:first + 1],
                                      volumes[first - 2:first + 1], 2), 0.0)
    return limit/cylinder, how


# ---- the exact axisymmetric bridges ---------------------------------------------
#
# The profile r(s), z(s) by arclength s from the lower edge (1, 0), phi the
# angle of its tangent from the z axis: r' = sin phi, z' = cos phi, and the
# Young-Laplace equation (mean curvature times 2 = p - B z, the curvatures
# k1 = -phi' along the profile and k2 = cos phi / r around it):
#     phi' = cos(phi)/r - p + B z.
# The bridge ends at the upper edge: r(L) = 1, z(L) = H. With three unknowns
# (phi(0), p, L) and two conditions, the bridges at given H and B form
# curves, followed here by pseudo-arclength continuation. Without gravity
# the bridges from the cylinder are mirror-symmetric about z = H/2, and those
# that aren't branch off them; to follow the symmetric ones past the branch
# points, the conditions are taken at the mid-plane instead: z(L/2) = H/2
# and phi(L/2) = 0 (q holds L/2 then).

def _rhs(s, y, p, B):
    r, z, phi, V = y
    return [np.sin(phi), np.cos(phi), np.cos(phi)/r - p + B*z, np.pi*r*r*np.cos(phi)]


def _pinch(s, y, p, B):
    return y[0] - 1e-3
_pinch.terminal = True


def _shoot(q, B, dense=False):
    phi0, p, L = q
    if not L > 0:
        raise RuntimeError("no profile")
    sol = solve_ivp(_rhs, (0.0, L), [1.0, 0.0, phi0, 0.0], args=(p, B), rtol=1e-12,
                    atol=1e-13, dense_output=dense, method="DOP853", events=_pinch)
    if sol.status == 1:               # a trial profile reaching the axis: give up on it
        raise RuntimeError("the profile pinches off")
    return sol


def _residual(q, H, B):
    y = _shoot(q, B).y[:, -1]
    if B == 0:
        return np.array([y[2], y[1] - H/2]), 2*y[3]
    return np.array([y[0] - 1.0, y[1] - H]), y[3]


def _full(q, B):
    """(phi(0), p, L) of the whole profile."""
    return (q[0], q[1], 2*q[2]) if B == 0 else tuple(q)


class Profile:
    """An axisymmetric bridge: ``volume``, ``pressure``, and the lowest
    eigenvalues of the second variation of the energy (area plus weight),
    with the contact lines pinned (the displacement vanishes there): for
    displacements u(s) cos(m theta) normal to the surface,
        J u = -(r u')'/r + (m^2/r^2 - k1^2 - k2^2 + B n_z) u,
    with the volume held for m = 0 (u r integrates to 0). A negative
    eigenvalue means an unstable bridge."""

    def __init__(self, q, H, B, n=600):
        self.q, self.H, self.B = np.asarray(q, float), H, B
        full = _full(self.q, B)
        sol = _shoot(full, B, dense=True)
        s = np.linspace(0.0, full[2], n + 2)
        r, z, phi, V = sol.sol(s)
        self.s, self.r, self.z, self.phi = s, r, z, phi
        self.volume, self.pressure = V[-1], self.q[1]
        k1 = -(np.cos(phi)/r - self.q[1] + B*z)
        k2 = np.cos(phi)/r
        self._q0 = -k1*k1 - k2*k2 - B*np.sin(phi)          # n_z = -sin(phi)
        # finite differences on the interior points, made symmetric: with
        # w = sqrt(r) u the problem is a symmetric tridiagonal matrix
        h = s[1] - s[0]
        rh = (r[1:] + r[:-1])/2                            # r at the midpoints
        ri = r[1:-1]
        self._d = (rh[:-1] + rh[1:])/h**2/ri + self._q0[1:-1]
        self._e = -rh[1:-1]/h**2/np.sqrt(ri[:-1]*ri[1:])
        self._ri = ri

    def eigenvalues(self, m):
        """The lowest eigenvalue for cos(m theta) (m = 0: volume held)."""
        d = self._d + m*m/self._ri**2
        if m:
            return eigh_tridiagonal(d, self._e, eigvals_only=True,
                                    select="i", select_range=(0, 0))[0]
        # the volume held: w orthogonal to sqrt(r); the lowest root of
        # sum c_k^2/(mu_k - lambda) lies between the two lowest mu
        mu, Q = eigh_tridiagonal(d, self._e)
        c2 = (Q.T @ np.sqrt(self._ri))**2

        def secular(lam):
            return np.sum(c2/(mu - lam))
        gap = mu[1] - mu[0]
        lo, hi = mu[0] + 1e-12*gap, mu[1] - 1e-12*gap
        if secular(lo)*secular(hi) > 0:
            return mu[0] if c2[0] < c2[1] else mu[1]
        return brentq(secular, lo, hi, xtol=1e-14*max(1.0, abs(mu[0])))

    def lowest(self):
        """The lowest eigenvalue with the volume held (m = 0) and for m = 1."""
        return self.eigenvalues(0), self.eigenvalues(1)

    def odd(self):
        """How much of the lowest m = 0 mode (volume held) is odd about the
        middle of the profile (0 to 1)."""
        n = len(self._ri)
        A = np.diag(self._d) + np.diag(self._e, 1) + np.diag(self._e, -1)
        Z = null_space(np.sqrt(self._ri)[None, :])
        w = Z @ eigh(Z.T @ A @ Z, subset_by_index=[0, 0])[1][:, 0]
        u = w/np.sqrt(self._ri)
        return np.sum((u - u[::-1])**2)/(2*np.sum(u**2 + u[::-1]**2)) if n else 0.0


def _correct(q, H, B, tangent=None, q0=None, ds=0.0, volume=None):
    """Newton's method on the end conditions, plus the arclength condition
    (or a target volume)."""
    q = np.array(q, float)
    for _ in range(30):
        f, V = _residual(q, H, B)
        extra = (V - volume) if volume is not None else (q - q0) @ tangent - ds
        F = np.append(f, extra)
        if np.abs(F).max() < 1e-11:
            return q
        J = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1e-7
            fj, Vj = _residual(q + e, H, B)
            ej = (Vj - volume) if volume is not None else (q + e - q0) @ tangent - ds
            J[:, j] = (np.append(fj, ej) - F)/1e-7
        # least squares: at a symmetric bridge's branch point J is singular
        dq = np.linalg.lstsq(J, -F, rcond=1e-9)[0]
        q += dq
        if np.abs(dq).max() < 1e-11:
            return q
    if np.abs(F).max() < 1e-8:
        return q
    raise RuntimeError("the profile didn't converge")


def _tangent(q, H, B, previous):
    f0, _ = _residual(q, H, B)
    J = np.zeros((2, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-7
        J[:, j] = (_residual(q + e, H, B)[0] - f0)/1e-7
    t = null_space(J)[:, 0]
    return t if t @ previous > 0 else -t


def _follow(q, H, B, direction, g, ds=0.05, steps=400):
    """Follow the bridges from q, the volume first changing with the sign of
    ``direction``, to the first where ``g(profile)`` changes sign; that
    bridge, found by root finding along the last step."""
    t = _tangent(q, H, B, np.ones(3))
    if (_residual(q + 1e-6*t, H, B)[1] - _residual(q, H, B)[1])*direction < 0:
        t = -t
    g0 = g(Profile(q, H, B))
    for _ in range(steps):
        qn = _correct(q + ds*t, H, B, t, q, ds)
        if g(Profile(qn, H, B))*g0 <= 0:
            def along(step):
                return g(Profile(_correct(q + step*t, H, B, t, q, step), H, B))
            step = brentq(along, 0.0, ds, xtol=1e-12)
            return Profile(_correct(q + step*t, H, B, t, q, step), H, B)
        t = _tangent(qn, H, B, qn - q)
        q = qn
    raise RuntimeError("no limit found")


def start(slenderness, bond=0.0, volume=1.0):
    """The axisymmetric bridge with ``volume`` times the cylinder's, followed
    from the one with the cylinder's volume."""
    H = 2*slenderness
    if bond == 0:
        q = _correct([0.0, 1.0, H/2], H, bond, volume=np.pi*H)
    else:                                   # gravity switched on in stages
        q = np.array([0.0, 1.0, H])
        for b in np.linspace(0, bond, 11)[1:]:
            q = _correct(q, H, b, volume=np.pi*H)
    if volume != 1.0:
        q = _follow(q, H, bond, np.sign(volume - 1),
                    lambda prof: prof.volume/(np.pi*H) - volume).q
    return q, H


def limits(slenderness, bond=0.0, volume=1.0, which=("less", "more")):
    """The stability limits of the bridges reached from the one holding
    ``volume`` times the cylinder's, by taking liquid away and by adding it:
    for each, the reduced volume V/(pi R^2 H) where the first eigenvalue
    crosses zero, and the kind of mode, as ``kind`` says. Returns
    {"less": (volume, kind), "more": (volume, kind)} (those in ``which``)."""
    q, H = start(slenderness, bond, volume)
    out = {}
    for name in which:
        direction = {"less": -1, "more": 1}[name]
        prof = _follow(q, H, bond, direction, lambda prof: min(prof.lowest()))
        l0, l1 = prof.lowest()
        how = ("sideways" if abs(l1) < abs(l0) else
               _axisymmetric(prof.odd()))
        out[name] = (prof.volume/(np.pi*H), how)
    return out
