"""Draining a periodic bead chain in a slit, from immersed beads to the first break.

Used by slit_drainage.ipynb. Units: the wall spacing is 1, the surface tension 1.
Beads of radius 0.48 sit midway between the walls z = -1/2 and z = 1/2, centre to
centre ell apart along x. By symmetry one eighth of a unit cell is computed:
0 <= x <= ell/2, y >= 0, 0 <= z <= 1/2, the bead centred at the origin.

Constraints: 1 the mid-plane z = 0, 2 the wall z = 1/2, 3 the bead, 4 x = 0,
5 x = ell/2 (the mirror between beads), 6 y = 0 (the mirror across the chain).
"""
import numpy as np
import pysurfaceevolver as pyse


class Cell:
    """The eighth cell, its starting shapes and its relaxation."""

    def __init__(self, ell, radius=0.48, theta_bead=40.0, theta_wall=60.0):
        self.ell, self.radius, self.half = ell, radius, ell/2
        self.theta_bead, self.theta_wall = theta_bead, theta_wall
        self.r_meniscus = 0.5/np.cos(np.radians(theta_wall))
        # the bead's wetting energy is counted from the dry bead
        self.energy_constant = -np.cos(np.radians(theta_bead))*np.pi/2*radius**2
        self.delta = 0.006        # "near" the mirror y = 0
        self.snap_gap = 0.0005    # the band snaps when it is this close to y = 0
        self.band_step = 0.01     # relative volume step on the band
        self.edge_h = None        # the band's remeshing length (see relax)

    # ---- the model ------------------------------------------------------------
    def constraints(self):
        def dphi(f, n):      # a line integral of f dphi around the z axis
            return f"{n}1: (-y*({f}))/(x^2 + y^2)\n{n}2: (x*({f}))/(x^2 + y^2)\n{n}3: 0\n"
        wall = ("formula: z = 0.5\n"
                "energy:\ne1: -cos(theta_wall*pi/180)*y\ne2: 0\ne3: 0\n"
                "content:\nc1: 0.5*y\nc2: 0\nc3: 0\n")
        bead = ("formula: x^2 + y^2 + z^2 = radius^2\n"
                "energy:\n" + dphi("cos(theta_bead*pi/180)*radius*z", "e")
                + "content:\n" + dphi("z^3/3", "c"))
        return {1: "formula: z = 0", 2: wall, 3: bead, 4: "formula: x = 0",
                5: "formula: x = ell/2", 6: "formula: y = 0"}

    def load(self, vertices, faces, on, volume):
        ev = pyse.Evolver()
        ev.load_string(pyse.make_datafile(
            np.asarray(vertices, float), faces,
            bodies=[pyse.Body(faces=range(len(faces)), volume=volume)],
            parameters={"radius": self.radius, "ell": self.ell,
                        "theta_bead": self.theta_bead, "theta_wall": self.theta_wall},
            constraints=self.constraints(),
            vertex_constraints={k: [i for i, c in enumerate(on) if k in c] for k in range(1, 7)}))
        # the liquid volume excludes the bead's eighth
        ev.command(f"set body[1] volconst {float(-np.pi/6*self.radius**3)!r}")
        return ev

    # ---- starting shapes ------------------------------------------------------
    def meniscus(self, y0, ns=6, nt=6):
        """The immersed state: a circular-arc meniscus between the walls, at y = y0
        on the mid-plane, straight along x; normals +y."""
        r = self.r_meniscus
        vertices, on = [], []
        zs, xs = np.linspace(0, 0.5, nt + 1), np.linspace(0, self.half, ns + 1)
        for z in zs:
            for x in xs:
                vertices.append((x, y0 + r - np.sqrt(r**2 - z**2), z))
                on.append({k for k, c in ((4, x == 0), (5, x == xs[-1]), (1, z == 0),
                                          (2, z == 0.5)) if c})
        n = len(xs)
        faces = [[i*n + j, (i + 1)*n + j, (i + 1)*n + j + 1, i*n + j + 1]
                 for i in range(nt) for j in range(ns)]
        return vertices, faces, on

    def meniscus_exact(self, y0):
        """Volume and energy of the immersed state (exact: the meniscus has
        Delta p = -2 cos(theta_wall) and does not see the bead)."""
        r, ell = self.r_meniscus, self.half
        sag = r - np.sqrt(r**2 - 0.25)
        area_under = 0.5*y0 + 0.5*r - (0.25*np.sqrt(r**2 - 0.25) + r**2/2*np.arcsin(0.5/r))
        volume = ell*area_under - np.pi*self.radius**3/6
        energy = (ell*r*np.arcsin(0.5/r) - np.cos(np.radians(self.theta_wall))*ell*(y0 + sag)
                  - np.cos(np.radians(self.theta_bead))*np.pi*self.radius**2/2)
        return volume, energy

    def sheet(self, w, ns=6, nt=6):
        """The start after the bead emerges: a flat sheet y = w with a dry spot on
        the bead facing +y; normals +y."""
        R, ell = self.radius, self.half
        rho = np.sqrt(R**2 - w**2)
        arc, seg = np.pi*rho/2, 0.5 - rho
        total = arc + seg

        def left(t):     # up the bead from the mid-plane, then up the x = 0 mirror
            s = t*total
            if s <= arc:
                return rho*np.cos(s/rho), rho*np.sin(s/rho), 3
            return 0.0, rho + s - arc, 4
        ts = np.unique(np.concatenate([np.linspace(0, arc/total, nt + 1),
                                       np.linspace(arc/total, 1, 4)]))
        ss = np.linspace(0, 1, ns + 1)
        vertices, on = [], []
        for t in ts:
            xl, zl, k = left(t)
            for s in ss:
                vertices.append(((1 - s)*xl + s*ell, w, (1 - s)*zl + s*0.5*t))
                c = set()
                if s == 0:
                    c.add(k)
                    if np.isclose(t, arc/total):
                        c |= {3, 4}
                if s == 1:
                    c.add(5)
                if t == 0:
                    c.add(1)
                if t == 1:
                    c.add(2)
                on.append(c)
        n = len(ss)
        faces = [[i*n + j, (i + 1)*n + j, (i + 1)*n + j + 1, i*n + j + 1]
                 for i in range(len(ts) - 1) for j in range(ns)]
        return vertices, faces, on

    # ---- distances --------------------------------------------------------------
    def gap(self, ev, k, hops=0):
        """Smallest distance to the mirror y = 0 (k = 6) or the bead (k = 3) of the
        vertices not on that constraint, leaving out those within `hops` edges of
        one on it (next to a contact line they are close by nature)."""
        m = ev.mesh()
        near = ev.values("vertex", f"on_constraint {k}") > 0
        a, b = m.edges[:, 0], m.edges[:, 1]
        for _ in range(hops):
            grow = near.copy()
            grow[a[near[b]]] = True
            grow[b[near[a]]] = True
            near = grow
        v = m.vertices
        d = v[:, 1] if k == 6 else np.linalg.norm(v, axis=1) - self.radius
        return d[~near].min() if (~near).any() else np.inf

    # ---- relaxation -----------------------------------------------------------
    def relax(self, ev, remesh=False, maxit=80):
        if remesh:
            # length-based remeshing, once per step: split edges longer than 1.6 h,
            # delete edges shorter than 0.5 h (h: the band's first median edge). As
            # the dry patch on the bead grows, its contact line stretches and the
            # surface next to it is squeezed; without this, thin triangles pile up
            # where the contact line meets the mid-plane and the pressure gets noisy.
            if self.edge_h is None:
                m = ev.mesh()
                self.edge_h = float(np.median(np.linalg.norm(
                    m.vertices[m.edges[:, 0]] - m.vertices[m.edges[:, 1]], axis=1)))
            # edges from the contact line into the surface are not split: their
            # midpoints (on a chord of the sphere) would land inside the bead
            ev.command("foreach edge ee where (ee.vertex[1].on_constraint 3) != "
                       "(ee.vertex[2].on_constraint 3) do set ee no_refine")
            ev.command(f"l {1.6*self.edge_h:.6g}; t {0.5*self.edge_h:.6g}")
            ev.command("unset edge no_refine")
        last = None
        for _ in range(maxit):
            ev.command("g 10; u; V")
            e = ev.total_energy
            if last is not None and abs(e - last) < 1e-9:
                break
            last = e
        for _ in range(3):     # Newton steps, undone if one crosses the mirror y = 0
            snapshot = ev.save()
            ev.command("hessian_seek")
            if self.gap(ev, 6) < -1e-4:
                ev.restore(snapshot)
                break


def _solve(f, target, lo, hi):
    """Bisection for an increasing f."""
    for _ in range(60):
        mid = (lo + hi)/2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return mid


def drain(cell, v_start, v_end, levels=2, log=None):
    """Drain from immersed beads at liquid volume v_start (per eighth cell) until
    v_end or the first break, whichever comes first.

    Returns the states (dicts with stage 0 immersed / 1 band, volume, pressure,
    energy, vertices, facets) and the events (bead emergence; the snap)."""
    y0 = _solve(lambda y: cell.meniscus_exact(y)[0], v_start, cell.radius, 2.0)
    ev = cell.load(*cell.meniscus(y0), v_start)
    cell.relax(ev)
    for _ in range(levels):
        ev.refine()
        cell.relax(ev)
    states, events, stage = [], [], 0

    def record():
        m = ev.mesh()
        states.append(dict(stage=stage, volume=ev.bodies().volume[0],
                           pressure=ev.bodies().pressure[0], energy=ev.total_energy,
                           vertices=m.vertices.copy(), facets=m.facets.copy()))
        if log:
            log(f"stage {stage} V {states[-1]['volume']:.5f} "
                f"p {states[-1]['pressure']:+.4f} facets {len(m.facets)}")
    record()
    # the immersed meniscus is an arc about (y0 + r, 0) in the y-z plane, straight
    # along x: it first touches the bead at (0, R, 0), when y0 = R
    v_emerge = cell.meniscus_exact(cell.radius)[0]
    volume = v_start
    while volume > v_end:
        if stage == 0:
            volume = max(v_end, v_emerge, 0.93*volume)
        else:
            near = cell.gap(ev, 6, hops=2) < 4*cell.delta
            volume = max(v_end, (0.995 if near else 1 - cell.band_step)*volume)
        ev.command(f"set body[1] target {float(volume)!r}")
        cell.relax(ev, remesh=stage == 1)
        if stage == 0 and volume <= v_emerge:
            # the bead breaks through the meniscus: restart from a sheet with a dry
            # spot on the bead, relaxed at this volume
            before = ev.bodies().pressure[0]
            stage = 1
            ev = cell.load(*cell.sheet(cell.radius - 0.03), volume)
            cell.relax(ev)
            for _ in range(levels):
                ev.refine()
                cell.relax(ev)
            events.append(dict(kind="bead emerges", volume=float(volume), before=float(before),
                               after=float(ev.bodies().pressure[0])))
        elif stage == 1 and cell.gap(ev, 6) < cell.snap_gap:
            # the band reaches the mirror y = 0: it snaps there, and the liquid
            # around each bead is no longer connected to the next row's
            events.append(dict(kind="snap", volume=float(volume),
                               before=float(ev.bodies().pressure[0])))
            record()
            break
        record()
    return states, events
