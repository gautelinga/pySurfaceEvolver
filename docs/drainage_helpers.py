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
from pysurfaceevolver import constraints as C


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
        # walls and the bead wet with the contact angles in the datafile's
        # parameters; the eighth bead's integrals leave out its wet north pole
        # (pyse.constraints gives the volume constant; the energy constant is
        # self.energy_constant)
        return {1: C.mirror("z"),
                2: C.plane((0, 0, -1), point=(0, 0, 0.5), contact_angle="theta_wall"),
                3: C.sphere((0, 0, 0), self.radius, contact_angle="theta_bead",
                            wet_poles=("north",), span=np.pi/2),
                4: C.mirror("x"), 5: C.mirror("x", self.half), 6: C.mirror("y")}

    def load(self, vertices, faces, on, volume):
        cons = self.constraints()
        ev = pyse.Evolver()
        ev.load_string(pyse.make_datafile(
            np.asarray(vertices, float), faces,
            bodies=[pyse.Body(faces=range(len(faces)), volume=volume,
                              volconst=cons[3].volconst)],
            parameters={"radius": self.radius, "ell": self.ell,
                        "theta_bead": self.theta_bead, "theta_wall": self.theta_wall},
            constraints=cons,
            vertex_constraints={k: [i for i, c in enumerate(on) if k in c] for k in range(1, 7)}))
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
        near = ev.on_constraint(k)
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
    def relax(self, ev, remesh=False):
        if remesh:
            # remeshing by edge length, once per step (h: the band's first median
            # edge). As the dry patch on the bead grows, its contact line stretches
            # and the surface next to it is squeezed; without this, thin triangles
            # pile up where the contact line meets the mid-plane and the pressure
            # gets noisy.
            if self.edge_h is None:
                self.edge_h = ev.mesh_quality().edge_median
            ev.remesh(target=self.edge_h)
        # to equilibrium (gradient steps with mesh upkeep, then Newton steps); a
        # Newton step that pushes the surface through the mirror y = 0 is undone
        ev.relax(stability=False, undo_if=lambda ev: self.gap(ev, 6) < -1e-4)


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
        ev.body(1).target = volume
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
