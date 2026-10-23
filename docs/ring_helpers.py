"""Two beads stacked in a box under a flat layer of liquid, drained through
the floor: used by drainage_topology.md (a pendular ring cut off) and
tools/drainage_topology_figure.py."""
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C

L = 3.0                                         # the box's side
BEADS = [(L/2, L/2, 1.015), (L/2, L/2, 3.045)]  # unit beads, 0.03 apart, the lower 0.015 off the floor


def flat(z, h):
    """The square [0, L]^2 at height z facing up, in nearly equilateral
    triangles of edge about h (rows offset by half an edge), and the rows
    on each side wall."""
    dy = h*np.sqrt(3)/2
    ny = max(2, int(round(L/dy)))
    nx = max(2, int(round(L/h)))
    rows, v = [], []
    for j, y in enumerate(np.linspace(0, L, ny + 1)):
        xs = np.linspace(0, L, nx + 1)
        if j % 2 and 0 < j < ny:
            xs = np.r_[0.0, (xs[:-1] + xs[1:])/2, L]
        rows.append(list(range(len(v), len(v) + len(xs))))
        v += [(x, y, z) for x in xs]
    v = np.array(v)
    f = []
    for a, b in zip(rows[:-1], rows[1:]):
        i = k = 0
        while i < len(a) - 1 or k < len(b) - 1:
            if k == len(b) - 1 or (i < len(a) - 1 and v[a[i + 1], 0] <= v[b[k + 1], 0]):
                f.append((a[i], a[i + 1], b[k]))
                i += 1
            else:
                f.append((a[i], b[k + 1], b[k]))
                k += 1
    tol = 1e-9
    sides = {"xlo": np.flatnonzero(v[:, 0] < tol), "xhi": np.flatnonzero(v[:, 0] > L - tol),
             "ylo": np.flatnonzero(v[:, 1] < tol), "yhi": np.flatnonzero(v[:, 1] > L - tol)}
    return v, np.array(f), sides


def two_beads(theta=20.0, h=0.3, top=4.05):
    """The liquid filling the box up to ``top`` over the two beads (contact
    angle ``theta``, flux gauge), the side walls and the floor ("bottom")
    at 90 degrees; relaxed."""
    v, f, sides = flat(top, h)
    walls = {"xlo": C.plane((1, 0, 0), 0.0, contact_angle=90.0),
             "xhi": C.plane((-1, 0, 0), -L, contact_angle=90.0),
             "ylo": C.plane((0, 1, 0), 0.0, contact_angle=90.0),
             "yhi": C.plane((0, -1, 0), -L, contact_angle=90.0),
             "bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0)}
    for i, c in enumerate(BEADS):
        walls[f"bead{i + 1}"] = C.sphere(c, 1.0, contact_angle=theta, gauge="flux")
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=walls, vertex_constraints=sides,
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    ev.relax()
    return ev


def at_floor(ev, step):
    """Whether the draining liquid has come down to the floor."""
    return ev.on_constraint("bottom").any() or any(
        e.kind == "solid" and str(e.constraint) == "bottom" and e.between == "body"
        for e in step.events)
