"""Periodic slabs for periodic.md: a flat periodic sheet, and two stacked
beads under a layer of liquid in a cell periodic in x and y."""
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C

L = 3.0                                          # the cell's side
BEADS = [(L/2, L/2, 1.015), (L/2, L/2, 3.045)]  # unit beads, 0.03 apart, 0.015 off the floor


def sheet(z, n):
    """A square sheet over the cell [0, L)^2 at height z, facing up: n x n
    rows of vertices, every other row offset by half a step (nearly
    equilateral triangles), joined across the cell's sides."""
    xs = np.arange(n)*L/n
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    X = X + 0.5*(np.arange(n)[None, :] % 2)*L/n
    v = np.c_[X.ravel(), Y.ravel(), np.full(n*n, z)]
    at = lambda i, j: (i % n)*n + (j % n)
    f = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)
            f += [(a, b, c), (b, d, c)] if j % 2 == 0 else [(a, b, d), (a, d, c)]
    return v, np.array(f)


def two_beads(theta=20.0, n=10, top=4.05):
    """The liquid filling the cell up to ``top`` over the two beads (contact
    angle ``theta``, flux gauge), the floor ("bottom") at 90 degrees; relaxed."""
    v, f = sheet(top, n)
    walls = {"bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0)}
    for i, c in enumerate(BEADS):
        walls[f"bead{i + 1}"] = C.sphere(c, 1.0, contact_angle=theta, gauge="flux")
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=walls, bodies=[pyse.Body(volume=None)], periods=(L, L))
    ev.body(1).target = ev.body(1).volume
    ev.relax()
    return ev
