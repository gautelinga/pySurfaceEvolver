"""Draw the picture at the top of the README: four surfaces from the examples, side by
side (a drop on a table, a double bubble, a barrel drop on a fibre, a catenoid
between rings). Needs PyVista; takes a few seconds.

    OMP_NUM_THREADS=2 python tools/readme_figure.py   # writes docs/_static/readme.png
"""
import os

import numpy as np
import pyvista as pv

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "_static", "readme.png")
LIQUID = dict(color="#a6cee3", show_edges=True, edge_color="#7aa9d0", line_width=0.3,
              ambient=0.25, diffuse=0.75)
SOLID = dict(color="#c8c8c8", ambient=0.3, diffuse=0.7)


def polydata(vertices, facets):
    return pv.PolyData(np.asarray(vertices, float),
                       np.hstack([np.full((len(facets), 1), 3), facets]).ravel())


def drop():
    """The README's drop: volume 1 on a table at 60 degrees."""
    cap = shapes.cap(radius=1, height=0.6)
    ev = pyse.Evolver()
    ev.load_arrays(cap.vertices, cap.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60)},
                   vertex_constraints={1: cap.rims["rim"]}, bodies=[pyse.Body(volume=1.0)])
    ev.relax(levels=1)
    m = ev.mesh()
    r = 1.25*np.hypot(m.vertices[:, 0], m.vertices[:, 1]).max()
    table = pv.Disc(inner=0, outer=r, c_res=64)
    return [(polydata(m.vertices, m.facets), LIQUID), (table, SOLID)], (0.0, -1.0, 0.6)


def double_bubble():
    """Two unit boxes sharing a face, relaxed, the second shrunk to 0.6."""
    box = shapes.box((1, 1, 1), n=2)
    right = box.vertices + [1, 0, 0]
    row = {tuple(np.round(p, 9)): i for i, p in enumerate(box.vertices)}
    shared = np.array([row.get(tuple(np.round(p, 9)), -1) for p in right])
    new = len(box.vertices) + np.cumsum(shared < 0) - 1
    vertices = np.vstack([box.vertices, right[shared < 0]])
    right_faces = np.where(shared >= 0, shared, new)[box.faces]
    wall = np.flatnonzero(np.isclose(box.vertices[box.faces].mean(axis=1)[:, 0], 1))
    right_faces = right_faces[~np.isclose(vertices[right_faces].mean(axis=1)[:, 0], 1)]
    faces = np.vstack([box.faces, right_faces])
    n = len(box.faces)
    ev = pyse.Evolver()
    ev.load_arrays(vertices, faces, bodies=[
        pyse.Body(1.0, faces=range(n)),
        pyse.Body(1.0, faces=list(wall) + list(range(n, len(faces))))])
    ev.relax(levels=2)
    ev.body(2).target = 0.6
    ev.relax()
    m = ev.mesh()
    film = dict(LIQUID, opacity=0.6)
    return [(polydata(m.vertices, m.facets), film)], (0.35, -1.0, 0.4)


def barrel():
    """A drop of volume 2 on a fibre of radius 0.2, at 30 degrees."""
    z = np.linspace(-1.2, 1.2, 13)
    s = shapes.revolve(np.sqrt(0.04 + (0.75**2 - 0.04)*(1 - (z/1.2)**2)), z, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(s.vertices, s.faces,
                   constraints={1: C.cylinder((0, 0, 0), (0, 0, 1), 0.2, contact_angle=30)},
                   vertex_constraints={1: s.rims["bottom"] + s.rims["top"]},
                   bodies=[pyse.Body(volume=2.0)])
    ev.relax(levels=1)
    m = ev.mesh()
    # lying along x, as a fibre would be drawn
    x = m.vertices[:, [2, 1, 0]]
    fibre = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=0.199, height=4.0,
                        resolution=48)
    return [(polydata(x, m.facets), LIQUID), (fibre, SOLID)], (0.25, -1.0, 0.5)


def catenoid():
    """A soap film between two rings of radius 1, 1 apart (short of the fold), the
    rings drawn as tubes."""
    z = np.linspace(-0.5, 0.5, 9)
    tube = shapes.revolve(np.ones_like(z), z, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces, fixed=tube.rims["bottom"] + tube.rims["top"])
    ev.relax(levels=2)
    m = ev.mesh()
    film = polydata(m.vertices, m.facets)
    rings = film.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                       manifold_edges=False, non_manifold_edges=False)
    return [(film, LIQUID), (rings.tube(radius=0.03, n_sides=16), SOLID)], (0.2, -1.0, 0.45)


def main():
    panels = [drop(), double_bubble(), barrel(), catenoid()]
    n = len(panels)
    p = pv.Plotter(shape=(1, n), off_screen=True, window_size=(420*n, 360), border=False,
                   lighting="three lights")
    p.set_background("white")
    for k, (meshes, view) in enumerate(panels):
        p.subplot(0, k)
        for mesh, style in meshes:
            p.add_mesh(mesh, smooth_shading=False, **style)
        liquid = meshes[0][0]
        c = np.array(liquid.center)
        size = np.ptp(liquid.points, axis=0).max()
        v = np.array(view, float)
        v /= np.linalg.norm(v)
        p.camera.focal_point = c
        p.camera.position = c + v*size*3
        p.camera.up = (0, 0, 1)
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 0.62*size
    p.screenshot(OUT)
    p.close()
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
