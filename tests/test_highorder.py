"""Quadratic and Lagrange elements: node layout and tessellation."""

import numpy as np
import pytest


def triangle_area(points, triangles):
    a, b, c = (points[triangles[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()


@pytest.fixture
def sphere(cube):
    cube.command("g 5; r; g 5")
    return cube


@pytest.mark.parametrize("model, order, nodes", [
    ("linear", 1, 3), ("quadratic", 2, 6),
    ("lagrange 2", 2, 6), ("lagrange 3", 3, 10), ("lagrange 5", 5, 21),
])
def test_node_layout(sphere, model, order, nodes):
    sphere.command(model)
    m = sphere.mesh()
    k = sphere.counts["facets"]
    assert m.order == order and not m.bezier
    assert m.facet_nodes.shape == (k, nodes)
    assert m.facet_node_index.shape == (nodes, 3)
    assert m.edge_nodes.shape == (sphere.counts["edges"], order + 1)
    assert m.edge_node_index.shape == (order + 1, 2)
    # barycentric multi-indices: all distinct, each sums to the order
    assert (m.facet_node_index.sum(axis=1) == order).all()
    assert len({tuple(r) for r in m.facet_node_index.tolist()}) == nodes
    assert (m.edge_node_index.sum(axis=1) == order).all()
    # nodes are valid vertex rows, and each facet's corners are its face
    assert m.facet_nodes.max() < len(m.vertices)
    corners = [int(np.flatnonzero(m.facet_node_index[:, i] == order)[0]) for i in range(3)]
    for f in range(k):
        assert set(m.facet_nodes[f, corners]) == set(m.faces[f])
    edge_ends = [0, order]
    assert (m.edge_nodes[:, edge_ends] == m.edges).all()


def test_nodes_lie_on_the_surface(sphere):
    # with interpolating (non-Bezier) nodes, tessellating at the node lattice
    # reproduces the node coordinates exactly
    sphere.command("lagrange 3; g 2")
    m = sphere.mesh()
    points, _ = m.tessellate(3, merge=False)
    per = len(points) // len(m.faces)
    first = points[:per]
    expected = m.vertices[m.facet_nodes[0]]
    for p in expected:
        assert np.min(np.linalg.norm(first - p, axis=1)) < 1e-12


@pytest.mark.parametrize("model", ["quadratic", "lagrange 2", "lagrange 4"])
def test_tessellation_converges_to_evolver_area(sphere, model):
    sphere.command(model + "; g 3")
    m = sphere.mesh()
    area = sphere.eval("total_area")
    errors = [abs(triangle_area(*m.tessellate(n)) - area) for n in (2, 4, 8, 16)]
    # flat triangles converge at second order
    assert all(b < a / 3 for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-3 * area
    # and do much better than the corner triangles alone
    assert errors[-1] < abs(triangle_area(m.vertices, m.faces) - area) / 50


def test_quadratic_and_lagrange_2_agree(sphere):
    sphere.command("quadratic; g 3")
    area_quadratic = triangle_area(*sphere.mesh().tessellate(8))
    sphere.command("lagrange 2")
    area_lagrange = triangle_area(*sphere.mesh().tessellate(8))
    assert area_lagrange == pytest.approx(area_quadratic, rel=1e-9)


def test_tessellation_keeps_face_orientation(sphere):
    sphere.command("lagrange 3; g 3")
    m = sphere.mesh()
    n = 16
    points, triangles = m.tessellate(n)
    a, b, c = (points[triangles[:, i]] for i in range(3))
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    # body 1 in front of a face: positive orientation for its volume
    face_sign = np.where(m.face_bodies[:, 0] == 1, 1.0, -1.0)
    volume = (np.repeat(face_sign, n * n) * signed).sum()
    assert abs(volume) == pytest.approx(sphere.eval("body[1].volume"), rel=1e-3)
    # every small triangle has the same orientation as its face
    a, b, c = (m.vertices[m.faces[:, i]] for i in range(3))
    face_normal = np.cross(b - a, c - a)
    a, b, c = (points[triangles[:, i]] for i in range(3))
    tri_normal = np.cross(b - a, c - a)
    agree = np.einsum("ij,ij->i", np.repeat(face_normal, n * n, axis=0), tri_normal)
    assert (agree > 0).all()


def test_linear_tessellation_is_the_faces(cube):
    m = cube.mesh()
    points, triangles = m.tessellate()
    assert triangles.shape == m.faces.shape
    assert triangle_area(points, triangles) == pytest.approx(cube.eval("total_area"), rel=1e-14)


def test_tessellate_edges(sphere):
    sphere.command("lagrange 3; g 2")
    m = sphere.mesh()
    points, segments = m.tessellate_edges(6)
    assert segments.shape == (len(m.edges) * 6, 2)
    # the ends of each tessellated edge are the edge's vertices
    starts = points[segments[::6, 0]]
    np.testing.assert_allclose(starts, m.vertices[m.edges[:, 0]], atol=1e-14)


def test_string_model_edges(load):
    # A drop whose three quadratic edges are strongly curved.
    ev = load("slidestr.fe")
    ev.command("quadratic; g 2")
    m = ev.mesh()
    assert m.faces is None and m.facet_nodes is None
    points, segments = m.tessellate_edges(256)
    length = np.linalg.norm(points[segments[:, 1]] - points[segments[:, 0]], axis=1)
    length = length.reshape(len(m.edges), -1).sum(axis=1)
    # Evolver integrates edge length with Gaussian quadrature; as its order
    # goes up, it converges to the arc length of the tessellated edges.
    errors = []
    for order in (3, 7, 15):
        ev.command(f"integration_order_1d := {order}; recalc")
        errors.append(np.abs(ev.values("edge", "length") - length).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-2
    with pytest.raises(ValueError):
        m.tessellate()


def test_tessellate_rejects_bad_n(cube):
    with pytest.raises(ValueError):
        cube.mesh().tessellate(0)
