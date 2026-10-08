"""Numerical results: analytic solutions and regression values."""

import math

import numpy as np
import pytest

# (datafile, commands, total_energy, total_area)
#
# Reference values come from the standalone Surface Evolver 2.70a binary
# (stock Makefile, gcc -O3, x86-64). The parallel facet loops sum in a
# different order, and runs that stop short of equilibrium amplify the
# round-off differences (to ~2e-9 for cube.fe below); the tolerance allows
# for that and for other compilers and CPUs. Agreement at equilibrium is
# tested in test_fastloops.py.
REGRESSION = [
    ("cube.fe", "g 10", 5.1124931277273991, 5.1124931277273991),
    ("cube.fe", "g 5; r; g 10; hessian", 4.9044731220583371, 4.9044731220583371),
    ("mound.fe", "g 10; r; g 10", 3.8915064799379624, 12.891506479937963),
    ("cat.fe", "g 10; r; g 10", 17.490771798639049, 17.490771798639049),
    ("column.fe", "g 10", 11.755388534562526, 0.025558569866275482),
    ("quad.fe", "g 10", 4.4721359549995796, 4.4721359549995796),
    ("knotty.fe", "g 10", 563.42743895010517, 23.99999961292427),
    ("100grain.fe", "g 5", 19.72705901165649, 19.72705901165649),
]


@pytest.mark.parametrize(
    "datafile, commands, energy, area", REGRESSION,
    ids=[f"{f}:{c}" for f, c, _, _ in REGRESSION],
)
def test_regression(load, datafile, commands, energy, area):
    ev = load(datafile)
    ev.command(commands)
    assert ev.eval("total_energy") == pytest.approx(energy, rel=1e-8)
    assert ev.eval("total_area") == pytest.approx(area, rel=1e-8)


def test_runs_are_reproducible(load):
    results = []
    for _ in range(2):
        ev = load("cube.fe")
        ev.command("g 5; r; g 5; hessian")
        results.append((ev.eval("total_energy"), ev.vertices))
    # equal to round-off: with several threads MUMPS's factoring (hessian)
    # is not bit-reproducible
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-12)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=1e-12)


def sphere_area(volume):
    radius = (3 * volume / (4 * math.pi)) ** (1 / 3)
    return 4 * math.pi * radius**2


def test_cube_relaxes_toward_sphere(cube):
    areas = [cube.eval("total_area")]
    for step in ["g 5", "r; g 5", "hessian", "r; g 5", "hessian"]:
        cube.command(step)
        areas.append(cube.eval("total_area"))
    # the area never goes up, never drops below the sphere's (isoperimetric
    # inequality), and ends within 0.5% of it
    assert all(b <= a + 1e-12 for a, b in zip(areas, areas[1:]))
    assert areas[-1] >= sphere_area(1.0)
    assert areas[-1] == pytest.approx(sphere_area(1.0), rel=5e-3)


def test_cube_to_sphere_high_order(cube):
    # gogo2 (from cube.fe) refines and uses Lagrange elements up to order 6
    out = cube.command("gogo2")
    assert cube.model == "lagrange" and cube.lagrange_order == 6
    assert cube.eval("total_area") == pytest.approx(sphere_area(1.0), rel=1e-10)
    spread = float(out.split("Vertex radius spread:")[1].split()[0])
    assert spread < 1e-7


def test_sphere_pressure_is_laplace(cube):
    # surface tension 1: pressure = 2/R (Young-Laplace)
    cube.command("g 5; r; g 5; hessian; r; g 5; hessian")
    radius = (3 / (4 * math.pi)) ** (1 / 3)
    assert cube.bodies().pressure[0] == pytest.approx(2 / radius, rel=1e-2)


def test_volume_constraint_holds(cube):
    # each iteration projects back onto the constraint; after a few steps
    # the volume stays on target through refinement and Newton steps
    cube.command("g 3")
    for step in ["r", "g 5", "hessian", "r", "g 5", "hessian"]:
        cube.command(step)
        assert cube.eval("body[1].volume") == pytest.approx(1.0, rel=1e-8)


def test_hessian_converges(cube):
    cube.command("g 5; r; g 10")
    cube.command("hessian")
    before = cube.eval("total_energy")
    cube.command("hessian")
    after = cube.eval("total_energy")
    # at a critical point another Newton step changes almost nothing
    assert abs(after - before) < 1e-8 * before


# Edges with a density in a soapfilm model with a metric: their length used
# the facet routines and read a third vertex the edge doesn't have (stack
# overflow, crash in quadm.fe). Now the metric length at the edge midpoint,
# with its gradient as the force.
def test_metric_edge_density_does_not_crash(run_python):
    result = run_python("""
        import warnings; warnings.simplefilter("ignore")
        from pysurfaceevolver import Evolver
        ev = Evolver("quadm.fe")
        ev.command("set edge density 2 where id % 5 == 0; g 3; r; g 3")
        print("ok", ev.total_energy)
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ok")


METRIC_FULL = """\
1+4*x1^2  4*x1*x2  2*x1
4*x1*x2  1+4*x2^2  2*x2
2*x1      2*x2      1
"""


@pytest.mark.parametrize("metric", [f"METRIC\n{METRIC_FULL}",
                                    "CONFORMAL_METRIC\n1 + x1^2 + 0.5*x2\n"],
                         ids=["metric", "conformal"])
def test_metric_edge_force_is_energy_gradient(load, fe_dir, tmp_path, metric):
    text = (fe_dir / "quadm.fe").read_text()
    start, end = text.index("METRIC"), text.index("constraint 1")
    path = tmp_path / "metric_edges.fe"
    path.write_text(text[:start] + metric + "\n" + text[end:])
    ev = load(str(path))
    ev.command("g 3; r; g 3; r; g 3")
    ev.command("set edge density 2 where not fixed and id % 3 == 0")
    out = ev.command('foreach edge ee where density > 1.5 do { if not ee.vertex[1].fixed '
                     'and sum(ee.vertex[1].edge, fixed) == 0 then printf "%d ", ee.vertex[1].id }')
    vertices = sorted({int(x) for x in out.split()})[:4]
    assert vertices
    ev.command("m 0; g 1")   # forces (velocities: no constraints here), no move

    def energy():
        ev.command("recalc")
        return ev.total_energy

    for v in vertices:
        force = [float(ev.command(f"print vertex[{v}].v_velocity[{k}]")) for k in (1, 2, 3)]
        for k, c in enumerate("xyz"):
            x0 = float(ev.command(f"print vertex[{v}].{c}"))
            h = 1e-6
            ev.command(f"set vertex[{v}] {c} {x0 + h!r}")
            ep = energy()
            ev.command(f"set vertex[{v}] {c} {x0 - h!r}")
            em = energy()
            ev.command(f"set vertex[{v}] {c} {x0!r}")
            grad = (ep - em) / (2 * h)
            assert abs(grad + force[k]) <= 1e-6 * max(1.0, abs(grad))
