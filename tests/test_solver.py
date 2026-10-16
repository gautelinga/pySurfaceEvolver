"""Sparse factoring for Newton steps: MUMPS (default when built in) and
Evolver's own minimal-degree factoring give the same steps and indices."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pysurfaceevolver as pyse
from conftest import FE_DIR


def has_mumps():
    current = pyse.solver()
    try:
        pyse.set_solver("mumps")
        return True
    except ValueError:
        return False
    finally:
        pyse.set_solver(current)


def test_solver_setting():
    current = pyse.solver()
    assert current in ("mumps", "evolver")
    pyse.set_solver("evolver")
    assert pyse.solver() == "evolver"
    with pytest.raises(ValueError):
        pyse.set_solver("nonsense")
    pyse.set_solver(current)
    assert pyse.solver() == current


NEWTON = r"""
import json, warnings
warnings.simplefilter("ignore")
import pysurfaceevolver as pyse
from pysurfaceevolver import Evolver
out = {"solver": pyse.solver()}
for name, commands in [
    ("cube.fe", "g 10; r; g 10; r; g 10; hessian; hessian"),
    ("mound.fe", "g 10; r; g 10; hessian; lagrange 3; g 3; hessian; hessian"),
    ("addload_example.fe", "g 10; r; g 10; hessian; hessian"),
]:
    ev = Evolver(name)
    ev.command(commands)
    out[name] = [ev.eval("total_energy"), [float(v) for v in ev.bodies().volume],
                 ev.command("eigenprobe 0").strip(), ev.command("eigenprobe 1").strip()]
print(json.dumps(out))
"""


def newton_run(solver):
    env = dict(os.environ, PYSE_SOLVER=solver)
    result = subprocess.run([sys.executable, "-c", NEWTON], capture_output=True, text=True,
                            cwd=FE_DIR, env=env, timeout=600)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not has_mumps(), reason="built without MUMPS")
def test_many_steps_on_one_pattern_are_the_same_after_a_reload(load):
    # a pattern factored six times is analysed again with PORD; a datafile
    # starts again with AMD, or the second run would factor its first steps
    # with the first run's PORD analysis (one thread: bit-reproducible)
    runs = []
    with pyse.threads_limit(1):
        for _ in range(2):
            ev = load("cube.fe")
            ev.command("g 5; r; g 2")
            steps = []
            for _ in range(8):
                ev.command("hessian")
                steps.append(ev.vertices)
            runs.append(steps)
    for a, b in zip(*runs):
        assert np.array_equal(a, b)


@pytest.mark.skipif(not has_mumps(), reason="built without MUMPS")
def test_mumps_newton_steps_match_evolver():
    mumps, evolver = newton_run("mumps"), newton_run("evolver")
    assert mumps.pop("solver") == "mumps" and evolver.pop("solver") == "evolver"
    for name, (e, vols, index0, index1) in mumps.items():
        e0, vols0, index00, index10 = evolver[name]
        assert abs(e - e0) <= 1e-10 * abs(e0), name
        for v, v0 in zip(vols, vols0):
            assert abs(v - v0) <= 1e-10 * abs(v0), name
        assert index0 == index00 and index1 == index10, name


def test_newton_leaves_a_free_surface_where_it_is(load):
    # nothing holds the cube: its rigid motions change neither energy nor
    # volume, and the Newton system is singular along them. They are taken
    # out of the step; left in, round-off moves the converged cube (24k
    # quadratic facets) by 4e-8 to 3e-7 a step, with the residual at 1e-10 to
    # 2e-9 (at 98k facets: 4e-6 and 2e-6, and relax() didn't converge). Taken
    # out: steps of 1e-14, the residual 1e-11 to 1e-10 (more with more threads)
    ev = load("cube.fe")
    ev.command("g 5; r; g 10; r; g 10; r; g 10; r; g 10; r; g 10; quadratic; g 5; "
               "hessian; hessian")
    for _ in range(3):
        x = ev.vertices
        ev.command("hessian")
        assert np.abs(ev.vertices - x).max() < 1e-11
    assert ev.residual() < 1e-9


@pytest.mark.parametrize("name", ["cat.fe", "mound.fe"])
def test_held_surfaces_get_evolvers_newton_steps(load, monkeypatch, name):
    # fixed rings, a drop on a plane: no rigid motion is free, and the steps
    # are the same to the bit with the check for free ones turned off
    runs = []
    with pyse.threads_limit(1):
        for off in (False, True):
            if off:
                monkeypatch.setenv("PYSE_NO_DEFLATE", "1")
            ev = load(name)
            ev.command("g 10; r; g 10; r; g 20")
            steps = []
            for _ in range(3):
                ev.command("hessian")
                steps.append(ev.vertices)
            runs.append(steps)
    for a, b in zip(*runs):
        assert np.array_equal(a, b)
