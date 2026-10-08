"""Sparse factoring for Newton steps: MUMPS (default when built in) and
Evolver's own minimal-degree factoring give the same steps and indices."""

import json
import os
import subprocess
import sys

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
def test_mumps_newton_steps_match_evolver():
    mumps, evolver = newton_run("mumps"), newton_run("evolver")
    assert mumps.pop("solver") == "mumps" and evolver.pop("solver") == "evolver"
    for name, (e, vols, index0, index1) in mumps.items():
        e0, vols0, index00, index10 = evolver[name]
        assert abs(e - e0) <= 1e-10 * abs(e0), name
        for v, v0 in zip(vols, vols0):
            assert abs(v - v0) <= 1e-10 * abs(v0), name
        assert index0 == index00 and index1 == index10, name
