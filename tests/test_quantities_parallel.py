"""Named quantities in parallel (src/fasthess.c): facet, edge and vertex
integral methods give the original loops' results, and integrand errors are
reported the same way."""

import json
import os
import subprocess
import sys

import pytest

from conftest import FE_DIR

QUANTITIES = """// cube.fe with named-quantity integrals
quantity grav energy method facet_scalar_integral global
scalar_integrand: 0.3*z^2 + 0.1*INTEGRAND
quantity vol fixed = 1 method facet_vector_integral global
vector_integrand:
q1: 0
q2: 0
q3: z
quantity rim energy method edge_scalar_integral global
scalar_integrand: 0.05*(1+x^2)
quantity pts energy method vertex_scalar_integral global
scalar_integrand: 0.01*(x^2+y^2)

"""


def datafile(tmp_path, integrand):
    cube = (FE_DIR / "cube.fe").read_text().replace("\r", "")
    body = cube[cube.index("vertices"):cube.index("bodies")]
    path = tmp_path / "qcube.fe"
    path.write_text(QUANTITIES.replace("INTEGRAND", integrand) + body)
    return str(path)


RUN = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver, EvolverError
out = []
try:
    ev = Evolver(sys.argv[1])
    for c in ["g 10; r; g 10; r; g 10; r; g 10", "hessian", "hessian",
              "lagrange 2", "g 3", "hessian"]:
        ev.command(c)
        out.append([ev.total_energy] + [q.value for q in ev.quantities().values()])
except EvolverError as e:
    out = str(e).strip().splitlines()[0]
print(json.dumps(out))
"""


def run(path, env_extra):
    env = dict(os.environ, **env_extra)
    result = subprocess.run([sys.executable, "-c", RUN, path], capture_output=True,
                            text=True, cwd=FE_DIR, env=env, timeout=600)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_integral_quantities_match_original(tmp_path):
    path = datafile(tmp_path, "sin(x)*y")
    original = run(path, {"PYSE_NO_FAST_LOOPS": "1"})
    for threads in ("1", "8"):
        fast = run(path, {"PYSE_THREADS": threads})
        assert len(fast) == len(original)
        for row, row0 in zip(fast, original):
            for v, v0 in zip(row, row0):
                assert v == pytest.approx(v0, rel=1e-9), threads


def test_integrand_errors_reported_as_before(tmp_path):
    path = datafile(tmp_path, "sqrt(z-0.5)")   # negative on the bottom facets
    original = run(path, {"PYSE_NO_FAST_LOOPS": "1"})
    fast = run(path, {"PYSE_THREADS": "8"})
    assert isinstance(original, str) and "Square root of negative" in original
    assert fast == original
