"""The parallel facet loops (src/fastloops.c) give exactly the results of
Evolver's original loops."""

import json
import subprocess
import sys

import pytest

from conftest import FE_DIR

SCRIPT = r"""
import hashlib, json, sys, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver
out = {}
for name in sys.argv[1:]:
    ev = Evolver(name)
    ev.command("g 5; r; g 5")
    out[name] = [ev.eval("total_energy"), ev.eval("total_area"),
                 [float(v) for v in ev.bodies().volume],
                 hashlib.sha1(ev.vertices.tobytes()).hexdigest()]
print(json.dumps(out))
"""

# soapfilm samples with bodies, gravity (tankex, mound), density (catbody)...
SAMPLES = ["cube.fe", "mound.fe", "qmound.fe", "catbody.fe", "column.fe", "tankex.fe",
           "octa.fe", "phelanc.fe", "symtest.fe", "sphere.fe", "mylarcube.fe"]


def run(env_extra):
    import os
    env = dict(os.environ, **env_extra)
    result = subprocess.run([sys.executable, "-c", SCRIPT, *SAMPLES], capture_output=True,
                            text=True, cwd=FE_DIR, env=env, timeout=600)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_fast_loops_are_bit_identical():
    original = run({"PYSE_NO_FAST_LOOPS": "1"})
    for threads in ("1", "4"):
        fast = run({"OMP_NUM_THREADS": threads, "PYSE_CHECK_FACET_CACHE": "1"})
        for name in SAMPLES:
            assert fast[name] == original[name], (name, threads)


def test_thread_setting():
    import pysurfaceevolver as pse
    default = pse.threads()
    assert default >= 1
    pse.set_threads(2)
    assert pse.threads() == 2
    pse.set_threads(0)
    assert pse.threads() == default
