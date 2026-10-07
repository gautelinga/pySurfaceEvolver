"""The parallel facet loops (src/fastloops.c) agree with Evolver's original
loops: single evaluations to round-off, relaxed equilibria to 1e-9, for any
thread count, and reproducibly for a given thread count."""

import json
import os
import subprocess
import sys

import pytest

from conftest import FE_DIR

# soapfilm samples with bodies (several meeting at vertices in
# addload_example), gravity (mound), density (catbody); refined
# past the size where the loops use threads
SURFACES = {
    "cube.fe": "g 5; r; g 5; r; g 5; r; g 5; r; g 2",
    "mound.fe": "g 5; r; g 5; r; g 5; r; g 5; r; g 2",
    "catbody.fe": "g 5; r; g 5; r; g 5; r; g 5; r; g 2",
    "tankex.fe": "g 5; r; g 5; r; g 5; r; g 5; r; r; g 2",
    "addload_example.fe": "g 5; r; g 5; r; g 5; r; g 2",   # 9 bodies
}

DUMP = r"""
import sys, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver
for name, commands, path in zip(*[iter(sys.argv[1:])]*3):
    ev = Evolver(name)
    ev.command(commands)
    ev.dump(path)
"""

EVALUATE = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver
out = {}
for path in sys.argv[1:]:
    ev = Evolver(path)
    ev.command("m 0; g 1")   # forces, without moving
    out[path] = dict(facets=ev.counts["facets"], energy=ev.eval("total_energy"),
                     area=ev.eval("total_area"),
                     volume=[float(v) for v in ev.bodies().volume],
                     force=[ev.values("vertex", f"v_force[{i}]").tolist() for i in (1, 2, 3)])
print(json.dumps(out))
"""

RELAX = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver
out = {}
for name in sys.argv[1:]:
    ev = Evolver(name)
    ev.command("g 5; r; g 5; r; g 5; r; g 5; r; g 10; hessian; hessian; hessian; hessian")
    out[name] = dict(facets=ev.counts["facets"], energy=ev.eval("total_energy"),
                     volume=[float(v) for v in ev.bodies().volume],
                     pressure=[float(v) for v in ev.bodies().pressure],
                     index=ev.command("eigenprobe 0").strip())
print(json.dumps(out))
"""

MODES = {
    "original": {"PYSE_NO_FAST_LOOPS": "1"},
    "1 thread": {"OMP_NUM_THREADS": "1", "PYSE_CHECK_FACET_CACHE": "1"},
    "4 threads": {"OMP_NUM_THREADS": "4", "PYSE_CHECK_FACET_CACHE": "1"},
    "8 threads": {"OMP_NUM_THREADS": "8", "PYSE_CHECK_FACET_CACHE": "1"},
}


def run(script, args, env_extra):
    env = dict(os.environ, **env_extra)
    result = subprocess.run([sys.executable, "-c", script, *args], capture_output=True,
                            text=True, cwd=FE_DIR, env=env, timeout=600)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout else None


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


@pytest.fixture(scope="module")
def dumps(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fastloops")
    args, paths = [], []
    for name, commands in SURFACES.items():
        path = str(tmp / (name + ".dmp"))
        args += [name, commands, path]
        paths.append(path)
    run(DUMP, args, {})
    return paths


def test_single_evaluation_matches_original(dumps):
    results = {mode: run(EVALUATE, dumps, env) for mode, env in MODES.items()}
    original = results.pop("original")
    for path in dumps:
        ref = original[path]
        assert ref["facets"] > 4096
        fmax = max(abs(f) for comp in ref["force"] for f in comp)
        assert fmax > 0
        for mode, result in results.items():
            got = result[path]
            assert rel(got["energy"], ref["energy"]) < 1e-12, (path, mode)
            assert rel(got["area"], ref["area"]) < 1e-12, (path, mode)
            for v, w in zip(got["volume"], ref["volume"]):
                assert rel(v, w) < 1e-12, (path, mode)
            for comp, refcomp in zip(got["force"], ref["force"]):
                assert max(abs(f - g) for f, g in zip(comp, refcomp)) < 1e-12 * fmax, (path, mode)


def test_relaxed_equilibria_match_original():
    names = ["cube.fe", "mound.fe"]   # these converge under Newton steps
    results = {mode: run(RELAX, names, env) for mode, env in MODES.items()}
    original = results.pop("original")
    for name in names:
        ref = original[name]
        assert ref["facets"] > 4096
        for mode, result in results.items():
            got = result[name]
            assert rel(got["energy"], ref["energy"]) < 1e-9, (name, mode)
            for v, w in zip(got["volume"], ref["volume"]):
                assert rel(v, w) < 1e-9, (name, mode)
            for p, q in zip(got["pressure"], ref["pressure"]):
                assert abs(p - q) < 1e-9 * max(abs(q), 1.0), (name, mode)
            assert got["index"] == ref["index"], (name, mode)


BODY_CHANGES = r"""
import json, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver
ev = Evolver("cube.fe")
ev.command("g 2; r; r; r; r; g 2")
out = []
for step in ["set facet noncontent where id <= 3000", "unset facet noncontent",
             "unset facet frontbody where id <= 2000", "set facet frontbody 1 where id <= 1000"]:
    ev.command(step + "; recalc")    # bodies change, topology doesn't
    out.append(float(ev.bodies().volume[0]))
print(json.dumps(out))
"""


def test_facet_body_changes(tmp_path):
    original = run(BODY_CHANGES, [], MODES["original"])
    fast = run(BODY_CHANGES, [], MODES["8 threads"])   # checks the body cache too
    assert len(set(original)) == 4
    for v, w in zip(fast, original):
        assert rel(v, w) < 1e-12


def test_reproducible_for_a_thread_count(dumps):
    env = {"OMP_NUM_THREADS": "4"}
    assert run(EVALUATE, dumps, env) == run(EVALUATE, dumps, env)


def test_thread_setting():
    import pysurfaceevolver as pse
    default = pse.threads()
    assert default >= 1
    pse.set_threads(2)
    assert pse.threads() == 2
    pse.set_threads(0)
    assert pse.threads() == default
