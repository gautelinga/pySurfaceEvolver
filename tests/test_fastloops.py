"""The parallel facet loops (src/fastloops.c) agree with Evolver's original
loops: single evaluations to round-off, relaxed equilibria to 1e-9, for any
thread count, and (single evaluations) reproducibly for a given thread count."""

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


LAGRANGE = r"""
import json, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver
ev = Evolver("cube.fe")
ev.command("g 10; r; g 10; r; g 10; r; g 10; hessian; hessian")
out = []
for n in (2, 4):
    ev.command(f"lagrange {n}; g 3; hessian; hessian")
    out.append([ev.eval("total_energy"), float(ev.bodies().volume[0]),
                float(ev.bodies().pressure[0]), ev.command("eigenprobe 0").strip()])
print(json.dumps(out))
"""


def test_lagrange_newton_steps_match_original():
    """The parallel Hessian assembly (src/fasthess.c) gives the original's
    Newton steps."""
    original = run(LAGRANGE, [], MODES["original"])
    for mode in ("1 thread", "4 threads"):
        got = run(LAGRANGE, [], MODES[mode])
        for (e, v, p, index), (e0, v0, p0, index0) in zip(got, original):
            assert rel(e, e0) < 1e-12 and rel(v, v0) < 1e-12, mode
            assert abs(p - p0) < 1e-9 * max(abs(p0), 1.0), mode
            assert index == index0, mode


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


def test_threads_limit_restores_the_setting():
    import pysurfaceevolver as pse
    from pysurfaceevolver import _core
    pse.set_threads(0)
    default = pse.threads()
    with pse.threads_limit(2):
        assert pse.threads() == 2
        with pse.threads_limit(3):
            assert pse.threads() == 3
        assert pse.threads() == 2
    assert pse.threads() == default and _core.thread_setting() == 0
    pse.set_threads(5)
    try:
        with pytest.raises(RuntimeError):
            with pse.threads_limit(1):
                raise RuntimeError
        assert pse.threads() == 5
    finally:
        pse.set_threads(0)


def test_threads_argument_for_one_call(load):
    ev = load("cube.fe")
    ev.relax(max_iter=5, threads=1)
    ev.hessian(threads=1)
    from pysurfaceevolver import _core
    assert _core.thread_setting() == 0


# The Newton-step matrix pattern kept across steps (src/fasthess.c): the same
# steps with and without it. symtest.fe adds entries the kept pattern lacks
# (the merge path); the others reuse it.
PATTERN_SAMPLES = {
    "cube.fe": "g 5; r; g 5; r; g 5; hessian; hessian; lagrange 3; g 2; hessian; hessian",
    "mound.fe": "g 5; r; g 5; hessian; hessian; hessian",
    "twointor.fe": "g 5; r; g 5; hessian; hessian; hessian",
    "quadm.fe": "g 5; r; g 5; hessian; hessian; hessian",
    "symtest.fe": "g 5; r; g 5; hessian; hessian; hessian; hessian",
    "100grain.fe": "g 5; hessian; hessian; hessian",
}

NEWTON = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver
out = {}
for name, commands in zip(*[iter(sys.argv[1:])]*2):
    ev = Evolver(name)
    ev.command(commands)
    out[name] = dict(energy=ev.eval("total_energy"),
                     volume=[float(v) for v in ev.bodies().volume])
print(json.dumps(out))
"""


@pytest.mark.parametrize("threads", ["1", "8"])
def test_kept_pattern_matches_hash_assembly(threads):
    args = [x for item in PATTERN_SAMPLES.items() for x in item]
    kept = run(NEWTON, args, {"OMP_NUM_THREADS": threads})
    hashed = run(NEWTON, args, {"OMP_NUM_THREADS": threads, "PYSE_NO_PATTERN": "1"})
    for name in PATTERN_SAMPLES:
        assert rel(kept[name]["energy"], hashed[name]["energy"]) < 1e-12, name
        for v, w in zip(kept[name]["volume"], hashed[name]["volume"]):
            assert rel(v, w) < 1e-12, name
