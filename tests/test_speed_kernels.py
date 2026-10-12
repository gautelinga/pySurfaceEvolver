"""Speed-ups in the engine's kernels agree with Evolver's own code: the
Lagrange-model vertex normals kept per topology (src/fastloops.c, vn_*),
checked vertex by vertex against new_calc_vertex_normal() under
PYSE_CHECK_FACET_CACHE=1 (a mismatch aborts) and against the original loops
(PYSE_NO_FAST_LOOPS=1) through the residual, a Newton step and the inertia,
each evaluated once from the same dumped surface."""

import json
import os
import subprocess
import sys

import pytest

from conftest import FE_DIR

# corner, edge and (order 3) facet nodes; constraints and a contact line
# (mound); triple edges where films meet (addload_example: those vertices
# keep Evolver's own normals); bodies (catbody)
LAGRANGE_SURFACES = {
    "cube.fe": "g 5; r; g 5; r; g 5; r; g 5; r; g 5; lagrange 2; g 2",
    "cube.fe ": "g 5; r; g 5; r; g 5; r; g 5; lagrange 3; g 2",
    "mound.fe": "g 5; r; g 5; r; g 5; r; g 5; r; g 5; lagrange 2; g 2",
    "addload_example.fe": "g 5; r; g 5; r; g 5; r; g 5; lagrange 2; g 2",
    "addload_example.fe ": "g 5; r; g 5; r; g 5; lagrange 3; g 2",
    "catbody.fe": "g 5; r; g 5; r; g 5; r; g 5; r; g 5; lagrange 2; g 2",
}

DUMP = r"""
import sys, warnings
warnings.simplefilter("ignore")
from pysurfaceevolver import Evolver
for name, commands, path in zip(*[iter(sys.argv[1:])]*3):
    ev = Evolver(name.strip())
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
    residual = ev.residual()            # shape normals
    index = ev.command("eigenprobe 0").strip()   # hessian_init's normals
    ev.command("hessian")
    out[path] = [ev.counts["vertices"], residual, index, ev.eval("total_energy")]
# the cache follows changes of the topology: facets reversed (the vertex
# count stays), the order changed
ev = Evolver(sys.argv[1])
ev.residual()
ev.command("reverse_orientation facet where id < 200")
out["reversed"] = [ev.counts["vertices"], ev.residual(), ev.command("eigenprobe 0").strip(), 0.0]
ev.command("lagrange 4")
out["order 4"] = [ev.counts["vertices"], ev.residual(), ev.command("eigenprobe 0").strip(), 0.0]
print(json.dumps(out))
"""


def run(script, args, env_extra):
    env = dict(os.environ, OMP_NUM_THREADS="2")
    env.update(env_extra)
    result = subprocess.run([sys.executable, "-c", script, *args], capture_output=True,
                            text=True, cwd=FE_DIR, env=env, timeout=900)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout else None


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


@pytest.fixture(scope="module")
def lagrange_dumps(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("speed_kernels")
    args, paths = [], []
    for i, (name, commands) in enumerate(LAGRANGE_SURFACES.items()):
        path = str(tmp / f"{i}_{name.strip()}.dmp")
        args += [name, commands, path]
        paths.append(path)
    run(DUMP, args, {})
    return paths


def test_lagrange_vertex_normals_match_evolver(lagrange_dumps):
    original = run(EVALUATE, lagrange_dumps, {"PYSE_NO_FAST_LOOPS": "1"})
    for threads in ("1", "2"):
        got = run(EVALUATE, lagrange_dumps,
                  {"OMP_NUM_THREADS": threads, "PYSE_CHECK_FACET_CACHE": "1"})
        for key, ref in original.items():
            vertices, residual, index, energy = got[key]
            assert vertices == ref[0] and vertices > 4096, key
            assert rel(residual, ref[1]) < 1e-10, (key, threads)
            assert index == ref[2], (key, threads)
            assert rel(energy, ref[3]) < 1e-12 or energy == ref[3], (key, threads)


# V's plain vertices from rings kept per topology (fastloops.c, av_*), bit
# for bit as find_vertex_average(): checked vertex by vertex under
# PYSE_CHECK_FACET_CACHE=1 (a mismatch aborts) and against the serial V
# (PYSE_NO_FAST_MESH=1) by the meshes' digests. Equiangulation and
# refinement between the averagings change the topology (the rings must be
# rebuilt); constraints, boundary wires, triple lines, bare edges and fixed
# vertices keep Evolver's search.
AVERAGES = r"""
import hashlib, json, sys, warnings
warnings.simplefilter("ignore")
import numpy as np
from pysurfaceevolver import Evolver
out = {}
for name, commands in zip(sys.argv[1::2], sys.argv[2::2]):
    ev = Evolver(name)
    for c in commands.split(";"):
        ev.command(c)
    mesh = ev.mesh()
    out[name] = hashlib.sha1(np.ascontiguousarray(mesh.vertices).tobytes()
                             + np.ascontiguousarray(mesh.facets).tobytes()).hexdigest()
print(json.dumps(out))
"""

JIGGLE = "set vertex x x+0.004*(random-0.5); set vertex y y+0.004*(random-0.5)"
AVERAGE_SAMPLES = {
    "cube.fe": f"g 5; r; r; r; r; r; {JIGGLE}; V; V; u; V; r; V; rawv; rawestv; V",
    "mound.fe": f"g 5; r; r; r; r; {JIGGLE}; V; u; V; V",
    "addload_example.fe": f"g 5; r; r; r; {JIGGLE}; V; u; V",
    "catbody.fe": "g 5; r; r; r; r; r; V; u; V",
    # quantity mode (gap energy): the parallel search's facet areas differ
    # from the serial one's at round-off, so only the vertex-by-vertex check
    "tankex.fe": "g 5; r; r; r; r; r; V; V",
}


def test_vertex_average_rings_match_serial():
    args = [x for item in AVERAGE_SAMPLES.items() for x in item]
    serial = run(AVERAGES, args, {"PYSE_NO_FAST_MESH": "1"})
    for threads in ("1", "2"):
        fast = run(AVERAGES, args, {"OMP_NUM_THREADS": threads, "PYSE_CHECK_FACET_CACHE": "1"})
        for name in AVERAGE_SAMPLES:
            if name != "tankex.fe":
                assert fast[name] == serial[name], (name, threads)
