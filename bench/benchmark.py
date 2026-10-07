"""Time pySE's bulk operations on large surfaces.

usage: python bench/benchmark.py [--levels 6 8] [--repeat 3] [--json out.json]

The test surface is the sample cube relaxed a little and refined
``level`` times (each refinement quadruples the facets: level 6 is 98k
facets, level 8 is 1.6M). Each timing is the best of ``--repeat`` runs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings

import numpy as np

from pysurfaceevolver import Evolver

warnings.simplefilter("ignore")


def best(f, repeat):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


def run(level: int, repeat: int) -> dict:
    ev = Evolver("cube.fe")
    ev.command("g 5")
    ev.refine(level)
    ev.command("g 1")
    facets = ev.counts["facets"]
    out = {"level": level, "facets": facets, "vertices": ev.counts["vertices"]}

    def fresh_mesh():
        ev.command("")          # bumps the surface version: no cached mesh
        return ev.mesh()

    mesh = fresh_mesh()
    z = ev.values("vertex", "z")
    tmp = tempfile.mkdtemp()
    ops = {
        "g 1": lambda: ev.command("g 1"),
        "mesh()": fresh_mesh,
        "vertices (read)": lambda: ev.vertices,
        "values(vertex, x)": lambda: ev.values("vertex", "x"),
        "values(facet, area)": lambda: ev.values("facet", "area"),
        "set_values(vertex, z)": lambda: ev.set_values("vertex", "z", z),
        "vertices = ...": lambda: setattr(ev, "vertices", mesh.vertices),
        "tessellate()": lambda: mesh.tessellate(),
        "body_surfaces()": lambda: mesh.body_surfaces(),
        "write .vtu": lambda: ev.write(os.path.join(tmp, "s.vtu")),
    }
    try:
        import pyvista  # noqa: F401
        ops["to_pyvista()"] = lambda: mesh.to_pyvista()
    except ImportError:
        pass
    for name, f in ops.items():
        out[name] = best(f, repeat)
    # curved elements: Lagrange order 3 tessellated 6 x 6 per facet (36 triangles
    # per facet, so only on the smaller surfaces)
    if facets > 200_000:
        return out
    ev.set_model("lagrange", 3)
    curved = ev.mesh()
    out["lagrange3 tessellate(6)"] = best(lambda: curved.tessellate(6), repeat)
    out["lagrange3 body_surfaces(n=6)"] = best(lambda: curved.body_surfaces(n=6), 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", type=int, nargs="+", default=[6, 8])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--json")
    args = p.parse_args()
    results = [run(level, args.repeat) for level in args.levels]
    names = [k for k in results[0] if k not in ("level", "facets", "vertices")]
    print(f"{'operation':32}" + "".join(f"{r['facets']:>12,} f" for r in results))
    for name in names:
        print(f"{name:32}" + "".join(
            f"{r[name] * 1e3:12.1f} ms" if name in r else f"{'-':>14}" for r in results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
