"""Time pySE on large surfaces: bulk operations, linear iterations and Newton
steps, and the Lagrange stage.

usage: python bench/benchmark.py [--levels 6 8] [--lagrange-levels 4 5]
                                 [--orders 2 4 6] [--threads 1 4]
                                 [--repeat 3] [--json out.json]

The test surface is the sample cube refined ``level`` times with ``g 10``
after each refinement, then relaxed with Newton steps (each refinement
quadruples the facets: level 4 is 6k facets, 5 is 24k, 6 is 98k, 8 is 1.6M).
Refining in steps keeps the surface near equilibrium, so the Newton steps
converge; refining several times at once and then taking Newton steps can
diverge.

Linear section (``--levels``): one iteration (``g 1``) and one Newton step
(``hessian``) on the relaxed surface, timed once; bulk operations, each the
best of ``--repeat`` runs.

Lagrange section (``--lagrange-levels``): as recommended in the Evolver
manual (sections 5.3 and 16.11), the triangulation is settled in the linear
model and then each order is a short final stage: ``lagrange n`` and
``g 5; hessian; hessian; hessian``, for each order in ``--orders`` in turn.
Each command is timed once.

Everything is repeated for each thread count in ``--threads``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings

import pysurfaceevolver as pse
from pysurfaceevolver import Evolver

warnings.simplefilter("ignore")


def best(f, repeat):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


def once(f):
    t = time.perf_counter()
    f()
    return time.perf_counter() - t


def progress(out: dict, name: str) -> None:
    print(f"  [{out['facets']:,} facets, {pse.threads()} threads] {name}: "
          f"{out[name] * 1e3:.1f} ms", flush=True)


def relaxed_cube(level: int) -> Evolver:
    """The cube refined `level` times, relaxed after each refinement, then
    converged with Newton steps."""
    ev = Evolver("cube.fe")
    ev.command("g 10")
    for _ in range(level):
        ev.command("r; g 10")
    for _ in range(4):
        before = ev.total_energy
        ev.command("hessian")
        if abs(ev.total_energy - before) <= 1e-12 * abs(ev.total_energy):
            break
    return ev


def run_linear(level: int, repeat: int) -> dict:
    t = time.perf_counter()
    ev = relaxed_cube(level)
    facets = ev.counts["facets"]
    print(f"  [{facets:,} facets, {pse.threads()} threads] relaxed in "
          f"{time.perf_counter() - t:.1f} s", flush=True)
    out = {"level": level, "facets": facets, "vertices": ev.counts["vertices"]}

    def fresh_mesh():
        ev.command("")          # bumps the surface version: no cached mesh
        return ev.mesh()

    mesh = fresh_mesh()
    z = ev.values("vertex", "z")
    tmp = tempfile.mkdtemp()
    for name in ("g 1", "hessian"):      # expensive: once
        out[name] = once(lambda: ev.command(name))
        progress(out, name)
    ops = {
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
        progress(out, name)
    # curved elements: Lagrange order 3 tessellated 6 x 6 per facet (36 triangles
    # per facet, so only on the smaller surfaces)
    if facets > 200_000:
        return out
    ev.set_model("lagrange", 3)
    curved = ev.mesh()
    out["lagrange3 tessellate(6)"] = best(lambda: curved.tessellate(6), repeat)
    out["lagrange3 body_surfaces(n=6)"] = best(lambda: curved.body_surfaces(n=6), 1)
    return out


def run_lagrange(level: int, orders: list) -> dict:
    ev = relaxed_cube(level)
    out = {"level": level, "facets": ev.counts["facets"], "vertices": ev.counts["vertices"]}
    for n in orders:
        steps = [(f"lagrange {n}", f"lagrange {n}"), (f"L{n} g 5", "g 5")]
        steps += [(f"L{n} hessian {i}", "hessian") for i in (1, 2, 3)]
        for name, command in steps:
            out[name] = once(lambda: ev.command(command))
            progress(out, name)
        out[f"L{n} energy"] = ev.total_energy
        out[f"L{n} index"] = ev.command("eigenprobe 0").strip()
    return out


def table(title: str, results: list) -> None:
    names = [k for k in results[0]
             if k not in ("level", "facets", "vertices", "threads")
             and "energy" not in k and "index" not in k]
    for r in results[1:]:
        names += [k for k in r if k not in names and k not in
                  ("level", "facets", "vertices", "threads")
                  and "energy" not in k and "index" not in k]
    print(f"\n{title:32}" + "".join(f"{r['facets']:>11,}f {r['threads']:>2}t" for r in results))
    for name in names:
        print(f"{name:32}" + "".join(
            f"{r[name] * 1e3:13.1f}ms" if name in r else f"{'-':>15}" for r in results))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", type=int, nargs="*", default=[6, 8],
                   help="refinement levels for the linear section")
    p.add_argument("--lagrange-levels", type=int, nargs="*", default=[4, 5],
                   help="refinement levels for the Lagrange section")
    p.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6],
                   help="Lagrange orders, run in turn")
    p.add_argument("--threads", type=int, nargs="+", default=[pse.threads()])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--json")
    args = p.parse_args()

    load = os.getloadavg()[0]
    if load > 0.5 * (os.cpu_count() or 1):
        print(f"warning: load average {load:.1f}; timings will be noisy")
    linear, lagrange = [], []
    for threads in args.threads:
        pse.set_threads(threads)
        for level in args.levels:
            linear.append(dict(run_linear(level, args.repeat), threads=threads))
        for level in args.lagrange_levels:
            lagrange.append(dict(run_lagrange(level, args.orders), threads=threads))
    pse.set_threads(0)
    if linear:
        table("linear", linear)
    if lagrange:
        table("lagrange", lagrange)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"linear": linear, "lagrange": lagrange,
                       "load": load, "cpus": os.cpu_count()}, f, indent=1)


if __name__ == "__main__":
    main()
