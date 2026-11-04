"""One corpus run with light snapshots for a film: every step, and breakthrough.

The liquid's facets (their corners, float16) and each facet's body are saved
at every step the continuation yields, and where the run ends (breakthrough
comes inside a descent, after the last step; corpus.py --frames keeps only
every so many steps, and not that). SNAPS/index.json lists them: per snapshot
(n, kind, step, V, trapped) with kind "step" or "end" and the volumes of the
last step yielded; SNAPS/beads.npz the beads' centres and the cell. The
surface is only read: at one thread the run goes as corpus.py's (its log
line for line).

usage: OMP_NUM_THREADS=1 python bench/topology/snapshots.py SNAPS --one NAME N L SEED THETA H DS OUT GOAL [options]

with corpus.py's --one arguments (a run as a set ran it: the set's name and
parameters, --frames 5 and --budget as the set had them; a resumed run
reloads its surface on its own schedule and may part from the original).
tools/drainage_topology_figure.py pack64 --snaps SNAPS draws the film.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import corpus  # noqa: E402
from pysurfaceevolver import recipes  # noqa: E402
from pysurfaceevolver._engine import surface  # noqa: E402

snaps = sys.argv[1]
os.makedirs(snaps, exist_ok=True)
index, last = [], {"ev": None, "step": -1, "V": None, "T": None}


def snap(ev, kind):
    m = ev.mesh()
    np.savez_compressed(os.path.join(snaps, f"s{len(index):06d}.npz"),
                        c=np.asarray(m.facet_corners(), np.float16),
                        b=np.asarray(m.facet_bodies).max(axis=1).astype(np.int16))
    index.append((len(index), kind, last["step"], last["V"], last["T"]))


arclength = recipes.arclength


def _arclength(ev, *args, **kwargs):
    last["ev"] = ev
    if not os.path.exists(os.path.join(snaps, "beads.npz")):
        centres = [np.asarray(w.slabflux[0], float) for w in surface.walls.values()
                   if getattr(w, "slabflux", None)]
        np.savez(os.path.join(snaps, "beads.npz"), centres=np.array(centres),
                 L=float(ev.periods[0]) if ev.periods else np.nan)
    for s in arclength(ev, *args, **kwargs):
        last.update(step=int(s.index), V=float(s.value), T=float(s.trapped))
        snap(ev, "step")
        yield s


recipes.arclength = _arclength
try:
    corpus.main(sys.argv[2:])
finally:
    if last["ev"] is not None:
        snap(last["ev"], "end")
    json.dump(index, open(os.path.join(snaps, "index.json"), "w"))
    print(f"{len(index)} snapshots in {snaps}", flush=True)
