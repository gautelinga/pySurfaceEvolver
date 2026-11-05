"""Wetting a bead pack: pendular bridges at the contacts growing together.

Beads settled in a cell periodic in x and y (pack.settle_periodic, lifted
LIFT off the floor), a bridge at every contact (pack.bridges_periodic), all
the liquid one body at one pressure (arclength's trap=False: bridges
exchange liquid as through films or the vapour). Liquid is added by
arclength(topology=True) until it fills SMAX of the pore space
(pack.pore_volume, between the lowest and highest centres), the budget runs
out or the liquid reaches the floor. Per step: the volume, the saturation,
the pressure, the surface's pieces and each bead's force
(Evolver.wall_force), in OUT/NAME.npz; the log in OUT/NAME.log, a summary in
OUT/NAME.json, and with --frames each step's liquid (facet corners, float16)
in OUT/frames/NAME/ for a film.

usage: python bench/topology/wet.py N SEED THETA OUT [--L 6] [--ds 0.05]
                                    [--smax 0.35] [--budget 3e6]
                                    [--frames]
"""
import argparse, json, logging, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pack import bridges_periodic, pore_volume, settle_periodic  # noqa: E402

LIFT = 0.5          # the beads' lowest points this far over the floor


def pieces(ev):
    """The liquid surface's pieces: (their count, the largest's share of the facets)."""
    from pysurfaceevolver import _clusters as K
    m = ev.mesh()
    p = K.pieces(np.asarray(m.facets), np.asarray(m.facet_bodies))
    counts = np.bincount(p[p >= 0])
    counts = counts[counts > 0]
    return len(counts), float(counts.max()/counts.sum())


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("theta", type=float)
    ap.add_argument("out")
    ap.add_argument("--L", type=float, default=6.0)
    ap.add_argument("--ds", type=float, default=0.05)
    ap.add_argument("--smax", type=float, default=0.35)
    ap.add_argument("--budget", type=float, default=3e6)
    ap.add_argument("--frames", action="store_true")
    a = ap.parse_args(argv)
    import pysurfaceevolver as pyse
    from pysurfaceevolver import _core, recipes
    name = f"w{a.n}_s{a.seed}_t{a.theta:g}"
    os.makedirs(a.out, exist_ok=True)
    frames = os.path.join(a.out, "frames", name) if a.frames else None
    if frames:
        os.makedirs(frames, exist_ok=True)
    log = open(os.path.join(a.out, f"{name}.log"), "w")
    t0 = time.time()
    at = {"step": -1}

    def say(text):
        log.write(f"{time.time() - t0:8.1f}s step {at['step']:4d}  {text}\n")
        log.flush()

    class Handler(logging.Handler):
        def emit(self, record):
            say(record.getMessage())

    logger = logging.getLogger("pysurfaceevolver.topology")
    logger.addHandler(Handler())
    logger.setLevel(logging.DEBUG)
    x, gap, _ = settle_periodic(a.n, a.L, seed=a.seed)
    x[:, 2] += LIFT
    ev, pairs = bridges_periodic(x, a.L, theta=a.theta)
    pores = pore_volume(x, a.L)
    names = [f"bead{k + 1}" for k in range(a.n)]
    np.savez(os.path.join(a.out, f"{name}_beads.npz"), centres=x, L=a.L, pairs=np.array(pairs),
             pores=pores)
    say(f"{a.n} beads, {len(pairs)} contacts, pore volume {pores:.3f}, "
        f"facets {ev.counts['facets']}")
    rows, forces = [], []
    stop = ""
    with_floor = ev.on_constraint("bottom")
    try:
        for s in recipes.arclength(ev, ev.body(1), a.ds, 100000, topology=True, trap=False,
                                   budget=a.budget):
            at["step"] = int(s.index)
            F = np.array([ev.wall_force(k) for k in names])
            count, share = pieces(ev)
            S = s.value/pores
            rows.append((s.index, s.value, S, s.pressures[0], count, share, bool(s.fell),
                         _core.work()))
            forces.append(F)
            say(f"V {s.value:.4f} S {S:.4f} p {s.pressures[0]:.4f} pieces {count} "
                f"largest {share:.2f} |F| {np.median(np.linalg.norm(F, axis=1)):.3f} "
                f"facets {ev.counts['facets']}{' fell' if s.fell else ''}")
            if frames:
                m = ev.mesh()
                np.savez_compressed(os.path.join(frames, f"s{s.index:05d}.npz"),
                                    c=np.asarray(m.facet_corners(), np.float16))
            if np.asarray(ev.on_constraint("bottom")).any() and not with_floor.any():
                stop = "the liquid reached the floor"
                break
            if S >= a.smax:
                stop = f"saturation {a.smax}"
                break
        else:
            stop = "steps"
    except Exception as e:      # (a run's crash ends that run, recorded)
        stop = f"{type(e).__name__}: {e}"
        say(f"stopped: {stop}")
    if not stop:
        stop = "the run ended (budget or a failure; see the log)"
    secs = time.time() - t0
    say(f"end: {stop}")
    np.savez(os.path.join(a.out, f"{name}.npz"), rows=np.array(rows, float),
             forces=np.array(forces))
    last = rows[-1] if rows else (0, 0.0, 0.0, 0.0, 0, 0.0, False, 0)
    json.dump(dict(name=name, beads=a.n, box=a.L, seed=a.seed, theta=a.theta, ds=a.ds,
                   contacts=len(pairs), pores=pores, steps=len(rows), V=last[1], S=last[2],
                   p=last[3], pieces=last[4], stop=stop, seconds=secs,
                   pyse=os.path.dirname(pyse.__file__)),
              open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
    print(f"{name}: {len(rows)} steps, S {last[2]:.3f}, {stop}, {secs:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
