"""Wetting a bead pack: pendular bridges at the contacts growing together.

Beads settled in a cell periodic in x and y (pack.settle_periodic, lifted
LIFT off the floor), a bridge at every contact (pack.bridges_periodic), all
the liquid one body at one pressure (arclength's trap=False: bridges
exchange liquid as through films or the vapour). Liquid is added by
arclength(topology=True) until it fills SMAX of the pore space
(pack.pore_volume, between the lowest and highest centres), the budget runs
out or the liquid reaches the floor, its edges held near REMESH (the
starting bridges' 48 facets are 6-10% off a bridge's pressure; at 0.06,
1-3%). Per step: the volume, the saturation,
the pressure, the surface's pieces and each bead's force
(Evolver.wall_force), in OUT/NAME.npz; the log in OUT/NAME.log, a summary in
OUT/NAME.json, and with --frames each step's liquid (facet corners, float16)
in OUT/frames/NAME/ for a film. With --checkpoints the last KEEP steps'
states (corpus.save_state, the run's own state with them) in
OUT/checkpoints/NAME/, each one to go on from exactly (--resume, with the
run's --pack and its other options; a new OUT).

usage: python bench/topology/wet.py N SEED THETA OUT [--L 6] [--ds 0.05]
                                    [--smax 0.35] [--budget 3e6] [--remesh 0.06]
                                    [--frames] [--pack NAME_beads.npz]
                                    [--checkpoints] [--resume OUT/checkpoints/NAME/stepNNNN]
"""
import argparse, json, logging, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pack import bridges_periodic, pore_volume, settle_periodic  # noqa: E402
from corpus import KEEP, restore_state, run_state, save_state  # noqa: E402

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
    ap.add_argument("--remesh", type=float, default=0.06, help="arclength's edge length")
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--pack", help="the beads from an earlier run's NAME_beads.npz (packs "
                    "settle differently from machine to machine)")
    ap.add_argument("--checkpoints", action="store_true",
                    help="the last KEEP steps' states kept, to resume exactly from")
    ap.add_argument("--resume", help="a checkpoint (its path without the extension; with "
                    "the run's --pack)")
    a = ap.parse_args(argv)
    import pysurfaceevolver as pyse
    from pysurfaceevolver import _core, recipes
    name = f"w{a.n}_s{a.seed}_t{a.theta:g}"
    os.makedirs(a.out, exist_ok=True)
    frames = os.path.join(a.out, "frames", name) if a.frames else None
    if frames:
        os.makedirs(frames, exist_ok=True)
    checkpoints = os.path.join(a.out, "checkpoints", name) if a.checkpoints else None
    if checkpoints:
        os.makedirs(checkpoints, exist_ok=True)
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
    if a.resume and not a.pack:
        raise SystemExit("--resume needs the run's --pack")
    if a.pack:
        x = np.load(a.pack)["centres"]
    else:
        x, gap, _ = settle_periodic(a.n, a.L, seed=a.seed)
        x[:, 2] += LIFT
    run = None
    if a.resume:
        ev, _, first = restore_state(a.resume)
        run = run_state(a.resume)
        pairs = [tuple(int(i) for i in p) for p in np.load(a.pack)["pairs"]]
        say(f"resumed from {a.resume} (step {first}"
            f"{'' if run else ': no run state there, the run afresh from the surface'})")
    else:
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
                                   remesh=a.remesh, budget=a.budget,
                                   resumable=bool(a.checkpoints or a.resume),
                                   resume=run["state"] if run else None):
            at["step"] = int(s.index)
            if checkpoints:
                own = s.state() if s.state else None
                save_state(ev, os.path.join(checkpoints, f"step{s.index:04d}"), 0.0, s.index,
                           run=None if own is None else {"state": own})
                stale = os.path.join(checkpoints, f"step{s.index - KEEP:04d}")
                for ext in (".fe", ".pkl"):
                    if os.path.exists(stale + ext):
                        os.remove(stale + ext)
            F = np.full((len(names), 3), np.nan)
            for i, k in enumerate(names):
                try:
                    F[i] = ev.wall_force(k)
                except ValueError as e:     # (recorded as unknown, the run goes on)
                    say(f"no force on {k}: {e}")
            count, share = pieces(ev)
            S = s.value/pores
            rows.append((s.index, s.value, S, s.pressures[0], count, share, bool(s.fell),
                         _core.work()))
            forces.append(F)
            say(f"V {s.value:.4f} S {S:.4f} p {s.pressures[0]:.4f} pieces {count} "
                f"largest {share:.2f} |F| {np.nanmedian(np.linalg.norm(F, axis=1)):.3f} "
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
    except Exception as e:      # (a run's crash ends that run, recorded)
        stop = f"{type(e).__name__}: {e}"
        say(f"stopped: {stop}")
    if not stop:
        stop = "the run ended (steps, budget, or arclength gave up; see the log)"
    secs = time.time() - t0
    say(f"end: {stop}")
    np.savez(os.path.join(a.out, f"{name}.npz"), rows=np.array(rows, float),
             forces=np.array(forces))
    last = rows[-1] if rows else (0, 0.0, 0.0, 0.0, 0, 0.0, False, 0)
    json.dump(dict(name=name, beads=a.n, box=a.L, seed=a.seed, theta=a.theta, ds=a.ds,
                   remesh=a.remesh, contacts=len(pairs), pores=pores, steps=len(rows),
                   V=last[1], S=last[2],
                   p=last[3], pieces=last[4], stop=stop, seconds=secs,
                   pyse=os.path.dirname(pyse.__file__)),
              open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
    print(f"{name}: {len(rows)} steps, S {last[2]:.3f}, {stop}, {secs:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
