"""Replay a set of saved frames for a fixed amount of work, and compare builds.

    replay_suite.py run MANIFEST OUT [--passes P] [--jobs N] [LABEL=PYSE ...]
    replay_suite.py report OUT

MANIFEST is a JSON list of frames, each {"name": NAME, "run": "c64_s1_t20",
"beads": 64, "box": 6.0, "seed": 1, "theta": 20, "frame": PATH (without
.fe/.pkl), "why": TEXT}. Each frame is resumed (corpus.one's resume=) with the
budget its run had spent there plus P passes over the surface, so a replay
stops after the same work however loaded the machine is, and (runs being
deterministic at a fixed thread count) a build replays the same way every
time. LABEL=PYSE: a build to run, PYSE a directory holding a
pysurfaceevolver package (put first on PYTHONPATH; "installed" for the
environment's own). Results go to OUT/LABEL/NAME/; report prints, per frame
and build, the volume reached (lower: further drained), the steps and changes
made, the passes and seconds spent and how the replay ended."""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

_ONE = """
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, {here!r})
import corpus
corpus.one({run!r}, {beads}, {box}, {seed}, {theta}, 0.2, -0.72, {out!r}, "breakthrough",
           flux_degree=8, resume={frame!r}, budget={budget!r})
"""


def _spent(frame: str) -> float:
    """The passes the frame's run had spent (its saved run state's)."""
    with open(frame + ".pkl", "rb") as f:
        run = pickle.load(f).get("run")
    if not run or "state" not in run:
        raise SystemExit(f"{frame}: no run state saved with it (corpus --frames)")
    return float(run["state"]["spent"])


def run(manifest: str, out: str, passes: float, jobs: int, builds: list) -> None:
    frames = json.load(open(manifest))
    todo = []
    for label, pyse in builds:
        for fr in frames:
            dest = os.path.join(out, label, fr["name"])
            if os.path.exists(os.path.join(dest, "done")):
                continue
            os.makedirs(dest, exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS="1")
            if pyse != "installed":
                env["PYTHONPATH"] = os.path.abspath(pyse) + os.pathsep + env.get("PYTHONPATH", "")
            code = _ONE.format(here=HERE, run=fr["run"], beads=fr.get("beads", 64),
                               box=fr.get("box", 6.0), seed=fr["seed"], theta=fr["theta"],
                               out=dest, frame=os.path.abspath(fr["frame"]),
                               budget=_spent(fr["frame"]) + passes)
            todo.append((label, fr["name"], dest, env, code))
    running: list = []
    while todo or running:
        while todo and len(running) < jobs:
            label, name, dest, env, code = todo.pop(0)
            log = open(os.path.join(dest, "stdout"), "w")
            p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
            running.append((p, dest, log, time.time()))
            print(f"started {label}/{name}", flush=True)
        for item in list(running):
            p, dest, log, t0 = item
            if p.poll() is not None:
                log.close()
                with open(os.path.join(dest, "done"), "w") as f:
                    f.write(f"{p.returncode} {time.time() - t0:.1f}\n")
                running.remove(item)
                print(f"done {dest} ({p.returncode})", flush=True)
        time.sleep(2)


def _result(dest: str) -> dict:
    for name in sorted(os.listdir(dest)):
        if name.endswith(".json"):
            return json.load(open(os.path.join(dest, name)))
    return {}


def report(out: str) -> None:
    labels = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
    names = sorted({n for lab in labels for n in os.listdir(os.path.join(out, lab))})
    print(f"{'frame':18s} {'build':10s} {'V end':>9s} {'steps':>6s} {'changes':>7s} "
          f"{'passes':>7s} {'seconds':>8s}  ended")
    for name in names:
        for lab in labels:
            dest = os.path.join(out, lab, name)
            if not os.path.isdir(dest):
                continue
            r = _result(dest)
            if not r:
                print(f"{name:18s} {lab:10s} (no result)")
                continue
            stop = "reached" if r.get("reached") else (r.get("stop") or "")
            stop = "budget" if "budget spent" in stop else stop
            print(f"{name:18s} {lab:10s} {r.get('end', float('nan')):9.3f} {r.get('steps', 0):6d} "
                  f"{r.get('changes', 0):7d} {r.get('passes', 0):7d} {r.get('seconds', 0):8.0f}  "
                  f"{stop[:60]}")
        print()
    # (each build's totals over the frames all builds have: the volume reached
    # summed, the passes and seconds)
    done = [n for n in names if all(_result(os.path.join(out, lab, n)) for lab in labels
                                    if os.path.isdir(os.path.join(out, lab, n)))]
    for lab in labels:
        rs = [_result(os.path.join(out, lab, n)) for n in done
              if os.path.isdir(os.path.join(out, lab, n))]
        print(f"{lab:10s} frames {len(rs)}  V end summed {sum(r.get('end', 0) for r in rs):9.3f}  "
              f"reached {sum(bool(r.get('reached')) for r in rs)}  passes {sum(r.get('passes', 0) for r in rs)}  "
              f"seconds {sum(r.get('seconds', 0) for r in rs):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("manifest")
    r.add_argument("out")
    r.add_argument("builds", nargs="*", default=["installed=installed"])
    r.add_argument("--passes", type=float, default=30000.0)
    r.add_argument("--jobs", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.manifest, a.out, a.passes, a.jobs, [tuple(b.split("=", 1)) for b in a.builds])
    else:
        report(a.out)


if __name__ == "__main__":
    main()
