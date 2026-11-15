"""Replay Hele-Shaw frames (checkpoints of tools/hele_shaw_figure.py runs) and
compare builds.

    hele_frames.py run MANIFEST OUT [--jobs N] [--seconds S] [LABEL=PYSE ...]
    hele_frames.py report OUT

MANIFEST is a JSON list of frames, each {"name": NAME, "checkpoint": PATH (a
.ckpt the tool's --states wrote), "cell": "pair11", "centres": [5.5],
"flags": "--rounding 0.25 --ds 12 --halvings 2 --width 14 --fine 1.5",
"steps": N, "why": TEXT}. Each frame is resumed with the tool's --resume and
--steps N, one thread (a run is the same every time at a fixed thread
count), at most S seconds (default 3600). LABEL=PYSE: a build to run, PYSE a
directory holding a pysurfaceevolver package (put first on PYTHONPATH;
"installed" for the environment's own). Results go to OUT/LABEL/NAME/
(the tool's log and rows). ``report`` reads the logs: per frame and build,
the steps made, the seconds, the descents' work (passes over the surface,
summed: deterministic, unlike the time), back-ups, runs judged crossed,
volume approaches, steps ended short, and the last volume and pressure.

The frames are where the finer-mesh cells stalled or failed (docs/
hele_shaw_defects.md, "Stalls on the finer mesh"): a replay that takes much
more work than the reference, or ends short, is a regression there."""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "..", "tools", "hele_shaw_figure.py")


def _replay(frame: dict, out: str, pyse: str, seconds: float) -> dict:
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    if pyse != "installed":
        env["PYTHONPATH"] = os.pathsep.join([pyse] + [p for p in [env.get("PYTHONPATH")] if p])
    cmd = ([sys.executable, TOOL, "run", frame["cell"]] + [str(c) for c in frame["centres"]]
           + frame["flags"].split() + ["--resume", frame["checkpoint"],
                                       "--steps", str(frame["steps"]), "--out", out])
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=seconds)
        how = "done" if p.returncode == 0 else f"failed ({p.returncode})"
        tail = p.stderr[-1500:] if p.returncode else ""
    except subprocess.TimeoutExpired:
        how, tail = f"stopped at {seconds:.0f} s", ""
    return dict(how=how, wall=time.perf_counter() - t, tail=tail)


def _read(log: str) -> dict:
    """What the tool's log says of a replay."""
    r = dict(steps=0, work=0.0, backups=0, crossed=0, approached=0, approach_steps=0,
             not_approached=0, ended_short=0, last="")
    if not os.path.exists(log):
        return r
    for line in open(log):
        text = re.sub(r"^\s*[\d.]+s ", "", line.rstrip("\n"))
        if re.match(r"(imbibition|drainage) V ", text):
            r["steps"] += 1
            r["last"] = " ".join(text.split()[1:5])
        elif m := re.search(r"descent's work: ([\d.e+]+)", text):
            r["work"] += float(m[1])
        elif "branch backed up" in text:
            r["backups"] += 1
        elif "implausible run" in text:
            r["crossed"] += 1
        elif m := re.search(r"volumes approached in (\d+) steps", text):
            r["approached"] += 1
            r["approach_steps"] += int(m[1])
        elif "volumes not approached" in text:
            r["not_approached"] += 1
        elif "ended short" in text:
            r["ended_short"] += 1
    return r


def run(args) -> None:
    frames = json.load(open(args.manifest))
    builds = [b.split("=", 1) for b in (args.builds or ["installed=installed"])]
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for label, pyse in builds:
            for f in frames:
                out = os.path.join(args.out, label, f["name"])
                jobs.append((label, f, out, pool.submit(_replay, f, out, pyse, args.seconds)))
        for label, f, out, job in jobs:
            r = dict(job.result(), frame=f["name"], build=label)
            json.dump(r, open(os.path.join(out, "replay.json"), "w"), indent=1)
            print(f"{label:10s} {f['name']:24s} {r['how']} in {r['wall']:.0f} s", flush=True)
            if r["tail"]:
                print("  " + r["tail"].replace("\n", "\n  "))


def report(args) -> None:
    rows = []
    for label in sorted(os.listdir(args.out)):
        folder = os.path.join(args.out, label)
        for name in sorted(os.listdir(folder)):
            done = os.path.join(folder, name, "replay.json")
            if not os.path.exists(done):
                continue
            r = json.load(open(done))
            logs = [p for p in os.listdir(os.path.join(folder, name)) if p.endswith(".log")]
            r.update(_read(os.path.join(folder, name, logs[0])) if logs else _read(""))
            rows.append(r)
    print(f"{'frame':24s} {'build':10s} {'steps':>5s} {'s':>6s} {'work':>9s} {'back':>4s} "
          f"{'cross':>5s} {'appr':>9s} {'short':>5s}  how / last")
    for r in sorted(rows, key=lambda r: (r["frame"], r["build"])):
        print(f"{r['frame']:24s} {r['build']:10s} {r['steps']:5d} {r['wall']:6.0f} "
              f"{r['work']:9.3g} {r['backups']:4d} {r['crossed']:5d} "
              f"{r['approached']:3d}/{r['approach_steps']:<4d}{'!' if r['not_approached'] else ' '}"
              f"{r['ended_short']:5d}  {r['how']}; {r['last']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="part", required=True)
    a = sub.add_parser("run")
    a.add_argument("manifest")
    a.add_argument("out")
    a.add_argument("builds", nargs="*", help="LABEL=PYSE (default: installed=installed)")
    a.add_argument("--jobs", type=int, default=1)
    a.add_argument("--seconds", type=float, default=3600.0)
    b = sub.add_parser("report")
    b.add_argument("out")
    args = ap.parse_args()
    run(args) if args.part == "run" else report(args)


if __name__ == "__main__":
    main()
