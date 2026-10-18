"""The topology corpus: how often the changes of topology fail.

Drainage of bead packs (see pack.py) with ``arclength(topology=True)``, over
pack sizes, seeds, contact angles and mesh sizes, each run on one thread
(deterministic: a failure comes back the same) in its own process (a crash
ends that run only). A run drains until the liquid is down to ``FRACTION`` of
its start or it stops; every change made, every change refused, every
descent undone and every stop is logged (the ``pysurfaceevolver.topology``
logger) with its class. The measure: changes per failure (a failure: a run
that stopped short, crashed or timed out) and how far the runs drained.

usage: python bench/topology/corpus.py [--set quick|full] [--jobs 4] [--out DIR]
                                       [--only NAME ...]
       python bench/topology/corpus.py --summary DIR
       python bench/topology/corpus.py --compare DIR_A DIR_B

With PYSE_CHECK_TOPOLOGY=1 in the environment, the topology check runs
after every step of the descents: a run stops at the first structural
problem, its class naming the problem and the step.

The quick set is 4- and 16-bead packs (18 runs, under an hour at 4 jobs);
full adds mesh sizes and 64-bead packs. Results go to DIR (default
bench/topology/out/<set>): NAME.json per run, NAME.log (the records),
summary.json. Run with another package (an overlay on PYTHONPATH, another
venv's python via --python) to compare code.
"""
import argparse, json, os, re, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FRACTION = 0.5      # a run succeeds when the liquid is down to this share of its start
STEPS = 2000        # arclength steps at most (a run needs about 150 at ds -0.5)


def configs(which):
    """(name, beads, box side, seed, contact angle, h, ds, timeout in s)."""
    out = []
    for n, L, timeout in ((4, 4.2, 1800), (16, 5.0, 3600)):
        for seed in (0, 1, 2):
            for theta in (15.0, 20.0, 25.0):
                out.append((n, L, seed, theta, 0.2, -0.5, timeout))
    if which == "full":
        for seed in (0, 1, 2):
            for h in (0.17, 0.23):
                out.append((16, 5.0, seed, 20.0, h, -0.5, 3600))
        for seed in (0, 1):
            out.append((64, 10.0, seed, 20.0, 0.2, -0.5, 4*3600))
    return [(f"n{n}_s{seed}_t{theta:g}_h{h:g}_ds{ds:g}", n, L, seed, theta, h, ds, timeout)
            for n, L, seed, theta, h, ds, timeout in out]


def _shape(text):
    """A message's class: numbers and element numbers out."""
    text = re.sub(r"\(\s*-?[\d.]+(e-?\d+)?,\s*-?[\d.]+(e-?\d+)?,\s*-?[\d.]+(e-?\d+)?\)", "(#)", text)
    text = re.sub(r"-?\d+(\.\d+)?(e-?\d+)?", "#", text)
    return text.strip()


def refusal_class(message):
    """'<event>: <op>(): why[: detail]' -> '<op>(): why'."""
    found = re.search(r"(pinch_off|merge|contact)\(\): (.*)", message)
    if not found:
        return _shape(message)[:80]
    why = found.group(2).split(": ")[0]
    return _shape(f"{found.group(1)}(): {why}")[:100]


def stop_class(message):
    for key, name in (("ran away", "descent ran away"),
                      ("no stable state", "no stable state"),
                      ("three falls", "three falls without a change"),
                      ("Evolver:", "Evolver error"),
                      ("relaxed, but", "relaxed but unsound"),
                      ("pieces didn't relax", "pieces didn't relax"),
                      ("vertices changed between points", "vertices changed between points")):
        if key in message:
            return name
    return _shape(message)[:80]


def one(name, n, L, seed, theta, h, ds, out):
    """One run, in this process: NAME.json and NAME.log in out."""
    import logging, warnings
    import numpy as np
    sys.path.insert(0, HERE)
    from pack import settle, load
    import pysurfaceevolver as pyse
    from pysurfaceevolver import recipes

    log = open(os.path.join(out, f"{name}.log"), "w")
    t0 = time.time()
    state = {"step": -1}
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            text = record.getMessage()
            records.append((state["step"], text))
            log.write(f"{time.time() - t0:8.1f}s step {state['step']:4d}  {text}\n")
            log.flush()

    logger = logging.getLogger("pysurfaceevolver.topology")
    logger.addHandler(Handler())
    logger.setLevel(logging.DEBUG)
    x, _, _ = settle(n, L, L, seed=seed)
    ev, _ = load(x, L, L, theta=theta, h=h)
    ev.relax()
    v0 = float(ev.body(1).volume)
    target = FRACTION*v0
    result = dict(name=name, beads=n, box=L, seed=seed, theta=theta, h=h, ds=ds,
                  start=v0, target=target, end=v0, steps=0, reached=False, stop="",
                  stop_class="", changes=0, kinds={}, falls=0, facets=ev.counts["facets"],
                  python=sys.executable, pyse=os.path.dirname(pyse.__file__))
    kinds = Counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            for step in recipes.arclength(ev, ev.body(1), ds, STEPS, topology=True):
                state["step"] = step.index
                kinds.update(e.kind for e in step.changes)
                result.update(end=float(step.value), steps=step.index,
                              changes=sum(kinds.values()), kinds=dict(kinds),
                              falls=result["falls"] + bool(step.fell),
                              facets=ev.counts["facets"])
                log.write(f"{time.time() - t0:8.1f}s step {step.index:4d}  V {step.value:.4f} "
                          f"facets {ev.counts['facets']} angle {ev.mesh_quality().angle_min:.1f}"
                          f"{' fell' if step.fell else ''}"
                          f"{''.join(f'; made {e}' for e in step.changes)}\n")
                log.flush()
                if step.value <= target:
                    result["reached"] = True
                    break
        except Exception as error:
            result.update(stop=f"{type(error).__name__}: {error}",
                          stop_class=f"exception {type(error).__name__}")
            if type(error).__name__ == "TopologyCheckError":
                # (PYSE_CHECK_TOPOLOGY=1: the first broken invariant and the
                # kind of step that broke it)
                what = re.sub(r":.*", "", error.what)
                first = re.sub(r" \(\w+ \d+\)$", "", error.problems[0][1])
                result["stop_class"] = f"check: {first}, after {what}"
    stops = [str(w.message) for w in caught
             if issubclass(w.category, UserWarning) and "arclength()" in str(w.message)]
    if not result["reached"] and not result["stop"]:
        result["stop"] = stops[-1] if stops else f"no stop and not there after {STEPS} steps"
        result["stop_class"] = stop_class(result["stop"]) if stops else "step budget"
    result["refused"] = dict(Counter(refusal_class(t[len("change refused: "):])
                                     for _, t in records if t.startswith("change refused: ")))
    result["undone"] = dict(Counter(_shape(t[len("run undone: "):].split(" (")[0])
                                    for _, t in records if t.startswith("run undone: ")))
    result["descent_stops"] = dict(Counter(stop_class(t) for _, t in records
                                           if t.startswith("descent stopped: ")))
    for key, prefix in (("poles_moved", "pole moved"), ("patches_closed", "patch closed"),
                        ("flat_dropped", "flat facets dropped")):
        result[key] = sum(t.startswith(prefix) for _, t in records)
    result["seconds"] = time.time() - t0
    result["drained"] = (v0 - result["end"])/(v0 - target)
    log.write(f"end: {'reached' if result['reached'] else result['stop']}\n")
    log.close()
    json.dump(result, open(os.path.join(out, f"{name}.json"), "w"), indent=1)


def launch(cfg, out, python):
    name, n, L, seed, theta, h, ds, timeout = cfg
    env = dict(os.environ, OMP_NUM_THREADS="1")
    path = os.path.join(out, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
    t = time.time()
    try:
        p = subprocess.run([python, os.path.abspath(__file__), "--one", name, str(n), str(L),
                            str(seed), str(theta), str(h), str(ds), out],
                           env=env, capture_output=True, text=True, timeout=timeout)
        code, err = p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        code, err = "timeout", (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    if os.path.exists(path):
        r = json.load(open(path))
    else:
        r = dict(name=name, beads=n, box=L, seed=seed, theta=theta, h=h, ds=ds, reached=False,
                 changes=None, drained=None, seconds=time.time() - t,
                 stop=f"process: {code}: {err.strip().splitlines()[-1] if err.strip() else ''}",
                 stop_class="timeout" if code == "timeout" else
                 f"crash ({'signal ' + str(-code) if isinstance(code, int) and code < 0 else 'exit ' + str(code)})")
        # (the changes before a crash, from the log)
        logpath = os.path.join(out, f"{name}.log")
        if os.path.exists(logpath):
            text = open(logpath).read()
            r["changes"] = text.count("; made ")
            vs = re.findall(r" V ([\d.]+) facets", text)
            first = re.findall(r"step +0  V ([\d.]+)", text)
            if vs and first:
                v0 = float(first[0])
                r["drained"] = (v0 - float(vs[-1]))/(v0*(1 - FRACTION))
        json.dump(r, open(path, "w"), indent=1)
    print(f"  {name:32s} {'reached' if r['reached'] else r['stop_class']:30s} "
          f"drained {r['drained'] if r['drained'] is None else round(r['drained'], 3)} "
          f"changes {r['changes']} {r['seconds']:.0f}s", flush=True)
    return r


def summary(results):
    rows = sorted(results, key=lambda r: r["name"])
    print(f"\n{'run':32s} {'end':>8s} {'drained':>8s} {'changes':>7s} {'refused':>7s} "
          f"{'undone':>6s} {'time':>7s}  stop")
    for r in rows:
        print(f"{r['name']:32s} {r.get('end', float('nan')):8.2f} "
              f"{r['drained'] if r['drained'] is not None else float('nan'):8.3f} "
              f"{r['changes'] if r['changes'] is not None else -1:7d} "
              f"{sum(r.get('refused', {}).values()):7d} {sum(r.get('undone', {}).values()):6d} "
              f"{r['seconds']:6.0f}s  {'reached' if r['reached'] else r['stop_class']}")
    changes = sum(r["changes"] or 0 for r in rows)
    failures = sum(not r["reached"] for r in rows)
    drained = sorted(r["drained"] for r in rows if r["drained"] is not None)
    stops = Counter(r["stop_class"] for r in rows if not r["reached"])
    refused = Counter()
    for r in rows:
        refused.update(r.get("refused", {}))
    total = dict(runs=len(rows), reached=len(rows) - failures, changes=changes,
                 failures=failures,
                 changes_per_failure=changes/failures if failures else None,
                 median_drained=drained[len(drained)//2] if drained else None,
                 stops=dict(stops), refused=dict(refused.most_common()))
    print(f"\n{total['runs']} runs, {total['reached']} reached {FRACTION:g} of the start; "
          f"{changes} changes, {failures} failures: "
          f"{'no failure' if not failures else f'{changes/failures:.1f} changes per failure'}; "
          f"median drained {total['median_drained']}")
    for k, v in stops.most_common():
        print(f"  stop  {v:3d}  {k}")
    for k, v in refused.most_common(12):
        print(f"  refused {v:4d}  {k}")
    return total


def load_dir(out):
    return [json.load(open(os.path.join(out, f))) for f in sorted(os.listdir(out))
            if f.endswith(".json") and f != "summary.json"]


def compare(a, b):
    ra = {r["name"]: r for r in load_dir(a)}
    rb = {r["name"]: r for r in load_dir(b)}
    print(f"{'run':32s} {'A drained':>9s} {'B drained':>9s}  A stop / B stop")
    for name in sorted(set(ra) | set(rb)):
        x, y = ra.get(name), rb.get(name)
        d = lambda r: "" if r is None or r["drained"] is None else f"{r['drained']:.3f}"
        s = lambda r: "-" if r is None else ("reached" if r["reached"] else r["stop_class"])
        print(f"{name:32s} {d(x):>9s} {d(y):>9s}  {s(x)} / {s(y)}")
    for label, d in (("A", a), ("B", b)):
        print(f"\n{label}: {d}")
        summary(load_dir(d))


def main(argv):
    if argv[:1] == ["--one"]:
        name, n, L, seed, theta, h, ds, out = argv[1:9]
        one(name, int(n), float(L), int(seed), float(theta), float(h), float(ds), out)
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="quick", choices=("quick", "full"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--summary")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args(argv)
    if args.summary:
        summary(load_dir(args.summary))
        return
    if args.compare:
        compare(*args.compare)
        return
    out = os.path.abspath(args.out or os.path.join(HERE, "out", args.set))
    os.makedirs(out, exist_ok=True)
    todo = [c for c in configs(args.set) if not args.only or c[0] in args.only]
    # the longest first, so the pool ends together
    todo.sort(key=lambda c: -c[1])
    print(f"{len(todo)} runs, {args.jobs} at a time, into {out}", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda c: launch(c, out, args.python), todo))
    total = summary(results)
    json.dump(total, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
