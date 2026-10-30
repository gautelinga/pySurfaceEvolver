"""The topology corpus: how often the changes of topology fail.

Drainage of bead packs (see pack.py) with ``arclength(topology=True)``, over
pack sizes, seeds, contact angles and mesh sizes, each run on one thread
(deterministic: a failure comes back the same) in its own process (a crash
ends that run only). A run drains until the liquid is down to ``FRACTION`` of
its start or it stops; every change made, every change refused, every
descent undone and every stop is logged (the ``pysurfaceevolver.topology``
logger) with its class. The measure: changes per failure (a failure: a run
that stopped short, crashed or timed out) and how far the runs drained.

usage: python bench/topology/corpus.py [--set quick|small|wide|wide-half|full]
                                       [--jobs 4] [--threads 1] [--out DIR]
                                       [--only NAME ...] [--h H] [--ds DS]
                                       [--gauge flux|pole] [--cpus fast|all|LIST]
                                       [--flux-degree P] [--ds-area]
                                       [--wet-area flux|lines]
       python bench/topology/corpus.py --summary DIR
       python bench/topology/corpus.py --compare DIR_A DIR_B

With PYSE_CHECK_TOPOLOGY=1 in the environment, the topology check runs
after every step of the descents: a run stops at the first structural
problem, its class naming the problem and the step.

The quick set is 4- and 16-bead packs (18 runs, under an hour at 4 jobs);
wide is 16-bead packs over six seeds, drained to breakthrough (the floor a
wall at 90 degrees, the run done when the gas reaches it: a "solid" event
on it with liquid between, or a dry patch there; wide-half: the
same runs to half the liquid); small is the same with 4-bead packs (fast);
full adds mesh sizes and 64-bead packs to quick; --h and --ds run a set at
another edge length or arclength step. Results go to DIR (default
bench/topology/out/<set>): NAME.json per run, NAME.log (the records),
summary.json. Run with another package (an overlay on PYTHONPATH, another
venv's python via --python) to compare code.
"""
import argparse, json, os, queue, re, shutil, subprocess, sys, time
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FRACTION = 0.5      # a run succeeds when the liquid is down to this share of its start
STEPS = 2000        # arclength steps at most (a run needs about 150 at ds -0.5)


def configs(which, h=None, ds=None, ds_area=False):
    """(name, beads, box side, seed, contact angle, h, ds, timeout in s,
    goal): goal "half" (the liquid down to FRACTION of its start) or
    "breakthrough" (the floor a wall, the gas reaching it). h, ds: the
    set's runs at that edge length, or arclength step, instead; ds_area:
    the step scaled with the cell's area (as a 5 x 5 cell's), the same mean
    move of the meniscus a step whatever the pack's size."""
    out = []
    for n, L, timeout in ((4, 4.2, 1800), (16, 5.0, 3600)):
        for seed in (0, 1, 2):
            for theta in (15.0, 20.0, 25.0):
                out.append((n, L, seed, theta, 0.2, -0.5, timeout))
    goal = "half"
    if which == "wide":
        # (16 beads only, six seeds, drained to breakthrough: the gas
        # reaching the floor)
        out = [(16, 5.0, seed, theta, 0.2, -0.5, 3*3600) for seed in range(6)
               for theta in (15.0, 20.0, 25.0)]
        goal = "breakthrough"
    if which == "small":
        # (4 beads, six seeds, to breakthrough: the wide set's mechanisms,
        # minutes a run)
        out = [(4, 4.2, seed, theta, 0.2, -0.5, 1800) for seed in range(6)
               for theta in (15.0, 20.0, 25.0)]
        goal = "breakthrough"
    if which == "wide-half":
        # (the wide set as it was before breakthrough was the goal)
        out = [(16, 5.0, seed, theta, 0.2, -0.5, 3600) for seed in range(6)
               for theta in (15.0, 20.0, 25.0)]
    if which in ("periodic", "periodic-16", "periodic-32", "periodic-64"):
        # (columns: a cell 3 bead diameters square, periodic in x and y, the
        # pack deeper with more beads, 16 to 64: 1.9 to 5.8 diameters; the
        # floor the outlet, to breakthrough. Until 2026-10-27 the cell grew
        # instead, 0.64 beads a unit area, and the packs stayed 2-3 diameters
        # deep: a 64-bead pack's front had nowhere to go)
        sizes = (16, 32, 64) if which == "periodic" else (int(which.split("-")[1]),)
        out = [(n, COLUMN, seed, theta, 0.2, -0.5, {16: 3600, 32: 2*3600, 64: 4*3600}[n])
               for n in sizes for seed in range(3) for theta in (15.0, 20.0, 25.0)]
        goal = "breakthrough"
    if which == "full":
        for seed in (0, 1, 2):
            for h in (0.17, 0.23):
                out.append((16, 5.0, seed, 20.0, h, -0.5, 3600))
        for seed in (0, 1):
            out.append((64, 10.0, seed, 20.0, 0.2, -0.5, 4*3600))
    if h is not None:
        out = [(n, L, seed, theta, h, d, timeout) for n, L, seed, theta, _, d, timeout in out]
    if ds is not None:
        out = [(n, L, seed, theta, h_, ds, timeout) for n, L, seed, theta, h_, _, timeout in out]
    if ds_area:
        out = [(n, L, seed, theta, h_, round(d*(L/5.0)**2, 3), timeout)
               for n, L, seed, theta, h_, d, timeout in out]
    if h is not None or ds is not None:
        # (the time limit grows with the work: facets as 1/h^2, steps as 1/ds)
        out = [(n, L, seed, theta, h_, d, int(t*(0.2/h_)**2*abs(0.5/d)))
               for n, L, seed, theta, h_, d, t in out]
    tag = "_bt" if goal == "breakthrough" else ""
    pre = "c" if which.startswith("periodic") else "n"
    return [(f"{pre}{n}_s{seed}_t{theta:g}_h{h_:g}_ds{d:g}{tag}", n, L, seed, theta, h_, d, timeout,
             goal) for n, L, seed, theta, h_, d, timeout in out]


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
                      ("budget spent", "budget spent"),
                      ("no liquid reaches the outlet", "no liquid at the outlet"),
                      ("no stable state", "no stable state"),
                      ("three falls", "three falls without a change"),
                      ("Evolver:", "Evolver error"),
                      ("relaxed, but", "relaxed but unsound"),
                      ("pieces didn't relax", "pieces didn't relax"),
                      ("vertices changed between points", "vertices changed between points")):
        if key in message:
            return name
    return _shape(message)[:80]


STALL = 300.0   # seconds without a log line before a run writes its stacks to its log


class _Watched:
    """A run's log, watched: no line for `stall` seconds and every thread's
    Python stack goes into it (again each `stall` while it lasts), so a run
    stuck in C shows the call it is in with no debugger attached (a 64-bead
    run hung an hour in one gradient step, found only by attaching to it).
    A thread of its own, holding the GIL as it reads the stacks (pySE lets
    it go in its commands; faulthandler's watchdog reads them without it and
    crashed a run). 0: off."""

    def __init__(self, f, stall):
        import threading
        self.f, self.stall = f, stall
        self.last = time.time()
        self.done = threading.Event()
        if stall > 0:
            threading.Thread(target=self._watch, daemon=True, name="stall watchdog").start()

    def _watch(self):
        import traceback
        while not self.done.wait(min(self.stall, max(0.01, self.last + self.stall - time.time()))):
            quiet = time.time() - self.last
            if quiet < self.stall:
                continue
            names = {t.ident: t.name for t in __import__("threading").enumerate()}
            text = [f"no log line for {quiet:.0f} s; the threads' stacks:\n"]
            for ident, frame in sys._current_frames().items():
                if names.get(ident) == "stall watchdog":
                    continue
                text.append(f"  thread {names.get(ident, ident)}:\n")
                text += ["    " + line for line in "".join(traceback.format_stack(frame)).splitlines(True)]
            self.f.write("".join(text))
            self.f.flush()
            self.last = time.time()

    def write(self, text):
        return self.f.write(text)

    def flush(self):
        self.f.flush()
        self.last = time.time()

    def close(self):
        self.done.set()
        self.f.close()


def one(name, n, L, seed, theta, h, ds, out, goal="half", checkpoints=None, resume=None,
        gauge=None, flux_degree=None, frames=None):
    """One run, in this process: NAME.json and NAME.log in out. checkpoints:
    a directory each step's state goes to (stepNNNN.fe and .pkl, the last
    KEEP kept); resume: such a state (its path without the extension), the
    run going on from it (the steps numbered on) instead of from the start;
    where an earlier attempt left NAME.part, its curve, counts and time are
    carried on and the log appended to (launch(): a crashed run resumed).
    frames: every this many steps the state kept in OUT/frames/NAME (for a
    film of the run: bench/topology/film.py)."""
    import logging, warnings
    import numpy as np
    sys.path.insert(0, HERE)
    from pack import settle, load, settle_periodic, load_periodic
    import pysurfaceevolver as pyse
    from pysurfaceevolver import recipes
    if flux_degree is not None:
        pyse.flux_degree = int(flux_degree)
    if os.environ.get("PYSE_WET_AREA"):
        pyse.wet_area = os.environ["PYSE_WET_AREA"]     # (--wet-area)
    if os.environ.get("PYSE_LOCAL_DESCENTS"):
        recipes.LOCAL_DESCENTS = True       # (falls by local descents: being validated)

    part = os.path.join(out, f"{name}.part")
    prior = json.load(open(part)) if resume and os.path.exists(part) else None
    log = _Watched(open(os.path.join(out, f"{name}.log"), "a" if resume else "w"),
                   float(os.environ.get("PYSE_CORPUS_STALL", STALL)))
    t0 = time.time() - (prior.get("seconds", 0.0) if prior else 0.0)
    if resume:
        log.write(f"{time.time() - t0:8.1f}s resumed from {resume}\n")
    state = {"step": -1}
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            text = record.getMessage()
            records.append((state["step"], text))
            # (the gas at the floor in a descent: its event made or refused)
            if goal == "breakthrough" and "constraint bottom" in text \
                    and "the liquid between thinning" in text and not state.get("floor"):
                state["floor"] = text
            log.write(f"{time.time() - t0:8.1f}s step {state['step']:4d}  {text}\n")
            log.flush()

    if checkpoints:
        os.makedirs(checkpoints, exist_ok=True)
    logger = logging.getLogger("pysurfaceevolver.topology")
    logger.addHandler(Handler())
    logger.setLevel(logging.DEBUG)
    first = 0
    if resume:
        ev, v0, first = restore_state(resume)
    else:
        # (the beads in the flux gauge where the run goes to breakthrough:
        # the pole gauge's volume bookkeeping lost whole beads' volumes where
        # lines crossed poles; the older "half" sets as they were)
        if gauge is None:
            gauge = "flux" if goal == "breakthrough" else "pole"
        if name[0] in "pc":
            # (a cell periodic in x and y: no side walls)
            x, _, _ = settle_periodic(n, L, seed=seed)
            ev, _ = load_periodic(x, L, theta=theta, h=h)
        else:
            x, _, _ = settle(n, L, L, seed=seed)
            ev, _ = load(x, L, L, theta=theta, h=h, floor=goal == "breakthrough", gauge=gauge)
        ev.relax()
        v0 = float(ev.body(1).volume)
    target = FRACTION*v0
    result = dict(name=name, beads=n, box=L, seed=seed, theta=theta, h=h, ds=ds, goal=goal,
                  start=v0, target=target, end=v0, steps=0, reached=False, stop="",
                  stop_class="", changes=0, kinds={}, falls=0, facets=ev.counts["facets"],
                  python=sys.executable, pyse=os.path.dirname(pyse.__file__),
                  flux_degree=pyse.flux_degree, wet_area=pyse.wet_area, threads=pyse.threads(),
                  cpus=sorted(os.sched_getaffinity(0)))
    # (per step: the kinds of the changes made, whether it fell; a step
    # the run backed up past is no longer the run's)
    made = {}
    # (per step the curve: the draining body's volume and pressure, the
    # liquid trapped, the energy; a step the run backed up past dropped)
    curve = {}
    # (the run's work, in passes over the surface, as arclength's budget
    # counts it: _core.work() between steps over the facets after each)
    from pysurfaceevolver import _core
    work = {"last": _core.work(), "passes": 0.0}
    # (resumed: the attempt before it, to its checkpoint, carried on)
    before = {"kinds": Counter(), "falls": 0}
    if prior:
        curve.update({int(k): tuple(rest) for k, *rest in prior.get("curve", []) if k < first})
        work["passes"] = float(prior.get("passes", 0.0))
        before.update(kinds=Counter(prior.get("kinds", {})), falls=int(prior.get("falls", 0)))
        result["resumed"] = prior.get("resumed", []) + [first]
    # (the liquid kept: at every step what the bodies hold is their targets,
    # the draining one's and the trapped ones'; the changes only move
    # targets between bodies, _clusters.reconcile logging any that doesn't)
    outlet = "bottom" if goal == "breakthrough" else None
    liquid = {"error": 0.0, "trapped": 0.0, "bodies": 0, "held": 0.0}

    def check_liquid(step):
        b = ev.bodies()
        fixed = np.asarray(b.fixed, bool)
        books = float(np.sum(np.asarray(b.volume)[fixed]) - np.sum(np.asarray(b.target)[fixed]))
        liquid["error"] = max(liquid["error"], abs(books))
        others = fixed & (np.asarray(b.ids) != 1)
        liquid["trapped"] = float(step.trapped)
        liquid["bodies"] = int(others.sum())
        if others.any():
            liquid["held"] = max(liquid["held"], float(np.max(np.abs(
                np.asarray(b.volume)[others] - np.asarray(b.target)[others]))))
        result.update(trapped=liquid["trapped"], trapped_bodies=liquid["bodies"],
                      conservation=liquid["error"], trapped_held=liquid["held"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            done, falls_at_start = False, 0
            while not done:
                taken = 0
                for step in recipes.arclength(ev, ev.body(1), ds, STEPS - first, topology=True,
                                              outlet=outlet):
                    taken += 1
                    check_liquid(step)
                    work["passes"] += (_core.work() - work["last"])/max(1, ev.counts["facets"])
                    result["passes"] = round(work["passes"])
                    step.index += first
                    state["step"] = step.index
                    if checkpoints:
                        save_state(ev, os.path.join(checkpoints, f"step{step.index:04d}"), v0,
                                   step.index)
                        stale = os.path.join(checkpoints, f"step{step.index - KEEP:04d}")
                        for ext in (".fe", ".pkl"):
                            if os.path.exists(stale + ext):
                                os.remove(stale + ext)
                    if frames and step.index % int(frames) == 0:
                        film = os.path.join(out, "frames", name)
                        os.makedirs(film, exist_ok=True)
                        save_state(ev, os.path.join(film, f"step{step.index:04d}"), v0, step.index)
                    for k in [k for k in made if k >= step.index]:
                        del made[k]
                    for k in [k for k in curve if k >= step.index]:
                        del curve[k]
                    curve[step.index] = (float(step.value), float(ev.body(1).pressure),
                                         float(step.trapped), float(step.energy), bool(step.fell),
                                         len(step.changes))
                    result["curve"] = [[k, *curve[k]] for k in sorted(curve)]
                    made[step.index] = ([e.kind for e in step.changes], step.fell)
                    kinds = before["kinds"] + Counter(kind for m, _ in made.values() for kind in m)
                    result.update(end=float(step.value), steps=step.index,
                                  changes=sum(kinds.values()), kinds=dict(kinds),
                                  falls=before["falls"] + sum(fell for _, fell in made.values()),
                                  facets=ev.counts["facets"])
                    # (the run so far, for a run stopped from outside: a timeout,
                    # or a crash, resumed from this step's checkpoint)
                    result["seconds"] = time.time() - t0
                    json.dump(result, open(part, "w"))
                    if not resume and str(step.index) == os.environ.get("PYSE_CORPUS_CRASH_AT"):
                        os.abort()      # (tests of launch()'s resuming)
                    log.write(f"{time.time() - t0:8.1f}s step {step.index:4d}  V {step.value:.4f} "
                              f"trapped {step.trapped:.4f} "
                              f"facets {ev.counts['facets']} angle {ev.mesh_quality().angle_min:.1f}"
                              f"{' fell' if step.fell else ''}"
                              f"{''.join(f'; made {e}' for e in step.changes)}\n")
                    log.flush()
                    work["last"] = _core.work()     # (not the checkpoints' and the log's)
                    if goal == "half" and step.value <= target:
                        result["reached"] = True
                        done = True
                        break
                    # (breakthrough: the gas at the floor, its film of liquid
                    # thinner than a change takes, or a dry patch made there;
                    # at the box's vertical edges the patch can't be made, a
                    # vertex would be on three walls)
                    if goal == "breakthrough" and (state.get("floor") or ev.on_constraint("bottom").any()
                                                   or any(e.kind == "solid"
                                                          and str(e.constraint) == "bottom"
                                                          and e.between == "body"
                                                          for e in step.events)):
                        result["reached"] = True
                        result["breakthrough"] = float(step.value)
                        done = True
                        break
                if done or not resume or taken > 1 or falls_at_start >= 3:
                    break
                # (resumed at a fold: a checkpoint is a branch's last point, and
                # from it the branch takes no step, as the run itself didn't; the
                # run went down from there, so the resumed one falls as it did)
                falls_at_start += 1
                from pysurfaceevolver._analysis import EVENT_FACTOR
                from pysurfaceevolver._surgery import descend
                log.write(f"{time.time() - t0:8.1f}s step {state['step']:4d}  resumed at a fold: "
                          "falling from it\n")
                log.flush()
                fell, _, why = descend(ev, EVENT_FACTOR, 1e-8, falling=True, keep=1)
                if fell is None:
                    raise RuntimeError(f"the fall from where the run resumed stopped: {why}")
                first = state["step"] + 1
        except Exception as error:
            import traceback
            log.write(traceback.format_exc())
            result.update(stop=f"{type(error).__name__}: {error}",
                          stop_class=f"exception {type(error).__name__}")
            if type(error).__name__ == "TopologyCheckError":
                # (PYSE_CHECK_TOPOLOGY=1: the first broken invariant and the
                # kind of step that broke it)
                what = re.sub(r":.*", "", error.what)
                first = re.sub(r" \(\w+ \d+\)$", "", error.problems[0][1])
                result["stop_class"] = f"check: {first}, after {what}"
    if goal == "breakthrough" and state.get("floor") and not result["reached"]:
        # (reached in the descent the run then stopped in)
        result["reached"] = True
        result["breakthrough"] = result["end"]
        result["stop"], result["stop_class"] = "", ""
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
    # (where the run would have stopped, backed up and gone on another way)
    result["backed_up"] = dict(Counter(stop_class(t) for _, t in records
                                       if t.startswith("branch backed up")))
    result["splits"] = sum(t.startswith("clusters: bodies") for _, t in records)
    result["targets_changed"] = sum("the targets' total changed" in t for _, t in records)
    result["mixed"] = sum("face both liquid and gas" in t for _, t in records)
    for key, prefix in (("poles_moved", "pole moved"), ("patches_closed", "patch closed"),
                        ("flat_dropped", "flat facets dropped")):
        result[key] = sum(t.startswith(prefix) for _, t in records)
    result["seconds"] = time.time() - t0
    # (breakthrough: the share of the liquid drained; half: how far toward it)
    left = result["end"] + result.get("trapped", 0.0)       # (trapped liquid stays)
    result["drained"] = ((v0 - left)/v0 if goal == "breakthrough"
                         else (v0 - left)/(v0 - target))
    log.write(f"end: {'reached' if result['reached'] else result['stop']}\n")
    log.close()
    json.dump(result, open(os.path.join(out, f"{name}.json"), "w"), indent=1)
    if os.path.exists(os.path.join(out, f"{name}.part")):
        os.remove(os.path.join(out, f"{name}.part"))


COLUMN = 6.0    # the periodic sets' cell side: 3 bead diameters
KEEP = 20       # checkpoints: the last this many steps' states kept


def save_state(ev, path, v0, index):
    """The surface (path.fe) and what pySE remembers of it (path.pkl), to
    go on from (restore_state)."""
    import pickle
    from pysurfaceevolver._engine import surface
    ev.dump(path + ".fe")
    with open(path + ".pkl", "wb") as f:
        pickle.dump({"walls": dict(surface.walls), "retired": set(surface.retired),
                     "energy_shift": list(surface.energy_shift),
                     "edge_scale": surface.edge_scale, "sizes": dict(surface.sizes),
                     "start": v0, "step": index}, f)


def restore_state(path):
    """A state save_state wrote: the surface loaded, pySE's memory of it put
    back. (The surface, the volume it started at, the step it was: a run
    resumed from it numbers its first point, this one again, so.)"""
    import pickle
    import pysurfaceevolver as pyse
    from pysurfaceevolver._engine import surface
    ev = pyse.Evolver()
    ev.load(path + ".fe")
    with open(path + ".pkl", "rb") as f:
        saved = pickle.load(f)
    surface.walls.clear()
    surface.walls.update(saved["walls"])
    surface.retired |= saved["retired"]
    surface.energy_shift[:] = saved["energy_shift"]
    surface.edge_scale = saved["edge_scale"]
    surface.sizes = dict(saved["sizes"])
    ev._register_native()
    return ev, float(saved["start"]), int(saved["step"])


def cpu_groups(spec, jobs, threads):
    """--cpus: None (no pinning: the runs where the system puts them, or a
    cluster's scheduler); "all" (the CPUs this process may use); "fast" (the
    highest clocked, one per physical core: a hybrid CPU's big cores); or a
    list "0,2,4-6". Split into groups of `threads` CPUs, one per run at a
    time (at most `jobs`)."""
    if spec is None:
        return None
    usable = sorted(os.sched_getaffinity(0))
    if spec in ("all", "fast"):
        cpus = usable
        if spec == "fast":
            def read(path, default):
                try:
                    return open(path).read().strip()
                except OSError:
                    return default
            top = "/sys/devices/system/cpu/cpu{}/"
            freq = {c: int(read(top.format(c) + "cpufreq/cpuinfo_max_freq", "0")) for c in usable}
            seen, cpus = set(), []
            for c in sorted(usable, key=lambda c: (-freq[c], c)):
                core = read(top.format(c) + "topology/core_cpus_list", str(c))
                if core not in seen:
                    seen.add(core)
                    cpus.append(c)
    else:
        cpus = []
        for part in spec.split(","):
            a, _, b = part.partition("-")
            cpus += list(range(int(a), int(b or a) + 1))
    groups = [cpus[k:k + threads] for k in range(0, len(cpus) - threads + 1, threads)]
    if len(groups) < jobs:
        raise SystemExit(f"--cpus {spec}: {len(cpus)} CPUs ({cpus}), not enough for {jobs} "
                         f"runs of {threads} threads")
    return groups[:jobs]


def latest_state(ck, part=None):
    """The checkpoint in ck to resume from (its path without the extension):
    the one of the last step NAME.part has (its curve and counts that far),
    else the newest; None if there is none."""
    steps = sorted(f[:-4] for f in os.listdir(ck) if f.endswith(".pkl")) \
        if os.path.isdir(ck) else []
    if part and os.path.exists(part):
        at = f"step{int(json.load(open(part)).get('steps', -1)):04d}"
        if at in steps:
            return os.path.join(ck, at)
    return os.path.join(ck, steps[-1]) if steps else None


def launch(cfg, out, python, threads=1, gauge=None, flux_degree=None, cpus=None, frames=None):
    """One run in its own process; cpus: the CPUs to pin it to (None: not
    pinned). Its checkpoints in OUT/ck/NAME (removed when it gets there); a
    run whose process died (no NAME.json, not a timeout: a crash) resumed
    once from the last of them, with the time it has left."""
    name, n, L, seed, theta, h, ds, timeout, goal = cfg
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    path = os.path.join(out, f"{name}.json")
    part = os.path.join(out, f"{name}.part")
    ck = os.path.join(out, "ck", name)
    for stale in (path, part):
        if os.path.exists(stale):
            os.remove(stale)
    if os.path.isdir(ck):
        shutil.rmtree(ck)
    t = time.time()
    resume, crash = None, None
    while True:
        try:
            p = subprocess.run([python, os.path.abspath(__file__), "--one", name, str(n), str(L),
                                str(seed), str(theta), str(h), str(ds), out, goal]
                               + (["--gauge", gauge] if gauge else [])
                               + (["--flux-degree", str(flux_degree)] if flux_degree else [])
                               + (["--frames", str(frames)] if frames else [])
                               + ["--checkpoints", ck] + (["--resume", resume] if resume else []),
                               env=env, capture_output=True, text=True,
                               timeout=max(1.0, timeout - (time.time() - t)),
                               preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None)
            code, err = p.returncode, p.stderr
        except subprocess.TimeoutExpired as e:
            code, err = "timeout", (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        if os.path.exists(path) or code == "timeout" or resume or not latest_state(ck):
            break
        # (crashed: once more from its last checkpoint)
        resume = latest_state(ck, part)
        crash = f"{'signal ' + str(-code) if isinstance(code, int) and code < 0 else 'exit ' + str(code)}: " \
                f"{err.strip().splitlines()[-1] if err.strip() else ''}"
        print(f"  {name}: crashed ({crash}); resuming from {os.path.basename(resume)}", flush=True)
    if os.path.exists(path):
        r = json.load(open(path))
    elif os.path.exists(part):
        # (stopped from outside: the run as far as it got, its curve kept)
        r = json.load(open(part))
        r.update(reached=False, seconds=time.time() - t, stop=f"process: {code}",
                 stop_class="timeout" if code == "timeout" else f"crash (exit {code})")
        left = r["end"] + r.get("trapped", 0.0)
        r["drained"] = (r["start"] - left)/r["start"]
        json.dump(r, open(path, "w"), indent=1)
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
    if crash:
        r["crashed"] = crash
        json.dump(r, open(path, "w"), indent=1)
    if r.get("reached") and os.path.isdir(ck):
        shutil.rmtree(ck)
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
    goal = "breakthrough" if any(r.get("goal") == "breakthrough" for r in rows) \
        else f"{FRACTION:g} of the start"
    print(f"\n{total['runs']} runs, {total['reached']} reached {goal}; "
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
        # --one NAME N L SEED THETA H DS OUT [GOAL] [--checkpoints DIR] [--resume STATE]
        options = {}
        while len(argv) > 2 and argv[-2] in ("--checkpoints", "--resume", "--gauge",
                                             "--flux-degree", "--frames"):
            options[argv[-2][2:].replace("-", "_")] = argv[-1]
            argv = argv[:-2]
        name, n, L, seed, theta, h, ds, out = argv[1:9]
        one(name, int(n), float(L), int(seed), float(theta), float(h), float(ds), out,
            argv[9] if len(argv) > 9 else "half", **options)
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="quick", choices=("quick", "small", "wide", "wide-half", "full",
                                                       "periodic", "periodic-16", "periodic-32",
                                                       "periodic-64"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1,
                    help="OpenMP threads per run (runs times threads up to the cores: a "
                         "run's threads idle a third of the time, so several runs side by "
                         "side do more an hour)")
    ap.add_argument("--cpus",
                    help="pin each run to its own CPUs: 'fast' (the highest clocked, one a "
                         "physical core), 'all' (this process's), or a list '0,2,4-6'; "
                         "default: not pinned (a cluster's scheduler places them)")
    ap.add_argument("--timeout", type=float, help="each run's time limit in seconds (default: the set's)")
    ap.add_argument("--frames", type=int,
                    help="keep every this many steps' state in OUT/frames/NAME (a film: film.py)")
    ap.add_argument("--flux-degree", type=int,
                    help="pyse.flux_degree for the runs (the flux gauge's fast sum)")
    ap.add_argument("--wet-area", choices=("flux", "lines"),
                    help="pyse.wet_area for the runs (a slab's flux spheres: fields or contact lines)")
    ap.add_argument("--gauge", choices=("flux", "pole"),
                    help="the beads' gauge (default: flux to breakthrough, else pole)")
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--h", type=float, help="the set's runs at this edge length")
    ap.add_argument("--ds", type=float, help="the set's runs at this arclength step")
    ap.add_argument("--ds-area", action="store_true",
                    help="the arclength step scaled with the cell's area (as a 5 x 5 cell's)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--summary")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args(argv)
    if args.wet_area:
        os.environ["PYSE_WET_AREA"] = args.wet_area     # (each run's process reads it)
    if args.threads < 1 or args.jobs < 1:
        ap.error("--threads and --jobs must be at least 1")
    if args.summary:
        summary(load_dir(args.summary))
        return
    if args.compare:
        compare(*args.compare)
        return
    out = os.path.abspath(args.out or os.path.join(HERE, "out", args.set))
    os.makedirs(out, exist_ok=True)
    todo = [c for c in configs(args.set, args.h, args.ds, args.ds_area) if not args.only or c[0] in args.only]
    if args.timeout:
        todo = [(*c[:7], args.timeout, c[8]) for c in todo]
    # the longest first, so the pool ends together
    todo.sort(key=lambda c: -c[1])
    groups = cpu_groups(args.cpus, min(args.jobs, len(todo)), args.threads)
    print(f"{len(todo)} runs, {args.jobs} at a time ({args.threads} thread"
          f"{'s' if args.threads > 1 else ''} each"
          + (f", on CPUs {groups}" if groups else "") + f"), into {out}", flush=True)
    free = queue.Queue()
    for g in groups or []:
        free.put(g)

    def run(c):
        cpus = free.get() if groups else None
        try:
            return launch(c, out, args.python, args.threads, args.gauge, args.flux_degree, cpus,
                          args.frames)
        finally:
            if cpus:
                free.put(cpus)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run, todo))
    total = summary(results)
    json.dump(total, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
