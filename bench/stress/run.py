"""Run the stress suite with pySE's default settings and print a report.

usage: python bench/stress/run.py [case numbers ...]   (from the repo, or anywhere)
"""
import importlib, json, os, sys, time, traceback, warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Result

CASES = ["case01_cylinder", "case02_catenoid", "case03_sessile", "case09_inflate",
         "case12_shrink"]


def main(argv):
    wanted = [a for a in argv if a.isdigit()]
    results = []
    for name in CASES:
        if wanted and str(int(name[4:6])) not in wanted:
            continue
        mod = importlib.import_module(name)
        print(f"--- {mod.NAME}", flush=True)
        t = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = mod.run()
        except Exception as e:
            r = Result(mod.NAME, False, f"CRASHED: {type(e).__name__}: {str(e)[:200]}",
                       seconds=time.perf_counter() - t)
            r.notes.append(traceback.format_exc()[-800:])
        print(f"    {'PASS' if r.passed else 'FAIL'}  {r.summary}", flush=True)
        results.append(r)
    print()
    print(f"{'case':36s} {'result':6s} {'error':>8s} {'time':>7s} {'g steps':>8s} {'newton':>6s} "
          f"{'unconv':>6s} {'min angle':>9s} {'neg eig':>7s}")
    for r in results:
        print(f"{r.case:36s} {'PASS' if r.passed else 'FAIL':6s} {r.error:8.2%} {r.seconds:6.1f}s "
              f"{r.gradient_steps:8d} {r.newton_steps:6d} {r.unconverged_steps:6d} "
              f"{r.min_angle:9.1f} {str(r.negative_eigenvalues):>7s}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    json.dump([r.__dict__ for r in results], open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1:])
