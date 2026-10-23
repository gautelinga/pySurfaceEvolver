"""Run the code in a Markdown page of the docs and compare it with the outputs
shown there: each ```python block followed by a ```text block is run in its
own process (from docs/, so the helpers import), and its printed lines are
compared with the text block, numbers within a relative tolerance (the last
digits vary with the thread count). A block whose first line starts with
``# continuing`` runs after the page's blocks before it (their output not
compared again).

    python tools/check_doc_snippets.py [--update] [--rel 1e-3] [--jobs n] [page.md ...]

Default pages: docs/stress_cases.md, docs/several_surfaces.md,
docs/bridge_rupture.md and docs/packing.md. Exit status 1 if an output
differs (or a block fails); ``--update`` writes the new outputs of the blocks
that differ into the page instead. ``--jobs n`` runs up to n blocks at a time (each in its own process
as always; give each fewer threads with OMP_NUM_THREADS).
"""
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
BLOCK = re.compile(r"```python\n(.*?)```\n\n```text\n(.*?)```", re.S)
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?")


def same(expected: str, got: str, rel: float) -> bool:
    """Whether two outputs agree: the same words, numbers within ``rel``."""
    a, b = expected.split(), got.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        nx, ny = NUMBER.findall(x), NUMBER.findall(y)
        if NUMBER.sub("#", x) != NUMBER.sub("#", y) or len(nx) != len(ny):
            return False
        for p, q in zip(nx, ny):
            u, v = float(p.rstrip("%")), float(q.rstrip("%"))
            # (a unit in the last digit shown, allowing for its binary rounding:
            # 0.48 - 0.47 is a little over 0.01)
            if abs(u - v) > rel*max(abs(u), abs(v)) and abs(u - v) > 1.000001*10**-_decimals(p):
                return False
    return True


def _decimals(text: str) -> int:
    """Decimal places shown: a difference in the last one always passes."""
    mantissa = text.rstrip("%").split("e")[0].split("E")[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def run(code: str) -> "tuple[str, float, str]":
    t = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", code], cwd=DOCS, capture_output=True, text=True)
    return p.stdout, time.perf_counter() - t, (p.stderr if p.returncode else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*", default=[os.path.join(DOCS, name) for name in
                                                 ("stress_cases.md", "several_surfaces.md",
                                                  "bridge_rupture.md", "packing.md",
                                                  "drainage_topology.md")])
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--rel", type=float, default=1e-3)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    status = 0
    pool = concurrent.futures.ThreadPoolExecutor(max(1, args.jobs))
    # every block of every page queued at once (each in its own process, up to
    # --jobs at a time), then reported page by page, in order
    queued = []
    for page in args.pages:
        text = open(page).read()
        before, runs = [], []
        for m in BLOCK.finditer(text):
            code = m.group(1)
            if code.startswith("# continuing"):
                code = ("import contextlib, io\nwith contextlib.redirect_stdout(io.StringIO()):\n"
                        + "".join(f"    exec({c!r})\n" for c in before) + code)
            before.append(m.group(1))
            runs.append(pool.submit(run, "import warnings; warnings.simplefilter('ignore')\n"
                                    + code))
        queued.append((page, text, runs))
    for page, text, runs in queued:
        out, pos = [], 0
        for k, (m, done) in enumerate(zip(BLOCK.finditer(text), runs), 1):
            got, seconds, error = done.result()
            ok = not error and same(m.group(2), got, args.rel)
            print(f"{os.path.basename(page)} block {k}: {'ok' if ok else 'DIFFERS'} "
                  f"({seconds:.0f} s)", flush=True)
            if error:
                print(error[-2000:])
            elif not ok:
                print("  shown:\n    " + m.group(2).rstrip().replace("\n", "\n    "))
                print("  now:\n    " + got.rstrip().replace("\n", "\n    "))
            status |= not ok
            # (only blocks that differ are rewritten: a block within the
            # tolerance keeps its digits)
            out.append(text[pos:m.start(2)] + (got if args.update and not ok and not error
                                               else m.group(2)))
            pos = m.end(2)
        if args.update:
            with open(page, "w") as fh:
                fh.write("".join(out) + text[pos:])
    return 0 if args.update else status


if __name__ == "__main__":
    sys.exit(main())
