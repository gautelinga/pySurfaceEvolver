"""Run many independent Evolver jobs in worker processes.

Surface Evolver allows one surface per process, so parallel sweeps use
processes. Each worker process has its own engine and runs jobs one after
another. A job that crashes its process (Evolver has had memory bugs) only
loses that job: the worker is replaced and the other jobs continue.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import traceback
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Any, Callable, Dict, Iterable, List, Optional

__all__ = ["map", "JobError", "WorkerCrashed", "WorkerStartError"]

_builtin_map = map


class WorkerCrashed(RuntimeError):
    """A job's worker process died (for example a segfault) while running it."""

    def __init__(self, exitcode: Optional[int]):
        super().__init__(f"worker process died while running the job (exit code {exitcode})")
        self.exitcode = exitcode


class WorkerStartError(RuntimeError):
    """Worker processes died before they could take a job.

    Workers are started with the ``spawn`` method, which re-imports the
    main script in every worker. That fails for a script read from stdin
    or ``python -c``, and a script whose top-level code calls
    :func:`map` without an ``if __name__ == "__main__":`` guard starts
    workers recursively.
    """

    def __init__(self, exitcode: Optional[int]):
        super().__init__(
            f"a worker process died while starting (exit code {exitcode}); its error "
            "output is above. pyse.map starts workers with the 'spawn' method, which "
            "re-imports the main script: run the script from a file (not stdin or "
            "python -c) and put the code that calls pyse.map under "
            "'if __name__ == \"__main__\":'.")
        self.exitcode = exitcode


@dataclass
class JobError(Exception):
    """A job raised an exception (or crashed its worker).

    ``index`` and ``param`` identify the job; ``error`` is the exception,
    ``traceback`` the worker's formatted traceback (if any).
    """

    index: int
    param: Any
    error: BaseException
    traceback: str = ""

    def __str__(self) -> str:
        return f"job {self.index} ({self.param!r}) failed: {self.error!r}"


def _dumps(obj: Any) -> bytes:
    """Pickle, with cloudpickle when available (functions defined in
    notebooks or __main__ then work too)."""
    try:
        import cloudpickle
        return cloudpickle.dumps(obj)
    except ImportError:
        return pickle.dumps(obj)


_READY = b"ready"


def _worker(conn, threads: int) -> None:
    os.environ["PYSE_THREADS"] = str(threads)
    conn.send_bytes(_READY)   # started: a later death is the job's fault
    while True:
        message = conn.recv_bytes()
        if not message:
            return
        index, payload = pickle.loads(message)
        try:
            fn, param = pickle.loads(payload)
            result = ("ok", fn(param))
        except BaseException as e:  # noqa: BLE001 - report every failure
            result = ("error", (e, traceback.format_exc()))
        try:
            data = pickle.dumps((index, result))
        except Exception as e:  # unpicklable result or exception
            data = pickle.dumps((index, ("error", (RuntimeError(
                f"could not send the job's result back: {e!r}"), traceback.format_exc()))))
        conn.send_bytes(data)


class _Worker:
    def __init__(self, ctx, threads: int):
        self.conn, child = ctx.Pipe()
        env_before = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = str(threads)   # inherited by the child
        try:
            self.process = ctx.Process(target=_worker, args=(child, threads), daemon=True)
            self.process.start()
        finally:
            if env_before is None:
                del os.environ["OMP_NUM_THREADS"]
            else:
                os.environ["OMP_NUM_THREADS"] = env_before
        child.close()
        self.job: Optional[int] = None
        self.started = False

    def stop(self) -> None:
        try:
            self.conn.send_bytes(b"")
        except (OSError, ValueError):
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
        self.conn.close()


def map(fn: Callable[[Any], Any], params: Iterable[Any], *, processes: Optional[int] = None,
        threads: Optional[int] = None, errors: str = "raise",
        callback: Optional[Callable[[int, Any], Any]] = None) -> List[Any]:
    """Run ``fn(param)`` for every param in worker processes, in parallel.

    ``fn`` typically builds or loads a surface with :class:`Evolver`, runs
    it, and returns numbers, arrays or a :class:`Snapshot`. Results come back
    in the order of ``params``.

    processes:
        Number of worker processes (default: number of CPUs, at most the
        number of jobs).
    threads:
        Threads per worker for Evolver's parallel loops and Newton steps
        (default 1: for sweeps, more processes beat threads); sets
        ``OMP_NUM_THREADS`` in the workers.
    errors:
        ``"raise"`` (default): after all jobs finish, raise the first
        failure as :class:`JobError` (its ``results`` attribute has all
        results, with failures as JobError objects). ``"return"``: put
        :class:`JobError` objects in the result list instead.
    callback:
        ``callback(index, result)`` is called in this process as each job
        finishes.

    ``fn`` must be picklable: a module-level function, or anything when
    ``cloudpickle`` is installed (which also covers functions defined in
    notebooks). A job that crashes its worker gives a :class:`JobError`
    wrapping :class:`WorkerCrashed`; the other jobs are unaffected. Workers
    that die while starting raise :class:`WorkerStartError` right away
    (see there for the usual causes).
    """
    if errors not in ("raise", "return"):
        raise ValueError("errors must be 'raise' or 'return'")
    params = list(params)
    if not params:
        return []
    cpus = os.cpu_count() or 1
    processes = max(1, min(processes or cpus, len(params)))
    threads = threads or 1
    ctx = mp.get_context("spawn")
    payloads = [_dumps((fn, p)) for p in params]
    results: Dict[int, Any] = {}
    pending = list(range(len(params)))[::-1]
    workers: List[_Worker] = []

    def assign(w: _Worker) -> None:
        if pending:
            w.job = pending.pop()
            w.conn.send_bytes(pickle.dumps((w.job, payloads[w.job])))
        else:
            w.job = None

    def finish(index: int, value: Any) -> None:
        results[index] = value
        if callback is not None:
            callback(index, value)

    try:
        for _ in range(processes):
            w = _Worker(ctx, threads)
            workers.append(w)
            assign(w)
        while len(results) < len(params):
            busy = {w.conn: w for w in workers if w.job is not None}
            sentinels = {w.process.sentinel: w for w in workers if w.job is not None}
            for ready in wait(list(busy) + list(sentinels)):
                found = busy.get(ready) or sentinels.get(ready)
                # a dead worker shows up twice (pipe and process): skip repeats
                if found is None or found.job is None or found not in workers:
                    continue
                w, index = found, found.job
                try:
                    message = w.conn.recv_bytes()
                except (EOFError, OSError):
                    w.process.join(timeout=5)
                    if not w.started:   # the same would happen to every worker
                        raise WorkerStartError(w.process.exitcode) from None
                    # the worker died: only its current job is lost
                    finish(index, JobError(index, params[index],
                                           WorkerCrashed(w.process.exitcode)))
                    w.job = None
                    w.conn.close()
                    workers.remove(w)
                    replacement = _Worker(ctx, threads)
                    workers.append(replacement)
                    assign(replacement)
                    continue
                if message == _READY:
                    w.started = True
                    continue
                _, (status, value) = pickle.loads(message)
                if status == "ok":
                    finish(index, value)
                else:
                    error, tb = value
                    finish(index, JobError(index, params[index], error, tb))
                assign(w)
    finally:
        for w in workers:
            if w.job is not None:
                w.process.terminate()
            w.stop()

    ordered = [results[i] for i in range(len(params))]
    if errors == "raise":
        for r in ordered:
            if isinstance(r, JobError):
                r.results = ordered  # type: ignore[attr-defined]
                raise r
    return ordered
