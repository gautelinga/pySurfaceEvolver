"""pse.map: sweeps in worker processes."""

import os
import subprocess
import sys

import numpy as np
import pytest

import pysurfaceevolver as pse
from pysurfaceevolver import JobError, WorkerCrashed


def relaxed_cube_area(volume):
    ev = pse.Evolver("cube.fe")
    ev.set_values("body", "target", volume)
    ev.relax(tol=1e-8, max_iter=300)
    return ev.eval("total_area")


def fails_for_two(x):
    if x == 2:
        raise ValueError("two is not allowed")
    return x * 10


def crashes_for_two(x):
    if x == 2:
        os.abort()                      # like a segfault in Evolver
    return x * 10


def report_threads(_):
    return os.environ.get("OMP_NUM_THREADS"), os.environ.get("PYSE_THREADS")


def test_map_runs_evolver_jobs_in_order():
    volumes = [0.5, 1.0, 2.0, 4.0]
    areas = pse.map(relaxed_cube_area, volumes, processes=2)
    # area scales like volume^(2/3) for the same shape
    np.testing.assert_allclose(np.array(areas) / np.array(volumes) ** (2 / 3),
                               areas[1], rtol=1e-6)


def test_map_raises_job_errors():
    with pytest.raises(JobError) as info:
        pse.map(fails_for_two, [1, 2, 3], processes=2)
    assert info.value.index == 1 and isinstance(info.value.error, ValueError)
    assert "two is not allowed" in info.value.traceback
    assert info.value.results[0] == 10 and info.value.results[2] == 30


def test_map_can_return_errors():
    results = pse.map(fails_for_two, [1, 2, 3], errors="return")
    assert results[0] == 10 and results[2] == 30
    assert isinstance(results[1], JobError)


def test_crash_only_loses_that_job():
    results = pse.map(crashes_for_two, [1, 2, 3, 4], processes=2, errors="return")
    assert [results[i] for i in (0, 2, 3)] == [10, 30, 40]
    assert isinstance(results[1], JobError)
    assert isinstance(results[1].error, WorkerCrashed)


def test_map_lambda_with_cloudpickle():
    pytest.importorskip("cloudpickle")
    assert pse.map(lambda x: x ** 2, range(5), processes=2) == [0, 1, 4, 9, 16]


def test_map_splits_threads():
    out = pse.map(report_threads, range(2), processes=2, threads=3)
    assert out == [("3", "3"), ("3", "3")]


def test_map_callback_and_empty():
    seen = []
    pse.map(fails_for_two, [1, 3], callback=lambda i, r: seen.append((i, r)))
    assert sorted(seen) == [(0, 10), (1, 30)]
    assert pse.map(fails_for_two, []) == []


def test_workers_that_cannot_start_fail_fast():
    # spawn re-imports the main script, which doesn't exist for stdin
    script = ("import pysurfaceevolver as pse\n"
              "def f(x): return x\n"
              "try:\n"
              "    pse.map(f, range(20), processes=2)\n"
              "except pse.WorkerStartError as e:\n"
              "    print('start error:', e)\n")
    out = subprocess.run([sys.executable, "-"], input=script, capture_output=True,
                         text=True, timeout=120)
    assert "start error:" in out.stdout
    assert "if __name__" in out.stdout
