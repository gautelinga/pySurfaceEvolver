"""Evolver(process=True): surfaces in worker processes of their own."""

import os
import signal
import time
import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import BodyView, recipes

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX only")


@pytest.fixture
def remote(fe_dir):
    """A cube in a worker process (closed after the test)."""
    ev = pyse.Evolver("cube.fe", process=True, threads=1)
    yield ev
    ev.close()


def test_two_surfaces_at_once(fe_dir):
    with pyse.Evolver("cube.fe", process=True, threads=1) as a, \
            pyse.Evolver(process=True, threads=1) as b:
        b.load("cube.fe")
        assert a.pid != b.pid != os.getpid()
        b.body(1).target = 2.0
        ra, rb = a.relax(levels=1), b.relax(levels=1)
        assert ra.converged and rb.converged
        assert a.body(1).volume == pytest.approx(1.0) and b.body(1).volume == pytest.approx(2.0)
        # the energy scales as volume^(2/3)
        assert b.energy/a.energy == pytest.approx(2**(2/3), rel=2e-2)
        # and the engine of this process is untouched
        local = pyse.Evolver("cube.fe")
        assert local.counts["facets"] == 24


def test_a_remote_surface_works_like_a_local_one(remote, cube):
    assert isinstance(remote, pyse.Evolver) and isinstance(remote.body(1), BodyView)
    assert "in process" in repr(remote)
    for ev in (remote, cube):
        ev.relax(levels=1)
    assert remote.energy == pytest.approx(cube.energy, rel=1e-9)
    assert remote.counts == cube.counts
    m = remote.mesh()
    assert np.allclose(m.vertices, cube.mesh().vertices)
    remote.define("pp", 0.5)
    assert remote["pp"] == 0.5
    remote["pp"] = 0.25
    assert remote.eval("pp") == 0.25
    snapshot = remote.save()
    remote.refine()
    remote.restore(snapshot)
    assert remote.counts["facets"] == 96
    with pytest.raises(pyse.EvolverError) as e:
        remote.command("this isn't a command")
    assert e.value.errnum and e.value.output is not None
    with pytest.raises(AttributeError):
        remote.no_such_attribute


def test_a_continuation_steps_a_remote_body(remote):
    remote.relax(levels=1)
    steps = list(recipes.continuation(remote, remote.body(1), [1.0, 1.1, 1.2, 1.3]))
    assert [s.volumes[0] for s in steps] == pytest.approx([1.0, 1.1, 1.2, 1.3], rel=1e-8)
    assert steps[-1].predicted


def test_callbacks_run_in_the_worker_and_print_here(remote, capsys):
    pytest.importorskip("cloudpickle")      # (to send a function defined in a function)

    def callback(ev, i):
        print("step", i, ev.counts["facets"])
        warnings.warn(f"at {i}", UserWarning)

    with pytest.warns(UserWarning, match="at 2"):
        remote.iterate(2, callback=callback)
    assert "step 1 24" in capsys.readouterr().out


def test_closing_and_a_dead_worker(fe_dir):
    # (as if a library had reset SIGPIPE: writing to the dead worker must
    # raise, not kill this process)
    previous = signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        _closing_and_a_dead_worker()
    finally:
        signal.signal(signal.SIGPIPE, previous)


def test_a_function_that_cant_be_sent(remote, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_cloudpickle(name, *args, **kwargs):
        if name == "cloudpickle":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cloudpickle)
    with pytest.raises(TypeError, match="cloudpickle"):
        remote.iterate(1, callback=lambda ev, i: None)
    assert remote.counts["facets"] == 24    # and the worker is fine


def _closing_and_a_dead_worker():
    ev = pyse.Evolver("cube.fe", process=True, threads=1)
    os.kill(ev.pid, signal.SIGKILL)
    time.sleep(0.2)
    with pytest.raises(pyse.WorkerCrashed, match="surface is lost"):
        ev.energy
    with pytest.raises(pyse.WorkerCrashed):
        ev.relax()
    ev = pyse.Evolver(process=True, threads=1)
    pid = ev.pid
    ev.close()
    with pytest.raises(RuntimeError, match="closed"):
        ev.energy
    time.sleep(0.1)
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_handles_stay_with_their_worker(fe_dir):
    with pyse.Evolver("cube.fe", process=True, threads=1) as a, \
            pyse.Evolver("cube.fe", process=True, threads=1) as b:
        with pytest.raises(ValueError, match="another worker"):
            b.restore(a.body(1))        # (a handle on a's surface, given to b)


def test_arguments_that_need_this_process(fe_dir):
    with pytest.raises(ValueError, match="input="):
        pyse.Evolver(process=True, input=lambda prompt: "y")
    with pytest.raises(ValueError, match="process=True"):
        pyse.Evolver(threads=2)


def test_a_remote_surface_plots_here(remote, can_render, tmp_path):
    remote.relax()
    path = tmp_path / "remote.png"
    remote.plot(off_screen=True, screenshot=str(path))
    assert path.stat().st_size > 0


def test_ctrl_c_interrupts_the_workers_call(run_python):
    result = run_python("""
        import os, signal, threading
        import pysurfaceevolver as pyse
        ev = pyse.Evolver("cube.fe", process=True, threads=1)
        ev.command("r; r; r")
        threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT)).start()
        try:
            ev.command("g 10000000")
        except KeyboardInterrupt:
            print("INTERRUPTED")
        print("VOLUME", ev.body(1).volume)
    """)
    assert "INTERRUPTED" in result.stdout, result.stderr
    volume = float(result.stdout.split("VOLUME")[1].split()[0])
    assert volume == pytest.approx(1.0, rel=1e-6)
