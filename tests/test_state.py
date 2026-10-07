"""Engine state: threads, invalid surfaces, Ctrl-C, packaging."""

import importlib.metadata
import importlib.resources
import threading
import time

import pytest

import pysurfaceevolver
from pysurfaceevolver import (
    Evolver,
    EvolverBusyError,
    EvolverError,
    InvalidSurfaceError,
)

BROKEN_DATAFILE = "vertices\n1 0 0 0\n2 1 0 0\nedges\n1 1 7\n"


# --- concurrent calls ------------------------------------------------------------

def test_rejected_call_from_other_thread_leaves_running_call_alone(load):
    # The worker's command stops at an interactive prompt; while it waits
    # there (holding the engine), the main thread tries a call of its own.
    worker_waiting = threading.Event()
    main_done = threading.Event()

    def answer(prompt):
        worker_waiting.set()
        main_done.wait(timeout=30)
        return "0"  # leave hessian_menu

    ev = load("cube.fe", input=answer)
    result = {}

    def worker():
        result["output"] = ev.command("g 3; hessian_menu; g 2")

    thread = threading.Thread(target=worker)
    thread.start()
    assert worker_waiting.wait(timeout=30)
    with pytest.raises(EvolverBusyError, match="another thread"):
        ev.eval("total_area")
    with pytest.raises(EvolverBusyError, match="busy"):
        ev.mesh()
    main_done.set()
    thread.join(timeout=30)

    # the worker kept all of its output: 3 + 2 iteration lines
    assert result["output"].count("scale:") == 5
    assert ev.eval("body[1].volume") == pytest.approx(1.0, rel=1e-6)


def _hold_engine(load):
    """A worker thread holding the engine at a prompt until release.set()."""
    holding, release = threading.Event(), threading.Event()

    def answer(prompt):
        holding.set()
        release.wait(timeout=30)
        return "0"

    ev = load("cube.fe", input=answer)
    thread = threading.Thread(target=ev.command, args=("g 1; hessian_menu",))
    thread.start()
    assert holding.wait(timeout=30)
    return ev, thread, release


def test_busy_timeout_waits_for_other_thread(load, monkeypatch):
    ev, thread, release = _hold_engine(load)
    monkeypatch.setattr(pysurfaceevolver, "busy_timeout", float("inf"))
    threading.Timer(0.3, release.set).start()
    start = time.perf_counter()
    assert ev.eval("body[1].volume") == pytest.approx(1.0, rel=1e-3)   # waited for the worker
    assert time.perf_counter() - start >= 0.25
    thread.join(timeout=30)


def test_busy_timeout_runs_out(load, monkeypatch):
    ev, thread, release = _hold_engine(load)
    monkeypatch.setattr(pysurfaceevolver, "busy_timeout", 0.2)
    with pytest.raises(EvolverBusyError, match="waited 0.2 s"):
        ev.counts
    assert isinstance(EvolverBusyError(), RuntimeError)
    release.set()
    thread.join(timeout=30)


def test_torus_display_mode_does_not_carry_over(load):
    ev = load("100grain.fe")             # sets "clipped"
    assert "clipped" in ev.save().text
    ev.load("cube.fe")
    assert "clipped" not in ev.save().text


# --- invalid surfaces ---------------------------------------------------------------

def test_failed_load_invalidates_surface(load):
    ev = load("cube.fe")
    with pytest.raises(EvolverError):
        ev.load_string(BROKEN_DATAFILE)
    assert not ev.valid
    assert "no valid surface" in repr(ev)
    blocked = [
        ev.mesh,
        ev.bodies,
        ev.quantities,
        lambda: ev.vertices,
        lambda: ev.command("g 1"),
        lambda: ev.eval("1 + 1"),
        lambda: ev.values("vertex", "x"),
        lambda: dict(ev.parameters),
    ]
    for call in blocked:
        with pytest.raises(InvalidSurfaceError):
            call()


def test_load_recovers_from_invalid_surface(load):
    ev = load("cube.fe")
    with pytest.raises(EvolverError):
        ev.load_string(BROKEN_DATAFILE)
    ev.load("mound.fe")
    assert ev.valid
    ev.command("g 2")
    assert ev.mesh().facets.shape == (ev.counts["facets"], 3)


def test_missing_file_keeps_surface_valid(load):
    ev = load("cube.fe")
    with pytest.raises(FileNotFoundError):
        ev.load("no_such_file.fe")
    assert ev.valid
    assert ev.eval("body[1].volume") == pytest.approx(1.0)


def test_invalid_surface_error_is_an_evolver_error():
    assert issubclass(InvalidSurfaceError, EvolverError)


# --- Ctrl-C ------------------------------------------------------------------------------

def test_one_ctrl_c_stops_a_while_loop(run_python):
    result = run_python("""
        import os, signal, threading
        from pysurfaceevolver import Evolver
        ev = Evolver("cube.fe")
        threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()
        try:
            ev.command("ii := 0; while 1 do ii := ii + 1")
        except KeyboardInterrupt:
            print("INTERRUPTED")
        print("COUNT_POSITIVE", ev.eval("ii") > 0)
    """)
    assert "INTERRUPTED" in result.stdout, result.stderr
    assert "COUNT_POSITIVE True" in result.stdout


def test_two_ctrl_c_abort_an_operation_that_ignores_the_first(run_python):
    # hessian doesn't check Evolver's break flag, so one Ctrl-C lets it
    # finish, while a second one aborts it.
    result = run_python("""
        import os, signal, threading, time
        from pysurfaceevolver import Evolver

        def setup():
            ev = Evolver("cube.fe")
            ev.command("r; r; r; r; r; r; r; g 1")
            return ev

        ev = setup()
        t = time.time()
        ev.command("hessian")
        baseline = time.time() - t
        print("BASELINE", baseline)
        if baseline < 0.4:
            print("TOO_FAST")
            raise SystemExit

        for presses in (1, 2):
            ev = setup()
            def fire():
                for _ in range(presses):
                    os.kill(os.getpid(), signal.SIGINT)
                    time.sleep(0.01)
            threading.Timer(0.2 * baseline, fire).start()
            t = time.time()
            try:
                ev.command("hessian")
            except KeyboardInterrupt:
                print("INTERRUPTED", presses, time.time() - t)
            print("USABLE", presses, ev.eval("body[1].volume"))
    """, timeout=300)
    if "TOO_FAST" in result.stdout:
        pytest.skip("hessian is too fast here to interrupt reliably")
    lines = dict((l.split()[0] + l.split()[1], l.split()[2])
                 for l in result.stdout.splitlines()
                 if l.startswith(("INTERRUPTED", "USABLE")))
    baseline = float(result.stdout.split("BASELINE")[1].split()[0])
    assert "INTERRUPTED1" in lines and "INTERRUPTED2" in lines, result.stderr
    # one press: hessian ran to completion; two: it stopped early
    assert float(lines["INTERRUPTED1"]) > 0.6 * baseline
    assert float(lines["INTERRUPTED2"]) < 0.6 * baseline
    assert float(lines["USABLE1"]) == pytest.approx(1.0, rel=1e-3)
    assert float(lines["USABLE2"]) == pytest.approx(1.0, rel=1e-3)


# --- packaging -----------------------------------------------------------------------------

def test_version_has_one_source():
    assert pysurfaceevolver.__version__ == importlib.metadata.version("pysurfaceevolver")


def test_type_information_is_installed():
    files = importlib.resources.files("pysurfaceevolver")
    assert files.joinpath("py.typed").is_file()
    stub = files.joinpath("_core.pyi").read_text()
    assert "class CallResult" in stub
    assert "def command(" in stub


def test_failed_hessian_twice(load):
    """A failed Newton step left pointers to freed memory that the next one
    freed again (tankex: gap energy has no Hessian)."""
    ev = load("tankex.fe")
    ev.command("g 5")
    for _ in range(2):
        with pytest.raises(EvolverError):
            ev.command("hessian")
    ev.command("g 5")
    assert ev.eval("total_energy") > 0
