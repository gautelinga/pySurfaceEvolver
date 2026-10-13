"""Surfaces in worker processes: ``Evolver(process=True)``.

Surface Evolver keeps its state in C globals, so a process holds one surface.
``Evolver(process=True)`` starts a worker process (a fresh interpreter running
:mod:`pysurfaceevolver._worker`) with an engine of its own, and returns a
handle that forwards attribute reads, assignments and method calls to it:
the API is the same, the surface lives in the worker. Views such as
``ev.body(1)`` or ``ev.parameters`` come back as handles too, and results
(meshes, IterationResults, arrays) as copies.
"""

from __future__ import annotations

import inspect
import os
import pickle
import signal
import socket
import subprocess
import sys
import threading
import warnings
import weakref
from multiprocessing.connection import Connection
from typing import Any, Optional, Tuple

from ._errors import EvolverBusyError
from ._parallel import WorkerCrashed, WorkerStartError

__all__ = ["RemoteHandle"]

# the classes whose objects are views of the worker's surface: they stay
# there, and the caller gets a handle (the rest is sent back as a copy)
_VIEWS = ("Evolver", "BodyView", "QuantityView", "Parameters")


def _dumps(obj: Any) -> bytes:
    """Pickle, with cloudpickle when available (functions defined in a
    script or notebook can then be sent too)."""
    try:
        import cloudpickle
        return cloudpickle.dumps(obj)
    except ImportError:
        return pickle.dumps(obj)


def _view_class(name: str) -> type:
    from . import _views
    from ._evolver import Evolver
    return Evolver if name == "Evolver" else getattr(_views, name)


class _Ref:
    """A handle sent back to its worker as an argument: there, the object."""

    def __init__(self, path: tuple):
        self.path = path


class _Worker:
    """A worker process and the connection to it."""

    def __init__(self, threads: Optional[int]):
        if os.name != "posix":
            raise NotImplementedError("Evolver(process=True) needs a POSIX system "
                                      "(Linux or macOS)")
        mine, theirs = socket.socketpair()
        env = dict(os.environ)
        # the worker imports this package from where this process found it
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = os.pathsep.join([here] + [p for p in
                                                      [env.get("PYTHONPATH")] if p])
        if threads is not None:
            env["PYSE_THREADS"] = str(int(threads))
        self.process = subprocess.Popen(
            [sys.executable, "-m", "pysurfaceevolver._worker", str(theirs.fileno())],
            pass_fds=(theirs.fileno(),), env=env, close_fds=True)
        theirs.close()
        self.conn = Connection(os.dup(mine.fileno()))
        mine.close()
        self.lock = threading.Lock()
        self.dead: Optional[str] = None
        self.owner: Any = None              # the RemoteEvolver (a weak reference)
        try:
            ready = self.conn.recv_bytes()
        except (EOFError, OSError):
            self.process.wait()
            raise WorkerStartError(self.process.returncode) from None
        self.pid = pickle.loads(ready)[1]

    def request(self, *message: Any) -> Any:
        if self.dead:
            raise self._crash() if self.dead == "crashed" else \
                RuntimeError("this Evolver's worker process was closed")
        from . import busy_timeout      # as for the engine of this process
        if busy_timeout is None:
            free = self.lock.acquire(blocking=False)
        elif busy_timeout == float("inf"):
            free = self.lock.acquire()
        else:
            free = self.lock.acquire(timeout=busy_timeout)
        if not free:
            raise EvolverBusyError("Surface Evolver is busy (another thread is using this "
                                   "worker)")
        try:
            try:
                self.conn.send_bytes(_dumps((os.getcwd(),) + message))
            except (OSError, ValueError):
                self._crashed()
            return self._reply()
        finally:
            self.lock.release()

    def _reply(self) -> Any:
        while True:
            try:
                # poll first: an interrupt then never cuts a message in half
                self.conn.poll(None)
                data = self.conn.recv_bytes()
            except KeyboardInterrupt:
                # Ctrl-C reaches this process only (the worker has a process
                # group of its own): pass it on; the worker answers with the
                # interrupt, raised below
                try:
                    os.kill(self.pid, signal.SIGINT)
                except OSError:
                    pass
                continue
            except (EOFError, OSError):
                self._crashed()
            kind, *rest = pickle.loads(data)
            if kind == "out":
                stream = sys.stderr if rest[0] else sys.stdout
                stream.write(rest[1])
                stream.flush()
                continue
            for category, text in rest[-1]:
                warnings.warn(text, category, stacklevel=5)     # the caller's line
            if kind == "ok":
                return rest[0]
            if kind == "view":
                return _View(rest[0])
            error, trace = rest[0], rest[1]
            if isinstance(error, BaseException):
                raise error
            raise RuntimeError(f"in the worker process: {error}\n{trace}")

    def _crashed(self) -> None:
        self.process.wait()
        self.dead = "crashed"
        raise self._crash() from None

    def _crash(self) -> WorkerCrashed:
        error = WorkerCrashed(self.process.returncode)
        error.args = (f"the worker process of Evolver(process=True) died (exit code "
                      f"{self.process.returncode}); its surface is lost",)
        return error

    def close(self) -> None:
        if self.dead:
            return
        self.dead = "closed"
        try:
            self.conn.send_bytes(b"")
        except (OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.conn.close()


class _View:
    """The worker's answer when the result is a view (stays there)."""

    def __init__(self, name: str):
        self.name = name


def _stop(worker: _Worker) -> None:
    worker.close()


def _static(cls: type, name: str) -> Any:
    try:
        return inspect.getattr_static(cls, name)
    except AttributeError:
        return None


class RemoteHandle:
    """A handle on an object in a worker process: an Evolver there, or one
    of its views (``ev.body(1)``, ``ev.parameters``, ...). Attribute reads,
    assignments, calls and item access go to the object in the worker.
    ``isinstance`` sees the class of the object it stands for."""

    __slots__ = ("_worker", "_path", "_cls", "__weakref__")

    def __init__(self, worker: _Worker, path: tuple, cls: type):
        object.__setattr__(self, "_worker", worker)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_cls", cls)

    @property                                   # type: ignore[misc]
    def __class__(self):                        # isinstance(ev, Evolver), recipes' checks
        return self._cls

    def _send(self, op: str, *args: Any, step: Optional[tuple] = None) -> Any:
        result = self._worker.request(op, self._path, *args)
        if isinstance(result, _View):
            if result.name == "Evolver":    # the worker's one engine: its handle
                owner = self._worker.owner and self._worker.owner()
                return owner if owner is not None else RemoteHandle(self._worker, (),
                                                                    _view_class("Evolver"))
            return RemoteHandle(self._worker, self._path + (step,), _view_class(result.name))
        return result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        static = _static(self._cls, name)
        if callable(static) and not isinstance(static, (property, type)):
            return _Method(self, name)
        return self._send("get", name, step=("get", name))

    def __setattr__(self, name: str, value: Any) -> None:
        self._send("set", name, value)

    def __getitem__(self, key: Any) -> Any:
        return self._send("call", "__getitem__", (key,), {}, step=("call", "__getitem__", (key,), {}))

    def __setitem__(self, key: Any, value: Any) -> None:
        self._send("call", "__setitem__", (key, value), {})

    def __delitem__(self, key: Any) -> None:
        self._send("call", "__delitem__", (key,), {})

    def __contains__(self, key: Any) -> bool:
        return self._send("call", "__contains__", (key,), {})

    def __len__(self) -> int:
        return self._send("call", "__len__", (), {})

    def __iter__(self):
        return iter(self._send("list", ()))

    def __repr__(self) -> str:
        return self._send("call", "__repr__", (), {})

    def __reduce__(self):
        return (_Ref, (self._path,))

    def __dir__(self):
        return sorted(set(dir(self._cls)) | {"close", "pid"})


class _Method:
    """A method of an object in the worker, called there."""

    def __init__(self, owner: RemoteHandle, name: str):
        self.owner, self.name = owner, name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for a in list(args) + list(kwargs.values()):
            if type(a) in (RemoteHandle, RemoteEvolver) and a._worker is not self.owner._worker:
                raise ValueError("a handle on another worker process's surface can't be "
                                 "passed here (each surface lives in its own process)")
        return self.owner._send("call", self.name, args, kwargs,
                                step=("call", self.name, args, kwargs))

    def __repr__(self) -> str:
        return f"<method {self.name} of {self.owner._cls.__name__} in process {self.owner._worker.pid}>"


class RemoteEvolver(RemoteHandle):
    """An :class:`Evolver` in a worker process of its own; see
    ``Evolver(process=True)``."""

    __slots__ = ()

    def __init__(self, datafile: Any = None, *, echo: bool = False, input: Any = None,
                 threads: Optional[int] = None):
        if input is not None:
            raise ValueError("Evolver(process=True) can't answer Evolver's prompts through "
                             "input= (it would run in the worker process)")
        from ._evolver import Evolver
        worker = _Worker(threads)
        super().__init__(worker, (), Evolver)
        worker.owner = weakref.ref(self)
        weakref.finalize(self, _stop, worker)
        if echo:
            self.echo = True
        if datafile is not None:
            self.load(datafile)

    @property
    def pid(self) -> int:
        """The worker process's id."""
        return self._worker.pid

    def close(self) -> None:
        """End the worker process (and the surface in it). Handles on it then
        raise; this also happens when the last of them is garbage collected,
        and at exit."""
        self._worker.close()

    def __enter__(self) -> "RemoteEvolver":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._worker.dead:
            return f"<Evolver in a {self._worker.dead} worker process>"
        return super().__repr__()[:-1] + f", in process {self.pid}>"

    # drawing happens here, from the worker's data
    def plot(self, *args: Any, **kwargs: Any) -> Any:
        from ._export import _Export
        return _Export.plot(self, *args, **kwargs)   # type: ignore[arg-type]

    def live_view(self, *args: Any, **kwargs: Any) -> Any:
        from ._export import _Export
        return _Export.live_view(self, *args, **kwargs)   # type: ignore[arg-type]

    def _run(self, function: str, *args: Any, **kwargs: Any) -> Any:
        """``function(ev, *args, **kwargs)`` in the worker (``function``: a
        dotted name in this package), with its Evolver."""
        return self._worker.request("run", (), function, args, kwargs)


def is_remote(ev: Any) -> bool:
    return type(ev) is RemoteEvolver


def resolve(root: Any, path: tuple) -> Any:
    """(In the worker) the object a handle's path leads to from the Evolver."""
    obj = root
    for step in path:
        if step[0] == "get":
            obj = getattr(obj, step[1])
        else:
            obj = getattr(obj, step[1])(*_local(step[2], root), **_local(step[3], root))
    return obj


def _local(args: Any, root: Any) -> Any:
    """(In the worker) handles among call arguments, as the objects."""
    if isinstance(args, dict):
        return {k: resolve(root, v.path) if isinstance(v, _Ref) else v for k, v in args.items()}
    return tuple(resolve(root, a.path) if isinstance(a, _Ref) else a for a in args)


def answer(root: Any, message: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """(In the worker) carry out one request; the reply without warnings."""
    op, path = message[0], message[1]
    obj = resolve(root, path)
    if op == "get":
        result = getattr(obj, message[2])
    elif op == "set":
        setattr(obj, message[2], message[3])
        return ("ok", None)
    elif op == "call":
        result = getattr(obj, message[2])(*_local(message[3], root), **_local(message[4], root))
    elif op == "list":
        return ("ok", list(obj))
    elif op == "run":
        import importlib
        module, _, name = message[2].rpartition(".")
        result = getattr(importlib.import_module(module), name)(root, *message[3], **message[4])
    else:
        raise ValueError(f"unknown request {op!r}")
    if type(result).__name__ in _VIEWS and type(result).__module__.startswith("pysurfaceevolver"):
        return ("view", type(result).__name__)
    return ("ok", result)
