"""The worker process of ``Evolver(process=True)``: an engine of its own,
serving the requests of the handle in the parent process (see _remote.py).

    python -m pysurfaceevolver._worker FD     (FD: the socket to the parent)
"""

from __future__ import annotations

import os
import pickle
import socket
import sys
import traceback
import warnings


class _Forward:
    """Text written here goes to the parent, printed there."""

    def __init__(self, conn, stream: int):
        self.conn, self.stream = conn, stream

    def write(self, text: str) -> int:
        if text:
            self.conn.send(pickle.dumps(("out", self.stream, text)))
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _sendable(reply: tuple) -> bytes:
    """The reply pickled; a result that doesn't pickle as a list (a keys
    view) or as an error."""
    try:
        return pickle.dumps(reply)
    except Exception as e:  # noqa: BLE001
        kind, *rest = reply
        if kind == "ok":
            try:
                return pickle.dumps(("ok", list(rest[0])) + tuple(rest[1:]))
            except Exception:  # noqa: BLE001
                pass
        return pickle.dumps(("error", f"the result can't be sent back: {e!r}", "", rest[-1]))


def main() -> None:
    from pysurfaceevolver._remote import Channel, answer
    conn = Channel(socket.socket(fileno=int(sys.argv[1])))
    # Ctrl-C at a terminal reaches the parent only, which passes it on once
    os.setpgid(0, 0)
    sys.stdout = _Forward(conn, 0)      # prints (echo=True, callbacks) to the parent
    sys.stderr = _Forward(conn, 1)
    import pysurfaceevolver as pyse
    ev = pyse.Evolver()
    conn.send(pickle.dumps(("ready", os.getpid())))
    while True:
        try:
            data = conn.recv()
        except (EOFError, OSError):
            return
        if not data:
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")   # the parent filters them
            try:
                message = pickle.loads(data)
                cwd = message[0]
                if cwd != os.getcwd():
                    os.chdir(cwd)           # relative paths as in the parent
                reply = answer(ev, message[1:])
            except BaseException as e:  # noqa: BLE001 - every failure goes back
                trace = traceback.format_exc()
                try:
                    pickle.loads(pickle.dumps(e))
                    reply = ("error", e, trace)
                except Exception:  # noqa: BLE001
                    reply = ("error", f"{type(e).__name__}: {e}", trace)
        found = [(w.category, str(w.message)) for w in caught]
        conn.send(_sendable(reply + (found,)))


if __name__ == "__main__":
    main()
