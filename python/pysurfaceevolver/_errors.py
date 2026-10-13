"""Errors and warnings, and helpers for command text."""

from __future__ import annotations

import re
from typing import Any, List, Optional

import numpy as np

from . import _core


def _rebuilt(cls: Any, args: tuple, state: dict) -> BaseException:
    error = cls.__new__(cls)
    error.args = args
    error.__dict__.update(state)
    return error


class EvolverError(RuntimeError):
    """An error reported by Surface Evolver. The surface is still usable.

    ``command`` is the command text that failed (set by :meth:`Evolver.command`).
    """

    def __init__(self, message: str, errnum: int = 0, output: str = ""):
        super().__init__(message)
        self.errnum = errnum
        self.output = output
        self.command: Optional[str] = None

    def __reduce__(self):
        # with errnum, output, ... (from a worker process, Evolver(process=True))
        return (_rebuilt, (type(self), self.args, self.__dict__))


# Mistakes that show up as Evolver syntax errors, with what to do instead
_COMMAND_HINTS = [
    (re.compile(r"\bnp\.\w+\("),
     "the command contains a numpy number's repr (e.g. 'np.float64(0.5)'): format "
     "float(x) instead, or set the value through the API (ev.body(i).target = x, "
     "ev.parameters[name] = x)"),
    (re.compile(r"(?<![\w.])(nan|inf)(?![\w(])", re.IGNORECASE),
     "the command contains nan or inf: a value was not finite"),
]


def _num(value: Any, what: str = "value") -> str:
    """A number for an Evolver command: exact, and plain for numpy scalars."""
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(f"{what} must be finite, got {x}")
    return repr(x)


def _command_hints(text: str) -> List[str]:
    return [hint for pattern, hint in _COMMAND_HINTS if pattern.search(text)]


class EvolverExit(EvolverError):
    """Evolver tried to exit, for example because of the ``quit`` command.

    The process keeps running and the engine is still usable.
    """

    def __init__(self, code: int, output: str = ""):
        super().__init__(
            f"Surface Evolver tried to exit (code {code}); the engine is still usable",
            code, output)
        self.code = code


class EvolverFatalError(EvolverError):
    """An unrecoverable Evolver error. Load a new datafile before going on."""


class InvalidSurfaceError(EvolverError):
    """There is no valid surface: the last datafile failed to load, or an
    unrecoverable error happened. Only :meth:`Evolver.load` works until a
    datafile loads successfully."""


EvolverBusyError = _core.EvolverBusyError
EvolverBusyError.__module__ = "pysurfaceevolver"
EvolverBusyError.__doc__ = """Evolver is running another call: from another thread (and
``pyse.busy_timeout`` is None, or ran out), or the running call itself (a
re-entrant call from a callback). A subclass of RuntimeError."""


class EvolverWarning(UserWarning):
    """A warning printed by Surface Evolver."""


class SharedEngineWarning(UserWarning):
    """An Evolver handle was used after another handle loaded a different
    surface: every handle is a view of the one engine in the process."""


class UnstableEquilibriumWarning(UserWarning):
    """relax() converged to an equilibrium that some motion makes lower in energy
    (a saddle): see :meth:`Evolver.stability`."""

# eval() and values() run `[foreach TYPE do] printf "@pyse@%.17g\n", (EXPR)`
# (see pyse_api.c); show just EXPR when Evolver echoes that line in an error.
_CAPTURE_WRAPPER = re.compile(
    r'(?:foreach \w+ do )?printf "@pyse@%\.17g\\n", \((.*?)\)?$', re.MULTILINE)


def _hide_capture_wrapper(error: EvolverError) -> EvolverError:
    message = str(error)
    if "@pyse@" not in message:
        return error
    cleaned = _CAPTURE_WRAPPER.sub(r"\1", message)
    return type(error)(cleaned, error.errnum, error.output)
