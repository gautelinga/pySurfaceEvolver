"""High-level Python interface to Surface Evolver."""

from __future__ import annotations

import contextlib
import functools
import os
import re
import sys
import tempfile
import threading
import warnings
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, Generator, List,
                    NamedTuple, Optional, Tuple, Union)

import numpy as np

from . import _core, _html
from ._build import Body, make_datafile
from ._mesh import Bodies, BodySurface, Mesh, MeshQuality, Quantity, is_watertight

if TYPE_CHECKING:
    from ._viz import LiveView

__all__ = [
    "Evolver",
    "IterationResult",
    "Parameters",
    "Snapshot",
    "EvolverBusyError",
    "EvolverError",
    "EvolverExit",
    "EvolverFatalError",
    "InvalidSurfaceError",
    "EvolverWarning",
]


@contextlib.contextmanager
def _threads_for_call(threads: Optional[int]) -> Generator[None, None, None]:
    """Use `threads` threads for one call (None: leave the setting alone)."""
    if threads is None:
        yield
        return
    previous = _core.thread_setting()
    _core.set_threads(int(threads))
    try:
        yield
    finally:
        _core.set_threads(previous)


class _Coords(NamedTuple):
    coords: np.ndarray
    stamp: int                    # Evolver's topology stamp when taken


class EigenCounts(NamedTuple):
    """Hessian eigenvalues below, at and above a shift; see :meth:`Evolver.eigen_counts`."""

    negative: int
    zero: int
    positive: int


class EvolverError(RuntimeError):
    """An error reported by Surface Evolver. The surface is still usable.

    ``command`` is the command text that failed (set by :meth:`Evolver.command`).
    """

    def __init__(self, message: str, errnum: int = 0, output: str = ""):
        super().__init__(message)
        self.errnum = errnum
        self.output = output
        self.command: Optional[str] = None


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


class UnstableEquilibriumWarning(UserWarning):
    """relax() converged to an equilibrium that some motion makes lower in energy
    (a saddle): see :meth:`Evolver.stability`."""


@dataclass(frozen=True)
class Stability:
    """See :meth:`Evolver.stability`."""

    stable: bool
    negative: int             # directions that lower the energy
    lowest: np.ndarray        # the eigenvalues nearest zero, ascending
    threshold: float          # eigenvalues below this counted as negative


@dataclass(frozen=True, repr=False)
class Health:
    """See :meth:`Evolver.health`. Gaps are in units of the median edge length."""

    residual: float
    stable: Optional[bool]          # None: not checked
    negative_modes: Optional[int]
    angle_min: float                # smallest facet angle, degrees
    skinny: int                     # facets with an angle below 5 degrees
    wall_gap: float                 # nearest approach to a constraint a vertex isn't on
    wall: Optional[int]             # that constraint's number
    crossed: Dict[int, int]         # constraint -> vertices on its far side
    self_gap: float                 # nearest approach of the surface to itself (inf: none under 1/2)
    junctions: int                  # junctions that break Plateau's laws (see Evolver.health)
    target_gap: float               # largest relative gap of a fixed volume or quantity to its target
    issues: List[str]               # what looks wrong, in words (empty: nothing)

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        """``"ok"``, or the issues in a line."""
        return "ok" if self.ok else "; ".join(self.issues)

    def __repr__(self) -> str:
        gaps = (f"wall gap {self.wall_gap:.2g} (constraint {self.wall})"
                if self.wall is not None else "no wall nearby")
        return (f"Health({self.summary()}: residual {self.residual:.2g}, smallest angle "
                f"{self.angle_min:.3g} degrees, {gaps}, self gap {self.self_gap:.2g})")


_REPRESENTATIONS = {1: "string", 2: "soapfilm", 3: "simplex"}
_MODELS = {1: "linear", 2: "quadratic", 3: "lagrange"}
_ELEMENT_TYPES = {
    "vertex": _core.VERTEX, "vertices": _core.VERTEX,
    "edge": _core.EDGE, "edges": _core.EDGE,
    "facet": _core.FACET, "facets": _core.FACET,
    "body": _core.BODY, "bodies": _core.BODY,
}
_QUANTITY_KINDS = {0: "energy", 1: "fixed", 2: "info", 3: "conserved"}
_ELEMENT_NAMES = {_core.VERTEX: "vertex", _core.EDGE: "edge",
                  _core.FACET: "facet", _core.BODY: "body"}
_STATEMENTS_PER_COMMAND = 2000


def _element_type(element: str) -> int:
    try:
        return _ELEMENT_TYPES[element]
    except KeyError:
        raise ValueError(
            f"unknown element type {element!r}; "
            "use 'vertex', 'edge', 'facet' or 'body'") from None


def _import_meshio():
    try:
        import meshio
    except ImportError:
        raise ImportError(
            "mesh file input/output needs meshio: pip install 'pysurfaceevolver[io]'") from None
    return meshio


def _target_format(path: str, file_format: Optional[str]) -> "tuple[Optional[str], str]":
    """meshio file format and native-cell flavor for writing path.

    meshio would read ".msh" as ANSYS; here it means Gmsh.
    """
    meshio = _import_meshio()
    ext = os.path.splitext(path)[1].lower()
    if file_format is None:
        if ext == ".msh":
            # Gmsh 2.2: meshio's 4.1 output lacks the $Entities section that
            # Gmsh itself needs to open the file
            file_format = "gmsh22"
        elif ext not in meshio.extension_to_filetypes:
            raise ValueError(f"unknown mesh file extension {ext!r}; pass file_format")
    fmt = file_format or meshio.extension_to_filetypes[ext][0]
    if fmt.startswith("gmsh"):
        flavor = "gmsh"
    elif fmt in ("vtu", "vtk"):
        flavor = "vtk"
    elif fmt == "xdmf":
        flavor = "xdmf"
    else:
        flavor = "linear"
    return file_format, flavor


@dataclass
class IterationResult:
    """What :meth:`Evolver.iterate` or :meth:`Evolver.relax` did.

    ``energy``, ``area``, ``scale`` and ``level`` (the refinement level, from
    0) have one entry per gradient iteration. ``converged`` and
    ``newton_steps`` are set by :meth:`Evolver.relax`.
    """

    energy: np.ndarray
    area: np.ndarray
    scale: np.ndarray
    output: str
    converged: Optional[bool] = None
    newton_steps: int = 0
    level: Optional[np.ndarray] = None
    residual: float = float("nan")    # Evolver.residual() at the end (relax)
    stable: Optional[bool] = None      # relax(): checked once converged (None: not checked)
    negative_modes: Optional[int] = None   # relax(): directions that lower the energy
    health: Optional["Health"] = None  # relax(): Evolver.health() at the end
    popped: int = 0                    # relax(): junctions popped (Evolver's 'o')
    collapsed: int = 0                 # relax(): collapsed edges deleted (pinch-off)
    # relax(): the residual stayed above tol, but the last gradient rounds ended
    # with the energy settled (surfaces without a Hessian; a crystalline energy,
    # which then counts as converged)
    settled: Optional[bool] = None

    def __repr__(self) -> str:
        n = len(self.energy)
        steps = f"{n} gradient step{'' if n == 1 else 's'}"
        if self.converged is None:          # iterate()
            energy = f", energy {self.energy[0]:.10g} -> {self.energy[-1]:.10g}" if n else ""
            return f"IterationResult({steps}{energy})"
        settled = " (energy settled)" if self.settled else ""
        parts = [("converged" if self.converged else "not converged") + settled
                 + f" after {steps} and {self.newton_steps} Newton",
                 f"residual {self.residual:.2g}"]
        if self.popped or self.collapsed:
            parts.append(f"{self.popped} popped, {self.collapsed} collapsed edges deleted")
        if self.stable is not None:
            parts.append("stable" if self.stable else
                         f"unstable ({self.negative_modes} negative modes)")
        if self.health is not None:
            parts.append(f"health {self.health.summary()}")
        return f"IterationResult({', '.join(parts)})"


class _Trace:
    """What the gradient steps of an iterate() or relax() call did."""

    def __init__(self) -> None:
        self.energy: List[float] = []
        self.area: List[float] = []
        self.scale: List[float] = []
        self.level: List[int] = []
        self.output: List[str] = []
        self.settled = False        # the last gradient phase ended with the energy settled

    def step(self, ev: "Evolver") -> None:
        self.output.append(ev.command("g 1"))
        self.energy.append(_core.total_energy())
        self.area.append(_core.total_area())
        self.scale.append(ev.eval("scale"))

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


# mesh() result for the engine's current surface version (see _core.surface_version)
_mesh_cache: "tuple[int, Mesh] | None" = None
# relax(): whether the last Newton-first attempt on this surface failed (the
# next call then skips it; reset when a surface is loaded)
_newton_first_failed = False
# relax(): failed last-resort attempts in a row on this surface, and the calls
# still to skip before the next attempt (reset when a surface is loaded)
_collapse_failures = 0
_collapse_skip = 0
_mesh_cache_lock = threading.Lock()


def _read_only(*arrays: Optional[np.ndarray]) -> None:
    for a in arrays:
        if a is not None:
            a.flags.writeable = False


@dataclass(frozen=True)
class Snapshot:
    """A saved surface from :meth:`Evolver.save`, as exact datafile text."""

    text: str
    datafile: str

    def write(self, path: Union[str, os.PathLike]) -> None:
        """Write the snapshot as a ``.fe`` datafile."""
        with open(path, "w") as f:
            f.write(self.text)


class Evolver:
    """A handle to the Surface Evolver engine of this process.

    Parameters
    ----------
    datafile:
        Optional path of a ``.fe`` datafile to load right away.
    echo:
        If true, print Evolver's output live while commands run. It's always
        returned from :meth:`command` either way.
    input:
        Optional callable ``input(prompt) -> str | None`` that answers
        Evolver's interactive prompts, for example in ``hessian_menu``.
        ``None`` (the default) answers every prompt with end-of-file.

    Notes
    -----
    Surface Evolver keeps its state in C globals, so there is one engine per
    process, and every ``Evolver`` object is a handle to it: they all see the
    same surface, and loading through one changes it for all. ``echo`` and
    ``input`` belong to each handle. Use :meth:`save` and :meth:`restore` to
    keep a surface around, and separate processes to work on several at
    once.

    Calls are serialized: a call made while another thread is running one
    raises ``RuntimeError`` instead of waiting.

    Ctrl-C during a call (from the main thread) stops Evolver at the next
    iteration or statement; pressing it again aborts the operation. Both
    raise ``KeyboardInterrupt``.
    """

    _no_hessian: bool = False      # set by _newton(): no Hessian for this surface
    _keep_mesh: bool = False       # relax(): no tidying (elastic, crystalline)

    def __init__(
        self,
        datafile: Optional[str | os.PathLike] = None,
        *,
        echo: bool = False,
        input: Optional[Callable[[str], Optional[str]]] = None,
    ):
        self.echo = echo
        self.input = input
        self._call(_core.initialize)
        if datafile is not None:
            self.load(datafile)

    # ------------------------------------------------------------------
    # Running Evolver

    def _call(self, fn, *args):
        """Run a guarded _core call; return (output, result) or raise.

        The output buffers are local to the call and the callbacks travel
        with it, so concurrent calls from other threads can't mix them up.
        """
        out: list = []
        err: list = []
        echo = self.echo

        def on_output(stream: int, text: str) -> None:
            (err if stream else out).append(text)
            if echo:
                target = sys.stderr if stream else sys.stdout
                target.write(text)
                target.flush()

        result = fn(*args, out=on_output, input=self.input,
                    sigint=threading.current_thread() is threading.main_thread())
        output = "".join(out)

        for message in result.warnings:
            warnings.warn(message.strip(), EvolverWarning, stacklevel=3)

        status = result.status
        if status == _core.OK:
            return output, result
        if status == _core.INTERRUPT:
            raise KeyboardInterrupt("Surface Evolver operation interrupted")
        if status == _core.EXIT:
            raise EvolverExit(result.exit_code, output)
        if status == _core.BUSY:
            raise EvolverBusyError("Surface Evolver is busy (calls are not re-entrant)")
        if status == _core.INVALID:
            raise InvalidSurfaceError(result.message.strip(), result.errnum, output)
        # Evolver's own error printout has the most context (input line, etc.)
        printed = "".join(err).strip()
        message = printed or f"ERROR {result.errnum}: {result.message.strip()}"
        if status == _core.FATAL:
            raise EvolverFatalError(
                message + "\nThe surface is now invalid; load a new datafile.",
                result.errnum, output)
        raise EvolverError(message, result.errnum, output)

    def load(self, datafile: str | os.PathLike) -> str:
        """Load a ``.fe`` datafile, replacing the current surface.

        Evolver looks for the file in the current directory, then in the
        directories listed in the ``EVOLVERPATH`` environment variable.
        Returns the output printed while loading.

        If the datafile has errors, this raises :class:`EvolverError`, and
        there is no valid surface until a datafile loads successfully.
        """
        global _newton_first_failed, _collapse_failures, _collapse_skip
        _newton_first_failed = False
        _collapse_failures = _collapse_skip = 0
        path = os.fspath(datafile)
        try:
            output, _ = self._call(_core.load, path)
        except EvolverError as e:
            if e.errnum == _core.ERR_NO_DATAFILE:
                raise FileNotFoundError(f"Cannot open datafile {path!r}") from None
            raise
        return output

    def load_string(self, text: str, name: str = "surface.fe") -> str:
        """Load a datafile given as a string.

        The text goes to a temporary file first, so ``#include`` and ``read``
        paths are resolved from the current directory.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            with open(path, "w") as f:
                f.write(text)
            return self.load(path)

    def command(self, text: str) -> str:
        """Run Evolver commands, such as ``"g 10; r; hessian"``.

        Returns everything Evolver printed to standard output. Raises
        :class:`EvolverError` if Evolver reports an error.
        """
        try:
            output, _ = self._call(_core.command, text)
        except EvolverError as error:
            if isinstance(error, (EvolverExit, InvalidSurfaceError)):
                raise
            error.command = text
            hints = _command_hints(text)
            if hints:
                error.args = (error.args[0] + "".join(f"\nHint: {h}" for h in hints),)
            raise
        return output

    __call__ = command

    def eval(self, expr: str) -> float:
        """Evaluate a numeric Evolver expression, like ``"body[1].volume"``.

        Nothing gets defined in Evolver's symbol table. Results are exact
        doubles, including infinities and NaN.
        """
        try:
            _, result = self._call(_core.eval, expr)
        except EvolverError as e:
            raise _hide_capture_wrapper(e) from None
        return result.value

    def values(self, element: str, expr: str) -> np.ndarray:
        """Evaluate an expression for every element of one type.

        ``element`` is ``"vertex"``, ``"edge"``, ``"facet"`` or ``"body"``.
        Inside ``expr``, attributes refer to the element at hand, for example
        ``ev.values("vertex", "x^2 + y^2")`` or ``ev.values("facet", "area")``.

        The result is aligned with the rows of :meth:`mesh` (``vertices``,
        ``edges``, ``facets``) and :meth:`bodies`.
        """
        element_type = _element_type(element)
        try:
            _, result = self._call(_core.values, element_type, expr)
        except EvolverError as e:
            raise _hide_capture_wrapper(e) from None
        return result.data

    def __getitem__(self, name: str) -> float:
        """``ev["total_area"]`` evaluates a variable or expression."""
        return self.eval(name)

    def __setitem__(self, name: str, value: float) -> None:
        """``ev["gravity"] = 0.5`` assigns a variable or parameter."""
        self.command(f"{name} := {float(value)!r}")

    # ------------------------------------------------------------------
    # Operations

    def iterate(self, n: int = 1, *, callback: Optional[Callable[["Evolver", int], Any]] = None,
                every: int = 1) -> IterationResult:
        """Run n gradient-descent iterations (Evolver's ``g``).

        Returns the energy, area and scale factor after each iteration.
        ``callback(ev, i)`` is called after every ``every``-th iteration and
        after the last one, for example to update a :class:`LiveView`.
        Ctrl-C stops between iterations with ``KeyboardInterrupt``.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        trace = _Trace()
        for i in range(1, n + 1):
            trace.step(self)
            if callback is not None and (i % every == 0 or i == n):
                callback(self, i)
        return IterationResult(np.array(trace.energy), np.array(trace.area),
                               np.array(trace.scale), "".join(trace.output))

    def relax(self, tol: float = 1e-8, max_iter: int = 1000, *, levels: int = 0,
              newton: Optional[int] = None, stability: Optional[bool] = None,
              adapt: Union[bool, float] = False,
              cg: Union[bool, str] = "auto", tidy: int = 10, energy_tol: Optional[float] = None,
              window: int = 3, seek: Optional[bool] = None, pinch: bool = True,
              undo_if: Optional[Callable[["Evolver"], bool]] = None,
              callback: Optional[Callable[["Evolver", int], Any]] = None,
              every: int = 1, threads: Optional[int] = None) -> IterationResult:
        """Relax the surface to an equilibrium: gradient steps that keep the
        mesh in shape, then safeguarded Newton steps, until :meth:`residual`
        is below ``tol``. The defaults are meant to work as they are; the
        result says whether it ``converged``, whether the equilibrium is
        ``stable``, and how the surface looks (``health``).

        Common options: ``levels=n`` refines and relaxes again n times;
        ``newton=0`` uses gradient steps only; ``stability=False`` skips the
        stability check (about two factorizations); ``adapt=True`` (linear
        model) then refines where the surface curves too much for its edges
        and coarsens where it has flattened (:meth:`adapt`; or give the
        turning angle in degrees, default 15), up to three passes or four
        times the facets the call began with. ``callback(ev, i)`` runs every
        ``every``-th gradient step and at the end of each level; ``threads``
        sets the threads for this call only (see
        :func:`pysurfaceevolver.threads_limit`).

        How it works:

        * In the quadratic and Lagrange models (a final stage from a mesh
          settled in the linear model) with ``levels=0``, five gradient steps
          and then safeguarded Newton steps come first (as the manual's
          ``g 5; hessian``), Newton as long as each step cuts the residual by
          10%; if that converges, that is all. If not, the
          surface goes back and the steps below follow; right after such a
          miss, the next call skips the attempt.
        * Gradient steps (Evolver's ``g``) come in rounds of ``tidy`` steps,
          each followed by equiangulation and vertex averaging (``u; V``;
          only ``V`` for strings and the Lagrange model, only ``u`` in the
          simplex representation; ``tidy=0``: none), until the energy changes by less than
          ``energy_tol`` (relative) over ``window`` rounds, or ``max_iter``
          steps. ``energy_tol`` is 1e-5 when Newton follows (it finishes faster
          and more reliably, and long gradient runs let slow mesh drift grow),
          1e-9 with ``newton=0``. When Newton follows, a round that more than
          doubles the residual from its lowest so far is undone and Newton
          takes over (on a coarse mesh a contact line can collapse within a
          few steps).
        * Then up to ``newton`` safeguarded Newton steps (see :meth:`newton`;
          ``seek`` and ``undo_if`` as there), until the residual is below
          ``tol`` or stops falling.
        * If that leaves the surface unconverged, ``cg="auto"`` tries once more
          with conjugate gradients (far faster on flat or stiff shapes, such as
          a drop at a 10 degree contact angle, but they wear the mesh down
          where contact lines travel), then Newton, with up to 200 of the
          steps the level left of ``max_iter``; kept only if that converges. ``cg=True`` uses
          conjugate gradients throughout, ``cg=False`` never (your own
          ``conj_grad`` setting is restored after).
        * Once converged (linear model), edges leaving a contact line that are
          longer than 1.5 times the contact-line edges there are split and the
          surface relaxed again, kept only if that converges: a shrinking
          contact circle (a drop at a high contact angle) would otherwise end
          up smaller than the facets next to it.
        * Then (linear model, films and strings) junctions that break
          Plateau's laws are popped (Evolver's ``o``: four films on an edge,
          four strings at a point, ...) and the surface relaxed again, up to
          five times. With ``pinch`` (default), if it is still unconverged, a
          last resort: up to 300 gradient steps without tidying (which can
          hold a shrinking neck open), then edges that have collapsed below a
          20th of the median edge are deleted (Evolver's ``t``: a film or neck
          shrinking to nothing, as before a foam's T1 change or a pinch-off),
          junctions popped and the surface relaxed, kept only if that
          converges, or lowers the energy on a sound mesh without raising the
          residual. A converged surface
          never loses edges. After k failed attempts in a row, the next
          2**k - 1 calls skip the last resort.
          ``popped`` and ``collapsed`` count them.
        * Once converged (or within 100 times ``tol``), with ``stability``,
          :meth:`stability` sets ``stable``
          and ``negative_modes`` and warns (:class:`UnstableEquilibriumWarning`)
          at a saddle, such as a liquid column past its Rayleigh-Plateau limit:
          the solver finds saddles too, so the warning is the only sign.

        * Without a Hessian (gap energies, simplex bodies outside the Lagrange
          model, ...) the Newton steps give way to gradient rounds settled as
          with ``newton=0``, while no facet is degenerate (below 1 degree); a
          crystalline (Wulff) energy takes gradient rounds only and counts as
          converged once its energy settles, since its residual doesn't
          vanish at the Wulff shape. ``settled`` says the energy settled with
          the residual above ``tol``.

        If the energy becomes non-finite, the surface is put back to the start
        of the gradient round where it happened and :class:`EvolverError` is
        raised. A gradient round that pushes a met body volume or fixed
        quantity off its target (Evolver's projection can fail on a degenerate
        mesh) is undone and ends the rounds.

        Returns an :class:`IterationResult`: one entry per gradient step in
        ``energy``, ``area``, ``scale`` and ``level`` (adaptive passes count as
        further levels), plus ``newton_steps``, ``residual``, ``converged``,
        ``settled``, ``stable``, ``negative_modes``, ``health``, ``popped``
        and ``collapsed``.
        """
        elastic = self._elastic()
        # a membrane: no mesh tidying (it would change the physics), and Newton
        # by line search, which keeps the surface from running off; a
        # crystalline energy depends on each facet's orientation, and flipping
        # and averaging undo the facets' alignment with the crystal. Rounds of
        # ``tidy`` steps still measure convergence.
        crystalline = _core.crystalline()
        keep_mesh = elastic or crystalline
        if elastic and seek is None:
            seek = True
        # no Hessian for a crystalline energy, and its residual doesn't vanish
        # at the Wulff shape (the facets sit at corners of the energy): gradient
        # rounds until the energy settles, which counts as converged. Nor for
        # simplex bodies outside the Lagrange model, where a failed attempt
        # leaves the surface unusable.
        why = ("a crystalline energy" if crystalline else
               "simplex bodies outside the Lagrange model"
               if _core.representation() == 3 and self.model != "lagrange" else None)
        if why is not None:
            if newton or stability:
                warnings.warn(f"relax(): no Newton steps or stability check for {why} "
                              "(Evolver has no Hessian for it)", EvolverWarning, stacklevel=2)
            newton, stability = 0, False
        newton = 20 if newton is None else newton
        stability = True if stability is None else stability
        if energy_tol is None:
            energy_tol = 1e-5 if newton else _GRADIENT_ONLY_TOL
        if tol <= 0 or energy_tol <= 0 or max_iter < 1 or window < 1:
            raise ValueError("tol and energy_tol must be positive; max_iter and window at least 1")
        if tidy < 0 or levels < 0 or newton < 0:
            raise ValueError("tidy, levels and newton must be non-negative")
        if cg not in ("auto", True, False):
            raise ValueError('cg must be "auto", True or False')
        with _threads_for_call(threads), self._conj_grad(cg is True), \
                self._mesh_kept(keep_mesh):
            trace = _Trace()
            facets0 = self.counts["facets"]
            run = functools.partial(self._gradient_phase, trace, energy_tol=energy_tol,
                                    window=window, callback=callback, every=every)
            first = self._newton_first(trace, tol, newton, seek, undo_if) \
                if newton and levels == 0 and self.model != "linear" else None
            if first is not None:
                steps, residual = first
            else:
                for lev in range(levels + 1):
                    if lev:
                        self.refine()
                    run(lev, max_iter, tidy, watch=newton > 0)
                steps, residual = self._finish(trace, levels, run, max_iter, tidy, tol,
                                               newton, seek, undo_if, fallback=cg == "auto")
            if self.model == "linear" and residual < tol and not keep_mesh:
                more, residual = self._contact_pass(trace, levels + 1, run, max_iter, tidy, tol,
                                                    newton, seek, undo_if, cg == "auto", residual)
                steps += more
            if adapt and self.model == "linear" and not elastic:
                turn = 15.0 if adapt is True else float(adapt)
                first_lev = max(trace.level, default=levels) + 1
                for lev in range(first_lev, first_lev + _ADAPT_PASSES):
                    if self.counts["facets"] > _ADAPT_GROWTH*facets0 or not self.adapt(turn):
                        break
                    run(lev, max_iter, tidy, watch=newton > 0)
                    more, residual = self._finish(trace, lev, run, max_iter, tidy, tol, newton,
                                                  seek, undo_if, fallback=cg == "auto")
                    steps += more
            popped = collapsed = 0
            if self.model == "linear" and _core.representation() != 3:
                for _ in range(_TOPOLOGY_PASSES):
                    pops = self._pop()
                    if not pops:
                        break
                    popped += pops
                    lev = max(trace.level, default=levels) + 1
                    run(lev, max_iter, tidy, watch=newton > 0)
                    more, residual = self._finish(trace, lev, run, max_iter, tidy, tol, newton,
                                                  seek, undo_if, fallback=cg == "auto")
                    steps += more
                if pinch and not residual < tol and not keep_mesh:
                    found = self._collapse(trace, run, max_iter, tidy, tol, newton, seek,
                                           undo_if, cg == "auto", levels)
                    if found is not None:
                        more, residual, pops, gone = found
                        steps += more
                        popped += pops
                        collapsed += gone
            stable: Optional[bool] = None
            modes: Optional[int] = None
            # just short of tol the eigenvalues are as good as at it (a step
            # of a continuation that stops a little early still gets checked)
            if stability and residual < _STABILITY_NEAR*tol:
                try:
                    check = self.stability()
                    stable, modes = check.stable, check.negative
                except EvolverError:
                    pass                    # no Hessian for this surface
                if stable is False:
                    lowest = check.lowest[0] if len(check.lowest) else float("nan")
                    warnings.warn(
                        f"relax() {'converged to' if residual < tol else 'ended near'} an "
                        f"unstable equilibrium: {modes} direction(s) lower the energy "
                        f"(lowest eigenvalue {lowest:.3g})",
                        UnstableEquilibriumWarning, stacklevel=2)
            try:
                health: Optional[Health] = self._health(residual, stable, modes, tol)
            except EvolverError:
                health = None
            settled = trace.settled and not residual < tol
        return IterationResult(np.array(trace.energy), np.array(trace.area),
                               np.array(trace.scale), "".join(trace.output),
                               bool(residual < tol or (crystalline and settled)), steps,
                               np.array(trace.level, dtype=int), residual, stable, modes,
                               health, popped, collapsed, settled)

    def _collapse(self, trace: "_Trace", run: Callable[..., None], max_iter: int, tidy: int,
                  tol: float, newton: int, seek: Optional[bool],
                  undo_if: Optional[Callable[["Evolver"], bool]], fallback: bool,
                  levels: int) -> "tuple[int, float, int, int] | None":
        """relax()'s last resort for an unconverged surface: gradient steps
        without tidying (which can hold a shrinking neck open as a tube of
        slivers), up to a few hundred, then delete what collapsed, pop and
        relax again. Kept if that converges, or lowers the energy without
        leaving degenerate facets; else the surface and the trace go back
        (None). Returns the Newton steps,
        residual, pops and deleted edges.

        After k failed attempts in a row the next 2**k - 1 calls skip it (a
        continuation that stays unconverged pays for a few attempts, not one
        per call); a kept attempt starts the count over."""
        global _collapse_failures, _collapse_skip
        if _collapse_skip:
            _collapse_skip -= 1
            return None
        snapshot = self.save()
        energy = _core.total_energy()
        residual0 = self.residual()
        marks = (len(trace.energy), len(trace.area), len(trace.scale), len(trace.level),
                 len(trace.output))
        lev = max(trace.level, default=levels) + 1
        steps = popped = gone = 0
        residual = float("inf")
        try:
            run(lev, _COLLAPSE_STEPS, 0, energy_tol=_GRADIENT_ONLY_TOL)
            gone = self._delete_collapsed()
            if gone:
                for _ in range(_TOPOLOGY_PASSES):
                    popped += self._pop()
                    run(lev, max_iter, tidy, watch=newton > 0)
                    more, residual = self._finish(trace, lev, run, max_iter, tidy, tol, newton,
                                                  seek, undo_if, fallback=fallback)
                    steps += more
                    extra = self._delete_collapsed() if not residual < tol else 0
                    gone += extra
                    if not extra:
                        break
        except EvolverError:
            gone = 0
        # a lower energy counts only on a sound mesh no farther from
        # equilibrium: a coarse one can lower it by collapsing facets into
        # what the mesh allows (tankex from 8 facets: energy 1.77 -> 0.24,
        # residual 0.94 -> 1.38).
        try:
            sound = gone and not self.mesh_quality(5.0).angle_min < 1.0
        except EvolverError:        # a facet left without its three edges
            sound, residual = False, float("inf")
        if gone and (residual < tol or (_core.total_energy() < energy and sound
                                        and residual <= residual0)):
            _collapse_failures = 0
            return steps, residual, popped, gone
        _collapse_failures += 1
        _collapse_skip = 2**_collapse_failures - 1
        self.restore(snapshot)
        del trace.energy[marks[0]:], trace.area[marks[1]:], trace.scale[marks[2]:]
        del trace.level[marks[3]:], trace.output[marks[4]:]
        return None

    def _pop(self) -> int:
        """Evolver's 'o': pop the junctions that break Plateau's laws; how many."""
        out = self.command("o")
        return sum(int(n) for n in re.findall(r"popped:\s*(\d+)", out))

    def _delete_collapsed(self) -> int:
        """Delete the edges (not fixed) that have collapsed below a 20th of the
        median edge (Evolver's 't'): a neck or film shrinking to nothing, as
        it does before a pinch-off or a foam's T1 change; how many."""
        lengths = self.values("edge", "length")
        free = lengths[self.values("edge", "fixed") == 0]
        if len(free) == 0:
            return 0
        cut = _COLLAPSED*float(np.median(free))
        if not (free < cut).any():
            return 0
        out = self.command(f"t {_num(cut, 'length')}")
        found = re.search(r"Deleted edges:\s*(\d+)", out)
        return int(found.group(1)) if found else 0

    def _newton_first(self, trace: "_Trace", tol: float, newton: int, seek: Optional[bool],
                      undo_if: Optional[Callable[["Evolver"], bool]]) -> "tuple[int, float] | None":
        """Five gradient steps, then safeguarded Newton steps as long as each
        cuts the residual by at least 10% (up to ``newton``), before the usual
        gradient rounds (relax() in the quadratic and Lagrange models; the
        manual's 'lagrange n; g 5; hessian'): the Newton steps and residual
        if they converge, else None with the coordinates put back. Skipped
        (None) right after a failed attempt on this surface, so a run where
        Newton rarely wins pays for every other attempt only. (Not in the
        linear model: Newton moves vertices only along the normal, so over a
        continuation the mesh doesn't follow a travelling contact line the way
        gradient rounds make it, and accuracy drops.)"""
        global _newton_first_failed
        if _newton_first_failed:
            _newton_first_failed = False
            return None
        coords = self._coords()
        for _ in range(_NEWTON_FIRST_SETTLE):   # as the manual's 'g 5; hessian':
            trace.step(self)                     # new high-order nodes settle first
        trace.level += [0]*_NEWTON_FIRST_SETTLE
        # no early stop on the residual: it doesn't see an unmet volume (a new
        # target), which the Newton steps restore
        with warnings.catch_warnings():     # an attempt; far from equilibrium the
            warnings.filterwarnings("ignore", "WARNING 1825", EvolverWarning)  # Hessian is indefinite
            steps = self._newton(newton, seek, None, undo_if, trace.output)
        residual = self.residual()
        if residual < tol and self._volumes_met():
            return steps, residual
        self._put_back(coords)
        _newton_first_failed = True
        return None

    def _contact_pass(self, trace: "_Trace", lev: int, run: Callable[..., None], max_iter: int,
                      tidy: int, tol: float, newton: int, seek: Optional[bool],
                      undo_if: Optional[Callable[["Evolver"], bool]], fallback: bool,
                      residual: float) -> "tuple[int, float]":
        """After convergence (linear model): split the edges leaving a contact
        line (an edge on one facet only) that are longer than 1.5 times the
        contact-line edges at their end, and relax again, up to twice; kept
        only if that converges, else the converged surface comes back. A
        contact circle that shrinks (a drop at a high contact angle) otherwise
        ends up smaller than the facets next to it. Returns the Newton steps
        and the residual."""
        mark = _long_spokes(self.mesh(), _SPOKE_RATIO)
        if not mark.any():
            return 0, residual
        before = self.save()
        self._split_marked(mark)
        steps = 0
        for _ in range(2):
            run(lev, max_iter, tidy, watch=newton > 0)
            more, second = self._finish(trace, lev, run, max_iter, tidy, tol, newton, seek,
                                        undo_if, fallback=fallback)
            steps += more
            if second < tol:
                return steps, second
        self.restore(before)
        return 0, residual

    def _elastic(self) -> bool:
        """Whether an energy quantity uses one of Evolver's elastic methods
        (in-plane forces are physical, and the facets remember a reference
        shape that retriangulating would corrupt)."""
        _, result = self._call(_core.energy_methods)
        return not _ELASTIC_METHODS.isdisjoint(result.data)

    @contextlib.contextmanager
    def _mesh_kept(self, on: bool) -> Iterator[None]:
        """No tidying commands in the gradient rounds for a block."""
        before = self._keep_mesh
        self._keep_mesh = on
        try:
            yield
        finally:
            self._keep_mesh = before

    def _tidy_kinds(self) -> Tuple[bool, bool]:
        """Whether Evolver can equiangulate ('u') and vertex-average ('V') this
        surface: 'u' only for linear or quadratic facets (not strings, simplices
        or the Lagrange model), 'V' for all but the simplex representation;
        neither while relax() keeps the mesh (elastic, crystalline)."""
        if self._keep_mesh:
            return False, False
        rep = _core.representation()            # 1 string, 2 soapfilm, 3 simplex
        return rep == 2 and self.model != "lagrange", rep != 3

    def _split_marked(self, mark: np.ndarray) -> int:
        """Split the marked edges (aligned with mesh() rows; not no_refine ones),
        equiangulate. Returns the edges split."""
        if "pyse_mark" not in self._edge_attributes():
            self.command("define edge attribute pyse_mark integer")
        self.set_values("edge", "pyse_mark", mark.astype(float))
        before = self.counts["vertices"]
        self.command("refine edge ee where ee.pyse_mark == 1 and not ee.no_refine")
        if self._tidy_kinds()[0]:
            self.command("u")
        return self.counts["vertices"] - before

    def _volumes_met(self, rel: float = 1e-6) -> bool:
        """Whether every body with a fixed volume is at its target."""
        b = self.bodies()
        if not len(b.ids):
            return True
        off = np.abs(b.volume - b.target)[b.fixed]
        return bool((off <= rel*np.maximum(np.abs(b.target[b.fixed]), 1e-300)).all())

    def _finish(self, trace: "_Trace", lev: int, run: Callable[..., None], max_iter: int,
                tidy: int, tol: float, newton: int, seek: Optional[bool],
                undo_if: Optional[Callable[["Evolver"], bool]], fallback: bool) -> "tuple[int, float]":
        """Newton steps, then (``fallback``) one conjugate-gradient pass and
        Newton again if still unconverged, kept only if that converges. The
        pass averages vertices but doesn't equiangulate, so the coordinates
        taken before it can undo it. Returns the Newton steps kept and the
        residual."""
        steps = self._newton(newton, seek, tol, undo_if, trace.output) if newton else 0
        if newton and self._no_hessian:
            # no Newton for this surface: plain gradient rounds instead, settled
            # as with newton=0 (the residual watch, there for Newton, is off)
            # only on a sound mesh: on degenerate facets a long gradient run
            # slides into what the mesh allows rather than the shape (tankex
            # from 8 facets: energy 1.6 -> 0.03)
            budget = max_iter - trace.level.count(lev)
            if budget > 0 and not self.mesh_quality(5.0).angle_min < 1.0:
                run(lev, budget, tidy, energy_tol=_GRADIENT_ONLY_TOL, sound=True)
            return 0, self.residual()
        residual = self.residual()
        budget = min(max_iter - trace.level.count(lev), _FALLBACK_STEPS)
        if not (fallback and newton and budget > 0 and np.isfinite(residual)
                and not residual < tol):
            return steps, residual
        before = self._coords()
        settled = trace.settled
        with self._conj_grad(True):
            run(lev, budget, tidy, watch=True, average_only=True)
        more = self._newton(newton, seek, tol, undo_if, trace.output) \
            if np.isfinite(_core.total_energy()) else 0
        second = self.residual()
        if second < tol or not self._put_back(before):   # (not: the topology changed)
            return steps + more, second
        trace.settled = settled             # the pass is undone: as the rounds before it
        return steps, residual

    @contextlib.contextmanager
    def _conj_grad(self, on: bool) -> Iterator[None]:
        """Conjugate gradients on (or off) for a block; the setting restored after."""
        before = self.eval("conj_grad") != 0
        if before != on:
            self.command(f"conj_grad {'on' if on else 'off'}")
        try:
            yield
        finally:
            if before != on and _core.surface_valid():
                self.command(f"conj_grad {'on' if before else 'off'}")

    def _edge_attributes(self) -> List[str]:
        try:
            self.eval("max(edge, pyse_mark)")
            return ["pyse_mark"]
        except EvolverError:
            return []

    def _gradient_phase(self, trace: "_Trace", lev: int, max_iter: int, tidy: int, *,
                        energy_tol: float, window: int,
                        callback: Optional[Callable[["Evolver", int], Any]], every: int,
                        watch: bool = False, average_only: bool = False,
                        sound: bool = False) -> None:
        """Gradient steps until the energy settles (see relax()). With tidy=k,
        'u; V' (``average_only``: 'V') every k steps and the energy compared
        round to round. With ``watch`` (Newton follows), also the residual at
        the end of each round, before the tidying (which disturbs it): a round
        that more than doubles it from its lowest is undone and ends the
        phase. The coordinates at the start of each round also undo a round
        whose energy turns non-finite (then EvolverError), or one that pushes
        a met volume or quantity off its target (which ends the phase). With
        ``sound``, the phase ends once a round leaves degenerate facets
        (below 1 degree)."""
        previous = _core.total_energy()
        quiet = 0
        best = np.inf
        first = len(trace.energy)
        watch = watch and tidy > 0
        start = self._coords()
        gap, _ = self._target_gap()
        for i in range(1, max_iter + 1):
            trace.step(self)
            if callback is not None and i % every == 0:
                callback(self, i)
            if not np.isfinite(trace.energy[-1]):
                bad = trace.energy[-1]
                where = ("the surface is back where that round began" if self._put_back(start)
                         else "its topology changed during the round, so it stays as it is")
                raise EvolverError(f"relax(): the energy became {bad} during gradient steps; "
                                   + where)
            if tidy and i % tidy:
                continue                  # mid-round: no convergence check
            # on a degenerate mesh Evolver's volume projection can fail without
            # an error: a round that pushes a volume off its target is undone
            # (a new target, still being approached, only counts if it grows)
            now, _ = self._target_gap()
            if now > _TARGET_LOST and now > 2*gap:
                if tidy and self._put_back(start):
                    trace.energy[-1] = _core.total_energy()
                    trace.area[-1] = _core.total_area()
                break
            gap = min(gap, now)
            if watch:
                res = self.residual()
                if res > 2*best:            # the round went wrong: undo it, hand over
                    self._put_back(start)       # (unless autopop etc. changed the topology)
                    trace.energy[-1] = _core.total_energy()
                    trace.area[-1] = _core.total_area()
                    break
                best = min(best, res)
            if tidy:
                equi, avg = self._tidy_kinds()
                cmd = "; ".join(c for c, on in (("u", equi and not average_only), ("V", avg)) if on)
                if cmd:
                    trace.output.append(self.command(cmd))
                trace.energy[-1] = _core.total_energy()
                trace.area[-1] = _core.total_area()
                start = self._coords()
                if sound and self.mesh_quality(5.0).angle_min < 1.0:
                    break
            change = abs(trace.energy[-1] - previous) / max(1.0, abs(trace.energy[-1]))
            previous = trace.energy[-1]
            quiet = quiet + 1 if change < energy_tol else 0
            if quiet >= window:
                break
        # vertex averaging moves volumes a little (on a degenerate mesh a lot),
        # and the next round's steps would project them back: at the end, one
        # step does
        if tidy and len(trace.energy) > first and self._target_gap()[0] > _TARGET_LOST:
            trace.step(self)
        trace.settled = quiet >= window
        steps = len(trace.energy) - first
        trace.level += [lev]*steps
        if callback is not None and steps % every != 0:
            callback(self, steps)

    def newton(self, steps: int = 1, *, seek: Optional[bool] = None, tol: Optional[float] = None,
               undo_if: Optional[Callable[["Evolver"], bool]] = None,
               threads: Optional[int] = None) -> int:
        """Up to ``steps`` safeguarded Newton steps.

        Each step is a plain Newton step (``hessian``), or, if that one is
        rejected, one with a line search (``hessian_seek``); ``seek=True`` or
        ``False`` uses only the one kind. A step is rejected (undone) if it
        makes the energy or the :meth:`residual` non-finite, raises both the
        energy and the residual, raises the energy by more than 10% plus what
        meeting the volume and quantity targets costs (pressure times the
        change), or makes ``undo_if(ev)`` true; a step that brings an unmet
        target (say a new body volume) at least halfway may raise both. With
        no step accepted, Newton stops. It also stops when neither the
        residual nor an unmet target is improving, or, with ``tol``, once the
        residual is below ``tol`` with the targets met.

        Newton converges to the equilibrium nearest in its own sense, saddles
        included: after a large jump of a target (a volume made 16 times
        larger in one go) it can end at an unstable one. Check with
        :meth:`stability`, or let :meth:`relax` (gradient steps first) or a
        continuation in smaller steps take such jumps.
        Returns the number of steps kept. (:meth:`hessian` takes one step with
        no checks.)
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        with _threads_for_call(threads):
            return self._newton(steps, seek, tol, undo_if, [])

    def _newton(self, steps: int, seek: Optional[bool], tol: Optional[float],
                undo_if: Optional[Callable[["Evolver"], bool]], output: List[str]) -> int:
        """Safeguarded Newton steps; see newton()."""
        kept = 0
        res0 = self.residual()
        # the residual doesn't see an unmet volume or quantity target (a new
        # one): restoring it is progress, and its cost an allowed energy rise
        gap0, cost = self._target_gap()
        if tol is not None and res0 < tol and gap0 <= _TARGET_LOST:
            return 0
        kinds = ["hessian_seek"] if seek is True else ["hessian"] if seek is False \
            else ["hessian", "hessian_seek"]
        self._no_hessian = False
        for step in range(steps):
            accepted = False
            failed = 0
            for kind in kinds:
                # a Newton step moves vertices only: their coordinates undo it
                # (a dump would cost several Newton steps on a large surface)
                snapshot = self._coords()
                e0 = _core.total_energy()
                try:
                    output.append(self.command(kind))
                except EvolverError:
                    self._put_back(snapshot)
                    failed += 1
                    continue
                e1 = _core.total_energy()
                res1 = self.residual()
                gap1, _ = self._target_gap()
                closer = gap0 > _TARGET_LOST and gap1 < 0.5*gap0
                # with constraints the energy alone is no merit (restoring a
                # volume can cost energy): reject what got worse on both, or
                # raised the energy by more than a volume correction could (far
                # from equilibrium a step can run off to a tangled surface
                # whose residual is small)
                bad = (not np.isfinite(e1) or not np.isfinite(res1)
                       or (e1 > e0 + 1e-12*max(1.0, abs(e0)) and res1 > 1.1*res0
                           and not closer)
                       or e1 > e0 + _NEWTON_MAX_RISE*abs(e0) + cost
                       or (undo_if is not None and undo_if(self)))
                if bad:
                    self._put_back(snapshot)
                    continue
                accepted = True
                break
            if not accepted:
                # no step at all: this surface has no Hessian (gap energies,
                # simplex volumes outside the Lagrange model, ...)
                self._no_hessian = step == 0 and failed == len(kinds)
                break
            kept += 1
            progress = res1 < 0.9*res0 or closer
            res0 = res1
            gap0, cost = self._target_gap()
            if tol is not None and res1 < tol and gap0 <= _TARGET_LOST:
                break
            if not progress:
                break
        return kept

    def _target_gap(self) -> "tuple[float, float]":
        """How far the fixed body volumes and fixed quantities are from their
        targets (the largest gap, relative to the target; absolute for a
        target of 0), and what closing the gaps costs in energy to first
        order (the sum of |pressure * gap|)."""
        gaps: List[float] = []
        cost = 0.0
        b = self.bodies()
        if len(b.ids) and b.fixed.any():
            target = b.target[b.fixed]
            off = target - b.volume[b.fixed]
            gaps += list(np.abs(off)/np.where(target != 0, np.abs(target), 1.0))
            cost += float(np.abs(b.pressure[b.fixed]*off).sum())
        for q in self.quantities().values():
            if q.kind == "fixed":
                gaps.append(abs(q.target - q.value)/(abs(q.target) if q.target else 1.0))
                cost += abs(q.pressure*(q.target - q.value))
        return max(gaps, default=0.0), cost

    def _coords(self) -> "_Coords":
        """The vertex coordinates, to undo steps that only move vertices."""
        return _Coords(self.vertices, _core.topology_stamp())

    def _put_back(self, snapshot: "_Coords") -> bool:
        """Restore coordinates from :meth:`_coords`; False (and nothing done)
        if the topology changed since, as Evolver's autopop and autochop do
        during gradient steps."""
        if snapshot.stamp != _core.topology_stamp() \
                or len(snapshot.coords) != _core.count(_core.VERTEX):
            return False
        self.vertices = snapshot.coords
        return True

    def remesh(self, target: Optional[float] = None, *, max_edge: Optional[float] = None,
               min_edge: Optional[float] = None, equiangulate: bool = True,
               average: bool = False, protect=None,
               keep_boundary: bool = True) -> Dict[str, int]:
        """Even out edge lengths: delete edges shorter than ``min_edge``
        (Evolver's ``t``), split edges longer than ``max_edge`` (``refine edge
        where length > max_edge``), then equiangulate (``u``) and, with
        ``average=True``, average the vertices (``V``). Deleting first: merging
        vertices lengthens the edges around them. One call splits each long
        edge once and doesn't delete every short edge (Evolver refuses unsafe
        merges); call it every few steps rather than expecting one call to
        finish the job.

        ``target=h`` sets ``max_edge=1.6*h`` and ``min_edge=0.5*h`` (explicit
        values win). Call it every few steps when the surface stretches or
        shrinks a lot, for example while a contact line moves.

        ``protect`` (an edge mask, aligned with :meth:`mesh` rows) keeps edges
        from being split, by setting ``no_refine`` on them for the split; their
        own flags are restored after. Edges flagged ``no_refine`` already are
        never split (unlike Evolver's ``l``, which ignores the flag). Edges leading from a contact line on a
        curved constraint into the surface are good candidates: their midpoint
        would land off the constraint, e.g. inside a solid sphere
        (``mesh.edges_touching(ev.on_constraint(k), "one")``).

        ``keep_boundary`` (default): edges on the surface's boundary (contact
        lines, wires) are never deleted: they are short where the geometry is
        small, and deleting them would coarsen it.

        Linear model only (Evolver deletes edges only there). Returns the
        numbers of edges ``deleted`` (about: each deletion removes three),
        ``split`` and ``switched``.
        """
        if self.model != "linear":
            raise ValueError("remesh() needs the linear model (Evolver's 't' doesn't "
                             "delete edges in the quadratic or Lagrange model)")
        if target is not None:
            max_edge = 1.6*float(target) if max_edge is None else max_edge
            min_edge = 0.5*float(target) if min_edge is None else min_edge
        if max_edge is not None and min_edge is not None and not 2*min_edge <= max_edge:
            raise ValueError("max_edge must be at least 2*min_edge: split halves "
                             "shorter than min_edge would be deleted again")
        counts = {"deleted": 0, "split": 0, "switched": 0}

        def number(text: str) -> int:
            found = re.findall(r"(\d+)\s*$", text.strip())
            return int(found[-1]) if found else 0

        if min_edge is not None and keep_boundary:
            before = self.counts["edges"]
            self.command(f"delete edge ee where ee.length < {_num(min_edge, 'min_edge')} "
                         "and ee.valence != 1")
            counts["deleted"] = max(0, (before - self.counts["edges"])//3)
        elif min_edge is not None:
            counts["deleted"] = number(self.command(f"t {_num(min_edge, 'min_edge')}"))
        if max_edge is not None:
            newly: np.ndarray = np.zeros(0, dtype=np.int64)
            if protect is not None:
                ids = self._ids(_core.EDGE)
                mask = self._mask(_core.EDGE, protect, len(ids))
                already = self.values("edge", "no_refine") != 0
                newly = ids[mask & ~already]
                self._run_statements(f"set edge[{i}] no_refine" for i in newly)
            try:
                counts["split"] = number(self.command(
                    f"refine edge where length > {_num(max_edge, 'max_edge')} and not no_refine"))
            finally:
                self._run_statements(f"unset edge[{i}] no_refine" for i in newly)
        if equiangulate:
            counts["switched"] = number(self.command("u"))
        if average:
            self.command("V")
        return counts

    def adapt(self, max_turn: float = 15.0, *, coarsen: bool = True) -> int:
        """Fit the mesh to the curvature: split the edges along which the
        surface turns by more than ``max_turn`` degrees (the angle between the
        vertex normals at its ends), such as on the strongly curved rim of a
        drop at a high contact angle, then equiangulate. One level per call.

        With ``coarsen`` (default), first merge the ends of short interior
        edges (under 3/4 of the bulk edge length, the 75th percentile) along
        which it turns by less than a third of that, where the surface has
        flattened since it was refined. A merge is refused if it would tilt a
        facet by more than ``max_turn/2``, flip one, or make an edge longer
        than 4/3 of the bulk length (checked in Evolver's edge deletion;
        ``collapse_max_tilt`` and ``collapse_max_edge``). Boundary edges
        (contact lines, wires) are never merged.

        Edges flagged ``no_refine`` are kept whole, and edges touching a
        junction of three or more facets (a triple line) are not judged (no
        normal there). Relax afterwards; ``relax(adapt=True)`` does both.

        Linear model only. Returns the number of edges split plus merged.
        """
        if self.model != "linear":
            raise ValueError("adapt() needs the linear model")
        if not max_turn > 0:
            raise ValueError("max_turn must be positive")
        m = self.mesh()
        if m.facets is None or len(m.facets) == 0:
            return 0
        lengths = _edge_lengths(m)
        bulk = float(np.percentile(lengths, 75))
        return self._remesh_pass(merge_below=0.75*bulk if coarsen else None,
                                 flat=max_turn/3, tilt=max_turn/2, longest=4/3*bulk,
                                 split_turn=max_turn, split_above=None)

    def _remesh_pass(self, *, merge_below: Optional[float], flat: float, tilt: float,
                     longest: float, split_turn: Optional[float],
                     split_above: Optional[float]) -> int:
        """Merge interior edges shorter than ``merge_below`` along which the
        surface turns by less than ``flat`` degrees (refused by the engine if a
        facet would tilt by more than ``tilt`` or flip, or an edge grow longer
        than ``longest``; not on a torus or with symmetry, where the engine
        can't check), then split edges turning more than ``split_turn`` or
        longer than ``split_above`` (not ``no_refine`` ones), equiangulate.
        Returns the edges merged plus split."""
        if "pyse_mark" not in self._edge_attributes():
            self.command("define edge attribute pyse_mark integer")
        m = self.mesh()
        merged = 0
        if merge_below is not None and not (self.eval("torus") or self.eval("symmetry_group")):
            valence = _edge_valence(m)
            turn, judged = _edge_turn(m, valence)
            mark = judged & (valence == 2) & (turn < flat) & (_edge_lengths(m) < merge_below)
            if mark.any():
                self.set_values("edge", "pyse_mark", mark.astype(float))
                before = self.counts["vertices"]
                self.command(f"collapse_max_tilt := {_num(tilt, 'tilt')}; "
                             f"collapse_max_edge := {_num(longest, 'length')}")
                try:
                    self.command("delete edge ee where ee.pyse_mark == 1")
                finally:
                    self.command("collapse_max_tilt := 0; collapse_max_edge := 0")
                merged = before - self.counts["vertices"]
                m = self.mesh()
        mark = np.zeros(len(m.edges), dtype=bool)
        if split_turn is not None:
            mark |= _edge_turn(m, _edge_valence(m))[0] > split_turn
        if split_above is not None:
            mark |= _edge_lengths(m) > split_above
        split = self._split_marked(mark) if mark.any() else 0
        if merged and not split and self._tidy_kinds()[0]:
            self.command("u")
        return merged + split

    def residual(self) -> float:
        """How far the surface is from equilibrium, as a dimensionless number
        (0 at an equilibrium; nothing moves).

        From the vertex velocities a gradient step would use (forces projected
        on the constraints, with the volume and quantity multipliers applied),
        the part that changes the shape: along each vertex's normal, projected
        into its constraints (at a contact line: the imbalance of the contact
        angle). Tangential parts, and freedoms that only slide a vertex along a
        wire, are mesh motion: Newton steps leave them to the gradient steps
        and so does the residual. Each velocity is about the vertex's share of
        the area times the local pressure imbalance, so the root mean square
        times N/sqrt(area) measures that imbalance relative to the surface's
        size: comparable across meshes and scales. With an elastic energy
        (Evolver's elastic methods: in-plane forces are physical there) all
        directions count, not only the normal. (For a string: the root
        mean square times N, the imbalance times the length; in general the
        imbalance times the surface measure to the power 1/dimension.)
        """
        _, result = self._call(_core.residual, self._elastic())
        r = np.asarray(result.data, dtype=float)
        if len(r) == 0:
            return 0.0
        area = _core.total_area()           # a string's length, a simplex's volume
        if area <= 0:
            return float(np.sqrt((r**2).sum()))
        rep = _core.representation()
        dim = 1 if rep == 1 else 2 if rep == 2 else int(self.eval("surface_dimension"))
        return float(np.sqrt((r**2).sum()*len(r))/area**((dim - 1)/dim))

    def health(self, tol: float = 1e-8, stability: bool = False) -> Health:
        """A check-up of the surface: is it in equilibrium (:meth:`residual`
        below ``tol``), stable (with ``stability=True``, :meth:`stability`),
        are its facets in shape, does it come close to (or through) a wall or
        mirror it isn't attached to, or to itself?

        The gaps are in units of the median edge length. ``wall_gap`` leaves
        out the vertices within two edges of a constraint's own vertices (next
        to a contact line the surface is close to its wall by design);
        ``self_gap`` measures between vertices more than two edges apart.
        ``junctions`` counts free junctions that break Plateau's laws (edges
        where four or more films meet, vertices whose tangent cone Evolver's
        ``o`` pops, such as more than four triple edges, or, in a string
        network, four or more strings meeting): the surface
        is in equilibrium for its topology, but popping them (Evolver's
        ``o``) lowers the energy. ``target_gap`` is the largest relative
        gap between a fixed body volume or quantity and its target (Evolver's
        projection onto them can fail on a degenerate mesh; an issue above
        1e-4). ``crossed`` counts vertices on the far side of a constraint: the
        forbidden side of a one-sided one, or a few vertices (at most 5%)
        against all the others for an equality constraint. ``issues`` puts
        what looks wrong in words: unconverged, unstable, facets below 1
        degree, crossings, gaps under 0.1, targets missed. (For a
        crystalline energy the residual is left out of ``issues``: it
        doesn't vanish at the Wulff shape.) :meth:`relax` attaches one to its
        result. Cost: about a gradient step (the stability check: two
        factorizations).
        """
        stable: Optional[bool] = None
        modes: Optional[int] = None
        if stability:
            check = self.stability()
            stable, modes = check.stable, check.negative
        return self._health(self.residual(), stable, modes, tol)

    def _health(self, residual: float, stable: Optional[bool], modes: Optional[int],
                tol: float) -> Health:
        m = self.mesh()
        quality = self.mesh_quality(5.0)
        e = m.edges
        n = len(m.vertices)
        h = quality.edge_median if np.isfinite(quality.edge_median) and quality.edge_median > 0 \
            else 1.0
        used = np.zeros(n, dtype=bool)
        used[e.ravel()] = True
        # fixed vertices can't approach anything (and often draw a wall, such
        # as a table top lying in the plane of a constraint)
        moving = used & (self.values("vertex", "fixed") == 0)
        wall_gap, wall = np.inf, None
        nums, attrs, dist, names = self._constraint_gaps()
        # a contact line on one wall may meet another (a corner, or a second
        # constraint drawn on the same wall): gaps count only for vertices free
        # of every constraint
        free = moving & ~np.isnan(dist).any(axis=1) if dist.size else moving
        label = {int(c): f"constraint {nm or c}" for c, nm in zip(nums, names)}
        for c in range(len(nums)):
            d = dist[:, c]
            on = np.isnan(d) & used
            off = ~np.isnan(d) & free
            if not off.any():
                continue
            far = off & ~_grow(on, e, 2)
            if far.any():
                g = float(np.abs(d[far]).min())/h
                if g < wall_gap:
                    wall_gap, wall = g, int(nums[c])
        crossed = _crossed(nums, attrs, dist, used, _CROSS_TOL*h)
        self_gap = _self_gap(m.vertices, e, moving, h)
        issues = []
        if not residual < tol and not _core.crystalline():
            issues.append(f"not in equilibrium (residual {residual:.2g})")
        if stable is False:
            issues.append(f"unstable: {modes} direction(s) lower the energy")
        if quality.angle_min < 1.0:
            issues.append(f"degenerate facets (smallest angle {quality.angle_min:.2g} degrees, "
                          f"{quality.skinny} below 5)")
        for c, k in crossed.items():
            issues.append(f"{k} {'vertex' if k == 1 else 'vertices'} on the far side of {label[c]}")
        if wall_gap < 0.1:
            issues.append(f"the surface nearly touches {label.get(wall or 0, 'a constraint')} "
                          f"({wall_gap:.2g} edge lengths away)")
        if self_gap < 0.1:
            issues.append(f"the surface nearly touches itself ({self_gap:.2g} edge lengths)")
        target_gap, _ = self._target_gap()
        if target_gap > _TARGET_LOST:
            issues.append(f"a fixed volume or quantity is off its target ({target_gap:.2g} "
                          "relative)")
        junctions = self._bad_junctions()
        if junctions:
            issues.append(f"{junctions} {'junction breaks' if junctions == 1 else 'junctions break'} "
                          "Plateau's laws (Evolver's 'o' pops them)")
        return Health(float(residual), stable, modes, float(quality.angle_min),
                      int(quality.skinny), float(wall_gap), wall, crossed, float(self_gap),
                      junctions, float(target_gap), issues)

    def _bad_junctions(self) -> int:
        """Junctions a minimal surface can't have, away from walls, wires and
        fixed elements: in a film, edges where four or more facets meet (not
        fixed, at least one end free) and free vertices whose tangent cone
        Evolver's 'o' pops (as it decides: more than four triple edges, a
        cube's cone, ...); in a string network, free vertices where four or
        more strings meet."""
        rep = _core.representation()
        if rep == 3 or self.model == "lagrange":
            return 0
        _, attached = self._call(_core.vertex_attached)
        free = ~np.asarray(attached.data, dtype=bool)
        if rep == 1:
            return int(((self.values("vertex", "valence") >= 4) & free).sum())
        m = self.mesh()
        if len(m.edges) == 0:
            return 0
        valence = self.values("edge", "valence")
        # along a wire or wall (both ends attached) more films may meet
        loose = (free[m.edges[:, 0]] | free[m.edges[:, 1]]) & (self.values("edge", "fixed") == 0)
        bad = int(((valence >= 4) & loose).sum())
        if not (valence >= 3).any():  # a film without junctions: every vertex is a disc
            return bad
        try:          # the tangent cones 'o' pops (cubes, Kraynik's, ...)
            _, cones = self._call(_core.poppable_vertices)
            return bad + int(cones.data)
        except EvolverError:        # a cone it can't analyze: more than four triple edges
            triple = valence == 3
            count = np.bincount(m.edges[triple].ravel(), minlength=len(free))
            return bad + int(((count > 4) & free).sum())

    def _constraint_gaps(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]":
        """Constraint numbers, attribute bits, per vertex the signed distance
        to each (NaN where on it), and their names ("" if unnamed)."""
        _, gaps = self._call(_core.constraint_gaps)
        return gaps.data

    def stability(self, nearest: int = 6) -> Stability:
        """Whether the surface sits at a stable equilibrium (a minimum, as far as
        second order tells; call it once converged, see :meth:`residual`).

        Counts the Hessian eigenvalues below zero (one factorization); if
        there are any, counts again below a small negative threshold: -1% of
        the scale of the lowest ones (the median magnitude of the positive
        ones among the ``nearest`` to zero, from Evolver's ``ritz``, which is
        slow on large surfaces). Exact zero modes from symmetries (a drop
        sliding on a plane, a barrel along its fibre) come out slightly above
        or below zero numerically; the threshold keeps them from reading as
        instabilities, at the price of noticing a real one slightly past its
        onset.
        """
        first = self.eigen_counts(0.0).negative
        if first == 0:
            return Stability(True, 0, np.array([]), 0.0)
        text = self.command(f"ritz(0, {max(int(nearest), first + 4)})")
        values = np.array(sorted(float(x) for x in re.findall(
            r"^\s*\d+\.\s+([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$", text, re.MULTILINE)))
        positive = np.abs(values[values > 0])
        scale = float(np.median(positive)) if len(positive) else float(np.abs(values).max(initial=1.0))
        threshold = -0.01*scale
        negative = self.eigen_counts(threshold).negative
        return Stability(negative == 0, negative, values, threshold)

    def eigen_counts(self, shift: float = 0.0) -> "EigenCounts":
        """How many Hessian eigenvalues lie below, at and above ``shift``
        (Evolver's ``eigenprobe``; costs one factorization).

        At an equilibrium, ``negative > 0`` means it is unstable (a saddle):
        some motion lowers the energy. With constraints, only motions that
        keep them count.
        """
        text = self.command(f"eigenprobe {_num(shift, 'shift')}")
        found = re.search(r"Eigencounts:\s*(\d+)\s*<,\s*(\d+)\s*==,\s*(\d+)\s*>", text)
        if not found:
            raise EvolverError(f"unexpected eigenprobe output: {text.strip()!r}")
        return EigenCounts(*(int(g) for g in found.groups()))

    def check(self) -> List[str]:
        """Evolver's consistency check of the surface's topology (``check``):
        the problems found, one per line; empty when it is sound."""
        return [line for line in self.command("check").splitlines() if line.strip()]

    def mesh_quality(self, skinny_angle: float = 15.0) -> "MeshQuality":
        """Edge lengths and facet shapes; see :meth:`Mesh.quality`. In a torus
        or quotient space, from Evolver's own edge lengths (vertex coordinates
        are wrapped, so facets across the cell's sides would look torn)."""
        if not _core.symmetry():
            return self.mesh().quality(skinny_angle)
        lengths = self.values("edge", "length")
        nan = float("nan")
        if len(lengths) == 0:
            return MeshQuality(nan, nan, nan, nan, nan, 0, 0, skinny_angle)
        edges = (float(lengths.min()), float(np.median(lengths)), float(lengths.max()))
        if self.counts["facets"] == 0 or self.eval("surface_dimension") != 2:
            return MeshQuality(*edges, nan, nan, 0, 0, skinny_angle)
        a, b, c = (self.values("facet", f"edge[{k}].length") for k in (1, 2, 3))
        angles = [np.degrees(np.arccos(np.clip((y*y + z*z - x*x)/np.maximum(2*y*z, 1e-300),
                                               -1, 1))) for x, y, z in ((a, b, c), (b, c, a), (c, a, b))]
        smallest = np.min(angles, axis=0)
        p = (a + b + c)/2
        area = np.sqrt(np.maximum(p*(p - a)*(p - b)*(p - c), 0))
        return MeshQuality(*edges, float(smallest.min()), float(np.median(smallest)),
                           int((smallest < skinny_angle).sum()),
                           int((area < 1e-12*np.median(area)).sum()), skinny_angle)

    def refine(self, times: int = 1) -> None:
        """Refine the surface: split every edge and facet (Evolver's ``r``)."""
        for _ in range(times):
            self.command("r")

    def equiangulate(self) -> None:
        """Flip edges to improve triangle shapes (Evolver's ``u``)."""
        self.command("u")

    def vertex_average(self) -> None:
        """Move vertices to the average of their neighbors (Evolver's ``V``)."""
        self.command("V")

    def hessian(self, seek: bool = False, *, threads: Optional[int] = None) -> None:
        """One Newton step (``hessian``), or a Newton line search (``hessian_seek``).
        ``threads`` sets the threads for this call only (see
        :func:`pysurfaceevolver.threads_limit`)."""
        with _threads_for_call(threads):
            self.command("hessian_seek" if seek else "hessian")

    def set_model(self, model: str, order: Optional[int] = None) -> None:
        """Switch to the ``"linear"``, ``"quadratic"`` or ``"lagrange"`` model.

        Lagrange needs an ``order``.
        """
        model = model.lower()
        if model == "lagrange":
            if order is None:
                raise ValueError("the Lagrange model needs an order")
            self.command(f"lagrange {int(order)}")
        elif model in ("linear", "quadratic"):
            if order is not None:
                raise ValueError(f"the {model} model has no order")
            self.command(model)
        else:
            raise ValueError("model must be 'linear', 'quadratic' or 'lagrange'")

    def recalc(self) -> None:
        """Recalculate energies and quantities."""
        self.command("recalc")

    def dump(self, path: Union[str, os.PathLike]) -> None:
        """Save the current surface as a datafile (Evolver's ``dump``).

        Numbers are written with 17 significant digits, so loading the file
        gives back the same coordinates exactly.
        """
        self.command(f'dump "{os.fspath(path)}"')

    def save(self) -> Snapshot:
        """Take a snapshot of the surface, to :meth:`restore` later.

        The snapshot is an exact dump: coordinates and energies come back
        bit for bit. Settings that dump doesn't record (most display
        options) are not part of it.
        """
        name = self.datafile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.fe")
            self.dump(path)
            with open(path) as f:
                return Snapshot(f.read(), name)

    def restore(self, snapshot: Snapshot) -> None:
        """Replace the surface with a snapshot from :meth:`save`."""
        global _newton_first_failed, _collapse_failures, _collapse_skip
        # the same surface, not a new one: relax()'s memory of failed attempts stays
        kept = _newton_first_failed, _collapse_failures, _collapse_skip
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.fe")
            with open(path, "w") as f:
                f.write(snapshot.text)
            self.load(path)
        _core.set_datafile(snapshot.datafile)
        _newton_first_failed, _collapse_failures, _collapse_skip = kept

    # ------------------------------------------------------------------
    # Writing element data

    def _ids(self, element_type: int) -> np.ndarray:
        if _core.count(element_type) == 0:
            return np.zeros(0, dtype=np.int64)
        return self.values(_ELEMENT_NAMES[element_type], "id").astype(np.int64)

    @staticmethod
    def _mask(element_type: int, where, n: int) -> np.ndarray:
        """A boolean mask over n elements from where (a mask or rows)."""
        mask = np.asarray(where)
        if mask.dtype != bool:
            rows = mask.astype(int)
            mask = np.zeros(n, dtype=bool)
            mask[rows] = True
        elif mask.shape != (n,):
            raise ValueError(f"where needs one entry per {_ELEMENT_NAMES[element_type]} "
                             f"({n}), got {mask.shape[0]}")
        return mask

    def _selected(self, element: str, where) -> "tuple[str, np.ndarray, np.ndarray]":
        element_type = _element_type(element)
        ids = self._ids(element_type)
        mask = (np.ones(len(ids), dtype=bool) if where is None
                else self._mask(element_type, where, len(ids)))
        return _ELEMENT_NAMES[element_type], ids, mask

    def _run_statements(self, statements: Iterable[str]) -> None:
        """Run many statements in batches, recalculating once at the end.

        With autorecalc on, Evolver would recalculate after every batch;
        that dominated the time of bulk writes (8x on 49k vertices).
        """
        batches: List[str] = []
        batch: List[str] = []
        for statement in statements:
            batch.append(statement)
            if len(batch) == _STATEMENTS_PER_COMMAND:
                batches.append("; ".join(batch))
                batch = []
        if batch:
            batches.append("; ".join(batch))
        if not batches:
            return
        autorecalc = self.eval("autorecalc") != 0
        if autorecalc:
            self.command("autorecalc off")
        try:
            for text in batches:
                self.command(text)
        finally:
            if autorecalc:
                self.command("autorecalc on")
                self.command("recalc")

    def set_values(self, element: str, attribute: str, values, *, where=None) -> None:
        """Set a numeric attribute on every element of a type.

        ``values`` is one number per element, aligned with :meth:`mesh` /
        :meth:`values` rows, or a single number for all. ``where`` (a
        boolean mask or rows) limits which elements change. Works for any
        attribute Evolver's ``set`` command accepts: coordinates (``x``,
        ``y``, ...), ``density``, ``tension``, ``target`` (bodies), extra
        attributes from :meth:`define_attribute`, ...

        Coordinates and scalar extra attributes are written directly in C;
        other attributes go through Evolver's ``set`` command, in batches.
        """
        element_type = _element_type(element)
        if _core.fast_attribute(element_type, attribute):
            n = _core.count(element_type)
            mask = None if where is None else self._mask(element_type, where, n)
            vals = np.ascontiguousarray(
                np.broadcast_to(np.asarray(values, dtype=float), (n,)))
            self._call(_core.set_values, element_type, attribute, vals,
                       None if mask is None else mask.astype(np.uint8))
            return
        name, ids, mask = self._selected(element, where)
        vals = np.broadcast_to(np.asarray(values, dtype=float), ids.shape)
        self._run_statements(f"set {name}[{i}] {attribute} {float(v)!r}"
                             for i, v in zip(ids[mask], vals[mask]))

    def set_flag(self, element: str, flag: str, where=None, on: bool = True) -> None:
        """Set (or with ``on=False`` clear) a yes/no attribute such as ``fixed``
        or ``no_refine`` on all elements of a type, or those selected by
        ``where`` (a boolean mask or rows, aligned with :meth:`mesh` rows).

        Prefer this over Evolver's ``foreach ... where ...`` in commands: there,
        a bare element type inside an aggregate (``max(vertex, ...)``) means all
        elements of the surface, and ``vertex[2]`` means global vertex 2, not an
        edge's second vertex (that is ``ee.vertex[2]`` with a named element).
        """
        if not re.fullmatch(r"[A-Za-z_]\w*", flag):
            raise ValueError(f"not an attribute name: {flag!r}")
        name, ids, mask = self._selected(element, where)
        verb = "set" if on else "unset"
        self._run_statements(f"{verb} {name}[{i}] {flag}" for i in ids[mask])

    def fix(self, element: str, where=None) -> None:
        """Fix vertices, edges or facets (all, or those selected by ``where``)."""
        self.set_flag(element, "fixed", where)

    def unfix(self, element: str, where=None) -> None:
        """Unfix vertices, edges or facets (all, or those selected by ``where``)."""
        self.set_flag(element, "fixed", where, on=False)

    def on_constraint(self, constraint: Union[int, str], element: str = "vertex") -> np.ndarray:
        """Which elements are on a constraint (number or name): one bool per
        :meth:`mesh` row of that element type."""
        if isinstance(constraint, str) and not re.fullmatch(r"[A-Za-z_]\w*", constraint):
            raise ValueError(f"not a constraint name: {constraint!r}")
        name = _ELEMENT_NAMES[_element_type(element)]
        if _core.count(_element_type(element)) == 0:
            return np.zeros(0, dtype=bool)
        return self.values(name, f"on_constraint {constraint}") != 0

    def set_constraint(self, element: str, constraint: Union[int, str], where=None,
                       on: bool = True) -> None:
        """Put elements on a constraint (or take them off with ``on=False``)."""
        name, ids, mask = self._selected(element, where)
        verb = "set" if on else "unset"
        self._run_statements(f"{verb} {name}[{i}] constraint {constraint}" for i in ids[mask])

    def define_attribute(self, element: str, name: str, dtype: str = "real") -> None:
        """Define an extra per-element attribute (``real`` or ``integer``).

        Read it with :meth:`values` and write it with :meth:`set_values`.
        """
        self.command(f"define {_ELEMENT_NAMES[_element_type(element)]} attribute {name} {dtype}")

    # ------------------------------------------------------------------
    # Building surfaces

    def load_arrays(self, vertices, faces=None, **kwargs: Any) -> str:
        """Load a surface given as arrays; see :func:`make_datafile` for options."""
        return self.load_string(make_datafile(vertices, faces, **kwargs))

    def load_mesh_file(self, path: Union[str, os.PathLike], *,
                       volume: Union[None, float, str] = None,
                       fix_boundary: bool = True, merge_tolerance: float = 1e-9,
                       **kwargs: Any) -> str:
        """Load a triangle surface from a mesh file (STL, OBJ, PLY, VTK, ...).

        Duplicate points (closer than ``merge_tolerance`` times the bounding
        box size) are merged, and quads are split into triangles.

        volume:
            ``None``: no body. A number: one body enclosed by the whole
            surface, with that target volume. ``"current"``: the same, with
            the volume the surface encloses now. The surface must be closed;
            its orientation is fixed up so the volume is positive.
        fix_boundary:
            Fix the vertices and edges on open boundaries, so a film spanning
            a wire keeps its wire.

        Other keyword arguments go to :func:`make_datafile`.
        """
        meshio = _import_meshio()
        mesh = meshio.read(os.fspath(path))
        tris = []
        for block in mesh.cells:
            if block.type == "triangle":
                tris.append(block.data)
            elif block.type == "quad":
                q = block.data
                tris.append(np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
        if not tris:
            raise ValueError(f"{os.fspath(path)} has no triangles or quads")
        points, faces = _merge_points(np.asarray(mesh.points, float), np.vstack(tris),
                                      merge_tolerance)
        if volume is not None:
            if not is_watertight(faces):
                raise ValueError("volume needs a closed, consistently oriented surface")
            current = _signed_volume(points, faces)
            sign = 1 if current >= 0 else -1
            target = abs(current) if volume == "current" else float(volume)
            kwargs.setdefault("bodies", [Body(faces=range(len(faces)), volume=target,
                                              orientation=[sign] * len(faces))])
        if fix_boundary and "fixed" not in kwargs:
            boundary = _boundary_vertices(faces)
            if len(boundary):
                kwargs["fixed"] = boundary
        return self.load_arrays(points, faces, **kwargs)

    # ------------------------------------------------------------------
    # Export and visualization

    def write(self, path: Union[str, os.PathLike], *, curved: str = "tessellate",
              n: Optional[int] = None, file_format: Optional[str] = None) -> None:
        """Write the surface to a mesh file with meshio (STL, OBJ, PLY, VTU,
        Gmsh .msh, XDMF, ...); the format follows the file extension.

        ``curved="tessellate"`` writes flat triangles sampled on curved
        elements (``n`` per facet edge); ``curved="native"`` writes the
        high-order elements themselves: Gmsh up to order 10, VTU/VTK any
        order, XDMF order 2; STL, OBJ, PLY etc. only take flat triangles.
        XDMF needs h5py. ``.msh`` means Gmsh format 2.2 (meshio alone would pick ANSYS, and
        its Gmsh 4.1 output can't be opened by Gmsh). See
        :meth:`Mesh.to_meshio` for cell data.
        """
        meshio = _import_meshio()
        file_format, flavor = _target_format(os.fspath(path), file_format)
        data = self.mesh().to_meshio(curved, n, flavor=flavor)
        meshio.write(os.fspath(path), data, file_format=file_format)

    def body_surfaces(self, *, curved: str = "tessellate", n: Optional[int] = None,
                      cap: bool = False) -> Dict[int, BodySurface]:
        """The closed surface around each body, with outward normals.

        See :meth:`Mesh.body_surfaces`. With ``cap=True``, an opening whose
        boundary vertices all lie on a constraint (a drop on a curved solid,
        say) is closed by a cap on that constraint: Evolver projects its
        points, so any constraint formula works (3D models; elsewhere caps
        are flat). ``cap_constraints`` says which constraint each cap is on
        (None: not on one; a flat cap).

        Projecting creates and dissolves temporary vertices: the surface is
        left as it was, but Evolver counts it as changed (the next Newton
        step rebuilds its matrix pattern, and later new elements may get
        other numbers than without the export).
        """
        mesh = self.mesh()
        if not cap or curved != "tessellate":
            return mesh.body_surfaces(curved, n, cap=cap)
        lookup: Dict[str, Any] = {}       # built on the first loop that needs it
        constraint_of: Dict[bytes, Optional[int]] = {}

        def constraint_for(loop_points) -> Optional[int]:
            key = np.asarray(loop_points).tobytes()
            if key not in constraint_of:
                if not lookup:
                    lookup["on"] = self._vertex_constraints()
                    lookup["index"] = _vertex_index(mesh.vertices)
                constraint_of[key] = self._loop_constraint(lookup["index"], lookup["on"],
                                                           loop_points)
            return constraint_of[key]

        def project(points, loop_points):
            if self.sdim != 3:
                return points
            con = constraint_for(loop_points)
            return points if con is None else self._project_points(points, con)

        surfaces = mesh.body_surfaces(curved, n, cap=True, project=project)
        for s in surfaces.values():   # the constraint of each cap, from its rim
            assert s.cap_ids is not None
            if not s.cap_constraints:
                continue
            rim = np.zeros(len(s.points), bool)
            rim[s.cells[s.cap_ids == 0].ravel()] = True
            for k in s.cap_constraints:
                cap_points = np.zeros(len(s.points), bool)
                cap_points[s.cells[s.cap_ids == k].ravel()] = True
                rim_points = s.points[cap_points & rim]
                s.cap_constraints[k] = (constraint_for(rim_points) if self.sdim == 3 else None)
        return surfaces

    def _vertex_constraints(self) -> Dict[int, np.ndarray]:
        """{constraint: mask of vertex rows on it}"""
        count = int(self.eval("high_constraint"))
        return {k: self.values("vertex", f"on_constraint {k}") > 0 for k in range(1, count + 1)}

    @staticmethod
    def _loop_constraint(index, on: Dict[int, np.ndarray], loop_points) -> Optional[int]:
        """The constraint all vertices of a boundary loop lie on, if any."""
        if not on:
            return None
        rows = _rows_of(index, np.asarray(loop_points))
        rows = rows[rows >= 0]
        if len(rows) == 0:
            return None
        common = [k for k, mask in on.items() if mask[rows].all()]
        if len(common) > 1:
            warnings.warn(f"an opening lies on constraints {common} at once; "
                          f"its cap follows constraint {common[0]}", stacklevel=4)
        return common[0] if common else None

    def _project_points(self, points, constraint: int) -> np.ndarray:
        """Points moved onto a constraint by Evolver itself: a temporary
        vertex is put on the constraint (which projects it), read and
        dissolved."""
        lines = [f"pse_tmp_v := new_vertex({x!r}, {y!r}, {z!r}); "
                 f"set vertex[pse_tmp_v] constraint {constraint}; "
                 'printf "@pse %.17g %.17g %.17g\\n", vertex[pse_tmp_v].x, '
                 "vertex[pse_tmp_v].y, vertex[pse_tmp_v].z; dissolve vertex[pse_tmp_v]"
                 for x, y, z in np.asarray(points, float)[:, :3].tolist()]
        out = []
        for i in range(0, len(lines), 200):
            out += [line for line in self.command("; ".join(lines[i:i + 200])).splitlines()
                    if line.startswith("@pse ")]
        return np.array([[float(v) for v in line.split()[1:]] for line in out])

    def write_bodies(self, pattern: str = "body_{id}.stl", *, curved: str = "tessellate",
                     n: Optional[int] = None, cap: bool = False,
                     require_watertight: bool = True,
                     file_format: Optional[str] = None) -> Dict[int, str]:
        """Write one closed surface per body, for volume meshing.

        ``pattern`` is formatted with the body ``id``. Raises ``ValueError``
        for a body whose surface isn't watertight (for example a drop that
        Evolver closes with a constraint plane) unless ``cap=True`` closes it
        or ``require_watertight=False``. Returns ``{body id: path}``.
        """
        meshio = _import_meshio()
        surfaces = self.body_surfaces(curved=curved, n=n, cap=cap)
        if require_watertight:
            open_ = [b for b, s in surfaces.items() if not s.watertight]
            if open_:
                raise ValueError(
                    f"bodies {open_} are not closed by facets (they likely end on a "
                    "constraint); use cap=True to close planar openings, or "
                    "require_watertight=False")
        written = {}
        for b, surface in surfaces.items():
            path = pattern.format(id=b)
            fmt, flavor = _target_format(path, file_format)
            meshio.write(path, surface.to_meshio(flavor), file_format=fmt)
            written[b] = path
        return written

    def plot(self, scalars: Union[None, str, np.ndarray] = None, *, element: Optional[str] = None,
             n: Optional[int] = None, off_screen: bool = False,
             screenshot: Optional[str] = None, mirror=None, **kwargs: Any) -> Any:
        """Show the surface with PyVista.

        ``scalars`` colors it: an Evolver expression evaluated per facet
        (or per vertex with ``element="vertex"``), or an array with one value
        per facet or vertex row. Curved elements are tessellated with ``n``
        subdivisions. ``mirror`` adds mirror images, for a symmetric piece of
        a surface: a list of planes, each ``"x"`` (x = 0), ``("x", c)`` (x = c)
        or ``(normal, point)``, applied in order, each doubling what is there
        (an eighth of a cell: ``["z", "y", "x"]``). Other keyword arguments go
        to ``Plotter.add_mesh``.
        """
        from ._viz import _import_pyvista, add_images, surface_dataset
        pv = _import_pyvista()
        dataset, name = surface_dataset(self, scalars, element, n)
        plotter = pv.Plotter(off_screen=off_screen)
        add_images(plotter, dataset, mirror, "evolver-surface", scalars=name,
                   **{"show_edges": True, **kwargs})
        return plotter.show(screenshot=screenshot)

    def live_view(self, scalars: Union[None, str, np.ndarray] = None, **kwargs: Any) -> "LiveView":
        """A PyVista view that redraws on :meth:`LiveView.update`; see :class:`LiveView`.

        Typical use: ``ev.iterate(200, callback=ev.live_view().update, every=10)``.
        """
        from ._viz import LiveView
        return LiveView(self, scalars, **kwargs)

    # ------------------------------------------------------------------
    # Surface data

    @property
    def datafile(self) -> str:
        """Name of the loaded datafile, or ``""`` if none."""
        return _core.datafile()

    @property
    def valid(self) -> bool:
        """Whether there is a usable surface (see :class:`InvalidSurfaceError`)."""
        return _core.surface_valid()

    @property
    def sdim(self) -> int:
        """Dimension of the ambient space."""
        return _core.sdim()

    @property
    def representation(self) -> str:
        """``"string"``, ``"soapfilm"`` or ``"simplex"``."""
        return _REPRESENTATIONS.get(_core.representation(), "unknown")

    @property
    def model(self) -> str:
        """``"linear"``, ``"quadratic"`` or ``"lagrange"``."""
        return _MODELS.get(_core.modeltype(), "unknown")

    @property
    def lagrange_order(self) -> int:
        return _core.lagrange_order()

    @property
    def torus(self) -> bool:
        """Whether the domain is periodic (torus model)."""
        return _core.torus()

    @property
    def total_energy(self) -> float:
        """Total energy as of the last iteration or recalculation."""
        return _core.total_energy()

    @property
    def total_area(self) -> float:
        """Total area as of the last iteration or recalculation."""
        return _core.total_area()

    @property
    def counts(self) -> dict:
        """Number of vertices, edges, facets and bodies."""
        return {
            "vertices": _core.count(_core.VERTEX),
            "edges": _core.count(_core.EDGE),
            "facets": _core.count(_core.FACET),
            "bodies": _core.count(_core.BODY),
        }

    @property
    def vertices(self) -> np.ndarray:
        """Vertex coordinates as an ``(n, sdim)`` array (a copy).

        Assigning an array of the same shape moves the vertices and
        recalculates energies. Constraints aren't re-projected until the next
        iteration.
        """
        _, result = self._call(_core.vertices)
        return result.data[0]

    @vertices.setter
    def vertices(self, xyz) -> None:
        arr = np.ascontiguousarray(xyz, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("vertex coordinates must be a 2-D array")
        self._call(_core.set_vertex_coords, arr)

    def mesh(self) -> Mesh:
        """Return the current geometry and connectivity as NumPy arrays.

        Faces exist only in the soapfilm representation. For quadratic and
        Lagrange models the high-order node layout is included too; see
        :meth:`Mesh.tessellate`.

        The result is cached until the surface changes (any command, load or
        coordinate write), so its arrays are read-only; copy them to modify.
        """
        global _mesh_cache
        version = _core.surface_version()
        with _mesh_cache_lock:
            if _mesh_cache is not None and _mesh_cache[0] == version:
                return _mesh_cache[1]
        mesh = self._build_mesh()
        with _mesh_cache_lock:
            if _core.surface_version() == version:   # nothing changed meanwhile
                _mesh_cache = (version, mesh)
        return mesh

    def _build_mesh(self) -> Mesh:
        _, r = self._call(_core.mesh)
        (xyz, vids, fixed, edges, eids, faces, fids, fbodies,
         edge_nodes, edge_index, facet_nodes, facet_index, order, bezier) = r.data
        if edge_nodes is None and facet_nodes is None:
            order, bezier = 1, False
        fixed = fixed.astype(bool)
        _read_only(xyz, edges, faces, vids, eids, fids, fbodies, fixed,
                   edge_nodes, edge_index, facet_nodes, facet_index)
        return Mesh(xyz, edges, faces, vids, eids, fids, fbodies, fixed,
                    order, bezier, edge_nodes, edge_index, facet_nodes, facet_index)

    def body(self, number: int) -> "BodyView":
        """A live handle on one body (1-based, as in the datafile).

        ``ev.body(1).target = 0.05`` sets the prescribed volume, and
        ``ev.body(1).pressure`` reads the current pressure. For all bodies at
        once, as arrays, use :meth:`bodies`.
        """
        number = int(number)
        if not 1 <= number <= _core.count(_core.BODY):
            raise IndexError(f"no body {number}: the surface has "
                             f"{_core.count(_core.BODY)} bodies")
        return BodyView(self, number)

    def bodies(self) -> Bodies:
        """Return volumes, target volumes and pressures of all bodies (a snapshot)."""
        _, result = self._call(_core.bodies)
        ids, volume, target, pressure, fixed = result.data
        return Bodies(ids, volume, target, pressure, fixed.astype(bool))

    @property
    def parameters(self) -> "Parameters":
        """The datafile's parameters, as a live dict-like view.

        ``ev.parameters["angle"] = 60`` assigns, and Evolver recalculates
        whatever depends on it.
        """
        return Parameters(self)

    def quantities(self) -> Dict[str, Quantity]:
        """Named quantities from the datafile, recalculated now."""
        _, result = self._call(_core.quantities)
        out = {}
        for name, value, target, modulus, pressure, kind in result.data:
            kind_name = _QUANTITY_KINDS.get(kind, "energy")
            if kind_name != "fixed":
                target = float("nan")
            out[name] = Quantity(name, value, target, modulus, pressure, kind_name)
        return out

    def __repr__(self) -> str:
        name = self.datafile or "no datafile"
        if not _core.surface_valid():
            return f"<Evolver {name!r}: no valid surface>"
        c = self.counts
        return (f"<Evolver {name!r}: {c['vertices']} vertices, {c['edges']} edges, "
                f"{c['facets']} facets, {c['bodies']} bodies>")

    def _repr_html_(self) -> Optional[str]:
        try:   # e.g. busy with a run in another thread: plain repr instead
            return self._html()
        except Exception:
            return None

    def _html(self) -> str:
        name = os.path.basename(self.datafile) or "no datafile"
        if not _core.surface_valid():
            return _html.fields("Evolver", [("datafile", name), ("surface", "none (not valid)")])
        model = self.model
        if model == "lagrange":
            model = f"Lagrange {self.lagrange_order}"
        c = self.counts
        rows = [
            ("datafile", name),
            ("model", f"{self.representation}, {model}" + (", torus" if self.torus else "")),
            ("elements", " · ".join(f"{_html.number(c[k])} {k if c[k] != 1 else one}"
                                    for k, one in (("vertices", "vertex"), ("edges", "edge"),
                                                   ("facets", "facet"), ("bodies", "body")))),
            ("energy", _html.number(self.total_energy)),
            ("area", _html.number(self.total_area)),
            ("threads", f"{_core.threads()} · solver {_core.solver()}"),
        ]
        out = _html.fields("Evolver", rows)
        if c["bodies"]:
            out += self.bodies()._repr_html_()
        return out


def _vertex_index(vertices: np.ndarray, tol: float = 1e-9):
    """Lookup from (rounded) position to vertex row, for :func:`_rows_of`."""
    step = tol * (float(np.abs(vertices).max()) or 1.0)
    keys = np.round(vertices / step).astype(np.int64)
    return step, {tuple(k): i for i, k in enumerate(keys.tolist())}


def _rows_of(index, points: np.ndarray) -> np.ndarray:
    """Vertex row at the position of each point, or -1."""
    step, rows = index
    keys = np.round(np.asarray(points, float) / step).astype(np.int64)
    return np.array([rows.get(tuple(k), -1) for k in keys.tolist()], dtype=np.int64)


def _merge_points(points: np.ndarray, faces: np.ndarray, tolerance: float):
    """Merge points closer than tolerance * bounding box size; drop unused ones."""
    size = float(np.ptp(points, axis=0).max()) or 1.0
    keys = np.round(points / (tolerance * size)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    faces = inverse.ravel()[faces]
    points = points[first]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                  & (faces[:, 0] != faces[:, 2])]
    used, local = np.unique(faces, return_inverse=True)
    return points[used], local.reshape(faces.shape)


def _signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = (points[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def _boundary_vertices(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                    axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1])


class BodyView:
    """A live handle on one body; see :meth:`Evolver.body`.

    Reading an attribute asks Evolver now; setting one changes the surface.
    """

    def __init__(self, ev: Evolver, number: int):
        self._ev = ev
        self.number = number

    def _get(self, attribute: str) -> float:
        return self._ev.eval(f"body[{self.number}].{attribute}")

    @property
    def volume(self) -> float:
        """The current volume."""
        return self._get("volume")

    @property
    def target(self) -> Optional[float]:
        """The prescribed volume, or None if the volume is free. Setting a
        number fixes the volume; setting None frees it."""
        return self._get("target") if self.fixed else None

    @target.setter
    def target(self, value: Optional[float]) -> None:
        if value is None:
            self._ev.command(f"unset body[{self.number}] target")
        else:
            self._ev.command(f"set body[{self.number}] target {_num(value, 'target')}")

    @property
    def volconst(self) -> float:
        """A constant added to the computed volume (for parts of the body the
        surface doesn't enclose, such as a solid inside it)."""
        return self._get("volconst")

    @volconst.setter
    def volconst(self, value: float) -> None:
        self._ev.command(f"set body[{self.number}] volconst {_num(value, 'volconst')}")

    @property
    def pressure(self) -> float:
        """The pressure: the Lagrange multiplier of the volume constraint."""
        return self._get("pressure")

    @property
    def fixed(self) -> bool:
        """Whether the volume is prescribed."""
        b = self._ev.bodies()
        return bool(b.fixed[list(b.ids).index(self.number)])

    def __repr__(self) -> str:
        target = self.target
        return (f"<body {self.number}: volume {self.volume:.6g}, "
                f"target {'free' if target is None else format(target, '.6g')}, "
                f"pressure {self.pressure:.6g}>")




def _crossed(nums: np.ndarray, attrs: np.ndarray, dist: np.ndarray, used: np.ndarray,
             eps: float) -> Dict[int, int]:
    """Per constraint, the vertices on its far side: the forbidden side of a
    one-sided constraint, or a few (at most 5%) against all the others."""
    out: Dict[int, int] = {}
    for c in range(len(nums)):
        d = dist[:, c]
        off = ~np.isnan(d) & used
        if not off.any():
            continue
        if attrs[c] & 2:            # NONNEGATIVE
            bad = int((d[off] < -eps).sum())
        elif attrs[c] & 1:          # NONPOSITIVE
            bad = int((d[off] > eps).sum())
        else:
            pos, neg = int((d[off] > eps).sum()), int((d[off] < -eps).sum())
            bad = min(pos, neg) if min(pos, neg) <= 0.05*(pos + neg) else 0
        if bad:
            out[int(nums[c])] = bad
    return out


def _grow(mask: np.ndarray, edges: np.ndarray, rings: int) -> np.ndarray:
    """The vertices within ``rings`` edges of the masked ones."""
    out = mask.copy()
    for _ in range(rings):
        nxt = out.copy()
        nxt[edges[out[edges[:, 1]], 0]] = True
        nxt[edges[out[edges[:, 0]], 1]] = True
        out = nxt
    return out


def _self_gap(x: np.ndarray, edges: np.ndarray, used: np.ndarray, h: float) -> float:
    """The smallest distance, in units of h, between used vertices more than
    two edges apart (inf if none are closer than h/2)."""
    idx = np.flatnonzero(used)
    if len(idx) < 2 or len(edges) == 0:
        return float("inf")
    r = h/2
    p = x[idx]
    # cells on the first three coordinates (zero-padded in the plane): pairs
    # closer than r in the full space are closer in those too
    grid = np.zeros((len(p), 3))
    grid[:, :min(3, p.shape[1])] = p[:, :3]
    cell = np.floor(grid/r).astype(np.int64)
    cell -= cell.min(axis=0) - 1                 # a margin: neighbours never wrap
    dims = cell.max(axis=0) + 2
    k = (cell[:, 0]*dims[1] + cell[:, 1])*dims[2] + cell[:, 2]
    order = np.argsort(k, kind="stable")
    sk = k[order]
    found_i, found_j, found_d = [], [], []
    offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
    offsets = offsets[[tuple(o) >= (0, 0, 0) for o in offsets]]   # each pair of cells once
    for off in offsets:
        same = not off.any()
        nk = sk + (off[0]*dims[1] + off[1])*dims[2] + off[2]       # sorted, like sk
        lo = np.searchsorted(sk, nk, "left")
        cnt = np.searchsorted(sk, nk, "right") - lo
        if not cnt.any():
            continue
        i = np.repeat(order, cnt)
        within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        j = order[np.repeat(lo, cnt) + within]
        keep = i < j if same else i != j
        i, j = i[keep], j[keep]
        d = np.linalg.norm(p[i] - p[j], axis=1)
        close = d < r
        found_i.append(i[close])
        found_j.append(j[close])
        found_d.append(d[close])
    d = np.concatenate(found_d)
    if len(d) == 0:
        return float("inf")
    i = idx[np.concatenate(found_i)]
    j = idx[np.concatenate(found_j)]
    # the nearest pair more than two edges apart (close pairs are few)
    a = np.concatenate([edges[:, 0], edges[:, 1]])
    b = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    start = np.searchsorted(a, np.arange(len(x)), "left")
    stop = np.searchsorted(a, np.arange(len(x)), "right")
    for k in np.argsort(d)[:5000]:
        ni = b[start[i[k]]:stop[i[k]]]
        nj = b[start[j[k]]:stop[j[k]]]
        if j[k] in ni or np.intersect1d(ni, nj).size:
            continue
        return float(d[k]/h)
    return float("inf")


# a vertex counts as crossed when beyond a constraint by this many median edge lengths:
# less is the chord of a curved wall next to a contact line
_CROSS_TOL = 0.01
_ADAPT_PASSES = 3
_NEWTON_FIRST_SETTLE = 5   # gradient steps before relax()'s Newton-first attempt
_SPOKE_RATIO = 1.5       # relax() splits edges leaving a contact line longer than this many contact-line edges
_FALLBACK_STEPS = 200           # gradient steps for relax()'s conjugate-gradient pass, at most
# Evolver's methods whose in-plane forces are physical (membranes)
_ELASTIC_METHODS = frozenset({
    "linear_elastic", "general_linear_elastic", "linear_elastic_B", "relaxed_elastic",
    "relaxed_elastic1", "relaxed_elastic2", "relaxed_elastic_A", "relaxed_elastic1_A",
    "relaxed_elastic2_A", "dirichlet_elastic", "SVK_elastic", "neo_hookean",
    "stress_integral"})
_TOPOLOGY_PASSES = 5          # relax(): pop / delete collapsed edges, relax again, at most
_COLLAPSED = 0.05              # an edge below this times the median has collapsed
_COLLAPSE_STEPS = 300          # relax(): untidied gradient steps of the last resort
_NEWTON_MAX_RISE = 0.1         # a Newton step raising the energy more (relative) is undone
_STABILITY_NEAR = 100          # relax() checks stability below this many times tol
# a body volume or quantity this far off its target (relative) is missed: a
# new target for Newton to meet, or a gradient round gone wrong (vertex
# averaging alone leaves volumes about 2e-5 off)
_TARGET_LOST = 1e-4
_GRADIENT_ONLY_TOL = 1e-9      # energy_tol when no Newton follows
_ADAPT_GROWTH = 4        # adaptive passes stop past this many times the facets relax() began with


def _edge_turn(m: Mesh, valence: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Per edge, the angle (degrees, 0-90) between the vertex normals at its
    ends, and whether it was judged: not for edges touching a vertex where
    three or more facets meet on an edge (their angle is 0). Vertex normals:
    area-weighted facet normals, each sign-aligned with one facet at the
    vertex (facet orientations needn't agree across a sheet). ``valence``:
    _edge_valence(m)."""
    f, x = m.facets, m.vertices
    assert f is not None
    n = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    ref = np.zeros_like(x)
    for k in range(3):
        ref[f[:, k]] = n
    vn = np.zeros_like(x)
    for k in range(3):
        s = np.where(np.einsum("ij,ij->i", n, ref[f[:, k]]) < 0, -1.0, 1.0)
        np.add.at(vn, f[:, k], n*s[:, None])
    norm = np.linalg.norm(vn, axis=1)
    vn = vn/np.where(norm > 0, norm, 1.0)[:, None]
    e = m.edges
    turn = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", vn[e[:, 0]], vn[e[:, 1]])),
                                        0.0, 1.0)))
    junction = np.zeros(len(x), dtype=bool)
    hub = e[valence > 2]
    junction[hub.ravel()] = True
    unjudged = junction[e[:, 0]] | junction[e[:, 1]] | (norm[e[:, 0]] == 0) | (norm[e[:, 1]] == 0)
    turn[unjudged] = 0.0
    return turn, ~unjudged


def _long_spokes(m: Mesh, ratio: float) -> np.ndarray:
    """Edges with one end on a contact line (a vertex of an edge on one facet
    only) and the other off it, longer than ``ratio`` times the mean length of
    the contact-line edges at that end."""
    if m.facets is None or len(m.facets) == 0:
        return np.zeros(len(m.edges), dtype=bool)
    e, n = m.edges, len(m.vertices)
    lengths, valence = _edge_lengths(m), _edge_valence(m)
    rim = valence == 1
    total, count = np.zeros(n), np.zeros(n)
    for k in (0, 1):
        np.add.at(total, e[rim, k], lengths[rim])
        np.add.at(count, e[rim, k], 1)
    on = count > 0
    mean = total/np.maximum(count, 1)
    spoke = (valence >= 2) & (on[e[:, 0]] != on[e[:, 1]])
    ref = np.where(on[e[:, 0]], mean[e[:, 0]], mean[e[:, 1]])
    return spoke & (lengths > ratio*ref)


def _edge_lengths(m: Mesh) -> np.ndarray:
    return np.linalg.norm(m.vertices[m.edges[:, 0]] - m.vertices[m.edges[:, 1]], axis=1)


def _edge_valence(m: Mesh) -> np.ndarray:
    """The number of facets on each edge (aligned with m.edges)."""
    if m.facets is None or len(m.facets) == 0:
        return np.zeros(len(m.edges), dtype=int)
    f = m.facets
    pairs = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    keys = pairs[:, 0]*(len(m.vertices) + 1) + pairs[:, 1]
    uniq, counts = np.unique(keys, return_counts=True)
    e = np.sort(m.edges, axis=1)
    ekeys = e[:, 0]*(len(m.vertices) + 1) + e[:, 1]
    idx = np.searchsorted(uniq, ekeys)
    idx = np.clip(idx, 0, len(uniq) - 1)
    return np.where(uniq[idx] == ekeys, counts[idx], 0)


class Parameters(MutableMapping):
    """Live view of the parameters declared in the loaded datafile.

    Only existing parameters can be assigned, and none can be deleted.
    """

    def __init__(self, evolver: Evolver):
        self._evolver = evolver

    def _snapshot(self):
        _, result = self._evolver._call(_core.parameters)
        return result.data

    def __getitem__(self, name: str) -> float:
        for key, value, _ in self._snapshot():
            if key == name:
                return value
        raise KeyError(name)

    def __setitem__(self, name: str, value: float) -> None:
        if name not in self:
            raise KeyError(f"{name!r} is not a parameter of this datafile")
        self._evolver.command(f"{name} := {float(value)!r}")

    def __delitem__(self, name: str) -> None:
        raise TypeError("Evolver parameters can't be deleted")

    def __iter__(self) -> Iterator[str]:
        return iter([name for name, _, _ in self._snapshot()])

    def __len__(self) -> int:
        return len(self._snapshot())

    def __contains__(self, name: object) -> bool:
        return any(key == name for key, _, _ in self._snapshot())

    @property
    def optimizing(self) -> frozenset:
        """Names of the optimizing parameters."""
        return frozenset(name for name, _, opt in self._snapshot() if opt)

    def __repr__(self) -> str:
        return repr({name: value for name, value, _ in self._snapshot()})

    def _repr_html_(self) -> str:
        rows = [(name, _html.number(value), "optimizing" if opt else "")
                for name, value, opt in self._snapshot()]
        if not rows:
            return _html.fields("Parameters", [("none", "")])
        return _html.table(rows, ["parameter", "value", ""], title="Parameters")
