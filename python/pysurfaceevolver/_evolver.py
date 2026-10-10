"""High-level Python interface to Surface Evolver."""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
import threading
import warnings
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, Generator, List,
                    NamedTuple, Optional, Union)

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
    remeshes: int = 0                  # automatic remeshings (relax)

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
        energy: List[float] = []
        area: List[float] = []
        scale: List[float] = []
        output: List[str] = []
        for i in range(1, n + 1):
            self._step(energy, area, scale, output)
            if callback is not None and (i % every == 0 or i == n):
                callback(self, i)
        return IterationResult(np.array(energy), np.array(area), np.array(scale),
                               "".join(output))

    def _step(self, energy: List[float], area: List[float], scale: List[float],
              output: List[str]) -> None:
        output.append(self.command("g 1"))
        energy.append(_core.total_energy())
        area.append(_core.total_area())
        scale.append(self.eval("scale"))

    def relax(self, tol: float = 1e-8, max_iter: int = 1000, *, energy_tol: float = 1e-9,
              window: int = 3, tidy: int = 10, remesh: bool = False, cg: bool = False,
              levels: int = 0,
              newton: int = 20, seek: Optional[bool] = None, undo_if: Optional[Callable[["Evolver"], bool]] = None,
              callback: Optional[Callable[["Evolver", int], Any]] = None,
              every: int = 1, threads: Optional[int] = None) -> IterationResult:
        """Relax the surface to an equilibrium: gradient steps that keep the
        mesh in shape, then safeguarded Newton steps, until :meth:`residual`
        is below ``tol``.

        Gradient steps (Evolver's ``g``; with ``cg=True``, as conjugate
        gradients: far faster on flat or stiff shapes, but they can diverge on
        films bounded by wires, so off by default; your own ``conj_grad``
        setting is restored after) come in rounds of ``tidy`` steps, each
        followed by equiangulation and vertex averaging (``u; V``), which keep
        facets from degenerating where the surface stretches or vertices drift
        (``tidy=0``: no tidying). With ``remesh=True`` (linear model; experimental, off by default), each round
        also checks the edge lengths against graded targets (the bulk edge
        size, but finer next to short boundary edges such as a small contact
        line): when one is more than twice its target, or an interior one less
        than 0.35 of it, as where a contact line travels or a surface grows or
        shrinks, edges are split and deleted toward the targets. The gradient
        phase ends when the energy
        changes by less than ``energy_tol`` (relative) over ``window``
        consecutive rounds, or after ``max_iter`` steps. ``levels=n`` refines
        and repeats it n times.

        Then up to ``newton`` safeguarded Newton steps (see :meth:`newton`;
        ``seek`` and ``undo_if`` as there), until the residual is below ``tol``
        or stops falling. ``newton=0``: gradient steps only.

        If the energy becomes non-finite during the gradient steps, the surface
        is restored to its state before the call and :class:`EvolverError` is
        raised.

        Returns the gradient steps' :class:`IterationResult`, with ``level``,
        ``newton_steps``, ``residual`` and ``converged`` (residual below
        ``tol``). ``callback(ev, i)`` runs every ``every``-th gradient step and
        at the end of each level; ``threads`` sets the threads for this call
        only (see :func:`pysurfaceevolver.threads_limit`).
        """
        if tol <= 0 or energy_tol <= 0 or max_iter < 1 or window < 1:
            raise ValueError("tol and energy_tol must be positive; max_iter and window at least 1")
        if tidy < 0 or levels < 0 or newton < 0:
            raise ValueError("tidy, levels and newton must be non-negative")
        with _threads_for_call(threads), self._conj_grad(cg):
            start = self.save()
            energy: List[float] = []
            area: List[float] = []
            scale: List[float] = []
            level: List[int] = []
            output: List[str] = []
            remeshes = 0
            for lev in range(levels + 1):
                if lev:
                    self.refine()
                n0 = len(energy)
                remeshes += self._relax_level(energy_tol, max_iter, window, tidy,
                                              remesh and self.model == "linear",
                                              callback, every, energy, area, scale, output)
                level += [lev]*(len(energy) - n0)
                if not np.isfinite(_core.total_energy()):
                    bad = _core.total_energy()
                    self.restore(start)
                    raise EvolverError(f"relax(): the energy became {bad} during gradient "
                                       "steps; the surface is back to its state before the call")
            steps = self._newton(newton, seek, tol, undo_if, output) if newton else 0
            residual = self.residual()
        return IterationResult(np.array(energy), np.array(area), np.array(scale),
                               "".join(output), bool(residual < tol), steps,
                               np.array(level, dtype=int), residual, remeshes)

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

    def _edge_targets(self) -> "tuple[Mesh, np.ndarray, np.ndarray, np.ndarray]":
        """Graded target lengths per edge: the bulk size (the 75th percentile of
        the edge lengths), but near a short boundary (a small contact line, a
        wire) no more than the boundary's edges plus half the distance from
        it. Returns the mesh, lengths, targets and which edges are interior."""
        m = self.mesh()
        a, b = m.edges[:, 0], m.edges[:, 1]
        lengths = np.linalg.norm(m.vertices[a] - m.vertices[b], axis=1)
        interior = _edge_valence(m) >= 2
        bulk = float(np.percentile(lengths, 75))
        h = np.full(len(m.vertices), bulk)
        boundary = ~interior
        np.minimum.at(h, a[boundary], lengths[boundary])
        np.minimum.at(h, b[boundary], lengths[boundary])
        for _ in range(100):                  # grow away from the boundary
            hn = h.copy()
            np.minimum.at(hn, a, h[b] + _GRADE*lengths)
            np.minimum.at(hn, b, h[a] + _GRADE*lengths)
            if np.array_equal(hn, h):
                break
            h = hn
        targets = (h[a] + h[b])/2
        targets[boundary] = bulk          # boundary edges: split only when long
        return m, lengths, targets, interior

    def _uneven(self) -> bool:
        """Whether some edge is far from its graded target length."""
        m = self.mesh()
        if len(m.edges) == 0 or m.facets is None:
            return False
        _, lengths, targets, interior = self._edge_targets()
        # well beyond the remeshing's own limits (1.6, 0.5): no back and forth
        return bool((lengths > 2.5*targets).any()
                    or (lengths[interior] < 0.2*targets[interior]).any())

    def _graded_remesh(self) -> None:
        """Delete interior edges shorter than half their target, split edges
        longer than 1.6 times it, equiangulate."""
        if "pyse_mark" not in self._edge_attributes():
            self.command("define edge attribute pyse_mark integer")
        _, lengths, targets, interior = self._edge_targets()
        self.set_values("edge", "pyse_mark", np.where(interior & (lengths < 0.5*targets), 1, 0))
        self.command("delete edge ee where ee.pyse_mark == 1 and ee.valence != 1")
        _, lengths, targets, interior = self._edge_targets()
        self.set_values("edge", "pyse_mark", np.where(lengths > 1.6*targets, 2, 0))
        self.command("refine edge ee where ee.pyse_mark == 2")
        self.command("u")

    def _edge_attributes(self) -> List[str]:
        try:
            self.eval("max(edge, pyse_mark)")
            return ["pyse_mark"]
        except EvolverError:
            return []

    def _relax_level(self, energy_tol: float, max_iter: int, window: int, tidy: int,
                     remesh: bool, callback: Optional[Callable[["Evolver", int], Any]],
                     every: int, energy: List[float], area: List[float],
                     scale: List[float], output: List[str]) -> int:
        """Gradient steps until the energy settles. With tidy=k, 'u; V' every k
        steps (and a remeshing when the edges have grown uneven), and the energy
        compared round to round. Returns the number of remeshings."""
        previous = _core.total_energy()
        quiet = 0
        first = len(energy)
        remeshes = 0
        for i in range(1, max_iter + 1):
            self._step(energy, area, scale, output)
            if callback is not None and i % every == 0:
                callback(self, i)
            if not np.isfinite(energy[-1]):
                break
            if tidy and i % tidy:
                continue                  # mid-round: no convergence check
            if tidy:
                output.append(self.command("u; V"))
                if remesh and remeshes < _MAX_REMESHES and self._uneven():
                    self._graded_remesh()
                    remeshes += 1
                    quiet = 0               # a new mesh: settle again
                energy[-1] = _core.total_energy()
                area[-1] = _core.total_area()
            change = abs(energy[-1] - previous) / max(1.0, abs(energy[-1]))
            previous = energy[-1]
            quiet = quiet + 1 if change < energy_tol else 0
            if quiet >= window:
                break
        steps = len(energy) - first
        if callback is not None and steps % every != 0:
            callback(self, steps)
        return remeshes

    def newton(self, steps: int = 1, *, seek: Optional[bool] = None, tol: Optional[float] = None,
               undo_if: Optional[Callable[["Evolver"], bool]] = None,
               threads: Optional[int] = None) -> int:
        """Up to ``steps`` safeguarded Newton steps.

        Each step is a plain Newton step (``hessian``), or, if that one is
        rejected, one with a line search (``hessian_seek``); ``seek=True`` or
        ``False`` uses only the one kind. A step is rejected (undone) if it
        makes the energy or the :meth:`residual` non-finite, raises both the
        energy and the residual, or makes ``undo_if(ev)`` true; with no step
        accepted, Newton stops. It also stops when the residual stops falling,
        or, with ``tol``, once it is below ``tol``.
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
        if tol is not None and res0 < tol:
            return 0
        kinds = ["hessian_seek"] if seek is True else ["hessian"] if seek is False \
            else ["hessian", "hessian_seek"]
        for _ in range(steps):
            accepted = False
            for kind in kinds:
                snapshot = self.save()
                e0 = _core.total_energy()
                try:
                    output.append(self.command(kind))
                except EvolverError:
                    self.restore(snapshot)
                    continue
                e1 = _core.total_energy()
                res1 = self.residual()
                # with constraints the energy alone is no merit (restoring a
                # volume can cost energy): reject only what got worse on both
                bad = (not np.isfinite(e1) or not np.isfinite(res1)
                       or (e1 > e0 + 1e-12*max(1.0, abs(e0)) and res1 > 1.1*res0)
                       or (undo_if is not None and undo_if(self)))
                if bad:
                    self.restore(snapshot)
                    continue
                accepted = True
                break
            if not accepted:
                break
            kept += 1
            progress = res1 < 0.9*res0
            res0 = res1
            if tol is not None and res1 < tol:
                break
            if not progress:
                break
        return kept

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

    def residual(self) -> float:
        """How far the surface is from equilibrium, as a dimensionless number
        (0 at an equilibrium; nothing moves).

        From the vertex velocities a gradient step would use (forces projected
        on the constraints, with the volume and quantity multipliers applied),
        the part along each vertex's normal: tangential parts only slide
        vertices within the surface (mesh motion, not shape), and Newton steps
        don't remove them. At a contact line the velocity already lies in the
        wall, so the normal part is the imbalance of the contact angle. Where
        a vertex has no single normal (where films meet) the whole velocity
        counts. Each velocity is about the vertex's share of the area times the
        local pressure imbalance, so the root mean square times N/sqrt(area)
        measures that imbalance relative to the surface's size: comparable
        across meshes and scales. Curved (quadratic, Lagrange) models: the
        facet corners only.
        """
        _, result = self._call(_core.residual)
        vel = np.asarray(result.data, dtype=float)
        if len(vel) == 0:
            return 0.0
        m = self.mesh()
        if m.facets is None or len(m.facets) == 0:
            r2 = (vel**2).sum(axis=1)
        else:
            p = m.vertices if m.vertices.shape[1] == 3 else np.column_stack(
                [m.vertices, np.zeros(len(m.vertices))])
            f = m.facets
            fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
            vn = np.zeros_like(p)
            spread = np.zeros(len(p))
            for k in range(3):
                np.add.at(vn, f[:, k], fn)
                np.add.at(spread, f[:, k], np.linalg.norm(fn, axis=1))
            corners = np.unique(f)
            vn, spread, vel = vn[corners], spread[corners], vel[corners]
            mag = np.linalg.norm(vn, axis=1)
            smooth = mag > 0.5*spread
            along = np.einsum("ij,ij->i", vel, vn)/np.where(mag > 0, mag, 1.0)
            r2 = np.where(smooth, along**2, (vel**2).sum(axis=1))
        area = _core.total_area()
        n = len(r2)
        if area <= 0:
            return float(np.sqrt(r2.sum()))
        return float(np.sqrt(r2.sum()*n/area))

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
        """Edge lengths and facet shapes; see :meth:`Mesh.quality`."""
        return self.mesh().quality(skinny_angle)

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
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.fe")
            with open(path, "w") as f:
                f.write(snapshot.text)
            self.load(path)
        _core.set_datafile(snapshot.datafile)

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


# how fast graded target edge lengths may grow with distance from a boundary
_GRADE = 1.0
# at most this many automatic remeshings per relax() call
_MAX_REMESHES = 10


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
