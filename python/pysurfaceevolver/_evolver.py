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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, Generator, List, Optional, Union

import numpy as np

from . import _core, _html
from ._build import Body, make_datafile
from ._mesh import Bodies, BodySurface, Mesh, Quantity, is_watertight

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


class EvolverError(RuntimeError):
    """An error reported by Surface Evolver. The surface is still usable."""

    def __init__(self, message: str, errnum: int = 0, output: str = ""):
        super().__init__(message)
        self.errnum = errnum
        self.output = output


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
``pse.busy_timeout`` is None, or ran out), or the running call itself (a
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

    ``energy``, ``area`` and ``scale`` have one entry per gradient iteration.
    ``converged`` and ``hessian_steps`` are set by :meth:`Evolver.relax`.
    """

    energy: np.ndarray
    area: np.ndarray
    scale: np.ndarray
    output: str
    converged: Optional[bool] = None
    hessian_steps: int = 0

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
        output, _ = self._call(_core.command, text)
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

    def relax(self, tol: float = 1e-10, max_iter: int = 1000, *, window: int = 5,
              hessian: bool = False, max_hessian: int = 10,
              callback: Optional[Callable[["Evolver", int], Any]] = None,
              every: int = 1, threads: Optional[int] = None) -> IterationResult:
        """Iterate until the energy stops changing.

        Runs gradient iterations until the relative energy change
        ``|E[i] - E[i-1]| / max(1, |E[i]|)`` stays below ``tol`` for
        ``window`` consecutive iterations, or ``max_iter`` is reached.
        With ``hessian=True``, Newton steps (``hessian``) follow, until one
        changes the energy by less than ``tol`` (at most ``max_hessian``).

        Returns the gradient iterations' :class:`IterationResult`, with
        ``converged`` and ``hessian_steps`` set. ``callback(ev, i)`` works as
        in :meth:`iterate`. It doesn't refine; refine and relax again for a
        finer surface. ``threads`` sets the threads for this call only (see
        :func:`pysurfaceevolver.threads_limit`).
        """
        with _threads_for_call(threads):
            return self._relax(tol, max_iter, window=window, hessian=hessian,
                               max_hessian=max_hessian, callback=callback, every=every)

    def _relax(self, tol: float, max_iter: int, *, window: int, hessian: bool,
               max_hessian: int, callback: Optional[Callable[["Evolver", int], Any]],
               every: int) -> IterationResult:
        """relax() without the thread setting."""
        if tol <= 0 or max_iter < 1 or window < 1:
            raise ValueError("tol must be positive; max_iter and window at least 1")
        energy: List[float] = []
        area: List[float] = []
        scale: List[float] = []
        output: List[str] = []
        previous = _core.total_energy()
        quiet = 0
        converged = False
        for i in range(1, max_iter + 1):
            self._step(energy, area, scale, output)
            if callback is not None and i % every == 0:
                callback(self, i)
            change = abs(energy[-1] - previous) / max(1.0, abs(energy[-1]))
            previous = energy[-1]
            quiet = quiet + 1 if change < tol else 0
            if quiet >= window:
                converged = True
                break
        if callback is not None and len(energy) % every != 0:
            callback(self, len(energy))
        steps = 0
        if hessian:
            converged = False
            for steps in range(1, max_hessian + 1):
                before = _core.total_energy()
                output.append(self.command("hessian"))
                after = _core.total_energy()
                if abs(after - before) / max(1.0, abs(after)) < tol:
                    converged = True
                    break
        return IterationResult(np.array(energy), np.array(area), np.array(scale),
                               "".join(output), converged, steps)

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

    def fix(self, element: str, where=None) -> None:
        """Fix vertices, edges or facets (all, or those selected by ``where``)."""
        name, ids, mask = self._selected(element, where)
        self._run_statements(f"set {name}[{i}] fixed" for i in ids[mask])

    def unfix(self, element: str, where=None) -> None:
        """Unfix vertices, edges or facets (all, or those selected by ``where``)."""
        name, ids, mask = self._selected(element, where)
        self._run_statements(f"unset {name}[{i}] fixed" for i in ids[mask])

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
        points, so any constraint formula works. ``cap_constraints`` says
        which constraint each cap is on (None: not on one; a flat cap).
        """
        mesh = self.mesh()
        if not cap:
            return mesh.body_surfaces(curved, n)
        on = self._vertex_constraints()
        index = _vertex_index(mesh.vertices)

        def project(points, loop_points):
            con = self._loop_constraint(index, on, loop_points)
            return points if con is None else self._project_points(points, con)

        surfaces = mesh.body_surfaces(curved, n, cap=True, project=project)
        for s in surfaces.values():   # the constraint of each cap, from its rim
            assert s.cap_ids is not None
            rim = np.zeros(len(s.points), bool)
            rim[s.cells[s.cap_ids == 0].ravel()] = True
            for k in s.cap_constraints:
                cap_points = np.zeros(len(s.points), bool)
                cap_points[s.cells[s.cap_ids == k].ravel()] = True
                s.cap_constraints[k] = self._loop_constraint(index, on, s.points[cap_points & rim])
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
             screenshot: Optional[str] = None, **kwargs: Any) -> Any:
        """Show the surface with PyVista.

        ``scalars`` colors it: an Evolver expression evaluated per facet
        (or per vertex with ``element="vertex"``), or an array with one value
        per facet or vertex row. Curved elements are tessellated with ``n``
        subdivisions. Other keyword arguments go to ``Plotter.add_mesh``.
        """
        from ._viz import _import_pyvista, surface_dataset
        pv = _import_pyvista()
        dataset, name = surface_dataset(self, scalars, element, n)
        plotter = pv.Plotter(off_screen=off_screen)
        plotter.add_mesh(dataset, scalars=name, **{"show_edges": True, **kwargs})
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

    def bodies(self) -> Bodies:
        """Return volumes, target volumes and pressures of all bodies."""
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
