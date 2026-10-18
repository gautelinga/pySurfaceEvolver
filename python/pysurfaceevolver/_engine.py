"""The engine: running Evolver commands, reading and writing the surface."""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import sys
import tempfile
import threading
import warnings
import weakref
from typing import Callable, Dict, Iterator, Generator, List, Optional, Tuple, Union

import numpy as np

from . import _core, _html
from ._errors import (EvolverBusyError, EvolverError, EvolverExit,
                      EvolverFatalError, EvolverWarning, InvalidSurfaceError,
                      SharedEngineWarning, _command_hints, _hide_capture_wrapper, _num)
from ._mesh import Bodies, Mesh, Quantity
from ._results import Snapshot
from ._views import BodyView, Parameters, QuantityView


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




# Evolver's own variables people set most, for suggestions on a typo
_COMMON_VARIABLES = ["scale_limit", "target_tolerance", "gravity_constant", "scale",
                     "constraint_tolerance", "brightness", "background", "diffusion",
                     "hessian_epsilon", "hessian_slant_cutoff", "jiggle_temperature",
                     "autochop_length", "random_seed", "thickness"]


def _scratch_name(element_type: int, what: str) -> str:
    """A scratch attribute's name: one per element type, since Evolver's
    attribute names are global (defining one on a second type fails)."""
    return f"pyse_{_ELEMENT_NAMES[element_type]}_{what}"


def _element_type(element: str) -> int:
    try:
        return _ELEMENT_TYPES[element]
    except KeyError:
        raise ValueError(
            f"unknown element type {element!r}; "
            "use 'vertex', 'edge', 'facet' or 'body'") from None

# mesh() result for the engine's current surface version (see _core.surface_version)
_mesh_cache: "tuple[int, Mesh] | None" = None
_mesh_cache_lock = threading.Lock()



# the names of the one-sided copies of walls that keep free vertices out of
# solids during changes of topology (_surgery.guard)
GUARD_PREFIX = "pyse_guard_"


class _SurfaceMemory:
    """What pySE remembers about the loaded surface: reset when a surface is
    loaded, kept by restore() (the same surface)."""

    def __init__(self) -> None:
        self.generation = 0                 # surfaces loaded so far in this process
        self.name = ""                      # the loaded datafile, as given
        self.reset()

    def reset(self) -> None:
        self.newton_first_failed = False    # relax(): the next call skips the Newton-first attempt
        self.collapse_failures = 0          # relax(): failed last-resort attempts in a row
        self.collapse_skip = 0              # relax(): calls still to skip before the next attempt
        self.walls: Dict[object, object] = {}   # load_arrays(): constraints with a shape
        # contact(): wet poles gained (+1) or lost (-1), as (sign, the wall's
        # pole_energy): the energy the walls' integrals leave out changes
        self.energy_shift: List[Tuple[int, str]] = []
        # move_poles(): the constraints moved off (names, or numbers where
        # unnamed; Evolver keeps them defined), left out of _constraint_gaps
        self.retired: set = set()
        self.unrelaxed_refines = 0          # refine() calls since relax() last converged


surface = _SurfaceMemory()


class _Saved:
    """A surface saved in memory (_Engine._save), for restore() into the same
    load: a copy of Evolver's elements, freed with this object. pySE's own
    save and restore inside its runs; Snapshot is the portable one. A
    restore() of a Snapshot reloads the surface, which the copy can't be
    restored into: it is turned into a Snapshot first (``dump``)."""

    __slots__ = ("handle", "walls", "generation", "energy_shift", "retired",
                 "dump", "__weakref__")

    def __init__(self, handle: int) -> None:
        self.handle = handle
        self.walls = dict(surface.walls)
        self.generation = surface.generation
        self.energy_shift = list(surface.energy_shift)
        self.retired = frozenset(surface.retired)
        self.dump: Optional[Snapshot] = None
        _live.add(self)

    def free(self) -> None:
        free = getattr(_core, "snapshot_free", None)   # (None at interpreter exit)
        if free is not None and self.handle >= 0:
            free(self.handle)
        self.handle = -1

    def __del__(self) -> None:
        self.free()

    def __reduce__(self):
        raise TypeError("a snapshot in memory belongs to the process that took it: "
                        "use Evolver.save() for one to keep or send")


_live: "weakref.WeakSet[_Saved]" = weakref.WeakSet()




def _read_only(*arrays: Optional[np.ndarray]) -> None:
    for a in arrays:
        if a is not None:
            a.flags.writeable = False

class _Engine:
    """The engine of this process: commands, and the surface's data (see :class:`Evolver`)."""

    def __init__(
        self,
        datafile: Optional[str | os.PathLike] = None,
        *,
        echo: bool = False,
        input: Optional[Callable[[str], Optional[str]]] = None,
        process: bool = False,
        threads: Optional[int] = None,
    ):
        if threads is not None:
            raise ValueError("threads= is for Evolver(process=True); for the engine of "
                             "this process use pyse.set_threads")
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
        self._check_handle()
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

    def _check_handle(self, stacklevel: int = 4) -> None:
        """Warn (once) when this handle loaded a surface and another has
        loaded a different one since."""
        mine = self.__dict__.get("_generation")
        if mine is not None and mine != surface.generation \
                and self.__dict__.get("_warned") != surface.generation:
            self._warned = surface.generation
            warnings.warn(
                f"this Evolver handle loaded {self.__dict__.get('_name', '')!r}, but another "
                f"handle has since loaded {surface.name!r}: every handle is a view of the one "
                "engine in this process, so this one now shows that surface too (keep "
                "surfaces with save()/restore(), or work on several at once with pyse.map)",
                SharedEngineWarning, stacklevel=stacklevel)

    def load(self, datafile: str | os.PathLike) -> str:
        """Load a ``.fe`` datafile, replacing the current surface.

        Evolver looks for the file in the current directory, then in the
        directories listed in the ``EVOLVERPATH`` environment variable.
        Returns the output printed while loading.

        If the datafile has errors, this raises :class:`EvolverError`, and
        there is no valid surface until a datafile loads successfully.
        """
        output = self._load(os.fspath(datafile))
        self._check_bodies()
        return output

    def _load(self, path: str) -> str:
        surface.reset()
        surface.generation += 1
        surface.name = path
        self._generation, self._name = surface.generation, path
        try:
            output, _ = self._call(_core.load, path)
        except EvolverError as e:
            if e.errnum == _core.ERR_NO_DATAFILE:
                raise FileNotFoundError(f"Cannot open datafile {path!r}") from None
            raise
        return output

    def _check_bodies(self) -> None:
        """Warn about bodies that start inside out: a negative volume against
        a positive target (faces oriented into the body)."""
        if not _core.count(_core.BODY):
            return
        b = self.bodies()
        for i, v, t, fixed in zip(b.ids, b.volume, b.target, b.fixed):
            if fixed and t > 0 and v < 0:
                warnings.warn(
                    f"body {i} starts with volume {v:.4g} against its target {t:.4g}: its "
                    "faces are probably oriented into it, and relaxing would turn it inside "
                    "out. Reverse their vertex order, or give the body's orientation "
                    "(Body(orientation=...), -face in a datafile).", UserWarning, stacklevel=3)

    def load_string(self, text: str, name: str = "surface.fe") -> str:
        """Load a datafile given as a string.

        The text goes to a temporary file first, so ``#include`` and ``read``
        paths are resolved from the current directory.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            with open(path, "w") as f:
                f.write(text)
            try:
                output = self._load(path)
            except EvolverError as e:       # name the text, not the temporary file
                e.args = (str(e).replace(path, name),) + e.args[1:]
                raise
        _core.set_datafile(name)            # not the temporary path
        surface.name = self._name = name
        self._check_bodies()
        return output

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
        """``ev["scale_limit"] = 0.5`` assigns a parameter or a variable that
        exists already (Evolver's own, or one from the datafile or
        :meth:`define`): a misspelt name raises ``KeyError`` instead of
        quietly creating a new variable."""
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise KeyError(f"not a variable name: {name!r}")
        if self.eval(f'is_defined("{name}")') == 0:
            close = difflib.get_close_matches(name, list(self.parameters) + _COMMON_VARIABLES, 3)
            raise KeyError(f"{name!r} is not a parameter or variable of this surface"
                           + (f"; did you mean {' or '.join(map(repr, close))}?" if close
                              else "") + " (ev.define(name, value) makes a new variable)")
        self.command(f"{name} := {_num(value, name)}")

    def define(self, name: str, value: float) -> None:
        """Make a new variable (or set one), for use in commands and
        expressions: ``ev.define("h", 0.1)``, then ``ev.command("t h")``."""
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError(f"not a variable name: {name!r}")
        self.command(f"{name} := {_num(value, name)}")

    @contextlib.contextmanager
    def _toggle(self, name: str, on: bool) -> Iterator[None]:
        """One of Evolver's toggles on (or off) for a block; restored after."""
        before = self.eval(name) != 0
        if before != on:
            self.command(f"{name} {'on' if on else 'off'}")
        try:
            yield
        finally:
            if before != on and _core.surface_valid():
                self.command(f"{name} {'on' if before else 'off'}")

    def refine(self, times: int = 1) -> None:
        """Refine the surface: split every edge and facet (Evolver's ``r``)."""
        if times < 0:
            raise ValueError("times must be non-negative")
        for _ in range(times):
            self.command("r")
            surface.unrelaxed_refines += 1

    def equiangulate(self) -> None:
        """Flip edges to improve triangle shapes (Evolver's ``u``)."""
        self.command("u")

    def vertex_average(self) -> None:
        """Move vertices to the average of their neighbors (Evolver's ``V``)."""
        self.command("V")

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
                return Snapshot(f.read(), name, dict(surface.walls), surface.generation,
                                list(surface.energy_shift), frozenset(surface.retired))

    def _save(self) -> "_Saved | Snapshot":
        """save() in memory, for restore() into this load (see _Saved): no
        dump and reload, the elements' memory kept. A surface not yet
        converted to quantities (Evolver converts it at the first Hessian,
        and can't go back) is saved as a dump: a restore after the
        conversion would mix the two."""
        if not _core.everything_quantities():
            return self.save()
        _, result = self._call(_core.snapshot_take)
        return _Saved(result.data)

    def restore(self, snapshot: "Snapshot | _Saved") -> None:
        """Replace the surface with a snapshot from :meth:`save`."""
        if isinstance(snapshot, _Saved):
            if snapshot.dump is not None:
                self.restore(snapshot.dump)
                return
            if snapshot.generation != surface.generation:
                raise ValueError("restore(): that snapshot was taken in memory from a "
                                 "surface loaded before this one")
            self._call(_core.snapshot_restore, snapshot.handle)
            surface.walls = dict(snapshot.walls)
            surface.energy_shift = list(snapshot.energy_shift)
            surface.retired = set(snapshot.retired)
            return
        # (the reload below ends the copies in memory of this surface: dumped
        # first. As relax() keeps a failed attempt in memory, goes back to its
        # start, saved before the first Hessian, and then maybe to the attempt)
        for saved in [k for k in _live if k.dump is None and k.handle >= 0
                      and k.generation == surface.generation]:
            self.restore(saved)
            saved.dump = self.save()
            saved.free()
        # back into the load it came from: the same surface, whose memory
        # (relax()'s failed attempts, the load count) stays; otherwise it is
        # a new surface to the other handles
        same = snapshot.generation == surface.generation
        kept = dict(vars(surface))
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            # what loading pySE's own dump prints is no news (edges with the
            # same endpoints in a torus, ...)
            warnings.simplefilter("ignore", EvolverWarning)
            path = os.path.join(tmp, "snapshot.fe")
            with open(path, "w") as f:
                f.write(snapshot.text)
            self._load(path)
        _core.set_datafile(snapshot.datafile)
        if same:
            vars(surface).update(kept)
        surface.walls = dict(snapshot.walls)
        surface.energy_shift = list(getattr(snapshot, "energy_shift", []))   # (older pickles)
        surface.retired = set(getattr(snapshot, "retired", ()))
        surface.name = snapshot.datafile
        self._generation, self._name = surface.generation, surface.name

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

    def _mark(self, element_type: int, where) -> str:
        """Mark the elements ``where`` selects (a mask or rows) in the scratch
        attribute ``pyse_<type>_mark``, for one ``set ... where`` command instead of
        a statement per element. Returns the where clause ("" for all)."""
        if where is None:
            return ""
        n = _core.count(element_type)
        mask = self._mask(element_type, where, n)
        name = _scratch_name(element_type, "mark")
        self._scratch(element_type, name, mask.astype(float), "integer")
        return f" where {name} == 1"

    def _scratch(self, element_type: int, name: str, values: np.ndarray, dtype: str) -> None:
        """Write a scratch attribute (defined on first use), one value per element."""
        if not _core.fast_attribute(element_type, name):
            self.command(f"define {_ELEMENT_NAMES[element_type]} attribute {name} {dtype}")
        self._call(_core.set_values, element_type, name, np.ascontiguousarray(values), None)

    def set_values(self, element: str, attribute: str, values, *, where=None) -> None:
        """Set a numeric attribute on every element of a type.

        ``values`` is one number per element, aligned with :meth:`mesh` /
        :meth:`values` rows, or a single number for all. ``where`` (a
        boolean mask or rows) limits which elements change. Works for any
        attribute Evolver's ``set`` command accepts: coordinates (``x``,
        ``y``, ...), ``density``, ``tension``, ``target`` (bodies), extra
        attributes from :meth:`define_attribute`, ...

        Coordinates and scalar extra attributes are written directly in C;
        other attributes through one Evolver ``set`` command (the values
        and selection go in scratch attributes, ``pyse_<type>_value`` and
        ``pyse_<type>_mark``).
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
        n = _core.count(element_type)
        if n == 0:
            return
        clause = self._mark(element_type, where)
        value = _scratch_name(element_type, "value")
        self._scratch(element_type, value,
                      np.broadcast_to(np.asarray(values, dtype=float), (n,)).astype(float), "real")
        self.command(f"set {_ELEMENT_NAMES[element_type]} {attribute} {value}{clause}")

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
        element_type = _element_type(element)
        if _core.count(element_type) == 0:
            return
        clause = self._mark(element_type, where)
        self.command(f"{'set' if on else 'unset'} {_ELEMENT_NAMES[element_type]} {flag}{clause}")

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
        nums, _, _, names = self._constraint_gaps()
        known = [int(c) for c in nums] + [n for n in names if n]
        if (constraint if isinstance(constraint, str) else int(constraint)) not in known:
            close = difflib.get_close_matches(str(constraint), [str(k) for k in known], 2)
            raise KeyError(f"no constraint {constraint!r}"
                           + (f"; did you mean {' or '.join(map(repr, close))}?" if close else
                              f" (the surface has {known})"))
        name = _ELEMENT_NAMES[_element_type(element)]
        if _core.count(_element_type(element)) == 0:
            return np.zeros(0, dtype=bool)
        return self.values(name, f"on_constraint {constraint}") != 0

    def _constraint_gaps(self, guards: bool = False
                         ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]":
        """Constraint numbers, attribute bits, per vertex the signed distance
        to each (NaN where on it), and their names ("" if unnamed). Without
        ``guards``, not the one-sided copies of walls that changes of
        topology keep free vertices out of solids with (named
        GUARD_PREFIX...: see _surgery.guard). Never the constraints
        _surgery.move_poles moved off (_SurfaceMemory.retired)."""
        _, gaps = self._call(_core.constraint_gaps)
        nums, attrs, dist, names = gaps.data
        keep = [i for i, n in enumerate(names)
                if (guards or not str(n).startswith(GUARD_PREFIX))
                and (n or int(nums[i])) not in surface.retired]
        if len(keep) == len(names):
            return gaps.data
        return (np.asarray(nums)[keep], np.asarray(attrs)[keep],
                np.asarray(dist)[:, keep] if np.asarray(dist).ndim == 2 else np.asarray(dist),
                [names[i] for i in keep])

    def set_constraint(self, element: str, constraint: Union[int, str], where=None,
                       on: bool = True) -> None:
        """Put elements on a constraint (or take them off with ``on=False``)."""
        if isinstance(constraint, str) and not re.fullmatch(r"[A-Za-z_]\w*", constraint):
            raise ValueError(f"not a constraint name: {constraint!r}")
        element_type = _element_type(element)
        if _core.count(element_type) == 0:
            return
        clause = self._mark(element_type, where)
        self.command(f"{'set' if on else 'unset'} {_ELEMENT_NAMES[element_type]} "
                     f"constraint {constraint}{clause}")

    def define_attribute(self, element: str, name: str, dtype: str = "real") -> None:
        """Define an extra per-element attribute (``real`` or ``integer``).

        Read it with :meth:`values` and write it with :meth:`set_values`.
        """
        self.command(f"define {_ELEMENT_NAMES[_element_type(element)]} attribute {name} {dtype}")

    # ------------------------------------------------------------------
    # Surface data

    @property
    def datafile(self) -> str:
        """Name of the loaded datafile, or ``""`` if none."""
        self._check_handle(3)
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
        self._check_handle(3)
        return _core.total_energy()

    @property
    def energy(self) -> float:
        """The total energy including what the walls' line integrals leave
        out (:attr:`pysurfaceevolver.constraints.Constraint.energy_constant`,
        added by :func:`make_datafile`; Evolver can't add a constant):
        :attr:`total_energy` plus that constant."""
        from ._build import ENERGY_CONSTANT
        energy = _core.total_energy()
        if self.eval(f'is_defined("{ENERGY_CONSTANT}")') != 0:
            energy += self.eval(f"{ENERGY_CONSTANT}()")
        for sign, text in surface.energy_shift:
            energy += sign*self.eval(text)
        return energy

    @property
    def total_area(self) -> float:
        """Total area as of the last iteration or recalculation."""
        self._check_handle(3)
        return _core.total_area()

    @property
    def counts(self) -> dict:
        """Number of vertices, edges, facets and bodies."""
        self._check_handle(3)
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
        # (by id: a body deleted, as two merged into one, leaves its number
        # unused, and the others keep theirs)
        ids = [int(i) for i in self.bodies().ids]
        if number not in ids:
            raise IndexError(f"no body {number}: the surface has bodies {ids}")
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

    def quantity(self, name: str) -> "QuantityView":
        """A live handle on a named quantity from the datafile.

        ``ev.quantity("stretch").modulus = 100`` changes a modulus,
        ``ev.quantity("vol").target = 2`` sets (and fixes) a target and
        ``= None`` frees it. For all quantities at once (a snapshot), use
        :meth:`quantities`.
        """
        if name not in self.quantities():
            raise KeyError(f"no quantity {name!r}; the datafile defines "
                           f"{sorted(self.quantities()) or 'none'}")
        return QuantityView(self, name)

    @property
    def gravity(self) -> float:
        """The gravitational constant (Evolver's ``gravity_constant``; 0: no
        gravity). Bodies weigh their ``density`` times it."""
        return self.eval("gravity_constant")

    @gravity.setter
    def gravity(self, value: float) -> None:
        self.command(f"gravity_constant := {_num(value, 'gravity')}")

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
