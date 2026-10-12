"""Live views of bodies, quantities and parameters."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Iterator, List, Optional


from . import _core, _html
from ._errors import _num
from ._mesh import Quantity

if TYPE_CHECKING:
    from ._engine import _Engine as Evolver


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
    def density(self) -> float:
        """The density, for gravity (see :attr:`Evolver.gravity`)."""
        return self._get("density")

    @density.setter
    def density(self, value: float) -> None:
        self._ev.command(f"set body[{self.number}] density {_num(value, 'density')}")

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


class QuantityView:
    """A live handle on one named quantity; see :meth:`Evolver.quantity`.

    Reading an attribute asks Evolver now; setting one changes the surface.
    """

    def __init__(self, ev: Evolver, name: str):
        self._ev = ev
        self.name = name

    def _now(self) -> Quantity:
        return self._ev.quantities()[self.name]

    @property
    def value(self) -> float:
        """The current value."""
        return self._now().value

    @property
    def kind(self) -> str:
        """``"energy"``, ``"fixed"``, ``"info"`` or ``"conserved"``."""
        return self._now().kind

    @property
    def target(self) -> Optional[float]:
        """The prescribed value, or None unless the quantity is fixed.
        Setting a number fixes it at that value; setting None frees it (it
        becomes an info quantity, as Evolver's ``unfix``)."""
        q = self._now()
        return q.target if q.kind == "fixed" else None

    @target.setter
    def target(self, value: Optional[float]) -> None:
        if value is None:
            self._ev.command(f"unfix {self.name}")
        else:
            self._ev.command(f"{self.name}.target := {_num(value, 'target')}; fix {self.name}")

    @property
    def modulus(self) -> float:
        """The factor its value enters the energy with (energy quantities)."""
        return self._now().modulus

    @modulus.setter
    def modulus(self, value: float) -> None:
        self._ev.command(f"{self.name}.modulus := {_num(value, 'modulus')}")

    @property
    def pressure(self) -> float:
        """The Lagrange multiplier of a fixed quantity."""
        return self._now().pressure

    def __repr__(self) -> str:
        q = self._now()
        target = "" if q.kind != "fixed" else f", target {q.target:.6g}"
        return f"<quantity {self.name} ({q.kind}): value {q.value:.6g}{target}>"

def _unknown(name: str, names: List[str]) -> str:
    import difflib
    close = difflib.get_close_matches(name, names, 3)
    return (f"{name!r} is not a parameter of this datafile"
            + (f"; did you mean {' or '.join(map(repr, close))}?" if close else
               f" (it has {', '.join(map(repr, names))})" if names else " (it has none)"))


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
        snapshot = self._snapshot()
        for key, value, _ in snapshot:
            if key == name:
                return value
        raise KeyError(_unknown(name, [key for key, _, _ in snapshot]))

    def __setitem__(self, name: str, value: float) -> None:
        if name not in self:
            raise KeyError(_unknown(name, list(self)))
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
        """Names of the optimizing parameters: ones the energy minimization
        moves too, like vertex coordinates (Evolver's
        ``optimizing_parameter``; see :meth:`optimize`)."""
        return frozenset(name for name, _, opt in self._snapshot() if opt)

    def optimize(self, name: str, on: bool = True) -> None:
        """Make a parameter optimizing (Evolver's ``unfix``): :meth:`Evolver.relax`
        then finds its value too, say the height a loaded joint settles at.
        ``on=False`` fixes it again at its current value."""
        if name not in self:
            raise KeyError(_unknown(name, list(self)))
        self._evolver.command(f"{'unfix' if on else 'fix'} {name}")

    def __repr__(self) -> str:
        return repr({name: value for name, value, _ in self._snapshot()})

    def _repr_html_(self) -> str:
        rows = [(name, _html.number(value), "optimizing" if opt else "")
                for name, value, opt in self._snapshot()]
        if not rows:
            return _html.fields("Parameters", [("none", "")])
        return _html.table(rows, ["parameter", "value", ""], title="Parameters")
