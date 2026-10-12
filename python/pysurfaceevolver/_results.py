"""What the engine's operations return."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from . import _core

if TYPE_CHECKING:
    from ._engine import _Engine


class _Coords(NamedTuple):
    coords: np.ndarray
    stamp: int                    # Evolver's topology stamp when taken
    # the optimizing parameters' values (steps move them with the vertices)
    parameters: Tuple[Tuple[str, float], ...] = ()


class EigenCounts(NamedTuple):
    """Hessian eigenvalues below, at and above a shift; see :meth:`Evolver.eigen_counts`."""

    negative: int
    zero: int
    positive: int

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

    def step(self, ev: "_Engine") -> None:
        self.output.append(ev.command("g 1"))
        self.energy.append(_core.total_energy())
        self.area.append(_core.total_area())
        self.scale.append(ev.eval("scale"))

    def refresh(self) -> None:
        """The last step's energy and area as they are now (after tidying it,
        or undoing a round)."""
        self.energy[-1] = _core.total_energy()
        self.area[-1] = _core.total_area()

    def mark(self) -> "tuple[int, ...]":
        """Where the trace stands, for :meth:`rewind`."""
        return tuple(len(x) for x in (self.energy, self.area, self.scale, self.level, self.output))

    def rewind(self, mark: "tuple[int, ...]") -> None:
        """Forget what came after :meth:`mark`."""
        lists: "tuple[list, ...]" = (self.energy, self.area, self.scale, self.level, self.output)
        for x, n in zip(lists, mark):
            del x[n:]

@dataclass(frozen=True)
class Snapshot:
    """A saved surface from :meth:`Evolver.save`, as exact datafile text."""

    text: str
    datafile: str
    # what pySE remembers with the surface: its walls (plot(walls=True)) and
    # which load it came from (restoring into the same one keeps relax()'s
    # memory of failed attempts and doesn't count as a new surface)
    walls: Dict[object, object] = field(default_factory=dict, compare=False, repr=False)
    generation: int = field(default=-1, compare=False, repr=False)

    def write(self, path: Union[str, os.PathLike]) -> None:
        """Write the snapshot as a ``.fe`` datafile."""
        with open(path, "w") as f:
            f.write(self.text)
