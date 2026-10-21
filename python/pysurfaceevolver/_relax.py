"""relax() and the steps it is made of: gradient rounds, safeguarded Newton, remeshing."""

from __future__ import annotations

import contextlib
import re
import sys
import warnings
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union,
                    cast)

import numpy as np

from . import _core
from ._analysis import _TARGET_LOST, _Analysis
from .constraints import ANGLE_FLOOR, ANGLE_FLOOR_TOP
from ._engine import _threads_for_call, surface
from ._errors import EvolverError, EvolverWarning, UnstableEquilibriumWarning, _num
from ._geometry import _edge_turn, _long_spokes
from ._results import Health, IterationResult, _Coords, _Trace

if TYPE_CHECKING:
    from ._events import Event
    from ._evolver import Evolver


_ADAPT_PASSES = 3
_NEWTON_FIRST_SETTLE = 5   # gradient steps before relax()'s Newton-first attempt
_SPOKE_RATIO = 1.5       # relax() splits edges leaving a contact line longer than this many contact-line edges
_FALLBACK_STEPS = 200           # gradient steps for relax()'s conjugate-gradient pass, at most

_TOPOLOGY_PASSES = 5          # relax(): pop / delete collapsed edges, relax again, at most
_COLLAPSED = 0.05              # an edge below this times the median has collapsed
_COLLAPSE_STEPS = 300          # relax(): untidied gradient steps of the last resort
_NEWTON_MAX_RISE = 0.1         # a Newton step raising the energy more (relative) is undone
_NEWTON_RESIDUAL_RISE = 1.1    # ... and one raising both the energy and the residual by this factor
_NEWTON_PROGRESS = 0.9         # Newton goes on while each step cuts the residual below this factor
_TARGET_HALFWAY = 0.5          # ... or brings an unmet target this much closer
_UNRELAXED_REFINES = 4         # relax() warns from this many refinements since it last converged
_NO_FREEDOM = (3004, 3005)      # Evolver: fewer degrees of freedom than constraints
_TRIAL_KEEP = 0.01             # ... and a Newton trial there is kept below this fraction of the residual
_SLOW_ROUND = 0.95             # a round leaving the residual above this fraction of the last's is slow
_NEWTON_STEPS = 20             # derivative(): Newton steps to reconverge, at most
_DERIVATIVE_TOL = 1e-10        # ... down to this residual
_STABILITY_NEAR = 100          # relax() checks stability below this many times tol
# relax() takes stability from the last Newton step's factorization if that
# step began this close to equilibrium (residual) and counted no negative
# eigenvalue: a step so small can't carry one across zero (except at the
# onset of an instability, which the threshold of stability() blurs anyway).
# It matched the full check in all of some 230 relaxes (stress cases, tests,
# Brakke's samples; steps from residuals up to 4e-3).
_INERTIA_NEAR = 1e-2

_ENERGY_TOL = 1e-5             # relax(): gradient rounds settle at this relative change...
_WINDOW = 3                    # ... over this many rounds
_GRADIENT_ONLY_TOL = 1e-9      # ... or this one when no Newton follows
_CONTACT_SPLITS = 3            # relax(): splits of long contact-line spokes, at most
_CONTACT_PASSES = 2           # relax(): relaxes after splitting long contact-line spokes, at most
_ROUND_BLOWUP = 2             # a gradient round more than doubling the residual or a target gap is undone
_STAGE_GAP = 0.25             # relax() stages a target further off than this (relative) if a round folds facets,
_STAGE = 0.1                  # ... each moving the targets by at most this (relative),
_STAGE_STEPS = 20             # ... with this many gradient steps (tidied as usual)
_TURN = 15.0                  # adapt(): default turning angle (degrees) across an edge
_ADAPT_SHORT = 0.75           # adapt() merges edges shorter than this times the bulk length,
_ADAPT_FLAT = 1/3              # ... turning by less than this times max_turn,
_ADAPT_TILT = 1/2              # ... unless a facet tilts more than this times max_turn
_ADAPT_LONGEST = 4/3           # ... or an edge grows longer than this times the bulk length
_SPLIT_ABOVE = 1.6             # remesh(target=h) splits edges longer than this times h
_MERGE_UNDER = 0.5             # ... and deletes those shorter than this times h
_ADAPT_GROWTH = 4        # adaptive passes stop past this many times the facets relax() began with
_SMALL_ANGLE = 15.0           # relax() stages contact angles below this when a start sinks into a wall,
_ANGLE_STEP = 0.7             # ... lowering them from ANGLE_FLOOR_TOP by this factor a stage
_SINK = 1e-3                  # ... sinking: a free vertex deeper in a wall than this times the median edge

class _Relaxing(_Analysis):
    """Moving the surface: gradient and Newton steps, relax(), remeshing."""

    _no_hessian: bool = False      # set by _newton(): no Hessian for this surface
    _keep_mesh: bool = False       # relax(): no tidying (elastic, crystalline)
    # relax(): no junctions popped nor collapsed edges deleted (a topology
    # descent's runs: it makes the changes itself, each one checked)
    _topology_held: bool = False
    # set by _newton(): the coordinates a Newton step left and the negative
    # eigenvalues its factorization counted (where it began)
    _inertia: "tuple[_Coords, int] | None" = None

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
                callback(cast("Evolver", self), i)
        return IterationResult(np.array(trace.energy), np.array(trace.area),
                               np.array(trace.scale), "".join(trace.output))

    def relax(self, tol: float = 1e-8, max_iter: int = 1000, *, levels: int = 0,
              newton: Optional[int] = None, stability: Optional[bool] = None,
              adapt: Union[bool, float] = False, tidy: int = 10, seek: Optional[bool] = None,
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

        * With ``levels=0``, a surface already in equilibrium (residual and
          the gaps to its volume and quantity targets below ``tol``, no
          degenerate facet) is left as it is: only the stability check below
          (and ``adapt``, if given) runs. Gradient steps on it would be mesh
          motion only, which next to a shallow contact line slides vertices
          through the wall.
        * In the quadratic and Lagrange models (a final stage from a mesh
          settled in the linear model) with ``levels=0``, five gradient steps
          and then safeguarded Newton steps come first (as the manual's
          ``g 5; hessian``), Newton as long as each step cuts the residual by
          10%; if that converges, that is all. If not, the
          surface goes back and the steps below follow; right after such a
          miss, the next call skips the attempt.
        * A body volume or fixed quantity whose target is more than 25% from
          its value: if the first gradient step toward it leaves degenerate
          facets (below 1 degree) the surface didn't have, it is undone and
          the target is approached in stages of at most 10%, each followed by
          20 gradient steps (tidied as below): in one go, Evolver's projection
          onto a far target can fold facets.
        * Gradient steps (Evolver's ``g``) come in rounds of ``tidy`` steps,
          each followed by equiangulation and vertex averaging (``u; V``;
          only ``V`` for strings and the Lagrange model, only ``u`` in the
          simplex representation; ``tidy=0``: none), until the energy changes by less than
          1e-5 (relative) over three rounds when Newton follows (it finishes
          faster and more reliably, and long gradient runs let slow mesh drift
          grow), 1e-9 with ``newton=0``, or ``max_iter`` steps. When Newton follows, a round that more than
          doubles the residual from its lowest so far is undone and Newton
          takes over (on a coarse mesh a contact line can collapse within a
          few steps). Rounds that have used half of ``max_iter`` and keep
          slowing without settling (the residual falling by less than 5% a
          round, three rounds running: a contact line creeping over a curved
          wall) get a trial of Newton steps,
          kept if they converge or cut the residual a hundredfold; otherwise
          the surface goes back and the rounds go on (a neck shrinking toward
          a pinch-off creeps the same way).
        * Then up to ``newton`` safeguarded Newton steps (see :meth:`newton`;
          ``seek`` and ``undo_if`` as there), until the residual is below
          ``tol`` or stops falling.
        * If that leaves the surface unconverged, one more try with conjugate
          gradients (far faster on flat or stiff shapes, but they wear the mesh
          down where contact lines travel), then Newton, with up to 200 of the
          steps the level left of ``max_iter``; kept only if that converges.
          Otherwise plain gradient steps (your own ``conj_grad`` setting is
          restored after).
        * With ``levels=n``, if the last level still ends unconverged: the
          levels again from the call's start refined once first, kept only if
          that converges. A coarse start can hold equilibria the shape doesn't
          have, and which one its gradient rounds find can turn on the last
          digit of a volume (a bridge at 20 degrees from a tube 12 facets
          around).
        * A surface that ends unconverged with vertices sunk into a wall from
          :mod:`pysurfaceevolver.constraints` whose contact angle is below 15
          degrees: the levels again from the call's start with the small angles
          raised to 20 degrees, then lowered in stages (by 0.7 each), relaxed
          at each, kept only if that converges at their own angles. At a small
          angle the liquid meets the solid almost tangentially, gradient steps
          from a crude start slide the vertices next to the contact line
          through it, and the wall's integrals reward that (a trimer between
          three beads at 5 degrees). Walls with such angles carry a datafile
          parameter ``pyse_angle_floor`` (0) for this.
        * Once converged (linear model), edges leaving a contact line that are
          longer than 1.5 times the contact-line edges there are split and the
          surface relaxed again, kept only if that converges: a shrinking
          contact circle (a drop at a high contact angle) would otherwise end
          up smaller than the facets next to it.
        * Then (linear model, films and strings) junctions that break
          Plateau's laws are popped (Evolver's ``o``: four films on an edge,
          four strings at a point, ...) and the surface relaxed again, up to
          five times. If it is still unconverged, a last resort: up to 300 gradient steps without tidying (which can
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
        caught: list = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                return self._relax(tol=tol, max_iter=max_iter, levels=levels, newton=newton,
                                   stability=stability, adapt=adapt, tidy=tidy, seek=seek,
                                   undo_if=undo_if, callback=callback, every=every,
                                   threads=threads)
        finally:                    # (after the block: inside it they'd be recorded again)
            if sys.exc_info()[0] is None:
                _reemit(caught)
            else:                   # an error is on its way: -W error mustn't replace it
                try:
                    _reemit(caught)
                except Warning:
                    pass

    def _relax(self, *, tol: float, max_iter: int, levels: int, newton: Optional[int],
               stability: Optional[bool], adapt: Union[bool, float], tidy: int,
               seek: Optional[bool], undo_if: Optional[Callable[["Evolver"], bool]],
               callback: Optional[Callable[["Evolver", int], Any]], every: int,
               threads: Optional[int]) -> IterationResult:
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
        energy_tol = _ENERGY_TOL if newton else _GRADIENT_ONLY_TOL
        if tol <= 0 or max_iter < 1:
            raise ValueError("tol must be positive and max_iter at least 1")
        if tidy < 0 or levels < 0 or newton < 0:
            raise ValueError("tidy, levels and newton must be non-negative")
        self._inertia = None                # only this call's Newton steps count
        refined = surface.unrelaxed_refines
        if refined >= _UNRELAXED_REFINES:
            warnings.warn(
                f"relax(): the surface was refined {refined} times since it last converged; "
                "relaxing it this fine is slow and may not converge (every refinement "
                "multiplies the gradient steps a shape change takes). relax(levels=n) from "
                "the coarser surface relaxes after each refinement instead (on cube.fe five "
                "refinements: 0.4 s converged, against 10 s unconverged).",
                EvolverWarning, stacklevel=3)
        with _threads_for_call(threads), self._conj_grad(False), \
                self._mesh_kept(keep_mesh):
            run = _Run(self, tol=tol, max_iter=max_iter, tidy=tidy, newton=newton, seek=seek,
                       undo_if=undo_if, energy_tol=energy_tol, callback=callback, every=every)
            trace = run.trace
            facets0 = self.counts["facets"]
            first = None
            # a surface already in equilibrium is left as it is: gradient
            # rounds on it are mesh motion only, and next to a shallow contact
            # line they slide vertices through the wall (a bridge at 20 degrees
            # relaxed a second time: 300 steps later 175 vertices sat inside a
            # bead, the energy falling all the way). A surface that passes is
            # checked again after Evolver's recalc, which puts the vertices on
            # their boundaries and constraints as they are now: a wire moved by
            # a parameter leaves them where they were, and the old equilibrium
            # passes. (Only then: recalc changes the path of a surface that
            # goes on to relax.)
            def met() -> bool:
                return (self.residual() < tol and self._target_gap()[0] < tol
                        and not self._degenerate())
            already = False
            if levels == 0 and met():
                self.command("recalc")
                already = met()
            if already:
                first = (0, self.residual())
            elif newton and levels == 0 and self.model != "linear":
                run.approach()
                first = run.newton_first()
            if first is not None:
                steps, residual = first
            else:
                small = None if keep_mesh else self._small_angle()
                start = self._save() if (levels or small is not None) and not keep_mesh else None
                for lev in range(levels + 1):
                    if lev:
                        self.refine()
                    try:
                        run.approach()
                        run.settle(lev)
                    except EvolverError as e:
                        # a start with nothing free to move (Brakke's bga-30:
                        # every vertex on a pad); refining gives it some
                        if e.errnum not in _NO_FREEDOM or lev == levels:
                            raise
                steps, residual = run.finish(levels)
                if start is not None and not residual < tol:
                    # sunk into a wall at a small contact angle: the angles staged
                    # down; otherwise a finer start
                    if small is not None and self._sunk():
                        retry = run.lower_angles(start, levels, small)
                    else:
                        retry = run.finer_start(start, levels) if levels else None
                    steps, residual = retry or (steps, residual)
            if self.model == "linear" and residual < tol and not keep_mesh and not already:
                more, residual = run.contact_pass(levels + 1, residual)
                steps += more
            if adapt and self.model == "linear" and not elastic:
                turn = _TURN if adapt is True else float(adapt)
                first_lev = run.next_level(levels)
                for lev in range(first_lev, first_lev + _ADAPT_PASSES):
                    if self.counts["facets"] > _ADAPT_GROWTH*facets0 or not self.adapt(turn):
                        break
                    more, residual = run.level(lev)
                    steps += more
            popped = collapsed = 0
            if self.model == "linear" and _core.representation() != 3 \
                    and not self._topology_held:
                for _ in range(_TOPOLOGY_PASSES):
                    pops = self.pop()
                    if not pops:
                        break
                    popped += pops
                    more, residual = run.level(run.next_level(levels))
                    steps += more
                if not residual < tol and not keep_mesh:
                    found = run.collapse(levels)
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
                if self._inertia_unchanged() == 0:
                    stable, modes = True, 0     # the last Newton step's factorization
                else:
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
            # (a topology descent's runs: health only where converged, the
            # only place the descent looks at it, and no events, which it
            # finds itself: both were a sixth of a 16-bead pack's descent)
            held = self._topology_held
            health: Optional[Health] = None
            events: Optional[list] = None
            if not held or residual < tol:
                try:
                    health = self._health(residual, stable, modes, tol)
                except EvolverError:
                    health = None
            if not held:
                try:
                    events = self.events()
                except EvolverError:
                    events = None
            settled = trace.settled and not residual < tol
        # (refinements this call made were relaxed after; a converged surface
        # starts the count over)
        surface.unrelaxed_refines = 0 if residual < tol else refined
        return IterationResult(np.array(trace.energy), np.array(trace.area),
                               np.array(trace.scale), "".join(trace.output),
                               bool(residual < tol or (crystalline and settled)), steps,
                               np.array(trace.level, dtype=int), residual, stable, modes,
                               health, popped, collapsed, settled, events)

    def derivative(self, name: str, step: Optional[float] = None, *,
                   move: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                   method: str = "newton") -> float:
        """The derivative of the equilibrium energy (:attr:`energy`) with
        respect to a parameter, by central differences: the parameter set to
        its value plus and minus ``step`` (default 1e-5 times its size, at
        least 1e-5; 1e-3 with ``method="relax"``), the energy found each time, the surface put back as it
        was. With a parameter that moves a wall or a pad, minus this is the
        force on it (Brakke's bga-4 tutorial compares seven ways).

        ``method``:

        * ``"newton"`` (default): safeguarded Newton steps reconverge the
          surface each time, as Brakke's ``zforce5``. Newton moves vertices
          along the normal only, so the mesh doesn't drift between the two
          and the difference is clean even for tiny steps.
        * ``"fixed"``: no reconverging, the energy corrected for the volume
          and quantity changes by their multipliers (minus pressure times
          change), as Brakke's preferred ``zforce4``: as accurate, and far
          cheaper on a large surface, but it needs ``move``.
        * ``"relax"``: :meth:`relax` each time; it tidies the mesh too, which
          changes the energy by more than a small step does, hence its
          larger default step.

        ``move(xyz)``: how the vertices follow a unit change of the
        parameter, an ``(n, sdim)`` array (a pad at height ``h``:
        ``lambda x: np.c_[0*x[:, :2], x[:, 2]/h]``, everything stretched with
        it); the surface is moved that way before the energy is taken, which
        also gives Newton a close start. An optimizing parameter is held
        fixed meanwhile: at its optimum the derivative is zero (with a load
        in the energy, the joint's force balances it). (A parameter that only
        enters a quantity's ``modulus``, which Evolver evaluates once, gives 0.)
        """
        if method not in ("newton", "fixed", "relax"):
            raise ValueError("method must be 'newton', 'fixed' or 'relax'")
        if method == "fixed" and move is None:
            raise ValueError("method='fixed' needs move: how the vertices follow the parameter")
        if name not in self.parameters:
            raise KeyError(f"{name!r} is not a parameter of this datafile")
        value = self.parameters[name]
        # relax() tidies the mesh too, which moves the energy by more than a
        # tiny step would: a larger default step there
        h = (1e-3 if method == "relax" else 1e-5)*max(abs(value), 1.0) if step is None \
            else float(step)
        if not h > 0:
            raise ValueError("step must be positive")
        snapshot = self.save()
        start = self._coords()
        energies = []
        optimizing = name in self.parameters.optimizing
        try:
            if optimizing:              # held where it is set, not optimized
                self.parameters.optimize(name, False)
            for sign in (1, -1):
                if move is not None:
                    x = start.coords
                    shift = np.asarray(move(x), dtype=float)
                    if shift.shape != x.shape:
                        raise ValueError(f"move() must give an {x.shape} array")
                    self.vertices = x + sign*h*shift
                self.parameters[name] = value + sign*h
                if method == "relax":
                    result = self.relax(stability=False)
                    ok = bool(result.converged)
                elif method == "newton":
                    # two steps at least: the residual doesn't see vertices
                    # left off a moved constraint, which a step projects back
                    self.newton(2)
                    self.newton(_NEWTON_STEPS, tol=_DERIVATIVE_TOL)
                    ok = self.residual() < _DERIVATIVE_TOL*100
                else:
                    self.recalc()
                    ok = True
                if not ok:
                    warnings.warn(f"derivative(): the surface didn't reconverge at {name} = "
                                  f"{value + sign*h:.6g}", EvolverWarning, stacklevel=2)
                energies.append(self.energy - (self._constraint_change() if method == "fixed"
                                               else 0.0))
                self.parameters[name] = value
                if sign == 1:       # back for the other side (the last: below)
                    # (Newton and relax() change the multipliers too: all back)
                    if method != "fixed" or not self._put_back(start):
                        self.restore(snapshot)
                        if optimizing:
                            self.parameters.optimize(name, False)
                    else:
                        self.recalc()
        finally:
            if method != "fixed" or not self._put_back(start):
                self.restore(snapshot)
            else:
                self.parameters[name] = value
                self.recalc()
            if optimizing:
                self.parameters.optimize(name)
        return (energies[0] - energies[1])/(2*h)

    def _constraint_change(self) -> float:
        """Pressure times the change of each fixed volume and quantity from
        its target: what holding them would add to the energy, to first order
        (taken off a moved surface's energy, as in Brakke's zforce)."""
        b = self.bodies()
        total = float((b.pressure[b.fixed]*(b.volume[b.fixed] - b.target[b.fixed])).sum()) \
            if len(b.ids) else 0.0
        for q in self.quantities().values():
            if q.kind == "fixed":
                total += q.pressure*(q.value - q.target)
        return total

    def pop(self) -> int:
        """Pop the junctions a minimal surface can't have (Evolver's ``o``:
        four films on an edge, a cube's cone at a vertex, four strings at a
        point, ...), so they can split into Plateau's junctions as the
        surface relaxes. Returns how many were popped. :meth:`relax` does
        this by default in the linear model."""
        out = self.command("o")
        return sum(int(n) for n in re.findall(r"popped:\s*(\d+)", out))

    def saddle(self) -> float:
        """Step off a saddle point (Evolver's ``saddle``): along the
        eigenvector of the lowest Hessian eigenvalue, if it is negative, as
        far as the energy falls. Safe far from equilibrium too (Brakke's
        eigenvalue tutorial). Returns how much the energy fell, 0 at a
        local minimum; follow with :meth:`relax` to settle into the minimum
        it leads to. (Evolver needs pressures from a gradient step first:
        one is taken if none was.)"""
        before = _core.total_energy()
        try:
            self.command("saddle")
        except EvolverError as e:
            if e.errnum != 1606:            # "Pressures invalid": not iterated yet
                raise
            self.command("g 1")
            before = _core.total_energy()
            self.command("saddle")
        return before - _core.total_energy()

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
        clause = self._mark(_core.EDGE, mark)
        before = self.counts["vertices"]
        self.command(f"refine edge{clause} and not no_refine")
        if self._tidy_kinds()[0]:
            self.command("u")
        return self.counts["vertices"] - before

    def _conj_grad(self, on: bool) -> "contextlib.AbstractContextManager[None]":
        """Conjugate gradients on (or off) for a block; the setting restored after."""
        return self._toggle("conj_grad", on)

    def newton(self, steps: int = 1, *, seek: Optional[bool] = None, tol: Optional[float] = None,
               undo_if: Optional[Callable[["Evolver"], bool]] = None,
               threads: Optional[int] = None) -> IterationResult:
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
        Returns an :class:`IterationResult`: ``newton_steps`` (the steps
        kept), ``residual``, and ``converged``: the residual below ``tol``
        (without one, :meth:`relax`'s default 1e-8) and the targets met.
        Stability isn't checked (:meth:`stability`). (:meth:`hessian` takes
        one step with no checks.)
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        output: List[str] = []
        with _threads_for_call(threads):
            kept, residual = self._newton(steps, seek, tol, undo_if, output)
        gap, _ = self._target_gap()
        none = np.empty(0)
        return IterationResult(none, none, none, "".join(output),
                               converged=bool(residual < (1e-8 if tol is None else tol)
                                              and gap <= _TARGET_LOST),
                               newton_steps=kept, residual=residual)

    def _newton(self, steps: int, seek: Optional[bool], tol: Optional[float],
                undo_if: Optional[Callable[["Evolver"], bool]],
                output: List[str], grace: bool = False) -> "tuple[int, float]":
        """Safeguarded Newton steps; see newton(). Returns the steps kept and
        the residual of the surface as it is left."""
        kept = 0
        res0 = self.residual()
        # the residual doesn't see an unmet volume or quantity target (a new
        # one): restoring it is progress, and its cost an allowed energy rise
        gap0, cost = self._target_gap()
        if tol is not None and res0 < tol and gap0 <= _TARGET_LOST:
            return 0, res0
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
                closer = gap0 > _TARGET_LOST and gap1 < _TARGET_HALFWAY*gap0
                # with constraints the energy alone is no merit (restoring a
                # volume can cost energy): reject what got worse on both, or
                # raised the energy by more than a volume correction could (far
                # from equilibrium a step can run off to a tangled surface
                # whose residual is small)
                bad = (not np.isfinite(e1) or not np.isfinite(res1)
                       or (e1 > e0 + 1e-12*max(1.0, abs(e0)) and res1 > _NEWTON_RESIDUAL_RISE*res0
                           and not closer)
                       or e1 > e0 + _NEWTON_MAX_RISE*abs(e0) + cost
                       or (undo_if is not None and undo_if(cast("Evolver", self))))
                if bad:
                    self._put_back(snapshot)
                    continue
                accepted = True
                # (both kinds factor the Hessian where the step begins)
                self._inertia = (self._coords(), int(self.eval("eigen_neg"))) \
                    if res0 < _INERTIA_NEAR else None
                break
            if not accepted:
                # no step at all: this surface has no Hessian (gap energies,
                # simplex volumes outside the Lagrange model, ...)
                self._no_hessian = step == 0 and failed == len(kinds)
                break
            kept += 1
            progress = res1 < _NEWTON_PROGRESS*res0 or closer
            res0 = res1
            gap0, cost = self._target_gap()
            if tol is not None and res1 < tol and gap0 <= _TARGET_LOST:
                break
            if not progress:
                if grace and step == 0:     # (a trial: one slow first step allowed)
                    continue
                break
        return kept, res0

    def _inertia_unchanged(self) -> Optional[int]:
        """The negative eigenvalues the last Newton step counted, if it
        began near equilibrium and the surface is as it left it; else None."""
        if self._inertia is None:
            return None
        coords, negative = self._inertia
        if coords.stamp != _core.topology_stamp() or not np.array_equal(coords.coords,
                                                                        self.vertices):
            return None
        return negative

    def _coords(self) -> "_Coords":
        """The vertex coordinates (and optimizing parameters), to undo steps
        that only move vertices."""
        _, result = self._call(_core.parameters)
        return _Coords(self.vertices, _core.topology_stamp(),
                       tuple((name, value) for name, value, opt in result.data if opt),
                       _core.multipliers())

    def _put_back(self, snapshot: "_Coords") -> bool:
        """Restore coordinates from :meth:`_coords`; False (and nothing done)
        if the topology changed since, as Evolver's autopop and autochop do
        during gradient steps."""
        if snapshot.stamp != _core.topology_stamp() \
                or len(snapshot.coords) != _core.count(_core.VERTEX):
            return False
        if snapshot.parameters:
            self.command("; ".join(f"{name} := {value!r}" for name, value in snapshot.parameters))
        self.vertices = snapshot.coords
        if snapshot.multipliers is not None \
                and len(snapshot.multipliers) == _core.multiplier_count():
            _core.set_multipliers(snapshot.multipliers)
        return True

    def _small_angle(self) -> Optional[float]:
        """The least contact angle below _SMALL_ANGLE degrees among the walls
        given to load_arrays() that relax() can raise (their angle takes
        constraints.ANGLE_FLOOR as its least), or None."""
        if ANGLE_FLOOR not in self.parameters:
            return None
        low: Optional[float] = None
        for wall in surface.walls.values():
            angle = getattr(wall, "contact_angle", None)
            if angle is None or not getattr(wall, "floored", False):
                continue
            try:
                angle = float(self.eval(angle)) if isinstance(angle, str) else float(angle)
            except (EvolverError, ValueError):
                continue
            if 0 <= angle < _SMALL_ANGLE:
                low = angle if low is None else min(low, angle)
        return low

    def _sunk(self) -> bool:
        """Whether a vertex on no constraint lies inside the solid of a wall
        with a small contact angle, deeper than _SINK times the median edge."""
        x = self.vertices
        _, _, dist, _ = self._constraint_gaps()
        free = ~np.isnan(dist).any(axis=1) if dist.size else np.ones(len(x), dtype=bool)
        if not free.any():
            return False
        h = self.mesh_quality().edge_median
        for wall in surface.walls.values():
            angle, shape = getattr(wall, "contact_angle", None), getattr(wall, "shape", None)
            if angle is None or shape is None or not getattr(wall, "floored", False):
                continue
            kind = shape[0]
            if kind == "sphere":
                c = np.array([float(self.eval(v)) if isinstance(v, str) else float(v)
                              for v in shape[1]])
                depth = shape[2] - np.linalg.norm(x[free] - c, axis=1)
            elif kind == "cylinder":
                p, d = np.asarray(shape[1], float), np.asarray(shape[2], float)
                r = x[free] - p
                depth = shape[3] - np.linalg.norm(r - np.outer(r @ d, d), axis=1)
            else:
                depth = shape[2] - x[free] @ np.asarray(shape[1], float)
            if (depth > _SINK*h).any():
                return True
        return False

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
            max_edge = _SPLIT_ABOVE*float(target) if max_edge is None else max_edge
            min_edge = _MERGE_UNDER*float(target) if min_edge is None else min_edge
        if max_edge is not None and min_edge is not None and not 2*min_edge <= max_edge:
            raise ValueError("max_edge must be at least 2*min_edge: split halves "
                             "shorter than min_edge would be deleted again")
        counts = {"deleted": 0, "split": 0, "switched": 0}
        # the protected edges by id: the deletion below renumbers the rows
        protected = None
        if protect is not None:
            protected = self.mesh().edge_ids[
                self._mask(_core.EDGE, protect, _core.count(_core.EDGE))]

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
            clause = ""
            if protected is not None:
                mask = np.isin(self.mesh().edge_ids, protected)
                newly = mask & (self.values("edge", "no_refine") == 0)
                if newly.any():
                    clause = self._mark(_core.EDGE, newly)
                    self.command(f"set edge no_refine{clause}")
            try:
                counts["split"] = number(self.command(
                    f"refine edge where length > {_num(max_edge, 'max_edge')} and not no_refine"))
            finally:
                if clause:
                    self.command(f"unset edge no_refine{clause}")
        if equiangulate:
            counts["switched"] = number(self.command("u"))
        if average:
            self.command("V")
        return counts

    def pinch_off(self, event: Optional[Event] = None, *, relax: bool = True) -> List[int]:
        """Cut the surface at a neck, an :class:`Event` of kind ``"neck"``
        from :meth:`events` (default: the first one there): the two rows of
        vertices round the neck either side of the plane across its
        narrowest point collapse to one vertex, Evolver deletes the collapsed
        edges, the vertex is split into one per side (a checked edit; Evolver's
        ``pop vertex`` where that is refused), so the two sides no longer
        touch, and Evolver makes the body's pieces bodies (``rebody``); the short
        edges round the two tips go (Evolver's edge deletion), and the two
        sides, cones at first, relax (``relax=False`` leaves them so).

        The piece holding the most liquid keeps the body's number. Each
        piece's ``volconst`` is that of the walls it reaches (constraints from
        :mod:`pysurfaceevolver.constraints` that close a volume); the rest of
        the body's, if any, stays with the largest piece. If the volume was
        fixed, the smaller pieces' targets are what they hold, plus what
        collapsing the ring took from their side, and the largest's the rest
        of the old target: they add up to it, to round-off. Relaxing goes in
        two parts: at the volumes the pieces hold after the cut (a few percent
        less, near a neck a couple of edges across), then to the targets by
        Newton steps (going there at once, a drop on a sphere at 40 degrees
        slid through it); a warning says if that fails. Returns the new
        bodies' numbers (none for a surface without bodies).

        The neck must be clear of fixed vertices, constraints and the
        surface's border, and round one body (a film between two bodies
        doesn't pinch this way); it needs a few edges round it (on a bridge
        meshed 16 edges round, the drops failed to relax). Linear model only.
        If the cut itself fails, the surface is put back and ValueError raised.
        """
        from ._surgery import pinch_off
        return pinch_off(self, event, relax)

    def merge(self, event: Optional[Event] = None, *, keep: Optional[int] = None,
              relax: bool = True) -> int:
        """Glue the surface where it is about to touch itself across the
        outside, an :class:`Event` of kind ``"touch"`` from :meth:`events`
        (default: the first one there): the facet nearest the touch and the
        nearest one facing it taken out and joined by a short tube of facets
        (a checked edit; Evolver's ``facet_merge`` with the merged facet
        opened where that is refused), and the bodies on either side made one (two drops; a surface
        touching itself, a clam shell closing round its fibre, keeps its
        body). The merged body has the number ``keep`` if it is one of them,
        else the lower; its ``volconst`` is theirs added and its target their
        targets added (free if either was).

        Then, with ``relax`` (default), the surface relaxes down to a stable
        state: :meth:`relax` in short runs, the mesh remeshed toward the
        median edge after each change (the neck widens), the changes that come on the way made
        (a merged neck coming down onto a solid wets it: see :meth:`contact`),
        and down along an unstable mode if it ends at a saddle; a warning says
        if it doesn't get there. Returns the merged body's number (0 without
        bodies). Linear model only.
        """
        from ._surgery import merge, relax_after
        if self.model != "linear":
            raise ValueError("merge() needs the linear model")
        if event is None:
            event = self._first_event("touch")
        body = merge(self, event, keep)
        if relax:
            relax_after(self, keep=body or None)
        return body

    def contact(self, event: Optional[Event] = None, *, keep: Optional[int] = None,
                relax: bool = True) -> None:
        """Make the contact an :class:`Event` of kind ``"solid"`` or
        ``"contact lines"`` announces (default: the first such event).

        A ``"solid"`` event: where the surface reaches the solid, its
        vertices are put on it, those facets go, and their border becomes a
        new contact line on that constraint: a wetted patch where the gas
        was between (a meniscus reaching a bead, a merged neck sinking onto a
        plane), a dry patch where the liquid was (a film thinning away). A
        ``"contact lines"`` event: two wetted patches about to meet across a
        dry gap are joined (a facet along each line glued, as :meth:`merge`
        does, ``keep`` as there), and a wet strip between dry ground about to
        split is cut as a dry patch.

        The volumes don't change in the surgery: each body's ``volconst``
        takes what the solid's content integrals change by. A new contact
        line that winds round the pole of a sphere from
        :mod:`pysurfaceevolver.constraints` changes the wetting energy the
        integrals leave out, which :attr:`energy` keeps track of. The new
        contact-line edges are stored the way their facets go round them,
        with orientation 1, as the ones from :func:`make_datafile` are.
        Then it relaxes as :meth:`merge` does. Linear model only.
        """
        from ._surgery import contact, relax_after
        if self.model != "linear":
            raise ValueError("contact() needs the linear model")
        if event is None:
            event = self._first_event("solid", "contact lines")
        contact(self, event, keep)
        if relax:
            relax_after(self, keep=keep)

    def _first_event(self, *kinds: str) -> Event:
        found = [e for e in self.events() if e.kind in kinds]
        if not found:
            raise ValueError(f"no event of kind {' or '.join(map(repr, kinds))} "
                             "(Evolver.events() finds none)")
        return found[0]

    def adapt(self, max_turn: float = _TURN, *, coarsen: bool = True) -> int:
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
        lengths = m.edge_lengths()
        bulk = float(np.percentile(lengths, 75))
        return self._remesh_pass(merge_below=_ADAPT_SHORT*bulk if coarsen else None,
                                 flat=_ADAPT_FLAT*max_turn, tilt=_ADAPT_TILT*max_turn,
                                 longest=_ADAPT_LONGEST*bulk,
                                 split_turn=max_turn)

    def _remesh_pass(self, *, merge_below: Optional[float], flat: float, tilt: float,
                     longest: float, split_turn: float) -> int:
        """Merge interior edges shorter than ``merge_below`` along which the
        surface turns by less than ``flat`` degrees (refused by the engine if a
        facet would tilt by more than ``tilt`` or flip, or an edge grow longer
        than ``longest``; not on a torus or with symmetry, where the engine
        can't check), then split edges turning more than ``split_turn`` or
        longer than ``split_above`` (not ``no_refine`` ones), equiangulate.
        Returns the edges merged plus split."""
        m = self.mesh()
        merged = 0
        if merge_below is not None and not _core.symmetry():
            valence = m.edge_valence()
            turn, judged = _edge_turn(m)
            mark = judged & (valence == 2) & (turn < flat) & (m.edge_lengths() < merge_below)
            if mark.any():
                clause = self._mark(_core.EDGE, mark)
                before = self.counts["vertices"]
                self.command(f"collapse_max_tilt := {_num(tilt, 'tilt')}; "
                             f"collapse_max_edge := {_num(longest, 'length')}")
                try:
                    self.command(f"delete edge{clause}")
                finally:
                    self.command("collapse_max_tilt := 0; collapse_max_edge := 0")
                merged = before - self.counts["vertices"]
                m = self.mesh()
        mark = _edge_turn(m)[0] > split_turn
        split = self._split_marked(mark) if mark.any() else 0
        if merged and not split and self._tidy_kinds()[0]:
            self.command("u")
        return merged + split


def _reemit(caught) -> None:
    """Warnings recorded during relax(), passed on to its caller: Evolver's own
    once per kind (WARNING n), with how often it came up (it repeats one per
    vertex and step), others as they were."""
    seen: Dict[str, list] = {}
    for w in caught:
        if issubclass(w.category, EvolverWarning):
            code = re.match(r"\s*(WARNING \d+)", str(w.message))
            key = code.group(1) if code else str(w.message)
            seen.setdefault(key, [w, 0])[1] += 1
        else:                       # where it was raised (a callback, a library)
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    for w, count in seen.values():
        more = f" (and {count - 1} more like it during relax())" if count > 1 else ""
        warnings.warn(f"{w.message}{more}", w.category, stacklevel=3)


class _Run:
    """One relax() call: its settings, the trace of its gradient steps, and
    the stages it is made of (gradient rounds, Newton with its fallback, the
    contact-line pass, the last resort)."""

    def __init__(self, ev: _Relaxing, *, tol: float, max_iter: int, tidy: int, newton: int,
                 seek: Optional[bool], undo_if: Optional[Callable[["Evolver"], bool]],
                 energy_tol: float, callback: Optional[Callable[["Evolver", int], Any]],
                 every: int):
        self.ev = ev
        self.tol, self.max_iter, self.tidy, self.newton = tol, max_iter, tidy, newton
        self.seek, self.undo_if = seek, undo_if
        self.energy_tol, self.callback, self.every = energy_tol, callback, every
        self.trace = _Trace()
        self.slowed = False         # the last rounds ended slowing down (see settle)
        self.early_newton = 0       # Newton steps a trial in settle() kept

    def approach(self) -> None:
        """A target far from the surface (a body volume or fixed quantity
        more than 25% off): first one gradient step toward it as usual; if
        that leaves degenerate facets (below 1 degree) the surface didn't
        have, it is undone and the targets are approached in stages of at
        most 10%, each with a few tidied gradient steps. Evolver's projection
        onto a target moves each vertex along its volume gradient, several
        edge lengths for a large change and unequally where the mesh is
        irregular: in one go it can fold facets (drops shrunk to half their
        volume at 90 and 120 degrees). Staging every far target instead
        lost coarse starts that the direct way converges (a bridge 2.6 times
        its volume: its contact lines drift during the stages)."""
        ev = self.ev
        # (not with optimizing parameters: what they balance may have no
        # equilibrium on the way, as bga-5's loaded pad at a smaller volume,
        # whose height then runs off)
        if ev._target_gap()[0] <= _STAGE_GAP or ev.parameters.optimizing or ev._degenerate():
            return
        targets: List[Tuple[Any, float, float]] = []
        b = ev.bodies()
        for i, fixed, now, target in zip(b.ids, b.fixed, b.volume, b.target):
            if fixed:
                targets.append((ev.body(int(i)), float(now), float(target)))
        for name, q in ev.quantities().items():
            if q.kind == "fixed":
                targets.append((ev.quantity(name), float(q.value), float(q.target)))
        # a trial step (the projection onto the targets, and one step),
        # undone either way: a surface it doesn't fold goes on as without it.
        # Its coordinates undo it (a restored dump isn't the same path: a
        # constraint with a parameter then relaxes differently)
        coords, start, mark = ev._coords(), ev._save(), self.trace.mark()
        scale = ev.eval("scale")            # (the step size the next step starts from)
        self.rounds(0, 1, 0)
        folded = ev._degenerate()
        if not ev._put_back(coords):        # (autopop or autochop changed the topology)
            ev.restore(start)
        ev.command(f"scale := {scale!r}")
        self.trace.rewind(mark)
        if not folded:
            return

        def stages(now: float, target: float) -> int:
            if now*target > 0:
                return int(np.ceil(abs(np.log(target/now))/np.log1p(_STAGE)))
            return int(np.ceil(abs(target - now)/(_STAGE*max(abs(now), abs(target), 1e-300))))

        def between(now: float, target: float, f: float) -> float:
            return now*(target/now)**f if now*target > 0 else now + f*(target - now)

        n = max(stages(now, target) for _, now, target in targets)
        try:
            for k in range(1, n):
                for view, now, target in targets:
                    view.target = between(now, target, k/n)
                self.rounds(0, _STAGE_STEPS, self.tidy)
        finally:
            for view, _, target in targets:
                view.target = target

    def next_level(self, levels: int) -> int:
        """The level after the last one with gradient steps (adaptive passes,
        pops and the last resort count as further levels)."""
        return max(self.trace.level, default=levels) + 1

    def level(self, lev: int) -> "tuple[int, float]":
        """Gradient rounds, then Newton (see finish): the Newton steps and residual."""
        self.settle(lev)
        return self.finish(lev)

    def settle(self, lev: int) -> None:
        """The level's gradient rounds, the residual watched when Newton
        follows. When the rounds have used half of ``max_iter`` and are
        slowing down without settling (a contact line creeping over a curved
        wall: the residual falling by less than 5% a round, three rounds
        running), Newton is tried there; kept if it
        converges or cuts the residual a hundredfold, else the surface goes
        back and the rounds carry on as before (a neck shrinking toward a pinch-off creeps the same way, and
        there the rounds must go on)."""
        ev, watch = self.ev, self.newton > 0
        self.slowed = False
        self.rounds(lev, self.max_iter, self.tidy, watch=watch, slow_exit=watch)
        if not self.slowed:
            return
        coords = ev._coords()
        before = ev.residual()
        kept, residual = self.newton_steps(grace=True)
        # kept if it converged, or got well on the way (the level's Newton
        # carries on from there; a soft mode can make Newton slow)
        keep = (residual < self.tol or residual < _TRIAL_KEEP*before) \
            and ev._target_gap()[0] <= _TARGET_LOST
        if keep:
            self.early_newton += kept
            return
        ev._put_back(coords)
        used = self.trace.level.count(lev)
        if used < self.max_iter:
            self.rounds(lev, self.max_iter - used, self.tidy, watch=watch)

    def newton_steps(self, grace: bool = False) -> "tuple[int, float]":
        # far from equilibrium the Hessian is indefinite, which Evolver
        # prints at every step; relax() checks stability itself at the end
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "WARNING 1825", EvolverWarning)
            return self.ev._newton(self.newton, self.seek, self.tol, self.undo_if,
                                   self.trace.output, grace)

    def newton_first(self) -> "tuple[int, float] | None":
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
        ev, trace = self.ev, self.trace
        if surface.newton_first_failed:
            surface.newton_first_failed = False
            return None
        coords = ev._coords()
        for _ in range(_NEWTON_FIRST_SETTLE):   # as the manual's 'g 5; hessian':
            trace.step(ev)                       # new high-order nodes settle first
        trace.level += [0]*_NEWTON_FIRST_SETTLE
        # stops at tol with the targets met (past it, steps in round-off cost
        # a factorization each)
        steps, residual = self.newton_steps()
        if residual < self.tol and ev._target_gap()[0] <= _TARGET_LOST:
            return steps, residual
        ev._put_back(coords)
        surface.newton_first_failed = True
        return None

    def finish(self, lev: int) -> "tuple[int, float]":
        """Newton steps, and the conjugate-gradient fallback (see _finish);
        counting the steps a trial in settle() kept."""
        steps, residual = self._finish(lev)
        early, self.early_newton = self.early_newton, 0
        return steps + early, residual

    def _finish(self, lev: int) -> "tuple[int, float]":
        """Newton steps, then one conjugate-gradient pass and Newton again if
        still unconverged, kept only if that converges. The pass averages
        vertices but doesn't equiangulate, so the coordinates taken before it
        can undo it. Returns the Newton steps kept and the residual."""
        ev, trace, tol = self.ev, self.trace, self.tol
        steps, residual = self.newton_steps() if self.newton else (0, ev.residual())
        if self.newton and ev._no_hessian:
            # no Newton for this surface: plain gradient rounds instead, settled
            # as with newton=0 (the residual watch, there for Newton, is off)
            # only on a sound mesh: on degenerate facets a long gradient run
            # slides into what the mesh allows rather than the shape (tankex
            # from 8 facets: energy 1.6 -> 0.03)
            budget = self.max_iter - trace.level.count(lev)
            if budget > 0 and not ev._degenerate():
                self.rounds(lev, budget, self.tidy, energy_tol=_GRADIENT_ONLY_TOL, sound=True)
            return 0, ev.residual()
        budget = min(self.max_iter - trace.level.count(lev), _FALLBACK_STEPS)
        if not (self.newton and budget > 0 and np.isfinite(residual) and not residual < tol):
            return steps, residual
        before = ev._coords()
        settled = trace.settled
        with ev._conj_grad(True):
            self.rounds(lev, budget, self.tidy, watch=True, average_only=True)
        more, second = self.newton_steps() if np.isfinite(_core.total_energy()) \
            else (0, ev.residual())
        if second < tol or not ev._put_back(before):   # (not: the topology changed)
            return steps + more, second
        trace.settled = settled             # the pass is undone: as the rounds before it
        return steps, residual

    def contact_pass(self, lev: int, residual: float) -> "tuple[int, float]":
        """After convergence (linear model): split the edges leaving a contact
        line (an edge on one facet only) that are longer than 1.5 times the
        contact-line edges at their end, and relax again, up to twice; kept
        only if that converges, else the converged surface comes back. A
        contact circle that shrinks (a drop at a high contact angle) otherwise
        ends up smaller than the facets next to it. Returns the Newton steps
        and the residual."""
        ev = self.ev
        total = 0
        for _ in range(_CONTACT_SPLITS):    # (until no spoke is long, each kept if it converges)
            mark = _long_spokes(ev.mesh(), _SPOKE_RATIO)
            if not mark.any():
                break
            before = ev._save()
            ev._split_marked(mark)
            steps, converged = 0, False
            for _ in range(_CONTACT_PASSES):
                more, second = self.level(lev)
                steps += more
                if second < self.tol:
                    converged = True
                    break
            if not converged:
                ev.restore(before)
                break
            total, residual = total + steps, second
        return total, residual

    def lower_angles(self, start: Any, levels: int, low: float) -> "tuple[int, float] | None":
        """relax()'s fallback for a crude start at a small contact angle that
        sank into a wall (gradient steps slide the vertices next to the
        contact line through a solid the liquid meets almost tangentially,
        and the wall's integrals reward it): the levels again from the call's
        start with the small angles raised to ANGLE_FLOOR_TOP (the datafile
        parameter ANGLE_FLOOR), then lowered by _ANGLE_STEP a stage, relaxed
        at each, down to their own (``low``, the least). Kept if it
        converges; else the surface and the trace go back (None). Returns
        the Newton steps and the residual."""
        ev, trace = self.ev, self.trace
        failed, mark = ev._save(), trace.mark()
        lev = self.next_level(levels)
        floors = [ANGLE_FLOOR_TOP]
        while floors[-1]*_ANGLE_STEP > max(low, 1.0):
            floors.append(floors[-1]*_ANGLE_STEP)
        steps, residual = 0, float("inf")
        try:
            ev.restore(start)
            ev.parameters[ANGLE_FLOOR] = floors[0]
            for k in range(levels + 1):
                if k:
                    ev.refine()
                self.approach()
                self.settle(lev + k)
            steps, residual = self.finish(lev + levels)
            for k, floor in enumerate(floors[1:] + [0.0], lev + levels + 1):
                if not residual < self.tol:
                    break
                ev.parameters[ANGLE_FLOOR] = floor
                more, residual = self.level(k)
                steps += more
        except EvolverError:
            residual = float("inf")
        if residual < self.tol and ev.parameters[ANGLE_FLOOR] == 0:
            return steps, residual
        ev.restore(failed)
        trace.rewind(mark)
        return None

    def finer_start(self, start: Any, levels: int) -> "tuple[int, float] | None":
        """relax(levels=n)'s fallback when the last level ends unconverged:
        the levels again from the call's start refined once first (n - 1
        refinements after). A coarse start can hold equilibria the shape
        doesn't have, and which one its gradient rounds find is chance (a
        bridge at 20 degrees from a tube 12 facets around: 6 in 13 starts
        failed, its volume moved by a few units in the last place; refined
        first, all converged, to the same energy). Kept if it converges; else
        the surface and the trace go back (None). Returns the Newton steps
        and the residual."""
        ev, trace = self.ev, self.trace
        failed, mark = ev._save(), trace.mark()
        first = self.next_level(levels)
        residual = float("inf")
        try:
            ev.restore(start)
            for k in range(levels):
                ev.refine()
                self.approach()
                self.settle(first + k)
            steps, residual = self.finish(first + levels - 1)
        except EvolverError:
            pass
        if residual < self.tol:
            return steps, residual
        ev.restore(failed)
        trace.rewind(mark)
        return None

    def collapse(self, levels: int) -> "tuple[int, float, int, int] | None":
        """relax()'s last resort for an unconverged surface: gradient steps
        without tidying (which can hold a shrinking neck open as a tube of
        slivers), up to a few hundred, then delete what collapsed, pop and
        relax again. Kept if that converges, or lowers the energy without
        leaving degenerate facets; else the surface and the trace go back
        (None). Returns the Newton steps, residual, pops and deleted edges.

        After k failed attempts in a row the next 2**k - 1 calls skip it (a
        continuation that stays unconverged pays for a few attempts, not one
        per call); a kept attempt starts the count over."""
        ev, trace, tol = self.ev, self.trace, self.tol
        memory = surface
        if memory.collapse_skip:
            memory.collapse_skip -= 1
            return None
        snapshot = ev._save()
        energy = _core.total_energy()
        residual0 = ev.residual()
        mark = trace.mark()
        lev = self.next_level(levels)
        steps = popped = gone = 0
        residual = float("inf")
        try:
            self.rounds(lev, _COLLAPSE_STEPS, 0, energy_tol=_GRADIENT_ONLY_TOL)
            gone = ev._delete_collapsed()
            if gone:
                for _ in range(_TOPOLOGY_PASSES):
                    popped += ev.pop()
                    more, residual = self.level(lev)
                    steps += more
                    extra = ev._delete_collapsed() if not residual < tol else 0
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
            sound = gone and not ev._degenerate()
        except EvolverError:        # a facet left without its three edges
            sound, residual = False, float("inf")
        if gone and (residual < tol or (_core.total_energy() < energy and sound
                                        and residual <= residual0)):
            memory.collapse_failures = 0
            return steps, residual, popped, gone
        memory.collapse_failures += 1
        memory.collapse_skip = 2**memory.collapse_failures - 1
        ev.restore(snapshot)
        trace.rewind(mark)
        return None

    def rounds(self, lev: int, max_iter: int, tidy: int, *, watch: bool = False,
               energy_tol: Optional[float] = None, average_only: bool = False,
               sound: bool = False, slow_exit: bool = False) -> None:
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
        ev, trace = self.ev, self.trace
        energy_tol = self.energy_tol if energy_tol is None else energy_tol
        callback, every = self.callback, self.every
        previous = _core.total_energy()
        quiet = 0
        best = np.inf
        first = len(trace.energy)
        watch = watch and tidy > 0
        start = ev._coords()
        gap, _ = ev._target_gap()
        slow, last_res = 0, 0.0
        for i in range(1, max_iter + 1):
            trace.step(ev)
            if callback is not None and i % every == 0:
                callback(cast("Evolver", ev), i)
            if not np.isfinite(trace.energy[-1]):
                bad = trace.energy[-1]
                where = ("the surface is back where that round began" if ev._put_back(start)
                         else "its topology changed during the round, so it stays as it is")
                raise EvolverError(f"relax(): the energy became {bad} during gradient steps; "
                                   + where)
            if tidy and i == 1 and ev._tidy_kinds()[0] and not average_only:
                # a coarse start triangulated badly is equiangulated before
                # it settles into what its mesh allows (no averaging: that
                # moves vertices, and every call would pay it)
                trace.output.append(ev.command("u"))
                trace.refresh()
                start = ev._coords()
            if tidy and i % tidy:
                continue                  # mid-round: no convergence check
            # on a degenerate mesh Evolver's volume projection can fail without
            # an error: a round that pushes a volume off its target is undone
            # (a new target, still being approached, only counts if it grows)
            now, _ = ev._target_gap()
            if now > _TARGET_LOST and now > _ROUND_BLOWUP*gap:
                if tidy and ev._put_back(start):
                    trace.refresh()
                break
            gap = min(gap, now)
            if watch:
                res = ev.residual()
                if res > _ROUND_BLOWUP*best:    # the round went wrong: undo it, hand over
                    ev._put_back(start)          # (unless autopop etc. changed the topology)
                    trace.refresh()
                    break
                best = min(best, res)
                if slow_exit:                   # falling, but slowly (see settle)
                    slow = slow + 1 if last_res and _SLOW_ROUND*last_res < res < last_res \
                        else 0
                    last_res = res
            if tidy:
                equi, avg = ev._tidy_kinds()
                cmd = "; ".join(c for c, on in (("u", equi and not average_only), ("V", avg)) if on)
                if cmd:
                    trace.output.append(ev.command(cmd))
                trace.refresh()
                start = ev._coords()
                if sound and ev._degenerate():
                    break
            change = abs(trace.energy[-1] - previous) / max(1.0, abs(trace.energy[-1]))
            previous = trace.energy[-1]
            quiet = quiet + 1 if change < energy_tol else 0
            if quiet >= _WINDOW:
                break
            # slowing down without settling, with half the budget gone (a
            # surface that settles normally never gets here)
            if slow >= _WINDOW and i >= max_iter//2:
                self.slowed = True
                break
        # vertex averaging moves volumes a little (on a degenerate mesh a lot),
        # and the next round's steps would project them back: at the end, one
        # step does
        if tidy and len(trace.energy) > first and ev._target_gap()[0] > _TARGET_LOST:
            trace.step(ev)
        trace.settled = quiet >= _WINDOW
        steps = len(trace.energy) - first
        trace.level += [lev]*steps
        if callback is not None and steps % every != 0:
            callback(cast("Evolver", ev), steps)
