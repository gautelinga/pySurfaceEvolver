"""Analysis of the surface: residual, health, stability, mesh quality."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

import numpy as np

from . import _core
from ._engine import _Engine
from ._errors import EvolverError, _num
from ._events import Event, find as _find_events
from ._geometry import _chains, _crossed, _grow, _self_gap
from ._mesh import Mesh, MeshQuality
from ._results import EigenCounts, Health, Modes, Stability


# a vertex counts as crossed when beyond a constraint by this many median edge lengths:
# less is the chord of a curved wall next to a contact line
_CROSS_TOL = 0.01

# a body volume or quantity this far off its target (relative) is missed: a
# new target for Newton to meet, or a gradient round gone wrong (vertex
# averaging alone leaves volumes about 2e-5 off)
_TARGET_LOST = 1e-4
_DEGENERATE = 1.0              # a facet with an angle below this (degrees) is degenerate
_NEAR = 0.1                    # health(): a gap below this many median edges is an issue
EVENT_FACTOR = 0.5             # events(): a gap below this many local edge lengths

# Evolver's methods whose in-plane forces are physical (membranes)
_ELASTIC_METHODS = frozenset({
    "linear_elastic", "general_linear_elastic", "linear_elastic_B", "relaxed_elastic",
    "relaxed_elastic1", "relaxed_elastic2", "relaxed_elastic_A", "relaxed_elastic1_A",
    "relaxed_elastic2_A", "dirichlet_elastic", "SVK_elastic", "neo_hookean",
    "stress_integral"})

# residual() of the surface as it last was, kept for its version
# (_core.surface_version(): any edit, command or move changes it): a 16-bead
# pack's descent asked 57% of its 12.6k residuals of a state it had just
# measured (a run's check before relax(), relax()'s own at its start), 73 s
# of a 1500 s replay. (Not _target_gap(): its first reading at a version can
# come before the quantities' recalculation it makes, and kept, it would
# stand in for the later, recalculated ones)
_measured: Dict[str, tuple] = {}

class _Analysis(_Engine):
    """How the surface stands: equilibrium, stability, mesh quality, health."""

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

    def _degenerate(self) -> bool:
        """Whether a facet is degenerate (an angle below 1 degree)."""
        return self.mesh_quality(5.0).angle_min < _DEGENERATE

    def _elastic(self) -> bool:
        """Whether an energy quantity uses one of Evolver's elastic methods
        (in-plane forces are physical, and the facets remember a reference
        shape that retriangulating would corrupt)."""
        _, result = self._call(_core.energy_methods)
        return not _ELASTIC_METHODS.isdisjoint(result.data)

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
        directions count, not only the normal. Optimizing parameters count
        too: the force on each (minus the derivative of the energy, with
        the volume and quantity multipliers applied), as if spread over the
        vertices. (For a string: the root
        mean square times N, the imbalance times the length; in general the
        imbalance times the surface measure to the power 1/dimension.)
        """
        version = _core.surface_version()
        kept = _measured.get("residual")
        if kept is not None and kept[0] == version:
            return kept[1]
        _, result = self._call(_core.residual, self._elastic())
        r = np.asarray(result.data, dtype=float)
        area = _core.total_area()           # a string's length, a simplex's volume
        if len(r) == 0:
            out = 0.0
        elif area <= 0:
            out = float(np.sqrt((r**2).sum()))
        else:
            rep = _core.representation()
            dim = 1 if rep == 1 else 2 if rep == 2 else int(self.eval("surface_dimension"))
            n = max(_core.count(_core.VERTEX), 1)      # (r: the vertices, then the parameters)
            out = float(np.sqrt((r**2).sum()*n)/area**((dim - 1)/dim))
        if _core.surface_version() == version:     # (nothing changed meanwhile)
            _measured["residual"] = (version, out)
        return out

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
        moving = used & ~m.fixed
        wall_gap, wall = np.inf, None
        nums, attrs, dist, names = self._constraint_gaps()
        # a contact line on one wall may meet another (a corner, or a second
        # constraint drawn on the same wall): gaps count only for vertices free
        # of every constraint
        free = moving & ~np.isnan(dist).any(axis=1) if dist.size else moving
        label = {int(c): f"constraint {nm or c}" for c, nm in zip(nums, names)}
        # an equality constraint only fixed vertices are on just places them
        # (a pad's rim, a solid drawn for display): not a wall to the surface;
        # one no vertex is on still is (the surface may come close)
        def placement(c: int) -> bool:
            on = np.isnan(dist[:, c]) & used
            return bool(not attrs[c] & 3 and on.any() and not (on & moving).any())
        active = np.array([not placement(c) for c in range(len(nums))], dtype=bool)
        for c in np.flatnonzero(active).tolist():
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
        crossed = _crossed(nums[active], attrs[active], dist[:, active], moving,
                           _CROSS_TOL*h)             # (fixed vertices never moved)
        self_gap = _self_gap(m.vertices, e, moving, h, periods=m.periods)
        issues = []
        if not residual < tol and not _core.crystalline():
            issues.append(f"not in equilibrium (residual {residual:.2g})")
        if stable is False:
            issues.append(f"unstable: {modes} direction(s) lower the energy")
        if quality.angle_min < _DEGENERATE:
            issues.append(f"degenerate facets (smallest angle {quality.angle_min:.2g} degrees, "
                          f"{quality.skinny} below 5)")
        for c, k in crossed.items():
            issues.append(f"{k} {'vertex' if k == 1 else 'vertices'} on the far side of {label[c]}")
        if wall_gap < _NEAR:
            issues.append(f"the surface nearly touches {label.get(wall or 0, 'a constraint')} "
                          f"({wall_gap:.2g} edge lengths away)")
        if self_gap < _NEAR:
            issues.append(f"the surface nearly touches itself ({self_gap:.2g} edge lengths)")
        target_gap, _ = self._target_gap()
        if target_gap > _TARGET_LOST:
            issues.append(f"a fixed volume or quantity is off its target ({target_gap:.2g} "
                          "relative)")
        junctions = self._bad_junctions(m)
        if junctions:
            issues.append(f"{junctions} {'junction breaks' if junctions == 1 else 'junctions break'} "
                          "Plateau's laws (Evolver's 'o' pops them)")
        return Health(float(residual), stable, modes, float(quality.angle_min),
                      int(quality.skinny), float(wall_gap), wall, crossed, float(self_gap),
                      junctions, float(target_gap), issues)

    def _bad_junctions(self, m: Optional[Mesh] = None) -> int:
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
        m = self.mesh() if m is None else m
        if len(m.edges) == 0:
            return 0
        # (in a torus two edges can join the same two vertices: Evolver counts)
        valence = self.values("edge", "valence") if _core.symmetry() else m.edge_valence()
        if not (valence >= 3).any():  # a film without junctions: every vertex is a disc
            return 0
        bad = 0
        if (valence >= 4).any():
            # along a wire or wall (both ends attached) more films may meet
            loose = (free[m.edges[:, 0]] | free[m.edges[:, 1]]) \
                & (self.values("edge", "fixed") == 0)
            bad = int(((valence >= 4) & loose).sum())
        try:          # the tangent cones 'o' pops (cubes, Kraynik's, ...)
            _, cones = self._call(_core.poppable_vertices)
            return bad + int(cones.data)
        except EvolverError:        # a cone it can't analyze: more than four triple edges
            triple = valence == 3
            count = np.bincount(m.edges[triple].ravel(), minlength=len(free))
            return bad + int(((count > 4) & free).sum())

    def events(self, factor: float = EVENT_FACTOR) -> List[Event]:
        """The changes of topology that are coming, closest first (relative to
        the edges there), as :class:`Event` records: a neck about to pinch
        off (the surface within ``factor`` local edge lengths of itself across
        a body), the surface about to touch itself across the outside, a
        stretch of surface that close to a constraint it isn't on (away from
        its contact lines there; or gone through it), and contact lines on one
        constraint that close to each other while far apart along them (or on
        different lines). Empty when nothing is that close.

        :func:`recipes.arclength` and :func:`recipes.continuation` put the
        events on each :class:`Step` and stop at the first point where one
        is coming (there at the last point too, and closer now; see their
        ``events=``); :meth:`relax` puts them on its result.

        The gaps are measured from vertices to facets and to the constraints'
        level sets (curved elements by their corners); a surface that has
        crossed itself between two looks isn't found. Not in a torus or with
        a symmetry group."""
        return _find_events(self, float(factor))

    def stability(self, nearest: int = 6, *, eigenvalues: bool = False) -> Stability:
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

        ``lowest`` and ``threshold`` are eigenvalues with Evolver's
        ``linear_metric`` (as its eigenvalue tutorial recommends): those of
        the continuous surface, which don't shrink as the mesh is refined
        (the counts are the same either way). On a stable surface ``lowest``
        is empty unless ``eigenvalues=True`` (then ``ritz`` runs anyway: the
        lowest eigenvalue falling toward zero over a continuation shows a
        fold or instability coming).
        """
        first = self.eigen_counts(0.0).negative
        if first == 0 and not eigenvalues:
            return Stability(True, 0, np.array([]), 0.0)
        with self._toggle("linear_metric", True):
            text = self.command(f"ritz(0, {max(int(nearest), first + 4)})")
            values = np.array(sorted(float(x) for x in re.findall(
                r"^\s*\d+\.\s+([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$", text, re.MULTILINE)))
            if first == 0:
                return Stability(True, 0, values, 0.0)
            positive = np.abs(values[values > 0])
            scale = float(np.median(positive)) if len(positive) \
                else float(np.abs(values).max(initial=1.0))
            threshold = -0.01*scale
            negative = self.eigen_counts(threshold).negative
        return Stability(negative == 0, negative, values, threshold)

    def modes(self, n: int = 1) -> Modes:
        """The ``n`` lowest Hessian eigenvalues and their modes: the motions
        of the vertices, with the constraints (volumes, quantities, walls)
        held.

        ``values`` are ascending, with Evolver's ``linear_metric`` as in
        :meth:`stability` (the continuous surface's eigenvalues: a negative
        one is a motion that lowers the energy). ``vectors[k]`` is mode
        ``k`` as a displacement per vertex (0 for fixed vertices), scaled to
        a root mean square of 1 over the vertices and signed so that its
        largest component is positive (among components within 1% of the
        largest, the first in vertex order), so a simple eigenvalue's mode
        comes out the same each time; a repeated eigenvalue's (a symmetry's
        rotations or translations) is any combination of its eigenspace.
        ``ev.vertices + 0.01*m.vectors[0]`` steps along the first. Unlike
        :meth:`saddle`, the surface is left as it is. Fewer than ``n`` come
        back when the surface has fewer degrees of freedom. A mode that
        moves optimizing parameters moves them too; ``vectors`` has only
        the vertices. Uses Evolver's ``ritz``, slow on large surfaces.
        """
        if int(n) < 1:
            raise ValueError("modes() needs n >= 1")
        with self._toggle("linear_metric", True):
            _, result = self._call(_core.modes, int(n))
        values, vectors = (np.asarray(a, dtype=float) for a in result.data)
        order = np.argsort(values, kind="stable")
        values, vectors = values[order], vectors[order]
        for v in vectors:
            rms = np.sqrt((v*v).sum(axis=1).mean())
            if rms > 0:
                v /= rms
            flat = v.ravel()
            big = np.flatnonzero(np.abs(flat) >= 0.99*np.abs(flat).max())
            if len(big) and flat[big[0]] < 0:
                v *= -1
        return Modes(values, vectors)

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
        or quotient space other than a periodic slab (whose mesh measures
        itself at the minimum image), from Evolver's own edge lengths (vertex
        coordinates are wrapped, so facets across the cell's sides would look
        torn)."""
        if not _core.symmetry() or self.periods is not None:
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

    def energies(self) -> Dict[str, float]:
        """What the energy is made of, term by term: ``{name: energy}`` for
        every energy method instance once Evolver has converted everything to
        named quantities (``default_area_inst`` for the facets' tension,
        ``constraint_1_energy_inst`` for a wall's line integral,
        ``gravity_inst``, ...).
        Brakke's way of checking a datafile (``convert_to_quantities;
        show_all_quantities; Q``): a term with the wrong sign or size points
        at the mistake. The terms add up to :attr:`total_energy`. Works on
        a copy: the surface is left as it was."""
        snapshot = self.save()
        try:
            self.command("convert_to_quantities")
            _, result = self._call(_core.energy_terms)
        finally:
            self.restore(snapshot)
        return {name: energy for name, _, energy in result.data}

    def contact_lines(self, constraint: Union[int, str]) -> List[np.ndarray]:
        """Where the surface meets a constraint (number or name): the edges on
        it that bound the surface (on one facet only; for strings, the
        vertices' chain on it), joined into polylines of vertex positions,
        longest first; a closed line repeats its first point. Curved models:
        through the corner vertices. For a drop's contact radius, say,
        ``np.linalg.norm(line[:-1, :2] - line[:-1, :2].mean(axis=0), axis=1).mean()``.
        In a periodic slab each line is unwrapped along itself from its first
        vertex (a line round the cell ends a period from where it starts)."""
        m = self.mesh()
        on = self.on_constraint(constraint)
        e = m.edges
        mask = on[e[:, 0]] & on[e[:, 1]]
        if m.facets is not None and len(m.facets):
            mask &= m.edge_valence() == 1
        if m.periods is not None:
            from ._periodic import unwrap_loop
            return [unwrap_loop(path, m.vertices, m.periods) for path in _chains(e[mask])]
        return [m.vertices[path] for path in _chains(e[mask])]

    def contact_angles(self, constraint: Union[int, str], body: Optional[int] = None) -> np.ndarray:
        """The contact angle (degrees, through the liquid) at each vertex where
        the surface meets a constraint (number or name): the angle between
        the surface's normal out of ``body`` (area-weighted over its flat
        facets) and the constraint's normal, the gradient of its formula.
        Vertices also on another constraint (corners) are left out.

        ``body``: the liquid; needed when the surface has several bodies
        (without bodies, facet normals are taken as they are). The gradient
        is taken to point out of the solid, as with the walls of
        :mod:`pysurfaceevolver.constraints` (a plane's normal into the
        liquid; spheres and cylinders solid); otherwise the angles come out
        as 180 degrees minus the contact angle.
        """
        m = self.mesh()
        if m.facets is None:
            raise ValueError("contact_angles() needs facets (the soapfilm model)")
        nums, _, dist, names = self._constraint_gaps()
        if isinstance(constraint, str):
            if constraint not in names:
                raise KeyError(f"no constraint named {constraint!r}")
            constraint = int(nums[names.index(constraint)])
        if int(constraint) not in [int(c) for c in nums]:
            raise KeyError(f"no constraint {constraint}")
        column = [int(c) for c in nums].index(int(constraint))
        sign = np.ones(len(m.facets))
        if m.facet_bodies is not None and m.facet_bodies.any():
            if body is None:
                found = np.unique(m.facet_bodies[m.facet_bodies > 0])
                if len(found) != 1:
                    raise ValueError(f"the surface has bodies {list(found)}: give body=")
                body = int(found[0])
            sign = np.where(m.facet_bodies[:, 0] == body, 1.0,
                            np.where(m.facet_bodies[:, 1] == body, -1.0, 0.0))
        xyz = m.vertices[:, :3]
        a, b, c = (xyz[m.facets[:, i]] for i in range(3))
        normals = np.cross(m.delta(a, b), m.delta(a, c))*sign[:, None]
        vn = np.zeros_like(xyz)
        for k in range(3):
            np.add.at(vn, m.facets[:, k], normals)
        on = np.isnan(dist)
        sel = on[:, column] & (on.sum(axis=1) == 1) & (np.linalg.norm(vn, axis=1) > 0)
        _, solid = self._call(_core.constraint_normals_at, int(constraint),
                              np.flatnonzero(sel).astype(np.int64))
        nl = vn[sel]/np.linalg.norm(vn[sel], axis=1)[:, None]
        return np.degrees(np.arccos(np.clip((nl*solid.data).sum(axis=1), -1.0, 1.0)))
