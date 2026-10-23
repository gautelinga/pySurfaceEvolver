"""Constraint builders: planes, mirrors, spheres and cylinders, with contact angles.

A solid wall wetted by the liquid contributes ``-tension*cos(contact_angle)``
times its wetted area to the energy, and the wetted part of the wall closes the
liquid's volume. Evolver sees neither directly: both are line integrals along
the contact line (a constraint's ``energy`` and ``content`` integrands). These
builders write those integrands, so a datafile needs only::

    from pysurfaceevolver import constraints as C
    make_datafile(..., constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60),
                                    2: C.sphere((0, 0, 0), 0.48, contact_angle=40,
                                                span=np.pi/2, wet_poles=("north",))})

Conventions (the usual ones for bodies):

* facets are oriented with their normals pointing out of the liquid;
* a plane's ``normal`` points into the liquid (away from the solid); spheres and
  cylinders are solid, with the liquid outside;
* ``contact_angle`` is in degrees, measured through the liquid; None (or a
  mirror) means no wetting energy. It may be a datafile parameter's name or an
  expression (``"theta"``), so ``ev.parameters["theta"] = 50`` changes it;
  then :attr:`Constraint.energy_constant` is NaN and the left-out energy is
  ``-tension*cos(theta)*area_constant``. An angle given as a parameter, or a
  number below 20 degrees, enters as ``maximum(angle, pyse_angle_floor)``
  with a datafile parameter ``pyse_angle_floor`` (0) that
  :meth:`~pysurfaceevolver.Evolver.relax` raises for a while when a crude start
  at a small angle fails (see there), and :attr:`Constraint.energy_constant`
  is NaN for it too;
* volumes are Evolver's default (``z dx dy`` over facets), not
  ``symmetric_content``.

The line integrals only see the contact line. Keep mirror planes vertical or in
z = 0, where they need no volume closure; a tilted mirror face bounded partly by
other walls can't be closed by line integrals. Faces of the liquid's region that
no contact line touches (a floor under the whole film, say) aren't closed by any
of them: add their part of Evolver's volume (``z dx dy``) to the body's
``volconst`` yourself (it is zero for a face in the plane z = 0 or parallel to
the z axis). Where the wetted region's boundary
also runs along a mirror plane, the integrands are chosen so that those parts
contribute nothing (for spheres and cylinders: mirror planes through the axis,
and for spheres the plane through the centre normal to the axis). A sphere's
wetted region may contain a pole (where the azimuth is undefined); say so with
``wet_poles``. :func:`pysurfaceevolver.make_datafile` then adds
:attr:`Constraint.volconst` to the volume of the body touching the wall (or of
the only body), and :attr:`Constraint.energy_constant` to
:attr:`pysurfaceevolver.Evolver.energy` (Evolver can't add a constant energy;
pressures don't need it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Constraint", "Boundary", "plane", "mirror", "sphere", "cylinder", "circle", "curve"]


def _n(x: float) -> str:
    """A number for a datafile formula (parenthesized when negative)."""
    x = float(x)
    if x == 0:
        return "0"
    return f"({x!r})" if x < 0 else repr(x)


def _lin(coeffs: "Sequence[float] | np.ndarray", const: float = 0.0) -> str:
    """const + a*x + b*y + c*z, without zero terms."""
    terms = [f"{_n(c)}*{v}" for c, v in zip(coeffs, "xyz") if c != 0]
    if const != 0 or not terms:
        terms.insert(0, _n(const))
    return "(" + " + ".join(terms) + ")"


ANGLE_FLOOR = "pyse_angle_floor"   # the datafile parameter relax() raises small angles to
ANGLE_FLOOR_TOP = 20.0             # ... from at most this many degrees


@dataclass(frozen=True)
class Constraint:
    """A constraint for :func:`make_datafile`: its datafile text, and the
    constants its line integrals leave out (see the module notes)."""

    formula: str
    energy: Optional[Sequence[str]] = None
    content: Optional[Sequence[str]] = None
    volconst: float = 0.0
    energy_constant: float = 0.0   # NaN when the contact angle is a parameter
    nonnegative: bool = False
    area_constant: float = 0.0     # wetted area the integrals leave out
    # the left-out energy as a datafile expression, when the contact angle
    # is a parameter (make_datafile adds it to Evolver.energy)
    energy_constant_text: Optional[str] = field(default=None, compare=False, repr=False)
    # datafile parameters the contact angle refers to (make_datafile checks
    # that they are declared)
    parameters: Tuple[str, ...] = field(default=(), compare=False, repr=False)
    # the wall's geometry, for drawing: ("plane", normal, offset),
    # ("sphere", center, radius) or ("cylinder", point, direction, radius)
    shape: Optional[Tuple[Any, ...]] = field(default=None, compare=False, repr=False)
    # the contact angle as given (degrees or a datafile expression)
    contact_angle: Any = field(default=None, compare=False, repr=False)
    # a sphere's: the wetting energy of one pole's share of the integrals'
    # constant (a contact line that comes to wind round a pole adds or takes
    # it: Evolver.contact), as a datafile expression
    pole_energy: Optional[str] = field(default=None, compare=False, repr=False)
    # a sphere's: the direction from its centre to its north pole, and
    # whether its integrals are regular at the south pole (sphere(axis=))
    pole_axis: Optional[Tuple[float, float, float]] = field(default=None, compare=False,
                                                            repr=False)
    south_regular: bool = field(default=False, compare=False, repr=False)
    # a sphere(axis=)'s integrands' data, for evaluating them in C (centre,
    # radius, frame e1 e2 e3, the energy's coefficient or NaN where it isn't
    # a number): Evolver's interpreter took 22% of a pack's gradient steps
    native: Optional[Tuple[Any, ...]] = field(default=None, compare=False, repr=False)
    # a sphere(gauge="flux")'s: (centre, radius, contact angle, tension), for
    # make_datafile to put together the integrals all walls share (_flux)
    flux: Optional[Tuple[Any, ...]] = field(default=None, compare=False, repr=False)
    # a plane()'s frame (origin o, in-plane u and v, normal n, tension), for
    # evaluating its own integrands in C beside a flux gauge's (the floor's
    # content, interpreted with sixteen beads' terms, was a third of a pack's
    # gradient step)
    frame: Optional[Tuple[Any, ...]] = field(default=None, compare=False, repr=False)

    @property
    def floored(self) -> bool:
        """Whether its contact angle takes ANGLE_FLOOR as its least (see
        :meth:`~pysurfaceevolver.Evolver.relax`). (A sphere(gauge="flux")'s
        wetting energy is in the facet integral make_datafile adds.)"""
        if self.flux is not None:
            k = _Coefficient(self.flux[2], self.flux[3])
            return k.expr is not None and ANGLE_FLOOR in k.expr
        return ANGLE_FLOOR in self.text()

    def text(self) -> str:
        """The constraint's body in a datafile (after ``constraint n``)."""
        out = "nonnegative\n" if self.nonnegative else ""
        out += f"formula: {self.formula}\n"
        if self.energy is not None:
            out += "energy:\n" + "".join(f"e{i + 1}: {c}\n" for i, c in enumerate(self.energy))
        if self.content is not None:
            out += "content:\n" + "".join(f"c{i + 1}: {c}\n" for i, c in enumerate(self.content))
        return out

    def __str__(self) -> str:
        return self.text()

    def to_pyvista(self, bounds=None, resolution: int = 60) -> "Any":
        """The wall as a PyVista surface, for drawing next to the liquid (see
        ``walls`` in :meth:`Evolver.plot`): a sphere, a cylinder cut to
        ``bounds`` (xmin, xmax, ymin, ymax, zmin, zmax) or a plane patch
        covering them. None for a constraint without known geometry."""
        if self.shape is None:
            return None
        import pyvista as pv
        lo, hi = (np.array([-1.0]*3), np.array([1.0]*3)) if bounds is None else \
            (np.asarray(bounds, float)[0::2], np.asarray(bounds, float)[1::2])
        middle, size = (lo + hi)/2, float(np.linalg.norm(hi - lo))
        kind = self.shape[0]
        if kind == "sphere":
            _, c, r = self.shape
            if any(isinstance(x, str) for x in c):
                raise ValueError("this sphere's centre follows parameters: draw it with "
                                 "Evolver.plot(walls=True), which puts it where it is")
            return pv.Sphere(radius=r, center=c, theta_resolution=resolution,
                             phi_resolution=resolution)
        if kind == "cylinder":
            _, p, d, r = self.shape
            d = np.asarray(d, float)
            center = np.asarray(p, float) + np.dot(middle - p, d)*d
            return pv.Cylinder(center=center, direction=d, radius=r, height=size,
                               resolution=resolution, capping=False)
        _, n, offset = self.shape
        n = np.asarray(n, float)
        center = middle - (np.dot(middle, n) - offset)*n
        return pv.Plane(center=center, direction=n, i_size=size, j_size=size)


def _unit(v, what: str) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError(f"{what} must be a nonzero 3-vector")
    return v/np.linalg.norm(v)


def _frame(n: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """u, v with (u, v, n) right-handed and orthonormal."""
    helper = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


_FUNCTIONS = {"pi", "e", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sqrt", "exp",
              "log", "abs", "sinh", "cosh", "tanh", "pow", "floor", "ceil", "minimum", "maximum"}


def _identifiers(expr: str) -> Tuple[str, ...]:
    """The names a datafile expression refers to (parameters), without
    Evolver's functions and constants."""
    import re
    # identifiers, not exponents (the "e3" of 1e3)
    return tuple(sorted(set(re.findall(r"(?<![\w.])[A-Za-z_]\w*", expr)) - _FUNCTIONS))


class _Coefficient:
    """tension*cos(contact_angle): a number, or a datafile expression when the
    angle is given as a parameter name or expression (e.g. ``"theta"``) or is
    below ANGLE_FLOOR_TOP; those take ANGLE_FLOOR as their least angle."""

    def __init__(self, contact_angle, tension: float):
        self.value: Optional[float] = None
        self.expr: Optional[str] = None
        self.names: Tuple[str, ...] = ()
        if isinstance(contact_angle, str):
            self.names = _identifiers(contact_angle)
        if contact_angle is None:
            self.value = 0.0
        elif isinstance(contact_angle, str):
            self.expr = (f"({_n(tension)}*cos(maximum(({contact_angle}), {ANGLE_FLOOR})"
                         "*pi/180))")
        elif 0 < float(contact_angle) < ANGLE_FLOOR_TOP:
            self.expr = (f"({_n(tension)}*cos(maximum({_n(float(contact_angle))}, "
                         f"{ANGLE_FLOOR})*pi/180))")
        else:
            self.value = float(tension)*float(np.cos(np.radians(contact_angle)))
            if abs(self.value) < 1e-12*abs(float(tension)):
                self.value = 0.0            # 90 degrees (cos is 6e-17 in floats)

    def __bool__(self) -> bool:
        return self.expr is not None or self.value != 0

    def times(self, x: float) -> str:
        """k*x as a datafile factor."""
        if self.expr is not None:
            return f"{self.expr}*{_n(x)}"
        assert self.value is not None
        return _n(self.value*x)

    def energy(self, area: float) -> float:
        if self.expr is not None:
            return float("nan")
        assert self.value is not None
        return -self.value*area


def plane(normal, offset: float = 0.0, contact_angle: Optional[float] = None, *,
          point=None, ref=None, origin=None, tension: float = 1.0) -> Constraint:
    """The plane ``normal . x = offset`` (or through ``point``), the liquid on
    the side ``normal`` points to. With ``contact_angle``, the wetted part
    adds ``-tension*cos(contact_angle)`` per area; the volume is closed over
    the wetted part.

    In-plane coordinates: s along ``ref`` (default: x, or y for planes normal
    to x), t across it, both from ``origin`` (default: the point of the plane
    nearest the coordinate origin). The integrands are ``t ds`` forms: they
    vanish along lines s = const and along t = 0. So where the wetted face's
    boundary runs along mirror planes, choose ``ref`` and ``origin`` so those
    lines are s = const or t = 0 (a ceiling z = 0.5 cut by mirrors x = 0,
    x = c and y = 0: the defaults).
    """
    n = _unit(normal, "normal")
    scale = np.linalg.norm(np.asarray(normal, dtype=float))
    d = float(np.dot(n, point)) if point is not None else float(offset)/scale
    if ref is None:
        ref = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    r = np.asarray(ref, dtype=float)
    r = r - np.dot(r, n)*n
    u = _unit(r, "ref (a direction in the plane)")
    v = np.cross(n, u)
    o = d*n if origin is None else np.asarray(origin, dtype=float) - (np.dot(n, origin) - d)*n
    k = _Coefficient(contact_angle, tension)
    # x = o + s u + t v; along the contact line (oriented by the facets) the
    # integral of t ds is minus the wetted area
    t = _lin(v, -np.dot(v, o))
    energy = None
    if k:
        energy = [f"{k.times(ui)}*{t}" if ui != 0 else "0" for ui in u]
    # Evolver's volume misses the wetted face's z dx dy: -n_z * integral of z dA,
    # z = o_z + s u_z + t v_z on the plane; as P ds with -dP/dt = -n_z z
    content = None
    if n[2] != 0:
        s = _lin(u, -np.dot(u, o))
        p = f"({_n(n[2])}*({_n(o[2])}*{t} + {_n(u[2])}*{s}*{t} + {_n(v[2]/2)}*{t}^2))"
        content = [f"{_n(ui)}*{p}" if ui != 0 else "0" for ui in u]
    return Constraint(f"{_lin(n)[1:-1]} = {float(d)!r}", energy, content,
                      shape=("plane", tuple(n), d), parameters=k.names,
                      contact_angle=contact_angle,
                      frame=(tuple(map(float, o)), tuple(map(float, u)), tuple(map(float, v)),
                             tuple(map(float, n)), float(tension)))


def mirror(axis, at: float = 0.0) -> Constraint:
    """A mirror plane (no wetting energy): ``"x"`` for x = ``at`` (also
    ``"y"``, ``"z"``), or ``(normal, point)``. The liquid may lie on either
    side; the volume is closed over the face on the mirror."""
    if isinstance(axis, str):
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y' or 'z', not {axis!r}")
        n = np.eye(3)["xyz".index(axis)]
        return plane(n, at)
    normal, point = axis
    return plane(normal, point=point)


def sphere(center, radius: float, contact_angle: Optional[float] = None, *,
           wet_poles: Iterable[str] = (), span: float = 2*np.pi,
           tension: float = 1.0, nonnegative: bool = False, axis=None,
           gauge: str = "pole") -> Constraint:
    """A solid sphere, the liquid outside. The azimuth runs about the z axis
    through the centre; its poles are ``center +- radius*z``, where the
    integrals are singular: keep contact lines off them.

    ``axis``: the azimuth runs about it instead, and the integrals are
    singular only at the north pole, ``center + radius*axis`` (regular at
    the south pole). In a pack of beads, point each bead's axis at a
    neighbour it nearly touches, where a ring of liquid stays.

    ``wet_poles``: the poles ("north", "south") inside the wetted region, and
    ``span``: the azimuthal extent of the computed piece (2 pi for a whole
    surface, pi/2 for a quarter cut by mirrors through the axis). They set
    :attr:`Constraint.volconst` and :attr:`Constraint.energy_constant`.
    ``nonnegative`` makes it one-sided (vertices may not enter the sphere).

    A coordinate of ``center`` may be a datafile parameter's name or an
    expression (``(0, 0, "1 + gap/2")``), so ``ev.parameters["gap"] = 0.5``
    moves the sphere (and :func:`pysurfaceevolver.recipes.arclength` can
    follow the parameter). The volume over a wet pole is then in the line
    integral itself rather than in ``volconst``, which can't follow the
    parameter; with both poles wet the centre must be numbers. With
    ``axis``, the centre must be numbers and ``span`` 2 pi.

    ``gauge="flux"``: no pole at all. The wetted area is the flux out of the
    liquid of the field ``radius^2 (x - c)/|x - c|^3`` (singular only at the
    centre, inside the solid) through the liquid's surface, a facet
    integral, plus smooth line integrals on the other walls' contact lines
    (:mod:`pysurfaceevolver._flux`, which :func:`make_datafile` puts
    together for all of them; the volume likewise). Its walls may only be
    spheres with this gauge and planes, every flat face the liquid meets a
    plane; the centre must be numbers and ``span`` 2 pi (``axis`` and
    ``wet_poles`` don't apply).
    """
    c = tuple(center)
    R = float(radius)
    if len(c) != 3 or R <= 0:
        raise ValueError("center must be a 3-vector and radius positive")
    moving = any(isinstance(x, str) for x in c)
    if gauge not in ("pole", "flux"):
        raise ValueError(f"gauge must be 'pole' or 'flux', not {gauge!r}")
    if gauge == "flux":
        if moving or span != 2*np.pi:
            raise ValueError("a sphere with gauge='flux' needs a centre of numbers and span 2 pi")
        return _flux_sphere(tuple(float(x) for x in c), R, contact_angle, tension, nonnegative)
    if axis is not None:
        if moving or span != 2*np.pi:
            raise ValueError("a sphere with an axis needs a centre of numbers and span 2 pi")
        return _axis_sphere(tuple(float(x) for x in c), R, contact_angle, _unit(axis, "axis"),
                            set(wet_poles), tension, nonnegative)
    text = [f"({x})" if isinstance(x, str) else _n(float(x)) for x in c]
    poles = set(wet_poles)
    if not poles <= {"north", "south"}:
        raise ValueError("wet_poles may hold 'north' and 'south'")
    if moving and len(poles) == 2:
        raise ValueError("a sphere with both poles wet needs a center of numbers")
    k = _Coefficient(contact_angle, tension)
    X, Y, Z = (f"(x - {text[0]})", f"(y - {text[1]})", f"(z - {text[2]})")
    rho2 = f"({X}^2 + {Y}^2)"
    dphi = (f"(-{Y})/{rho2}", f"{X}/{rho2}", "0")       # d(azimuth) about z
    energy = None
    if k:
        # wetted area = -integral of R h dphi (+ R^2 span per wet pole, with sign)
        energy = [f"{k.times(R)}*{Z}*{d}" if d != "0" else "0" for d in dphi]
    # volume over the wetted face: integral of Q(h) dphi, Q = c_z h^2/2 + h^3/3
    north, south = "north" in poles, "south" in poles
    if moving:
        # with the wet pole's Q taken off inside: around that pole the azimuth
        # turns by -span (north) or span (south) along the contact line, so
        # the integral takes in what volconst holds for a fixed centre
        h0 = R if north else -R if south else 0.0
        q = (f"(({text[2]})/2*({Z}^2 - {_n(h0*h0)}) + ({Z}^3 - {_n(h0**3)})/3)" if poles
             else f"(({text[2]})/2*{Z}^2 + {Z}^3/3)")
        volconst = 0.0
    else:
        cz = float(c[2])
        q = f"({_n(cz/2)}*{Z}^2 + {Z}^3/3)"
        Q = lambda h: cz*h*h/2 + h**3/3
        volconst = -span*((Q(R) if north else 0.0) - (Q(-R) if south else 0.0))
    content = [f"{q}*{d}" if d != "0" else "0" for d in dphi]
    area = R*R*span*((1.0 if north else 0.0) + (1.0 if south else 0.0))
    formula = f"{X}^2 + {Y}^2 + {Z}^2 = {float(R*R)!r}"
    names = k.names + tuple(n for x in c if isinstance(x, str) for n in _identifiers(x))
    return Constraint(formula, energy, content, volconst, k.energy(area), nonnegative, area,
                      shape=("sphere", c if moving else tuple(float(x) for x in c), R),
                      energy_constant_text=(f"-{k.expr}*{_n(area)}" if k.expr and area else None),
                      parameters=tuple(sorted(set(names))), contact_angle=contact_angle,
                      pole_energy=f"-({k.times(R*R*span)})" if k else None,
                      pole_axis=(0.0, 0.0, 1.0))


def _axis_sphere(c: Tuple[float, ...], R: float, contact_angle: Optional[float],
                 e3: np.ndarray, poles: set, tension: float,
                 nonnegative: bool) -> Constraint:
    """sphere() with an axis e3: the integrals singular only at the north
    pole c + R e3, smooth everywhere else near the sphere.

    The wetted area is the integral of R (R + h) dphi about e3 (x - c = a e1 +
    b e2 + h e3), smooth at the south pole. Evolver's volume, z dx dy, on the
    sphere is c_z dx dy + R (n_z^2 - 1/3) dA + R/3 dA: the first two have no
    total over the sphere and polynomial forms (-c_z X dY, -(Z/3)(X dY - Y dX)),
    the last is the area's. Evolver integrates along chords, a little inside
    the sphere, so where the forms are singular in space counts too: dphi's
    denominator a^2 + b^2 (zero along the whole axis, which the chords near
    the south pole pass) is written a^2 + b^2 + (R^2 - r^2)(R - h)/(2R),
    equal on the sphere, zero inside it only on the axis at the north pole.
    (Two earlier constructions from the z frame left points where singular
    terms cancelled only on the sphere: dry patches opened there ran away.)"""
    if not poles <= {"north", "south"}:
        raise ValueError("wet_poles may hold 'north' and 'south'")
    e1, e2 = _frame(e3)
    k = _Coefficient(contact_angle, tension)
    C = np.asarray(c)
    a, b, h = (_lin(e, -float(np.dot(e, C))) for e in (e1, e2, e3))
    X, Y, Z = (f"(x - {_n(c[0])})", f"(y - {_n(c[1])})", f"(z - {_n(c[2])})")
    cz = float(C[2])
    D = (f"({a}^2 + {b}^2 + ({_n(R*R)} - {a}^2 - {b}^2 - {h}^2)*({_n(R)} - {h})/{_n(2*R)})")
    # R (R + h) dphi, dphi = (a db - b da)/D
    area = [f"{_n(R)}*({_n(R)} + {h})*({_n(e2[i])}*{a} - {_n(e1[i])}*{b})/{D}" for i in range(3)]
    energy = [f"{k.times(1.0)}*{t}" for t in area] if k else None
    dX, dY = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    content = []
    for i in range(3):
        terms = [f"{_n(R/3)}*{area[i]}"]
        xy = dY[i]*1.0                       # -c_z X dY - (Z/3)(X dY - Y dX)
        if i == 1:
            terms.append(f"(-({_n(cz)} + {Z}/3)*{X})")
        if i == 0:
            terms.append(f"({Z}/3*{Y})")
        del xy
        content.append(" + ".join(terms))
    del dX
    span = 2*np.pi
    north = "north" in poles
    r_a = 2*R*R                              # R (R + h) at the north pole
    volconst = -span*r_a*R/3 if north else 0.0
    area_c = span*r_a if north else 0.0
    formula = (f"(x - {_n(c[0])})^2 + (y - {_n(c[1])})^2 + (z - {_n(c[2])})^2 = "
               f"{float(R*R)!r}")
    return Constraint(formula, energy, content, volconst, k.energy(area_c), nonnegative, area_c,
                      shape=("sphere", c, R),
                      energy_constant_text=(f"-{k.expr}*{_n(area_c)}" if k.expr and area_c
                                            else None),
                      parameters=k.names, contact_angle=contact_angle,
                      pole_energy=f"-({k.times(span*r_a)})" if k else None,
                      pole_axis=(float(e3[0]), float(e3[1]), float(e3[2])), south_regular=True,
                      native=(tuple(float(v) for v in C), float(R), tuple(map(float, e1)),
                              tuple(map(float, e2)), tuple(map(float, e3)),
                              float(k.value) if k.expr is None and k.value is not None
                              else float("nan")))


def _flux_sphere(c: Tuple[float, ...], R: float, contact_angle: Optional[float], tension: float,
                 nonnegative: bool) -> Constraint:
    """sphere(gauge="flux"): its own line integral only the volume's part
    with no total over the sphere (Evolver's z dx dy there, less the area's
    R/3 dA, which the facets' field takes: -c_z X dY - (Z/3)(X dY - Y dX),
    as _axis_sphere's); the rest make_datafile adds (_flux)."""
    k = _Coefficient(contact_angle, tension)
    X, Y, Z = (f"(x - {_n(c[0])})", f"(y - {_n(c[1])})", f"(z - {_n(c[2])})")
    content = [f"({Z}/3*{Y})", f"(-({_n(c[2])} + {Z}/3)*{X})", "0"]
    formula = (f"(x - {_n(c[0])})^2 + (y - {_n(c[1])})^2 + (z - {_n(c[2])})^2 = "
               f"{float(R*R)!r}")
    return Constraint(formula, None, content, 0.0, 0.0, nonnegative, 0.0,
                      shape=("sphere", c, R), parameters=k.names, contact_angle=contact_angle,
                      flux=(c, R, contact_angle, tension))


def cylinder(point, direction, radius: float, contact_angle: Optional[float] = None, *,
             gauge: str = "azimuthal", ref=None, tension: float = 1.0,
             nonnegative: bool = False) -> Constraint:
    """A solid cylinder (a fibre) of ``radius`` about the line through ``point``
    along ``direction``, the liquid outside.

    Where the wetted region's boundary runs along a mirror plane, the line
    integrals must vanish there; pick the form that does:

    * ``gauge="azimuthal"`` (default): integrals of l dphi (l along the axis,
      from ``point``; phi around it). Vanishes on mirrors through the axis and
      on the mirror normal to the axis through ``point``. For example a
      vertical fibre through a film, cut by mirrors through its axis.
    * ``gauge="axial"``: integrals of phi dl, phi measured from ``ref`` (a
      direction normal to the axis). Vanishes on every mirror normal to the
      axis and on the mirror through the axis containing ``ref``. For example
      a fibre along x, cut by mirrors x = const and by z = 0 (``ref`` along y).
      The wetted region must not cross phi = +-pi (opposite ``ref``).
    """
    p = np.asarray(point, dtype=float)
    d = _unit(direction, "direction")
    R = float(radius)
    if p.shape != (3,) or R <= 0:
        raise ValueError("point must be a 3-vector and radius positive")
    if gauge not in ("azimuthal", "axial"):
        raise ValueError(f"gauge must be 'azimuthal' or 'axial', not {gauge!r}")
    if ref is None:
        e1, e2 = _frame(d)
    else:
        r = np.asarray(ref, dtype=float)
        r = r - np.dot(r, d)*d
        e1 = _unit(r, "ref (normal to the axis)")
        e2 = np.cross(d, e1)
    k = _Coefficient(contact_angle, tension)
    # coordinates: x = p + l d + a e1 + b e2, with a^2 + b^2 = R^2
    ell = _lin(d, -np.dot(d, p))
    a = _lin(e1, -np.dot(e1, p))
    b = _lin(e2, -np.dot(e2, p))
    # the wetted face's part of Evolver's volume (z dx dy), with the face's
    # normal out of the liquid (into the fibre): -R z n_z dphi dl, where
    # R n_z = a e1_z + b e2_z and z = p_z + l d_z + a e1_z + b e2_z
    rnz = f"({a}*{_n(e1[2])} + {b}*{_n(e2[2])})"
    has_content = e1[2] != 0 or e2[2] != 0
    if gauge == "azimuthal":
        dphi = [f"({_n(e2[i])}*{a} - {_n(e1[i])}*{b})/{_n(R*R)}" for i in range(3)]
        energy = [f"{k.times(R)}*{ell}*{dp}" for dp in dphi] if k else None
        # along the contact line (oriented by the facets) the integral of
        # R l dphi is minus the wetted area, as for the energy: so Q is the
        # integral over l of +z R n_z (a horizontal fibre's volume came out
        # with the wetted face's part added instead of taken off)
        q = f"({rnz}*({ell}*({_n(p[2])} + {rnz}) + {_n(d[2]/2)}*{ell}^2))"
        content = [f"{q}*{dp}" for dp in dphi] if has_content else None
    else:
        phi = f"atan2({b}, {a})"
        energy = [f"{k.times(-R*d[i])}*{phi}" if d[i] != 0 else "0"
                  for i in range(3)] if k else None
        # P = -R * integral_0^phi z n_z dphi, with cos(phi) = a/R, sin(phi) = b/R
        e1z, e2z = e1[2], e2[2]
        w1 = f"({_n(e1z)}*{b} - {_n(e2z)}*({a} - {_n(R)}))/{_n(R)}"      # int w
        w2 = (f"({_n((e1z**2 + e2z**2)/2)}*{phi} + {_n((e1z**2 - e2z**2)/2)}*{a}*{b}/{_n(R*R)}"
              f" + {_n(e1z*e2z)}*{b}^2/{_n(R*R)})")                       # int w^2
        pp = f"({_n(-R)}*(({_n(p[2])} + {_n(d[2])}*{ell})*{w1} + {_n(R)}*{w2}))"
        content = ([f"{_n(d[i])}*{pp}" if d[i] != 0 else "0" for i in range(3)]
                   if has_content else None)
    formula = f"{a}^2 + {b}^2 = {float(R*R)!r}"
    return Constraint(formula, energy, content, 0.0, 0.0, nonnegative,
                      shape=("cylinder", tuple(p), tuple(d), R), parameters=k.names,
                      contact_angle=contact_angle)


@dataclass(frozen=True)
class Boundary:
    """A curve for :func:`make_datafile` (Evolver's ``boundary``): refining
    puts new vertices on it, so a film keeps a curved wire however much it
    is refined (a wire of fixed vertices alone stays the polygon it started
    as).

    ``formulas`` give x, y, z in terms of the parameter ``p1``;
    ``parameter(xyz)`` gives each vertex's parameter from its coordinates
    (an ``(n, 3)`` array in, ``n`` values out).

    ``content``, three expressions in x, y, z, is a vector field integrated
    along the body's edges on the wire and added to its volume (Evolver's
    content integral, oriented by the facets as on a constraint): the volume
    of a face that bounds the body but isn't in the surface. For a flat face
    at height h spanning the wire, ``("-h*y/2", "h*x/2", "0")`` adds its
    z dx dy (a bridge pinned on the rim of a disk at z = h).
    """

    formulas: Tuple[str, str, str]
    parameter: Any = field(compare=False)
    content: Optional[Tuple[str, str, str]] = None

    def text(self) -> str:
        """The boundary's body in a datafile (after ``boundary n parameters 1``)."""
        out = "".join(f"x{i + 1}: {f}\n" for i, f in enumerate(self.formulas))
        if self.content is not None:
            out += "content:\n" + "".join(f"c{i + 1}: {c}\n" for i, c in enumerate(self.content))
        return out


def curve(x: str, y: str, z: str, parameter, content=None) -> Boundary:
    """A curve x(p1), y(p1), z(p1) given as datafile expressions, with
    ``parameter(xyz)`` mapping vertex coordinates to p1, and an optional
    ``content`` integral (see :class:`Boundary`)."""
    return Boundary((x, y, z), parameter, None if content is None else tuple(content))


def circle(center=(0.0, 0.0, 0.0), radius: float = 1.0, normal=(0.0, 0.0, 1.0),
           content=None) -> Boundary:
    """A circular wire of ``radius`` about ``center`` in the plane normal to
    ``normal``, parametrized by the angle p1 (radians), with an optional
    ``content`` integral (see :class:`Boundary`)."""
    c = np.asarray(center, dtype=float)
    n = _unit(normal, "normal")
    u, v = _frame(n)
    R = float(radius)
    if c.shape != (3,) or R <= 0:
        raise ValueError("center must be a 3-vector and radius positive")
    formulas = tuple(f"{_n(c[i])} + {_n(R*u[i])}*cos(p1) + {_n(R*v[i])}*sin(p1)"
                     for i in range(3))

    def parameter(xyz):
        d = np.asarray(xyz, dtype=float)[:, :3] - c
        return np.arctan2(d @ v, d @ u)
    return Boundary(formulas, parameter,  # type: ignore[arg-type]
                    None if content is None else tuple(content))
