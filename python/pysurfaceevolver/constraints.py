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
  ``-tension*cos(theta)*area_constant``;
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
``wet_poles``, and add :attr:`Constraint.volconst` to the body's ``volconst``
and :attr:`Constraint.energy_constant` to the energy you report (Evolver can't
add a constant energy; pressures don't need it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["Constraint", "plane", "mirror", "sphere", "cylinder"]


def _n(x: float) -> str:
    """A number for a datafile formula (parenthesized when negative)."""
    x = float(x)
    if x == 0:
        return "0"
    return f"({x!r})" if x < 0 else repr(x)


def _lin(coeffs: Sequence[float], const: float = 0.0) -> str:
    """const + a*x + b*y + c*z, without zero terms."""
    terms = [f"{_n(c)}*{v}" for c, v in zip(coeffs, "xyz") if c != 0]
    if const != 0 or not terms:
        terms.insert(0, _n(const))
    return "(" + " + ".join(terms) + ")"


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


class _Coefficient:
    """tension*cos(contact_angle): a number, or a datafile expression when the
    angle is given as a parameter name or expression (e.g. ``"theta"``)."""

    def __init__(self, contact_angle, tension: float):
        self.value: Optional[float] = None
        self.expr: Optional[str] = None
        if contact_angle is None:
            self.value = 0.0
        elif isinstance(contact_angle, str):
            self.expr = f"({_n(tension)}*cos(({contact_angle})*pi/180))"
        else:
            self.value = float(tension)*float(np.cos(np.radians(contact_angle)))

    def __bool__(self) -> bool:
        return self.expr is not None or self.value != 0

    def times(self, x: float) -> str:
        """k*x as a datafile factor."""
        return _n(self.value*x) if self.expr is None else f"{self.expr}*{_n(x)}"

    def energy(self, area: float) -> float:
        return float("nan") if self.expr is not None else -self.value*area


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
    return Constraint(f"{_lin(n)[1:-1]} = {d!r}", energy, content)


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
           tension: float = 1.0, nonnegative: bool = False) -> Constraint:
    """A solid sphere, the liquid outside. The azimuth runs about the z axis
    through the centre; its poles are ``center +- radius*z``.

    ``wet_poles``: the poles ("north", "south") inside the wetted region, and
    ``span``: the azimuthal extent of the computed piece (2 pi for a whole
    surface, pi/2 for a quarter cut by mirrors through the axis). They set
    :attr:`Constraint.volconst` and :attr:`Constraint.energy_constant`.
    ``nonnegative`` makes it one-sided (vertices may not enter the sphere).
    """
    c = np.asarray(center, dtype=float)
    R = float(radius)
    if c.shape != (3,) or R <= 0:
        raise ValueError("center must be a 3-vector and radius positive")
    poles = set(wet_poles)
    if not poles <= {"north", "south"}:
        raise ValueError("wet_poles may hold 'north' and 'south'")
    k = _Coefficient(contact_angle, tension)
    X, Y, Z = (f"(x - {_n(c[0])})", f"(y - {_n(c[1])})", f"(z - {_n(c[2])})")
    rho2 = f"({X}^2 + {Y}^2)"
    dphi = (f"(-{Y})/{rho2}", f"{X}/{rho2}", "0")       # d(azimuth) about z
    energy = None
    if k:
        # wetted area = -integral of R h dphi (+ R^2 span per wet pole, with sign)
        energy = [f"{k.times(R)}*{Z}*{d}" if d != "0" else "0" for d in dphi]
    # volume over the wetted face: integral of Q(h) dphi, Q = c_z h^2/2 + h^3/3
    q = f"({_n(c[2]/2)}*{Z}^2 + {Z}^3/3)"
    content = [f"{q}*{d}" if d != "0" else "0" for d in dphi]
    north, south = "north" in poles, "south" in poles
    Q = lambda h: c[2]*h*h/2 + h**3/3
    volconst = -span*((Q(R) if north else 0.0) - (Q(-R) if south else 0.0))
    area = R*R*span*((1.0 if north else 0.0) + (1.0 if south else 0.0))
    formula = f"{X}^2 + {Y}^2 + {Z}^2 = {R*R!r}"
    return Constraint(formula, energy, content, volconst, k.energy(area), nonnegative, area)


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
        q = f"(-{rnz}*({ell}*({_n(p[2])} + {rnz}) + {_n(d[2]/2)}*{ell}^2))"
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
    formula = f"{a}^2 + {b}^2 = {R*R!r}"
    return Constraint(formula, energy, content, 0.0, 0.0, nonnegative)
