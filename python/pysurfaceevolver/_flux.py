"""Spheres without poles: the integrals of :func:`constraints.sphere`'s
``gauge="flux"``, which couple every wall, put together by make_datafile.

A sphere's wetted area has no line integral singular nowhere on the sphere
(its area form isn't exact), and sphere(axis=)'s pole had to be kept off
the contact lines, moved as they swept across. Here, with
``F_i = R_i^2 (x - c_i)/|x - c_i|^3`` outside the sphere (divergence-free
there; ``F_i . n = 1`` on it) and ``(x - c_i)/R_i`` inside, where no
liquid is (bounded: no jump for a surface pushed through the solid), the divergence
theorem over the liquid gives bead i's wetted area as the flux of F_i out
of the liquid through its surface (a facet integral) plus that through the
other walls' wetted parts. On another sphere j and on the planes F_i's flux
is exact, a smooth line integral along the contact lines there: of the
monopole potential ``R_i^2 (d x w)/(r (r - d . w))`` (w = x - c_i,
r = |w|), whose curl is F_i, singular only on the ray from c_i along d. On
sphere j, d points away from j; on the planes one d for all (so where two
planes meet the pieces along their edge cancel), a direction no plane is
across from the centre on.

So, with k_i = tension cos(angle):

* the walls' wetting energy, -sum k_i A_i: a global facet integral of
  ``-sum k_i F_i`` (the liquid in front of each facet), and on each wall's
  contact lines ``-sum k_i A_i^d`` over the spheres but its own (a plane's
  own wetting as plane() has it);
* the volume: the bodies' volume method a facet integral of
  ``(0, 0, z) - sum (R_i/3) F_i`` (divergence 1 in the liquid), and each
  wall's content its own (plane()'s, sphere's exact part) plus
  ``-sum (R_i/3) A_i^d`` over the spheres but its own.

No constants are left out: volconst and the energy constant are 0.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple

import numpy as np

VOLUME_METHOD = "pyse_flux_volume"
ENERGY_QUANTITY = "pyse_flux_wetting"


def _n(x: float) -> str:
    x = float(x)
    if x == 0:
        return "0"
    return f"({x!r})" if x < 0 else repr(x)


def _w(c) -> Tuple[str, str, str]:
    return tuple(f"(x{k + 1} - {_n(c[k])})" for k in range(3))     # type: ignore[return-value]


def source(c, R: float) -> List[str]:
    """F = R^2 (x - c)/max(|x - c|, R)^3, per component: outside the sphere
    the point source; inside, where liquid never is, (x - c)/R, bounded
    (singular at the centre, a surface pushed across it changed the wetted
    area by 4 pi R^2 in one gradient step, Evolver's line search found the
    jump, and a pack's descent ran away: every change refused)."""
    w = _w(c)
    r3 = f"maximum({w[0]}^2 + {w[1]}^2 + {w[2]}^2, {float(R*R)!r})^1.5"
    return [f"{_n(R*R)}*{w[k]}/{r3}" for k in range(3)]


def monopole(c, R: float, d) -> List[str]:
    """R^2 (d x w)/(r (r - d . w)), w = x - c: curl F, singular only on the
    ray from c along d."""
    d = np.asarray(d, float)/np.linalg.norm(d)
    w = _w(c)
    r = f"sqrt({w[0]}^2 + {w[1]}^2 + {w[2]}^2)"
    dw = " + ".join(f"{_n(d[k])}*{w[k]}" for k in range(3) if d[k] != 0) or "0"
    den = f"({r}*({r} - ({dw})))"
    cross = [(1, 2), (2, 0), (0, 1)]
    out = []
    for i, (a, b) in enumerate(cross):
        terms = []
        if d[a] != 0:
            terms.append(f"{_n(d[a])}*{w[b]}")
        if d[b] != 0:
            terms.append(f"-{_n(d[b])}*{w[a]}")
        out.append(f"{_n(R*R)}*({' + '.join(terms)})/{den}" if terms else "0")
    return out


def _plane_direction(planes: List[Tuple[np.ndarray, float]], centres: List[np.ndarray]) -> np.ndarray:
    """A direction no plane is across from any centre on: n . d >= 0 for every
    plane's normal n (into the liquid), so the ray from a centre along d
    meets none of them."""
    candidates = [np.eye(3)[k]*s for k in (2, 0, 1) for s in (1.0, -1.0)]
    normals = [n for n, _ in planes]
    for d in candidates:
        if all(float(n @ d) >= -1e-12 for n in normals):
            return d
    # (any direction in the cone the normals leave: their mean, if it works)
    if normals:
        d = np.mean(normals, axis=0)
        if np.linalg.norm(d) > 0:
            d = d/np.linalg.norm(d)
            if all(float(n @ d) >= -1e-12 for n in normals):
                return d
    raise ValueError("gauge='flux': the planes leave no direction to run the spheres' strings "
                     "out along (a closed box): not supported")


def _sum(parts: List[str]) -> str:
    parts = [p for p in parts if p and p != "0"]
    return " + ".join(parts) if parts else "0"


def assemble(cons: Dict[Any, Any]) -> Tuple[Dict[Any, Any], str]:
    """The constraints with the flux gauge's line integrals added, and the
    datafile lines its facet integrals need (before the vertices)."""
    from .constraints import _Coefficient
    spheres = [(key, spec) for key, spec in cons.items() if getattr(spec, "flux", None)]
    if not spheres:
        return cons, ""
    planes = []
    for key, spec in cons.items():
        shape = getattr(spec, "shape", None)
        if getattr(spec, "flux", None):
            continue
        if shape is None or shape[0] != "plane":
            raise ValueError(f"constraint {key}: with spheres of gauge='flux', every wall must be "
                             "such a sphere or a plane (constraints.plane/mirror)")
        planes.append((key, np.asarray(shape[1], float), float(shape[2])))
    info = []
    for key, spec in spheres:
        c, R, angle, tension = spec.flux
        info.append((key, np.asarray(c, float), float(R), _Coefficient(angle, tension)))
    d_planes = _plane_direction([(n, d) for _, n, d in planes], [c for _, c, _, _ in info])

    def coefficient(k, factor: float) -> str:
        """-k*factor as a datafile factor ('' for 0)."""
        if not k:
            return ""
        return f"(-{k.times(factor)})" if k.expr is not None else _n(-float(k.value)*factor)

    out = dict(cons)
    # each wall's line integrals: the spheres' but its own
    for key, spec in cons.items():
        energy: List[List[str]] = [[], [], []]
        content: List[List[str]] = [[], [], []]
        own_sphere = getattr(spec, "flux", None)
        for skey, c, R, k in info:
            if skey == key:
                continue
            if own_sphere:
                d = c - np.asarray(spec.flux[0], float)        # away from this sphere
            else:
                d = d_planes
            A = monopole(c, R, d)
            ke = coefficient(k, 1.0)
            for m in range(3):
                if A[m] == "0":
                    continue
                if ke:
                    energy[m].append(f"{ke}*{A[m]}")
                content[m].append(f"{_n(-R/3)}*{A[m]}")
        base_e = list(spec.energy) if spec.energy is not None else ["0", "0", "0"]
        base_c = list(spec.content) if spec.content is not None else ["0", "0", "0"]
        new_e = [_sum([base_e[m]] + energy[m]) for m in range(3)]
        new_c = [_sum([base_c[m]] + content[m]) for m in range(3)]
        out[key] = dataclasses.replace(
            spec, energy=new_e if any(e != "0" for e in new_e) else None,
            content=new_c if any(e != "0" for e in new_c) else None)
    # the facet integrals: the volume method and the wetting energy
    vol = [[], [], ["x3"]]
    wet: List[List[str]] = [[], [], []]
    for _, c, R, k in info:
        F = source(c, R)
        ke = coefficient(k, 1.0)
        for m in range(3):
            vol[m].append(f"{_n(-R/3)}*{F[m]}")
            if ke:
                wet[m].append(f"{ke}*{F[m]}")
    # (three Gauss points a facet: the fields are smooth on the facets, at
    # least a radius from the centres; the exact plane-sphere bridge came out
    # the same to six digits as with twelve, and a pack's step 3.7 times faster)
    lines = ["integral_order_2d 2\n",
             f"method_instance {VOLUME_METHOD} method facet_vector_integral\n",
             "vector_integrand:\n"]
    lines += [f"q{m + 1}: {_sum(vol[m])}\n" for m in range(3)]
    lines.append(f'volume_method_name "{VOLUME_METHOD}"\n')
    if any(wet[m] for m in range(3)):
        lines += [f"quantity {ENERGY_QUANTITY} energy method facet_vector_integral global\n",
                  "vector_integrand:\n"]
        lines += [f"q{m + 1}: {_sum(wet[m])}\n" for m in range(3)]
    return out, "".join(lines)


def native_specs(walls: Dict[Any, Any]) -> "tuple[list, list]":
    """What bindings/pyse_native.c evaluates for the flux gauge, for walls
    (key -> Constraint): per wall, (key, which, base, bc, terms) for its
    energy (which 0) and content (1) line integrands; and per facet method,
    (name, base, terms). Terms are 11 numbers each (centre, radius,
    direction or zeros, energy flag, angle, tension, 0). Empty where a wall
    has terms C doesn't evaluate (a plane's own wetting or content, an
    angle given as an expression): those stay with Evolver's interpreter."""
    spheres = [(k, w) for k, w in walls.items() if getattr(w, "flux", None)]
    if not spheres:
        return [], []
    planes = [(k, w) for k, w in walls.items() if not getattr(w, "flux", None)]
    if any(getattr(w, "shape", (None,))[0] != "plane" for _, w in planes):
        return [], []
    d_planes = _plane_direction([(np.asarray(w.shape[1], float), float(w.shape[2]))
                                 for _, w in planes], [])

    def numeric(angle) -> bool:
        return angle is None or not isinstance(angle, str)

    def term(c, R, d, energy, angle, tension) -> List[float]:
        a = 90.0 if angle is None else float(angle)
        return [*map(float, c), float(R), *map(float, d), float(energy), a, float(tension), 0.0]

    if not all(numeric(w.flux[2]) for _, w in spheres):
        return [], []
    lines = []
    for key, w in walls.items():
        own = getattr(w, "flux", None)
        if not own:
            # a plane's own integrands: C knows none (fine where it has none)
            if w.energy is not None or w.content is not None:
                continue
        energy, content = [], []
        for skey, s in spheres:
            if skey == key:
                continue
            c, R, angle, tension = s.flux
            d = np.asarray(c, float) - np.asarray(own[0], float) if own else d_planes
            d = d/np.linalg.norm(d)
            if angle is not None and np.cos(np.radians(float(angle))) != 0:
                energy += term(c, R, d, 1, angle, tension)
            content += term(c, R, d, 0, angle, tension)
        if energy:
            lines.append((key, 0, 0, [0.0, 0.0, 0.0], energy))
        lines.append((key, 1, 2 if own else 0, list(map(float, own[0])) if own else [0.0]*3,
                      content))
    vol, wet = [], []
    for _, s in spheres:
        c, R, angle, tension = s.flux
        vol += term(c, R, (0, 0, 0), 0, angle, tension)
        if angle is not None and np.cos(np.radians(float(angle))) != 0:
            wet += term(c, R, (0, 0, 0), 1, angle, tension)
    facets = [(VOLUME_METHOD, 1, vol)]
    if wet:
        facets.append((ENERGY_QUANTITY, 0, wet))
    return lines, facets
