# Getting started

A drop on a table, end to end, and the conventions every surface built from arrays
follows: which way normals point, what `ev.energy` counts, how to change the surface
tension, how accurate the measured contact angles are, and how Evolver's numbers
relate to array rows.

## A drop on a table

A drop of volume 1 on the plane z = 0 with a contact angle of 60 degrees, from a
spherical cap that holds less than that: relaxing meets the target volume.

```python
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes

drop = shapes.cap(radius=1, height=0.6)            # vertices, faces and the rim's rows
ev = pyse.Evolver()
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60)},  # normal into the liquid
               vertex_constraints={1: drop.rims["rim"]},   # the rim slides on the table
               bodies=[pyse.Body(volume=1.0)])             # faces: all, out of the liquid
result = ev.relax(levels=2)                        # refine twice, relaxing after each
print(f"converged {result.converged}, stable {result.stable}, {ev.counts['facets']} facets")
print(f"energy {ev.energy:.5f}, pressure {ev.body(1).pressure:.5f}")

line = ev.contact_lines(1)[0]                      # the contact line, in order (closed)
radius = np.linalg.norm(line[:-1, :2] - line[:-1, :2].mean(axis=0), axis=1).mean()
print(f"contact radius {radius:.5f}, contact angle {ev.contact_angles(1).mean():.2f}")
```

```text
converged True, stable True, 7680 facets
energy 2.60490, pressure 1.73661
contact radius 0.99770, contact angle 59.23
```

The exact spherical cap has energy 2.60470, pressure 1.73647 and contact radius
0.99746. Next steps:

```python
ev.plot(walls=True)                       # the drop with the table (walls given to load_arrays)
ev.write_bodies("drop_{id}.stl", cap=True)   # a closed surface, the wetted disc as a cap
surface = ev.body_surfaces(cap=True)[1]
surface.volume, surface.areas()           # enclosed volume; area of the drop, of the cap
```

`pyse.shapes` has other starts: `disc`, `tube` (a liquid bridge or a film between
rings), `revolve` (any meridian), `box`. Starts needn't hold the target volume;
they need the right topology and contact lines on the right walls.

## Conventions

**Orientation.** Facets point out of the liquid (out of the body). A plane's normal,
given to `C.plane`, points into the liquid, away from the solid; spheres and cylinders
(`C.sphere`, `C.cylinder`) are solid, with the liquid outside. When a body's faces close
a surface, `make_datafile` orients them outward itself; otherwise it takes their vertex
order (or `Body(orientation=...)`). A body whose faces point into it, or a plane facing
the solid, gets a warning when it loads.

**Numbers and rows.** Evolver numbers elements and bodies from 1: `ev.body(1)`,
constraint 1 in `vertex_constraints`, `ev.on_constraint(1)`, and commands such as
`ev.eval("vertex[5].z")`. Arrays are NumPy rows from 0: `ev.mesh().vertices[0]`,
`ev.values("facet", "area")[0]`, `ev.bodies().pressure[0]` (body 1). The Evolver
number of each row is in `mesh.vertex_ids`, `edge_ids`, `facet_ids`; after refining
or remeshing, rows and numbers no longer differ by one, so map through these.
Constraints may also be named (`constraints={"table": C.plane(...)}`, avoiding
Evolver's reserved words such as `floor`), and then referred to by name.

## Energies: `ev.energy` and `ev.total_energy`

A wall's wetting energy, $-\gamma\cos\theta$ per wetted area, and its part of the
liquid's volume are line integrals along the contact line, which `pyse.constraints`
writes. Where those integrals can't cover the whole wetted region (a sphere's wetted
pole, say), the constraint knows the rest: the volume goes into the body's `volconst`
automatically, summed over the walls the body's vertices touch, and the energy into a
constant that Evolver can't add itself. With a single body, a solid that sits inside the
liquid without touching the surface (a bead under a meniscus) counts too. With several
bodies, a wall that several of them touch, or that none touches, is left to you, with a
warning: give each body's share yourself (`Body(volume=..., faces=..., volconst=...)`;
an explicit value wins).

* `ev.total_energy` is Evolver's own total: without that constant.
* `ev.energy` is `ev.total_energy` plus the constant: the energy of the wetted
  configuration counted from dry walls, the one to compare with a reference.

Pressures, forces and stability don't depend on a constant, so they are the same
either way. For a drop on a plane the integrals cover the whole wetted disc, and the
two energies agree.

## A surface tension other than 1

Facets have tension 1 unless told otherwise. Set it per facet after loading (refined
facets inherit it), and give the walls the same tension, so that their wetting energy
is $-\gamma\cos\theta$ and Young's law holds:

```python
gamma = 0.072
ev.load_arrays(drop.vertices, drop.faces,
               constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60, tension=gamma)},
               vertex_constraints={1: drop.rims["rim"]}, bodies=[pyse.Body(volume=1.0)])
ev.set_values("facet", "tension", gamma)       # or per facet: one value per facet row
ev.relax(levels=2)                             # energy and pressure scale with gamma
```

Without gravity the shape doesn't change; with gravity, the capillary length
$\sqrt{\gamma/\rho g}$ does.

## How accurate are the contact angles?

`ev.contact_angles(k)` measures the angle at each contact-line vertex from the flat
facets next to it, so it is first order in the edge length: the error halves with each
refinement. For the drop above (set to 60 degrees):

| facets | mean contact angle | energy |
|---|---|---|
| 7680 | 59.23 | 2.60490 |
| 30720 | 59.62 | 2.60475 |
| 122880 | 59.81 | 2.60471 |
| exact | 60 | 2.60470 |

The energy converges much faster (second order). A degree off on a mesh of a few
thousand facets is normal and doesn't mean the contact angle is wrong in the model:
it is imposed through the energy, exactly. Compare refined meshes, or measure
something smoother (the contact radius, the pressure).
