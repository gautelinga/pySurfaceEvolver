"""Generate the notebooks on Brakke's sample datafiles (docs/brakke_*.ipynb).

    python tools/make_brakke_notebooks.py [output directory, default docs/]

Edit the notebooks here and regenerate: they are written without outputs (the
documentation build executes them), with cell ids numbered so a regeneration
changes only what was edited. Needs nbformat.
"""
import os
import sys

import nbformat as nbf

DOCS = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")

SETUP = '''import warnings
import numpy as np
import pyvista as pv
import pysurfaceevolver as pyse
import brakke_helpers as bh

pv.set_jupyter_backend("static")
warnings.simplefilter("ignore", pyse.EvolverWarning)'''

INTRO_COMMON = """Each sample is run twice from its datafile: once with the recipe Brakke wrote into
it (its `gogo` command, or the usage its comments describe), once with a default
`ev.relax()` given as many refinements as the recipe does (`levels`). The table
shows the energy each ends with, the residual (how far from equilibrium, see
[Relaxing a surface](relaxing.md)), the facet count, the time, and what `relax()`
reports: converged, stable, and any health issues. A lower energy on the same mesh
is not always the better answer: a coarse mesh can lower its energy in ways the
smooth surface can't, so where an exact value is known, the error against it is
the measure. Times are from the documentation build and vary."""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(name, title, intro, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.cells = [md(f"# {title}\n\n{intro}\n\n{INTRO_COMMON}"), code(SETUP)] + cells
    for i, cell in enumerate(nb.cells):
        cell.id = f"{name}-{i}"
    nbf.write(nb, f"{DOCS}/{name}.ipynb")


# ---- 1. films and foams -----------------------------------------------------------

notebook("brakke_films", "Brakke's samples: soap films and foams", """
Soap films spanning wires, a bubble, and foams in a periodic box, from the sample
datafiles that come with Surface Evolver (`pyse.examples`). The helpers are in
[`docs/brakke_helpers.py`](https://github.com/gautelinga/pySurfaceEvolver/blob/main/docs/brakke_helpers.py).""", [
    md("""
## `cube.fe`: a cube becoming a sphere

A unit cube of fixed volume 1, its surface area the energy: it relaxes toward a sphere
(area $(36\\pi)^{1/3} \\approx 4.836$). The recipe refines twice with gradient and Newton
steps in between."""),
    code('''print(bh.recipe("cube.fe"))
runs = bh.compare("cube.fe")
bh.table(runs, reference=(36*np.pi)**(1/3))
bh.show(runs)'''),
    md("""
Both meshes are polyhedra inscribed in the sphere, with a little more area than it;
`relax()` takes its residual to about 1e-9 where the recipe stops sooner, at nearly the
same energy."""),
    md("""
## `catman.fe`: the catenoid of the manual's tutorial

A film between two rings of radius 1 at height $\\pm 0.55$; the stable catenoid has
area 6.511674. The recipe's comments say what happens on this coarse mesh: after 50
steps "at this point have nearly a saddle point", then "triangulation twists around to
lower energy". Twisting lowers the area of the coarse mesh, not its error."""),
    code('''print(bh.recipe("catman.fe"))
runs = bh.compare("catman.fe")
bh.table(runs, reference=6.511674)
bh.show(runs)'''),
    md("""
Both are below the smooth catenoid (the rings are coarse polygons); the twisted mesh
of the recipe is further below. `relax()` stops at the untwisted equilibrium, and
leaves sliding along the wires (where the twist happens) to its gradient rounds."""),
    md("""
## `cat.fe`: a catenoid past its limit

Here the rings are as far apart as a catenoid can span (`RMAX` set at the fold). The
neck collapses, and the recipe deletes the tiny edges there (`t .05`) and pops the
pinched vertex (`o`): the film becomes two flat discs. `relax()` can't converge the
film, so as a last resort it takes gradient steps without tidying, lets the neck
collapse, deletes the collapsed edges and pops (`collapsed`, `popped`)."""),
    code('''print(bh.recipe("cat.fe"))
runs = bh.compare("cat.fe")
R = 1.5088795                       # RMAX: the rings' radius
discs = 2*3*R**2                    # two 12-gons inscribed in the rings (after one refinement)
bh.table(runs, reference=discs)
bh.show(runs)'''),
    md("""
The recipe stops after `g 5` on the discs; `relax()` finishes them: two regular
12-gons, the polygons the rings' vertices make after one refinement."""),
    md("""
## `catbody.fe`: a catenoid enclosing a fixed volume

The same catenoid shape, now closed by plane ends and holding a fixed volume, so it
has a pressure. The file adjusts that volume to find zero pressure (the unstable
catenoid) in a separate script; here only its `gogo`."""),
    code('''print(bh.recipe("catbody.fe"))
runs = bh.compare("catbody.fe")
bh.show(runs)'''),
    md("""
## `octa.fe`: a film in an octahedral frame

A symmetric film from the edges of an octahedral wire frame to its centre: six edges
there carry four films each. Plateau's laws (soap films meet three at a time along an
edge, at 120 degrees, and such edges four at a time at a point) say that is unstable:
splitting the junction lowers the area. The recipe pops junctions (`o`), deletes small
edges (`t .1`) and iterates, several times; `relax()` pops after converging, relaxes
again, and deletes the films that shrink away."""),
    code('''print(bh.recipe("octa.fe"))
runs = bh.compare("octa.fe")
bh.show(runs)'''),
    md("""
Both reach the same film, with a central tetrahedral point."""),
    md("""
## Kelvin against Weaire-Phelan: `twointor.fe` and `phelanc.fe`

Lord Kelvin asked in 1887 how to divide space into cells of equal volume with the
least interface area, and proposed the cell of `twointor.fe` (two cells in a
periodic box). In 1993 Weaire and Phelan found a structure with two kinds of cells
(`phelanc.fe`, eight cells) with about 0.3% less area. Per cell, the area over the
cell volume to the power 2/3 compares them independently of the box."""),
    code('''kelvin = bh.compare("twointor.fe")
wp = bh.compare("phelanc.fe")

def per_cell(name, levels):
    ev = pyse.Evolver(name)
    ev.relax(levels=levels)
    n = len(ev.bodies().volume)
    V = float(np.sum(ev.bodies().volume))
    return ev.total_energy/n/(V/n)**(2/3)

k, w = per_cell("twointor.fe", 2), per_cell("phelanc.fe", 2)
print(f"area per cell / volume^(2/3): Kelvin {k:.5f}, Weaire-Phelan {w:.5f}, "
      f"{1 - w/k:.2%} less")'''),
    code('''bh.show(wp)'''),
    md("""
## `symtest.fe`: the Kelvin foam in a quotient space

The same two Kelvin cells, but in Evolver's quotient-space representation (a symmetry
group instead of the periodic box). The recipe is `twointor.fe`'s."""),
    code('''runs = bh.compare("symtest.fe", bh.recipe("twointor.fe"))'''),
    md("""
The energy is that of `twointor.fe`, but `relax()` reports the surface unstable, with
one negative Hessian eigenvalue (as Brakke's Evolver 2.70a does), and `twointor.fe`
stable. The instability is real: the volumes in this file's `bodies` section are
comments (`// volume 0.500`), so its two cells have no volume constraint, and one can
grow at the expense of the other. Moving along that mode (Evolver's `saddle`) lowers
the energy, and relaxing from there shrinks one cell away."""),
    code('''ev = pyse.Evolver("symtest.fe")
print(ev.bodies().fixed)                    # no volume constraints
ev.relax(levels=1)
e0 = ev.total_energy
ev.saddle()                                 # a step along the negative mode
print(f"energy {e0:.6f} -> {ev.total_energy:.6f}")'''),
    md("""
## Triply periodic minimal surfaces: `pcell.fe`, `dcell.fe`, `IWP.fe`

From Brakke's collection of periodic minimal surfaces online (included with pySE):
Schwarz's P and D surfaces and Schoen's I-WP, each a small piece of the surface in a
quotient space (a symmetry group generates the rest)."""),
    code('''for name in ("pcell.fe", "dcell.fe", "IWP.fe"):
    runs = bh.compare(name)'''),
    md("""
`relax()` agrees with the recipes of P and D to about 1e-5. On I-WP the recipe
refines the edges along the boundary once more and stops unconverged, so the two
end on different meshes."""),
])

# ---- 2. drops and capillarity --------------------------------------------------------

notebook("brakke_capillarity", "Brakke's samples: drops and capillarity", """
Liquid volumes against walls: contact angles, gravity, curved tanks, and the forces a
liquid bridge exerts.""", [
    md("""
## `mound.fe`: a drop on a table

A drop of fixed volume on the plane $z = 0$ with a contact angle (here 90 degrees, a
parameter) and gravity. The contact angle comes from an energy integral on the
contact line; the table top is drawn with fixed facets. `re` refines the contact line
only."""),
    code('''print(bh.recipe("mound.fe"))
runs = bh.compare("mound.fe")
bh.show(runs)'''),
    md("""
The recipe's `re` refines the edges on the table first, so its mesh has more facets
there; the energies agree to about 1e-5. The file's other recipes change the contact
angle and gravity; `gogo2` sets 45 degrees:"""),
    code('''runs = bh.compare("mound.fe", "gogo2", levels=2,
                  setup=lambda ev: ev.parameters.update(angle=45))
bh.show(runs)'''),
    md("""
## `addload_example.fe`: nine drops at once

On loading, the file's `setup` script loads the mound eight more times next to the
first (`addload`), each drop with its own contact angle on its edges (from 50 to 130
degrees)."""),
    code('''runs = bh.compare("addload_example.fe")
bh.show(runs)'''),
    md("""
## `column.fe`: a column of solder

A liquid solder column between two pads, its upper pad offset sideways (a skewed
catenoid shape with a volume and gravity; units cgs). The file goes on to compute the
forces on the pads by central differences."""),
    code('''runs = bh.compare("column.fe")
bh.show(runs)'''),
    md("""
## `sphere.fe`: liquid in a spherical tank

A spherical tank partly filled, with a contact angle on the wall. Straight edges can't
follow the curved wall, so the file gives the gaps between them and the wall an
energy and volume (a "convex" constraint). Evolver has no Hessian for that gap energy,
so `relax()` uses gradient steps only; the recipe uses conjugate gradients at the
end."""),
    code('''print(bh.recipe("sphere.fe"))
runs = bh.compare("sphere.fe")
bh.show(runs)'''),
    md("""
Neither run reaches a residual near zero: gradient steps converge slowly, and the
energies agree to about 3e-4."""),
    md("""
## `tankex.fe`: liquid in a cylindrical tank

A flat-ended cylindrical tank on its side, gravity pulling the liquid to the wall, gap
energies as in `sphere.fe`. The file has no recipe; here two refinements and plain
gradient steps (it starts with 8 facets, too coarse to keep its shape on its own).
With no Hessian, `relax()` goes on with gradient rounds while no facet is degenerate;
from fewer refinements (`levels=1`) its facets degenerate first, and it stops there and
says so instead of sliding into what the degenerate mesh allows."""),
    code('''runs = bh.compare("tankex.fe", "r; r; g 1000", levels=2)
bh.show(runs)'''),
    md("""
## Solder joints: `bga-1.fe`, `cbga1.fe`, `cbga2.fe`

Three files from Brakke's online examples (included with pySE), in cgs units
(surface tension 430-480 erg/cm²). `bga-1.fe` is the first of his ball grid array
tutorial: solder between two round pads."""),
    code('''runs = bh.compare("bga-1.fe")
bh.show(runs)'''),
    md("""
`cbga1.fe` holds solder between a pad and a sphere, its contact with the sphere as
edge integrals. It starts as a hexagonal prism of quads that Evolver splits at their
centres; the recipe first swaps the six vertical edges by hand (`edgeswap`), and
without that, gradient steps settle at an energy of 11 on the badly triangulated prism.
`relax()` equiangulates after its first step, which swaps them. (The recipe ends
unconverged.)"""),
    code('''runs = bh.compare("cbga1.fe")
bh.show(runs)'''),
    md("""
`cbga2.fe` is the same joint with explicit contact facets on the sphere. Its comments
say that at high contact angles the symmetric bulge is unstable, bulging to one side,
and suggest Evolver's `saddle` to move off it. `relax()` finds the symmetric bulge and
reports it unstable; `ev.saddle()` steps along the unstable mode, and relaxing from
there finds the drop leaning over, at a lower energy."""),
    code('''import warnings
ev = pyse.Evolver("cbga2.fe")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", pyse.UnstableEquilibriumWarning)
    r = ev.relax(levels=2)
print(f"symmetric: energy {ev.total_energy:.6f}, {r.negative_modes} unstable directions")
print(f"saddle() lowers the energy by {ev.saddle():.4f}")
r = ev.relax()
print(f"leaning:   energy {ev.total_energy:.6f}, stable: {r.stable}")
ev.plot(color="sandybrown", show_edges=False, smooth_shading=True)'''),
])

# ---- 3. crystals and elastic membranes ---------------------------------------------

notebook("brakke_crystals_membranes", "Brakke's samples: crystals and elastic membranes", """
Energies that are not plain surface tension: a crystalline surface energy, and elastic
membranes that remember their unstretched shape. `relax()` recognizes both and leaves
the mesh as it is (no equiangulation or vertex averaging).""", [
    md("""
## `crystal.fe`: a crystal growing into its Wulff shape

A cube of fixed volume whose surface energy depends on each facet's orientation
through a set of Wulff vectors (`octa.wlf`). The minimizing shape is the Wulff shape
of those vectors. A crystalline energy has a corner at its minimum, so the residual
(a force balance) stays above zero there; Evolver has no Hessian for it either.
`relax()` takes gradient rounds and counts the surface as converged once its energy
settles (it says `energy settled`). The recipe is ten gradient steps."""),
    code('''print(bh.recipe("crystal.fe"))
runs = bh.compare("crystal.fe")
bh.show(runs)'''),
    md("""
`relax()` goes on until the energy settles (what a thousand plain gradient steps
reach), past the recipe's ten."""),
    md("""
## `mylarcube.fe`: a mylar balloon

A cube of mylar inflated to the most volume it can hold without stretching: a
`relaxed_elastic_A` energy that resists stretching but lets the sheet wrinkle freely.
The file redefines `r` to keep each facet's reference shape and warns that any other
retriangulation corrupts it; its recipe raises the elastic modulus as it refines and
ends with `hessian_seek`. `relax()` takes the in-plane forces into its residual and
uses `hessian_seek` for its Newton steps. Here `relax()` follows the recipe's ramp:
the same refinements, the modulus raised from 10 to 100, 1000 and 10000 at the same
points."""),
    code('''print(bh.recipe("mylarcube.fe"))
recipe = bh.run_recipe("mylarcube.fe", label="recipe: gogo")

def ramp(ev):
    ev.relax(levels=1)
    ev.quantity("stretch").modulus = 100
    ev.relax(levels=2)
    ev.quantity("stretch").modulus = 1000
    ev.relax()
    ev.quantity("stretch").modulus = 10000
    return ev.relax()

import time
ev = pyse.Evolver("mylarcube.fe")
t = time.perf_counter()
r = ramp(ev)
relaxed = bh.Run("relax() along the same ramp", ev.total_energy, r.residual,
                 ev.counts["facets"], time.perf_counter() - t, r.converged, r.stable,
                 [], ev.mesh())
bh.table([recipe, relaxed])
bh.show([recipe, relaxed])'''),
    md("""
Neither converges: the wrinkling sheet has many directions that cost almost no energy,
and the file's comments note indefinite Hessians at lower Poisson ratios. The recipe
ends about 0.1% lower (more volume): its conjugate-gradient runs get further on this
slowly settling sheet than `relax()` within its 1000 steps per call. The residual here
counts all force directions, in units of the load (the pressure, 1 here): with a
stiffer sheet the same residual means a smaller strain error."""),
    md("""
## `relaxed.fe`: the elastic method on one triangle

A test of the `relaxed_elastic` energy on a single unstretched triangle (two extra
coordinates hold its reference shape): its energy is 0. Stretched by moving a vertex,
it gains energy, and relaxing lets it shrink back (the method gives compression no
energy, so any shape no larger than the reference has none)."""),
    code('''ev = pyse.Evolver("relaxed.fe")
print("unstretched:", ev.total_energy)
ev.command("set vertex[2] x 1.5")
print("stretched:  ", ev.total_energy)
r = ev.relax()
print("relaxed:    ", ev.total_energy, r)'''),
])

# ---- 4. strings, metrics and higher dimensions -------------------------------------

notebook("brakke_strings", "Brakke's samples: strings, metrics and other dimensions", """
One-dimensional "surfaces" (strings) in the plane and in space, a background metric,
and a three-dimensional bubble in four-dimensional space. For strings, `relax()`
averages vertices but can't equiangulate.""", [
    md("""
## `popstr.fe`: popping a string junction

Four strings meeting at a point. For strings, Plateau's rule is three at a point, at
120 degrees: the recipe pops the junction into two triple points (`o`), and so does
`relax()` after converging."""),
    code('''print(bh.recipe("popstr.fe"))
runs = bh.compare("popstr.fe")
bh.show(runs, view="xy")'''),
    md("""
## `slidestr.fe`: a drop held in a corner

A drop (in two dimensions: an area bounded by a string) on a hyperbolic slide, gravity
pulling it down into the corner that holds it."""),
    code('''print(bh.recipe("slidestr.fe"))
runs = bh.compare("slidestr.fe")
bh.show(runs, view="xy")'''),
    md("""
## `100grain.fe`: a two-dimensional foam coarsening

A Voronoi diagram of 100 cells in a periodic square, with automatic popping and
chopping (`autopop`, `autochop`): edges that shrink away are deleted and the junctions
popped as the foam coarsens. The cells have no fixed areas, so there is no equilibrium
to reach; the file says to "just do 'g' steps in big chunks". Here 1000 of them,
against a default `relax()`, which reports the foam unconverged."""),
    code('''runs = bh.compare("100grain.fe", "g 1000")
bh.show(runs, view="xy")'''),
    md("""
## `metric.fe`: a string in a curved metric

A background metric that turns the punctured plane into a catenoid: the string loop
shrinks onto the catenoid's throat (drawn in the plane's coordinates)."""),
    code('''print(bh.recipe("metric.fe"))
runs = bh.compare("metric.fe")
bh.show(runs, view="xy")'''),
    md("""
## `knotty.fe`: a trefoil knot

A trefoil (3,2 torus knot) with a knot energy that keeps the string from crossing
itself. The file defines several energies and length constraints; this is its default
combination. Knot energies have many local minima: the recipe uses conjugate gradients
(`U`) and may end in a different one."""),
    code('''print(bh.recipe("knotty.fe"))
runs = bh.compare("knotty.fe")
bh.show(runs)'''),
    md("""
## `quadm.fe`: a film in a metric

A soap film on a quadrilateral wire, in a background metric."""),
    code('''print(bh.recipe("quadm.fe"))
runs = bh.compare("quadm.fe")
bh.show(runs)'''),
    md("""
## `simplex3.fe`: a bubble in four dimensions

A three-dimensional bubble in four-dimensional space, starting as a simplex and
becoming a 3-sphere, in Evolver's simplex representation. Evolver has no Hessian for
bodies there (outside the Lagrange model), so `relax()` takes gradient steps only, and
there is nothing to draw in three dimensions."""),
    code('''print(bh.recipe("simplex3.fe"))
runs = bh.compare("simplex3.fe")'''),
])

# ---- 5. quadratic and Lagrange elements --------------------------------------------

notebook("brakke_lagrange", "Brakke's samples: quadratic and Lagrange elements", """
Curved elements: the quadratic model (edges with midpoints) and Lagrange elements of
higher order, a short final stage on a mesh settled in the linear model. `relax()` in
those models first does what the manual recommends, five gradient steps and then
Newton steps.""", [
    md("""
## `quad.fe`: a skew quadrilateral

A film on a skew quadrilateral wire in the quadratic model."""),
    code('''print(bh.recipe("quad.fe"))
runs = bh.compare("quad.fe")
bh.show(runs)'''),
    md("""
## `qmound.fe`: the drop on a table, quadratic

`mound.fe` in the quadratic model, its volume a named quantity."""),
    code('''print(bh.recipe("qmound.fe"))
runs = bh.compare("qmound.fe")
bh.show(runs)'''),
    md("""
## The Lagrange ladders of `cube.fe` and `catman.fe`

`cube.fe`'s `gogo2` refines twice in the linear model, then switches to Lagrange
elements of order 2, 4 and 6 with `g 5; hessian` at each. The same with `relax()`:
relax with two refinements, then for each order `ev.set_model("lagrange", n)` and
relax."""),
    code('''import time

def ladder(name, orders, levels=2, setup=None):
    ev = pyse.Evolver(name)
    if setup:
        setup(ev)
    t = time.perf_counter()
    ev.relax(levels=levels)
    for n in orders:
        ev.set_model("lagrange", n)
        r = ev.relax()
    return bh.Run(f"relax ladder, orders {orders}", ev.total_energy, r.residual,
                  ev.counts["facets"], time.perf_counter() - t, r.converged, r.stable, [],
                  ev.mesh())

print(bh.recipe("cube.fe", "gogo2"))
cube = [bh.run_recipe("cube.fe", "gogo2", label="recipe: gogo2"), ladder("cube.fe", (2, 4, 6))]
bh.table(cube, reference=(36*np.pi)**(1/3))'''),
    md("""
At order 6 both match the sphere's area to rounding error (the recipe's own check
prints an area error of about 1e-15): sixth-order elements on this mesh represent the
sphere essentially exactly.

`catman.fe`'s `gogo3` goes up to order 8. It sets `rmax := cosh(height)` and compares
with the area of a catenoid through rings of radius $\\cosh(0.55)$, but `rmax` is a
parameter of `cat.fe`: in `catman.fe` the rings keep radius 1, so its printed
"Difference from true area" compares with the wrong catenoid. The exact catenoid for
rings of radius 1 at $\\pm 0.55$ has area 6.5116735 (from $a \\cosh(0.55/a) = 1$)."""),
    code('''lo, hi = 0.6, 1.0                   # the larger root of a cosh(0.55/a) = 1: the stable catenoid
for _ in range(100):
    a = (lo + hi)/2
    lo, hi = (lo, a) if a*np.cosh(0.55/a) > 1 else (a, hi)
exact = np.pi*a*(2*0.55 + a*np.sinh(2*0.55/a))
print(f"exact catenoid area {exact:.10f}")
catman = [bh.run_recipe("catman.fe", "gogo3", label="recipe: gogo3"),
          ladder("catman.fe", (2, 4, 6, 8))]
bh.table(catman, reference=exact)
bh.show(catman)'''),
])
print(f"written to {os.path.normpath(DOCS)}")
