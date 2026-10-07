"""Python frontend to Ken Brakke's Surface Evolver.

>>> from pysurfaceevolver import Evolver
>>> ev = Evolver("cube.fe")          # sample datafiles are found anywhere
>>> ev.iterate(5); ev.refine(); ev.iterate(5)
>>> ev.eval("body[1].volume")
>>> ev.values("vertex", "x^2 + y^2")   # one value per vertex
>>> mesh = ev.mesh()                   # NumPy arrays: mesh.vertices, mesh.faces, ...
>>> ev.write("surface.vtu")            # any meshio format
>>> ev.plot("area")                    # PyVista
"""

from ._build import Body, make_datafile
from ._evolver import (
    Evolver,
    EvolverError,
    EvolverExit,
    EvolverFatalError,
    EvolverWarning,
    InvalidSurfaceError,
    IterationResult,
    Parameters,
    Snapshot,
)
from ._mesh import Bodies, BodySurface, Mesh, Quantity, is_watertight
from ._viz import LiveView
from . import examples

examples._add_to_evolverpath()

__all__ = [
    "Bodies",
    "Body",
    "BodySurface",
    "Evolver",
    "EvolverError",
    "EvolverExit",
    "EvolverFatalError",
    "EvolverWarning",
    "InvalidSurfaceError",
    "IterationResult",
    "LiveView",
    "Mesh",
    "Parameters",
    "Quantity",
    "Snapshot",
    "is_watertight",
    "examples",
    "make_datafile",
]
__version__ = "0.3.0"
