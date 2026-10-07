"""Surface Evolver's sample datafiles and command scripts, installed with
the package.

The directory is also appended to ``EVOLVERPATH`` when the package is
imported, so ``Evolver("cube.fe")`` and commands like ``read "obj.cmd"``
find these files from any working directory (files in the working
directory, and directories already on ``EVOLVERPATH``, come first).

>>> from pysurfaceevolver import Evolver, examples
>>> examples.names()[:3]
['100grain.fe', 'addload_example.fe', 'cat.fe']
>>> ev = Evolver(examples.path("cube.fe"))
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

__all__ = ["directory", "names", "path"]


def directory() -> Path:
    """The directory holding the sample files."""
    return Path(__file__).resolve().parent / "data" / "fe"


def names(pattern: str = "*.fe") -> List[str]:
    """Names of the sample files matching a glob pattern (datafiles by default;
    ``"*.cmd"`` lists the command scripts)."""
    return sorted(p.name for p in directory().glob(pattern))


def path(name: str) -> str:
    """Full path of a sample file, e.g. ``path("cube.fe")``."""
    p = directory() / name
    if not p.exists():
        raise FileNotFoundError(f"no sample file {name!r}; see examples.names()")
    return str(p)


def _add_to_evolverpath() -> None:
    d = str(directory())
    if not os.path.isdir(d):
        return
    entries = [e for e in os.environ.get("EVOLVERPATH", "").split(os.pathsep) if e]
    if d not in entries:
        os.environ["EVOLVERPATH"] = os.pathsep.join(entries + [d])
