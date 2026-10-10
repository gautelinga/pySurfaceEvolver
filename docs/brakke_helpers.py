"""Helpers for the notebooks on Brakke's sample datafiles: run a sample's own
recipe (its ``gogo`` command, or the usage its comments describe) and a
default ``ev.relax()`` side by side, tabulate both, and draw the surfaces.
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from html import escape
from typing import Callable, List, Optional, Sequence

import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import examples


@dataclass
class Run:
    """What one way of evolving a sample ended with."""
    label: str
    energy: float
    residual: float
    facets: int
    seconds: float
    converged: Optional[bool] = None       # relax() only
    stable: Optional[bool] = None
    issues: List[str] = field(default_factory=list)
    mesh: Optional[pyse.Mesh] = None


def recipe(name: str, macro: str = "gogo") -> str:
    """The text of a command macro defined in a sample datafile."""
    text = re.sub(r"//[^\n]*", "", open(examples.path(name)).read())
    found = re.search(rf"^\s*{macro}\s*:=\s*\{{", text, re.M)
    if not found:
        raise ValueError(f"{name} defines no {macro}")
    depth, start = 1, found.end()
    for i in range(start, len(text)):         # to the matching brace
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return " ".join(text[start:i].split())
    raise ValueError(f"{name}: {macro} is not closed")


def refinements(commands: str) -> int:
    """How many times a command string refines the whole surface ('r')."""
    return len(re.findall(r"(?<![\w.])r(?![\w.])", commands))


def _load(name: str, setup: Optional[Callable] = None) -> pyse.Evolver:
    ev = pyse.Evolver(examples.path(name))
    if setup is not None:
        setup(ev)
    return ev


def _health(ev: pyse.Evolver) -> List[str]:
    try:
        return list(ev.health().issues)
    except pyse.EvolverError:
        return []


def run_recipe(name: str, commands: str = "gogo", setup: Optional[Callable] = None,
               label: Optional[str] = None) -> Run:
    """Load a sample and run Evolver commands on it (by default its gogo)."""
    ev = _load(name, setup)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        ev.command(commands)
        seconds = time.perf_counter() - t
    issues = [i for i in _health(ev) if not i.startswith("not in equilibrium")]
    return Run(label or f"recipe: {commands}", ev.total_energy, ev.residual(),
               ev.counts["facets"], seconds, issues=issues, mesh=ev.mesh())


def run_relax(name: str, levels: int = 0, setup: Optional[Callable] = None,
              label: Optional[str] = None, **kwargs) -> Run:
    """Load a sample and relax it with defaults (and ``levels`` refinements)."""
    ev = _load(name, setup)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        r = ev.relax(levels=levels, **kwargs)
        seconds = time.perf_counter() - t
    extra = f", {r.popped} popped" if r.popped else ""
    extra += f", {r.collapsed} collapsed edges" if r.collapsed else ""
    return Run(label or f"relax(levels={levels}){extra}", ev.total_energy, r.residual,
               ev.counts["facets"], seconds, r.converged, r.stable,
               [i for i in (r.health.issues if r.health else [])
                if not i.startswith("not in equilibrium")], ev.mesh())


def compare(name: str, commands: str = "gogo", levels: Optional[int] = None,
            setup: Optional[Callable] = None, **kwargs) -> List[Run]:
    """The sample's recipe against a default relax() with as many
    refinements; shows the table and returns both runs."""
    if commands == "gogo":
        commands_text = recipe(name)
    else:
        commands_text = commands
    if levels is None:
        levels = refinements(commands_text)
    runs = [run_recipe(name, commands, setup, label=f"recipe: {commands_text}"),
            run_relax(name, levels, setup, **kwargs)]
    table(runs)
    return runs


def _fmt(x: Optional[float], spec: str) -> str:
    return "" if x is None else format(x, spec)


def table(runs: Sequence[Run], reference: Optional[float] = None) -> None:
    """Show runs as a table (and their error against a reference energy)."""
    from IPython.display import HTML, display
    head = ["", "energy"] + (["error"] if reference is not None else []) + \
        ["residual", "facets", "seconds", "converged", "stable", "notes"]
    rows = []
    for r in runs:
        cells = [escape(r.label), f"{r.energy:.10g}"]
        if reference is not None:
            cells.append(f"{r.energy/reference - 1:+.2e}")
        cells += [f"{r.residual:.1e}", str(r.facets), f"{r.seconds:.2f}",
                  "" if r.converged is None else ("yes" if r.converged else "no"),
                  "" if r.stable is None else ("yes" if r.stable else "no"),
                  escape("; ".join(r.issues))]
        rows.append(cells)
    style = 'style="text-align:left; padding:2px 8px"'
    html = "<table><tr>" + "".join(f"<th {style}>{h}</th>" for h in head) + "</tr>"
    for cells in rows:
        html += "<tr>" + "".join(f"<td {style}>{c}</td>" for c in cells) + "</tr>"
    display(HTML(html + "</table>"))


def show(runs: Sequence[Run], mirror=None, view: Optional[str] = None, **kwargs) -> None:
    """Draw the surfaces the runs ended with, side by side (strings as lines)."""
    import pyvista as pv
    meshes = [r.mesh for r in runs if r.mesh is not None]
    if not meshes or meshes[0].vertices.shape[1] > 3:
        return
    p = pv.Plotter(shape=(1, len(meshes)), window_size=(380*len(meshes), 340))
    for k, (run, m) in enumerate(zip(runs, meshes)):
        p.subplot(0, k)
        if m.facets is not None and len(m.facets):
            poly = (m.mirrored(mirror) if mirror else m).to_pyvista()
            p.add_mesh(poly, color=kwargs.get("color", "lightblue"),
                       show_edges=kwargs.get("show_edges", True), edge_color="#3a6f99")
        else:
            pts = np.zeros((len(m.vertices), 3))
            pts[:, :m.vertices.shape[1]] = m.vertices
            lines = np.hstack([np.full((len(m.edges), 1), 2), m.edges]).ravel()
            p.add_mesh(pv.PolyData(pts, lines=lines), color="#1f4e79", line_width=2)
        p.add_text(run.label if len(run.label) < 40 else run.label[:37] + "...",
                   font_size=8, color="black")
        if view == "xy":
            p.view_xy()
        elif view is not None:
            getattr(p, f"view_{view}")()
    p.link_views()
    p.show()
