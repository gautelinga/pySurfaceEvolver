"""8. The drainage band at ell = 1.1 (docs/slit_drainage.ipynb): beads in a
slit drained from immersed, through the bead emergence, along the band to its
snap at the mirror y = 0. Geometry and events from docs/drainage_helpers.py,
run as the notebook does: ev.remesh(target=h) before each band step's relax()
(h: the band's first median edge length). Reference: the stored finer-mesh run
(docs/_static/drainage_runs.npz, "l1.1": level 3, snap at 0.00707 per eighth
cell). Refinement level 2; band steps of 3% (as the notebook's live run).

The one case that isn't plain relax(): on defaults alone the receding contact
line on the bead squeezes the free vertices behind it into slivers at the
mid-plane (from the first band step), nearly every relax ends unconverged and
single runs scatter from 3% to over 50% at worst. Merging or moving the
squeezed vertices inside relax() did not help over repeated runs (October 2026;
docs/relaxing.md). Checked here: the remeshing workflow on a contact line
receding over a curved solid: the median band-pressure error, the snap, and
the share of band steps that converge."""

import os
import sys

import numpy as np

from common import Result, Tracker, rel

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")
sys.path.insert(0, ROOT)
import drainage_helpers as dh       # noqa: E402

NAME = "8 drainage band, ell = 1.1, remeshed"
MEDIAN_LIMIT = 1e-3       # the band pressure, median over the states
CONVERGED_SHARE = 0.75    # band steps whose relax() converges, at least


class TrackedCell(dh.Cell):
    """The helper's cell, its relaxations recorded (remeshing as the notebook)."""

    def __init__(self, track, *args, **kw):
        super().__init__(*args, **kw)
        self.track = track
        self.band_calls = []

    def relax(self, ev, remesh=False):
        if remesh:
            if self.edge_h is None:
                self.edge_h = ev.mesh_quality().edge_median
            ev.remesh(target=self.edge_h)
        r = self.track.relax(ev, stability=False)
        if remesh:
            self.band_calls.append(bool(r.converged))


def run(quick=False):
    track = Tracker()
    cell = TrackedCell(track, 1.1)
    cell.band_step = 0.03
    res = Result(NAME, False, "")
    ref = np.load(os.path.join(ROOT, "_static", "drainage_runs.npz"))
    rv, rp, rs = ref["l1.1_volume"], ref["l1.1_pressure"], ref["l1.1_stage"]
    band_v, band_p = rv[rs == 1], rp[rs == 1]
    order = np.argsort(band_v)
    snap_ref = float(ref["l1.1_events"][1])
    try:
        states, events = dh.drain(cell, cell.meniscus_exact(0.55)[0], 0.004, levels=2)
    except Exception as e:
        res.summary = f"CRASHED in the drainage: {type(e).__name__}: {str(e)[:150]}"
        return res
    errs = []
    for s in states:
        if s["stage"] != 1 or not (band_v.min() < s["volume"] < 0.08):
            continue
        p_ref = np.interp(s["volume"], band_v[order], band_p[order])
        e = rel(s["pressure"], p_ref)
        errs.append(e)
        res.rows.append(dict(volume=s["volume"], pressure=s["pressure"], pressure_ref=p_ref, error=e))
    snap = [e["volume"] for e in events if e["kind"] == "snap"]
    snap_err = rel(snap[0], snap_ref) if snap else float("inf")
    # the last state is the band at the snap; the surface itself is gone with the
    # helper's local Evolver, so the tracker reports on the engine's current one
    import pysurfaceevolver as pyse
    track.finish(pyse.Evolver(), res, eigen=False)
    median = float(np.median(errs)) if errs else float("inf")
    worst = max(errs) if errs else float("inf")
    converged = (sum(cell.band_calls)/len(cell.band_calls)) if cell.band_calls else 0.0
    res.error = max(median, snap_err)
    res.passed = median < MEDIAN_LIMIT and snap_err < 0.02 and converged >= CONVERGED_SHARE
    res.summary = (f"band pressure against the stored level-3 curve: median {median:.3%}, worst "
                   f"{worst:.2%} over {len(errs)} states; snap at {snap[0] if snap else None} "
                   f"(stored {snap_ref:.5f}, {snap_err:.1%} off); {converged:.0%} of the "
                   f"{len(cell.band_calls)} band steps converged")
    return res
