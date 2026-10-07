"""Every sample datafile in fe/ loads and iterates."""

import numpy as np
import pytest

from conftest import FE_DIR

# Sample datafiles expected to fail, with the reason (none at the moment).
KNOWN_BAD: dict = {}

DATAFILES = [
    pytest.param(
        p.name,
        marks=pytest.mark.xfail(reason=KNOWN_BAD[p.name], strict=True)
        if p.name in KNOWN_BAD else (),
    )
    for p in sorted(FE_DIR.glob("*.fe"))
]


@pytest.mark.filterwarnings("ignore::pysurfaceevolver.EvolverWarning")
@pytest.mark.parametrize("datafile", DATAFILES)
def test_datafile_loads_and_iterates(load, datafile):
    ev = load(datafile)
    assert ev.counts["vertices"] > 0
    if ev.representation != "simplex":  # the simplex model has no edges
        assert ev.counts["edges"] > 0

    ev.command("g 3")
    assert np.isfinite(ev.total_energy)
    counts = ev.counts  # some datafiles change topology while iterating

    m = ev.mesh()
    assert m.vertices.shape == (counts["vertices"], ev.sdim)
    assert np.isfinite(m.vertices).all()
    assert m.edges.shape == (counts["edges"], 2)
    assert (m.edges < len(m.vertices)).all()
    if ev.representation == "soapfilm":
        assert m.faces.shape == (counts["facets"], 3)
        assert m.faces.max() < len(m.vertices)
    else:
        assert m.faces is None

    b = ev.bodies()
    assert len(b.ids) == counts["bodies"]
    assert np.isfinite(b.volume).all()
