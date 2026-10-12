"""The gradient step's merged passes over the elements.

With named quantities only (the Lagrange model, or `convert_to_quantities`)
a gradient step takes the energy gradients in the same pass as the volume
gradients, and each trial move evaluates the energy in its last volume
projection pass. The energy a step reports must be what a fresh evaluation
gives, and the steps must follow the trajectory of the separate passes
(reference values from before the merge; the linear model, which keeps its
passes, as a control).
"""
import pytest


def _step_and_check(ev, n):
    energies = []
    for _ in range(n):
        ev.command("g 1")
        energy, area = ev.total_energy, ev.eval("total_area")
        volume = ev.eval("body[1].volume")
        ev.command("recalc")
        assert ev.total_energy == pytest.approx(energy, rel=1e-13, abs=0)
        assert ev.eval("total_area") == pytest.approx(area, rel=1e-13, abs=0)
        assert ev.eval("body[1].volume") == pytest.approx(volume, rel=1e-13, abs=0)
        energies.append(energy)
    return energies


CASES = {
    "lagrange 2": ("r; r; g 5; lagrange 2", 8, [
        4.91059894538056, 4.90292718449706, 4.89726013739729, 4.8922730452867,
        4.88781032832514, 4.88372032068658, 4.87999440099702, 4.87655999422211]),
    # large enough for the parallel element loops
    "lagrange 2, 24k facets": ("r; r; r; r; g 5; r; lagrange 2", 4, [
        5.56326688128116, 5.55982298056169, 5.5573550670638, 5.55532543973151]),
    "quantities, linear": ("convert_to_quantities; r; r; r", 10, [
        5.61982355899099, 5.47445963095805, 5.37960148563045, 5.31015819246852,
        5.25464058710434, 5.20944726544523, 5.17104249185636, 5.13836231815958,
        5.10988653407386, 5.08513369697065]),
    # fixed scale: one move per step, large enough to need projecting
    "lagrange 3, fixed scale": ("r; r; g 5; lagrange 3; m 0.05", 4, [
        4.92692672594599, 4.92195116930647, 4.91844039658259, 4.91571301501855]),
    "linear": ("r; r; r; g 5", 10, [
        5.20944726544523, 5.17104249185636, 5.13836231815958, 5.10988653407386,
        5.08513369697065, 5.06337002138641, 5.04424153993844, 5.02737961105931,
        5.01235480482593, 4.99903819152524]),
    "linear, fixed scale": ("r; r; r; g 5; m 0.3", 4, [
        5.22438972905785, 5.32766687663722, 5.64715089057768, 5.87752131862583]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_steps_report_the_fresh_energy_and_follow_the_reference(cube, case):
    setup, n, reference = CASES[case]
    cube.command(setup)
    energies = _step_and_check(cube, n)
    assert energies == pytest.approx(reference, rel=1e-11, abs=0)

