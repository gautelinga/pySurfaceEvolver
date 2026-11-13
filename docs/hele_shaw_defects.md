# Defects in a Hele-Shaw cell: when an interface jumps

An interface pushed slowly through a rough fracture or a porous medium does not move
smoothly: it sticks where the walls hold it and then jumps, and each jump turns some
of the work done on it into heat. That is where the hysteresis of capillary pressure
against saturation comes from. Holtzman et al. built the simplest case. In a quasi-2D
**Hele-Shaw cell**, two plates 3.6 mm apart, they printed **defects**, places 10 mm
wide where the gap narrows, and pushed silicone oil in and out of the cell by its
pressure. Each defect alone is **weak**: the interface bulges over it and back without
a jump. Their model predicts that two weak defects closer than a critical separation
act together as a **strong** one, which the interface crosses in jumps, so dissipation
emerges from units that dissipate nothing on their own. In their cell one defect
gave no hysteresis, a pair 13 mm apart (centre to centre) a loop, and a pair 11 mm
apart a wider one.

Their model treats the interface as a line in the plane of the cell. The meniscus
spans the gap, though, and where the gap narrows it does so steeply. This page builds
the cell in 3D, its upper plate a curved wall, and follows the three cases of the
experiment with pySE: the meniscus's shape across the gap, its contact lines on both
plates and gravity in the tilted cell, through imbibition and drainage.

The cell is in `docs/hele_shaw_helpers.py`, and `tools/hele_shaw_figure.py` runs it
and draws the pictures.

## The upper plate as a curved wall

The defects come from Appendix D of the paper. Along a defect's ramp, 10 mm long, the
gap b narrows so that the capillary pressure 2γ/b rises linearly by 60 Pa. A plateau
5 mm long follows at the narrowest gap, 0.56 mm, and then the gap opens back to 3.6 mm.
The printed edges are smoothed over 0.25 mm here. The upper plate is the surface
z = b(x, y), a formula in x and y, and {func}`~pysurfaceevolver.constraints.heightfield`
makes it a wall. The wetted area, the oil's volume under the plate and its weight there
have no closed form as integrals along the contact line, so `heightfield` tabulates
them over a rectangle and evaluates them in C as cubic splines, derivatives included.
The lower plate is the plane z = 0. The oil (contact angle 20°) fills the cell up to
the meniscus, and mirror planes at x = 0 and x = X stand for the rest of the cell.

Without gravity, a meniscus between parallel plates is a circular arc of radius
b/(2 cos θ). That gives a check of the curved wall. Here the defect spans the cell's
whole width, a channel 2 mm wide, and a flat meniscus is relaxed at y = -4, where the
gap is the open 3.6 mm, and at y = 12, on the plateau. The second start has the ramp
between the plates' integrals' first row (y = -12) and the meniscus, so its volume
checks the tables along the ramp:

```python
import numpy as np
from hele_shaw_helpers import cell, contact_line, gap, meniscus

for y0 in (-4.0, 12.0):                          # the open cell; the plateau
    ev = cell(None, y0, width=2.0, nx=5, gravity=False)
    y = np.linspace(-12, y0, 200001)
    b = gap(y)
    exact = 2.0*np.sum((b[1:] + b[:-1])/2*np.diff(y))     # the oil up to y0
    print(f"start at y = {y0:g}: gap {b[-1]:.4f}, volume {ev.body(1).volume:.4f} "
          f"(exact {exact:.4f})")
    ev.relax(levels=2)
    p, dry = meniscus(b[-1])
    print(f"  relaxed: pressure {ev.body(1).pressure:.4f} (exact {p:.4f}), "
          f"contact line at y = {contact_line(ev):.3f} (exact {y0 + dry/b[-1]:.3f})")
```

```text
start at y = -4: gap 3.6000, volume 57.6000 (exact 57.6000)
  relaxed: pressure -0.5225 (exact -0.5221), contact line at y = -3.086 (exact -3.082)
start at y = 12: gap 0.5625, volume 113.0703 (exact 113.0703)
  relaxed: pressure -3.3414 (exact -3.3411), contact line at y = 12.140 (exact 12.143)
```

The pressures agree to under a thousandth and the contact lines to 0.004 mm, the
error of a mesh with 24 edges across the gap. The volume agrees to every digit shown, which checks
the tables' integrals along the ramp.

## The three cells

Each run follows the experiment's cycle. A flat meniscus starts 4 mm below the defects,
and oil goes in until the meniscus far from them is 13.5 mm up (the defects end at 15);
then it is drawn back out to the start. The runs control the volume, not the
pressure: `recipes.arclength` steps the oil's volume by 12 mm³, halved at most twice
where a step finds no equilibrium, and makes the changes of topology on the way
(contact lines meeting, dry patches opening on the plate). Mirror planes bound each
cell, at x = 0 and at x = 10 for one defect, x = 14 for the pairs. The three runs:

```text
OMP_NUM_THREADS=3 python tools/hele_shaw_figure.py run single 0   --rounding 0.25 --ds 12 --halvings 2 --width 10
OMP_NUM_THREADS=3 python tools/hele_shaw_figure.py run pair13 6.5 --rounding 0.25 --ds 12 --halvings 2 --width 14
OMP_NUM_THREADS=3 python tools/hele_shaw_figure.py run pair11 5.5 --rounding 0.25 --ds 12 --halvings 2 --width 14
python tools/hele_shaw_figure.py draw
```

![The three cells through imbibition and drainage](_static/hele_shaw.png)

The top row follows the paper's figures: the bulge of the meniscus over the defect, η<sub>m</sub>
(the meniscus at the middle of the defect less the meniscus far from it), against
where the far meniscus is, h<sub>f</sub>. The bottom row shows the oil's pressure
against its volume, less the pressure of the meniscus undisturbed: a line in the
volume fitted while the defects are still far (the first 100 mm³), which takes out
gravity's steady rise. Under volume control the path is continuous. The experiment
controls the pressure, and there the interface jumps wherever the pressure would have
to fall: the dashed segments. Each jump dissipates the area it cuts off under the
pressure curve:

| cell | jumps (imbibition + drainage) | dissipated over the cycle |
|---|---|---|
| one defect | 18 + 12 | 0.33 γ mm² (6.6 nJ) |
| two, 13 mm apart | 29 + 12 | 0.41 γ mm² (8.2 nJ) |
| two, 11 mm apart | 43 + 13 | 0.43 γ mm² (8.6 nJ) |

All three cells share the main shape. The bulge jumps to about 1 mm as the meniscus
reaches the defect's foot (h<sub>f</sub> ≈ −1). It climbs steeply as the meniscus
rises up the ramp (h<sub>f</sub> 3 to 5). It sits at 5.5–6.5 mm while the meniscus
crosses the plateau, and it relaxes once the far meniscus passes the defect's end.
Drainage runs above imbibition across the plateau, which is the loop. The dissipation
grows from one defect to the pair 13 mm apart to the pair 11 mm apart, as in the
experiment. The pairs' loops are not much wider than the single defect's, though, and
the single defect is not reversible here: most of its jumps come on the ramp, where
the bulge grows fastest.

Run times on 3 threads of a cluster node: 39 minutes for one defect and 88 for the pair
13 mm apart (63 in an earlier run of the same cell). The pair 11 mm apart took 3 h 50
min in two jobs: the run was resumed from a saved state (`--states`, `--resume`)
when the first job's 3 h ran out late in drainage.

## Notes and limits

* **Rounded edges.** The printed edges are rounded over 0.25 mm. A printed part has
  no sharp edges, and edges rounded over 0.1 mm made the runs churn for 20 minutes
  and more at each defect's side edges, where the oil wicks along the step's concave
  foot. With 0.25 mm rounding, the 1 mm channel between the defects 11 mm apart is a
  rounded groove whose roof is 3.5 mm, not 3.6.
* **Narrow cells.** Mirror planes 10 and 14 mm from the middle leave the "far"
  meniscus only 2.5 to 5 mm beyond the defects. A cell 20 mm wide gave four times the
  facets, and an estimated 2 h a cycle for one defect.
* **No contact-angle hysteresis.** The contact angle is 20° on both plates,
  advancing or receding. All the dissipation here comes from the meniscus's jumps.
* **The groove's end.** Where the 1 mm channel opens past the defects' rounded
  corners, the oil's contact line has to wrap round a corner. Each dry patch the
  descent made there moved the line by a whole edge, the line slid back, and the
  same spot was dried and wetted 52 times; the run never got past V ≈ 510 mm³. pySE
  now makes the mesh finer where a contact line on a heightfield swings like this,
  and the run goes through.
* **A remaining slow spot.** Where the meniscus first enters the 1 mm channel between
  the defects 11 mm apart (V ≈ 80 mm³), the surface crowds the mirror plane at the
  channel's middle. It takes about 13 minutes there in imbibition and 30 in drainage.

## References

* R. Holtzman, M. Dentz, M. Moura, M. Chubynsky, R. Planet and J. Ortín, Emergence of
  dissipation and hysteresis from interactions among reversible, non-dissipative
  units: the case of fluid-fluid interfaces, *Phys. Rev. Fluids* 9 (2024) 064001,
  [doi:10.1103/PhysRevFluids.9.064001](https://doi.org/10.1103/PhysRevFluids.9.064001),
  [arXiv:2401.13077](https://arxiv.org/abs/2401.13077).
