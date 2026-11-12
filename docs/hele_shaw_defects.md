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

<!-- results go here -->

## Notes and limits

<!-- notes go here -->

## References

* R. Holtzman, M. Dentz, M. Moura, M. Chubynsky, R. Planet and J. Ortín, Emergence of
  dissipation and hysteresis from interactions among reversible, non-dissipative
  units: the case of fluid-fluid interfaces, *Phys. Rev. Fluids* 9 (2024) 064001,
  [doi:10.1103/PhysRevFluids.9.064001](https://doi.org/10.1103/PhysRevFluids.9.064001),
  [arXiv:2401.13077](https://arxiv.org/abs/2401.13077).
