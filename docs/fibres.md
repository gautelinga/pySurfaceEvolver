# Drops on fibres

A drop on a fibre either wraps all the way round it, a **barrel**, or sits on one
side, a **clam shell**. Between two fibres, liquid either fills the narrow space along
them as a **column** or gathers into a drop; where two fibres cross, it spreads into a
column along the narrow gap on both sides of the crossing. Which shape the liquid takes
decides how wet fibrous materials (filters, textiles, paper) hold and lose it. This
page computes the three cases with pySE, on fibres of radius 1 (surface tension 1, no
gravity), and checks them against exact references where there are some:

* where a barrel rolls up into a clam shell: an exact onset from the second variation
  of its energy, and how the two shapes' energies compare,
* the column between two parallel fibres: Princen's cross-section, and the widest gap
  that holds one,
* the column at two crossed fibres: its length against the model of Sauret et al., and
  how far it goes.

The starting surfaces and the references are in `docs/fibres_helpers.py`. Each example
takes about a minute.

## One fibre: where a barrel rolls up

A barrel is a surface of revolution around the fibre; its profile is known in closed
form (Carroll; `barrel` integrates it). Shrinking a barrel at a fixed contact angle,
it becomes unstable at some volume to shifting to one side, and rolls up into a clam
shell. McHale, Newton and Carroll related its stability to an inflexion in its profile.
The second variation of the energy makes this exact for that motion. Shifting the
drop sideways moves its free surface rigidly, so its normal part solves the Jacobi
equation. The contact line adds a condition for a motion that slides it along the
fibre at a fixed angle: u' = q u, with q the meridian's curvature there times
cot(theta) (the fibre itself is straight along its axis; sliding the drop along the
fibre, an exact symmetry, meets the condition). The sideways shift meets it only when
the meridian's curvature vanishes at the contact line. There the shift is a neutral
mode, the lowest of its kind, and at any smaller volume it lowers the energy. So the
barrel turns unstable as the inflexion of its profile reaches the contact line, where
its pressure equals cos(theta) and its equator radius is (1 + sin theta)/cos theta.
`barrel_onset` integrates the profile to there for the volume.

`barrel_drop` relaxes a barrel twice refined. Shrinking it through the onset in steps
of 5%, the lowest eigenvalue whose mode shifts the drop sideways (`shift`) changes
sign:

```python
import numpy as np
from pysurfaceevolver import recipes
from fibres_helpers import barrel_drop, barrel_onset, shift

for theta in (30, 45, 60):
    exact = barrel_onset(theta)
    volumes = exact["volume"]*np.arange(1.3, 0.79, -0.05)    # shrinking the drop
    ev = barrel_drop(theta, volumes[0])
    for step in recipes.continuation(ev, ev.body(1), volumes):
        now = (step.value, shift(ev), step.pressures[0])      # sideways
        if now[1] < 0:                                        # it would roll up
            break
        last = now
    w = last[1]/(last[1] - now[1])                            # where the eigenvalue is zero
    onset, pressure = last[0] + w*(now[0] - last[0]), last[2] + w*(now[2] - last[2])
    print(f"contact angle {theta}: rolls up below volume {onset:.2f} (exact {exact['volume']:.2f}, "
          f"{onset/exact['volume'] - 1:+.1%}), pressure {pressure:.3f} "
          f"(cos theta {np.cos(np.radians(theta)):.3f})")
```

```text
contact angle 30: rolls up below volume 17.71 (exact 17.83, -0.7%), pressure 0.868 (cos theta 0.866)
contact angle 45: rolls up below volume 54.99 (exact 55.36, -0.7%), pressure 0.709 (cos theta 0.707)
contact angle 60: rolls up below volume 212.02 (exact 214.74, -1.3%), pressure 0.502 (cos theta 0.500)
```

The onset comes out about 1% low, from the mesh (refined once instead of twice, 4% low
at 60 degrees), and the pressure there within 0.5% of cos(theta). The sideways shift is
a pair of modes, one for each direction across the fibre; sliding along the fibre is an
exact zero mode, which `shift` leaves aside. No other mode turns unstable first.

## The clam shell

`clamshell` starts a drop as a ball touching the fibre on a small patch and relaxes
it; it wets one side of the fibre. Its energy against the barrel's (exact) at the same
volume, at 30 degrees, on two meshes:

```python
import numpy as np
from pysurfaceevolver import recipes
from fibres_helpers import barrel, barrel_onset, clamshell

theta = 30
print(f"the barrel is unstable below volume {barrel_onset(theta)['volume']:.1f}")
near = {}                                       # around the crossover, on each mesh
for levels, volumes in ((1, [30, 35, 40]), (2, [10, 15, 20, 25, 30, 35, 40, 50]), (3, [30, 35, 40])):
    ev = clamshell(theta, volumes[0], levels=levels)
    rows = []
    for step in recipes.continuation(ev, ev.body(1), volumes):
        rows.append((step.value, ev.energy, barrel(theta, step.value)["energy"], step.result.stable))
    near[levels] = np.array([[clam - bar for v, clam, bar, _ in rows if v == w][0]
                             for w in (30, 35, 40)])
    if levels == 1:
        continue
    print(f"{ev.counts['facets']} facets:")
    for v, clam, bar, stable in rows:
        print(f"  volume {v:2.0f}: clam shell {clam:7.3f} ({'stable' if stable else 'unstable'}), "
              f"barrel {bar:7.3f}")
    print(f"  equal energies at volume {np.interp(0, near[levels], [30, 35, 40]):.1f}")
# extrapolated to no mesh size (Richardson), at the order the three meshes show
order = np.log2((near[1] - near[2])/(near[2] - near[3]))
limit = near[3] + (near[3] - near[2])/(2**order - 1)
print(f"order of convergence {order.min():.2f} to {order.max():.2f}; extrapolated, "
      f"equal energies at volume {np.interp(0, limit, [30, 35, 40]):.1f}")
```

```text
the barrel is unstable below volume 17.8
5868 facets:
  volume 10: clam shell  12.339 (stable), barrel  12.844
  volume 15: clam shell  17.115 (stable), barrel  17.525
  volume 20: clam shell  21.570 (stable), barrel  21.877
  volume 25: clam shell  25.793 (stable), barrel  25.993
  volume 30: clam shell  29.835 (stable), barrel  29.928
  volume 35: clam shell  33.729 (stable), barrel  33.716
  volume 40: clam shell  37.496 (stable), barrel  37.379
  volume 50: clam shell  44.721 (stable), barrel  44.399
  equal energies at volume 34.4
23464 facets:
  volume 30: clam shell  29.786 (stable), barrel  29.928
  volume 35: clam shell  33.673 (stable), barrel  33.716
  volume 40: clam shell  37.435 (stable), barrel  37.379
  equal energies at volume 37.2
order of convergence 1.56 to 1.60; extrapolated, equal energies at volume 38.7
```

Small drops sit best on one side, large ones round the fibre. The two energies differ
by a fraction of a percent near the crossover, so it moves with the mesh: from 34.4 to
37.2 here, the clam shell's energy still falling with refinement (the barrel's is
exact). Its differences from mesh to mesh shrink at an order of about 1.6, a little
below the second order of a smooth surface (the contact line on the curved fibre), and
extrapolated to no mesh size the two shapes have equal energies at a volume of about
39. Between the barrel's onset and the crossover both shapes are stable: a barrel
shrunk below the crossover stays a barrel until its onset, and a clam shell grown past
it stays a clam shell. Chou et al. mapped this coexistence with Surface Evolver, and
Eral et al. switched drops between the two shapes with electrowetting. Every barrel with
the lower energy is stable, as McHale and Newton found. Large clam shells wrap further
round the fibre (most of the way at a volume of 200); where the liquid would close round
it, the clam shell has to become a barrel, a change of topology the mesh can't follow.

## Two parallel fibres: the column

Between two parallel fibres a gap 2d apart, a long column has a uniform middle. The
force along the fibres through a cross-section is the same everywhere along the column
(the fibres push on the liquid only at right angles to themselves) and zero past its
ends. So in the middle the energy per length equals the pressure times the area
(Princen): the cross-section with the least energy per volume. `princen` solves for it:
two circular menisci meeting the fibres at the contact angle. The wider the gap, the
further the liquid wraps round the fibres; for a perfectly wetting liquid it reaches all
the way round at d = sqrt(2), and at wider gaps there is no column, only drops (the
transition Protière et al. studied).

`column_slice` computes a slice of such a middle: a quarter of the cross-section cut
by mirror planes, between two more mirror planes across the fibres, so that it repeats
along them. Its volume sets the area of the cross-section; `column_excess` relaxes it
and returns its pressure minus its energy per volume, zero for Princen's column. Here
for a perfectly wetting liquid:

```python
from fibres_helpers import column_area, column_excess, column_slice, princen

for d in (0.5, 0.75, 1.0, 1.3):
    ev = column_slice(d, 0.0)
    b = column_area(ev)                        # the flat start's
    gb = column_excess(ev, b)
    a, ga = b, gb
    while ga*gb > 0:                           # 10% steps until the balance changes sign
        a, ga = b, gb
        b *= 0.9 if gb > 0 else 1.1
        gb = column_excess(ev, b)
    for _ in range(5):                         # then home in on it, keeping it bracketed
        c = b - gb*(b - a)/(gb - ga)
        gc = column_excess(ev, c)
        a, ga, b, gb = (a, ga, c, gc) if gc*ga < 0 else (c, gc, b, gb)
    exact = princen(d, 0.0)
    print(f"gap {2*d:.1f}: area {c:.3f} (Princen {exact['area']:.3f}), "
          f"pressure {ev.bodies().pressure[0]:.4f} (Princen {exact['pressure']:.4f}), "
          f"wetting {exact['alpha']:.0f} degrees around each fibre")
```

```text
gap 1.0: area 2.438 (Princen 2.441), pressure -0.1073 (Princen -0.1073), wetting 82 degrees around each fibre
gap 1.5: area 5.374 (Princen 5.376), pressure 0.1579 (Princen 0.1579), wetting 109 degrees around each fibre
gap 2.0: area 9.899 (Princen 9.902), pressure 0.2523 (Princen 0.2522), wetting 132 degrees around each fibre
gap 2.6: area 19.457 (Princen 19.492), pressure 0.2896 (Princen 0.2896), wetting 160 degrees around each fibre
```

At the widest gap shown the liquid already reaches 160 degrees round each fibre from the
side facing the other, and the column's cross-section is twenty times the fibre's; at
the limit (gap 2.83) the area reaches 30. At a contact angle above zero, Princen's
balance has no solution before the liquid reaches the far side: the column ends at a
fold, at a gap of 2.47 for 20 degrees (`princen_limit`).

## Crossed fibres

Two fibres crossing at a small angle delta, touching where they cross, leave a gap
that widens away from the crossing. Sauret et al. found that a perfectly wetting liquid
spreads into a column along it, symmetric about the crossing, and modelled each
cross-section as Princen's at the local gap: the column then ends where the gap reaches
Princen's limit. Given more liquid than that column holds, they saw it gather into a
drop at one end of the column, breaking the symmetry. `crossed_column` gives the
model's column length for a volume, and the longest column with the volume it holds.

`crossed` starts a column between the fibres, crossing at 30 degrees, with flat sides
and ends, and relaxes it. From such a start a perfectly wetting liquid doesn't relax
(its contact lines lie along the fibres in the narrow gap near the crossing), so it
relaxes at 20 degrees first, the contact angle a datafile parameter `theta`, and
{func}`~pysurfaceevolver.recipes.continuation` lowers it to 0. Adding liquid, the
column grows along the gap. `shift` is the eigenvalue of the mode that moves the liquid
along the fibres, toward one end, and `wrap_angle` how far round the fibres the liquid
reaches (180: the far side):

```python
from pysurfaceevolver import recipes
from fibres_helpers import crossed, crossed_column, shift, wrap_angle

delta = 30.0
model = crossed_column(delta, 0.0)
print(f"model: longest column {model['half_length_max']:.2f} each way, "
      f"holding {model['volume_max']:.1f}")
ev = crossed(delta, 20.0, 5.0)
ev.relax(levels=1)
for step in recipes.continuation(ev, "theta", [15, 10, 5, 2, 0]):
    pass                                                      # down to perfect wetting
for volume in (5, 10, 20, 30, 40, 50, 60, 70):
    ev.body(1).target = volume
    ev.remesh(target=0.22, keep_boundary=False)
    ev.relax()
    x = ev.vertices
    half = (x[:, 0].max() - x[:, 0].min())/2                  # the column's half length
    L = crossed_column(delta, 0.0, volume)["half_length"]
    print(f"volume {volume:2d}: half length {half:.2f} (model {L:.2f}, {half/L - 1:+.1%}), "
          f"pressure {ev.bodies().pressure[0]:.3f}, shift along the fibres {shift(ev, axes=(0,)):.3f}, "
          f"wraps {wrap_angle(ev, delta):.0f} degrees")
```

```text
model: longest column 8.20 each way, holding 87.7
volume  5: half length 3.77 (model 4.20, -10.2%), pressure -0.468, shift along the fibres 2.323, wraps 94 degrees
volume 10: half length 4.53 (model 4.96, -8.6%), pressure -0.066, shift along the fibres 1.313, wraps 115 degrees
volume 20: half length 5.55 (model 5.86, -5.2%), pressure 0.180, shift along the fibres 0.585, wraps 136 degrees
volume 30: half length 6.27 (model 6.45, -2.8%), pressure 0.273, shift along the fibres 0.354, wraps 149 degrees
volume 40: half length 6.79 (model 6.90, -1.6%), pressure 0.323, shift along the fibres 0.257, wraps 158 degrees
volume 50: half length 7.31 (model 7.27, +0.6%), pressure 0.348, shift along the fibres 0.185, wraps 166 degrees
volume 60: half length 7.70 (model 7.57, +1.7%), pressure 0.366, shift along the fibres 0.144, wraps 172 degrees
volume 70: half length 8.04 (model 7.83, +2.7%), pressure 0.379, shift along the fibres 0.116, wraps 178 degrees
```

The column's length follows the model: 10% shorter for a small column, whose rounded
ends count for more, and within 3% from a volume of 30. The model's column ends at a
volume of 88, where its end cross-sections reach Princen's limit, the liquid all the
way round the fibres. The computed column gets there first: its ends wrap further round
the fibres than a long column's cross-section, and close round them between volumes 70
and 80 (at 178 degrees at 70), a change of topology the mesh can't follow; past it the
runs don't converge. Up to there, moving the liquid toward one end raises the energy:
the eigenvalue falls but stays positive. So here the symmetric column doesn't break its
symmetry by itself before its ends close round the fibres, and what Sauret et al. saw,
liquid gathering at one end, would come after that change of topology. These fibres
cross at a wider angle than most of their experiments, where the columns are far
longer than wide and the model fits better.

## Pictures

```{figure} _static/fibres.png
:alt: The sideways eigenvalue of shrinking barrels, the clam shell's energy against the barrel's, the pressure and area of long columns against Princen's, and a crossed-fibre column's length against the model
:width: 100%

From the left: the eigenvalue of the sideways shift of three barrels as they shrink,
against the volume over the exact onset; the clam shell's energy minus the barrel's at
30 degrees (dotted, the barrel's onset); the pressure (black) and area (grey) of a long
column's middle against the gap, Princen's curves and pySE's points (dotted, the limit
2 sqrt(2)); the half length of the column at crossed fibres against the volume, the
model and pySE's points, with how far round the fibres the liquid reaches (red,
dashed). Drawn by `tools/fibres_figure.py`.
```

```{figure} _static/fibres_shapes.png
:alt: A clam shell and a barrel of the same volume on a fibre, and liquid at two crossed fibres at volumes 30 and 70
:width: 100%

Left, a clam shell and a barrel holding the same volume (30, at 30 degrees). Middle
and right, the column at crossed fibres at volumes 30 and 70 (perfectly wetting), the
second nearly closed round the fibres near its ends.
```

## Notes and limits

* **Topology changes are where the runs stop.** A clam shell closing round its fibre, a
  column closing round two: the mesh can't change how the liquid is connected, so the
  runs find where that happens but not what follows.
* **Why a slice for the parallel fibres.** Relaxing a whole finite column also gives
  Princen's middle, but only once the column has the right length, and its length
  changes by moving vertices along the surface, which `relax()`'s residual and Newton
  steps leave aside (gradient steps do move it, slowly): started 5% short, a column
  ends 5% short with a residual of 1e-9. The slice has no ends.
* **Small contact angles.** For parallel fibres at 0 degrees, gaps below about 1 leave
  the two menisci of the slice nearly touching in its middle, and the flat start
  doesn't relax. At crossed fibres the column relaxes at 20 degrees and is lowered to 0
  by continuation in the contact angle.
* **The clam shell's energy** converges a little more slowly than second order near its
  contact line on the curved fibre; the crossover is extrapolated from three meshes.
  A fourth (93 thousand facets) took minutes and came out above the third at two of the
  three volumes, which a converging mesh doesn't, so it isn't used.

## References

* B. J. Carroll, The accurate measurement of contact angle, phase contact areas, drop
  volume, and Laplace excess pressure in drop-on-fiber systems, *J. Colloid Interface
  Sci.* 57 (1976) 488–495,
  [doi:10.1016/0021-9797(76)90227-7](https://doi.org/10.1016/0021-9797(76)90227-7).
  The barrel's profile.
* B. J. Carroll, Equilibrium conformations of liquid drops on thin cylinders under
  forces of capillarity. A theory for the roll-up process, *Langmuir* 2 (1986) 248–250,
  [doi:10.1021/la00068a024](https://doi.org/10.1021/la00068a024).
* G. McHale, M. I. Newton and B. J. Carroll, The shape and stability of small liquid
  drops on fibers, *Oil & Gas Science and Technology* 56 (2001) 47–54,
  [doi:10.2516/ogst:2001006](https://doi.org/10.2516/ogst:2001006). The inflexion in the
  barrel's profile and its stability; clam shells computed by finite elements.
* G. McHale and M. I. Newton, Global geometry and the equilibrium shapes of liquid drops
  on fibers, *Colloids Surf. A* 206 (2002) 79–86,
  [doi:10.1016/S0927-7757(02)00081-X](https://doi.org/10.1016/S0927-7757(02)00081-X).
  Barrel and clam shell energies, 14 to 70 degrees.
* H. B. Eral, J. de Ruiter, R. de Ruiter, J. M. Oh, C. Semprebon, M. Brinkmann and F.
  Mugele, Drops on functional fibers: from barrels to clamshells and back, *Soft
  Matter* 7 (2011) 5138–5143, [doi:10.1039/C0SM01403F](https://doi.org/10.1039/C0SM01403F).
* T.-H. Chou, S.-J. Hong, Y.-E. Liang, H.-K. Tsao and Y.-J. Sheng, Equilibrium phase
  diagram of drop-on-fiber: coexistent states and gravity effect, *Langmuir* 27 (2011)
  3685–3692, [doi:10.1021/la2000969](https://doi.org/10.1021/la2000969). Barrel, clam
  shell and coexistence regions in volume and contact angle, with Surface Evolver.
* H. M. Princen, Capillary phenomena in assemblies of parallel cylinders: III. Liquid
  columns between horizontal parallel cylinders, *J. Colloid Interface Sci.* 34 (1970)
  171–184, [doi:10.1016/0021-9797(70)90167-0](https://doi.org/10.1016/0021-9797(70)90167-0).
* S. Protière, C. Duprat and H. A. Stone, Wetting on two parallel fibers: drop to column
  transitions, *Soft Matter* 9 (2013) 271–276,
  [doi:10.1039/C2SM27075G](https://doi.org/10.1039/C2SM27075G).
* A. Sauret, A. D. Bick, C. Duprat and H. A. Stone, Wetting of crossed fibers: multiple
  steady states and symmetry breaking, *EPL* 105 (2014) 56006,
  [doi:10.1209/0295-5075/105/56006](https://doi.org/10.1209/0295-5075/105/56006).
* A. Sauret, F. Boulogne, D. Cébron, E. Dressaire and H. A. Stone, Wetting morphologies
  on an array of fibers of different radii, *Soft Matter* 11 (2015) 4034–4040,
  [doi:10.1039/C5SM00401B](https://doi.org/10.1039/C5SM00401B). Princen's cross-section
  and its limit, d = sqrt(2) for equal fibres.
