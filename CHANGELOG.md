# Changelog

## Unreleased

### API

- `constraints.Boundary`, `curve()` and `circle()` take a `content` integral: the
  volume of a face that bounds a body but isn't in the surface, such as the disk a
  bridge is pinned on (Evolver's content integral on a boundary).

- `Evolver.events()`: the changes of topology that are coming, as `Event` records
  (`pysurfaceevolver.Event`): a neck about to pinch off, the surface about to touch
  itself, a solid it isn't on (away from its contact lines there, or gone through
  it), or contact lines on one solid about to meet, each with its gap, the edge length
  there, the place, the vertices and the constraint. `recipes.arclength` and
  `recipes.continuation` put them on each `Step` (`Step.events`) and stop, with a
  warning, at the first point where one is coming (`events=True` by default; a number
  sets the gap's factor, `False` turns it off); a point whose surface has gone through
  a solid is a failed step. `relax()` puts them on its result (`result.events`).

- `Evolver.pinch_off(event)`: cut the surface at a neck (an `Event` of kind
  `"neck"`): the two rows of vertices round it collapse to a point, which Evolver's
  `pop` splits, `rebody` makes the pieces bodies, and the two sides relax. The largest
  piece keeps the body's number; each piece gets the `volconst` of the walls it
  reaches; with a fixed volume the targets add up to the old one, each smaller piece
  given back what the collapse took from its side.

- `Evolver.merge(event)`: glue the surface where it is about to touch itself
  across the outside (a `"touch"` event): the nearest facets merged (Evolver's
  `facet_merge`) and opened, the bodies either side made one (its volconst and
  target theirs added), then relaxed down to a stable state (`relax()` in short
  runs, the edges kept in range, the changes on the way made).

- `Evolver.contact(event)`: make a contact a `"solid"` or `"contact lines"` event
  announces: where the surface reaches a solid, its facets there go and their
  border becomes a new contact line (a wetted patch where the gas was between, a
  dry patch where the liquid was); contact lines meeting across dry ground are
  joined (their facing edges glued, or the facets roofing a dry channel over
  taken away), a wet strip between them cut. The volumes don't change in the
  surgery (volconst); a new contact line round a sphere's pole changes the
  wetting energy the integrals leave out, which `Evolver.energy` keeps.

- `recipes.arclength(..., topology=True)` goes on through coming events (cuts,
  merges, contacts made instead of stopping) and where the branch loses its
  stability: it falls from the first unstable point (past a fold, one step past the
  last stable point) by steps along the unstable mode, cuts the necks that come
  (`pinch_off`), and relaxes into a stable state, yielded as a `Step` with `fell`
  set and the necks cut in `Step.changes`; the run goes on with the largest piece,
  the others keeping their volumes. Other events coming on the way (a solid only
  once gone through) end the run.
  Tested on a bridge between spheres breaking at its branch point into two drops
  (exact caps to 0.5%), a bridge between disks pinching at its fold and a catenoid
  falling into two disks. How the liquid splits depends on the way down (a descent
  of the energy, not the flow): 17-25% for the smaller drop of the bridge between
  spheres as the fall's start, steps and thread count change. Also tested: two free
  drops merging into the sphere, two drops on a plane merging into one cap (their
  contact lines joined), a drop growing onto a sphere into the exact plane-sphere
  bridge, a dry patch opening on a sphere inside a draining drop; a pilot bead-pack
  pore drained from volume 0.69 to 0.05 through ten drying changes (a coarse mesh;
  one level finer it got to 0.41).

- `constraints.sphere(axis=...)`: the sphere's integrals singular only at its north
  pole, `center + radius*axis`, so that it can point at a neighbouring bead where a
  ring of liquid stays, and contact lines are free to cover the rest. (The default
  azimuth about z is singular at both poles.) The volume's form is made of the
  wetted area's about the axis and two polynomial forms; Evolver integrates along
  chords just inside the sphere, so the forms are written to be smooth there too.
  The same bridge turned as a whole gives the same answer to 1e-4.

- Changes of topology keep free vertices out of solids: in the descent after a merge
  or contact, each wall with a contact angle gets a one-sided copy (a constraint named
  `pyse_guard_<wall>`, without integrals) on the free vertices near it, its two
  nearest. The surface resting on a solid, or contact lines that have met, are
  changes to make (they stay put, so they never came closer); a descent may end
  close to a wall (a contact to come), and a run of it that went wrong (a volume off
  its target, the energy fallen by more than a quarter of the area) is undone for
  plain gradient steps. `Evolver.events()` and the rest leave the guards out.

- A `sphere(axis=...)`'s pole moves off its contact lines. In `arclength(topology=True)`
  and the descent after a change, a sphere whose line comes within half its radius of
  the pole is given a new pole at the point farthest from its lines (a new constraint;
  the old one, which Evolver keeps, is left out of `events()` and the rest). The
  volumes and energy shift by the whole spheres the two poles' integrals differ by,
  counted from the lines' solid angles, not from the integrals (which a line near the
  old pole, or with long edges, puts off by up to half a sphere). A run of the
  descent that takes a line within a quarter radius of a pole, or across it, is done
  again with that pole moved first. Contact-line edges stored against their facet
  (seen once in a pack, its integral of the wrong sign) are turned round. In a
  16-bead pack the beads' integrals now stay with their lines' geometry (a line
  crossing a pole had put them a whole sphere off), but for lines a fall leaves with
  long edges.

- `arclength(topology=True)` goes further through a bead pack: a contact-line loop
  that has shrunk within half an edge (a patch closing up) is covered over; falls
  keep contact-line edges within two median edges (their chord integrals went 1.5
  off); where a change or fall ends, the mesh is evened (vertex averaging and edge
  flips, Newton steps back between: Newton's steps from there had converged only
  linearly); and where the branch still can't take a step from there, the change
  nearest to coming is made. Each removed a way a 16-bead pack's runs ended; they
  still end, from 87.5 at 57 to 83 run to run.

- Fixed in the changes of topology: contact lines joined where they ran the same
  way left a facet turned over (Evolver's pop then failed with an internal error);
  they aren't joined there now. Facets lying flat on a solid with next to no area
  (a merged neck sunk onto the plane between two drops) are taken away as a
  roofed-over channel's are; they had ended the descent as degenerate facets, and
  a test of two drops merging failed one run in four.

- Evolver fix (src/skeleton.c): putting an edge or a vertex on a constraint with
  energy and content integrals read its constraint list after Evolver had moved it
  (applying the energy method can expand the element attributes): a crash in a
  contact on a finer mesh (AddressSanitizer: heap-use-after-free).

- `arclength(topology=True)` passes over a change it can't make (descending from
  there instead of stopping), and a descent may go on for 200 steps after each
  change it makes (2000 in all), not 200 in all. Three falls without a change stop
  a run only when no step was taken between them.

- `arclength(topology=True)` backs up where it would stop: where a change or a
  descent fails (runs away, stalls, finds no stable state, Evolver or an edit fails
  on the way) or falls stop getting anywhere, the run goes back to a state it went
  through (its last stable point, then 2, 5 and 10 points back) and on from there
  another way (half the step, a quarter, remeshed afresh, edges 0.9 and 1.1 times as
  long), the changes on the way made where they now come; four tries at a place,
  twelve in a run. The states are kept in memory (pySE's memory of the surface with
  them), a few at a time: 0.3 ms a step on a 16-bead pack. Each try is logged
  ("branch backed up 2 steps: ...") and the steps after it are numbered on from the
  state it went back to. A 16-bead pack's runs end where the way down from one state
  fails, and from a state next to it the way down rarely fails too: resumed at each
  of nine failed runs' last state, all nine reached breakthrough.

- `contact()` splits a contact line that runs through a vertex twice by making one
  fan of facets again on a new vertex there (Evolver's `pop`, used before, made a
  junction of three facets on one edge, a facet turned over, and Evolver's popping
  in `relax()` later failed with an internal error). Falls flip edges where they
  split contact-line edges (facets had gone to 0 degrees in a pack's fall), and shrunk contact-line loops
  are covered only within a fifth of an edge (within half, a contact's one-facet
  hole on a wall was opened and covered again over and over).

- The topology runs log what they do to the `pysurfaceevolver.topology` logger
  (debug records: each change made or refused, each run undone, poles moved,
  patches closed, why a descent stopped). With `PYSE_CHECK_TOPOLOGY=1` in the
  environment a check runs after every step of their descents and a run stops,
  naming the step, at the first problem no change may leave: an edge on no facet
  or (one body or none) on more than two, facets turned against each other across
  an edge, a pinched vertex (two fans of facets), a border edge on no constraint
  or on one its vertex isn't on, a contact-line edge stored against its facet;
  degenerate facets, folds, vertices or edges inside a wall and edges through
  facets are logged (`bench/topology/corpus.py` sorts the runs by them).

- Checked local mesh edits in C (bindings/pyse_edit.c, used inside pySE):
  collapse, flip, split, smooth (along the surface; a contact line's vertex along
  the line), snap, separate (one vertex per fan of facets) and degenerate-facet
  removal, each refused by its rules (a contact line's vertex never collapses
  inward, constraints kept, no facet turned too far, no edge made too long, a
  bounded volume) or undone where the surface round it comes out worse (the
  topology check's problems counted before and after); a batch keeps the edits
  made before one undone. The topology runs use them: edges kept in range after a
  change and the mesh evened there before it relaxes, flips instead of Evolver's
  equiangulation, `contact()` separating a vertex its new contact line goes
  through twice (refused where it can't), the mesh evened at the end of a change
  or fall. Descents don't pass the surface through itself: a run, a fall's
  gradient steps, a step along the mode or a change that adds edges through
  facets is undone (a run evened out and tried again; a step halved; gradient
  steps taken one at a time).

- The topology runs remesh after each change (a checked isotropic remesher in C:
  edges split over 4/3 of the median edge, collapsed under 4/5, flipped toward
  valence 6, 4 on a border, smoothed along the surface, each edit checked, contact
  lines kept). Their changes are checked edits first: `merge()` joins the two facing
  facets by a short tube of facets, contact lines meeting across dry ground are
  bridged by two facets (lines at an angle, or sharing a vertex already, too), and
  `pinch_off()` splits the vertex where the cones meet one per side; Evolver's
  `facet_merge`, `vertex_merge` and `pop` stay as the fallbacks where an edit is
  refused (`fallback:` records on the `pysurfaceevolver.topology` logger).

- Topology runs to breakthrough (bench/topology/corpus.py: a pack's floor a wall
  at 90 degrees, the run done when the gas reaches it; sets `small`, 4 beads, and
  `wide`, 16 beads, six seeds at 15, 20 and 25 degrees each; `--h` for another
  edge length; a run's traceback in its log). Wet and dry patches on a wall are a
  checked edit in C (`patch`: the place's vertices onto the wall, a vertex on one
  other wall a corner of both, none on three nor where two walls touch, pinches
  separated), dissolving facets by hand the fallback; where a pinch can't be
  separated, the patch is grown over it. The descents (`arclength(...,
  topology=True)`):
  - where the surface would pass through itself (a run, a fall's steps, a step
    along the mode held back), the nearest change coming is made instead;
  - a change, and the tidying (patches closed, poles moved), is undone unless
    Evolver can take a gradient step from it; where Evolver fails later on, the
    last change is undone and not made again there; a change made three times at
    one place is not made there again (a join was made 160 times);
  - a branch that stops short at a fold of the volume sets the volume a typical
    step past it (twice that each time it comes back);
  - facets crushed flat in a fall are taken away (they had held a fall at 1e-4 a
    step); a descent that goes nowhere for its 200 steps is remeshed once and
    goes on; one whose runs and gradient steps all go wrong makes the nearest
    change, or is remeshed and tried once more; a relaxed state not yet sound is
    descended further instead of ending the run;
  - a run is undone for its volumes only where it ends farther off its targets
    than it began, and for its energy only where it also jumped by a tenth of the
    area in one step (menisci merging give up area step by step);
  - a dry patch is not opened again in a descent where one closed (a pack's run
    opened and closed patches on the top beads 37 times);
  - a pole moved because a run swept a line across it goes clear of where the run
    took the line too (a bead's pole had moved 39 times to one place), and runs
    that keep sweeping lines across poles are shortened;
  - a Newton step that finds a body with no vertex free to move lets go of its
    volume's target (the solids hold it).
  `Evolver.events()`: vertices on two other constraints make no solid events (a
  box's corner vertex lay through the third wall by round-off at every step).
  4-bead runs reaching breakthrough: 10 of 18 before, 12 to 16 of 18 now (paths
  differ run to run with any change); 16-bead runs: see the PLAN.

- Topology runs: one pole per contact-line loop. A sphere's loop near its pole
  moves alone onto a copy of the sphere's constraint with its own pole (volumes and
  energy shifted by the whole spheres its integrals differ by), the others stay;
  the copies are one solid to `Evolver.events()` (a vertex on any is on it, lines on
  different copies meet, an event names the copy whose pole is farthest from it)
  and to contacts (the lines a contact touches moved onto its copy first). A
  heavily wetted bead's single pole had moved 96 times in 12 steps.

- Topology runs: necks against a wall are cut (`pinch_off`): a ring from a wall's
  contact line round to it again collapses onto its vertex on the wall nearest its
  middle, the line then split one per side; a cut that leaves the liquid joined
  elsewhere takes a handle off it. Half the necks a round of 16-bead runs refused
  were against a side wall, most of the rest against a bead. A cut whose pieces
  don't settle at once goes on down in the descent's runs.

- `constraints.sphere(..., gauge="flux")`: a sphere without poles. Its wetted area
  is the flux of a point source at its centre through the liquid's facets plus
  smooth line integrals on the other walls' contact lines (the divergence theorem
  over the liquid), so no pole has to be kept off the contact lines and no
  volume or area constant is left out; all spheres and planes of a datafile are
  coupled through the integrals. The field is capped at half the radius, inside
  the solid, where no liquid is (bounded at the centre; capped on the sphere
  itself, its kink sat on the contact lines and broke Newton steps there).
  Needs an open side (no closed box). bench/topology's corpus uses it for runs
  to breakthrough.

### Speed

- `Evolver.residual()` is kept for the surface's version (`_core.surface_version()`:
  any command, edit or move changes it): a 16-bead pack's descent asked 57% of its
  12.6k residuals of a state it had just measured. The same values; with the guards'
  change below, six arclength steps of a 16-bead pack from step 240 at one thread
  4.95e12 -> 3.53e12 instructions (867 -> 637 s, with the volumes' gap kept too,
  since dropped), the same run line for line.

- Remeshing: the checked edits read each vertex's kept wall gap inline and look at
  whether an edge is on a wall only where it comes near it: a 16-bead pack's whole
  remeshing at 14k facets 3.96e9 -> 3.33e9 instructions, the same mesh bit for bit.
  Taking the guards off keeps the mesh and the gap table without them (nothing
  moves), which a remeshing then needn't take again (75 million instructions at 28k
  facets); `events()` takes the border edges once, not the whole edge list per
  solid.

- `events()`' search for the surface close to itself (and `_sizes`' at an edge
  across) keeps a grid of the facets per way they face (the axis of their side's
  largest component, and its sign) and leaves out, for each vertex, the grids whose
  facets can't face it: most of a vertex's candidates were its own sheet's facets.
  At a 16-bead pack's 28k facets 109 -> 64 million instructions at half an edge,
  214 -> 93 at one edge; the same pairs, threads or not.

- The table of each vertex's gap to each constraint (events, guards, remeshing
  sizes, health) and the constraint normals at chosen vertices look each formula's
  closed form up once per call, not per vertex, and the table's rows go in parallel
  (threads) where every formula has one: at a 16-bead pack's 14k facets 108 -> 38
  million instructions a table at one thread, `events()` 294 -> 217; the same
  values bit for bit, threads or not. `guard()` builds its tables only for the
  vertices within reach of a wall (122 -> 104 million instructions at 14k facets for
  one that changes nothing).

- Topology runs: the guards go on and off the vertices in one call
  (`_core.set_vertex_constraints`, one recalc) instead of a command per guard, each
  with its recalc, and only where they change; and writing pySE's scratch
  attributes (`pyse_..._mark`) no longer recalculates. A 16-bead pack's `guard()` at
  2k facets 736 -> 108 million instructions (one that changes nothing 33), the same
  guards and surface bit for bit; it was 16% of a descent.

- Evolver's volume projection (project_all) evaluates the energy with the content
  only in its first pass after a restore, where that is likely the last; each pass
  did, and only the last one's was used: 16% fewer instructions in the projection
  of a 16-bead pack's slow gradient steps (ten passes), the same results bit for
  bit.

- The contact-line integrands of `constraints.sphere(..., axis=)` walls are
  evaluated in C (bindings/pyse_native.c, with their first and second derivatives
  as dual numbers) instead of through Evolver's expression interpreter, in the
  linear model's edge integrals once the surface is in quantities: the same values
  to round-off (1e-15 in a Newton step), a 16-bead pack's gradient step 8.6 -> 5.5
  ms and its Newton step 17.8 -> 13.9 ms. `_core.native_enable(False)` goes back
  to the expressions.

- Topology runs: the remesher's and edits' checks find a crossing's candidates in
  the grid's cells, changed elements included (a list of those scanned whole by
  every query), with cells an edge across and a box test before the determinants,
  and keep each vertex's wall gaps while it stays put: a 16-bead pack's remesh
  3.4 -> 1.6 s, its checked smoothing 1.0 -> 0.5 s (instructions 4.0e9 -> 2.5e9).
  `on_constraint()`, unguarding and crossing counts quicker (see the guard fix).

- pySE's own saves and restores inside `relax()` and the topology runs keep the
  surface in memory once it is converted to quantities (Evolver converts it at the
  first Hessian of a contact angle's integrals): a copy of Evolver's element blocks,
  restored bit for bit, instead of a dump and a reload; 43 ms against 1.1 s for a
  save and restore at 98k facets, 0.3 against 10 ms at a 16-bead pack's 1.5k.
  `Evolver.save()` is still the dump.

- Newton steps on a surface free to move as a whole reuse the last step's null
  directions while the topology is the same (checked by a product with the matrix)
  instead of a solve for each: about 0.1-0.2 s of a 1.1 s step on the free 98k
  quadratic cube.

- The flux gauge's integrands in C at any number of spheres (no 64-term limit;
  beyond it Evolver's interpreter, whose second derivatives of these powers are
  wrong): four terms at a time (AVX2, a scalar path the same to the bit), value,
  gradient and Hessian in closed form, one sum for the fields that share their
  spheres (volume and wetting). Gradient step per facet at 64 beads 23.5 -> 5.8 us
  (pole gauge 1.1), at 256 beads 2209 -> 14 us. Evolver's volume projection no
  longer scans every energy expression for a pressure at each pass (known from the
  parse; 19% of a 64-bead step).

- Topology events (`Evolver.events`, and every look in a descent): the search's
  loops in C (bindings/pyse_events.c: the surface near itself, with threads;
  contact-line chains and pairs; distances to segments; grouping), the same events:
  4-6 times faster at a 16-bead pack's 2-3k facets, 20-35 times with a large
  factor.

- The remesher: an edit's crossing checks only on the elements it changed (on a
  surface without crossings at the start), smoothing moves' checks reduced to what
  a move can change, grids hashed (only cells in use) with a box per entry,
  snapshots per edits made (every facets/64 on a large surface) instead of per
  candidate, no grid re-binning after smoothing moves, contact-line crossings by
  grid: the same meshes to the bit, a full remesh of 14k facets 8.6 -> 2.4 s, of
  106k 121 -> 25 s (about linear in the facets).

- Newton steps: facet vector integrals' Hessians in closed form (linear model, 3D)
  and the flux point sources' second derivatives as ten symmetric components; the
  arclength corrector's two Newton steps from one factorization (197 -> 103 ms);
  descents' gradient steps skip the move at scale 0 where nothing changed and take
  a parabola's step from the energy's slope (11 -> 7 ms a step; off elsewhere).

- Descents' gradient steps evaluate the integrands once per move instead of twice:
  each move's volumes are predicted for its restoring motion (where the step began,
  plus the slope along the motion from the volume gradients, plus the curvature the
  first trial measured), and the energies the line search compares are taken as if
  the volumes were restored, to first order with the pressures. The restore's own
  pass checks the volumes and restores again if they are out of tolerance; the
  final move's volumes land as close as the stock projection's. The search's
  energies are nearer the restored ones than the stock projection's own (a trial
  after vertex averaging: 4e-7 off against 2.5e-5); 200 steps of a 16-bead pack end
  3e-7 from the stock projection's energy. Instructions a step, flux gauge: 16
  beads at 2k facets 64 -> 43 million, at 14k 407 -> 281, 64 beads (5.9k facets)
  249 -> 156. Every gradient step (stock search too, the same to the bit): the
  content isn't evaluated again after the move where the projection's last pass
  found it there (16-bead pack 108 -> 99 million at 2k facets, 616 -> 568 at 14k).

- Topology events and a descent's target edge sizes: the table of constraint gaps
  kept for the surface version (the sizing took it, then the event search again);
  constraint normals at the vertices that need them only (`_core.constraint_normals_at`);
  points' distances to contact-line segments through a tree of boxes (5-20 times
  faster than against every segment); the bodies either side of a vertex and the
  sizes' grading without unbuffered `np.maximum.at`/`np.minimum.at`. The same events
  and sizes; at a 30k-facet pack the search 130 -> 104 ms, the sizing 156 -> 112 ms,
  a remesh near a change 640 -> 565 ms.

### Documentation

- New page "Draining through changes of topology" (docs/drainage_topology.md): a drop
  draining over a sphere inside it (a dry patch opening and growing, its contact angle
  as set) and one pore of a bead pack (dry patches on one bead, then the liquid's jump
  at a fixed volume, curvature 4.5 to 3.1), with what it can't do yet.

- Four pages of classic results, each against an exact reference computed on the
  page's helpers:
  - `disk_bridge.md`: the stability limits of a liquid bridge between two equal
    disks, without gravity (the least volume, pinching or moving to one disk; the
    largest, sideways where the surface meets the edges flat) and at three Bond
    numbers (Slobozhanin and Perales' Table I to its last digit).
  - `pendant_drop.md`: the largest drop a ring can hold against the Bond number
    (0.2-0.3%), and on wide rings the drop tipping sideways first (Michael and
    Williams' 3.219 capillary lengths).
  - `fibres.md`: a barrel on a fibre rolling up into a clam shell (an exact onset,
    where the inflexion of its profile reaches the contact line; the energies'
    crossover extrapolated from three meshes), Princen's column between two fibres,
    and a perfectly wetting column at crossed fibres (relaxed at 20 degrees, lowered
    by continuation in the contact angle) against Sauret et al.'s model.
  - `floating_lens.md`: a lens floating on a liquid, three tensions meeting at
    Neumann's triangle (1e-4 against the exact caps), and under gravity against the
    axisymmetric equations and Langmuir's puddle thickness.

### Fixes

- Evolver's walks round the mesh's rings (an edge's facets, a vertex's edges, a
  facet's edges) on the paths the topology runs take are bounded, as pySE's own
  are: edge deletion, its unstarring and link checks, edge refinement and
  division, facet triangulation and dissolving, edge reversal, equiangulation's
  ear and doubled-edge tests, vertex averaging, popping (`pop_one_edge`,
  `try_prop`, `versplit`, `two_split`, `cone_analyze`), facet ordering round an
  edge, and the vertex-normal and vertex-facet walks every step takes. A ring that
  doesn't come back to its start is refused before anything changes (the edge not
  deleted, refined or swapped, the vertex not moved), or, mid-change, stops with a
  recoverable error (6370) instead of spinning: a 16-bead pack's remeshing spun in
  one for half an hour (pySE's `link_kept()`, guarded since). Where the guards in
  edge deletion's checks ran out they let the deletion go on; they refuse it.

- Topology runs at a box's side walls (16-bead packs drained to breakthrough):
  - The descents' guards: each vertex on the walls nearest it, as many as make
    three with the walls it is on (not two, planes first: free vertices in a
    box's corner went into the bead there, run after run undone; contact lines'
    vertices, unguarded, went out through the side wall their wall meets);
    contact-line edges split during gradient steps give the midpoint the
    guards both ends have.
  - A sheet (a neck whose waist normals lie along one direction) is an event
    within the gap two sheets are, not four times it: sheets against side walls
    ruptured 1-1.8 edges thick, and the liquid came back over the cut again
    and again.
  - A neck against one wall is cut across an axis along the wall (its waist's
    vertices are the flanks beside the wall, whose normals put the axis across
    it: the cut squashed the flanks together and left the neck, the body one
    piece); where the collapse leaves the neck's two ends on the wall joined by
    an edge, the checked `unzip` edit parts them.
  - Shrunk patches are closed against the edges round them where the mesh is
    finer than the edge scale (a one-facet patch made on a refined mesh was
    closed at once, and the surface lay on the wall with its contact refused);
    pieces of surface with a border and next to no area (lone facets left
    along a box's edge) are taken away as crumbs.
  - The checked edits count each vertex of a patch once (a separation beside
    another pinched vertex was refused as worse).
  - A neck's waist (whose normals tell a sheet from a tube and set the cut's
    axis) is its vertices near its narrowest point, not all of the event's: a
    thin film curving between three beads, 344 vertices, passed for a tube
    whose ring came out in two pieces and was refused at every look. A sheet's
    hole: the mesh refined where it opens, the facet straight across taken,
    and folds allowed in the checked merge when the faces are closer than
    their facets are wide (the tube between them can't but fold back).

- The remesher's collapse walks a facet ring with a guard, as its other walks do: a
  16-bead replay spent half an hour in a ring that never came back to its start;
  the collapse is now refused.

- The descents lift chords through a curved wall: an edge between two contact-line
  vertices of one bead (on two facets, off the bead itself) whose middle lies more
  than a tenth of an edge inside it is split, its new vertex put out on the bead,
  free. A band of facets with every corner on a bead's lines has nothing to hold it
  off the bead, and the remeshing can't split its edges (their middles are inside
  the solid): a 4-bead run's fall left one, its lines moved apart and its chords
  were stretched to 1.3 (edges of 0.2), 0.24 deep in the bead; every run after
  crossed there, and the descent made 70 changes in 900 s round it, now 22 in 21 s.

- A descent whose runs leave the state as it was (the counts the same, the energy
  within 1e-6 of itself, the residual not halved) goes to its budget's end at once
  (the change nearest to coming, else the remeshing), and ends if it stalls again
  after it: a 16-bead replay's runs stopped moving at residual 2 and the descent
  went on so 200 times (215 s); another's crept at residual 3.9 for 75 runs. Likewise a relaxed but unsound state that
  the tidying and relaxing bring back to itself (it went round, remeshed twice a
  round, to the end of the budget), which is taken as the descent's end where all
  it lacks is a better mesh (degenerate facets, but none under the degree
  `arclength` stops at; the surface near a wall or itself).

- A descent doesn't make a change where it just made, undid or refused one: where
  the opposite change was made or undone (a cut and a merge, drying and wetting, a
  footprint dried and its lines joined back) on the same solid (or both on none)
  within three of the local edges, the same change (its kind, its way) within one
  (it came back: it didn't take), or one of its kind refused within three, the next
  is put off until it is nearer than half that one's gap (through a solid twice as
  deep; a neck, touch or contact lines met), or for 20 of the descent's iterations
  at most (a sheet at a side wall, its lines joined, thinned again at the joint and
  was held there until the descent's budget ran out). The guards and the runs hold
  the surface meanwhile; where a run would cross, the changes not put off are tried
  first, then these; and where the descent gets nowhere (its runs stall, or its
  budget runs out), the change nearest to coming is made, these included, before it
  remeshes or gives up (a 16-bead replay's runs stopped against a bead, a dry patch
  there put off, and the descent gave up). In `arclength(topology=True)` the changes
  are remembered from one descent to the next until the branch steps on: a sheet's
  footprint on a side wall dried in a fall, and the next descent began by gluing the
  sheet back there, its first event, and ran away. (The same change beside one made
  is no repeat: a film drying in patches over a bead, the next two edges from the
  last, put off, left the descent wandering.) A sheet's neck at a wall where one was
  cut in the descent ruptures by a hole beside its contact line instead of its
  footprint dried again. 16-bead replays had a sheet's footprint on a side wall
  dried three times and its lines joined back, the runs crossing between, until the
  descent ran away.

- A descent that would run away (its runs, then single gradient steps, going wrong,
  and every change near refused) undoes the change it made last instead, if any, and
  goes on without it, as it does where Evolver fails after one: after a 16-bead
  replay's contact lines were joined in a box's corner, every run and step went
  wrong and every change there was refused.

- A neck against one wall whose ring Evolver's deletion won't collapse ("the ring
  didn't collapse to one vertex (3 left)": a 4-bead pack's neck on a bead refused
  so at every look, a 16-bead pack's on a side wall until its descent ran away) has
  its footprint on the wall dried across instead, as a neck against two solids is
  cut: the liquid parts at the wall, the pieces made bodies as a cut's are. Where
  the neck is still there after it, the refusal stands.

- A neck collapsed below the mesh whose one side is a spike (the facets on its
  ring's edges there all meet at one vertex with no other neighbours: a filament
  thinned to a cone 3e-5 across at its base) is taken off, the apex and the ring
  collapsed to one point and the rest of the surface closed there. Cut as a neck,
  the spike's facets would be one point's: Evolver refused every edge ("the ring
  didn't collapse to one vertex (3 left)"), at every look of a 16-bead descent. A
  spike's tip may be up to three vertices joined only to the ring and each other: a
  tip of two, cut as a neck, left two vertices ("2 left") and the neck uncut.

- The relaxing after a cut (`pinch_off`'s, and `arclength`'s after a neck it cuts)
  makes no changes of topology of its own, as a descent's runs don't: `relax()`'s
  pops and edge collapses after a sheet's footprint on a side wall was dried left a
  vertex there pinched (two fans of facets, edges through facets), which no contact
  on that wall could separate after; every one was refused and the descents ran away
  (two of the 18 16-bead runs drained to breakthrough).

- A cut (`pinch_off`, a neck collapsed below the mesh) whose piece is a crumb of the
  collapsed tube itself (left by the seam's tidying without facets, or holding next to
  nothing in a lone facet) takes the crumb away, its target back to the body cut. Left,
  a body with a fixed volume and nothing to hold it made Evolver's constraint
  projection singular (ERROR 3005), or the fast gradient steps' volume prediction blew
  the energy up to 1e9; 16-bead packs' collapsed necks were left so, uncut, until the
  descents stalled round them.

- Snapshots in memory (`relax()`'s and the descents' undo) put back after a body
  was dissolved and another made: Evolver gives the freed volume quantity's and
  method instance's slots to the next ones, and the restore copied flags back slot
  by slot, so the new body's volume was marked deleted under its live method (its
  volume read 0; dumps declared a method they never wrote and didn't load). Freed
  slots are now kept while a snapshot is held, and a restore brings them back.

- Topology events: each vertex's edge length was the mean of other edges'
  lengths (bincount weights out of order: 7-17% of a pack's vertices off by more
  than a fifth), and the body behind or ahead of a vertex the last facet's, not
  the largest. The neck factor (`NECK`) goes from 3 to 4, which keeps necks found
  before they pinch with the true lengths.

- `close_patches` asks for the walls' edges again after closing a patch (the
  lists it had no longer matched the mesh: "operands could not be broadcast").

- Evolver's Hessian setup with constraints kept apart (bk.c,
  `bk_constraint_setup`) counted each constraint's entries from each row's end,
  never looking at a row's first entry: a row without its diagonal, its
  constraint entry first, went uncounted, and the entries were then written one
  past the end of the arrays (heap corruption; a 4-bead topology run ended in
  "double free or corruption" at the next error). All entries are counted, as they
  are placed.

- `Evolver.body(n)` goes by the body's id: after a body was deleted (two merged
  into one) the others keep their numbers, and `body()` refused the highest
  ones.

- MUMPS analysed a much-used pattern again with PORD on small systems too, where
  PORD can find more multisector stages than its compressed graph has vertices and
  exits the process (knotty.fe's string relaxed with `relax(levels=0)`). PORD only
  from 10000 rows.

- `Evolver.contact()` on contact lines that already meet at a vertex (the border
  through it twice, after a pore's liquid rearranged): Evolver's `pop` splits the
  vertex, and the lines are joined as two.

- Evolver's equiangulation (`u`, and `edgeswap`) could swap two quadrilaterals'
  diagonals onto one pair of vertices in a single pass: a doubled edge with four
  non-manifold facets, and later ERROR 1303 in `o` (cone_analyze). A swap onto
  vertices an edge already joins is refused.

- Evolver's edge deletion (`t`, `delete`) keeps the strict link condition for an edge
  of the surface's border (on one facet, as along a contact line): merging its ends
  where they share another neighbour pinched the border into a figure eight, and a
  contact line's spike came out as a facet with a corner twice.

- `recipes.arclength` from a surface `relax()` found in equilibrium without
  iterating (a flat meniscus): Newton steps failed with "Pressures invalid"; a
  gradient step is taken first, as `saddle()` does.

- `Evolver.remesh(protect=...)` with `min_edge`: the mask was applied after the
  deletion had renumbered the edges (now carried by id).

- `recipes.arclength`: settling back after tidying takes up to 12 Newton steps where
  8 left the residual within 10 tol (a free drop's rigid motions slow them).

- packing.md's 5-degree dimer ran on into bead 3 (from volume 0.203 to 0.196, where
  `arclength` stalled): it now stops at 0.206, where its meniscus comes down onto the
  bead. The pore wedge with a narrow bottom throat now stops at volume 0.204, where its
  liquid is about to pinch off at a mirror plane.

- Newton steps carry the mesh along a moving contact or triple line. With
  `hessian_normal` a step moves each vertex only along its normal (across the line,
  on a line), so a line that moved further than the edges beside it inverted them:
  a lens near spreading started 5% off its rim, and the lens under gravity refined
  twice, ended unconverged with collapsed facets. Where the plain step squashes the
  facets beside a line (and none inside a surface), the vertices beside the line now
  move along their surface with it (Jacobi sweeps of the lines' motion in each
  vertex's tangent directions); any other step is Newton's own, so convergence is as
  before, and `arclength`'s corrector takes plain steps. Both lenses now converge to
  their references (rims within 0.02%). `PYSE_NO_FOLLOW=1` turns it off.
- `constraints.cylinder` with its default `gauge="azimuthal"`: on a fibre that isn't
  vertical, a body's volume had the wetted face's part added instead of taken off
  (a band of liquid round a fibre along x came out 2x too large in the test; the
  energy was right). Found while computing drops on crossed fibres.
- `recipes.arclength` turns sharp folds: where a step's change is mostly the value's
  (a drop hanging from a ring of radius 1 at its largest volume), the secant of the
  last two points points off the branch by an angle that halving the step doesn't
  shrink, and the run stopped at the fold. Once a step fails, the shorter ones go
  along the branch's tangent at the last point (two more Newton steps).
- `recipes.arclength` in the quadratic and Lagrange models no longer spreads the
  vertices between points: Newton steps move nodes only along the normals, so the
  spreading's rearrangement stayed, and the surface settled on a slightly different
  equilibrium each time (a bridge between disks drifted 2.3% past its fold).
- `recipes.arclength` halves a failed step up to six times by default (three left a
  bridge between disks under gravity, and a drop on a ring with large steps, short of
  their folds), but not below `ds/4**halvings`, where a run near a dead end only crept
  on.
- Evolver's `rebody` command in quantity mode (pySE's) left a wire's content integral
  with the old body: after a bridge between disks pinched off, the drop at the top
  disk lost the disk's volume and the bottom one gained it (1.98 and 1.98 read as
  11.17 and -7.21), and refining the wire changed it again. `rebody` now rebuilds the
  bodies' quantities (Evolver's `reconvert_bodies_to_quantities`, which only `tordup`
  called).

## 0.7.5 (2026-10-15)

Python 3.11 to 3.14. Pseudo-arclength continuation through folds (`recipes.arclength`)
and a predicted start for each `continuation()` step; surfaces in worker processes of
their own (`Evolver(process=True)`); the quadratic model in parallel, and Newton steps
that converge on surfaces free to move as a whole. Breaking: `newton()` returns an
`IterationResult`.

### Requirements

- Python 3.11 to 3.14: 3.9 and 3.10 are past their end of life (October 2025 and
  October 2026), and NumPy and SciPy releases have left them; 3.14 is new (wheels and
  CI). Wheels and CI cover 3.11, 3.12, 3.13 and 3.14.

### API

- `constraints.sphere()`: a coordinate of the centre may be a datafile parameter's name
  or an expression (`(0, 0, "1 + gap/2")`), so setting the parameter moves the sphere and
  `recipes.arclength` can follow it (a bridge pulled apart at a fixed volume, say). The
  volume over a wet pole is then in the line integral (the pole's value taken off inside)
  instead of `volconst`, which can't follow a parameter; the same bridge comes out the
  same to 2e-6 in energy, converging alike under refinement. Both poles wet needs a
  centre of numbers. `plot(walls=True)` draws such a sphere where it is.

- `pyse.Evolver(process=True)`: a surface in a worker process of its own, so several
  surfaces at once. The handle works like any `Evolver`: attribute reads, assignments
  and method calls go to the worker (a round trip is about 0.04 ms plus copying),
  views such as `ev.body(1)` and `ev.parameters` are handles too (`recipes.continuation`
  works on them), and results come back as copies. Prints, warnings and errors from the
  worker show in the caller, Ctrl-C interrupts the worker's call, callables given to
  methods run in the worker, and `plot()` draws in the caller. `threads=` sets the
  worker's threads; `close()`, `with`, or the last handle going away ends it, and a
  dead worker raises `WorkerCrashed`. The worker is a new interpreter
  (`python -m pysurfaceevolver._worker`), so scripts need no `__main__` guard.
  POSIX only. Errors keep `errnum` and `output` when pickled.

- `ev.newton()` returns an `IterationResult`, as `relax()` does, instead of the number
  of steps kept: `newton_steps` (the steps kept), `residual`, and `converged` (the
  residual below `tol`, or 1e-8 without one, with the volume and quantity targets met).
  Breaking: code using the number reads `ev.newton(...).newton_steps`.

- `ev.modes(n=1)`: the `n` lowest Hessian eigenvalues with the constraints held
  (`linear_metric`, as `stability()` reports them) and their modes as displacements
  per vertex, `Modes(values, vectors)` with `vectors` shaped (n, vertices, 3), each
  scaled to a root mean square of 1 and signed so its largest component is positive:
  a simple eigenvalue's mode comes out the same on any thread count. Evolver's `ritz`
  below the lowest eigenvalue, each vector turned into vertex motion through Evolver's
  own move (boundaries, constraint projection), undone. Unlike `saddle()`, the surface
  is left as it is and the mode carries no line search's even part.

### Continuation

- `recipes.arclength(ev, what, ds, steps, bounds=)`: pseudo-arclength continuation,
  following a branch of equilibria through a fold (where `continuation()` stops) and on
  along the unstable branch. Each point is `ds` along the branch from the last (the
  value's change and the vertices' RMS displacement together), predicted along the
  secant and corrected by Newton's method on the equilibrium plus the arclength
  condition, built from two of Evolver's Newton steps per iteration (bordering with
  its own factorization; no gradient steps, which would leave an unstable branch).
  Between points the vertices are averaged along the surface and edges flipped (Newton
  steps move vertices only along the normal; without this the error grew to 0.5% and
  the run broke down before the fold); the vertices themselves stay the same, and the
  step is measured along the surface's normals (the tidying's sliding isn't a step;
  counting it, a run turned back at the bridge's fold). Steps yield
  `result.negative_modes`, counted with the constraints held (`eigen_counts`). On the catenoid between round wires (stress case 17)
  it turns at the fold and follows the unstable catenoid within 0.03% of the exact
  areas. On a liquid bridge between two spheres (gap 0.6, 40 degrees) shrunk at fixed
  volume it passes the rupture fold (V 0.07306 against the axisymmetric 0.07307,
  refined four times) onto the thinner-necked branch within 5e-4 of the reference
  energies, and finds one negative mode from a neck of about 0.13 (V 0.08), before the
  fold, steady under refinement: an axisymmetric mode odd about the mid-plane (Vogel's
  small volume instability; see "Where a shrinking bridge breaks" below); following it
  with `saddle()` lowers the energy. Stops with a warning when a step fails after
  halving or the mesh degenerates.
  The two Newton steps of each iteration start from the same Lagrange multipliers
  (pressures), and the corrector carries them along with the vertices: Evolver's Newton
  step uses them in its Hessian and updates them, so the second step had started from
  the first one's, and away from an equilibrium the change with the value came out
  several times too large. The corrector had then converged slowly or not at all, and
  the bridge above (8192 facets) stopped at its fold in about half the runs; now each
  point takes 3-5 iterations at the full `ds` and 8 of 8 runs pass the fold, in 7 s
  instead of about 30. Undoing a step (in `relax()` too) puts the multipliers back with
  the vertices: the stress cases and 40 of the 42 sample datafiles that finish
  `relax(levels=2)` come out the same (cat.fe varies run to run either way); bga-6, an
  overloaded pad with no equilibrium, undoes 20 steps instead of 12 before giving up
  (5 s instead of 1).

- `arclength(..., along=mode)`: the first step goes along a displacement per vertex
  (such as `ev.modes(1).vectors[0]`) at the starting value instead of in the value,
  onto a branch that crosses there, and the run follows that branch. From the first
  unstable point of a shrinking bridge (a mode turning negative without a fold), it
  follows the off-centre bridges, with pressures within 1e-4 of the exact ones and one
  negative mode throughout; the two signs of `ds` give mirror images (the rupture page
  has the example). Not found for the packing page's pore: its symmetric branch stops
  before its triply degenerate mode turns negative.
- `arclength()` past folds and soft modes: a corrected point farther from its
  prediction than the step itself is a failed step (halved), not a point (a pore's
  liquid, stepped 0.01 in volume, had landed at -0.49); a point that doesn't settle
  again after tidying is tried with the averaging undone, and failing that it is a
  failed step too: the run goes back to the last point by the corrector along the same
  tangent and halves the step, where it used to stop. A catenoid on a coarse mesh in
  steps of 0.3 now turns at its fold (it stopped at wires 1.1 apart); a trimer between
  three beads at 5 degrees goes through its snap-in fold and on (it stopped there).
  A remeshed surface that doesn't settle is put back and the run goes on.
- `arclength(..., remesh=h)`: keeps the edges in range on a branch that stretches or
  shrinks the surface, with `ev.remesh(target=h)` between points when an edge has left
  0.5h-1.6h (or a dict of `remesh()`'s arguments, or a function of the surface called
  between points). The last step is carried over to the new vertices (by vertex id, and
  for a split edge's midpoint the mean of its neighbours') and the remeshed surface is
  brought back to the branch by the corrector. A catenoid between round wires, through
  its fold and down the unstable branch: without remeshing its facets degrade and at
  wires 0.85 apart the run is back to no negative mode with areas 30% off; with
  `remesh=h` it follows the unstable catenoid to wires 0.43 apart within 0.4% of the
  exact areas.

- `continuation()` predicts each step: from the third step on, relax starts from every
  vertex extrapolated from the last two steps (a secant predictor), which saves a fifth
  to a half of the gradient steps. Over the stress cases that step a value, the gradient
  steps went from 4820 to 2970 (case 2), 3010 to 1690 (case 6), 3116 to 1606 (case 12),
  3410 to 2520 (case 17) and 5870 to 5110 (case 14), with the same results. It applies
  only while the vertices stay the same (not after a refinement, remeshing or pop, in
  relax or in the loop), and not to a step more than twice the last (a jump such as
  1, 10, 100, where a straight line would overshoot). A predicted step that comes out unconverged is relaxed again
  from the last step's positions before `stop_on` or halving sees it. `Step.predicted`
  says whether a step kept the prediction; `predictor=False` turns it off.
- Tried and not adopted for stress case 8 (the drainage band on defaults, without
  remeshing): extrapolating the band pushes the vertices squeezed by the receding
  contact line further, and over 3 paired runs the worst errors went from 4-9% to
  170-1550%, with the snap 15-100% off (from 3-4%).

### Speed

- Newton steps: a sparsity pattern factored six times is analysed again with MUMPS's
  PORD ordering instead of AMD (slower to analyse, less fill). At 4 threads the steps
  after it are 10-15% faster in Lagrange 2 at 98k facets, Lagrange 4 at 24k and the
  linear model at 1.6M, and about the same at 393k facets and below (its extra
  analysis is 0.1 s at 98k, 1.25 s at 1.6M). `relax()` takes 2-3 steps on a pattern and
  is unchanged; `arclength()` takes 6-9 a point: six points in Lagrange 2 at 98k
  facets take 37-40 s instead of 41-42. Each datafile load starts again with AMD, so a
  run's results don't depend on what ran before it.
- The quadratic model runs in parallel like the linear and Lagrange ones: its facet
  energies, forces, volumes and volume gradients, its named-quantity loops and Hessian
  assembly (after `hessian` converts to quantities) and its vertex normals, from a
  cached table of each facet's six control points. The volume and its gradient are
  summed over pairs of control points (about a sixth of the products; results agree
  with Evolver's to round-off). Cube at 98k facets, 4 threads: `g 1` 0.65-0.84 ->
  0.17-0.19 s (0.48-0.57 -> 0.21-0.27 s after a Newton step), a Newton step 1.8-2.1
  -> 0.95 s, a default `relax()` 67-71 -> 34-38 s. Only the area and volume methods
  and vertex integrals are taken in parallel in the quadratic model; other named
  quantities run Evolver's loops as before.
- The quadratic model's facet tangents and Gauss points, and the Lagrange model's edge
  setup, multiply their small matrices inline instead of through `mat_mult` (the same
  sums in the same order: bit-identical).
- Not changed: `r` stays serial (refining 393k facets to 1.6M takes 1.4 s, less than
  half a Newton step there; about 40% of it is zeroing new elements in fresh memory,
  the rest walking edge loops: no single hotspot).

### Documentation

- "Liquid in a sphere packing" (`docs/packing.md`): the curvature at which gas passes
  the throat between three spheres of a layer (contact angles 20-60 degrees, against
  the Mayer-Stowe-Princen estimate: 2% below to 6% above), a trimer between three
  beads drained until the menisci in its throat meet (Semprebon et al.'s burst; at 20
  degrees it comes before any fold at gaps 0.09-0.25), and a pore full of liquid
  between four spheres, whose curvature peaks at its throats' entry curvature (9.886
  against 9.871) as a triply degenerate mode goes soft. Starting surfaces from flat
  patches between the spheres in `docs/packing_helpers.py`; figures by
  `tools/packing_figure.py`; checked by `tools/check_doc_snippets.py` (now a default
  page). At 5 degrees, the paper's angle, relaxing these starts slides vertices through
  the spheres (in Notes and limits). The `docs` extra now includes SciPy (the notebooks' reference
  helpers use it).

- "Liquid in a sphere packing": which throat a pore empties through. `pore_wedge` relaxes
  the full pore with a three-fold axis through one sphere and the opposite (bottom)
  throat, and cuts it down to the sixth between two mirror planes, which holds the
  bottom throat against the other three exactly (the full pore's mesh had split the
  triply degenerate mode). With the bottom throat the widest (gaps 0.03 against 0.02),
  the curvature peaks at 9.533, that throat's entry curvature (9.520 in the layer); its
  meniscus then goes on into the pore while the others pull back, and the branch turns
  at a fold: the pore empties through it. With it the narrowest, the other three go
  first (at 10.002). `bottom_meniscus` and the cut in the helpers (`_pore_surface`
  shared with `filled_pore`, unchanged).
- The 5-degree trimers near Semprebon et al.'s kink, on a finer mesh: the snap-in at
  gaps 0.13-0.15 with the throat's menisci meeting just past it, and at 0.11 a burst
  with no fold before it, so the change lies between gaps 0.11 and 0.13 near volume 0.1
  (the paper: 0.125 and 0.09); the figure shows both lines.
- Docs CI: the notebooks run side by side (`sphinx-build -j auto`, one thread each: 3
  minutes locally instead of 9 of notebooks), and the snippet pages in two jobs, each
  running its blocks four at a time (`tools/check_doc_snippets.py --jobs n`, which
  queues every page's blocks at once and reports them in order: all pages in 6 minutes
  locally instead of about 15).
- "Liquid in a sphere packing": a section on trimers at 5 degrees, Semprebon et al.'s
  angle: `arclength` with `remesh=` follows the trimer through its snap-in fold, along
  the unstable branch, through the pop-out fold and down the dimer branch (gap 0.17:
  bistable between volumes 0.185 and 0.224), and the two folds, mapped over gaps,
  meet in a cusp at about volume 0.267 and gap 0.226 (the paper: 0.26 and 0.22).
  `meniscus_distance` in the helpers; figures by `tools/packing_figure.py cusp`. The
  notes no longer say 5-degree starts fail.
- "Where a shrinking bridge breaks": a section following the off-centre bridges with
  `arclength(along=...)` against the exact ones. `tools/check_doc_snippets.py` runs a
  block starting with `# continuing` after the page's blocks before it.
- `bridge_spheres_branch` (docs helpers) weighs the neck's sliding by the neck's
  curvature: unweighted, the slope it solves for also vanishes at the cylindrical
  bridge (neck R cos(theta), in range above about 53 degrees), where sliding the neck
  moves nothing. That was the unexplained second zero at 60 degrees, where Evolver
  found no instability. The branch points it returns are the same.

- Liquid bridge example: "How far a bridge stretches". Bridges of the volumes that break
  at gaps 0.4 and 0.7 (20 and 60 degrees, from the exact axisymmetric bridges) pulled
  apart with `arclength` in the gap, the beads following the parameter: at 20 degrees
  they end at the fold (the neck pinches), at 60 degrees at the branch point well before
  it (the liquid goes to one bead); Evolver's widest stable bridge within 0.01 of the
  exact gap, Lian et al.'s rupture distance within 5%. The notebook's beads follow
  `gap` throughout. `bridge_spheres_branch` in `docs/stress_helpers.py` returns the branch
  point where shrinking bridges lose stability: at 60 degrees there is a second, at
  large volumes, where Evolver finds none, and the helper had taken that one. References
  to Surface Evolver work on liquid in bead packs on the bridge, slit drainage and
  bridge rupture pages.

- "Several surfaces at once" (`docs/several_surfaces.md`): `save()`/`restore()`,
  `pyse.map` and `Evolver(process=True)` compared, with an example of four drops relaxed
  and shrunk side by side from threads (twice as fast as one after another), what
  crosses between the processes, and the limits. Its example is checked by
  `tools/check_doc_snippets.py` with the stress page's.
- The stress page's catenoid between round wires goes past the fold with
  `recipes.arclength` (a runnable section).
- "Where a shrinking bridge breaks" (`docs/bridge_rupture.md`): a liquid bridge between
  two spheres (gap 0.6, 40 degrees) shrunk with `recipes.arclength` turns unstable at
  volume 0.079, 8% above the fold at 0.0731 where an axisymmetric calculation puts its
  end, and the unstable mode moves the liquid toward one sphere. An axisymmetric
  reference without the mirror symmetry checks it: bridges with the neck off the
  mid-plane branch off at volume 0.07895, exist only at larger volumes and have more
  energy than the symmetric one (a subcritical pitchfork; T. I. Vogel's small volume
  instability, 2013). New helpers in `docs/stress_helpers.py`
  (`bridge_spheres_fold`, `bridge_spheres_branch`, `bridge_spheres_offset`), a figure
  drawn by `tools/bridge_rupture_figure.py` along with pictures of the bridge along
  the run, its unstable mode and the off-centre bridge in 3D, and the page's code
  checked by `tools/check_doc_snippets.py`. The unstable mode comes from `ev.modes()`.

- A shorter README (92 lines, was 261): what pySE is, a picture of four relaxed
  surfaces, install, one example with its output, and where to read on, with badges.
  Its API tour moved to the docs as "A tour of the API" (`docs/tour.md`), and building,
  testing and the code layout to "Development" (`docs/development.md`).
  `tools/readme_figure.py` redraws the picture.

### Fixes

- Newton steps on a surface free to move as a whole (nothing fixed, no wall, as
  `cube.fe`): moving it changes neither the energy nor the volume, so the Newton system
  is singular along its rigid motions, and round-off moved the surface along them, the
  more the finer the mesh. On `cube.fe` at 98k quadratic facets each step moved it by
  up to 4e-6 and the residual stayed near 2e-6: `relax()` stopped unconverged after 75
  gradient steps and 3 Newton steps (34-38 s). The rigid motions (translations, and
  rotations about any axis) are now checked with one product with the matrix each, and
  those along which the system is singular (an eigenvalue below 1e-10 of the mean
  diagonal) are taken out of the step, made exact first by one solve each with the
  factored matrix; `relax()` there converges after 5 gradient steps and 2 Newton steps
  (residual 1e-9, 6 s). The solves cost about 20% of a Newton step, on free surfaces
  near equilibrium only. In the Lagrange model the same cube's residual settles at
  2e-10 instead of 1e-9 to 3e-9. Surfaces held in place get Evolver's steps to the bit
  (tested on `cat.fe` and `mound.fe`); `PYSE_NO_DEFLATE=1` turns it off.
- `arclength()` in the Lagrange model and for strings: tidying between points
  equiangulated, which Evolver refuses there ("Cannot equiangulate LAGRANGE model"), so
  the first point failed. It now tidies as `relax()` does (vertex averaging only). And
  on fine Lagrange meshes the corrector waited for steps below `tol` that round-off
  doesn't allow: in Lagrange 2 at 98k facets the vertices slide 2e-8 to 1e-7 a step
  along the surface with the residual at 1e-9, so points failed, steps halved and runs
  stopped after one or two points, at different points from run to run. A point is now
  also converged when it is in equilibrium (residual below `tol`) and the steps have
  stopped shrinking; a run of six points takes 38 s there.
- The edge force of a string with a conformal metric (`CONFORMAL_METRIC`), fixed on
  2026-10-08 (the metric's derivative term lacked a 1/4) but thought unreachable, is
  reached by the linear string model outside quantities mode, and now tested against
  the energy's gradient (`metric.fe` made conformal; the old formula was off by order
  1 near the throat).

- `arclength()` stopping at a point that doesn't settle after tidying leaves the surface
  at the last point (its vertices on the tidied mesh, Newton there), not at the
  unsettled one: a catenoid stepped past its fold was left with its wires 1.45 apart.
  `_core.multipliers()` is typed (a float64 array) in the stubs.

- `arclength()` halves its first step too, as it did the later ones (up to `halvings`
  times), instead of stopping with "try a smaller ds"; the attempts are undone from a
  copy once tidying has flipped edges. Finer meshes and shallow angles had needed a
  smaller `ds` by hand (the packing page's pore refined once, a throat at 20 degrees).

- Small contact angles: a surface that `relax()` leaves unconverged with vertices sunk
  into a wall from `constraints` whose contact angle is below 15 degrees is relaxed
  again from the call's start with those angles raised to 20 degrees, then lowered in
  stages (by 0.7 each) to their own, kept only if that converges. At a small angle the
  liquid meets the solid almost tangentially, gradient steps from a crude start slide
  the vertices next to the contact line through it, and the wall's integrals reward
  that (the energy fell without end). Trimers between three beads at 5 degrees (gaps
  0.25 and 0.30) and a bridge from a fat tube, which sank, now converge, within 2e-4
  of the same surfaces lowered by hand (the bridge within 7e-4 of the exact one).
  Surfaces that relaxed before come out the same, with the same speed. For this, a
  wall's contact angle given as a parameter, or as a number below 20 degrees, enters its
  integrals as `maximum(angle, pyse_angle_floor)`, a datafile parameter declared at 0
  (`Constraint.floored` says which); such a wall's `energy_constant` is NaN, as for a
  parameter angle, and `Evolver.energy` adds the left-out part.
- `relax(levels=n)` starts again one level finer when its last level ends
  unconverged: from the call's start refined once, then the levels as before (kept only
  if that converges). A coarse start can hold equilibria the shape doesn't have, and
  which one the gradient rounds found was chance: a 20-degree bridge from a tube 12
  facets around, with its volume moved by a few units in the last place, failed 6 times
  in 13 (the "different builds" of the bridge notebook: different SciPy versions
  computed its volume 1 ulp apart); now all 13 converge. Stress cases unchanged; the
  sample datafiles at levels 1 and 2 converge as before, to the same energies (those
  that don't, try the finer start too: up to 1.7 s more, tankex.fe).
- `relax()` leaves a surface that is already in equilibrium as it is (with `levels=0`:
  residual and target gaps below `tol`, no degenerate facet, also after Evolver's
  `recalc`, which puts the vertices on wires moved by a parameter): only the stability
  check runs. Its gradient rounds were mesh motion only, and next to a shallow contact
  line they slid vertices through the wall: a bridge between beads at 20 degrees,
  relaxed a second time, ended unconverged with 14 vertices inside a bead (175 after
  300 plain gradient steps, the energy falling throughout), and `arclength()`, which
  relaxes first, failed from it. Tests, the 43 sample datafiles and the stress errors
  are as before; stress case 6's continuation, whose first step re-relaxed the
  converged barrel, now needs a retry at 60 degrees (2020 gradient steps instead of
  1690).

- `relax()` with a body volume or fixed quantity far from its target (more than 25%):
  if the first gradient step leaves degenerate facets (below 1 degree) the surface
  didn't have, it is undone and the target approached in stages of at most 10%, each
  with 20 tidied gradient steps. Evolver's projection onto a far target moves each
  vertex along its volume gradient, several edge lengths and unequally where the mesh
  is irregular, and folded facets: drops at 30-120 degrees taken from volume 1 to
  0.25-4 in one relax() failed 15 of 72 times (3 perturbed starts each), now 9 (the
  rest mostly a contact line receding far, as stress case 8: a continuation or
  remeshing). The trial step is undone otherwise (coordinates and step size), so
  other surfaces relax exactly as before (same stress results and gradient steps).
  Staging every far target instead lost coarse starts (stress case 5's bridge, 2.6
  times its volume: its contact lines drift during the stages); not with optimizing
  parameters (bga-5's loaded pad has no equilibrium at the smaller volumes on the way).

- With optimizing parameters, constraint projection takes one more step for a vertex
  that ends within `constraint_tolerance` of its constraints but not well inside it
  (above 1e-3 times the tolerance): projection converges quadratically, so the vertex
  lands on them to rounding. Left up to the tolerance off, the vertices stalled
  Newton's residual on fine meshes: `bga-5.fe` refined four times ended "not converged"
  at a residual of 2.6e-7 after 11 s; it now converges (6.5e-10) in under 1 s. The
  limit was blamed on Evolver's finite-difference parameter Hessian; varying its step
  (`pdelta`) didn't move it. Surfaces without optimizing parameters project as before:
  applied to all, the extra step left stress case 8 (the remeshed drainage band)
  unconverged at two threads. The solder notebook drops its `tol=1e-6`.

- `continuation()`'s warning when it stops names the step as a plain number; with
  NumPy 2 and NumPy steps it said `np.float64(1.4)`.

## 0.7.4 (2026-10-12)

Four new stress cases, all fourteen passing; `relax()` converges where gradient rounds
used to creep (a contact line moving slowly over a curved wall) and keeps a shrinking
contact circle resolved; a warning when a surface is refined too far before relaxing.

### Stress cases

- Four new cases (`bench/stress/`, and sections in the stress cases page), each on
  `relax()`'s defaults against an exact or converged reference:
  - 14, a drop on a sphere, its contact angle swept from 20 to 150 degrees as a
    parameter (exact spherical caps): energies within 0.2%. It found the defect
    fixed below.
  - 15, a double bubble from two boxes, the second volume shrunk to 0.2 (the standard
    double bubble): area and pressures within 0.2%.
  - 16, a solder joint's force on its pad (Brakke's bga-3) against his axisymmetric
    model (bga-30): both converge at second order, the extrapolations agree to 5e-5.
  - 17, a catenoid between round wires pulled past its fold with
    `continuation(stop_on="unconverged")`: areas within 0.04%, stopped at the first
    step past the fold.
- `docs/stress_helpers.py` has the exact references (`drop_on_sphere`,
  `double_bubble`).
- Case 8, the drainage band, is now run as its notebook runs it, with
  `ev.remesh(target=h)` before each band step, and checked on what is stable from run
  to run: over three runs the median band-pressure error was 0.017-0.022%, the snap
  0.1-0.4% off, 87-89% of steps converged. On defaults alone its worst state scattered
  from 3% to 52% between identical runs, too noisy to catch regressions. Why it fails
  on defaults is in the guide: the contact line receding over the bead squeezes the
  vertices behind it into slivers where it meets the mid-plane. It now has figures and
  a section with a runnable snippet on the stress page, like every other case. All 14
  stress cases pass.

### relax()

- Gradient rounds that creep without settling (a contact line moving slowly over a
  curved wall) used up `max_iter`, and Newton then gave up after one slow step:
  stress case 14 ended most steps unconverged. Now, once the rounds have used half of
  `max_iter` with the residual falling by less than 5% a round (three rounds
  running), Newton is tried there; kept if it converges or cuts the residual a
  hundredfold, otherwise the surface goes back and the rounds go on (a neck
  shrinking toward a pinch-off creeps the same way, and must go on: case 2). Case 14
  now converges at every step, in half the time; every other case gives the results
  it gave before. Tried and dropped on the way: counting an energy drop as Newton
  progress (case 2 failed past its fold), a budget of its own for the
  conjugate-gradient fallback (it broke `max_iter`), and the trial without the
  half-budget condition (fewer tidying rounds cost accuracy: the gravity puddle
  went from 0.22% to 0.54%).
- The split of long edges leaving a contact line (after convergence) repeats, up to
  three times per call, while some are long: a contact circle that shrinks a lot in
  one step (a drop from 160 to 170 degrees) needed more than one. Stress case 3 now
  passes at 170 degrees (the implied contact angle 1.3 degrees off, was 1.7; the
  limit is 1.5).
- `relax()` warns when the surface was refined four or more times since it last
  converged: relaxing that fine is slow and may not converge (`cube.fe` refined five
  times: 10 s unconverged, against 0.4 s converged with `relax(levels=5)`).
### Documentation

- Figures for stress cases 13 to 17.

- The site reads as one path: grouped navigation (getting started, guides,
  examples, Brakke's samples, reference), every page saying what it adds and
  pointing to the next, conventions in one place (Getting started), the examples
  on the current API, and README's threading notes moved to the performance page.

## 0.7.3 (2026-10-12)

Replaces 0.7.2, whose x86 wheels and source distribution did not build. Faster
Newton steps and Lagrange stages, fewer silent mistakes when building surfaces, Brakke's
solder-joint series, and fixes to three bugs in Evolver itself.

### Fixes

- The 0.7.2 test of `relax()` on `tankex.fe` at one refinement level asserted a
  stopping energy that depends on the thread count (0.7.1 failed it too, on one
  thread); it now asserts what is guaranteed: no slide, volume met, failure
  reported. The 0.7.2 wheels for x86 and the source distribution did not build
  because of it.

### Breaking changes

- `ev["name"] = value` assigns only parameters and variables that exist (a typo
  raised nothing and created a new variable); `ev.define(name, value)` makes a new
  one. `continuation()` with a misspelt name raises `KeyError` with suggestions.
- `Body(volume=..., faces=None, ...)`: `volume` comes first, and `faces` defaults to
  all faces. `volconst` defaults to the sum of `Constraint.volconst` over the walls
  from `pysurfaceevolver.constraints` that the body touches, and with a single body
  over untouched ones too (a solid immersed in the liquid); explicit values win.
- Vertices on a `boundaries=` curve are fixed (free, they bunch up along the wire).
- `continuation()`'s `Step.energy` is `Evolver.energy` (with the walls' constants).

### Found in review

- Two more Evolver bugs fixed. Popping a fixed edge between two free facets divided
  it and retried without end, so `relax()` (which pops by default) hung; it was so in
  0.7.1 too. Trimming the warning-message buffer could read past its text.
- Vertex averaging's new ring cache moved vertices fixed after it was built;
  `energies()` counted an instance's modulus twice; the fused Lagrange energy pass
  ran before the pressures were recomputed, which mattered for an energy reading a
  pressure (now skipped there).
- Bulk writes on a second element type failed (Evolver's attribute names are
  global): the scratch attributes are now per type (`pyse_vertex_mark`, ...); they
  are written into dumps.
- `make_datafile`: a closing wall that several bodies touch (or, with several bodies,
  none) is no longer added to each body's `volconst`, with a warning; several bodies
  must each give their faces; `optimizing_parameter` angles and exponents in angle
  expressions pass the parameter check; per-vertex constraints take numpy integers
  (0: none); single-letter names are refused; string-model bodies are oriented
  counterclockwise.
- `health()` checks walls no vertex is on again (only constraints that only fixed
  vertices are on are skipped).
- `continuation(stop_on=...)` also stops where `relax()` pinches the surface off.
- `restore()` brings back the snapshot's walls and name; relax()'s memory of failed
  attempts only for the same load.
- Warnings from `relax()` callbacks keep their location, and `-W error` no longer
  replaces an error from `relax()`.

### Solder joints (Brakke's ball grid array series)

- Vendored `bga-2.fe` to `bga-10.fe`, `bga-30.fe` and Brakke's force command files
  (`zforce.cmd`, ...); a *Solder joints* notebook: the force on a pad (three ways,
  against Brakke's variational integral), force against height with the unstable
  range, the height a loaded chip floats at, and a chip pulling itself into line.
  (`bga-11.fe`, `bga-12.fe` and `bga-31.fe` run away even with Brakke's own
  recipes in Evolver 2.70, so they are left out.)
- `residual()` counts the force on optimizing parameters, so `relax()` balances
  them; `Parameters.optimize(name, on=True)` frees or fixes one.
- An Evolver bug fixed: reading a datafile reset the pressures for every named
  quantity without a Lagrange multiplier, energies included, so after a restore
  Newton refused to run ("Pressures invalid") on surfaces with an energy quantity
  and fixed volumes, inside `relax()` too.
- `relax()` undoing a step puts optimizing parameters back with the coordinates.
- `derivative(name, move=..., method="newton" | "fixed" | "relax")`: with `move`
  (how the vertices follow the parameter) Brakke's `zforce5` (Newton re-converged,
  the default) or `zforce4` (no relaxing, pressure-corrected); they agree with his
  variational integral to 8 digits. The surface, multipliers included, comes back.
- `relax(levels=n)` refines past a start with nothing free to move.
- `health()` no longer reports pad rims or display solids (equality constraints only
  fixed vertices are on) as walls, nor fixed vertices as crossing them.
- `relax()` passes on each kind of Evolver warning once, with how often it came up
  (19,430 "doesn't converge to constraint" lines became one).

### Faster

- Against 0.7.2 (2 threads, energies identical to 15 digits): a Newton step on a
  linear cube of 98k facets 313 -> 197 ms (393k: 1.46 -> 1.14 s), `relax()` 1.14 ->
  0.96 s; Lagrange 2, 98k facets: gradient step 373 -> 313 ms, Newton step 1.44 ->
  1.19 s, residual 255 -> 205 ms, `relax()` 5.25 -> 4.09 s. From: no serial normal
  in the Hessian setup for vertices on no constraint; Lagrange gradient and value
  kernels with their per-point factors computed once; Lagrange vertex normals from
  the per-topology normal cache; vertex averaging from per-topology rings (bit-identical).
- Gradient steps with named quantities only (Lagrange model, `convert_to_quantities`):
  9 passes over the elements instead of 14 (energy gradients with the volume
  gradients, the energy with the last volume projection pass; bit-identical).
  Lagrange 2, 98k facets: `g 1` 330 -> 292 ms; Lagrange 4, 24k facets: 218 -> 196 ms.

### Fewer silent mistakes

- A body whose faces point into it (negative starting volume against a positive
  target) warns when it loads; it used to turn inside out and report "converged,
  stable, health ok". So does a wetted plane whose normal points into the solid
  (the drop came out at 180 degrees minus the contact angle).
- `SharedEngineWarning`: an `Evolver` handle used after another handle loaded a
  surface (every handle is a view of the one engine).
- `make_datafile` errors name the mistake: faces given 1-based, a vertex put on an
  undefined constraint, `z == 0` for `z = 0`, a contact angle naming an undeclared
  parameter. `refine(-1)` raises; `parameters["typo"]` suggests names;
  `load_string` surfaces are named `surface.fe`, not a temporary path.
- `relax()` keeps the "Hessian not positive definite" warnings of its trial Newton
  steps to itself (it checks stability at the end); `restore()` those of loading its
  own dump.

### Simpler building and measuring

- `pyse.shapes`: `cap`, `disc`, `tube`, `revolve` and `box` starting shapes with
  named rims. The start needn't have the target volume.
- `Evolver.energy`: the total energy plus the wetting energy the walls' line integrals
  leave out (`Constraint.energy_constant`, also when the contact angle is a
  parameter); `make_datafile` records it.
- Constraints may be named (`{"table": C.plane(...)}`); `vertex_constraints` may give
  each vertex's constraints; `boundaries=` with `constraints.circle()` or `curve()`
  for films on curved wires (refining keeps the wire round).
- `load_arrays` remembers its walls: `plot(walls=True)`, `live_view(walls=True)`.
- `contact_lines(k)`: where the surface meets constraint k, as ordered polylines.
  `BodySurface.volume`, `.area`, `.areas()`; `Mesh.facet_areas()`, `edge_lengths()`,
  `edge_valence()`.
- `derivative(name)`: the derivative of the equilibrium energy with respect to a
  parameter (minus it: the force on a wall it moves; Brakke's "reconverge" method).
- `energies()`: the energy term by term, Brakke's datafile check.
- `continuation(stop_on="unconverged" | "unstable", halvings=k)`: stops at a fold
  with the surface back at the last good step, after retrying in smaller steps.
- `stability(eigenvalues=True)` reports the lowest eigenvalues when stable too.

### Simpler code

- `_evolver.py` (2,750 lines) is split into modules: errors, results, views,
  geometry, engine, analysis, relax and export. `relax()` keeps its settings and
  stages in one object per call instead of passing ten arguments around; magic
  numbers are named constants.
- Bulk writes (`set_flag`, `set_constraint`, `set_values` of other attributes,
  `remesh(protect=)`) are one `set ... where` command over scratch attributes
  instead of a statement per element (fixing half of 12k vertices: 4 ms).

### Tried and dropped (Brakke's tips, measured)

- Storage reordering along a space-filling curve (`reorder_storage`): Newton steps
  10-18% and vertex averaging 40% faster, but gradient steps and the residual 1.6-2.3x
  slower, on cubes of 98k and 393k facets.
- Conjugate gradients in the gradient rounds, history cleared after each tidying:
  fewer steps in several stress cases, but the catenoid case fails (degenerate
  facets) and the bridge case is slower.
- `sobolev_mode` Newton steps (positive definite) before the gradient rounds: the
  catenoid case fails and most cases are slower.
- `immediate_autopop`: coarsening of `100grain.fe` slows down a lot (1185 vertices left
  after 20,000 steps against 712) with no sign the default does harm; not switched on.
- Lowering `target_tolerance` for tiny volumes (`cbga1.fe` in cgs units): volumes
  were met exactly either way.

### Documentation and examples

- A *Getting started* page: a drop on a table end to end, and the conventions
  (normals out of the liquid, plane normals into it, spheres and cylinders solid;
  `ev.energy` against `ev.total_energy`; a surface tension other than 1; contact
  angles first order in the edge length; 1-based Evolver numbers against 0-based array
  rows). The API reference lists `pyse.shapes`, `constraints.circle`/`curve`/`Boundary`,
  `Stability` and `SharedEngineWarning`.
- The notebooks, their helpers, the stress cases page and the stress suite use the
  new API: starting shapes from `pyse.shapes` (the copies of `cap`, `tube` and
  `revolve` in `docs/stress_helpers.py` and `bench/stress/common.py` are gone),
  `Body(volume=...)` without face lists or hand-summed `volconst`, `ev.energy` instead
  of adding wall energy constants by hand, `load_arrays` with `walls=True` in plots,
  per-vertex constraint sets, `BodySurface.volume`/`areas()`. Starts are no longer
  fitted to the target volume by bisection (the liquid bridge, the slit droplet, the
  drainage, stress case 5): relaxing meets it. Results agree with before to the
  discretization error (the bridge's forces within 0.3%).
- The Brakke notebooks no longer silence `EvolverWarning` globally; `brakke_helpers`
  silences only the instability warning of `relax()` (its table reports stability)
  and Evolver's messages during a sample's own recipe.

## 0.7.2 (2026-10-12)

Faster `relax()`, a simpler API (with removals, listed first) and Brakke's online
examples.

### Breaking changes

- Removed `ev(...)` (use `ev.command(...)`) and `ev.hessian()` (an unguarded Newton
  step; use `ev.newton()`, or `ev.command("hessian")`).
- `relax()` no longer takes `cg`, `window`, `energy_tol` or `pinch`: the defaults
  were the only settings worth having (the conjugate-gradient retry stays, see below).
- `recipes.continuation(ev, what, values)`: `what` is a parameter or variable name,
  `ev.body(i)` (its target volume), `ev.quantity(name)` (its target) or a callable;
  `recipes.body_target()` and `recipes.parameter()` are gone.
- `stability()` reports `lowest` and `threshold` with Evolver's `linear_metric`
  (eigenvalues of the continuous surface, which don't shrink as the mesh is refined;
  the counts are unchanged).

### Faster

- `relax()` against 0.7.1 (2 threads, same energies to 12 digits): linear cube,
  98k facets 1.60 -> 1.09 s, 393k facets 9.7 -> 7.0 s; Lagrange 2, 98k facets
  8.3 -> 5.4 s; Lagrange 4, 24k facets about 7 -> 4.2 s. From:
  - the Lagrange stage's Newton steps stop once the residual is below `tol` (two or
    three of four or five were in round-off);
  - the stability check takes the negative-eigenvalue count of the last Newton step's
    factorization when that step began within a residual of 1e-2 and the surface is
    as it left it (it matched the full check in all of some 230 relaxes: stress cases,
    tests, Brakke's samples); otherwise the full check as before;
  - vertex normals in the plain linear film (the residual, the Hessian setup) from
    facet normals computed once per facet and the facets around each vertex kept per
    topology: the residual 3x faster (checked against Evolver's own normals under
    `PYSE_CHECK_FACET_CACHE`);
  - no second residual after Newton; `quantities()` skips its recalculation when
    there are none; `health()` reads flags and valences from the mesh.
- `restore()` (reading datafiles) about 20% faster: the lexer reads with
  `getc_unlocked`.

### More robust

- `relax()` equiangulates once after the first gradient step of each phase: a coarse
  start triangulated badly (Brakke's `cbga1.fe`, a prism of quads split at their
  centres, whose recipe swaps the edges by hand) used to settle at an energy of 11
  instead of 5.3, and `relax()` ended degenerate with the volume 49% off; it now
  converges at every level. Stress suite: same results, fewer steps (case 6: 60 -> 44
  Newton steps).
- Tried and dropped: a watch on the scale factor ("a scale factor going to 0 does not
  mean convergence", Brakke's hints) stopped `relax()` from recognising settled
  energies (the scale legitimately falls more than 1000-fold within a phase); fewer
  vertex averagings inside the conjugate-gradient retry rescued no more steps. The
  retry stays: without it the drainage case is ~30% faster but its median error
  doubles.
- Two topology tests (octa's popping, cat's pinch-off) run with one thread: their
  path varies at round-off with several threads (also in 0.7.1), and so, now and
  then, the final state.

### New

- `ev.quantity(name)`: a live handle on a named quantity (`value`, `target` settable,
  `None` frees it, `modulus`, `pressure`, `kind`). `ev.gravity`, `BodyView.density`,
  `make_datafile(gravity=)`.
- `ev.saddle()`: steps off a saddle along its unstable mode (Evolver's `saddle`, with
  the gradient step it needs first); `ev.relax()` from there finds the minimum it
  leads to. `ev.pop()`: Evolver's `o`, public.
- `ev.contact_angles(constraint)`: the contact angle at each contact-line vertex, from
  the facet normals and the constraint's gradient (first order in the edge length).
- `Mesh.slice(normal, point)`: a cross-section as ordered polylines.
- Constraints from `pysurfaceevolver.constraints` keep their geometry:
  `Constraint.to_pyvista()`, and `ev.plot(walls=[...])` / `ev.live_view(walls=[...])`
  draw them with the surface.
- Torus models plot each facet whole, unwrapped (they were stretched across the cell).
- `ev.write()` writes glTF (`.gltf`, `.glb`) and VRML (`.wrl`) scenes through PyVista;
  PLY files no longer trigger meshio's 64-bit integer warning.
- `make_datafile` orients a body's faces itself when they close a surface (consistent
  and outward); otherwise all +1 as before.
- Eight of Brakke's online examples ship with the samples: `cbga1.fe`, `cbga2.fe` and
  `bga-1.fe` (solder joints), `pcell.fe`, `dcell.fe`, `IWP.fe` (periodic minimal
  surfaces), `TwoSpheres.fe` and `TwoSpheres-explicit.fe`, with tests and notebook
  sections. Stress case 13: Schwarz's P and D surfaces against their standard areas
  (extrapolated to 1e-6 and 2e-5).

### Docs

- The notebooks use the new API instead of hand-written helpers (contact angles,
  cross-sections, walls) and raw commands; the tutorial's bodies need no orientation.

## 0.7.1 (2026-10-11)

The v0.7.0 tag's aarch64 wheels failed to build; 0.7.1 is the first release with
wheels for all platforms, and carries the fixes below.

### relax() on all of Brakke's samples

Each of the 25 sample datafiles was relaxed with defaults and compared with its own
`gogo` recipe; what that showed is fixed:

- Fixed: `relax()` crashed on strings ("Cannot equiangulate string model") and in the
  simplex representation ("V not implemented"): its tidying uses only the commands the
  model has. Bodies in the simplex representation (outside the Lagrange model) get no
  Newton steps: Evolver has no Hessian for them, and a failed attempt left the
  surface unusable.
- `relax()` on surfaces without a Hessian (gap energies on curved walls, crystalline
  integrands): plain gradient rounds settled to 1e-9 instead of stopping when the
  residual watch fired (tankex: energy 1.75 -> 1.50).
- Newton steps that raise the energy by more than 10% are undone, whatever the residual
  does: on a 100-cell foam a step ran off to a tangled surface (energy 19 -> 8e17) with
  a smaller residual, and `relax()` reported it converged.
- Fixed: undoing a gradient round or Newton step by the vertex coordinates when Evolver
  had changed the topology meanwhile (`autopop`, `autochop`) wrecked the surface. Such
  undos are now skipped (new binding: the topology stamp).
- Elastic energies (Evolver's elastic methods, such as `relaxed_elastic_A` in
  `mylarcube.fe`): no mesh tidying (it corrupted the facets' reference shapes), a
  residual over all force directions, Newton by `hessian_seek`.
- Elastic and crystalline energies keep relax()'s rounds of 10 steps for measuring
  convergence (only the tidying is skipped): `tidy=0` measured step to step and stopped
  early (`mylarcube.fe` along its recipe's modulus ramp: -1.2518 -> -1.2563, recipe
  -1.2577).
- Facet angles in quotient spaces from Evolver's edge lengths too (`symtest.fe`).
  `symtest.fe` reads as unstable and `twointor.fe`, the same surface, doesn't: rightly,
  since `symtest.fe`'s volumes are only comments, so its cells can trade volume
  (Evolver's `saddle` lowers the energy).
- Crystalline (Wulff) energies: no mesh tidying in `relax()` (flipping and averaging undid
  the facets' alignment with the crystal: `crystal.fe` 3.608 -> 3.3256, as plain gradient
  steps reach). Their residual stays above zero at the minimum (the energy has a corner
  there), so `relax()` takes gradient rounds only and counts them converged once the
  energy settles; `health()` leaves their residual out.
- `ev.residual()`: plain forces under `area_normalization` (it divided each by its
  vertex's area), and dimensionless for strings and simplices too.
- `ev.health()`: `junctions` counts junctions that break Plateau's laws (`octa.fe`: 7,
  popped by `o`): edges where four or more films meet, and vertices whose tangent cone
  `o` would pop, found as `o` finds them (new binding). No more false alarms from fixed vertices drawing a wall (mound's table
  top) or from contact lines on a constraint drawn twice (slidestr); facet angles in a
  torus from Evolver's edge lengths (wrapped coordinates made the foams' facets look
  degenerate); planar and simplex surfaces handled.
- `relax()` pops junctions that break Plateau's laws (Evolver's `o`) and relaxes again;
  if the surface still won't converge, edges collapsed below a 20th of the median are
  deleted (`pinch=True`, the default; a converged surface never loses edges).
  `octa.fe` and `popstr.fe` now reach the energies of Brakke's recipes on defaults.
  `IterationResult.popped` and `.collapsed` count them. The deletion runs as a last
  resort: untidied gradient steps let a shrinking neck or film collapse (the tidying
  held it open as slivers), and the outcome is kept only if it converges or lowers the
  energy on a sound mesh. `cat.fe` and the stress catenoid past its fold pinch off into
  two discs (stress case 2 now checks the discs). After k failed attempts in a row,
  the next 2^k - 1 calls skip the last resort (without that it made the drainage
  stress case on plain defaults, with its ~120 unconverged steps, about 3x slower).
- Edge deletion refuses only merges that would make two facets one (the case that lost
  surfaces in Evolver 2.70a). The check from 0.7.0 also refused shared neighbours and
  parallel edges without that, and loop edges, which is how a collapsing neck's ring
  of edges closes: `cat.fe`'s recipe (`t .05; o`) now pinches the catenoid into two
  discs as in 2.70a (13.71288; was stuck at 15.53).
- Fixed: stress case 3 counted only its second sweep's time and steps.

### Found afterwards

- Fixed: `ev.newton()` couldn't meet a new volume target (`ev.body(1).target = 2.0`
  then `ev.newton()` took no step; also in 0.7.0). The residual doesn't see an unmet
  volume, so the safeguard read the step restoring it as making things worse. A step
  that brings a missed volume or fixed-quantity target (more than 1e-4 off) at least
  halfway is now accepted, and the energy rise allowed is 10% plus what meeting the
  targets costs (pressure times the change). (With a threshold of 1e-6 instead, the
  2e-5 that vertex averaging leaves let nearly every step skip the safeguard: the
  drainage stress case's worst band pressure went from 6% to 89% off.) This hit `relax()` too: the last vertex averaging
  leaves a volume slightly off, and the conjugate-gradient pass was what rescued a
  drop spread to 10 degrees; Newton now converges it directly.
- Fixed: on a degenerate mesh, `relax()`'s last vertex averaging could lose a body's
  volume without a word (`tankex.fe` from 8 facets: 0.354 against 0.6), and plain
  gradient rounds slid into what the degenerate mesh allows (energy 1.6 -> 0.03). A
  gradient step now follows tidying that leaves a volume off; a round that pushes a
  met volume off its target is undone and ends the rounds; the gradient rounds that
  replace Newton (no Hessian) run only while no facet is below 1 degree. `ev.health()`
  reports volumes and fixed quantities more than 1e-4 off their targets (`target_gap`).
- Fixed: the last resort crashed when edge deletion left a facet without its three
  edges (it now counts as unsound and the surface goes back).
- The last resort keeps an unconverged lower energy only if the residual didn't rise
  (`tankex.fe` from 8 facets, at 2 threads: energy 1.77 -> 0.24 by collapsing facets,
  residual 0.94 -> 1.38, and that was kept).
- `relax()` warns when `newton` or `stability=True` is asked for a surface without a
  Hessian (crystalline energies, simplex bodies outside the Lagrange model); the
  defaults are now `None` (20 Newton steps, stability checked).
- `settled` describes the gradient rounds, not an undone conjugate-gradient pass.
- The wheels workflow runs on main only when what goes into a wheel changes.
- `health()` skips the cone analysis on films without junctions (no edge where three
  films meet): 55 ms at 98k facets, on every `relax()`.
- `relax()` checks stability when it ends within 100 times `tol` too (a continuation
  step stopping just short skipped the check).
- `IterationResult.settled`: the energy settled with the residual above `tol`
  (surfaces without a Hessian; shown as `not converged (energy settled)`).
- `ev.restore()` keeps `relax()`'s memory of failed attempts (it reloads a dump, which
  counted as a new surface).
- Drainage stress case on plain defaults, after all of the above: about 2.2x faster
  than before the last-resort backoff (at 2 threads, 4 runs each: 500 s against
  790-1210 s). Median band pressure 0.2% off as before; the worst state varies from run
  to run (the case is path-sensitive, also at a fixed thread count): mostly 3-8%, but
  over 10% in about a third of 11 runs before (once 147%) and a quarter of some 30
  runs now, so no change in the tail.
- The wheels workflow also runs on pushes to main and weekly (the aarch64 wheels broke
  unnoticed until the release tag built them).

### Docs

- Five notebooks on Brakke's 25 sample datafiles (films and foams, drops and capillarity,
  crystals and membranes, strings and other dimensions, quadratic and Lagrange): each
  sample's own recipe against a default `relax()` with as many refinements, with a
  table and pictures (`docs/brakke_helpers.py`). Among them Kelvin's foam against
  Weaire-Phelan's (0.35% less area per cell), and the Lagrange ladders of `cube.fe` and
  `catman.fe` (`relax()` agrees with the recipes to 1e-14; `catman.fe`'s own `gogo3`
  compares with the wrong catenoid).
- `tools/make_brakke_notebooks.py` generates the Brakke notebooks;
  `tools/check_doc_snippets.py` runs the code on `stress_cases.md` (or another page)
  and compares the outputs shown there (`--update` refreshes them).
- Stress cases: figures for each case (result against its reference, error per step,
  what `relax()` did, and the surface at a few steps), drawn by
  `bench/stress/figures.py`.

## 0.7.0 (2026-10-10)

`ev.relax()` now takes a surface to a verified equilibrium with its defaults: it
reports convergence, stability and the mesh's health, and the examples use it as is.
Of the nine hard cases in `bench/stress/`, seven pass on plain defaults. A friendlier
API (phase F) comes with it: remeshing, live body handles, constraint helpers,
continuations and diagnostics as values.

### Robust by default (phase G)

- Fixed: `relax()` failed in the Lagrange model ("Cannot equiangulate LAGRANGE model"):
  its tidying rounds now only average vertices there.
- `ev.stability()` (and so `relax()`'s check) needs `ritz` only when some eigenvalue is
  below zero; a stable surface costs one factorization. `relax()` at 98k facets:
  80 -> 1.0 s, at 393k: 275 -> 4.2 s.
- `relax()` in the quadratic and Lagrange models first tries five gradient steps and then
  safeguarded Newton steps while each cuts the residual by 10% (the manual's
  `g 5; hessian`), and keeps them if they converge (volumes at their targets
  included); otherwise it goes back and relaxes as before, and skips the attempt on its
  next call. The Lagrange stage of a 24k-facet cube: order 2 2.7 -> 1.2 s, order 4
  6.1 -> 5.2 s; the tutorial's double bubble at 45k facets reaches the manual ladder's
  energies at every order (without the settling steps, Newton diverged there at order 2). Not in the linear model, where continuations with travelling contact
  lines lost accuracy without the gradient rounds' mesh upkeep.
- `relax()` (linear model), once converged, splits the edges leaving a contact line that
  are longer than 1.5 times the contact-line edges there and relaxes again, kept only
  if that converges. A drop stepped to 160 degrees now comes out within 1.5 degrees of
  its contact angle (was 3.8 off; 170: 1.7, was 7.6), the barrel on a fibre gets
  better-shaped facets (smallest angle 49 vs 13 degrees). A cap shrinking evenly is
  left alone.
- Faster safeguards: a safeguarded Newton step undoes itself by the vertex
  coordinates instead of a dump (`ev.newton(1)` at 393k facets 3.85 -> 1.77 s, raw
  `hessian` 1.2 s), `relax()` no longer dumps the surface at its start, and
  `ev.residual()` finds the shape directions in parallel (393k facets, 8 threads:
  0.39 -> 0.14 s).
- `relax()` reorganized: the experimental `remesh=True` and `IterationResult.remeshes` are
  gone (`ev.adapt()` and `ev.remesh()` remain); a result prints as a summary
  (`IterationResult(converged after 130 gradient steps and 2 Newton, residual 2.8e-09,
  stable, health ok)`), and so does `Health`. Health messages name named constraints.
- `relax()` hands over to Newton sooner: `energy_tol` now defaults to 1e-5 when Newton
  steps follow (1e-9 with `newton=0`). The tidying in every round keeps the energy from
  settling to 1e-9, so the gradient phase used to run its full 1000 steps, long enough
  for slow mesh drift to wreck a surface (contact-line vertices sliding together on a
  coarse liquid bridge). Stress suite: 7 of 9 (the bridge passes), far fewer gradient
  steps (cylinder 26000 -> 1060, shrinking cap 25880 -> 3040).
- `relax()` undoes a gradient round that more than doubles the residual (when Newton
  follows) and hands over to Newton: on a coarse start a contact line can collapse
  within a few steps. The liquid bridge relaxes on plain defaults (`relax(levels=3)`)
  at every gap and contact angle of its notebook, which no longer needs its own
  relaxation recipe.
- `relax(cg="auto")` (new default): when the first pass ends unconverged, a second pass
  of conjugate gradient steps (at most 200, averaging vertices) and Newton, kept only if
  it converges. A sessile drop spread to 10 degrees now converges. `cg=True` still uses
  conjugate gradients throughout; `cg=False` never.
- `ev.adapt(max_turn=15)`: splits the edges along which the surface turns by more than
  `max_turn` degrees (between the vertex normals at their ends), one level per call,
  never coarsening; `relax(adapt=True)` (opt-in) alternates it with relaxation, up to
  three passes or 4x the facets. A sessile drop at 170 degrees: implied contact angle
  off by 3.5 degrees with 734 facets, against 3.9 with 1920 refined uniformly.
- `ev.adapt()` also coarsens (`coarsen=True`): short interior edges along which the
  surface turns by less than a third of `max_turn` are merged, unless that would tilt a
  facet by more than `max_turn/2`, flip one, or make an edge longer than 4/3 of the bulk
  length. Engine: Evolver's edge deletion checks this when the new variables
  `collapse_max_tilt` (degrees) and `collapse_max_edge` are set (both 0 by default:
  no change for Evolver scripts).
- `ev.health()` and `relax()`'s `result.health` (`pyse.Health`): residual, stability,
  smallest facet angle and skinny facets, the nearest approach to a wall or mirror a vertex
  isn't on (beyond two edges of its contact line) and of the surface to itself (in edge
  lengths), vertices on the far side of a constraint, and `issues` in words (empty: `ok`).
  No warnings. About 0.2 s at 98k facets. New C binding: per-vertex signed distances to
  the constraints.
- The drainage helper (docs) relaxes with plain `ev.relax(stability=False)` after its
  remeshing; its Newton rollback (`undo_if` on the mirror gap) is no longer needed
  (band pressures within 0.02% median of the stored level-3 run, snap 0.4% off).
- Newton and eigenvalue counts no longer see spurious modes from vertices on wires
  (curves where two constraints meet): their only freedom, sliding along the wire, is
  left to the gradient steps (`hessian_slant_cutoff` now 0.05, and its test uses the
  raw surface normal and the freedom's magnitude). A stable catenoid counts 0 negative
  eigenvalues (was about one per wire vertex).
- `ev.residual()` balances the pressures (volume and quantity multipliers) in the shape
  directions, as Newton does; Evolver's own fit over the full velocities, tangential
  mesh forces included, left a uniform remainder at a Newton solution (the double
  bubble stalled at 5.9e-7; now 4e-11).
- The examples (tutorial, liquid bridge, slit droplet, drainage) relax with plain
  `ev.relax()` / `ev.relax(levels=3)`.
- `ev.stability()` and, once `relax()` converges, `result.stable` /
  `result.negative_modes` with an `UnstableEquilibriumWarning`: unstable equilibria
  (a liquid column past Rayleigh-Plateau, a barrel drop rolling up) are reported
  instead of silently returned; symmetry zero modes are not mistaken for them.
- `ev.residual()`: how far the surface is from equilibrium (the normal part of the
  projected vertex forces, dimensionless; 0 at an equilibrium). New C binding.
- `relax()` defaults: rounds of 10 gradient steps with equiangulation and vertex
  averaging, then up to 20 safeguarded Newton steps, converged when the residual is
  below `tol` (now a residual tolerance, default 1e-8; `energy_tol` ends the gradient
  phase). A NaN surface is put back to the start of the gradient round where it
  happened and raised, not reported as converged.
- Newton steps (`relax`, `ev.newton()`) are safeguarded: plain Newton first, a line
  search if that is rejected; a step that makes things non-finite, or raises both the
  energy and the residual, is undone.
- `bench/stress/`: nine hard cases with exact or axisymmetric references, run with
  default settings.

### API (phase F: a friendlier API)

- `ev.relax(...)` does the whole recipe: `levels=n` (refine and relax again),
  `tidy=k` (equiangulation and vertex averaging every k steps, convergence measured
  round to round),
  `newton=n` with `seek=` and `undo_if=` (a Newton step is undone when
  `undo_if(ev)` is true). Replaces `hessian=`/`max_hessian=`, and
  `IterationResult.hessian_steps` is now `newton_steps` (breaking); the result has a
  per-iteration `level`. New `ev.newton(steps, seek=, tol=, undo_if=)`.
- `ev.remesh(target=h)` (or `max_edge=`, `min_edge=`): deletes short edges, splits
  long ones and equiangulates; `protect=` keeps an edge mask whole. Splitting honours
  `no_refine` (Evolver's `l` ignores it).
- Diagnostics as values: `ev.eigen_counts(shift)` (negative/zero/positive Hessian
  eigenvalues: an unstable equilibrium shows as `negative > 0`), `ev.check()` (topology
  problems, empty when sound), `Mesh.quality()` / `ev.mesh_quality()` (edge lengths,
  smallest angles, skinny and degenerate facets).
- Mirror images for symmetric pieces: `ev.plot(mirror=["z", "y", ("x", c)])`,
  `ev.live_view(mirror=...)` (the images share the data, so updates stay fast), and
  `Mesh.mirrored(planes)` for one combined linear mesh.
- `pyse.constraints`: planes, mirrors, spheres and cylinders with contact angles, for
  `make_datafile(constraints=)`: they write the wetting-energy and volume line
  integrals (and give the sphere's `volconst` and energy constant for wet poles);
  `Body(volconst=)`.
- `pyse.recipes.continuation(ev, set_value, values, relax=, checkpoint=, resume=)`:
  follows a family of equilibria (a body's volume, a parameter), with checkpoints that
  a stopped run resumes from; setters `recipes.body_target(i)`, `recipes.parameter(name)`.
- `ev.body(i)`: a live handle on one body (`.volume`, `.target` and `.volconst`
  settable, `.pressure`, `.fixed`); `target = None` frees the volume.
- `Bodies.target_volume` is now `Bodies.target` (breaking).
- `ev.on_constraint(k, element="vertex")`, `Mesh.edges_touching(mask, how=)` and
  `ev.set_flag(element, flag, where=, on=)` select elements with numpy masks instead
  of Evolver `where` clauses.
- `EvolverError.command` holds the failing command; errors hint at numpy reprs
  (`np.float64(...)`) and non-finite numbers in the command text.

### Fixes

- The engine is compiled without fused multiply-adds (`-ffp-contract=off`): GCC fused
  them on aarch64, which shifted unconverged runs by up to 1e-5 against x86 and
  Evolver's reference values. No change on x86-64 builds.
- Tiny-edge deletion (`t`, also used by `w` and `delete`) no longer merges the ends of an
  edge when they share neighbours other than the third vertices of its facets (the link
  condition). Evolver 2.70a merged them, making parallel and loop edges; later deletions
  then freed elements twice, hung in `edge_valence()`, or lost the surface.

### Docs

- New pages: "Relaxing a surface" (what `relax()` does, reading its result, mesh
  resolution, limits) and "Stress cases" (eight hard problems against exact references,
  with defaults; helpers in `docs/stress_helpers.py`). `docs/performance.md` compares
  `relax()` with the hand-written commands at scale (1.3-1.9x their time, for the
  convergence, stability and health checks).
- New example, `slit_drainage`: liquid drained from a bead chain in a slit, from immersed
  beads through bead emergence and the band to the snap. Runs a coarse case live and shows
  stored full runs and movies for three bead spacings.

### Speed

- Quantity mode: the value and gradient passes of the fast element loop sum into
  per-thread buffers merged in thread order, instead of recording every entry for a
  serial replay (`g 10` in quantity mode at 98k facets, 8 threads: 0.83 -> 0.61 s;
  results unchanged and deterministic).

## 0.6.0 (2026-10-08)

### Speed

Cube sample, 8 threads, against 0.5.0 (`bench/benchmark.py`): `g 1` at 98k facets
75 -> 10 ms, at 1.6M 1.34 -> 0.40 s; a Newton step at 98k 0.49 -> 0.10 s, at 1.6M
25.5 -> 2.8 s. Against the original Evolver 2.70a see `docs/performance.md`
(Lagrange 6 Newton step at 24k facets: 22.9 s there, 1.42 s here).

- Newton steps: MUMPS factoring (instead of Evolver's own), with its analysis kept
  across steps; the matrix pattern kept across steps and filled in parallel; vertex
  normals in parallel; all volume and quantity constraint columns solved at once.
- Lagrange models: the facet setup as a packed, vectorized kernel; the area and
  volume element Hessians as BLAS matrix products; each element's terms projected
  and entered once.
- Gradient steps: parallel named quantities (facet, edge and vertex integrals);
  per-thread sums kept on separate cache lines; attribute selections kept until the
  surface changes.
- `u` and `V`: the swap tests and vertex-average search in parallel, identical
  results (393k facets: `u` 0.30 -> 0.022 s, `V` 0.60 -> 0.12 s).
- Builds prefer the system's OpenMP OpenBLAS and link it statically with private
  symbols (no clash with NumPy's OpenBLAS); wheels bundle OpenBLAS 0.3.34 (OpenMP).

### Threads

- Default thread count: physical cores; `pyse.set_threads()`, `pyse.threads_limit()`
  and `threads=` per call; `pyse.map` workers run single-threaded.
- Safe Ctrl-C with threads: aborts wait until the parallel region or MUMPS call ends.
- Results agree with Evolver's serial loops to round-off and are reproducible to
  round-off from run to run.

### Library

- `ev.body_surfaces(cap=True)`: closed per-body surfaces, openings capped on the
  constraint their rim lies on (points projected by Evolver); `BodySurface.cap_ids`
  and `cap_constraints`; `BodySurface.volume_mesh()` for tetrahedral meshes with
  Gmsh (physical groups for the surface, caps and body).
- Notebook display: HTML summaries of `Evolver`, `Mesh`, bodies and parameters
  (`jupyter` extra); faster live view; a warning before very large tessellations.
- `busy_timeout` / `EvolverBusyError`: optionally wait for a running call from
  another thread instead of failing at once.
- `pyse.map` fails fast with `WorkerStartError` when workers can't start.

### Breaking changes

- `Mesh` arrays renamed to `facets`, `facet_ids`, `facet_bodies`; unused low-level
  bindings (`edges`, `facets`, `element_nodes`) removed.

### Fixes

- Edges with a density in a soapfilm model with a metric crashed (stack overflow,
  also in the original Evolver); they now use the metric edge length.
- Switching `bezier_basis` in a Lagrange model gave wrong energies (stale cached
  basis tables).
- Double free after a failed Newton step; signed overflow in the sparse Hessian
  hash; `metis_factor` falling through to the MKL setup; METIS 5 API.
- The torus display mode no longer carries over to the next loaded datafile.

### Documentation and tooling

- Sphinx documentation (API reference, double-bubble tutorial, liquid-bridge and
  slit-droplet examples with mesh export), published to GitHub Pages by CI.
- Performance comparison with the original Evolver 2.70a.
- MIT license for pySE's own code; third-party license notices shipped in wheels.
- Sanitizer runs include an OpenMP build with Newton steps, `u` and `V`;
  `tools/valgrind.supp`.
