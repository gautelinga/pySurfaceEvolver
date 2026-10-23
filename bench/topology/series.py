"""A bead pack's drainage as an XDMF time series for ParaView.

One run of the topology corpus (see corpus.py), every step's state kept,
then the states written as one XDMF file (a grid per step, since the
surface's topology changes; the time is the share of the liquid drained,
the heights a point field) with its data in HDF5 beside it. The beads and
the box go into static .vtp files, and a ParaView script opens all three.

usage: python bench/topology/series.py [--beads 16] [--seed 0] [--theta 15]
                                       [--h 0.2] [--ds -0.5] [--goal breakthrough]
                                       [--threads 1] [--out DIR]
       python bench/topology/series.py --from CKDIR [--beads 16] [--seed 0] [--out DIR]

The first form runs the pack (at one thread a run is deterministic) and
writes the series; --from writes the series of states a run already kept
(corpus.py --one ... --checkpoints CKDIR, kept with corpus.KEEP large
enough). DIR defaults to bench/topology/out/series_NAME; open the result
with  paraview --script=open_NAME.py  from DIR. Needs h5py and pyvista.
"""
import argparse, glob, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = {4: 4.2, 16: 5.0, 64: 10.0}       # the corpus's box sides by bead count


def run(name, n, L, seed, theta, h, ds, goal, out, threads):
    """The run, in its own process (as corpus.py runs one), every step kept
    in out/ck."""
    code = (f"import sys; sys.path.insert(0, {HERE!r}); import corpus; corpus.KEEP = 10**6; "
            f"corpus.main(sys.argv[1:])")
    args = ["--one", name, str(n), str(L), str(seed), str(theta), str(h), str(ds), out, goal,
            "--checkpoints", os.path.join(out, "ck")]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    subprocess.run([sys.executable, "-c", code] + args, env=env, check=True)
    with open(os.path.join(out, f"{name}.log")) as f:
        ends = [line for line in f if line.startswith("end")]
    print(ends[-1].strip() if ends else "no end line in the log", flush=True)


def write(ck, n, L, seed, out, base):
    """The states in ck (stepNNNN.fe and .pkl) as base.xdmf and base.h5,
    the beads and box as base_beads.vtp and base_box.vtp, and open_base.py."""
    import numpy as np, h5py
    sys.path.insert(0, HERE)
    import corpus
    from pack import settle
    paths = sorted(glob.glob(os.path.join(ck, "step*.fe")))
    if not paths:
        sys.exit(f"no states (stepNNNN.fe) in {ck}")
    h5name = base + ".h5"
    grids, top = [], None
    with h5py.File(os.path.join(out, h5name), "w") as h5:
        for path in paths:
            ev, v0, _ = corpus.restore_state(path[:-3])
            m = ev.mesh()
            x = np.asarray(m.vertices, np.float64)
            f = np.asarray(m.facets, np.int64)
            # (the liquid: every body, once a cut has split it)
            liquid = sum(float(ev.body(b).volume) for b in ev.bodies().ids)
            k = int(os.path.basename(path)[4:8])
            g = h5.create_group(f"s{k:04d}")
            g["xyz"], g["tri"], g["z"] = x, f, x[:, 2]
            grids.append((k, len(x), len(f), (v0 - liquid)/v0))
            if top is None:
                top = float(x[:, 2].max()) + 0.2
    # (the time: the share drained, never back; states at the same share a
    # little apart, so ParaView keeps them in order)
    times, last = [], -1.0
    for *_, drained in grids:
        last = max(drained, last + 1e-9, 0.0)
        times.append(last)
    xml = ['<?xml version="1.0" ?>', '<Xdmf Version="3.0">', ' <Domain>',
           '  <Grid Name="liquid surface" GridType="Collection" CollectionType="Temporal">']
    for (k, nv, nf, _), t in zip(grids, times):
        item = f'NumberType="{{}}" Precision="8" Format="HDF">{h5name}:/s{k:04d}'
        xml += [f'   <Grid Name="step {k}" GridType="Uniform">',
                f'    <Time Value="{t:.9f}"/>',
                f'    <Topology TopologyType="Triangle" NumberOfElements="{nf}">',
                f'     <DataItem Dimensions="{nf} 3" {item.format("Int")}/tri</DataItem>',
                '    </Topology>',
                '    <Geometry GeometryType="XYZ">',
                f'     <DataItem Dimensions="{nv} 3" {item.format("Float")}/xyz</DataItem>',
                '    </Geometry>',
                '    <Attribute Name="height" AttributeType="Scalar" Center="Node">',
                f'     <DataItem Dimensions="{nv}" {item.format("Float")}/z</DataItem>',
                '    </Attribute>',
                '   </Grid>']
    xml += ['  </Grid>', ' </Domain>', '</Xdmf>']
    with open(os.path.join(out, base + ".xdmf"), "w") as f:
        f.write("\n".join(xml) + "\n")

    import pyvista as pv
    beads, _, _ = settle(n, L, L, seed=seed)
    spheres = [pv.Sphere(radius=1.0, center=c, theta_resolution=40, phi_resolution=40)
               for c in beads]
    spheres[0].merge(spheres[1:]).save(os.path.join(out, base + "_beads.vtp"))
    pv.Box(bounds=(0, L, 0, L, 0, top)).save(os.path.join(out, base + "_box.vtp"))
    with open(os.path.join(out, f"open_{base}.py"), "w") as f:
        f.write(PARAVIEW.format(base=base, eye=[2.6*L, -2*L, 2.6*L], focus=[L/2, L/2, top/2]))
    print(f"{os.path.join(out, base + '.xdmf')}: {len(grids)} steps ({grids[0][0]} to "
          f"{grids[-1][0]}), drained 0 to {times[-1]:.3f}")


PARAVIEW = '''\
# ParaView: paraview --script=open_{base}.py   (run from the folder with the files)
import os
from paraview.simple import *
here = os.getcwd()
surface = XDMFReader(registrationName="liquid surface", FileNames=[os.path.join(here, "{base}.xdmf")])
beads = XMLPolyDataReader(registrationName="beads", FileName=[os.path.join(here, "{base}_beads.vtp")])
box = XMLPolyDataReader(registrationName="box", FileName=[os.path.join(here, "{base}_box.vtp")])
view = GetActiveViewOrCreate("RenderView")
s = Show(surface, view)
ColorBy(s, ("POINTS", "height"))
s.SetScalarBarVisibility(view, True)
s.RescaleTransferFunctionToDataRangeOverTime()
s.EdgeColor = [0.1, 0.2, 0.4]
s.SetRepresentationType("Surface With Edges")
b = Show(beads, view)
b.DiffuseColor = [0.79, 0.73, 0.6]
b.Opacity = 0.25
w = Show(box, view)
w.SetRepresentationType("Outline")
w.DiffuseColor = [0.3, 0.3, 0.3]
GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
view.ResetCamera()
view.CameraPosition = {eye}
view.CameraFocalPoint = {focus}
view.CameraViewUp = [0, 0, 1]
Render()
'''


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--beads", type=int, default=16, choices=sorted(BOX))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta", type=float, default=15.0)
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--ds", type=float, default=-0.5)
    ap.add_argument("--goal", default="breakthrough", choices=("breakthrough", "half"))
    ap.add_argument("--threads", type=int, default=1, choices=(1, 2, 3, 4))
    ap.add_argument("--out")
    ap.add_argument("--from", dest="ck", help="write the series of the states kept here")
    args = ap.parse_args(argv)
    n, L = args.beads, BOX[args.beads]
    tag = "_bt" if args.goal == "breakthrough" else ""
    name = f"n{n}_s{args.seed}_t{args.theta:g}_h{args.h:g}_ds{args.ds:g}{tag}"
    out = os.path.abspath(args.out or os.path.join(HERE, "out", f"series_{name}"))
    os.makedirs(out, exist_ok=True)
    ck = args.ck
    if ck is None:
        run(name, n, L, args.seed, args.theta, args.h, args.ds, args.goal, out, args.threads)
        ck = os.path.join(out, "ck")
    write(os.path.abspath(ck), n, L, args.seed, out, f"pack{n}_s{args.seed}_t{args.theta:g}")


if __name__ == "__main__":
    main(sys.argv[1:])
