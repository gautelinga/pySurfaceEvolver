"""A bead pack's drainage as an XDMF time series for ParaView.

One run of the topology corpus (see corpus.py), every step's state kept,
then written as one XDMF file at equally spaced volumes of the liquid: each
frame the nearer of the states the run kept either side of that volume,
taken to it by Newton steps (where that can't be without a change of
topology, from either side, the nearer state as it is, at its own volume).
A grid a frame, since the surface's topology changes; the time is the share
of the liquid drained (the liquid: the draining body and the trapped ones),
the heights a point field, each facet's body (the draining liquid 1, each
trapped cluster its own) and whether it is trapped cell fields; the data in
HDF5 beside it. The beads and the box go into static .vtp files, and a ParaView script
opens all three.

usage: python bench/topology/series.py [--beads 16] [--seed 0] [--theta 15]
                                       [--h 0.2] [--ds -0.5] [--goal breakthrough]
                                       [--threads 1] [--out DIR]
                                       [--frames 200 | --dv DV | --every-step]
       python bench/topology/series.py --from CKDIR [--beads 16] [--seed 0] [--out DIR] ...

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


def at_volume(states, V):
    """The surface holding liquid V (the draining body and the trapped
    ones) from the nearer of two states the run kept, either side of V
    (else the other), the draining body's target set to V less what that
    state holds trapped: Newton steps there, as the run's corrector takes,
    else relax(). Not where the surface has gone through a solid on the way
    (a step there needs a change of topology first). (ev, step, volume,
    how), how None where neither got there: the nearer state as the run
    kept it."""
    import corpus
    from pysurfaceevolver import recipes
    for k, path, Vk, Tk in sorted(states, key=lambda s: abs(s[2] - V)):
        for how in ("Newton steps", "relaxed"):
            ev, v0, _ = corpus.restore_state(path)
            ev.body(1).target = V - Tk
            if how == "Newton steps":
                ok = recipes._arc_back(ev, lambda v: None, V - Tk, 1e-8, None, None)[0]
            else:
                result = ev.relax(stability=False)
                ok = result.converged and not getattr(result, "collapsed", 0)
            if ok and not recipes._through(ev.events()):
                return ev, k, V, how
    k, path, Vk, _ = min(states, key=lambda s: abs(s[2] - V))
    ev, _, _ = corpus.restore_state(path)
    return ev, k, Vk, None


def write(ck, n, L, seed, out, base, frames=200, dv=None):
    """The states in ck (stepNNNN.fe and .pkl) as base.xdmf and base.h5,
    the beads and box as base_beads.vtp and base_box.vtp, and open_base.py.
    The series is at equally spaced liquid volumes (``frames`` of them from
    the start to the end, or ``dv`` apart): each from the states the run kept
    either side of that volume (at_volume). frames=None: every state kept,
    as it is."""
    import warnings
    import numpy as np, h5py
    sys.path.insert(0, HERE)
    import corpus
    from pack import settle
    warnings.simplefilter("ignore")
    paths = sorted(glob.glob(os.path.join(ck, "step*.fe")))
    if not paths:
        sys.exit(f"no states (stepNNNN.fe) in {ck}")
    # (the liquid: body 1, the one the run drains, and the trapped bodies,
    # cut off, at the volumes they hold)
    kept = []
    for path in paths:
        ev, v0, _ = corpus.restore_state(path[:-3])
        volumes = np.asarray(ev.bodies().volume)
        kept.append((int(os.path.basename(path)[4:8]), path[:-3], float(volumes.sum()),
                     float(volumes.sum() - ev.body(1).volume)))
    # (at a fold the run falls at one volume and the branch may turn back a
    # little: the volumes it has got down to)
    low = np.minimum.accumulate([V for _, _, V, _ in kept])
    if frames is None and dv is None:
        wanted = [(i, None) for i in range(len(kept))]
    else:
        if dv is None:
            dv = (low[0] - low[-1])/(frames - 1)
        targets = low[0] - dv*np.arange(int((low[0] - low[-1])/dv + 1e-9) + 1)
        wanted = [(int(np.flatnonzero(low >= V - 1e-9*v0)[-1]), float(V)) for V in targets]
    h5name = base + ".h5"
    grids, top, as_is = [], None, 0
    with h5py.File(os.path.join(out, h5name), "w") as h5:
        for j, (i, V) in enumerate(wanted):
            k, path, Vk, _ = kept[i]
            ev, v0, _ = corpus.restore_state(path)
            label, vol = f"step {k}", Vk if V is None else V
            if V is not None and abs(Vk - V) > 1e-6*v0:
                ev, k, vol, how = at_volume(kept[i:i + 2], V)
                if how is None:
                    as_is += 1
                label = f"step {k} as it is" if how is None else f"step {k} {how} to V {V:.4f}"
            m = ev.mesh()
            x = np.asarray(m.vertices, np.float64)
            f = np.asarray(m.facets, np.int64)
            g = h5.create_group(f"s{j:04d}")
            g["xyz"], g["tri"], g["z"] = x, f, x[:, 2]
            # (each facet's liquid: its body, a cluster, and whether trapped)
            body = np.asarray(m.facet_bodies).max(axis=1)
            g["body"], g["trapped"] = body.astype(np.int64), (body > 1).astype(np.int64)
            grids.append((j, len(x), len(f), (v0 - vol)/v0, label))
            if top is None:
                top = float(x[:, 2].max()) + 0.2
    # (the time: the share drained, never back; states at the same share a
    # little apart, so ParaView keeps them in order)
    times, last = [], -1.0
    for *_, drained, _ in grids:
        last = max(drained, last + 1e-9, 0.0)
        times.append(last)
    xml = ['<?xml version="1.0" ?>', '<Xdmf Version="3.0">', ' <Domain>',
           '  <Grid Name="liquid surface" GridType="Collection" CollectionType="Temporal">']
    for (k, nv, nf, _, label), t in zip(grids, times):
        item = f'NumberType="{{}}" Precision="8" Format="HDF">{h5name}:/s{k:04d}'
        xml += [f'   <Grid Name="{label}" GridType="Uniform">',
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
                '    <Attribute Name="body" AttributeType="Scalar" Center="Cell">',
                f'     <DataItem Dimensions="{nf}" {item.format("Int")}/body</DataItem>',
                '    </Attribute>',
                '    <Attribute Name="trapped" AttributeType="Scalar" Center="Cell">',
                f'     <DataItem Dimensions="{nf}" {item.format("Int")}/trapped</DataItem>',
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
    spacing = "every state kept" if dv is None else f"volumes {dv:.4g} apart"
    print(f"{os.path.join(out, base + '.xdmf')}: {len(grids)} frames ({spacing}), drained "
          f"0 to {times[-1]:.3f}" + (f"; {as_is} not relaxed (no equilibrium there without "
                                     f"a change of topology)" if as_is else ""))


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
    ap.add_argument("--frames", type=int, default=200,
                    help="frames at equally spaced volumes, start to end (default 200)")
    ap.add_argument("--dv", type=float, help="the volume between frames, instead of --frames")
    ap.add_argument("--every-step", action="store_true",
                    help="a frame per state kept, as it is, instead")
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
    write(os.path.abspath(ck), n, L, args.seed, out, f"pack{n}_s{args.seed}_t{args.theta:g}",
          frames=None if args.every_step else args.frames, dv=args.dv)


if __name__ == "__main__":
    main(sys.argv[1:])
