"""A film of a corpus run: its states (corpus.py --frames, or its checkpoints)
drawn with PyVista off screen, then put together with ffmpeg.

usage: python bench/topology/film.py STATES_DIR OUT.mp4 [--fps 4] [--hot 25 26 ...]

STATES_DIR holds stepNNNN.fe/.pkl (corpus.save_state). Each frame: the gas-
liquid interface (blue), the beads faint (the --hot ones red), the floor;
two views a quarter turn apart. The frames stay beside OUT (OUT minus .mp4).
ffmpeg: imageio_ffmpeg's, else the one on the path.
"""
import argparse, glob, os, shutil, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def frame(state, path, hot):
    import pyvista as pv
    sys.path.insert(0, HERE)
    import corpus
    from pysurfaceevolver._engine import surface
    ev, _, step = corpus.restore_state(state)
    L = float(ev.periods[0]) if getattr(ev, "periods", None) else None
    liquid = ev.mesh().to_pyvista()
    beads = [(str(k), np.asarray(w.slabflux[0], float)) for k, w in surface.walls.items()
             if getattr(w, "slabflux", None)]
    top = max(c[2] for _, c in beads) + 1.0 if beads else float(liquid.bounds[5])
    lo, hi = (0.0, L) if L else (float(liquid.bounds[0]), float(liquid.bounds[1]))
    mid = np.array([(lo + hi)/2, (lo + hi)/2, top/2])
    p = pv.Plotter(off_screen=True, window_size=(1600, 1000), shape=(1, 2), border=False)
    for panel, turn in enumerate((0.0, np.pi/2)):
        p.subplot(0, panel)
        p.set_background("white")
        p.add_mesh(liquid, color="#2f7fd6", smooth_shading=True)
        shifts = [(dx, dy) for dx in (-L, 0, L) for dy in (-L, 0, L)] if L else [(0, 0)]
        for key, c in beads:
            for dx, dy in shifts:
                cc = c + (dx, dy, 0)
                if L and (cc[0] < lo - 1 or cc[0] > hi + 1 or cc[1] < lo - 1 or cc[1] > hi + 1):
                    continue
                h = key.replace("bead", "") in hot
                p.add_mesh(pv.Sphere(radius=1.0, center=cc, theta_resolution=28, phi_resolution=28),
                           color="#d0483a" if h else "#9c9a92", opacity=0.45 if h else 0.1,
                           smooth_shading=True)
        p.add_mesh(pv.Plane(center=(mid[0], mid[1], 0), direction=(0, 0, 1), i_size=hi - lo,
                            j_size=hi - lo), color="#e9e7e0")
        eye = mid + np.array([np.cos(turn - 0.9), np.sin(turn - 0.9), 0.35])
        p.camera_position = [tuple(eye), tuple(mid), (0, 0, 1)]
        p.reset_camera(bounds=(lo, hi, lo, hi, 0.0, top))     # (the cell, seen whole)
        if panel == 0:
            p.add_text(f"step {step}   V {ev.body(1).volume:.3f}   facets {ev.counts['facets']}",
                       font_size=12, color="black")
    p.screenshot(path)
    p.close()


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("states")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--hot", nargs="*", default=[])
    args = ap.parse_args(argv)
    import pyvista as pv
    pv.OFF_SCREEN = True
    frames = os.path.splitext(args.out)[0]
    os.makedirs(frames, exist_ok=True)
    states = sorted(glob.glob(os.path.join(args.states, "step*.pkl")))
    for k, s in enumerate(states):
        frame(s[:-4], os.path.join(frames, f"f{k:05d}.png"), set(args.hot))
        print(f"{k + 1}/{len(states)}", flush=True)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit(f"no ffmpeg: the frames are in {frames}")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", os.path.join(frames, "f%05d.png"),
                    "-vf", "scale=1600:-2,format=yuv420p", "-c:v", "libx264", "-crf", "18",
                    args.out], check=True)
    print(f"{len(states)} frames: {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
