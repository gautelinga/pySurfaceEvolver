# Several surfaces at once

Surface Evolver keeps its surface in C global variables, so a Python process holds one
surface: every `Evolver` object in it is a handle on the same one. There are three ways
to work with more than one:

* **`save()` and `restore()`** keep surfaces as snapshots in one process and switch
  between them. Cheap for a few, but only one is live at a time.
* **`pyse.map(fn, params)`** runs `fn(param)` in parallel worker processes for a sweep:
  each job builds its surface, works on it and returns numbers, arrays or a snapshot;
  the surfaces are gone when the jobs end.
* **`pyse.Evolver(process=True)`** gives a surface in a worker process of its own that
  stays there while you work with it, through a handle with the same API as any
  `Evolver`. This page is about this one.

## Drops side by side

Four drops of volume 1 on a table, at four contact angles, each in its own worker. A
call on a handle waits for its worker, so a thread per surface runs them at the same
time (here twice as fast as one after another). Then each drop is shrunk to half its
volume with {func}`~pysurfaceevolver.recipes.continuation`, which works on the handles as
on any surface, and its height compared with the exact cap's.

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes

def drop(theta):
    """A drop of volume 1 on a table at the contact angle theta, in a worker of its own."""
    cap = shapes.cap(radius=1, height=0.6)
    ev = pyse.Evolver(process=True, threads=1)
    ev.load_arrays(cap.vertices, cap.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=theta)},
                   vertex_constraints={1: cap.rims["rim"]}, bodies=[pyse.Body(volume=1.0)])
    return ev

angles = [30, 60, 90, 120]
drops = [drop(theta) for theta in angles]
print(len({ev.pid for ev in drops}), "worker processes")

# each call waits for its own worker, so threads relax the four at once
with ThreadPoolExecutor(len(drops)) as pool:
    results = list(pool.map(lambda ev: ev.relax(levels=2), drops))
for theta, ev, r in zip(angles, drops, results):
    print(f"{theta:3d} degrees: converged {r.converged}, pressure {ev.body(1).pressure:.4f}, "
          f"contact angle {ev.contact_angles(1).mean():.1f}")

# the surfaces stay: shrink each to half its volume, again all at once
def shrink(ev):
    steps = list(recipes.continuation(ev, ev.body(1), np.linspace(0.9, 0.5, 5)))
    return all(s.result.converged for s in steps)

with ThreadPoolExecutor(len(drops)) as pool:
    converged = list(pool.map(shrink, drops))
for theta, ev, ok in zip(angles, drops, converged):
    c = np.cos(np.radians(theta))         # the exact cap: V = pi R^3 (2 - 3c + c^3)/3
    exact = (1.5/(np.pi*(2 - 3*c + c**3)))**(1/3)*(1 - c)
    height = ev.mesh().vertices[:, 2].max()
    print(f"{theta:3d} degrees, volume {ev.body(1).volume:.2f}: height {height:.4f} "
          f"(exact {exact:.4f}), every step converged: {ok}")
for ev in drops:
    ev.close()
```

```text
4 worker processes
 30 degrees: converged True, pressure 0.7554, contact angle 29.5
 60 degrees: converged True, pressure 1.7366, contact angle 59.2
 90 degrees: converged True, pressure 2.5591, contact angle 89.0
120 degrees: converged True, pressure 3.0469, contact angle 119.2
 30 degrees, volume 0.50: height 0.2821 (exact 0.2816), every step converged: True
 60 degrees, volume 0.50: height 0.4577 (exact 0.4571), every step converged: True
 90 degrees, volume 0.50: height 0.6212 (exact 0.6204), every step converged: True
120 degrees, volume 0.50: height 0.7827 (exact 0.7816), every step converged: True
```

Interactively, the handles are just as usable one at a time: `drops[2].plot()` draws
that drop here (from a copy of its mesh), `drops[0].write("drop.stl")` writes a file
from the worker (relative paths are this process's), and
`drops[1].save()` returns a snapshot that any other `Evolver` can `restore()`.

## What crosses between the processes

* **Calls, attribute reads and assignments** go to the worker: `ev.relax()`,
  `ev.energy`, `ev.body(1).target = 2.0`, `ev.parameters["theta"] = 50`. Each costs a
  round trip of a fraction of a millisecond, plus copying its arguments and result.
* **Views stay in the worker:** `ev.body(1)`, `ev.quantity(name)` and `ev.parameters`
  come back as handles too (and pass `isinstance` checks, so code written for an
  `Evolver` works unchanged).
* **Results come back as copies:** meshes, `IterationResult`s, arrays, snapshots. A
  mesh of 25k facets takes about 10 ms to come over.
* **Functions given to methods** (`relax(callback=...)`, `undo_if=`) are copied to the
  worker and run there, with the worker's surface; what they print shows here, but
  changes they make to your objects stay in the worker. Sending a function defined
  inside another function, as a lambda or in a notebook needs
  [cloudpickle](https://github.com/cloudpipe/cloudpickle) (`pip install cloudpickle`).
* **Prints, warnings and errors** from the worker show here, as from a local surface.
  Ctrl-C interrupts the worker's call.

## Starting, stopping and threads

A worker starts in about 0.1 s: it is a fresh Python interpreter
(`python -m pysurfaceevolver._worker`), so scripts need no
`if __name__ == "__main__":` guard (unlike `pyse.map`). `ev.close()`, a `with` block, or
the last handle going away ends it; `ev.pid` is its process id. If a worker dies (say
Evolver crashes), its handle raises {class}`~pysurfaceevolver.WorkerCrashed` and the
surface is lost; the other workers and this process carry on.

`threads=` sets a worker's threads for Evolver's parallel loops and Newton steps (the
default, as for the engine of this process, is the number of physical cores). Several
busy workers share the cores, so with a surface per core `threads=1` is the right
choice.

## Limits

`input=` (answering Evolver's prompts from Python) isn't available, and a live view
(`ev.live_view()`) updates when you call its `update()`, not from a callback (that runs
in the worker). Workers need a POSIX system (Linux or macOS).
