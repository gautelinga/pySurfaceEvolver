/*
 * pyse_internal.h -- helpers shared by the glue's C files (pyse_api.c,
 * pyse_check.c, pyse_edit.c). All run inside a guarded call.
 */
#ifndef PYSE_INTERNAL_H
#define PYSE_INTERNAL_H

int pyse_surface_call(void (*body)(void *), void *arg);
/* after a call that changed the surface: Python's cached mesh is old */
void pyse_surface_changed(void);

/* which load the surface is (pyse_api.c): pySE's state keyed to it */
long pyse_load_serial(void);
/* slabs periodic in x and y (pyse_api.c): the periods, a minimum-image
   difference, a wrap from coordinates, an edge's wrap set from them */
int pyse_slab(REAL *L);
void pyse_delta(const REAL *a, const REAL *b, REAL *d);
WRAPTYPE pyse_wrap_between(vertex_id tail, vertex_id head);
void pyse_edge_rewrap(edge_id e);
/* where a slab's walls want their vertices: 0 the same at every copy, 1 a
   sphere (centre into c), -1 unknown; whether a vertex must stay at its copy */
int pyse_slab_home(int con, REAL *c);
int pyse_slab_pinned(vertex_id v_id);

/* changes to the native integrands (pyse_native.c): registrations, toggles */
long pyse_native_serial(void);
/* a constraint formula's closed form (pyse_native.c): its slot (-1: none,
   Evolver's interpreter), and its value and gradient there */
int pyse_native_formula_slot(struct expnode *ex);
void pyse_native_formula_at(int slot, const REAL *x, REAL *f, REAL *grad);

/* geometry (pyse_check.c) */
REAL pyse_det3(const REAL *a, const REAL *b, const REAL *c, const REAL *d);
int pyse_crosses(const REAL *p, const REAL *q, const REAL *a, const REAL *b, const REAL *c,
                 REAL eps);
void pyse_facet_normal(facet_id f_id, REAL *n, REAL *area);  /* |n| = twice the area */
double pyse_wall_gap(int c, REAL *x);       /* formula over its gradient's length */
int pyse_corner(facet_id f_id, vertex_id v_id);
double pyse_mean_edge(void);

/* snapshots inside a guarded call (pyse_api.c) */
struct snapshot;
struct snapshot *pyse_snapshot_hold(void);
void pyse_snapshot_put_back(struct snapshot *s);
void pyse_snapshot_drop(struct snapshot *s);

#endif
