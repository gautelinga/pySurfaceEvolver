/*
 * pyse_internal.h -- helpers shared by the glue's C files (pyse_api.c,
 * pyse_check.c, pyse_edit.c). All run inside a guarded call.
 */
#ifndef PYSE_INTERNAL_H
#define PYSE_INTERNAL_H

int pyse_surface_call(void (*body)(void *), void *arg);
/* after a call that changed the surface: Python's cached mesh is old */
void pyse_surface_changed(void);

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
