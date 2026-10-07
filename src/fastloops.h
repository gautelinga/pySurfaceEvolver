/*
 * fastloops.h -- cached facet topology and parallel facet loops.
 *
 * Not part of the original Surface Evolver; see fastloops.c.
 */
#ifndef FASTLOOPS_H
#define FASTLOOPS_H

/* Three corner vertices per facet ordinal (tail order), or NULL when the
   cache doesn't apply (it is only built for soapfilm surfaces). */
vertex_id *fl_facet_corners(void);
/* Facets in FOR_ALL_FACETS order; *n gets their number. */
facet_id *fl_facet_list(long *n);
/* Nonzero when PYSE_CHECK_FACET_CACHE is set: verify every cached answer. */
int fl_check(void);

/* Classic facet volumes for linear soapfilm surfaces, computed in parallel
   with per-thread sums merged in thread order (agrees with the serial loop
   to round-off).
   Returns 0, doing nothing, when the case isn't covered. */
int fl_facet_volumes(void);

/* Facet surface energies (area, density, gravity) for linear soapfilm
   surfaces, as facet_energy_l(f_id,ALL_ENERGIES) for every facet: areas
   are stored, and the area and energy sums accumulated per thread.
   Returns 0, doing nothing, when the case isn't covered. */
int fl_facet_energies(void);

/* Facet tension and gravity forces for linear soapfilm surfaces, as
   facet_force_l() for every facet: areas stored, forces added to the
   vertices via per-thread arrays. Returns 0, doing nothing, when not covered. */
int fl_facet_forces(void);

/* Body volume gradients for linear soapfilm surfaces, as film_grad_l(),
   in parallel and with identical results. Returns 0, doing nothing, when
   the case isn't covered. */
int fl_film_grad(void);

/* Per-vertex loops over all vertices, in parallel (iterate.c, fixvol.c,
   calcforc.c). Each returns 0, doing nothing, when it doesn't apply.
   Boundary vertices (and in fl_volume_restore() constraint vertices)
   are left to the caller's serial loop: they evaluate expressions. */
int fl_zero_forces(void);
int fl_move_vertices(REAL scale, int dim);
int fl_save_coords(REAL (*coord)[MAXCOORD]);     /* NULL: to __oldx */
int fl_restore_coords(REAL (*coord)[MAXCOORD]);  /* NULL: from __oldx */
int fl_volume_restore(REAL stepsize, REAL *vol_restore, int fixcount);
/* DV^T DV added to dense rleftside, degrees of freedom to *degfree */
int fl_calc_leftside(REAL **rleftside, int fixcount, int *degfree);

/* Serial loops over the vertices or edges whose attributes have any of
   `bits` (all of them for bits 0), in FOR_ALL order, found by a parallel
   scan; Evolver's own traversal when that doesn't apply. `site` is a
   small number unique to the call site (its own result buffer).
     fl_sel sel;
     FL_FOR_SELECTED(sel,EDGE,e_id,BDRY_ENERGY|DENSITY,0) { ... }  */
typedef struct
{ element_id *list;
  long n, k;
  element_id id;
  int type;
  ATTR bits;
} fl_sel;
void fl_sel_begin(fl_sel *s, int type, ATTR bits, int site);
int fl_sel_next(fl_sel *s, element_id *id);
#define FL_FOR_SELECTED(sel,type,id,bits,site) \
  for ( fl_sel_begin(&(sel),(type),(bits),(site)) ; fl_sel_next(&(sel),&(id)) ; )

/* Bumped when facet NONCONTENT attributes change or bodies are deleted
   (the loops cache facet bodies; set_facet_body() bumps top_timestamp). */
extern long fl_body_stamp;

/* Number of threads for the parallel loops (1 without OpenMP). */
int fl_threads(void);
void fl_set_threads(int n);

#endif
