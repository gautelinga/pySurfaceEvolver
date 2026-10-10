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
/* Vertices or edges in FOR_ALL order (cached); *n gets their number. */
element_id *fl_element_list(int type, long *n);
/* Facets in FOR_ALL_FACETS order; *n gets their number. */
facet_id *fl_facet_list(long *n);
/* Critical sections for interrupts: parallel regions and MUMPS run inside
   fl_enter()/fl_leave(); an abort requested meanwhile (fl_abort_pending)
   is made by fl_abort_hook at fl_leave(). */
#include <signal.h>
extern volatile sig_atomic_t fl_critical;
extern volatile sig_atomic_t fl_abort_pending;
extern void (*fl_abort_hook)(void);
void fl_enter(void);
void fl_leave(void);

/* Per-thread Evolver state in parallel loops: GET_THREAD_DATA gives each
   OpenMP thread its own struct thread_data (eval stack, q_info); worker
   threads' are set up by fl_prepare_threads() before a parallel region. */
struct thread_data *fl_thread_data(void);
void fl_prepare_threads(int threads);
int fl_in_parallel(void);
/* Errors in parallel loops: kb_error() calls fl_trap_error() first, which
   jumps back to the loop if its thread set a trap (fl_trap_set()), so the
   loop can redo that element serially; otherwise it returns. */
void fl_trap_error(void);
void fl_reset_state(void);   /* in Evolver's error recovery */
#include <setjmp.h>
extern __thread jmp_buf *fl_trap;

/* Nonzero when PYSE_NO_FAST_LOOPS is set: use Evolver's original loops. */
int fl_disabled(void);
/* The element loops of calc_quants(), calc_quant_grads(), calc_quant_hess()
   in parallel for elements of `type` (fasthess.c); each returns 0, doing
   nothing, when not covered (methods not known to be thread-safe, etc.). */
int fl_quant_values(int type, int mode, int global_needs);
int fl_quant_grads(int type, int mode, int global_needs);
int fl_quant_hess(struct linsys *S, int type, int hess_mode, int mode, REAL *rhs,
                  int global_needs);
/* web.total_area_addends += area, thread-safe in the parallel facet loops
   (fasthess.c) */
void fl_total_area_add(REAL area);


/* Linear-model area and body volume Hessians (hessian3.c) in parallel
   (fasthess.c); each returns 0, doing nothing, when not covered. */
int fl_area_hessian(struct linsys *S, REAL *rhs);
int fl_body_hessian_linear(struct linsys *S, REAL *rhs, REAL *Z);
int fl_body_hessian_quadratic(struct linsys *S, REAL *Z);

/* lagrange_facet_tension_hess() for 2D facets with BLAS (fasthess.c): adds
   the gradient and Hessian, sets *energy; returns 0 (nothing done) when not
   covered. */
int fl_lagrange_tension_hess(struct qinfo *f_info, REAL density, REAL *energy);
/* the same for lagrange_facet_volume_all() in METHOD_HESSIAN mode */
int fl_lagrange_volume_hess(struct qinfo *f_info, REAL *volume);

/* MUMPS factoring (mumpsfactor.c), the MUMPS_FACTORING mode */
void mumps_factor(struct linsys *S, int mtype);
void mumps_solve(struct linsys *S, REAL *b, REAL *x, int mtype);
void mumps_solve_multi(struct linsys *S, REAL **b, REAL **x, int nrhs, int mtype);
void mumps_free_system(struct linsys *S);
int fl_have_mumps(void);   /* nonzero if this build has MUMPS */

/* The Newton-step matrix pattern kept across steps (fasthess.c):
   hessian_init() calls fl_pattern_begin(S) after sp_hash_init();
   sp_hash_search() first tries fl_pattern_add() (nonzero: added);
   sp_hash_end() first tries fl_pattern_end() (>= 0: done, its return
   value) and calls fl_pattern_store() at its end. */
void fl_pattern_begin(struct linsys *S);
int fl_pattern_add(struct linsys *S, int row, int col, REAL value);
int fl_pattern_end(struct linsys *S, int rows, int cols, int index_start);
void fl_pattern_store(struct linsys *S, int rows, int cols, int index_start);

/* PYSE_DUMP_HESSIAN debugging hooks around the factoring (fasthess.c) */
void fl_hessian_before_factor(struct linsys *S);
void fl_hessian_after_factor(struct linsys *S);
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
/* DV^T DV added to the sparse system S, or if S is NULL to the dense
   rleftside; degrees of freedom added to *degfree */
int fl_calc_leftside(REAL **rleftside, struct linsys *S, int fixcount, int *degfree);
/* hessian_init()'s normals (NORMAL_MOTION) in parallel: for the k-th vertex
   of *list (FOR_ALL order, *n of them) that is not fixed, on a boundary or
   no_hessian_normal and has no constraints, new_calc_vertex_normal() and
   project_vertex_normals() into v_normal[k]; returns the malloc'ed
   dimensions, -1 for vertices left to the caller. NULL: nothing done. */
int *fl_vertex_normals(REAL ***v_normal, vertex_id **list, long *n);
/* pySE residual's shape directions in parallel: for the k-th vertex of *list
   (FOR_ALL order) with no constraints and not on a boundary,
   new_calc_vertex_normal() then gram_schmidt() into v_normal[k]; dims[k] is
   their number (0 for fixed vertices, -1 for none), -2 for vertices left to
   the caller. Returns the malloc'ed dims, or NULL: nothing done. */
int *fl_shape_normals(REAL ***v_normal, vertex_id **list, long *n);

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

/* Positions (gauss_pt) and, if need_side, tangents (sides) at the Gauss
   points of a Lagrange facet from its control points x, as the mat_mult()
   calls of q_facet_setup_lagrange(). Returns 0, doing nothing, when the
   case isn't covered. */
/* bumped whenever gauss_lagrange_setup() rebuilds the Lagrange tables */
extern long fl_lagrange_tables_version;
int fl_lagrange_facet_setup(struct gauss_lag *gl, int dim, int ctrl,
                            REAL **x, REAL **gauss_pt, REAL ***sides, int need_side);

/* Bumped when facet NONCONTENT attributes change or bodies are deleted
   (the loops cache facet bodies; set_facet_body() bumps top_timestamp). */
extern long fl_body_stamp;

/* vertex_average()'s search (find_vertex_average() for every vertex) in
   parallel; stride: bytes between consecutive vertex ordinals' x and status.
   Returns 0, doing nothing, when the case isn't covered (or PYSE_NO_FAST_MESH
   is set, for both functions here). Inside it,
   find_vertex_average() takes facet areas from fl_facet_area() (not stored). */
int fl_vertex_averages(int mode, REAL *x0, int *status0, size_t stride);
int fl_vertex_average_active(void);
REAL fl_facet_area(facet_id f_id);
/* calc_edge() for all edges in parallel (equiangulate()); 0 when not covered */
int fl_calc_edges(void);
/* test(e) for all edges in parallel, as 0/1 by edge ordinal (shared buffer,
   valid until the next call); NULL when not covered or a test failed */
unsigned char *fl_edge_marks(int (*test)(edge_id));

/* Number of threads for the parallel loops (1 without OpenMP). */
int fl_threads(void);
int fl_thread_setting(void);   /* as set: 0 for the default */
void fl_set_threads(int n);

#endif
