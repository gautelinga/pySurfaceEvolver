/*
 * pyse_api.h -- small C API around Surface Evolver, used by the nanobind
 * module.  Every function that walks or changes the surface goes through a
 * setjmp guard inside pyse_api.c, so Evolver's longjmp-based error handling
 * never unwinds through C++ frames.
 *
 * Surface Evolver keeps all of its state in globals, so there is exactly one
 * Evolver per process and none of these functions are thread safe; the
 * binding serializes all calls.
 */
#ifndef PYSE_API_H
#define PYSE_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes returned by the guarded calls. */
enum {
  PYSE_OK = 0,
  PYSE_ERROR = 1,      /* Evolver reported an error; state is consistent */
  PYSE_EXIT = 2,       /* Evolver called my_exit() / the quit command */
  PYSE_INTERRUPT = 3,  /* stopped by SIGINT (Ctrl-C) */
  PYSE_FATAL = 4,      /* unrecoverable error; the surface is now invalid */
  PYSE_BUSY = 5,       /* re-entrant call while another call is running */
  PYSE_INVALID = 6     /* no valid surface (failed load or fatal error) */
};

/* Error numbers used by the glue itself (Evolver's own are below 9000). */
#define PYSE_ERR_NO_DATAFILE 9001     /* pyse_load(): file not found */
#define PYSE_ERR_INVALID_SURFACE 9002 /* see PYSE_INVALID */
#define PYSE_ERR_BAD_VALUE 9003       /* expression did not give numbers */
#define PYSE_ERR_BAD_ARGUMENT 9004    /* wrong array size, element type... */

/* Element types (same numbering as Evolver's storage.h). */
enum { PYSE_VERTEX = 0, PYSE_EDGE = 1, PYSE_FACET = 2, PYSE_BODY = 3 };

/* Size of the name buffers filled by the parameter and quantity queries. */
#define PYSE_NAME_SIZE 128

typedef void (*pyse_output_fn)(int stream /* 0 stdout, 1 stderr */,
                               const char *text, void *userdata);
/* Return 1 with a line in buf, 0 for EOF, or -1 if no input source is set. */
typedef int (*pyse_input_fn)(const char *prompt, char *buf, int max,
                             void *userdata);

void pyse_set_output_callback(pyse_output_fn fn, void *userdata);
void pyse_set_input_callback(pyse_input_fn fn, void *userdata);
/* Handle SIGINT during guarded calls (main thread only).  The first Ctrl-C
   asks Evolver to stop at the next iteration or statement; a second one
   aborts the operation. */
void pyse_set_handle_sigint(int flag);

/* ---- Guarded calls.  All return one of the status codes above. ---------- */

int pyse_initialize(void);
int pyse_load(const char *path);
int pyse_command(const char *text);
/* Evaluate a numeric expression without defining any Evolver variable. */
int pyse_eval(const char *expr, double *value);
/* Evaluate expr for every element of a type, in the order of the
   pyse_get_* arrays.  n must equal pyse_count(type). */
int pyse_values(int type, const char *expr, double *out, long n);
int pyse_set_vertex_coords(const double *xyz, long n, int sdim);
/* Fast element writes, without going through Evolver commands.
   pyse_fast_attribute() says whether an attribute can be written this way:
   vertex coordinates (x, y, z, x1, x2, ...) and scalar real or integer
   extra attributes without an on-assign procedure.  pyse_set_values()
   writes values[i] to element i (in pyse_get_* order) where mask[i] is
   nonzero (mask may be NULL), then recalculates if autorecalc is on. */
int pyse_fast_attribute(int type, const char *attribute);
int pyse_set_values(int type, const char *attribute, const double *values,
                    const unsigned char *mask, long n);
/* Put vertices on constraints (on != 0; projected there as Evolver's set
   vertex constraint does) or take them off: vertex i (pyse_get_* order) and
   constraint nums[j] where mask[i*k + j] is nonzero; one recalc after. */
int pyse_set_vertex_constraints(const int *nums, int k, const unsigned char *mask, long n,
                                int on);

int pyse_get_vertices(double *xyz, int64_t *ids, unsigned char *fixed, long n,
                      int sdim);
int pyse_get_bodies(int64_t *ids, double *volume, double *target,
                    double *pressure, unsigned char *fixed, long n);

/* High-order elements.  pyse_element_node_count(type) is the number of
   nodes per edge or facet: 2/3 linear, 3/6 quadratic, order+1 and
   (order+1)(order+2)/2 Lagrange; pyse_get_mesh() returns the nodes of each
   element as vertex rows.  pyse_node_layout() fills a
   (nodes, dim+1) array with each node's barycentric multi-index, whose
   entries sum to pyse_element_order(). */
/* Vertices (as pyse_get_vertices), edges, facets and element nodes in one
   call with one vertex lookup table.  Pointers for parts that don't exist
   (facets outside the soapfilm model, node layouts) must be NULL. */
struct pyse_mesh_arrays {
  double *xyz; int64_t *vertex_ids; unsigned char *fixed; long nv; int sdim;
  int64_t *edges; int64_t *edge_ids; long ne;
  int64_t *facets; int64_t *facet_ids; int64_t *facet_bodies; long nf;
  int64_t *edge_nodes; int edge_nodes_per;
  int64_t *facet_nodes; int facet_nodes_per;
};
int pyse_get_mesh(struct pyse_mesh_arrays *m);

int pyse_element_node_count(int type);
int pyse_node_layout(int type, int *index, int nodes_per);
int pyse_element_order(void);   /* 1 linear, 2 quadratic, Lagrange order */
int pyse_bezier(void);          /* Lagrange nodes are Bezier control points */

/* Parameters declared in the datafile, and named quantities. */
long pyse_parameter_count(void);
int pyse_get_parameters(char (*names)[PYSE_NAME_SIZE], double *values,
                        unsigned char *optimizing, long n);
long pyse_quantity_count(void);
/* the Evolver method names of the energy quantities' method instances (at
   most max rows; count: all of them) */
int pyse_energy_methods(char (*names)[PYSE_NAME_SIZE], int max, int *count);
/* the energy quantities' method instances: names, method names and what
   each adds to the energy (at most max rows; count: all of them) */
int pyse_energy_terms(char (*names)[PYSE_NAME_SIZE], char (*methods)[PYSE_NAME_SIZE],
                      double *energy, int max, int *count);
int pyse_get_quantities(char (*names)[PYSE_NAME_SIZE], double *value,
                        double *target, double *modulus, double *pressure,
                        int *kind /* 0 energy, 1 fixed, 2 info, 3 conserved */,
                        long n);

/* Threads for the parallel facet loops (1 without OpenMP).  n <= 0 means
   the OpenMP default.  PYSE_THREADS sets the initial value. */
void pyse_set_threads(int n);
int pyse_threads(void);
int pyse_thread_setting(void);   /* 0: the default */
/* Factoring for Newton steps: "mumps" (default when built with MUMPS) or
   "evolver" (Evolver's own minimal degree). Returns 0 if unknown/unavailable. */
int pyse_set_solver(const char *name);
/* pySE's sphere walls' contact-line integrands evaluated in C (pyse_native.c):
   constraint con registered with its centre c, radius R, frame e1 e2 e3 (e3
   its axis) and energy coefficient k (cos(angle)*tension; NaN: the content
   only); 0 if it can't be. pyse_native_enable(0) goes back to Evolver's
   expressions (for comparison). */
int pyse_native_sphere(int con, const double *c, double R, const double *e1,
                       const double *e2, const double *e3, double k);
void pyse_native_enable(int on);
/* the flux gauge's integrands in C (constraints.sphere(gauge="flux")): a
   method instance's facet integrand (method) or constraint con's energy
   (which 0) or content (1) line integrand; base 0 none, 1 (0,0,z), 2 a
   sphere's exact content part about bc, 3 a plane's content (bc n_z, o, u,
   v), 4 a plane's wetting (bc angle, tension, o, u, v); terms 11 numbers each (centre,
   radius, direction (0 for a point source), energy flag, angle, tension,
   spare). 0 if it can't be registered. */
int pyse_native_flux(const char *method, int con, int which, int base, const double *bc,
                     int nbc, const double *terms, int nterm);
/* the flux gauge in a periodic slab (pyse_slabflux.c, _periodic.py) */
int pyse_slabflux_setup(const double *L, double zf, int nbead, const double *beads,
                        double floor_angle, double floor_tension, const double *prm,
                        int ns, const double *stencil, int nh, const double *H,
                        const double *pinv, int nsrc, const double *src, const double *grid);
int pyse_slabflux_key(const char *method, int con, int which, int kind, int wall);
int pyse_slabflux_probe(const double *x, double *out);
int pyse_slabflux_field(int kind, int wall, const double *x, double *out);
/* the flux gauge's sums over the spheres with AVX2 where the processor has
   it (1, the default) or the scalar kernels (0), which agree to the bit;
   returns the previous setting */
int pyse_native_simd(int on);
/* constraint con's formula evaluated in closed form, a plane (kind 0: p = n,
   d) or a sphere (kind 1: p = c, R^2), where that matches the interpreter at
   test points (1; else 0). */
int pyse_native_formula_register(int con, int kind, const double *p);
/* Gradient steps with fewer energy evaluations (src/iterate.c,
   pyse_fast_steps: the move at scale 0 left out where nothing changed since
   the last step, the scale from a parabola with the energy's slope where it
   can be, each move's content predicted for its restoring motion; off by
   default); returns the previous setting. */
int pyse_set_fast_steps(int on);
double pyse_work_count(void);
/* For each edge and each of the constraints nums, whether the edge is on it
   (out: edges x ncon, row-major). */
int pyse_edges_on_constraints(const int *nums, int ncon, unsigned char *out, long n);
/* Two Newton steps from the same point (see recipes._arc_correct): at body
   number body's target and at that plus delta, from one factorization; the
   vertices (nv x 3) and multipliers (nm) each ends at; the surface put back.
   PYSE_OK or an error status. */
int pyse_newton_pair(int body, double delta, double *x0, double *m0, double *x1, double *m1,
                     long nv, long nm);
const char *pyse_solver(void);
/* Whether Newton steps carry the mesh along a line they move past the edges
   beside it (on by default); returns the previous setting. */
int pyse_set_follow(int on);
/* With length > 0, each gradient step ends with the contact-line edges (border
   edges on a constraint) longer than length split, the midpoints onto their
   constraints (0: off, the default); returns the previous setting. */
double pyse_line_chop(double length);

/* ---- Information about the last guarded call. ---------------------------- */
int pyse_last_errnum(void);
const char *pyse_last_errmsg(void);
int pyse_exit_code(void);
int pyse_warning_count(void);
const char *pyse_warning(int i);

/* ---- Unguarded state: plain reads of Evolver fields.  They never run
   Evolver code, so they cannot raise Evolver errors. ------------------------ */
int pyse_is_initialized(void);
int pyse_surface_valid(void);
/* Goes up whenever a call may have changed the surface (commands, loads,
   coordinate writes), so callers can cache snapshots.  Expression
   evaluation (pyse_eval, pyse_values) and the snapshot getters don't
   change it. */
long pyse_surface_version(void);
long pyse_count(int type);
int pyse_sdim(void);
int pyse_representation(void);  /* 1 string, 2 soapfilm, 3 simplex */
int pyse_modeltype(void);       /* 1 linear, 2 quadratic, 3 lagrange */
int pyse_lagrange_order(void);
int pyse_torus(void);
int pyse_slab_periods(double *L);   /* 1 and (Lx, Ly) for a slab periodic in x, y */
int pyse_slab_set_home(int con, int kind, const double *c);  /* 0 the same at every copy, 1 a sphere */
int pyse_crystalline(void);   /* a Wulff (crystalline) integrand is in use */
int pyse_symmetry(void);      /* a torus or quotient space: coordinates are wrapped */
/* goes up with every topology change (Evolver's top_timestamp) */
long pyse_topology_stamp(void);
/* the n lowest Hessian eigenvalues with the constraints held and their
   eigenvectors as vertex displacements (vectors: n * nv * sdim); found: how
   many there are (at most n) */
int pyse_modes(int n, double *values, double *vectors, long nv, int *found);
/* the Lagrange multipliers: bodies' pressures, then quantities' (n must be
   pyse_multiplier_count()) */
long pyse_multiplier_count(void);
void pyse_get_multipliers(double *out, long n);
void pyse_set_multipliers(const double *in, long n);
double pyse_total_energy(void);
/* per vertex, the shape part of its velocity (projected force): along its
   normal(s), projected into its constraints; one per vertex in vertex order */
/* full: all directions instead (for energies with in-plane forces, such
   as elastic membranes) */
int pyse_residual(double *out, long n, int full);   /* n: vertices + optimizing parameters */
int pyse_optparam_count(void);
/* per vertex, 1 if fixed, on a constraint or on a boundary */
int pyse_vertex_attached(unsigned char *out, long n);
/* how many free vertices (not fixed, on no constraint or boundary) have a
   tangent cone that Evolver's 'o' would pop (soapfilm; 0 otherwise) */
int pyse_poppable_vertices(long *count);
/* the constraints in use: how many (*count), and up to max of their numbers,
   attribute bits and names ("" if unnamed; Evolver's own strings); the
   arrays may be NULL to only count */
int pyse_constraint_list(int *nums, int *attrs, const char **names, int max, int *count);
/* per vertex (rows) and listed constraint (columns): the constraint's value
   over its gradient's length, a signed distance to first order; NaN where the
   vertex is on that constraint or the gradient vanishes */
int pyse_constraint_gaps(const int *nums, int ncon, double *out, long n);
/* Per vertex, the unit gradient of constraint num's level-set function
   (n x 3; NaN where it vanishes). */
int pyse_constraint_normals(int num, double *out, long n);
/* The same at the listed vertex rows only (out: nrows x 3, in their order;
   a query from the topology runs needs the solid's few vertices close by:
   the whole column was most of its cost). */
int pyse_constraint_normals_rows(int num, const int64_t *rows, long nrows, double *out, long n);
/* Per facet (linear soapfilm), its three corners as Evolver sees them, the
   second and third unwrapped to the first across a torus's sides (n x 3 x 3). */
int pyse_facet_corners(double *out, long n);
double pyse_total_area(void);
const char *pyse_datafilename(void);

/* In-memory snapshots of the surface (pySE's own save/restore inside its
   runs; see pyse_api.c). take() fills handle; restore() puts the surface
   back as it was, into the same load only; free() may be called anytime. */
int pyse_snapshot_take(long *handle);
/* The topology check (pyse_check.c): problems found, as records.
   walls/sides: solid constraints and which side of each is solid (+1:
   where its formula is negative); tol: depth allowed, in mean edge lengths;
   manifold: edges on more than two facets are problems too. out gets up to
   max records; count: how many were found. */
enum { PYSE_PROBLEM_BARE_EDGE = 1, PYSE_PROBLEM_MANY_FACETS, PYSE_PROBLEM_ORIENTATION,
       PYSE_PROBLEM_PINCHED, PYSE_PROBLEM_OPEN_BORDER, PYSE_PROBLEM_BORDER_VERTEX,
       PYSE_PROBLEM_BORDER_REVERSED, PYSE_PROBLEM_DEGENERATE, PYSE_PROBLEM_FOLD,
       PYSE_PROBLEM_VERTEX_IN_SOLID, PYSE_PROBLEM_EDGE_IN_SOLID, PYSE_PROBLEM_INTERSECTION };
struct pyse_problem { int code; int type; long id; long other; double value; };
int pyse_check_topology(const int *walls, const int *sides, int nwalls, double tol,
                        int manifold, struct pyse_problem *out, int max, int *count);

/* Mesh edits (pyse_edit.c), a batch of one kind: ids are element numbers
   (edges, vertices, facets; snap: vertex pairs, n = 2 per pair); results
   one per edit (per pair), PYSE_EDIT_DONE or why it was refused. params: [0] largest
   turn of a facet's normal (degrees), [1] largest volume swept (in h^3),
   [2] longest new edge (in h, 0: any), [3] depth into a wall allowed (in h),
   [4] flip mode (0 any, 1 more equiangular, 2 valences nearer 6),
   [5] smoothing factor, [6] separation offset (in h), [7] a degenerate
   facet's area (in h^2), [8] h (0: the mean edge length), [9] remesh sweeps.
   merge: facet pairs; bridge: border-edge pairs on one wall; patch: the wall's
   number then the facets of a wet or dry patch (one edit); unzip: edges;
   remesh: no ids, results the splits, collapses, flips and moves made
   (four). */
enum { PYSE_OP_COLLAPSE = 1, PYSE_OP_FLIP, PYSE_OP_SPLIT, PYSE_OP_SMOOTH, PYSE_OP_SNAP,
       PYSE_OP_SEPARATE, PYSE_OP_DEGENERATE, PYSE_OP_MERGE, PYSE_OP_BRIDGE, PYSE_OP_REMESH,
       PYSE_OP_PATCH, PYSE_OP_UNZIP };
enum { PYSE_EDIT_DONE = 0, PYSE_EDIT_GONE, PYSE_EDIT_NOT_APPLICABLE, PYSE_EDIT_NOT_BETTER,
       PYSE_EDIT_FIXED, PYSE_EDIT_CONSTRAINTS, PYSE_EDIT_LINE, PYSE_EDIT_LINK,
       PYSE_EDIT_ORIENTATION, PYSE_EDIT_TURN, PYSE_EDIT_LONG, PYSE_EDIT_VOLUME,
       PYSE_EDIT_EVOLVER, PYSE_EDIT_STRUCTURE, PYSE_EDIT_PINCH, PYSE_EDIT_DEGENERATE,
       PYSE_EDIT_FOLD, PYSE_EDIT_SOLID, PYSE_EDIT_CROSSING, PYSE_EDIT_REPLAY };
int pyse_edit(int op, const long *ids, int n, const double *params, const int *walls,
              const int *sides, int nwalls, int manifold, int *results,
              const double *size, long nsize);
/* size (or NULL): a target edge length per vertex, by Evolver's vertex
   number - 1 (<= 0: the h of params), for remesh's splits and collapses
   and a collapse's longest edge; a split's new vertex takes its ends' mean.
   For remesh, ids (if any) are a region of vertices: only the edges with an
   end there are split, collapsed and flipped, only its vertices smoothed
   (a split's new vertex joins it). */
/* Edges through facets on the whole surface (for descent steps); far: only
   those more than a ring of edges from the facet (not folds); with contact
   lines crossing each other on a wall. where (or NULL): the places, up to
   max (x, y, z each: the edge's midpoint, the two lines' middle). */
int pyse_crossings(int far, long *count, double *where, long max);

/* The topology events' loops (pyse_events.c, for _events.py): on arrays
   only, never Evolver's state, so not guarded and safe to run beside a
   call. Rows as in the mesh's arrays (x: n x 3, f: nf x 3, pairs n x 2).
   Each returns 0, -1 when memory ran out, -2 for a row out of range;
   arrays returned through pointers are malloc'ed, the caller frees them.
   threads: at most that many OpenMP threads, where the work is enough.
   L: a slab's periods in x and y (coordinates at any image, differences
   of nearby points at the minimum image), or NULL.

   frame: the facets' unique edges (smaller end first, sorted), each
   vertex's unit normal (the facets' areas summed) and whether it lies in a
   fold (a facet round it facing against it). */
int pyse_events_frame(long n, const double *x, long nf, const int64_t *f, const double *L,
                      int64_t **edges, long *ne, double *normal, unsigned char *folded);
/* self: the surface close to itself, (vertex, facet) pairs in order (vi,
   ti): vi in vmask, a corner of the facet in pmask facing vi (normals
   apart by more than 120 degrees) within R, the facet not round vi and no
   corner of it folded, facing vi (its corners' normals summed), within
   reach*r[vi] of its centroid less its radius, the gap to its closest
   point (closest) within r[vi] (a neck, neck_factor times that), along
   vi's normal for more than half of it, and no corner within two edges of
   vi along the mesh (edges). neck: a body lies across (with_bodies;
   behind/ahead: the bodies on each side of each vertex), or (without)
   the gap is behind vi. */
int pyse_events_self(long n, const double *x, long nf, const int64_t *f, const double *L,
                     long ne, const int64_t *edges, const double *normal, const double *r,
                     const unsigned char *vmask, const unsigned char *pmask,
                     const unsigned char *folded, const int64_t *behind, const int64_t *ahead,
                     int with_bodies, double R, double reach, double neck_factor, int threads,
                     long *count, int64_t **vi, int64_t **ti, double **gap, double **closest,
                     unsigned char **neck);
/* lines: one constraint's contact lines (its contact-line edges, joined
   into paths as _geometry._chains does: line_vertices, nl of them, the
   paths' in turn, a loop's first vertex once) and the pairs (i < j, rows
   of line_vertices, np of them, in order) closer than factor times the
   largest h there and than factor times their mean h (scale), on two
   lines or farther apart along theirs than three times that or their
   distance (d). */
int pyse_events_lines(long n, const double *x, const double *h, long ne, const int64_t *edges,
                      double factor, const double *L, long *nl, int64_t **line_vertices,
                      long *np,
                      int64_t **pi, int64_t **pj, double **pd, double **pscale);
/* segments: each point's distance to the nearest of the segments (a, b)
   (in a slab, of their images) */
int pyse_events_segments(long np, const double *p, long ns, const double *a, const double *b,
                         const double *L, int threads, double *out);
/* groups: the connected groups of masked vertices, joined by links (nl
   pairs) between masked ones: per vertex its group's smallest member (-1
   where not masked). */
int pyse_events_groups(long n, const unsigned char *mask, long nl, const int64_t *links,
                       int64_t *label);

/* Whether the surface is converted to quantities (everything_quantities;
   the first Hessian converts it): a snapshot restores only into the mode
   it was taken in. */
int pyse_everything_quantities(void);
int pyse_snapshot_restore(long handle);
void pyse_snapshot_free(long handle);
/* Set the name reported as the current datafile (e.g. after restoring a
   snapshot from a temporary file). */
void pyse_set_datafilename(const char *name);

#ifdef __cplusplus
}
#endif

#endif
