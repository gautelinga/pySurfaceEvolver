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
const char *pyse_solver(void);

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
double pyse_total_energy(void);
double pyse_total_area(void);
const char *pyse_datafilename(void);
/* Set the name reported as the current datafile (e.g. after restoring a
   snapshot from a temporary file). */
void pyse_set_datafilename(const char *name);

#ifdef __cplusplus
}
#endif

#endif
