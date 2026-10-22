/*
 * pyse_api.c -- C glue between Surface Evolver and the Python binding.
 *
 * Surface Evolver handles errors with setjmp/longjmp: kb_error() jumps to
 * jumpbuf[subshell_depth] (or cmdbuf, or loadjumpbuf for the "load"
 * command), and my_exit() calls exit().  Every entry point here that walks
 * or changes the surface goes through protected_call(), which sets all of
 * those jump buffers in a C frame.  That keeps longjmp from ever crossing
 * C++ frames, and turns errors and exits into status codes the binding can
 * raise as Python exceptions.
 */

#include "include.h"
#include "pyse_hooks.h"
#include "pyse_api.h"
#include "fastloops.h"
#include "pyse_internal.h"

#include <signal.h>
#include <pthread.h>

int pyse_library_mode = 0;

static int initialized = 0;
static int in_protected = 0;
static jmp_buf exit_jmp;
static int exit_code = 0;

/* Whether the web holds a usable surface.  Cleared while a datafile loads
   (set again once it has parsed) and after an unrecoverable error. */
static int surface_valid = 1;
/* Globals below this index were declared by the datafile itself. */
static int datafile_global_limit = 0;
/* See pyse_surface_version(). */
static long surface_version = 0;
/* Datafiles loaded (pyse_load() or the "load" command): snapshots restore
   only into the load they came from. */
static long load_serial = 0;
long pyse_load_serial(void) { return load_serial; }

/* Map from vertex ordinal to row in the pyse_get_vertices() arrays. */
struct vertex_map { long *rows; long size; };
/* Set by mesh_body() so the snapshot bodies share one table; reset at the
   end of every guarded call. */
static struct vertex_map *shared_vertex_map = NULL;

static void inconsistent(const char *what);
static void expect_rows(long row, long n);

static pyse_output_fn out_cb = NULL;
static void *out_ud = NULL;
static pyse_input_fn in_cb = NULL;
static void *in_ud = NULL;
static int handle_sigint = 0;
static int suppress_input_warning = 0;

/* What happened during the current guarded call. */
#define ERRBUF_SIZE 4096
static int err_num = 0;
static int err_mode = -1;
static char err_msg[ERRBUF_SIZE];
static int fatal_seen = 0;
static int input_eof_seen = 0;

#define MAX_WARNINGS 64
#define WARNING_SIZE 512
static int warning_count = 0;
static char warnings[MAX_WARNINGS][WARNING_SIZE];

/* SIGINT state.  Text Evolver prints while we are inside the signal
   handler is held here and delivered after the call. */
static volatile sig_atomic_t sigint_count = 0;
static volatile sig_atomic_t in_signal = 0;
#define SIGBUF_SIZE 4096
static char sig_buf[SIGBUF_SIZE];

/* Numbers printed by our own printf commands (see run_capture()) are
   intercepted in pyse_outstring() instead of being shown. */
#define CAPTURE_TAG "@pyse@"
static int capture_active = 0;
static double *capture_out = NULL;
static long capture_max = 0;
static long capture_count = 0;
static int capture_bad = 0;

/**************************************************************************
 * Hooks called from the patched Evolver sources.
 */

void pyse_exit(int code)
{
  exit_code = code;
  if ( in_protected )
    longjmp(exit_jmp,1);
  exit(code);
}

static void hold_signal_text(const char *text)
{ size_t used = strlen(sig_buf);
  strncat(sig_buf,text,SIGBUF_SIZE-1-used);
}

static void capture_line(const char *text)
{ const char *p = text + strlen(CAPTURE_TAG);
  char *end;
  double v = strtod(p,&end);  /* also reads inf and nan */
  if ( end == p ) capture_bad = 1;
  if ( capture_count < capture_max )
    capture_out[capture_count] = v;
  capture_count++;
}

int pyse_outstring(const char *text)
{
  if ( !pyse_library_mode ) return 0;
  if ( capture_active && strncmp(text,CAPTURE_TAG,strlen(CAPTURE_TAG)) == 0 )
    capture_line(text);
  else if ( in_signal )
    hold_signal_text(text);
  else if ( out_cb )
    out_cb(0,text,out_ud);
  return 1;
}

int pyse_erroutstring(const char *text)
{
  if ( !pyse_library_mode ) return 0;
  if ( in_signal )
    hold_signal_text(text);
  else if ( out_cb )
    out_cb(1,text,out_ud);
  return 1;
}

static void append_error_text(const char *emsg)
{ size_t used = strlen(err_msg);
  if ( used && used < ERRBUF_SIZE-2 && err_msg[used-1] != '\n' )
  { strcat(err_msg,"\n"); used++; }
  strncat(err_msg,emsg,ERRBUF_SIZE-1-used);
}

static void add_warning(int errnum, const char *emsg)
{ if ( warning_count >= MAX_WARNINGS ) return;
  /* long messages are cut to fit the slot */
  snprintf(warnings[warning_count],WARNING_SIZE,"WARNING %d: %.480s",errnum,emsg);
  warning_count++;
}

void pyse_note_error(int errnum, const char *emsg, int mode)
{ int i;
  if ( !pyse_library_mode ) return;
  if ( emsg == NULL ) emsg = "";
  switch ( mode )
  { case WARNING:
      for ( i = 0 ; i < warnings_suppressed_count ; i++ )
        if ( errnum == warnings_suppressed[i] ) return;
      add_warning(errnum,emsg);
      return;
    case RECOVERABLE_QUIET:
      /* used for the Ctrl-C abort and a few silent bailouts */
      if ( err_mode < 0 ) { err_num = errnum; err_mode = mode; }
      return;
    case UNRECOVERABLE:
      fatal_seen = 1;
      /* fall through */
    default:
      if ( err_mode < 0 || err_mode == RECOVERABLE_QUIET )
      { err_num = errnum; err_mode = mode; }
      append_error_text(emsg);
      return;
  }
}

int pyse_read_stdin(const char *prompt, char *buf, int max)
{
  if ( !pyse_library_mode ) return -1;
  buf[0] = 0;
  if ( in_cb )
  { int k = in_cb(prompt ? prompt : "",buf,max,in_ud);
    if ( k >= 0 ) return k ? 1 : 0;
  }
  if ( prompt && strncmp(prompt,"Enter new datafile name",23) == 0 )
    return 0;  /* from startup(NULL), e.g. the "q" command: keep the surface */
  if ( !suppress_input_warning )
  { char text[WARNING_SIZE];
    snprintf(text,sizeof(text),
      "Evolver asked for interactive input (prompt: \"%s\"); "
      "no input callback is set, so it got end-of-file.",
      prompt ? prompt : "");
    add_warning(0,text);
    input_eof_seen = 1;
  }
  return 0;
}

/**************************************************************************
 * Guarded execution.
 */

/* First Ctrl-C: set Evolver's break flag, which stops iterations and
   command loops at the next safe point, outside this handler.  Second
   Ctrl-C: abort right away, the way Evolver's own handler does.  Nothing
   here calls back into Python. */
static pthread_t engine_thread;   /* the thread running Evolver */

static void pyse_abort(void)
{ kb_error(1357,"",RECOVERABLE_QUIET);  /* longjmps to protected_call() */
}

static void pyse_sigint(int sig)
{ (void)sig;
  /* any thread may get the signal (OpenMP, BLAS workers): only the
     engine's own thread may longjmp */
  if ( !pthread_equal(pthread_self(),engine_thread) )
  { pthread_kill(engine_thread,SIGINT);
    return;
  }
  sigint_count++;
  if ( sigint_count == 1 )
  { breakflag = BREAKFULL;
    return;
  }
  if ( fl_critical > 0 )   /* in a parallel region or MUMPS: abort later */
  { fl_abort_pending = 1;
    return;
  }
  in_signal = 1;
  kb_error(1357,"",RECOVERABLE_QUIET);  /* longjmps to protected_call() */
}

typedef void (*body_fn)(void *);

/* Commands after "read" in a datafile are left on the command-file stack by
   startup(); the interactive loop would run them next.  Run them now, until
   input falls back to the base "stdin" entry. */
static void run_datafile_commands(void)
{ int old_quiet = quiet_load_flag;
  datafile_flag = 0;
  quiet_load_flag = 1;  /* don't echo the commands as they are read */
  exec_commands(stdin,"Enter command: ");
  quiet_load_flag = old_quiet;
}

/* Load a datafile.  The surface counts as valid once startup() has parsed
   it, even if one of the datafile's trailing commands then fails. */
static void load_datafile(char *name)
{
  /* The torus display mode, which Evolver keeps from one datafile to the
     next for its own graphics (pySE has none); it would show up as
     "clipped" in later dumps, and so in save()/restore(). Here so that
     both pyse_load() and Evolver's own "load" command reset it. */
  web.torus_clip_flag = 0;
  web.torus_body_flag = 0;
  torus_display_mode = TORUS_DEFAULT_MODE;
  former_torus_display_mode = TORUS_DEFAULT_MODE;
  mumps_new_surface();   /* the factoring's ordering starts again too */
  surface_valid = 0;
  startup(name);
  surface_valid = 1;
  datafile_global_limit = web.global_count;
  run_datafile_commands();
}

static void reset_call_state(void)
{ err_num = 0; err_mode = -1; err_msg[0] = 0;
  fatal_seen = 0; input_eof_seen = 0;
  warning_count = 0;
  exit_code = 0;
  sigint_count = 0; in_signal = 0; sig_buf[0] = 0;
}

/* Report an error from the glue itself, outside a guarded call. */
static int glue_error(int status, int errnum, const char *text)
{ reset_call_state();
  err_num = errnum;
  err_mode = RECOVERABLE;
  append_error_text(text);
  return status;
}

static int invalid_surface(void)
{ return glue_error(PYSE_INVALID,PYSE_ERR_INVALID_SURFACE,
    "No valid surface: the last datafile failed to load, or an "
    "unrecoverable error occurred.  Load a datafile first.");
}

static int protected_call(body_fn body, void *arg)
{
  volatile int status = PYSE_OK;
  struct sigaction old_action, new_action;
  volatile int sigint_installed = 0;

  if ( in_protected ) return PYSE_BUSY;
  in_protected = 1;
  pyse_library_mode = 1;
  reset_call_state();
  subshell_depth = 0;

  engine_thread = pthread_self();
  fl_critical = 0;
  fl_abort_pending = 0;
  fl_abort_hook = pyse_abort;
  if ( handle_sigint )
  { memset(&new_action,0,sizeof(new_action));
    new_action.sa_handler = pyse_sigint;
    sigemptyset(&new_action.sa_mask);
    /* SA_NODEFER: the second Ctrl-C longjmps out of the handler */
    new_action.sa_flags = SA_NODEFER;
    if ( sigaction(SIGINT,&new_action,&old_action) == 0 )
      sigint_installed = 1;
  }

  if ( setjmp(exit_jmp) )
  { /* my_exit() was called */
    status = fatal_seen ? PYSE_FATAL : PYSE_EXIT;
    goto done;
  }
  if ( setjmp(jumpbuf[0]) )
  { /* kb_error() bailout, or the "q" command */
    status = PYSE_ERROR;
    goto done;
  }
  if ( setjmp(loadjumpbuf) )
  { /* the "load" command jumps here to start a new datafile */
    if ( list && (list != permlist) )
    { myfree((char*)list); list = NULL; }
    load_serial++;
    load_datafile(loadfilename);
    goto done;
  }

  body(arg);

done:
  shared_vertex_map = NULL;
  if ( sigint_installed )
    sigaction(SIGINT,&old_action,NULL);
  in_signal = 0;
  capture_active = 0;
  subshell_depth = 0;
  datafile_flag = 0;
  breakflag = 0;
  iterate_flag = 0;
  quiet_flag = 0;
  /* EOF on the input hook can pop the base "stdin" entry; restore it */
  if ( commandfd == NULL )
    push_commandfd(stdin,"stdin");
  if ( outfd == NULL ) outfd = stdout;
  if ( erroutfd == NULL ) erroutfd = stderr;
  in_protected = 0;

  if ( sig_buf[0] && out_cb )
    out_cb(1,sig_buf,out_ud);

  if ( status == PYSE_ERROR && err_mode < 0 )
    status = PYSE_EXIT;  /* the "q"/"quit" command jumps out without an error */
  if ( status == PYSE_EXIT && input_eof_seen )
  { status = PYSE_ERROR;
    append_error_text("Evolver needed interactive input to continue.");
  }
  if ( fatal_seen )
  { status = PYSE_FATAL;
    surface_valid = 0;
  }
  else if ( sigint_count )
    status = PYSE_INTERRUPT;
  else if ( status == PYSE_OK && err_mode >= 0 && err_mode != RECOVERABLE_QUIET )
    status = PYSE_ERROR;  /* error reported without a longjmp (parse errors) */
  return status;
}

/* protected_call() for anything that needs a valid surface. */
static int surface_call(body_fn body, void *arg)
{
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  return protected_call(body,arg);
}

/* for the glue's other files (pyse_internal.h) */
int pyse_surface_call(body_fn body, void *arg) { return surface_call(body,arg); }
void pyse_surface_changed(void) { surface_version++; }

/**************************************************************************
 * Starting up and loading.
 */

static void init_body(void *arg)
{ (void)arg;
  msgmax = 2000;
  if ( !msg ) msg = my_list_calloc(1,msgmax,ETERNAL_BLOCK);
  set_ctypes();
  outfd = stdout;
  erroutfd = stderr;

  /* machine precision, as in main() */
  { REAL eps,one = 1.0;
    for ( eps = 1.0 ; one + eps != one ; eps /= 2.0 ) ;
    machine_eps = 2.0*eps;
    root8machine_eps = sqrt(sqrt(sqrt(machine_eps)));
    DPREC = (int)floor(-log(machine_eps)/log(10.0));
    DWIDTH = DPREC + 3;
  }

  print_express(NULL,0); /* initializes string allocation */
  find_cpu_speed();
  nprocs = procs_requested;
  if ( thread_data_ptrs == NULL )
  { thread_data_ptrs = &default_thread_data_ptr;
    thread_data_ptrs[0] = &default_thread_data;
  }
  scoeff_init();
  vcoeff_init();
  fl_set_threads(getenv("PYSE_THREADS") ? atoi(getenv("PYSE_THREADS")) : 0);
  if ( getenv("PYSE_SOLVER") ) pyse_set_solver(getenv("PYSE_SOLVER"));
  push_commandfd(stdin,"stdin");
  subshell_depth = 0;

  /* Start with an empty surface: startup(NULL) asks for a datafile name,
     gets end-of-file from the input hook, and resets the web. */
  suppress_input_warning = 1;
  startup(NULL);
  suppress_input_warning = 0;
  datafile_flag = 0;
  datafile_global_limit = web.global_count;
}

int pyse_initialize(void)
{ int status;
  pyse_input_fn saved_cb = in_cb;
  if ( in_protected ) return PYSE_BUSY;
  if ( initialized )
  { reset_call_state();  /* don't report the previous call's warnings */
    return PYSE_OK;
  }
  in_cb = NULL;  /* the startup prompt must get EOF */
  surface_version++;
  status = protected_call(init_body,NULL);
  in_cb = saved_cb;
  if ( status == PYSE_OK ) initialized = 1;
  return status;
}

static void load_body(void *arg)
{ /* startup() would ask for another name if the file is missing; check
     first, using Evolver's own search (EVOLVERPATH, .fe suffix) */
  FILE *fd = path_open((char*)arg,NOTDATAFILENAME);
  if ( fd == NULL )
  { sprintf(errmsg,"Cannot open datafile %s.\n",(char*)arg);
    kb_error(PYSE_ERR_NO_DATAFILE,errmsg,RECOVERABLE);
  }
  fclose(fd);
  load_datafile((char*)arg);
}

int pyse_load(const char *path)
{ char name[PATHSIZE];
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized ) return invalid_surface();
  if ( strlen(path) >= sizeof(name) )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,"Datafile path is too long.");
  strcpy(name,path);
  surface_version++;
  load_serial++;
  return protected_call(load_body,name);
}

/**************************************************************************
 * Commands and expressions.
 */

struct command_args { char *text; int history; };

static void command_body(void *arg)
{ struct command_args *a = (struct command_args *)arg;
  /* housekeeping done between commands by exec_commands() */
  temp_free_all();
  free_discards(DISCARDS_ALL);
  if ( a->history )
    old_menu(a->text);
  else
  { command(a->text,NO_HISTORY);
    if ( change_flag ) recalc();
  }
}

/* Evolver's parser keeps pointers into the command text, so give it a
   private, writable copy that outlives the call. */
static char *command_text = NULL;

static int run_text(const char *text, int history)
{ struct command_args a;
  size_t n = strlen(text);
  char *copy = (char*)malloc(n+1);
  if ( !copy ) return PYSE_FATAL;
  memcpy(copy,text,n+1);
  free(command_text);
  command_text = copy;
  a.text = copy;
  a.history = history;
  return surface_call(command_body,&a);
}

int pyse_command(const char *text)
{ int status = run_text(text,1);
  surface_version++;  /* after, too: a failed command may have changed things */
  return status;
}

/* Run a command whose printf output tagged with CAPTURE_TAG is collected
   into out[0..n-1] instead of being printed.  %.17g reproduces a double
   exactly, and nothing gets defined in Evolver's symbol table. */
static int run_capture(const char *text, double *out, long n)
{ int status;
  capture_out = out;
  capture_max = n;
  capture_count = 0;
  capture_bad = 0;
  capture_active = 1;  /* switched off again at the end of protected_call() */
  status = run_text(text,0);
  capture_active = 0;
  if ( status != PYSE_OK ) return status;
  if ( capture_bad )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_VALUE,
                      "The expression did not evaluate to a number.");
  if ( capture_count != n )
  { char text2[200];
    snprintf(text2,sizeof(text2),
      "The expression gave %ld values, but %ld were expected.",capture_count,n);
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_VALUE,text2);
  }
  return PYSE_OK;
}

int pyse_eval(const char *expr, double *value)
{ int status;
  size_t n = strlen(expr) + 64;
  char *text = (char*)malloc(n);
  if ( !text ) return PYSE_FATAL;
  snprintf(text,n,"printf \"%s%%.17g\\n\", (%s)",CAPTURE_TAG,expr);
  status = run_capture(text,value,1);
  free(text);
  return status;
}

static const char *element_name(int type)
{ switch ( type )
  { case PYSE_VERTEX: return "vertex";
    case PYSE_EDGE: return "edge";
    case PYSE_FACET: return "facet";
    case PYSE_BODY: return "body";
  }
  return NULL;
}

int pyse_values(int type, const char *expr, double *out, long n)
{ int status;
  size_t len;
  char *text;
  const char *name = element_name(type);
  if ( in_protected ) return PYSE_BUSY;
  if ( name == NULL )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,"Unknown element type.");
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( n != web.skel[type].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Output array does not match the element count.");
  len = strlen(expr) + 96;
  text = (char*)malloc(len);
  if ( !text ) return PYSE_FATAL;
  /* inside foreach, unqualified attributes refer to the loop element */
  snprintf(text,len,"foreach %s do printf \"%s%%.17g\\n\", (%s)",
           name,CAPTURE_TAG,expr);
  status = run_capture(text,out,n);
  free(text);
  return status;
}

struct coords_args { const double *xyz; long n; int sdim; };

static void set_coords_body(void *arg)
{ struct coords_args *a = (struct coords_args *)arg;
  vertex_id v_id;
  long row = 0;
  int i;
  FOR_ALL_VERTICES(v_id)
  { REAL *x = get_coord(v_id);
    if ( row >= a->n ) break;
    for ( i = 0 ; i < a->sdim ; i++ )
      x[i] = (REAL)a->xyz[row*a->sdim + i];
    row++;
  }
  recalc();
}

int pyse_set_vertex_coords(const double *xyz, long n, int sdim)
{ struct coords_args a;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( n != web.skel[VERTEX].count || sdim != SDIM )
  { char text[200];
    snprintf(text,sizeof(text),
      "Expected a (%ld, %d) array of vertex coordinates, got (%ld, %d).",
      (long)web.skel[VERTEX].count,SDIM,n,sdim);
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,text);
  }
  a.xyz = xyz; a.n = n; a.sdim = sdim;
  surface_version++;
  return surface_call(set_coords_body,&a);
}

/* Fast writes: a vertex coordinate index (>= 0), or an extra attribute. */
struct fast_target { int coord; int extra; };

static int coordinate_index(const char *name)
{ if ( name[1] == 0 )
  { if ( name[0] == 'x' ) return 0;
    if ( name[0] == 'y' ) return 1;
    if ( name[0] == 'z' ) return 2;
    return -1;
  }
  if ( name[0] == 'x' && name[1] >= '1' && name[1] <= '9' )
  { char *end;
    long k = strtol(name+1,&end,10);
    if ( *end == 0 ) return (int)k - 1;
  }
  return -1;
}

static int find_fast_target(int type, const char *attribute, struct fast_target *t)
{ int n;
  struct extra *ex;
  t->coord = -1; t->extra = -1;
  if ( type < PYSE_VERTEX || type > PYSE_BODY ) return 0;
  if ( type == PYSE_VERTEX )
  { int k = coordinate_index(attribute);
    if ( k >= 0 ) { t->coord = k; return k < SDIM; }
  }
  ex = EXTRAS(type);
  for ( n = 0 ; n < web.skel[type].extra_count ; n++, ex++ )
    if ( strcmp(ex->name,attribute) == 0 )
    { if ( ex->array_spec.dim != 0
           || (ex->flags & (FUNCTION_ATTR|READ_ONLY_ATTR|VIRTUAL_ATTR|INTERNAL_ATTR)) )
        return 0;
      if ( ex->type != REAL_TYPE && ex->type != INTEGER_TYPE ) return 0;
      t->extra = n;
      return 1;
    }
  return 0;
}

int pyse_fast_attribute(int type, const char *attribute)
{ struct fast_target t;
  if ( !initialized || !surface_valid ) return 0;
  return find_fast_target(type,attribute,&t);
}

struct set_values_args
{ int type; struct fast_target t; const double *values; const unsigned char *mask; long n; };

static void set_values_body(void *arg)
{ struct set_values_args *a = (struct set_values_args *)arg;
  element_id id;
  long row = 0;
  FOR_ALL_ELEMENTS(a->type,id)
  { if ( row >= a->n ) inconsistent("element count changed");
    if ( !a->mask || a->mask[row] )
    { double v = a->values[row];
      if ( a->t.coord >= 0 )
        get_coord(id)[a->t.coord] = (REAL)v;
      else if ( EXTRAS(a->type)[a->t.extra].type == REAL_TYPE )
        *(REAL*)get_extra(id,a->t.extra) = (REAL)v;
      else
        *(int*)get_extra(id,a->t.extra) = (int)v;
    }
    row++;
  }
  expect_rows(row,a->n);
  /* moved vertices move the clock (what it stamps, a gradient step's
     record of where the last one ended among them, is stale) */
  if ( a->t.coord >= 0 ) global_timestamp++;
  if ( autorecalc_flag ) recalc();
}

int pyse_set_values(int type, const char *attribute, const double *values,
                    const unsigned char *mask, long n)
{ struct set_values_args a;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( !find_fast_target(type,attribute,&a.t) )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "No fast write for this attribute; use Evolver's set command.");
  if ( n != web.skel[type].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Value array does not match the element count.");
  a.type = type; a.values = values; a.mask = mask; a.n = n;
  surface_version++;
  return surface_call(set_values_body,&a);
}

/**************************************************************************
 * The residual: the vertex velocities a gradient step would use (forces
 * projected on the constraints and adjusted by the volume and quantity
 * multipliers), without moving anything. Zero at an equilibrium.
 */

/* out: one entry per vertex, then one per optimizing parameter */
struct residual_args { double *out; long n; int full; };

/* A vertex's shape directions: its surface normal(s), projected into its
   constraints, leaving out those almost perpendicular to the normal (sliding
   along a wire: a mesh freedom), as hessian_normal does. Returns how many. */
static int shape_directions(vertex_id v_id, REAL dirs[MAXCOORD][MAXCOORD])
{ MAT2D(nrm,MAXCOORD,MAXCOORD);
  MAT2D(tmp,MAXCOORD,MAXCOORD);
  int kk, q, i, count = 0;
  if ( get_vattr(v_id) & FIXED ) return 0;
  kk = new_calc_vertex_normal(v_id,nrm);
  kk = gram_schmidt(nrm,kk,SDIM);
  if ( kk <= 0 ) return -1;              /* no normal: use the whole velocity */
  for ( q = 0 ; q < kk ; q++ )
  { REAL *d = nrm[q];
    if ( get_vattr(v_id) & CONSTRAINT )
    { for ( i = 0 ; i < SDIM ; i++ ) tmp[0][i] = nrm[q][i];
      if ( project_vertex_normals(v_id,tmp,1) < 1 ) continue;
      if ( fabs(SDIM_dot(tmp[0],nrm[q])) < hessian_slant_cutoff ) continue;
      d = tmp[0];
    }
    for ( i = 0 ; i < SDIM ; i++ ) dirs[count][i] = d[i];
    count++;
  }
  return count;
}

/* The force balance in the shape directions. The vertex forces and the
   gradients of the fixed volumes and quantities, as vectors (what a gradient
   step uses before Evolver's own multiplier fit), are projected onto each
   vertex's shape directions; the multipliers (pressures) are fitted by least
   squares there, as Newton balances them, and what is left at each vertex is
   its residual. (Evolver's calc_lagrange() fits them over the full velocities,
   tangential mesh forces included, which leaves a uniform normal remainder
   at a Newton solution.) */
static void residual_body(void *arg)
{ struct residual_args *a = (struct residual_args *)arg;
  vertex_id v_id;
  long row = 0, k;
  int Q = 0, i, j, q, r, nd;
  double *A = NULL, *bvec = NULL, *lam = NULL, *proj = NULL, *popt = NULL;
  int *ndir = NULL;
  int mode = CALC_FORCE|CALC_VOLGRADS;
  REAL ***pre_normal = NULL;       /* shape directions found in parallel */
  int *pre_dims = NULL;
  vertex_id *pre_list = NULL;
  long pre_n = 0;
  find_fixed();
  calc_volgrads(DO_OPTS);
  calc_force();
  pressure_forces();
  partner_shift_grads(mode);
  convert_forms_to_vectors(mode);
  if ( a->n >= 4096 && !a->full )
  { pre_normal = temp_dmatrix3(a->n,MAXCOORD,MAXCOORD);
    pre_dims = fl_shape_normals(pre_normal,&pre_list,&pre_n);
    if ( pre_dims && pre_n != a->n ) { free(pre_dims); pre_dims = NULL; }
  }
  FOR_ALL_VERTICES(v_id)
  { volgrad *vg;
    for ( vg = get_vertex_vgrad(v_id) ; vg ; vg = vg->chain )
      if ( vg->fixnum + 1 > Q ) Q = vg->fixnum + 1;
  }
  /* per vertex and direction: the force, then the Q gradients, projected */
  proj = (double*)temp_calloc((size_t)a->n*MAXCOORD*(Q + 1) + 1,sizeof(double));
  ndir = (int*)temp_calloc((size_t)a->n + 1,sizeof(int));
  A = (double*)temp_calloc((size_t)Q*Q + 1,sizeof(double));
  bvec = (double*)temp_calloc((size_t)Q + 1,sizeof(double));
  lam = (double*)temp_calloc((size_t)Q + 1,sizeof(double));
  FOR_ALL_VERTICES(v_id)
  { REAL dirs[MAXCOORD][MAXCOORD];
    REAL *vel = get_velocity(v_id);
    volgrad *vg;
    double *pv;
    if ( row >= a->n ) inconsistent("vertex count changed");
    if ( pre_dims && pre_dims[row] != -2 )
    { nd = pre_dims[row];
      for ( k = 0 ; k < (nd > 0 ? nd : 0) ; k++ )
        for ( i = 0 ; i < SDIM ; i++ ) dirs[k][i] = pre_normal[row][k][i];
    }
    else nd = a->full ? -1 : shape_directions(v_id,dirs);
    if ( nd < 0 )              /* no normal, or all directions: the axes */
    { nd = SDIM;
      for ( i = 0 ; i < SDIM ; i++ )
        for ( j = 0 ; j < SDIM ; j++ ) dirs[i][j] = (i == j);
    }
    ndir[row] = nd;
    pv = proj + (size_t)row*MAXCOORD*(Q + 1);
    for ( k = 0 ; k < nd ; k++ )
    { pv[k*(Q + 1)] = (double)SDIM_dot(vel,dirs[k]);
      for ( vg = get_vertex_vgrad(v_id) ; vg ; vg = vg->chain )
        if ( vg->fixnum >= 0 && vg->velocity )
          pv[k*(Q + 1) + 1 + vg->fixnum] = (double)SDIM_dot(vg->velocity,dirs[k]);
    }
    row++;
  }
  expect_rows(row,a->n);
  /* each optimizing parameter: its force and the fixed quantities' gradients,
     scaled by 1/sqrt(n): a force on a parameter counts as that force spread
     over the vertices would */
  if ( optparamcount > 0 )
  { double scale = 1.0/sqrt((double)(a->n > 0 ? a->n : 1));
    popt = (double*)temp_calloc((size_t)optparamcount*(Q + 1) + 1,sizeof(double));
    for ( i = 0 ; i < optparamcount ; i++ )
    { double *e = popt + (size_t)i*(Q + 1);
      e[0] = -(double)optparam[i].grad*scale;
      if ( optparam_congrads )
        for ( q = 0 ; q < gen_quant_count ; q++ )
        { struct gen_quant *gq = GEN_QUANT(q);
          if ( (gq->flags & Q_DELETED) || gq->fixnum < 0 || gq->fixnum >= Q ) continue;
          e[1 + gq->fixnum] = (double)optparam_congrads[i][gq->vol_number]*scale;
        }
      for ( q = 0 ; q < Q ; q++ )
      { bvec[q] += e[0]*e[1 + q];
        for ( r = 0 ; r < Q ; r++ ) A[q*Q + r] += e[1 + q]*e[1 + r];
      }
    }
  }
  for ( k = 0 ; k < a->n ; k++ )        /* normal equations for the multipliers */
  { double *pv = proj + (size_t)k*MAXCOORD*(Q + 1);
    for ( i = 0 ; i < ndir[k] ; i++ )
    { double *e = pv + i*(Q + 1);
      for ( q = 0 ; q < Q ; q++ )
      { bvec[q] += e[0]*e[1 + q];
        for ( r = 0 ; r < Q ; r++ ) A[q*Q + r] += e[1 + q]*e[1 + r];
      }
    }
  }
  for ( q = 0 ; q < Q ; q++ ) lam[q] = bvec[q];
  for ( q = 0 ; q < Q ; q++ )            /* Gaussian elimination, partial pivoting */
  { int piv = q;
    for ( r = q + 1 ; r < Q ; r++ )
      if ( fabs(A[r*Q + q]) > fabs(A[piv*Q + q]) ) piv = r;
    if ( piv != q )
    { for ( j = 0 ; j < Q ; j++ )
      { double t = A[q*Q + j]; A[q*Q + j] = A[piv*Q + j]; A[piv*Q + j] = t; }
      { double t = lam[q]; lam[q] = lam[piv]; lam[piv] = t; }
    }
    if ( fabs(A[q*Q + q]) < 1e-300 ) continue;
    for ( r = q + 1 ; r < Q ; r++ )
    { double f = A[r*Q + q]/A[q*Q + q];
      for ( j = q ; j < Q ; j++ ) A[r*Q + j] -= f*A[q*Q + j];
      lam[r] -= f*lam[q];
    }
  }
  for ( q = Q - 1 ; q >= 0 ; q-- )
  { double t = lam[q];
    for ( j = q + 1 ; j < Q ; j++ ) t -= A[q*Q + j]*lam[j];
    lam[q] = fabs(A[q*Q + q]) < 1e-300 ? 0.0 : t/A[q*Q + q];
  }
  for ( k = 0 ; k < a->n ; k++ )
  { double *pv = proj + (size_t)k*MAXCOORD*(Q + 1);
    double sum = 0.0;
    for ( i = 0 ; i < ndir[k] ; i++ )
    { double *e = pv + i*(Q + 1);
      double c = e[0];
      for ( q = 0 ; q < Q ; q++ ) c -= lam[q]*e[1 + q];
      sum += c*c;
    }
    a->out[k] = sqrt(sum);
  }
  for ( i = 0 ; i < optparamcount ; i++ )
  { double *e = popt + (size_t)i*(Q + 1);
    double c = e[0];
    for ( q = 0 ; q < Q ; q++ ) c -= lam[q]*e[1 + q];
    a->out[a->n + i] = fabs(c);
  }
  if ( pre_dims ) free(pre_dims);
  if ( pre_normal ) free_temp_matrix3(pre_normal);
  vgrad_end();
}

/* The shape part of each vertex's velocity (one per vertex, in vertex order),
   then each optimizing parameter's force (see residual_body); n counts both. */
int pyse_residual(double *out, long n, int full)
{ struct residual_args a;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( n != web.skel[VERTEX].count + optparamcount )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Output array does not match the vertex count.");
  a.out = out; a.n = n - optparamcount; a.full = full;
  { /* plain forces: area normalization would divide each by its vertex's
       area (and in a string model can delete dangling edges) */
    int area_norm = web.area_norm_flag, status;
    web.area_norm_flag = 0;
    status = surface_call(residual_body,&a);
    web.area_norm_flag = area_norm;
    return status;
  }
}

/* Per vertex: 1 if it is fixed, on a constraint or on a boundary. */
struct attached_args { unsigned char *out; long n; };

static void attached_body(void *arg)
{ struct attached_args *a = (struct attached_args *)arg;
  vertex_id v_id;
  long row = 0;
  FOR_ALL_VERTICES(v_id)
  { if ( row >= a->n ) inconsistent("too many vertices");
    a->out[row++] = (get_vattr(v_id) & (FIXED|CONSTRAINT|BOUNDARY)) ? 1 : 0;
  }
  expect_rows(row,a->n);
}

int pyse_vertex_attached(unsigned char *out, long n)
{ struct attached_args a;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( n != web.skel[VERTEX].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Output array does not match the vertex count.");
  a.out = out; a.n = n;
  return surface_call(attached_body,&a);
}

/* Free vertices (not fixed, on no constraint or boundary) whose tangent cone
   Evolver's 'o' would pop: the cone types find_vertex_to_pop() pops, found
   the same way but without popping. Soapfilm only. */
struct poppable_args { long count; };

/* cone_analyze()'s types that find_vertex_to_pop() pops (src/popfilm.c
   defines them locally) */
#define PYSE_OTHERCONE   0
#define PYSE_KRAYNIKCONE 7
#define PYSE_CUBECONE    8
#define PYSE_ODD4CONE    9

static int pyse_vfcomp(const void *a, const void *b)
{ const struct verfacet *x = (const struct verfacet *)a;
  const struct verfacet *y = (const struct verfacet *)b;
  return x->v_id < y->v_id ? -1 : x->v_id > y->v_id ? 1 : 0;
}

static void poppable_body(void *arg)
{ struct poppable_args *a = (struct poppable_args *)arg;
  struct verfacet *vflist;
  facet_id f_id;
  facetedge_id fe;
  long count = 0, spot, dups;
  int k;
  a->count = 0;
  if ( web.representation != SOAPFILM || web.skel[FACET].count == 0 ) return;
  vflist = (struct verfacet *)temp_calloc(sizeof(struct verfacet),
                                          3*(size_t)web.skel[FACET].count);
  FOR_ALL_FACETS(f_id)
  { fe = get_facet_fe(f_id);
    for ( k = 0 ; k < 3 ; k++ )
    { vflist[count].f_id = f_id;
      vflist[count].v_id = get_fe_tailv(fe);
      count++;
      fe = get_next_edge(fe);
    }
  }
  qsort((char *)vflist,(size_t)count,sizeof(struct verfacet),pyse_vfcomp);
  for ( spot = 0 ; spot < count ; spot += dups )
  { vertex_id v_id = vflist[spot].v_id;
    struct cone_info cinfo;
    int type;
    for ( dups = 1 ; spot + dups < count ; dups++ )
      if ( !equal_id(vflist[spot+dups].v_id,v_id) ) break;
    if ( get_vattr(v_id) & (FIXED|CONSTRAINT|BOUNDARY) ) continue;
    type = cone_analyze(vflist + spot,(int)dups,&cinfo);
    temp_free((char *)cinfo.felist);
    if ( type == PYSE_KRAYNIKCONE || type == PYSE_CUBECONE || type == PYSE_ODD4CONE
         || type == PYSE_OTHERCONE )
      a->count++;
  }
  temp_free((char *)vflist);
}

int pyse_poppable_vertices(long *count)
{ struct poppable_args a;
  int status;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  a.count = 0;
  status = surface_call(poppable_body,&a);
  if ( count ) *count = a.count;
  return status;
}

/* Constraints in use and each vertex's signed distance to the ones it isn't on. */
struct conlist_args { int *nums; int *attrs; const char **names; int max; int count; };

static void conlist_body(void *arg)
{ struct conlist_args *a = (struct conlist_args *)arg;
  int i;
  a->count = 0;
  for ( i = 0 ; i < web.maxcon ; i++ )
  { struct constraint *con = get_constraint(i);
    if ( !(con->attr & IN_USE) ) continue;
    if ( a->nums && a->count < a->max )
    { a->nums[a->count] = i;
      if ( a->attrs ) a->attrs[a->count] = (int)con->attr;
      if ( a->names ) a->names[a->count] = (con->attr & NAMED_THING) ? con->name : "";
    }
    a->count++;
  }
}

int pyse_constraint_list(int *nums, int *attrs, const char **names, int max, int *count)
{ struct conlist_args a;
  int status;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  a.nums = nums; a.attrs = attrs; a.names = names; a.max = max; a.count = 0;
  status = surface_call(conlist_body,&a);
  if ( count ) *count = a.count;
  return status;
}

struct gaps_args { const int *nums; int ncon; double *out; long n; };

static void gaps_body(void *arg)
{ struct gaps_args *a = (struct gaps_args *)arg;
  vertex_id v_id;
  long row = 0;
  int c,i;
  for ( c = 0 ; c < a->ncon ; c++ )
    if ( a->nums[c] < 0 || a->nums[c] >= web.maxcon
         || !(get_constraint(a->nums[c])->attr & IN_USE) )
      inconsistent("constraint list changed");
  FOR_ALL_VERTICES(v_id)
  { REAL *x = get_coord(v_id);
    if ( row >= a->n ) inconsistent("too many vertices");
    for ( c = 0 ; c < a->ncon ; c++ )
    { double *o = a->out + (size_t)row*a->ncon + c;
      if ( v_on_constraint(v_id,a->nums[c]) ) *o = NAN;
      else
      { REAL f = 0.0, grad[MAXCOORD], g = 0.0;
        eval_all(get_constraint(a->nums[c])->formula,x,SDIM,&f,grad,v_id);
        for ( i = 0 ; i < SDIM ; i++ ) g += grad[i]*grad[i];
        *o = g > 0.0 ? (double)(f/sqrt(g)) : NAN;
      }
    }
    row++;
  }
  if ( row != a->n ) inconsistent("element count changed");
}

/* For each edge (mesh row order) and each of the constraints nums, whether
   the edge is on it (Evolver's e_on_constraint): one call for what the
   topology runs' tidying asked a wall at a time, 0.85 ms a query, 27k
   queries in ten minutes of a 16-bead pack. */
struct edges_on_args { const int *nums; int ncon; unsigned char *out; long n; };

static void edges_on_body(void *arg)
{ struct edges_on_args *a = (struct edges_on_args *)arg;
  edge_id e_id;
  long row = 0;
  int c;
  for ( c = 0 ; c < a->ncon ; c++ )
    if ( a->nums[c] < 0 || a->nums[c] >= web.maxcon
         || !(get_constraint(a->nums[c])->attr & IN_USE) )
      inconsistent("constraint list changed");
  FOR_ALL_EDGES(e_id)
  { if ( row >= a->n ) inconsistent("too many edges");
    for ( c = 0 ; c < a->ncon ; c++ )
      a->out[(size_t)row*a->ncon + c] = (unsigned char)(e_on_constraint(e_id,a->nums[c]) != 0);
    row++;
  }
  if ( row != a->n ) inconsistent("element count changed");
}

int pyse_edges_on_constraints(const int *nums, int ncon, unsigned char *out, long n)
{ struct edges_on_args a;
  if ( n != web.skel[EDGE].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "edges_on_constraints: output array does not match the edge count.");
  a.nums = nums; a.ncon = ncon; a.out = out; a.n = n;
  return surface_call(edges_on_body,&a);
}

struct corners_args { double *out; long n; };

static void corners_body(void *arg)
{ struct corners_args *a = (struct corners_args *)arg;
  facet_id f_id;
  long row = 0;
  int i,j;
  MAT2D(x,FACET_VERTS+1,MAXCOORD);
  FOR_ALL_FACETS(f_id)
  { double *o = a->out + (size_t)row*9;
    if ( row >= a->n ) inconsistent("too many facets");
    get_facet_verts(f_id,x,NULL);       /* unwrapped across a torus's sides */
    for ( i = 0 ; i < 3 ; i++ )
      for ( j = 0 ; j < 3 ; j++ )
        o[3*i+j] = j < SDIM ? (double)x[i][j] : 0.0;
    row++;
  }
  if ( row != a->n ) inconsistent("element count changed");
}

int pyse_facet_corners(double *out, long n)
{ struct corners_args a;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( web.representation != SOAPFILM || web.modeltype != LINEAR )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Facet corners need linear facets.");
  if ( n != web.skel[FACET].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Output array does not match the facet count.");
  a.out = out; a.n = n;
  return surface_call(corners_body,&a);
}

struct cnormals_args { int num; double *out; long n; const int64_t *rows; long nrows; };

static void cnormals_body(void *arg)
{ struct cnormals_args *a = (struct cnormals_args *)arg;
  vertex_id v_id;
  long row = 0, k;
  int i;
  double *out = a->out;    /* (with rows: per vertex, then gathered) */
  unsigned char *want = NULL;
  if ( a->num < 0 || a->num >= web.maxcon || !(get_constraint(a->num)->attr & IN_USE) )
    inconsistent("no such constraint");
  if ( a->rows )
  { out = (double*)temp_calloc((size_t)(a->n > 0 ? a->n : 1)*3,sizeof(double));
    want = (unsigned char*)temp_calloc((size_t)(a->n > 0 ? a->n : 1),1);
    for ( k = 0 ; k < a->nrows ; k++ )
    { if ( a->rows[k] < 0 || a->rows[k] >= a->n ) inconsistent("vertex row out of range");
      want[a->rows[k]] = 1;
    }
  }
  FOR_ALL_VERTICES(v_id)
  { REAL f = 0.0, grad[MAXCOORD], g = 0.0;
    double *o = out + (size_t)row*3;
    if ( row >= a->n ) inconsistent("too many vertices");
    if ( want && !want[row] ) { row++; continue; }
    for ( i = 0 ; i < MAXCOORD ; i++ ) grad[i] = 0.0;
    eval_all(get_constraint(a->num)->formula,get_coord(v_id),SDIM,&f,grad,v_id);
    for ( i = 0 ; i < SDIM ; i++ ) g += grad[i]*grad[i];
    g = sqrt(g);
    for ( i = 0 ; i < 3 ; i++ )
      o[i] = (g > 0.0 && i < SDIM) ? (double)(grad[i]/g) : (g > 0.0 ? 0.0 : NAN);
    row++;
  }
  if ( row != a->n ) inconsistent("element count changed");
  if ( a->rows )
  { for ( k = 0 ; k < a->nrows ; k++ )
      for ( i = 0 ; i < 3 ; i++ )
        a->out[k*3 + i] = out[a->rows[k]*3 + i];
    temp_free((char*)out);
    temp_free((char*)want);
  }
}

int pyse_constraint_normals(int num, double *out, long n)
{ return pyse_constraint_normals_rows(num,NULL,0,out,n);
}

int pyse_constraint_normals_rows(int num, const int64_t *rows, long nrows, double *out, long n)
{ struct cnormals_args a;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( n != web.skel[VERTEX].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Vertex count changed.");
  a.num = num; a.out = out; a.n = n; a.rows = rows; a.nrows = nrows;
  return surface_call(cnormals_body,&a);
}

int pyse_constraint_gaps(const int *nums, int ncon, double *out, long n)
{ struct gaps_args a;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( n != web.skel[VERTEX].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Output array does not match the vertex count.");
  a.nums = nums; a.ncon = ncon; a.out = out; a.n = n;
  return surface_call(gaps_body,&a);
}

/**************************************************************************
 * Surface snapshots.  These run inside the guard too, so an inconsistency
 * becomes an Evolver error instead of a bad memory read.
 */

static void inconsistent(const char *what)
{ sprintf(errmsg,"Surface is inconsistent: %s.\n",what);
  kb_error(PYSE_ERR_INVALID_SURFACE,errmsg,RECOVERABLE);
}

/* Vertex maps are allocated with temp_calloc(), which Evolver frees
   between commands. */

static void vertex_map_build(struct vertex_map *m)
{ long k, row = 0;
  vertex_id v_id;
  m->size = (long)web.skel[VERTEX].max_ord + 1;
  if ( m->size < 1 ) m->size = 1;
  m->rows = (long*)temp_calloc(m->size,sizeof(long));
  for ( k = 0 ; k < m->size ; k++ ) m->rows[k] = -1;
  FOR_ALL_VERTICES(v_id)
  { long ord = ordinal(v_id);
    if ( ord < 0 || ord >= m->size ) inconsistent("vertex number out of range");
    m->rows[ord] = row++;
  }
}

static void vertex_map_get(struct vertex_map *m)
{ if ( shared_vertex_map ) *m = *shared_vertex_map;
  else vertex_map_build(m);
}

/* Free a table from vertex_map_get() unless it is the shared one. (After
   an error the table stays allocated until Evolver's next temp_free_all.) */
static void vertex_map_release(struct vertex_map *m)
{ if ( !shared_vertex_map || m->rows != shared_vertex_map->rows )
    temp_free((char*)m->rows);
}

static int64_t vertex_row(struct vertex_map *m, vertex_id v_id)
{ long ord;
  if ( !valid_id(v_id) ) inconsistent("element refers to a missing vertex");
  ord = ordinal(v_id);
  if ( ord < 0 || ord >= m->size || m->rows[ord] < 0 )
    inconsistent("element refers to a deleted vertex");
  return m->rows[ord];
}

static void expect_rows(long row, long n)
{ if ( row != n ) inconsistent("element count changed");
}

struct vertices_args
{ double *xyz; int64_t *ids; unsigned char *fixed; long n; int sdim; };

static void vertices_body(void *arg)
{ struct vertices_args *a = (struct vertices_args *)arg;
  vertex_id v_id;
  long row = 0;
  int i;
  if ( a->sdim != SDIM ) inconsistent("space dimension changed");
  FOR_ALL_VERTICES(v_id)
  { REAL *x = get_coord(v_id);
    if ( row >= a->n ) inconsistent("too many vertices");
    for ( i = 0 ; i < a->sdim ; i++ ) a->xyz[row*a->sdim+i] = (double)x[i];
    a->ids[row] = ordinal(v_id) + 1;
    a->fixed[row] = (get_vattr(v_id) & FIXED) ? 1 : 0;
    row++;
  }
  expect_rows(row,a->n);
}

int pyse_get_vertices(double *xyz, int64_t *ids, unsigned char *fixed, long n,
                      int sdim)
{ struct vertices_args a;
  a.xyz = xyz; a.ids = ids; a.fixed = fixed; a.n = n; a.sdim = sdim;
  return surface_call(vertices_body,&a);
}

struct edges_args { int64_t *verts; int64_t *ids; long n; };

static void edges_body(void *arg)
{ struct edges_args *a = (struct edges_args *)arg;
  struct vertex_map m;
  edge_id e_id;
  long row = 0;
  vertex_map_get(&m);
  FOR_ALL_EDGES(e_id)
  { if ( row >= a->n ) inconsistent("too many edges");
    a->verts[2*row]   = vertex_row(&m,get_edge_tailv(e_id));
    a->verts[2*row+1] = vertex_row(&m,get_edge_headv(e_id));
    a->ids[row] = ordinal(e_id) + 1;
    row++;
  }
  expect_rows(row,a->n);
  vertex_map_release(&m);
}

static int64_t body_number(body_id b_id)
{ return valid_id(b_id) ? ordinal(b_id) + 1 : 0; }

struct facets_args { int64_t *verts; int64_t *ids; int64_t *bodies; long n; };

static void facets_body(void *arg)
{ struct facets_args *a = (struct facets_args *)arg;
  struct vertex_map m;
  vertex_id *corners;
  facet_id f_id;
  long row = 0;
  if ( web.representation != SOAPFILM )
    kb_error(PYSE_ERR_BAD_ARGUMENT,
      "Facet triangles exist only in the soapfilm representation.\n",RECOVERABLE);
  vertex_map_get(&m);
  corners = fl_facet_corners();  /* same corners as walking the edges */
  FOR_ALL_FACETS(f_id)
  { int i;
    if ( row >= a->n ) inconsistent("too many facets");
    if ( corners )
    { vertex_id *c = corners + 3*ordinal(f_id);
      for ( i = 0 ; i < 3 ; i++ ) a->verts[3*row+i] = vertex_row(&m,c[i]);
    }
    else
    { facetedge_id fe = get_facet_fe(f_id);
      for ( i = 0 ; i < 3 ; i++ )
      { if ( !valid_id(fe) ) inconsistent("facet with a missing edge");
        a->verts[3*row+i] = vertex_row(&m,get_fe_tailv(fe));
        fe = get_next_edge(fe);
      }
    }
    a->ids[row] = ordinal(f_id) + 1;
    a->bodies[2*row]   = body_number(get_facet_body(f_id));
    a->bodies[2*row+1] = body_number(get_facet_body(inverse_id(f_id)));
    row++;
  }
  expect_rows(row,a->n);
  vertex_map_release(&m);
}

struct bodies_args
{ int64_t *ids; double *volume, *target, *pressure; unsigned char *fixed; long n; };

static void bodies_body(void *arg)
{ struct bodies_args *a = (struct bodies_args *)arg;
  body_id b_id;
  long row = 0;
  FOR_ALL_BODIES(b_id)
  { if ( row >= a->n ) inconsistent("too many bodies");
    a->ids[row] = ordinal(b_id) + 1;
    a->volume[row] = (double)get_body_volume(b_id);
    a->fixed[row] = (get_battr(b_id) & FIXEDVOL) ? 1 : 0;
    a->target[row] = a->fixed[row] ? (double)get_body_fixvol(b_id) : NAN;
    a->pressure[row] = (double)get_body_pressure(b_id);
    row++;
  }
  expect_rows(row,a->n);
}

int pyse_get_bodies(int64_t *ids, double *volume, double *target,
                    double *pressure, unsigned char *fixed, long n)
{ struct bodies_args a;
  a.ids = ids; a.volume = volume; a.target = target; a.pressure = pressure;
  a.fixed = fixed; a.n = n;
  return surface_call(bodies_body,&a);
}

/**************************************************************************
 * High-order elements.
 */

int pyse_element_order(void)
{ switch ( web.modeltype )
  { case QUADRATIC: return 2;
    case LAGRANGE: return web.lagrange_order;
    default: return 1;
  }
}

int pyse_bezier(void)
{ return web.modeltype == LAGRANGE && bezier_flag; }

int pyse_element_node_count(int type)
{ int p = pyse_element_order();
  if ( web.representation == SIMPLEX ) return -1;
  if ( type == PYSE_EDGE ) return p + 1;
  if ( type == PYSE_FACET && web.representation == SOAPFILM )
    return (p+1)*(p+2)/2;
  return -1;
}

int pyse_node_layout(int type, int *index, int nodes_per)
{ int p = pyse_element_order();
  int dim = (type == PYSE_EDGE) ? 1 : 2;
  int k, m, inx[3];
  if ( nodes_per != pyse_element_node_count(type) ) return -1;
  if ( web.modeltype == QUADRATIC )
  { /* our own order: corners, then edge midpoints (see nodes_body) */
    static const int edge2[3][2] = { {2,0}, {1,1}, {0,2} };
    static const int facet2[6][3] =
      { {2,0,0}, {0,2,0}, {0,0,2}, {1,1,0}, {0,1,1}, {1,0,1} };
    for ( k = 0 ; k < nodes_per ; k++ )
      for ( m = 0 ; m <= dim ; m++ )
        index[k*(dim+1)+m] = (dim == 1) ? edge2[k][m] : facet2[k][m];
    return 0;
  }
  if ( web.modeltype != LAGRANGE )
  { for ( k = 0 ; k < nodes_per ; k++ )
      for ( m = 0 ; m <= dim ; m++ )
        index[k*(dim+1)+m] = (k == m);
    return 0;
  }
  /* Lagrange: the order Evolver stores nodes in (see model.c) */
  for ( m = 0 ; m <= dim ; m++ ) inx[m] = 0;
  inx[0] = p;
  k = 0;
  do
  { for ( m = 0 ; m <= dim ; m++ ) index[k*(dim+1)+m] = inx[m];
    k++;
  } while ( k < nodes_per && increment_lagrange_index(dim,inx) );
  return 0;
}

struct nodes_args { int type; int64_t *nodes; long n; int nodes_per; };

static void nodes_body(void *arg)
{ struct nodes_args *a = (struct nodes_args *)arg;
  struct vertex_map m;
  long row = 0;
  int k, np = a->nodes_per;
  if ( np != pyse_element_node_count(a->type) )
    kb_error(PYSE_ERR_BAD_ARGUMENT,
      "No node layout for this element type and model.\n",RECOVERABLE);
  vertex_map_get(&m);
  if ( a->type == PYSE_EDGE )
  { edge_id e_id;
    FOR_ALL_EDGES(e_id)
    { int64_t *out = a->nodes + row*np;
      if ( row >= a->n ) inconsistent("too many edges");
      if ( web.modeltype == LAGRANGE )
      { vertex_id *v = get_edge_vertices(e_id);
        for ( k = 0 ; k < np ; k++ ) out[k] = vertex_row(&m,v[k]);
      }
      else
      { out[0] = vertex_row(&m,get_edge_tailv(e_id));
        if ( web.modeltype == QUADRATIC )
          out[1] = vertex_row(&m,get_edge_midv(e_id));
        out[np-1] = vertex_row(&m,get_edge_headv(e_id));
      }
      row++;
    }
  }
  else
  { facet_id f_id;
    FOR_ALL_FACETS(f_id)
    { int64_t *out = a->nodes + row*np;
      if ( row >= a->n ) inconsistent("too many facets");
      if ( web.modeltype == LAGRANGE )
      { vertex_id *v = get_facet_vertices(f_id);
        for ( k = 0 ; k < np ; k++ ) out[k] = vertex_row(&m,v[k]);
      }
      else
      { facetedge_id fe = get_facet_fe(f_id);
        for ( k = 0 ; k < 3 ; k++ )
        { if ( !valid_id(fe) ) inconsistent("facet with a missing edge");
          out[k] = vertex_row(&m,get_fe_tailv(fe));
          if ( web.modeltype == QUADRATIC )
            out[3+k] = vertex_row(&m,get_fe_midv(fe));
          fe = get_next_edge(fe);
        }
      }
      row++;
    }
  }
  expect_rows(row,a->n);
  vertex_map_release(&m);
}

static void mesh_body(void *arg)
{ struct pyse_mesh_arrays *m = (struct pyse_mesh_arrays *)arg;
  struct vertex_map map;
  struct vertices_args va;
  struct edges_args ea;
  vertex_map_build(&map);
  shared_vertex_map = &map;
  va.xyz = m->xyz; va.ids = m->vertex_ids; va.fixed = m->fixed; va.n = m->nv;
  va.sdim = m->sdim;
  vertices_body(&va);
  ea.verts = m->edges; ea.ids = m->edge_ids; ea.n = m->ne;
  edges_body(&ea);
  if ( m->facets )
  { struct facets_args fa;
    fa.verts = m->facets; fa.ids = m->facet_ids; fa.bodies = m->facet_bodies;
    fa.n = m->nf;
    facets_body(&fa);
  }
  if ( m->edge_nodes )
  { struct nodes_args na;
    na.type = PYSE_EDGE; na.nodes = m->edge_nodes; na.n = m->ne;
    na.nodes_per = m->edge_nodes_per;
    nodes_body(&na);
  }
  if ( m->facet_nodes )
  { struct nodes_args na;
    na.type = PYSE_FACET; na.nodes = m->facet_nodes; na.n = m->nf;
    na.nodes_per = m->facet_nodes_per;
    nodes_body(&na);
  }
  shared_vertex_map = NULL;
  temp_free((char*)map.rows);
}

int pyse_get_mesh(struct pyse_mesh_arrays *m)
{ return surface_call(mesh_body,m); }

/**************************************************************************
 * Parameters and named quantities.
 */

static int is_datafile_parameter(int i)
{ struct global *g;
  if ( i >= datafile_global_limit || i >= web.global_count ) return 0;
  g = globals(i);
  if ( !(g->flags & ORDINARY_PARAM) ) return 0;
  if ( g->flags & (INTERNAL_NAME|READONLY|ARRAY_PARAM|STRINGVAL|SUBROUTINE
                   |FILE_VALUES|QUANTITY_TYPES|METHOD_TYPES|PROCEDURE_NAME
                   |FUNCTION_NAME|GLOB_LOCALVAR|DYNAMIC_LOAD_FUNC) )
    return 0;
  return 1;
}

long pyse_parameter_count(void)
{ long count = 0;
  int i;
  if ( !initialized || !surface_valid ) return 0;
  for ( i = 0 ; i < web.global_count ; i++ )
    if ( is_datafile_parameter(i) ) count++;
  return count;
}

struct parameters_args
{ char (*names)[PYSE_NAME_SIZE]; double *values; unsigned char *optimizing; long n; };

static void parameters_body(void *arg)
{ struct parameters_args *a = (struct parameters_args *)arg;
  long row = 0;
  int i;
  for ( i = 0 ; i < web.global_count ; i++ )
  { struct global *g;
    if ( !is_datafile_parameter(i) ) continue;
    if ( row >= a->n ) inconsistent("parameter count changed");
    g = globals(i);
    snprintf(a->names[row],PYSE_NAME_SIZE,"%s",g->name);
    a->values[row] = (double)g->value.real;
    a->optimizing[row] = (g->flags & OPTIMIZING_PARAMETER) ? 1 : 0;
    row++;
  }
  expect_rows(row,a->n);
}

int pyse_get_parameters(char (*names)[PYSE_NAME_SIZE], double *values,
                        unsigned char *optimizing, long n)
{ struct parameters_args a;
  a.names = names; a.values = values; a.optimizing = optimizing; a.n = n;
  return surface_call(parameters_body,&a);
}

static int is_named_quantity(int i)
{ struct gen_quant *q = GEN_QUANT(i);
  return !(q->flags & (Q_DELETED|DEFAULT_QUANTITY));
}

int pyse_optparam_count(void)
{ return ( initialized && surface_valid ) ? optparamcount : 0; }

long pyse_quantity_count(void)
{ long count = 0;
  int i;
  if ( !initialized || !surface_valid ) return 0;
  for ( i = 0 ; i < gen_quant_count ; i++ )
    if ( is_named_quantity(i) ) count++;
  return count;
}

struct quantities_args
{ char (*names)[PYSE_NAME_SIZE]; double *value, *target, *modulus, *pressure;
  int *kind; long n; };

static void quantities_body(void *arg)
{ struct quantities_args *a = (struct quantities_args *)arg;
  long row = 0;
  int i;
  /* bring every quantity's value up to date (none to report: skip it, as
     relax() asks after every step) */
  if ( a->n > 0 )
    calc_content(Q_ENERGY|Q_FIXED|Q_INFO|Q_CONSERVED);
  for ( i = 0 ; i < gen_quant_count ; i++ )
  { struct gen_quant *q = GEN_QUANT(i);
    if ( !is_named_quantity(i) ) continue;
    if ( row >= a->n ) inconsistent("quantity count changed");
    snprintf(a->names[row],PYSE_NAME_SIZE,"%s",q->name);
    a->value[row] = (double)q->value;
    a->target[row] = (double)q->target;
    a->modulus[row] = (double)q->modulus;
    a->pressure[row] = (double)q->pressure;
    if ( q->flags & Q_FIXED ) a->kind[row] = 1;
    else if ( q->flags & Q_CONSERVED ) a->kind[row] = 3;
    else if ( q->flags & Q_INFO ) a->kind[row] = 2;
    else a->kind[row] = 0;
    row++;
  }
  expect_rows(row,a->n);
}

/* The methods (by their Evolver names) of the energy quantities, one row per
   method instance, at most max; count gets the total. */
struct energy_methods_args { char (*names)[PYSE_NAME_SIZE]; int max; int count; };

static void energy_methods_body(void *arg)
{ struct energy_methods_args *a = (struct energy_methods_args *)arg;
  int i,j;
  a->count = 0;
  for ( i = 0 ; i < gen_quant_count ; i++ )
  { struct gen_quant *q = GEN_QUANT(i);
    if ( (q->flags & Q_DELETED) || !(q->flags & Q_ENERGY) ) continue;
    for ( j = 0 ; j < q->method_count ; j++ )
    { struct method_instance *mi = METH_INSTANCE(q->meth_inst[j]);
      if ( a->count < a->max )
        snprintf(a->names[a->count],PYSE_NAME_SIZE,"%s",
                 basic_gen_methods[mi->gen_method].name);
      a->count++;
    }
  }
}

int pyse_energy_methods(char (*names)[PYSE_NAME_SIZE], int max, int *count)
{ struct energy_methods_args a;
  int status;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  a.names = names; a.max = max; a.count = 0;
  status = surface_call(energy_methods_body,&a);
  if ( count ) *count = a.count;
  return status;
}

struct energy_terms_args
{ char (*names)[PYSE_NAME_SIZE]; char (*methods)[PYSE_NAME_SIZE]; double *energy;
  int max; int count; };

/* each energy quantity's method instances with what they add to the energy
   (quantity modulus * instance value, which includes the instance's modulus) */
static void energy_terms_body(void *arg)
{ struct energy_terms_args *a = (struct energy_terms_args *)arg;
  int i,j;
  a->count = 0;
  calc_content(Q_ENERGY|Q_FIXED|Q_INFO|Q_CONSERVED);
  for ( i = 0 ; i < gen_quant_count ; i++ )
  { struct gen_quant *q = GEN_QUANT(i);
    if ( (q->flags & Q_DELETED) || !(q->flags & Q_ENERGY) ) continue;
    for ( j = 0 ; j < q->method_count ; j++ )
    { struct method_instance *mi = METH_INSTANCE(q->meth_inst[j]);
      if ( a->count < a->max )
      { snprintf(a->names[a->count],PYSE_NAME_SIZE,"%s",mi->name);
        snprintf(a->methods[a->count],PYSE_NAME_SIZE,"%s",
                 basic_gen_methods[mi->gen_method].name);
        a->energy[a->count] = (double)(q->modulus*mi->value);  /* (value has its modulus) */
      }
      a->count++;
    }
  }
}

int pyse_energy_terms(char (*names)[PYSE_NAME_SIZE], char (*methods)[PYSE_NAME_SIZE],
                      double *energy, int max, int *count)
{ struct energy_terms_args a;
  int status;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  a.names = names; a.methods = methods; a.energy = energy; a.max = max; a.count = 0;
  status = surface_call(energy_terms_body,&a);
  if ( count ) *count = a.count;
  return status;
}

int pyse_get_quantities(char (*names)[PYSE_NAME_SIZE], double *value,
                        double *target, double *modulus, double *pressure,
                        int *kind, long n)
{ struct quantities_args a;
  a.names = names; a.value = value; a.target = target; a.modulus = modulus;
  a.pressure = pressure; a.kind = kind; a.n = n;
  return surface_call(quantities_body,&a);
}

/**************************************************************************
 * Settings and results.
 */

void pyse_set_output_callback(pyse_output_fn fn, void *userdata)
{ out_cb = fn; out_ud = userdata; }

void pyse_set_input_callback(pyse_input_fn fn, void *userdata)
{ in_cb = fn; in_ud = userdata; }

void pyse_set_handle_sigint(int flag) { handle_sigint = flag; }

void pyse_set_threads(int n) { fl_set_threads(n); }
int pyse_threads(void) { return fl_threads(); }
int pyse_thread_setting(void) { return fl_thread_setting(); }

int pyse_set_solver(const char *name)
{ int newh, quiet = quiet_flag;
  if ( strcmp(name,"mumps") == 0 )
  { if ( !fl_have_mumps() ) return 0;
    newh = MUMPS_FACTORING;
  }
  else if ( strcmp(name,"evolver") == 0 ) newh = MINDEG_FACTORING;
  else return 0;
  if ( newh != ysmp_flag )
  { quiet_flag = 1;   /* change_hessian_functions() reports the switch */
    change_hessian_functions(ysmp_flag,newh);
    quiet_flag = quiet;
  }
  return 1;
}

const char *pyse_solver(void)
{ switch ( ysmp_flag )
  { case MUMPS_FACTORING: return "mumps";
    case MINDEG_FACTORING: return "evolver";
    case YSMP_FACTORING: return "ysmp";
    case METIS_FACTORING: return "metis";
    case MKL_FACTORING: return "mkl";
  }
  return "unknown";
}

int pyse_last_errnum(void) { return err_num; }
const char *pyse_last_errmsg(void) { return err_msg; }
int pyse_exit_code(void) { return exit_code; }
int pyse_warning_count(void) { return warning_count; }
const char *pyse_warning(int i)
{ return ( i >= 0 && i < warning_count ) ? warnings[i] : ""; }

/**************************************************************************
 * Unguarded state.
 */

int pyse_is_initialized(void) { return initialized; }
int pyse_surface_valid(void) { return initialized && surface_valid; }
long pyse_surface_version(void) { return surface_version; }

long pyse_count(int type)
{ if ( type < 0 || type > PYSE_BODY ) return -1;
  return web.skel[type].count;
}

int pyse_sdim(void) { return SDIM; }
int pyse_representation(void) { return web.representation; }
int pyse_modeltype(void) { return web.modeltype; }
int pyse_lagrange_order(void) { return web.lagrange_order; }
int pyse_torus(void) { return web.torus_flag; }
int pyse_crystalline(void) { return web.wulff_flag; }
int pyse_symmetry(void) { return web.torus_flag || web.symmetry_flag; }
long pyse_topology_stamp(void) { return top_timestamp; }
/* The n lowest eigenpairs of the Hessian with the constraints held (as
   saddle and eigenprobe set it up, with linear_metric or area normalization
   if on): Evolver's ritz shifted below every eigenvalue, so the nearest are
   the lowest. Each eigenvector as a displacement per vertex: a central
   difference of Evolver's own move along it (through the boundaries'
   parameters and the projection onto constraints), undone. values (n),
   vectors (n * vertices * SDIM); found: how many ritz gave (at most n). */
extern REAL *evalues;              /* do_ritz()'s eigenvalues (hessian2.c) */

struct modes_args { int n; double *values; double *vectors; long nv; int found; };

static void modes_body(void *arg)
{ struct modes_args *a = (struct modes_args *)arg;
  struct linsys S;
  REAL *rhs = NULL, **vecs = NULL, t;
  struct oldcoord saver;
  vertex_id v_id;
  int k, i, quiet = quiet_flag;
  long row;

  memset(&S,0,sizeof(S));
  memset(&saver,0,sizeof(saver));
  hmode = hessian_normal_flag;
  hessian_init(&S,&rhs);
  hess_flag = 1; rhs_flag = 1;
  hessian_fill(&S,&rhs);
  (*sp_AIJ_setup_func)(S.A_rows,&S);
  (*sp_constraint_setup_func)(web.skel[BODY].max_ord+1 + gen_quant_count,&S);
  if ( sp_ordering_func ) (*sp_ordering_func)(&S);
  if ( hessian_linear_metric_flag ) linear_metric_setup(&S,&Met);
  else if ( web.area_norm_flag ) star_metric_setup(&S,&Met);

  /* a shift below every eigenvalue (as lowest_eigenpair() finds one) */
  S.lambda = -0.01;
  sp_factor(&S,MKL_INDEF);
  (*sp_hess_project_setup_func)(&S);
  while ( S.neg > 0 )
  { S.lambda *= 10;
    sp_factor(&S,MKL_INDEF);
    (*sp_hess_project_setup_func)(&S);
  }
  vecs = dmatrix(0,a->n-1,0,S.N+S.concount);
  quiet_flag = 1;
  do_ritz(&S,S.lambda,a->n,vecs);
  quiet_flag = quiet;
  a->found = a->n;
  if ( a->found > S.N - (augmented_hessian_mode ? 2 : 1)*S.CN )
    a->found = S.N - (augmented_hessian_mode ? 2 : 1)*S.CN;
  if ( a->found < 0 ) a->found = 0;

  /* ritz scales each vector to a largest component of a tenth of the
     surface's size: step a millionth of that */
  t = 1e-6;
  for ( k = 0 ; k < a->found ; k++ )
  { double *d = a->vectors + (size_t)k*a->nv*SDIM;
    a->values[k] = (double)evalues[k];
    save_coords(&saver,SAVE_SEPARATE);
    hessian_move(t,ACTUAL_MOVE,vecs[k]);
    row = 0;
    FOR_ALL_VERTICES(v_id)
    { REAL *x = get_coord(v_id);
      if ( row >= a->nv ) inconsistent("vertex count changed");
      for ( i = 0 ; i < SDIM ; i++ ) d[row*SDIM + i] = (double)x[i];
      row++;
    }
    restore_coords(&saver,SAVE_SEPARATE);
    hessian_move(-t,ACTUAL_MOVE,vecs[k]);
    row = 0;
    FOR_ALL_VERTICES(v_id)
    { REAL *x = get_coord(v_id);
      for ( i = 0 ; i < SDIM ; i++ )
        d[row*SDIM + i] = (d[row*SDIM + i] - (double)x[i])/(2*t);
      row++;
    }
    restore_coords(&saver,SAVE_SEPARATE);
    unsave_coords(&saver,SAVE_SEPARATE);
  }
  free_matrix(vecs);
  hessian_exit(NULL);
  free_system(&S);
  if ( hessian_linear_metric_flag || web.area_norm_flag ) free_system(&Met);
  temp_free((char*)rhs);
  recalc();
}

int pyse_modes(int n, double *values, double *vectors, long nv, int *found)
{ struct modes_args a;
  int status;
  if ( n < 1 )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,"modes: n must be at least 1.");
  if ( nv != web.skel[VERTEX].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "modes: output array does not match the vertex count.");
  a.n = n; a.values = values; a.vectors = vectors; a.nv = nv; a.found = 0;
  status = surface_call(modes_body,&a);
  *found = a.found;
  return status;
}

/* Two Newton steps from the same point, at a body's target and at that plus
   delta (arclength's corrector: the step's change with the value), from one
   factorization: only the constraint rows of the right side change with the
   target (hessian_fill's last loops), the Hessian doesn't. Each as Evolver's
   hessian command makes it (the multipliers updated by the solve, the move
   with the mesh not following, projected at that step's target), the
   vertices and multipliers it ends at returned; the surface, its targets and
   multipliers put back as they were. The corrector made the second step by
   a second hessian, factorizing the same matrix again: half its cost, a
   fifth of a 16-bead pack's run. */
struct pair_args { int body; double delta; double *x0, *m0, *x1, *m1; long nv, nm; };

static void constraint_rows(struct linsys *S, REAL *rhs)
{ body_id b_id;
  int k;
  if ( !everything_quantities_flag )
  { FOR_ALL_BODIES(b_id)
    { REAL *current = rhs + S->bodyrowstart + loc_ordinal(b_id);
      if ( get_battr(b_id) & FIXEDVOL )
        *current = -(get_body_fixvol(b_id) - get_body_volume(b_id));
    }
  }
  for ( k = 0 ; k < gen_quant_count ; k++ )
  { struct gen_quant *q = GEN_QUANT(k);
    if ( q->flags & (Q_DELETED|Q_REDUNDANT) ) continue;
    if ( q->flags & Q_FIXED )
      rhs[S->quanrowstart + k] = -(q->target - q->value);
  }
}

static void pair_coords(double *out, long nv)
{ vertex_id v_id;
  long row = 0;
  int j;
  FOR_ALL_VERTICES(v_id)
  { REAL *x = get_coord(v_id);
    if ( row >= nv ) inconsistent("too many vertices");
    for ( j = 0 ; j < SDIM ; j++ ) out[row*SDIM + j] = (double)x[j];
    row++;
  }
  if ( row != nv ) inconsistent("vertex count changed");
}

static void pair_body(void *arg)
{ struct pair_args *a = (struct pair_args *)arg;
  struct linsys S;
  REAL *rhs = NULL, *rhs2 = NULL, *X = NULL, *X2 = NULL;
  body_id b_id = get_ordinal_id(BODY,a->body - 1);
  double *p0;
  REAL target;
  int was;
  if ( !valid_id(b_id) || !(get_battr(b_id) & FIXEDVOL) )
    kb_error(1307,"newton_pair: the body has no target volume.\n",RECOVERABLE);
  target = get_body_fixvol(b_id);
  p0 = (double *)temp_calloc(a->nm > 0 ? a->nm : 1,sizeof(double));
  pyse_get_multipliers(p0,a->nm);
  was = fl_set_follow(0);
  memset(&S,0,sizeof(S));
  hmode = hessian_normal_flag;
  hessian_cleanup();
  hessian_init(&S,&rhs);
  hess_flag = 1; rhs_flag = 1;
  hessian_fill(&S,&rhs);
  rhs2 = (REAL *)temp_calloc(S.total_rows,sizeof(REAL));
  memcpy(rhs2,rhs,S.total_rows*sizeof(REAL));
  set_body_fixvol(b_id,target + a->delta);
  constraint_rows(&S,rhs2);
  set_body_fixvol(b_id,target);
  sp_Hessian_solver(&S,rhs,&X);                 /* factorized here */
  pyse_get_multipliers(a->m0,a->nm);
  pyse_set_multipliers(p0,a->nm);
  X2 = (REAL *)temp_calloc(S.N,sizeof(REAL));
  sp_hessian_solve(&S,rhs2,X2,SET_PRESSURE);    /* the same factors */
  fl_hessian_deflate(&S,X2);
  pyse_get_multipliers(a->m1,a->nm);
  hessian_iterate_flag = 1;
  pyse_set_multipliers(a->m0,a->nm);
  hessian_move((REAL)1.0,ACTUAL_MOVE,X);
  pair_coords(a->x0,a->nv);
  restore_coords(&saved,SAVE_IN_ATTR);
  set_body_fixvol(b_id,target + a->delta);
  pyse_set_multipliers(a->m1,a->nm);
  hessian_move((REAL)1.0,ACTUAL_MOVE,X2);
  pair_coords(a->x1,a->nv);
  restore_coords(&saved,SAVE_IN_ATTR);
  set_body_fixvol(b_id,target);
  hessian_iterate_flag = 0;
  pyse_set_multipliers(p0,a->nm);
  hessian_exit(NULL);
  free_system(&S);
  fl_set_follow(was);
  temp_free((char *)rhs);
  temp_free((char *)rhs2);
  temp_free((char *)X);
  temp_free((char *)X2);
  temp_free((char *)p0);
  recalc();
}

int pyse_newton_pair(int body, double delta, double *x0, double *m0, double *x1, double *m1,
                     long nv, long nm)
{ struct pair_args a;
  if ( nv != web.skel[VERTEX].count || nm != pyse_multiplier_count() )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "newton_pair: output arrays don't match the vertices or multipliers.");
  a.body = body; a.delta = delta; a.x0 = x0; a.m0 = m0; a.x1 = x1; a.m1 = m1;
  a.nv = nv; a.nm = nm;
  return surface_call(pair_body,&a);
}

/* The Lagrange multipliers: each body's pressure (in body order), then each
   quantity's (in quantity order). A Newton step (hessian) uses them in the
   Hessian and adds its increments to them, so undoing a step undoes them too. */
long pyse_multiplier_count(void)
{ return pyse_count(PYSE_BODY) + gen_quant_count; }

void pyse_get_multipliers(double *out, long n)
{ body_id b_id;
  long row = 0;
  int i;
  FOR_ALL_BODIES(b_id)
    if ( row < n ) out[row++] = (double)get_body_pressure(b_id);
  for ( i = 0 ; i < gen_quant_count && row < n ; i++ )
    out[row++] = (double)GEN_QUANT(i)->pressure;
}

void pyse_set_multipliers(const double *in, long n)
{ body_id b_id;
  long row = 0;
  int i;
  surface_version++;       /* (the pressures: an edit's recalc() isn't skipped) */
  FOR_ALL_BODIES(b_id)
    if ( row < n ) set_body_pressure(b_id,(REAL)in[row++]);
  for ( i = 0 ; i < gen_quant_count && row < n ; i++ )
    GEN_QUANT(i)->pressure = (REAL)in[row++];
}

int pyse_set_follow(int on) { return fl_set_follow(on); }

double pyse_total_energy(void) { return (double)web.total_energy; }
double pyse_total_area(void) { return (double)web.total_area; }
const char *pyse_datafilename(void) { return datafilename; }

void pyse_set_datafilename(const char *name)
{ strncpy(datafilename,name,PATHSIZE-1);
  datafilename[PATHSIZE-1] = 0;
}

/**************************************************************************
 * Snapshots in memory: a copy of the element blocks and of what goes with
 * them (the free and in-use chains, attribute layouts, body count, scale,
 * quantities' targets and multipliers, the datafile's parameters), restored
 * into the same load. pySE's own save and restore inside its runs: a dump
 * and reload parses the whole surface and moves every element. Elements
 * refer to each other by id, never by pointer, so their bytes copy as they
 * are. Not restored: whatever was defined since (constraints, attributes,
 * quantities, procedures stay; attributes defined since come back zero).
 */

struct snap_type
{ int size;                        /* web.sizes[] */
  struct skeleton skel;
  int extra_count;
  struct extra *extras;            /* their layout then */
  int nblocks;
  struct blocklist_struct *blocks; /* blockptr: a copy of the block */
};

struct snapshot
{ long load;
  int quantities;                  /* everything_quantities_flag */
  int sdim, representation, modeltype, lagrange_order;
  struct snap_type t[NUMELEMENTS];
  int bodycount;
  body_id outside_body;
  REAL scale;
  int quant_count;
  struct gen_quant *quants;
  int inst_count;                  /* method instances: names, flags, quantities */
  struct inst_rec *insts;
  int global_count;
  REAL *params;                    /* NAN where not a datafile parameter */
};

static struct snapshot **snapshots = NULL;
static long snapshot_slots = 0;
static pthread_mutex_t snapshot_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ---- quantities' and method instances' slots while snapshots are held -----
   Evolver frees a dissolved body's volume quantity and its instances onto
   free lists and gives the slots to the next ones made; elements and bodies
   name them by slot. A snapshot put back after a body went and another came
   (a neck's cut undone, a stray piece folded back: the new body's volume in
   the old one's slots) had its quantities' flags copied back slot by slot:
   the new body's volume marked deleted under its live method, which a dump
   declared and never wrote (a 16-bead pack's checkpoints didn't load: ERROR
   3226). So while a snapshot is held, the freed slots are parked, off the
   free lists, and a restore brings those freed since back in place (Evolver's
   freeing keeps all but a slot's name and flags), and frees a body's volume
   made since, as a dissolve does. */
struct inst_rec { char name[MNAMESIZE]; int flags; int quants[MMAXQUANTS]; };
static int slot_holds = 0;          /* live snapshots (snapshot_mutex) */
static int *parked[2], parked_n[2], parked_max[2];  /* quantities, instances */

static int holding(void)
{ int held;
  pthread_mutex_lock(&snapshot_mutex);
  held = slot_holds;
  pthread_mutex_unlock(&snapshot_mutex);
  return held;
}

/* (src/quantity.c, as a slot is freed: whether it is kept off the free list) */
int pyse_park(int kind, int slot)
{ if ( !holding() ) return 0;
  if ( parked_n[kind] == parked_max[kind] )
  { int more = parked_max[kind] ? 2*parked_max[kind] : 64;
    int *grown = (int *)realloc(parked[kind],more*sizeof(int));
    if ( !grown ) return 0;
    parked[kind] = grown;
    parked_max[kind] = more;
  }
  parked[kind][parked_n[kind]++] = slot;
  return 1;
}

/* (before an allocation) the parked slots onto the free lists once no
   snapshot is held */
void pyse_unpark_all(void)
{ int i;
  if ( !(parked_n[0] + parked_n[1]) || holding() ) return;
  for ( i = 0 ; i < parked_n[0] ; i++ )
  { GEN_QUANT(parked[0][i])->next = (int)gen_quant_free_head;
    gen_quant_free_head = parked[0][i];
  }
  for ( i = 0 ; i < parked_n[1] ; i++ )
  { METH_INSTANCE(parked[1][i])->next = (int)meth_inst_free_head;
    meth_inst_free_head = parked[1][i];
  }
  parked_n[0] = parked_n[1] = 0;
}

/* (a new surface: its lists start empty) */
void pyse_park_reset(void) { parked_n[0] = parked_n[1] = 0; }

static int unpark(int kind, int slot)
{ int i;
  for ( i = 0 ; i < parked_n[kind] ; i++ )
    if ( parked[kind][i] == slot )
    { parked[kind][i] = parked[kind][--parked_n[kind]];
      return 1;
    }
  return 0;
}

static void snapshot_release(struct snapshot *s)
{ int type, b;
  if ( !s ) return;
  pthread_mutex_lock(&snapshot_mutex);
  slot_holds--;
  pthread_mutex_unlock(&snapshot_mutex);
  for ( type = 0 ; type < NUMELEMENTS ; type++ )
  { struct snap_type *t = s->t + type;
    for ( b = 0 ; b < t->nblocks ; b++ )
      free(t->blocks[b].blockptr);
    free(t->blocks);
    free(t->extras);
  }
  free(s->quants);
  free(s->insts);
  free(s->params);
  free(s);
}

static void *snap_alloc(size_t n, size_t size)
{ void *p = calloc(n ? n : 1,size ? size : 1);
  if ( !p ) kb_error(PYSE_ERR_BAD_ARGUMENT,"Out of memory for a snapshot.\n",RECOVERABLE);
  return p;
}

struct snapshot_take_args { struct snapshot *s; };

static void snapshot_take_body(void *arg)
{ struct snapshot_take_args *a = (struct snapshot_take_args *)arg;
  struct snapshot *s;
  int type, b, i;

  free_discards(DISCARDS_ALL);  /* (freed elements onto the free list) */
  s = (struct snapshot *)snap_alloc(1,sizeof(*s));
  pthread_mutex_lock(&snapshot_mutex);
  slot_holds++;
  pthread_mutex_unlock(&snapshot_mutex);
  a->s = s;
  s->load = load_serial;
  s->quantities = everything_quantities_flag;
  s->sdim = SDIM;
  s->representation = web.representation;
  s->modeltype = web.modeltype;
  s->lagrange_order = web.lagrange_order;
  for ( type = 0 ; type < NUMELEMENTS ; type++ )
  { struct snap_type *t = s->t + type;
    t->size = web.sizes[type];
    t->skel = web.skel[type];
    t->extra_count = web.skel[type].extra_count;
    t->extras = (struct extra *)snap_alloc(t->extra_count,sizeof(struct extra));
    if ( t->extra_count )
      memcpy(t->extras,EXTRAS(type),t->extra_count*sizeof(struct extra));
    t->blocks = (struct blocklist_struct *)snap_alloc(blockcount[type],
                                                      sizeof(struct blocklist_struct));
    for ( b = 0 ; b < blockcount[type] ; b++ )
    { struct blocklist_struct *src = blocklist[type] + b;
      size_t bytes = (size_t)src->count*t->size;
      t->blocks[b].start_ord = src->start_ord;
      t->blocks[b].count = src->count;
      /* (not zeroed first, copied over whole: the zeroing was a quarter
         of a snapshot's time) */
      t->blocks[b].blockptr = (struct element *)malloc(bytes ? bytes : 1);
      if ( !t->blocks[b].blockptr )
        kb_error(PYSE_ERR_BAD_ARGUMENT,"Out of memory for a snapshot.\n",RECOVERABLE);
      t->nblocks = b + 1;
      memcpy(t->blocks[b].blockptr,src->blockptr,bytes);
    }
  }
  s->bodycount = web.bodycount;
  s->outside_body = web.outside_body;
  s->scale = web.scale;
  s->quant_count = gen_quant_count;
  s->quants = (struct gen_quant *)snap_alloc(gen_quant_count,sizeof(struct gen_quant));
  if ( gen_quant_count )
    memcpy(s->quants,GEN_QUANT(0),gen_quant_count*sizeof(struct gen_quant));
  s->inst_count = meth_inst_count;
  s->insts = (struct inst_rec *)snap_alloc(meth_inst_count,sizeof(struct inst_rec));
  for ( i = LOW_INST ; i < meth_inst_count ; i++ )
  { struct method_instance *mi = METH_INSTANCE(i);
    memcpy(s->insts[i].name,mi->name,sizeof(mi->name));
    s->insts[i].flags = mi->flags;
    memcpy(s->insts[i].quants,mi->quants,sizeof(mi->quants));
  }
  s->global_count = web.global_count;
  s->params = (REAL *)snap_alloc(web.global_count,sizeof(REAL));
  for ( i = 0 ; i < web.global_count ; i++ )
    s->params[i] = is_datafile_parameter(i) ? globals(i)->value.real : NAN;
}

/* inside a guarded call (pyse_internal.h): edits that undo themselves */
struct snapshot *pyse_snapshot_hold(void)
{ struct snapshot_take_args a;
  a.s = NULL;
  snapshot_take_body(&a);
  return a.s;
}

void pyse_snapshot_drop(struct snapshot *s) { snapshot_release(s); }

int pyse_everything_quantities(void) { return everything_quantities_flag; }

int pyse_snapshot_take(long *handle)
{ struct snapshot_take_args a;
  int status;
  long h;
  a.s = NULL;
  status = surface_call(snapshot_take_body,&a);
  if ( status != PYSE_OK )
  { snapshot_release(a.s);
    return status;
  }
  pthread_mutex_lock(&snapshot_mutex);
  for ( h = 0 ; h < snapshot_slots && snapshots[h] ; h++ ) ;
  if ( h == snapshot_slots )
  { long more = snapshot_slots ? 2*snapshot_slots : 16;
    struct snapshot **grown = (struct snapshot **)realloc(snapshots,more*sizeof(*grown));
    if ( !grown )
    { pthread_mutex_unlock(&snapshot_mutex);
      snapshot_release(a.s);
      return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,"Out of memory for a snapshot.");
    }
    memset(grown + snapshot_slots,0,(more - snapshot_slots)*sizeof(*grown));
    snapshots = grown;
    snapshot_slots = more;
  }
  snapshots[h] = a.s;
  pthread_mutex_unlock(&snapshot_mutex);
  *handle = h;
  return PYSE_OK;
}

void pyse_snapshot_free(long handle)
{ struct snapshot *s = NULL;
  pthread_mutex_lock(&snapshot_mutex);
  if ( handle >= 0 && handle < snapshot_slots )
  { s = snapshots[handle];
    snapshots[handle] = NULL;
  }
  pthread_mutex_unlock(&snapshot_mutex);
  snapshot_release(s);
}

/* One element from a snapshot into today's layout (the shapes as they were,
   offsets maybe not: alignment padding round an attribute that grew and
   shrank back): the fixed fields, then attribute by attribute; attributes
   defined since stay zero, as when defined. */
static void snapshot_copy_element(char *to, const char *from, int type, struct snap_type *t)
{ int k;
  int now_count = web.skel[type].extra_count;
  size_t head = t->extra_count ? (size_t)t->extras[0].offset : (size_t)t->size;
  if ( now_count && (size_t)EXTRAS(type)[0].offset < head )
    head = (size_t)EXTRAS(type)[0].offset;
  memcpy(to,from,head);
  for ( k = 0 ; k < t->extra_count ; k++ )
  { size_t was_end = k + 1 < t->extra_count ? (size_t)t->extras[k+1].offset : (size_t)t->size;
    size_t now_end = k + 1 < now_count ? (size_t)EXTRAS(type)[k+1].offset
                                       : (size_t)web.sizes[type];
    size_t was_span = was_end - (size_t)t->extras[k].offset;
    size_t now_span = now_end - (size_t)EXTRAS(type)[k].offset;
    memcpy(to + EXTRAS(type)[k].offset,from + t->extras[k].offset,
           was_span < now_span ? was_span : now_span);
  }
}

/* The quantities and method instances as at the snapshot (see pyse_park()):
   a body's volume made since freed with its instances, as a dissolve does;
   then those freed since back in their slots (in that order: a body made
   again since has its old volume's names) */
static int alive_at(const struct snapshot *s, int kind, int i)
{ if ( kind == 0 ) return i < s->quant_count && !(s->quants[i].flags & Q_DELETED);
  return i >= LOW_INST && i < s->inst_count && !(s->insts[i].flags & Q_DELETED);
}

static void restore_slots(struct snapshot *s)
{ int i, k, j;
  for ( i = LOW_INST ; i < meth_inst_count ; i++ )
  { struct method_instance *mi = METH_INSTANCE(i);
    if ( !alive_at(s,1,i) && !(mi->flags & Q_DELETED) && (mi->flags & BODY_INSTANCE) )
      free_method_instance(mi);
  }
  for ( i = 0 ; i < gen_quant_count ; i++ )
  { struct gen_quant *q = GEN_QUANT(i);
    if ( !alive_at(s,0,i) && !(q->flags & Q_DELETED) && valid_id(q->b_id) )
      free_quantity(q);
  }
  for ( i = LOW_INST ; i < meth_inst_count ; i++ )
  { struct method_instance *mi = METH_INSTANCE(i);
    if ( alive_at(s,1,i) && (mi->flags & Q_DELETED) && unpark(1,i) )
    { memcpy(mi->name,s->insts[i].name,sizeof(mi->name));
      mi->flags = s->insts[i].flags;
      memcpy(mi->quants,s->insts[i].quants,sizeof(mi->quants));
      lookup_global_hash(mi->name,i,METHODNAME,HASH_ADD);
    }
  }
  for ( i = 0 ; i < gen_quant_count ; i++ )
  { struct gen_quant *q = GEN_QUANT(i);
    if ( alive_at(s,0,i) && (q->flags & Q_DELETED) && unpark(0,i) )
    { memcpy(q->name,s->quants[i].name,sizeof(q->name));
      q->flags = s->quants[i].flags;
      lookup_global_hash(q->name,i,QUANTITYNAME,HASH_ADD);
    }
  }
  /* (the quantities' lists without the instances freed) */
  for ( i = 0 ; i < gen_quant_count ; i++ )
  { struct gen_quant *q = GEN_QUANT(i);
    if ( q->flags & (Q_DELETED|Q_COMPOUND) ) continue;
    for ( k = j = 0 ; k < q->method_count ; k++ )
      if ( !(METH_INSTANCE(q->meth_inst[k])->flags & Q_DELETED) )
        q->meth_inst[j++] = q->meth_inst[k];
    q->method_count = j;
  }
}

static void snapshot_restore_body(void *arg)
{ struct snapshot *s = (struct snapshot *)arg;
  int type, b, i, k;

  /* the attributes' layout as it was (a constraint list grown since moved
     the attributes after it), by Evolver's own reshaping; all types before
     any is replaced, so a refusal leaves the surface as it was */
  for ( type = 0 ; type < NUMELEMENTS ; type++ )
  { struct snap_type *t = s->t + type;
    struct extra *now;
    for ( k = 0 ; k < t->extra_count ; k++ )
    { struct extra *was = t->extras + k;
      int sizes[MAXARRAYDIMS];
      now = EXTRAS(type) + k;
      if ( now->array_spec.datacount == was->array_spec.datacount
           && now->array_spec.dim == was->array_spec.dim
           && !memcmp(now->array_spec.sizes,was->array_spec.sizes,sizeof(sizes)) )
        continue;
      memcpy(sizes,was->array_spec.sizes,sizeof(sizes));
      if ( was->array_spec.dim == 0 ) sizes[0] = was->array_spec.datacount;
      expand_attribute(type,k,sizes);
    }
  }
  for ( type = 0 ; type < NUMELEMENTS ; type++ )
  { struct snap_type *t = s->t + type;
    INDIRECT_TYPE *ibase;
    char *spot;
    /* the same layout (the usual case: nothing defined or grown since):
       the blocks copy whole */
    int same = web.sizes[type] == t->size && web.skel[type].extra_count == t->extra_count;
    for ( k = 0 ; same && k < t->extra_count ; k++ )
      same = EXTRAS(type)[k].offset == t->extras[k].offset;

    /* the blocks as they were, at today's size (attributes defined since:
       zero) */
    for ( b = 0 ; b < blockcount[type] ; b++ )
      myfree((char*)blocklist[type][b].blockptr);
    blockcount[type] = 0;
    if ( blockmax[type] < t->nblocks )
    { blocklist[type] = (struct blocklist_struct *)kb_realloc((char*)blocklist[type],
                          (t->nblocks + BATCHSIZE)*sizeof(struct blocklist_struct));
      blockmax[type] = t->nblocks + BATCHSIZE;
    }
    if ( web.skel[type].ibase ) myfree((char*)web.skel[type].ibase);
    ibase = (INDIRECT_TYPE *)mycalloc(t->skel.ialloc ? t->skel.ialloc : 1,
                                      sizeof(INDIRECT_TYPE));
    for ( b = 0 ; b < t->nblocks ; b++ )
    { struct blocklist_struct *was = t->blocks + b;
      char *block = (char*)mycalloc(was->count,web.sizes[type]);
      if ( same )
        memcpy(block,was->blockptr,(size_t)was->count*t->size);
      else
        for ( i = 0 ; i < was->count ; i++ )
          snapshot_copy_element(block + (size_t)i*web.sizes[type],
                                (char*)was->blockptr + (size_t)i*t->size,type,t);
      for ( i = 0, spot = block ; i < was->count ; i++, spot += web.sizes[type] )
      { struct element *e = (struct element *)spot;
        ORDTYPE ord = ordinal(e->self_id);
        if ( ord < 0 || ord >= t->skel.ialloc )
          kb_error(PYSE_ERR_BAD_ARGUMENT,"Internal error: a snapshot element's id is out of "
                   "range.\n",RECOVERABLE);
        ibase[ord] = e;
      }
      blocklist[type][b].blockptr = (struct element *)block;
      blocklist[type][b].start_ord = was->start_ord;
      blocklist[type][b].count = was->count;
      blockcount[type] = b + 1;
    }
    { struct skeleton keep = web.skel[type];
      web.skel[type] = t->skel;
      web.skel[type].ibase = ibase;
      web.skel[type].dy_extras = keep.dy_extras;
      web.skel[type].maxextra = keep.maxextra;
      web.skel[type].extra_count = keep.extra_count;
    }
    switch ( type )
    { case VERTEX: vibase = ibase; break;
      case EDGE: eibase = ibase; break;
      case FACET: fibase = ibase; break;
      case BODY: bibase = ibase; break;
      case FACETEDGE: feibase = ibase; break;
    }
    parallel_update_flag[type] = 1;
  }
  web.bodycount = s->bodycount;
  web.outside_body = s->outside_body;
  web.scale = s->scale;
  restore_slots(s);
  for ( i = 0 ; i < gen_quant_count ; i++ )
  { struct gen_quant *q = GEN_QUANT(i);
    if ( i < s->quant_count && !(s->quants[i].flags & Q_DELETED) && !(q->flags & Q_DELETED) )
    { struct gen_quant *was = s->quants + i;
      q->flags = was->flags;
      q->target = was->target;
      q->value = was->value;
      q->modulus = was->modulus;
      q->pressure = was->pressure;
      q->oldpressure = was->oldpressure;
      q->volconst = was->volconst;
      q->b_id = was->b_id;
      q->vol_number = was->vol_number;
      q->fixnum = was->fixnum;
    }
    q->timestamp = 0;
  }
  for ( i = 0 ; i < s->global_count && i < web.global_count ; i++ )
    if ( !isnan(s->params[i]) && is_datafile_parameter(i) )
      globals(i)->value.real = s->params[i];
  web_timestamp = top_timestamp = ++global_timestamp;
  fl_body_stamp++;
  FL_ATTR_CHANGED();
  recalc();
}

void pyse_snapshot_put_back(struct snapshot *s) { snapshot_restore_body(s); }

int pyse_snapshot_restore(long handle)
{ struct snapshot *s = NULL;
  int status;
  pthread_mutex_lock(&snapshot_mutex);
  if ( handle >= 0 && handle < snapshot_slots ) s = snapshots[handle];
  pthread_mutex_unlock(&snapshot_mutex);
  if ( !s )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,"No such snapshot.");
  if ( s->load != load_serial )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "The snapshot is of a surface loaded before the current one.");
  if ( s->quantities != everything_quantities_flag )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "The surface was converted to quantities since the snapshot (as the "
                      "first Hessian does): it can't be restored in memory.");
  if ( s->sdim != SDIM || s->representation != web.representation
       || s->modeltype != web.modeltype || s->lagrange_order != web.lagrange_order )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "The model changed since the snapshot (dimension, representation or "
                      "element order): it can't be restored in memory.");
  status = surface_call(snapshot_restore_body,s);
  surface_version++;
  return status;
}

extern int pyse_fast_steps;   /* src/iterate.c */

int pyse_set_fast_steps(int on)
{ int was = pyse_fast_steps;
  pyse_fast_steps = on != 0;
  return was;
}
