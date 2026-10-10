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

struct residual_args { double *out; long n; };

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
  double *A = NULL, *bvec = NULL, *lam = NULL, *proj = NULL;
  int *ndir = NULL;
  int mode = CALC_FORCE|CALC_VOLGRADS;
  find_fixed();
  calc_volgrads(DO_OPTS);
  calc_force();
  pressure_forces();
  partner_shift_grads(mode);
  convert_forms_to_vectors(mode);
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
    nd = shape_directions(v_id,dirs);
    if ( nd < 0 )                         /* no normal: the velocity's axes */
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
  vgrad_end();
}

/* The shape part of each vertex's velocity (one per vertex, in vertex order). */
int pyse_residual(double *out, long n)
{ struct residual_args a;
  if ( in_protected ) return PYSE_BUSY;
  if ( !initialized || !surface_valid ) return invalid_surface();
  if ( n != web.skel[VERTEX].count )
    return glue_error(PYSE_ERROR,PYSE_ERR_BAD_ARGUMENT,
                      "Output array does not match the vertex count.");
  a.out = out; a.n = n;
  return surface_call(residual_body,&a);
}

/* Constraints in use and each vertex's signed distance to the ones it isn't on. */
struct conlist_args { int *nums; int *attrs; int max; int count; };

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
    }
    a->count++;
  }
}

int pyse_constraint_list(int *nums, int *attrs, int max)
{ struct conlist_args a;
  int status;
  if ( in_protected ) return -1;
  if ( !initialized || !surface_valid ) return -1;
  a.nums = nums; a.attrs = attrs; a.max = max; a.count = 0;
  status = surface_call(conlist_body,&a);
  return status == PYSE_OK ? a.count : -1;
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
  /* bring every quantity's value up to date */
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
double pyse_total_energy(void) { return (double)web.total_energy; }
double pyse_total_area(void) { return (double)web.total_area; }
const char *pyse_datafilename(void) { return datafilename; }

void pyse_set_datafilename(const char *name)
{ strncpy(datafilename,name,PATHSIZE-1);
  datafilename[PATHSIZE-1] = 0;
}
