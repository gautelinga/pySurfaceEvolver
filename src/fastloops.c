/*
 * fastloops.c -- cached facet topology and parallel facet loops.
 *
 * Not part of the original Surface Evolver.
 *
 * Evolver walks facet -> facet-edge -> edge -> vertex every time it needs a
 * facet's corners, dozens of times per facet per iteration. The facet
 * corner cache keeps the three corner vertices of every facet, plus the
 * facets in FOR_ALL_FACETS order, rebuilt whenever the topology changes
 * (top_timestamp) or facets are added or removed.
 *
 * The parallel loops use the cache: per-facet values are pure arithmetic
 * on coordinates (no globals, no expression evaluation). Each thread sums
 * its static chunk of facets (compensated sums for energies and volumes,
 * its own force array), and the partial sums are merged in thread order.
 * Results differ from the original serial loops at round-off level and
 * are reproducible for a given thread count.
 *
 * Without OpenMP everything compiles and runs serially.
 */

#include "include.h"
#include "fastloops.h"

extern REAL wee_area;   /* filml.c */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static vertex_id *fv_cache = NULL;      /* 3 per facet ordinal */
static long fv_cache_size = 0;
static facet_id *fv_list = NULL;        /* facets in traversal order */
static long fv_list_size = 0;
static long fv_list_count = 0;
static long fv_cache_stamp = -1;
static long fv_cache_count = -1;
static long fv_cache_maxord = -1;
static vertex_id *fv_verts = NULL;     /* corner vertices by ordinal, or NULLID */
static long fv_verts_size = 0;
static long fv_verts_count = 0;
static int fv_check = -1;

/* PYSE_NO_FAST_LOOPS turns the parallel loops off (the original loops run),
   for comparing results. */
int fl_disabled(void)
{ static int disabled = -1;
  if ( disabled < 0 ) disabled = getenv("PYSE_NO_FAST_LOOPS") != NULL;
  return disabled;
}

int fl_check(void)
{ if ( fv_check < 0 )
    fv_check = getenv("PYSE_CHECK_FACET_CACHE") != NULL;
  return fv_check;
}

static int ensure_size(void **p, long *size, long want, size_t item)
{ if ( want <= *size ) return 1;
  free(*p);
  *p = malloc((size_t)want*item);
  if ( !*p ) { *size = 0; return 0; }
  *size = want;
  return 1;
}

static int cache_valid(void)
{ return fv_cache && fv_cache_stamp == top_timestamp
      && fv_cache_count == web.skel[FACET].count
      && fv_cache_maxord == (long)web.skel[FACET].max_ord
      && fv_verts_count == (long)web.skel[VERTEX].max_ord + 1;
}

static int build_cache(void)
{ facet_id f_id;
  long k = 0;
  if ( web.representation != SOAPFILM ) return 0;
  if ( !ensure_size((void**)&fv_cache,&fv_cache_size,
                    3*((long)web.skel[FACET].max_ord + 1),sizeof(vertex_id))
       || !ensure_size((void**)&fv_list,&fv_list_size,
                       web.skel[FACET].count + 1,sizeof(facet_id))
       || !ensure_size((void**)&fv_verts,&fv_verts_size,
                       (long)web.skel[VERTEX].max_ord + 1,sizeof(vertex_id)) )
    return 0;
  fv_verts_count = (long)web.skel[VERTEX].max_ord + 1;
  for ( k = 0 ; k < fv_verts_count ; k++ ) fv_verts[k] = NULLID;
  k = 0;
  FOR_ALL_FACETS(f_id)
  { facetedge_id fe = get_facet_fe(f_id);
    vertex_id *c = fv_cache + 3*ordinal(f_id);
    c[0] = get_fe_tailv(fe); fe = get_next_edge(fe);
    c[1] = get_fe_tailv(fe); fe = get_next_edge(fe);
    c[2] = get_fe_tailv(fe);
    fv_verts[ordinal(c[0])] = c[0];
    fv_verts[ordinal(c[1])] = c[1];
    fv_verts[ordinal(c[2])] = c[2];
    if ( k < fv_list_size ) fv_list[k] = f_id;
    k++;
  }
  fv_list_count = k;
  fv_cache_stamp = top_timestamp;
  fv_cache_count = web.skel[FACET].count;
  fv_cache_maxord = (long)web.skel[FACET].max_ord;
  return k == web.skel[FACET].count;
}

vertex_id *fl_facet_corners(void)
{ if ( cache_valid() ) return fv_cache;
  return build_cache() ? fv_cache : NULL;
}

facet_id *fl_facet_list(long *n)
{ if ( !fl_facet_corners() ) return NULL;
  *n = fv_list_count;
  return fv_list;
}

/* Verify the cached traversal order against FOR_ALL_FACETS. */
static void check_list(facet_id *list, long n)
{ facet_id f_id;
  long k = 0;
  FOR_ALL_FACETS(f_id)
  { if ( k >= n || !equal_id(list[k],f_id) )
    { fprintf(stderr,"facet list cache mismatch at position %ld\n",k);
      abort();
    }
    k++;
  }
  if ( k != n )
  { fprintf(stderr,"facet list cache has %ld facets, surface %ld\n",n,k);
    abort();
  }
}

/**************************************************************************
 * Facet bodies
 *
 * The front and back body of every facet in list order, NULLID for
 * NONCONTENT facets: the loops read these instead of the (large) facet
 * records. Rebuilt when the topology changes (set_facet_body() counts as
 * one), or NONCONTENT attributes or bodies are changed or deleted
 * (fl_body_stamp, bumped where Evolver does that).
 */

long fl_body_stamp = 0;

static int loop_threads(long n);

static body_id *fb_cache = NULL;    /* 2 per facet in list order */
static long fb_cache_size = 0;
static long fb_top_stamp = -1;
static long fb_body_stamp = -1;
static long fb_count = -1;

static void facet_bodies_of(facet_id f_id, body_id *b)
{ if ( get_fattr(f_id) & NONCONTENT ) b[0] = b[1] = NULLID;
  else
  { b[0] = get_facet_body(f_id);
    b[1] = get_facet_body(facet_inverse(f_id));
  }
}

/* The body table for the facet list (n facets), or NULL. */
static body_id *facet_bodies(facet_id *list, long n)
{ long k;
  if ( fb_cache && fb_top_stamp == fv_cache_stamp
       && fb_body_stamp == fl_body_stamp && fb_count == n )
  { if ( fl_check() )
      for ( k = 0 ; k < n ; k++ )
      { body_id b[2];
        facet_bodies_of(list[k],b);
        if ( !equal_id(b[0],fb_cache[2*k]) || !equal_id(b[1],fb_cache[2*k+1]) )
        { fprintf(stderr,"facet body cache mismatch at position %ld\n",k);
          abort();
        }
      }
    return fb_cache;
  }
  if ( !ensure_size((void**)&fb_cache,&fb_cache_size,2*n,sizeof(body_id)) )
    return NULL;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
    facet_bodies_of(list[k],fb_cache + 2*k);
  fb_top_stamp = fv_cache_stamp;
  fb_body_stamp = fl_body_stamp;
  fb_count = n;
  return fb_cache;
}

/**************************************************************************
 * Threads
 */

static int fl_thread_count = 0;   /* 0: the default */

/* Physical cores available to this process (hyperthreads add little to
   memory-bound loops and dense factoring); 0 if unknown. */
static int physical_cores(void)
{ int cores = 0;
#if defined(__linux__)
  { long n = sysconf(_SC_NPROCESSORS_ONLN), i, k;
    long *seen = n > 0 ? (long*)malloc(n*sizeof(long)) : NULL;
    for ( i = 0 ; seen && i < n ; i++ )
    { char path[128];
      int core = -1, pkg = 0;
      FILE *f;
      sprintf(path,"/sys/devices/system/cpu/cpu%ld/topology/core_id",i);
      if ( (f = fopen(path,"r")) ) { if ( fscanf(f,"%d",&core) != 1 ) core = -1; fclose(f); }
      sprintf(path,"/sys/devices/system/cpu/cpu%ld/topology/physical_package_id",i);
      if ( (f = fopen(path,"r")) ) { if ( fscanf(f,"%d",&pkg) != 1 ) pkg = 0; fclose(f); }
      if ( core < 0 ) continue;
      for ( k = 0 ; k < cores ; k++ )
        if ( seen[k] == ((long)pkg << 20 | core) ) break;
      if ( k == cores ) seen[cores++] = (long)pkg << 20 | core;
    }
    free(seen);
  }
#elif defined(__APPLE__)
  { size_t size = sizeof(cores);
    if ( sysctlbyname("hw.physicalcpu",&cores,&size,NULL,0) != 0 ) cores = 0;
  }
#endif
  return cores;
}

/* Default threads: OMP_NUM_THREADS if set, else the physical cores (at
   most the processors this process may use). */
static int default_threads(void)
{
#ifdef _OPENMP
  static int dflt = 0;
  if ( dflt == 0 )
  { int procs = omp_get_num_procs(), cores = physical_cores();
    if ( getenv("OMP_NUM_THREADS") ) dflt = omp_get_max_threads();
    else dflt = (cores > 0 && cores < procs) ? cores : procs;
    if ( dflt < 1 ) dflt = 1;
  }
  return dflt;
#else
  return 1;
#endif
}

int fl_threads(void)
{ return fl_thread_count > 0 ? fl_thread_count : default_threads();
}

int fl_thread_setting(void) { return fl_thread_count; }

/* Also sets OpenMP's default, which MUMPS and an OpenMP BLAS use. */
void fl_set_threads(int n)
{ fl_thread_count = n > 0 ? n : 0;
#ifdef _OPENMP
  omp_set_num_threads(fl_threads());
#endif
}

/* Below this many facets the loops stay serial (thread start-up costs
   more than it saves). */
#define FL_PARALLEL_MIN 4096

/* threads to use for n facets */
static int loop_threads(long n)
{ return n >= FL_PARALLEL_MIN ? fl_threads() : 1;
}

#ifdef _OPENMP
#define THREAD_NUM() omp_get_thread_num()
#define THREAD_COUNT() omp_get_num_threads()
#else
#define THREAD_NUM() 0
#define THREAD_COUNT() 1
#endif

/* Per-thread blocks of `count` items in one array: the stride, rounded up to
   whole cache lines plus one, so no two threads' items share a line (with a
   handful of bodies, unpadded blocks put all threads' sums on one line and
   every addition bounced it between cores) */
#define FL_CACHE_LINE 64
static long padded(long count, size_t item)
{ long per_line = FL_CACHE_LINE/(long)item;
  if ( per_line < 1 ) per_line = 1;
  return (count + per_line - 1)/per_line*per_line + per_line;
}

/* Compensated (Neumaier) sum: each thread's partial sum is accurate to
   about one rounding however many facets it adds. */
typedef struct { double s, c; } csum;

static void csum_add(csum *a, double x)
{ double t = a->s + x;
  if ( fabs(a->s) >= fabs(x) ) a->c += (a->s - t) + x;
  else a->c += (x - t) + a->s;
  a->s = t;
}

/**************************************************************************
 * Facet volumes
 */

static csum *vol_sums = NULL;      /* per thread, per body ordinal */
static long vol_sums_size = 0;
static double *vol_abs = NULL;     /* per thread, per body ordinal */
static long vol_abs_size = 0;

static int fl_facet_volumes_impl(void)
{ long n, nb, ns, na;
  facet_id *list;
  vertex_id *corners;
  body_id b_id;
  body_id *bodies;
  int threads, t;

  if ( fl_disabled() ) return 0;
  if ( web.representation != SOAPFILM || web.torus_flag || web.symmetry_flag
       || web.symmetric_content || web.modeltype != LINEAR || SDIM != 3
       || calc_facet_volume != facet_volume_l || threadflag )
    return 0;
  corners = fl_facet_corners();
  list = fl_facet_list(&n);
  if ( !corners || !list ) return 0;
  if ( fl_check() ) check_list(list,n);
  bodies = facet_bodies(list,n);
  if ( !bodies ) return 0;
  threads = loop_threads(n);
  nb = (long)web.skel[BODY].max_ord + 1;
  ns = padded(nb,sizeof(csum));
  na = padded(nb,sizeof(double));
  if ( !ensure_size((void**)&vol_sums,&vol_sums_size,threads*ns,sizeof(csum))
       || !ensure_size((void**)&vol_abs,&vol_abs_size,threads*na,sizeof(double)) )
    return 0;
  memset(vol_sums,0,threads*ns*sizeof(csum));
  memset(vol_abs,0,threads*na*sizeof(double));

  /* each thread sums the signed volumes of its facets per body */
#ifdef _OPENMP
  #pragma omp parallel num_threads(threads)
#endif
  { csum *sums = vol_sums + THREAD_NUM()*ns;
    double *abss = vol_abs + THREAD_NUM()*na;
    long k;
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( k = 0 ; k < n ; k++ )
    { facet_id f_id = list[k];
      vertex_id *c = corners + 3*ordinal(f_id);
      REAL *x0, *x1, *x2, vol;
      body_id b0 = bodies[2*k], b1 = bodies[2*k+1];
      if ( !valid_id(b0) && !valid_id(b1) ) continue;
      x0 = get_coord(c[0]); x1 = get_coord(c[1]); x2 = get_coord(c[2]);
      /* as facet_volume_l() */
      vol = (x0[2]+x1[2]+x2[2])/6*
         ((x1[0]-x0[0])*(x2[1]-x0[1])-(x1[1]-x0[1])*(x2[0]-x0[0]));
      if ( valid_id(b0) )
      { csum_add(sums + ordinal(b0),vol);
        abss[ordinal(b0)] += fabs(vol);
      }
      if ( valid_id(b1) )
      { csum_add(sums + ordinal(b1),-vol);
        abss[ordinal(b1)] += fabs(vol);
      }
    }
  }

  /* merge in thread order */
  FOR_ALL_BODIES(b_id)
  { struct body *b = bptr(b_id);
    long ord = ordinal(b_id);
    for ( t = 0 ; t < threads ; t++ )
    { REAL v = vol_sums[t*ns+ord].s + vol_sums[t*ns+ord].c;
      if ( v != 0.0 ) binary_tree_add(b->volume_addends,v);
      b->abstotal += vol_abs[t*na+ord];
    }
  }
  return 1;
}

/**************************************************************************
 * Facet energies
 */

static csum *energy_sums = NULL;   /* per thread: area, energy */
static long energy_sums_size = 0;

/* SDIM_dot for SDIM 3, summed in the same order as dot() */
#define DOT3(a,b) (((a)[0]*(b)[0] + (a)[1]*(b)[1]) + (a)[2]*(b)[2])

static int fl_facet_energies_impl(void)
{ long n;
  facet_id *list;
  vertex_id *corners;
  int threads, t;
  REAL grav = web.grav_const;
  int gravflag = web.gravflag;

  if ( fl_disabled() ) return 0;
  if ( web.representation != SOAPFILM || web.torus_flag || web.symmetry_flag
       || web.modeltype != LINEAR || SDIM != 3 || web.metric_flag
       || web.wulff_flag || calc_facet_energy != facet_energy_l || threadflag
       || ((square_curvature_flag | mean_curv_int_flag) & EVALUATE) )
    return 0;
  corners = fl_facet_corners();
  list = fl_facet_list(&n);
  if ( !corners || !list ) return 0;
  if ( fl_check() ) check_list(list,n);
  threads = loop_threads(n);
  if ( !ensure_size((void**)&energy_sums,&energy_sums_size,2*threads,sizeof(csum)) )
    return 0;
  memset(energy_sums,0,2*threads*sizeof(csum));

  /* each facet's area and energy, as facet_energy_l() computes them */
#ifdef _OPENMP
  #pragma omp parallel num_threads(threads)
#endif
  { csum area_sum = {0.0,0.0}, energy_sum = {0.0,0.0};
    long k;
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( k = 0 ; k < n ; k++ )
    { facet_id f_id = list[k];
      vertex_id *c = corners + 3*ordinal(f_id);
      REAL *x0 = get_coord(c[0]), *x1 = get_coord(c[1]), *x2 = get_coord(c[2]);
      REAL s0[3], s1[3], ss, st, tt, det, energy;
      ATTR attr = get_fattr(f_id);
      int j;
      for ( j = 0 ; j < 3 ; j++ )
      { s0[j] = x1[j] - x0[j];
        s1[j] = x2[j] - x1[j];
      }
      ss = DOT3(s0,s0);
      st = DOT3(s0,s1);
      tt = DOT3(s1,s1);
      det = ss*tt - st*st;
      energy = det > 0.0 ? sqrt(det)/2 : 0.0;
      set_facet_area(f_id,energy);
      csum_add(&area_sum,energy);
      if ( attr & DENSITY )
        energy = energy * get_facet_density(f_id);
      if ( gravflag && !(attr & NONCONTENT) )
      { REAL z0 = x0[2], z1 = x1[2], z2 = x2[2];
        REAL zz = (z0*z0+z1*z1+z2*z2+z0*z1+z1*z2+z0*z2)/6;
        REAL u = zz*(s0[0]*s1[1]-s0[1]*s1[0])/2/2;
        body_id b_id = get_facet_body(f_id);
        if ( valid_id(b_id) )
          energy += u*get_body_density(b_id)*grav;
        b_id = get_facet_body(facet_inverse(f_id));
        if ( valid_id(b_id) )
          energy -= u*get_body_density(b_id)*grav;
      }
      csum_add(&energy_sum,energy);
    }
    energy_sums[2*THREAD_NUM()] = area_sum;
    energy_sums[2*THREAD_NUM()+1] = energy_sum;
  }

  /* merge in thread order */
  for ( t = 0 ; t < threads ; t++ )
  { REAL a = energy_sums[2*t].s + energy_sums[2*t].c;
    REAL e = energy_sums[2*t+1].s + energy_sums[2*t+1].c;
    if ( a != 0.0 ) binary_tree_add(web.total_area_addends,a);
    if ( e != 0.0 ) binary_tree_add(web.total_energy_addends,e);
  }
  return 1;
}

/**************************************************************************
 * Facet forces
 */

/* One facet's forces on its three corners, with the same operations as
   facet_force_l(); sets the facet area. Returns 1 for a zero-area facet
   (no forces). */
static int facet_force(facet_id f_id, vertex_id *c, REAL grav, int gravflag,
                       REAL f[3][3])
{ REAL *x[3];
  REAL side[3][3], z[3], ss, st, tt, area;
  REAL density = get_facet_density(f_id);
  int i, j, m;
  x[0] = get_coord(c[0]); x[1] = get_coord(c[1]); x[2] = get_coord(c[2]);
  for ( i = 0 ; i < 3 ; i++ )
  { int ii = (i+1)%3;
    for ( j = 0 ; j < 3 ; j++ )
    { side[i][j] = x[ii][j] - x[i][j];
      f[i][j] = 0.0;
    }
    z[i] = x[i][2];
  }
  ss = DOT3(side[0],side[0]);
  st = DOT3(side[0],side[1]);
  tt = DOT3(side[1],side[1]);
  area = ss*tt-st*st;
  if ( area < 0.0 ) area = 0.0;
  area = sqrt(area)/2;
  set_facet_area(f_id,area);
  if ( area <= wee_area ) return 1;
  { REAL coeff = density/4/area;
    for ( m = 0 ; m < 3 ; m++ )
    { f[0][m] += coeff*(side[0][m]*tt - st*side[1][m]);
      f[1][m] -= coeff*(side[0][m]*tt - st*side[1][m]);
      f[1][m] += coeff*(ss*side[1][m] - st*side[0][m]);
      f[2][m] -= coeff*(ss*side[1][m] - st*side[0][m]);
    }
  }
  if ( gravflag && !(get_fattr(f_id) & NONCONTENT) )
  { REAL zz, gdensity, normz;
    body_id b_id;
    zz = (z[0]*z[0]+z[1]*z[1]+z[2]*z[2]+z[0]*z[1]+z[1]*z[2]+z[0]*z[2])/6;
    b_id = get_facet_body(f_id);
    gdensity = 0.0;
    if ( valid_id(b_id) ) gdensity += get_body_density(b_id);
    b_id = get_facet_body(facet_inverse(f_id));
    if ( valid_id(b_id) ) gdensity -= get_body_density(b_id);
    normz = side[0][0]*side[1][1] - side[0][1]*side[1][0];
    for ( i = 0 ; i < 3 ; i++ )
    { j = (i+1)%3;
      f[i][0] += grav*gdensity*side[j][1]*zz/4;
      f[i][1] -= grav*gdensity*side[j][0]*zz/4;
      f[i][2] -= grav*gdensity*normz*(z[i]+z[0]+z[1]+z[2])/24;
    }
  }
  return 0;
}

static double *force_sums = NULL;   /* per thread: 3 per vertex ordinal */
static long force_sums_size = 0;
static unsigned char *wee_buf = NULL;
static long wee_buf_size = 0;

static int fl_facet_forces_impl(void)
{ long n, nv, k;
  facet_id *list;
  vertex_id *corners;
  int threads;
  long wee_count = 0;
  REAL grav = web.grav_const;
  int gravflag = web.gravflag;

  if ( fl_disabled() ) return 0;
  if ( web.representation != SOAPFILM || web.torus_flag || web.symmetry_flag
       || web.modeltype != LINEAR || SDIM != 3 || web.metric_flag
       || web.wulff_flag || calc_facet_forces != facet_force_l || threadflag
       || itdebug || ((square_curvature_flag | mean_curv_int_flag) & EVALUATE) )
    return 0;
  corners = fl_facet_corners();
  list = fl_facet_list(&n);
  if ( !corners || !list ) return 0;
  if ( fl_check() ) check_list(list,n);
  threads = loop_threads(n);
  nv = fv_verts_count;
  if ( !ensure_size((void**)&wee_buf,&wee_buf_size,n,1)
       || !ensure_size((void**)&force_sums,&force_sums_size,threads*3*nv,sizeof(double)) )
    return 0;

  /* each thread adds its facets' forces to its own force array */
#ifdef _OPENMP
  #pragma omp parallel num_threads(threads) reduction(+:wee_count)
#endif
  { double *fsum = force_sums + THREAD_NUM()*3*nv;
    long kk;
    memset(fsum,0,3*nv*sizeof(double));
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( kk = 0 ; kk < n ; kk++ )
    { vertex_id *c = corners + 3*ordinal(list[kk]);
      REAL f[3][3];
      int i;
      wee_buf[kk] = (unsigned char)facet_force(list[kk],c,grav,gravflag,f);
      if ( wee_buf[kk] ) { wee_count++; continue; }
      for ( i = 0 ; i < 3 ; i++ )
      { double *fv = fsum + 3*ordinal(c[i]);
        fv[0] += f[i][0];
        fv[1] += f[i][1];
        fv[2] += f[i][2];
      }
    }

    /* add the thread sums to the vertices, in thread order */
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( kk = 0 ; kk < nv ; kk++ )
    { vertex_id v_id = fv_verts[kk];
      REAL *force;
      int t;
      if ( !valid_id(v_id) ) continue;
      force = get_force(v_id);
      for ( t = 0 ; t < THREAD_COUNT() ; t++ )
      { double *fv = force_sums + (t*nv + kk)*3;
        force[0] += fv[0];
        force[1] += fv[1];
        force[2] += fv[2];
      }
    }
  }

  /* zero-area warnings in the order facet_force_l() would print them */
  if ( wee_count )
    for ( k = 0 ; k < n ; k++ )
      if ( wee_buf[k] )
      { sprintf(errmsg,"WARNING! Zero area for facet %s.\n",ELNAME(list[k]));
        outstring(errmsg);
      }
  return 1;
}

/**************************************************************************
 * Body volume gradients
 *
 * film_grad_l() for linear soapfilm surfaces without torus or symmetry:
 * each facet adds its volume gradient at its corners to the volgrad
 * structure of each of its fixed-volume or pressure bodies. Here the
 * per-facet terms are computed in parallel, then each vertex (in parallel)
 * builds its chain and sums its facets' terms in facet order. Chains are in
 * the same order and the sums in the same order as film_grad_l(), so the
 * gradients are identical.
 */

extern int vgrad_attr;  /* fixvol.c */

/* For each vertex ordinal, the volume gradient terms that go to it:
   vg_start[ord]..vg_start[ord+1] index vg_terms, which holds 3k+j for
   term j of the k-th facet in the list (term j goes to corner (j+2)%3,
   as film_grad_l() visits them). Built with the facet corner cache. */
static long *vg_start = NULL;
static long vg_start_size = 0;
static long *vg_terms = NULL;
static long vg_terms_size = 0;
static long vg_stamp = -1;
static long vg_count = -1;

static int build_vertex_terms(facet_id *list, vertex_id *corners, long n, long nv)
{ long k, v;
  int j;
  if ( !ensure_size((void**)&vg_start,&vg_start_size,nv+1,sizeof(long))
       || !ensure_size((void**)&vg_terms,&vg_terms_size,3*n,sizeof(long)) )
    return 0;
  for ( v = 0 ; v <= nv ; v++ ) vg_start[v] = 0;
  for ( k = 0 ; k < n ; k++ )
  { vertex_id *c = corners + 3*ordinal(list[k]);
    for ( j = 0 ; j < 3 ; j++ ) vg_start[ordinal(c[j])+1]++;
  }
  for ( v = 0 ; v < nv ; v++ ) vg_start[v+1] += vg_start[v];
  for ( k = 0 ; k < n ; k++ )
  { vertex_id *c = corners + 3*ordinal(list[k]);
    for ( j = 0 ; j < 3 ; j++ )
      vg_terms[vg_start[ordinal(c[(j+2)%3])]++] = 3*k + j;
  }
  for ( v = nv ; v > 0 ; v-- ) vg_start[v] = vg_start[v-1];
  vg_start[0] = 0;
  vg_stamp = fv_cache_stamp;
  vg_count = n;
  return 1;
}

static double *vg_grad = NULL;     /* 3 terms x 3 per facet */
static long vg_grad_size = 0;
static int *vg_fixnum = NULL;      /* per facet: front, back body fixnum or -1 */
static long vg_fixnum_size = 0;
static body_id *vg_body = NULL;    /* per facet: front, back body */
static long vg_body_size = 0;
static long *vg_first = NULL;      /* per vertex ordinal: first volgrad */
static long vg_first_size = 0;
static volgrad **vg_pool = NULL;   /* the volgrads, by vertex */
static long vg_pool_size = 0;

/* most bodies at one vertex for the parallel version */
#define VG_MAXBODIES 64

static int fl_film_grad_impl(void)
{ long n, nv, k, total;
  facet_id *list;
  vertex_id *corners;
  int threads, overflow = 0;

  if ( fl_disabled() ) return 0;
  if ( web.representation != SOAPFILM || web.torus_flag || web.symmetry_flag
       || web.symmetric_content || web.modeltype != LINEAR || SDIM != 3
       || film_grad != film_grad_l || threadflag || one_sided_present )
    return 0;
  corners = fl_facet_corners();
  list = fl_facet_list(&n);
  if ( !corners || !list || n == 0 ) return 0;
  if ( fl_check() ) check_list(list,n);
  nv = fv_verts_count;
  if ( vg_stamp != fv_cache_stamp || vg_count != n || vg_start_size < nv+1 )
    if ( !build_vertex_terms(list,corners,n,nv) ) return 0;
  if ( !ensure_size((void**)&vg_grad,&vg_grad_size,9*n,sizeof(double))
       || !ensure_size((void**)&vg_fixnum,&vg_fixnum_size,2*n,sizeof(int))
       || !ensure_size((void**)&vg_body,&vg_body_size,2*n,sizeof(body_id))
       || !ensure_size((void**)&vg_first,&vg_first_size,nv+1,sizeof(long)) )
    return 0;
  threads = loop_threads(n);

  /* pass 1: each facet's bodies and gradient terms, as film_grad_l() */
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for ( k = 0 ; k < n ; k++ )
  { facet_id f_id = list[k];
    vertex_id *c = corners + 3*ordinal(f_id);
    body_id bi_id = get_facet_body(f_id);
    body_id bj_id = get_facet_body(facet_inverse(f_id));
    REAL side[3][3], *x[3], z, normal2;
    double *g = vg_grad + 9*k;
    int i, j;
    vg_fixnum[2*k] = vg_fixnum[2*k+1] = -1;
    vg_body[2*k] = bi_id;
    vg_body[2*k+1] = bj_id;
    if ( valid_id(bi_id) && (get_battr(bi_id) & (FIXEDVOL|PRESSURE)) )
      vg_fixnum[2*k] = get_body_fixnum(bi_id);
    if ( valid_id(bj_id) && (get_battr(bj_id) & (FIXEDVOL|PRESSURE)) )
      vg_fixnum[2*k+1] = get_body_fixnum(bj_id);
    if ( vg_fixnum[2*k] < 0 && vg_fixnum[2*k+1] < 0 ) continue;
    for ( i = 0 ; i < 3 ; i++ )
    { REAL *t = get_coord(c[i]), *h = get_coord(c[(i+1)%3]);
      for ( j = 0 ; j < 3 ; j++ ) side[i][j] = h[j] - t[j];
      x[i] = h;
    }
    for ( i = 0, z = 0.0 ; i < 3 ; i++ ) z += x[i][2];
    normal2 = side[0][0]*side[1][1] - side[0][1]*side[1][0];
    for ( j = 0 ; j < 3 ; j++ )
    { g[3*j] = -side[j][1]*z/6.0;
      g[3*j+1] = side[j][0]*z/6.0;
      g[3*j+2] = normal2/6.0;
    }
  }

  /* pass 2: how many bodies each vertex needs a volgrad for */
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:overflow)
#endif
  for ( k = 0 ; k < nv ; k++ )
  { vertex_id v_id = fv_verts[k];
    int seen[VG_MAXBODIES], count = 0, m;
    long e;
    vg_first[k] = 0;
    if ( !valid_id(v_id) || (get_vattr(v_id) & FIXED) ) continue;
    for ( e = vg_start[k] ; e < vg_start[k+1] ; e++ )
    { long f = vg_terms[e]/3;
      int s;
      for ( s = 0 ; s < 2 ; s++ )
      { int fixnum = vg_fixnum[2*f+s];
        if ( fixnum < 0 ) continue;
        for ( m = 0 ; m < count ; m++ ) if ( seen[m] == fixnum ) break;
        if ( m < count ) continue;
        if ( count == VG_MAXBODIES ) { overflow = 1; break; }
        seen[count++] = fixnum;
      }
    }
    vg_first[k] = count;
  }
  if ( overflow ) return 0;   /* nothing changed yet: the original runs */

  /* allocate, in vertex order */
  for ( k = 0, total = 0 ; k < nv ; k++ )
  { long count = vg_first[k];
    vg_first[k] = total;
    total += count;
  }
  vg_first[nv] = total;
  if ( !ensure_size((void**)&vg_pool,&vg_pool_size,total+1,sizeof(volgrad*)) )
    return 0;
  for ( k = 0 ; k < total ; k++ ) vg_pool[k] = new_vgrad();

  /* pass 3: each vertex links its chain in the order film_grad_l() would
     and sums its terms in facet order */
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for ( k = 0 ; k < nv ; k++ )
  { volgrad **mine = vg_pool + vg_first[k];
    int count = 0, m;
    long e;
    if ( vg_first[k+1] == vg_first[k] ) continue;
    for ( e = vg_start[k] ; e < vg_start[k+1] ; e++ )
    { long f = vg_terms[e]/3;
      double *g = vg_grad + 3*vg_terms[e];
      int s;
      for ( s = 0 ; s < 2 ; s++ )
      { int fixnum = vg_fixnum[2*f+s];
        volgrad *vg;
        if ( fixnum < 0 ) continue;
        for ( m = 0 ; m < count ; m++ ) if ( mine[m]->fixnum == fixnum ) break;
        vg = mine[m];
        if ( m == count )
        { vg->fixnum = fixnum;
          vg->bb_id = vg_body[2*f+s];
          vg->chain = NULL;
          if ( count ) mine[count-1]->chain = vg;
          count++;
        }
        if ( s == 0 )
        { vg->grad[0] += g[0];
          vg->grad[1] += g[1];
          vg->grad[2] += g[2];
        }
        else
        { vg->grad[0] -= g[0];
          vg->grad[1] -= g[1];
          vg->grad[2] -= g[2];
        }
      }
    }
    set_vertex_vgrad(fv_verts[k],mine[0]);
  }

  /* film_grad_l() leaves int_val at the last facet */
  int_val = ordinal(get_original(list[n-1])) + 1;
  return 1;
}

/**************************************************************************
 * Vertices and selected elements
 *
 * The vertex list (FOR_ALL_VERTICES order, cached like the facet list)
 * lets per-vertex loops run in parallel. Loops that must stay serial
 * (expression evaluation, ordered sums) but act on few elements use
 * FL_FOR_SELECTED: a parallel scan for the elements with given attribute
 * bits, then the loop body for those, in traversal order.
 */

static vertex_id *vl_list = NULL;
static long vl_list_size = 0;
static long vl_count = 0;
static long vl_stamp = -1;
static long vl_vcount = -1;
static long vl_maxord = -1;

/* traversal-order list of VERTEX or EDGE elements, or NULL */
static element_id *type_list(int type, long *n)
{ element_id **list;
  long *size, *count, *stamp, *ecount, *maxord;
  element_id id;
  long k;
  static edge_id *edges = NULL;
  static long edges_size = 0, edges_count = 0, edges_stamp = -1,
              edges_ecount = -1, edges_maxord = -1;

  if ( type == VERTEX )
  { list = &vl_list; size = &vl_list_size; count = &vl_count;
    stamp = &vl_stamp; ecount = &vl_vcount; maxord = &vl_maxord;
  }
  else
  { list = &edges; size = &edges_size; count = &edges_count;
    stamp = &edges_stamp; ecount = &edges_ecount; maxord = &edges_maxord;
  }
  if ( !(*list && *stamp == top_timestamp && *ecount == web.skel[type].count
         && *maxord == (long)web.skel[type].max_ord) )
  { if ( !ensure_size((void**)list,size,web.skel[type].count + 1,sizeof(element_id)) )
      return NULL;
    k = 0;
    for ( id = web.skel[type].used ; valid_id(id) ; id = elptr(id)->forechain )
    { if ( !valid_element(id) ) continue;
      if ( k < *size ) (*list)[k] = id;
      k++;
    }
    if ( k != web.skel[type].count ) return NULL;
    *count = k;
    *stamp = top_timestamp;
    *ecount = web.skel[type].count;
    *maxord = (long)web.skel[type].max_ord;
  }
  else if ( fl_check() )
  { k = 0;
    for ( id = web.skel[type].used ; valid_id(id) ; id = elptr(id)->forechain )
    { if ( !valid_element(id) ) continue;
      if ( k >= *count || !equal_id((*list)[k],id) )
      { fprintf(stderr,"%s list cache mismatch at position %ld\n",
                typenames[type],k);
        abort();
      }
      k++;
    }
    if ( k != *count )
    { fprintf(stderr,"%s list cache has %ld, surface %ld\n",typenames[type],*count,k);
      abort();
    }
  }
  *n = *count;
  return *list;
}

/* traversal-order list of the vertices or edges (cached), or NULL */
element_id *fl_element_list(int type, long *n)
{ if ( type != VERTEX && type != EDGE ) return NULL;
  return type_list(type,n);
}

/* the vertex list, when the parallel vertex loops apply */
static vertex_id *vertex_list(long *n)
{ if ( fl_disabled() || threadflag ) return NULL;
  return type_list(VERTEX,n);
}

#define FL_SELECT_SITES 8
static element_id *sel_buf[FL_SELECT_SITES];
static long sel_buf_size[FL_SELECT_SITES];
static unsigned char *sel_mark = NULL;
static long sel_mark_size = 0;

long fl_attr_stamp = 0;   /* bumped by set_attr() and the like (inline.h) */

/* per site: what its selection was made from */
static struct { int type; ATTR bits; long top, attr, count, maxord, n; } sel_key[FL_SELECT_SITES];
static int sel_key_valid[FL_SELECT_SITES];

static void fl_sel_begin_impl(fl_sel *s, int type, ATTR bits, int site)
{ element_id *list;
  long n, k, hits = 0;
  int cached;
  s->type = type;
  s->bits = bits;
  s->list = NULL;
  s->n = s->k = 0;
  s->id = NULLID;
  if ( fl_disabled() || threadflag || site < 0 || site >= FL_SELECT_SITES )
    return;
  list = type_list(type,&n);
  if ( !list ) return;
  if ( bits == 0 ) { s->list = list; s->n = n; return; }
  /* the same elements and attributes as last time: the same selection */
  cached = sel_key_valid[site] && sel_key[site].type == type && sel_key[site].bits == bits
           && sel_key[site].top == top_timestamp && sel_key[site].attr == fl_attr_stamp
           && sel_key[site].count == n && sel_key[site].maxord == (long)web.skel[type].max_ord;
  if ( cached && !fl_check() )
  { s->list = sel_buf[site]; s->n = sel_key[site].n; return; }
  if ( !ensure_size((void**)&sel_mark,&sel_mark_size,n,1) ) return;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(n)) reduction(+:hits)
#endif
  for ( k = 0 ; k < n ; k++ )
  { sel_mark[k] = (elptr(list[k])->attr & bits) != 0;
    hits += sel_mark[k];
  }
  if ( !ensure_size((void**)&sel_buf[site],&sel_buf_size[site],hits+1,sizeof(element_id)) )
    return;
  if ( cached )   /* fl_check(): the cached selection must be this one */
  { long m = 0;
    for ( k = 0 ; k < n ; k++ )
      if ( sel_mark[k] && (m >= sel_key[site].n || !equal_id(sel_buf[site][m++],list[k])) )
      { fprintf(stderr,"stale %s selection at site %d\n",typenames[type],site);
        abort();
      }
    if ( m != sel_key[site].n || hits != m )
    { fprintf(stderr,"stale %s selection at site %d (%ld, now %ld)\n",typenames[type],
              site,sel_key[site].n,hits);
      abort();
    }
  }
  sel_key_valid[site] = 0;
  s->n = 0;
  for ( k = 0 ; s->n < hits ; k++ )
    if ( sel_mark[k] ) sel_buf[site][s->n++] = list[k];
  s->list = sel_buf[site];
  sel_key[site].type = type;
  sel_key[site].bits = bits;
  sel_key[site].top = top_timestamp;
  sel_key[site].attr = fl_attr_stamp;
  sel_key[site].count = n;
  sel_key[site].maxord = (long)web.skel[type].max_ord;
  sel_key[site].n = s->n;
  sel_key_valid[site] = 1;
}

int fl_sel_next(fl_sel *s, element_id *id)
{ if ( s->list )
  { if ( s->k >= s->n ) return 0;
    *id = s->list[s->k++];
    return 1;
  }
  /* fallback: Evolver's own traversal, same selection */
  if ( !valid_id(s->id) ) s->id = (s->k++ == 0) ? web.skel[s->type].used : NULLID;
  else s->id = elptr(s->id)->forechain;
  while ( valid_id(s->id) && (!valid_element(s->id)
            || (s->bits && !(elptr(s->id)->attr & s->bits))) )
    s->id = elptr(s->id)->forechain;
  if ( !valid_id(s->id) ) return 0;
  *id = s->id;
  return 1;
}

static int fl_zero_forces_impl(void)
{ vertex_id *list;
  long n, k;
  if ( !(list = vertex_list(&n)) ) return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { REAL *f = get_force(list[k]);
    int i;
    for ( i = 0 ; i < SDIM ; i++ ) f[i] = 0.0;
    set_vertex_valence(list[k],0);
  }
  return 1;
}

static int fl_move_vertices_impl(REAL scale, int dim)
{ vertex_id *list;
  long n, k;
  if ( !(list = vertex_list(&n)) ) return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    ATTR attr = get_vattr(v_id);
    REAL *velocity, *x;
    int i;
    if ( attr & FIXED ) continue;
    velocity = get_velocity(v_id);
    if ( attr & BOUNDARY )
    { int pcount = get_boundary(v_id)->pcount;
      REAL *param = get_param(v_id);
      for ( i = 0 ; i < pcount ; i++ )
        param[i] += scale*velocity[i];
    }
    else
    { x = get_coord(v_id);
      for ( i = 0 ; i < dim ; i++ )
        x[i] += scale*velocity[i];
    }
  }
  return 1;
}

static int fl_save_coords_impl(REAL (*coord)[MAXCOORD])
{ vertex_id *list;
  long n, k;
  if ( !(list = vertex_list(&n)) ) return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    REAL *to = coord ? coord[loc_ordinal(v_id)] : get_oldcoord(v_id);
    if ( get_vattr(v_id) & BOUNDARY )
      memcpy(to,get_param(v_id),sizeof(REAL)*web.maxparam);
    else
      memcpy(to,get_coord(v_id),sizeof(REAL)*SDIM);
  }
  return 1;
}

static int fl_restore_coords_impl(REAL (*coord)[MAXCOORD])
{ vertex_id *list;
  long n, k;
  if ( !(list = vertex_list(&n)) ) return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    if ( get_vattr(v_id) & BOUNDARY ) continue;   /* serial: evaluates */
    memcpy(get_coord(v_id),coord ? coord[loc_ordinal(v_id)] : get_oldcoord(v_id),
           sizeof(REAL)*SDIM);
  }
  return 1;
}

static int fl_volume_restore_impl(REAL stepsize, REAL *vol_restore, int fixcount)
{ vertex_id *list;
  long n, k;
  if ( approx_curve_flag ) return 0;
  if ( !(list = vertex_list(&n)) ) return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    ATTR attr = get_vattr(v_id);
    REAL *x;
    volgrad *vgptr;
    int bi, i;
    if ( attr & (CONSTRAINT|BOUNDARY|FIXED) ) continue;  /* serial or none */
    x = get_coord(v_id);
    for ( vgptr = get_vertex_vgrad(v_id) ; vgptr ; vgptr = vgptr->chain )
    { bi = vgptr->fixnum;
      if ( (bi < 0) || (bi >= fixcount) ) continue;
      for ( i = 0 ; i < SDIM ; i++ )
        x[i] += stepsize*vol_restore[bi]*vgptr->velocity[i];
    }
  }
  return 1;
}

/* dense DV^T DV for at most this many constraints */
#define FL_LEFTSIDE_MAX 256

static double *ls_buf = NULL;
static long ls_buf_size = 0;

static int fl_calc_leftside_impl(REAL **rleftside, struct linsys *S, int fixcount, int *degfree)
{ vertex_id *list;
  long n, k, m = (long)fixcount*fixcount, ms = padded(m,sizeof(double));
  int threads, t, deg = 0;
  if ( approx_curve_flag || fixcount > FL_LEFTSIDE_MAX ) return 0;
  if ( !(list = vertex_list(&n)) ) return 0;
  threads = loop_threads(n);
  if ( !ensure_size((void**)&ls_buf,&ls_buf_size,threads*ms,sizeof(double)) )
    return 0;
  memset(ls_buf,0,threads*ms*sizeof(double));
#ifdef _OPENMP
  #pragma omp parallel num_threads(threads) reduction(+:deg)
#endif
  { double *a = ls_buf + THREAD_NUM()*ms;
    long kk;
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for ( kk = 0 ; kk < n ; kk++ )
    { vertex_id v_id = list[kk];
      volgrad *vgi, *vgj;
      if ( get_vattr(v_id) & FIXED ) continue;
      for ( vgi = get_vertex_vgrad(v_id) ; vgi ; vgi = vgi->chain )
      { int bi = vgi->fixnum;
        if ( (bi < 0) || (bi >= fixcount) ) continue;
        deg++;
        a[bi*fixcount+bi] += SDIM_dot(vgi->velocity,vgi->grad);
        for ( vgj = vgi->chain ; vgj ; vgj = vgj->chain )
        { int bj = vgj->fixnum;
          REAL tmp = SDIM_dot(vgi->grad,vgj->velocity);
          if ( (bj < 0) || (bj >= fixcount) ) continue;
          a[bi*fixcount+bj] += tmp;
          a[bj*fixcount+bi] += tmp;
        }
      }
    }
  }
  for ( t = 1 ; t < threads ; t++ )
    for ( k = 0 ; k < m ; k++ )
      ls_buf[k] += ls_buf[t*ms + k];
  if ( S )   /* sparse: upper triangle */
  { int i, j;
    for ( i = 0 ; i < fixcount ; i++ )
      for ( j = i ; j < fixcount ; j++ )
        sp_hash_search(S,i,j,ls_buf[i*fixcount + j]);
  }
  else
    for ( k = 0 ; k < m ; k++ )
      rleftside[k/fixcount][k%fixcount] += ls_buf[k];
  *degfree += deg;
  return 1;
}

/* Vertex normals in the plain linear film (the usual case of
   new_calc_vertex_normal(): no triple edge at the vertex, not axial): the
   sum of its facets' normals, each flipped to agree with the sum so far, in
   Evolver's order around the vertex. The facets around each vertex, in that
   order and with their orientation, are kept per topology (vn_*), and each
   facet's normal is computed once per call instead of once per corner.
   Results agree with new_calc_vertex_normal() to round-off (an inverted
   facet's normal is the negated one).

   The Lagrange model likewise: lagrange_facet_normal() gives a facet's unit
   normal at one of its nodes, from all the facet's nodes, so each facet's
   normals at all its nodes are computed at once (vn_lnorm, by facet ordinal
   and node) instead of one get_facet_verts() per node and facet. A vertex
   then combines them as new_calc_vertex_normal() does: a corner as above
   (VN_ORIENTED); a node inside an edge of one or two facets, their plain
   sum, and a node inside a facet, its facet's normal (VN_SUM). Any other
   vertex (triple or bare edges, degenerate facets) is left to
   new_calc_vertex_normal(). */
#define VN_OTHER 0
#define VN_ORIENTED 1
#define VN_SUM 2
static long *vn_start = NULL, vn_start_size = 0;   /* [n+1] into vn_fac */
static long *vn_fac = NULL, vn_fac_size = 0;       /* +-(facet ordinal + 1) */
static int *vn_node = NULL;                        /* Lagrange: node in facet */
static long vn_node_size = 0;
static unsigned char *vn_kind = NULL;              /* per vertex: VN_* */
static long vn_kind_size = 0;
static long vn_stamp = -1, vn_count = -1, vn_maxord = -1;
static int vn_model = -1, vn_order = -1;
static vertex_id *vn_list = NULL;
static double (*vn_fnorm)[3] = NULL;
static long vn_fnorm_size = 0;
static double (*vn_lnorm)[3] = NULL;   /* Lagrange: [facet ordinal*ctrl + node] */
static long vn_lnorm_size = 0;

static int vn_applicable(void)
{ return web.representation == SOAPFILM
      && (web.modeltype == LINEAR || (web.modeltype == LAGRANGE && web.dimension == 2))
      && !web.symmetry_flag && SDIM == 3 && !hessian_special_normal_flag
      && !hessian_normal_one_flag;
}

/* whether new_calc_vertex_normal() takes the usual case at this vertex */
static int vn_usual(vertex_id v_id)
{ edge_id e_id, start_e;
  if ( get_vattr(v_id) & (AXIAL_POINT|Q_MIDPOINT|Q_MIDEDGE|Q_MIDFACET) ) return 0;
  e_id = start_e = get_vertex_edge(v_id);
  if ( !valid_id(e_id) ) return 0;
  do
  { if ( get_edge_valence(e_id) >= 3 ) return 0;
    e_id = get_next_tail_edge(e_id);
  } while ( !equal_id(e_id,start_e) );
  return valid_id(get_vertex_fe(v_id));
}

/* append facet f_id (as oriented) to the list; Lagrange: with v_id's node */
static int vn_push(long *m, long n, facet_id f_id, vertex_id v_id, int lag)
{ long o = (long)ordinal(f_id) + 1;
  if ( *m >= vn_fac_size || (lag && *m >= vn_node_size) )
  { long want = vn_fac_size ? 2*vn_fac_size : 8*n + 64;
    long *grown = (long*)realloc(vn_fac,(size_t)want*sizeof(long));
    if ( !grown ) return 0;
    vn_fac = grown; vn_fac_size = want;
    if ( lag )
    { int *ngrown = (int*)realloc(vn_node,(size_t)want*sizeof(int));
      if ( !ngrown ) return 0;
      vn_node = ngrown; vn_node_size = want;
    }
  }
  if ( lag )
  { vertex_id *v = get_facet_vertices(f_id);
    int ctrl = web.skel[FACET].ctrlpts, i;
    for ( i = 0 ; i < ctrl ; i++ ) if ( v[i] == v_id ) break;
    if ( i >= ctrl ) return 0;
    vn_node[*m] = i;
  }
  vn_fac[(*m)++] = inverted(f_id) ? -o : o;
  return 1;
}

/* new_calc_vertex_normal()'s case for a Lagrange vertex, with its facets
   listed; -1: out of memory */
static int vn_lagrange_vertex(vertex_id v_id, long *m, long n)
{ ATTR attr = get_vattr(v_id);
  if ( attr & Q_MIDFACET )
    return vn_push(m,n,get_vertex_facet(v_id),v_id,1) ? VN_SUM : -1;
  if ( attr & Q_MIDEDGE )
  { edge_id e_id = get_vertex_edge(v_id);
    facetedge_id sidea, sideb;
    int valence;
    if ( !valid_id(e_id) ) return VN_OTHER;
    valence = get_edge_valence(e_id);   /* the only edge it counts */
    if ( valence < 1 || valence > 2 ) return VN_OTHER;
    sidea = get_vertex_fe(v_id);
    if ( !valid_id(sidea) ) return VN_OTHER;
    if ( inverted(get_fe_facet(sidea)) ) sidea = inverse_id(sidea);
    if ( !equal_id(get_next_facet(sidea),get_prev_facet(sidea)) ) return VN_OTHER;
    if ( !vn_push(m,n,get_fe_facet(sidea),v_id,1) ) return -1;
    sideb = get_next_facet(sidea);
    if ( !equal_id(sidea,sideb) )
      if ( !vn_push(m,n,get_fe_facet(inverse_id(sideb)),v_id,1) ) return -1;
    return VN_SUM;
  }
  if ( vn_usual(v_id) )
  { facetedge_id fe, start_fe;
    fe = start_fe = get_vertex_first_facet(v_id);
    do
    { if ( !vn_push(m,n,get_fe_facet(fe),v_id,1) ) return -1;
      fe = get_next_vertex_facet(v_id,fe);
    } while ( !equal_element(fe,start_fe) );
    return VN_ORIENTED;
  }
  return VN_OTHER;
}

static int vn_build(vertex_id *list, long n)
{ long k, m = 0;
  int lag = web.modeltype == LAGRANGE;
  if ( vn_list == list && vn_stamp == top_timestamp && vn_count == n
       && vn_maxord == (long)web.skel[VERTEX].max_ord && vn_model == web.modeltype
       && vn_order == web.lagrange_order )
    return 1;
  if ( !ensure_size((void**)&vn_start,&vn_start_size,n + 1,sizeof(long))
       || !ensure_size((void**)&vn_kind,&vn_kind_size,n,1) )
    return 0;
  vn_stamp = -1;
  vn_start[0] = 0;
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    int kind = VN_OTHER;
    long m0 = m;
    if ( lag )
    { kind = vn_lagrange_vertex(v_id,&m,n);
      if ( kind < 0 ) return 0;
    }
    else if ( vn_usual(v_id) )
    { facetedge_id fe, start_fe;
      kind = VN_ORIENTED;
      fe = start_fe = get_vertex_first_facet(v_id);
      do
      { if ( !vn_push(&m,n,get_fe_facet(fe),v_id,0) ) return 0;
        fe = get_next_vertex_facet(v_id,fe);
      } while ( !equal_element(fe,start_fe) );
    }
    if ( kind == VN_OTHER ) m = m0;
    vn_kind[k] = (unsigned char)kind;
    vn_start[k+1] = m;
  }
  vn_list = list;
  vn_count = n;
  vn_maxord = (long)web.skel[VERTEX].max_ord;
  vn_model = web.modeltype;
  vn_order = web.lagrange_order;
  vn_stamp = top_timestamp;
  return 1;
}

/* the facet normals (length = area), by ordinal, as get_facet_normal() */
static int vn_facet_normals(void)
{ vertex_id *corners = fl_facet_corners();
  facet_id *facets;
  long nf, j;
  if ( !corners || !(facets = fl_facet_list(&nf)) ) return 0;
  if ( !ensure_size((void**)&vn_fnorm,&vn_fnorm_size,(long)web.skel[FACET].max_ord + 1,
                    sizeof(double[3])) )
    return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(nf))
#endif
  for ( j = 0 ; j < nf ; j++ )
  { long o = (long)ordinal(facets[j]);
    vertex_id *c = corners + 3*o;
    REAL *x0 = get_coord(c[0]), *x1 = get_coord(c[1]), *x2 = get_coord(c[2]);
    REAL s1[3], s2[3], nrm[3];
    int i;
    for ( i = 0 ; i < 3 ; i++ ) { s1[i] = x1[i] - x0[i]; s2[i] = x2[i] - x1[i]; }
    cross_prod(s1,s2,nrm);
    for ( i = 0 ; i < 3 ; i++ ) vn_fnorm[o][i] = nrm[i]/2;
  }
  return 1;
}

/* Lagrange: each facet's unit normal at each of its nodes, as
   lagrange_facet_normal() computes it for the facet's positive orientation
   (same sums, same kernel_basis_rows()); NAN where that is not one vector */
static int vn_lagrange_normals(void)
{ facet_id *facets;
  long nf, j;
  int ctrl = web.skel[FACET].ctrlpts;
  struct gauss_lag *gl = &gauss_lagrange[web.dimension][web.gauss2D_order];
  if ( !(facets = fl_facet_list(&nf)) || ctrl > MAXVCOUNT || !gl->lpolypart ) return 0;
  if ( !ensure_size((void**)&vn_lnorm,&vn_lnorm_size,
                    ((long)web.skel[FACET].max_ord + 1)*ctrl,sizeof(double[3])) )
    return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(nf))
#endif
  for ( j = 0 ; j < nf ; j++ )
  { facet_id f_id = positive_id(facets[j]);
    long o = (long)ordinal(f_id);
    MAT2D(sides,MAXCOORD,MAXCOORD);
    MAT2D(vx,MAXVCOUNT,MAXCOORD);
    MAT2D(nrm,MAXCOORD,MAXCOORD);
    int i, jj, k, kk;
    get_facet_verts(f_id,vx,NULL);
    for ( k = 0 ; k < ctrl ; k++ )
    { double *out = vn_lnorm[o*ctrl + k];
      for ( i = 1 ; i <= 2 ; i++ )
        for ( jj = 0 ; jj < SDIM ; jj++ )
          for ( kk = 0, sides[i][jj] = 0.0 ; kk < ctrl ; kk++ )
            sides[i][jj] += gl->lpolypart[k][i-1][kk]*vx[kk][jj];
      if ( kernel_basis_rows(sides+1,nrm,2,SDIM) == 1 )
        for ( i = 0 ; i < 3 ; i++ ) out[i] = nrm[0][i];
      else
        out[0] = out[1] = out[2] = NAN;
    }
  }
  return 1;
}

/* new_calc_vertex_normal() for the k-th vertex of the list vn_build() saw */
static int vn_vertex_normal(long k, vertex_id v_id, REAL **norm)
{ long j;
  int i, d;
  int lag = vn_model == LAGRANGE;
  int ctrl = web.skel[FACET].ctrlpts;
  if ( vn_kind[k] == VN_OTHER ) return new_calc_vertex_normal(v_id,norm);
  for ( d = 0 ; d < SDIM ; d++ )
    for ( i = 0 ; i < SDIM ; i++ ) norm[d][i] = 0.0;
  for ( j = vn_start[k] ; j < vn_start[k+1] ; j++ )
  { long o = vn_fac[j];
    double *f = lag ? vn_lnorm[((o < 0 ? -o : o) - 1)*ctrl + vn_node[j]]
                    : vn_fnorm[(o < 0 ? -o : o) - 1];
    REAL fn[3];
    if ( isnan(f[0]) ) return new_calc_vertex_normal(v_id,norm);
    for ( i = 0 ; i < 3 ; i++ ) fn[i] = o < 0 ? -f[i] : f[i];
    if ( vn_kind[k] == VN_ORIENTED && j > vn_start[k] && SDIM_dot(norm[0],fn) < 0 )
      for ( i = 0 ; i < 3 ; i++ ) norm[0][i] -= fn[i];
    else
      for ( i = 0 ; i < 3 ; i++ ) norm[0][i] += fn[i];
  }
  if ( fl_check() )   /* against Evolver's own */
  { MAT2D(ref,MAXCOORD,MAXCOORD);
    int kk = new_calc_vertex_normal(v_id,ref);
    REAL scale = sqrt(SDIM_dot(ref[0],ref[0])) + 1e-300;
    for ( i = 0 ; i < 3 ; i++ )
      if ( kk != 1 || fabs(ref[0][i] - norm[0][i]) > 1e-12*scale )
      { fprintf(stderr,"vertex normal cache mismatch at vertex %ld\n",(long)ordinal(v_id));
        abort();
      }
  }
  return 1;
}

/* set up the cached normals for the vertices of list; 0: not applicable */
static int vn_prepare(vertex_id *list, long n)
{ if ( !vn_applicable() || !vn_build(list,n) ) return 0;
  return web.modeltype == LAGRANGE ? vn_lagrange_normals() : vn_facet_normals();
}

/* hessian_init()'s vertex normals, see fastloops.h. The normals only read
   the surface; an error (kb_error) leaves the vertex to the caller, which
   redoes it serially so Evolver reports it as usual. */
static int *fl_vertex_normals_impl(REAL ***v_normal, vertex_id **listp, long *np)
{ vertex_id *list;
  int *dims;
  long n, k;
  int fast;
  if ( web.representation == SIMPLEX || hessian_special_normal_flag
       || (web.symmetry_flag && !web.torus_flag) )
    return NULL;
  if ( !(list = vertex_list(&n)) || n < FL_PARALLEL_MIN ) return NULL;
  if ( !(dims = (int*)malloc(n*sizeof(int))) ) return NULL;
  fast = vn_prepare(list,n);
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic,256) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    jmp_buf trap;
    dims[k] = -1;
    if ( get_vattr(v_id) & (FIXED|BOUNDARY|NO_HESSIAN_NORMAL_ATTR) ) continue;
    if ( get_v_constraint_map(v_id)[0] ) continue;
    if ( setjmp(trap) == 0 )
    { int kk;
      fl_trap = &trap;   /* kb_error() comes back here */
      kk = fast ? vn_vertex_normal(k,v_id,v_normal[k])
                : new_calc_vertex_normal(v_id,v_normal[k]);
      dims[k] = project_vertex_normals(v_id,v_normal[k],kk);
      fl_trap = NULL;
    }
    else dims[k] = -1;
  }
  *listp = list;
  *np = n;
  return dims;
}

static int *fl_shape_normals_impl(REAL ***v_normal, vertex_id **listp, long *np)
{ vertex_id *list;
  int *dims;
  long n, k;
  int fast;
  if ( web.representation == SIMPLEX || (web.symmetry_flag && !web.torus_flag) )
    return NULL;
  if ( !(list = vertex_list(&n)) || n < FL_PARALLEL_MIN ) return NULL;
  if ( !(dims = (int*)malloc(n*sizeof(int))) ) return NULL;
  fast = vn_prepare(list,n);
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic,256) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    jmp_buf trap;
    dims[k] = -2;
    if ( get_vattr(v_id) & FIXED ) { dims[k] = 0; continue; }
    if ( get_vattr(v_id) & BOUNDARY ) continue;
    if ( get_v_constraint_map(v_id)[0] ) continue;
    if ( setjmp(trap) == 0 )
    { int kk;
      fl_trap = &trap;   /* kb_error() comes back here */
      kk = fast ? vn_vertex_normal(k,v_id,v_normal[k])
                : new_calc_vertex_normal(v_id,v_normal[k]);
      kk = gram_schmidt(v_normal[k],kk,SDIM);
      dims[k] = kk > 0 ? kk : -1;
      fl_trap = NULL;
    }
    else dims[k] = -2;
  }
  *listp = list;
  *np = n;
  return dims;
}

/**************************************************************************
 * Lagrange facet setup
 *
 * q_facet_setup_lagrange() maps a facet's control points to positions and
 * tangents at the Gauss points: gauss_pt = gpoly x and, per Gauss point g,
 * sides[g] = gpolypart[g] x, through the generic mat_mult() on REAL**
 * matrices. Here the basis matrices are packed once, column-major over all
 * output rows (Gauss points, then each Gauss point's tangents), so each
 * coordinate is one contiguous, vectorizable loop per control point. Sums
 * run over control points in the same order as mat_mult().
 */

static double *lg_packed = NULL;    /* [ctrl][rows] */
static long lg_packed_size = 0;
static REAL **lg_gpoly = NULL;      /* what the packing was made from */
static int lg_ctrl = -1, lg_gnumpts = -1, lg_dim = -1;
static long lg_version = -1;
/* bumped by gauss_lagrange_setup() (model.c), which can rebuild the tables in
   place (same arrays, same sizes: switching bezier_basis, say) */
long fl_lagrange_tables_version = 0;

#define LG_MAXROWS 1024
#define LG_MAXCTRL 64

/* out_k[r] = sum_l col_l[r] x_k[l], k = 0..2, summed over l in order; an AVX2
   version is picked at run time where available (no FMA: same rounding).
   Blocks of four rows keep their sums in registers over all control points
   (the packed columns are padded to a multiple of four rows). */
#if defined(__GNUC__)
typedef double lg_v4 __attribute__((vector_size(32)));
#endif
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
__attribute__((target_clones("avx2","default")))
#endif
static void lg_kernel(const double *restrict packed, int stride, int rows, int ctrl,
                      double X[3][LG_MAXCTRL], double out[3][LG_MAXROWS])
{ int l, r;
#if defined(__GNUC__)
  for ( r = 0 ; r < rows ; r += 4 )
  { lg_v4 a0 = {0.0,0.0,0.0,0.0}, a1 = a0, a2 = a0;
    for ( l = 0 ; l < ctrl ; l++ )
    { lg_v4 c;
      memcpy(&c,packed + (long)l*stride + r,sizeof c);
      a0 += c*X[0][l];
      a1 += c*X[1][l];
      a2 += c*X[2][l];
    }
    memcpy(out[0] + r,&a0,sizeof a0);
    memcpy(out[1] + r,&a1,sizeof a1);
    memcpy(out[2] + r,&a2,sizeof a2);
  }
#else
  int q;
  for ( r = 0 ; r < rows ; r += 4 )
  { double a0[4] = {0.0,0.0,0.0,0.0}, a1[4] = {0.0,0.0,0.0,0.0},
           a2[4] = {0.0,0.0,0.0,0.0};
    for ( l = 0 ; l < ctrl ; l++ )
    { const double *col = packed + (long)l*stride + r;
      for ( q = 0 ; q < 4 ; q++ )
      { a0[q] += col[q]*X[0][l];
        a1[q] += col[q]*X[1][l];
        a2[q] += col[q]*X[2][l];
      }
    }
    for ( q = 0 ; q < 4 ; q++ )
    { out[0][r+q] = a0[q]; out[1][r+q] = a1[q]; out[2][r+q] = a2[q]; }
  }
#endif
}

int fl_lagrange_facet_setup(struct gauss_lag *gl, int dim, int ctrl,
                            REAL **x, REAL **gauss_pt, REAL ***sides, int need_side)
{ int rows, l, g, d, k, stride;
  double X[3][LG_MAXCTRL];
  double out[3][LG_MAXROWS];

  if ( fl_disabled() || SDIM != 3 ) return 0;
  rows = gl->gnumpts*(1 + dim);
  stride = (rows + 3) & ~3;
  if ( stride > LG_MAXROWS || ctrl > LG_MAXCTRL || ctrl != gl->lagpts ) return 0;

  if ( lg_gpoly != gl->gpoly || lg_ctrl != ctrl || lg_gnumpts != gl->gnumpts
       || lg_dim != dim || lg_version != fl_lagrange_tables_version )
  { if ( !ensure_size((void**)&lg_packed,&lg_packed_size,(long)ctrl*stride,
                      sizeof(double)) )
      return 0;
    for ( l = 0 ; l < ctrl ; l++ )
    { double *col = lg_packed + (long)l*stride;
      for ( g = 0 ; g < gl->gnumpts ; g++ )
      { col[g] = gl->gpoly[g][l];
        for ( d = 0 ; d < dim ; d++ )
          col[gl->gnumpts + g*dim + d] = gl->gpolypart[g][d][l];
      }
      for ( g = rows ; g < stride ; g++ ) col[g] = 0.0;
    }
    lg_gpoly = gl->gpoly;
    lg_version = fl_lagrange_tables_version;
    lg_ctrl = ctrl;
    lg_gnumpts = gl->gnumpts;
    lg_dim = dim;
  }
  if ( !need_side ) rows = gl->gnumpts;

  for ( l = 0 ; l < ctrl ; l++ )
    for ( k = 0 ; k < 3 ; k++ )
      X[k][l] = x[l][k];
  lg_kernel(lg_packed,stride,rows,ctrl,X,out);
  for ( g = 0 ; g < gl->gnumpts ; g++ )
    for ( k = 0 ; k < 3 ; k++ )
      gauss_pt[g][k] = out[k][g];
  if ( need_side )
    for ( g = 0 ; g < gl->gnumpts ; g++ )
      for ( d = 0 ; d < dim ; d++ )
        for ( k = 0 ; k < 3 ; k++ )
          sides[g][d][k] = out[k][gl->gnumpts + g*dim + d];
  return 1;
}

/**************************************************************************
 * Interrupts
 *
 * pySE aborts a command on a second Ctrl-C by longjmp-ing out of the
 * signal handler. That must not happen inside a parallel region or MUMPS:
 * the functions with parallel regions run inside fl_enter()/fl_leave(), and
 * an abort that arrives there is held until fl_leave().
 */

volatile sig_atomic_t fl_critical = 0;
volatile sig_atomic_t fl_abort_pending = 0;
void (*fl_abort_hook)(void) = NULL;

void fl_enter(void) { fl_critical++; }

void fl_leave(void)
{ if ( fl_critical > 0 ) fl_critical--;
  if ( fl_critical == 0 && fl_abort_pending )
  { fl_abort_pending = 0;
    if ( fl_abort_hook ) (*fl_abort_hook)();
  }
}

int fl_facet_volumes(void)
{ int r; fl_enter(); r = fl_facet_volumes_impl(); fl_leave(); return r; }

int fl_facet_energies(void)
{ int r; fl_enter(); r = fl_facet_energies_impl(); fl_leave(); return r; }

int fl_facet_forces(void)
{ int r; fl_enter(); r = fl_facet_forces_impl(); fl_leave(); return r; }

int fl_film_grad(void)
{ int r; fl_enter(); r = fl_film_grad_impl(); fl_leave(); return r; }

int fl_zero_forces(void)
{ int r; fl_enter(); r = fl_zero_forces_impl(); fl_leave(); return r; }

int *fl_vertex_normals(REAL ***v_normal, vertex_id **list, long *n)
{ int *r; fl_enter(); r = fl_vertex_normals_impl(v_normal,list,n); fl_leave(); return r; }

int *fl_shape_normals(REAL ***v_normal, vertex_id **list, long *n)
{ int *r; fl_enter(); r = fl_shape_normals_impl(v_normal,list,n); fl_leave(); return r; }

int fl_move_vertices(REAL scale, int dim)
{ int r; fl_enter(); r = fl_move_vertices_impl(scale,dim); fl_leave(); return r; }

int fl_save_coords(REAL (*coord)[MAXCOORD])
{ int r; fl_enter(); r = fl_save_coords_impl(coord); fl_leave(); return r; }

int fl_restore_coords(REAL (*coord)[MAXCOORD])
{ int r; fl_enter(); r = fl_restore_coords_impl(coord); fl_leave(); return r; }

int fl_volume_restore(REAL stepsize, REAL *vol_restore, int fixcount)
{ int r; fl_enter(); r = fl_volume_restore_impl(stepsize,vol_restore,fixcount); fl_leave(); return r; }

int fl_calc_leftside(REAL **rleftside, struct linsys *S, int fixcount, int *degfree)
{ int r; fl_enter(); r = fl_calc_leftside_impl(rleftside,S,fixcount,degfree); fl_leave(); return r; }

void fl_sel_begin(fl_sel *s, int type, ATTR bits, int site)
{ fl_enter(); fl_sel_begin_impl(s,type,bits,site); fl_leave(); }

/**************************************************************************
 * Per-thread Evolver state and error traps for parallel loops
 */

#define FL_MAXTHREADS 256
static struct thread_data *fl_tdata[FL_MAXTHREADS];   /* [0] unused */

struct thread_data *fl_thread_data(void)
{
#ifdef _OPENMP
  int t = omp_get_thread_num();   /* 0 outside parallel regions */
  if ( t > 0 && t < FL_MAXTHREADS && fl_tdata[t] ) return fl_tdata[t];
#endif
  return &default_thread_data;
}

/* Serially, before a parallel region with `threads` threads: give each
   worker thread its own eval stack (eval_all() and eval_second() push
   function arguments before eval() can grow it). */
void fl_prepare_threads(int threads)
{ int t;
  for ( t = 1 ; t < threads && t < FL_MAXTHREADS ; t++ )
  { struct thread_data *td = fl_tdata[t];
    if ( !td )
    { td = (struct thread_data *)calloc(1,sizeof(struct thread_data));
      if ( !td ) return;
      td->eval_stack_size = 1000;
      td->eval_stack = (REAL *)malloc(td->eval_stack_size*sizeof(REAL));
      if ( !td->eval_stack ) { free(td); return; }
      td->worker_id = t;
      fl_tdata[t] = td;
    }
    td->stack_top = td->eval_stack;
    td->frame_spot = 0;
    td->eval_stack[td->eval_stack_size-1] = STACKMAGIC;
  }
}

int fl_in_parallel(void)
{
#ifdef _OPENMP
  return omp_in_parallel();
#else
  return 0;
#endif
}

__thread jmp_buf *fl_trap = NULL;

void fh_reset(void);   /* fasthess.c */

/* After an error (Evolver's recovery): no loop is running any more */
void fl_reset_state(void)
{ fl_trap = NULL;
  fl_critical = 0;
  fl_abort_pending = 0;
  fh_reset();
}

void fl_trap_error(void)
{ if ( fl_trap )
  { jmp_buf *trap = fl_trap;
    fl_trap = NULL;
    longjmp(*trap,1);
  }
}

/**************************************************************************
 * Vertex averaging (V) and equiangulation's edge lengths (u)
 *
 * vertex_average() finds every vertex's new position before moving any, so
 * the search runs in parallel for vertices without constraints or
 * boundaries (those keep the serial path: constraint formulas). The search
 * weighs neighbours by facet areas, and get_facet_area() recomputes and
 * stores the area at every call (facet_energy_l(AREA_ONLY), or in quantity
 * mode the default area quantity through a shared qinfo): here every facet's
 * area is computed once beforehand, each by one thread, and the search reads
 * it, as vertex_average()'s serial loop does. Positions don't change in
 * between, so the results are those of the serial loop: bit for bit, or to
 * round-off in quantity mode (there the area is computed by Gram-Schmidt).
 */

int find_vertex_average(vertex_id v_id, REAL *vx, int mode);   /* veravg.c */

static int fast_mesh_disabled(void)   /* PYSE_NO_FAST_MESH=1: these two off */
{ static int disabled = -1;
  if ( disabled < 0 ) disabled = getenv("PYSE_NO_FAST_MESH") != NULL;
  return disabled;
}

static __thread int va_active = 0;

/* facet_energy_l(f_id,AREA_ONLY)'s area (corners in f_id's orientation), not stored */
REAL fl_facet_area(facet_id f_id)
{ REAL unwrap_x[FACET_VERTS][MAXCOORD];
  REAL *x[FACET_VERTS];
  REAL side[2][MAXCOORD];
  REAL ss, st, tt, det;
  int i, j;
  for ( i = 0 ; i < FACET_VERTS ; i++ ) x[i] = unwrap_x[i];
  get_facet_verts(f_id,x,NULL);
  for ( i = 0 ; i < 2 ; i++ )
    for ( j = 0 ; j < SDIM ; j++ )
      side[i][j] = x[i+1][j] - x[i][j];
  ss = SDIM_dot(side[0],side[0]);
  st = SDIM_dot(side[0],side[1]);
  tt = SDIM_dot(side[1],side[1]);
  det = ss*tt - st*st;
  return det > 0.0 ? sqrt(det)/2 : 0.0;
}

int fl_vertex_average_active(void) { return va_active; }

/* In quantity mode get_facet_area() reads the area the default area
   quantity's value stores: q_facet_tension_value() of its default instance
   stores the plain geometric area (before density or moduli; to round-off,
   being computed by Gram-Schmidt) when that quantity has just this one
   global facet_area instance. */
static int quantity_area_is_plain(void)
{ struct gen_quant *q;
  struct method_instance *mi;
  if ( default_area_quant_num < 0 || default_area_quant_num >= gen_quant_count ) return 0;
  q = GEN_QUANT(default_area_quant_num);
  if ( q->method_count != 1 ) return 0;
  mi = METH_INSTANCE(q->meth_inst[0]);
  return mi->type == FACET && !(mi->flags & Q_DELETED)
         && (mi->flags & GLOBAL_INST) && (mi->flags & DEFAULT_INSTANCE)
         && basic_gen_methods[mi->gen_method].value == q_facet_tension_value;
}

/* find_vertex_average() for the plain vertex of a linear film, from a ring
   kept per topology (va_*): a vertex that is not fixed, axial, constrained
   or on a boundary, whose edges all have exactly two facets (no bare,
   single or triple edges) and which has a facet ring for its normal. For
   each of its edges in Evolver's order around the vertex (tail edges) the
   ring holds the head vertex and the edge's two facets in the order the
   search adds their areas; then the vertex's facets with their orientation,
   in new_calc_vertex_normal()'s order. The sums, the facet normals (as
   get_facet_normal() computes them, for either orientation) and the
   projection are Evolver's, in the same order: the result is the same bit
   for bit, which PYSE_CHECK_FACET_CACHE=1 checks vertex by vertex. */
typedef struct { vertex_id head; long fa, fb; } av_edge;   /* facet ordinals */
static long *av_estart = NULL, av_estart_size = 0;   /* [n+1] into av_e */
static av_edge *av_e = NULL;
static long av_e_size = 0;
static long *av_fstart = NULL, av_fstart_size = 0;   /* [n+1] into av_f */
static long *av_f = NULL, av_f_size = 0;              /* +-(facet ordinal + 1) */
static long av_stamp = -1, av_count = -1, av_maxord = -1;
static vertex_id *av_list = NULL;
static double *av_area = NULL;           /* by facet ordinal: the stored area */
static long av_area_size = 0;
static double (*av_norm)[2][3] = NULL;   /* by facet ordinal: normal, as is and inverted */
static long av_norm_size = 0;

static int av_grow(void **p, long *size, long want, size_t item)
{ void *q;
  if ( want <= *size ) return 1;
  want = want < 2*(*size) ? 2*(*size) : want;
  if ( !(q = realloc(*p,(size_t)want*item)) ) return 0;
  *p = q; *size = want;
  return 1;
}

/* the ring of a plain vertex into e and f (growing them); 0 if it isn't
   one, -1 out of memory */
static int av_vertex_ring(vertex_id v_id, av_edge **e, long *e_size, long *me,
                          long **f, long *f_size, long *mf)
{ edge_id e_id, start_e;
  facetedge_id fe, start_fe;
  if ( get_vattr(v_id) & (FIXED|AXIAL_POINT|CONSTRAINT|BOUNDARY|Q_MIDPOINT|Q_MIDEDGE|Q_MIDFACET) )
    return 0;
  e_id = start_e = get_vertex_edge(v_id);
  if ( !valid_id(e_id) || !valid_id(get_vertex_fe(v_id)) ) return 0;
  do
  { av_edge *ve;
    fe = get_edge_fe(e_id);
    if ( !valid_id(fe) || (get_eattr(e_id) & BARE_NAKED) || get_edge_valence(e_id) != 2
         || equal_id(fe,get_next_facet(fe))
         || !equal_id(get_next_facet(fe),get_prev_facet(fe))
         || !equal_id(get_edge_tailv(e_id),v_id) )
      return 0;
    if ( !av_grow((void**)e,e_size,*me + 1,sizeof(av_edge)) ) return -1;
    ve = *e + (*me)++;
    ve->head = get_edge_headv(e_id);
    ve->fa = (long)ordinal(get_fe_facet(fe));
    ve->fb = (long)ordinal(get_fe_facet(get_next_facet(fe)));
    e_id = get_next_tail_edge(e_id);
  } while ( !equal_id(e_id,start_e) );
  fe = start_fe = get_vertex_first_facet(v_id);
  do
  { facet_id f_id = get_fe_facet(fe);
    long o = (long)ordinal(f_id) + 1;
    if ( !av_grow((void**)f,f_size,*mf + 1,sizeof(long)) ) return -1;
    (*f)[(*mf)++] = inverted(f_id) ? -o : o;
    fe = get_next_vertex_facet(v_id,fe);
  } while ( !equal_element(fe,start_fe) );
  return 1;
}

/* the rings, built in parallel: each thread walks a block of the list into
   buffers of its own, which are then joined in list order */
#define AV_MAXT 256
static int av_build(vertex_id *list, long n)
{ static av_edge *te[AV_MAXT];
  static long *tf[AV_MAXT];
  static long te_size[AV_MAXT], tf_size[AV_MAXT];
  long tme[AV_MAXT], tmf[AV_MAXT];
  int threads = loop_threads(n), t, failed = 0;
  long k, me, mf;
  if ( av_list == list && av_stamp == top_timestamp && av_count == n
       && av_maxord == (long)web.skel[VERTEX].max_ord )
    return 1;
  if ( threads > AV_MAXT ) threads = AV_MAXT;
  if ( !ensure_size((void**)&av_estart,&av_estart_size,n + 1,sizeof(long))
       || !ensure_size((void**)&av_fstart,&av_fstart_size,n + 1,sizeof(long)) )
    return 0;
  av_stamp = -1;
#ifdef _OPENMP
  #pragma omp parallel num_threads(threads)
#endif
  { int me_t = 0, nt = 1;
    long lo, hi, j, m_e = 0, m_f = 0;
#ifdef _OPENMP
    me_t = omp_get_thread_num(); nt = omp_get_num_threads();
#endif
    lo = n*me_t/nt; hi = n*(me_t + 1)/nt;
    for ( j = lo ; j < hi ; j++ )
    { long me0 = m_e, mf0 = m_f;
      int r = av_vertex_ring(list[j],&te[me_t],&te_size[me_t],&m_e,
                             &tf[me_t],&tf_size[me_t],&m_f);
      if ( r < 0 )
      {
#ifdef _OPENMP
        #pragma omp atomic write
#endif
        failed = 1;
        break;
      }
      if ( r == 0 ) { m_e = me0; m_f = mf0; }
      av_estart[j+1] = m_e;   /* within the thread's block for now */
      av_fstart[j+1] = m_f;
    }
    tme[me_t] = m_e; tmf[me_t] = m_f;
    if ( me_t == 0 ) threads = nt;
  }
  if ( failed ) return 0;
  for ( t = 0, me = mf = 0 ; t < threads ; t++ ) { me += tme[t]; mf += tmf[t]; }
  if ( !av_grow((void**)&av_e,&av_e_size,me + 1,sizeof(av_edge))
       || !av_grow((void**)&av_f,&av_f_size,mf + 1,sizeof(long)) )
    return 0;
  av_estart[0] = av_fstart[0] = 0;
  for ( t = 0, me = mf = 0 ; t < threads ; t++ )
  { long lo = n*t/threads, hi = n*(t + 1)/threads;
    if ( tme[t] ) memcpy(av_e + me,te[t],(size_t)tme[t]*sizeof(av_edge));
    if ( tmf[t] ) memcpy(av_f + mf,tf[t],(size_t)tmf[t]*sizeof(long));
    for ( k = lo ; k < hi ; k++ ) { av_estart[k+1] += me; av_fstart[k+1] += mf; }
    me += tme[t]; mf += tmf[t];
  }
  av_list = list;
  av_count = n;
  av_maxord = (long)web.skel[VERTEX].max_ord;
  av_stamp = top_timestamp;
  return 1;
}

/* get_facet_normal() of the facet with corners c in its loop order */
static void av_facet_normal(vertex_id c0, vertex_id c1, vertex_id c2, double *out)
{ REAL *x0 = get_coord(c0), *x1 = get_coord(c1), *x2 = get_coord(c2);
  REAL side1[MAXCOORD], side2[MAXCOORD], normal[MAXCOORD];
  int i;
  for ( i = 0 ; i < SDIM ; i++ )
  { side1[i] = x1[i] - x0[i];
    side2[i] = x2[i] - x1[i];
  }
  cross_prod(side1,side2,normal);
  for ( i = 0 ; i < SDIM ; i++ ) out[i] = normal[i]/2;
}

/* find_vertex_average() of the k-th vertex of the list, a plain one */
static int av_average(long k, vertex_id v_id, REAL *vx, int mode)
{ REAL *oldx = get_coord(v_id);
  REAL xsum[MAXCOORD], x[MAXCOORD];
  REAL weight, total_weight = 0.0;
  long j;
  int i, cnt = 0;
  for ( i = 0 ; i < SDIM ; i++ ) { vx[i] = oldx[i]; xsum[i] = 0.0; }
  for ( j = av_estart[k] ; j < av_estart[k+1] ; j++ )
  { av_edge *ve = av_e + j;
    REAL *z;
    weight = 0.0;
    weight += av_area[ve->fa];
    weight += av_area[ve->fb];
    if ( weight == 0.0 ) continue;
    z = get_coord(ve->head);
    for ( i = 0 ; i < SDIM ; i++ ) x[i] = z[i] - oldx[i];
    for ( i = 0 ; i < SDIM ; i++ ) xsum[i] += weight*weight*x[i];
    total_weight += weight*weight;
    cnt++;
  }
  if ( cnt <= 1 ) return 0;
  if ( mode == VOLKEEP )   /* new_calc_vertex_normal()'s usual case */
  { MAT2D(norm,MAXCOORD,MAXCOORD);
    int d, n, dm;
    for ( d = 0 ; d < SDIM ; d++ )
      for ( i = 0 ; i < SDIM ; i++ ) norm[d][i] = 0.0;
    for ( j = av_fstart[k] ; j < av_fstart[k+1] ; j++ )
    { long o = av_f[j];
      double *fn = av_norm[(o < 0 ? -o : o) - 1][o < 0];
      if ( j > av_fstart[k] && SDIM_dot(norm[0],fn) < 0 )
        for ( i = 0 ; i < SDIM ; i++ ) norm[0][i] -= fn[i];
      else
        for ( i = 0 ; i < SDIM ; i++ ) norm[0][i] += fn[i];
    }
    dm = gram_schmidt(norm,1,SDIM);
    for ( n = 0 ; n < dm ; n++ )
    { REAL c = SDIM_dot(xsum,norm[n]);
      for ( i = 0 ; i < SDIM ; i++ )
        xsum[i] -= c*norm[n][i];
    }
  }
  for ( i = 0 ; i < SDIM ; i++ )
    vx[i] += 0.25*xsum[i]/total_weight;
  return 1;
}

/* set up the rings and per-facet data; 0: search everything as before */
static int av_prepare(vertex_id *list, long n, facet_id *facets, long nf, int mode)
{ vertex_id *corners;
  long j;
  if ( web.symmetry_flag || SDIM != 3 || FACET_VERTS != 3 || hessian_normal_one_flag
       || !(corners = fl_facet_corners()) )
    return 0;
  if ( !av_build(list,n) ) return 0;
  if ( mode != VOLKEEP ) return 1;
  if ( !ensure_size((void**)&av_norm,&av_norm_size,(long)web.skel[FACET].max_ord + 1,
                    sizeof(double[2][3])) )
    return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(nf))
#endif
  for ( j = 0 ; j < nf ; j++ )
  { long o = (long)ordinal(facets[j]);
    vertex_id *c = corners + 3*o;
    av_facet_normal(c[0],c[1],c[2],av_norm[o][0]);
    av_facet_normal(c[1],c[0],c[2],av_norm[o][1]);   /* inverted: corners 1, 0, 2 */
  }
  return 1;
}

static int fl_vertex_averages_impl(int mode, char *xbase, char *sbase, size_t stride)
{ vertex_id *list;
  facet_id *facets;
  long n, nf, k;
  int failed = 0, ring;
  if ( fast_mesh_disabled() || web.modeltype != LINEAR || web.representation != SOAPFILM
       || web.metric_flag || (everything_quantities_flag && !quantity_area_is_plain())
       || (!everything_quantities_flag && calc_facet_energy != facet_energy_l)
       || hessian_special_normal_flag || (web.symmetry_flag && !web.torus_flag) )
    return 0;
  if ( !(list = vertex_list(&n)) || n < FL_PARALLEL_MIN ) return 0;
  if ( !(facets = fl_facet_list(&nf)) ) return 0;
  if ( !ensure_size((void**)&av_area,&av_area_size,(long)web.skel[FACET].max_ord + 1,
                    sizeof(double)) )
    return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(nf))
#endif
  for ( k = 0 ; k < nf ; k++ )
  { REAL a = fl_facet_area(facets[k]);
    set_facet_area(facets[k],a);
    av_area[ordinal(facets[k])] = a;
  }
  ring = av_prepare(list,n,facets,nf,mode);
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic,256) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    long o = loc_ordinal(v_id);
    jmp_buf trap;
    if ( get_vattr(v_id) & (CONSTRAINT|BOUNDARY) ) continue;
    /* (fixed now but not when the rings were built, which only a topology
       change rebuilds: Evolver's own routine, which leaves it alone) */
    if ( ring && av_estart[k] < av_estart[k+1] && !(get_vattr(v_id) & (FIXED|AXIAL_POINT)) )
    { REAL *vx = (REAL*)(xbase + o*stride);
      int st = av_average(k,v_id,vx,mode);
      *(int*)(sbase + o*stride) = st;
      if ( fl_check() )   /* against Evolver's own, bit for bit */
      { REAL ref[MAXCOORD];
        int i, rst;
        va_active = 1;
        rst = find_vertex_average(v_id,ref,mode);
        va_active = 0;
        for ( i = 0 ; i < SDIM ; i++ )
          if ( rst != st || ref[i] != vx[i] )
          { fprintf(stderr,"vertex average ring mismatch at vertex %ld\n",(long)ordinal(v_id));
            abort();
          }
      }
      continue;
    }
    if ( setjmp(trap) == 0 )
    { fl_trap = &trap;   /* kb_error() comes back here */
      va_active = 1;
      *(int*)(sbase + o*stride) = find_vertex_average(v_id,(REAL*)(xbase + o*stride),mode);
      va_active = 0;
      fl_trap = NULL;
    }
    else
    { va_active = 0;
#ifdef _OPENMP
      #pragma omp atomic write
#endif
      failed = 1;
    }
  }
  if ( failed ) return 0;   /* nothing moved yet: the serial loop redoes it */
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    long o = loc_ordinal(v_id);
    if ( get_vattr(v_id) & (CONSTRAINT|BOUNDARY) )
      *(int*)(sbase + o*stride) = find_vertex_average(v_id,(REAL*)(xbase + o*stride),mode);
  }
  return 1;
}

/* calc_edge() over all edges, for linear models whose edge length is the
   plain one (no metric; with everything_quantities, the default
   edge_length instance, which sets the same length). Its total-area
   addends (quantity mode) are reset by the next energy calculation unused. */
static int fl_calc_edges_impl(void)
{ edge_id *list;
  long n, k;
  if ( fl_disabled() || threadflag || fast_mesh_disabled() ) return 0;
  if ( web.modeltype != LINEAR || web.metric_flag || klein_metric_flag ) return 0;
  if ( everything_quantities_flag )
  { struct method_instance *mi;
    int found = 0;
    if ( length_method_number <= 0 ) return 0;
    mi = METH_INSTANCE(length_method_number);
    if ( mi->type != EDGE || !(mi->flags & DEFAULT_INSTANCE)
         || basic_gen_methods[mi->gen_method].value != q_edge_tension_value )
      return 0;
    for ( k = 0 ; k < global_meth_inst_count[EDGE] ; k++ )
      if ( global_meth_inst[EDGE][k] == length_method_number ) found = 1;
    if ( !found ) return 0;
  }
  if ( !(list = type_list(EDGE,&n)) || n < FL_PARALLEL_MIN ) return 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { REAL s[MAXCOORD];
    get_edge_side(list[k],s);
    set_edge_length(list[k],sqrt(SDIM_dot(s,s)));
  }
  return 1;
}

/* test(e) for every edge in parallel, as marks by edge ordinal; NULL when
   not run (or on an error in a test) */
static unsigned char *em_marks = NULL;
static long em_marks_size = 0;

static unsigned char *fl_edge_marks_impl(int (*test)(edge_id))
{ edge_id *list;
  long n, k;
  int failed = 0;
  if ( fl_disabled() || threadflag || fast_mesh_disabled() ) return NULL;
  if ( !(list = type_list(EDGE,&n)) || n < FL_PARALLEL_MIN ) return NULL;
  if ( !ensure_size((void**)&em_marks,&em_marks_size,web.skel[EDGE].max_ord+1,1) )
    return NULL;
  memset(em_marks,0,web.skel[EDGE].max_ord+1);
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic,1024) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { jmp_buf trap;
    if ( setjmp(trap) == 0 )
    { fl_trap = &trap;
      em_marks[loc_ordinal(list[k])] = (unsigned char)((*test)(list[k]) != 0);
      fl_trap = NULL;
    }
    else
    {
#ifdef _OPENMP
      #pragma omp atomic write
#endif
      failed = 1;
    }
  }
  return failed ? NULL : em_marks;
}

unsigned char *fl_edge_marks(int (*test)(edge_id))
{ unsigned char *r; fl_enter(); r = fl_edge_marks_impl(test); fl_leave(); return r; }

int fl_vertex_averages(int mode, REAL *x0, int *status0, size_t stride)
{ int r; fl_enter();
  r = fl_vertex_averages_impl(mode,(char*)x0,(char*)status0,stride);
  fl_leave(); return r; }

int fl_calc_edges(void)
{ int r; fl_enter(); r = fl_calc_edges_impl(); fl_leave(); return r; }
