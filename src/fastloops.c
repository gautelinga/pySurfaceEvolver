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
{ long n, nb;
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
  if ( !ensure_size((void**)&vol_sums,&vol_sums_size,threads*nb,sizeof(csum))
       || !ensure_size((void**)&vol_abs,&vol_abs_size,threads*nb,sizeof(double)) )
    return 0;
  memset(vol_sums,0,threads*nb*sizeof(csum));
  memset(vol_abs,0,threads*nb*sizeof(double));

  /* each thread sums the signed volumes of its facets per body */
#ifdef _OPENMP
  #pragma omp parallel num_threads(threads)
#endif
  { csum *sums = vol_sums + THREAD_NUM()*nb;
    double *abss = vol_abs + THREAD_NUM()*nb;
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
    { REAL v = vol_sums[t*nb+ord].s + vol_sums[t*nb+ord].c;
      if ( v != 0.0 ) binary_tree_add(b->volume_addends,v);
      b->abstotal += vol_abs[t*nb+ord];
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

static void fl_sel_begin_impl(fl_sel *s, int type, ATTR bits, int site)
{ element_id *list;
  long n, k, hits = 0;
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
  s->n = 0;
  for ( k = 0 ; s->n < hits ; k++ )
    if ( sel_mark[k] ) sel_buf[site][s->n++] = list[k];
  s->list = sel_buf[site];
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
  long n, k, m = (long)fixcount*fixcount;
  int threads, t, deg = 0;
  if ( approx_curve_flag || fixcount > FL_LEFTSIDE_MAX ) return 0;
  if ( !(list = vertex_list(&n)) ) return 0;
  threads = loop_threads(n);
  if ( !ensure_size((void**)&ls_buf,&ls_buf_size,threads*m,sizeof(double)) )
    return 0;
  memset(ls_buf,0,threads*m*sizeof(double));
#ifdef _OPENMP
  #pragma omp parallel num_threads(threads) reduction(+:deg)
#endif
  { double *a = ls_buf + THREAD_NUM()*m;
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
      ls_buf[k] += ls_buf[t*m + k];
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

/* hessian_init()'s vertex normals, see fastloops.h. The normals only read
   the surface; an error (kb_error) leaves the vertex to the caller, which
   redoes it serially so Evolver reports it as usual. */
static int *fl_vertex_normals_impl(REAL ***v_normal, vertex_id **listp, long *np)
{ vertex_id *list;
  int *dims;
  long n, k;
  if ( web.representation == SIMPLEX || hessian_special_normal_flag
       || (web.symmetry_flag && !web.torus_flag) )
    return NULL;
  if ( !(list = vertex_list(&n)) || n < FL_PARALLEL_MIN ) return NULL;
  if ( !(dims = (int*)malloc(n*sizeof(int))) ) return NULL;
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
      kk = new_calc_vertex_normal(v_id,v_normal[k]);
      dims[k] = project_vertex_normals(v_id,v_normal[k],kk);
      fl_trap = NULL;
    }
    else dims[k] = -1;
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
   version is picked at run time where available (no FMA: same rounding) */
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
__attribute__((target_clones("avx2","default")))
#endif
static void lg_kernel(const double *packed, int stride, int rows, int ctrl,
                      double X[3][LG_MAXCTRL], double out[3][LG_MAXROWS])
{ int l, r;
  double *o0 = out[0], *o1 = out[1], *o2 = out[2];
  for ( r = 0 ; r < rows ; r++ ) o0[r] = o1[r] = o2[r] = 0.0;
  for ( l = 0 ; l < ctrl ; l++ )
  { const double *col = packed + (long)l*stride;
    double x0 = X[0][l], x1 = X[1][l], x2 = X[2][l];
    for ( r = 0 ; r < rows ; r++ )
    { double c = col[r];
      o0[r] += c*x0;
      o1[r] += c*x1;
      o2[r] += c*x2;
    }
  }
}

int fl_lagrange_facet_setup(struct gauss_lag *gl, int dim, int ctrl,
                            REAL **x, REAL **gauss_pt, REAL ***sides, int need_side)
{ int rows, l, g, d, k;
  double X[3][LG_MAXCTRL];
  double out[3][LG_MAXROWS];

  if ( fl_disabled() || SDIM != 3 ) return 0;
  rows = gl->gnumpts*(1 + dim);
  if ( rows > LG_MAXROWS || ctrl > LG_MAXCTRL || ctrl != gl->lagpts ) return 0;

  if ( lg_gpoly != gl->gpoly || lg_ctrl != ctrl || lg_gnumpts != gl->gnumpts
       || lg_dim != dim || lg_version != fl_lagrange_tables_version )
  { if ( !ensure_size((void**)&lg_packed,&lg_packed_size,(long)ctrl*rows,
                      sizeof(double)) )
      return 0;
    for ( l = 0 ; l < ctrl ; l++ )
    { double *col = lg_packed + (long)l*rows;
      for ( g = 0 ; g < gl->gnumpts ; g++ )
      { col[g] = gl->gpoly[g][l];
        for ( d = 0 ; d < dim ; d++ )
          col[gl->gnumpts + g*dim + d] = gl->gpolypart[g][d][l];
      }
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
  lg_kernel(lg_packed,lg_gnumpts*(1 + dim),rows,ctrl,X,out);
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
 * computes missing facet areas and stores them (facet_energy_l(AREA_ONLY));
 * here the areas are computed alike, the facets marked, and the areas stored
 * after the loop. Positions don't change in between, so the results are
 * those of the serial loop.
 */

int find_vertex_average(vertex_id v_id, REAL *vx, int mode);   /* veravg.c */

static int fast_mesh_disabled(void)   /* PYSE_NO_FAST_MESH=1: these two off */
{ static int disabled = -1;
  if ( disabled < 0 ) disabled = getenv("PYSE_NO_FAST_MESH") != NULL;
  return disabled;
}

static unsigned char *va_mark = NULL;    /* per facet ordinal */
static long va_mark_size = 0;
static __thread int va_active = 0;

/* facet_energy_l(f_id,AREA_ONLY)'s area, not stored */
static REAL linear_facet_area(facet_id f_id)
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

REAL fl_lazy_facet_area(facet_id f_id)
{ long o = loc_ordinal(f_id);
#ifdef _OPENMP
  #pragma omp atomic write
#endif
  va_mark[o] = 1;
  return linear_facet_area(f_id);
}

static int fl_vertex_averages_impl(int mode, char *xbase, char *sbase, size_t stride)
{ vertex_id *list;
  facet_id *facets;
  long n, nf, k;
  int failed = 0;
  if ( fast_mesh_disabled() || web.modeltype != LINEAR || web.representation != SOAPFILM
       || web.metric_flag || calc_facet_energy != facet_energy_l || hessian_special_normal_flag
       || (web.symmetry_flag && !web.torus_flag) )
    return 0;
  if ( !(list = vertex_list(&n)) || n < FL_PARALLEL_MIN ) return 0;
  if ( !(facets = fl_facet_list(&nf)) ) return 0;
  if ( !ensure_size((void**)&va_mark,&va_mark_size,web.skel[FACET].max_ord+1,1) )
    return 0;
  memset(va_mark,0,web.skel[FACET].max_ord+1);
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic,256) num_threads(loop_threads(n))
#endif
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    long o = loc_ordinal(v_id);
    jmp_buf trap;
    if ( get_vattr(v_id) & (CONSTRAINT|BOUNDARY) ) continue;
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
  if ( failed ) return 0;   /* nothing stored yet: the serial loop redoes it */
  for ( k = 0 ; k < n ; k++ )
  { vertex_id v_id = list[k];
    long o = loc_ordinal(v_id);
    if ( get_vattr(v_id) & (CONSTRAINT|BOUNDARY) )
      *(int*)(sbase + o*stride) = find_vertex_average(v_id,(REAL*)(xbase + o*stride),mode);
  }
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(loop_threads(nf))
#endif
  for ( k = 0 ; k < nf ; k++ )
    if ( va_mark[loc_ordinal(facets[k])] && get_facet_area(facets[k]) == 0.0 )
      set_facet_area(facets[k],linear_facet_area(facets[k]));
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
