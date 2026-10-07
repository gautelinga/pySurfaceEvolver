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
static int fl_disabled(void)
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
 * Threads
 */

static int fl_thread_count = 0;   /* 0: OpenMP's default */

int fl_threads(void)
{
#ifdef _OPENMP
  return fl_thread_count > 0 ? fl_thread_count : omp_get_max_threads();
#else
  return 1;
#endif
}

void fl_set_threads(int n)
{ fl_thread_count = n > 0 ? n : 0;
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

int fl_facet_volumes(void)
{ long n, nb;
  facet_id *list;
  vertex_id *corners;
  body_id b_id;
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
  threads = loop_threads(n);
  nb = (long)web.skel[BODY].max_ord + 1;
  if ( !ensure_size((void**)&vol_sums,&vol_sums_size,threads*nb,sizeof(csum))
       || !ensure_size((void**)&vol_abs,&vol_abs_size,threads*nb,sizeof(double)) )
    return 0;
  memset(vol_sums,0,threads*nb*sizeof(csum));
  memset(vol_abs,0,threads*nb*sizeof(double));

  /* each thread sums the signed volumes of its facets per body. Bodies
     are looked up per call (they can change without a topology change). */
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
      body_id b0, b1;
      if ( get_fattr(f_id) & NONCONTENT ) continue;
      b0 = get_facet_body(f_id);
      b1 = get_facet_body(facet_inverse(f_id));
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

int fl_facet_energies(void)
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

int fl_facet_forces(void)
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
