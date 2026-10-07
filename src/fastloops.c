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
 * The parallel loops use the cache in two passes: per-facet values are
 * computed in parallel (pure arithmetic on coordinates, no globals, no
 * expression evaluation), then accumulated serially in facet order, so
 * the sums are exactly those of the original serial loops.
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
      && fv_cache_maxord == (long)web.skel[FACET].max_ord;
}

static int build_cache(void)
{ facet_id f_id;
  long k = 0;
  if ( web.representation != SOAPFILM ) return 0;
  if ( !ensure_size((void**)&fv_cache,&fv_cache_size,
                    3*((long)web.skel[FACET].max_ord + 1),sizeof(vertex_id))
       || !ensure_size((void**)&fv_list,&fv_list_size,
                       web.skel[FACET].count + 1,sizeof(facet_id)) )
    return 0;
  FOR_ALL_FACETS(f_id)
  { facetedge_id fe = get_facet_fe(f_id);
    vertex_id *c = fv_cache + 3*ordinal(f_id);
    c[0] = get_fe_tailv(fe); fe = get_next_edge(fe);
    c[1] = get_fe_tailv(fe); fe = get_next_edge(fe);
    c[2] = get_fe_tailv(fe);
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

/**************************************************************************
 * Facet volumes
 */

static double *vol_buf = NULL;
static long vol_buf_size = 0;
static body_id *body_buf = NULL;   /* front and back body per facet */
static long body_buf_size = 0;

int fl_facet_volumes(void)
{ long n, k;
  facet_id *list;
  vertex_id *corners;
  int threads;

  if ( fl_disabled() ) return 0;
  if ( web.representation != SOAPFILM || web.torus_flag || web.symmetry_flag
       || web.symmetric_content || web.modeltype != LINEAR || SDIM != 3
       || calc_facet_volume != facet_volume_l || threadflag )
    return 0;
  corners = fl_facet_corners();
  list = fl_facet_list(&n);
  if ( !corners || !list ) return 0;
  if ( fl_check() ) check_list(list,n);
  if ( !ensure_size((void**)&vol_buf,&vol_buf_size,n,sizeof(double))
       || !ensure_size((void**)&body_buf,&body_buf_size,2*n,sizeof(body_id)) )
    return 0;
  threads = fl_threads();
  (void)threads;

  /* pass 1: each facet's bodies and signed volume contribution. Bodies
     are recorded per call (they can change without a topology change). */
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads) if(n >= FL_PARALLEL_MIN)
#endif
  for ( k = 0 ; k < n ; k++ )
  { facet_id f_id = list[k];
    vertex_id *c = corners + 3*ordinal(f_id);
    REAL *x0, *x1, *x2;
    body_id b0 = NULLID, b1 = NULLID;
    if ( !(get_fattr(f_id) & NONCONTENT) )
    { b0 = get_facet_body(f_id);
      b1 = get_facet_body(facet_inverse(f_id));
    }
    body_buf[2*k] = b0;
    body_buf[2*k+1] = b1;
    if ( !valid_id(b0) && !valid_id(b1) ) continue;
    x0 = get_coord(c[0]); x1 = get_coord(c[1]); x2 = get_coord(c[2]);
    /* same expression as facet_volume_l(), for identical rounding */
    vol_buf[k] = (x0[2]+x1[2]+x2[2])/6*
       ((x1[0]-x0[0])*(x2[1]-x0[1])-(x1[1]-x0[1])*(x2[0]-x0[0]));
  }

  /* pass 2: accumulate in facet order, as facet_volume_l() does */
  for ( k = 0 ; k < n ; k++ )
  { body_id b0 = body_buf[2*k], b1 = body_buf[2*k+1];
    if ( valid_id(b0) ) add_body_volume(b0,vol_buf[k]);
    if ( valid_id(b1) ) add_body_volume(b1,-vol_buf[k]);
  }
  return 1;
}

/**************************************************************************
 * Facet energies
 */

static double *area_buf = NULL;
static long area_buf_size = 0;
static double *energy_buf = NULL;
static long energy_buf_size = 0;

/* SDIM_dot for SDIM 3, summed in the same order as dot() */
#define DOT3(a,b) (((a)[0]*(b)[0] + (a)[1]*(b)[1]) + (a)[2]*(b)[2])

int fl_facet_energies(void)
{ long n, k;
  facet_id *list;
  vertex_id *corners;
  int threads;
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
  if ( !ensure_size((void**)&area_buf,&area_buf_size,n,sizeof(double))
       || !ensure_size((void**)&energy_buf,&energy_buf_size,n,sizeof(double)) )
    return 0;
  threads = fl_threads();
  (void)threads;

  /* pass 1: each facet's area and energy, as facet_energy_l() computes them */
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads) if(n >= FL_PARALLEL_MIN)
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
    area_buf[k] = energy;
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
    energy_buf[k] = energy;
  }

  /* pass 2: the sums, in facet order */
  for ( k = 0 ; k < n ; k++ )
  { binary_tree_add(web.total_area_addends,area_buf[k]);
    binary_tree_add(web.total_energy_addends,energy_buf[k]);
  }
  return 1;
}

/**************************************************************************
 * Facet forces
 */

static double *force_buf = NULL;     /* 9 per facet: 3 vertices x 3 */
static long force_buf_size = 0;
static unsigned char *wee_buf = NULL;
static long wee_buf_size = 0;

int fl_facet_forces(void)
{ long n, k;
  facet_id *list;
  vertex_id *corners;
  int threads;
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
  if ( !ensure_size((void**)&force_buf,&force_buf_size,9*n,sizeof(double))
       || !ensure_size((void**)&wee_buf,&wee_buf_size,n,1) )
    return 0;
  threads = fl_threads();
  (void)threads;

  /* pass 1: each facet's forces on its three vertices, with the same
     operations, in the same order, as facet_force_l() */
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads) if(n >= FL_PARALLEL_MIN)
#endif
  for ( k = 0 ; k < n ; k++ )
  { facet_id f_id = list[k];
    vertex_id *c = corners + 3*ordinal(f_id);
    REAL *x[3];
    REAL side[3][3], z[3], f[3][3], ss, st, tt, area;
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
    wee_buf[k] = area <= wee_area;
    if ( wee_buf[k] ) continue;
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
    memcpy(force_buf + 9*k,f,sizeof(f));
  }

  /* pass 2: add to the vertices in facet order; zero-area warnings in the
     same order as facet_force_l() would print them */
  for ( k = 0 ; k < n ; k++ )
  { facet_id f_id = list[k];
    vertex_id *c = corners + 3*ordinal(f_id);
    int i, j;
    if ( wee_buf[k] )
    { sprintf(errmsg,"WARNING! Zero area for facet %s.\n",ELNAME(f_id));
      outstring(errmsg);
      continue;
    }
    for ( i = 0 ; i < 3 ; i++ )
    { REAL *force = get_force(c[i]);
      for ( j = 0 ; j < 3 ; j++ )
        force[j] += force_buf[9*k + 3*i + j];
    }
  }
  return 1;
}
