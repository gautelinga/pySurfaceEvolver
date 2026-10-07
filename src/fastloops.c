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

int fl_film_grad(void)
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
