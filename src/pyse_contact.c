/*
 * pyse_contact.c -- spheres' wetted areas and volumes from their contact
 * lines alone (pySE's gauge="lines" in a periodic slab).
 *
 * The wetted region of a sphere (centre c, radius R) bounded by geodesic
 * polygons through the contact-line vertices has, per loop taken with the
 * wetted region on its left (seen from outside the sphere), the area of the
 * fan of geodesic triangles from a point P to its edges, R^2 times their
 * signed solid angles (Van Oosterom and Strackee's), up to whole spheres: P
 * any point, chosen each pass per loop as far as can be from the loop's
 * antipodes (where a triangle's angle is undefined), the sum and its
 * derivatives the same for any. A region bounded by several loops is the
 * sum of theirs less whole spheres: with S the sum over a body's loops on a
 * sphere, its wetted area is S - 4 pi R^2 n, n = floor(S / 4 pi R^2) (exact
 * unless the region is empty or the whole sphere, which a loop can't bound).
 * Each contact edge carries its own triangle, so the method is an edge
 * method on the edge's two vertices. (Until 2026-10-30 the area was R^2 (2 pi
 * - the turning angles at the vertices) per loop, Gauss-Bonnet: where the
 * line doubled back on itself, a turn of pi, the angle's sign was round-off,
 * and a move of 3e-7 changed a pack's wetted area by 2 pi R^2, its volume by
 * half a bead; the fan's angles have no such point but its antipode.)
 *
 * With the bodies' volume a facet integral of (0, 0, z - z_f) (divergence 1,
 * zero on a floor z = z_f), the wetted part of sphere i adds
 * -R A / 3 + the line integral of B = (wy wz/3 + h wy/2, -wx wz/3 - h wx/2, 0)
 * (w = x - c, h = c_z - z_f) along its loops: curl B = (x - c)/3 - (0, 0, z - z_f),
 * so the divergence-free difference of the two volume fields goes through
 * the lines. The wetting energy is -k A, k = tension cos(angle).
 *
 * A bead with no contact line of a body is inside it or not: a vertical
 * ray's crossings with the body's facets and the cones from each sphere's
 * centre to its contact lines (which close the body without the wetted
 * caps; the cones lie inside the spheres) decide, once per change of the
 * lines' topology. Its whole sphere then goes to the body's volume and the
 * energy, shared out over the body's contact edges (and the energy's over
 * all of them).
 *
 * Nothing here is local to a pass's elements but the constants, which
 * change only where the lines' topology does; Evolver's constraint energy
 * and per-body content integrals of the registered spheres are switched to
 * this method (pyse_contact_adopt), keeping how they are applied to edges.
 * Linear model only.
 */

#include "include.h"
#include "fastloops.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__)
extern int pyse_slab(REAL *L) __attribute__((weak));
extern long pyse_load_serial(void) __attribute__((weak));
#define PC_SLAB(L) (pyse_slab ? pyse_slab(L) : 0)
#define PC_LOAD() (pyse_load_serial ? pyse_load_serial() : 0)
#else
#define PC_SLAB(L) 0
#define PC_LOAD() 0
#endif

#define PC_FLOOR_TOP 20.0          /* constraints.ANGLE_FLOOR_TOP */

struct pc_sphere { int con; REAL c[3], R, angle, tension, zf; };

static struct pc_sphere *sph = NULL;
static int nsph = 0, maxsph = 0;
static int *sph_of_con = NULL;     /* constraint number -> sphere index + 1 */
static int sph_of_con_n = 0;
static int evi_method = -1, pc_method = -1;

/* a contact edge of a body on a sphere, wetted region on its left along
   dir (+1: tail to head of the positive edge) */
struct pc_entry
{ edge_id e;
  int s, b;                 /* sphere index, body ordinal */
  int dir, sapp;            /* direction; the sign Evolver applies the body's content with */
  vertex_id A, B;           /* the line's direction: A -> B */
  int prev, next;           /* entries before and after along the line, -1 if none */
  int group;                /* the (sphere, body)'s group */
  int other;                /* another entry of the same edge, or -1 */
  int loop;                 /* its loop, -1 in a group with a branch or an open line */
  REAL om;                  /* its fan triangle's solid angle (this pass) */
};

struct pc_group { int s, b, first, count, loops, bad; REAL S, K; };  /* K: the whole spheres' constant area */
struct pc_loop { int g; REAL P[3], near[7]; };   /* P: the fan's apex (unit, from the centre) */

static struct pc_entry *ent = NULL;
static int nent = 0, maxent = 0;
static struct pc_group *grp = NULL;
static int ngrp = 0, maxgrp = 0;
static struct pc_loop *lp = NULL;
static int nlp = 0, maxlp = 0;
static int *slot = NULL;            /* edge ordinal -> first entry, -1 */
static int nslot = 0;
static int pc_bad = 0;
static int pc_geom_done = 0;         /* angles and constants since the last pyse_contact_refresh() */
static int pc_stale = 1;             /* the table to be built at the next init */
static long pc_stamp[6];            /* the elements, load and topology it was built for */
extern long pyse_edge_conmap_stamp;  /* (skeleton.c: an edge's constraints changed) */
static unsigned long long topo_key = 0;
static int *imm_body = NULL;        /* per sphere: the body (ordinal) it is inside, -1 */
static int nimm_alloc = 0;
static int *imm_count = NULL;       /* per body ordinal (for pyse_contact_status) */
static int nbody_alloc = 0;
static REAL pc_L[2];
static int pc_slab = 0;

/* ---- registration ---- */

static int method_index(const char *name)
{ int k;
  for ( k = 0 ; basic_gen_methods[k].name[0] ; k++ )
    if ( strcmp(basic_gen_methods[k].name,name) == 0 ) return k;
  return -1;
}

void pyse_contact_reset(void)
{ int k;
  if ( evi_method < 0 ) evi_method = method_index("edge_vector_integral");
  if ( pc_method < 0 ) pc_method = method_index("pyse_contact_integral");
  /* back to Evolver's integrals */
  for ( k = LOW_INST ; k < meth_inst_count ; k++ )
  { struct method_instance *mi = METH_INSTANCE(k);
    if ( !(mi->flags & Q_DELETED) && mi->gen_method == pc_method )
      mi->gen_method = evi_method;
  }
  nsph = 0;
  if ( sph_of_con ) memset(sph_of_con,0,sph_of_con_n*sizeof(int));
  pc_stale = 1;
  topo_key = 0;
}

int pyse_contact_sphere(int con, const REAL *c, REAL R, REAL angle, REAL tension, REAL zf)
{ int k;
  if ( con < 0 || R <= 0 ) return 0;
  if ( con >= sph_of_con_n )
  { int n = con + 64;
    int *m = (int*)realloc(sph_of_con,n*sizeof(int));
    if ( !m ) return 0;
    memset(m + sph_of_con_n,0,(n - sph_of_con_n)*sizeof(int));
    sph_of_con = m; sph_of_con_n = n;
  }
  if ( nsph == maxsph )
  { int n = maxsph ? 2*maxsph : 64;
    struct pc_sphere *p = (struct pc_sphere *)realloc(sph,n*sizeof(*p));
    if ( !p ) return 0;
    sph = p; maxsph = n;
  }
  k = sph_of_con[con] ? sph_of_con[con] - 1 : nsph++;
  sph[k].con = con;
  for ( int i = 0 ; i < 3 ; i++ ) sph[k].c[i] = c[i];
  sph[k].R = R; sph[k].angle = angle; sph[k].tension = tension; sph[k].zf = zf;
  sph_of_con[con] = k + 1;
  topo_key = 0;
  return 1;
}

int pyse_contact_active(void) { return nsph > 0; }

static int sphere_of(int con)
{ con &= CONMASK;
  return (con < sph_of_con_n && sph_of_con[con]) ? sph_of_con[con] - 1 : -1;
}

/* an edge-integral instance of a registered sphere switched to this method */
void pyse_contact_adopt(int meth)
{ struct method_instance *mi;
  if ( !nsph || meth <= 0 || meth >= meth_inst_count ) return;
  if ( evi_method < 0 ) evi_method = method_index("edge_vector_integral");
  if ( pc_method < 0 ) pc_method = method_index("pyse_contact_integral");
  mi = METH_INSTANCE(meth);
  if ( mi->gen_method == evi_method && sphere_of(mi->connum) >= 0 )
    mi->gen_method = pc_method;
}

int pyse_contact_adopt_all(void)
{ int k, n = 0;
  if ( web.modeltype != LINEAR ) return -1;
  for ( k = LOW_INST ; k < meth_inst_count ; k++ )
  { struct method_instance *mi = METH_INSTANCE(k);
    if ( mi->flags & Q_DELETED ) continue;
    pyse_contact_adopt(k);
    if ( mi->gen_method == pc_method ) n++;
  }
  pc_stale = 1;
  return n;
}

/* ---- geometry ---- */

static void pc_delta(const REAL *c, const REAL *x, REAL *w)
{ int i;
  for ( i = 0 ; i < 3 ; i++ ) w[i] = x[i] - c[i];
  if ( pc_slab )
    for ( i = 0 ; i < 2 ; i++ ) w[i] -= pc_L[i]*floor(w[i]/pc_L[i] + 0.5);
}

/* the signed solid angle of the geodesic triangle (p, a, b), p a unit
   vector, a and b positions from the centre: positive where a -> b runs
   left about p seen from outside. tan(omega/2) = det(p, a, b)/(|a||b| +
   (p.a)|b| + (a.b) + (p.b)|a|), as for unit a, b times |a||b| */
static REAL omega(const REAL *p, const REAL *a, const REAL *b)
{ REAL la = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
  REAL lb = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  REAL det = p[0]*(a[1]*b[2] - a[2]*b[1]) - p[1]*(a[0]*b[2] - a[2]*b[0])
           + p[2]*(a[0]*b[1] - a[1]*b[0]);
  REAL den = la*lb + (p[0]*a[0] + p[1]*a[1] + p[2]*a[2])*lb
           + (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) + (p[0]*b[0] + p[1]*b[1] + p[2]*b[2])*la;
  return 2*atan2(det,den);
}

/* the angle floor's global looked up again only outside the threads (a load's
   first pass: pc_current() does it before the threads read it; two threads
   writing it, one read the new load with the old index, a floor of 0) */
static REAL angle_floor(void)
{ static long load = -2;
  static int k = -1;
#ifdef _OPENMP
  if ( !omp_in_parallel() )
#endif
  if ( load != PC_LOAD() ) { k = lookup_global("pyse_angle_floor"); load = PC_LOAD(); }
  return k >= 0 ? globals(k)->value.real : 0.0;
}

static REAL sphere_k(const struct pc_sphere *p)
{ REAL angle = p->angle, fl = angle_floor();
  if ( angle > 0 && angle < PC_FLOOR_TOP && fl > angle ) angle = fl;
  return p->tension*cos(angle*M_PI/180.0);
}

/* ---- the lines' table ---- */

static int ent_add(void)
{ if ( nent == maxent )
  { int n = maxent ? 2*maxent : 1024;
    struct pc_entry *p = (struct pc_entry *)realloc(ent,n*sizeof(*p));
    if ( !p ) return -1;
    ent = p; maxent = n;
  }
  return nent++;
}

static int cmp_key(const void *a, const void *b)
{ const struct pc_entry *x = ent + *(const int *)a, *y = ent + *(const int *)b;
  if ( x->s != y->s ) return x->s < y->s ? -1 : 1;
  if ( x->b != y->b ) return x->b < y->b ? -1 : 1;
  return (*(const int *)a < *(const int *)b) ? -1 : 1;
}

static vertex_id *sort_v = NULL;
static int cmp_by_vertex(const void *a, const void *b)
{ vertex_id x = sort_v[*(const int *)a], y = sort_v[*(const int *)b];
  return x < y ? -1 : x > y ? 1 : 0;
}

static unsigned long long mix(unsigned long long h, unsigned long long x)
{ h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

/* whether a vertical ray up from p crosses triangle T */
static int pc_cross(REAL T[3][3], const REAL *p)
{ REAL d0, d1, d2, den, z;
  REAL ax = T[0][0] - p[0], ay = T[0][1] - p[1], bx = T[1][0] - p[0], by = T[1][1] - p[1],
       cx = T[2][0] - p[0], cy = T[2][1] - p[1];
  d0 = ax*by - ay*bx; d1 = bx*cy - by*cx; d2 = cx*ay - cy*ax;
  if ( !((d0 > 0 && d1 > 0 && d2 > 0) || (d0 < 0 && d1 < 0 && d2 < 0)) ) return 0;
  den = d0 + d1 + d2;
  z = (d1*T[0][2] + d2*T[1][2] + d0*T[2][2])/den;
  return z > p[2];
}

/* the beads with no line (has[s] 0): each inside the first body (in the
   bodies' order) a vertical ray from its centre crosses an odd number of
   times, the body's facets and the cones over its contact lines; the facets'
   shapes taken once, the beads over the threads */
struct pc_tri { const REAL *x0; REAL w1[3], w2[3]; int b0, b1; };

static void pc_immersed(int nb, const char *has)
{ facet_id f_id;
  struct pc_tri *tri;
  int *cand;
  char *odd;
  int ntri = 0, maxtri = (int)web.skel[FACET].count + 16, ncand = 0, i, s;
  int threads = fl_threads();
  (void)threads;
  tri = (struct pc_tri *)malloc(maxtri*sizeof(*tri));
  cand = (int*)malloc((nsph ? nsph : 1)*sizeof(int));
  if ( !tri || !cand ) { free(tri); free(cand); return; }
  for ( s = 0 ; s < nsph ; s++ ) if ( !has[s] ) cand[ncand++] = s;
  odd = (char*)calloc((size_t)(ncand ? ncand : 1)*nb,1);
  if ( !odd ) { free(tri); free(cand); return; }
  FOR_ALL_FACETS(f_id)
  { body_id fb = get_facet_body(f_id), bb = get_facet_body(inverse_id(f_id));
    facetedge_id fe;
    struct pc_tri *t;
    int b0 = valid_id(fb) ? ordinal(fb) : -1, b1 = valid_id(bb) ? ordinal(bb) : -1;
    if ( b0 >= nb ) b0 = -1;
    if ( b1 >= nb || b1 == b0 ) b1 = -1;
    if ( b0 < 0 && b1 < 0 ) continue;
    if ( ntri == maxtri )
    { struct pc_tri *m = (struct pc_tri *)realloc(tri,2*maxtri*sizeof(*tri));
      if ( !m ) { free(tri); free(cand); free(odd); return; }
      tri = m; maxtri *= 2;
    }
    t = tri + ntri++;
    t->b0 = b0; t->b1 = b1;
    fe = get_facet_fe(f_id);
    t->x0 = get_coord(get_fe_tailv(fe));
    fe = get_next_edge(fe);
    pc_delta(t->x0,get_coord(get_fe_tailv(fe)),t->w1);
    fe = get_next_edge(fe);
    pc_delta(t->x0,get_coord(get_fe_tailv(fe)),t->w2);
  }
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic,1) num_threads(threads) if ( threads > 1 && ncand > 1 )
#endif
  for ( i = 0 ; i < ncand ; i++ )
  { char *o = odd + (size_t)i*nb;
    const REAL *c0 = sph[cand[i]].c;
    REAL p[3], T[3][3], w[3];
    int j, k;
    /* (off any vertex or edge's projection: the ray nudged) */
    p[0] = c0[0] + 1.2345678e-7; p[1] = c0[1] + 2.3456789e-7; p[2] = c0[2];
    for ( j = 0 ; j < ntri ; j++ )
    { const struct pc_tri *t = tri + j;
      pc_delta(p,t->x0,w);
      for ( k = 0 ; k < 3 ; k++ )
      { T[0][k] = p[k] + w[k]; T[1][k] = T[0][k] + t->w1[k]; T[2][k] = T[0][k] + t->w2[k]; }
      if ( pc_cross(T,p) )
      { if ( t->b0 >= 0 ) o[t->b0] ^= 1;
        if ( t->b1 >= 0 ) o[t->b1] ^= 1;
      }
    }
    for ( j = 0 ; j < nent ; j++ )
    { const struct pc_entry *q = ent + j;
      const REAL *c = sph[q->s].c;
      if ( q->b < 0 || q->b >= nb ) continue;
      pc_delta(p,c,w);
      for ( k = 0 ; k < 3 ; k++ ) T[0][k] = p[k] + w[k];
      pc_delta(c,get_coord(q->A),w);
      for ( k = 0 ; k < 3 ; k++ ) T[1][k] = T[0][k] + w[k];
      pc_delta(c,get_coord(q->B),w);
      for ( k = 0 ; k < 3 ; k++ ) T[2][k] = T[0][k] + w[k];
      if ( pc_cross(T,p) ) o[q->b] ^= 1;
    }
  }
  { body_id b_id;
    FOR_ALL_BODIES(b_id)
    { int b = ordinal(b_id);
      if ( b >= nb ) continue;
      for ( i = 0 ; i < ncand ; i++ )
        if ( imm_body[cand[i]] < 0 && odd[(size_t)i*nb + b] )
        { imm_body[cand[i]] = b; imm_count[b]++; }
    }
  }
  free(tri); free(cand); free(odd);
}

static int bodies_alloc(int n)
{ if ( nsph > nimm_alloc )
  { imm_body = (int*)realloc(imm_body,nsph*sizeof(int));
    if ( !imm_body ) { nimm_alloc = 0; return 0; }
    nimm_alloc = nsph;
  }
  if ( n <= nbody_alloc ) return 1;
  imm_count = (int*)realloc(imm_count,n*sizeof(int));
  if ( !imm_count ) { nbody_alloc = 0; return 0; }
  nbody_alloc = n;
  return 1;
}

/* the loops' apexes, the fan triangles' angles, each group's area, n and
   the whole spheres' constant: every pass (the coordinates move; the
   table's topology is kept) */
static REAL *pc_gx = NULL;           /* the entries' A and B as the angles were last taken at */
static long pc_gx_size = 0;
static int pc_gx_valid = 0;          /* (0: the table was rebuilt since) */
static REAL *pc_ax = NULL;           /* per entry: A from the centre, its length, its 1 + P.a/|a| */
static long pc_ax_size = 0;

static void pc_geometry(void)
{ int i, g, same;
  int threads = fl_threads();
  (void)threads;
  /* (a gradient step's second pass at the coordinates of its first: the
     angles and constants as they are) */
  if ( 6L*nent > pc_gx_size )
  { REAL *p = (REAL*)realloc(pc_gx,(6L*nent + 1024)*sizeof(REAL));
    if ( !p ) { free(pc_gx); pc_gx = NULL; pc_gx_size = 0; pc_gx_valid = 0; }
    else { pc_gx = p; pc_gx_size = 6L*nent + 1024; pc_gx_valid = 0; }
  }
  same = pc_gx_valid && pc_gx;
  if ( pc_gx )
  { int moved = 0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:moved) if ( threads > 1 && nent >= 512 )
#endif
    for ( i = 0 ; i < nent ; i++ )
    { REAL *x = pc_gx + 6L*i;
      const REAL *a = get_coord(ent[i].A), *b = get_coord(ent[i].B);
      if ( memcmp(x,a,3*sizeof(REAL)) || memcmp(x + 3,b,3*sizeof(REAL)) )
      { moved = 1;
        memcpy(x,a,3*sizeof(REAL)); memcpy(x + 3,b,3*sizeof(REAL));
      }
    }
    if ( moved ) same = 0;
  }
  pc_gx_valid = pc_gx != NULL;
  if ( same ) { pc_geom_done = 1; return; }
  if ( 11L*nent > pc_ax_size )
  { REAL *p = (REAL*)realloc(pc_ax,(11L*nent + 1024)*sizeof(REAL));
    if ( p ) { pc_ax = p; pc_ax_size = 11L*nent + 1024; }
  }
  if ( 11L*nent > pc_ax_size )   /* (out of memory: the apexes up the z axis) */
  { int l;
    for ( l = 0 ; l < nlp ; l++ ) { lp[l].P[0] = lp[l].P[1] = 0.0; lp[l].P[2] = 1.0; }
  }
  else
  /* each loop's apex: of the loop's mean direction and the six axes', the
     one whose antipode is furthest from its vertices (the least 1 + P.a/|a|
     over them greatest); each entry's a and |a|, then its seven 1 + P.a/|a|,
     on the threads, the sums and least values per loop in entry order */
  { static const REAL axes[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    int l, k;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads) if ( threads > 1 && nent >= 512 )
#endif
    for ( i = 0 ; i < nent ; i++ )
    { REAL *a = pc_ax + 11L*i;
      if ( ent[i].loop < 0 ) continue;
      pc_delta(sph[ent[i].s].c,get_coord(ent[i].A),a);
      a[3] = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    }
    for ( l = 0 ; l < nlp ; l++ )
    { lp[l].P[0] = lp[l].P[1] = lp[l].P[2] = 0.0;
      for ( k = 0 ; k < 7 ; k++ ) lp[l].near[k] = 2.0;
    }
    for ( i = 0 ; i < nent ; i++ )
    { const REAL *a = pc_ax + 11L*i;
      if ( ent[i].loop < 0 || a[3] <= 0 ) continue;
      for ( k = 0 ; k < 3 ; k++ ) lp[ent[i].loop].P[k] += a[k]/a[3];
    }
    for ( l = 0 ; l < nlp ; l++ )
    { REAL m = sqrt(lp[l].P[0]*lp[l].P[0] + lp[l].P[1]*lp[l].P[1] + lp[l].P[2]*lp[l].P[2]);
      if ( m > 0 ) for ( k = 0 ; k < 3 ; k++ ) lp[l].P[k] /= m;
      else lp[l].near[6] = -1.0;            /* (no mean direction: an axis) */
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads) private(k) if ( threads > 1 && nent >= 512 )
#endif
    for ( i = 0 ; i < nent ; i++ )
    { REAL *a = pc_ax + 11L*i, la = a[3];
      const struct pc_loop *L;
      if ( ent[i].loop < 0 || la <= 0 ) continue;
      L = lp + ent[i].loop;
      for ( k = 0 ; k < 6 ; k++ )
        a[4 + k] = 1.0 + (axes[k][0]*a[0] + axes[k][1]*a[1] + axes[k][2]*a[2])/la;
      if ( L->near[6] >= 0 )
        a[10] = 1.0 + (L->P[0]*a[0] + L->P[1]*a[1] + L->P[2]*a[2])/la;
    }
    for ( i = 0 ; i < nent ; i++ )
    { const REAL *a = pc_ax + 11L*i;
      struct pc_loop *L;
      if ( ent[i].loop < 0 || a[3] <= 0 ) continue;
      L = lp + ent[i].loop;
      for ( k = 0 ; k < 6 ; k++ ) if ( a[4 + k] < L->near[k] ) L->near[k] = a[4 + k];
      if ( L->near[6] >= 0 && a[10] < L->near[6] ) L->near[6] = a[10];
    }
    for ( l = 0 ; l < nlp ; l++ )
    { int best = 6;
      for ( k = 0 ; k < 6 ; k++ ) if ( lp[l].near[k] > lp[l].near[best] ) best = k;
      if ( best < 6 ) for ( k = 0 ; k < 3 ; k++ ) lp[l].P[k] = axes[best][k];
    }
  }
  /* (the triangles' angles on the threads, twice a gradient step; summed in order) */
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads) if ( threads > 1 && nent >= 512 )
#endif
  for ( i = 0 ; i < nent ; i++ )
  { struct pc_entry *q = ent + i;
    REAL a[3], b[3];
    const REAL *c = sph[q->s].c;
    if ( q->loop < 0 ) { q->om = 0.0; continue; }
    pc_delta(c,get_coord(q->A),a);
    pc_delta(c,get_coord(q->B),b);
    q->om = omega(lp[q->loop].P,a,b);
  }
  for ( g = 0 ; g < ngrp ; g++ ) grp[g].S = 0.0;
  for ( i = 0 ; i < nent ; i++ ) grp[ent[i].group].S += ent[i].om;
  for ( g = 0 ; g < ngrp ; g++ )
  { struct pc_group *G = grp + g;
    REAL R = sph[G->s].R, S, whole = 4*M_PI*R*R, n;
    S = R*R*G->S;
    n = G->bad ? 0.0 : floor(S/whole);
    G->S = S - n*whole;
    G->K = G->bad ? 0.0 : -n*whole;
  }
  pc_geom_done = 1;
}

/* the table: contact entries, their groups and lines, the fan triangles' angles
   and the constants */
static void pc_build(void)
{ edge_id e_id;
  int i, j, g, nb;
  int *order = NULL, *by = NULL;
  unsigned long long key = 0;

  pc_slab = PC_SLAB(pc_L);
  nent = 0; ngrp = 0; nlp = 0; pc_bad = 0;
  pc_gx_valid = 0;
  if ( (int)web.skel[EDGE].max_ord + 1 > nslot )
  { nslot = (int)web.skel[EDGE].max_ord + 1 + 1024;
    slot = (int*)realloc(slot,nslot*sizeof(int));
  }
  for ( i = 0 ; i < nslot ; i++ ) slot[i] = -1;
  nb = (int)web.skel[BODY].max_ord + 1;
  if ( !bodies_alloc(nb + 1) ) return;

  if ( EXTRAS(EDGE)[E_CONSTR_LIST_ATTR].array_spec.datacount )
  FOR_ALL_EDGES(e_id)
  { conmap_t *map = get_e_constraint_map(e_id);
    int m;
    for ( m = 1 ; m <= (int)map[0] ; m++ )
    { int s = sphere_of((int)(map[m] & CONMASK));
      facetedge_id fe, first;
      int bods[16], dirs[16], sapps[16], cnt[16], nbods = 0, t;
      if ( s < 0 ) continue;
      first = fe = get_edge_fe(e_id);
      if ( !valid_id(fe) ) continue;
      do
      { facet_id f = get_fe_facet(fe);
        int along = equal_id(get_fe_edge(fe),e_id) ? 1 : -1;
        for ( t = 0 ; t < 2 ; t++ )
        { body_id bid = get_facet_body(t ? inverse_id(f) : f);
          int bo, u;
          if ( !valid_id(bid) ) continue;
          bo = ordinal(bid);
          for ( u = 0 ; u < nbods ; u++ ) if ( bods[u] == bo ) break;
          if ( u == nbods )
          { if ( nbods == 16 ) continue;
            bods[u] = bo; cnt[u] = 0; nbods++;
            dirs[u] = (t ? -1 : 1)*along;
            sapps[u] = t ? -1 : 1;
          }
          cnt[u]++;
        }
        fe = get_next_facet(fe);
      } while ( valid_id(fe) && !equal_id(fe,first) );
      for ( t = 0 ; t < nbods ; t++ )
      { struct pc_entry *q;
        int k;
        if ( cnt[t] != 1 ) continue;
        k = ent_add();
        if ( k < 0 ) return;
        q = ent + k;
        q->e = e_id; q->s = s; q->b = bods[t]; q->dir = dirs[t]; q->sapp = sapps[t];
        q->A = q->dir > 0 ? get_edge_tailv(e_id) : get_edge_headv(e_id);
        q->B = q->dir > 0 ? get_edge_headv(e_id) : get_edge_tailv(e_id);
        q->prev = q->next = -1;
        q->other = -1;
        q->loop = -1;
      }
    }
  }
  for ( i = 0 ; i < nent ; i++ )
  { int o = ordinal(ent[i].e);
    ent[i].other = slot[o];
    slot[o] = i;
  }

  /* groups: (sphere, body), and each entry's neighbours along its line */
  if ( nent )
  { order = (int*)malloc(nent*sizeof(int));
    by = (int*)malloc(nent*sizeof(int));
    sort_v = (vertex_id*)malloc(nent*sizeof(vertex_id));
  }
  for ( i = 0 ; i < nent ; i++ ) order[i] = i;
  if ( nent ) qsort(order,nent,sizeof(int),cmp_key);
  for ( i = 0 ; i < nent ; )
  { struct pc_group *G;
    int n, a, lo, hi;
    for ( j = i ; j < nent && ent[order[j]].s == ent[order[i]].s
                  && ent[order[j]].b == ent[order[i]].b ; j++ ) ;
    if ( ngrp == maxgrp )
    { maxgrp = maxgrp ? 2*maxgrp : 256;
      grp = (struct pc_group *)realloc(grp,maxgrp*sizeof(*grp));
    }
    G = grp + ngrp;
    G->s = ent[order[i]].s; G->b = ent[order[i]].b;
    G->first = i; G->count = n = j - i; G->loops = 0; G->bad = 0; G->S = 0.0;
    for ( a = i ; a < j ; a++ ) ent[order[a]].group = ngrp;
    /* by A: each entry's next is the one whose A is its B */
    for ( a = 0 ; a < n ; a++ ) { by[a] = order[i + a]; sort_v[order[i + a]] = ent[order[i + a]].A; }
    qsort(by,n,sizeof(int),cmp_by_vertex);
    for ( a = 0 ; a + 1 < n ; a++ )
      if ( ent[by[a]].A == ent[by[a+1]].A ) G->bad = 1;     /* a branch */
    for ( a = 0 ; a < n ; a++ )
    { struct pc_entry *q = ent + order[i + a];
      lo = 0; hi = n - 1;
      while ( lo <= hi )
      { int mid = (lo + hi)/2;
        vertex_id v = ent[by[mid]].A;
        if ( v == q->B ) { q->next = by[mid]; break; }
        if ( v < q->B ) lo = mid + 1; else hi = mid - 1;
      }
      if ( q->next < 0 ) G->bad = 1;                        /* an open line */
      else if ( ent[q->next].prev >= 0 ) G->bad = 1;       /* two into one: a tail */
      else ent[q->next].prev = order[i + a];
    }
    /* loops */
    if ( !G->bad )
    { for ( a = i ; a < j ; a++ ) ent[order[a]].group = -1 - ngrp;   /* unvisited */
      for ( a = i ; a < j ; a++ )
      { int k = order[a];
        if ( ent[k].group >= 0 ) continue;
        G->loops++;
        if ( nlp == maxlp )
        { maxlp = maxlp ? 2*maxlp : 256;
          lp = (struct pc_loop *)realloc(lp,maxlp*sizeof(*lp));
        }
        lp[nlp].g = ngrp;
        while ( ent[k].group < 0 ) { ent[k].group = ngrp; ent[k].loop = nlp; k = ent[k].next; }
        nlp++;
      }
    }
    else pc_bad++;
    for ( a = i ; a < j ; a++ ) ent[order[a]].group = ngrp;
    ngrp++;
    i = j;
  }

  /* the topology's key: its entries in order */
  key = mix(0,(unsigned long long)nent);
  key = mix(key,(unsigned long long)web.skel[FACET].count);
  key = mix(key,(unsigned long long)web.skel[BODY].count);
  key = mix(key,(unsigned long long)PC_LOAD());
  for ( i = 0 ; i < nent ; i++ )
  { key = mix(key,(unsigned long long)ent[i].e);
    key = mix(key,(unsigned long long)(ent[i].s*4096 + ent[i].b*2 + (ent[i].dir > 0)));
  }

  pc_geometry();

  /* beads with no line: inside a body or not (when the lines' topology changed) */
  if ( key != topo_key )
  { int s, b;
    char *has = (char*)calloc(nsph ? nsph : 1,1);
    for ( b = 0 ; b <= nb ; b++ ) imm_count[b] = 0;
    for ( s = 0 ; s < nsph ; s++ ) imm_body[s] = -1;
    for ( i = 0 ; i < nent ; i++ ) has[ent[i].s] = 1;
    pc_immersed(nb,has);
    free(has);
    topo_key = key;
  }
  free(order); free(by); free(sort_v); sort_v = NULL;
}

/* the table rebuilt where the topology changed since it was built (the
   element counts, the load, Evolver's topology stamp, which edges' changes
   of facets, vertices and bodies and a snapshot's restore bump, and the
   edges' constraints), its angles and constants at the first init of each
   pass (stale from the end of the last one, pyse_contact_refresh()); every
   pass's rebuild was a fifth of a gradient step at 4 threads */

static int pc_changed(void)
{ long now[6] = { (long)web.skel[VERTEX].count, (long)web.skel[EDGE].count,
                  (long)web.skel[EDGE].max_ord, PC_LOAD(), (long)top_timestamp,
                  pyse_edge_conmap_stamp };
  int k, changed = 0;
  for ( k = 0 ; k < 6 ; k++ ) if ( now[k] != pc_stamp[k] ) { changed = 1; pc_stamp[k] = now[k]; }
  return changed;
}

static void pc_current(void)
{ if ( pc_changed() || pc_stale ) { pc_build(); pc_stale = 0; }
  else if ( !pc_geom_done ) pc_geometry();
  (void)angle_floor();      /* (its global looked up here, serially) */
}

void pyse_contact_init(int mode, struct method_instance *mi)
{ (void)mode; (void)mi;
  if ( web.modeltype != LINEAR )
    kb_error(9901,"pyse_contact_integral: linear model only.\n",RECOVERABLE);
  pc_current();
}

/* ---- second-order duals in the edge's two vertices' six coordinates ---- */

#define ND 6
typedef struct { REAL v, g[ND], h[ND][ND]; } d12;

/* (each takes the order o the pass needs: 0 value, 1 gradient, 2 Hessian; a
   thread-local one cost a TLS lookup at every operation) */

/* r = t, the parts the pass needs (a whole d12 is 1.2 KB: copied whole,
   a value or gradient pass spent most of its contact edges' time on it) */
static void dset(int o, d12 *r, const d12 *t)
{ r->v = t->v;
  if ( o >= 1 ) memcpy(r->g,t->g,sizeof(r->g));
  if ( o >= 2 ) memcpy(r->h,t->h,sizeof(r->h));
}

static void dz(int o, d12 *a, REAL v)
{ a->v = v;
  if ( o >= 1 ) memset(a->g,0,sizeof(a->g));
  if ( o >= 2 ) memset(a->h,0,sizeof(a->h));
}

static void dvar(int o, d12 *a, REAL v, int i)
{ dz(o,a,v);
  if ( o >= 1 ) a->g[i] = 1.0;
}

static void dadd(int o, d12 *r, const d12 *a, const d12 *b, REAL sb)   /* r = a + sb b */
{ int i, j;
  r->v = a->v + sb*b->v;
  if ( o >= 1 ) for ( i = 0 ; i < ND ; i++ ) r->g[i] = a->g[i] + sb*b->g[i];
  if ( o >= 2 )
    for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ ) r->h[i][j] = a->h[i][j] + sb*b->h[i][j];
}

static void dmul(int o, d12 *r, const d12 *a, const d12 *b)
{ d12 t;
  int i, j;
  t.v = a->v*b->v;
  if ( o >= 1 ) for ( i = 0 ; i < ND ; i++ ) t.g[i] = a->v*b->g[i] + b->v*a->g[i];
  if ( o >= 2 )
    for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ )
      t.h[i][j] = a->v*b->h[i][j] + b->v*a->h[i][j] + a->g[i]*b->g[j] + a->g[j]*b->g[i];
  dset(o,r,&t);
}

static void dscale(int o, d12 *a, REAL s)
{ int i, j;
  a->v *= s;
  if ( o >= 1 ) for ( i = 0 ; i < ND ; i++ ) a->g[i] *= s;
  if ( o >= 2 ) for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ ) a->h[i][j] *= s;
}

static void dsqrt(int o, d12 *r, const d12 *a)
{ REAL s = sqrt(a->v);
  int i, j;
  d12 t;
  t.v = s;
  if ( o >= 1 ) for ( i = 0 ; i < ND ; i++ ) t.g[i] = a->g[i]/(2*s);
  if ( o >= 2 )
    for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ )
      t.h[i][j] = a->h[i][j]/(2*s) - a->g[i]*a->g[j]/(4*s*s*s);
  dset(o,r,&t);
}

static void datan2(int o, d12 *r, const d12 *Y, const d12 *X)
{ REAL r2 = X->v*X->v + Y->v*Y->v;
  int i, j;
  d12 t;
  t.v = atan2(Y->v,X->v);
  if ( o >= 1 && r2 > 0 )
    for ( i = 0 ; i < ND ; i++ ) t.g[i] = (X->v*Y->g[i] - Y->v*X->g[i])/r2;
  else if ( o >= 1 ) memset(t.g,0,sizeof(t.g));
  if ( o >= 2 && r2 > 0 )
    for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ )
      t.h[i][j] = (X->g[j]*Y->g[i] + X->v*Y->h[i][j] - Y->g[j]*X->g[i] - Y->v*X->h[i][j])/r2
        - (X->v*Y->g[i] - Y->v*X->g[i])*(2*X->v*X->g[j] + 2*Y->v*Y->g[j])/(r2*r2);
  else if ( o >= 2 ) memset(t.h,0,sizeof(t.h));
  dset(o,r,&t);
}

static void ddot(int o, d12 *r, const d12 *a, const d12 *b)
{ d12 t;
  dmul(o,r,a,b);
  dmul(o,&t,a+1,b+1); dadd(o,r,r,&t,1.0);
  dmul(o,&t,a+2,b+2); dadd(o,r,r,&t,1.0);
}

/* omega(p, a, b) (a and b three duals each, p constant) */
static void domega(int o, d12 *r, const REAL *p, const d12 *a, const d12 *b)
{ d12 la, lb, det, den, t, s;
  int k;
  ddot(o,&la,a,a); dsqrt(o,&la,&la);
  ddot(o,&lb,b,b); dsqrt(o,&lb,&lb);
  /* det(p, a, b) = p . (a x b) */
  dmul(o,&t,a+1,b+2); dmul(o,&s,a+2,b+1); dadd(o,&det,&t,&s,-1.0); dscale(o,&det,p[0]);
  dmul(o,&t,a+2,b);   dmul(o,&s,a,b+2);   dadd(o,&t,&t,&s,-1.0); dadd(o,&det,&det,&t,p[1]);
  dmul(o,&t,a,b+1);   dmul(o,&s,a+1,b);   dadd(o,&t,&t,&s,-1.0); dadd(o,&det,&det,&t,p[2]);
  /* |a||b| + (p.a)|b| + a.b + (p.b)|a| */
  dmul(o,&den,&la,&lb);
  dz(o,&t,0.0); for ( k = 0 ; k < 3 ; k++ ) dadd(o,&t,&t,a+k,p[k]);
  dmul(o,&t,&t,&lb); dadd(o,&den,&den,&t,1.0);
  ddot(o,&t,a,b); dadd(o,&den,&den,&t,1.0);
  dz(o,&t,0.0); for ( k = 0 ; k < 3 ; k++ ) dadd(o,&t,&t,b+k,p[k]);
  dmul(o,&t,&t,&la); dadd(o,&den,&den,&t,1.0);
  datan2(o,r,&det,&den);
  dscale(o,r,2.0);
}

/* an entry's term: wetted area share (*area) and the volume line term
   (*line), as duals in the element's slots 0, 1 (tail, head) */
static void entry_terms(int o, const struct pc_entry *q, struct qinfo *e_info, d12 *area, d12 *line)
{ const struct pc_sphere *p = sph + q->s;
  d12 X[2][3], t, s;
  int slotA, slotB, i, k;
  REAL w[3];
  for ( i = 0 ; i < 2 ; i++ )
  { pc_delta(p->c,get_coord(e_info->v[i]),w);
    for ( k = 0 ; k < 3 ; k++ ) dvar(o,&X[i][k],w[k],3*i + k);
  }
  if ( get_edge_tailv(e_info->id) == q->A ) { slotA = 0; slotB = 1; }
  else { slotA = 1; slotB = 0; }
  if ( q->loop >= 0 )
  { domega(o,area,lp[q->loop].P,X[slotA],X[slotB]);
    dscale(o,area,p->R*p->R);
  }
  else dz(o,area,0.0);
  /* line: -(a_x b_y - a_y b_x)((a_z + b_z)/6 + h/2), a = A - c, b = B - c */
  { d12 *a = X[slotA], *b = X[slotB];
    REAL h = p->c[2] - p->zf;
    dmul(o,&t,a,b+1); dmul(o,&s,a+1,b); dadd(o,&t,&t,&s,-1.0);
    dadd(o,&s,a+2,b+2,1.0); dscale(o,&s,1.0/6.0); s.v += h/2;
    dmul(o,line,&t,&s); dscale(o,line,-1.0);
  }
}

/* entry_terms' value and gradient (order 0 or 1) in closed form: the
   values in the duals' order of operations, bit for bit; the gradients
   (g: the tail's three, then the head's) by hand. (The duals, 1.2 KB each,
   took a quarter of a 64-bead pack's gradient step, its value pass as much
   as its gradient's) */
static void entry_terms_1(int o, const struct pc_entry *q, struct qinfo *e_info,
                          REAL *area, REAL *ag, REAL *line, REAL *lg)
{ const struct pc_sphere *p = sph + q->s;
  REAL X[2][3], *a, *b, h = p->c[2] - p->zf, t, sv;
  int i, k, ia, ib;
  for ( i = 0 ; i < 2 ; i++ ) pc_delta(p->c,get_coord(e_info->v[i]),X[i]);
  if ( get_edge_tailv(e_info->id) == q->A ) { ia = 0; ib = 1; }
  else { ia = 1; ib = 0; }
  a = X[ia]; b = X[ib];
  if ( o >= 1 ) for ( k = 0 ; k < 6 ; k++ ) ag[k] = lg[k] = 0.0;
  *area = 0.0;
  if ( q->loop >= 0 )
  { const REAL *P = lp[q->loop].P;
    REAL la = sqrt((a[0]*a[0] + a[1]*a[1]) + a[2]*a[2]);
    REAL lb = sqrt((b[0]*b[0] + b[1]*b[1]) + b[2]*b[2]);
    REAL det = ((a[1]*b[2] - a[2]*b[1])*P[0] + P[1]*(a[2]*b[0] - a[0]*b[2]))
               + P[2]*(a[0]*b[1] - a[1]*b[0]);
    REAL pa = (P[0]*a[0] + P[1]*a[1]) + P[2]*a[2], pb = (P[0]*b[0] + P[1]*b[1]) + P[2]*b[2];
    REAL ab = (a[0]*b[0] + a[1]*b[1]) + a[2]*b[2];
    REAL den = ((la*lb + pa*lb) + ab) + pb*la;
    REAL RR = p->R*p->R;
    *area = 2.0*atan2(det,den)*RR;
    if ( o >= 1 )
    { REAL r2 = den*den + det*det, ca = (lb + pb)/la, cb = (la + pa)/lb;
      REAL dda[3], ddb[3], dna[3], dnb[3];
      /* det = p . (a x b): d/da = b x p, d/db = p x a */
      dda[0] = b[1]*P[2] - b[2]*P[1]; dda[1] = b[2]*P[0] - b[0]*P[2]; dda[2] = b[0]*P[1] - b[1]*P[0];
      ddb[0] = P[1]*a[2] - P[2]*a[1]; ddb[1] = P[2]*a[0] - P[0]*a[2]; ddb[2] = P[0]*a[1] - P[1]*a[0];
      for ( k = 0 ; k < 3 ; k++ )
      { dna[k] = ca*a[k] + lb*P[k] + b[k];
        dnb[k] = cb*b[k] + la*P[k] + a[k];
      }
      if ( r2 > 0 )
        for ( k = 0 ; k < 3 ; k++ )
        { ag[3*ia + k] = 2.0*RR*(den*dda[k] - det*dna[k])/r2;
          ag[3*ib + k] = 2.0*RR*(den*ddb[k] - det*dnb[k])/r2;
        }
    }
  }
  /* line: -(a_x b_y - a_y b_x)((a_z + b_z)/6 + h/2) */
  t = a[0]*b[1] - a[1]*b[0];
  sv = (a[2] + b[2])*(1.0/6.0) + h/2;
  *line = t*sv*-1.0;
  if ( o >= 1 )
  { lg[3*ia + 0] = -sv*b[1]; lg[3*ia + 1] = sv*b[0]; lg[3*ia + 2] = -t/6.0;
    lg[3*ib + 0] = sv*a[1];  lg[3*ib + 1] = -sv*a[0]; lg[3*ib + 2] = -t/6.0;
  }
}

/* pc_eval for order 0 or 1, by entry_terms_1 */
static REAL pc_eval_1(struct qinfo *e_info, int o, int s, int body, int eo)
{ REAL tot = 0.0, tg[6] = {0,0,0,0,0,0};
  int k, i, a;
  for ( k = slot[eo] ; k >= 0 ; k = ent[k].other )
  { const struct pc_entry *q = ent + k;
    REAL area, ag[6], line, lg[6];
    if ( q->s != s || (body >= 0 && q->b != body) ) continue;
    entry_terms_1(o,q,e_info,&area,ag,&line,lg);
    if ( body >= 0 )
    { REAL c = -sph[s].R/3.0, sa = (REAL)q->sapp;
      tot = tot + sa*(area*c + line);
      if ( o >= 1 ) for ( i = 0 ; i < 6 ; i++ ) tg[i] += sa*(ag[i]*c + lg[i]);
    }
    else
    { REAL kk = -sphere_k(sph + s);
      tot = tot + kk*area;
      if ( o >= 1 ) for ( i = 0 ; i < 6 ; i++ ) tg[i] += kk*ag[i];
    }
  }
  if ( o >= 1 )
    for ( a = 0 ; a < e_info->vcount && a < 2 ; a++ )
      for ( i = 0 ; i < 3 ; i++ ) e_info->grad[a][i] = tg[3*a + i];
  return tot;
}

/* the element's value (and derivatives) for instance mi */
static REAL pc_eval(struct qinfo *e_info, int order)
{ const int o = order;
  struct method_instance *mi = METH_INSTANCE(e_info->method);
  int s = sphere_of(mi->connum), k, eo, body = -1, i, j, a, b2;
  d12 tot, area, line;
  dz(o,&tot,0.0);
  if ( s < 0 || !nent ) goto out;
  eo = ordinal(e_info->id);
  if ( eo >= nslot ) goto out;
  if ( mi->flags & BODY_INSTANCE )
  { for ( j = 0 ; j < MMAXQUANTS ; j++ )
      if ( mi->quants[j] >= 0 )
      { body_id bid = GEN_QUANT(mi->quants[j])->b_id;
        if ( valid_id(bid) ) body = ordinal(bid);
        break;
      }
    if ( body < 0 ) goto out;
  }
  if ( o < 2 ) return pc_eval_1(e_info,o,s,body,eo);
  for ( k = slot[eo] ; k >= 0 ; k = ent[k].other )
  { const struct pc_entry *q = ent + k;
    if ( q->s != s || (body >= 0 && q->b != body) ) continue;
    entry_terms(o,q,e_info,&area,&line);
    if ( body >= 0 )
    { /* the body's content, as Evolver applies it: times sapp */
      dscale(o,&area,-sph[s].R/3.0);
      dadd(o,&area,&area,&line,1.0);
      dadd(o,&tot,&tot,&area,(REAL)q->sapp);
    }
    else
    { dadd(o,&tot,&tot,&area,-sphere_k(sph + s));
    }
  }
out:
  if ( order >= 1 )
    for ( a = 0 ; a < e_info->vcount && a < 2 ; a++ )
      for ( i = 0 ; i < 3 ; i++ ) e_info->grad[a][i] = tot.g[3*a + i];
  if ( order >= 2 )
    for ( a = 0 ; a < e_info->vcount && a < 2 ; a++ )
      for ( b2 = 0 ; b2 < e_info->vcount && b2 < 2 ; b2++ )
        for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ )
          e_info->hess[a][b2][i][j] = tot.h[3*a + i][3*b2 + j];
  return tot.v;
}

/* before a pass's quantities are put together: the table built if no
   instance of ours started it (a pass of the volumes alone where no body
   has a line) */
void pyse_contact_refresh(void)
{ if ( nsph ) pc_current();
  pc_geom_done = 0;
}

/* what a body's volume (quantity k) adds beyond its edges: the groups'
   constants on each sphere times -R/3, and the spheres inside it whole
   (a body's content instance on a sphere it has no edge on isn't part of
   its volume: Evolver attaches them as it applies them) */
REAL pyse_contact_body_constant(int k)
{ struct gen_quant *q = GEN_QUANT(k);
  int g, s, body;
  REAL v = 0.0;
  if ( !nsph || !valid_id(q->b_id) || get_body_volquant(q->b_id) != k ) return 0.0;
  body = ordinal(q->b_id);
  for ( g = 0 ; g < ngrp ; g++ )
    if ( grp[g].b == body ) v += -sph[grp[g].s].R/3.0*grp[g].K;
  for ( s = 0 ; imm_body && s < nsph && s < nimm_alloc ; s++ )
    if ( imm_body[s] == body ) v += -4.0/3.0*M_PI*sph[s].R*sph[s].R*sph[s].R;
  return v;
}

/* what an energy instance adds beyond its edges: the groups' constants (less
   the whole spheres n), and the spheres inside a body whole
   (their area and volume); for the energy times -k */
REAL pyse_contact_constant(struct method_instance *mi)
{ int s, g;
  REAL v = 0.0;
  if ( pc_method < 0 || mi->gen_method != pc_method ) return 0.0;
  s = sphere_of(mi->connum);
  if ( s < 0 ) return 0.0;
  if ( mi->flags & BODY_INSTANCE ) return 0.0;   /* (the body's: pyse_contact_body_constant) */
  for ( g = 0 ; g < ngrp ; g++ )
    if ( grp[g].s == s ) v += grp[g].K;
  if ( imm_body && s < nimm_alloc && imm_body[s] >= 0 )
    v += 4*M_PI*sph[s].R*sph[s].R;
  return -sphere_k(sph + s)*v;
}

REAL pyse_contact_value(struct qinfo *e_info) { return pc_eval(e_info,0); }
REAL pyse_contact_grad(struct qinfo *e_info) { return pc_eval(e_info,1); }
REAL pyse_contact_hess(struct qinfo *e_info) { return pc_eval(e_info,2); }

/* diagnostics: entries, groups, groups with a branch or an open line,
   bodies with whole spheres in them but no line to carry them, the
   immersed spheres per body (out, up to n) */
int pyse_contact_status(int *out, int n)
{ int b;
  pc_build();
  if ( n > 0 ) out[0] = nent;
  if ( n > 1 ) out[1] = ngrp;
  if ( n > 2 ) out[2] = pc_bad;
  if ( n > 3 ) out[3] = 0;
  for ( b = 0 ; b + 4 < n ; b++ ) out[b + 4] = b < nbody_alloc ? imm_count[b] : 0;
  return 1;
}
