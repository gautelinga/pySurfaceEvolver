/*
 * pyse_stepcap.c -- a gradient step capped so the surface doesn't pass
 * through itself (pyse_step_cap, called by iterate() once its line search
 * has the scale of the move, x += scale*velocity).
 *
 * Each pair of a vertex and a facet, and of two edges, more than a ring of
 * edges apart (no vertex shared, none joined by an edge: as the crossings
 * counted, pyse_edit.c's far ones) closes its gap at most by the share
 * cap_frac in one step, to first order: the gap's rate along the line
 * between the pair's nearest points, the vertices' velocities interpolated
 * there. A pair moving apart takes nothing off the step. So two sheets
 * coming together slow down step by step, the gap closing geometrically,
 * until the descent's run ends and the change between them is made
 * (a 64-bead pack's cascades threw away 28-41% of their descents' runs,
 * each run 50 steps that took sheets through each other, 31 places of a
 * run at the median, two thirds of them sheets far apart on the surface).
 * And no vertex moves more than a mean edge in a step (the gaps looked
 * for are the move's reach: a run's step took a vertex 70 across).
 *
 * Linear soap films in 3D, in a box or a slab (periodic in x and y,
 * pyse_slab: the pairs at their nearest images); else the step as it is.
 */

#include "include.h"
#include "pyse_api.h"
#include "pyse_internal.h"

static REAL cap_frac = 0.0;     /* 0: off */
static REAL contact_factor = 0.0;   /* a capped pair within this many local edges: contact */
static int contact = 0;         /* such a pair met since pyse_step_cap_contact cleared it */
static int cap_keep = 0;        /* keep the last step's (pyse_step_cap_last) */
static REAL *last_X = NULL, *last_V = NULL, last_s = 0.0;
static long last_n = 0;
static long cap_steps = 0, cap_capped = 0, cap_moved = 0;

double pyse_set_step_cap(double frac, double contact_at)
{ double was = cap_frac;
  cap_frac = frac > 0.0 && frac < 1.0 ? frac : 0.0;
  contact_factor = contact_at > 0.0 ? contact_at : 0.0;
  return was;
}

/* whether a capped step met a pair within contact_factor local edges (a
   change to make: the run should end) since last cleared; clear: clear it */
int pyse_step_cap_contact(int clear)
{ int was = contact;
  if ( clear ) contact = 0;
  return was;
}

/* steps looked at, capped by a pair, capped by the longest move */
void pyse_step_cap_counts(long *out)
{ out[0] = cap_steps; out[1] = cap_capped; out[2] = cap_moved;
}

/* the last capped step's scale, and its vertices' coordinates and velocities by
   ordinal (n of them; NULL where none kept): a debugging aid */
long pyse_step_cap_last(int keep, double *scale, const double **x, const double **v)
{ cap_keep = keep;
  *scale = last_s; *x = last_X; *v = last_V;
  return last_n;
}

/* ---- geometry -------------------------------------------------------- */

static REAL dot3c(const REAL *a, const REAL *b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

static int slab_on;
static REAL slab_L[2];

static void img(const REAL *ref, const REAL *x, REAL *out)
{ out[0] = x[0]; out[1] = x[1]; out[2] = x[2];
  if ( slab_on )
  { out[0] -= slab_L[0]*floor((x[0] - ref[0])/slab_L[0] + 0.5);
    out[1] -= slab_L[1]*floor((x[1] - ref[1])/slab_L[1] + 0.5);
  }
}

/* the point of triangle abc nearest p, as barycentric weights w (Ericson,
   Real-Time Collision Detection 5.1.5) */
static void nearest_on_triangle(const REAL *p, const REAL *a, const REAL *b, const REAL *c,
                                REAL *w)
{ REAL ab[3], ac[3], ap[3], bp[3], cp[3], d1, d2, d3, d4, d5, d6, va, vb, vc, den;
  int i;
  for ( i = 0 ; i < 3 ; i++ )
  { ab[i] = b[i] - a[i]; ac[i] = c[i] - a[i]; ap[i] = p[i] - a[i];
    bp[i] = p[i] - b[i]; cp[i] = p[i] - c[i];
  }
  d1 = dot3c(ab,ap); d2 = dot3c(ac,ap);
  if ( d1 <= 0 && d2 <= 0 ) { w[0] = 1; w[1] = w[2] = 0; return; }
  d3 = dot3c(ab,bp); d4 = dot3c(ac,bp);
  if ( d3 >= 0 && d4 <= d3 ) { w[1] = 1; w[0] = w[2] = 0; return; }
  vc = d1*d4 - d3*d2;
  if ( vc <= 0 && d1 >= 0 && d3 <= 0 )
  { REAL v = d1/(d1 - d3); w[0] = 1 - v; w[1] = v; w[2] = 0; return; }
  d5 = dot3c(ab,cp); d6 = dot3c(ac,cp);
  if ( d6 >= 0 && d5 <= d6 ) { w[2] = 1; w[0] = w[1] = 0; return; }
  vb = d5*d2 - d1*d6;
  if ( vb <= 0 && d2 >= 0 && d6 <= 0 )
  { REAL v = d2/(d2 - d6); w[0] = 1 - v; w[2] = v; w[1] = 0; return; }
  va = d3*d6 - d5*d4;
  if ( va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 )
  { REAL v = (d4 - d3)/((d4 - d3) + (d5 - d6)); w[1] = 1 - v; w[2] = v; w[0] = 0; return; }
  den = 1/(va + vb + vc);
  w[1] = vb*den; w[2] = vc*den; w[0] = 1 - w[1] - w[2];
}

static REAL clamp01(REAL x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

/* the nearest points of segments p1q1 and p2q2, as their parameters s, t
   (Ericson 5.1.9) */
static void nearest_on_segments(const REAL *p1, const REAL *q1, const REAL *p2, const REAL *q2,
                                REAL *s, REAL *t)
{ REAL d1[3], d2[3], r[3], a, e, f, c, b, den;
  int i;
  for ( i = 0 ; i < 3 ; i++ ) { d1[i] = q1[i] - p1[i]; d2[i] = q2[i] - p2[i]; r[i] = p1[i] - p2[i]; }
  a = dot3c(d1,d1); e = dot3c(d2,d2); f = dot3c(d2,r);
  if ( a <= 1e-300 && e <= 1e-300 ) { *s = *t = 0; return; }
  if ( a <= 1e-300 ) { *s = 0; *t = clamp01(f/e); return; }
  c = dot3c(d1,r);
  if ( e <= 1e-300 ) { *t = 0; *s = clamp01(-c/a); return; }
  b = dot3c(d1,d2); den = a*e - b*b;
  *s = den > 0 ? clamp01((b*f - c*e)/den) : 0;
  *t = (b*(*s) + f)/e;
  if ( *t < 0 ) { *t = 0; *s = clamp01(-c/a); }
  else if ( *t > 1 ) { *t = 1; *s = clamp01((b - c)/a); }
}

/* the scale at which the pair at nearest points x (moving at u) and y
   (moving at v) closes cap_frac of its gap; HUGE where it opens */
static REAL pair_scale(const REAL *x, const REAL *y, const REAL *u, const REAL *v, REAL tiny,
                       REAL *gap)
{ REAL d[3], g, closing;
  int i;
  for ( i = 0 ; i < 3 ; i++ ) d[i] = x[i] - y[i];
  g = sqrt(dot3c(d,d));
  *gap = g;
  if ( g <= tiny ) return HUGE_VAL;     /* (touching: the change's to make) */
  closing = 0;
  for ( i = 0 ; i < 3 ; i++ ) closing -= d[i]*(u[i] - v[i]);
  closing /= g;
  return closing > 0 ? cap_frac*g/closing : HUGE_VAL;
}

/* ---- the vertices, their velocities, a grid of the facets and edges ---- */

static REAL (*X)[3], (*V)[3];   /* by vertex ordinal: coordinates, velocity (0 if fixed) */
static REAL *speed;

struct cell_grid
{ REAL c[3]; long n[2];         /* cell sides; cells a period (slab) */
  long buckets, *head, *next, *item, count, cap;
  long *key;
};

static long cell_hash(const struct cell_grid *g, long i, long j, long k)
{ unsigned long h = (unsigned long)i*73856093UL ^ (unsigned long)j*19349663UL
                    ^ (unsigned long)k*83492791UL;
  return (long)(h % (unsigned long)g->buckets);
}

static void wrap_index(const struct cell_grid *g, long *i, long *j)
{ if ( slab_on )
  { *i %= g->n[0]; if ( *i < 0 ) *i += g->n[0];
    *j %= g->n[1]; if ( *j < 0 ) *j += g->n[1];
  }
}

/* the cells a box covers: lo..hi indices (unwrapped) */
static void cell_range(const struct cell_grid *g, const REAL *lo, const REAL *hi, long *a, long *b)
{ int d;
  for ( d = 0 ; d < 3 ; d++ )
  { a[d] = (long)floor(lo[d]/g->c[d]);
    b[d] = (long)floor(hi[d]/g->c[d]);
    if ( slab_on && d < 2 && b[d] - a[d] >= g->n[d] ) b[d] = a[d] + g->n[d] - 1;
  }
}

static void grid_add(struct cell_grid *g, const REAL *lo, const REAL *hi, long item)
{ long a[3], b[3], i, j, k;
  cell_range(g,lo,hi,a,b);
  for ( i = a[0] ; i <= b[0] ; i++ )
    for ( j = a[1] ; j <= b[1] ; j++ )
      for ( k = a[2] ; k <= b[2] ; k++ )
      { long ii = i, jj = j, h;
        wrap_index(g,&ii,&jj);
        if ( g->count == g->cap )
        { g->cap = 2*g->cap + 1024;
          g->next = (long*)realloc(g->next,g->cap*sizeof(long));
          g->item = (long*)realloc(g->item,g->cap*sizeof(long));
          g->key = (long*)realloc(g->key,3*g->cap*sizeof(long));
        }
        h = cell_hash(g,ii,jj,k);
        g->item[g->count] = item;
        g->key[3*g->count] = ii; g->key[3*g->count+1] = jj; g->key[3*g->count+2] = k;
        g->next[g->count] = g->head[h];
        g->head[h] = g->count++;
      }
}

static void grid_free(struct cell_grid *g)
{ free(g->head); free(g->next); free(g->item); free(g->key);
}

/* a box's lower corner moved into the cell (x, y in [0, L)), its upper with it */
static void box_home(REAL *lo, REAL *hi)
{ int d;
  if ( !slab_on ) return;
  for ( d = 0 ; d < 2 ; d++ )
  { REAL s = slab_L[d]*floor(lo[d]/slab_L[d]);
    lo[d] -= s; hi[d] -= s;
  }
}

static int far_apart(vertex_id a, vertex_id b)
{ edge_id e, e0;
  int guard;
  if ( equal_id(a,b) ) return 0;
  /* (pyse_edit.c's EACH_EDGE_AT) */
  for ( e0 = get_vertex_edge(a), e = e0, guard = 0 ; valid_id(e) && guard < 100000 ;
        e = get_next_tail_edge(e), guard = equal_id(e,e0) ? 100000 : guard + 1 )
    if ( equal_id(get_edge_headv(e),b) ) return 0;
  return 1;
}

static void facet_corners(facet_id f, vertex_id *v)
{ facetedge_id fe = get_facet_fe(positive_id(f));
  int k;
  for ( k = 0 ; k < 3 ; k++, fe = get_next_edge(fe) ) v[k] = get_fe_tailv(fe);
}

REAL pyse_step_cap(REAL scale)
{ long nv, nf, ne, k, fi, *fstamp = NULL, *estamp = NULL, stamp = 0;
  vertex_id v_id;
  edge_id e_id;
  facet_id f_id;
  facet_id *F = NULL;
  edge_id *E = NULL;
  vertex_id (*FV)[3] = NULL;
  REAL hsum = 0, hbar, vmax = 0, s = scale, tiny;
  struct cell_grid gf, ge;
  int moved = 0, capped = 0;

  if ( cap_frac <= 0.0 || scale <= 0.0 || SDIM != 3 || web.modeltype != LINEAR
       || web.representation != SOAPFILM )
    return scale;
  slab_on = web.torus_flag && pyse_slab(slab_L);
  if ( web.torus_flag && !slab_on ) return scale;
  cap_steps++;

  nv = web.skel[VERTEX].max_ord + 1;
  X = (REAL(*)[3])calloc(nv,sizeof(REAL[3]));
  V = (REAL(*)[3])calloc(nv,sizeof(REAL[3]));
  speed = (REAL*)calloc(nv,sizeof(REAL));
  FOR_ALL_VERTICES(v_id)
  { long o = ordinal(v_id);
    REAL *x = get_coord(v_id);
    ATTR attr = get_vattr(v_id);
    X[o][0] = x[0]; X[o][1] = x[1]; X[o][2] = x[2];
    if ( !(attr & (FIXED|BOUNDARY)) )
    { REAL *u = get_velocity(v_id);
      V[o][0] = u[0]; V[o][1] = u[1]; V[o][2] = u[2];
      speed[o] = sqrt(dot3c(V[o],V[o]));
      if ( speed[o] > vmax ) vmax = speed[o];
    }
  }
  ne = 0;
  FOR_ALL_EDGES(e_id) ne++;
  E = (edge_id*)malloc((ne + 1)*sizeof(edge_id));
  ne = 0;
  FOR_ALL_EDGES(e_id)
  { REAL y[3], *x = X[ordinal(get_edge_tailv(e_id))];
    img(x,X[ordinal(get_edge_headv(e_id))],y);
    hsum += sqrt((y[0]-x[0])*(y[0]-x[0]) + (y[1]-x[1])*(y[1]-x[1]) + (y[2]-x[2])*(y[2]-x[2]));
    E[ne++] = e_id;
  }
  if ( ne == 0 || vmax <= 0 ) goto done;
  hbar = hsum/ne;
  tiny = 1e-9*hbar;
  /* (no vertex further than a mean edge) */
  if ( s*vmax > hbar ) { s = hbar/vmax; moved = 1; }

  nf = 0;
  FOR_ALL_FACETS(f_id) nf++;
  F = (facet_id*)malloc((nf + 1)*sizeof(facet_id));
  FV = (vertex_id(*)[3])malloc((nf + 1)*sizeof(vertex_id[3]));
  nf = 0;
  FOR_ALL_FACETS(f_id) { F[nf] = f_id; facet_corners(f_id,FV[nf]); nf++; }

  /* the grids: cells of two mean edges, a whole number a period */
  memset(&gf,0,sizeof gf); memset(&ge,0,sizeof ge);
  for ( k = 0 ; k < 3 ; k++ ) gf.c[k] = 2*hbar;
  if ( slab_on )
    for ( k = 0 ; k < 2 ; k++ )
    { gf.n[k] = (long)floor(slab_L[k]/gf.c[k]);
      if ( gf.n[k] < 1 ) gf.n[k] = 1;
      gf.c[k] = slab_L[k]/gf.n[k];
    }
  ge = gf;
  gf.buckets = ge.buckets = 4*(nf + ne) + 1;
  gf.head = (long*)malloc(gf.buckets*sizeof(long));
  ge.head = (long*)malloc(ge.buckets*sizeof(long));
  for ( k = 0 ; k < gf.buckets ; k++ ) gf.head[k] = ge.head[k] = -1;

  /* each facet's box, padded by its reach: where a pair with it could close
     cap_frac of its gap at this scale (the speeds' sum times the scale over
     cap_frac), its share of that its corners' fastest */
  for ( fi = 0 ; fi < nf ; fi++ )
  { REAL c[3][3], lo[3], hi[3], pad = 0;
    int a, d;
    for ( a = 0 ; a < 3 ; a++ )
    { long o = ordinal(FV[fi][a]);
      img(X[ordinal(FV[fi][0])],X[o],c[a]);
      if ( speed[o] > pad ) pad = speed[o];
    }
    pad *= s/cap_frac;
    for ( d = 0 ; d < 3 ; d++ )
    { lo[d] = fmin(c[0][d],fmin(c[1][d],c[2][d])) - pad;
      hi[d] = fmax(c[0][d],fmax(c[1][d],c[2][d])) + pad;
    }
    box_home(lo,hi);
    grid_add(&gf,lo,hi,fi);
  }
  for ( k = 0 ; k < ne ; k++ )
  { long t = ordinal(get_edge_tailv(E[k])), h = ordinal(get_edge_headv(E[k]));
    REAL y[3], lo[3], hi[3], pad = (speed[t] > speed[h] ? speed[t] : speed[h])*s/cap_frac;
    int d;
    img(X[t],X[h],y);
    for ( d = 0 ; d < 3 ; d++ )
    { lo[d] = fmin(X[t][d],y[d]) - pad; hi[d] = fmax(X[t][d],y[d]) + pad; }
    box_home(lo,hi);
    grid_add(&ge,lo,hi,k);
  }

  /* vertex against facet */
  fstamp = (long*)calloc(nf + 1,sizeof(long));
  FOR_ALL_VERTICES(v_id)
  { long o = ordinal(v_id), a[3], b[3], i, j, kk;
    REAL lo[3], hi[3], p[3], pad = speed[o]*s/cap_frac;
    int d;
    for ( d = 0 ; d < 3 ; d++ ) { p[d] = X[o][d]; lo[d] = p[d] - pad; hi[d] = p[d] + pad; }
    box_home(lo,hi);
    cell_range(&gf,lo,hi,a,b);
    stamp++;
    for ( i = a[0] ; i <= b[0] ; i++ )
      for ( j = a[1] ; j <= b[1] ; j++ )
        for ( kk = a[2] ; kk <= b[2] ; kk++ )
        { long ii = i, jj = j, it;
          wrap_index(&gf,&ii,&jj);
          for ( it = gf.head[cell_hash(&gf,ii,jj,kk)] ; it >= 0 ; it = gf.next[it] )
          { long f = gf.item[it];
            REAL c[3][3], q[3], u[3], w[3], pp[3], sc, gap;
            int m;
            if ( gf.key[3*it] != ii || gf.key[3*it+1] != jj || gf.key[3*it+2] != kk ) continue;
            if ( fstamp[f] == stamp ) continue;
            fstamp[f] = stamp;
            if ( !far_apart(v_id,FV[f][0]) || !far_apart(v_id,FV[f][1])
                 || !far_apart(v_id,FV[f][2]) )
              continue;
            img(p,X[ordinal(FV[f][0])],c[0]);
            img(c[0],X[ordinal(FV[f][1])],c[1]);
            img(c[0],X[ordinal(FV[f][2])],c[2]);
            img(c[0],p,pp);
            nearest_on_triangle(pp,c[0],c[1],c[2],w);
            for ( m = 0 ; m < 3 ; m++ )
            { q[m] = w[0]*c[0][m] + w[1]*c[1][m] + w[2]*c[2][m];
              u[m] = w[0]*V[ordinal(FV[f][0])][m] + w[1]*V[ordinal(FV[f][1])][m]
                     + w[2]*V[ordinal(FV[f][2])][m];
            }
            sc = pair_scale(pp,q,V[o],u,tiny,&gap);
            if ( sc < s ) { s = sc; capped = 1; }
            if ( sc < scale && contact_factor > 0 && !contact )
            { REAL e = (sqrt((c[1][0]-c[0][0])*(c[1][0]-c[0][0]) + (c[1][1]-c[0][1])*(c[1][1]-c[0][1])
                             + (c[1][2]-c[0][2])*(c[1][2]-c[0][2]))
                        + sqrt((c[2][0]-c[1][0])*(c[2][0]-c[1][0]) + (c[2][1]-c[1][1])*(c[2][1]-c[1][1])
                               + (c[2][2]-c[1][2])*(c[2][2]-c[1][2]))
                        + sqrt((c[0][0]-c[2][0])*(c[0][0]-c[2][0]) + (c[0][1]-c[2][1])*(c[0][1]-c[2][1])
                               + (c[0][2]-c[2][2])*(c[0][2]-c[2][2])))/3;
              if ( gap < contact_factor*e ) contact = 1;
            }
          }
        }
  }

  /* edge against edge (each pair once: the other's index higher) */
  estamp = (long*)calloc(ne + 1,sizeof(long));
  for ( k = 0 ; k < ne ; k++ )
  { vertex_id et = get_edge_tailv(E[k]), eh = get_edge_headv(E[k]);
    long t = ordinal(et), h = ordinal(eh), a[3], b[3], i, j, kk;
    REAL y[3], lo[3], hi[3], pad = (speed[t] > speed[h] ? speed[t] : speed[h])*s/cap_frac;
    int d;
    img(X[t],X[h],y);
    for ( d = 0 ; d < 3 ; d++ )
    { lo[d] = fmin(X[t][d],y[d]) - pad; hi[d] = fmax(X[t][d],y[d]) + pad; }
    box_home(lo,hi);
    cell_range(&ge,lo,hi,a,b);
    stamp++;
    for ( i = a[0] ; i <= b[0] ; i++ )
      for ( j = a[1] ; j <= b[1] ; j++ )
        for ( kk = a[2] ; kk <= b[2] ; kk++ )
        { long ii = i, jj = j, it;
          wrap_index(&ge,&ii,&jj);
          for ( it = ge.head[cell_hash(&ge,ii,jj,kk)] ; it >= 0 ; it = ge.next[it] )
          { long o2 = ge.item[it];
            vertex_id ot, oh;
            REAL p2[3], q2[3], x1[3], x2[3], u1[3], u2[3], ss, tt, sc, gap;
            long t2, h2;
            int m;
            if ( ge.key[3*it] != ii || ge.key[3*it+1] != jj || ge.key[3*it+2] != kk ) continue;
            if ( o2 <= k || estamp[o2] == stamp ) continue;
            estamp[o2] = stamp;
            ot = get_edge_tailv(E[o2]); oh = get_edge_headv(E[o2]);
            if ( !far_apart(et,ot) || !far_apart(et,oh) || !far_apart(eh,ot)
                 || !far_apart(eh,oh) )
              continue;
            t2 = ordinal(ot); h2 = ordinal(oh);
            img(X[t],X[t2],p2);
            img(p2,X[h2],q2);
            nearest_on_segments(X[t],y,p2,q2,&ss,&tt);
            for ( m = 0 ; m < 3 ; m++ )
            { x1[m] = X[t][m] + ss*(y[m] - X[t][m]);
              x2[m] = p2[m] + tt*(q2[m] - p2[m]);
              u1[m] = (1 - ss)*V[t][m] + ss*V[h][m];
              u2[m] = (1 - tt)*V[t2][m] + tt*V[h2][m];
            }
            sc = pair_scale(x1,x2,u1,u2,tiny,&gap);
            if ( sc < s ) { s = sc; capped = 1; }
            if ( sc < scale && contact_factor > 0 && !contact )
            { REAL l1 = sqrt((y[0]-X[t][0])*(y[0]-X[t][0]) + (y[1]-X[t][1])*(y[1]-X[t][1])
                             + (y[2]-X[t][2])*(y[2]-X[t][2]));
              REAL l2 = sqrt((q2[0]-p2[0])*(q2[0]-p2[0]) + (q2[1]-p2[1])*(q2[1]-p2[1])
                             + (q2[2]-p2[2])*(q2[2]-p2[2]));
              if ( gap < contact_factor*(l1 + l2)/2 ) contact = 1;
            }
          }
        }
  }
  grid_free(&gf); grid_free(&ge);

done:
  if ( cap_keep )
  { free(last_X); free(last_V);
    last_X = (REAL*)X; last_V = (REAL*)V; last_n = nv; last_s = s < scale ? s : scale;
    X = V = NULL;
  }
  free(X); free(V); free(speed); free(F); free(FV); free(E); free(fstamp); free(estamp);
  X = V = NULL; speed = NULL;
  if ( capped ) cap_capped++;
  else if ( moved ) cap_moved++;
  return s < scale ? s : scale;
}
