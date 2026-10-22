/*
 * pyse_events.c -- the loops of pySE's search for coming changes of
 * topology (_events.py: Evolver.events(), and a descent's changes and
 * sizing, several calls per iteration) in C. They work on the mesh's
 * NumPy arrays, never on Evolver's own state, so they need no guard and run
 * with the GIL released, in OpenMP threads where that pays. What counts as
 * an event, its group and its record stay in Python; these compute the
 * numbers the NumPy code they replaced did, by the same formulas (equal to
 * round-off: some sums are added in another order).
 *
 *   frame     the facets' edges, each vertex's normal and whether it lies
 *             in a fold of the mesh;
 *   self      the surface close to itself: (vertex, facet) pairs facing
 *             each other across a gap, not near each other along the mesh;
 *   lines     contact-line vertices on one constraint close to each other
 *             while far apart along the lines;
 *   segments  points' distances to the nearest of a set of segments;
 *   groups    the connected groups of masked vertices (union-find).
 *
 * Each returns 0, or -1 if memory ran out (-2: a row out of range).
 * Arrays returned through pointers are malloc'ed; the caller frees them.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "pyse_api.h"

/* Below this much work (candidate vertices; points times segments) the
   loops stay serial: thread start-up costs more than it saves. (The
   surface close to itself at 0.5 edges, a 16-bead pack's 3k facets: 1.0
   ms, 0.56 with two threads, 0.42 with four; at 15k facets 5.9, 3.5, 2.4.) */
#define PAR_MIN_VERTICES 1024
#define PAR_MIN_WORK 200000L

static double dot3(const double *a, const double *b)
{ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

static void sub3(const double *a, const double *b, double *c)
{ c[0] = a[0] - b[0]; c[1] = a[1] - b[1]; c[2] = a[2] - b[2]; }

static void cross3(const double *a, const double *b, double *c)
{ c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}

static int finite3(const double *p)
{ return isfinite(p[0]) && isfinite(p[1]) && isfinite(p[2]); }

/* NumPy's maximum(a, tiny): NaN stays NaN */
static double at_least(double a, double tiny)
{ return (a != a || a > tiny) ? a : tiny; }

/* NumPy's clip(t, 0, 1): NaN stays NaN */
static double clip01(double t)
{ return t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t; }

/* NumPy's minimum and maximum: NaN wins */
static double nan_min(double a, double b) { return (a != a || a < b) ? a : b; }
static double nan_max(double a, double b) { return (a != a || a > b) ? a : b; }

/* ---- growable arrays ------------------------------------------------------ */

struct buf { char *data; long count, cap; size_t size; };

static int buf_push(struct buf *b, const void *item)
{ if ( b->count == b->cap )
  { long cap = b->cap ? 2*b->cap : 16;
    char *data = (char *)realloc(b->data,(size_t)cap*b->size);
    if ( !data ) return -1;
    b->data = data;
    b->cap = cap;
  }
  memcpy(b->data + (size_t)b->count*b->size,item,b->size);
  b->count++;
  return 0;
}

static size_t at_least_one(long n) { return n > 0 ? (size_t)n : 1; }

/* ---- the frame -------------------------------------------------------------- */

/* sort a short run (one vertex's edges) */
static void sort_short(int64_t *a, long n)
{ long i, j;
  for ( i = 1 ; i < n ; i++ )
  { int64_t v = a[i];
    for ( j = i ; j > 0 && a[j-1] > v ; j-- ) a[j] = a[j-1];
    a[j] = v;
  }
}

int pyse_events_frame(long n, const double *x, long nf, const int64_t *f,
                      int64_t **edges, long *ne, double *normal, unsigned char *folded)
{ long *start = NULL, *fill = NULL, t, v, k, m = 0;
  int64_t *hi = NULL, *e = NULL;
  double *area = NULL, *agree = NULL;
  int i;

  *edges = NULL;
  *ne = 0;
  for ( t = 0 ; t < 3*nf ; t++ )
    if ( f[t] < 0 || f[t] >= n ) return -2;
  /* the edges by their smaller end (a counting sort), each one's larger
     ends sorted and once: the order of a unique over the rows */
  start = (long *)calloc((size_t)n + 1,sizeof(long));
  fill = (long *)malloc(at_least_one(n)*sizeof(long));
  hi = (int64_t *)malloc(at_least_one(3*nf)*sizeof(int64_t));
  area = (double *)malloc(at_least_one(nf)*3*sizeof(double));
  agree = (double *)malloc(at_least_one(n)*sizeof(double));
  if ( !start || !fill || !hi || !area || !agree ) goto fail;
  for ( t = 0 ; t < nf ; t++ )
    for ( i = 0 ; i < 3 ; i++ )
    { int64_t a = f[3*t+i], b = f[3*t+(i+1)%3];
      start[(a < b ? a : b) + 1]++;
    }
  for ( v = 0 ; v < n ; v++ ) start[v+1] += start[v];
  memcpy(fill,start,(size_t)n*sizeof(long));
  for ( t = 0 ; t < nf ; t++ )
    for ( i = 0 ; i < 3 ; i++ )
    { int64_t a = f[3*t+i], b = f[3*t+(i+1)%3];
      if ( a < b ) hi[fill[a]++] = b;
      else hi[fill[b]++] = a;
    }
  for ( v = 0 ; v < n ; v++ )
  { long s = start[v], c = start[v+1] - s, j;
    sort_short(hi + s,c);
    for ( j = 0 ; j < c ; j++ )
      if ( j == 0 || hi[s+j] != hi[s+j-1] ) m++;
  }
  e = (int64_t *)malloc(at_least_one(m)*2*sizeof(int64_t));
  if ( !e ) goto fail;
  for ( v = 0, k = 0 ; v < n ; v++ )
  { long s = start[v], c = start[v+1] - s, j;
    for ( j = 0 ; j < c ; j++ )
      if ( j == 0 || hi[s+j] != hi[s+j-1] )
      { e[2*k] = v; e[2*k+1] = hi[s+j]; k++; }
  }

  /* normals: the facets' areas summed round each vertex, made unit */
  memset(normal,0,(size_t)n*3*sizeof(double));
  for ( t = 0 ; t < nf ; t++ )
  { const double *p0 = x + 3*f[3*t], *p1 = x + 3*f[3*t+1], *p2 = x + 3*f[3*t+2];
    double u[3], w[3], c[3];
    sub3(p1,p0,u);
    sub3(p2,p0,w);
    cross3(u,w,c);
    for ( i = 0 ; i < 3 ; i++ ) area[3*t+i] = 0.5*c[i];
  }
  for ( i = 0 ; i < 3 ; i++ )
    for ( t = 0 ; t < nf ; t++ )
    { double *nv = normal + 3*f[3*t+i];
      nv[0] += area[3*t]; nv[1] += area[3*t+1]; nv[2] += area[3*t+2];
    }
  for ( v = 0 ; v < n ; v++ )
  { double *nv = normal + 3*v, len = at_least(sqrt(dot3(nv,nv)),1e-300);
    nv[0] /= len; nv[1] /= len; nv[2] /= len;
  }
  /* in a fold: a facet round the vertex facing against its normal */
  for ( v = 0 ; v < n ; v++ ) agree[v] = 1.0;
  for ( t = 0 ; t < nf ; t++ )
  { double *a = area + 3*t, len = at_least(sqrt(dot3(a,a)),1e-300), unit[3];
    unit[0] = a[0]/len; unit[1] = a[1]/len; unit[2] = a[2]/len;
    for ( i = 0 ; i < 3 ; i++ )
    { int64_t w = f[3*t+i];
      agree[w] = nan_min(dot3(unit,normal + 3*w),agree[w]);
    }
  }
  for ( v = 0 ; v < n ; v++ ) folded[v] = agree[v] < 0;
  free(start); free(fill); free(hi); free(area); free(agree);
  *edges = e;
  *ne = m;
  return 0;
fail:
  free(start); free(fill); free(hi); free(area); free(agree); free(e);
  return -1;
}

/* ---- a uniform grid of points ------------------------------------------------ */

struct grid { double lo[3], cell; long n[3]; long *start; int64_t *items; };

static long grid_cell(const struct grid *g, const double *p, int d)
{ double c = floor((p[d] - g->lo[d])/g->cell);
  return !(c > 0) ? 0 : c > g->n[d] - 1 ? g->n[d] - 1 : (long)c;
}

static long grid_index(const struct grid *g, const double *p)
{ return (grid_cell(g,p,2)*g->n[1] + grid_cell(g,p,1))*g->n[0] + grid_cell(g,p,0); }

static void grid_free(struct grid *g)
{ free(g->start); free(g->items); g->start = NULL; g->items = NULL; }

/* the points (those with use[i] nonzero and finite) in cells of about
   cell, at most 128 a side; each cell's items increasing */
static int grid_make(struct grid *g, const double *p, long np, const unsigned char *use,
                     double cell)
{ double top[3] = { 0, 0, 0 };
  long i, ncells, *fill;
  int d, any = 0;
  for ( d = 0 ; d < 3 ; d++ ) g->lo[d] = 0.0;
  for ( i = 0 ; i < np ; i++ )
  { const double *q = p + 3*i;
    if ( !use[i] ) continue;
    for ( d = 0 ; d < 3 ; d++ )
    { if ( !any || q[d] < g->lo[d] ) g->lo[d] = q[d];
      if ( !any || q[d] > top[d] ) top[d] = q[d];
    }
    any = 1;
  }
  g->cell = cell > 0 && isfinite(cell) ? cell : 1.0;
  for ( d = 0 ; d < 3 ; d++ )
    if ( (top[d] - g->lo[d])/128 > g->cell ) g->cell = (top[d] - g->lo[d])/128;
  for ( d = 0 ; d < 3 ; d++ )
  { double cells = floor((top[d] - g->lo[d])/g->cell) + 1;
    g->n[d] = !(cells > 1) ? 1 : cells > 129 ? 129 : (long)cells;
  }
  ncells = g->n[0]*g->n[1]*g->n[2];
  g->start = (long *)calloc((size_t)ncells + 1,sizeof(long));
  g->items = (int64_t *)malloc(at_least_one(np)*sizeof(int64_t));
  fill = (long *)malloc((size_t)ncells*sizeof(long));
  if ( !g->start || !g->items || !fill ) { free(fill); grid_free(g); return -1; }
  for ( i = 0 ; i < np ; i++ )
    if ( use[i] ) g->start[grid_index(g,p + 3*i) + 1]++;
  for ( i = 0 ; i < ncells ; i++ ) g->start[i+1] += g->start[i];
  memcpy(fill,g->start,(size_t)ncells*sizeof(long));
  for ( i = 0 ; i < np ; i++ )
    if ( use[i] ) g->items[fill[grid_index(g,p + 3*i)]++] = i;
  free(fill);
  return 0;
}

/* the box of cells within r of p: lo[d] to hi[d] */
static void grid_box(const struct grid *g, const double *p, double r, long *lo, long *hi)
{ int d;
  for ( d = 0 ; d < 3 ; d++ )
  { double a = floor((p[d] - r - g->lo[d])/g->cell), b = floor((p[d] + r - g->lo[d])/g->cell);
    lo[d] = !(a > 0) ? 0 : a > g->n[d] - 1 ? g->n[d] - 1 : (long)a;
    hi[d] = !(b < g->n[d] - 1) ? g->n[d] - 1 : b < 0 ? 0 : (long)b;
  }
}

/* ---- the surface close to itself --------------------------------------------- */

/* The point of triangle (a, b, c) closest to p: the projection onto its
   plane if inside it, else the nearest of its sides' closest points (the
   first, as argmin picks). */
static void closest_on_triangle(const double *p, const double *a, const double *b,
                                const double *c, double *out)
{ const double *s[3], *e[3];
  double ab[3], ac[3], ap[3], nrm[3], t1[3], t2[3], nn, w0, w1, w2, best = 0.0;
  int d, j;
  sub3(b,a,ab);
  sub3(c,a,ac);
  cross3(ab,ac,nrm);
  nn = dot3(nrm,nrm);
  sub3(p,a,ap);
  cross3(ab,ap,t1);
  cross3(ap,ac,t2);
  w2 = dot3(t1,nrm)/at_least(nn,1e-300);
  w1 = dot3(t2,nrm)/at_least(nn,1e-300);
  w0 = 1 - w1 - w2;
  if ( w0 >= 0 && w1 >= 0 && w2 >= 0 && nn > 0 )
  { for ( d = 0 ; d < 3 ; d++ ) out[d] = a[d] + w1*ab[d] + w2*ac[d];
    return;
  }
  s[0] = a; s[1] = b; s[2] = c;
  e[0] = b; e[1] = c; e[2] = a;
  for ( j = 0 ; j < 3 ; j++ )
  { double st[3], q[3], k[3], dd[3], t, dist;
    sub3(e[j],s[j],st);
    sub3(p,s[j],q);
    t = clip01(dot3(q,st)/at_least(dot3(st,st),1e-300));
    for ( d = 0 ; d < 3 ; d++ ) k[d] = s[j][d] + t*st[d];
    sub3(p,k,dd);
    dist = dot3(dd,dd);
    if ( j == 0 || (best == best && (dist < best || dist != dist)) )
    { best = dist;
      memcpy(out,k,sizeof(k));
    }
  }
}

struct self_hit { int64_t v, t; double gap, c[3]; unsigned char neck; };

static int compare_hits(const void *a, const void *b)
{ int64_t s = ((const struct self_hit *)a)->t, t = ((const struct self_hit *)b)->t;
  return s < t ? -1 : s > t;
}

/* Per facet: its centroid, its farthest corner's distance from it, its
   side (its corners' normals summed, made unit: an inverted facet's own
   normal faces back into the surface) and whether it can be across a gap
   (no corner in a fold, a finite centroid). */
static void facet_frames(long nf, const int64_t *f, const double *x, const double *normal,
                         const unsigned char *folded, double *centre, double *radius, double *tn,
                         unsigned char *usable)
{ long t;
  for ( t = 0 ; t < nf ; t++ )
  { const int64_t *c = f + 3*t;
    double *m = centre + 3*t, *s = tn + 3*t, len, far = 0.0;
    int d, j;
    for ( d = 0 ; d < 3 ; d++ )
    { m[d] = (x[3*c[0]+d] + x[3*c[1]+d] + x[3*c[2]+d])/3;
      s[d] = normal[3*c[0]+d] + normal[3*c[1]+d] + normal[3*c[2]+d];
    }
    for ( j = 0 ; j < 3 ; j++ )
    { double u[3];
      sub3(x + 3*c[j],m,u);
      far = j ? nan_max(far,sqrt(dot3(u,u))) : sqrt(dot3(u,u));
    }
    radius[t] = far;
    len = at_least(sqrt(dot3(s,s)),1e-300);
    s[0] /= len; s[1] /= len; s[2] /= len;
    usable[t] = !folded[c[0]] && !folded[c[1]] && !folded[c[2]] && finite3(m) && far == far;
  }
}

/* One vertex's candidates: the facets whose centroid is within q (a box
   of the grid's cells), tested as _events._self did: not round the vertex,
   facing it, within reach (near) of their centroid, a corner facing it
   within R (the vertex pairs), the gap to their closest point within the
   vertex's r (necks neck_factor times it) and mostly along its normal, and
   no corner within two edges of it along the mesh. */
struct self_args
{ const double *x, *normal, *r, *centre, *radius, *tn;
  const double *gc;          /* the centroids in the grid's order */
  const int64_t *f, *behind, *ahead, *nbr;
  const long *nstart;
  const unsigned char *pmask;
  const struct grid *g;
  int with_bodies;
  double R, reach, neck_factor, rmax;
};

static int vertex_hits(const struct self_args *s, long w, long *stamp, struct buf *out)
{ const double *p = s->x + 3*w, *nw = s->normal + 3*w;
  double near = s->reach*s->r[w], q, q2, R2 = s->R*s->R;
  long lo[3], hi[3], b, c, first = out->count;
  int ringed = 0;
  if ( !finite3(p) || near != near ) return 0;
  q = (fmin(near,s->R) + s->rmax)*(1 + 1e-9);
  q2 = q*q;
  grid_box(s->g,p,q,lo,hi);
  /* a row of cells along x is one run of the grid's items, their
     centroids copied in that order: read straight through */
  for ( c = lo[2] ; c <= hi[2] ; c++ )
  for ( b = lo[1] ; b <= hi[1] ; b++ )
  { long row = (c*s->g->n[1] + b)*s->g->n[0], k, end = s->g->start[row + hi[0] + 1];
    for ( k = s->g->start[row + lo[0]] ; k < end ; k++ )
    { const double *m = s->gc + 3*k;
      int64_t t;
      const int64_t *fc;
      double u[3], cp[3], gap, along;
      int j, ok = 0, neck;
      struct self_hit hit;
      u[0] = m[0] - p[0]; u[1] = m[1] - p[1]; u[2] = m[2] - p[2];
      if ( dot3(u,u) > q2 ) continue;
      t = s->g->items[k];
      fc = s->f + 3*t;
      if ( fc[0] == w || fc[1] == w || fc[2] == w ) continue;
      if ( !(dot3(s->tn + 3*t,nw) < -0.5) ) continue;
      if ( !(sqrt(dot3(u,u)) - s->radius[t] < near) ) continue;
      for ( j = 0 ; j < 3 && !ok ; j++ )
      { int64_t o = fc[j];
        double dv[3];
        if ( !s->pmask[o] ) continue;
        sub3(p,s->x + 3*o,dv);
        ok = dot3(nw,s->normal + 3*o) < -0.5 && dot3(dv,dv) < R2;
      }
      if ( !ok ) continue;
      /* the gap; a neck when a body lies across (no bodies: behind) */
      closest_on_triangle(p,s->x + 3*fc[0],s->x + 3*fc[1],s->x + 3*fc[2],cp);
      sub3(cp,p,u);
      gap = sqrt(dot3(u,u));
      along = dot3(u,nw);
      neck = s->with_bodies ? (along < 0 ? s->behind[w] : s->ahead[w]) > 0 : along < 0;
      if ( !(gap < (neck ? s->neck_factor : 1.0)*s->r[w]) || !(fabs(along) > 0.5*gap) )
        continue;
      /* within two edges: a corner in the vertex's closed ring, or next
         to one */
      if ( !ringed )
      { long m;
        stamp[w] = w;
        for ( m = s->nstart[w] ; m < s->nstart[w+1] ; m++ ) stamp[s->nbr[m]] = w;
        ringed = 1;
      }
      for ( j = 0 ; j < 3 && ok ; j++ )
      { int64_t o = fc[j];
        long m;
        if ( stamp[o] == w ) ok = 0;
        for ( m = s->nstart[o] ; m < s->nstart[o+1] && ok ; m++ )
          if ( stamp[s->nbr[m]] == w ) ok = 0;
      }
      if ( !ok ) continue;
      hit.v = w; hit.t = t; hit.gap = gap; hit.neck = (unsigned char)neck;
      memcpy(hit.c,cp,sizeof(cp));
      if ( buf_push(out,&hit) ) return -1;
    }
  }
  /* its facets in increasing order, as the pairs came */
  if ( out->count - first > 1 )
    qsort(out->data + (size_t)first*out->size,(size_t)(out->count - first),out->size,
          compare_hits);
  return 0;
}

int pyse_events_self(long n, const double *x, long nf, const int64_t *f, long ne,
                     const int64_t *edges, const double *normal, const double *r,
                     const unsigned char *vmask, const unsigned char *pmask,
                     const unsigned char *folded, const int64_t *behind, const int64_t *ahead,
                     int with_bodies, double R, double reach, double neck_factor, int threads,
                     long *count, int64_t **vi, int64_t **ti, double **gap, double **closest,
                     unsigned char **neck)
{ struct self_args s;
  struct grid g = { { 0, 0, 0 }, 0, { 0, 0, 0 }, NULL, NULL };
  struct buf *chunks = NULL;
  double *centre = NULL, *radius = NULL, *tn = NULL, *gc = NULL, rmax = 0.0, qsum = 0.0;
  unsigned char *usable = NULL;
  long *nstart = NULL, *fill = NULL, t, v, k, nq = 0, nchunks = 0, total = 0, failed = 0;
  int64_t *nbr = NULL;
  const long CHUNK = 32;

  *count = 0;
  *vi = *ti = NULL; *gap = *closest = NULL; *neck = NULL;
  for ( t = 0 ; t < 3*nf ; t++ )
    if ( f[t] < 0 || f[t] >= n ) return -2;
  for ( k = 0 ; k < 2*ne ; k++ )
    if ( edges[k] < 0 || edges[k] >= n ) return -2;
  if ( n == 0 || nf == 0 || !(R > 0) ) return 0;
  centre = (double *)malloc((size_t)nf*3*sizeof(double));
  radius = (double *)malloc((size_t)nf*sizeof(double));
  tn = (double *)malloc((size_t)nf*3*sizeof(double));
  usable = (unsigned char *)malloc((size_t)nf);
  if ( !centre || !radius || !tn || !usable ) goto fail;
  facet_frames(nf,f,x,normal,folded,centre,radius,tn,usable);
  for ( t = 0 ; t < nf ; t++ )
    if ( usable[t] && radius[t] > rmax ) rmax = radius[t];
  /* a facet within reach of a vertex has its centroid within reach plus
     its radius, and a corner facing it within R: so its centroid within
     min(reach, R) plus the largest radius. The grid's cells about that */
  for ( v = 0 ; v < n ; v++ )
    if ( vmask[v] && pmask[v] && isfinite(r[v]) )
    { qsum += fmin(reach*r[v],R) + rmax;
      nq++;
    }
  if ( nq == 0 ) goto done;
  if ( grid_make(&g,centre,nf,usable,qsum/nq) ) goto fail;
  { long items = g.start[g.n[0]*g.n[1]*g.n[2]];
    gc = (double *)malloc(at_least_one(items)*3*sizeof(double));
    if ( !gc ) goto fail;
    for ( k = 0 ; k < items ; k++ ) memcpy(gc + 3*k,centre + 3*g.items[k],3*sizeof(double));
  }
  /* each vertex's neighbours along the mesh */
  nstart = (long *)calloc((size_t)n + 1,sizeof(long));
  fill = (long *)malloc((size_t)n*sizeof(long));
  nbr = (int64_t *)malloc(at_least_one(2*ne)*sizeof(int64_t));
  if ( !nstart || !fill || !nbr ) goto fail;
  for ( k = 0 ; k < ne ; k++ ) { nstart[edges[2*k]+1]++; nstart[edges[2*k+1]+1]++; }
  for ( v = 0 ; v < n ; v++ ) nstart[v+1] += nstart[v];
  memcpy(fill,nstart,(size_t)n*sizeof(long));
  for ( k = 0 ; k < ne ; k++ )
  { nbr[fill[edges[2*k]]++] = edges[2*k+1];
    nbr[fill[edges[2*k+1]]++] = edges[2*k];
  }

  s.x = x; s.normal = normal; s.r = r; s.centre = centre; s.radius = radius; s.tn = tn;
  s.gc = gc;
  s.f = f; s.behind = behind; s.ahead = ahead; s.nstart = nstart; s.nbr = nbr;
  s.pmask = pmask; s.g = &g; s.with_bodies = with_bodies; s.R = R; s.reach = reach;
  s.neck_factor = neck_factor; s.rmax = rmax;
  /* the vertices in chunks, each chunk's hits in a list of its own, the
     lists joined in order: the same pairs, in the same order, threads or
     not */
  nchunks = (n + CHUNK - 1)/CHUNK;
  chunks = (struct buf *)calloc((size_t)nchunks,sizeof(struct buf));
  if ( !chunks ) goto fail;
  for ( k = 0 ; k < nchunks ; k++ ) chunks[k].size = sizeof(struct self_hit);
#ifdef _OPENMP
  #pragma omp parallel num_threads(nq >= PAR_MIN_VERTICES && threads > 1 ? threads : 1) \
      reduction(+:failed)
#endif
  { /* a stamp per vertex: the ring of the vertex being looked at */
    long *stamp = (long *)malloc((size_t)n*sizeof(long)), ch, w;
    if ( !stamp ) failed++;
    else
    { for ( w = 0 ; w < n ; w++ ) stamp[w] = -1;
#ifdef _OPENMP
      #pragma omp for schedule(dynamic,4)
#endif
      for ( ch = 0 ; ch < nchunks ; ch++ )
        for ( w = ch*CHUNK ; w < n && w < (ch+1)*CHUNK ; w++ )
          if ( vmask[w] && pmask[w] && vertex_hits(&s,w,stamp,chunks + ch) )
          { failed++;
            break;
          }
      free(stamp);
    }
  }
  if ( failed ) goto fail;
  for ( k = 0 ; k < nchunks ; k++ ) total += chunks[k].count;
  if ( total )
  { long at = 0;
    *vi = (int64_t *)malloc((size_t)total*sizeof(int64_t));
    *ti = (int64_t *)malloc((size_t)total*sizeof(int64_t));
    *gap = (double *)malloc((size_t)total*sizeof(double));
    *closest = (double *)malloc((size_t)total*3*sizeof(double));
    *neck = (unsigned char *)malloc((size_t)total);
    if ( !*vi || !*ti || !*gap || !*closest || !*neck ) goto fail;
    for ( k = 0 ; k < nchunks ; k++ )
    { const struct self_hit *h = (const struct self_hit *)chunks[k].data;
      long j;
      for ( j = 0 ; j < chunks[k].count ; j++, at++ )
      { (*vi)[at] = h[j].v; (*ti)[at] = h[j].t; (*gap)[at] = h[j].gap;
        memcpy(*closest + 3*at,h[j].c,3*sizeof(double));
        (*neck)[at] = h[j].neck;
      }
    }
  }
  *count = total;
done:
  if ( chunks ) for ( k = 0 ; k < nchunks ; k++ ) free(chunks[k].data);
  free(chunks); free(centre); free(radius); free(tn); free(gc); free(usable); free(nstart);
  free(fill); free(nbr);
  grid_free(&g);
  return 0;
fail:
  free(*vi); free(*ti); free(*gap); free(*closest); free(*neck);
  *vi = *ti = NULL; *gap = *closest = NULL; *neck = NULL;
  *count = 0;
  if ( chunks ) for ( k = 0 ; k < nchunks ; k++ ) free(chunks[k].data);
  free(chunks); free(centre); free(radius); free(tn); free(gc); free(usable); free(nstart);
  free(fill); free(nbr);
  grid_free(&g);
  return -1;
}

/* ---- contact lines close to each other ------------------------------------------ */

/* NumPy's sum of a float array (pairwise, in blocks of eight): a closed
   line's length, which a pair's distance along it is compared with */
static double pairwise_sum(const double *a, long n)
{ if ( n < 8 )
  { double s = 0.0;
    long i;
    for ( i = 0 ; i < n ; i++ ) s += a[i];
    return s;
  }
  if ( n <= 128 )
  { double r[8], s;
    long i;
    int j;
    for ( j = 0 ; j < 8 ; j++ ) r[j] = a[j];
    for ( i = 8 ; i < n - n%8 ; i += 8 )
      for ( j = 0 ; j < 8 ; j++ ) r[j] += a[i+j];
    s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for ( ; i < n ; i++ ) s += a[i];
    return s;
  }
  { long n2 = n/2;
    n2 -= n2%8;
    return pairwise_sum(a,n2) + pairwise_sum(a + n2,n - n2);
  }
}

static int compare_ids(const void *a, const void *b)
{ int64_t s = *(const int64_t *)a, t = *(const int64_t *)b;
  return s < t ? -1 : s > t;
}

/* the position of key among the sorted keys (it is there) */
static long key_slot(const int64_t *keys, long n, int64_t key)
{ long lo = 0, hi = n;
  while ( lo < hi )
  { long mid = lo + (hi - lo)/2;
    if ( keys[mid] < key ) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

static int64_t pair_key(int64_t u, int64_t w, long n)
{ return (u < w ? u : w)*(int64_t)n + (u < w ? w : u); }

struct path { long start, len, index; };

static int compare_paths(const void *a, const void *b)
{ const struct path *p = (const struct path *)a, *q = (const struct path *)b;
  if ( p->len != q->len ) return p->len > q->len ? -1 : 1;     /* longest first */
  return p->index < q->index ? -1 : p->index > q->index;      /* stable */
}

struct sweep { double x; int64_t i; };

static int compare_sweep(const void *a, const void *b)
{ const struct sweep *p = (const struct sweep *)a, *q = (const struct sweep *)b;
  if ( p->x != q->x ) return p->x < q->x ? -1 : 1;
  return p->i < q->i ? -1 : p->i > q->i;
}

struct pair { int64_t i, j; };

static int compare_pairs(const void *a, const void *b)
{ const struct pair *p = (const struct pair *)a, *q = (const struct pair *)b;
  if ( p->i != q->i ) return p->i < q->i ? -1 : 1;
  return p->j < q->j ? -1 : p->j > q->j;
}

int pyse_events_lines(long n, const double *x, const double *h, long ne, const int64_t *edges,
                      double factor, long *nl, int64_t **line_vertices, long *np,
                      int64_t **pi, int64_t **pj, double **pd, double **pscale)
{ long *slot = NULL, *astart = NULL, *apos = NULL, *line = NULL;
  int64_t *order = NULL, *adj = NULL, *keys = NULL, *verts = NULL, *lv = NULL;
  unsigned char *used = NULL, *closed = NULL;
  double *s = NULL, *total = NULL, *seg = NULL;
  struct path *paths = NULL;
  struct sweep *sw = NULL;
  struct buf pairs = { NULL, 0, 0, sizeof(struct pair) };
  long nv = 0, k, nkeys = 0, npaths = 0, nverts = 0, L = 0, kept = 0, pass;
  double hmax = 0.0, rr;

  *nl = 0; *np = 0;
  *line_vertices = *pi = *pj = NULL; *pd = *pscale = NULL;
  for ( k = 0 ; k < 2*ne ; k++ )
    if ( edges[k] < 0 || edges[k] >= n ) return -2;
  if ( ne == 0 ) return 0;
  /* _geometry._chains as it goes: the vertices in the order they first
     come, each one's neighbours in the order of the edges */
  slot = (long *)malloc((size_t)n*sizeof(long));
  order = (int64_t *)malloc((size_t)2*ne*sizeof(int64_t));
  astart = (long *)calloc((size_t)2*ne + 1,sizeof(long));
  apos = (long *)malloc((size_t)2*ne*sizeof(long));
  adj = (int64_t *)malloc((size_t)2*ne*sizeof(int64_t));
  keys = (int64_t *)malloc((size_t)ne*sizeof(int64_t));
  used = (unsigned char *)calloc((size_t)ne,1);
  verts = (int64_t *)malloc((size_t)2*ne*sizeof(int64_t));
  paths = (struct path *)malloc((size_t)ne*sizeof(struct path));
  if ( !slot || !order || !astart || !apos || !adj || !keys || !used || !verts || !paths )
    goto fail;
  for ( k = 0 ; k < n ; k++ ) slot[k] = -1;
  for ( k = 0 ; k < 2*ne ; k++ )
  { int64_t u = edges[k];
    if ( slot[u] < 0 ) { slot[u] = nv; order[nv++] = u; }
    astart[slot[u]+1]++;
  }
  for ( k = 0 ; k < nv ; k++ ) astart[k+1] += astart[k];
  memcpy(apos,astart,(size_t)nv*sizeof(long));
  for ( k = 0 ; k < ne ; k++ )
  { int64_t u = edges[2*k], w = edges[2*k+1];
    adj[apos[slot[u]]++] = w;
    adj[apos[slot[w]]++] = u;
  }
  /* the edges walked, by their ends */
  for ( k = 0 ; k < ne ; k++ ) keys[k] = pair_key(edges[2*k],edges[2*k+1],n);
  qsort(keys,(size_t)ne,sizeof(int64_t),compare_ids);
  for ( k = 0 ; k < ne ; k++ )
    if ( k == 0 || keys[k] != keys[nkeys-1] ) keys[nkeys++] = keys[k];
  /* paths from the ends and junctions first (not two neighbours), then
     from any vertex (closed loops); each to a junction or back to its
     start */
  for ( pass = 0 ; pass < 2 ; pass++ )
    for ( k = 0 ; k < nv ; k++ )
    { long m;
      int64_t start = order[k];
      if ( pass == 0 && astart[k+1] - astart[k] == 2 ) continue;
      for ( m = astart[k] ; m < astart[k+1] ; m++ )
      { int64_t prev = start, cur = adj[m];
        long at = key_slot(keys,nkeys,pair_key(start,cur,n));
        if ( used[at] ) continue;
        used[at] = 1;
        paths[npaths].start = nverts;
        paths[npaths].index = npaths;
        verts[nverts++] = start;
        for ( ;; )
        { long c = slot[cur];
          int64_t follow;
          verts[nverts++] = cur;
          if ( astart[c+1] - astart[c] != 2 || cur == start ) break;
          follow = adj[astart[c]+1] == prev ? adj[astart[c]] : adj[astart[c]+1];
          at = key_slot(keys,nkeys,pair_key(cur,follow,n));
          if ( used[at] ) break;
          used[at] = 1;
          prev = cur;
          cur = follow;
        }
        paths[npaths].len = nverts - paths[npaths].start;
        npaths++;
      }
    }
  qsort(paths,(size_t)npaths,sizeof(struct path),compare_paths);

  /* each line's vertices with their distance along it from its start (a
     loop's first vertex once), its length, whether it is closed */
  lv = (int64_t *)malloc(at_least_one(nverts)*sizeof(int64_t));
  s = (double *)malloc(at_least_one(nverts)*sizeof(double));
  total = (double *)malloc(at_least_one(nverts)*sizeof(double));
  seg = (double *)malloc(at_least_one(nverts)*sizeof(double));
  closed = (unsigned char *)malloc(at_least_one(nverts));
  line = (long *)malloc(at_least_one(nverts)*sizeof(long));
  if ( !lv || !s || !total || !seg || !closed || !line ) goto fail;
  for ( k = 0 ; k < npaths ; k++ )
  { const int64_t *p = verts + paths[k].start;
    long len = paths[k].len, j, keep;
    int loop = len > 2 && p[0] == p[len-1];
    double cum = 0.0, length;
    for ( j = 0 ; j + 1 < len ; j++ )
    { double d[3];
      sub3(x + 3*p[j+1],x + 3*p[j],d);
      seg[j] = sqrt(dot3(d,d));
    }
    length = pairwise_sum(seg,len - 1);
    keep = loop ? len - 1 : len;
    for ( j = 0 ; j < keep ; j++ )
    { lv[L] = p[j]; s[L] = cum; total[L] = length; closed[L] = (unsigned char)loop;
      line[L] = k;
      hmax = L ? nan_max(hmax,h[p[j]]) : h[p[j]];
      L++;
      if ( j + 1 < len ) cum += seg[j];
    }
  }

  /* pairs closer than factor times the largest edge there (sorted along
     x, each against those after it within that), in order (i, j); kept:
     closer than factor times their own edges, and far apart along the
     lines (more than three times that or their distance) or on two */
  rr = factor*hmax;
  if ( L >= 2 && rr > 0 )
  { long a, nsw = 0;
    sw = (struct sweep *)malloc((size_t)L*sizeof(struct sweep));
    if ( !sw ) goto fail;
    for ( a = 0 ; a < L ; a++ )
      if ( finite3(x + 3*lv[a]) ) { sw[nsw].x = x[3*lv[a]]; sw[nsw].i = a; nsw++; }
    qsort(sw,(size_t)nsw,sizeof(struct sweep),compare_sweep);
    for ( a = 0 ; a < nsw ; a++ )
    { long b;
      for ( b = a + 1 ; b < nsw && sw[b].x - sw[a].x <= rr*(1 + 1e-9) ; b++ )
      { struct pair pr;
        double d[3];
        pr.i = sw[a].i < sw[b].i ? sw[a].i : sw[b].i;
        pr.j = sw[a].i < sw[b].i ? sw[b].i : sw[a].i;
        sub3(x + 3*lv[pr.i],x + 3*lv[pr.j],d);
        if ( sqrt(dot3(d,d)) < rr && buf_push(&pairs,&pr) ) goto fail;
      }
    }
    if ( pairs.count > 1 )
      qsort(pairs.data,(size_t)pairs.count,sizeof(struct pair),compare_pairs);
  }
  if ( pairs.count )
  { const struct pair *pr = (const struct pair *)pairs.data;
    *pi = (int64_t *)malloc((size_t)pairs.count*sizeof(int64_t));
    *pj = (int64_t *)malloc((size_t)pairs.count*sizeof(int64_t));
    *pd = (double *)malloc((size_t)pairs.count*sizeof(double));
    *pscale = (double *)malloc((size_t)pairs.count*sizeof(double));
    if ( !*pi || !*pj || !*pd || !*pscale ) goto fail;
    for ( k = 0 ; k < pairs.count ; k++ )
    { int64_t i = pr[k].i, j = pr[k].j;
      double d[3], dist, scale, apart;
      sub3(x + 3*lv[i],x + 3*lv[j],d);
      dist = sqrt(dot3(d,d));
      scale = 0.5*(h[lv[i]] + h[lv[j]]);
      if ( !(dist < factor*scale) ) continue;
      apart = fabs(s[i] - s[j]);
      if ( closed[i] ) apart = nan_min(apart,total[i] - apart);
      if ( !(line[i] != line[j] || apart > 3*nan_max(dist,scale)) ) continue;
      (*pi)[kept] = i; (*pj)[kept] = j; (*pd)[kept] = dist; (*pscale)[kept] = scale;
      kept++;
    }
  }
  free(slot); free(order); free(astart); free(apos); free(adj); free(keys); free(used);
  free(verts); free(paths); free(s); free(total); free(seg); free(closed); free(line);
  free(sw); free(pairs.data);
  *line_vertices = lv;
  *nl = L;
  *np = kept;
  return 0;
fail:
  free(slot); free(order); free(astart); free(apos); free(adj); free(keys); free(used);
  free(verts); free(paths); free(s); free(total); free(seg); free(closed); free(line);
  free(sw); free(pairs.data); free(lv);
  free(*pi); free(*pj); free(*pd); free(*pscale);
  *pi = *pj = NULL; *pd = *pscale = NULL;
  return -1;
}

/* ---- distances to segments ---------------------------------------------------- */

/* a point's squared distance to segment j (a + t ab, ab and |ab|^2 in ab) */
static double segment_d2(const double *p, const double *a, const double *ab, long j)
{ const double *e = ab + 4*j;
  double q[3], d[3], t;
  sub3(p,a + 3*j,q);
  t = clip01(dot3(q,e)/e[3]);
  d[0] = q[0] - t*e[0]; d[1] = q[1] - t*e[1]; d[2] = q[2] - t*e[2];
  return dot3(d,d);
}

/* The segments in a tree of boxes: each node the box round its segments,
   split at the median of their middles along its longest side, down to
   SEG_LEAF a leaf. A point's nearest segment is then found in the boxes
   closer than the nearest so far: the same minimum as over all of them.
   (Over all of them for every point, a 30k-facet pack's search spent 35 ms
   of its 130 here: its 5k points close to a wall against the contact
   lines' 1.3k edges, per wall.) */
#define SEG_LEAF 4
#define SEG_STACK 128

struct seg_node { double lo[3], hi[3]; long left, start, count; };  /* left < 0: a leaf */

/* the k-th smallest of key[idx[lo..hi]] put at lo + k, smaller before it */
static void select_k(long *idx, const double *key, long lo, long hi, long k)
{ k += lo;
  while ( hi > lo )
  { double pivot = key[idx[(lo + hi)/2]];
    long i = lo, j = hi;
    while ( i <= j )
    { while ( key[idx[i]] < pivot ) i++;
      while ( key[idx[j]] > pivot ) j--;
      if ( i <= j ) { long t = idx[i]; idx[i] = idx[j]; idx[j] = t; i++; j--; }
    }
    if ( k <= j ) hi = j;
    else if ( k >= i ) lo = i;
    else return;
  }
}

static long seg_tree(long ns, const double *a, const double *b, long *idx,
                     struct seg_node *node, double *key)
{ long count = 1, stack[SEG_STACK], top = 0, k;
  for ( k = 0 ; k < ns ; k++ ) idx[k] = k;
  node[0].start = 0; node[0].count = ns;
  stack[top++] = 0;
  while ( top )
  { struct seg_node *nd = node + stack[--top];
    int c, axis = 0;
    for ( c = 0 ; c < 3 ; c++ ) { nd->lo[c] = INFINITY; nd->hi[c] = -INFINITY; }
    for ( k = nd->start ; k < nd->start + nd->count ; k++ )
      for ( c = 0 ; c < 3 ; c++ )
      { double u = a[3*idx[k]+c], v = b[3*idx[k]+c];
        if ( u < nd->lo[c] ) nd->lo[c] = u;
        if ( v < nd->lo[c] ) nd->lo[c] = v;
        if ( u > nd->hi[c] ) nd->hi[c] = u;
        if ( v > nd->hi[c] ) nd->hi[c] = v;
      }
    nd->left = -1;
    if ( nd->count <= SEG_LEAF || top + 2 > SEG_STACK ) continue;
    for ( c = 1 ; c < 3 ; c++ )
      if ( nd->hi[c] - nd->lo[c] > nd->hi[axis] - nd->lo[axis] ) axis = c;
    for ( k = nd->start ; k < nd->start + nd->count ; k++ )
      key[idx[k]] = a[3*idx[k]+axis] + b[3*idx[k]+axis];
    select_k(idx,key,nd->start,nd->start + nd->count - 1,nd->count/2);
    nd->left = count;
    node[count].start = nd->start; node[count].count = nd->count/2;
    node[count+1].start = nd->start + nd->count/2;
    node[count+1].count = nd->count - nd->count/2;
    stack[top++] = count;
    stack[top++] = count + 1;
    count += 2;
  }
  return count;
}

static double box_d2(const struct seg_node *nd, const double *p)
{ double s = 0.0;
  int c;
  for ( c = 0 ; c < 3 ; c++ )
  { double d = p[c] < nd->lo[c] ? nd->lo[c] - p[c] : p[c] > nd->hi[c] ? p[c] - nd->hi[c] : 0.0;
    s += d*d;
  }
  return s;
}

static double seg_nearest_d2(const struct seg_node *node, const long *idx, const double *p,
                             const double *a, const double *ab)
{ long stack[SEG_STACK], top = 0, k;
  double best = INFINITY;
  stack[top++] = 0;
  while ( top )
  { const struct seg_node *nd = node + stack[--top];
    if ( !(box_d2(nd,p) < best) ) continue;
    if ( nd->left < 0 || top + 2 > SEG_STACK )   /* (a leaf; or all its segments) */
    { for ( k = nd->start ; k < nd->start + nd->count ; k++ )
      { double d2 = segment_d2(p,a,ab,idx[k]);
        if ( d2 < best ) best = d2;
      }
      continue;
    }
    { double dl = box_d2(node + nd->left,p), dr = box_d2(node + nd->left + 1,p);
      long near = dl <= dr ? nd->left : nd->left + 1;   /* the nearer looked in first */
      stack[top++] = near == nd->left ? nd->left + 1 : nd->left;
      stack[top++] = near;
    }
  }
  return best;
}

int pyse_events_segments(long np, const double *p, long ns, const double *a, const double *b,
                         int threads, double *out)
{ double *ab = (double *)malloc(at_least_one(ns)*4*sizeof(double));
  long *idx = NULL, i, k;
  struct seg_node *node = NULL;
  double *key = NULL;
  int finite = 1;
  if ( !ab ) return -1;
  /* each segment's direction and its length squared */
  for ( k = 0 ; k < ns ; k++ )
  { sub3(b + 3*k,a + 3*k,ab + 4*k);
    ab[4*k+3] = at_least(dot3(ab + 4*k,ab + 4*k),1e-300);
    finite = finite && finite3(a + 3*k) && finite3(b + 3*k);
  }
  /* (the tree for more than a few, all finite: else each point against
     all of them, as before, NaN for a NaN among them) */
  if ( finite && ns > 2*SEG_LEAF )
  { idx = (long *)malloc((size_t)ns*sizeof(long));
    node = (struct seg_node *)malloc((size_t)2*ns*sizeof(struct seg_node));
    key = (double *)malloc((size_t)ns*sizeof(double));
    if ( !idx || !node || !key ) { free(ab); free(idx); free(node); free(key); return -1; }
    seg_tree(ns,a,b,idx,node,key);
  }
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) \
      num_threads((double)np*(node ? 16 : ns) >= PAR_MIN_WORK && threads > 1 ? threads : 1)
#endif
  for ( i = 0 ; i < np ; i++ )
  { double best = INFINITY;
    long j;
    if ( node && finite3(p + 3*i) )
      best = seg_nearest_d2(node,idx,p + 3*i,a,ab);
    else
      for ( j = 0 ; j < ns ; j++ )
        best = nan_min(segment_d2(p + 3*i,a,ab,j),best);
    out[i] = sqrt(best);
  }
  free(ab); free(idx); free(node); free(key);
  return 0;
}

/* ---- groups ----------------------------------------------------------------------- */

int pyse_events_groups(long n, const unsigned char *mask, long nl, const int64_t *links,
                       int64_t *label)
{ long v, k;
  for ( k = 0 ; k < 2*nl ; k++ )
    if ( links[k] < 0 || links[k] >= n ) return -2;
  /* union-find, each root its group's smallest member */
  for ( v = 0 ; v < n ; v++ ) label[v] = mask[v] ? v : -1;
  for ( k = 0 ; k < nl ; k++ )
  { int64_t a = links[2*k], b = links[2*k+1];
    if ( !mask[a] || !mask[b] ) continue;
    while ( label[a] != a ) a = label[a] = label[label[a]];
    while ( label[b] != b ) b = label[b] = label[label[b]];
    if ( a < b ) label[b] = a;
    else if ( b < a ) label[a] = b;
  }
  /* (a root is smaller than all below it: one pass in increasing order) */
  for ( v = 0 ; v < n ; v++ )
    if ( label[v] >= 0 ) label[v] = label[label[v]];
  return 0;
}
