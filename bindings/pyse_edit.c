/*
 * pyse_edit.c -- local mesh edits for pySE's changes of topology and
 * remeshing, each checked: refused before it changes anything when its
 * rules or its predicted shape say no, and undone when the surface round it
 * comes out worse than before (the topology check's problems, counted on
 * the patch before and after: structure, pinches, degenerate facets, folds,
 * vertices or edges in a solid wall, edges through facets).
 *
 * A batch of one kind of edit takes a snapshot first; an edit that comes
 * out worse puts it back, and the batch's edits made before it are made
 * again (the same state, so the same results), that one refused. Nothing
 * else restores, so the snapshot costs only when an edit fails afterwards.
 *
 * The edits (linear soap films in 3D):
 *   collapse  an edge to a point: Evolver's delete_edge (its shape checks),
 *             our rules first: the strict link condition, a contact line's vertex
 *             only along the line, constraints kept, no facet turned more
 *             than max_turn, edges not longer than max_len, the volume swept
 *             under max_volume;
 *   flip      an edge between two facets (Evolver's edgeswap): orientation,
 *             no new chord between two contact lines' vertices, the new
 *             facets turned at most max_turn from the old, the volume
 *             between them under max_volume; mode 1 only where it makes the
 *             facets more equiangular, mode 2 only where it brings the
 *             vertices' valences nearer 6 (4 on a border);
 *   split     an edge at its midpoint (Evolver's edge_refine), projected; a
 *             fixed edge stays;
 *   smooth    a vertex toward its neighbours, along the surface (in the
 *             tangent plane; a contact line's vertex along the line), by
 *             lambda, at most a quarter edge, projected onto its
 *             constraints; one with more than one constraint, fixed or on a
 *             boundary stays;
 *   snap      two vertices merged into one (Evolver's vertex_merge) where
 *             their stars share no vertex and their constraints agree;
 *   separate  a vertex where several fans of facets meet: one vertex per
 *             fan, moved offset edges toward its fan;
 *   degenerate a facet with next to no area: its shortest edge collapsed
 *             (a needle) or its longest flipped (a cap); twin facets (the
 *             same three edges) both dissolved;
 *   unzip     an edge between two facets whose ends are on one wall's
 *             contact lines made two contact-line edges, one per facet, and
 *             its ends separated, one vertex per fan.
 */

#include "include.h"
#include "pyse_api.h"
#include "pyse_internal.h"

/* ---- id lists and marks ------------------------------------------------- */

struct ids { element_id *a; int n, max; };

static void push(struct ids *l, element_id id)
{ if ( l->n == l->max )
  { int m = l->max ? 2*l->max : 64;
    l->a = l->a ? (element_id *)temp_realloc((char *)l->a,m*sizeof(element_id))
                : (element_id *)temp_calloc(m,sizeof(element_id));
    l->max = m;
  }
  l->a[l->n++] = id;
}

/* an element in a grid's cell: its ordinal and a box round it (where it
   was put in, padded: it stays inside while the element is unchanged, so a
   query passes over most candidates by these without going to the element,
   whose facetedges' pointer-chasing was a third of a remesh's time) */
struct gitem { float lo[3], hi[3]; int ord; };

/* elements by the cells of side cell their boxes are in, the cells kept by
   a hash of their numbers (only those with something in them: a dense
   array of the box's cells, capped at 128 a side, was mostly empty and
   cleared at every build, and at 50k facets its cells grew past h, each
   query's candidates with them). Per cell, the items put in when the grid
   was built and a list (through xnext, from xhead, -1 none) of those of
   elements made or changed since. In a slab, x and y are periodic: cells
   of sides pcell dividing the periods, their numbers taken round them;
   each element's box at the image of its first vertex, a query's matched
   to an item's image (see apart). */
struct grid
{ REAL lo[3], cell, pcell[2];
  long n[3];
  int per;
  long cap, used, *key, *bstart, *xhead;    /* cells by hash: key -1, an empty slot */
  int *bcount;
  struct gitem *items, *xitems;
  long total, *xnext, xn, xmax;
  int built;
};

/* the edges and facets of a patch before an edit (each with its
   vertices and where they were), to tell after it which it changed */
struct sig { element_id id, v[3]; REAL x[9]; };

struct ctx
{ const int *walls, *sides;
  int nwalls;
  int manifold;
  double h, tol, cos_turn, max_volume, max_len, lambda, offset, flat;
  int mode;
  int fold_ok;                     /* (a sheet's hole: folds may grow, see merge_facets) */
  int *stamp[NUMELEMENTS];
  long stamp_size[NUMELEMENTS];
  int stamp_now;
  struct grid fgrid, egrid;        /* facets and edges */
  REAL pad;                        /* the grids' boxes padded by this */
  int epoch;                       /* the grids' builds */
  int *moved;                      /* per vertex: the build it was smoothed in */
  long moved_size;
  struct sig *sigs;                /* the patch before the edit: see before() */
  int nsigs, maxsigs, sig_now;
  int *sig_at[NUMELEMENTS];        /* per edge, facet: its sig's place in sigs, */
  int *sig_ep[NUMELEMENTS];        /* when this is sig_now */
  long sig_size[NUMELEMENTS];
  REAL *wref;                      /* in a slab, per wall: a vertex on it, where it was */
  int *wref_ok;                    /* (none: 0) */
  long nper[2];                    /* in a slab: near_key's cells a period */
  REAL *gaps;                      /* per vertex and wall: x, y, z, the gap there */
  long gaps_size;                  /* vertex ordinals it has room for */
  REAL *reach;                     /* per vertex: x, y, z, vertex_reach there */
  long reach_size;
  REAL box_lo[3], box_hi[3];
  int clean;                       /* no crossings when the remesh began: see patch() */
  int in_cells;                    /* a smoothing batch: its moves leave elements in their cells */
  int fresh;                       /* its grids built for it (else its moves put their elements in) */
  int readd;                       /* this move's elements put in again all the same */
  struct snapshot *snap;           /* the batch's: see before() */
  int snapping, snap_k, cur_k;     /* the batch takes them; the edit it was taken for; now */
  struct edit_args *batch;         /* the batch's, its keys taken when its first edit is made */
  int grids_due;                   /* the grids to be built when first needed, padded so */
  REAL due_pad;
  int near_on;                     /* a remesh near a change, clean near it: see near_clean() */
  long ncap, *nkey;                /* the cells near its region (side h), by hash: key -1 none */
  int *vok, vok_now;               /* per vertex: its elements found clean (near_verify), */
  long vok_size;                   /* when this is vok_now */
  int changed;                     /* an edit made (or undone) */
  REAL cell_h;                     /* the grids' cells, times h (0: 1) */
  int fcache;                      /* facets' vertices kept in fv (a remesh): see fverts() */
  vertex_id *fv;                   /* per facet ordinal: its three vertices */
  long fv_size;
};

/* a fresh mark for this pass; 1 if id wasn't marked yet */
static int mark(struct ctx *c, element_id id)
{ int type = id_type(id);
  long ord = ordinal(id);
  if ( !valid_id(id) || ord < 0 ) return 0;     /* (a freed element's: NULLID) */
  if ( ord >= c->stamp_size[type] )
  { long m = 2*ord + 64, i;
    c->stamp[type] = c->stamp[type] ? (int *)temp_realloc((char *)c->stamp[type],m*sizeof(int))
                                    : (int *)temp_calloc(m,sizeof(int));
    for ( i = c->stamp_size[type] ; i < m ; i++ ) c->stamp[type][i] = 0;
    c->stamp_size[type] = m;
  }
  if ( c->stamp[type][ord] == c->stamp_now ) return 0;
  c->stamp[type][ord] = c->stamp_now;
  return 1;
}

/* ---- adjacency ---------------------------------------------------------- */

#define EACH_EDGE_AT(v,e,e0,guard) \
  for ( e0 = get_vertex_edge(v), e = e0, guard = 0 ; valid_id(e) && guard < 100000 ; \
        e = get_next_tail_edge(e), guard = equal_id(e,e0) ? 100000 : guard + 1 )

static int valence(edge_id e)
{ facetedge_id fe0 = get_edge_fe(e), fe = fe0;
  int n = 0;
  if ( !valid_id(fe0) ) return 0;
  do { n++; fe = get_next_facet(fe); } while ( valid_id(fe) && !equal_id(fe,fe0) && n < 1000 );
  return n;
}

static void facet_verts(facet_id f, vertex_id *v)
{ facetedge_id fe = get_facet_fe(positive_id(f));
  int k;
  for ( k = 0 ; k < 3 ; k++, fe = get_next_edge(fe) ) v[k] = get_fe_tailv(fe);
}

static int on_border(vertex_id v)
{ edge_id e, e0;
  int guard;
  EACH_EDGE_AT(v,e,e0,guard)
    if ( valence(e) == 1 ) return 1;
  return 0;
}

static int joined(vertex_id a, vertex_id b)
{ edge_id e, e0;
  int guard;
  EACH_EDGE_AT(a,e,e0,guard)
    if ( equal_id(get_edge_headv(e),b) ) return 1;
  return 0;
}

static int common_wall(vertex_id a, vertex_id b)
{ conmap_t *map;
  unsigned int m;
  if ( !(get_vattr(a) & CONSTRAINT) || !(get_vattr(b) & CONSTRAINT) ) return 0;
  map = get_v_constraint_map(a);
  for ( m = 1 ; m <= map[0] ; m++ )
    if ( v_on_constraint(b,(int)(map[m] & CONMASK)) ) return 1;
  return 0;
}

static int constraint_count(vertex_id v)
{ return (get_vattr(v) & CONSTRAINT) ? (int)get_v_constraint_map(v)[0] : 0; }

/* whether v, put onto wall too, would be on two constraints whose normals
   there are within 10 degrees of each other (a bead touching a side wall:
   a vertex on both at the touching point left Evolver's constraint
   projection singular, ERROR 3005); at, where v goes onto it (wall_image) */
static int tangent_walls(vertex_id v, int wall, REAL *at)
{ conmap_t *map;
  unsigned int m;
  REAL *x = get_coord(v), f, a[MAXCOORD], b[MAXCOORD], aa = 0, bb = 0, ab = 0;
  int i;
  if ( !(get_vattr(v) & CONSTRAINT) ) return 0;
  eval_all(get_constraint(wall)->formula,at,SDIM,&f,a,v);
  map = get_v_constraint_map(v);
  for ( m = 1 ; m <= map[0] ; m++ )
  { int o = (int)(map[m] & CONMASK);
    if ( o == wall ) continue;
    eval_all(get_constraint(o)->formula,x,SDIM,&f,b,v);
    aa = bb = ab = 0;
    for ( i = 0 ; i < SDIM ; i++ ) { aa += a[i]*a[i]; bb += b[i]*b[i]; ab += a[i]*b[i]; }
    if ( aa > 0 && bb > 0 && fabs(ab) > 0.985*sqrt(aa*bb) ) return 1;
  }
  return 0;
}

static void neighbours(vertex_id v, struct ids *out)
{ edge_id e, e0;
  int guard;
  EACH_EDGE_AT(v,e,e0,guard) push(out,get_edge_headv(e));
}

static void tri_normal(const REAL *a, const REAL *b, const REAL *c, REAL *n)
{ REAL u[3], w[3];
  int i;
  for ( i = 0 ; i < 3 ; i++ ) { u[i] = b[i] - a[i]; w[i] = c[i] - a[i]; }
  n[0] = u[1]*w[2] - u[2]*w[1];
  n[1] = u[2]*w[0] - u[0]*w[2];
  n[2] = u[0]*w[1] - u[1]*w[0];
}

static REAL dot3(const REAL *a, const REAL *b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static REAL norm3(const REAL *a) { return sqrt(dot3(a,a)); }

static REAL dist3(const REAL *a, const REAL *b)
{ REAL d[3] = { a[0]-b[0], a[1]-b[1], a[2]-b[2] };
  return norm3(d);
}

static void project(vertex_id v)
{ if ( get_vattr(v) & CONSTRAINT ) project_v_constr(v,ACTUAL_MOVE,KEEP_ONESIDEDNESS);
}

/* ---- slabs periodic in x and y ------------------------------------------ */

/* (pyse_check.c) */
void pyse_image(const REAL *ref, const REAL *x, REAL *out);
double pyse_slab_gap(int c, double side, const REAL *ref, const REAL *x, REAL *at);

/* A slab (pyse_slab): coordinates aren't wrapped into the cell and every
   edge is short, so the geometry of a few nearby vertices is taken with
   them at one image, each nearest a reference point (one of them); a new
   position for a vertex is written at its own image. Set as each call
   begins. */
static int slab_on;
static REAL slab_L[2];

static void slab_set(void) { slab_on = web.torus_flag && pyse_slab(slab_L); }

/* x's image nearest ref (x itself outside a slab) */
static inline void img(const REAL *ref, const REAL *x, REAL *out)
{ out[0] = x[0]; out[1] = x[1]; out[2] = x[2];
  if ( slab_on )
  { out[0] -= slab_L[0]*floor((x[0] - ref[0])/slab_L[0] + 0.5);
    out[1] -= slab_L[1]*floor((x[1] - ref[1])/slab_L[1] + 0.5);
  }
}

/* v's coordinates at the image nearest ref: in buf in a slab, else its own */
static inline REAL *vimg(const REAL *ref, vertex_id v, REAL *buf)
{ REAL *x = get_coord(v);
  if ( !slab_on ) return x;
  img(ref,x,buf);
  return buf;
}

/* v moved by whole periods to its image nearest p (Evolver's wrap_vertex,
   its edges' wraps kept right) */
static void wrap_near(vertex_id v, const REAL *p)
{ REAL *x = get_coord(v);
  int kx, ky;
  if ( !slab_on ) return;
  kx = (int)floor((p[0] - x[0])/slab_L[0] + 0.5);
  ky = (int)floor((p[1] - x[1])/slab_L[1] + 0.5);
  if ( kx || ky ) wrap_vertex(v,(kx & WRAPMASK) | ((ky & WRAPMASK) << TWRAPBITS));
}

/* the distance between vertices a and b (nearest images) */
static REAL vdist(vertex_id a, vertex_id b)
{ REAL buf[3], *x = get_coord(a);
  return dist3(x,vimg(x,b,buf));
}

static REAL edge_length(edge_id e)
{ return vdist(get_edge_tailv(e),get_edge_headv(e)); }

/* ---- a target edge length per vertex ------------------------------------ */

/* (set for one pyse_edit call: finer where the surface comes close to a wall
   or to itself, as at a 16-bead pack's near-contacts, whose 0.03 gaps a
   uniform mesh of 0.2 couldn't resolve) by vertex ordinal; <= 0: c->h */
static REAL *sizes = NULL;
static long sizes_n = 0;

static REAL vertex_size(struct ctx *c, vertex_id v)
{ long o = ordinal(v);
  return sizes && o >= 0 && o < sizes_n && sizes[o] > 0 ? sizes[o] : c->h;
}

/* the target length of edge e: its ends' mean */
static REAL edge_target(struct ctx *c, edge_id e)
{ return 0.5*(vertex_size(c,get_edge_tailv(e)) + vertex_size(c,get_edge_headv(e))); }

static void set_size(vertex_id v, REAL size)
{ long o = ordinal(v), i;
  if ( !sizes || o < 0 ) return;
  if ( o >= sizes_n )
  { long m = 2*o + 64;
    REAL *grown = (REAL *)realloc(sizes,m*sizeof(REAL));
    if ( !grown ) return;
    sizes = grown;
    for ( i = sizes_n ; i < m ; i++ ) sizes[i] = -1.0;
    sizes_n = m;
  }
  sizes[o] = size;
}

/* (set for one remesh: the vertices it works on, by ordinal; NULL: all. A
   descent's changes are local, and remeshing all of a refined pack's
   surface after each, its 20k smoothing moves, took most of a descent) */
static unsigned char *region = NULL;
static long region_n = 0;

static int in_region(vertex_id v)
{ long o = ordinal(v);
  return !region || (o >= 0 && o < region_n && region[o]);
}

static void add_to_region(vertex_id v)
{ long o = ordinal(v), i;
  if ( !region || o < 0 ) return;
  if ( o >= region_n )
  { long m = 2*o + 64;
    unsigned char *grown = (unsigned char *)realloc(region,m);
    if ( !grown ) return;
    region = grown;
    for ( i = region_n ; i < m ; i++ ) region[i] = 0;
    region_n = m;
  }
  region[o] = 1;
}

/* ---- facets' vertices, kept ---------------------------------------------- */

/* A remesh keeps each facet's vertices by its ordinal (c->fv): the grids'
   queries test hundreds of candidate facets an edit, and facet_verts' walk
   round the facetedges, ten dependent loads, was a sixth of a remesh's time.
   Taken for every facet as the grids are built (after a snapshot is put back
   too), and for the facets of each patch after an edit, before its counts:
   the facets an edit makes or changes are all in its patch (they have a
   vertex of it, the patch() logic's premise), a smoothing move changes no
   facet's vertices, and a freed facet is told by valid_element. */
static void fcache_put(struct ctx *c, facet_id f, const vertex_id *v)
{ long o = ordinal(f), i;
  if ( o < 0 ) return;
  if ( o >= c->fv_size )
  { long m = 2*o + 64;
    c->fv = c->fv ? (vertex_id *)temp_realloc((char *)c->fv,3*m*sizeof(vertex_id))
                  : (vertex_id *)temp_calloc(3*m,sizeof(vertex_id));
    for ( i = 3*c->fv_size ; i < 3*m ; i++ ) c->fv[i] = NULLID;
    c->fv_size = m;
  }
  c->fv[3*o] = v[0]; c->fv[3*o+1] = v[1]; c->fv[3*o+2] = v[2];
}

/* facet f's vertices: kept (c->fcache), else walked round into buf */
static const vertex_id *fverts(struct ctx *c, facet_id f, vertex_id *buf)
{ long o = ordinal(f);
  if ( c && c->fcache && o >= 0 && o < c->fv_size && valid_id(c->fv[3*o]) )
  {
#ifdef PYSE_FCACHE_CHECK
    facet_verts(f,buf);
    if ( !equal_id(buf[0],c->fv[3*o]) || !equal_id(buf[1],c->fv[3*o+1])
         || !equal_id(buf[2],c->fv[3*o+2]) )
    { fprintf(stderr,"pyse_edit: facet %ld's kept vertices are stale\n",o + 1);
      abort();
    }
#endif
    return c->fv + 3*o;
  }
  facet_verts(f,buf);
  return buf;
}

/* ---- spatial grids ------------------------------------------------------ */

static void elem_box(struct ctx *c, element_id id, REAL *lo, REAL *hi, REAL pad)
{ vertex_id buf[3];
  const vertex_id *v = buf;
  REAL *x0 = NULL, xb[3];
  int n, k, d;
  if ( id_type(id) == EDGE ) { buf[0] = get_edge_tailv(id); buf[1] = get_edge_headv(id); n = 2; }
  else { v = fverts(c,id,buf); n = 3; }
  for ( k = 0 ; k < n ; k++ )
  { const REAL *x = k ? vimg(x0,v[k],xb) : (x0 = get_coord(v[0]));
    for ( d = 0 ; d < 3 ; d++ )
    { if ( k == 0 || x[d] < lo[d] ) lo[d] = x[d];
      if ( k == 0 || x[d] > hi[d] ) hi[d] = x[d];
    }
  }
  for ( d = 0 ; d < 3 ; d++ ) { lo[d] -= pad; hi[d] += pad; }
}

/* the cells a box is in (those past the grid's box: its last; in a
   slab, x and y: as many as there are at most, numbered on past the period) */
static void cells(struct grid *g, const REAL *lo, const REAL *hi, long *a, long *b)
{ int d;
  for ( d = 0 ; d < 3 ; d++ )
  { REAL s, t;
    if ( g->per && d < 2 )
    { s = floor(lo[d]/g->pcell[d]); t = floor(hi[d]/g->pcell[d]);
      if ( !(fabs(s) < 1e15 && fabs(t) < 1e15) ) s = t = 0;     /* (not a number) */
      a[d] = (long)s;
      b[d] = t - s > g->n[d] - 1 ? a[d] + g->n[d] - 1 : (long)t;
      continue;
    }
    s = floor((lo[d] - g->lo[d])/g->cell); t = floor((hi[d] - g->lo[d])/g->cell);
    a[d] = !(s > 0) ? 0 : s > g->n[d] - 1 ? g->n[d] - 1 : (long)s;
    b[d] = !(t > 0) ? 0 : t > g->n[d] - 1 ? g->n[d] - 1 : (long)t;
  }
}

static inline long cell_key(const struct grid *g, long i, long j, long k)
{ if ( g->per )
  { i %= g->n[0]; if ( i < 0 ) i += g->n[0];
    j %= g->n[1]; if ( j < 0 ) j += g->n[1];
  }
  return (k*g->n[1] + j)*g->n[0] + i;
}

#define CELL(g,i,j,k) cell_key(g,i,j,k)

/* cell key's slot, or the empty one where it would go */
static long slot_probe(const struct grid *g, long key)
{ unsigned long u = (unsigned long)key*0x9E3779B97F4A7C15UL;
  long m = g->cap - 1, s = (long)((u ^ (u >> 32)) & (unsigned long)m);
  while ( g->key[s] >= 0 && g->key[s] != key ) s = (s + 1) & m;
  return s;
}

static void slots_alloc(struct grid *g, long cap)
{ long i;
  g->cap = cap;
  g->used = 0;
  g->key = (long *)temp_calloc(cap,sizeof(long));
  g->bstart = (long *)temp_calloc(cap,sizeof(long));
  g->xhead = (long *)temp_calloc(cap,sizeof(long));
  g->bcount = (int *)temp_calloc(cap,sizeof(int));
  for ( i = 0 ; i < cap ; i++ ) g->key[i] = g->xhead[i] = -1;
}

static void slots_free(struct grid *g)
{ temp_free((char *)g->key); temp_free((char *)g->bstart);
  temp_free((char *)g->xhead); temp_free((char *)g->bcount);
}

/* cell key's slot, made if it has none (the slots twice as many once half
   are used) */
static long slot_add(struct grid *g, long key)
{ long s = slot_probe(g,key);
  if ( g->key[s] == key ) return s;
  if ( 2*(g->used + 1) > g->cap )
  { struct grid old = *g;
    long i;
    slots_alloc(g,2*old.cap);
    for ( i = 0 ; i < old.cap ; i++ )
      if ( old.key[i] >= 0 )
      { long t = slot_probe(g,old.key[i]);
        g->key[t] = old.key[i]; g->bstart[t] = old.bstart[i];
        g->bcount[t] = old.bcount[i]; g->xhead[t] = old.xhead[i];
        g->used++;
      }
    slots_free(&old);
    s = slot_probe(g,key);
  }
  g->key[s] = key;
  g->bstart[s] = 0; g->bcount[s] = 0; g->xhead[s] = -1;
  g->used++;
  return s;
}

/* floats round a double from below and above */
static float below(REAL x) { float f = (float)x; return (REAL)f > x ? nextafterf(f,-HUGE_VALF) : f; }
static float above(REAL x) { float f = (float)x; return (REAL)f < x ? nextafterf(f,HUGE_VALF) : f; }

static void item_of(struct gitem *it, element_id id, const REAL *lo, const REAL *hi)
{ int d;
  it->ord = ordinal(id);
  for ( d = 0 ; d < 3 ; d++ ) { it->lo[d] = below(lo[d]); it->hi[d] = above(hi[d]); }
}

/* (a slab's: the box moved to the image whose centre is nearest the item's) */
static int apart_slab(const struct gitem *it, const REAL *lo, const REAL *hi, REAL tol)
{ REAL l[3], h[3];
  int d;
  for ( d = 0 ; d < 2 ; d++ )
  { REAL s = slab_L[d]*floor(((it->lo[d] + it->hi[d]) - (lo[d] + hi[d]))/(2*slab_L[d]) + 0.5);
    l[d] = lo[d] + s; h[d] = hi[d] + s;
  }
  l[2] = lo[2]; h[2] = hi[2];
  return h[0] < it->lo[0] - tol || l[0] > it->hi[0] + tol
      || h[1] < it->lo[1] - tol || l[1] > it->hi[1] + tol
      || h[2] < it->lo[2] - tol || l[2] > it->hi[2] + tol;
}

/* whether a box is further than tol from an item's (so from its element's) */
static inline int apart(const struct gitem *it, const REAL *lo, const REAL *hi, REAL tol)
{ if ( slab_on ) return apart_slab(it,lo,hi,tol);
  return hi[0] < it->lo[0] - tol || lo[0] > it->hi[0] + tol
      || hi[1] < it->lo[1] - tol || lo[1] > it->hi[1] + tol
      || hi[2] < it->lo[2] - tol || lo[2] > it->hi[2] + tol;
}

/* the grids' padding: c->pad, and 1e-6 h more (the crossing tests' boxes
   are 1e-9 h apart at most: none is missed across a cell's side) */
static REAL grid_pad(struct ctx *c) { return c->pad + 1e-6*c->h; }

static void grid_build(struct ctx *c, struct grid *g, int type)
{ element_id id;
  long nt = 0, mt = 0, *tkey = NULL, i, s, run, cap, a[3], b[3], x, y, z;
  struct gitem *tit = NULL, it;
  REAL lo[3], hi[3], pad = grid_pad(c);
  int d;
  if ( g->built )
  { slots_free(g);
    temp_free((char *)g->items);
    if ( g->xitems ) { temp_free((char *)g->xitems); temp_free((char *)g->xnext); }
  }
  memset(g,0,sizeof(*g));
  for ( d = 0 ; d < 3 ; d++ ) g->lo[d] = c->box_lo[d];
  /* (2 h: twice the candidates a query, but for a remesh, whose star
     queries scan their sets' boxes once: there 11% fewer instructions than
     with h at 28k facets, 6% at 14k) */
  g->cell = c->cell_h > 0 ? c->cell_h*c->h : c->h;
  for ( d = 0 ; d < 3 ; d++ )
  { double n = floor((c->box_hi[d] - c->box_lo[d])/g->cell) + 1;
    g->n[d] = n > 1024 ? 1024 : (long)n;
  }
  for ( d = slab_on ? 2 : 0 ; d < 3 ; d++ )
    if ( (c->box_hi[d] - c->box_lo[d])/g->n[d] > g->cell )
      g->cell = (c->box_hi[d] - c->box_lo[d])/g->n[d]*1.0000001;
  if ( slab_on )
  { REAL want = c->cell_h > 0 ? c->cell_h*c->h : c->h;
    g->per = 1;
    for ( d = 0 ; d < 2 ; d++ )
    { double n = floor(slab_L[d]/want);
      g->n[d] = n > 1024 ? 1024 : n < 1 ? 1 : (long)n;
      g->pcell[d] = slab_L[d]/g->n[d];
    }
  }
  for ( cap = 64 ; cap < 2*web.skel[type].count ; cap *= 2 ) ;
  slots_alloc(g,cap);
  /* (each element's box once: the cells' keys and its item, in turn) */
  for ( id = web.skel[type].used ; valid_id(id) ; id = elptr(id)->forechain )
  { if ( !(elptr(id)->attr & ALLOCATED) ) continue;
    if ( type == FACET && !valid_id(get_facet_fe(id)) ) continue;
    if ( type == FACET && c->fcache )
    { vertex_id v[3];
      facet_verts(id,v);
      fcache_put(c,id,v);
    }
    elem_box(c,id,lo,hi,pad);
    item_of(&it,id,lo,hi);
    cells(g,lo,hi,a,b);
    for ( z = a[2] ; z <= b[2] ; z++ ) for ( y = a[1] ; y <= b[1] ; y++ )
      for ( x = a[0] ; x <= b[0] ; x++ )
      { if ( nt == mt )
        { mt = mt ? 2*mt : 4096;
          tkey = tkey ? (long *)temp_realloc((char *)tkey,mt*sizeof(long))
                      : (long *)temp_calloc(mt,sizeof(long));
          tit = tit ? (struct gitem *)temp_realloc((char *)tit,mt*sizeof(struct gitem))
                    : (struct gitem *)temp_calloc(mt,sizeof(struct gitem));
        }
        tkey[nt] = CELL(g,x,y,z);
        tit[nt++] = it;
        s = slot_add(g,tkey[nt-1]);       /* (it may move the slots) */
        g->bcount[s]++;
      }
  }
  for ( s = 0, run = 0 ; s < g->cap ; s++ )
    if ( g->key[s] >= 0 ) { g->bstart[s] = run; run += g->bcount[s]; g->bcount[s] = 0; }
  g->items = (struct gitem *)temp_calloc(nt ? nt : 1,sizeof(struct gitem));
  for ( i = 0 ; i < nt ; i++ )
  { s = slot_probe(g,tkey[i]);
    g->items[g->bstart[s] + g->bcount[s]++] = tit[i];
  }
  if ( tkey ) { temp_free((char *)tkey); temp_free((char *)tit); }
  g->total = nt;
  g->built = 1;
}

#define XGRID_LIMIT (1L << 18)

static void xgrid_add(struct ctx *c, struct grid *g, element_id id)
{ REAL lo[3], hi[3];
  long a[3], b[3], x, y, z;
  struct gitem it;
  if ( !valid_id(id) || !valid_element(id) ) return;
  if ( id_type(id) == FACET && !valid_id(get_facet_fe(id)) ) return;
  elem_box(c,id,lo,hi,grid_pad(c));
  item_of(&it,id,lo,hi);
  cells(g,lo,hi,a,b);
  for ( z = a[2] ; z <= b[2] ; z++ ) for ( y = a[1] ; y <= b[1] ; y++ )
    for ( x = a[0] ; x <= b[0] ; x++ )
    { long s = slot_add(g,CELL(g,x,y,z));
      if ( g->xn == g->xmax )
      { long m = g->xmax ? 2*g->xmax : 1024;
        g->xnext = g->xnext ? (long *)temp_realloc((char *)g->xnext,m*sizeof(long))
                            : (long *)temp_calloc(m,sizeof(long));
        g->xitems = g->xitems ? (struct gitem *)temp_realloc((char *)g->xitems,m*sizeof(struct gitem))
                              : (struct gitem *)temp_calloc(m,sizeof(struct gitem));
        g->xmax = m;
      }
      g->xitems[g->xn] = it;
      g->xnext[g->xn] = g->xhead[s];
      g->xhead[s] = g->xn++;
    }
}

static void grids_build(struct ctx *c)
{ vertex_id v;
  int first = 1, d;
  FOR_ALL_VERTICES(v)
  { REAL *x = get_coord(v);
    for ( d = 0 ; d < 3 ; d++ )
    { if ( first || x[d] < c->box_lo[d] ) c->box_lo[d] = x[d];
      if ( first || x[d] > c->box_hi[d] ) c->box_hi[d] = x[d];
    }
    first = 0;
  }
  if ( first ) for ( d = 0 ; d < 3 ; d++ ) c->box_lo[d] = c->box_hi[d] = 0;
  c->epoch++;
  grid_build(c,&c->fgrid,FACET);
  grid_build(c,&c->egrid,EDGE);
}

/* elements an edit made or changed: in the cells their boxes are in now,
   which queries look at too (a list of them all, scanned by every query,
   made the remesher's checked edits quadratic in a batch: 1.6-2.5 s a
   remesh of a 16-bead pack's 1.5k facets); the grids made again once they
   are as many as the grids' own (hundreds of splits in one place, a pack's
   refined patch, piled up on its cells' lists: 641 splits took 11 s, the
   first 100 of them 0.4) or many */
static void changed(struct ctx *c, struct ids *f, struct ids *e)
{ int i;
  for ( i = 0 ; i < f->n ; i++ ) xgrid_add(c,&c->fgrid,f->a[i]);
  for ( i = 0 ; i < e->n ; i++ ) xgrid_add(c,&c->egrid,e->a[i]);
  if ( c->fgrid.xn + c->egrid.xn > XGRID_LIMIT
       || c->fgrid.xn + c->egrid.xn > c->fgrid.total + c->egrid.total + 4096 ) grids_build(c);
}

/* ---- what an edit changed ------------------------------------------------ */

static void sig_of(element_id id, struct sig *s)
{ vertex_id v[3];
  int k, n;
  s->id = positive_id(id);
  if ( id_type(id) == EDGE ) { v[0] = get_edge_tailv(id); v[1] = get_edge_headv(id); n = 2; }
  else { facet_verts(id,v); n = 3; }
  for ( k = 0 ; k < 3 ; k++ )
  { REAL *x = k < n ? get_coord(v[k]) : NULL;
    s->v[k] = k < n ? v[k] : NULLID;
    s->x[3*k] = x ? x[0] : 0; s->x[3*k+1] = x ? x[1] : 0; s->x[3*k+2] = x ? x[2] : 0;
  }
}

/* the edges and facets of the patch before an edit, as they are */
static void sigs_record(struct ctx *c, struct ids *E, struct ids *F)
{ int i, j;
  c->sig_now++;
  c->nsigs = 0;
  for ( j = 0 ; j < 2 ; j++ )
  { struct ids *l = j ? F : E;
    for ( i = 0 ; i < l->n ; i++ )
    { element_id id = l->a[i];
      int type = id_type(id);
      long ord = ordinal(id), k;
      if ( ord < 0 ) continue;
      if ( ord >= c->sig_size[type] )
      { long m = 2*ord + 64;
        c->sig_at[type] = c->sig_at[type] ? (int *)temp_realloc((char *)c->sig_at[type],m*sizeof(int))
                                          : (int *)temp_calloc(m,sizeof(int));
        c->sig_ep[type] = c->sig_ep[type] ? (int *)temp_realloc((char *)c->sig_ep[type],m*sizeof(int))
                                          : (int *)temp_calloc(m,sizeof(int));
        for ( k = c->sig_size[type] ; k < m ; k++ ) c->sig_ep[type][k] = 0;
        c->sig_size[type] = m;
      }
      if ( c->nsigs == c->maxsigs )
      { int m = c->maxsigs ? 2*c->maxsigs : 256;
        c->sigs = c->sigs ? (struct sig *)temp_realloc((char *)c->sigs,m*sizeof(struct sig))
                          : (struct sig *)temp_calloc(m,sizeof(struct sig));
        c->maxsigs = m;
      }
      sig_of(id,&c->sigs[c->nsigs]);
      c->sig_ep[type][ord] = c->sig_now;
      c->sig_at[type][ord] = c->nsigs++;
    }
  }
}

/* whether id is as it was before the edit: in the patch then, on the
   same vertices, they where they were (what a crossing test looks at) */
static int unchanged(struct ctx *c, element_id id)
{ int type = id_type(id), k;
  long ord = ordinal(id);
  struct sig now, *was;
  if ( ord < 0 || ord >= c->sig_size[type] || c->sig_ep[type][ord] != c->sig_now ) return 0;
  was = &c->sigs[c->sig_at[type][ord]];
  sig_of(id,&now);
  if ( !equal_id(was->id,now.id) ) return 0;
  for ( k = 0 ; k < 3 ; k++ ) if ( !equal_id(was->v[k],now.v[k]) ) return 0;
  for ( k = 0 ; k < 9 ; k++ ) if ( !(was->x[k] == now.x[k]) ) return 0;
  return 1;
}

/* ---- the patch's problems ----------------------------------------------- */

/* wall w's gap at x (pyse_wall_gap; in a slab at x's image deepest in its
   solid, pyse_slab_gap) */
static double wall_gap(struct ctx *c, int w, REAL *x)
{ if ( !slab_on ) return pyse_wall_gap(c->walls[w],x);
  return pyse_slab_gap(c->walls[w],c->sides[w] < 0 ? -1.0 : 1.0,
                       c->wref_ok[w] ? c->wref + 3*w : NULL,x,NULL);
}

/* v's coordinates at its image nearest wall w (deepest in its solid) */
static void wall_image(struct ctx *c, int w, vertex_id v, REAL *at)
{ int k;
  if ( !slab_on ) { for ( k = 0 ; k < SDIM ; k++ ) at[k] = get_coord(v)[k]; return; }
  pyse_slab_gap(c->walls[w],c->sides[w] < 0 ? -1.0 : 1.0,c->wref_ok[w] ? c->wref + 3*w : NULL,
                get_coord(v),at);
}

/* wall w's gap at vertex v (pyse_wall_gap), kept while v stays where it was
   (a checked edit looks at each vertex round it before and after, most of
   them unmoved: the walls' formulas took 19% of a checked smoothing) */
static double vertex_gap_fill(struct ctx *c, int w, vertex_id v)
{ long ord = ordinal(v), i;
  REAL *x = get_coord(v), *g;
  if ( ord >= c->gaps_size )
  { long m = 2*ord + 64;
    c->gaps = c->gaps ? (REAL *)temp_realloc((char *)c->gaps,m*c->nwalls*4*sizeof(REAL))
                      : (REAL *)temp_calloc(m*c->nwalls*4,sizeof(REAL));
    for ( i = c->gaps_size*c->nwalls*4 ; i < m*c->nwalls*4 ; i++ ) c->gaps[i] = NAN;
    c->gaps_size = m;
  }
  g = c->gaps + (ord*c->nwalls + w)*4;
  if ( !(g[0] == x[0] && g[1] == x[1] && g[2] == x[2]) )
  { g[0] = x[0]; g[1] = x[1]; g[2] = x[2];
    g[3] = wall_gap(c,w,x);
  }
  return (double)g[3];
}

/* (the kept gap read inline, the rest out of line: the checks ask it for
   every wall at both ends of every edge of every patch, a sixth of a
   remesh's instructions spent in the call) */
static inline double vertex_gap(struct ctx *c, int w, vertex_id v)
{ long ord = ordinal(v);
  if ( ord < c->gaps_size )
  { const REAL *g = c->gaps + (ord*c->nwalls + w)*4, *x = get_coord(v);
    if ( g[0] == x[0] && g[1] == x[1] && g[2] == x[2] ) return (double)g[3];
  }
  return vertex_gap_fill(c,w,v);
}

struct counts { int structure, pinched, degenerate, fold, solid, crossing; };

static int far_only;
static int near_pair(edge_id e, facet_id f);

static int pair_crosses(struct ctx *c, edge_id e, facet_id f)
{ vertex_id buf[3], t, hd;
  const vertex_id *v;
  const REAL *p, *q, *a, *b, *w;
  REAL pad, qb[3], ab[3], bb[3], wb[3];
  int d;
  if ( !valid_element(e) || !valid_element(f) || !valid_id(get_facet_fe(f)) ) return 0;
  t = get_edge_tailv(e); hd = get_edge_headv(e);
  v = fverts(c,f,buf);
  if ( equal_id(v[0],t) || equal_id(v[1],t) || equal_id(v[2],t)
       || equal_id(v[0],hd) || equal_id(v[1],hd) || equal_id(v[2],hd) ) return 0;
  /* (boxes apart: no crossing; most of a query's candidates, which the
     determinants below took 30% of a checked smoothing's time on) */
  p = get_coord(t); q = get_coord(hd);
  a = get_coord(v[0]); b = get_coord(v[1]); w = get_coord(v[2]);
  if ( slab_on )
  { q = vimg(p,hd,qb);
    a = vimg(p,v[0],ab); b = vimg(a,v[1],bb); w = vimg(a,v[2],wb);
  }
  pad = 1e-9*c->h;
  for ( d = 0 ; d < 3 ; d++ )
  { REAL elo = p[d] < q[d] ? p[d] : q[d], ehi = p[d] < q[d] ? q[d] : p[d];
    REAL flo = a[d], fhi = a[d];
    if ( b[d] < flo ) flo = b[d];
    if ( b[d] > fhi ) fhi = b[d];
    if ( w[d] < flo ) flo = w[d];
    if ( w[d] > fhi ) fhi = w[d];
    if ( ehi < flo - pad || elo > fhi + pad ) return 0;
  }
  if ( far_only && near_pair(e,f) ) return 0;
  return pyse_crosses(p,q,a,b,w,1e-12*c->h*c->h*c->h);
}

/* candidate it against id: not past its item's box, not seen yet */
static int candidate(struct ctx *c, element_id id, int other, const struct gitem *it,
                     const REAL *lo, const REAL *hi)
{ element_id o;
  if ( apart(it,lo,hi,1e-9*c->h) ) return 0;
  o = get_ordinal_id(other,it->ord);
  if ( !mark(c,o) ) return 0;
  return other == FACET ? pair_crosses(c,id,o) : pair_crosses(c,o,id);
}

static int crossings_of(struct ctx *c, element_id id, struct ids *near)
{ struct grid *g = id_type(id) == EDGE ? &c->fgrid : &c->egrid;
  int other = id_type(id) == EDGE ? FACET : EDGE;
  REAL lo[3], hi[3];
  long a[3], b[3], x, y, z, k;
  int i, n = 0;
  elem_box(c,id,lo,hi,0.0);
  cells(g,lo,hi,a,b);
  c->stamp_now++;
  for ( z = a[2] ; z <= b[2] ; z++ ) for ( y = a[1] ; y <= b[1] ; y++ )
    for ( x = a[0] ; x <= b[0] ; x++ )
    { long s = slot_probe(g,CELL(g,x,y,z));
      if ( g->key[s] < 0 ) continue;
      for ( k = g->bstart[s] ; k < g->bstart[s] + g->bcount[s] ; k++ )
        n += candidate(c,id,other,&g->items[k],lo,hi);
      for ( k = g->xhead[s] ; k >= 0 ; k = g->xnext[k] )
        n += candidate(c,id,other,&g->xitems[k],lo,hi);
    }
  for ( i = 0 ; i < near->n ; i++ )
    if ( mark(c,near->a[i]) )
      n += other == FACET ? pair_crosses(c,id,near->a[i]) : pair_crosses(c,near->a[i],id);
  return n;
}

/* The crossings of edges QE against all facets (and nearF) and of facets QF
   against all edges (and nearE): the sum of crossings_of() over them, in one
   pass over the cells of each set's box, every candidate tried against each
   query, its vertices and box looked up once (a smoothing move's twelve
   queries, its star's edges and facets, scanned the same cells' items twelve
   times: a fifth of a remesh's time). A candidate counts for a query as in
   pair_crosses(), the same tests in the same order; one crossing a query is
   in the cells of the query's box, which are among the set's, so each query
   finds the same partners. */
/* (in a slab, the points at one image, in x: see star_crossings) */
struct qedge { vertex_id t, h; const REAL *p, *q; REAL lo[3], hi[3], x[2][3]; int ok; };
struct qfacet { vertex_id v[3]; const REAL *a, *b, *w; REAL lo[3], hi[3], x[3][3]; int ok; };

static void box3(const REAL *const *x, int n, REAL *lo, REAL *hi)
{ int k, d;
  for ( d = 0 ; d < 3 ; d++ ) lo[d] = hi[d] = x[0][d];
  for ( k = 1 ; k < n ; k++ )
    for ( d = 0 ; d < 3 ; d++ )
    { if ( x[k][d] < lo[d] ) lo[d] = x[k][d];
      if ( x[k][d] > hi[d] ) hi[d] = x[k][d];
    }
}

static void qedge_of(struct qedge *e, edge_id id)
{ const REAL *x[2];
  e->ok = valid_element(id);
  if ( !e->ok ) return;
  e->t = get_edge_tailv(id); e->h = get_edge_headv(id);
  x[0] = e->p = get_coord(e->t); x[1] = e->q = get_coord(e->h);
  if ( slab_on )
  { img(e->p,e->p,e->x[0]); img(e->p,e->q,e->x[1]);
    x[0] = e->p = e->x[0]; x[1] = e->q = e->x[1];
  }
  box3(x,2,e->lo,e->hi);
}

static void qfacet_of(struct ctx *c, struct qfacet *f, facet_id id)
{ vertex_id buf[3];
  const vertex_id *v;
  const REAL *x[3];
  f->ok = valid_element(id) && valid_id(get_facet_fe(id));
  if ( !f->ok ) return;
  v = fverts(c,id,buf);
  f->v[0] = v[0]; f->v[1] = v[1]; f->v[2] = v[2];
  x[0] = f->a = get_coord(v[0]); x[1] = f->b = get_coord(v[1]); x[2] = f->w = get_coord(v[2]);
  if ( slab_on )
  { img(f->a,f->a,f->x[0]); img(f->a,f->b,f->x[1]); img(f->a,f->w,f->x[2]);
    x[0] = f->a = f->x[0]; x[1] = f->b = f->x[1]; x[2] = f->w = f->x[2];
  }
  box3(x,3,f->lo,f->hi);
}

/* a query moved by whole periods to the image nearest ref (a slab's) */
static void qedge_near(struct qedge *e, const REAL *ref)
{ int d, k;
  for ( d = 0 ; d < 2 ; d++ )
  { REAL s = slab_L[d]*floor((e->x[0][d] - ref[d])/slab_L[d] + 0.5);
    if ( s == 0 ) continue;
    for ( k = 0 ; k < 2 ; k++ ) e->x[k][d] -= s;
    e->lo[d] -= s; e->hi[d] -= s;
  }
}

static void qfacet_near(struct qfacet *f, const REAL *ref)
{ int d, k;
  for ( d = 0 ; d < 2 ; d++ )
  { REAL s = slab_L[d]*floor((f->x[0][d] - ref[d])/slab_L[d] + 0.5);
    if ( s == 0 ) continue;
    for ( k = 0 ; k < 3 ; k++ ) f->x[k][d] -= s;
    f->lo[d] -= s; f->hi[d] -= s;
  }
}

static int qpair(struct ctx *c, const struct qedge *e, const struct qfacet *f, edge_id eid,
                 facet_id fid);

/* (a slab's: the facet moved to the edge's image first; out of line, the
   copy kept off qpair's way) */
static int __attribute__((noinline)) qpair_slab(struct ctx *c, const struct qedge *e,
                                                const struct qfacet *f, edge_id eid, facet_id fid)
{ struct qfacet moved;
  memcpy(&moved,f,sizeof(moved));
  qfacet_near(&moved,e->p);
  moved.a = moved.x[0]; moved.b = moved.x[1]; moved.w = moved.x[2];
  return qpair(c,e,&moved,eid,fid);
}

/* (pair_crosses on the looked-up pair) */
static int qpair(struct ctx *c, const struct qedge *e, const struct qfacet *f, edge_id eid,
                 facet_id fid)
{ REAL pad = 1e-9*c->h;
  int d;
  if ( !e->ok || !f->ok ) return 0;
  if ( equal_id(f->v[0],e->t) || equal_id(f->v[1],e->t) || equal_id(f->v[2],e->t)
       || equal_id(f->v[0],e->h) || equal_id(f->v[1],e->h) || equal_id(f->v[2],e->h) ) return 0;
  if ( slab_on
       && (fabs(f->a[0] - e->p[0]) > 0.5*slab_L[0] || fabs(f->a[1] - e->p[1]) > 0.5*slab_L[1]) )
    return qpair_slab(c,e,f,eid,fid);
  for ( d = 0 ; d < 3 ; d++ )
    if ( e->hi[d] < f->lo[d] - pad || e->lo[d] > f->hi[d] + pad ) return 0;
  if ( far_only && near_pair(eid,fid) ) return 0;
  return pyse_crosses(e->p,e->q,f->a,f->b,f->w,1e-12*c->h*c->h*c->h);
}

/* candidate o (a facet in pass 0, an edge in pass 1) against each query */
static int star_try(struct ctx *c, int pass, element_id o, struct ids *QE, struct qedge *qe,
                    struct ids *QF, struct qfacet *qf)
{ int i, n = 0;
  if ( pass )
  { struct qedge oe;
    qedge_of(&oe,o);
    for ( i = 0 ; i < QF->n ; i++ ) n += qpair(c,&oe,qf + i,o,QF->a[i]);
  }
  else
  { struct qfacet of;
    qfacet_of(c,&of,o);
    for ( i = 0 ; i < QE->n ; i++ ) n += qpair(c,qe + i,&of,QE->a[i],o);
  }
  return n;
}

#define STAR_MAX 16

static int star_crossings(struct ctx *c, struct ids *QE, struct ids *nearF, struct ids *QF,
                          struct ids *nearE)
{ struct qedge qe[STAR_MAX];
  struct qfacet qf[STAR_MAX];
  int pass, i, n = 0;
  if ( QE->n > STAR_MAX || QF->n > STAR_MAX )
  { /* (a patch's: its box many queries' cells, each candidate tried against them all) */
    for ( i = 0 ; i < QE->n ; i++ ) n += crossings_of(c,QE->a[i],nearF);
    for ( i = 0 ; i < QF->n ; i++ ) n += crossings_of(c,QF->a[i],nearE);
    return n;
  }
  for ( i = 0 ; i < QE->n ; i++ ) qedge_of(qe + i,QE->a[i]);
  for ( i = 0 ; i < QF->n ; i++ ) qfacet_of(c,qf + i,QF->a[i]);
  if ( slab_on )
  { /* (the queries at one image, round a vertex: their box a patch's) */
    const REAL *ref = NULL;
    REAL r[3];
    for ( i = 0 ; i < QE->n && !ref ; i++ ) if ( qe[i].ok ) ref = qe[i].p;
    for ( i = 0 ; i < QF->n && !ref ; i++ ) if ( qf[i].ok ) ref = qf[i].a;
    if ( ref )
    { r[0] = ref[0]; r[1] = ref[1]; r[2] = ref[2];
      for ( i = 0 ; i < QE->n ; i++ ) if ( qe[i].ok ) qedge_near(qe + i,r);
      for ( i = 0 ; i < QF->n ; i++ ) if ( qf[i].ok ) qfacet_near(qf + i,r);
    }
  }
  for ( pass = 0 ; pass < 2 ; pass++ )
  { struct ids *Q = pass ? QF : QE, *near = pass ? nearE : nearF;
    struct grid *g = pass ? &c->egrid : &c->fgrid;
    int type = pass ? EDGE : FACET, any = 0, d;
    REAL lo[3], hi[3];
    long a[3], b[3], x, y, z, k;
    for ( i = 0 ; i < Q->n ; i++ )
    { const REAL *ql = pass ? qf[i].lo : qe[i].lo, *qh = pass ? qf[i].hi : qe[i].hi;
      if ( !(pass ? qf[i].ok : qe[i].ok) ) continue;
      for ( d = 0 ; d < 3 ; d++ )
      { if ( !any || ql[d] < lo[d] ) lo[d] = ql[d];
        if ( !any || qh[d] > hi[d] ) hi[d] = qh[d];
      }
      any = 1;
    }
    if ( !any ) continue;
    cells(g,lo,hi,a,b);
    c->stamp_now++;
    for ( z = a[2] ; z <= b[2] ; z++ ) for ( y = a[1] ; y <= b[1] ; y++ )
      for ( x = a[0] ; x <= b[0] ; x++ )
      { long s = slot_probe(g,CELL(g,x,y,z));
        if ( g->key[s] < 0 ) continue;
        for ( k = g->bstart[s] ; k < g->bstart[s] + g->bcount[s] ; k++ )
          if ( !apart(&g->items[k],lo,hi,1e-9*c->h) )
          { element_id o = get_ordinal_id(type,g->items[k].ord);
            if ( mark(c,o) ) n += star_try(c,pass,o,QE,qe,QF,qf);
          }
        for ( k = g->xhead[s] ; k >= 0 ; k = g->xnext[k] )
          if ( !apart(&g->xitems[k],lo,hi,1e-9*c->h) )
          { element_id o = get_ordinal_id(type,g->xitems[k].ord);
            if ( mark(c,o) ) n += star_try(c,pass,o,QE,qe,QF,qf);
          }
      }
    for ( i = 0 ; i < near->n ; i++ )
      if ( mark(c,near->a[i]) ) n += star_try(c,pass,near->a[i],QE,qe,QF,qf);
  }
  return n;
}

/* ---- contact lines crossing each other ---------------------------------- */

/* a contact-line edge: on one facet, on a constraint */
static int is_line(edge_id e)
{ return valid_element(e) && (get_eattr(e) & CONSTRAINT) && valence(e) == 1; }

/* whether x lies on constraint con's surface (within 1e-6 h) */
static int on_surface(int con, REAL *x, REAL h)
{ REAL f, g[MAXCOORD];
  eval_all(get_constraint(con)->formula,x,SDIM,&f,g,NULLID);
  return fabs(f) <= 1e-6*h*norm3(g);
}

/* whether w is within depth contact-line edges of v along the lines */
static int line_near(vertex_id v, vertex_id w, int depth)
{ edge_id e, e0;
  int guard;
  if ( equal_id(v,w) ) return 1;
  if ( depth == 0 ) return 0;
  EACH_EDGE_AT(v,e,e0,guard)
    if ( is_line(e) && line_near(get_edge_headv(e),w,depth - 1) ) return 1;
  return 0;
}

/* whether contact-line edges e and o, sharing no vertex, on one wall (or
   o's ends on e's wall's surface: a pole copy of a bead is a constraint of
   its own), cross as seen along the wall's normal at their middle (mid,
   when they do); far: not two within two edges of each other along the
   lines (a fold of a line, as crossings_of leaves folds one ring across: a
   finger of a pack's line two edges long, narrower than an edge, folded
   over) */
static int lines_cross(edge_id e, edge_id o, REAL h, int far, REAL *mid)
{ vertex_id a0 = get_edge_tailv(e), a1 = get_edge_headv(e);
  vertex_id b0 = get_edge_tailv(o), b1 = get_edge_headv(o);
  REAL qb[3], rb[3], tb[3], *p = get_coord(a0), *q = vimg(p,a1,qb), *r = vimg(p,b0,rb),
       *t = vimg(p,b1,tb);
  REAL g[MAXCOORD], f, gl, e1[3], e2[3], P[4][2], d1, d2, d3, d4, *pts[4];
  conmap_t *map = get_e_constraint_map(e);
  unsigned int m;
  int con = -1, k;
  if ( equal_id(a0,b0) || equal_id(a0,b1) || equal_id(a1,b0) || equal_id(a1,b1) ) return 0;
  for ( k = 0 ; k < 3 ; k++ )
  { REAL alo = p[k] < q[k] ? p[k] : q[k], ahi = p[k] < q[k] ? q[k] : p[k];
    REAL blo = r[k] < t[k] ? r[k] : t[k], bhi = r[k] < t[k] ? t[k] : r[k];
    if ( ahi < blo - 1e-9*h || bhi < alo - 1e-9*h ) return 0;
  }
  if ( far && (line_near(a0,b0,2) || line_near(a0,b1,2) || line_near(a1,b0,2)
               || line_near(a1,b1,2)) ) return 0;
  for ( m = 1 ; m <= map[0] && con < 0 ; m++ )
    if ( e_on_constraint(o,(int)(map[m] & CONMASK))
         || (on_surface((int)(map[m] & CONMASK),r,h) && on_surface((int)(map[m] & CONMASK),t,h)) )
      con = (int)(map[m] & CONMASK);
  if ( con < 0 ) return 0;
  for ( k = 0 ; k < 3 ; k++ ) mid[k] = (p[k] + q[k] + r[k] + t[k])/4;
  eval_all(get_constraint(con)->formula,mid,SDIM,&f,g,NULLID);
  gl = norm3(g);
  if ( !(gl > 0) ) return 0;
  for ( k = 0 ; k < 3 ; k++ ) g[k] /= gl;
  /* e1, e2 across the normal */
  if ( fabs(g[0]) < 0.6 ) { e1[0] = 0; e1[1] = g[2]; e1[2] = -g[1]; }
  else { e1[0] = -g[2]; e1[1] = 0; e1[2] = g[0]; }
  gl = norm3(e1);
  for ( k = 0 ; k < 3 ; k++ ) e1[k] /= gl;
  e2[0] = g[1]*e1[2] - g[2]*e1[1];
  e2[1] = g[2]*e1[0] - g[0]*e1[2];
  e2[2] = g[0]*e1[1] - g[1]*e1[0];
  pts[0] = p; pts[1] = q; pts[2] = r; pts[3] = t;
  for ( k = 0 ; k < 4 ; k++ )
  { P[k][0] = dot3(pts[k],e1) - dot3(mid,e1);
    P[k][1] = dot3(pts[k],e2) - dot3(mid,e2);
  }
#define ORIENT(o_,a_,b_) ((P[a_][0]-P[o_][0])*(P[b_][1]-P[o_][1]) - (P[a_][1]-P[o_][1])*(P[b_][0]-P[o_][0]))
  d1 = ORIENT(0,1,2); d2 = ORIENT(0,1,3); d3 = ORIENT(2,3,0); d4 = ORIENT(2,3,1);
#undef ORIENT
  return d1*d2 < 0 && d3*d4 < 0;
}

/* contact-line edges crossing contact-line edge e (from the edge grid and
   the changed lists, and near) */
static int line_candidate(struct ctx *c, edge_id e, const struct gitem *it, const REAL *lo,
                          const REAL *hi)
{ REAL mid[MAXCOORD];
  edge_id o;
  if ( apart(it,lo,hi,1e-9*c->h) ) return 0;
  o = get_ordinal_id(EDGE,it->ord);
  if ( !mark(c,o) || equal_id(o,e) || !is_line(o) ) return 0;
  return lines_cross(e,o,c->h,1,mid);
}

static int line_crossings_of(struct ctx *c, edge_id e, struct ids *near)
{ struct grid *g = &c->egrid;
  REAL lo[3], hi[3], mid[MAXCOORD];
  long a[3], b[3], x, y, z, k;
  int i, n = 0;
  if ( !is_line(e) ) return 0;
  elem_box(c,e,lo,hi,0.0);
  cells(g,lo,hi,a,b);
  c->stamp_now++;
  for ( z = a[2] ; z <= b[2] ; z++ ) for ( y = a[1] ; y <= b[1] ; y++ )
    for ( x = a[0] ; x <= b[0] ; x++ )
    { long s = slot_probe(g,CELL(g,x,y,z));
      if ( g->key[s] < 0 ) continue;
      for ( k = g->bstart[s] ; k < g->bstart[s] + g->bcount[s] ; k++ )
        n += line_candidate(c,e,&g->items[k],lo,hi);
      for ( k = g->xhead[s] ; k >= 0 ; k = g->xnext[k] )
        n += line_candidate(c,e,&g->xitems[k],lo,hi);
    }
  for ( i = 0 ; i < near->n ; i++ )
    if ( mark(c,near->a[i]) && !equal_id(near->a[i],e) && is_line(near->a[i]) )
      n += lines_cross(e,near->a[i],c->h,1,mid);
  return n;
}

/* edge e's problems of structure, and its fold, into k */
static void edge_problems(struct ctx *c, edge_id e, struct counts *k)
{ facetedge_id fe0;
  facet_id f[3];
  int n = 0, w;
  vertex_id t, hh;
  t = get_edge_tailv(e); hh = get_edge_headv(e);
  fe0 = get_edge_fe(e);
  if ( !valid_id(fe0) ) { k->structure++; return; }
  { facetedge_id fe = fe0;
    do { if ( n < 3 ) f[n] = get_fe_facet(fe); n++; fe = get_next_facet(fe); }
    while ( valid_id(fe) && !equal_id(fe,fe0) && n < 1000 );
  }
  if ( n > 2 ) { if ( c->manifold ) k->structure++; return; }
  if ( n == 2 )
  { int s1 = inverted(f[0]), s2 = inverted(f[1]);
    body_id fr1 = get_facet_body(positive_id(f[0])), bk1 = get_facet_body(inverse_id(positive_id(f[0])));
    body_id fr2 = get_facet_body(positive_id(f[1])), bk2 = get_facet_body(inverse_id(positive_id(f[1])));
    int alike = equal_id(fr1,fr2) && equal_id(bk1,bk2);
    int swapped = equal_id(fr1,bk2) && equal_id(bk1,fr2);
    if ( (alike && s1 == s2) || (!alike && swapped && s1 != s2) ) k->structure++;
    else if ( alike || swapped )
    { REAL n1[3], n2[3], a1, a2, d;
      pyse_facet_normal(positive_id(f[0]),n1,&a1);
      pyse_facet_normal(positive_id(f[1]),n2,&a2);
      d = dot3(n1,n2)/(4*a1*a2 + 1e-300);
      if ( swapped && !alike ) d = -d;
      if ( a1 > 0 && a2 > 0 && d < -0.95 ) k->fold++;
    }
    return;
  }
  { ATTR attr = get_eattr(e);
    if ( !(attr & (CONSTRAINT|BOUNDARY|FIXED)) ) k->structure++;
    if ( attr & CONSTRAINT )
    { conmap_t *map = get_e_constraint_map(e);
      unsigned int m;
      for ( m = 1 ; m <= map[0] ; m++ )
        if ( !v_on_constraint(t,(int)(map[m]&CONMASK)) || !v_on_constraint(hh,(int)(map[m]&CONMASK)) )
          k->structure++;
    }
    for ( w = 0 ; w < c->nwalls ; w++ )
      if ( e_on_constraint(e,c->walls[w]) && (inverted(f[0]) || (attr & NEGBOUNDARY)) )
        k->structure++;
  }
}

/* whether vertex v is in wall w's solid */
static int vertex_solid(struct ctx *c, int w, vertex_id v)
{ double side = c->sides[w] < 0 ? -1.0 : 1.0;
  if ( !valid_element(v) || v_on_constraint(v,c->walls[w]) ) return 0;
  return -side*vertex_gap(c,w,v) > c->tol*c->h;
}

/* whether the middle of the edge from t (at p) to hh (at q, p's image),
   len long, is in wall w's solid */
static int edge_solid(struct ctx *c, int w, vertex_id t, vertex_id hh, REAL *p, REAL *q,
                      double len)
{ int wc = c->walls[w], d;
  double side = c->sides[w] < 0 ? -1.0 : 1.0, deep, dh;
  REAL mid[3];
  /* (the midpoint can't be in deeper than its ends by more than about
     half the edge, the gap growing about as the distance does: with
     twice that to spare, not evaluated; looked at before whether the
     edge is on the wall, which most edges are far from) */
  deep = -side*vertex_gap(c,w,t);
  dh = -side*vertex_gap(c,w,hh);
  if ( dh > deep ) deep = dh;
  if ( deep + len < c->tol*c->h ) return 0;
  if ( v_on_constraint(t,wc) && v_on_constraint(hh,wc) ) return 0;
  for ( d = 0 ; d < 3 ; d++ ) mid[d] = (p[d] + q[d])/2;
  return -side*wall_gap(c,w,mid) > c->tol*c->h;
}

/* how far vertex v is in the solid of the wall it is deepest in (-side
   times its gap, the most over all walls, those it is on too: about 0
   there; infinite where a gap isn't a number), kept while v stays where
   it was: no wall's solid has v or an edge of it unless this is near
   (the walls looked at one by one for each vertex and edge of a patch,
   20 walls on a 16-bead pack, were a tenth of a remesh) */
static double vertex_reach(struct ctx *c, vertex_id v)
{ long ord = ordinal(v), i;
  REAL *x = get_coord(v), *r;
  if ( ord >= c->reach_size )
  { long m = 2*ord + 64;
    c->reach = c->reach ? (REAL *)temp_realloc((char *)c->reach,m*4*sizeof(REAL))
                        : (REAL *)temp_calloc(m*4,sizeof(REAL));
    for ( i = c->reach_size*4 ; i < m*4 ; i++ ) c->reach[i] = NAN;
    c->reach_size = m;
  }
  r = c->reach + 4*ord;
  if ( !(r[0] == x[0] && r[1] == x[1] && r[2] == x[2]) )
  { double most = -HUGE_VAL;
    int w, nan = 0;
    for ( w = 0 ; w < c->nwalls ; w++ )
    { double d = -(c->sides[w] < 0 ? -1.0 : 1.0)*vertex_gap(c,w,v);
      if ( d != d ) nan = 1;
      else if ( d > most ) most = d;
    }
    r[0] = x[0]; r[1] = x[1]; r[2] = x[2];
    r[3] = nan ? HUGE_VAL : most;
  }
  return (double)r[3];
}

/* vertex v's and edge e's counts in the walls' solids (those of each
   wall it is near) */
static int vertex_solids(struct ctx *c, vertex_id v)
{ int w, n = 0;
  if ( !valid_element(v) || !(vertex_reach(c,v) > c->tol*c->h) ) return 0;
  for ( w = 0 ; w < c->nwalls ; w++ ) n += vertex_solid(c,w,v);
  return n;
}

static int edge_solids(struct ctx *c, edge_id e)
{ vertex_id t = get_edge_tailv(e), hh = get_edge_headv(e);
  REAL qb[3], *p = get_coord(t), *q = slab_on ? vimg(p,hh,qb) : get_coord(hh);
  double deep = vertex_reach(c,t), len = dist3(p,q);
  int w, n = 0;
  if ( vertex_reach(c,hh) > deep ) deep = vertex_reach(c,hh);
  /* (as edge_solid skips each wall: deeper in none) */
  if ( deep + len < c->tol*c->h ) return 0;
  for ( w = 0 ; w < c->nwalls ; w++ ) n += edge_solid(c,w,t,hh,p,q,len);
  return n;
}

/* a patch's stage: before an edit, after it, or only looked at */
enum { PATCH_BEFORE, PATCH_AFTER, PATCH_CHECK };

/* the patch round verts: its problems; before an edit, its edges and
   facets as they are (sigs_record); after it, those it changed (in out_e,
   out_f) */
static void patch(struct ctx *c, struct ids *verts, struct counts *k, int stage,
                  struct ids *out_e, struct ids *out_f)
{ struct ids E = {0,0,0}, F = {0,0,0}, CE = {0,0,0}, CF = {0,0,0}, V = {0,0,0};
  int i, guard;
  edge_id e, e0;
  memset(k,0,sizeof(*k));
  c->stamp_now++;
  /* (each vertex once: a separation's patch after it is its vertex's star
     and the new vertices', which share neighbours, and a pinched neighbour
     counted twice made the separation look worse; the other end of an
     unzipped edge, pinched until its own separation) */
  for ( i = 0 ; i < verts->n ; i++ )
    if ( valid_element(verts->a[i]) && mark(c,verts->a[i]) ) push(&V,verts->a[i]);
  verts = &V;
  for ( i = 0 ; i < verts->n ; i++ )
  { vertex_id v = verts->a[i];
    if ( !valid_element(v) ) continue;
    EACH_EDGE_AT(v,e,e0,guard)
      if ( mark(c,positive_id(e)) ) push(&E,positive_id(e));
  }
  for ( i = 0 ; i < E.n ; i++ )
  { facetedge_id fe0 = get_edge_fe(E.a[i]), fe = fe0;
    int n = 0;
    if ( valid_id(fe0) ) do
    { facet_id f = positive_id(get_fe_facet(fe));
      if ( valid_id(f) && mark(c,f) ) push(&F,f);
      fe = get_next_facet(fe);
    } while ( valid_id(fe) && !equal_id(fe,fe0) && ++n < 1000 );
  }

  /* edges */
  for ( i = 0 ; i < E.n ; i++ ) edge_problems(c,E.a[i],k);

  /* fans round each vertex */
  for ( i = 0 ; i < verts->n ; i++ )
  { vertex_id v = verts->a[i];
    facet_id fs[64];
    int parent[64], nf = 0, a, b, roots = 0, many = 0;
    if ( !valid_element(v) ) continue;
    EACH_EDGE_AT(v,e,e0,guard)
    { facetedge_id fe0 = get_edge_fe(e), fe = fe0;
      int n = 0;
      if ( valence(e) > 2 ) many = 1;
      if ( valid_id(fe0) ) do
      { facet_id f = positive_id(get_fe_facet(fe));
        for ( a = 0 ; a < nf && !equal_id(fs[a],f) ; a++ ) ;
        if ( a == nf && nf < 64 ) { fs[nf] = f; parent[nf] = nf; nf++; }
        fe = get_next_facet(fe);
      } while ( valid_id(fe) && !equal_id(fe,fe0) && ++n < 1000 );
    }
    if ( many ) continue;
    EACH_EDGE_AT(v,e,e0,guard)
    { facetedge_id fe0 = get_edge_fe(e);
      if ( valence(e) == 2 )
      { facet_id f1 = positive_id(get_fe_facet(fe0)), f2 = positive_id(get_fe_facet(get_next_facet(fe0)));
        int r1, r2;
        for ( a = 0 ; a < nf && !equal_id(fs[a],f1) ; a++ ) ;
        for ( b = 0 ; b < nf && !equal_id(fs[b],f2) ; b++ ) ;
        if ( a == nf || b == nf ) continue;
        for ( r1 = a ; parent[r1] != r1 ; r1 = parent[r1] ) ;
        for ( r2 = b ; parent[r2] != r2 ; r2 = parent[r2] ) ;
        if ( r1 != r2 ) parent[r1 < r2 ? r2 : r1] = r1 < r2 ? r1 : r2;
      }
    }
    for ( a = 0 ; a < nf ; a++ ) if ( parent[a] == a ) roots++;
    if ( roots > 1 ) k->pinched++;
  }

  /* facets */
  for ( i = 0 ; i < F.n ; i++ )
  { REAL n[3], area;
    pyse_facet_normal(F.a[i],n,&area);
    if ( area < 1e-10*c->h*c->h ) k->degenerate++;
  }

  /* solids */
  for ( i = 0 ; i < verts->n ; i++ ) k->solid += vertex_solids(c,verts->a[i]);
  for ( i = 0 ; i < E.n ; i++ ) k->solid += edge_solids(c,E.a[i]);

  /* the elements the edit changed (all, without a record from before it);
     the patch's facets' vertices kept again (see fverts) */
  if ( stage == PATCH_AFTER )
  { if ( c->fcache )
      for ( i = 0 ; i < F.n ; i++ )
      { vertex_id v[3];
        facet_verts(F.a[i],v);
        fcache_put(c,F.a[i],v);
      }
    for ( i = 0 ; i < E.n ; i++ ) if ( !unchanged(c,E.a[i]) ) push(&CE,E.a[i]);
    for ( i = 0 ; i < F.n ; i++ ) if ( !unchanged(c,F.a[i]) ) push(&CF,F.a[i]);
    c->sig_now++;                    /* (the record used up) */
  }

  /* crossings: the patch's edges against facets, its facets against edges;
     and contact lines through each other (the patch's against all). Not
     counted before an edit (the counts compared with after) where the
     remesh began with none on the whole surface: every edit kept since
     lets no patch's count grow and changes nothing outside it, so there
     are none to count (the counts before and after were half a remeshing's
     time, 1.2 s on a 16-bead pack); and after it only the elements it
     changed against all, the others' crossings being among those none
     (a smoothing move changes its vertex's 12 of a patch's 60; the
     patch's own lines still, whose crossings go by how near they are
     along the lines, which an edit changes) */
  if ( !(c->clean && stage != PATCH_AFTER) )
  { int few = c->clean && stage == PATCH_AFTER && !far_only;
    struct ids *QE = few ? &CE : &E, *QF = few ? &CF : &F;
    k->crossing += star_crossings(c,QE,&F,QF,&E);
    for ( i = 0 ; i < E.n ; i++ ) k->crossing += line_crossings_of(c,E.a[i],&E);
  }

  if ( stage == PATCH_BEFORE ) sigs_record(c,&E,&F);
  if ( out_e ) for ( i = 0 ; i < CE.n ; i++ ) push(out_e,CE.a[i]);
  if ( out_f ) for ( i = 0 ; i < CF.n ; i++ ) push(out_f,CF.a[i]);
  if ( E.a ) temp_free((char *)E.a);
  if ( F.a ) temp_free((char *)F.a);
  if ( CE.a ) temp_free((char *)CE.a);
  if ( CF.a ) temp_free((char *)CF.a);
  if ( V.a ) temp_free((char *)V.a);
}

/* the patch round verts before an edit: its problems, and its elements
   as they are, which after() tells the changed of. The batch's snapshot
   taken here if it has none (every edit calls this before it changes
   anything): a batch that changes nothing takes none (a remesh near a
   change took nine, each the whole surface, for a few edits) */
static void keys_now(struct edit_args *a);
static void near_verify(struct ctx *c, struct ids *verts);

/* the grids built if they are due (see run_batch) */
static void ensure_grids(struct ctx *c)
{ if ( !c->grids_due ) return;
  c->grids_due = 0;
  c->pad = c->due_pad;
  grids_build(c);
}

static void before(struct ctx *c, struct ids *verts, struct counts *was)
{ if ( c->snapping && !c->snap ) { c->snap = pyse_snapshot_hold(); c->snap_k = c->cur_k; }
  if ( c->batch ) keys_now(c->batch);
  ensure_grids(c);
  if ( c->clean && c->near_on ) near_verify(c,verts);
  patch(c,verts,was,PATCH_BEFORE,NULL,NULL);
}

/* PYSE_EDIT_* of the first count that grew (pinches may grow where allowed) */
static int worse(struct counts *was, struct counts *now, int pinch_ok)
{ if ( now->structure > was->structure ) return PYSE_EDIT_STRUCTURE;
  if ( !pinch_ok && now->pinched > was->pinched ) return PYSE_EDIT_PINCH;
  if ( now->degenerate > was->degenerate ) return PYSE_EDIT_DEGENERATE;
  if ( now->fold > was->fold ) return PYSE_EDIT_FOLD;
  if ( now->solid > was->solid ) return PYSE_EDIT_SOLID;
  if ( now->crossing > was->crossing ) return PYSE_EDIT_CROSSING;
  return 0;
}

/* the patch after an edit against before: 0, or why it is undone; the
   patch's elements go on the changed lists */
static int after(struct ctx *c, struct ids *verts, struct counts *was, int pinch_ok)
{ struct counts now;
  struct ids E = {0,0,0}, F = {0,0,0};
  int why;
  patch(c,verts,&now,PATCH_AFTER,&E,&F);
  if ( c->fold_ok && now.fold > was->fold ) now.fold = was->fold;
  why = worse(was,&now,pinch_ok);
  /* (a smoothing move, once a vertex in a batch, by no more than the
     grids' padding, leaves its elements in their items' boxes: added
     again, a pass over 15k facets' vertices made the grids again 76 times,
     a fifth of a remesh) */
  if ( !why && (!c->in_cells || c->readd) ) changed(c,&F,&E);
  if ( E.a ) temp_free((char *)E.a);
  if ( F.a ) temp_free((char *)F.a);
  return why;
}

/* the patch round vertex v as a smoothing move sees it, on a surface clean
   when the remesh began: what the move can change of the whole patch's
   counts (its edges and facets unchanged, no vertex's fans: the folds of
   the edges at v and across from it, its facets' areas, v and its edges
   in a solid; after the move, the crossings of its edges and facets, and
   of the patch's contact lines, as in patch()). The same differences
   before and after as the patch's, so the same verdict (the whole patch,
   twice a move, was 40% of a remesh; a move changes 18 of its 60
   elements) */
static void moved_patch(struct ctx *c, vertex_id v, struct counts *k, int stage,
                        struct ids *out_e, struct ids *out_f)
{ struct ids E = {0,0,0}, F = {0,0,0}, L = {0,0,0};
  edge_id e, e0;
  int i, guard;
  ensure_grids(c);
  memset(k,0,sizeof(*k));
  c->stamp_now++;
  EACH_EDGE_AT(v,e,e0,guard)
    if ( mark(c,positive_id(e)) ) push(&E,positive_id(e));
  for ( i = 0 ; i < E.n ; i++ )
  { facetedge_id fe0 = get_edge_fe(E.a[i]), fe = fe0;
    int n = 0;
    if ( valid_id(fe0) ) do
    { facet_id f = positive_id(get_fe_facet(fe));
      if ( valid_id(f) && mark(c,f) ) push(&F,f);
      fe = get_next_facet(fe);
    } while ( valid_id(fe) && !equal_id(fe,fe0) && ++n < 1000 );
  }
  for ( i = 0 ; i < F.n ; i++ )      /* the edges across from v */
  { facetedge_id fe = get_facet_fe(F.a[i]);
    int j;
    for ( j = 0 ; j < 3 ; j++, fe = get_next_edge(fe) )
      if ( mark(c,positive_id(get_fe_edge(fe))) ) push(&L,positive_id(get_fe_edge(fe)));
  }
  for ( i = 0 ; i < E.n ; i++ ) edge_problems(c,E.a[i],k);
  for ( i = 0 ; i < L.n ; i++ ) edge_problems(c,L.a[i],k);
  for ( i = 0 ; i < F.n ; i++ )
  { REAL n[3], area;
    pyse_facet_normal(F.a[i],n,&area);
    if ( area < 1e-10*c->h*c->h ) k->degenerate++;
  }
  k->solid += vertex_solids(c,v);
  for ( i = 0 ; i < E.n ; i++ ) k->solid += edge_solids(c,E.a[i]);
  if ( stage == PATCH_AFTER )
  { struct ids P = {0,0,0};        /* the patch's edges: at v and its neighbours */
    k->crossing += star_crossings(c,&E,&F,&F,&E);
    c->stamp_now++;
    EACH_EDGE_AT(v,e,e0,guard)
    { vertex_id u = get_edge_headv(e);
      edge_id d, d0;
      int g2;
      if ( mark(c,positive_id(e)) ) push(&P,positive_id(e));
      EACH_EDGE_AT(u,d,d0,g2)
        if ( mark(c,positive_id(d)) ) push(&P,positive_id(d));
    }
    for ( i = 0 ; i < P.n ; i++ ) k->crossing += line_crossings_of(c,P.a[i],&P);
    if ( P.a ) temp_free((char *)P.a);
  }
  if ( out_e ) for ( i = 0 ; i < E.n ; i++ ) push(out_e,E.a[i]);
  if ( out_f ) for ( i = 0 ; i < F.n ; i++ ) push(out_f,F.a[i]);
  if ( E.a ) temp_free((char *)E.a);
  if ( F.a ) temp_free((char *)F.a);
  if ( L.a ) temp_free((char *)L.a);
}

static int moved_after(struct ctx *c, vertex_id v, struct counts *was)
{ struct counts now;
  struct ids E = {0,0,0}, F = {0,0,0};
  int why;
  moved_patch(c,v,&now,PATCH_AFTER,&E,&F);
  why = worse(was,&now,0);
  if ( !why && (!c->in_cells || c->readd) ) changed(c,&F,&E);
  if ( E.a ) temp_free((char *)E.a);
  if ( F.a ) temp_free((char *)F.a);
  return why;
}

static void star(vertex_id v, struct ids *out)
{ push(out,v);
  neighbours(v,out);
}

/* ---- the edits ---------------------------------------------------------- */

/* the facets round u or w (not on both) moved to p: worst cosine of the
   turn, smallest doubled area, longest new edge, and the volume swept (in
   a slab, the facets at p's image) */
static void moved_shape(struct ctx *c, vertex_id u, vertex_id w, REAL *p, REAL *worst_cos,
                        REAL *least, REAL *longest, REAL *volume)
{ vertex_id ends[2];
  int s, k, guard;
  edge_id e, e0;
  ends[0] = u; ends[1] = w;
  *worst_cos = 1.0; *least = 1e300; *longest = 0.0; *volume = 0.0;
  c->stamp_now++;
  for ( s = 0 ; s < 2 ; s++ )
  { if ( !valid_id(ends[s]) ) continue;
    EACH_EDGE_AT(ends[s],e,e0,guard)
    { facetedge_id fe0 = get_edge_fe(e), fe = fe0;
      int n = 0;
      vertex_id o = get_edge_headv(e);
      if ( !equal_id(o,u) && !equal_id(o,w) )
      { REAL ob[3], l = dist3(p,vimg(p,o,ob));
        if ( l > *longest ) *longest = l;
      }
      if ( valid_id(fe0) ) do
      { facet_id f = positive_id(get_fe_facet(fe));
        if ( valid_id(f) && mark(c,f) )
        { vertex_id v[3];
          REAL *x[3], *y[3], n0[3], n1[3], a0, a1, xb[3][3];
          int hits = 0;
          facet_verts(f,v);
          for ( k = 0 ; k < 3 ; k++ )
          { x[k] = y[k] = vimg(p,v[k],xb[k]);
            if ( equal_id(v[k],u) || equal_id(v[k],w) ) { y[k] = p; hits++; }
          }
          *volume += pyse_det3(p,x[0],x[1],x[2])/6;
          if ( hits == 1 )
          { tri_normal(x[0],x[1],x[2],n0);
            tri_normal(y[0],y[1],y[2],n1);
            a0 = norm3(n0); a1 = norm3(n1);
            if ( a1 < *least ) *least = a1;
            /* (not a facet of next to no area before: its normal is
               round-off, and the turn it made refused every collapse that
               would mend it) */
            if ( a0 > 1e-6*c->h*c->h && a1 > 0 && dot3(n0,n1)/(a0*a1) < *worst_cos )
              *worst_cos = dot3(n0,n1)/(a0*a1);
          }
        }
        fe = get_next_facet(fe);
      } while ( valid_id(fe) && !equal_id(fe,fe0) && ++n < 1000 );
    }
  }
}

static int sub_attr(vertex_id w, vertex_id u)
{ int r = compare_vertex_attr(w,u);
  return r == A_EQ_B || r == A_SUB_B;
}

/* whether merging the ends of e keeps the surface a manifold: no other edge
   between them, and no neighbour common to both but the third corners of
   e's facets (Evolver's delete_edge lets a common neighbour by when no two
   facets would coincide, and merged, the ends then join it by two edges:
   the surface pinched along an edge, a 16-bead pack's 80 of them, and its
   constraint projection failed, ERROR 3005) */
static int link_kept(edge_id e)
{ vertex_id a = get_edge_tailv(e), b = get_edge_headv(e), third[64], x;
  facetedge_id fe, fe0;
  edge_id d, d0;
  int n = 0, k, guard, round = 0;
  fe0 = fe = get_edge_fe(e);
  if ( valid_id(fe) )
    do
    { if ( n < 64 ) third[n++] = get_fe_headv(get_next_edge(fe));
      fe = get_next_facet(fe);
      /* (a ring that doesn't come back to its start: refused; unguarded,
         a 16-bead pack's remeshing after a change spun here for half an
         hour) */
      if ( ++round > 1000 ) return 0;
    } while ( valid_id(fe) && !equal_id(fe,fe0) );
  EACH_EDGE_AT(a,d,d0,guard)
  { x = get_edge_headv(d);
    if ( equal_id(x,b) )
    { if ( !equal_element(d,e) ) return 0;
      continue;
    }
    if ( !joined(b,x) ) continue;
    for ( k = 0 ; k < n ; k++ ) if ( equal_id(third[k],x) ) break;
    if ( k == n ) return 0;
  }
  return 1;
}

static int collapse(struct ctx *c, edge_id e, int relaxed)
{ vertex_id a, b, u = NULLID, w = NULLID;
  struct ids verts = {0,0,0};
  struct counts was;
  REAL p[3], cosw, least, longest, volume;
  int border_edge, k, n, why, try;
  long stamp;

  if ( get_eattr(e) & FIXED ) return PYSE_EDIT_FIXED;
  a = get_edge_tailv(e); b = get_edge_headv(e);
  border_edge = valence(e) == 1;
  for ( try = 0 ; try < 2 ; try++ )
  { vertex_id keep = try ? b : a, gone = try ? a : b;
    if ( get_vattr(gone) & (FIXED|BOUNDARY) ) continue;
    if ( !sub_attr(gone,keep) ) continue;
    /* a contact line's vertex goes only along the line */
    if ( on_border(gone) && !border_edge ) continue;
    u = keep; w = gone;
    break;
  }
  if ( !valid_id(w) )
  { if ( (get_vattr(a) & FIXED) && (get_vattr(b) & FIXED) ) return PYSE_EDIT_FIXED;
    if ( (on_border(a) || on_border(b)) && !border_edge ) return PYSE_EDIT_LINE;
    return PYSE_EDIT_CONSTRAINTS;
  }
  if ( !link_kept(e) ) return PYSE_EDIT_LINK;
  if ( compare_vertex_attr(u,w) == A_EQ_B && !(get_vattr(u) & FIXED) )
  { REAL wb[3], *xw = vimg(get_coord(u),w,wb);
    for ( k = 0 ; k < 3 ; k++ ) p[k] = (get_coord(u)[k] + xw[k])/2;
  }
  else
    for ( k = 0 ; k < 3 ; k++ ) p[k] = get_coord(u)[k];
  moved_shape(c,u,w,p,&cosw,&least,&longest,&volume);
  if ( least < 2e-10*c->h*c->h && !relaxed ) return PYSE_EDIT_DEGENERATE;
  if ( cosw < c->cos_turn && !relaxed ) return PYSE_EDIT_TURN;
  if ( c->max_len > 0 && longest > c->max_len*edge_target(c,e) ) return PYSE_EDIT_LONG;
  if ( fabs(volume) > c->max_volume*c->h*c->h*c->h ) return PYSE_EDIT_VOLUME;

  star(u,&verts); star(w,&verts);
  before(c,&verts,&was);
  stamp = top_timestamp;
  /* Evolver eliminates the head where it may: w at the head */
  if ( !equal_id(get_edge_headv(e),w) ) e = inverse_id(e);
  n = delete_edge(e);
  if ( n && valid_element(e) ) free_element(e);
  if ( !n )
  { if ( verts.a ) temp_free((char *)verts.a);
    return top_timestamp != stamp ? -PYSE_EDIT_EVOLVER : PYSE_EDIT_EVOLVER;
  }
  if ( !valid_element(u) || valid_element(w) )
  { if ( verts.a ) temp_free((char *)verts.a);
    return -PYSE_EDIT_EVOLVER;
  }
  /* (in a slab, delete_edge may have moved u into the cell: back, a
     sphere's vertices kept at the image near its centre) */
  wrap_near(u,p);
  for ( k = 0 ; k < 3 ; k++ ) get_coord(u)[k] = p[k];
  project(u);
  verts.n = 0;
  star(u,&verts);
  why = after(c,&verts,&was,0);
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? -why : PYSE_EDIT_DONE;
}

static REAL angle_at(const REAL *o, const REAL *p, const REAL *q)
{ REAL u[3], w[3], x;
  int i;
  for ( i = 0 ; i < 3 ; i++ ) { u[i] = p[i] - o[i]; w[i] = q[i] - o[i]; }
  x = dot3(u,w)/(norm3(u)*norm3(w) + 1e-300);
  return acos(x > 1 ? 1 : (x < -1 ? -1 : x));
}

static int target_valence(vertex_id v) { return on_border(v) ? 4 : 6; }

static int vertex_valence(vertex_id v)
{ edge_id e, e0;
  int guard, n = 0;
  EACH_EDGE_AT(v,e,e0,guard) n++;
  return n;
}

static int flip(struct ctx *c, edge_id e, int mode)
{ facetedge_id fe0;
  facet_id f1, f2;
  vertex_id a, b, p, q, cc, d, v1[3], v2[3];
  REAL n1[3], n2[3], m1[3], m2[3], avg[3], an, l1, l2, qb[3], cb[3], db[3], *xp, *xq, *xc, *xd;
  struct ids verts = {0,0,0};
  struct counts was;
  int k, why, s1, s2;
  body_id fr1, bk1, fr2, bk2;

  if ( get_eattr(e) & FIXED ) return PYSE_EDIT_FIXED;
  if ( valence(e) != 2 ) return PYSE_EDIT_NOT_APPLICABLE;
  if ( get_eattr(e) & (CONSTRAINT|BOUNDARY) ) return PYSE_EDIT_CONSTRAINTS;
  fe0 = get_edge_fe(e);
  f1 = get_fe_facet(fe0); f2 = get_fe_facet(get_next_facet(fe0));
  s1 = inverted(f1); s2 = inverted(f2);
  fr1 = get_facet_body(positive_id(f1)); bk1 = get_facet_body(inverse_id(positive_id(f1)));
  fr2 = get_facet_body(positive_id(f2)); bk2 = get_facet_body(inverse_id(positive_id(f2)));
  if ( !(equal_id(fr1,fr2) && equal_id(bk1,bk2) && s1 != s2) ) return PYSE_EDIT_ORIENTATION;
  a = get_edge_tailv(e); b = get_edge_headv(e);
  facet_verts(f1,v1); facet_verts(f2,v2);
  /* f1 as (p,q,cc) round its orientation; f2 is (q,p,d) */
  for ( k = 0 ; k < 3 ; k++ )
    if ( (equal_id(v1[k],a) || equal_id(v1[k],b))
         && (equal_id(v1[(k+1)%3],a) || equal_id(v1[(k+1)%3],b)) ) break;
  if ( k == 3 ) return PYSE_EDIT_NOT_APPLICABLE;
  p = v1[k]; q = v1[(k+1)%3]; cc = v1[(k+2)%3];
  d = NULLID;
  for ( k = 0 ; k < 3 ; k++ ) if ( !equal_id(v2[k],a) && !equal_id(v2[k],b) ) d = v2[k];
  if ( !valid_id(d) || equal_id(cc,d) || joined(cc,d) ) return PYSE_EDIT_NOT_APPLICABLE;
  /* no chord between two contact lines' vertices */
  if ( on_border(cc) && on_border(d) && common_wall(cc,d) ) return PYSE_EDIT_LINE;
  /* (the four at p's image) */
  xp = get_coord(p); xq = vimg(xp,q,qb); xc = vimg(xp,cc,cb); xd = vimg(xp,d,db);
  if ( mode == 1 )
  { if ( angle_at(xc,xp,xq) + angle_at(xd,xp,xq) <= M_PI + 1e-9 )
      return PYSE_EDIT_NOT_BETTER;
  }
  else if ( mode == 2 )
  { int before = abs(vertex_valence(p) - target_valence(p)) + abs(vertex_valence(q) - target_valence(q))
               + abs(vertex_valence(cc) - target_valence(cc)) + abs(vertex_valence(d) - target_valence(d));
    int later = abs(vertex_valence(p) - 1 - target_valence(p)) + abs(vertex_valence(q) - 1 - target_valence(q))
              + abs(vertex_valence(cc) + 1 - target_valence(cc)) + abs(vertex_valence(d) + 1 - target_valence(d));
    if ( later >= before ) return PYSE_EDIT_NOT_BETTER;
  }
  tri_normal(xp,xq,xc,n1);
  tri_normal(xq,xp,xd,n2);
  tri_normal(xp,xd,xc,m1);
  tri_normal(xq,xc,xd,m2);
  l1 = norm3(n1); l2 = norm3(n2);
  if ( l1 <= 0 && l2 <= 0 ) return PYSE_EDIT_DEGENERATE;
  /* (weighted by area: a facet of next to no area has a normal that is
     round-off, and given the other's weight it turned every flip that
     would mend it: a pack's remeshed surface kept 109 such, refused) */
  for ( k = 0 ; k < 3 ; k++ ) avg[k] = n1[k] + n2[k];
  an = norm3(avg);
  if ( norm3(m1) < 2e-10*c->h*c->h || norm3(m2) < 2e-10*c->h*c->h ) return PYSE_EDIT_DEGENERATE;
  if ( an <= 0 || dot3(m1,avg) < c->cos_turn*norm3(m1)*an || dot3(m2,avg) < c->cos_turn*norm3(m2)*an )
    return PYSE_EDIT_TURN;
  if ( fabs(pyse_det3(xp,xq,xc,xd))/6
       > c->max_volume*c->h*c->h*c->h )
    return PYSE_EDIT_VOLUME;

  push(&verts,p); push(&verts,q); push(&verts,cc); push(&verts,d);
  before(c,&verts,&was);
  if ( !edgeswap(e) )
  { if ( verts.a ) temp_free((char *)verts.a);
    return PYSE_EDIT_EVOLVER;
  }
  why = after(c,&verts,&was,0);
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? -why : PYSE_EDIT_DONE;
}

static int split(struct ctx *c, edge_id e)
{ struct ids verts = {0,0,0};
  struct counts was;
  vertex_id a = get_edge_tailv(e), b = get_edge_headv(e), m;
  int why;
  if ( get_eattr(e) & FIXED ) return PYSE_EDIT_FIXED;    /* (its midpoint projected would move it) */
  star(a,&verts); star(b,&verts);
  before(c,&verts,&was);
  if ( !valid_id(edge_refine(e)) )
  { if ( verts.a ) temp_free((char *)verts.a);
    return PYSE_EDIT_EVOLVER;
  }
  m = get_edge_headv(e);                 /* e is the tail half now */
  project(m);
  if ( sizes ) set_size(m,0.5*(vertex_size(c,a) + vertex_size(c,b)));
  if ( region && (in_region(a) || in_region(b)) ) add_to_region(m);
  verts.n = 0;
  star(m,&verts); push(&verts,a); push(&verts,b);
  why = after(c,&verts,&was,0);
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? -why : PYSE_EDIT_DONE;
}

static int smooth(struct ctx *c, vertex_id v)
{ REAL *x = get_coord(v), old[3], target[3], d[3], n[3], len;
  struct ids verts = {0,0,0};
  struct counts was;
  edge_id e, e0;
  int guard, k, cnt = 0, border = 0, why, few;
  vertex_id line[2];

  if ( get_vattr(v) & (FIXED|BOUNDARY) ) return PYSE_EDIT_FIXED;
  if ( constraint_count(v) > 1 ) return PYSE_EDIT_CONSTRAINTS;
  for ( k = 0 ; k < 3 ; k++ ) { target[k] = 0; n[k] = 0; old[k] = x[k]; }
  EACH_EDGE_AT(v,e,e0,guard)
  { int val = valence(e);
    REAL yb[3], *y = vimg(x,get_edge_headv(e),yb);
    if ( val > 2 || val == 0 ) return PYSE_EDIT_NOT_APPLICABLE;
    if ( val == 1 )
    { if ( border < 2 ) line[border] = get_edge_headv(e);
      border++;
    }
    for ( k = 0 ; k < 3 ; k++ ) target[k] += y[k];
    cnt++;
  }
  if ( !cnt ) return PYSE_EDIT_NOT_APPLICABLE;
  if ( border && border != 2 ) return PYSE_EDIT_NOT_APPLICABLE;
  if ( border == 2 )
  { /* along the contact line, between its two neighbours there */
    REAL b0[3], b1[3], *y0 = vimg(x,line[0],b0), *y1 = vimg(x,line[1],b1), t[3], s;
    if ( !(get_vattr(v) & CONSTRAINT) ) return PYSE_EDIT_CONSTRAINTS;
    for ( k = 0 ; k < 3 ; k++ ) t[k] = y1[k] - y0[k];
    len = norm3(t);
    if ( len <= 0 ) return PYSE_EDIT_NOT_APPLICABLE;
    s = 0;
    for ( k = 0 ; k < 3 ; k++ ) s += ((y0[k] + y1[k])/2 - x[k])*t[k]/len;
    for ( k = 0 ; k < 3 ; k++ ) d[k] = c->lambda*s*t[k]/len;
  }
  else
  { /* in the tangent plane, toward the neighbours' centroid */
    c->stamp_now++;
    EACH_EDGE_AT(v,e,e0,guard)
    { facetedge_id fe0 = get_edge_fe(e), fe = fe0;
      int m = 0;
      if ( valid_id(fe0) ) do
      { facet_id f = positive_id(get_fe_facet(fe));
        if ( mark(c,f) )
        { REAL fn[3], a;
          pyse_facet_normal(f,fn,&a);
          for ( k = 0 ; k < 3 ; k++ ) n[k] += fn[k];
        }
        fe = get_next_facet(fe);
      } while ( valid_id(fe) && !equal_id(fe,fe0) && ++m < 1000 );
    }
    len = norm3(n);
    for ( k = 0 ; k < 3 ; k++ ) d[k] = c->lambda*(target[k]/cnt - x[k]);
    if ( len > 0 )
    { REAL s = dot3(d,n)/len;
      for ( k = 0 ; k < 3 ; k++ ) d[k] -= s*n[k]/len;
    }
  }
  len = norm3(d);
  if ( len > 0.25*c->h ) for ( k = 0 ; k < 3 ; k++ ) d[k] *= 0.25*c->h/len;
  if ( len < 1e-12*c->h ) return PYSE_EDIT_NOT_BETTER;
  /* (not the turn of each facet: smoothing is for the facets whose normal
     is next to nothing; none turned over against the vertex's normal) */
  { REAL p[3], fn[3], vn[3] = {0,0,0}, vlen;
    vertex_id fv[3];
    c->stamp_now++;
    EACH_EDGE_AT(v,e,e0,guard)
    { facetedge_id fe0 = get_edge_fe(e), fe = fe0;
      int m = 0;
      if ( valid_id(fe0) ) do
      { facet_id f = positive_id(get_fe_facet(fe));
        REAL a;
        if ( mark(c,f) ) { pyse_facet_normal(f,fn,&a); for ( k = 0 ; k < 3 ; k++ ) vn[k] += fn[k]; }
        fe = get_next_facet(fe);
      } while ( valid_id(fe) && !equal_id(fe,fe0) && ++m < 1000 );
    }
    vlen = norm3(vn);
    for ( k = 0 ; k < 3 ; k++ ) p[k] = x[k] + d[k];
    c->stamp_now++;
    EACH_EDGE_AT(v,e,e0,guard)
    { facetedge_id fe0 = get_edge_fe(e), fe = fe0;
      int m = 0;
      if ( valid_id(fe0) ) do
      { facet_id f = positive_id(get_fe_facet(fe));
        if ( mark(c,f) )
        { REAL *y[3], yb[3][3];
          facet_verts(f,fv);
          for ( k = 0 ; k < 3 ; k++ ) y[k] = equal_id(fv[k],v) ? p : vimg(p,fv[k],yb[k]);
          tri_normal(y[0],y[1],y[2],fn);
          if ( norm3(fn) < 2e-10*c->h*c->h ) return PYSE_EDIT_DEGENERATE;
          if ( vlen > 0 && dot3(fn,vn) <= 0 ) return PYSE_EDIT_TURN;
        }
        fe = get_next_facet(fe);
      } while ( valid_id(fe) && !equal_id(fe,fe0) && ++m < 1000 );
    }
  }
  if ( c->clean && c->near_on )
  { /* (the patch before() would look at: v's star) */
    ensure_grids(c);
    star(v,&verts);
    near_verify(c,&verts);
    verts.n = 0;
  }
  few = c->clean && !far_only;
  if ( few ) moved_patch(c,v,&was,PATCH_BEFORE,NULL,NULL);
  else { star(v,&verts); before(c,&verts,&was); }
  for ( k = 0 ; k < 3 ; k++ ) x[k] += d[k];
  project(v);
  { /* its elements put in the grids again where the move (projected) went
       past their items' padding, or v moved before since the grids were
       built, or they weren't built for this batch */
    long o = ordinal(v), i;
    REAL far = 0;
    for ( k = 0 ; k < 3 ; k++ ) if ( !(fabs(x[k] - old[k]) <= far) ) far = fabs(x[k] - old[k]);
    if ( o >= c->moved_size )
    { long m = 2*o + 64;
      c->moved = c->moved ? (int *)temp_realloc((char *)c->moved,m*sizeof(int))
                          : (int *)temp_calloc(m,sizeof(int));
      for ( i = c->moved_size ; i < m ; i++ ) c->moved[i] = 0;
      c->moved_size = m;
    }
    c->readd = !c->fresh || !(far <= c->pad) || c->moved[o] == c->epoch;
  }
  why = few ? moved_after(c,v,&was) : after(c,&verts,&was,0);
  c->readd = 0;
  if ( why ) for ( k = 0 ; k < 3 ; k++ ) x[k] = old[k];    /* (a move undoes exactly) */
  else c->moved[ordinal(v)] = c->epoch;
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? why : PYSE_EDIT_DONE;
}

static int snap(struct ctx *c, vertex_id a, vertex_id b)
{ struct ids verts = {0,0,0}, ra = {0,0,0}, rb = {0,0,0};
  struct counts was;
  vertex_id keep, gone;
  REAL p[3];
  int i, j, k, why, r;

  if ( equal_id(a,b) || joined(a,b) ) return PYSE_EDIT_NOT_APPLICABLE;
  if ( (get_vattr(a) & FIXED) && (get_vattr(b) & FIXED) ) return PYSE_EDIT_FIXED;
  neighbours(a,&ra); neighbours(b,&rb);
  for ( i = 0 ; i < ra.n ; i++ ) for ( j = 0 ; j < rb.n ; j++ )
    if ( equal_id(ra.a[i],rb.a[j]) )
    { temp_free((char *)ra.a); temp_free((char *)rb.a);
      return PYSE_EDIT_LINK;
    }
  if ( ra.a ) temp_free((char *)ra.a);
  if ( rb.a ) temp_free((char *)rb.a);
  r = compare_vertex_attr(a,b);
  if ( r == INCOMPARABLE ) return PYSE_EDIT_CONSTRAINTS;
  keep = (r == A_SUB_B || (get_vattr(b) & FIXED)) ? b : a;
  gone = equal_id(keep,a) ? b : a;
  if ( get_vattr(gone) & FIXED ) return PYSE_EDIT_FIXED;
  { REAL bb[3], *xb = vimg(get_coord(a),b,bb);
    for ( k = 0 ; k < 3 ; k++ )
      p[k] = (r == A_EQ_B && !(get_vattr(keep) & FIXED))
             ? (get_coord(a)[k] + xb[k])/2 : get_coord(keep)[k];
  }
  star(a,&verts); star(b,&verts);
  before(c,&verts,&was);
  merge_vertex(keep,gone);
  img(get_coord(keep),p,p);             /* (at keep's image, as its edges' wraps go) */
  for ( k = 0 ; k < 3 ; k++ ) get_coord(keep)[k] = p[k];
  project(keep);
  verts.n = 0;
  star(keep,&verts);
  why = after(c,&verts,&was,1);
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? -why : PYSE_EDIT_DONE;
}

static int separate(struct ctx *c, vertex_id v)
{ facet_id fs[64];
  int parent[64], nf = 0, a, b, guard, k, why, nroots = 0, root[64];
  edge_id e, e0, es[64];
  int ne = 0, efan[64];
  struct ids verts = {0,0,0};
  struct counts was;
  REAL x[3];

  /* (a copy: making a vertex can expand the attributes, moving the blocks) */
  for ( k = 0 ; k < 3 ; k++ ) x[k] = get_coord(v)[k];
  EACH_EDGE_AT(v,e,e0,guard)
  { facetedge_id fe0 = get_edge_fe(e), fe = fe0;
    int n = 0, val = valence(e);
    if ( val == 0 || val > 2 ) return PYSE_EDIT_NOT_APPLICABLE;
    if ( ne == 64 ) return PYSE_EDIT_NOT_APPLICABLE;
    es[ne++] = e;
    do
    { facet_id f = positive_id(get_fe_facet(fe));
      for ( a = 0 ; a < nf && !equal_id(fs[a],f) ; a++ ) ;
      if ( a == nf )
      { if ( nf == 64 ) return PYSE_EDIT_NOT_APPLICABLE;
        fs[nf] = f; parent[nf] = nf; nf++;
      }
      fe = get_next_facet(fe);
    } while ( valid_id(fe) && !equal_id(fe,fe0) && ++n < 1000 );
  }
  for ( k = 0 ; k < ne ; k++ )
    if ( valence(es[k]) == 2 )
    { facetedge_id fe0 = get_edge_fe(es[k]);
      facet_id f1 = positive_id(get_fe_facet(fe0)), f2 = positive_id(get_fe_facet(get_next_facet(fe0)));
      int r1, r2;
      for ( a = 0 ; !equal_id(fs[a],f1) ; a++ ) ;
      for ( b = 0 ; !equal_id(fs[b],f2) ; b++ ) ;
      for ( r1 = a ; parent[r1] != r1 ; r1 = parent[r1] ) ;
      for ( r2 = b ; parent[r2] != r2 ; r2 = parent[r2] ) ;
      if ( r1 != r2 ) parent[r1 < r2 ? r2 : r1] = r1 < r2 ? r1 : r2;
    }
  for ( a = 0 ; a < nf ; a++ )
  { int r;
    for ( r = a ; parent[r] != r ; r = parent[r] ) ;
    for ( b = 0 ; b < nroots && root[b] != r ; b++ ) ;
    if ( b == nroots ) root[nroots++] = r;
    parent[a] = r;
  }
  if ( nroots < 2 ) return PYSE_EDIT_NOT_APPLICABLE;
  /* each edge's fan: its first facet's */
  for ( k = 0 ; k < ne ; k++ )
  { facet_id f = positive_id(get_fe_facet(get_edge_fe(es[k])));
    for ( a = 0 ; !equal_id(fs[a],f) ; a++ ) ;
    for ( b = 0 ; root[b] != parent[a] ; b++ ) ;
    efan[k] = b;
  }
  star(v,&verts);
  before(c,&verts,&was);
  for ( b = 0 ; b < nroots ; b++ )
  { vertex_id u = v;
    REAL centre[3] = {0,0,0}, dir[3], len;
    int cnt = 0;
    if ( b > 0 ) u = new_vertex(x,v);
    for ( k = 0 ; k < ne ; k++ )
      if ( efan[k] == b )
      { REAL yb[3], *y = vimg(x,get_edge_headv(es[k]),yb);
        int i;
        for ( i = 0 ; i < 3 ; i++ ) centre[i] += y[i];
        cnt++;
        if ( b > 0 ) set_edge_tailv(es[k],u);
      }
    for ( k = 0 ; k < 3 ; k++ ) dir[k] = centre[k]/(cnt ? cnt : 1) - get_coord(u)[k];
    len = norm3(dir);
    if ( len > 0 ) for ( k = 0 ; k < 3 ; k++ ) get_coord(u)[k] += c->offset*c->h*dir[k]/len;
    project(u);
    if ( b > 0 ) star(u,&verts);
  }
  top_timestamp = ++global_timestamp;
  /* (mode 1, a cut's tips: the cones a neck collapsed below the mesh
     leaves are as crumpled as the neck was, their facets folded back on
     each other; folds may grow, as they relax out. Refused, Evolver's pop
     split a pack's such vertex instead, leaving two edges between the
     same two vertices, and the descent stalled round them) */
  c->fold_ok = c->mode == 1;
  why = after(c,&verts,&was,0);
  c->fold_ok = 0;
  if ( !why )
  { /* the pinch is gone */
    struct counts now;
    patch(c,&verts,&now,PATCH_CHECK,NULL,NULL);
    if ( now.pinched >= was.pinched ) why = PYSE_EDIT_PINCH;
  }
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? -why : PYSE_EDIT_DONE;
}

static int degenerate(struct ctx *c, facet_id f)
{ vertex_id v[3];
  edge_id es[3];
  REAL n[3], area, len[3];
  facetedge_id fe;
  int k, shortest = 0, longest = 0, r;

  f = positive_id(f);
  pyse_facet_normal(f,n,&area);
  if ( area >= c->flat*c->h*c->h ) return PYSE_EDIT_NOT_APPLICABLE;
  fe = get_facet_fe(f);
  for ( k = 0 ; k < 3 ; k++, fe = get_next_edge(fe) )
  { es[k] = get_fe_edge(fe);
    v[k] = get_fe_tailv(fe);
  }
  /* twins: another facet on all three edges */
  { facetedge_id f0 = get_edge_fe(es[0]), g = f0;
    int m = 0;
    do
    { facet_id t = positive_id(get_fe_facet(g));
      if ( !equal_id(t,f) )
      { int all = 1, j;
        for ( j = 1 ; j < 3 && all ; j++ )
        { facetedge_id h0 = get_edge_fe(es[j]), hh = h0;
          int found = 0, mm = 0;
          do
          { if ( equal_id(positive_id(get_fe_facet(hh)),t) ) found = 1;
            hh = get_next_facet(hh);
          } while ( valid_id(hh) && !equal_id(hh,h0) && ++mm < 1000 );
          all = found;
        }
        if ( all )
        { struct ids verts = {0,0,0};
          struct counts was;
          int why;
          for ( j = 0 ; j < 3 ; j++ ) star(v[j],&verts);
          before(c,&verts,&was);
          if ( !dissolve_facet(f) || !dissolve_facet(t) )
          { if ( verts.a ) temp_free((char *)verts.a);
            return -PYSE_EDIT_EVOLVER;
          }
          for ( j = 0 ; j < 3 ; j++ )
            if ( valid_element(es[j]) && !valid_id(get_edge_fe(es[j])) ) dissolve_edge(es[j]);
          for ( j = 0 ; j < 3 ; j++ )
            if ( valid_element(v[j]) && !valid_id(get_vertex_edge(v[j])) ) dissolve_vertex(v[j]);
          why = after(c,&verts,&was,0);
          if ( verts.a ) temp_free((char *)verts.a);
          return why ? -why : PYSE_EDIT_DONE;
        }
      }
      g = get_next_facet(g);
    } while ( valid_id(g) && !equal_id(g,f0) && ++m < 1000 );
  }
  for ( k = 0 ; k < 3 ; k++ )
  { len[k] = edge_length(es[k]);
    if ( len[k] < len[shortest] ) shortest = k;
    if ( len[k] > len[longest] ) longest = k;
  }
  /* a needle: its short edge collapsed; a cap: its long edge flipped */
  if ( len[shortest] < 0.3*len[longest] )
  { r = collapse(c,positive_id(es[shortest]),1);
    if ( r == PYSE_EDIT_DONE || r < 0 ) return r;
  }
  return flip(c,positive_id(es[longest]),0);
}

/* ---- events: merge and bridge -------------------------------------------- */

/* a facet on three oriented edges (the new_facet command's way), with its
   bodies and its tension as like's */
static facet_id make_facet(edge_id *e, body_id front, body_id back, facet_id like)
{ facet_id f = new_facet();
  facetedge_id fe = NULLID, old = NULLID, first = NULLID;
  int i;
  for ( i = 0 ; i < 3 ; i++ )
  { facetedge_id edge_fe;
    fe = new_facetedge(f,e[i]);
    if ( !valid_id(first) ) { set_facet_fe(f,fe); first = fe; }
    else { set_next_edge(old,fe); set_prev_edge(fe,old); }
    old = fe;
    edge_fe = get_edge_fe(e[i]);
    if ( valid_id(edge_fe) && !equal_id(edge_fe,fe) )
    { set_next_facet(fe,get_next_facet(edge_fe));
      set_prev_facet(fe,edge_fe);
      set_prev_facet(get_next_facet(fe),fe);
      set_next_facet(edge_fe,fe);
    }
    else
    { set_next_facet(fe,fe);
      set_prev_facet(fe,fe);
      set_edge_fe(e[i],fe);
    }
  }
  set_next_edge(old,first);
  set_prev_edge(first,old);
  if ( valid_id(like) )
  { set_facet_density(f,get_facet_density(like));
    if ( get_fattr(like) & DENSITY ) set_attr(f,DENSITY);
  }
  for ( i = 0 ; i < 3 ; i++ ) fe_reorder(positive_id(e[i]));
  recalc_facet_area(f);
  need_fe_reorder_flag = 1;
  if ( web.skel[BODY].count )
  { set_facet_body(f,front);
    set_facet_body(inverse_id(f),back);
  }
  top_timestamp = ++global_timestamp;
  return f;
}

/* an oriented edge from a to b: an existing one, or a new one (in a slab,
   its wrap from its ends: to b's image nearest a) */
static edge_id edge_between(vertex_id a, vertex_id b)
{ edge_id e, e0;
  int guard;
  EACH_EDGE_AT(a,e,e0,guard)
    if ( equal_id(get_edge_headv(e),b) ) return e;
  e = new_edge(a,b,NULLID);
  if ( slab_on ) pyse_edge_rewrap(e);
  return e;
}

/* the side of facet f (as its loop goes) toward point p: 1 front, 0 back */
static int faces(facet_id f, const REAL *p)
{ vertex_id v[3];
  REAL n[3], d[3], b0[3], b1[3], b2[3], *x0, *x1, *x2;
  int k;
  facet_verts(f,v);
  x0 = vimg(p,v[0],b0); x1 = vimg(x0,v[1],b1); x2 = vimg(x0,v[2],b2);
  tri_normal(x0,x1,x2,n);
  for ( k = 0 ; k < 3 ; k++ )
    d[k] = p[k] - (x0[k] + x1[k] + x2[k])/3;
  return dot3(n,d) > 0;
}

/* merge two facing facets: both taken out, a short tube of six facets
   joins their edges (each matched to the nearest of the other's, round the
   way that keeps the surface oriented); the bodies inside the two sheets
   now meet through the tube (one body: the caller's). The facets' edges
   must be on two facets each, not on a constraint or fixed, and the sheets
   must face each other across one body (or none). */
static int merge_facets(struct ctx *c, facet_id f1, facet_id f2)
{ vertex_id a[3], b[3];
  edge_id ea[3], eb[3];
  facetedge_id fe;
  REAL ca[3] = {0,0,0}, cb[3] = {0,0,0};
  body_id front, back, front2, back2;
  struct ids verts = {0,0,0};
  struct counts was;
  int i, k, best = 0, side1, side2, why;
  REAL bestd = 1e300;

  f1 = positive_id(f1); f2 = positive_id(f2);
  if ( equal_id(f1,f2) ) return PYSE_EDIT_NOT_APPLICABLE;
  fe = get_facet_fe(f1);
  for ( i = 0 ; i < 3 ; i++, fe = get_next_edge(fe) ) { a[i] = get_fe_tailv(fe); ea[i] = get_fe_edge(fe); }
  fe = get_facet_fe(f2);
  for ( i = 0 ; i < 3 ; i++, fe = get_next_edge(fe) ) { b[i] = get_fe_tailv(fe); eb[i] = get_fe_edge(fe); }
  for ( i = 0 ; i < 3 ; i++ ) for ( k = 0 ; k < 3 ; k++ )
    if ( equal_id(a[i],b[k]) || joined(a[i],b[k]) ) return PYSE_EDIT_LINK;
  for ( i = 0 ; i < 3 ; i++ )
  { if ( valence(ea[i]) != 2 || valence(eb[i]) != 2 ) return PYSE_EDIT_LINE;
    if ( (get_eattr(ea[i]) | get_eattr(eb[i])) & (FIXED|CONSTRAINT|BOUNDARY) )
      return PYSE_EDIT_CONSTRAINTS;
  }
  /* (in a slab, both at a[0]'s image) */
  for ( i = 0 ; i < 3 ; i++ )
  { REAL ab[3], bb[3], *xa = vimg(get_coord(a[0]),a[i],ab), *xb = vimg(get_coord(a[0]),b[i],bb);
    for ( k = 0 ; k < 3 ; k++ ) { ca[k] += xa[k]/3; cb[k] += xb[k]/3; }
  }
  side1 = faces(f1,cb); side2 = faces(f2,ca);
  front = get_facet_body(f1); back = get_facet_body(inverse_id(f1));
  front2 = get_facet_body(f2); back2 = get_facet_body(inverse_id(f2));
  if ( side1 != side2 )
  { /* two facets all but on each other (a sheet of liquid 6e-6 thick, its
       hole refused run after run): which side each faces is round-off, so
       by their normals, facing each other, the gap on the side whose bodies
       agree */
    REAL n1[3], n2[3], gap[3], l1, l2, y[4][3], *xa, *xb;
    vertex_id va[3], vb[3];
    facet_verts(f1,va); facet_verts(f2,vb);
    xa = get_coord(va[0]); xb = get_coord(vb[0]);
    tri_normal(xa,vimg(xa,va[1],y[0]),vimg(xa,va[2],y[1]),n1);
    tri_normal(xb,vimg(xb,vb[1],y[2]),vimg(xb,vb[2],y[3]),n2);
    for ( k = 0 ; k < 3 ; k++ ) gap[k] = cb[k] - ca[k];
    l1 = norm3(n1); l2 = norm3(n2);
    if ( !(l1 > 0 && l2 > 0) || dot3(n1,n2) > -0.5*l1*l2
         || fabs(dot3(n1,gap)) > 1e-3*l1*sqrt(l1) || fabs(dot3(n2,gap)) > 1e-3*l2*sqrt(l2) )
      return PYSE_EDIT_ORIENTATION;
    if ( equal_id(front,front2) ) side1 = side2 = 0;
    else if ( equal_id(back,back2) ) side1 = side2 = 1;
    else return PYSE_EDIT_ORIENTATION;
  }
  /* across the gap (where the normals point: out of the front body), the
     same body or none */
  if ( !equal_id(side1 ? back : front, side2 ? back2 : front2) ) return PYSE_EDIT_ORIENTATION;
  /* a[i] with b[(k - i) % 3]: the tube's facets go round both edges as the
     facets did */
  for ( k = 0 ; k < 3 ; k++ )
  { REAL d = 0;
    for ( i = 0 ; i < 3 ; i++ ) d += vdist(a[i],b[((k - i) % 3 + 3) % 3]);
    if ( d < bestd ) { bestd = d; best = k; }
  }
  for ( i = 0 ; i < 3 ; i++ ) { star(a[i],&verts); star(b[i],&verts); }
  before(c,&verts,&was);
  /* out with the two facets (their bodies first: dissolve_facet keeps a
     facet between two bodies) */
  set_facet_body(f1,NULLID); set_facet_body(inverse_id(f1),NULLID);
  set_facet_body(f2,NULLID); set_facet_body(inverse_id(f2),NULLID);
  if ( !dissolve_facet(f1) || !dissolve_facet(f2) )
  { if ( verts.a ) temp_free((char *)verts.a);
    return -PYSE_EDIT_EVOLVER;
  }
  for ( i = 0 ; i < 3 ; i++ )
  { int i1 = (i + 1) % 3;
    int j = ((best - i - 1) % 3 + 3) % 3, j1 = (j + 1) % 3;   /* a[i1] ~ b[j], a[i] ~ b[j1] */
    edge_id t1[3], t2[3];
    edge_id rung_i = edge_between(a[i],b[j1]), rung_i1 = edge_between(a[i1],b[j]);
    edge_id diag = edge_between(b[j],a[i]);
    edge_id eaa = equal_id(get_edge_tailv(ea[i]),a[i]) ? ea[i] : inverse_id(ea[i]);
    edge_id ebb = NULLID;
    for ( k = 0 ; k < 3 ; k++ )
      if ( equal_id(get_edge_tailv(eb[k]),b[j]) && equal_id(get_edge_headv(eb[k]),b[j1]) ) ebb = eb[k];
      else if ( equal_id(get_edge_tailv(eb[k]),b[j1]) && equal_id(get_edge_headv(eb[k]),b[j]) )
        ebb = inverse_id(eb[k]);
    if ( !valid_id(ebb) ) { if ( verts.a ) temp_free((char *)verts.a); return -PYSE_EDIT_EVOLVER; }
    t1[0] = eaa; t1[1] = rung_i1; t1[2] = diag;                       /* a_i a_i1 b_j */
    t2[0] = inverse_id(diag); t2[1] = ebb; t2[2] = inverse_id(rung_i); /* a_i b_j b_j1 */
    make_facet(t1,front,back,f1);
    make_facet(t2,front,back,f1);
  }
  /* (mode 1, a sheet's hole: its faces closer than about a fifth of the
     facets' size, the tube between them lies all but flat and its facets
     fold back on the faces' (normals over 162 degrees apart) whatever the
     pair: a pack's sheet 0.003 thin on facets of 0.04 refused every pair,
     and the descent ran away; the hole's rim opens as it relaxes) */
  c->fold_ok = c->mode == 1;
  why = after(c,&verts,&was,0);
  c->fold_ok = 0;
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? -why : PYSE_EDIT_DONE;
}

/* the oriented way a border edge's facet goes along it */
static edge_id along_facet(edge_id e)
{ facet_id f = get_fe_facet(get_edge_fe(e));
  return inverted(f) ? inverse_id(e) : e;
}

/* two contact lines joined across the dry ground between them on one
   wall: facing border edges a1->a2 and b1->b2 (as their facets go) bridged
   by two facets (one where the lines already share a vertex), whose outer
   edges are the contact line now; the two edges inside, off the wall. The
   lines must run opposite ways (a1 across from b2). */
static int bridge(struct ctx *c, edge_id e1, edge_id e2)
{ vertex_id a1, a2, b1, b2;
  edge_id g1, g2, outer[2] = {NULLID,NULLID};
  facet_id fa, fb;
  body_id front, back;
  conmap_t *map;
  unsigned int m;
  int wall = -1, w, why, shared = 0;
  struct ids verts = {0,0,0};
  struct counts was;

  if ( valence(e1) != 1 || valence(e2) != 1 ) return PYSE_EDIT_NOT_APPLICABLE;
  if ( equal_id(positive_id(e1),positive_id(e2)) ) return PYSE_EDIT_NOT_APPLICABLE;
  g1 = along_facet(e1); g2 = along_facet(e2);
  a1 = get_edge_tailv(g1); a2 = get_edge_headv(g1);
  b1 = get_edge_tailv(g2); b2 = get_edge_headv(g2);
  /* a wall both edges are on */
  if ( get_eattr(g1) & CONSTRAINT )
  { map = get_e_constraint_map(g1);
    for ( m = 1 ; m <= map[0] && wall < 0 ; m++ )
      for ( w = 0 ; w < c->nwalls ; w++ )
        if ( (int)(map[m] & CONMASK) == c->walls[w] && e_on_constraint(g2,c->walls[w]) )
          wall = c->walls[w];
  }
  if ( wall < 0 ) return PYSE_EDIT_CONSTRAINTS;
  fa = get_fe_facet(get_edge_fe(g1)); fb = get_fe_facet(get_edge_fe(g2));
  front = get_facet_body(positive_id(fa)); back = get_facet_body(inverse_id(positive_id(fa)));
  /* the same body on one side (the gas, say); on the other the same, or two
     bodies the caller makes one (two drops' liquids) */
  { body_id front2 = get_facet_body(positive_id(fb)), back2 = get_facet_body(inverse_id(positive_id(fb)));
    if ( !equal_id(front,front2) && !equal_id(back,back2) ) return PYSE_EDIT_ORIENTATION;
    if ( !equal_id(front,front2) && !(valid_id(front) && valid_id(front2)) ) return PYSE_EDIT_ORIENTATION;
    if ( !equal_id(back,back2) && !(valid_id(back) && valid_id(back2)) ) return PYSE_EDIT_ORIENTATION;
  }
  if ( equal_id(a1,b2) || equal_id(a2,b1) ) shared = 1;
  else if ( equal_id(a1,b1) || equal_id(a2,b2) ) return PYSE_EDIT_ORIENTATION;
  else
  { /* running opposite ways: a1 across from b2 */
    if ( vdist(a1,b2) + vdist(a2,b1) >= vdist(a1,b1) + vdist(a2,b2) )
      return PYSE_EDIT_ORIENTATION;
    if ( joined(a1,b2) || joined(a2,b1) ) return PYSE_EDIT_LINK;
  }
  star(a1,&verts); star(a2,&verts); star(b1,&verts); star(b2,&verts);
  before(c,&verts,&was);
  if ( shared )
  { /* one facet: (a2, a1, b1) where a1 = b2, or (a2, a1, b2) where a2 = b1 */
    edge_id t[3];
    if ( equal_id(a1,b2) )
    { outer[0] = edge_between(b1,a2);
      t[0] = inverse_id(g1); t[1] = inverse_id(g2); t[2] = outer[0];
    }
    else
    { outer[0] = edge_between(a1,b2);
      t[0] = inverse_id(g1); t[1] = outer[0]; t[2] = inverse_id(g2);
    }
    make_facet(t,front,back,positive_id(fa));
  }
  else
  { edge_id t1[3], t2[3], diag;
    outer[0] = edge_between(a1,b2);
    outer[1] = edge_between(b1,a2);
    if ( vdist(a2,b2) <= vdist(a1,b1) )
    { diag = edge_between(b2,a2);                                     /* a2 a1 b2 | a2 b2 b1 */
      t1[0] = inverse_id(g1); t1[1] = outer[0]; t1[2] = diag;
      t2[0] = inverse_id(diag); t2[1] = inverse_id(g2); t2[2] = outer[1];
    }
    else
    { diag = edge_between(a1,b1);                                     /* a2 a1 b1 | a1 b2 b1 */
      t1[0] = inverse_id(g1); t1[1] = diag; t1[2] = outer[1];
      t2[0] = outer[0]; t2[1] = inverse_id(g2); t2[2] = inverse_id(diag);
    }
    make_facet(t1,front,back,positive_id(fa));
    make_facet(t2,front,back,positive_id(fa));
  }
  /* the new contact line on the wall (after the facets: its content
     integrals go by their bodies), the old one off it */
  set_e_constraint_map(outer[0],wall);
  if ( valid_id(outer[1]) ) set_e_constraint_map(outer[1],wall);
  unset_e_constraint_map(g1,wall);
  unset_e_constraint_map(g2,wall);
  top_timestamp = ++global_timestamp;
  why = after(c,&verts,&was,0);
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? -why : PYSE_EDIT_DONE;
}

/* a wet or dry patch on wall c: the facets taken away, their vertices put
   on c first (projected; one on another wall too becomes a corner, on
   both: the gas reaching the floor along a side wall), the edges and
   vertices left bare gone, the new border edges on c as contact lines
   stored the way their facet goes (orientation 1), a vertex the new line
   goes through twice separated. The caller keeps the volumes. */
static int patch_on(struct ctx *c, int wall, element_id *fs, int nf)
{ struct ids verts = {0,0,0}, edges = {0,0,0}, ring = {0,0,0};
  struct counts was;
  int i, k, w, why, known = 0;
  facetedge_id fe;

  for ( w = 0 ; w < c->nwalls ; w++ ) if ( c->walls[w] == wall ) known = w + 1;
  if ( !known ) return PYSE_EDIT_CONSTRAINTS;
  if ( nf < 1 ) return PYSE_EDIT_NOT_APPLICABLE;
  c->stamp_now++;
  for ( i = 0 ; i < nf ; i++ )
  { facet_id f = positive_id(fs[i]);
    if ( !valid_element(f) ) return PYSE_EDIT_GONE;
    fe = get_facet_fe(f);
    for ( k = 0 ; k < 3 ; k++, fe = get_next_edge(fe) )
    { vertex_id v = get_fe_tailv(fe);
      edge_id e = positive_id(get_fe_edge(fe));
      if ( mark(c,v) )
      { if ( get_vattr(v) & (FIXED|BOUNDARY) ) return PYSE_EDIT_FIXED;
        if ( !v_on_constraint(v,wall) && constraint_count(v) > 1 ) return PYSE_EDIT_CONSTRAINTS;
        if ( !v_on_constraint(v,wall) )
        { REAL at[MAXCOORD];
          wall_image(c,known - 1,v,at);
          if ( tangent_walls(v,wall,at) ) return PYSE_EDIT_CONSTRAINTS;
        }
        push(&verts,v);
      }
      if ( mark(c,e) ) push(&edges,e);
    }
  }
  for ( i = 0 ; i < verts.n ; i++ ) star(verts.a[i],&ring);
  before(c,&ring,&was);
  /* onto the wall (with the others it is on: a corner); in a slab, from
     its image nearest the wall's (a sphere's vertices kept there) */
  for ( i = 0 ; i < verts.n ; i++ )
    if ( !v_on_constraint(verts.a[i],wall) )
    { if ( slab_on )
      { REAL at[MAXCOORD];
        wall_image(c,known - 1,verts.a[i],at);
        wrap_near(verts.a[i],at);
      }
      set_v_constraint_map(verts.a[i],wall);
      project_v_constr(verts.a[i],ACTUAL_MOVE,RESET_ONESIDEDNESS);
    }
  for ( i = 0 ; i < nf ; i++ )
  { facet_id f = positive_id(fs[i]);
    set_facet_body(f,NULLID); set_facet_body(inverse_id(f),NULLID);
    if ( !dissolve_facet(f) ) { why = PYSE_EDIT_EVOLVER; goto undo; }
  }
  for ( i = 0 ; i < edges.n ; i++ )
  { edge_id e = edges.a[i];
    if ( !valid_element(e) ) continue;
    if ( !valid_id(get_edge_fe(e)) ) { dissolve_edge(e); continue; }
    if ( valence(e) == 1 && !e_on_constraint(e,wall) )
    { /* stored the way its facet goes, orientation 1, then on the wall */
      if ( inverted(get_fe_facet(get_edge_fe(e))) ) reverse_orientation_edge(e);
      unset_attr(e,NEGBOUNDARY);
      set_e_constraint_map(e,wall);
    }
  }
  for ( i = 0 ; i < verts.n ; i++ )
    if ( valid_element(verts.a[i]) && !valid_id(get_vertex_edge(verts.a[i])) )
      dissolve_vertex(verts.a[i]);
  /* a vertex the new line goes through twice: one per fan */
  for ( i = 0 ; i < verts.n ; i++ )
  { vertex_id v = verts.a[i];
    edge_id e, e0;
    int guard, borders = 0;
    if ( !valid_element(v) ) continue;
    EACH_EDGE_AT(v,e,e0,guard) if ( valence(e) == 1 ) borders++;
    if ( borders >= 4 )
    { int r = separate(c,v);
      if ( r < 0 ) { why = -r; goto undo; }       /* (worse: the whole patch undone) */
      if ( r != PYSE_EDIT_DONE && r != PYSE_EDIT_NOT_APPLICABLE ) { why = PYSE_EDIT_PINCH; goto undo; }
    }
  }
  top_timestamp = ++global_timestamp;
  { struct ids keep = {0,0,0};
    for ( i = 0 ; i < ring.n ; i++ ) if ( valid_element(ring.a[i]) ) push(&keep,ring.a[i]);
    why = after(c,&keep,&was,0);
    if ( keep.a ) temp_free((char *)keep.a);
  }
  if ( !why )
  { if ( verts.a ) temp_free((char *)verts.a);
    if ( edges.a ) temp_free((char *)edges.a);
    if ( ring.a ) temp_free((char *)ring.a);
    return PYSE_EDIT_DONE;
  }
undo:
  if ( verts.a ) temp_free((char *)verts.a);
  if ( edges.a ) temp_free((char *)edges.a);
  if ( ring.a ) temp_free((char *)ring.a);
  return -why;
}

/* an edge two facets share whose ends are both on wall's contact lines made
   two contact-line edges on the wall, one for each facet (a new edge for
   the second, the facet made again on it), and its ends, where the line
   now goes through twice, separated one vertex per fan: pinch_off's cut of
   a neck against a wall collapses its ring onto the wall, the two strands
   of contact line either side of the neck ending in two vertices there,
   and Evolver keeps the edge between them (deleting it would pinch the
   line), which held the two sides together: the cut left one piece, its
   vertices pinched, and the neck came again where it was cut, over and
   over (a 16-bead pack's run cut one ten times). */
static int unzip(struct ctx *c, edge_id e)
{ facetedge_id fe1, fe;
  facet_id f1, f2;
  vertex_id a, b, ends[2];
  edge_id t[3], e2 = NULLID;
  body_id front, back;
  struct ids verts = {0,0,0};
  struct counts was;
  int wall = -1, w, k, why;

  e = positive_id(e);
  if ( get_eattr(e) & (FIXED|CONSTRAINT|BOUNDARY) ) return PYSE_EDIT_CONSTRAINTS;
  if ( valence(e) != 2 ) return PYSE_EDIT_NOT_APPLICABLE;
  a = get_edge_tailv(e); b = get_edge_headv(e);
  if ( !on_border(a) || !on_border(b) ) return PYSE_EDIT_NOT_APPLICABLE;
  if ( (get_vattr(a) | get_vattr(b)) & (FIXED|BOUNDARY) ) return PYSE_EDIT_FIXED;
  for ( w = 0 ; w < c->nwalls && wall < 0 ; w++ )
    if ( v_on_constraint(a,c->walls[w]) && v_on_constraint(b,c->walls[w]) ) wall = c->walls[w];
  if ( wall < 0 ) return PYSE_EDIT_CONSTRAINTS;
  fe1 = get_edge_fe(e);
  f1 = positive_id(get_fe_facet(fe1));
  f2 = positive_id(get_fe_facet(get_next_facet(fe1)));
  if ( equal_id(f1,f2) || (get_fattr(f2) & FIXED) ) return PYSE_EDIT_NOT_APPLICABLE;
  star(a,&verts); star(b,&verts);
  before(c,&verts,&was);
  /* f2's edges as its loop goes, e's place taken by a new edge the same way */
  fe = get_facet_fe(f2);
  for ( k = 0 ; k < 3 ; k++, fe = get_next_edge(fe) )
  { t[k] = get_fe_edge(fe);
    if ( equal_id(positive_id(t[k]),e) )
    { e2 = new_edge(get_edge_tailv(t[k]),get_edge_headv(t[k]),NULLID);
      if ( slab_on ) set_edge_wrap(e2,get_edge_wrap(t[k]));    /* (to the same image) */
      t[k] = e2;
    }
  }
  if ( !valid_id(e2) ) { if ( verts.a ) temp_free((char *)verts.a); return PYSE_EDIT_NOT_APPLICABLE; }
  front = get_facet_body(f2); back = get_facet_body(inverse_id(f2));
  set_facet_body(f2,NULLID); set_facet_body(inverse_id(f2),NULLID);
  if ( !dissolve_facet(f2) ) { if ( verts.a ) temp_free((char *)verts.a); return -PYSE_EDIT_EVOLVER; }
  make_facet(t,front,back,f1);
  /* both on the wall, stored the way their facet goes (orientation 1) */
  { edge_id both[2];
    both[0] = e; both[1] = positive_id(e2);
    for ( k = 0 ; k < 2 ; k++ )
    { if ( inverted(get_fe_facet(get_edge_fe(both[k]))) ) reverse_orientation_edge(both[k]);
      unset_attr(both[k],NEGBOUNDARY);
      set_e_constraint_map(both[k],wall);
    }
  }
  star(get_edge_tailv(e2),&verts);
  /* the ends: one vertex per fan */
  ends[0] = a; ends[1] = b;
  for ( k = 0 ; k < 2 ; k++ )
  { int r = separate(c,ends[k]);
    if ( r < 0 ) { why = -r; goto out; }
    if ( r != PYSE_EDIT_DONE ) { why = PYSE_EDIT_PINCH; goto out; }
  }
  top_timestamp = ++global_timestamp;
  { struct ids keep = {0,0,0};
    for ( k = 0 ; k < verts.n ; k++ ) if ( valid_element(verts.a[k]) ) push(&keep,verts.a[k]);
    why = after(c,&keep,&was,0);
    if ( keep.a ) temp_free((char *)keep.a);
  }
out:
  if ( verts.a ) temp_free((char *)verts.a);
  return why ? -why : PYSE_EDIT_DONE;
}

/* ---- batches ------------------------------------------------------------ */

struct edit_args
{ int op;
  const long *ids;
  int n;
  const double *params;
  const int *walls, *sides;
  int nwalls;
  int manifold;
  int *results;
  long *keys;
  int keyed;                       /* keys taken (else the elements are as when the batch began) */
};

static int op_type(int op)
{ switch ( op )
  { case PYSE_OP_COLLAPSE: case PYSE_OP_FLIP: case PYSE_OP_SPLIT: case PYSE_OP_BRIDGE:
    case PYSE_OP_UNZIP:
      return EDGE;
    case PYSE_OP_DEGENERATE: case PYSE_OP_MERGE: case PYSE_OP_PATCH: return FACET;
  }
  return VERTEX;
}

static int paired(int op)
{ return op == PYSE_OP_SNAP || op == PYSE_OP_MERGE || op == PYSE_OP_BRIDGE; }

static element_id item(int op, long num)
{ int type = op_type(op);
  element_id id;
  if ( num < 1 || num > web.skel[type].max_ord + 1 ) return NULLID;
  id = get_ordinal_id(type,num - 1);
  return valid_element(id) ? id : NULLID;
}

/* an edge or facet still the one asked for: the same ends */
static long key(element_id id)
{ if ( !valid_id(id) ) return -1;
  if ( id_type(id) == EDGE )
  { long a = ordinal(get_edge_tailv(id)), b = ordinal(get_edge_headv(id));
    return a < b ? a*1000003 + b : b*1000003 + a;
  }
  if ( id_type(id) == FACET )
  { vertex_id v[3];
    facet_verts(id,v);
    return ordinal(v[0]) + ordinal(v[1]) + ordinal(v[2]);
  }
  return ordinal(id);
}

/* the keys of the batch's elements as they are, before its first edit
   changes anything (their keys then: a batch of a whole surface's facets,
   few of them edited, took them all first, a quarter of its time); until
   then each element is the one asked for (its key taken twice to say so was
   a third of a degenerate batch's time over a 15k-facet pack, none made) */
static void keys_now(struct edit_args *a)
{ int k;
  if ( a->keyed ) return;
  for ( k = a->op == PYSE_OP_PATCH ? 1 : 0 ; k < a->n ; k++ ) a->keys[k] = key(item(a->op,a->ids[k]));
  a->keyed = 1;
}

static int one(struct ctx *c, struct edit_args *a, int k)
{ element_id id, id2;
  int op = a->op;
  if ( op == PYSE_OP_PATCH )
  { /* ids: the wall, then the facets (one edit) */
    element_id *fs = (element_id *)temp_calloc(a->n ? a->n : 1,sizeof(element_id));
    int i, r;
    for ( i = 1 ; i < a->n ; i++ )
    { fs[i-1] = item(op,a->ids[i]);
      if ( !valid_id(fs[i-1]) || (a->keyed && key(fs[i-1]) != a->keys[i]) )
      { temp_free((char *)fs); return PYSE_EDIT_GONE; }
    }
    r = patch_on(c,(int)a->ids[0],fs,a->n - 1);
    temp_free((char *)fs);
    return r;
  }
  if ( paired(op) )
  { id = item(op,a->ids[2*k]);
    id2 = item(op,a->ids[2*k+1]);
    if ( !valid_id(id) || !valid_id(id2)
         || (a->keyed && (key(id) != a->keys[2*k] || key(id2) != a->keys[2*k+1])) )
      return PYSE_EDIT_GONE;
    switch ( op )
    { case PYSE_OP_SNAP: return snap(c,id,id2);
      case PYSE_OP_MERGE: return merge_facets(c,id,id2);
      case PYSE_OP_BRIDGE: return bridge(c,id,id2);
    }
  }
  id = item(op,a->ids[k]);
  if ( !valid_id(id) || (a->keyed && key(id) != a->keys[k]) ) return PYSE_EDIT_GONE;
  switch ( op )
  { case PYSE_OP_COLLAPSE: return collapse(c,id,0);
    case PYSE_OP_FLIP: return flip(c,id,c->mode);
    case PYSE_OP_SPLIT: return split(c,id);
    case PYSE_OP_SMOOTH: return smooth(c,id);
    case PYSE_OP_SEPARATE: return separate(c,id);
    case PYSE_OP_DEGENERATE: return degenerate(c,id);
    case PYSE_OP_UNZIP: return unzip(c,id);
  }
  return PYSE_EDIT_NOT_APPLICABLE;
}

/* one batch: each edit in turn; one that comes out worse puts the batch's
   snapshot back, the edits made before it made again */
/* edits a batch replays at most after one is undone: the surface is
   saved again every so many (from the batch's start, a refined patch's 641
   splits, a dozen undone, replayed thousands: 11 s, 4 in chunks of 100) */
#define BATCH_CHECKPOINT 32

/* ... and on a large surface every facets/BATCH_SPAN of them until one is
   undone in the batch: a snapshot copies the whole surface, and every 32
   edits made they were a quarter of a remesh of 58k facets, quadratic in
   the facets (a refused edit then replays at most that many, once) */
#define BATCH_SPAN 64

/* the grids' padding in a smoothing batch, times h: its moves are a
   quarter edge at most before they are projected (none of a 16-bead pack's
   remesh's 90k went further; one that does puts its elements in again).
   0.15: fewer items a cell, and few moves go further (0.3 took 5% more
   instructions in a remesh of 14k facets, 0.1 and 0.06 no fewer than 0.15) */
#define SMOOTH_PAD 0.15

static void run_batch(struct ctx *c, struct edit_args *a)
{ int k, j, base, made = 0, undone = 0, count = a->op == PYSE_OP_PATCH ? 1 : paired(a->op) ? a->n/2 : a->n;
  a->keys = (long *)temp_calloc(a->n ? a->n : 1,sizeof(long));
  a->keyed = 0;
  c->batch = a;
  c->in_cells = a->op == PYSE_OP_SMOOTH;
  /* (the grids of the batch before kept where their changed lists are
     short, and either the batch is small, its edits' elements then put in
     the changed lists, or for one of splits, collapses or flips, they have
     no padding; else built again, for a smoothing batch padded so that its
     moves leave their elements in their items. A remesh near a change on a
     30k-facet pack built them 13 times for 100 vertices, a fifth of its
     time; and a remesh near a change, its batches a region's, keeps them
     where its candidates are a sixteenth of the elements: its flips and
     moves after the grids were padded or not built them again each sweep,
     a third of its time on a 15k-facet pack) */
  { long total = c->fgrid.total + c->egrid.total;
    int few = (region ? 16L : 256L)*count < total;
    int keep = c->fgrid.built && 4*(c->fgrid.xn + c->egrid.xn) <= total
               && (few || (!c->in_cells && c->pad == 0));
    /* (built when first needed: a batch whose candidates are all refused
       before they change anything builds none) */
    c->grids_due = !keep;
    c->due_pad = c->in_cells ? SMOOTH_PAD*c->h : 0.0;
    c->fresh = c->in_cells && !keep;
  }
  c->snap = NULL;
  c->snapping = a->op != PYSE_OP_SMOOTH;
  for ( k = 0 ; k < count ; k++ )
  { int r;
    c->cur_k = k;
    if ( c->snap && made >= (undone || web.skel[FACET].count < BATCH_SPAN*BATCH_CHECKPOINT
                             ? BATCH_CHECKPOINT : web.skel[FACET].count/BATCH_SPAN) )
    { /* (the edits before k kept: replays start at the next one's
         snapshot; edits made, not candidates, counted: a snapshot is the
         whole surface, and every 32 candidates a pass over all edges took
         one per 32 of them, 2700 in a remesh of 15k facets, a fifth of its
         8 s, quadratic in the facets) */
      pyse_snapshot_drop(c->snap);
      c->snap = NULL;
      made = 0;
    }
    r = one(c,a,k);
    if ( r >= 0 )
    { a->results[k] = r;
      made += r == PYSE_EDIT_DONE;
      if ( r == PYSE_EDIT_DONE ) c->changed = 1;
      continue;
    }
    a->results[k] = -r;
    undone = c->changed = 1;
    if ( !c->snap )
      kb_error(PYSE_ERR_BAD_ARGUMENT,"Internal error: an edit changed the surface before "
               "its checks.\n",RECOVERABLE);
    base = c->snap_k;
    for ( ;; )
    { int again = 0;
      pyse_snapshot_put_back(c->snap);
      grids_build(c);
      c->vok_now++;                  /* (those found clean were of a later surface) */
      for ( j = base ; j < k && !again ; j++ )
        if ( a->results[j] == PYSE_EDIT_DONE )
        { int rr = one(c,a,j);
          if ( rr != PYSE_EDIT_DONE )
          { /* (not the same again: refused, and the replay starts over
               without it where it changed something) */
            a->results[j] = PYSE_EDIT_REPLAY;
            again = rr < 0;
          }
        }
      if ( !again ) break;
    }
  }
  if ( c->snap ) pyse_snapshot_drop(c->snap);
  else if ( c->snapping && count )
    free_discards(DISCARDS_ALL);   /* (as the snapshot it didn't take: the same numbering after) */
  c->snap = NULL;
  c->snapping = 0;
  temp_free((char *)a->keys);
  a->keys = NULL;
  c->batch = NULL;
}

static void setup(struct ctx *c, struct edit_args *a)
{ int w;
  slab_set();
  if ( web.representation != SOAPFILM || web.modeltype != LINEAR || SDIM != 3
       || ((web.torus_flag || web.symmetry_flag) && !slab_on) )
    kb_error(PYSE_ERR_BAD_ARGUMENT,"The mesh edits are for linear soap films in 3D, not in "
             "torus or symmetry spaces (but slabs periodic in x and y).\n",RECOVERABLE);
  memset(c,0,sizeof(*c));
  c->walls = a->walls; c->sides = a->sides; c->nwalls = a->nwalls; c->manifold = a->manifold;
  if ( slab_on )
  { /* each wall's vertices at the image near it (a sphere's): one of them */
    vertex_id v;
    c->wref = (REAL *)temp_calloc(3*(c->nwalls + 1),sizeof(REAL));
    c->wref_ok = (int *)temp_calloc(c->nwalls + 1,sizeof(int));
    for ( w = 0 ; w < c->nwalls ; w++ )
      FOR_ALL_VERTICES(v)
        if ( v_on_constraint(v,c->walls[w]) )
        { REAL *x = get_coord(v);
          c->wref[3*w] = x[0]; c->wref[3*w+1] = x[1]; c->wref[3*w+2] = x[2];
          c->wref_ok[w] = 1;
          break;
        }
  }
  c->h = a->params[8] > 0 ? a->params[8] : pyse_mean_edge();
  c->cos_turn = cos(a->params[0]*M_PI/180);
  c->max_volume = a->params[1];
  c->max_len = a->params[2];
  c->tol = a->params[3];
  c->mode = (int)a->params[4];
  c->lambda = a->params[5];
  c->offset = a->params[6];
  c->flat = a->params[7];
}

/* recalc() after a call, but where it made no edit and the surface is as
   the last edit or remesh left it, recalculated (nothing set since: no
   command, coordinates, values, multipliers, native integrands or threads), the rest
   of what recalc() does only: it projects every vertex and evaluates every
   quantity again, to the same values (a remesh's degenerate batch, none
   made, spent 60% of its time there, a tenth of a remesh near a change on a
   15k-facet pack) */
static long recalc_mark[4] = {-1,-1,-1,-1};
static int recalc_done;

static void recalc_unless(struct ctx *c)
{ web_timestamp = top_timestamp = ++global_timestamp;
  if ( c->changed || recalc_mark[0] != pyse_surface_version()
       || recalc_mark[1] != pyse_load_serial() || recalc_mark[2] != pyse_native_serial()
       || recalc_mark[3] != pyse_threads() )
    recalc();
  else
  { global_timestamp++;
    reset_rot_order();
    change_flag = 0;
    reset_conj_grad();
    if ( normal_motion_flag ) begin_normal_motion();
  }
  recalc_done = 1;
}

static void edit_body(void *arg)
{ struct edit_args *a = (struct edit_args *)arg;
  struct ctx c;
  setup(&c,a);
  run_batch(&c,a);
  recalc_unless(&c);
}

/* ---- the remesher ------------------------------------------------------- */

static int by_length_up(const void *x, const void *y)
{ const REAL *p = (const REAL *)x, *q = (const REAL *)y;
  return p[0] < q[0] ? -1 : p[0] > q[0];
}

/* edges as element numbers: longer than above (longest first), or shorter
   than below (shortest first), or (both 0) those on two facets */
/* edges longer than above, or shorter than below, times their targets
   (longest, shortest first, as their ratio to it), or with two facets */
static long *edges_by(struct ctx *c, double above, double below, int *n)
{ edge_id e;
  long count = 0, k = 0;
  REAL *pairs;
  long *out;
  FOR_ALL_EDGES(e) count++;
  pairs = (REAL *)temp_calloc(2*(count ? count : 1),sizeof(REAL));
  FOR_ALL_EDGES(e)
  { REAL l;
    int keep;
    if ( !in_region(get_edge_tailv(e)) && !in_region(get_edge_headv(e)) ) continue;
    l = edge_length(e)/edge_target(c,e);
    keep = above > 0 ? l > above : below > 0 ? l < below : valence(e) == 2;
    if ( !keep ) continue;
    pairs[2*k] = above > 0 ? -l : l;
    pairs[2*k+1] = (REAL)(ordinal(e) + 1);
    k++;
  }
  qsort(pairs,(size_t)k,2*sizeof(REAL),by_length_up);
  out = (long *)temp_calloc(k ? k : 1,sizeof(long));
  for ( count = 0 ; count < k ; count++ ) out[count] = (long)pairs[2*count+1];
  temp_free((char *)pairs);
  *n = (int)k;
  return out;
}

static long *all_vertices(int *n)
{ vertex_id v;
  long count = 0, k = 0, *out;
  FOR_ALL_VERTICES(v) count++;
  out = (long *)temp_calloc(count ? count : 1,sizeof(long));
  FOR_ALL_VERTICES(v) if ( in_region(v) ) out[k++] = ordinal(v) + 1;
  *n = (int)k;
  return out;
}

static int count_done(int *r, int n)
{ int k, d = 0;
  for ( k = 0 ; k < n ; k++ ) d += r[k] == PYSE_EDIT_DONE;
  return d;
}

/* isotropic remeshing toward edges of h (Botsch and Kobbelt's): per sweep,
   edges over 4/3 h split, under 4/5 h collapsed (none made over 4/3 h),
   flips toward valence 6 (4 on a border), vertices smoothed along the
   surface; every edit checked as in a batch, contact lines kept (they
   are the edits' features). results: splits, collapses, flips, moves. */
static int surface_clean(struct ctx *c);
static int near_clean(struct ctx *c);

static void remesh_body(void *arg)
{ struct edit_args *a = (struct edit_args *)arg;
  struct ctx c;
  int sweep, sweeps = (int)a->params[9], n, pass;
  double keep_len, keep_mode;
  int k;
  setup(&c,a);
  c.fcache = 1;
  c.cell_h = 2.0;
  if ( a->n > 0 )
  { /* the region: the vertices numbered (the edges with an end there) */
    region = (unsigned char *)calloc(1,1);
    region_n = 1;
    for ( k = 0 ; k < a->n ; k++ )
    { vertex_id v = item(PYSE_OP_SMOOTH,a->ids[k]);
      if ( valid_id(v) ) add_to_region(v);
    }
  }
  c.clean = region ? near_clean(&c) : surface_clean(&c);
  keep_len = c.max_len; keep_mode = c.mode;
  for ( pass = 0 ; pass < 4 ; pass++ ) a->results[pass] = 0;
  for ( sweep = 0 ; sweep < (sweeps > 0 ? sweeps : 3) ; sweep++ )
  { struct edit_args b = *a;
    int moved = 0;
    for ( pass = 0 ; pass < 4 ; pass++ )
    { long *ids;
      int *res;
      if ( pass == 0 ) { ids = edges_by(&c,4.0/3,0,&n); b.op = PYSE_OP_SPLIT; }
      else if ( pass == 1 ) { ids = edges_by(&c,0,0.8,&n); b.op = PYSE_OP_COLLAPSE; c.max_len = 4.0/3; }
      else if ( pass == 2 ) { ids = edges_by(&c,0,0,&n); b.op = PYSE_OP_FLIP; c.mode = 2; }
      else { ids = all_vertices(&n); b.op = PYSE_OP_SMOOTH; }
      res = (int *)temp_calloc(n ? n : 1,sizeof(int));
      b.ids = ids; b.n = n; b.results = res;
      if ( n ) run_batch(&c,&b);
      a->results[pass] += count_done(res,n);
      if ( pass < 3 ) moved += count_done(res,n);
      c.max_len = keep_len; c.mode = (int)keep_mode;
      temp_free((char *)res);
      temp_free((char *)ids);
    }
    if ( !moved ) break;            /* (topology settled: smoothing alone doesn't repeat) */
  }
  if ( c.nkey ) temp_free((char *)c.nkey);
  if ( c.vok ) temp_free((char *)c.vok);
  if ( c.fv ) temp_free((char *)c.fv);
  free(region);
  region = NULL; region_n = 0;
  recalc_unless(&c);
}

int pyse_edit(int op, const long *ids, int n, const double *params, const int *walls,
              const int *sides, int nwalls, int manifold, int *results,
              const double *size, long nsize)
{ struct edit_args *a;
  int status, k;
  if ( op < PYSE_OP_COLLAPSE || op > PYSE_OP_UNZIP )
    return PYSE_ERROR;
  a = (struct edit_args *)calloc(1,sizeof(*a));
  if ( !a ) return PYSE_FATAL;
  if ( size && nsize > 0 )
  { sizes = (REAL *)malloc(nsize*sizeof(REAL));
    if ( !sizes ) { free(a); return PYSE_FATAL; }
    for ( k = 0 ; k < nsize ; k++ ) sizes[k] = size[k];
    sizes_n = nsize;
  }
  a->op = op; a->ids = ids; a->n = n; a->params = params; a->walls = walls;
  a->sides = sides; a->nwalls = nwalls; a->manifold = manifold; a->results = results;
  recalc_done = 0;
  if ( op == PYSE_OP_REMESH )
    status = pyse_surface_call(remesh_body,a);
  else
  { for ( k = 0 ; k < (op == PYSE_OP_PATCH ? 1 : paired(op) ? n/2 : n) ; k++ )
      results[k] = PYSE_EDIT_NOT_APPLICABLE;
    status = pyse_surface_call(edit_body,a);
  }
  pyse_surface_changed();
  recalc_mark[0] = recalc_done && status == PYSE_OK ? pyse_surface_version() : -1;
  recalc_mark[1] = pyse_load_serial();
  recalc_mark[2] = pyse_native_serial();
  recalc_mark[3] = pyse_threads();
  free(a);
  free(sizes);
  sizes = NULL; sizes_n = 0;
  free(region);                /* (an Evolver error jumps past remesh_body's) */
  region = NULL; region_n = 0;
  return status;
}

/* ---- crossings (for descent steps) -------------------------------------- */

struct crossing_args { long count; int far; double *where; long max; };

/* whether an edge's end and a facet's corner are joined by an edge: a fold
   one ring across, not two sheets through each other */
static int near_pair(edge_id e, facet_id f)
{ vertex_id v[3], ends[2];
  int i, k;
  ends[0] = get_edge_tailv(e); ends[1] = get_edge_headv(e);
  facet_verts(f,v);
  for ( i = 0 ; i < 2 ; i++ ) for ( k = 0 ; k < 3 ; k++ )
    if ( joined(ends[i],v[k]) ) return 1;
  return 0;
}


/* where a crossing is (up to max kept) */
static void record(struct crossing_args *a, long n, const REAL *x)
{ int k;
  if ( !a->where ) { a->count += n; return; }
  for ( ; n > 0 && a->count < a->max ; n--, a->count++ )
    for ( k = 0 ; k < 3 ; k++ ) a->where[3*a->count + k] = x[k];
  a->count += n;
}

/* contact lines through each other: two border edges on one wall,
   sharing no vertex, crossing as seen along the constraint's normal at their
   middle (both lie on the wall, so neither goes through a facet and
   crossings_of misses them: two lines on a 16-bead pack's bead crossed, and
   the runs after were undone for the crossings that came of it later).
   Each pair once, in the order of a scan of all pairs (by c's edge grid:
   all pairs, 3k lines on a 30k-facet pack, were most of a count) */
struct line_hit { int j; REAL mid[3]; };

/* line L[i] against the line on item it (at: L's places + 1 by ordinal):
   a crossing with one after it in L into hits */
static void line_hit(struct ctx *c, struct crossing_args *a, struct ids *L, int i, int *at,
                     const struct gitem *it, const REAL *lo, const REAL *hi,
                     struct line_hit **hits, int *nh, int *mh)
{ edge_id o;
  REAL mid[MAXCOORD];
  if ( apart(it,lo,hi,1e-9*c->h) || it->ord > web.skel[EDGE].max_ord ) return;
  if ( at[it->ord] <= i + 1 ) return;          /* (not a line, or not after L[i]) */
  o = get_ordinal_id(EDGE,it->ord);
  if ( !mark(c,o) || !lines_cross(L->a[i],o,c->h,a->far,mid) ) return;
  if ( *nh == *mh )
  { *mh = *mh ? 2**mh : 16;
    *hits = *hits ? (struct line_hit *)temp_realloc((char *)*hits,*mh*sizeof(**hits))
                  : (struct line_hit *)temp_calloc(*mh,sizeof(**hits));
  }
  (*hits)[*nh].j = at[it->ord];
  (*hits)[*nh].mid[0] = mid[0]; (*hits)[*nh].mid[1] = mid[1]; (*hits)[*nh].mid[2] = mid[2];
  (*nh)++;
}

static int by_j(const void *x, const void *y)
{ return ((const struct line_hit *)x)->j - ((const struct line_hit *)y)->j; }

static long line_crossings(struct ctx *c, struct crossing_args *a)
{ struct ids L = {0,0,0};
  struct grid *g = &c->egrid;
  struct line_hit *hits = NULL;
  edge_id e;
  long n = 0, k, x, y, z, lo_c[3], hi_c[3];
  int i, nh = 0, mh = 0, *at;
  FOR_ALL_EDGES(e)
    if ( is_line(e) ) push(&L,e);
  /* each line's place in L, + 1, by ordinal */
  at = (int *)temp_calloc(web.skel[EDGE].max_ord + 1,sizeof(int));
  for ( i = 0 ; i < L.n ; i++ ) at[ordinal(L.a[i])] = i + 1;
  for ( i = 0 ; i < L.n ; i++ )
  { REAL lo[3], hi[3];
    elem_box(c,L.a[i],lo,hi,0.0);
    cells(g,lo,hi,lo_c,hi_c);
    c->stamp_now++;
    nh = 0;
    for ( z = lo_c[2] ; z <= hi_c[2] ; z++ ) for ( y = lo_c[1] ; y <= hi_c[1] ; y++ )
      for ( x = lo_c[0] ; x <= hi_c[0] ; x++ )
      { long s = slot_probe(g,CELL(g,x,y,z));
        if ( g->key[s] < 0 ) continue;
        for ( k = g->bstart[s] ; k < g->bstart[s] + g->bcount[s] ; k++ )
          line_hit(c,a,&L,i,at,&g->items[k],lo,hi,&hits,&nh,&mh);
        for ( k = g->xhead[s] ; k >= 0 ; k = g->xnext[k] )
          line_hit(c,a,&L,i,at,&g->xitems[k],lo,hi,&hits,&nh,&mh);
      }
    if ( nh > 1 ) qsort(hits,(size_t)nh,sizeof(*hits),by_j);
    for ( k = 0 ; k < nh ; k++ ) { n++; record(a,1,hits[k].mid); }
  }
  temp_free((char *)at);
  if ( hits ) temp_free((char *)hits);
  if ( L.a ) temp_free((char *)L.a);
  return n;
}

/* whether the surface has no crossings at all (edges through facets,
   contact lines through each other), with c's grids */
static int surface_clean(struct ctx *c)
{ struct crossing_args a;
  struct ids none = {0,0,0};
  edge_id e;
  memset(&a,0,sizeof(a));
  far_only = 0;
  grids_build(c);                  /* (the first batch keeps them) */
  FOR_ALL_EDGES(e)
    if ( crossings_of(c,e,&none) ) return 0;
  line_crossings(c,&a);
  return a.count == 0;
}

/* A remesh near a change: whether it may skip the counts before its edits
   (c->clean, see patch()) asked near its region only. Every edit's patch
   is within a few rings of the region, and whether the whole surface was
   clean (surface_clean: every edge asked against the facets, 30k of them
   on a 15k-facet pack, half the time of a remesh near a change) decides
   nothing more than whether crossings in the patches have to be counted:
   the verdicts are the same either way where a patch has none, since an
   edit kept on a clean patch leaves none in it and changes nothing outside
   it. So here: the contact lines asked as surface_clean does (all of them:
   few), the edges and facets with a vertex in the cells within NEAR_CELLS
   of the region's (cells of side h, a patch's two rings at most 2.7 h
   across) asked against all; and where an edit's patch has a vertex outside
   those cells, that vertex's edges and facets asked before it (near_verify:
   a crossing found there was on the surface when the remesh began, the
   edits making none, so surface_clean would have found it: the counts are
   taken from then on, as they would have been all along, the verdicts
   before the same). The same edits as with surface_clean, but for contact
   lines crossing on two walls (lines_cross isn't symmetric there). */
#define NEAR_CELLS 3

/* (in a slab, x and y in nper cells a period, numbered round it) */
static long near_wrap(struct ctx *c, int i, long k)
{ if ( !slab_on || i > 1 ) return k & 0x1FFFFF;
  k %= c->nper[i];
  return k < 0 ? k + c->nper[i] : k;
}

static long near_key(struct ctx *c, const REAL *x, long dx, long dy, long dz)
{ long k[3], d[3] = {dx,dy,dz}, key = 0;
  int i;
  for ( i = 0 ; i < 3 ; i++ )
  { REAL s = slab_on && i < 2 ? floor(x[i]*c->nper[i]/slab_L[i]) : floor(x[i]/c->h);
    if ( !(fabs(s) < 1e6) ) return -1;      /* (not a number, or far out: outside) */
    k[i] = near_wrap(c,i,(long)s + d[i]);
    key = (key << 21) | k[i];
  }
  return key;
}

static long near_slot(struct ctx *c, long key)
{ unsigned long u = (unsigned long)key*0x9E3779B97F4A7C15UL;
  long m = c->ncap - 1, s = (long)((u ^ (u >> 32)) & (unsigned long)m);
  while ( c->nkey[s] >= 0 && c->nkey[s] != key ) s = (s + 1) & m;
  return s;
}

static void near_add(struct ctx *c, long key, long *used)
{ long s, i;
  if ( key < 0 ) return;
  if ( 2*(*used + 1) > c->ncap )
  { long old = c->ncap, *was = c->nkey;
    c->ncap = old ? 2*old : 1024;
    c->nkey = (long *)temp_calloc(c->ncap,sizeof(long));
    for ( i = 0 ; i < c->ncap ; i++ ) c->nkey[i] = -1;
    for ( i = 0 ; i < old ; i++ ) if ( was[i] >= 0 ) c->nkey[near_slot(c,was[i])] = was[i];
    if ( was ) temp_free((char *)was);
  }
  s = near_slot(c,key);
  if ( c->nkey[s] < 0 ) { c->nkey[s] = key; (*used)++; }
}

/* whether v is in the cells near the region */
static int near_in(struct ctx *c, vertex_id v)
{ long key;
  if ( !c->ncap ) return 0;
  key = near_key(c,get_coord(v),0,0,0);
  return key >= 0 && c->nkey[near_slot(c,key)] == key;
}

static int near_clean(struct ctx *c)
{ struct crossing_args a;
  struct ids none = {0,0,0}, E = {0,0,0}, F = {0,0,0};
  vertex_id v;
  long used = 0, i, dx, dy, dz, *seed = NULL, nseed = 0, mseed = 0;
  int clean = 1, guard;
  memset(&a,0,sizeof(a));
  far_only = 0;
  grids_build(c);                  /* (the first batch keeps them) */
  c->near_on = 1;
  for ( i = 0 ; i < 2 && slab_on ; i++ )
  { double n = floor(slab_L[i]/c->h);
    c->nper[i] = n > 0x100000 ? 0x100000 : n < 1 ? 1 : (long)n;
  }
  c->vok_now = 1;
  line_crossings(c,&a);
  if ( a.count ) return 0;
  /* the region's cells, then those within NEAR_CELLS of them */
  FOR_ALL_VERTICES(v)
    if ( in_region(v) ) near_add(c,near_key(c,get_coord(v),0,0,0),&used);
  for ( i = 0 ; i < c->ncap ; i++ )
    if ( c->nkey[i] >= 0 )
    { if ( nseed == mseed )
      { mseed = mseed ? 2*mseed : 1024;
        seed = seed ? (long *)temp_realloc((char *)seed,mseed*sizeof(long))
                    : (long *)temp_calloc(mseed,sizeof(long));
      }
      seed[nseed++] = c->nkey[i];
    }
  for ( i = 0 ; i < nseed ; i++ )
  { long k[3], key;
    int j;
    for ( j = 0 ; j < 3 ; j++ ) k[j] = (seed[i] >> (21*(2-j))) & 0x1FFFFF;
    for ( dz = -NEAR_CELLS ; dz <= NEAR_CELLS ; dz++ )
      for ( dy = -NEAR_CELLS ; dy <= NEAR_CELLS ; dy++ )
        for ( dx = -NEAR_CELLS ; dx <= NEAR_CELLS ; dx++ )
        { key = (near_wrap(c,0,k[0] + dx) << 42) | (near_wrap(c,1,k[1] + dy) << 21)
                | near_wrap(c,2,k[2] + dz);
          near_add(c,key,&used);
        }
  }
  if ( seed ) temp_free((char *)seed);
  /* the edges and facets with a vertex there, against all */
  c->stamp_now++;
  FOR_ALL_VERTICES(v)
  { edge_id e, e0;
    if ( !used || !near_in(c,v) ) continue;
    EACH_EDGE_AT(v,e,e0,guard)
      if ( mark(c,positive_id(e)) ) push(&E,positive_id(e));
  }
  for ( i = 0 ; i < E.n ; i++ )
  { facetedge_id fe0 = get_edge_fe(E.a[i]), fe = fe0;
    int n = 0;
    if ( valid_id(fe0) ) do
    { facet_id f = positive_id(get_fe_facet(fe));
      if ( valid_id(f) && mark(c,f) ) push(&F,f);
      fe = get_next_facet(fe);
    } while ( valid_id(fe) && !equal_id(fe,fe0) && ++n < 1000 );
  }
  for ( i = 0 ; i < E.n && clean ; i++ ) if ( crossings_of(c,E.a[i],&none) ) clean = 0;
  for ( i = 0 ; i < F.n && clean ; i++ ) if ( crossings_of(c,F.a[i],&none) ) clean = 0;
  if ( E.a ) temp_free((char *)E.a);
  if ( F.a ) temp_free((char *)F.a);
  return clean;
}

/* before an edit of a remesh near a change, clean near it: the edges and
   facets at each of verts outside the cells near the region asked against
   all, once each (until the surface is put back); one crossing, and the
   remesh counts them from here on (c->clean 0) */
static void near_verify(struct ctx *c, struct ids *verts)
{ struct ids none = {0,0,0}, E = {0,0,0}, F = {0,0,0};
  int i, k, guard;
  for ( i = 0 ; i < verts->n && c->clean ; i++ )
  { vertex_id v = verts->a[i];
    long o = ordinal(v);
    edge_id e, e0;
    if ( !valid_element(v) || near_in(c,v) ) continue;
    if ( o < c->vok_size && c->vok[o] == c->vok_now ) continue;
    E.n = F.n = 0;
    c->stamp_now++;
    EACH_EDGE_AT(v,e,e0,guard)
      if ( mark(c,positive_id(e)) ) push(&E,positive_id(e));
    for ( k = 0 ; k < E.n ; k++ )
    { facetedge_id fe0 = get_edge_fe(E.a[k]), fe = fe0;
      int n = 0;
      if ( valid_id(fe0) ) do
      { facet_id f = positive_id(get_fe_facet(fe));
        if ( valid_id(f) && mark(c,f) ) push(&F,f);
        fe = get_next_facet(fe);
      } while ( valid_id(fe) && !equal_id(fe,fe0) && ++n < 1000 );
    }
    for ( k = 0 ; k < E.n && c->clean ; k++ ) if ( crossings_of(c,E.a[k],&none) ) c->clean = 0;
    for ( k = 0 ; k < F.n && c->clean ; k++ ) if ( crossings_of(c,F.a[k],&none) ) c->clean = 0;
    if ( !c->clean ) break;
    if ( o >= c->vok_size )
    { long m = 2*o + 64, j;
      c->vok = c->vok ? (int *)temp_realloc((char *)c->vok,m*sizeof(int))
                      : (int *)temp_calloc(m,sizeof(int));
      for ( j = c->vok_size ; j < m ; j++ ) c->vok[j] = 0;
      c->vok_size = m;
    }
    c->vok[o] = c->vok_now;
  }
  if ( E.a ) temp_free((char *)E.a);
  if ( F.a ) temp_free((char *)F.a);
}

static void crossing_body(void *arg)
{ struct crossing_args *a = (struct crossing_args *)arg;
  struct ctx c;
  edge_id e;
  struct ids none = {0,0,0};
  memset(&c,0,sizeof(c));
  slab_set();
  if ( web.representation != SOAPFILM || SDIM != 3
       || ((web.torus_flag || web.symmetry_flag) && !slab_on) )
    kb_error(PYSE_ERR_BAD_ARGUMENT,"Crossings are counted for soap films in 3D only (not "
             "in torus or symmetry spaces but slabs periodic in x and y).\n",RECOVERABLE);
  c.h = pyse_mean_edge();
  grids_build(&c);
  a->count = 0;
  far_only = a->far;
  FOR_ALL_EDGES(e)
  { long n = crossings_of(&c,e,&none);
    if ( n && a->where )
    { REAL qb[3], *p = get_coord(get_edge_tailv(e)), *q = vimg(p,get_edge_headv(e),qb), mid[3];
      int k;
      for ( k = 0 ; k < 3 ; k++ ) mid[k] = (p[k] + q[k])/2;
      record(a,n,mid);
    }
    else a->count += n;
  }
  far_only = 0;
  line_crossings(&c,a);
}

int pyse_crossings(int far, long *count, double *where, long max)
{ struct crossing_args a;
  int status;
  a.count = 0;
  a.far = far;
  a.where = where;
  a.max = where ? max : 0;
  far_only = 0;
  status = pyse_surface_call(crossing_body,&a);
  far_only = 0;
  *count = a.count;
  return status;
}

/* ---- contact lines kept fine through gradient steps ---------------------- */

/* A contact-line edge on a curved wall is a chord: grown long, it cuts into
   the wall and its integrals go wrong, and the gradient follows the error (a
   16-bead pack's relax() run stretched one on a bead of radius 1 from 0.36 to
   1.9 in 18 steps, the energy 1.4 lower; split back onto the bead, it was 0.4
   higher than where the run began). With a length set, each gradient step
   (iterate()) ends with the border edges on a constraint longer than it split,
   the midpoints onto their constraints, until none is. A midpoint takes the
   one-sided constraints both ends are on too (the guards a descent keeps the
   vertices out of the solids with: a midpoint without them, on a side wall,
   went 0.05 out through the side wall it meets at the box's edge, and the
   run was undone). */
static REAL line_chop = 0.0;

double pyse_line_chop(double length)
{ double was = line_chop;
  line_chop = length > 0 ? length : 0.0;
  return was;
}

int pyse_chop_lines(void)
{ struct ids long_ = {0,0,0};
  edge_id e;
  int round, k, total = 0;
  if ( line_chop <= 0 || web.representation != SOAPFILM ) return 0;
  slab_set();
  for ( round = 0 ; round < 6 ; round++ )
  { long_.n = 0;
    FOR_ALL_EDGES(e)
    { REAL *a, *b, d[3], bb[3];
      if ( (get_eattr(e) & (CONSTRAINT|FIXED)) != CONSTRAINT || valence(e) != 1 ) continue;
      a = get_coord(get_edge_tailv(e)); b = vimg(a,get_edge_headv(e),bb);
      for ( k = 0 ; k < 3 ; k++ ) d[k] = b[k] - a[k];
      if ( norm3(d) > line_chop ) push(&long_,e);
    }
    if ( !long_.n ) break;
    for ( k = 0 ; k < long_.n ; k++ )
    { vertex_id t, h, mid;
      unsigned int m;
      e = long_.a[k];
      if ( !valid_element(e) ) continue;
      t = get_edge_tailv(e); h = get_edge_headv(e);
      if ( !valid_id(edge_refine(e)) ) continue;
      mid = get_edge_headv(e);           /* e is the tail half now */
      if ( get_vattr(t) & CONSTRAINT )
        for ( m = 1 ; m <= get_v_constraint_map(t)[0] ; m++ )
        { /* (the map read afresh: setting one can move the attributes) */
          int c = (int)(get_v_constraint_map(t)[m] & CONMASK);
          if ( (get_constraint(c)->attr & (NONNEGATIVE|NONPOSITIVE))
               && v_on_constraint(h,c) && !v_on_constraint(mid,c) )
            set_v_constraint_map(mid,c);
        }
      project(mid);
      total++;
    }
  }
  if ( long_.a ) temp_free((char *)long_.a);
  if ( total )
  { if ( count_fixed_vol() || web.pressure_flag ) calc_content(Q_FIXED);
    calc_pressure();
    calc_energy();
  }
  return total;
}
