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
 *   collapse  an edge to a point: Evolver's delete_edge (its link condition
 *             and shape checks), our rules first: a contact line's vertex
 *             only along the line, constraints kept, no facet turned more
 *             than max_turn, edges not longer than max_len, the volume swept
 *             under max_volume;
 *   flip      an edge between two facets (Evolver's edgeswap): orientation,
 *             no new chord between two contact lines' vertices, the new
 *             facets turned at most max_turn from the old, the volume
 *             between them under max_volume; mode 1 only where it makes the
 *             facets more equiangular, mode 2 only where it brings the
 *             vertices' valences nearer 6 (4 on a border);
 *   split     an edge at its midpoint (Evolver's edge_refine), projected;
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
 *             same three edges) both dissolved.
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

struct grid { REAL lo[3], cell; int n[3]; long *start, *items; int built; };

struct ctx
{ const int *walls, *sides;
  int nwalls;
  int manifold;
  double h, tol, cos_turn, max_volume, max_len, lambda, offset, flat;
  int mode;
  int *stamp[NUMELEMENTS];
  long stamp_size[NUMELEMENTS];
  int stamp_now;
  struct grid fgrid, egrid;        /* facets and edges where the batch began */
  struct ids extra_f, extra_e;     /* made or changed since */
  REAL box_lo[3], box_hi[3];
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

/* ---- spatial grids ------------------------------------------------------ */

static void elem_box(element_id id, REAL *lo, REAL *hi, REAL pad)
{ vertex_id v[3];
  int n, k, d;
  if ( id_type(id) == EDGE ) { v[0] = get_edge_tailv(id); v[1] = get_edge_headv(id); n = 2; }
  else { facet_verts(id,v); n = 3; }
  for ( k = 0 ; k < n ; k++ )
  { REAL *x = get_coord(v[k]);
    for ( d = 0 ; d < 3 ; d++ )
    { if ( k == 0 || x[d] < lo[d] ) lo[d] = x[d];
      if ( k == 0 || x[d] > hi[d] ) hi[d] = x[d];
    }
  }
  for ( d = 0 ; d < 3 ; d++ ) { lo[d] -= pad; hi[d] += pad; }
}

static void cells(struct ctx *c, struct grid *g, const REAL *lo, const REAL *hi, int *a, int *b)
{ int d;
  (void)c;
  for ( d = 0 ; d < 3 ; d++ )
  { a[d] = (int)floor((lo[d] - g->lo[d])/g->cell);
    b[d] = (int)floor((hi[d] - g->lo[d])/g->cell);
    if ( a[d] < 0 ) a[d] = 0;
    if ( b[d] < 0 ) b[d] = 0;
    if ( a[d] > g->n[d] - 1 ) a[d] = g->n[d] - 1;
    if ( b[d] > g->n[d] - 1 ) b[d] = g->n[d] - 1;
  }
}

#define CELL(g,i,j,k) (((long)(k)*(g)->n[1] + (j))*(g)->n[0] + (i))

/* elements' boxes padded by half an edge: a vertex smoothed by at most a
   quarter edge in the batch leaves its elements in their cells */
static void grid_build(struct ctx *c, struct grid *g, int type)
{ element_id id;
  long ncells, total = 0, i, *fill;
  int a[3], b[3], d, x, y, z;
  REAL lo[3], hi[3];
  if ( g->built ) { temp_free((char *)g->start); temp_free((char *)g->items); }
  for ( d = 0 ; d < 3 ; d++ ) g->lo[d] = c->box_lo[d];
  g->cell = 2*c->h;
  for ( d = 0 ; d < 3 ; d++ )
  { double n = floor((c->box_hi[d] - c->box_lo[d])/g->cell) + 1;
    g->n[d] = n > 128 ? 128 : (int)n;
  }
  for ( d = 0 ; d < 3 ; d++ )
    if ( (c->box_hi[d] - c->box_lo[d])/g->n[d] > g->cell )
      g->cell = (c->box_hi[d] - c->box_lo[d])/g->n[d]*1.0000001;
  ncells = (long)g->n[0]*g->n[1]*g->n[2];
  g->start = (long *)temp_calloc(ncells + 1,sizeof(long));
  for ( id = web.skel[type].used ; valid_id(id) ; id = elptr(id)->forechain )
  { if ( !(elptr(id)->attr & ALLOCATED) ) continue;
    if ( type == EDGE || valid_id(get_facet_fe(id)) )
    { elem_box(id,lo,hi,c->h/2);
      cells(c,g,lo,hi,a,b);
      for ( z = a[2] ; z <= b[2] ; z++ ) for ( y = a[1] ; y <= b[1] ; y++ )
        for ( x = a[0] ; x <= b[0] ; x++ ) { g->start[CELL(g,x,y,z) + 1]++; total++; }
    }
  }
  for ( i = 0 ; i < ncells ; i++ ) g->start[i+1] += g->start[i];
  g->items = (long *)temp_calloc(total ? total : 1,sizeof(long));
  fill = (long *)temp_calloc(ncells,sizeof(long));
  for ( id = web.skel[type].used ; valid_id(id) ; id = elptr(id)->forechain )
  { if ( !(elptr(id)->attr & ALLOCATED) ) continue;
    if ( type == EDGE || valid_id(get_facet_fe(id)) )
    { elem_box(id,lo,hi,c->h/2);
      cells(c,g,lo,hi,a,b);
      for ( z = a[2] ; z <= b[2] ; z++ ) for ( y = a[1] ; y <= b[1] ; y++ )
        for ( x = a[0] ; x <= b[0] ; x++ )
        { i = CELL(g,x,y,z);
          g->items[g->start[i] + fill[i]++] = ordinal(id);
        }
    }
  }
  temp_free((char *)fill);
  g->built = 1;
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
  grid_build(c,&c->fgrid,FACET);
  grid_build(c,&c->egrid,EDGE);
  c->extra_f.n = c->extra_e.n = 0;
}

/* elements an edit made or changed: queries look at them too, and the grids
   are made again once they are many */
static void changed(struct ctx *c, struct ids *f, struct ids *e)
{ int i;
  for ( i = 0 ; i < f->n ; i++ ) push(&c->extra_f,f->a[i]);
  for ( i = 0 ; i < e->n ; i++ ) push(&c->extra_e,e->a[i]);
  if ( c->extra_f.n + c->extra_e.n > 1024 ) grids_build(c);
}

/* ---- the patch's problems ----------------------------------------------- */

struct counts { int structure, pinched, degenerate, fold, solid, crossing; };

static int shares_vertex(edge_id e, facet_id f)
{ vertex_id v[3], t = get_edge_tailv(e), hh = get_edge_headv(e);
  int k;
  facet_verts(f,v);
  for ( k = 0 ; k < 3 ; k++ ) if ( equal_id(v[k],t) || equal_id(v[k],hh) ) return 1;
  return 0;
}

static int far_only;
static int near_pair(edge_id e, facet_id f);

static int pair_crosses(struct ctx *c, edge_id e, facet_id f)
{ vertex_id v[3];
  if ( !valid_element(e) || !valid_element(f) || !valid_id(get_facet_fe(f)) ) return 0;
  if ( shares_vertex(e,f) ) return 0;
  facet_verts(f,v);
  if ( far_only && near_pair(e,f) ) return 0;
  return pyse_crosses(get_coord(get_edge_tailv(e)),get_coord(get_edge_headv(e)),
                      get_coord(v[0]),get_coord(v[1]),get_coord(v[2]),1e-12*c->h*c->h*c->h);
}

static int crossings_of(struct ctx *c, element_id id, struct ids *near)
{ struct grid *g = id_type(id) == EDGE ? &c->fgrid : &c->egrid;
  struct ids *extra = id_type(id) == EDGE ? &c->extra_f : &c->extra_e;
  int other = id_type(id) == EDGE ? FACET : EDGE;
  REAL lo[3], hi[3];
  int a[3], b[3], x, y, z, i, n = 0;
  long k;
  elem_box(id,lo,hi,0.0);
  cells(c,g,lo,hi,a,b);
  c->stamp_now++;
  for ( z = a[2] ; z <= b[2] ; z++ ) for ( y = a[1] ; y <= b[1] ; y++ )
    for ( x = a[0] ; x <= b[0] ; x++ )
    { long cc = CELL(g,x,y,z);
      for ( k = g->start[cc] ; k < g->start[cc+1] ; k++ )
      { element_id o = get_ordinal_id(other,g->items[k]);
        if ( !mark(c,o) ) continue;
        n += other == FACET ? pair_crosses(c,id,o) : pair_crosses(c,o,id);
      }
    }
  for ( i = 0 ; i < extra->n ; i++ )
    if ( mark(c,positive_id(extra->a[i])) )
      n += other == FACET ? pair_crosses(c,id,positive_id(extra->a[i]))
                          : pair_crosses(c,positive_id(extra->a[i]),id);
  for ( i = 0 ; i < near->n ; i++ )
    if ( mark(c,near->a[i]) )
      n += other == FACET ? pair_crosses(c,id,near->a[i]) : pair_crosses(c,near->a[i],id);
  return n;
}

/* the patch round verts: its edges and facets (out, when wanted) and its
   problems */
static void patch(struct ctx *c, struct ids *verts, struct counts *k, struct ids *out_e,
                  struct ids *out_f)
{ struct ids E = {0,0,0}, F = {0,0,0};
  int i, w, guard;
  edge_id e, e0;
  memset(k,0,sizeof(*k));
  c->stamp_now++;
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
  for ( i = 0 ; i < E.n ; i++ )
  { facetedge_id fe0;
    facet_id f[3];
    int n = 0;
    vertex_id t, hh;
    e = E.a[i];
    t = get_edge_tailv(e); hh = get_edge_headv(e);
    fe0 = get_edge_fe(e);
    if ( !valid_id(fe0) ) { k->structure++; continue; }
    { facetedge_id fe = fe0;
      do { if ( n < 3 ) f[n] = get_fe_facet(fe); n++; fe = get_next_facet(fe); }
      while ( valid_id(fe) && !equal_id(fe,fe0) && n < 1000 );
    }
    if ( n > 2 ) { if ( c->manifold ) k->structure++; continue; }
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
      continue;
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
  for ( w = 0 ; w < c->nwalls ; w++ )
  { int wc = c->walls[w];
    double side = c->sides[w] < 0 ? -1.0 : 1.0;
    for ( i = 0 ; i < verts->n ; i++ )
    { vertex_id v = verts->a[i];
      if ( !valid_element(v) || v_on_constraint(v,wc) ) continue;
      if ( -side*pyse_wall_gap(wc,get_coord(v)) > c->tol*c->h ) k->solid++;
    }
    for ( i = 0 ; i < E.n ; i++ )
    { vertex_id t = get_edge_tailv(E.a[i]), hh = get_edge_headv(E.a[i]);
      REAL *p = get_coord(t), *q = get_coord(hh), mid[3];
      int d;
      if ( v_on_constraint(t,wc) && v_on_constraint(hh,wc) ) continue;
      for ( d = 0 ; d < 3 ; d++ ) mid[d] = (p[d] + q[d])/2;
      if ( -side*pyse_wall_gap(wc,mid) > c->tol*c->h ) k->solid++;
    }
  }

  /* crossings: the patch's edges against facets, its facets against edges */
  for ( i = 0 ; i < E.n ; i++ ) k->crossing += crossings_of(c,E.a[i],&F);
  for ( i = 0 ; i < F.n ; i++ ) k->crossing += crossings_of(c,F.a[i],&E);

  if ( out_e ) for ( i = 0 ; i < E.n ; i++ ) push(out_e,E.a[i]);
  if ( out_f ) for ( i = 0 ; i < F.n ; i++ ) push(out_f,F.a[i]);
  if ( E.a ) temp_free((char *)E.a);
  if ( F.a ) temp_free((char *)F.a);
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
  patch(c,verts,&now,&E,&F);
  why = worse(was,&now,pinch_ok);
  if ( !why ) changed(c,&F,&E);
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
   turn, smallest doubled area, longest new edge, and the volume swept */
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
      { REAL l = dist3(p,get_coord(o));
        if ( l > *longest ) *longest = l;
      }
      if ( valid_id(fe0) ) do
      { facet_id f = positive_id(get_fe_facet(fe));
        if ( valid_id(f) && mark(c,f) )
        { vertex_id v[3];
          REAL *x[3], *y[3], n0[3], n1[3], a0, a1;
          int hits = 0;
          facet_verts(f,v);
          for ( k = 0 ; k < 3 ; k++ )
          { x[k] = y[k] = get_coord(v[k]);
            if ( equal_id(v[k],u) || equal_id(v[k],w) ) { y[k] = p; hits++; }
          }
          *volume += pyse_det3(p,x[0],x[1],x[2])/6;
          if ( hits == 1 )
          { tri_normal(x[0],x[1],x[2],n0);
            tri_normal(y[0],y[1],y[2],n1);
            a0 = norm3(n0); a1 = norm3(n1);
            if ( a1 < *least ) *least = a1;
            if ( a0 > 0 && a1 > 0 && dot3(n0,n1)/(a0*a1) < *worst_cos )
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
  if ( compare_vertex_attr(u,w) == A_EQ_B && !(get_vattr(u) & FIXED) )
    for ( k = 0 ; k < 3 ; k++ ) p[k] = (get_coord(u)[k] + get_coord(w)[k])/2;
  else
    for ( k = 0 ; k < 3 ; k++ ) p[k] = get_coord(u)[k];
  moved_shape(c,u,w,p,&cosw,&least,&longest,&volume);
  if ( least < 2e-10*c->h*c->h && !relaxed ) return PYSE_EDIT_DEGENERATE;
  if ( cosw < c->cos_turn && !relaxed ) return PYSE_EDIT_TURN;
  if ( c->max_len > 0 && longest > c->max_len*c->h ) return PYSE_EDIT_LONG;
  if ( fabs(volume) > c->max_volume*c->h*c->h*c->h ) return PYSE_EDIT_VOLUME;

  star(u,&verts); star(w,&verts);
  patch(c,&verts,&was,NULL,NULL);
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
  REAL n1[3], n2[3], m1[3], m2[3], avg[3], an, l1, l2;
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
  if ( mode == 1 )
  { if ( angle_at(get_coord(cc),get_coord(p),get_coord(q))
         + angle_at(get_coord(d),get_coord(p),get_coord(q)) <= M_PI + 1e-9 )
      return PYSE_EDIT_NOT_BETTER;
  }
  else if ( mode == 2 )
  { int before = abs(vertex_valence(p) - target_valence(p)) + abs(vertex_valence(q) - target_valence(q))
               + abs(vertex_valence(cc) - target_valence(cc)) + abs(vertex_valence(d) - target_valence(d));
    int later = abs(vertex_valence(p) - 1 - target_valence(p)) + abs(vertex_valence(q) - 1 - target_valence(q))
              + abs(vertex_valence(cc) + 1 - target_valence(cc)) + abs(vertex_valence(d) + 1 - target_valence(d));
    if ( later >= before ) return PYSE_EDIT_NOT_BETTER;
  }
  tri_normal(get_coord(p),get_coord(q),get_coord(cc),n1);
  tri_normal(get_coord(q),get_coord(p),get_coord(d),n2);
  tri_normal(get_coord(p),get_coord(d),get_coord(cc),m1);
  tri_normal(get_coord(q),get_coord(cc),get_coord(d),m2);
  l1 = norm3(n1); l2 = norm3(n2);
  if ( l1 <= 0 || l2 <= 0 ) return PYSE_EDIT_DEGENERATE;
  for ( k = 0 ; k < 3 ; k++ ) avg[k] = n1[k]/l1 + n2[k]/l2;
  an = norm3(avg);
  if ( norm3(m1) < 2e-10*c->h*c->h || norm3(m2) < 2e-10*c->h*c->h ) return PYSE_EDIT_DEGENERATE;
  if ( an <= 0 || dot3(m1,avg) < c->cos_turn*norm3(m1)*an || dot3(m2,avg) < c->cos_turn*norm3(m2)*an )
    return PYSE_EDIT_TURN;
  if ( fabs(pyse_det3(get_coord(p),get_coord(q),get_coord(cc),get_coord(d)))/6
       > c->max_volume*c->h*c->h*c->h )
    return PYSE_EDIT_VOLUME;

  push(&verts,p); push(&verts,q); push(&verts,cc); push(&verts,d);
  patch(c,&verts,&was,NULL,NULL);
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
  star(a,&verts); star(b,&verts);
  patch(c,&verts,&was,NULL,NULL);
  if ( !valid_id(edge_refine(e)) )
  { if ( verts.a ) temp_free((char *)verts.a);
    return PYSE_EDIT_EVOLVER;
  }
  m = get_edge_headv(e);                 /* e is the tail half now */
  project(m);
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
  int guard, k, cnt = 0, border = 0, why;
  vertex_id line[2];

  if ( get_vattr(v) & (FIXED|BOUNDARY) ) return PYSE_EDIT_FIXED;
  if ( constraint_count(v) > 1 ) return PYSE_EDIT_CONSTRAINTS;
  for ( k = 0 ; k < 3 ; k++ ) { target[k] = 0; n[k] = 0; old[k] = x[k]; }
  EACH_EDGE_AT(v,e,e0,guard)
  { int val = valence(e);
    REAL *y = get_coord(get_edge_headv(e));
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
    REAL *y0 = get_coord(line[0]), *y1 = get_coord(line[1]), t[3], s;
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
        { REAL *y[3];
          facet_verts(f,fv);
          for ( k = 0 ; k < 3 ; k++ ) y[k] = equal_id(fv[k],v) ? p : get_coord(fv[k]);
          tri_normal(y[0],y[1],y[2],fn);
          if ( norm3(fn) < 2e-10*c->h*c->h ) return PYSE_EDIT_DEGENERATE;
          if ( vlen > 0 && dot3(fn,vn) <= 0 ) return PYSE_EDIT_TURN;
        }
        fe = get_next_facet(fe);
      } while ( valid_id(fe) && !equal_id(fe,fe0) && ++m < 1000 );
    }
  }
  star(v,&verts);
  patch(c,&verts,&was,NULL,NULL);
  for ( k = 0 ; k < 3 ; k++ ) x[k] += d[k];
  project(v);
  why = after(c,&verts,&was,0);
  if ( why ) for ( k = 0 ; k < 3 ; k++ ) x[k] = old[k];    /* (a move undoes exactly) */
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
  for ( k = 0 ; k < 3 ; k++ )
    p[k] = (r == A_EQ_B && !(get_vattr(keep) & FIXED))
           ? (get_coord(a)[k] + get_coord(b)[k])/2 : get_coord(keep)[k];
  star(a,&verts); star(b,&verts);
  patch(c,&verts,&was,NULL,NULL);
  merge_vertex(keep,gone);
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
  patch(c,&verts,&was,NULL,NULL);
  for ( b = 0 ; b < nroots ; b++ )
  { vertex_id u = v;
    REAL centre[3] = {0,0,0}, dir[3], len;
    int cnt = 0;
    if ( b > 0 ) u = new_vertex(x,v);
    for ( k = 0 ; k < ne ; k++ )
      if ( efan[k] == b )
      { REAL *y = get_coord(get_edge_headv(es[k]));
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
  why = after(c,&verts,&was,0);
  if ( !why )
  { /* the pinch is gone */
    struct counts now;
    patch(c,&verts,&now,NULL,NULL);
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
          patch(c,&verts,&was,NULL,NULL);
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
  { len[k] = dist3(get_coord(get_edge_tailv(es[k])),get_coord(get_edge_headv(es[k])));
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

/* an oriented edge from a to b: an existing one, or a new one */
static edge_id edge_between(vertex_id a, vertex_id b)
{ edge_id e, e0;
  int guard;
  EACH_EDGE_AT(a,e,e0,guard)
    if ( equal_id(get_edge_headv(e),b) ) return e;
  return new_edge(a,b,NULLID);
}

/* the side of facet f (as its loop goes) toward point p: 1 front, 0 back */
static int faces(facet_id f, const REAL *p)
{ vertex_id v[3];
  REAL n[3], d[3];
  int k;
  facet_verts(f,v);
  tri_normal(get_coord(v[0]),get_coord(v[1]),get_coord(v[2]),n);
  for ( k = 0 ; k < 3 ; k++ )
    d[k] = p[k] - (get_coord(v[0])[k] + get_coord(v[1])[k] + get_coord(v[2])[k])/3;
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
  for ( i = 0 ; i < 3 ; i++ ) for ( k = 0 ; k < 3 ; k++ )
  { ca[k] += get_coord(a[i])[k]/3; cb[k] += get_coord(b[i])[k]/3; }
  side1 = faces(f1,cb); side2 = faces(f2,ca);
  if ( side1 != side2 ) return PYSE_EDIT_ORIENTATION;
  front = get_facet_body(f1); back = get_facet_body(inverse_id(f1));
  front2 = get_facet_body(f2); back2 = get_facet_body(inverse_id(f2));
  /* across the gap (where the normals point: out of the front body), the
     same body or none */
  if ( !equal_id(side1 ? back : front, side2 ? back2 : front2) ) return PYSE_EDIT_ORIENTATION;
  /* a[i] with b[(k - i) % 3]: the tube's facets go round both edges as the
     facets did */
  for ( k = 0 ; k < 3 ; k++ )
  { REAL d = 0;
    for ( i = 0 ; i < 3 ; i++ ) d += dist3(get_coord(a[i]),get_coord(b[((k - i) % 3 + 3) % 3]));
    if ( d < bestd ) { bestd = d; best = k; }
  }
  for ( i = 0 ; i < 3 ; i++ ) { star(a[i],&verts); star(b[i],&verts); }
  patch(c,&verts,&was,NULL,NULL);
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
  why = after(c,&verts,&was,0);
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
    if ( dist3(get_coord(a1),get_coord(b2)) + dist3(get_coord(a2),get_coord(b1))
         >= dist3(get_coord(a1),get_coord(b1)) + dist3(get_coord(a2),get_coord(b2)) )
      return PYSE_EDIT_ORIENTATION;
    if ( joined(a1,b2) || joined(a2,b1) ) return PYSE_EDIT_LINK;
  }
  star(a1,&verts); star(a2,&verts); star(b1,&verts); star(b2,&verts);
  patch(c,&verts,&was,NULL,NULL);
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
    if ( dist3(get_coord(a2),get_coord(b2)) <= dist3(get_coord(a1),get_coord(b1)) )
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
};

static int op_type(int op)
{ switch ( op )
  { case PYSE_OP_COLLAPSE: case PYSE_OP_FLIP: case PYSE_OP_SPLIT: case PYSE_OP_BRIDGE:
      return EDGE;
    case PYSE_OP_DEGENERATE: case PYSE_OP_MERGE: return FACET;
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

static int one(struct ctx *c, struct edit_args *a, int k)
{ element_id id, id2;
  int op = a->op;
  if ( paired(op) )
  { id = item(op,a->ids[2*k]);
    id2 = item(op,a->ids[2*k+1]);
    if ( !valid_id(id) || !valid_id(id2) || key(id) != a->keys[2*k]
         || key(id2) != a->keys[2*k+1] ) return PYSE_EDIT_GONE;
    switch ( op )
    { case PYSE_OP_SNAP: return snap(c,id,id2);
      case PYSE_OP_MERGE: return merge_facets(c,id,id2);
      case PYSE_OP_BRIDGE: return bridge(c,id,id2);
    }
  }
  id = item(op,a->ids[k]);
  if ( !valid_id(id) || key(id) != a->keys[k] ) return PYSE_EDIT_GONE;
  switch ( op )
  { case PYSE_OP_COLLAPSE: return collapse(c,id,0);
    case PYSE_OP_FLIP: return flip(c,id,c->mode);
    case PYSE_OP_SPLIT: return split(c,id);
    case PYSE_OP_SMOOTH: return smooth(c,id);
    case PYSE_OP_SEPARATE: return separate(c,id);
    case PYSE_OP_DEGENERATE: return degenerate(c,id);
  }
  return PYSE_EDIT_NOT_APPLICABLE;
}

/* one batch: each edit in turn; one that comes out worse puts the batch's
   snapshot back, the edits made before it made again */
static void run_batch(struct ctx *c, struct edit_args *a)
{ struct snapshot *s = NULL;
  int k, j, count = paired(a->op) ? a->n/2 : a->n;
  a->keys = (long *)temp_calloc(a->n ? a->n : 1,sizeof(long));
  for ( k = 0 ; k < a->n ; k++ ) a->keys[k] = key(item(a->op,a->ids[k]));
  grids_build(c);
  for ( k = 0 ; k < count ; k++ )
  { int r;
    if ( !s && a->op != PYSE_OP_SMOOTH ) s = pyse_snapshot_hold();
    r = one(c,a,k);
    if ( r >= 0 ) { a->results[k] = r; continue; }
    a->results[k] = -r;
    for ( ;; )
    { int again = 0;
      pyse_snapshot_put_back(s);
      grids_build(c);
      for ( j = 0 ; j < k && !again ; j++ )
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
  if ( s ) pyse_snapshot_drop(s);
  temp_free((char *)a->keys);
  a->keys = NULL;
}

static void setup(struct ctx *c, struct edit_args *a)
{ if ( web.representation != SOAPFILM || web.modeltype != LINEAR || SDIM != 3
       || web.torus_flag || web.symmetry_flag )
    kb_error(PYSE_ERR_BAD_ARGUMENT,"The mesh edits are for linear soap films in 3D, not in "
             "torus or symmetry spaces.\n",RECOVERABLE);
  memset(c,0,sizeof(*c));
  c->walls = a->walls; c->sides = a->sides; c->nwalls = a->nwalls; c->manifold = a->manifold;
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

static void edit_body(void *arg)
{ struct edit_args *a = (struct edit_args *)arg;
  struct ctx c;
  setup(&c,a);
  run_batch(&c,a);
  web_timestamp = top_timestamp = ++global_timestamp;
  recalc();
}

/* ---- the remesher ------------------------------------------------------- */

static REAL edge_length(edge_id e)
{ return dist3(get_coord(get_edge_tailv(e)),get_coord(get_edge_headv(e))); }

static int by_length_up(const void *x, const void *y)
{ const REAL *p = (const REAL *)x, *q = (const REAL *)y;
  return p[0] < q[0] ? -1 : p[0] > q[0];
}

/* edges as element numbers: longer than above (longest first), or shorter
   than below (shortest first), or (both 0) those on two facets */
static long *edges_by(double above, double below, int *n)
{ edge_id e;
  long count = 0, k = 0;
  REAL *pairs;
  long *out;
  FOR_ALL_EDGES(e) count++;
  pairs = (REAL *)temp_calloc(2*(count ? count : 1),sizeof(REAL));
  FOR_ALL_EDGES(e)
  { REAL l = edge_length(e);
    int keep = above > 0 ? l > above : below > 0 ? l < below : valence(e) == 2;
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
  FOR_ALL_VERTICES(v) out[k++] = ordinal(v) + 1;
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
static void remesh_body(void *arg)
{ struct edit_args *a = (struct edit_args *)arg;
  struct ctx c;
  int sweep, sweeps = (int)a->params[9], n, pass;
  double h, keep_len, keep_mode;
  setup(&c,a);
  h = c.h;
  keep_len = c.max_len; keep_mode = c.mode;
  for ( pass = 0 ; pass < 4 ; pass++ ) a->results[pass] = 0;
  for ( sweep = 0 ; sweep < (sweeps > 0 ? sweeps : 3) ; sweep++ )
  { struct edit_args b = *a;
    int moved = 0;
    for ( pass = 0 ; pass < 4 ; pass++ )
    { long *ids;
      int *res;
      if ( pass == 0 ) { ids = edges_by(4.0/3*h,0,&n); b.op = PYSE_OP_SPLIT; }
      else if ( pass == 1 ) { ids = edges_by(0,0.8*h,&n); b.op = PYSE_OP_COLLAPSE; c.max_len = 4.0/3; }
      else if ( pass == 2 ) { ids = edges_by(0,0,&n); b.op = PYSE_OP_FLIP; c.mode = 2; }
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
  web_timestamp = top_timestamp = ++global_timestamp;
  recalc();
}

int pyse_edit(int op, const long *ids, int n, const double *params, const int *walls,
              const int *sides, int nwalls, int manifold, int *results)
{ struct edit_args *a;
  int status, k;
  if ( op < PYSE_OP_COLLAPSE || op > PYSE_OP_REMESH )
    return PYSE_ERROR;
  a = (struct edit_args *)calloc(1,sizeof(*a));
  if ( !a ) return PYSE_FATAL;
  a->op = op; a->ids = ids; a->n = n; a->params = params; a->walls = walls;
  a->sides = sides; a->nwalls = nwalls; a->manifold = manifold; a->results = results;
  if ( op == PYSE_OP_REMESH )
    status = pyse_surface_call(remesh_body,a);
  else
  { for ( k = 0 ; k < (paired(op) ? n/2 : n) ; k++ ) results[k] = PYSE_EDIT_NOT_APPLICABLE;
    status = pyse_surface_call(edit_body,a);
  }
  pyse_surface_changed();
  free(a);
  return status;
}

/* ---- crossings (for descent steps) -------------------------------------- */

struct crossing_args { long count; int far; };

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


static void crossing_body(void *arg)
{ struct crossing_args *a = (struct crossing_args *)arg;
  struct ctx c;
  edge_id e;
  struct ids none = {0,0,0};
  memset(&c,0,sizeof(c));
  if ( web.representation != SOAPFILM || SDIM != 3 || web.torus_flag || web.symmetry_flag )
    kb_error(PYSE_ERR_BAD_ARGUMENT,"Crossings are counted for soap films in 3D only.\n",
             RECOVERABLE);
  c.h = pyse_mean_edge();
  grids_build(&c);
  a->count = 0;
  far_only = a->far;
  FOR_ALL_EDGES(e) a->count += crossings_of(&c,e,&none);
  far_only = 0;
}

int pyse_crossings(int far, long *count)
{ struct crossing_args a;
  int status;
  a.count = 0;
  a.far = far;
  far_only = 0;
  status = pyse_surface_call(crossing_body,&a);
  far_only = 0;
  *count = a.count;
  return status;
}
