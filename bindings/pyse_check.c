/*
 * pyse_check.c -- invariants of a soap-film surface beyond Evolver's own
 * "check" (its lists and links): what pySE's changes of topology must keep.
 *
 *   - each edge on one facet (a border) or two (manifold= also refuses more);
 *   - two facets across an edge oriented alike (the same bodies on the same
 *     sides), which Evolver's cone analysis (pop) needs;
 *   - one fan of facets round each vertex (two: a pinch);
 *   - border edges on a constraint (or a boundary, or fixed), their ends on
 *     its constraints too; on a wall, stored the way their facet goes and not
 *     reversed (orientation 1), or the wall's integrals change sign;
 *   - no degenerate facet, no fold (facets across an edge turned back onto
 *     each other);
 *   - no vertex inside a solid wall, no edge through one (its midpoint
 *     inside, unless both its ends are on that wall: a chord of the contact
 *     line);
 *   - no edge through a facet (a spatial hash, edges against the facets
 *     near them that share no vertex with them).
 *
 * Linear soap films only (the topology runs' surfaces): other models and
 * symmetric spaces aren't checked (an error says so). Problems come back as
 * records: what, the element (its number, as Evolver prints it), another
 * element or constraint, and a value (depth, cosine, ...).
 */

#include "include.h"
#include "pyse_api.h"
#include "pyse_internal.h"

struct check_args
{ const int *walls;          /* constraint numbers of solid walls */
  const int *sides;          /* +1: the solid where the formula is negative; -1: positive */
  int nwalls;
  double tol;                /* in mean edge lengths */
  int manifold;
  struct pyse_problem *out;
  int max;
  int count;                 /* all found, may exceed max */
};

static void add(struct check_args *a, int code, int type, element_id id, long other, double value)
{ if ( a->count < a->max )
  { struct pyse_problem *p = a->out + a->count;
    p->code = code;
    p->type = type;
    p->id = valid_id(id) ? (long)ordinal(id) + 1 : 0;
    p->other = other;
    p->value = value;
  }
  a->count++;
}

/* union-find over facet corners (3 per facet ordinal) */
static long find_root(long *parent, long i)
{ while ( parent[i] != i )
  { parent[i] = parent[parent[i]];
    i = parent[i];
  }
  return i;
}

static void unite(long *parent, long i, long j)
{ i = find_root(parent,i);
  j = find_root(parent,j);
  if ( i != j ) parent[i < j ? j : i] = i < j ? i : j;
}

int pyse_corner(facet_id f_id, vertex_id v_id)
{ facetedge_id fe = get_facet_fe(positive_id(f_id));   /* (corners as FOR_ALL_FACETS numbers them) */
  int k;
  for ( k = 0 ; k < FACET_VERTS ; k++, fe = get_next_edge(fe) )
    if ( equal_id(get_fe_tailv(fe),v_id) ) return k;
  return -1;
}

/* ---- slabs periodic in x and y (pyse_slab) -------------------------------
   Coordinates aren't wrapped into the cell, every edge is short: a nearby
   point is taken at its image nearest a reference point. */

/* x's image nearest ref (x itself outside a slab: whole periods off x) */
void pyse_image(const REAL *ref, const REAL *x, REAL *out)
{ REAL L[2];
  int i;
  for ( i = 0 ; i < 3 ; i++ ) out[i] = x[i];
  if ( web.torus_flag && pyse_slab(L) )
    for ( i = 0 ; i < 2 ; i++ ) out[i] -= L[i]*floor((x[i] - ref[i])/L[i] + 0.5);
}

/* Wall c's gap (pyse_wall_gap) at x's image deepest into its solid (side
   +1: the solid where its formula is negative), of the 3 x 3 round x's
   image nearest ref (a vertex on the wall, or NULL: x), into at (may be
   NULL). A wall's vertices sit at the image near its centre (a sphere's),
   the others anywhere: so a solid ball less than half a period across is
   found whichever image a point is at, its radius under half a period; a
   floor is the same at all. Outside
   a slab, the gap at x. */
double pyse_slab_gap(int c, double side, const REAL *ref, const REAL *x, REAL *at)
{ REAL L[2], y[MAXCOORD], z[MAXCOORD], best[MAXCOORD];
  double g, most;
  int i, j, k;
  for ( k = 0 ; k < SDIM ; k++ ) best[k] = z[k] = y[k] = x[k];
  most = pyse_wall_gap(c,z);
  if ( web.torus_flag && pyse_slab(L) )
  { if ( ref ) pyse_image(ref,x,y);
    for ( i = -1 ; i <= 1 ; i++ ) for ( j = -1 ; j <= 1 ; j++ )
    { for ( k = 0 ; k < SDIM ; k++ ) z[k] = y[k];
      z[0] += i*L[0]; z[1] += j*L[1];
      if ( z[0] == x[0] && z[1] == x[1] ) continue;      /* (x's own: done) */
      g = pyse_wall_gap(c,z);
      if ( -side*g > -side*most || most != most )
      { most = g;
        for ( k = 0 ; k < SDIM ; k++ ) best[k] = z[k];
      }
    }
  }
  if ( at ) for ( k = 0 ; k < SDIM ; k++ ) at[k] = best[k];
  return most;
}

void pyse_facet_normal(facet_id f_id, REAL *n, REAL *area)
{ facetedge_id fe = get_facet_fe(f_id);
  REAL *x0 = get_coord(get_fe_tailv(fe));
  REAL *x1 = get_coord(get_fe_headv(fe));
  REAL *x2 = get_coord(get_fe_headv(get_next_edge(fe)));
  REAL a[3], b[3];
  int i;
  for ( i = 0 ; i < 3 ; i++ ) { a[i] = x1[i] - x0[i]; b[i] = x2[i] - x0[i]; }
  if ( web.torus_flag && pyse_slab(NULL) ) { pyse_delta(x0,x1,a); pyse_delta(x0,x2,b); }
  n[0] = a[1]*b[2] - a[2]*b[1];
  n[1] = a[2]*b[0] - a[0]*b[2];
  n[2] = a[0]*b[1] - a[1]*b[0];
  *area = 0.5*sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
}

double pyse_wall_gap(int c, REAL *x)
{ REAL f = 0.0, grad[MAXCOORD], g = 0.0;
  int i;
  eval_all(get_constraint(c)->formula,x,SDIM,&f,grad,NULLID);
  for ( i = 0 ; i < SDIM ; i++ ) g += grad[i]*grad[i];
  return g > 0.0 ? (double)(f/sqrt(g)) : 0.0;
}

REAL pyse_det3(const REAL *a, const REAL *b, const REAL *c, const REAL *d)
{ REAL u[3], v[3], w[3];
  int i;
  for ( i = 0 ; i < 3 ; i++ ) { u[i] = b[i] - a[i]; v[i] = c[i] - a[i]; w[i] = d[i] - a[i]; }
  return u[0]*(v[1]*w[2] - v[2]*w[1]) - u[1]*(v[0]*w[2] - v[2]*w[0])
         + u[2]*(v[0]*w[1] - v[1]*w[0]);
}

/* Whether segment pq goes through triangle abc: p and q on either side of
   its plane, each beyond eps (an edge lying in the plane is left out), and
   the line through the triangle or its edges and corners (within eps: the
   two triangles beside an edge it crosses at both see it on their border). */
int pyse_crosses(const REAL *p, const REAL *q, const REAL *a, const REAL *b, const REAL *c,
                 REAL eps)
{ REAL sp = pyse_det3(a,b,c,p), sq = pyse_det3(a,b,c,q), s1, s2, s3;
  if ( !((sp > eps && sq < -eps) || (sp < -eps && sq > eps)) ) return 0;
  s1 = pyse_det3(p,q,a,b);
  s2 = pyse_det3(p,q,b,c);
  s3 = pyse_det3(p,q,c,a);
  return (s1 > -eps && s2 > -eps && s3 > -eps) || (s1 < eps && s2 < eps && s3 < eps);
}

double pyse_mean_edge(void)
{ edge_id e_id;
  double h = 0.0;
  long n = 0;
  int slab = web.torus_flag && pyse_slab(NULL);
  FOR_ALL_EDGES(e_id)
  { REAL *p = get_coord(get_edge_tailv(e_id)), *q = get_coord(get_edge_headv(e_id)), d[3];
    if ( slab ) { pyse_delta(p,q,d); h += sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]); }
    else h += sqrt((p[0]-q[0])*(p[0]-q[0]) + (p[1]-q[1])*(p[1]-q[1]) + (p[2]-q[2])*(p[2]-q[2]));
    n++;
  }
  return n ? h/n : 1.0;
}

/* (in a slab, x and y periodic: cells of a side dividing the period,
   their numbers taken round it) */
struct grid { REAL lo[3], cell, pcell[2]; int n[3], per; long *start, *items; };

static long cell_of(struct grid *g, int i, int j, int k)
{ if ( g->per )
  { i %= g->n[0]; if ( i < 0 ) i += g->n[0];
    j %= g->n[1]; if ( j < 0 ) j += g->n[1];
  }
  return ((long)k*g->n[1] + j)*g->n[0] + i;
}

static void cell_range(struct grid *g, const REAL *lo, const REAL *hi, int *a, int *b)
{ int d;
  for ( d = 0 ; d < 3 ; d++ )
  { if ( g->per && d < 2 )
    { a[d] = (int)floor(lo[d]/g->pcell[d]);
      b[d] = (int)floor(hi[d]/g->pcell[d]);
      if ( b[d] - a[d] > g->n[d] - 1 ) b[d] = a[d] + g->n[d] - 1;
      continue;
    }
    a[d] = (int)floor((lo[d] - g->lo[d])/g->cell);
    b[d] = (int)floor((hi[d] - g->lo[d])/g->cell);
    if ( a[d] < 0 ) a[d] = 0;
    if ( b[d] > g->n[d] - 1 ) b[d] = g->n[d] - 1;
  }
}

/* facet f's corners, at one image (in a slab: the first's, or near ref) */
static void facet_corners(facet_id f_id, const REAL *ref, REAL x[3][3])
{ facetedge_id fe = get_facet_fe(f_id);
  int k, d;
  for ( k = 0 ; k < FACET_VERTS ; k++, fe = get_next_edge(fe) )
  { REAL *y = get_coord(get_fe_tailv(fe));
    if ( k == 0 && ref ) pyse_image(ref,y,x[0]);
    else if ( k == 0 ) for ( d = 0 ; d < 3 ; d++ ) x[0][d] = y[d];
    else pyse_image(x[0],y,x[k]);
  }
}

static void facet_box(facet_id f_id, REAL *lo, REAL *hi)
{ REAL x[3][3];
  int k, d;
  facet_corners(f_id,NULL,x);
  for ( k = 0 ; k < FACET_VERTS ; k++ )
    for ( d = 0 ; d < 3 ; d++ )
    { if ( k == 0 || x[k][d] < lo[d] ) lo[d] = x[k][d];
      if ( k == 0 || x[k][d] > hi[d] ) hi[d] = x[k][d];
    }
}

static void intersections(struct check_args *a, REAL h)
{ struct grid g;
  facet_id f_id;
  edge_id e_id;
  long nf = web.skel[FACET].max_ord + 1, total = 0, ncells, c;
  long *stamp;
  REAL lo[3], hi[3], top[3], L[2];
  vertex_id v_id;
  int first = 1, d, i, j, k, ia[3], ib[3];

  FOR_ALL_VERTICES(v_id)
  { REAL *x = get_coord(v_id);
    for ( d = 0 ; d < 3 ; d++ )
    { if ( first || x[d] < g.lo[d] ) g.lo[d] = x[d];
      if ( first || x[d] > top[d] ) top[d] = x[d];
    }
    first = 0;
  }
  if ( first ) return;
  g.cell = 2*h;
  for ( d = 0 ; d < 3 ; d++ )
  { double cells = floor((top[d] - g.lo[d])/g.cell) + 1;
    g.n[d] = cells > 256 ? 256 : (int)cells;
  }
  for ( d = 0 ; d < 3 ; d++ )
    if ( (top[d] - g.lo[d])/g.n[d] > g.cell ) g.cell = (top[d] - g.lo[d])/g.n[d]*1.0000001;
  g.per = pyse_slab(L);
  if ( g.per )
    for ( d = 0 ; d < 2 ; d++ )
    { double cells = floor(L[d]/(2*h));
      g.n[d] = cells > 256 ? 256 : cells < 1 ? 1 : (int)cells;
      g.pcell[d] = L[d]/g.n[d];
    }
  ncells = (long)g.n[0]*g.n[1]*g.n[2];
  g.start = (long *)temp_calloc(ncells + 1,sizeof(long));
  /* counting pass, then filling */
  FOR_ALL_FACETS(f_id)
  { facet_box(f_id,lo,hi);
    cell_range(&g,lo,hi,ia,ib);
    for ( k = ia[2] ; k <= ib[2] ; k++ )
      for ( j = ia[1] ; j <= ib[1] ; j++ )
        for ( i = ia[0] ; i <= ib[0] ; i++ )
        { g.start[cell_of(&g,i,j,k) + 1]++; total++; }
  }
  for ( c = 0 ; c < ncells ; c++ ) g.start[c+1] += g.start[c];
  g.items = (long *)temp_calloc(total ? total : 1,sizeof(long));
  { long *fill = (long *)temp_calloc(ncells,sizeof(long));
    FOR_ALL_FACETS(f_id)
    { facet_box(f_id,lo,hi);
      cell_range(&g,lo,hi,ia,ib);
      for ( k = ia[2] ; k <= ib[2] ; k++ )
        for ( j = ia[1] ; j <= ib[1] ; j++ )
          for ( i = ia[0] ; i <= ib[0] ; i++ )
          { c = cell_of(&g,i,j,k);
            g.items[g.start[c] + fill[c]++] = ordinal(f_id);
          }
    }
    temp_free((char *)fill);
  }
  stamp = (long *)temp_calloc(nf,sizeof(long));
  for ( c = 0 ; c < nf ; c++ ) stamp[c] = -1;
  FOR_ALL_EDGES(e_id)
  { vertex_id t = get_edge_tailv(e_id), hv = get_edge_headv(e_id);
    REAL *p = get_coord(t), *q = get_coord(hv), qq[3];
    long eo = ordinal(e_id);
    if ( g.per ) { pyse_image(p,q,qq); q = qq; }
    for ( d = 0 ; d < 3 ; d++ )
    { lo[d] = p[d] < q[d] ? p[d] : q[d];
      hi[d] = p[d] < q[d] ? q[d] : p[d];
    }
    cell_range(&g,lo,hi,ia,ib);
    for ( k = ia[2] ; k <= ib[2] ; k++ )
      for ( j = ia[1] ; j <= ib[1] ; j++ )
        for ( i = ia[0] ; i <= ib[0] ; i++ )
        { c = cell_of(&g,i,j,k);
          for ( total = g.start[c] ; total < g.start[c+1] ; total++ )
          { long fo = g.items[total];
            facet_id ff = get_ordinal_id(FACET,fo);
            facetedge_id fe;
            REAL *x[3], y[3][3];
            int m, shared = 0;
            if ( stamp[fo] == eo ) continue;
            stamp[fo] = eo;
            fe = get_facet_fe(ff);
            for ( m = 0 ; m < 3 ; m++, fe = get_next_edge(fe) )
            { vertex_id w = get_fe_tailv(fe);
              if ( equal_id(w,t) || equal_id(w,hv) ) shared = 1;
              x[m] = get_coord(w);
            }
            if ( shared ) continue;
            if ( g.per )
            { facet_corners(ff,p,y);
              for ( m = 0 ; m < 3 ; m++ ) x[m] = y[m];
            }
            if ( pyse_crosses(p,q,x[0],x[1],x[2],1e-12*h*h*h) )
              add(a,PYSE_PROBLEM_INTERSECTION,PYSE_EDGE,e_id,fo + 1,0.0);
          }
        }
  }
  temp_free((char *)stamp);
  temp_free((char *)g.items);
  temp_free((char *)g.start);
}

static void check_body(void *arg)
{ struct check_args *a = (struct check_args *)arg;
  long nf = web.skel[FACET].max_ord + 1, nv = web.skel[VERTEX].max_ord + 1, i;
  long *parent, *root_of, *fans;
  unsigned char *nonmanifold;
  edge_id e_id;
  facet_id f_id;
  vertex_id v_id;
  REAL h;
  int w, slab = pyse_slab(NULL);

  if ( web.representation != SOAPFILM || web.modeltype != LINEAR || SDIM != 3 )
    kb_error(PYSE_ERR_BAD_ARGUMENT,"The topology check is for linear soap films in 3D.\n",
             RECOVERABLE);
  if ( (web.torus_flag || web.symmetry_flag) && !pyse_slab(NULL) )
    kb_error(PYSE_ERR_BAD_ARGUMENT,"The topology check doesn't do torus or symmetry "
             "spaces (but slabs periodic in x and y).\n",RECOVERABLE);
  for ( w = 0 ; w < a->nwalls ; w++ )
    if ( a->walls[w] < 0 || a->walls[w] >= web.maxcon
         || !(get_constraint(a->walls[w])->attr & IN_USE) )
      kb_error(PYSE_ERR_BAD_ARGUMENT,"No such constraint for the topology check.\n",
               RECOVERABLE);

  h = pyse_mean_edge();

  parent = (long *)temp_calloc(3*nf,sizeof(long));
  for ( i = 0 ; i < 3*nf ; i++ ) parent[i] = i;
  nonmanifold = (unsigned char *)temp_calloc(nv,1);

  /* edges: valence, orientation across, folds, borders */
  FOR_ALL_EDGES(e_id)
  { facetedge_id start = get_edge_fe(e_id), fe;
    facet_id f[3];
    int n = 0;
    vertex_id t = get_edge_tailv(e_id), hv = get_edge_headv(e_id);
    if ( !valid_id(start) )
    { add(a,PYSE_PROBLEM_BARE_EDGE,PYSE_EDGE,e_id,0,0.0);
      continue;
    }
    fe = start;
    do
    { if ( n < 3 ) f[n] = get_fe_facet(fe);
      n++;
      fe = get_next_facet(fe);
    } while ( valid_id(fe) && !equal_id(fe,start) && n < 1000 );
    if ( n > 2 )
    { nonmanifold[ordinal(t)] = nonmanifold[ordinal(hv)] = 1;
      if ( a->manifold ) add(a,PYSE_PROBLEM_MANY_FACETS,PYSE_EDGE,e_id,n,0.0);
      continue;
    }
    if ( n == 2 )
    { int s1 = inverted(f[0]), s2 = inverted(f[1]);
      body_id fr1 = get_facet_body(positive_id(f[0])), bk1 = get_facet_body(inverse_id(positive_id(f[0])));
      body_id fr2 = get_facet_body(positive_id(f[1])), bk2 = get_facet_body(inverse_id(positive_id(f[1])));
      int alike = equal_id(fr1,fr2) && equal_id(bk1,bk2);
      int swapped = equal_id(fr1,bk2) && equal_id(bk1,fr2);
      int bad = (alike && s1 == s2) || (!alike && swapped && s1 != s2);
      long c1 = 3*ordinal(f[0]), c2 = 3*ordinal(f[1]);
      int k1, k2;
      if ( bad )
        add(a,PYSE_PROBLEM_ORIENTATION,PYSE_EDGE,e_id,(long)ordinal(f[1]) + 1,0.0);
      else if ( alike || swapped )
      { /* a fold: the facets' normals, as the surface goes, nearly opposite */
        REAL n1[3], n2[3], a1, a2, dot;
        pyse_facet_normal(positive_id(f[0]),n1,&a1);
        pyse_facet_normal(positive_id(f[1]),n2,&a2);
        dot = (n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2])/(4*a1*a2 + 1e-300);
        if ( swapped && !alike ) dot = -dot;
        if ( a1 > 0 && a2 > 0 && dot < -0.95 )
          add(a,PYSE_PROBLEM_FOLD,PYSE_EDGE,e_id,(long)ordinal(f[1]) + 1,dot);
      }
      /* the two facets' corners at each end belong to one fan */
      k1 = pyse_corner(f[0],t); k2 = pyse_corner(f[1],t);
      if ( k1 >= 0 && k2 >= 0 ) unite(parent,c1 + k1,c2 + k2);
      k1 = pyse_corner(f[0],hv); k2 = pyse_corner(f[1],hv);
      if ( k1 >= 0 && k2 >= 0 ) unite(parent,c1 + k1,c2 + k2);
      continue;
    }
    /* a border edge */
    { ATTR attr = get_eattr(e_id);
      if ( !(attr & (CONSTRAINT|BOUNDARY|FIXED)) )
        add(a,PYSE_PROBLEM_OPEN_BORDER,PYSE_EDGE,e_id,0,0.0);
      if ( attr & CONSTRAINT )
      { conmap_t *map = get_e_constraint_map(e_id);
        unsigned int m;
        for ( m = 1 ; m <= map[0] ; m++ )
        { int c = (int)(map[m] & CONMASK);
          if ( !v_on_constraint(t,c) || !v_on_constraint(hv,c) )
            add(a,PYSE_PROBLEM_BORDER_VERTEX,PYSE_EDGE,e_id,c,0.0);
        }
      }
      for ( w = 0 ; w < a->nwalls ; w++ )
        if ( e_on_constraint(e_id,a->walls[w])
             && (inverted(f[0]) || (attr & NEGBOUNDARY)) )
          add(a,PYSE_PROBLEM_BORDER_REVERSED,PYSE_EDGE,e_id,a->walls[w],0.0);
    }
  }

  /* fans round each vertex */
  root_of = (long *)temp_calloc(nv,sizeof(long));
  fans = (long *)temp_calloc(nv,sizeof(long));
  for ( i = 0 ; i < nv ; i++ ) root_of[i] = -1;
  FOR_ALL_FACETS(f_id)
  { facetedge_id fe = get_facet_fe(f_id);
    int k;
    REAL n[3], area;
    pyse_facet_normal(f_id,n,&area);
    if ( area < 1e-10*h*h )
      add(a,PYSE_PROBLEM_DEGENERATE,PYSE_FACET,f_id,0,area/(h*h));
    for ( k = 0 ; k < FACET_VERTS ; k++, fe = get_next_edge(fe) )
    { long vo = ordinal(get_fe_tailv(fe));
      long r = find_root(parent,3*ordinal(f_id) + k);
      if ( root_of[vo] < 0 ) { root_of[vo] = r; fans[vo] = 1; }
      else if ( root_of[vo] != r )
      { /* a second fan, unless one already counted: mark roots by
           pointing them at the first (cheap; exact counts need more) */
        long r0 = find_root(parent,root_of[vo]);
        if ( r0 != r ) { unite(parent,r0,r); fans[vo]++; }
      }
    }
  }
  FOR_ALL_VERTICES(v_id)
  { long vo = ordinal(v_id);
    if ( fans[vo] > 1 && !nonmanifold[vo] )
      add(a,PYSE_PROBLEM_PINCHED,PYSE_VERTEX,v_id,fans[vo],0.0);
  }
  temp_free((char *)fans);
  temp_free((char *)root_of);

  /* solids (in a slab, each point at its image deepest in: see
     pyse_slab_gap) */
  for ( w = 0 ; w < a->nwalls ; w++ )
  { int c = a->walls[w];
    double side = a->sides[w] < 0 ? -1.0 : 1.0, depth;
    REAL *ref = NULL;
    if ( slab )
    { FOR_ALL_VERTICES(v_id)
        if ( v_on_constraint(v_id,c) ) { ref = get_coord(v_id); break; }
    }
    FOR_ALL_VERTICES(v_id)
    { if ( v_on_constraint(v_id,c) ) continue;
      depth = -side*(slab ? pyse_slab_gap(c,side,ref,get_coord(v_id),NULL)
                          : pyse_wall_gap(c,get_coord(v_id)));
      if ( depth > a->tol*h )
        add(a,PYSE_PROBLEM_VERTEX_IN_SOLID,PYSE_VERTEX,v_id,c,depth);
    }
    FOR_ALL_EDGES(e_id)
    { vertex_id t = get_edge_tailv(e_id), hv = get_edge_headv(e_id);
      REAL *p = get_coord(t), *q = get_coord(hv), mid[MAXCOORD], qq[3];
      int d;
      if ( v_on_constraint(t,c) && v_on_constraint(hv,c) ) continue;
      if ( slab ) { pyse_image(p,q,qq); q = qq; }
      for ( d = 0 ; d < SDIM ; d++ ) mid[d] = (p[d] + q[d])/2;
      depth = -side*(slab ? pyse_slab_gap(c,side,ref,mid,NULL) : pyse_wall_gap(c,mid));
      if ( depth > a->tol*h )
        add(a,PYSE_PROBLEM_EDGE_IN_SOLID,PYSE_EDGE,e_id,c,depth);
    }
  }

  intersections(a,h);
  temp_free((char *)nonmanifold);
  temp_free((char *)parent);
}

int pyse_check_topology(const int *walls, const int *sides, int nwalls, double tol,
                        int manifold, struct pyse_problem *out, int max, int *count)
{ struct check_args a;
  int status;
  a.walls = walls; a.sides = sides; a.nwalls = nwalls; a.tol = tol;
  a.manifold = manifold; a.out = out; a.max = max; a.count = 0;
  status = pyse_surface_call(check_body,&a);
  *count = a.count;
  return status;
}
