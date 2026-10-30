/*
 * pyse_contact.c -- spheres' wetted areas and volumes from their contact
 * lines alone (pySE's gauge="lines" in a periodic slab).
 *
 * The wetted region of a sphere (centre c, radius R) bounded by geodesic
 * polygons through the contact-line vertices has, by Gauss-Bonnet (Girard),
 * area R^2 (2 pi - sum of the turning angles) per loop, the loop taken with
 * the wetted region on its left (seen from outside the sphere). A region
 * bounded by several loops is the sum of theirs less whole spheres: with S
 * the sum over a body's loops on a sphere, its wetted area is S - 4 pi R^2 n,
 * n = floor(S / 4 pi R^2) (exact unless the region is empty or the whole
 * sphere, which a loop can't bound). A turning angle needs a vertex and its
 * two neighbours along the line: each contact edge carries half of the
 * angle at each of its ends, so the method is an edge method seeing four
 * vertices (the edge's, and the neighbour of each along its line).
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
  REAL tauA;                /* turning angle at A (this pass) */
};

struct pc_group { int s, b, first, count, loops, bad; REAL S, K; };  /* K: the loops' constant area */

static struct pc_entry *ent = NULL;
static int nent = 0, maxent = 0;
static struct pc_group *grp = NULL;
static int ngrp = 0, maxgrp = 0;
static int *slot = NULL;            /* edge ordinal -> first entry, -1 */
static int nslot = 0;
static int pc_bad = 0;
static int pc_built = 0;
static int pc_stale = 1;             /* the table to be built at the next init */
static long pc_stamp[4];            /* the elements and load it was built for */            /* built since the last pyse_contact_refresh() */
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

/* the turning angle at v from u to w (positions from the centre), left
   turns about v positive */
static REAL turning(const REAL *u, const REAL *v, const REAL *w)
{ REAL vv = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
  REAL uv = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
  REAL vw = v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
  REAL uw = u[0]*w[0] + u[1]*w[1] + u[2]*w[2];
  REAL det = u[0]*(v[1]*w[2] - v[2]*w[1]) - u[1]*(v[0]*w[2] - v[2]*w[0])
           + u[2]*(v[0]*w[1] - v[1]*w[0]);
  return atan2(sqrt(vv)*det,uv*vw - uw*vv);
}

static REAL angle_floor(void)
{ static long load = -2;
  static int k = -1;
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

/* whether point p is inside body b (ordinal): a vertical ray's crossings
   with its facets and the cones over its contact lines */
static int inside_body(int b, const REAL *p0)
{ facet_id f_id;
  REAL p[3];
  int count = 0, i, k;
  /* (off any vertex or edge's projection: the ray nudged) */
  p[0] = p0[0] + 1.2345678e-7; p[1] = p0[1] + 2.3456789e-7; p[2] = p0[2];
#define PC_CROSS(T) \
  { REAL d0, d1, d2, den, z; \
    REAL ax = T[0][0] - p[0], ay = T[0][1] - p[1], bx = T[1][0] - p[0], by = T[1][1] - p[1], \
         cx = T[2][0] - p[0], cy = T[2][1] - p[1]; \
    d0 = ax*by - ay*bx; d1 = bx*cy - by*cx; d2 = cx*ay - cy*ax; \
    if ( (d0 > 0 && d1 > 0 && d2 > 0) || (d0 < 0 && d1 < 0 && d2 < 0) ) \
    { den = d0 + d1 + d2; \
      z = (d1*T[0][2] + d2*T[1][2] + d0*T[2][2])/den; \
      if ( z > p[2] ) count++; \
    } \
  }
  FOR_ALL_FACETS(f_id)
  { body_id fb = get_facet_body(f_id), bb = get_facet_body(inverse_id(f_id));
    facetedge_id fe;
    REAL T[3][3], w[3];
    REAL *x0;
    if ( !((valid_id(fb) && ordinal(fb) == b) || (valid_id(bb) && ordinal(bb) == b)) ) continue;
    fe = get_facet_fe(f_id);
    x0 = get_coord(get_fe_tailv(fe));
    pc_delta(p,x0,w);
    for ( k = 0 ; k < 3 ; k++ ) T[0][k] = p[k] + w[k];
    for ( i = 1 ; i < 3 ; i++ )
    { fe = get_next_edge(fe);
      pc_delta(x0,get_coord(get_fe_tailv(fe)),w);
      for ( k = 0 ; k < 3 ; k++ ) T[i][k] = T[0][k] + w[k];
    }
    PC_CROSS(T)
  }
  for ( i = 0 ; i < nent ; i++ )
  { struct pc_entry *q = ent + i;
    REAL T[3][3], w[3];
    const REAL *c;
    if ( q->b != b ) continue;
    c = sph[q->s].c;
    pc_delta(p,c,w);
    for ( k = 0 ; k < 3 ; k++ ) T[0][k] = p[k] + w[k];
    pc_delta(c,get_coord(q->A),w);
    for ( k = 0 ; k < 3 ; k++ ) T[1][k] = T[0][k] + w[k];
    pc_delta(c,get_coord(q->B),w);
    for ( k = 0 ; k < 3 ; k++ ) T[2][k] = T[0][k] + w[k];
    PC_CROSS(T)
  }
#undef PC_CROSS
  return count & 1;
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

/* the table: contact entries, their groups and lines, the turning angles
   and the constants */
static void pc_build(void)
{ edge_id e_id;
  int i, j, g, nb;
  int *order = NULL, *by = NULL;
  unsigned long long key = 0;

  pc_slab = PC_SLAB(pc_L);
  pc_built = 1;
  nent = 0; ngrp = 0; pc_bad = 0;
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
      else ent[q->next].prev = order[i + a];
    }
    /* loops */
    if ( !G->bad )
    { for ( a = i ; a < j ; a++ ) ent[order[a]].group = -1 - ngrp;   /* unvisited */
      for ( a = i ; a < j ; a++ )
      { int k = order[a];
        if ( ent[k].group >= 0 ) continue;
        G->loops++;
        while ( ent[k].group < 0 ) { ent[k].group = ngrp; k = ent[k].next; }
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

  /* turning angles, each group's area, n and loops' constant */
  for ( i = 0 ; i < nent ; i++ )
  { struct pc_entry *q = ent + i;
    REAL u[3], v[3], w[3];
    const REAL *c = sph[q->s].c;
    if ( q->prev < 0 ) { q->tauA = 0.0; continue; }
    pc_delta(c,get_coord(ent[q->prev].A),u);
    pc_delta(c,get_coord(q->A),v);
    pc_delta(c,get_coord(q->B),w);
    q->tauA = turning(u,v,w);
  }
  for ( g = 0 ; g < ngrp ; g++ )
  { struct pc_group *G = grp + g;
    REAL R = sph[G->s].R, sum = 0.0, S, whole = 4*M_PI*R*R, n;
    for ( j = G->first ; j < G->first + G->count ; j++ ) sum += ent[order[j]].tauA;
    S = R*R*(2*M_PI*G->loops - sum);
    n = G->bad ? 0.0 : floor(S/whole);
    G->S = S - n*whole;
    G->K = G->bad ? 0.0 : R*R*(2*M_PI*G->loops - 4*M_PI*n);
  }

  /* beads with no line: inside a body or not (when the lines' topology changed) */
  if ( key != topo_key )
  { int s, b;
    char *has = (char*)calloc(nsph ? nsph : 1,1);
    for ( b = 0 ; b <= nb ; b++ ) imm_count[b] = 0;
    for ( s = 0 ; s < nsph ; s++ ) imm_body[s] = -1;
    for ( i = 0 ; i < nent ; i++ ) has[ent[i].s] = 1;
    body_id b_id;
    FOR_ALL_BODIES(b_id)
    { b = ordinal(b_id);
      if ( b >= nb ) continue;
      for ( s = 0 ; s < nsph ; s++ )
        if ( !has[s] && imm_body[s] < 0 && inside_body(b,sph[s].c) )
        { imm_body[s] = b; imm_count[b]++; }
    }
    free(has);
    topo_key = key;
  }
  free(order); free(by); free(sort_v); sort_v = NULL;
}

/* the table rebuilt at the first init of each pass (stale from the end of
   the last one, pyse_contact_refresh(), and whenever the elements or the
   load changed since it was built) */

static int pc_changed(void)
{ long now[4] = { (long)web.skel[VERTEX].count, (long)web.skel[EDGE].count,
                  (long)web.skel[EDGE].max_ord, PC_LOAD() };
  int k, changed = 0;
  for ( k = 0 ; k < 4 ; k++ ) if ( now[k] != pc_stamp[k] ) { changed = 1; pc_stamp[k] = now[k]; }
  return changed;
}

void pyse_contact_init(int mode, struct method_instance *mi)
{ (void)mode; (void)mi;
  if ( web.modeltype != LINEAR )
    kb_error(9901,"pyse_contact_integral: linear model only.\n",RECOVERABLE);
  if ( pc_changed() || pc_stale ) { pc_build(); pc_stale = 0; }
}

/* the edge's neighbours along its line: v[2] beside the tail, v[3] the head */
void pyse_contact_wings(struct qinfo *e_info)
{ int o, k;
  edge_id e = e_info->id;
  if ( !nent ) return;
  o = ordinal(e);
  if ( o >= nslot || (k = slot[o]) < 0 ) return;
  { struct pc_entry *q = ent + k;
    vertex_id tail = get_edge_tailv(e), head = get_edge_headv(e);
    vertex_id nt, nh;
    if ( q->prev < 0 || q->next < 0 ) return;
    /* (along the line P -> A -> B -> N) */
    if ( tail == q->A ) { nt = ent[q->prev].A; nh = ent[q->next].B; }
    else { nt = ent[q->next].B; nh = ent[q->prev].A; }
    e_info->v[2] = nt;
    e_info->v[3] = nh;
    e_info->x[2] = get_coord(nt);
    e_info->x[3] = get_coord(nh);
    e_info->vcount = 4;
  }
}

/* ---- second-order duals in the four vertices' twelve coordinates ---- */

#define ND 12
typedef struct { REAL v, g[ND], h[ND][ND]; } d12;

static __thread int dorder;     /* 0 value, 1 gradient, 2 Hessian (per call) */

static void dz(d12 *a, REAL v)
{ a->v = v;
  if ( dorder >= 1 ) memset(a->g,0,sizeof(a->g));
  if ( dorder >= 2 ) memset(a->h,0,sizeof(a->h));
}

static void dvar(d12 *a, REAL v, int i)
{ dz(a,v);
  if ( dorder >= 1 ) a->g[i] = 1.0;
}

static void dadd(d12 *r, const d12 *a, const d12 *b, REAL sb)   /* r = a + sb b */
{ int i, j;
  r->v = a->v + sb*b->v;
  if ( dorder >= 1 ) for ( i = 0 ; i < ND ; i++ ) r->g[i] = a->g[i] + sb*b->g[i];
  if ( dorder >= 2 )
    for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ ) r->h[i][j] = a->h[i][j] + sb*b->h[i][j];
}

static void dmul(d12 *r, const d12 *a, const d12 *b)
{ d12 t;
  int i, j;
  t.v = a->v*b->v;
  if ( dorder >= 1 ) for ( i = 0 ; i < ND ; i++ ) t.g[i] = a->v*b->g[i] + b->v*a->g[i];
  if ( dorder >= 2 )
    for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ )
      t.h[i][j] = a->v*b->h[i][j] + b->v*a->h[i][j] + a->g[i]*b->g[j] + a->g[j]*b->g[i];
  *r = t;
}

static void dscale(d12 *a, REAL s)
{ int i, j;
  a->v *= s;
  if ( dorder >= 1 ) for ( i = 0 ; i < ND ; i++ ) a->g[i] *= s;
  if ( dorder >= 2 ) for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ ) a->h[i][j] *= s;
}

static void dsqrt(d12 *r, const d12 *a)
{ REAL s = sqrt(a->v);
  int i, j;
  d12 t;
  t.v = s;
  if ( dorder >= 1 ) for ( i = 0 ; i < ND ; i++ ) t.g[i] = a->g[i]/(2*s);
  if ( dorder >= 2 )
    for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ )
      t.h[i][j] = a->h[i][j]/(2*s) - a->g[i]*a->g[j]/(4*s*s*s);
  *r = t;
}

static void datan2(d12 *r, const d12 *Y, const d12 *X)
{ REAL r2 = X->v*X->v + Y->v*Y->v;
  int i, j;
  d12 t;
  t.v = atan2(Y->v,X->v);
  if ( dorder >= 1 && r2 > 0 )
    for ( i = 0 ; i < ND ; i++ ) t.g[i] = (X->v*Y->g[i] - Y->v*X->g[i])/r2;
  else if ( dorder >= 1 ) memset(t.g,0,sizeof(t.g));
  if ( dorder >= 2 && r2 > 0 )
    for ( i = 0 ; i < ND ; i++ ) for ( j = 0 ; j < ND ; j++ )
      t.h[i][j] = (X->g[j]*Y->g[i] + X->v*Y->h[i][j] - Y->g[j]*X->g[i] - Y->v*X->h[i][j])/r2
        - (X->v*Y->g[i] - Y->v*X->g[i])*(2*X->v*X->g[j] + 2*Y->v*Y->g[j])/(r2*r2);
  else if ( dorder >= 2 ) memset(t.h,0,sizeof(t.h));
  *r = t;
}

static void ddot(d12 *r, const d12 *a, const d12 *b)
{ d12 t;
  dmul(r,a,b);
  dmul(&t,a+1,b+1); dadd(r,r,&t,1.0);
  dmul(&t,a+2,b+2); dadd(r,r,&t,1.0);
}

/* the turning angle at v from u to w (each three duals) */
static void dturning(d12 *r, const d12 *u, const d12 *v, const d12 *w)
{ d12 vv, uv, vw, uw, det, t, s, X, Y;
  ddot(&vv,v,v); ddot(&uv,u,v); ddot(&vw,v,w); ddot(&uw,u,w);
  /* det(u, v, w) = u . (v x w) */
  dmul(&t,v+1,w+2); dmul(&s,v+2,w+1); dadd(&t,&t,&s,-1.0); dmul(&det,u,&t);
  dmul(&t,v+2,w);   dmul(&s,v,w+2);   dadd(&t,&t,&s,-1.0); dmul(&s,u+1,&t); dadd(&det,&det,&s,1.0);
  dmul(&t,v,w+1);   dmul(&s,v+1,w);   dadd(&t,&t,&s,-1.0); dmul(&s,u+2,&t); dadd(&det,&det,&s,1.0);
  dsqrt(&t,&vv); dmul(&Y,&t,&det);
  dmul(&X,&uv,&vw); dmul(&t,&uw,&vv); dadd(&X,&X,&t,-1.0);
  datan2(r,&Y,&X);
}

/* an entry's term: wetted area share (*area) and the volume line term
   (*line), as duals in the element's slots 0..3 (tail, head, beside the
   tail, beside the head) */
static void entry_terms(const struct pc_entry *q, struct qinfo *e_info, d12 *area, d12 *line)
{ const struct pc_sphere *p = sph + q->s;
  d12 X[4][3], ta, tb, t, s;
  int slotA, slotB, slotP, slotN, i, k;
  REAL w[3];
  for ( i = 0 ; i < 4 ; i++ )
  { /* (slots 2 and 3 only where the setup found the line's neighbours) */
    if ( i < e_info->vcount ) pc_delta(p->c,get_coord(e_info->v[i]),w);
    else w[0] = w[1] = w[2] = 0.0;
    for ( k = 0 ; k < 3 ; k++ ) dvar(&X[i][k],w[k],3*i + k);
  }
  if ( get_edge_tailv(e_info->id) == q->A ) { slotA = 0; slotB = 1; slotP = 2; slotN = 3; }
  else { slotA = 1; slotB = 0; slotP = 3; slotN = 2; }
  if ( e_info->vcount == 4 && q->prev >= 0 && q->next >= 0
       && e_info->v[slotP] == ent[q->prev].A && e_info->v[slotN] == ent[q->next].B )
  { dturning(&ta,X[slotP],X[slotA],X[slotB]);
    dturning(&tb,X[slotA],X[slotB],X[slotN]);
    dadd(area,&ta,&tb,1.0);
    dscale(area,-0.5*p->R*p->R);
  }
  else dz(area,0.0);
  /* line: -(a_x b_y - a_y b_x)((a_z + b_z)/6 + h/2), a = A - c, b = B - c */
  { d12 *a = X[slotA], *b = X[slotB];
    REAL h = p->c[2] - p->zf;
    dmul(&t,a,b+1); dmul(&s,a+1,b); dadd(&t,&t,&s,-1.0);
    dadd(&s,a+2,b+2,1.0); dscale(&s,1.0/6.0); s.v += h/2;
    dmul(line,&t,&s); dscale(line,-1.0);
  }
}

/* the element's value (and derivatives) for instance mi */
static REAL pc_eval(struct qinfo *e_info, int order)
{ struct method_instance *mi = METH_INSTANCE(e_info->method);
  int s = sphere_of(mi->connum), k, o, body = -1, i, j, a, b2;
  d12 tot, area, line;
  dorder = order;
  dz(&tot,0.0);
  if ( s < 0 || !nent ) goto out;
  o = ordinal(e_info->id);
  if ( o >= nslot ) goto out;
  if ( mi->flags & BODY_INSTANCE )
  { for ( j = 0 ; j < MMAXQUANTS ; j++ )
      if ( mi->quants[j] >= 0 )
      { body_id bid = GEN_QUANT(mi->quants[j])->b_id;
        if ( valid_id(bid) ) body = ordinal(bid);
        break;
      }
    if ( body < 0 ) goto out;
  }
  for ( k = slot[o] ; k >= 0 ; k = ent[k].other )
  { const struct pc_entry *q = ent + k;
    if ( q->s != s || (body >= 0 && q->b != body) ) continue;
    entry_terms(q,e_info,&area,&line);
    if ( body >= 0 )
    { /* the body's content, as Evolver applies it: times sapp */
      dscale(&area,-sph[s].R/3.0);
      dadd(&area,&area,&line,1.0);
      dadd(&tot,&tot,&area,(REAL)q->sapp);
    }
    else
    { dadd(&tot,&tot,&area,-sphere_k(sph + s));
    }
  }
out:
  if ( order >= 1 )
    for ( a = 0 ; a < e_info->vcount && a < 4 ; a++ )
      for ( i = 0 ; i < 3 ; i++ ) e_info->grad[a][i] = tot.g[3*a + i];
  if ( order >= 2 )
    for ( a = 0 ; a < e_info->vcount && a < 4 ; a++ )
      for ( b2 = 0 ; b2 < e_info->vcount && b2 < 4 ; b2++ )
        for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ )
          e_info->hess[a][b2][i][j] = tot.h[3*a + i][3*b2 + j];
  return tot.v;
}

/* before a pass's quantities are put together: the table built if no
   instance of ours started it (a pass of the volumes alone where no body
   has a line) */
void pyse_contact_refresh(void)
{ if ( nsph && (!pc_built || pc_changed()) ) pc_build();
  pc_built = 0;
  pc_stale = 1;
}

/* what a body's volume (quantity k) adds beyond its edges: the loops'
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

/* what an energy instance adds beyond its edges: the loops' constants (2 pi R^2
   a loop less the whole spheres n), and the spheres inside a body whole
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
