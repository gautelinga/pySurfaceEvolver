/*
 * pyse_native.c -- the contact-line integrands of pySE's sphere walls in C.
 *
 * A wall from constraints.sphere(..., axis=) has energy and content
 * integrands that Evolver evaluates through its expression interpreter at
 * each Gauss point of each contact-line edge, with first and second
 * derivatives (edge_vector_integral and its gradient and Hessian): 22% of a
 * 16-bead pack's slow gradient steps. Registered here by their expression
 * nodes (the method instances made from a constraint share them), they are
 * evaluated directly, the derivatives carried along as second-order dual
 * numbers: the same values to round-off.
 *
 * The integrands (constraints._axis_sphere), with a, b, h the coordinates
 * of x - c in the frame e1, e2, e3:
 *   D = a^2 + b^2 + (R^2 - a^2 - b^2 - h^2)(R - h)/(2R)
 *   area_i = R (R + h)(e2_i a - e1_i b)/D
 *   energy_i = k area_i
 *   content_i = R/3 area_i + (Z/3 Y, -(c_z + Z/3) X, 0)_i,  X = x - c etc.
 */

#include "include.h"
#include "pyse_api.h"
#include "pyse_internal.h"

/* a value with its gradient and Hessian in x, y, z */
typedef struct { REAL v, g[3], h[3][3]; } dual;

static dual d_lin(const REAL *coef, REAL constant, const REAL *x)
{ dual r;
  int i, j;
  r.v = constant;
  for ( i = 0 ; i < 3 ; i++ )
  { r.v += coef[i]*x[i];
    r.g[i] = coef[i];
    for ( j = 0 ; j < 3 ; j++ ) r.h[i][j] = 0.0;
  }
  return r;
}

static dual d_const(REAL v)
{ dual r;
  memset(&r,0,sizeof(r));
  r.v = v;
  return r;
}

static dual d_add(dual a, dual b)
{ int i, j;
  a.v += b.v;
  for ( i = 0 ; i < 3 ; i++ )
  { a.g[i] += b.g[i];
    for ( j = 0 ; j < 3 ; j++ ) a.h[i][j] += b.h[i][j];
  }
  return a;
}

static dual d_scale(dual a, REAL s)
{ int i, j;
  a.v *= s;
  for ( i = 0 ; i < 3 ; i++ )
  { a.g[i] *= s;
    for ( j = 0 ; j < 3 ; j++ ) a.h[i][j] *= s;
  }
  return a;
}

static dual d_mul(dual a, dual b)
{ dual r;
  int i, j;
  r.v = a.v*b.v;
  for ( i = 0 ; i < 3 ; i++ )
  { r.g[i] = a.v*b.g[i] + b.v*a.g[i];
    for ( j = 0 ; j < 3 ; j++ )
      r.h[i][j] = a.v*b.h[i][j] + b.v*a.h[i][j] + a.g[i]*b.g[j] + b.g[i]*a.g[j];
  }
  return r;
}

static dual d_inv(dual b)
{ dual r;
  REAL v = 1.0/b.v, v2 = v*v, v3 = v2*v;
  int i, j;
  r.v = v;
  for ( i = 0 ; i < 3 ; i++ )
  { r.g[i] = -b.g[i]*v2;
    for ( j = 0 ; j < 3 ; j++ ) r.h[i][j] = -b.h[i][j]*v2 + 2.0*b.g[i]*b.g[j]*v3;
  }
  return r;
}

#define NATIVE_MAX 4096

struct native
{ struct expnode *energy, *content;   /* the constraint's integrands' first nodes */
  REAL c[3], R, e[3][3], k;           /* centre, radius, frame e1 e2 e3, cos(angle)*tension */
};

static struct native natives[NATIVE_MAX];
static int native_count = 0;
static long native_load = -1;
static int native_on = 1;

static long native_serial = 0;   /* bumped at every registration or toggle */
long pyse_native_serial(void) { return native_serial; }

void pyse_native_enable(int on) { native_on = on; native_serial++; }

/* register constraint con as a sphere wall (k: its energy coefficient, NaN
   where it isn't a number: the content only); 0 if it can't be (no
   integrands, too many) */
int pyse_native_sphere(int con, const double *c, double R, const double *e1,
                       const double *e2, const double *e3, double k)
{ struct constraint *cn;
  struct native *n;
  int i;
  native_serial++;
  if ( native_load != pyse_load_serial() ) { native_count = 0; native_load = pyse_load_serial(); }
  if ( con < 0 || con >= web.maxcon || native_count >= NATIVE_MAX ) return 0;
  cn = get_constraint(con);
  if ( !cn || !cn->convect[0] ) return 0;
  for ( i = 0, n = NULL ; i < native_count ; i++ )
    if ( natives[i].content == cn->convect[0] ) n = natives + i;   /* (again: updated) */
  if ( !n ) n = natives + native_count++;
  n->energy = isnan(k) ? NULL : cn->envect[0];   /* (an angle that is a parameter) */
  n->content = cn->convect[0];
  for ( i = 0 ; i < 3 ; i++ )
  { n->c[i] = c[i]; n->e[0][i] = e1[i]; n->e[1][i] = e2[i]; n->e[2][i] = e3[i]; }
  n->R = R;
  n->k = k;
  return 1;
}

static int flux_eval(struct expnode *key, REAL *x, int order, REAL *val,
                     REAL (*d)[MAXCOORD], REAL ***s);

/* the integrand vector of the wall whose integrand key is (its first
   node), at x: values, and with order 1 or 2 their first (and second)
   derivatives; 0 where key isn't a registered wall's */
int pyse_native_eval(struct expnode *key, REAL *x, int order, REAL *val,
                     REAL (*d)[MAXCOORD], REAL ***s)
{ struct native *n = NULL;
  int i, j, m, content = 0;
  dual a, b, h, D, rr, lead, area[3];
  REAL off;
  if ( !native_on || !key || SDIM != 3 ) return 0;
  if ( native_load == pyse_load_serial() )
    for ( i = 0 ; i < native_count ; i++ )
    { if ( natives[i].energy == key ) { n = natives + i; break; }
      if ( natives[i].content == key ) { n = natives + i; content = 1; break; }
    }
  if ( !n ) return flux_eval(key,x,order,val,d,s);
  for ( i = 0, off = 0.0 ; i < 3 ; i++ ) off -= n->e[0][i]*n->c[i];
  a = d_lin(n->e[0],off,x);
  for ( i = 0, off = 0.0 ; i < 3 ; i++ ) off -= n->e[1][i]*n->c[i];
  b = d_lin(n->e[1],off,x);
  for ( i = 0, off = 0.0 ; i < 3 ; i++ ) off -= n->e[2][i]*n->c[i];
  h = d_lin(n->e[2],off,x);
  /* D = a^2 + b^2 + (R^2 - a^2 - b^2 - h^2)(R - h)/(2R) */
  rr = d_add(d_add(d_mul(a,a),d_mul(b,b)),d_mul(h,h));
  D = d_add(d_add(d_mul(a,a),d_mul(b,b)),
            d_scale(d_mul(d_add(d_const(n->R*n->R),d_scale(rr,-1.0)),
                          d_add(d_const(n->R),d_scale(h,-1.0))),1.0/(2.0*n->R)));
  /* R (R + h)/D */
  lead = d_mul(d_scale(d_add(d_const(n->R),h),n->R),d_inv(D));
  for ( m = 0 ; m < 3 ; m++ )
    area[m] = d_mul(lead,d_add(d_scale(a,n->e[1][m]),d_scale(b,-n->e[0][m])));
  for ( m = 0 ; m < 3 ; m++ )
  { dual t = d_scale(area[m],content ? n->R/3.0 : n->k);
    if ( content && m < 2 )
    { REAL ex[3] = {1,0,0}, ey[3] = {0,1,0}, ez[3] = {0,0,1};
      dual X = d_lin(ex,-n->c[0],x), Y = d_lin(ey,-n->c[1],x), Z = d_lin(ez,-n->c[2],x);
      /* Z/3 Y; -(c_z + Z/3) X */
      if ( m == 0 ) t = d_add(t,d_mul(d_scale(Z,1.0/3.0),Y));
      else t = d_add(t,d_scale(d_mul(d_add(d_const(n->c[2]),d_scale(Z,1.0/3.0)),X),-1.0));
    }
    val[m] = t.v;
    if ( order >= 1 ) for ( i = 0 ; i < 3 ; i++ ) d[m][i] = t.g[i];
    if ( order >= 2 )
      for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ ) s[m][i][j] = t.h[i][j];
  }
  return 1;
}

/* ---- the flux gauge (constraints.sphere(gauge="flux"), _flux.py) ---------
 *
 * Its integrands sum over the spheres: on the facets, point sources
 * R^2 (x - c)/|x - c|^3, R^2 (x - c)/K^3 inside a cap of radius K =
 * FLUX_CORE R (bounded where no liquid is: no jump at the centre); on each
 * wall's contact lines, monopole potentials
 * R^2 (d x w)/(r (r - d . w)), w = x - c; each term times -k (the wetting
 * energy, k = tension cos(max(angle, pyse_angle_floor)) for angles under
 * ANGLE_FLOOR_TOP, else tension cos(angle)) or -R/3 (the volume). Plus the
 * wall's own terms: (0, 0, z) for the volume's facets, a flux sphere's exact
 * content part (Z/3 Y, -(c_z + Z/3) X, 0) with Z = z - c_z etc. Registered
 * by the integrand's first expression node, as the spheres above.
 *
 * Every sphere's term at every Gauss point, exactly: the field's far part
 * has no cheaper smooth stand-in at these sizes. A multipole or Taylor
 * expansion of the far spheres converges as (a/D)^p, a the evaluation
 * cell's radius and D the distance to its nearest far sphere: to 1e-10 of
 * the field with a 3 x 3 x 3 block of cells near (a/D up to 0.6) p is over
 * 40, with 5 x 5 x 5 (0.35) over 20, a polynomial of 1771 terms and its
 * derivatives at each point, against the 64 sources of the 64-bead pack's
 * whole box; and the switch between near and far, as a vertex moves, must
 * be blended (an energy's jump or kink there broke Evolver's line search
 * and Newton). So the direct sum, made cheap: the terms of a field put in
 * a group (their centres, directions and caps as arrays), four at a time
 * with AVX2; a group's sum shared by the fields that differ only in the
 * coefficients (the volume's and the wetting's integrands, each body's
 * copy of the volume's), once a Gauss point; second derivatives in closed
 * form. */

#define FLUX_FLOOR_TOP 20.0       /* constraints.ANGLE_FLOOR_TOP */
#define FLUX_CORE 0.5             /* _flux.CORE: the field's cap radius, of R */
#define FLUX_LANES 4              /* terms at a time, a group padded to it */
#define FLUX_PAD 1e8              /* a padding term's centre (its coefficient 0) */
#define FLUX_COLS 2               /* coefficient columns a group evaluates at once */
#define FLUX_OUT 30               /* a kernel's results, at most (flux_kernel) */
#define FLUX_CACHE 16             /* a thread's last Gauss points' sums */

struct flux_term
{ REAL c[3], R2, d[3];           /* d zero: a point source */
  REAL K2, K3;                   /* the cap's radius (FLUX_CORE R) squared, cubed */
  int energy;                    /* -k R^2 (else -R/3 R^2) */
  REAL angle, tension, R;
  REAL coef;                     /* the coefficient with the floor below the angle */
};

/* a field's terms as arrays: point sources (kind 0, a facet integrand) or
   monopoles (kind 1, a contact line's), padded to FLUX_LANES with terms of
   coefficient 0; and up to FLUX_COLS columns of coefficients, the fields'
   (c0, raised to kc cos(floor) where the floor is over athr, the angle).
   Uniform where every column's coefficients are the same for all its terms
   (equal spheres at one angle, a pack's): the sum is then taken once with
   coefficients 1 (one: 1, 0 for the padding) and each column is it times
   its coefficient (the volume's projection takes the volume alone at most
   Gauss points, where the wetting's column cost half as much again). */
struct flux_group
{ int kind, n, np;
  REAL *cx, *cy, *cz, *dx, *dy, *dz, *K2, *iK3, *one;
  int ncol, uniform;
  REAL *c0[FLUX_COLS], *kc[FLUX_COLS], *athr[FLUX_COLS], amin[FLUX_COLS];
};

struct flux_field
{ struct expnode *key;
  int base;                      /* 0 none, 1 (0,0,z), 2 a sphere's content part,
                                    3 a plane's content, 4 a plane's wetting */
  REAL bc[12];                   /* 2: the sphere's centre; 3: n_z, o, u, v;
                                    4: angle, tension, o, u, v */
  int nterm;
  struct flux_term *term;
  int group, col;                /* its group and column (-1: term by term) */
};

/* (as many as registered: a 256-bead pack has 522 line integrands of 255
   terms each; capped at 512 fields of 64 terms, the rest went to the
   interpreter, a gradient step 2.2 ms a facet against 0.9 with poles) */
static struct flux_field *fluxes = NULL;
static int flux_count = 0, flux_cap = 0;
static struct flux_group *flux_groups = NULL;
static int flux_ngroups = 0, flux_gcap = 0;
static int *flux_hash = NULL;    /* key -> field index + 1, open addressing */
static int flux_hash_size = 0;
static long flux_load = -1;
static long flux_gen = 0;        /* bumped at each registration: the caches' */

static void flux_group_free(struct flux_group *g)
{ int c;
  free(g->cx); free(g->cy); free(g->cz); free(g->dx); free(g->dy); free(g->dz);
  free(g->K2); free(g->iK3); free(g->one);
  for ( c = 0 ; c < g->ncol ; c++ ) { free(g->c0[c]); free(g->kc[c]); free(g->athr[c]); }
  memset(g,0,sizeof(*g));
}

static void flux_reset(void)
{ int i;
  for ( i = 0 ; i < flux_count ; i++ ) free(fluxes[i].term);
  for ( i = 0 ; i < flux_ngroups ; i++ ) flux_group_free(flux_groups + i);
  flux_count = flux_ngroups = 0;
  for ( i = 0 ; i < flux_hash_size ; i++ ) flux_hash[i] = 0;
  flux_gen++;
}

static unsigned flux_slot(const struct expnode *key, int size)
{ unsigned long long h = (unsigned long long)(size_t)key*0x9E3779B97F4A7C15ULL;
  return (unsigned)(h >> 32) & (unsigned)(size - 1);
}

/* the field of key, or NULL (read only: called from the OpenMP threads; a
   linear search over the fields, 10 kB apart, was an eighth of the time in
   flux_eval on a 4-bead pack) */
static struct flux_field *flux_find(const struct expnode *key)
{ unsigned s;
  if ( !flux_hash_size ) return NULL;
  for ( s = flux_slot(key,flux_hash_size) ; flux_hash[s] ; s = (s + 1) & (flux_hash_size - 1) )
    if ( fluxes[flux_hash[s]-1].key == key ) return fluxes + flux_hash[s] - 1;
  return NULL;
}

static int flux_hash_put(int index)
{ unsigned s;
  if ( 2*(index + 1) > flux_hash_size )
  { int size = flux_hash_size ? 2*flux_hash_size : 64, i;
    int *h = (int*)calloc(size,sizeof(int));
    if ( !h ) return 0;
    free(flux_hash);
    flux_hash = h;
    flux_hash_size = size;
    for ( i = 0 ; i < index ; i++ )
    { s = flux_slot(fluxes[i].key,size);
      while ( h[s] ) s = (s + 1) & (size - 1);
      h[s] = i + 1;
    }
  }
  s = flux_slot(fluxes[index].key,flux_hash_size);
  while ( flux_hash[s] ) s = (s + 1) & (flux_hash_size - 1);
  flux_hash[s] = index + 1;
  return 1;
}

static dual d_pow(dual a, REAL p)
{ dual r;
  REAL v = pow(a.v,p), v1 = p*pow(a.v,p-1.0), v2 = p*(p-1.0)*pow(a.v,p-2.0);
  int i, j;
  r.v = v;
  for ( i = 0 ; i < 3 ; i++ )
  { r.g[i] = v1*a.g[i];
    for ( j = 0 ; j < 3 ; j++ ) r.h[i][j] = v1*a.h[i][j] + v2*a.g[i]*a.g[j];
  }
  return r;
}

/* the angle floor's value (its global looked up once a load: by name, at
   every term and Gauss point, the lookup was a quarter of a step) */
static REAL flux_floor(void)
{ static long load = -2;
  static int k = -1;
  if ( load != pyse_load_serial() ) { k = lookup_global("pyse_angle_floor"); load = pyse_load_serial(); }
  return k >= 0 ? globals(k)->value.real : 0.0;
}

static REAL flux_coefficient(const struct flux_term *t, REAL floor)
{ /* (read only: called from the OpenMP threads; the cosine only where the
     floor is raised past the angle, relax()'s staging) */
  if ( t->energy && t->angle > 0 && t->angle < FLUX_FLOOR_TOP && floor > t->angle )
    return -t->tension*cos(floor*M_PI/180.0)*t->R2;
  return t->coef;
}

/* a plane's wetting coefficient tension cos(angle), the floor raised past
   an angle under FLUX_FLOOR_TOP as for the spheres (constraints._Coefficient) */
static REAL plane_coefficient(const struct flux_field *f, REAL floor)
{ REAL angle = f->bc[0], tension = f->bc[1];
  if ( angle > 0 && angle < FLUX_FLOOR_TOP && floor > angle ) angle = floor;
  return tension*cos(angle*M_PI/180.0);
}

static REAL *flux_array(int n)
{ return (REAL*)malloc((size_t)(n > 0 ? n : 1)*sizeof(REAL));
}

/* field f's group and column (kind 0 or 1): one with the same terms but
   for their coefficients, its column the same coefficients or a new one;
   else a new group. 0 if out of memory. */
static int flux_group_for(struct flux_field *f, int kind)
{ int n = f->nterm, np = (n + FLUX_LANES - 1)/FLUX_LANES*FLUX_LANES, i, c, gi;
  struct flux_group t, *g = NULL;
  REAL *c0 = flux_array(np), *kc = flux_array(np), *athr = flux_array(np), amin = HUGE_VAL;
  size_t bytes = (size_t)np*sizeof(REAL);
  memset(&t,0,sizeof(t));
  t.kind = kind; t.n = n; t.np = np;
  t.cx = flux_array(np); t.cy = flux_array(np); t.cz = flux_array(np);
  t.dx = flux_array(np); t.dy = flux_array(np); t.dz = flux_array(np);
  t.K2 = flux_array(np); t.iK3 = flux_array(np); t.one = flux_array(np);
  if ( !c0 || !kc || !athr || !t.cx || !t.cy || !t.cz || !t.dx || !t.dy || !t.dz || !t.K2
       || !t.iK3 || !t.one )
    goto fail;
  for ( i = 0 ; i < np ; i++ )
  { if ( i < n )
    { const struct flux_term *u = f->term + i;
      int floored = u->energy && u->angle > 0 && u->angle < FLUX_FLOOR_TOP;
      t.cx[i] = u->c[0]; t.cy[i] = u->c[1]; t.cz[i] = u->c[2];
      t.dx[i] = kind ? u->d[0] : 0.0; t.dy[i] = kind ? u->d[1] : 0.0;
      t.dz[i] = kind ? u->d[2] : 0.0;
      t.K2[i] = kind ? 0.0 : u->K2; t.iK3[i] = kind ? 0.0 : 1.0/u->K3;
      t.one[i] = 1.0;
      c0[i] = u->coef;
      kc[i] = floored ? -u->tension*u->R2 : 0.0;
      athr[i] = floored ? u->angle : HUGE_VAL;
      if ( athr[i] < amin ) amin = athr[i];
    }
    else
    { t.cx[i] = t.cy[i] = t.cz[i] = FLUX_PAD;
      t.dx[i] = t.dy[i] = 0.0; t.dz[i] = kind ? 1.0 : 0.0;
      t.K2[i] = t.iK3[i] = t.one[i] = c0[i] = kc[i] = 0.0;
      athr[i] = HUGE_VAL;
    }
  }
  for ( gi = 0 ; gi < flux_ngroups ; gi++ )
  { struct flux_group *h = flux_groups + gi;
    if ( h->kind != kind || h->n != n || memcmp(h->cx,t.cx,bytes) || memcmp(h->cy,t.cy,bytes)
         || memcmp(h->cz,t.cz,bytes) || memcmp(h->dx,t.dx,bytes) || memcmp(h->dy,t.dy,bytes)
         || memcmp(h->dz,t.dz,bytes) || memcmp(h->K2,t.K2,bytes) )
      continue;
    for ( c = 0 ; c < h->ncol ; c++ )
      if ( !memcmp(h->c0[c],c0,bytes) && !memcmp(h->kc[c],kc,bytes)
           && !memcmp(h->athr[c],athr,bytes) )
      { f->group = gi; f->col = c;
        flux_group_free(&t);
        free(c0); free(kc); free(athr);
        return 1;
      }
    if ( h->ncol < FLUX_COLS ) { g = h; break; }
  }
  if ( g ) flux_group_free(&t);
  else
  { if ( flux_ngroups >= flux_gcap )
    { int cap = flux_gcap ? 2*flux_gcap : 8;
      struct flux_group *ng = (struct flux_group*)realloc(flux_groups,cap*sizeof(*ng));
      if ( !ng ) goto fail;
      flux_groups = ng;
      flux_gcap = cap;
    }
    g = flux_groups + flux_ngroups++;
    *g = t;
    g->uniform = 1;
  }
  for ( i = 1 ; i < n ; i++ )
    if ( c0[i] != c0[0] || kc[i] != kc[0] || athr[i] != athr[0] ) g->uniform = 0;
  c = g->ncol++;
  g->c0[c] = c0; g->kc[c] = kc; g->athr[c] = athr; g->amin[c] = amin;
  f->group = (int)(g - flux_groups);
  f->col = c;
  return 1;
fail:
  flux_group_free(&t);
  free(c0); free(kc); free(athr);
  return 0;
}

/* register a field: target a method instance (its name) or a constraint's
   energy (which 0) or content (1) integrand; terms 11 numbers each (c, R,
   d, energy flag, angle, tension, and a spare); 0 if it can't be */
int pyse_native_flux(const char *method, int con, int which, int base, const double *bc,
                     int nbc, const double *terms, int nterm)
{ struct expnode *key = NULL;
  struct flux_field *f = NULL;
  int i, k, points, poles;
  native_serial++;
  if ( flux_load != pyse_load_serial() ) { flux_reset(); flux_load = pyse_load_serial(); }
  if ( nterm < 0 || SDIM != 3 || nbc > 12 ) return 0;
  if ( method && method[0] )
  { int m = find_method_instance((char *)method);
    if ( m < 0 ) return 0;
    key = METH_INSTANCE(m)->expr[0];
  }
  else
  { struct constraint *cn;
    if ( con < 0 || con >= web.maxcon ) return 0;
    cn = get_constraint(con);
    if ( !cn ) return 0;
    key = which ? cn->convect[0] : cn->envect[0];
  }
  if ( !key ) return 0;
  flux_gen++;
  f = flux_find(key);
  if ( f )
  { free(f->term);
    f->term = NULL;
  }
  else
  { if ( flux_count >= flux_cap )
    { int cap = flux_cap ? 2*flux_cap : 64;
      struct flux_field *nf = (struct flux_field*)realloc(fluxes,cap*sizeof(*nf));
      if ( !nf ) return 0;
      fluxes = nf;
      flux_cap = cap;
    }
    f = fluxes + flux_count;
    memset(f,0,sizeof(*f));
    f->key = key;
    if ( !flux_hash_put(flux_count) ) return 0;
    flux_count++;
  }
  f->group = -1;
  f->nterm = 0;
  f->base = base;
  for ( k = 0 ; k < 12 ; k++ ) f->bc[k] = bc && k < nbc ? bc[k] : 0.0;
  f->term = (struct flux_term*)calloc(nterm > 0 ? nterm : 1,sizeof(struct flux_term));
  if ( !f->term ) return 0;
  f->nterm = nterm;
  points = poles = nterm > 0;
  for ( i = 0 ; i < nterm ; i++ )
  { const double *t = terms + 11*i;
    struct flux_term *u = f->term + i;
    for ( k = 0 ; k < 3 ; k++ ) { u->c[k] = t[k]; u->d[k] = t[4+k]; }
    u->R = t[3]; u->R2 = t[3]*t[3];
    u->K2 = FLUX_CORE*FLUX_CORE*u->R2; u->K3 = u->K2*FLUX_CORE*u->R;
    u->energy = t[7] != 0;
    u->angle = t[8]; u->tension = t[9];
    u->coef = u->energy ? -u->tension*cos(u->angle*M_PI/180.0)*u->R2 : -u->R/3.0*u->R2;
    if ( u->d[0] != 0 || u->d[1] != 0 || u->d[2] != 0 || !(u->K3 > 0) ) points = 0;
    if ( u->d[0] == 0 && u->d[1] == 0 && u->d[2] == 0 ) poles = 0;
  }
  if ( (points || poles) && !flux_group_for(f,poles) )
    f->group = -1;              /* (out of memory: term by term) */
  return 1;
}

/* The kernels: a group's terms at x, for each of its ncol coefficient
   columns, out[col][]. Each of FLUX_LANES lanes sums its terms, the lanes
   added at the end as ((0 + 1) + (2 + 3)); the AVX2 kernels and the scalar
   ones take the same IEEE operations in the same order (no fused ones:
   -ffp-contract=off), so they agree to the bit.

   Point sources (kind 0): F = coef w/r^3, w = x - c (coef w/K^3 inside the
   cap). With a = coef/r^3, b = 3a/r^2, e = 15a/r^4 (b = e = 0 inside):
     dF_m/dx_i = a d_mi - b w_m w_i,
     d2F_m/dx_i dx_j = -b (d_mi w_j + d_mj w_i + d_ij w_m) + e w_m w_i w_j;
   out: F 3, dF 6 (00 11 22 01 02 12), d2F 10 (000 111 222 001 002 110 112
   220 221 012: F = -grad of a potential, symmetric in all indices).

   Monopoles (kind 1): A = coef (d x w) phi, phi = 1/D, D = r (r - d . w):
     D_i = (2 - q) w_i - r d_i, q = d . w/r,
     D_ij = (2 - q) d_ij + q w_i w_j/r^2 - (w_i d_j + w_j d_i)/r,
     phi_i = -D_i phi^2, phi_ij = 2 D_i D_j phi^3 - D_ij phi^2,
     dA_m/dx_i = coef ((d x e_i)_m phi + (d x w)_m phi_i),
     d2A_m/dx_i dx_j = coef ((d x e_i)_m phi_j + (d x e_j)_m phi_i + (d x w)_m phi_ij)
   (1/r = (r - d . w) phi: no second division); out: A 3, dA 9 (m i),
   d2A 18 (m: 00 11 22 01 02 12). */
#define FLUX_POINT_TERM(T, ADD, SUB, MUL, DIV, SQRT, LT, SEL, ZERO, ONE, LOADC)              \
  { T w0 = SUB(X0,LOADC(g->cx)), w1 = SUB(X1,LOADC(g->cy)), w2 = SUB(X2,LOADC(g->cz));      \
    T r2 = ADD(ADD(MUL(w0,w0),MUL(w1,w1)),MUL(w2,w2)), r = SQRT(r2);                         \
    T u = DIV(ONE,MUL(r2,r)), q = MUL(u,r), iK3 = LOADC(g->iK3), s;                          \
    int c;                                                                                   \
    LT(in,r2,LOADC(g->K2));                                                                  \
    s = SEL(in,iK3,u);                                                                       \
    for ( c = 0 ; c < ncol ; c++ )                                                           \
    { T a = MUL(LOADC(coef[c]),s), *A = acc[c];                                              \
      A[0] = ADD(A[0],MUL(w0,a)); A[1] = ADD(A[1],MUL(w1,a)); A[2] = ADD(A[2],MUL(w2,a));    \
      if ( order >= 1 )                                                                      \
      { T b = SEL(in,ZERO,MUL(MUL(K_3,a),q));                                                \
        T bw0 = MUL(b,w0), bw1 = MUL(b,w1), bw2 = MUL(b,w2);                                 \
        A[3] = ADD(A[3],SUB(a,MUL(bw0,w0)));                                                 \
        A[4] = ADD(A[4],SUB(a,MUL(bw1,w1)));                                                 \
        A[5] = ADD(A[5],SUB(a,MUL(bw2,w2)));                                                 \
        A[6] = SUB(A[6],MUL(bw0,w1));                                                        \
        A[7] = SUB(A[7],MUL(bw0,w2));                                                        \
        A[8] = SUB(A[8],MUL(bw1,w2));                                                        \
        if ( order >= 2 )                                                                    \
        { T e = SEL(in,ZERO,MUL(MUL(K_5,b),q));                                              \
          T ew0 = MUL(e,w0), s0 = MUL(ew0,w0), s1 = MUL(MUL(e,w1),w1), s2 = MUL(MUL(e,w2),w2); \
          T b3 = MUL(K_3,b);                                                                 \
          A[9] = ADD(A[9],MUL(SUB(s0,b3),w0));                                               \
          A[10] = ADD(A[10],MUL(SUB(s1,b3),w1));                                             \
          A[11] = ADD(A[11],MUL(SUB(s2,b3),w2));                                             \
          A[12] = ADD(A[12],MUL(SUB(s0,b),w1));                                              \
          A[13] = ADD(A[13],MUL(SUB(s0,b),w2));                                              \
          A[14] = ADD(A[14],MUL(SUB(s1,b),w0));                                              \
          A[15] = ADD(A[15],MUL(SUB(s1,b),w2));                                              \
          A[16] = ADD(A[16],MUL(SUB(s2,b),w0));                                              \
          A[17] = ADD(A[17],MUL(SUB(s2,b),w1));                                              \
          A[18] = ADD(A[18],MUL(MUL(ew0,w1),w2));                                            \
        }                                                                                    \
      }                                                                                      \
    }                                                                                        \
  }

/* (the indices of the six i <= j pairs) */
#define FLUX_PAIRS { {0,0},{1,1},{2,2},{0,1},{0,2},{1,2} }

#define FLUX_POLE_TERM(T, ADD, SUB, MUL, DIV, SQRT, LT, SEL, ZERO, ONE, LOADC)               \
  { T w[3], d[3], C[3], dC[3][3], r2, r, dw, rd, phi;                                        \
    int c, m, i, k;                                                                          \
    w[0] = SUB(X0,LOADC(g->cx)); w[1] = SUB(X1,LOADC(g->cy)); w[2] = SUB(X2,LOADC(g->cz));   \
    d[0] = LOADC(g->dx); d[1] = LOADC(g->dy); d[2] = LOADC(g->dz);                           \
    r2 = ADD(ADD(MUL(w[0],w[0]),MUL(w[1],w[1])),MUL(w[2],w[2]));                             \
    r = SQRT(r2);                                                                            \
    dw = ADD(ADD(MUL(d[0],w[0]),MUL(d[1],w[1])),MUL(d[2],w[2]));                             \
    rd = SUB(r,dw);                                                                          \
    phi = DIV(ONE,MUL(r,rd));                                                                \
    C[0] = SUB(MUL(d[1],w[2]),MUL(d[2],w[1]));                                               \
    C[1] = SUB(MUL(d[2],w[0]),MUL(d[0],w[2]));                                               \
    C[2] = SUB(MUL(d[0],w[1]),MUL(d[1],w[0]));                                               \
    dC[0][0] = ZERO; dC[0][1] = SUB(ZERO,d[2]); dC[0][2] = d[1];                             \
    dC[1][0] = d[2]; dC[1][1] = ZERO; dC[1][2] = SUB(ZERO,d[0]);                             \
    dC[2][0] = SUB(ZERO,d[1]); dC[2][1] = d[0]; dC[2][2] = ZERO;                             \
    { T pg[3], ph[6], ir = ZERO, q = ZERO;                                                   \
      if ( order >= 1 )                                                                      \
      { T p2 = MUL(phi,phi), tq;                                                             \
        ir = MUL(rd,phi); q = MUL(dw,ir); tq = SUB(K_2,q);                                   \
        for ( i = 0 ; i < 3 ; i++ )                                                          \
          pg[i] = SUB(ZERO,MUL(SUB(MUL(tq,w[i]),MUL(r,d[i])),p2));                           \
        if ( order >= 2 )                                                                    \
        { static const int pr[6][2] = FLUX_PAIRS;                                            \
          T p3 = MUL(p2,phi), qr2 = MUL(MUL(q,ir),ir), Dg[3];                                \
          for ( i = 0 ; i < 3 ; i++ ) Dg[i] = SUB(MUL(tq,w[i]),MUL(r,d[i]));                 \
          for ( k = 0 ; k < 6 ; k++ )                                                        \
          { int a = pr[k][0], b = pr[k][1];                                                  \
            T Dh = SUB(MUL(MUL(w[a],w[b]),qr2),MUL(ADD(MUL(w[a],d[b]),MUL(w[b],d[a])),ir)); \
            if ( a == b ) Dh = ADD(tq,Dh);                                                   \
            ph[k] = SUB(MUL(MUL(K_2,MUL(Dg[a],Dg[b])),p3),MUL(Dh,p2));                       \
          }                                                                                  \
        }                                                                                    \
      }                                                                                      \
      for ( c = 0 ; c < ncol ; c++ )                                                         \
      { T cf = LOADC(coef[c]), a = MUL(cf,phi), cC[3], *A = acc[c];                          \
        for ( m = 0 ; m < 3 ; m++ )                                                          \
        { cC[m] = MUL(cf,C[m]);                                                              \
          A[m] = ADD(A[m],MUL(C[m],a));                                                      \
        }                                                                                    \
        if ( order >= 1 )                                                                    \
          for ( m = 0 ; m < 3 ; m++ )                                                        \
            for ( i = 0 ; i < 3 ; i++ )                                                      \
              A[3+3*m+i] = ADD(A[3+3*m+i],ADD(MUL(a,dC[m][i]),MUL(cC[m],pg[i])));            \
        if ( order >= 2 )                                                                    \
        { static const int pr[6][2] = FLUX_PAIRS;                                            \
          T G[3];                                                                            \
          for ( i = 0 ; i < 3 ; i++ ) G[i] = MUL(cf,pg[i]);                                  \
          for ( m = 0 ; m < 3 ; m++ )                                                        \
            for ( k = 0 ; k < 6 ; k++ )                                                      \
            { int a2 = pr[k][0], b2 = pr[k][1];                                              \
              A[12+6*m+k] = ADD(A[12+6*m+k],ADD(ADD(MUL(dC[m][a2],G[b2]),                    \
                                                    MUL(dC[m][b2],G[a2])),                   \
                                                MUL(cC[m],ph[k])));                          \
            }                                                                                \
        }                                                                                    \
      }                                                                                      \
    }                                                                                        \
  }

static int flux_nout(int kind, int order)
{ return kind ? (order >= 2 ? 30 : order ? 12 : 3) : (order >= 2 ? 19 : order ? 9 : 3);
}

#define S_ADD(a,b) ((a) + (b))
#define S_SUB(a,b) ((a) - (b))
#define S_MUL(a,b) ((a) * (b))
#define S_DIV(a,b) ((a) / (b))
#define S_SQRT(a) sqrt(a)
#define S_LT(m,a,b) int m = (a) < (b)
#define S_SEL(m,a,b) ((m) ? (a) : (b))

static void flux_kernel_scalar(const struct flux_group *g, int ncol, const REAL *const *coef,
                               const REAL *x, int order, REAL out[FLUX_COLS][FLUX_OUT])
{ REAL lane[FLUX_COLS][FLUX_OUT][FLUX_LANES];
  const REAL X0 = x[0], X1 = x[1], X2 = x[2], K_2 = 2.0, K_3 = 3.0, K_5 = 5.0;
  int t, l, c, k, nout = flux_nout(g->kind,order);
  (void)K_5;
  for ( l = 0 ; l < FLUX_LANES ; l++ )
  { REAL acc[FLUX_COLS][FLUX_OUT];
    memset(acc,0,sizeof(acc));
    for ( t = l ; t < g->np ; t += FLUX_LANES )
    {
#define S_LOAD(p) (p)[t]
      if ( g->kind )
        FLUX_POLE_TERM(REAL,S_ADD,S_SUB,S_MUL,S_DIV,S_SQRT,S_LT,S_SEL,0.0,1.0,S_LOAD)
      else
        FLUX_POINT_TERM(REAL,S_ADD,S_SUB,S_MUL,S_DIV,S_SQRT,S_LT,S_SEL,0.0,1.0,S_LOAD)
#undef S_LOAD
    }
    for ( c = 0 ; c < ncol ; c++ )
      for ( k = 0 ; k < nout ; k++ ) lane[c][k][l] = acc[c][k];
  }
  for ( c = 0 ; c < ncol ; c++ )
    for ( k = 0 ; k < nout ; k++ )
      out[c][k] = (lane[c][k][0] + lane[c][k][1]) + (lane[c][k][2] + lane[c][k][3]);
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(LONGDOUBLE) \
    && !defined(FLOAT128)
#include <immintrin.h>
#define FLUX_SIMD 1
#define V_LT(m,a,b) __m256d m = _mm256_cmp_pd(a,b,_CMP_LT_OQ)
#define V_SEL(m,a,b) _mm256_blendv_pd(b,a,m)
#define V_LOAD(p) _mm256_loadu_pd((p) + t)

#define FLUX_AVX2_BODY(TERM)                                                                 \
{ __m256d acc[FLUX_COLS][FLUX_OUT];                                                          \
  const __m256d X0 = _mm256_set1_pd(x[0]), X1 = _mm256_set1_pd(x[1]), X2 = _mm256_set1_pd(x[2]); \
  const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0),                       \
                K_2 = _mm256_set1_pd(2.0), K_3 = _mm256_set1_pd(3.0), K_5 = _mm256_set1_pd(5.0); \
  int t, c, k, nout = flux_nout(g->kind,order);                                              \
  (void)K_2; (void)K_3; (void)K_5;                                                           \
  for ( c = 0 ; c < ncol ; c++ )                                                             \
    for ( k = 0 ; k < nout ; k++ ) acc[c][k] = zero;                                         \
  for ( t = 0 ; t < g->np ; t += FLUX_LANES )                                                \
    TERM(__m256d,_mm256_add_pd,_mm256_sub_pd,_mm256_mul_pd,_mm256_div_pd,_mm256_sqrt_pd,     \
         V_LT,V_SEL,zero,one,V_LOAD)                                                         \
  for ( c = 0 ; c < ncol ; c++ )                                                             \
    for ( k = 0 ; k < nout ; k++ )                                                           \
    { REAL l[4];                                                                             \
      _mm256_storeu_pd(l,acc[c][k]);                                                         \
      out[c][k] = (l[0] + l[1]) + (l[2] + l[3]);                                             \
    }                                                                                        \
}

__attribute__((target("avx2"),always_inline))
static inline void flux_point_avx2(const struct flux_group *g, int ncol, const REAL *const *coef,
                                   const REAL *x, int order, REAL out[FLUX_COLS][FLUX_OUT])
FLUX_AVX2_BODY(FLUX_POINT_TERM)

__attribute__((target("avx2"),always_inline))
static inline void flux_pole_avx2(const struct flux_group *g, int ncol, const REAL *const *coef,
                                  const REAL *x, int order, REAL out[FLUX_COLS][FLUX_OUT])
FLUX_AVX2_BODY(FLUX_POLE_TERM)

/* (each kind, column count and order its own function: the loops over them
   unrolled, the accumulators in registers) */
typedef void (*flux_kernel_fn)(const struct flux_group *, const REAL *const *, const REAL *,
                               REAL [FLUX_COLS][FLUX_OUT]);
#define FLUX_AVX2(NAME, KERNEL, NCOL, ORDER)                                                \
  __attribute__((target("avx2")))                                                           \
  static void NAME(const struct flux_group *g, const REAL *const *coef, const REAL *x,      \
                   REAL out[FLUX_COLS][FLUX_OUT])                                           \
  { KERNEL(g,NCOL,coef,x,ORDER,out); }
FLUX_AVX2(flux_avx2_p10,flux_point_avx2,1,0)
FLUX_AVX2(flux_avx2_p11,flux_point_avx2,1,1)
FLUX_AVX2(flux_avx2_p12,flux_point_avx2,1,2)
FLUX_AVX2(flux_avx2_p20,flux_point_avx2,2,0)
FLUX_AVX2(flux_avx2_p21,flux_point_avx2,2,1)
FLUX_AVX2(flux_avx2_p22,flux_point_avx2,2,2)
FLUX_AVX2(flux_avx2_m10,flux_pole_avx2,1,0)
FLUX_AVX2(flux_avx2_m11,flux_pole_avx2,1,1)
FLUX_AVX2(flux_avx2_m12,flux_pole_avx2,1,2)
FLUX_AVX2(flux_avx2_m20,flux_pole_avx2,2,0)
FLUX_AVX2(flux_avx2_m21,flux_pole_avx2,2,1)
FLUX_AVX2(flux_avx2_m22,flux_pole_avx2,2,2)
static const flux_kernel_fn flux_avx2[2][2][3] =
  {{{flux_avx2_p10,flux_avx2_p11,flux_avx2_p12},{flux_avx2_p20,flux_avx2_p21,flux_avx2_p22}},
   {{flux_avx2_m10,flux_avx2_m11,flux_avx2_m12},{flux_avx2_m20,flux_avx2_m21,flux_avx2_m22}}};
#endif

static int flux_simd = -1;        /* (a benign race: every thread finds the same) */

static void flux_kernel(const struct flux_group *g, int ncol, const REAL *const *coef,
                        const REAL *x, int order, REAL out[FLUX_COLS][FLUX_OUT])
{
#ifdef FLUX_SIMD
  if ( flux_simd < 0 ) flux_simd = __builtin_cpu_supports("avx2") ? 1 : 0;
  if ( flux_simd )
  { flux_avx2[g->kind][ncol-1][order](g,coef,x,out);
    return;
  }
#endif
  flux_kernel_scalar(g,ncol,coef,x,order,out);
}

/* pyse_native_simd(0): the scalar kernels (tests compare the two) */
int pyse_native_simd(int on)
{
#ifdef FLUX_SIMD
  int was;
  native_serial++;
  if ( flux_simd < 0 ) flux_simd = __builtin_cpu_supports("avx2") ? 1 : 0;
  was = flux_simd;
  flux_simd = on && __builtin_cpu_supports("avx2") ? 1 : 0;
  return was;
#else
  (void)on;
  return 0;
#endif
}

/* A thread's last Gauss points' sums, every column of the group: the
   volume's and the wetting's integrands are taken at the same Gauss points
   one after the other (Evolver's element loop: each facet's methods in
   turn, an edge's energy and content), the second from here (a quarter of
   a 64-bead pack's sums in its gradient steps: the volume's projection
   takes the volume alone, at new points each pass). Keyed by the point's
   bits, the group, the order, the floor and the registration, in a slot
   hashed from the point (eight slots searched in turn, the comparisons'
   branches were a sixth of a 4-bead pack's step); a value at an order is
   the same, to the bit, as the one taken with a higher order. */
struct flux_cached
{ long gen;
  int group, order;
  REAL floor;
  unsigned long long x[3];
  REAL out[FLUX_COLS][FLUX_OUT];
};
struct flux_thread
{ struct flux_cached cache[FLUX_CACHE];
  REAL *coefbuf;                 /* the columns' coefficients, the floor raised */
  int coefcap, cos_ok;
  REAL cos_floor, cos_value;     /* cos(floor), the last floor's */
};
static __thread struct flux_thread flux_tls;   /* (one lookup a call: each is a call
                                                  to __tls_get_addr in a module) */

/* cos(floor), the thread's last */
static REAL flux_cos(struct flux_thread *T, REAL floor)
{ if ( !T->cos_ok || floor != T->cos_floor )
  { T->cos_value = cos(floor*M_PI/180.0);
    T->cos_floor = floor;
    T->cos_ok = 1;
  }
  return T->cos_value;
}

/* field f's terms at x (its group's column), to the order, into
   res[flux_nout()]; 0 if out of memory */
static int flux_group_terms(const struct flux_field *f, const REAL *x, int order, REAL floor,
                            REAL *res)
{ const struct flux_group *g = flux_groups + f->group;
  struct flux_thread *T = &flux_tls;
  const REAL *coef[FLUX_COLS];
  struct flux_cached *e;
  unsigned long long b[3], h;
  int c, t, k, nout = flux_nout(g->kind,order);
  memcpy(b,x,sizeof(b));
  h = b[0]*0x9E3779B97F4A7C15ULL ^ b[1]*0xC2B2AE3D27D4EB4FULL ^ b[2]*0x165667B19E3779F9ULL;
  e = T->cache + ((h ^ (h >> 29)) >> 32) % FLUX_CACHE;
  if ( !(e->gen == flux_gen && e->x[0] == b[0] && e->x[1] == b[1] && e->x[2] == b[2]
         && e->group == f->group && e->order >= order && e->floor == floor) )
  { if ( g->uniform )
    { coef[0] = g->one;
      flux_kernel(g,1,coef,x,order,e->out);
    }
    else
    { for ( c = 0 ; c < g->ncol ; c++ )
      { coef[c] = g->c0[c];
        if ( floor > g->amin[c] )
        { /* relax()'s staging: the angles under the floor raised to it */
          REAL cosf = flux_cos(T,floor);
          if ( T->coefcap < FLUX_COLS*g->np )
          { REAL *nb = (REAL*)realloc(T->coefbuf,(size_t)FLUX_COLS*g->np*sizeof(REAL));
            if ( !nb ) return 0;
            T->coefbuf = nb;
            T->coefcap = FLUX_COLS*g->np;
          }
          for ( t = 0 ; t < g->np ; t++ )
            T->coefbuf[c*g->np+t] = floor > g->athr[c][t] ? g->kc[c][t]*cosf : g->c0[c][t];
          coef[c] = T->coefbuf + c*g->np;
        }
      }
      flux_kernel(g,g->ncol,coef,x,order,e->out);
    }
    e->gen = flux_gen;
    e->group = f->group;
    e->order = order;
    e->floor = floor;
    for ( c = 0 ; c < 3 ; c++ ) e->x[c] = b[c];
  }
  if ( g->uniform )
  { /* (every term's coefficient, the floor raised past its angle or not) */
    c = f->col;
    REAL a = floor > g->athr[c][0] ? g->kc[c][0]*flux_cos(T,floor) : g->c0[c][0];
    for ( k = 0 ; k < nout ; k++ ) res[k] = a*e->out[0][k];
  }
  else
    for ( k = 0 ; k < nout ; k++ ) res[k] = e->out[f->col][k];
  return 1;
}

/* a field's own terms (base): values v, first derivatives g */
static void flux_base(const struct flux_field *f, const REAL *x, REAL floor, REAL v[3],
                      REAL g[3][3])
{ int i, m;
  for ( m = 0 ; m < 3 ; m++ )
  { v[m] = 0.0;
    for ( i = 0 ; i < 3 ; i++ ) g[m][i] = 0.0;
  }
  if ( f->base == 1 ) { v[2] = x[2]; g[2][2] = 1.0; }
  else if ( f->base == 2 )
  { REAL XX = x[0] - f->bc[0], YY = x[1] - f->bc[1], ZZ = x[2] - f->bc[2];
    v[0] = ZZ/3.0*YY;        g[0][1] = ZZ/3.0; g[0][2] = YY/3.0;
    v[1] = -(f->bc[2] + ZZ/3.0)*XX;  g[1][0] = -(f->bc[2] + ZZ/3.0); g[1][2] = -XX/3.0;
  }
  else if ( f->base == 3 || f->base == 4 )
  { /* a plane's: x = o + s u + t v; content u n_z (o_z t + u_z s t + v_z t^2/2),
       wetting k u t (constraints.plane) */
    const REAL *o = f->bc + (f->base == 3 ? 1 : 2), *u = o + 3, *w = o + 6;
    REAL ss = 0, tt = 0, p, dp[3];
    for ( i = 0 ; i < 3 ; i++ ) { ss += u[i]*(x[i] - o[i]); tt += w[i]*(x[i] - o[i]); }
    if ( f->base == 3 )
    { REAL nz = f->bc[0];
      p = nz*(o[2]*tt + u[2]*ss*tt + 0.5*w[2]*tt*tt);
      for ( i = 0 ; i < 3 ; i++ )
        dp[i] = nz*(o[2]*w[i] + u[2]*(tt*u[i] + ss*w[i]) + w[2]*tt*w[i]);
    }
    else
    { REAL k = plane_coefficient(f,floor);
      p = k*tt;
      for ( i = 0 ; i < 3 ; i++ ) dp[i] = k*w[i];
    }
    for ( m = 0 ; m < 3 ; m++ )
    { v[m] = u[m]*p;
      for ( i = 0 ; i < 3 ; i++ ) g[m][i] = u[m]*dp[i];
    }
  }
}

/* a field's own terms' second derivatives (base): h[m][i][j] */
static void flux_base_second(const struct flux_field *f, REAL h[3][3][3])
{ int i, j, m;
  memset(h,0,27*sizeof(REAL));
  if ( f->base == 2 )
  { /* (Z/3 Y, -(c_z + Z/3) X, 0) */
    h[0][1][2] = h[0][2][1] = 1.0/3.0;
    h[1][0][2] = h[1][2][0] = -1.0/3.0;
  }
  else if ( f->base == 3 )
  { /* u_m n_z (u_z s t + v_z t^2/2): u_m n_z (u_z (u_i w_j + w_i u_j) + v_z w_i w_j) */
    const REAL *o = f->bc + 1, *u = o + 3, *w = o + 6;
    REAL nz = f->bc[0];
    for ( m = 0 ; m < 3 ; m++ )
      for ( i = 0 ; i < 3 ; i++ )
        for ( j = 0 ; j < 3 ; j++ )
          h[m][i][j] = u[m]*nz*(u[2]*(u[i]*w[j] + w[i]*u[j]) + w[2]*w[i]*w[j]);
  }
}

static int flux_eval(struct expnode *key, REAL *x, int order, REAL *val,
                     REAL (*d)[MAXCOORD], REAL ***s)
{ struct flux_field *f = NULL;
  REAL floor, v[3], g[3][3];
  int i, j, m, t;
  if ( flux_load != pyse_load_serial() ) return 0;
  f = flux_find(key);
  if ( !f ) return 0;
  floor = flux_floor();
  flux_base(f,x,floor,v,g);
  if ( f->group >= 0 )
  { const struct flux_group *G = flux_groups + f->group;
    REAL o[FLUX_OUT];
    if ( !flux_group_terms(f,x,order,floor,o) ) return 0;
    for ( m = 0 ; m < 3 ; m++ ) v[m] += o[m];
    if ( order && !G->kind )
    { g[0][0] += o[3]; g[1][1] += o[4]; g[2][2] += o[5];
      g[0][1] += o[6]; g[1][0] += o[6];
      g[0][2] += o[7]; g[2][0] += o[7];
      g[1][2] += o[8]; g[2][1] += o[8];
    }
    else if ( order )
      for ( m = 0 ; m < 3 ; m++ )
        for ( i = 0 ; i < 3 ; i++ ) g[m][i] += o[3+3*m+i];
    for ( m = 0 ; m < 3 ; m++ )
    { val[m] = v[m];
      if ( order ) for ( i = 0 ; i < 3 ; i++ ) d[m][i] = g[m][i];
    }
    if ( order >= 2 )
    { /* (the ten of a point source's, the six of each monopole component's) */
      static const int sym[3][3][3] = {{{0,3,4},{3,5,9},{4,9,7}},
                                       {{3,5,9},{5,1,6},{9,6,8}},
                                       {{4,9,7},{9,6,8},{7,8,2}}};
      static const int pair[3][3] = {{0,3,4},{3,1,5},{4,5,2}};
      REAL h[3][3][3];
      flux_base_second(f,h);
      for ( m = 0 ; m < 3 ; m++ )
        for ( i = 0 ; i < 3 ; i++ )
          for ( j = 0 ; j < 3 ; j++ )
            s[m][i][j] = h[m][i][j] + (G->kind ? o[12 + 6*m + pair[i][j]] : o[9 + sym[m][i][j]]);
    }
    return 1;
  }
  if ( order < 2 )
  { /* term by term (a field of mixed terms; none of _flux's) */
    for ( t = 0 ; t < f->nterm ; t++ )
    { struct flux_term *u = f->term + t;
      REAL coef = flux_coefficient(u,floor), X[3], r2, r;
      int point = u->d[0] == 0 && u->d[1] == 0 && u->d[2] == 0;
      for ( m = 0 ; m < 3 ; m++ ) X[m] = x[m] - u->c[m];
      r2 = X[0]*X[0] + X[1]*X[1] + X[2]*X[2];
      r = sqrt(r2);
      if ( point && r2 < u->K2 )
      { REAL ir3 = coef/u->K3;             /* inside the cap: coef (x - c)/K^3 */
        for ( m = 0 ; m < 3 ; m++ )
        { v[m] += X[m]*ir3;
          if ( order ) g[m][m] += ir3;
        }
      }
      else if ( point )
      { REAL ir3 = coef/(r2*r), ir5 = 3.0*ir3/r2;
        for ( m = 0 ; m < 3 ; m++ )
        { v[m] += X[m]*ir3;
          if ( order ) for ( i = 0 ; i < 3 ; i++ ) g[m][i] += (m == i ? ir3 : 0.0) - ir5*X[m]*X[i];
        }
      }
      else
      { const REAL *dd = u->d;
        REAL dw = dd[0]*X[0] + dd[1]*X[1] + dd[2]*X[2];
        REAL D = r*(r - dw), ig = coef/D;
        REAL C[3] = { dd[1]*X[2] - dd[2]*X[1], dd[2]*X[0] - dd[0]*X[2], dd[0]*X[1] - dd[1]*X[0] };
        REAL dC[3][3] = {{0,-dd[2],dd[1]},{dd[2],0,-dd[0]},{-dd[1],dd[0],0}};
        for ( m = 0 ; m < 3 ; m++ ) v[m] += C[m]*ig;
        if ( order )
        { REAL dD[3];
          for ( i = 0 ; i < 3 ; i++ ) dD[i] = 2.0*X[i] - X[i]/r*dw - r*dd[i];
          for ( m = 0 ; m < 3 ; m++ )
            for ( i = 0 ; i < 3 ; i++ )
              g[m][i] += dC[m][i]*ig - C[m]*ig*dD[i]/D;
        }
      }
    }
    for ( m = 0 ; m < 3 ; m++ )
    { val[m] = v[m];
      if ( order ) for ( i = 0 ; i < 3 ; i++ ) d[m][i] = g[m][i];
    }
    return 1;
  }
  { /* second derivatives term by term, as dual numbers */
    dual out[3], X[3];
    REAL ex[3][3] = {{1,0,0},{0,1,0},{0,0,1}}, h[3][3][3];
    flux_base_second(f,h);
    for ( m = 0 ; m < 3 ; m++ )
    { out[m].v = v[m];
      for ( i = 0 ; i < 3 ; i++ )
      { out[m].g[i] = g[m][i];
        for ( j = 0 ; j < 3 ; j++ ) out[m].h[i][j] = h[m][i][j];
      }
    }
    for ( t = 0 ; t < f->nterm ; t++ )
    { struct flux_term *u = f->term + t;
      REAL coef = flux_coefficient(u,floor);
      dual r2;
      for ( m = 0 ; m < 3 ; m++ ) X[m] = d_lin(ex[m],-u->c[m],x);
      r2 = d_add(d_add(d_mul(X[0],X[0]),d_mul(X[1],X[1])),d_mul(X[2],X[2]));
      if ( u->d[0] == 0 && u->d[1] == 0 && u->d[2] == 0 )
      { dual inv = r2.v < u->K2 ? d_const(coef/u->K3)   /* inside the cap: (x - c)/K^3 */
                                : d_scale(d_pow(r2,-1.5),coef);   /* coef (x - c)/r^3 */
        for ( m = 0 ; m < 3 ; m++ ) out[m] = d_add(out[m],d_mul(X[m],inv));
      }
      else
      { dual r = d_pow(r2,0.5), dw = d_const(0.0), den, inv;
        for ( m = 0 ; m < 3 ; m++ ) dw = d_add(dw,d_scale(X[m],u->d[m]));
        den = d_mul(r,d_add(r,d_scale(dw,-1.0)));
        inv = d_scale(d_inv(den),coef);
        /* (d x w) */
        out[0] = d_add(out[0],d_mul(d_add(d_scale(X[2],u->d[1]),d_scale(X[1],-u->d[2])),inv));
        out[1] = d_add(out[1],d_mul(d_add(d_scale(X[0],u->d[2]),d_scale(X[2],-u->d[0])),inv));
        out[2] = d_add(out[2],d_mul(d_add(d_scale(X[1],u->d[0]),d_scale(X[0],-u->d[1])),inv));
      }
    }
    for ( m = 0 ; m < 3 ; m++ )
    { val[m] = out[m].v;
      for ( i = 0 ; i < 3 ; i++ )
      { d[m][i] = out[m].g[i];
        for ( j = 0 ; j < 3 ; j++ ) s[m][i][j] = out[m].h[i][j];
      }
    }
  }
  return 1;
}

/* ---- constraint formulas in closed form ----------------------------------
 *
 * A plane's formula n . x = d and a sphere's |x - c|^2 = R^2 (constraints.py,
 * and the guards' one-sided copies) evaluated without Evolver's interpreter:
 * Evolver evaluates every constraint at every vertex for its projection, and
 * pySE's gap query (every vertex, every wall and guard) did so ten thousand
 * times a call, a sixth of a 16-bead pack's main thread. The expression is
 * marked by PYSE_NATIVE_FORMULA in its flag (src/eval_all.c, evaltree.c test
 * it); registered only where the closed form matches the interpreter bit for
 * bit at test points, so the datafile's formula stays the definition. */

#define PYSE_NATIVE_FORMULA 0x40000000
#define FORMULA_NATIVES 512

struct formula_native
{ struct expnode *key;
  int kind;                      /* 0 plane: p = n, d; 1 sphere: p = c, R^2 */
  REAL p[4];
  REAL sign;                     /* the interpreter's value is sign times ours */
};

static struct formula_native formulas[FORMULA_NATIVES];
static int formula_count = 0;
static long formula_load = -1;

static void formula_value(const struct formula_native *n, const REAL *x, REAL *f, REAL *g)
{ /* in the interpreter's order (the datafile's A + B + C = D taken as
     ((A + B) + C) - D, a plane's zero terms left out), so as to agree with it
     bit for bit: a last bit's difference changed a relaxed drop's energy in
     its seventh digit and the edits after it */
  int i, first = 1;
  REAL v = 0.0;
  if ( n->kind == 0 )
  { for ( i = 0 ; i < 3 ; i++ )
      if ( n->p[i] != 0.0 )
      { REAL t = n->p[i]*x[i];
        v = first ? t : v + t;
        first = 0;
      }
    v = v - n->p[3];
    if ( g ) for ( i = 0 ; i < 3 ; i++ ) g[i] = n->p[i];
  }
  else
  { REAL X = x[0] - n->p[0], Y = x[1] - n->p[1], Z = x[2] - n->p[2];
    v = (X*X + Y*Y) + Z*Z - n->p[3];
    if ( g ) { g[0] = 2.0*X; g[1] = 2.0*Y; g[2] = 2.0*Z; }
  }
  if ( n->sign < 0 )
  { v = -v;
    if ( g ) for ( i = 0 ; i < 3 ; i++ ) g[i] = -g[i];
  }
  *f = v;
}

int pyse_native_formula(struct expnode *ex, REAL *x, REAL *f, REAL *grad)
{ int i;
  if ( formula_load != pyse_load_serial() ) return 0;
  for ( i = 0 ; i < formula_count ; i++ )
    if ( formulas[i].key == ex )
    { formula_value(formulas + i,x,f,grad);
      return 1;
    }
  return 0;
}

/* register constraint con's formula as a plane (kind 0: n, d) or a sphere
   (kind 1: c, R^2); 1 if registered (the closed form matching the
   interpreter, value and gradient, at test points), else 0 */
int pyse_native_formula_register(int con, int kind, const double *p)
{ struct constraint *cn;
  struct expnode *ex;
  struct formula_native n;
  REAL pts[8][3] = {{0.31,-0.27,0.73},{1.9,0.4,-0.6},{-0.8,2.3,1.1},{0.05,0.11,3.7},
                    {-1.37,-0.91,0.29},{0.7071,0.7071,0.0},{2.9,-3.1,-0.4},{0.013,-0.6,-2.2}};
  int i, k, j;
  native_serial++;
  if ( formula_load != pyse_load_serial() )
  { for ( i = 0 ; i < formula_count ; i++ )
      formulas[i].key = NULL;    /* (the old load's expressions are gone) */
    formula_count = 0;
    formula_load = pyse_load_serial();
  }
  if ( SDIM != 3 || (kind != 0 && kind != 1) || con < 0 || con >= web.maxcon ) return 0;
  cn = get_constraint(con);
  if ( !cn || !(cn->attr & IN_USE) || !cn->formula ) return 0;
  ex = cn->formula;
  n.key = ex;
  n.kind = kind;
  n.sign = 1.0;
  ex->flag &= ~PYSE_NATIVE_FORMULA;      /* (the interpreter for the test) */
  { /* the constants as Evolver parsed them, read back from its interpreter
       (exactly: a sphere's gradient at 0 is -2c, its value at c -R^2; a
       plane's gradient n, its value at 0 -d), not as given: its parser and
       Python's round some decimals apart, and the closed form then missed
       the interpreter in the last bit at most points */
    REAL zero[3] = {0,0,0}, f0, g0[2*MAXCOORD];
    eval_all(ex,zero,SDIM,&f0,g0,NULLID);
    if ( kind == 1 )
    { /* s (|x - c|^2 - R^2), s = +1 or -1: with s = 1, c = -g/2 and the
         value there -R^2 < 0; else s = -1, c = g/2, the value there R^2 */
      REAL c[3], fc, gc[2*MAXCOORD];
      for ( j = 0 ; j < 3 ; j++ ) c[j] = -g0[j]/2.0;
      eval_all(ex,c,SDIM,&fc,gc,NULLID);
      if ( !(fc < 0) )
      { n.sign = -1.0;
        for ( j = 0 ; j < 3 ; j++ ) c[j] = g0[j]/2.0;
        eval_all(ex,c,SDIM,&fc,gc,NULLID);
        fc = -fc;
      }
      for ( j = 0 ; j < 3 ; j++ ) n.p[j] = c[j];
      n.p[3] = -fc;
    }
    else
    { /* s (n . x - d): read back as s n and s d, the same thing */
      for ( j = 0 ; j < 3 ; j++ ) n.p[j] = g0[j];
      n.p[3] = -f0;
    }
    /* (and close to what pySE gave: else not the formula it meant) */
    for ( j = 0 ; j < 4 ; j++ )
      if ( fabs(fabs(n.p[j]) - fabs(p[j])) > 1e-9*(1.0 + fabs(p[j])) ) return 0;
  }
  for ( k = 0 ; k < 8 ; k++ )
  { REAL x[3], fi, gi[2*MAXCOORD], fn, gn[3];
    for ( j = 0 ; j < 3 ; j++ ) x[j] = (kind ? p[j] : 0.0) + pts[k][j];
    eval_all(ex,x,SDIM,&fi,gi,NULLID);
    formula_value(&n,x,&fn,gn);
    /* (bit for bit: else the interpreter's own arithmetic, a rotated
       frame's say, and this one's would part in the last bits) */
    if ( fi != fn ) return 0;
    for ( j = 0 ; j < 3 ; j++ )
      if ( gi[j] != gn[j] ) return 0;
  }
  for ( i = 0 ; i < formula_count ; i++ )
    if ( formulas[i].key == ex ) break;
  if ( i == formula_count )
  { if ( formula_count >= FORMULA_NATIVES ) return 0;
    formula_count++;
  }
  formulas[i] = n;
  ex->flag |= PYSE_NATIVE_FORMULA;
  return 1;
}
