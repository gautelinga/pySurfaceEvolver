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
/* the flux gauge in a periodic slab (pyse_slabflux.c): evaluated whether or
   not native evaluation is on (Evolver's interpreter can't sum the lattice) */
int pyse_slabflux_eval(struct expnode *key, REAL *x, int order, REAL *val, REAL (*d)[MAXCOORD],
                       REAL ***s);
int pyse_slabflux_active(void);

/* the integrand vector of the wall whose integrand key is (its first
   node), at x: values, and with order 1 or 2 their first (and second)
   derivatives; 0 where key isn't a registered wall's */
/* ---- walls z = b(x, y) by tables (constraints.heightfield) ---------------
 *
 * A curved wall's contact-line integrands have no closed form (its wetted
 * area is the integral of sqrt(1 + |grad b|^2)), so Python tabulates them:
 * the energy's and the content's x components (the integrands are P(x, y) dx
 * forms, zero along lines x = const and along the table's first row y = y0),
 * as tensor-product cubic B-splines over a uniform grid, natural at its
 * edges (constraints._spline_coefficients); past the grid each extends
 * linearly. Evaluated here always: the integrands in the datafile are
 * placeholders (pyse_native_enable(0) leaves these on). */

#define TABLE_MAX 64

struct native_table
{ struct expnode *energy, *content;   /* the constraint's integrands' first nodes */
  int nx, ny;                         /* grid nodes; (nx + 2)(ny + 2) coefficients */
  REAL x0, y0, hx, hy;
  REAL *e, *c;                        /* coefficients, x-major (NULL: no energy) */
};

static struct native_table tables[TABLE_MAX];
static int table_count = 0;
static long table_load = -1;

static void table_free(struct native_table *t)
{ free(t->e); free(t->c);
  t->e = t->c = NULL;
}

/* register constraint con's integrands as tables (E may be NULL); 0 if it
   can't be (no integrands, too many) */
int pyse_native_table(int con, int nx, int ny, double x0, double y0, double hx, double hy,
                      const double *E, const double *C)
{ struct constraint *cn;
  struct native_table *t;
  long n = (long)(nx + 2)*(ny + 2);
  int i;
  native_serial++;
  if ( table_load != pyse_load_serial() )
  { for ( i = 0 ; i < table_count ; i++ ) table_free(tables + i);
    table_count = 0;
    table_load = pyse_load_serial();
  }
  if ( con < 0 || con >= web.maxcon || nx < 2 || ny < 2 || hx <= 0 || hy <= 0 ) return 0;
  cn = get_constraint(con);
  if ( !cn || !cn->convect[0] || (E && !cn->envect[0]) ) return 0;
  for ( i = 0, t = NULL ; i < table_count ; i++ )
    if ( tables[i].content == cn->convect[0] ) { t = tables + i; table_free(t); }
  if ( !t )
  { if ( table_count >= TABLE_MAX ) return 0;
    t = tables + table_count++;
  }
  t->energy = E ? cn->envect[0] : NULL;
  t->content = cn->convect[0];
  t->nx = nx; t->ny = ny;
  t->x0 = x0; t->y0 = y0; t->hx = hx; t->hy = hy;
  t->c = (REAL*)malloc(n*sizeof(REAL));
  t->e = E ? (REAL*)malloc(n*sizeof(REAL)) : NULL;
  if ( !t->c || (E && !t->e) ) { table_free(t); table_count--; return 0; }
  for ( i = 0 ; i < n ; i++ ) t->c[i] = C[i];
  if ( E ) for ( i = 0 ; i < n ; i++ ) t->e[i] = E[i];
  return 1;
}

/* the cubic B-spline's four weights at t in [0, 1] and their first and
   second derivatives in t */
static void bspline_weights(REAL t, REAL *w, REAL *w1, REAL *w2)
{ REAL u = 1.0 - t, t2 = t*t, t3 = t2*t;
  w[0] = u*u*u/6.0;
  w[1] = (3.0*t3 - 6.0*t2 + 4.0)/6.0;
  w[2] = (-3.0*t3 + 3.0*t2 + 3.0*t + 1.0)/6.0;
  w[3] = t3/6.0;
  w1[0] = -u*u/2.0;
  w1[1] = (3.0*t2 - 4.0*t)/2.0;
  w1[2] = (-3.0*t2 + 2.0*t + 1.0)/2.0;
  w1[3] = t2/2.0;
  w2[0] = u;
  w2[1] = 3.0*t - 2.0;
  w2[2] = 1.0 - 3.0*t;
  w2[3] = t;
}

/* the cell and the place in it of u (grid units) among n nodes; *out the
   distance past the grid (0 inside) */
static int table_cell(REAL u, int n, REAL *t, REAL *out)
{ int i;
  *out = 0.0;
  /* (a NaN coordinate gives a NaN value, not a cell far off the table) */
  if ( u != u ) { *out = u; u = 0.0; }
  else if ( u < 0.0 ) { *out = u; u = 0.0; }
  else if ( u > n - 1 ) { *out = u - (n - 1); u = n - 1; }
  i = (int)floor(u);
  if ( i > n - 2 ) i = n - 2;
  *t = u - i;
  return i;
}

/* the table's value at x, y with its gradient and Hessian in x, y (past the
   grid: the edge's value and slope carried on linearly) */
static void table_value(const struct native_table *T, const REAL *coef, REAL x, REAL y,
                        REAL *f, REAL *fx, REAL *fy, REAL *fxx, REAL *fxy, REAL *fyy)
{ REAL tx, ty, ox, oy, wx[4], wx1[4], wx2[4], wy[4], wy1[4], wy2[4];
  int i = table_cell((x - T->x0)/T->hx,T->nx,&tx,&ox);
  int j = table_cell((y - T->y0)/T->hy,T->ny,&ty,&oy);
  int a, b, stride = T->ny + 2;
  REAL v = 0, vx = 0, vy = 0, vxx = 0, vxy = 0, vyy = 0;
  bspline_weights(tx,wx,wx1,wx2);
  bspline_weights(ty,wy,wy1,wy2);
  for ( a = 0 ; a < 4 ; a++ )
  { const REAL *row = coef + (long)(i + a)*stride + j;
    REAL r = 0, r1 = 0, r2 = 0;
    for ( b = 0 ; b < 4 ; b++ )
    { r += row[b]*wy[b]; r1 += row[b]*wy1[b]; r2 += row[b]*wy2[b]; }
    v += wx[a]*r; vx += wx1[a]*r; vy += wx[a]*r1;
    vxx += wx2[a]*r; vxy += wx1[a]*r1; vyy += wx[a]*r2;
  }
  /* (grid units to lengths; past the grid, linear in the distance) */
  vx /= T->hx; vy /= T->hy;
  vxx /= T->hx*T->hx; vxy /= T->hx*T->hy; vyy /= T->hy*T->hy;
  if ( ox != 0.0 ) { v += vx*ox*T->hx; vy += vxy*ox*T->hx; vxx = vxy = 0.0; }
  if ( oy != 0.0 ) { v += vy*oy*T->hy; vx += vxy*oy*T->hy; vyy = vxy = 0.0; }
  *f = v; *fx = vx; *fy = vy; *fxx = vxx; *fxy = vxy; *fyy = vyy;
}

static int table_eval(struct expnode *key, REAL *x, int order, REAL *val,
                      REAL (*d)[MAXCOORD], REAL ***s)
{ const struct native_table *T = NULL;
  const REAL *coef = NULL;
  REAL f, fx, fy, fxx, fxy, fyy;
  int i, j, m;
  if ( table_load != pyse_load_serial() ) return 0;
  for ( i = 0 ; i < table_count ; i++ )
  { if ( tables[i].energy == key ) { T = tables + i; coef = T->e; break; }
    if ( tables[i].content == key ) { T = tables + i; coef = T->c; break; }
  }
  if ( !T ) return 0;
  table_value(T,coef,x[0],x[1],&f,&fx,&fy,&fxx,&fxy,&fyy);
  for ( m = 0 ; m < 3 ; m++ )
  { val[m] = m == 0 ? f : 0.0;
    if ( order >= 1 )
      for ( i = 0 ; i < 3 ; i++ ) d[m][i] = (m == 0 && i < 2) ? (i == 0 ? fx : fy) : 0.0;
    if ( order >= 2 )
      for ( i = 0 ; i < 3 ; i++ )
        for ( j = 0 ; j < 3 ; j++ )
          s[m][i][j] = (m == 0 && i < 2 && j < 2)
                       ? (i == 0 ? (j == 0 ? fxx : fxy) : (j == 0 ? fxy : fyy)) : 0.0;
  }
  return 1;
}

/* ---- a flat plate with blocks on it (constraints.plate) -----------------
 *
 * The plate z = base, the liquid below it, carries blocks, each over the
 * footprint x0 <= x <= x1, y0 <= y <= y1: its top z = d(y) (d(y0) = base),
 * its end face y = y1, its sides x = x0, x1. Each face is a constraint of
 * its own; all share one energy and one content integrand, valid across
 * every edge, so a contact line adds up whichever faces it crosses
 * (constraints.plate gives the derivation). Per block, on its footprint
 * (A), with S the arclength along the profile from the plate's first row
 * y = ylo, dd the profile extended flat (base before y0, d(y1) after y1),
 * Psi = int (base - dd) dy, Dc = int dd dy, Gam = int G(y, dd) dy and
 * G = gy y z + gz z^2/2 the weight:
 *   energy  = (-k (S + z - dd + kappa Psi) + Gam + [y > y1] W(y, z),
 *              -k kappa (x - xc)(base - z), 0)
 *   content = (Dc + [y > y1] (y - y1)(z - d1), 0, 0)
 * with kappa = 2/(x1 - x0), xc the middle, W the weight's part past the end
 * face. Beside the blocks (B), in each stretch [a, b] of x between
 * footprints or mirror planes:
 *   energy  = (-k (y - ylo + z - base) + Gam_open + sum_e H_e'(x) mu_e(y),
 *              -k s(x)(base - z) + sum_e H_e(x) mu_e'(y), 0)
 *   content = (base (y - ylo) + sum_e H_e'(x) nu_e(y), sum_e H_e(x) nu_e'(y), 0)
 * the sum over the stretch's footprint edges e, H_e the cubic with
 * H(x_e) = 0, H'(x_e) = 1 and H = H' = 0 at the stretch's other end, mu_e
 * and nu_e the jumps of the footprint's dx parts across its edge on the
 * plate (a smooth step along the block, where no face crosses the edge),
 * s(x) the sides' orientation (-1 at a left edge, +1 at a right one); past
 * a block's end, its edges' terms carry on its dx parts' z dependence too
 * (which vanishes on the plate), so the forms join across the edge at any
 * z. The functions of y are cubic B-splines over each footprint's y range,
 * carried on in closed form outside it. */

#define PLATE_MAX 16
#define PLATE_KEYS 256
#define PLATE_FUNCS 7      /* S, dd, Psi, Dc, Gam, mu, nu */
#define PLATE_EPS 1e-9     /* a point this near a footprint is on it (its sides) */

struct plate_block
{ REAL x0, x1, y0, y1, xc, kappa;   /* the sides' dy part kappa (x - xc)(base - z) */
  int n;                   /* nodes over [y0, y1] */
  REAL h;
  REAL *coef;              /* PLATE_FUNCS x (n + 2) */
  REAL end[PLATE_FUNCS];   /* the functions at y1 */
};

struct plate_stretch { REAL a, b, sa, sb; int ea, eb; };   /* ea, eb: blocks or -1 */

struct plate
{ REAL base, ylo, k, gy, gz;
  int nblocks, nstretches;
  struct plate_block *blocks;
  struct plate_stretch *stretches;
};

static struct plate plates[PLATE_MAX];
static int plate_count = 0;
static long plate_load = -1;
static struct { struct expnode *key; int plate, content; } plate_keys[PLATE_KEYS];
static int plate_nkeys = 0;
/* the faces: constraint, plate, kind (FACE_OPEN ... below), block */
static struct { int con, plate, kind, block; } plate_faces[PLATE_KEYS];
static int plate_nfaces = 0;

static void plate_free(struct plate *P)
{ int i;
  for ( i = 0 ; i < P->nblocks ; i++ ) free(P->blocks[i].coef);
  free(P->blocks); free(P->stretches);
  P->blocks = NULL; P->stretches = NULL; P->nblocks = P->nstretches = 0;
}

/* function f of block B at y in [y0, y1]: value and three derivatives */
static void plate_spline(const struct plate_block *B, int f, REAL y, REAL *out)
{ REAL tt, o, w[4], w1[4], w2[4], w3[4] = {-1.0, 3.0, -3.0, 1.0};
  int i = table_cell((y - B->y0)/B->h,B->n,&tt,&o), a;
  const REAL *c = B->coef + (long)f*(B->n + 2) + i;
  bspline_weights(tt,w,w1,w2);
  out[0] = out[1] = out[2] = out[3] = 0.0;
  for ( a = 0 ; a < 4 ; a++ )
  { out[0] += c[a]*w[a]; out[1] += c[a]*w1[a]; out[2] += c[a]*w2[a]; out[3] += c[a]*w3[a]; }
  out[1] /= B->h; out[2] /= B->h*B->h; out[3] /= B->h*B->h*B->h;
}

/* register a plate: the constraints (its faces) cons[0..ncons-1], their
   kinds (0 the plate, 1 a top, 2 an end, 3 a left side, 4 a right one) and
   blocks (of), whose integrands are evaluated as the plate's;
   scalars base, ylo, k, gy, gz; per block x0, x1, y0, y1, n, xc, kappa
   (blocks, 7 each) and its PLATE_FUNCS x (n + 2) coefficients (in order,
   coef); per
   stretch a, b, sa, sb, ea, eb (6 each). 0 if it can't be. */
int pyse_native_plate(const int *cons, const int *kinds, const int *of, int ncons,
                      const double *scalars, const double *blocks, int nblocks,
                      const double *coef, long ncoef, const double *stretches, int nstretches)
{ struct plate *P;
  long used = 0;
  int i, j, nkeys = 0;
  native_serial++;
  if ( plate_load != pyse_load_serial() )
  { for ( i = 0 ; i < plate_count ; i++ ) plate_free(plates + i);
    plate_count = plate_nkeys = plate_nfaces = 0;
    plate_load = pyse_load_serial();
  }
  if ( plate_count >= PLATE_MAX || nblocks < 0 || nstretches < 1 ) return 0;
  for ( i = 0 ; i < ncons ; i++ )
  { struct constraint *cn;
    if ( cons[i] < 0 || cons[i] >= web.maxcon ) return 0;
    cn = get_constraint(cons[i]);
    if ( !cn || !cn->convect[0] ) return 0;
    nkeys += cn->envect[0] ? 2 : 1;
  }
  if ( plate_nkeys + nkeys > PLATE_KEYS || plate_nfaces + ncons > PLATE_KEYS ) return 0;
  P = plates + plate_count;
  P->base = scalars[0]; P->ylo = scalars[1]; P->k = scalars[2];
  P->gy = scalars[3]; P->gz = scalars[4];
  P->blocks = (struct plate_block*)calloc(nblocks > 0 ? nblocks : 1,sizeof(struct plate_block));
  P->stretches = (struct plate_stretch*)calloc(nstretches,sizeof(struct plate_stretch));
  P->nblocks = nblocks; P->nstretches = nstretches;
  if ( !P->blocks || !P->stretches ) { plate_free(P); return 0; }
  for ( i = 0 ; i < nblocks ; i++ )
  { struct plate_block *B = P->blocks + i;
    long size;
    B->x0 = blocks[7*i]; B->x1 = blocks[7*i + 1];
    B->y0 = blocks[7*i + 2]; B->y1 = blocks[7*i + 3];
    B->n = (int)blocks[7*i + 4];
    B->xc = blocks[7*i + 5]; B->kappa = blocks[7*i + 6];
    if ( B->n < 2 || !(B->y1 > B->y0) || !(B->x1 > B->x0) ) { plate_free(P); return 0; }
    B->h = (B->y1 - B->y0)/(B->n - 1);
    size = (long)PLATE_FUNCS*(B->n + 2);
    if ( used + size > ncoef ) { plate_free(P); return 0; }
    B->coef = (REAL*)malloc(size*sizeof(REAL));
    if ( !B->coef ) { plate_free(P); return 0; }
    for ( j = 0 ; j < size ; j++ ) B->coef[j] = coef[used + j];
    used += size;
    for ( j = 0 ; j < PLATE_FUNCS ; j++ )
    { REAL v[4];
      plate_spline(B,j,B->y1,v);
      B->end[j] = v[0];
    }
  }
  for ( i = 0 ; i < nstretches ; i++ )
  { struct plate_stretch *s = P->stretches + i;
    const double *r = stretches + 6*i;
    s->a = r[0]; s->b = r[1]; s->sa = r[2]; s->sb = r[3];
    s->ea = (int)r[4]; s->eb = (int)r[5];
    if ( !(s->b > s->a) || s->ea >= nblocks || s->eb >= nblocks ) { plate_free(P); return 0; }
  }
  for ( i = 0 ; i < ncons ; i++ )
  { struct constraint *cn = get_constraint(cons[i]);
    plate_faces[plate_nfaces].con = cons[i];
    plate_faces[plate_nfaces].plate = plate_count;
    plate_faces[plate_nfaces].kind = kinds[i];
    plate_faces[plate_nfaces++].block = of[i];
    plate_keys[plate_nkeys].key = cn->convect[0];
    plate_keys[plate_nkeys].plate = plate_count;
    plate_keys[plate_nkeys++].content = 1;
    if ( cn->envect[0] )
    { plate_keys[plate_nkeys].key = cn->envect[0];
      plate_keys[plate_nkeys].plate = plate_count;
      plate_keys[plate_nkeys++].content = 0;
    }
  }
  plate_count++;
  return 1;
}

/* a function of y alone as a dual (value, d/dy, d2/dy2) */
static dual d_of_y(REAL v, REAL v1, REAL v2)
{ dual r = d_const(v);
  r.g[1] = v1;
  r.h[1][1] = v2;
  return r;
}

/* block B's function f at y as a dual (and, with deriv, its derivative's),
   carried on outside [y0, y1] as the plate's integrands need (see above) */
static dual plate_fn(const struct plate *P, const struct plate_block *B, int f, REAL y,
                     int deriv)
{ REAL v[4], d1 = B->end[1];
  if ( y >= B->y0 && y <= B->y1 )
  { plate_spline(B,f,y,v);
    return deriv ? d_of_y(v[1],v[2],v[3]) : d_of_y(v[0],v[1],v[2]);
  }
  if ( y < B->y0 )
    switch ( f )
    { case 0: v[0] = y - P->ylo; v[1] = 1.0; v[2] = 0.0; break;            /* S */
      case 1: v[0] = P->base; v[1] = v[2] = 0.0; break;                    /* dd */
      case 3: v[0] = P->base*(y - P->ylo); v[1] = P->base; v[2] = 0.0; break;
      case 4:                                                               /* Gam_open */
        v[0] = P->gy*P->base*(y*y - P->ylo*P->ylo)/2 + P->gz*P->base*P->base/2*(y - P->ylo);
        v[1] = P->gy*P->base*y + P->gz*P->base*P->base/2;
        v[2] = P->gy*P->base;
        break;
      default: v[0] = v[1] = v[2] = 0.0;                                   /* Psi, mu, nu */
    }
  else
  { REAL t = y - B->y1;
    switch ( f )
    { case 0: v[0] = B->end[0] + t; v[1] = 1.0; v[2] = 0.0; break;
      case 1: v[0] = d1; v[1] = v[2] = 0.0; break;
      case 3: v[0] = B->end[3] + d1*t; v[1] = d1; v[2] = 0.0; break;
      case 4:
        v[0] = B->end[4] + P->gy*d1*(y*y - B->y1*B->y1)/2 + P->gz*d1*d1/2*t;
        v[1] = P->gy*d1*y + P->gz*d1*d1/2;
        v[2] = P->gy*d1;
        break;
      default: v[0] = B->end[f]; v[1] = v[2] = 0.0;
    }
  }
  if ( deriv ) return d_of_y(v[1],v[2],0.0);
  return d_of_y(v[0],v[1],v[2]);
}

static int plate_eval(struct expnode *key, REAL *x, int order, REAL *val,
                      REAL (*d)[MAXCOORD], REAL ***s)
{ const struct plate *P = NULL;
  const struct plate_block *B = NULL;
  REAL ex[3] = {1,0,0}, ey[3] = {0,1,0}, ez[3] = {0,0,1};
  dual X, Y, Z, out[3];
  int i, j, m, content = 0;
  if ( plate_load != pyse_load_serial() ) return 0;
  for ( i = 0 ; i < plate_nkeys ; i++ )
    if ( plate_keys[i].key == key )
    { P = plates + plate_keys[i].plate; content = plate_keys[i].content; break; }
  if ( !P ) return 0;
  X = d_lin(ex,0.0,x); Y = d_lin(ey,0.0,x); Z = d_lin(ez,0.0,x);
  out[2] = d_const(0.0);
  for ( i = 0 ; i < P->nblocks ; i++ )
    if ( x[0] >= P->blocks[i].x0 - PLATE_EPS && x[0] <= P->blocks[i].x1 + PLATE_EPS )
    { B = P->blocks + i; break; }
  if ( B )
  { REAL kappa = B->kappa, xc = B->xc, d1 = B->end[1];
    int past = x[1] > B->y1;
    if ( content )
    { out[0] = plate_fn(P,B,3,x[1],0);
      if ( past )
        out[0] = d_add(out[0],d_mul(d_add(Y,d_const(-B->y1)),d_add(Z,d_const(-d1))));
      out[1] = d_const(0.0);
    }
    else
    { /* -k (S + z - dd + kappa Psi) + Gam + [past] W */
      dual a = d_add(d_add(plate_fn(P,B,0,x[1],0),Z),
                     d_add(d_scale(plate_fn(P,B,1,x[1],0),-1.0),
                           d_scale(plate_fn(P,B,2,x[1],0),kappa)));
      out[0] = d_add(d_scale(a,-P->k),plate_fn(P,B,4,x[1],0));
      if ( past )
      { /* gy (y^2 - y1^2)/2 (z - d1) + gz (y - y1)(z^2 - d1^2)/2 */
        dual w = d_mul(d_scale(d_add(d_mul(Y,Y),d_const(-B->y1*B->y1)),P->gy/2),
                       d_add(Z,d_const(-d1)));
        w = d_add(w,d_mul(d_scale(d_add(Y,d_const(-B->y1)),P->gz/2),
                          d_add(d_mul(Z,Z),d_const(-d1*d1))));
        out[0] = d_add(out[0],w);
      }
      out[1] = d_scale(d_mul(d_add(X,d_const(-xc)),d_add(d_const(P->base),d_scale(Z,-1.0))),
                       -P->k*kappa);
    }
  }
  else
  { const struct plate_stretch *st = NULL;
    for ( i = 0 ; i < P->nstretches ; i++ )
      if ( x[0] >= P->stretches[i].a - PLATE_EPS && x[0] <= P->stretches[i].b + PLATE_EPS )
      { st = P->stretches + i; break; }
    if ( !st ) st = P->stretches + (x[0] < P->stretches[0].a ? 0 : P->nstretches - 1);
    if ( content )
    { out[0] = d_scale(d_add(Y,d_const(-P->ylo)),P->base);
      out[1] = d_const(0.0);
    }
    else
    { REAL L = st->b - st->a;
      dual sx = d_add(d_const(st->sa),d_scale(d_add(X,d_const(-st->a)),(st->sb - st->sa)/L));
      out[0] = d_add(d_scale(d_add(d_add(Y,d_const(-P->ylo)),d_add(Z,d_const(-P->base))),-P->k),
                     d_of_y(P->gy*P->base*(x[1]*x[1] - P->ylo*P->ylo)/2
                              + P->gz*P->base*P->base/2*(x[1] - P->ylo),
                            P->gy*P->base*x[1] + P->gz*P->base*P->base/2, P->gy*P->base));
      out[1] = d_scale(d_mul(sx,d_add(d_const(P->base),d_scale(Z,-1.0))),-P->k);
    }
    for ( j = 0 ; j < 2 ; j++ )
    { int e = j ? st->eb : st->ea;
      REAL xe = j ? st->b : st->a, L = st->b - st->a;
      dual t, one_t, H, H1, f, f1;
      if ( e < 0 ) continue;
      /* t from the edge into the stretch; H = (x - xe)(1 - t)^2,
         H' = (1 - t)(1 - 3t) (d/dx, either side) */
      t = d_scale(d_add(X,d_const(-xe)),j ? -1.0/L : 1.0/L);
      one_t = d_add(d_const(1.0),d_scale(t,-1.0));
      H = d_mul(d_add(X,d_const(-xe)),d_mul(one_t,one_t));
      H1 = d_mul(one_t,d_add(d_const(1.0),d_scale(t,-3.0)));
      f = plate_fn(P,P->blocks + e,content ? 6 : 5,x[1],0);
      f1 = plate_fn(P,P->blocks + e,content ? 6 : 5,x[1],1);
      if ( x[1] > P->blocks[e].y1 )
      { /* past the end: the footprint's z terms less theirs on the plate,
           (y - y1)(z - base) for the content, the weight's W(y, z) - W(y, base),
           with their y derivatives */
        REAL y1 = P->blocks[e].y1;
        dual zb = d_add(Z,d_const(-P->base));
        if ( content )
        { f = d_add(f,d_mul(d_add(Y,d_const(-y1)),zb));
          f1 = d_add(f1,zb);
        }
        else
        { dual zz = d_add(d_mul(Z,Z),d_const(-P->base*P->base));
          f = d_add(f,d_add(d_mul(d_scale(d_add(d_mul(Y,Y),d_const(-y1*y1)),P->gy/2),zb),
                            d_mul(d_scale(d_add(Y,d_const(-y1)),P->gz/2),zz)));
          f1 = d_add(f1,d_add(d_scale(d_mul(Y,zb),P->gy),d_scale(zz,P->gz/2)));
        }
      }
      out[0] = d_add(out[0],d_mul(H1,f));
      out[1] = d_add(out[1],d_mul(H,f1));
    }
  }
  for ( m = 0 ; m < 3 ; m++ )
  { val[m] = out[m].v;
    if ( order >= 1 ) for ( i = 0 ; i < 3 ; i++ ) d[m][i] = out[m].g[i];
    if ( order >= 2 )
      for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ ) s[m][i][j] = out[m].h[i][j];
  }
  return 1;
}

int pyse_native_eval(struct expnode *key, REAL *x, int order, REAL *val,
                     REAL (*d)[MAXCOORD], REAL ***s)
{ struct native *n = NULL;
  int i, j, m, content = 0;
  dual a, b, h, D, rr, lead, area[3];
  REAL off;
  if ( key && SDIM == 3 && pyse_slabflux_active() && pyse_slabflux_eval(key,x,order,val,d,s) )
    return 1;
  if ( key && SDIM == 3 && table_count && table_eval(key,x,order,val,d,s) )
    return 1;   /* (whether native_on or not: the datafile's are placeholders) */
  if ( key && SDIM == 3 && plate_count && plate_eval(key,x,order,val,d,s) )
    return 1;   /* (the same) */
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
  int base;                      /* 0 none, 1 (0,0,z - z_f), 2 a sphere's content part,
                                    3 a plane's content, 4 a plane's wetting */
  REAL bc[12];                   /* 1: z_f; 2: the sphere's centre; 3: n_z, o, u, v;
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
  pyse_slabflux_simd(on);
  return was;
#else
  pyse_slabflux_simd(on);
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
  int cols, used;                /* the columns taken, those asked for */
  REAL out[FLUX_COLS][FLUX_OUT];
};

/* A facet's Gauss points' sums (pyse_native_points): each facet's methods
   in turn ask for the same points, the volume's and the wetting's, and the
   points hashed one by one into the cache above collided (twelve of a
   facet in sixteen slots) and cost a hash, a slot's check and a TLS lookup
   each: a fifth of a 4-bead pack's gradient step with the kernels' setup.
   The last facet's, all points at once. */
#define FLUX_PTS 37               /* Gauss points a facet at most (Evolver's 2D rules) */
struct flux_facet
{ long gen;
  int group, order, n;
  REAL floor;
  int cols, used;
  unsigned long long b[FLUX_PTS][3];
  REAL out[FLUX_PTS][FLUX_COLS][FLUX_OUT];
};

struct flux_thread
{ struct flux_cached cache[FLUX_CACHE];
  struct flux_facet facet;
  REAL *coefbuf;                 /* the columns' coefficients, the floor raised */
  int coefcap, cos_ok;
  REAL cos_floor, cos_value;     /* cos(floor), the last floor's */
  int single, waste;             /* columns taken one at a time: see flux_columns() */
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

/* column c's coefficients, the floor raised past the angles under it
   (relax()'s staging); NULL if out of memory */
static const REAL *flux_coef(struct flux_thread *T, const struct flux_group *g, int c, REAL floor)
{ int t;
  REAL cosf;
  if ( !(floor > g->amin[c]) ) return g->c0[c];
  cosf = flux_cos(T,floor);
  if ( T->coefcap < FLUX_COLS*g->np )
  { REAL *nb = (REAL*)realloc(T->coefbuf,(size_t)FLUX_COLS*g->np*sizeof(REAL));
    if ( !nb ) return NULL;
    T->coefbuf = nb;
    T->coefcap = FLUX_COLS*g->np;
  }
  for ( t = 0 ; t < g->np ; t++ )
    T->coefbuf[c*g->np+t] = floor > g->athr[c][t] ? g->kc[c][t]*cosf : g->c0[c][t];
  return T->coefbuf + c*g->np;
}

/* A group of several columns (spheres of more than one radius or angle: a
   polydisperse pack) has them all summed at once where both are asked for
   at the same points, the energy's and the volume's (each term's distance
   and powers shared); but the volume's projection asks for the volume's
   alone, pass after pass, and the other column was a third of the kernels'
   time there. So: all columns while each entry's are all asked for, one at
   a time after four entries in a row (point or facet) had only one asked
   for, and all again once another is asked for at a point taken. Each
   column's sums are the same, to the bit, either way. */
static int flux_columns(struct flux_thread *T, const struct flux_group *g, int cols, int used,
                        int need)
{ int all = (1 << g->ncol) - 1;
  if ( cols && cols != used )
  { if ( used == (used & -used) && ++T->waste >= 4 ) T->single = 1; }
  else if ( cols ) T->waste = 0;
  return T->single ? need : all;
}

/* columns cols of group g at x, to the order, into out (a uniform group: its one
   sum); 0 if out of memory */
static int flux_sum(struct flux_thread *T, const struct flux_group *g, int cols, const REAL *x,
                    int order, REAL floor, REAL out[FLUX_COLS][FLUX_OUT])
{ const REAL *coef[FLUX_COLS];
  int c, k;
  if ( g->uniform )
  { /* (the sum with coefficients 1: each column is it times its own) */
    coef[0] = g->one;
    flux_kernel(g,1,coef,x,order,out);
    return 1;
  }
  if ( cols == (1 << g->ncol) - 1 )
  { for ( c = 0 ; c < g->ncol ; c++ )
      if ( !(coef[c] = flux_coef(T,g,c,floor)) ) return 0;
    flux_kernel(g,g->ncol,coef,x,order,out);
    return 1;
  }
  for ( c = 0 ; c < g->ncol ; c++ )
    if ( cols & (1 << c) )
    { REAL one[FLUX_COLS][FLUX_OUT];
      if ( !(coef[0] = flux_coef(T,g,c,floor)) ) return 0;
      flux_kernel(g,1,coef,x,order,one);
      for ( k = 0 ; k < flux_nout(g->kind,order) ; k++ ) out[c][k] = one[0][k];
    }
  return 1;
}

/* field f's share of its group's sums out, into res[flux_nout()] */
static void flux_share(struct flux_thread *T, const struct flux_field *f,
                       const struct flux_group *g, REAL floor, int nout,
                       REAL out[FLUX_COLS][FLUX_OUT], REAL *res)
{ int c = f->col, k;
  if ( g->uniform )
  { /* (every term's coefficient, the floor raised past its angle or not) */
    REAL a = floor > g->athr[c][0] ? g->kc[c][0]*flux_cos(T,floor) : g->c0[c][0];
    for ( k = 0 ; k < nout ; k++ ) res[k] = a*out[0][k];
  }
  else
    for ( k = 0 ; k < nout ; k++ ) res[k] = out[c][k];
}

/* field f's terms at x (its group's column), to the order, into
   res[flux_nout()]; 0 if out of memory */
static int flux_group_terms(const struct flux_field *f, const REAL *x, int order, REAL floor,
                            REAL *res)
{ const struct flux_group *g = flux_groups + f->group;
  struct flux_thread *T = &flux_tls;
  struct flux_cached *e;
  unsigned long long b[3], h;
  int c, need = g->uniform ? 1 : 1 << f->col;
  memcpy(b,x,sizeof(b));
  h = b[0]*0x9E3779B97F4A7C15ULL ^ b[1]*0xC2B2AE3D27D4EB4FULL ^ b[2]*0x165667B19E3779F9ULL;
  e = T->cache + ((h ^ (h >> 29)) >> 32) % FLUX_CACHE;
  if ( e->gen == flux_gen && e->x[0] == b[0] && e->x[1] == b[1] && e->x[2] == b[2]
       && e->group == f->group && e->order >= order && e->floor == floor )
  { if ( !(e->cols & need) )
    { /* (another column at a point taken: both asked for) */
      if ( !flux_sum(T,g,need,x,e->order,floor,e->out) ) return 0;
      e->cols |= need;
      T->single = T->waste = 0;
    }
  }
  else
  { int cols = g->uniform ? 1 : flux_columns(T,g,e->gen == flux_gen ? e->cols : 0,e->used,need);
    if ( !flux_sum(T,g,cols,x,order,floor,e->out) ) return 0;
    e->gen = flux_gen;
    e->group = f->group;
    e->order = order;
    e->floor = floor;
    e->cols = cols;
    e->used = 0;
    for ( c = 0 ; c < 3 ; c++ ) e->x[c] = b[c];
  }
  e->used |= need;
  flux_share(T,f,g,floor,flux_nout(g->kind,order),e->out,res);
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
  if ( f->base == 1 ) { v[2] = x[2] - f->bc[0]; g[2][2] = 1.0; }   /* (bc[0]: a floor's z_f) */
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

/* a grouped field's integrand at x from its group's share o
   (flux_group_terms): values V, with order 1 or 2 derivatives D, S */
static void flux_assemble(const struct flux_field *f, const REAL *x, int order, REAL floor,
                          const REAL *o, REAL V[3], REAL D[3][3], REAL S[3][3][3])
{ const struct flux_group *G = flux_groups + f->group;
  REAL v[3], g[3][3];
  int i, j, m;
  flux_base(f,x,floor,v,g);
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
  { V[m] = v[m];
    if ( order ) for ( i = 0 ; i < 3 ; i++ ) D[m][i] = g[m][i];
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
          S[m][i][j] = h[m][i][j] + (G->kind ? o[12 + 6*m + pair[i][j]] : o[9 + sym[m][i][j]]);
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
  if ( f->group >= 0 )
  { REAL o[FLUX_OUT], V[3], D[3][3], S[3][3][3];
    if ( !flux_group_terms(f,x,order,floor,o) ) return 0;
    flux_assemble(f,x,order,floor,o,V,D,S);
    for ( m = 0 ; m < 3 ; m++ )
    { val[m] = V[m];
      if ( order ) for ( i = 0 ; i < 3 ; i++ ) d[m][i] = D[m][i];
      if ( order >= 2 )
        for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ ) s[m][i][j] = S[m][i][j];
    }
    return 1;
  }
  flux_base(f,x,floor,v,g);
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

/* The integrand vectors of the flux field whose integrand key is at the n
   points x[m] (a facet's or an edge's Gauss points): V[m], with order 1 or
   2 D[m], S[m], each the same to the bit as pyse_native_eval's at the point;
   0 (nothing done) where key isn't a grouped flux field's or n is over
   FLUX_PTS: the caller then asks point by point. */
int pyse_native_points(struct expnode *key, REAL **x, int n, int order, REAL (*V)[3],
                       REAL (*D)[3][3], REAL (*S)[3][3][3])
{ struct flux_field *f;
  const struct flux_group *g;
  struct flux_thread *T;
  struct flux_facet *e;
  REAL floor, o[FLUX_OUT];
  int m, need, nout;
  if ( !native_on || !key || SDIM != 3 || n > FLUX_PTS || n < 1 ) return 0;
  if ( flux_load != pyse_load_serial() ) return 0;
  f = flux_find(key);
  if ( !f || f->group < 0 ) return 0;
  floor = flux_floor();
  g = flux_groups + f->group;
  T = &flux_tls;
  e = &T->facet;
  need = g->uniform ? 1 : 1 << f->col;
  nout = flux_nout(g->kind,order);
  if ( e->gen == flux_gen && e->group == f->group && e->n == n && e->order >= order
       && e->floor == floor && !memcmp(e->b[0],x[0],sizeof(e->b[0])) )
  { for ( m = 1 ; m < n && !memcmp(e->b[m],x[m],sizeof(e->b[m])) ; m++ ) ;
    if ( m < n ) e->gen = -1;
  }
  else e->gen = -1;
  if ( e->gen != -1 )
  { if ( !(e->cols & need) )
    { /* (another column at the points taken: both asked for) */
      for ( m = 0 ; m < n ; m++ )
        if ( !flux_sum(T,g,need,x[m],e->order,floor,e->out[m]) ) return 0;
      e->cols |= need;
      T->single = T->waste = 0;
    }
  }
  else
  { int cols = g->uniform ? 1 : flux_columns(T,g,e->cols,e->used,need);
    for ( m = 0 ; m < n ; m++ )
    { if ( !flux_sum(T,g,cols,x[m],order,floor,e->out[m]) ) return 0;
      memcpy(e->b[m],x[m],sizeof(e->b[m]));
    }
    e->gen = flux_gen;
    e->group = f->group;
    e->order = order;
    e->n = n;
    e->floor = floor;
    e->cols = cols;
    e->used = 0;
  }
  e->used |= need;
  for ( m = 0 ; m < n ; m++ )
  { flux_share(T,f,g,floor,nout,e->out[m],o);
    flux_assemble(f,x[m],order,floor,o,V[m],order ? D[m] : NULL,order >= 2 ? S[m] : NULL);
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
  int kind;                      /* 0 plane: p = n, d; 1 sphere: p = c, R^2; 2 a
                                    slab's sphere at the nearest image in x, y
                                    (a guard: _surgery._guard_formula), L its periods */
  REAL p[4];
  REAL L[2];
  REAL sign;                     /* the interpreter's value is sign times ours */
  REAL *P;                       /* kind 3: spheres (c, R^2) by fours, the least of */
  int np;                        /* kind 2's terms over them (a guard against any) */
  int *cand, *cstart;            /* kind 3: by a grid's cells, the spheres that can be */
  int nc[3];                     /* the least there (formula_grid), cstart[c] to */
  REAL lo[3], h[3];              /* cstart[c+1] in cand; the grid x, y over a period */
};

static struct formula_native formulas[FORMULA_NATIVES];
static int formula_count = 0;
static long formula_load = -1;

/* the slot of a formula's expression, by a hash of its address (a scan of the
   ~130 of a 64-bead slab, its walls and guards, at every vertex's projection
   was 3% of a gradient step): slot + 1, 0 empty; read by the threads, written
   only at registration */
#define FORMULA_HASH 1024           /* (a power of two, twice FORMULA_NATIVES) */
static int formula_hash[FORMULA_HASH];

static unsigned formula_bucket(const struct expnode *ex)
{ return (unsigned)((((unsigned long long)(size_t)ex >> 4)*0x9E3779B97F4A7C15ULL) >> 54)
         & (FORMULA_HASH - 1);
}

static int formula_find(const struct expnode *ex)
{ unsigned h = formula_bucket(ex);
  int k;
  while ( (k = formula_hash[h]) != 0 )
  { if ( formulas[k - 1].key == ex ) return k - 1;
    h = (h + 1) & (FORMULA_HASH - 1);
  }
  return -1;
}

static void formula_free(struct formula_native *n)
{ free(n->P); free(n->cand); free(n->cstart);
  n->P = NULL; n->cand = n->cstart = NULL;
}

/* 1D: the least and the most distance from [u0, u1] (in [0, L)) to c's
   images (c in [0, L)), widened by e */
static void circle_reach(REAL u0, REAL u1, REAL c, REAL L, REAL e, REAL *lo, REAL *hi)
{ REAL d0 = fabs(u0 - c), d1 = fabs(u1 - c), far = c + L/2;
  d0 = d0 < L - d0 ? d0 : L - d0;
  d1 = d1 < L - d1 ? d1 : L - d1;
  if ( far >= L ) far -= L;
  *lo = (c >= u0 && c <= u1) ? 0.0 : (d0 < d1 ? d0 : d1);
  *hi = (far >= u0 && far <= u1) ? L/2 : (d0 > d1 ? d0 : d1);
  *lo = *lo > e ? *lo - e : 0.0;
  *hi += e;
}

/* kind 3's grid: x, y over a period [0, L), z over the spheres' heights and
   two radii past, cells of half the largest radius; each cell's spheres are
   those whose term's least there is no more than the least of the terms'
   most (the bounds widened by a millionth of the period: the terms are
   computed from x, the cells from x mod L). The least term at a point is
   then among its cell's, the same one, and the loop over a 64-bead pack's
   all at every guarded vertex's projection was 13% of a gradient step. 0 if
   out of memory (the loop over all then) */
static int formula_grid(struct formula_native *n)
{ int j, k, c, ncell, count = 0, cap;
  REAL rmax = 0.0, zmin = 0.0, zmax = 0.0, e = 1e-6*(n->L[0] + n->L[1]);
  REAL *lb, *ub;
  for ( k = 0 ; k < n->np ; k++ )
  { REAL r = sqrt(fabs(n->P[4*k + 3])), z = n->P[4*k + 2];
    if ( r > rmax ) rmax = r;
    if ( k == 0 || z < zmin ) zmin = z;
    if ( k == 0 || z > zmax ) zmax = z;
  }
  if ( !(rmax > 0) ) return 0;
  n->lo[0] = n->lo[1] = 0.0;
  n->lo[2] = zmin - 2*rmax;
  for ( j = 0 ; j < 3 ; j++ )
  { REAL span = j < 2 ? n->L[j] : (zmax + 2*rmax) - n->lo[2];
    n->nc[j] = (int)ceil(span/(rmax/2));
    if ( n->nc[j] < 1 ) n->nc[j] = 1;
    if ( n->nc[j] > 64 ) n->nc[j] = 64;
    n->h[j] = span/n->nc[j];
  }
  ncell = n->nc[0]*n->nc[1]*n->nc[2];
  lb = (REAL*)malloc(2*n->np*sizeof(REAL));
  n->cstart = (int*)malloc((ncell + 1)*sizeof(int));
  cap = 8*ncell;
  n->cand = (int*)malloc(cap*sizeof(int));
  if ( !lb || !n->cstart || !n->cand )
  { free(lb); free(n->cstart); free(n->cand); n->cstart = n->cand = NULL; return 0; }
  ub = lb + n->np;
  for ( c = 0 ; c < ncell ; c++ )
  { int ix = c % n->nc[0], iy = (c/n->nc[0]) % n->nc[1], iz = c/(n->nc[0]*n->nc[1]);
    REAL least = 0.0;
    for ( k = 0 ; k < n->np ; k++ )
    { const REAL *q = n->P + 4*k;
      REAL dl[3], dh[3];
      for ( j = 0 ; j < 2 ; j++ )
      { int i_ = j ? iy : ix;
        REAL cc = q[j] - n->L[j]*floor(q[j]/n->L[j]);
        circle_reach(i_*n->h[j],(i_ + 1)*n->h[j],cc,n->L[j],e,dl + j,dh + j);
      }
      { REAL z0 = n->lo[2] + iz*n->h[2], z1 = z0 + n->h[2];
        dl[2] = q[2] < z0 ? z0 - q[2] : q[2] > z1 ? q[2] - z1 : 0.0;
        dh[2] = fabs(z0 - q[2]) > fabs(z1 - q[2]) ? fabs(z0 - q[2]) : fabs(z1 - q[2]);
        dl[2] = dl[2] > e ? dl[2] - e : 0.0;
        dh[2] += e;
      }
      lb[k] = dl[0]*dl[0] + dl[1]*dl[1] + dl[2]*dl[2] - q[3];
      ub[k] = dh[0]*dh[0] + dh[1]*dh[1] + dh[2]*dh[2] - q[3];
      if ( k == 0 || ub[k] < least ) least = ub[k];
    }
    n->cstart[c] = count;
    for ( k = 0 ; k < n->np ; k++ )
      if ( lb[k] <= least + e*e )
      { if ( count == cap )
        { int *more = (int*)realloc(n->cand,2*cap*sizeof(int));
          if ( !more )
          { free(lb); free(n->cstart); free(n->cand); n->cstart = n->cand = NULL; return 0; }
          n->cand = more; cap *= 2;
        }
        n->cand[count++] = k;
      }
  }
  n->cstart[ncell] = count;
  free(lb);
  return 1;
}

static void formula_value(const struct formula_native *n, const REAL *x, REAL *f, REAL *g);

/* whether kind 3's grid gives what the loop over all its spheres does, bit
   for bit, at each cell's centre and near its lowest corner */
static int formula_grid_agrees(struct formula_native *n)
{ int c, j, ncell = n->nc[0]*n->nc[1]*n->nc[2];
  int *cand = n->cand;
  for ( c = 0 ; c < ncell ; c++ )
  { int at[3] = {c % n->nc[0], (c/n->nc[0]) % n->nc[1], c/(n->nc[0]*n->nc[1])}, s;
    for ( s = 0 ; s < 2 ; s++ )
    { REAL x[3], fg, gg[3], fa, ga[3];
      for ( j = 0 ; j < 3 ; j++ )
        x[j] = n->lo[j] + (at[j] + (s ? 1e-7 : 0.5))*n->h[j];
      formula_value(n,x,&fg,gg);
      n->cand = NULL;
      formula_value(n,x,&fa,ga);
      n->cand = cand;
      if ( fg != fa || gg[0] != ga[0] || gg[1] != ga[1] || gg[2] != ga[2] ) return 0;
    }
  }
  return 1;
}

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
  else if ( n->kind == 2 )
  { /* (x - c - L floor((x - c)/L + 0.5)), as the formula has it */
    REAL a = x[0] - n->p[0], b = x[1] - n->p[1];
    REAL X = a - n->L[0]*floor(a/n->L[0] + 0.5), Y = b - n->L[1]*floor(b/n->L[1] + 0.5);
    REAL Z = x[2] - n->p[2];
    v = (X*X + Y*Y) + Z*Z - n->p[3];
    if ( g ) { g[0] = 2.0*X; g[1] = 2.0*Y; g[2] = 2.0*Z; }
  }
  else if ( n->kind == 3 )
  { /* minimum(t_1, minimum(t_2, ... t_n)), Evolver's minimum: the second
       where the first is greater */
    int k, from = 0, to = n->np, cell = -1;
    REAL gg[3] = {0,0,0};
    if ( n->cand )
    { /* (the spheres of x's cell: formula_grid; the loop over all where x is
         past the grid in z, or not finite) */
      REAL u = x[0] - n->L[0]*floor(x[0]/n->L[0]), w = x[1] - n->L[1]*floor(x[1]/n->L[1]);
      REAL zz = (x[2] - n->lo[2])/n->h[2];
      if ( zz >= 0 && zz < n->nc[2] && u >= 0 && w >= 0 )
      { int ix = (int)(u/n->h[0]), iy = (int)(w/n->h[1]);
        if ( ix >= n->nc[0] ) ix = n->nc[0] - 1;
        if ( iy >= n->nc[1] ) iy = n->nc[1] - 1;
        cell = ix + n->nc[0]*(iy + n->nc[1]*(int)zz);
        from = n->cstart[cell]; to = n->cstart[cell + 1];
      }
    }
    for ( i = to - 1 ; i >= from ; i-- )
    { const REAL *q;
      REAL a, b, X, Y, Z, t;
      k = cell >= 0 ? n->cand[i] : i;
      q = n->P + 4*k;
      a = x[0] - q[0]; b = x[1] - q[1];
      X = a - n->L[0]*floor(a/n->L[0] + 0.5); Y = b - n->L[1]*floor(b/n->L[1] + 0.5);
      Z = x[2] - q[2];
      t = (X*X + Y*Y) + Z*Z - q[3];
      if ( i == to - 1 || !(t > v) )
      { v = t; gg[0] = 2.0*X; gg[1] = 2.0*Y; gg[2] = 2.0*Z; }
    }
    if ( g ) for ( i = 0 ; i < 3 ; i++ ) g[i] = gg[i];
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
  if ( (i = formula_find(ex)) < 0 ) return 0;
  formula_value(formulas + i,x,f,grad);
  return 1;
}

/* The slot of the closed form eval_all() takes for ex (as above), or -1:
   looked up once for a loop over the vertices (pyse_api.c's gap and normal
   queries), whose evaluations then go straight to it, the same values */
int pyse_native_formula_slot(struct expnode *ex)
{ if ( !ex || !(ex->flag & PYSE_NATIVE_FORMULA) || formula_load != pyse_load_serial() )
    return -1;
  return formula_find(ex);
}

void pyse_native_formula_at(int slot, const REAL *x, REAL *f, REAL *grad)
{ formula_value(formulas + slot,x,f,grad);
}

/* register constraint con's formula as a plane (kind 0: n, d), a sphere
   (kind 1: c, R^2) or a slab's sphere at the nearest image (kind 2: c, R^2,
   the slab's periods); 1 if registered (the closed form matching the
   interpreter, value and gradient, at test points), else 0 */
static int formula_register(int con, int kind, const double *p, int np);

int pyse_native_formula_register(int con, int kind, const double *p)
{ return formula_register(con,kind,p,1);
}

/* kind 3: the least of np spheres' kind-2 terms (p: c, R^2 by fours) */
int pyse_native_formula_register_many(int con, const double *p, int np)
{ return formula_register(con,3,p,np);
}

static int formula_register(int con, int kind, const double *p, int np)
{ struct constraint *cn;
  struct expnode *ex;
  struct formula_native n;
  REAL pts[8][3] = {{0.31,-0.27,0.73},{1.9,0.4,-0.6},{-0.8,2.3,1.1},{0.05,0.11,3.7},
                    {-1.37,-0.91,0.29},{0.7071,0.7071,0.0},{2.9,-3.1,-0.4},{0.013,-0.6,-2.2}};
  int i, k, j;
  native_serial++;
  if ( formula_load != pyse_load_serial() )
  { for ( i = 0 ; i < formula_count ; i++ )
    { formulas[i].key = NULL;    /* (the old load's expressions are gone) */
      formula_free(formulas + i);
    }
    formula_count = 0;
    memset(formula_hash,0,sizeof(formula_hash));
    formula_load = pyse_load_serial();
  }
  if ( SDIM != 3 || kind < 0 || kind > 3 || con < 0 || con >= web.maxcon || np < 1 ) return 0;
  if ( (kind == 2 || kind == 3) && !pyse_slab(n.L) ) return 0;
  n.P = NULL; n.np = 0; n.cand = n.cstart = NULL;
  cn = get_constraint(con);
  if ( !cn || !(cn->attr & IN_USE) || !cn->formula ) return 0;
  ex = cn->formula;
  n.key = ex;
  n.kind = kind;
  n.sign = 1.0;
  ex->flag &= ~PYSE_NATIVE_FORMULA;      /* (the interpreter for the test) */
  if ( kind == 3 )
  { /* (as given: the test below takes them bit for bit or not at all) */
    n.P = (REAL*)malloc(4*np*sizeof(REAL));
    if ( !n.P ) return 0;
    n.np = np;
    /* each sphere's constants read back as kind 2's, at its centre (where
       its term, -R^2, is the least) */
    for ( i = 0 ; i < np ; i++ )
    { REAL x[3], c[3], f0, g0[2*MAXCOORD], fc, gc[2*MAXCOORD];
      for ( j = 0 ; j < 3 ; j++ ) x[j] = p[4*i + j];
      eval_all(ex,x,SDIM,&f0,g0,NULLID);
      for ( j = 0 ; j < 3 ; j++ ) c[j] = x[j] - g0[j]/2.0;
      eval_all(ex,c,SDIM,&fc,gc,NULLID);
      for ( j = 0 ; j < 3 ; j++ ) n.P[4*i + j] = c[j];
      n.P[4*i + 3] = -fc;
      for ( j = 0 ; j < 4 ; j++ )
        if ( fabs(n.P[4*i + j] - p[4*i + j]) > 1e-9*(1.0 + fabs(p[4*i + j])) )
        { formula_free(&n); return 0; }
    }
    for ( j = 0 ; j < 4 ; j++ ) n.p[j] = n.P[j];
    if ( formula_grid(&n) && !formula_grid_agrees(&n) )
    { free(n.cand); free(n.cstart); n.cand = n.cstart = NULL; }
  }
  else
  { /* the constants as Evolver parsed them, read back from its interpreter
       (exactly: a sphere's gradient at 0 is -2c, its value at c -R^2; a
       plane's gradient n, its value at 0 -d), not as given: its parser and
       Python's round some decimals apart, and the closed form then missed
       the interpreter in the last bit at most points */
    REAL zero[3] = {0,0,0}, f0, g0[2*MAXCOORD];
    if ( kind == 2 )   /* (read back from the centre's guess: periodic about it) */
      for ( j = 0 ; j < 3 ; j++ ) zero[j] = p[j];
    eval_all(ex,zero,SDIM,&f0,g0,NULLID);
    if ( kind == 2 )
    { REAL c[3], fc, gc[2*MAXCOORD];
      for ( j = 0 ; j < 3 ; j++ ) c[j] = p[j] - g0[j]/2.0;
      eval_all(ex,c,SDIM,&fc,gc,NULLID);
      for ( j = 0 ; j < 3 ; j++ ) n.p[j] = c[j];
      n.p[3] = -fc;
    }
    else if ( kind == 1 )
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
  for ( k = 0 ; k < 8*(kind == 3 ? (np < 16 ? np : 16) : 1) ; k++ )
  { REAL x[3], fi, gi[2*MAXCOORD], fn, gn[3];
    const double *at = kind == 3 ? p + 4*(k/8)*(np < 16 ? 1 : np/16) : p;
    for ( j = 0 ; j < 3 ; j++ ) x[j] = (kind ? at[j] : 0.0) + (kind == 3 ? 0.5 : 1.0)*pts[k % 8][j];
    eval_all(ex,x,SDIM,&fi,gi,NULLID);
    formula_value(&n,x,&fn,gn);
    /* (bit for bit: else the interpreter's own arithmetic, a rotated
       frame's say, and this one's would part in the last bits) */
    if ( fi != fn ) { formula_free(&n); return 0; }
    for ( j = 0 ; j < 3 ; j++ )
      if ( gi[j] != gn[j] ) { formula_free(&n); return 0; }
  }
  if ( (i = formula_find(ex)) >= 0 ) formula_free(formulas + i);
  else
  { unsigned h = formula_bucket(ex);
    if ( formula_count >= FORMULA_NATIVES ) { formula_free(&n); return 0; }
    i = formula_count++;
    while ( formula_hash[h] ) h = (h + 1) & (FORMULA_HASH - 1);
    formula_hash[h] = i + 1;
  }
  formulas[i] = n;
  ex->flag |= PYSE_NATIVE_FORMULA;
  return 1;
}

int pyse_fixcount(void);  /* src/fixvol.c */

/* ---- a plate's contact lines over its edges (constraints.plate) --------
 *
 * A vertex on one face that has gone past one of its bounds (the top past
 * its foot, its end or a side; the plate into a footprint; ...) is put on
 * the edge there: on the face beyond as well, which projects it onto the
 * edge line; past an end of an edge line, onto the corner (three faces). A
 * vertex on an edge leaves it onto one of its faces where the force on it
 * points into that face, away from the edge, and the meniscus leaves the
 * edge on that face's liquid side (Gibbs at a convex edge: the meniscus
 * turns through the edge's angle before the line moves on; a concave edge
 * passes it on at once); from a corner, likewise along one of its edges.
 * The forms are the same on every face, so a contact-line edge integrates
 * them once whichever face it is on: each keeps one, one both its ends are
 * on where there is one. Onto edges at the end of each gradient step
 * (iterate.c) and after Newton's, off concave ones there too; off convex
 * ones from an equilibrium (pyse_plate_settle, the forces there: relax()
 * relaxes again after). */

#define PLATE_TOL 1e-9            /* past a bound by more than this: crossed */
enum { FACE_OPEN, FACE_TOP, FACE_END, FACE_LEFT, FACE_RIGHT };


static int plate_face_of_con(int con)
{ int i;
  for ( i = 0 ; i < plate_nfaces ; i++ ) if ( plate_faces[i].con == con ) return i;
  return -1;
}

static int plate_face_find(int plate, int kind, int block)
{ int i;
  for ( i = 0 ; i < plate_nfaces ; i++ )
    if ( plate_faces[i].plate == plate && plate_faces[i].kind == kind
         && (kind == FACE_OPEN || plate_faces[i].block == block) ) return i;
  return -1;
}

/* a bound of a face: slack >= 0 on the face, its gradient, the face beyond */
struct plate_bound { REAL slack, grad[3]; int beyond; };

/* the profile of block B at y: d and d' */
static void plate_profile(const struct plate_block *B, REAL y, REAL *d, REAL *d1)
{ REAL v[4];
  if ( y < B->y0 ) y = B->y0;
  if ( y > B->y1 ) y = B->y1;
  plate_spline(B,1,y,v);
  *d = v[0]; *d1 = v[1];
}

static int plate_bound_add(struct plate_bound *b, int n, REAL slack, REAL gx, REAL gy, REAL gz,
                           int beyond)
{ if ( beyond < 0 ) return n;
  b[n].slack = slack; b[n].grad[0] = gx; b[n].grad[1] = gy; b[n].grad[2] = gz;
  b[n].beyond = beyond;
  return n + 1;
}

/* face i's bounds at x (up to 4; the plate's: the one it is least far
   past, in a footprint it is in) */
static int plate_bounds(int i, const REAL *x, struct plate_bound *b)
{ int p = plate_faces[i].plate, k = plate_faces[i].kind, bi = plate_faces[i].block, n = 0;
  const struct plate *P = plates + p;
  int left, right, top, end, open = plate_face_find(p,FACE_OPEN,0);
  const struct plate_block *B;
  REAL d, d1;
  if ( k == FACE_OPEN )
  { int j;
    for ( j = 0 ; j < P->nblocks ; j++ )
    { REAL s[4];
      int beyond[4], m, best = -1;
      B = P->blocks + j;
      beyond[0] = plate_face_find(p,FACE_TOP,j);
      beyond[1] = plate_face_find(p,FACE_END,j);
      beyond[2] = plate_face_find(p,FACE_LEFT,j);
      beyond[3] = plate_face_find(p,FACE_RIGHT,j);
      /* (in the footprint: an edge without a side, on a mirror plane, is
         no end of it, the block goes on as its image) */
      if ( !(x[1] > B->y0 && x[1] < B->y1
             && (beyond[2] >= 0 ? x[0] > B->x0 : x[0] >= B->x0 - PLATE_TOL)
             && (beyond[3] >= 0 ? x[0] < B->x1 : x[0] <= B->x1 + PLATE_TOL)) ) continue;
      s[0] = x[1] - B->y0;
      s[1] = B->y1 - x[1];
      s[2] = x[0] - B->x0;
      s[3] = B->x1 - x[0];
      for ( m = 0 ; m < 4 ; m++ )
        if ( beyond[m] >= 0 && (best < 0 || s[m] < s[best]) ) best = m;
      if ( best < 0 ) continue;
      n = plate_bound_add(b,n,-s[best],best == 2 ? -1.0 : best == 3 ? 1.0 : 0.0,
                          best == 0 ? -1.0 : best == 1 ? 1.0 : 0.0,0.0,beyond[best]);
    }
    return n;
  }
  B = P->blocks + bi;
  top = plate_face_find(p,FACE_TOP,bi); end = plate_face_find(p,FACE_END,bi);
  left = plate_face_find(p,FACE_LEFT,bi); right = plate_face_find(p,FACE_RIGHT,bi);
  switch ( k )
  { case FACE_TOP:
      n = plate_bound_add(b,n,x[1] - B->y0,0,1,0,open);
      n = plate_bound_add(b,n,B->y1 - x[1],0,-1,0,end);
      n = plate_bound_add(b,n,x[0] - B->x0,1,0,0,left);
      n = plate_bound_add(b,n,B->x1 - x[0],-1,0,0,right);
      break;
    case FACE_END:
      n = plate_bound_add(b,n,x[2] - B->end[1],0,0,1,top);
      n = plate_bound_add(b,n,P->base - x[2],0,0,-1,open);
      n = plate_bound_add(b,n,x[0] - B->x0,1,0,0,left);
      n = plate_bound_add(b,n,B->x1 - x[0],-1,0,0,right);
      break;
    default:                      /* a side */
      plate_profile(B,x[1],&d,&d1);
      n = plate_bound_add(b,n,x[2] - d,0,-d1,1,top);
      n = plate_bound_add(b,n,P->base - x[2],0,0,-1,open);
      n = plate_bound_add(b,n,B->y1 - x[1],0,-1,0,end);
      /* (and its foot, where it has no height: the plate before it; without
         it, a contact line on the plate before the block was near the side
         along its plane) */
      n = plate_bound_add(b,n,x[1] - B->y0,0,1,0,open);
  }
  return n;
}

/* face i's normal into the liquid at x (its formula's gradient) */
static void plate_normal(int i, const REAL *x, REAL *nrm)
{ const struct plate *P = plates + plate_faces[i].plate;
  REAL d, d1, len;
  nrm[0] = nrm[1] = nrm[2] = 0.0;
  switch ( plate_faces[i].kind )
  { case FACE_OPEN: nrm[2] = -1.0; break;
    case FACE_END: nrm[1] = 1.0; break;
    case FACE_LEFT: nrm[0] = -1.0; break;
    case FACE_RIGHT: nrm[0] = 1.0; break;
    default:
      plate_profile(P->blocks + plate_faces[i].block,x[1],&d,&d1);
      nrm[1] = d1; nrm[2] = -1.0;
  }
  len = sqrt(nrm[0]*nrm[0] + nrm[1]*nrm[1] + nrm[2]*nrm[2]);
  nrm[0] /= len; nrm[1] /= len; nrm[2] /= len;
}

/* the plate faces vertex v is on (indices into plate_faces), all of one plate */
static int plate_faces_at(vertex_id v, int *out)
{ conmap_t *map;
  unsigned int m;
  int n = 0, i;
  if ( !(get_vattr(v) & CONSTRAINT) ) return 0;
  map = get_v_constraint_map(v);
  for ( m = 1 ; m <= map[0] ; m++ )
    if ( (i = plate_face_of_con((int)(map[m] & CONMASK))) >= 0 && n < 3 ) out[n++] = i;
  return n;
}

static int plate_has(const int *s, int n, int i)
{ int j;
  for ( j = 0 ; j < n ; j++ ) if ( s[j] == i ) return 1;
  return 0;
}

static int plate_valence(edge_id e)
{ facetedge_id fe0 = get_edge_fe(e), fe = fe0;
  int n = 0;
  if ( !valid_id(fe0) ) return 0;
  do { n++; fe = get_next_facet(fe); } while ( valid_id(fe) && !equal_id(fe,fe0) && n < 1000 );
  return n;
}

#define PLATE_EDGES_AT(v,e,e0,guard) \
  for ( e0 = get_vertex_edge(v), e = e0, guard = 0 ; valid_id(e) && guard < 100000 ; \
        e = get_next_tail_edge(e), guard = equal_id(e,e0) ? 100000 : guard + 1 )

/* v's contact-line edges on the plate: on one face, one both ends are on
   where there is one */
static void plate_fix_edges(vertex_id v)
{ edge_id e, e0;
  int guard;
  PLATE_EDGES_AT(v,e,e0,guard)
  { int on[8], n = 0, a[3], na, b[3], nb, keep = -1, j;
    conmap_t *map;
    unsigned int m;
    if ( !(get_eattr(e) & CONSTRAINT) || plate_valence(e) != 1 ) continue;
    map = get_e_constraint_map(e);
    for ( m = 1 ; m <= map[0] && n < 8 ; m++ )
    { int i = plate_face_of_con((int)(map[m] & CONMASK));
      if ( i >= 0 ) on[n++] = i;
    }
    if ( !n ) continue;
    na = plate_faces_at(get_edge_tailv(e),a);
    nb = plate_faces_at(get_edge_headv(e),b);
    for ( j = 0 ; j < na && keep < 0 ; j++ ) if ( plate_has(b,nb,a[j]) ) keep = a[j];
    if ( keep < 0 ) keep = on[0];
    for ( j = 0 ; j < n ; j++ )
      if ( on[j] != keep ) unset_e_constraint_map(e,plate_faces[on[j]].con);
    if ( !plate_has(on,n,keep) ) set_e_constraint_map(e,plate_faces[keep].con);
  }
}

/* whether the contact line crosses the edge between faces i and j at v:
   its neighbours along the line on the plate, one on i (not j) and one on j
   (not i) */
static int plate_crossing(vertex_id v, int i, int j)
{ edge_id e, e0;
  int guard, on_i = 0, on_j = 0;
  PLATE_EDGES_AT(v,e,e0,guard)
  { vertex_id w = get_edge_headv(e);
    int s[3], n;
    if ( plate_valence(e) != 1 ) continue;
    n = plate_faces_at(w,s);
    if ( plate_has(s,n,i) && !plate_has(s,n,j) ) on_i = 1;
    if ( plate_has(s,n,j) && !plate_has(s,n,i) ) on_j = 1;
  }
  return on_i && on_j;
}

/* the meniscus's direction from v: to its neighbours off the plate */
static int plate_departure(vertex_id v, REAL *m)
{ edge_id e, e0;
  int guard, n = 0, s[3];
  REAL *x = get_coord(v);
  m[0] = m[1] = m[2] = 0.0;
  PLATE_EDGES_AT(v,e,e0,guard)
  { vertex_id w = get_edge_headv(e);
    REAL *y = get_coord(w);
    int k;
    if ( plate_faces_at(w,s) ) continue;
    for ( k = 0 ; k < 3 ; k++ ) m[k] += y[k] - x[k];
    n++;
  }
  return n;
}

/* the net force on v (with the volumes' pressures): the step's, or anew */
static void plate_force(vertex_id v, REAL *f)
{ REAL *g = get_force(v);
  volgrad *vg;
  int k;
  for ( k = 0 ; k < 3 ; k++ ) f[k] = g[k];
  for ( vg = get_vertex_vgrad(v) ; vg ; vg = vg->chain )
    if ( vg->fixnum >= 0 && vg->fixnum < pyse_fixcount() && vpressures )
      for ( k = 0 ; k < 3 ; k++ ) f[k] -= vpressures[vg->fixnum]*vg->grad[k];
}

static REAL dot3(const REAL *a, const REAL *b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

static void cross3(const REAL *a, const REAL *b, REAL *c)
{ c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}

/* whether the edge between faces i and j is concave (the liquid's side
   under 180 degrees): the plate with a block's top (its foot: the top goes
   down into the liquid from the plate), end or sides; the others (the top
   with the end and sides, a side with the end) are convex */
static int plate_concave(int i, int j)
{ int a = plate_faces[i].kind, b = plate_faces[j].kind;
  return (a == FACE_OPEN) != (b == FACE_OPEN);
}

#define PLATE_PROBE 1e-5          /* a short step off an edge, to see which way a face lies */

/* whether a short step from x along unit dir stays within the faces k */
static int plate_within(const int *k, int nk, const REAL *x, const REAL *dir)
{ REAL y[3];
  int j, q;
  for ( j = 0 ; j < 3 ; j++ ) y[j] = x[j] + PLATE_PROBE*dir[j];
  for ( j = 0 ; j < nk ; j++ )
  { struct plate_bound b[8];
    int nb = plate_bounds(k[j],y,b);
    for ( q = 0 ; q < nb ; q++ ) if ( b[q].slack < -PLATE_PROBE/2 ) return 0;
  }
  return 1;
}

/* a vertex not on the plate that a step took into its solid (the plate,
   a block) put back on its surface, across the bound nearest it: the plate
   is a one-sided wall to every vertex, at every step (only the descents
   guard against the walls; a meniscus that a line's jump up a block's ramp
   pushed through it went on through, weight toward the plate lowering the
   energy without end). Its other constraints after (a mirror's, which no
   block's bound across it moves). 1 if moved. */
static double plate_solid(int p, const REAL *x);

static int plate_push_out(vertex_id v)
{ int p, k, moved = 0;
  REAL *x = get_coord(v);
  for ( p = 0 ; p < plate_count ; p++ )
    for ( k = 0 ; k < 3 && plate_solid(p,x) < -PLATE_TOL ; k++ )
    { const struct plate *P = plates + p;
      double best = P->base - x[2];
      int which = -1, axis = 2, j;
      REAL to = P->base;
      for ( j = 0 ; j < P->nblocks ; j++ )
      { const struct plate_block *B = P->blocks + j;
        REAL d, d1, bound[5], at[5];
        int ax[5], nb = 0, m, top = -1;
        double most = -1e300;
        plate_profile(B,x[1],&d,&d1);
        bound[nb] = B->y0 - x[1]; ax[nb] = 1; at[nb++] = B->y0;
        bound[nb] = x[1] - B->y1; ax[nb] = 1; at[nb++] = B->y1;
        bound[nb] = d - x[2]; ax[nb] = 2; at[nb] = d; top = nb++;
        if ( plate_face_find(p,FACE_LEFT,j) >= 0 ) { bound[nb] = B->x0 - x[0]; ax[nb] = 0; at[nb++] = B->x0; }
        if ( plate_face_find(p,FACE_RIGHT,j) >= 0 ) { bound[nb] = x[0] - B->x1; ax[nb] = 0; at[nb++] = B->x1; }
        for ( m = 0 ; m < nb ; m++ ) if ( bound[m] > most ) { most = bound[m]; which = m; }
        if ( most < best )
        { best = most; axis = ax[which]; to = at[which];
          (void)top;
        }
      }
      x[axis] = to;
      moved = 1;
    }
  if ( moved && (get_vattr(v) & CONSTRAINT) )
    project_v_constr(v,ACTUAL_MOVE,RESET_ONESIDEDNESS);
  return moved;
}

/* onto edges and corners, off them: release, off any from an equilibrium
   (the forces where the surface is now); concave, off concave edges with
   the forces at hand (a gradient step's); returns the vertices changed */
static int plate_pass(int release, int concave)
{ vertex_id v;
  int changed = 0;
  if ( plate_load != pyse_load_serial() || !plate_nfaces ) return 0;
  FOR_ALL_VERTICES(v)
  { int s[3], n = plate_faces_at(v,s), j, q, add[3], nadd = 0, total;
    REAL *x = get_coord(v);
    if ( !n )
    { /* (not on the plate: out of its solid, if a step took it in) */
      changed += plate_push_out(v);
      continue;
    }
    /* past a bound of a face it is on (to a face it isn't on): across a
       convex edge onto that face too (the edge, where the line can pin);
       across a concave one off the face and onto that one (a concave edge
       holds no line along it: a vertex left on a block's foot while the
       line beside it went up the ramp folded the line there, and the
       descents churned), unless the line crosses the edge at the vertex */
    total = (int)get_v_constraint_map(v)[0];
    { int drop[3], ndrop = 0;
      for ( j = 0 ; j < n ; j++ )
      { struct plate_bound b[8];
        int nb = plate_bounds(s[j],x,b);
        for ( q = 0 ; q < nb ; q++ )
          if ( b[q].slack < -PLATE_TOL && !plate_has(s,n,b[q].beyond)
               && !plate_has(add,nadd,b[q].beyond) && nadd < 3 )
          { add[nadd++] = b[q].beyond;
            if ( plate_concave(s[j],b[q].beyond) && !plate_has(drop,ndrop,s[j])
                 && !plate_crossing(v,s[j],b[q].beyond) )
              drop[ndrop++] = s[j];
          }
      }
      while ( nadd && total - ndrop + nadd > 3 ) nadd--;   /* (a mirror's corner: the first) */
      if ( nadd )
      { /* onto the edge (both faces: a short move back to where it crossed),
           then off the face it left, not projected again: the edge's point
           is on the face beyond (projected onto it from where the step left
           it, past a block's 63 degree foot, a vertex dropped 0.36 on edges
           of 0.12 and the mesh folded through itself) */
        for ( j = 0 ; j < nadd ; j++ ) set_v_constraint_map(v,plate_faces[add[j]].con);
        project_v_constr(v,ACTUAL_MOVE,RESET_ONESIDEDNESS);
        for ( j = 0 ; j < ndrop ; j++ ) unset_v_constraint_map(v,plate_faces[drop[j]].con);
        plate_fix_edges(v);
        changed++;
        continue;                   /* (its force was for where it was) */
      }
    }
    /* (off a convex edge only from an equilibrium, release; a concave one
       holds no line, Gibbs's range there empty: off it at every step, or
       the meniscus behind a line held at a block's foot went on up the ramp,
       through it) */
    if ( n < 2 || (!release && (!concave || n != 2)) ) continue;
    /* (where the line crosses the edge here, from one face to the other,
       the vertex is the crossing: kept, a filament's tip or a line rounding
       a block, whose vertices hopped face to face and folded the mesh) */
    if ( n == 2 && plate_crossing(v,s[0],s[1]) ) continue;
    { REAL f[3], m[3], best = 0.0, dir[3];
      int drop = -1, o;
      plate_force(v,f);
      if ( !plate_departure(v,m) ) continue;
      for ( o = 0 ; o < n ; o++ )
      { /* leaving face s[o]: into what is left (a face, or along an edge
           line), the way a short step stays within it */
        int keep[2], nk = 0, ok = 1;
        REAL g[3], nrm[3][3], len;
        for ( j = 0 ; j < n ; j++ ) if ( j != o ) keep[nk++] = s[j];
        for ( j = 0 ; j < nk ; j++ ) plate_normal(keep[j],x,nrm[j]);
        plate_normal(s[o],x,nrm[2]);
        if ( nk == 1 )
        { /* in the face, across the edge line it shares with s[o] */
          REAL line[3];
          cross3(nrm[0],nrm[2],line);
          cross3(line,nrm[0],g);
          /* the meniscus on that face's liquid side, across the edge line */
          { REAL mm[3], t2 = dot3(m,line)/dot3(line,line);
            for ( j = 0 ; j < 3 ; j++ ) mm[j] = m[j] - t2*line[j];
            if ( dot3(mm,nrm[0]) <= 0 ) ok = 0;
          }
        }
        else cross3(nrm[0],nrm[1],g);       /* along the edge line */
        len = sqrt(dot3(g,g));
        if ( !ok || len < 1e-12 ) continue;
        for ( j = 0 ; j < 3 ; j++ ) g[j] /= len;
        if ( !plate_within(keep,nk,x,g) )
        { for ( j = 0 ; j < 3 ; j++ ) g[j] = -g[j];
          if ( !plate_within(keep,nk,x,g) ) continue;
        }
        /* (a concave edge: the face left lies on the liquid's side of the one
           kept, into which the step goes) */
        if ( !release && dot3(g,nrm[2]) <= 0 ) continue;
        if ( dot3(f,g) > best ) { best = dot3(f,g); drop = o; for ( j = 0 ; j < 3 ; j++ ) dir[j] = g[j]; }
      }
      if ( drop < 0 ) continue;
      unset_v_constraint_map(v,plate_faces[s[drop]].con);
      project_v_constr(v,ACTUAL_MOVE,RESET_ONESIDEDNESS);
      plate_fix_edges(v);
      changed++;
      (void)dir;
    }
  }
  if ( changed )
  { if ( count_fixed_vol() || web.pressure_flag ) calc_content(Q_FIXED);
    calc_pressure();
    calc_energy();
  }
  return changed;
}

/* after Evolver's projection onto the constraints (project_all, at every
   trial of a line search and Newton's moves): the vertices not on a plate
   out of its solid (plate_push_out); the vertices moved */
int pyse_plate_project(void)
{ vertex_id v;
  int s[3], moved = 0;
  if ( plate_load != pyse_load_serial() || !plate_nfaces ) return 0;
  FOR_ALL_VERTICES(v)
    if ( !plate_faces_at(v,s) ) moved += plate_push_out(v);
  return moved;
}

/* after pyse_chop_lines split contact-line edges (pyse_edit.c): their
   midpoints, on the edge's one face, onto edges and faces as the pass puts
   them (a midpoint of an edge on the plate whose other end had gone onto a
   block's side lay on the plate inside the block, and a descent's run was
   judged through a wall); no release, the step's forces gone */
int pyse_plate_after_split(void)
{ return plate_pass(0,0);
}

/* at the end of a gradient step: onto edges only. (Off them only from an
   equilibrium, pyse_plate_settle: with a step's forces, short of one, a
   vertex pinned at an edge let go of it and was put back, step after step,
   and the relaxing never converged.) */
int pyse_plate_step(void)
{ return plate_pass(0,1);
}

/* after Newton or anywhere: onto edges, and (release) off them by the
   forces where the surface is now */
static void plate_settle_body(void *arg)
{ int *a = (int *)arg;
  if ( a[0] )
  { find_fixed();
    calc_volgrads(DO_OPTS);
    calc_force();
    pressure_forces();
  }
  a[1] = plate_pass(a[0],a[0]);
  if ( a[0] ) vgrad_end();
}

int pyse_plate_settle(int release)
{ int a[2];
  int r;
  a[0] = release; a[1] = 0;
  if ( plate_load != pyse_load_serial() || !plate_nfaces ) return 0;
  r = pyse_surface_call(plate_settle_body,a);
  if ( r ) return -1;
  if ( a[1] ) pyse_surface_changed();
  return a[1];
}

/* plate p's solid at x: positive where the liquid may be, negative in the
   solid (the plate and its blocks), as constraints.Plate.solid: the least
   of base - z and per block the most of its bounds' distances past them,
   its top at y held to its range */
static double plate_solid(int p, const REAL *x)
{ const struct plate *P = plates + p;
  double out = P->base - x[2];
  int j;
  for ( j = 0 ; j < P->nblocks ; j++ )
  { const struct plate_block *B = P->blocks + j;
    REAL d, d1;
    double t = B->y0 - x[1];
    if ( x[1] - B->y1 > t ) t = x[1] - B->y1;
    plate_profile(B,x[1],&d,&d1);
    if ( d - x[2] > t ) t = d - x[2];
    if ( plate_face_find(p,FACE_LEFT,j) >= 0 && B->x0 - x[0] > t ) t = B->x0 - x[0];
    if ( plate_face_find(p,FACE_RIGHT,j) >= 0 && x[0] - B->x1 > t ) t = x[0] - B->x1;
    if ( t < out ) out = t;
  }
  return out;
}

/* a plate's face con: its gap at x, from its formula's (gap, the formula
   over its gradient's length): the distance to the face, past its bounds
   too (its formula holds past the face, through the liquid beside it and
   behind it); negative only in the plate's solid (deeper than PLATE_TOL: a
   vertex on another face is on its surface) and for the faces nearest
   there, which the point went through (each face of a block reporting the
   depth, a change would be tried on the plate where it isn't). 0 if con
   is no plate's face (gap as it was). */
int pyse_plate_gap(int con, const REAL *x, double *gap)
{ struct plate_bound b[8];
  int i, n, q;
  double out = 0.0, depth, d;
  if ( plate_load != pyse_load_serial() || (i = plate_face_of_con(con)) < 0 ) return 0;
  n = plate_bounds(i,x,b);
  for ( q = 0 ; q < n ; q++ ) if ( b[q].slack < 0 ) out += b[q].slack*b[q].slack;
  d = sqrt((*gap)*(*gap) + out);
  depth = -plate_solid(plate_faces[i].plate,x);
  *gap = depth > PLATE_TOL && d <= depth*(1 + 1e-6) + PLATE_TOL ? -d : d;
  return 1;
}
