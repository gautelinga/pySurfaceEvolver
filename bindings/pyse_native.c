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

void pyse_native_enable(int on) { native_on = on; }

/* register constraint con as a sphere wall (k: its energy coefficient, NaN
   where it isn't a number: the content only); 0 if it can't be (no
   integrands, too many) */
int pyse_native_sphere(int con, const double *c, double R, const double *e1,
                       const double *e2, const double *e3, double k)
{ struct constraint *cn;
  struct native *n;
  int i;
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
 * R^2 (x - c)/|x - c|^3, (x - c)/R inside the sphere (bounded where no
 * liquid is: no jump at the centre); on each wall's contact lines, monopole potentials
 * R^2 (d x w)/(r (r - d . w)), w = x - c; each term times -k (the wetting
 * energy, k = tension cos(max(angle, pyse_angle_floor)) for angles under
 * ANGLE_FLOOR_TOP, else tension cos(angle)) or -R/3 (the volume). Plus the
 * wall's own terms: (0, 0, z) for the volume's facets, a flux sphere's exact
 * content part (Z/3 Y, -(c_z + Z/3) X, 0) with Z = z - c_z etc. Registered
 * by the integrand's first expression node, as the spheres above. */

#define FLUX_FIELDS 512
#define FLUX_TERMS 64
#define FLUX_FLOOR_TOP 20.0       /* constraints.ANGLE_FLOOR_TOP */

struct flux_term
{ REAL c[3], R2, d[3];           /* d zero: a point source */
  int energy;                    /* -k R^2 (else -R/3 R^2) */
  REAL angle, tension, R;
  REAL coef;                     /* the coefficient with the floor below the angle */
};

struct flux_field
{ struct expnode *key;
  int base;                      /* 0 none, 1 (0,0,z), 2 a sphere's content part,
                                    3 a plane's content, 4 a plane's wetting */
  REAL bc[12];                   /* 2: the sphere's centre; 3: n_z, o, u, v;
                                    4: angle, tension, o, u, v */
  int nterm;
  struct flux_term term[FLUX_TERMS];
  /* point sources only (a facet integrand): their centres, R^2 and R^3 as
     arrays, for flux_points */
  int points;
  REAL pc[3][FLUX_TERMS], pR2[FLUX_TERMS], pR3[FLUX_TERMS];
};

static struct flux_field fluxes[FLUX_FIELDS];
static int flux_count = 0;
static long flux_load = -1;

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

/* register a field: target a method instance (its name) or a constraint's
   energy (which 0) or content (1) integrand; terms 11 numbers each (c, R,
   d, energy flag, angle, tension, and a spare); 0 if it can't be */
int pyse_native_flux(const char *method, int con, int which, int base, const double *bc,
                     int nbc, const double *terms, int nterm)
{ struct expnode *key = NULL;
  struct flux_field *f = NULL;
  int i, k;
  if ( flux_load != pyse_load_serial() ) { flux_count = 0; flux_load = pyse_load_serial(); }
  if ( nterm > FLUX_TERMS || SDIM != 3 || nbc > 12 ) return 0;
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
  for ( i = 0 ; i < flux_count ; i++ )
    if ( fluxes[i].key == key ) f = fluxes + i;
  if ( !f )
  { if ( flux_count >= FLUX_FIELDS ) return 0;
    f = fluxes + flux_count++;
  }
  f->key = key;
  f->base = base;
  for ( k = 0 ; k < 12 ; k++ ) f->bc[k] = bc && k < nbc ? bc[k] : 0.0;
  f->nterm = nterm;
  for ( i = 0 ; i < nterm ; i++ )
  { const double *t = terms + 11*i;
    struct flux_term *u = f->term + i;
    for ( k = 0 ; k < 3 ; k++ ) { u->c[k] = t[k]; u->d[k] = t[4+k]; }
    u->R = t[3]; u->R2 = t[3]*t[3];
    u->energy = t[7] != 0;
    u->angle = t[8]; u->tension = t[9];
    u->coef = u->energy ? -u->tension*cos(u->angle*M_PI/180.0)*u->R2 : -u->R/3.0*u->R2;
  }
  f->points = nterm > 0;
  for ( i = 0 ; i < nterm ; i++ )
  { struct flux_term *u = f->term + i;
    if ( u->d[0] != 0 || u->d[1] != 0 || u->d[2] != 0 ) f->points = 0;
    for ( k = 0 ; k < 3 ; k++ ) f->pc[k][i] = u->c[k];
    f->pR2[i] = u->R2;
    f->pR3[i] = u->R2*u->R;
  }
  return 1;
}

/* The point sources' 1/r^3 and 3/r^5 factors, the costly part of a facet
   integrand (most of a 16-bead pack's gradient step): ir3 = coef/(r^2 r)
   (coef/R^3 inside, where r^2 < R^2) and, with want5, ir5 = 3 ir3/r^2 (0
   inside). Four terms at a time with AVX2, two with SSE2, picked at run
   time; IEEE square roots, quotients and products, no fused operations
   (-ffp-contract=off), so each term's factors are the scalar ones to the
   bit, and the sums after them are taken in the same order as before. */
static void flux_inv_scalar(int n, const REAL *r2, const REAL *coef, const REAL *R2,
                            const REAL *R3, int want5, REAL *ir3, REAL *ir5)
{ int t;
  for ( t = 0 ; t < n ; t++ )
    if ( r2[t] < R2[t] )
    { ir3[t] = coef[t]/R3[t];
      if ( want5 ) ir5[t] = 0.0;
    }
    else
    { REAL r = sqrt(r2[t]);
      ir3[t] = coef[t]/(r2[t]*r);
      if ( want5 ) ir5[t] = 3.0*ir3[t]/r2[t];
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(LONGDOUBLE) \
    && !defined(FLOAT128)
#include <immintrin.h>
#define FLUX_SIMD 1
__attribute__((target("avx2")))
static void flux_inv_avx2(int n, const REAL *r2, const REAL *coef, const REAL *R2,
                          const REAL *R3, int want5, REAL *ir3, REAL *ir5)
{ int t = 0;
  const __m256d three = _mm256_set1_pd(3.0), zero = _mm256_setzero_pd();
  for ( ; t + 4 <= n ; t += 4 )
  { __m256d q = _mm256_loadu_pd(r2 + t), c = _mm256_loadu_pd(coef + t);
    __m256d in = _mm256_cmp_pd(q, _mm256_loadu_pd(R2 + t), _CMP_LT_OQ);
    __m256d out3 = _mm256_div_pd(c,_mm256_mul_pd(q,_mm256_sqrt_pd(q)));
    __m256d i3 = _mm256_blendv_pd(out3,_mm256_div_pd(c,_mm256_loadu_pd(R3 + t)),in);
    _mm256_storeu_pd(ir3 + t,i3);
    if ( want5 )
      _mm256_storeu_pd(ir5 + t,_mm256_blendv_pd(_mm256_div_pd(_mm256_mul_pd(three,out3),q),
                                                zero,in));
  }
  flux_inv_scalar(n - t,r2 + t,coef + t,R2 + t,R3 + t,want5,ir3 + t,ir5 + t);
}

static void flux_inv_sse2(int n, const REAL *r2, const REAL *coef, const REAL *R2,
                          const REAL *R3, int want5, REAL *ir3, REAL *ir5)
{ int t = 0;
  const __m128d three = _mm_set1_pd(3.0);
  for ( ; t + 2 <= n ; t += 2 )
  { __m128d q = _mm_loadu_pd(r2 + t), c = _mm_loadu_pd(coef + t);
    __m128d in = _mm_cmplt_pd(q,_mm_loadu_pd(R2 + t));
    __m128d out3 = _mm_div_pd(c,_mm_mul_pd(q,_mm_sqrt_pd(q)));
    __m128d in3 = _mm_div_pd(c,_mm_loadu_pd(R3 + t));
    _mm_storeu_pd(ir3 + t,_mm_or_pd(_mm_and_pd(in,in3),_mm_andnot_pd(in,out3)));
    if ( want5 )
      _mm_storeu_pd(ir5 + t,_mm_andnot_pd(in,_mm_div_pd(_mm_mul_pd(three,out3),q)));
  }
  flux_inv_scalar(n - t,r2 + t,coef + t,R2 + t,R3 + t,want5,ir3 + t,ir5 + t);
}
#endif

static void flux_inv(int n, const REAL *r2, const REAL *coef, const REAL *R2, const REAL *R3,
                     int want5, REAL *ir3, REAL *ir5)
{
#ifdef FLUX_SIMD
  static int level = -1;            /* (a benign race: every thread finds the same) */
  if ( level < 0 ) level = __builtin_cpu_supports("avx2") ? 2 : 1;
  if ( level == 2 ) { flux_inv_avx2(n,r2,coef,R2,R3,want5,ir3,ir5); return; }
  flux_inv_sse2(n,r2,coef,R2,R3,want5,ir3,ir5);
#else
  flux_inv_scalar(n,r2,coef,R2,R3,want5,ir3,ir5);
#endif
}

/* a field of point sources only at x, in flux_eval's order: v and (order)
   g added to */
static void flux_points(const struct flux_field *f, const REAL *x, int order, REAL floor,
                        REAL v[3], REAL g[3][3])
{ REAL X[3][FLUX_TERMS], r2[FLUX_TERMS], coef[FLUX_TERMS], ir3[FLUX_TERMS], ir5[FLUX_TERMS];
  int t, m, i, n = f->nterm;
  for ( t = 0 ; t < n ; t++ )
  { for ( m = 0 ; m < 3 ; m++ ) X[m][t] = x[m] - f->pc[m][t];
    r2[t] = X[0][t]*X[0][t] + X[1][t]*X[1][t] + X[2][t]*X[2][t];
    coef[t] = flux_coefficient(f->term + t,floor);
  }
  flux_inv(n,r2,coef,f->pR2,f->pR3,order,ir3,ir5);
  for ( t = 0 ; t < n ; t++ )
  { REAL a = ir3[t];
    if ( r2[t] < f->pR2[t] )           /* inside: coef (x - c)/R^3 */
    { for ( m = 0 ; m < 3 ; m++ )
      { v[m] += X[m][t]*a;
        if ( order ) g[m][m] += a;
      }
    }
    else
    { REAL b = order ? ir5[t] : 0.0;
      for ( m = 0 ; m < 3 ; m++ )
      { v[m] += X[m][t]*a;
        if ( order ) for ( i = 0 ; i < 3 ; i++ ) g[m][i] += (m == i ? a : 0.0) - b*X[m][t]*X[i][t];
      }
    }
  }
}

static int flux_eval(struct expnode *key, REAL *x, int order, REAL *val,
                     REAL (*d)[MAXCOORD], REAL ***s)
{ struct flux_field *f = NULL;
  dual out[3], X[3];
  REAL ex[3][3] = {{1,0,0},{0,1,0},{0,0,1}}, floor;
  int i, j, m, t;
  if ( flux_load != pyse_load_serial() ) return 0;
  for ( i = 0 ; i < flux_count ; i++ )
    if ( fluxes[i].key == key ) { f = fluxes + i; break; }
  if ( !f ) return 0;
  floor = flux_floor();
  if ( order < 2 )
  { /* values (and first derivatives) in closed form: no second-order duals,
       most of the cost of a gradient step */
    REAL v[3] = {0,0,0}, g[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
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
    if ( f->points )
      flux_points(f,x,order,floor,v,g);
    else
    for ( t = 0 ; t < f->nterm ; t++ )
    { struct flux_term *u = f->term + t;
      REAL coef = flux_coefficient(u,floor), X[3], r2, r;
      for ( m = 0 ; m < 3 ; m++ ) X[m] = x[m] - u->c[m];
      r2 = X[0]*X[0] + X[1]*X[1] + X[2]*X[2];
      r = sqrt(r2);
      if ( u->d[0] == 0 && u->d[1] == 0 && u->d[2] == 0 && r2 < u->R2 )
      { REAL ir3 = coef/(u->R2*u->R);      /* inside: coef (x - c)/R^3 */
        for ( m = 0 ; m < 3 ; m++ )
        { v[m] += X[m]*ir3;
          if ( order ) g[m][m] += ir3;
        }
      }
      else if ( u->d[0] == 0 && u->d[1] == 0 && u->d[2] == 0 )
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
  if ( f->points && (f->base == 0 || f->base == 1) )
  { /* second derivatives in closed form for point sources (a facet
       integrand): with w = x - c, a = coef/r^3, F = a w,
       dF_m/dx_i = a (d_mi - 3 w_m w_i/r^2),
       d2F_m/dx_i dx_j = -3 a/r^2 (d_mi w_j + d_mj w_i + d_ij w_m)
                         + 15 a w_m w_i w_j/r^4;
       inside, F = coef w/R^3, linear. (The dual numbers, with pow, were a
       quarter of a 16-bead pack's Newton step.) */
    REAL v[3] = {0,0,0}, g[3][3] = {{0,0,0},{0,0,0},{0,0,0}}, h[3][3][3];
    memset(h,0,sizeof(h));
    if ( f->base == 1 ) { v[2] = x[2]; g[2][2] = 1.0; }
    for ( t = 0 ; t < f->nterm ; t++ )
    { REAL w[3], r2, coef = flux_coefficient(f->term + t,floor);
      int k;
      for ( m = 0 ; m < 3 ; m++ ) w[m] = x[m] - f->pc[m][t];
      r2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
      if ( r2 < f->pR2[t] )
      { REAL a = coef/f->pR3[t];
        for ( m = 0 ; m < 3 ; m++ ) { v[m] += w[m]*a; g[m][m] += a; }
        continue;
      }
      { REAL a = coef/(r2*sqrt(r2)), b = 3.0*a/r2, e = 15.0*a/(r2*r2);
        for ( m = 0 ; m < 3 ; m++ )
        { v[m] += w[m]*a;
          for ( i = 0 ; i < 3 ; i++ )
          { g[m][i] += (m == i ? a : 0.0) - b*w[m]*w[i];
            for ( k = 0 ; k < 3 ; k++ )
              h[m][i][k] += -b*((m == i ? w[k] : 0.0) + (m == k ? w[i] : 0.0)
                                + (i == k ? w[m] : 0.0)) + e*w[m]*w[i]*w[k];
          }
        }
      }
    }
    for ( m = 0 ; m < 3 ; m++ )
    { val[m] = v[m];
      for ( i = 0 ; i < 3 ; i++ )
      { d[m][i] = g[m][i];
        for ( j = 0 ; j < 3 ; j++ ) s[m][i][j] = h[m][i][j];
      }
    }
    return 1;
  }
  for ( m = 0 ; m < 3 ; m++ ) out[m] = d_const(0.0);
  if ( f->base == 1 ) out[2] = d_lin(ex[2],0.0,x);
  else if ( f->base == 2 )
  { dual XX = d_lin(ex[0],-f->bc[0],x), YY = d_lin(ex[1],-f->bc[1],x), ZZ = d_lin(ex[2],-f->bc[2],x);
    out[0] = d_mul(d_scale(ZZ,1.0/3.0),YY);
    out[1] = d_scale(d_mul(d_add(d_const(f->bc[2]),d_scale(ZZ,1.0/3.0)),XX),-1.0);
  }
  else if ( f->base == 3 || f->base == 4 )
  { const REAL *o = f->bc + (f->base == 3 ? 1 : 2), *u = o + 3, *w = o + 6;
    dual ss = d_const(0.0), tt = d_const(0.0), p;
    for ( i = 0 ; i < 3 ; i++ )
    { dual xi = d_lin(ex[i],-o[i],x);
      ss = d_add(ss,d_scale(xi,u[i]));
      tt = d_add(tt,d_scale(xi,w[i]));
    }
    if ( f->base == 3 )
      p = d_scale(d_add(d_add(d_scale(tt,o[2]),d_scale(d_mul(ss,tt),u[2])),
                        d_scale(d_mul(tt,tt),0.5*w[2])),f->bc[0]);
    else
      p = d_scale(tt,plane_coefficient(f,floor));
    for ( m = 0 ; m < 3 ; m++ ) out[m] = d_scale(p,u[m]);
  }
  for ( t = 0 ; t < f->nterm ; t++ )
  { struct flux_term *u = f->term + t;
    REAL coef = flux_coefficient(u,floor);
    dual r2;
    for ( m = 0 ; m < 3 ; m++ ) X[m] = d_lin(ex[m],-u->c[m],x);
    r2 = d_add(d_add(d_mul(X[0],X[0]),d_mul(X[1],X[1])),d_mul(X[2],X[2]));
    if ( u->d[0] == 0 && u->d[1] == 0 && u->d[2] == 0 )
    { dual inv = r2.v < u->R2 ? d_const(coef/(u->R2*u->R))   /* inside: (x - c)/R^3 */
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
    if ( order >= 1 ) for ( i = 0 ; i < 3 ; i++ ) d[m][i] = out[m].g[i];
    if ( order >= 2 )
      for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ ) s[m][i][j] = out[m].h[i][j];
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
