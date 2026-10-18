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

/* the integrand vector of the wall whose integrand key is (its first
   node), at x: values, and with order 1 or 2 their first (and second)
   derivatives; 0 where key isn't a registered wall's */
int pyse_native_eval(struct expnode *key, REAL *x, int order, REAL *val,
                     REAL (*d)[MAXCOORD], REAL ***s)
{ struct native *n = NULL;
  int i, j, m, content = 0;
  dual a, b, h, D, rr, lead, area[3];
  REAL off;
  if ( !native_on || !key || native_load != pyse_load_serial() || SDIM != 3 ) return 0;
  for ( i = 0 ; i < native_count ; i++ )
  { if ( natives[i].energy == key ) { n = natives + i; break; }
    if ( natives[i].content == key ) { n = natives + i; content = 1; break; }
  }
  if ( !n ) return 0;
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
