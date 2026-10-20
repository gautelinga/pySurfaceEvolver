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
 * R^2 (x - c)/|x - c|^3; on each wall's contact lines, monopole potentials
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
  int base;                      /* 0 none, 1 (0,0,z), 2 a sphere's content part */
  REAL bc[3];                    /* that sphere's centre */
  int nterm;
  struct flux_term term[FLUX_TERMS];
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

/* register a field: target a method instance (its name) or a constraint's
   energy (which 0) or content (1) integrand; terms 11 numbers each (c, R,
   d, energy flag, angle, tension, and a spare); 0 if it can't be */
int pyse_native_flux(const char *method, int con, int which, int base, const double *bc,
                     const double *terms, int nterm)
{ struct expnode *key = NULL;
  struct flux_field *f = NULL;
  int i, k;
  if ( flux_load != pyse_load_serial() ) { flux_count = 0; flux_load = pyse_load_serial(); }
  if ( nterm > FLUX_TERMS || SDIM != 3 ) return 0;
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
  for ( k = 0 ; k < 3 ; k++ ) f->bc[k] = bc ? bc[k] : 0.0;
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
  return 1;
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
    for ( t = 0 ; t < f->nterm ; t++ )
    { struct flux_term *u = f->term + t;
      REAL coef = flux_coefficient(u,floor), X[3], r2, r;
      for ( m = 0 ; m < 3 ; m++ ) X[m] = x[m] - u->c[m];
      r2 = X[0]*X[0] + X[1]*X[1] + X[2]*X[2];
      r = sqrt(r2);
      if ( u->d[0] == 0 && u->d[1] == 0 && u->d[2] == 0 )
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
  for ( m = 0 ; m < 3 ; m++ ) out[m] = d_const(0.0);
  if ( f->base == 1 ) out[2] = d_lin(ex[2],0.0,x);
  else if ( f->base == 2 )
  { dual XX = d_lin(ex[0],-f->bc[0],x), YY = d_lin(ex[1],-f->bc[1],x), ZZ = d_lin(ex[2],-f->bc[2],x);
    out[0] = d_mul(d_scale(ZZ,1.0/3.0),YY);
    out[1] = d_scale(d_mul(d_add(d_const(f->bc[2]),d_scale(ZZ,1.0/3.0)),XX),-1.0);
  }
  for ( t = 0 ; t < f->nterm ; t++ )
  { struct flux_term *u = f->term + t;
    REAL coef = flux_coefficient(u,floor);
    dual r2;
    for ( m = 0 ; m < 3 ; m++ ) X[m] = d_lin(ex[m],-u->c[m],x);
    r2 = d_add(d_add(d_mul(X[0],X[0]),d_mul(X[1],X[1])),d_mul(X[2],X[2]));
    if ( u->d[0] == 0 && u->d[1] == 0 && u->d[2] == 0 )
    { dual inv = d_scale(d_pow(r2,-1.5),coef);         /* coef (x - c)/r^3 */
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
