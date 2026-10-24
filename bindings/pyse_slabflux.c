/*
 * pyse_slabflux.c -- the flux gauge's fields in a slab periodic in x and y.
 *
 * constraints.sphere(gauge="flux") takes a sphere's wetted area as the flux
 * of R^2 (x - c)/|x - c|^3 out of the liquid (_flux.py). In a slab
 * (make_datafile(periods=), _periodic.py) each sphere's field is the sum
 * over the lattice of its source and, with a floor z = z_f, of its mirror
 * image below the floor: periodic, and with no normal part on the floor (no
 * line integrals there). The sum converges only conditionally; the far part
 * of it is taken here to a tolerance, the near part exactly:
 *
 *   explicit sources: each sphere's source and mirror at the lattice images
 *     |n|inf <= M (sf.M);
 *   the tail: the rest, their field Ewald's sum less the explicit ones',
 *     fitted once per layer of the slab (height min(Lx, Ly)) by a harmonic
 *     polynomial of degree pT from samples on a sphere (the fit's operator
 *     from Python: harmonic basis H and the samples' pseudo-inverse);
 *   regions: boxes of side ~2h tiling the cell, and each sphere's ball. A
 *     region's near sources (within kappa times its radius) are summed
 *     directly; the other explicit ones by their Taylor coefficients at its
 *     centre (a recurrence for 1/r), with the tail's polynomial moved there,
 *     to degree p.
 *
 * Fields come in classes: the volume's (strength -R^3/3) and one per
 * distinct (angle, tension) of the wetting energy (strength R^2, the
 * coefficient -tension cos(angle) applied at evaluation, where relax()'s
 * angle floor may raise the angle).
 *
 * The facets' integrands: the volume (0, 0, z - z_f) + F_vol, the wetting
 * sum_k coef_k F_k - k_f z-hat (k_f the floor's tension cos(angle)). A
 * sphere j's line integrands: the flux through its wetted part of every
 * source but its own, a primitive A with curl A = -F (Evolver's
 * orientation, as _flux's monopoles): near sources by their monopole
 * potentials, strings away from c_j; far ones by the field's local
 * expansion's own potential, A = -sum_l (x - c) x grad Phi_l/(l + 1) (Phi_l
 * the expansion's homogeneous parts, F = -grad Phi: smooth in the ball, no
 * strings). Plus, as _periodic's without the flux gauge, the floor's z-hat
 * through the wetted part, B = (x - c) x z-hat/2 (energy -k_f B, content
 * -z_f B), and the sphere's own content part (_flux_sphere's).
 *
 * Regions and layers are made when a point first needs them (under a lock;
 * made once, then read only).
 */

#include "include.h"
#include "pyse_api.h"
#include "pyse_internal.h"

#define SF_MAXCLS 8
#define SF_PMAX 24
#define SF_FLOOR_TOP 20.0         /* constraints.ANGLE_FLOOR_TOP */
#define SF_CORE 0.5               /* _flux.CORE */

/* monomials u^a v^b w^c of degree <= SF_PMAX, by degree */
static int sf_mono_n[SF_PMAX + 2];        /* count of degree <= p is sf_mono_n[p + 1] */
static unsigned char sf_mono[(SF_PMAX + 1)*(SF_PMAX + 2)*(SF_PMAX + 3)/6][3];
static int sf_mono_idx[SF_PMAX + 1][SF_PMAX + 1][SF_PMAX + 1];
static int sf_mono_ready = 0;

static void sf_monos(void)
{ int n, a, b, k = 0;
  if ( sf_mono_ready ) return;
  for ( n = 0 ; n <= SF_PMAX ; n++ )
  { sf_mono_n[n] = k;
    for ( a = n ; a >= 0 ; a-- )
      for ( b = n - a ; b >= 0 ; b-- )
      { sf_mono[k][0] = (unsigned char)a; sf_mono[k][1] = (unsigned char)b;
        sf_mono[k][2] = (unsigned char)(n - a - b);
        sf_mono_idx[a][b][n - a - b] = k;
        k++;
      }
  }
  sf_mono_n[SF_PMAX + 1] = k;
  sf_mono_ready = 1;
}

#define NMONO(p) (sf_mono_n[(p) + 1])

struct sf_src { REAL c[3]; REAL q[SF_MAXCLS]; REAL K2, iK3; int bead, own; };

struct sf_region
{ REAL c[3], rho;               /* centre, the scale of its polynomials (u = (x - c)/rho) */
  int p;
  int nnear;
  struct sf_src *near;
  REAL *phi;                    /* [cls][NMONO(p)]: Phi, F = -grad Phi */
  REAL *A;                      /* walls: [cls][3][NMONO(p)], A's components */
};

struct sf_layer { REAL c[3], rho; REAL *phi; };  /* [cls][NMONO(pT)] */

static struct
{ long load;
  int on;
  REAL L[2], zf, z0;            /* periods, the floor (NaN: none), boxes' base height */
  int mirror, lattice;            /* the floor's images; periodic in x and y */
  REAL ox, oy;                    /* the boxes' grid's corner */
  int nbead;
  REAL (*bead)[3]; REAL *R;
  int ncls, nen;                /* classes: nen energy ones, then the volume's */
  REAL cls_angle[SF_MAXCLS], cls_tension[SF_MAXCLS];
  int floor_energy; REAL floor_angle, floor_tension;
  REAL h, kappa, kappa_w; int p, pw, M, pT;
  /* Ewald */
  REAL alpha; int nr, nk;
  /* the explicit sources */
  int nexp; struct sf_src *exp;
  /* the base ones (one cell): for Ewald */
  int nbase; struct sf_src *base;
  /* the tail's fit */
  int ns, nh; REAL *stencil, *H, *pinv;
  /* boxes */
  int nx, ny, nz, izmin; REAL bx, by, bz;
  struct sf_region **box;
  /* layers */
  int nl, ilmin; REAL hl;
  struct sf_layer **layer;
  /* walls (one per sphere) */
  struct sf_region **wall;
  /* every explicit source, near (outside the grid without a lattice) */
  struct sf_region all;
  REAL all_phi[SF_MAXCLS];
} sf;

static long sf_gen = 1;          /* bumped at each setup (the caches' key) */

/* fields evaluated: by expression key */
struct sf_key { struct expnode *key; int kind, wall; };
static struct sf_key *sf_keys;
static int sf_nkeys, sf_capkeys;
static long sf_keys_load = -2;

static REAL sf_floor(void)
{ static long load = -2;
  static int k = -1;
  if ( load != pyse_load_serial() ) { k = lookup_global("pyse_angle_floor"); load = pyse_load_serial(); }
  return k >= 0 ? globals(k)->value.real : 0.0;
}

static REAL sf_coef(REAL angle, REAL tension, REAL floor)
{ if ( angle > 0 && angle < SF_FLOOR_TOP && floor > angle ) angle = floor;
  return -tension*cos(angle*M_PI/180.0);
}

static void sf_free(void)
{ int i;
  if ( sf.box )
  { for ( i = 0 ; i < sf.nx*sf.ny*sf.nz ; i++ )
      if ( sf.box[i] ) { free(sf.box[i]->near); free(sf.box[i]->phi); free(sf.box[i]); }
    free(sf.box);
  }
  if ( sf.layer )
  { for ( i = 0 ; i < sf.nl ; i++ )
      if ( sf.layer[i] ) { free(sf.layer[i]->phi); free(sf.layer[i]); }
    free(sf.layer);
  }
  if ( sf.wall )
  { for ( i = 0 ; i < sf.nbead ; i++ )
      if ( sf.wall[i] ) { free(sf.wall[i]->near); free(sf.wall[i]->phi); free(sf.wall[i]->A);
                          free(sf.wall[i]); }
    free(sf.wall);
  }
  free(sf.exp); free(sf.base); free(sf.bead); free(sf.R);
  free(sf.stencil); free(sf.H); free(sf.pinv);
  sf.box = NULL; sf.layer = NULL; sf.wall = NULL; sf.exp = sf.base = NULL;
  sf.bead = NULL; sf.R = NULL; sf.stencil = sf.H = sf.pinv = NULL;
  sf.on = 0;
}

/* ---- Ewald: the field sum_n (w - n)/|w - n|^3 over the lattice ---------- */

static void sf_ewald(const REAL *w0, REAL *E)
{ REAL w[3], Lx = sf.L[0], Ly = sf.L[1], A = Lx*Ly, a = sf.alpha, z;
  int i, j;
  w[0] = w0[0] - Lx*floor(w0[0]/Lx + 0.5);
  w[1] = w0[1] - Ly*floor(w0[1]/Ly + 0.5);
  w[2] = w0[2];
  E[0] = E[1] = E[2] = 0.0;
  for ( i = -sf.nr ; i <= sf.nr ; i++ )
    for ( j = -sf.nr ; j <= sf.nr ; j++ )
    { REAL d0 = w[0] - i*Lx, d1 = w[1] - j*Ly, d2 = w[2];
      REAL r2 = d0*d0 + d1*d1 + d2*d2, r = sqrt(r2);
      REAL f = erfc(a*r)/(r2*r) + 2.0*a/sqrt(M_PI)*exp(-a*a*r2)/r2;
      E[0] += d0*f; E[1] += d1*f; E[2] += d2*f;
    }
  z = w[2];
  for ( i = -sf.nk ; i <= sf.nk ; i++ )
    for ( j = -sf.nk ; j <= sf.nk ; j++ )
    { REAL g0, g1, g, ph, a1, a2, t1, t2, S, dS, c;
      if ( i == 0 && j == 0 ) continue;
      g0 = 2*M_PI*i/Lx; g1 = 2*M_PI*j/Ly; g = sqrt(g0*g0 + g1*g1);
      ph = w[0]*g0 + w[1]*g1;
      a1 = g/(2*a) + a*z; a2 = g/(2*a) - a*z;
      t1 = a1 > 26.0 ? 0.0 : exp(g*z)*erfc(a1);
      t2 = a2 > 26.0 ? 0.0 : exp(-g*z)*erfc(a2);
      S = t1 + t2;
      dS = g*(t1 - t2);       /* (the Gaussians' parts, e^{gz - a1^2} = e^{-gz - a2^2}, cancel) */
      c = M_PI/(A*g);
      E[0] += c*sin(ph)*g0*S;
      E[1] += c*sin(ph)*g1*S;
      E[2] -= c*cos(ph)*dS;
    }
  E[2] += 2*M_PI/A*erf(a*z);
}

/* ---- polynomials ----------------------------------------------------------- */

/* Taylor coefficients (in t = x - c) of 1/|x - y| about c, to degree p, added
   times q into out[NMONO(p)], scaled to u = t/rho */
static void sf_taylor_add(const REAL *c, const REAL *y, int p, REAL rho, const REAL *q, int ncls,
                          REAL *out, int stride, REAL *work)
{ REAL d[3] = { c[0] - y[0], c[1] - y[1], c[2] - y[2] };
  REAL r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
  int k, i, n = NMONO(p), cl;
  work[0] = 1.0/sqrt(r2);
  for ( k = 1 ; k < n ; k++ )
  { int a[3] = { sf_mono[k][0], sf_mono[k][1], sf_mono[k][2] }, deg = a[0] + a[1] + a[2];
    REAL s1 = 0.0, s2 = 0.0;
    for ( i = 0 ; i < 3 ; i++ )
    { if ( a[i] >= 1 )
      { a[i]--; s1 += d[i]*work[sf_mono_idx[a[0]][a[1]][a[2]]]; a[i]++; }
      if ( a[i] >= 2 )
      { a[i] -= 2; s2 += work[sf_mono_idx[a[0]][a[1]][a[2]]]; a[i] += 2; }
    }
    work[k] = (-(2*deg - 1)*s1 - (deg - 1)*s2)/(deg*r2);
  }
  { REAL sc = 1.0;
    int deg = 0;
    for ( k = 0 ; k < n ; k++ )
    { int dk = sf_mono[k][0] + sf_mono[k][1] + sf_mono[k][2];
      while ( deg < dk ) { sc *= rho; deg++; }
      for ( cl = 0 ; cl < ncls ; cl++ )
        if ( q[cl] != 0.0 ) out[cl*stride + k] += q[cl]*work[k]*sc;
    }
  }
}

/* the derivatives of a polynomial: for each multi-index b (|b| <= 3, by
   sf_mono_idx) the monomials m >= b of degree <= SF_PMAX, by degree: m's
   index, (m - b)'s, and the falling factorials' product */
struct sf_term { int k, r; REAL f; };
static struct sf_term *sf_dlist[20];
static int sf_dcount[20][SF_PMAX + 2];      /* terms with degree(m) <= p: [b][p + 1] */
static int sf_parent[(SF_PMAX + 1)*(SF_PMAX + 2)*(SF_PMAX + 3)/6][2];  /* m = parent + e_axis */

static int sf_dlists(void)
{ int b, k, n = NMONO(SF_PMAX);
  if ( sf_dlist[0] ) return 1;
  for ( k = 1 ; k < n ; k++ )
  { int a = sf_mono[k][0] ? 0 : sf_mono[k][1] ? 1 : 2, m[3] = { sf_mono[k][0], sf_mono[k][1],
                                                                 sf_mono[k][2] };
    m[a]--;
    sf_parent[k][0] = sf_mono_idx[m[0]][m[1]][m[2]]; sf_parent[k][1] = a;
  }
  for ( b = 0 ; b < 20 ; b++ )
  { int b0 = sf_mono[b][0], b1 = sf_mono[b][1], b2 = sf_mono[b][2], cnt = 0, deg;
    struct sf_term *t = (struct sf_term*)malloc((size_t)n*sizeof(struct sf_term));
    if ( !t ) return 0;
    for ( deg = 0 ; deg <= SF_PMAX ; deg++ )
    { for ( k = sf_mono_n[deg] ; k < sf_mono_n[deg + 1] ; k++ )
      { int m0 = sf_mono[k][0], m1 = sf_mono[k][1], m2 = sf_mono[k][2], q;
        REAL f = 1.0;
        if ( b0 > m0 || b1 > m1 || b2 > m2 ) continue;
        for ( q = 0 ; q < b0 ; q++ ) f *= m0 - q;
        for ( q = 0 ; q < b1 ; q++ ) f *= m1 - q;
        for ( q = 0 ; q < b2 ; q++ ) f *= m2 - q;
        t[cnt].k = k; t[cnt].r = sf_mono_idx[m0-b0][m1-b1][m2-b2]; t[cnt].f = f;
        cnt++;
      }
      sf_dcount[b][deg + 1] = cnt;
    }
    sf_dlist[b] = t;
  }
  return 1;
}

/* the monomials' values at u, to degree p */
static void sf_mono_values(const REAL *u, int p, REAL *mv)
{ int k, n = NMONO(p);
  mv[0] = 1.0;
  for ( k = 1 ; k < n ; k++ ) mv[k] = mv[sf_parent[k][0]]*u[sf_parent[k][1]];
}

/* the derivatives of sum_k a[k] u^mono[k] (degree p) up to order `order`
   from the monomials' values mv: D[] indexed by sf_mono_idx of the
   multi-index (up to order 3) */
static void sf_poly_derivs(const REAL *a, int p, const REAL *mv, int order, REAL *D)
{ int b, nd = NMONO(order);
  for ( b = 0 ; b < nd ; b++ )
  { const struct sf_term *t = sf_dlist[b];
    int i, n = sf_dcount[b][p + 1];
    REAL sum = 0.0;
    for ( i = 0 ; i < n ; i++ ) sum += t[i].f*a[t[i].k]*mv[t[i].r];
    D[b] = sum;
  }
}

static REAL sf_binom[SF_PMAX + 1][SF_PMAX + 1];

/* the polynomial a (degree pa, u_a = (x - ca)/rhoa) re-expanded about c (u =
   (x - c)/rho), to degree p, added to out */
static void sf_shift_add(const REAL *a, int pa, const REAL *ca, REAL rhoa, const REAL *c, REAL rho,
                         int p, REAL *out)
{ REAL s = rho/rhoa, e[3], ep[3][SF_PMAX + 1], sp[SF_PMAX + 1];
  int k, i, n = NMONO(pa);
  for ( i = 0 ; i < 3 ; i++ )
  { int t;
    e[i] = (c[i] - ca[i])/rhoa;
    ep[i][0] = 1.0;
    for ( t = 1 ; t <= pa ; t++ ) ep[i][t] = ep[i][t-1]*e[i];
  }
  sp[0] = 1.0;
  for ( i = 1 ; i <= pa ; i++ ) sp[i] = sp[i-1]*s;
  for ( k = 0 ; k < n ; k++ )
  { int m0 = sf_mono[k][0], m1 = sf_mono[k][1], m2 = sf_mono[k][2], b0, b1, b2;
    REAL ak = a[k];
    if ( ak == 0.0 ) continue;
    for ( b0 = 0 ; b0 <= m0 ; b0++ )
      for ( b1 = 0 ; b1 <= m1 && b0 + b1 <= p ; b1++ )
        for ( b2 = 0 ; b2 <= m2 && b0 + b1 + b2 <= p ; b2++ )
          out[sf_mono_idx[b0][b1][b2]] += ak*sf_binom[m0][b0]*sf_binom[m1][b1]*sf_binom[m2][b2]
            *ep[0][m0-b0]*ep[1][m1-b1]*ep[2][m2-b2]*sp[b0+b1+b2];
  }
}

/* ---- the tail's layers ---------------------------------------------------- */

static struct sf_layer *sf_layer_make(int il)
{ struct sf_layer *t = (struct sf_layer*)calloc(1,sizeof(*t));
  int nm = NMONO(sf.pT), k, m, cl, ns = sf.ns;
  REAL *samples, *coef, E[3];
  if ( !t ) return NULL;
  t->c[0] = sf.ox + sf.L[0]/2; t->c[1] = sf.oy + sf.L[1]/2; t->c[2] = sf.z0 + (il + 0.5)*sf.hl;
  t->rho = 1.05*sqrt(sf.L[0]*sf.L[0]/4 + sf.L[1]*sf.L[1]/4 + sf.hl*sf.hl/4);
  t->phi = (REAL*)calloc((size_t)sf.ncls*nm,sizeof(REAL));
  samples = (REAL*)calloc((size_t)sf.ncls*3*ns,sizeof(REAL));
  coef = (REAL*)calloc((size_t)sf.nh,sizeof(REAL));
  if ( !t->phi || !samples || !coef ) { free(t->phi); free(t); free(samples); free(coef); return NULL; }
  /* samples of -F (F = -grad Phi: the fit is of grad Phi) times rho (in u) */
  for ( k = 0 ; k < ns ; k++ )
  { REAL x[3];
    int s;
    for ( m = 0 ; m < 3 ; m++ ) x[m] = t->c[m] + t->rho*sf.stencil[3*k+m];
    for ( s = 0 ; s < sf.nbase ; s++ )
    { REAL w[3];
      for ( m = 0 ; m < 3 ; m++ ) w[m] = x[m] - sf.base[s].c[m];
      sf_ewald(w,E);
      for ( cl = 0 ; cl < sf.ncls ; cl++ )
        if ( sf.base[s].q[cl] != 0.0 )
          for ( m = 0 ; m < 3 ; m++ ) samples[cl*3*ns + 3*k + m] -= sf.base[s].q[cl]*E[m]*t->rho;
    }
    for ( s = 0 ; s < sf.nexp ; s++ )
    { REAL w[3], r2, ir3;
      for ( m = 0 ; m < 3 ; m++ ) w[m] = x[m] - sf.exp[s].c[m];
      r2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
      ir3 = 1.0/(r2*sqrt(r2));
      for ( cl = 0 ; cl < sf.ncls ; cl++ )
        if ( sf.exp[s].q[cl] != 0.0 )
          for ( m = 0 ; m < 3 ; m++ ) samples[cl*3*ns + 3*k + m] += sf.exp[s].q[cl]*w[m]*ir3*t->rho;
    }
  }
  for ( cl = 0 ; cl < sf.ncls ; cl++ )
  { int h;
    for ( h = 0 ; h < sf.nh ; h++ )
    { REAL sum = 0.0;
      for ( k = 0 ; k < 3*ns ; k++ ) sum += sf.pinv[(size_t)h*3*ns + k]*samples[cl*3*ns + k];
      coef[h] = sum;
    }
    for ( k = 0 ; k < nm ; k++ )
    { REAL sum = 0.0;
      for ( h = 0 ; h < sf.nh ; h++ ) sum += sf.H[(size_t)k*sf.nh + h]*coef[h];
      t->phi[cl*nm + k] = sum;
    }
  }
  free(samples); free(coef);
  return t;
}

static struct sf_layer *sf_layer_at(REAL z)
{ int il = (int)floor((z - sf.z0)/sf.hl) - sf.ilmin;
  struct sf_layer *t;
  if ( il < 0 ) il = 0;
  if ( il >= sf.nl ) il = sf.nl - 1;
  t = __atomic_load_n(&sf.layer[il],__ATOMIC_ACQUIRE);
  if ( t ) return t;
#ifdef _OPENMP
#pragma omp critical(pyse_slabflux_layer)   /* (a box's lock is held: another name) */
#endif
  { t = sf.layer[il];
    if ( !t )
    { t = sf_layer_make(il + sf.ilmin);
      __atomic_store_n(&sf.layer[il],t,__ATOMIC_RELEASE);
    }
  }
  return t;
}

/* ---- regions --------------------------------------------------------------- */

/* a region about c (radius r: the points it serves): near sources within
   kappa r directly, the others by its polynomial; skip: the bead whose own
   source (n = 0, not the mirror) it leaves out (-1: none) */
static struct sf_region *sf_region_make(const REAL *c, REAL r, REAL kappa, int p, int skip)
{ struct sf_region *g = (struct sf_region*)calloc(1,sizeof(*g));
  int nm = NMONO(p), s, cl, cap = 64;
  REAL *work = (REAL*)malloc((size_t)NMONO(SF_PMAX)*sizeof(REAL)), D = kappa*r;
  struct sf_layer *t;
  if ( !g || !work ) { free(g); free(work); return NULL; }
  memcpy(g->c,c,sizeof(g->c));
  g->rho = r; g->p = p;
  g->phi = (REAL*)calloc((size_t)sf.ncls*nm,sizeof(REAL));
  g->near = (struct sf_src*)malloc(cap*sizeof(struct sf_src));
  if ( !g->phi || !g->near ) goto fail;
  for ( s = 0 ; s < sf.nexp ; s++ )
  { const struct sf_src *e = sf.exp + s;
    REAL w[3] = { e->c[0] - c[0], e->c[1] - c[1], e->c[2] - c[2] };
    if ( skip >= 0 && e->bead == skip && e->own ) continue;
    if ( w[0]*w[0] + w[1]*w[1] + w[2]*w[2] < D*D )
    { if ( g->nnear >= cap )
      { struct sf_src *nn = (struct sf_src*)realloc(g->near,2*cap*sizeof(struct sf_src));
        if ( !nn ) goto fail;
        g->near = nn; cap *= 2;
      }
      g->near[g->nnear++] = *e;
    }
    else sf_taylor_add(c,e->c,p,r,e->q,sf.ncls,g->phi,nm,work);
  }
  /* the tail, from the layer at the region's height */
  if ( sf.lattice )
  { t = sf_layer_at(c[2]);
    if ( !t ) goto fail;
    for ( cl = 0 ; cl < sf.ncls ; cl++ )
      sf_shift_add(t->phi + cl*NMONO(sf.pT),sf.pT,t->c,t->rho,c,r,p,g->phi + cl*nm);
  }
  free(work);
  return g;
fail:
  free(work); free(g->phi); free(g->near); free(g);
  return NULL;
}

static struct sf_region *sf_box_at(const REAL *x, REAL *xc)
{ int ix, iy, iz, i;
  struct sf_region *g;
  xc[0] = x[0]; xc[1] = x[1]; xc[2] = x[2];
  if ( sf.lattice )
  { xc[0] -= sf.L[0]*floor((x[0] - sf.ox)/sf.L[0]);
    xc[1] -= sf.L[1]*floor((x[1] - sf.oy)/sf.L[1]);
  }
  ix = (int)floor((xc[0] - sf.ox)/sf.bx);
  iy = (int)floor((xc[1] - sf.oy)/sf.by);
  if ( sf.lattice && ix == sf.nx && sf.nx*sf.bx >= sf.L[0]*(1 - 1e-12) ) ix = sf.nx - 1;
  if ( sf.lattice && iy == sf.ny && sf.ny*sf.by >= sf.L[1]*(1 - 1e-12) ) iy = sf.ny - 1;
  if ( ix < 0 || ix >= sf.nx || iy < 0 || iy >= sf.ny ) return NULL;
  iz = (int)floor((xc[2] - sf.z0)/sf.bz) - sf.izmin;
  if ( iz < 0 || iz >= sf.nz ) return NULL;
  i = (iz*sf.ny + iy)*sf.nx + ix;
  g = __atomic_load_n(&sf.box[i],__ATOMIC_ACQUIRE);
  if ( g ) return g;
#ifdef _OPENMP
#pragma omp critical(pyse_slabflux_make)
#endif
  { g = sf.box[i];
    if ( !g )
    { REAL c[3] = { sf.ox + (ix + 0.5)*sf.bx, sf.oy + (iy + 0.5)*sf.by,
                    sf.z0 + (iz + sf.izmin + 0.5)*sf.bz };
      REAL r = 0.5*sqrt(sf.bx*sf.bx + sf.by*sf.by + sf.bz*sf.bz);
      g = sf_region_make(c,r,sf.kappa,sf.p,-1);
      __atomic_store_n(&sf.box[i],g,__ATOMIC_RELEASE);
    }
  }
  return g;
}

/* sphere j's region: its polynomial turned into A's components */
static struct sf_region *sf_wall_make(int j)
{ REAL r = 1.02*sf.R[j];
  struct sf_region *g = sf_region_make(sf.bead[j],r,sf.kappa_w,sf.pw,j);
  int nm, cl, k;
  if ( !g ) return NULL;
  nm = NMONO(sf.pw);
  g->A = (REAL*)calloc((size_t)sf.ncls*3*nm,sizeof(REAL));
  if ( !g->A ) { free(g->near); free(g->phi); free(g); return NULL; }
  /* A = -sum_l (x - c) x grad Phi_l/(l + 1) = -sum_l u x grad_u Phi_l/(l + 1)
     (rho cancels): for the monomial u^m, grad = (m0 u^(m-e0), ...) */
  for ( cl = 0 ; cl < sf.ncls ; cl++ )
    for ( k = 1 ; k < nm ; k++ )
    { int m[3] = { sf_mono[k][0], sf_mono[k][1], sf_mono[k][2] }, l = m[0] + m[1] + m[2], i;
      REAL a = g->phi[cl*nm + k];
      if ( a == 0.0 ) continue;
      /* (u x g)_0 = u1 g2 - u2 g1, etc.; u_i d_j u^m = m_j u^(m - e_j + e_i) */
      { static const int cr[3][2] = { {1,2}, {2,0}, {0,1} };
        for ( i = 0 ; i < 3 ; i++ )
        { int p1 = cr[i][0], p2 = cr[i][1], e[3];
          /* + u_p1 d_p2 */
          if ( m[p2] > 0 )
          { e[0] = m[0]; e[1] = m[1]; e[2] = m[2]; e[p2]--; e[p1]++;
            g->A[(cl*3 + i)*nm + sf_mono_idx[e[0]][e[1]][e[2]]] -= a*m[p2]/(l + 1);
          }
          /* - u_p2 d_p1 */
          if ( m[p1] > 0 )
          { e[0] = m[0]; e[1] = m[1]; e[2] = m[2]; e[p1]--; e[p2]++;
            g->A[(cl*3 + i)*nm + sf_mono_idx[e[0]][e[1]][e[2]]] += a*m[p1]/(l + 1);
          }
        }
      }
    }
  return g;
}

/* ---- setup ----------------------------------------------------------------- */

/* L (Lx, Ly); zf the floor's height (NaN: no floor, no mirrors); beads n x 6
   (centre, R, angle (NaN: no energy), tension); the floor's angle (NaN: no
   wetting) and tension; prm: h, kappa, p, kappa_w, p_w, M, pT; the tail's fit:
   stencil ns x 3 (unit sphere), H NMONO(pT) x nh, pinv nh x 3ns. 1 if made. */
int pyse_slabflux_setup(const double *L, double zf, int nbead, const double *beads,
                        double floor_angle, double floor_tension, const double *prm,
                        int ns, const double *stencil, int nh, const double *H,
                        const double *pinv, int nsrc, const double *src, const double *grid)
{ int i, j, k, cl;
  REAL zmin = HUGE_VAL, zmax = -HUGE_VAL, Lmin;
  sf_monos();
  if ( !sf_dlists() ) return 0;
  for ( i = 0 ; i <= SF_PMAX ; i++ )
    for ( j = 0 ; j <= SF_PMAX ; j++ )
      sf_binom[i][j] = j > i ? 0.0 : (j == 0 || j == i) ? 1.0 : sf_binom[i-1][j-1] + sf_binom[i-1][j];
  sf_free();
  sf.load = pyse_load_serial();
  sf.L[0] = L[0]; sf.L[1] = L[1];
  sf.lattice = L[0] > 0 && L[1] > 0;
  sf.zf = zf; sf.mirror = !isnan(zf);
  sf.h = prm[0]; sf.kappa = prm[1]; sf.p = (int)prm[2]; sf.kappa_w = prm[3]; sf.pw = (int)prm[4];
  sf.M = sf.lattice ? (int)prm[5] : 0; sf.pT = (int)prm[6];
  if ( sf.p > SF_PMAX || sf.pw > SF_PMAX || sf.pT > SF_PMAX || nbead <= 0
       || NMONO(sf.pT)*nh <= 0 ) return 0;
  sf.nbead = nbead;
  sf.bead = (REAL(*)[3])malloc(nbead*sizeof(*sf.bead));
  sf.R = (REAL*)malloc(nbead*sizeof(REAL));
  if ( !sf.bead || !sf.R ) { sf_free(); return 0; }
  /* classes */
  sf.nen = 0;
  for ( i = 0 ; i < nbead ; i++ )
  { const double *b = beads + 6*i;
    for ( k = 0 ; k < 3 ; k++ ) sf.bead[i][k] = b[k];
    sf.R[i] = b[3];
    if ( !isnan(b[4]) && cos(b[4]*M_PI/180.0) != 0 )
    { for ( cl = 0 ; cl < sf.nen ; cl++ )
        if ( sf.cls_angle[cl] == b[4] && sf.cls_tension[cl] == b[5] ) break;
      if ( cl == sf.nen )
      { if ( sf.nen >= SF_MAXCLS - 1 ) { sf_free(); return 0; }
        sf.cls_angle[cl] = b[4]; sf.cls_tension[cl] = b[5]; sf.nen++;
      }
    }
    if ( b[2] - b[3] < zmin ) zmin = b[2] - b[3];
    if ( b[2] + b[3] > zmax ) zmax = b[2] + b[3];
  }
  sf.ncls = sf.nen + 1;
  sf.floor_energy = !isnan(floor_angle) && cos(floor_angle*M_PI/180.0) != 0;
  sf.floor_angle = floor_angle; sf.floor_tension = floor_tension;
  /* the base sources (given: x, y, z, bead, own; else each bead's and its
     mirror in the floor), and the explicit images */
  sf.nbase = nsrc > 0 ? nsrc : nbead*(sf.mirror ? 2 : 1);
  sf.base = (struct sf_src*)calloc(sf.nbase,sizeof(struct sf_src));
  sf.nexp = sf.nbase*(2*sf.M + 1)*(2*sf.M + 1);
  sf.exp = (struct sf_src*)calloc(sf.nexp,sizeof(struct sf_src));
  if ( !sf.base || !sf.exp ) { sf_free(); return 0; }
  for ( i = 0, k = 0 ; i < (nsrc > 0 ? nsrc : nbead) ; i++ )
  { int bi = nsrc > 0 ? (int)src[5*i+3] : i, mi;
    const double *b;
    if ( bi < 0 || bi >= nbead ) { sf_free(); return 0; }
    b = beads + 6*bi;
    for ( mi = 0 ; mi < (nsrc > 0 ? 1 : sf.mirror ? 2 : 1) ; mi++, k++ )
    { struct sf_src *s = sf.base + k;
      if ( nsrc > 0 )
      { s->c[0] = src[5*i]; s->c[1] = src[5*i+1]; s->c[2] = src[5*i+2];
        s->own = src[5*i+4] != 0;
      }
      else
      { s->c[0] = b[0]; s->c[1] = b[1]; s->c[2] = mi ? 2*zf - b[2] : b[2];
        s->own = !mi;
      }
      s->bead = bi;
      s->K2 = SF_CORE*SF_CORE*b[3]*b[3]; s->iK3 = 1.0/(s->K2*SF_CORE*b[3]);
      s->q[sf.nen] = -b[3]*b[3]*b[3]/3.0;
      if ( !isnan(b[4]) && cos(b[4]*M_PI/180.0) != 0 )
      { for ( cl = 0 ; cl < sf.nen ; cl++ )
          if ( sf.cls_angle[cl] == b[4] && sf.cls_tension[cl] == b[5] ) break;
        s->q[cl] = b[3]*b[3];
      }
    }
  }
  for ( k = 0, i = 0 ; i < sf.nbase ; i++ )
  { int a, b;
    for ( a = -sf.M ; a <= sf.M ; a++ )
      for ( b = -sf.M ; b <= sf.M ; b++, k++ )
      { sf.exp[k] = sf.base[i];
        sf.exp[k].c[0] += a*sf.L[0]; sf.exp[k].c[1] += b*sf.L[1];
        sf.exp[k].own = sf.base[i].own && a == 0 && b == 0;
      }
  }
  /* Ewald: erfc(alpha r) and the k-space terms down to ~1e-17 */
  Lmin = sf.lattice ? (sf.L[0] < sf.L[1] ? sf.L[0] : sf.L[1]) : grid[2] < grid[3] ? grid[2] : grid[3];
  if ( sf.lattice )
  { sf.alpha = sqrt(M_PI/(sf.L[0]*sf.L[1]));
    sf.nr = (int)ceil(6.0/(sf.alpha*Lmin) + 0.5);
    sf.nk = (int)ceil(12.0*sf.alpha*(sf.L[0] > sf.L[1] ? sf.L[0] : sf.L[1])/(2*M_PI));
  }
  /* the tail's fit */
  sf.ns = ns; sf.nh = nh;
  sf.stencil = (REAL*)malloc((size_t)3*ns*sizeof(REAL));
  sf.H = (REAL*)malloc((size_t)NMONO(sf.pT)*nh*sizeof(REAL));
  sf.pinv = (REAL*)malloc((size_t)nh*3*ns*sizeof(REAL));
  if ( !sf.stencil || !sf.H || !sf.pinv ) { sf_free(); return 0; }
  for ( i = 0 ; i < 3*ns ; i++ ) sf.stencil[i] = stencil[i];
  for ( i = 0 ; i < NMONO(sf.pT)*nh ; i++ ) sf.H[i] = H[i];
  for ( i = 0 ; i < nh*3*ns ; i++ ) sf.pinv[i] = pinv[i];
  /* boxes: the cell in x and y, layers of 2h from below the lowest bead (or
     the floor) to well above the highest */
  /* (the grid: grid[0..1] its corner, [2..3] its size in x and y; with a
     lattice, points are taken into it by whole periods) */
  sf.ox = grid[0]; sf.oy = grid[1];
  sf.nx = (int)ceil(grid[2]/(2*sf.h)); sf.ny = (int)ceil(grid[3]/(2*sf.h));
  sf.bx = grid[2]/sf.nx; sf.by = grid[3]/sf.ny; sf.bz = 2*sf.h;
  sf.z0 = sf.mirror ? zf : zmin;
  sf.izmin = (int)floor((zmin - sf.z0)/sf.bz) - 2;
  if ( sf.izmin > -2 ) sf.izmin = -2;
  sf.nz = (int)ceil((zmax - sf.z0)/sf.bz) - sf.izmin + (int)ceil(4*Lmin/sf.bz) + 8;
  sf.box = (struct sf_region**)calloc((size_t)sf.nx*sf.ny*sf.nz,sizeof(*sf.box));
  sf.hl = Lmin;
  sf.ilmin = (int)floor((zmin - sf.z0)/sf.hl) - 1;
  if ( sf.ilmin > -1 ) sf.ilmin = -1;
  sf.nl = (int)ceil((zmax - sf.z0)/sf.hl) - sf.ilmin + 6;
  sf.layer = (struct sf_layer**)calloc(sf.nl,sizeof(*sf.layer));
  sf.wall = (struct sf_region**)calloc(nbead,sizeof(*sf.wall));
  if ( !sf.box || !sf.layer || !sf.wall ) { sf_free(); return 0; }
  for ( i = 0 ; i < nbead ; i++ )
    if ( !(sf.wall[i] = sf_wall_make(i)) ) { sf_free(); return 0; }
  memset(&sf.all,0,sizeof(sf.all));
  memset(sf.all_phi,0,sizeof(sf.all_phi));
  sf.all.rho = 1.0; sf.all.p = 0; sf.all.nnear = sf.nexp; sf.all.near = sf.exp;
  sf.all.phi = sf.all_phi;
  sf.on = 1;
  sf_gen++;
  return 1;
}

/* a field to evaluate: a method instance's (method) or a constraint's energy
   (which 0) or content (1) integrand; kind 0 the facets' volume, 1 their
   wetting, 2 sphere wall's energy, 3 its content. 1 if registered. */
int pyse_slabflux_key(const char *method, int con, int which, int kind, int wall)
{ struct expnode *key = NULL;
  int i;
  if ( !sf.on || sf.load != pyse_load_serial() ) return 0;
  if ( sf_keys_load != pyse_load_serial() ) { sf_nkeys = 0; sf_keys_load = pyse_load_serial(); }
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
  if ( !key || (kind >= 2 && (wall < 0 || wall >= sf.nbead)) ) return 0;
  for ( i = 0 ; i < sf_nkeys ; i++ )
    if ( sf_keys[i].key == key ) break;
  if ( i == sf_nkeys )
  { if ( sf_nkeys >= sf_capkeys )
    { int cap = sf_capkeys ? 2*sf_capkeys : 32;
      struct sf_key *n = (struct sf_key*)realloc(sf_keys,cap*sizeof(*n));
      if ( !n ) return 0;
      sf_keys = n; sf_capkeys = cap;
    }
    sf_nkeys++;
  }
  sf_keys[i].key = key; sf_keys[i].kind = kind; sf_keys[i].wall = wall;
  return 1;
}

int pyse_slabflux_active(void)
{ return sf.on && sf.load == pyse_load_serial() && sf_keys_load == pyse_load_serial()
         && sf_nkeys > 0;
}

/* ---- evaluation ------------------------------------------------------------ */

/* index of a symmetric multi-index's derivative in D[] (sf_poly_derivs) */
#define DI(a,b,c) sf_mono_idx[a][b][c]

/* the class fields F_cl at x (box region), with derivatives: V[cl][m],
   G[cl][m][i], S[cl][m][i][j] */
static int sf_fields(const REAL *x, int order, REAL V[][3], REAL G[][3][3], REAL S[][3][3][3])
{ REAL xc[3], u[3], D[20], mv[(SF_PMAX + 1)*(SF_PMAX + 2)*(SF_PMAX + 3)/6];
  struct sf_region *g = sf_box_at(x,xc);
  int cl, s, m, i, j, nm;
  if ( !g && !sf.lattice )
  { g = &sf.all;               /* (outside the grid, no lattice: every source) */
    for ( m = 0 ; m < 3 ; m++ ) xc[m] = x[m];
  }
  if ( !g ) return 0;
  memset(V,0,sf.ncls*sizeof(*V));
  if ( order >= 1 ) memset(G,0,sf.ncls*sizeof(*G));
  if ( order >= 2 ) memset(S,0,sf.ncls*sizeof(*S));
  /* near sources (their own images: x in the cell, they around it) */
  for ( s = 0 ; s < g->nnear ; s++ )
  { const struct sf_src *e = g->near + s;
    REAL w[3] = { xc[0] - e->c[0], xc[1] - e->c[1], xc[2] - e->c[2] };
    REAL r2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2], a, b = 0.0, f = 0.0;
    REAL F[3], dF[3][3], d2F[3][3][3];
    int in = r2 < e->K2;
    if ( in ) a = e->iK3;
    else { REAL r = sqrt(r2); a = 1.0/(r2*r); b = 3.0*a/r2; f = 15.0*a/(r2*r2); }
    for ( m = 0 ; m < 3 ; m++ ) F[m] = a*w[m];
    if ( order >= 1 )
      for ( m = 0 ; m < 3 ; m++ )
        for ( i = 0 ; i < 3 ; i++ ) dF[m][i] = (m == i ? a : 0.0) - b*w[m]*w[i];
    if ( order >= 2 )
      for ( m = 0 ; m < 3 ; m++ )
        for ( i = 0 ; i < 3 ; i++ )
          for ( j = 0 ; j < 3 ; j++ )
            d2F[m][i][j] = -b*((m == i ? w[j] : 0.0) + (m == j ? w[i] : 0.0)
                               + (i == j ? w[m] : 0.0)) + f*w[m]*w[i]*w[j];
    for ( cl = 0 ; cl < sf.ncls ; cl++ )
    { REAL q = e->q[cl];
      if ( q == 0.0 ) continue;
      for ( m = 0 ; m < 3 ; m++ )
      { V[cl][m] += q*F[m];
        if ( order >= 1 ) for ( i = 0 ; i < 3 ; i++ ) G[cl][m][i] += q*dF[m][i];
        if ( order >= 2 )
          for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ ) S[cl][m][i][j] += q*d2F[m][i][j];
      }
    }
  }
  /* the far part: F = -grad Phi, derivatives in u = (x - c)/rho */
  nm = NMONO(g->p);
  for ( m = 0 ; m < 3 ; m++ ) u[m] = (xc[m] - g->c[m])/g->rho;
  sf_mono_values(u,g->p,mv);
  for ( cl = 0 ; cl < sf.ncls ; cl++ )
  { REAL r1 = -1.0/g->rho, r2 = r1/g->rho, r3 = r2/g->rho;
    sf_poly_derivs(g->phi + cl*nm,g->p,mv,order + 1,D);
    for ( m = 0 ; m < 3 ; m++ )
    { int e[3] = { 0, 0, 0 };
      e[m]++;
      V[cl][m] += r1*D[DI(e[0],e[1],e[2])];
      if ( order >= 1 )
        for ( i = 0 ; i < 3 ; i++ )
        { e[i]++;
          G[cl][m][i] += r2*D[DI(e[0],e[1],e[2])];
          if ( order >= 2 )
            for ( j = 0 ; j < 3 ; j++ )
            { e[j]++;
              S[cl][m][i][j] += r3*D[DI(e[0],e[1],e[2])];
              e[j]--;
            }
          e[i]--;
        }
    }
  }
  return 1;
}

/* sphere j's class primitives A_cl at x: near monopoles and the region's A */
static void sf_wall_fields(int j, const REAL *x0, int order, REAL V[][3], REAL G[][3][3],
                           REAL S[][3][3][3])
{ struct sf_region *g = sf.wall[j];
  const REAL *c = sf.bead[j];
  REAL x[3], u[3], D[10], mv[(SF_PMAX + 1)*(SF_PMAX + 2)*(SF_PMAX + 3)/6];
  int s, cl, m, i, k, nm = NMONO(g->p);
  /* (the image of x nearest the sphere: its vertices are kept there) */
  for ( m = 0 ; m < 3 ; m++ ) x[m] = x0[m];
  if ( sf.lattice )
    for ( m = 0 ; m < 2 ; m++ ) x[m] = c[m] + (x[m] - c[m]) - sf.L[m]*floor((x[m] - c[m])/sf.L[m] + 0.5);
  memset(V,0,sf.ncls*sizeof(*V));
  if ( order >= 1 ) memset(G,0,sf.ncls*sizeof(*G));
  if ( order >= 2 ) memset(S,0,sf.ncls*sizeof(*S));
  for ( s = 0 ; s < g->nnear ; s++ )
  { const struct sf_src *e = g->near + s;
    REAL dd[3] = { e->c[0] - c[0], e->c[1] - c[1], e->c[2] - c[2] }, n, w[3], r2, r, dw, Dn, ph;
    REAL C[3], Di[3], Dij[3][3], phi_i[3], phi_ij[3][3], dC[3][3], q;
    n = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
    for ( m = 0 ; m < 3 ; m++ ) { dd[m] /= n; w[m] = x[m] - e->c[m]; }
    r2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2]; r = sqrt(r2);
    dw = dd[0]*w[0] + dd[1]*w[1] + dd[2]*w[2];
    Dn = r*(r - dw); ph = 1.0/Dn;
    C[0] = dd[1]*w[2] - dd[2]*w[1]; C[1] = dd[2]*w[0] - dd[0]*w[2]; C[2] = dd[0]*w[1] - dd[1]*w[0];
    dC[0][0] = 0; dC[0][1] = -dd[2]; dC[0][2] = dd[1];
    dC[1][0] = dd[2]; dC[1][1] = 0; dC[1][2] = -dd[0];
    dC[2][0] = -dd[1]; dC[2][1] = dd[0]; dC[2][2] = 0;
    if ( order >= 1 )
    { REAL qq = dw/r;
      for ( i = 0 ; i < 3 ; i++ ) { Di[i] = (2 - qq)*w[i] - r*dd[i]; phi_i[i] = -Di[i]*ph*ph; }
      if ( order >= 2 )
        for ( i = 0 ; i < 3 ; i++ )
          for ( k = 0 ; k < 3 ; k++ )
          { Dij[i][k] = (i == k ? 2 - qq : 0.0) + qq*w[i]*w[k]/r2 - (w[i]*dd[k] + w[k]*dd[i])/r;
            phi_ij[i][k] = 2*Di[i]*Di[k]*ph*ph*ph - Dij[i][k]*ph*ph;
          }
    }
    for ( cl = 0 ; cl < sf.ncls ; cl++ )
    { if ( (q = e->q[cl]) == 0.0 ) continue;
      for ( m = 0 ; m < 3 ; m++ )
      { V[cl][m] += q*C[m]*ph;
        if ( order >= 1 )
          for ( i = 0 ; i < 3 ; i++ )
          { G[cl][m][i] += q*(dC[m][i]*ph + C[m]*phi_i[i]);
            if ( order >= 2 )
              for ( k = 0 ; k < 3 ; k++ )
                S[cl][m][i][k] += q*(dC[m][i]*phi_i[k] + dC[m][k]*phi_i[i] + C[m]*phi_ij[i][k]);
          }
      }
    }
  }
  for ( m = 0 ; m < 3 ; m++ ) u[m] = (x[m] - g->c[m])/g->rho;
  sf_mono_values(u,g->p,mv);
  for ( cl = 0 ; cl < sf.ncls ; cl++ )
  { REAL r1 = 1.0/g->rho, r2 = r1/g->rho;
    for ( m = 0 ; m < 3 ; m++ )
    { int e[3] = { 0, 0, 0 };
      sf_poly_derivs(g->A + (cl*3 + m)*nm,g->p,mv,order,D);
      V[cl][m] += D[0];
      if ( order >= 1 )
        for ( i = 0 ; i < 3 ; i++ )
        { e[i]++;
          G[cl][m][i] += r1*D[DI(e[0],e[1],e[2])];
          if ( order >= 2 )
            for ( k = 0 ; k < 3 ; k++ )
            { e[k]++;
              S[cl][m][i][k] += r2*D[DI(e[0],e[1],e[2])];
              e[k]--;
            }
          e[i]--;
        }
    }
  }
}

/* the integrand of a registered key at x (Evolver's facet or edge Gauss
   point), with derivatives to order: 1 if it is one of ours */
static int sf_eval(const struct sf_key *k, const REAL *x, int order, REAL *v, REAL g[3][3],
                   REAL h[3][3][3]);

int pyse_slabflux_eval(struct expnode *key, REAL *x, int order, REAL *val, REAL (*d)[MAXCOORD],
                       REAL ***s)
{ const struct sf_key *k = NULL;
  REAL v[3], g[3][3], h[3][3][3];
  int i, j, m;
  if ( !sf.on || sf_keys_load != pyse_load_serial() || sf.load != pyse_load_serial() ) return 0;
  for ( i = 0 ; i < sf_nkeys ; i++ )
    if ( sf_keys[i].key == key ) { k = sf_keys + i; break; }
  if ( !k ) return 0;
  if ( !sf_eval(k,x,order,v,g,h) ) return 0;
  for ( m = 0 ; m < 3 ; m++ )
  { val[m] = v[m];
    if ( order >= 1 ) for ( i = 0 ; i < 3 ; i++ ) d[m][i] = g[m][i];
    if ( order >= 2 ) for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ ) s[m][i][j] = h[m][i][j];
  }
  return 1;
}

/* checks: field kind (wall) at x, value, first and second derivatives
   (out 3 + 9 + 27) */
int pyse_slabflux_field(int kind, int wall, const double *x, double *out)
{ struct sf_key k;
  REAL v[3], g[3][3], h[3][3][3];
  int i, j, m;
  if ( !sf.on || kind < 0 || kind > 3 || (kind >= 2 && (wall < 0 || wall >= sf.nbead)) ) return 0;
  k.key = NULL; k.kind = kind; k.wall = wall;
  if ( !sf_eval(&k,x,2,v,g,h) ) return 0;
  for ( m = 0 ; m < 3 ; m++ )
  { out[m] = v[m];
    for ( i = 0 ; i < 3 ; i++ )
    { out[3 + 3*m + i] = g[m][i];
      for ( j = 0 ; j < 3 ; j++ ) out[12 + 9*m + 3*i + j] = h[m][i][j];
    }
  }
  return 1;
}

/* the last point's class fields, one for the facets' and one for a wall's:
   the volume's and the wetting's integrands ask at the same points */
struct sf_cached { long gen; int order, wall; REAL x[3];
                   REAL V[SF_MAXCLS][3], G[SF_MAXCLS][3][3], S[SF_MAXCLS][3][3][3]; };
static __thread struct sf_cached sf_cache[2];

static int sf_eval(const struct sf_key *k, const REAL *x, int order, REAL *v, REAL g[3][3],
                   REAL h[3][3][3])
{ REAL coef[SF_MAXCLS], afloor;
  REAL (*V)[3], (*G)[3][3], (*S)[3][3][3];
  struct sf_cached *C = sf_cache + (k->kind >= 2);
  int i, j, m, cl, lo, hi;
  v[0] = v[1] = v[2] = 0.0;
  afloor = sf_floor();
  memset(g,0,9*sizeof(REAL)); memset(h,0,27*sizeof(REAL));
  for ( cl = 0 ; cl < sf.nen ; cl++ ) coef[cl] = sf_coef(sf.cls_angle[cl],sf.cls_tension[cl],afloor);
  coef[sf.nen] = 1.0;
  /* the classes this field takes: the volume's or the wetting's */
  if ( k->kind == 0 || k->kind == 3 ) { lo = sf.nen; hi = sf.nen + 1; }
  else { lo = 0; hi = sf.nen; }
  V = C->V; G = C->G; S = C->S;
  if ( C->gen != sf_gen || C->order < order || C->x[0] != x[0] || C->x[1] != x[1]
       || C->x[2] != x[2] || (k->kind >= 2 && C->wall != k->wall) )
  { if ( k->kind <= 1 )
    { if ( !sf_fields(x,2 < order ? 2 : order,V,G,S) )
      { C->gen = 0;
        kb_error(PYSE_ERR_BAD_ARGUMENT,"The periodic flux gauge's field was asked for far "
                 "above the slab's beads.\n",RECOVERABLE);
        return 0;
      }
    }
    else sf_wall_fields(k->wall,x,order,V,G,S);
    C->gen = sf_gen; C->order = order; C->wall = k->wall;
    C->x[0] = x[0]; C->x[1] = x[1]; C->x[2] = x[2];
  }
  for ( cl = lo ; cl < hi ; cl++ )
    for ( m = 0 ; m < 3 ; m++ )
    { v[m] += coef[cl]*V[cl][m];
      if ( order >= 1 ) for ( i = 0 ; i < 3 ; i++ ) g[m][i] += coef[cl]*G[cl][m][i];
      if ( order >= 2 )
        for ( i = 0 ; i < 3 ; i++ ) for ( j = 0 ; j < 3 ; j++ ) h[m][i][j] += coef[cl]*S[cl][m][i][j];
    }
  /* the bases */
  { REAL kf = sf.floor_energy ? -sf_coef(sf.floor_angle,sf.floor_tension,afloor) : 0.0;
    REAL zf = sf.mirror ? sf.zf : 0.0;
    if ( k->kind == 0 ) { v[2] += x[2] - zf; g[2][2] += 1.0; }
    else if ( k->kind == 1 ) v[2] += -kf;
    else
    { const REAL *c = sf.bead[k->wall];
      REAL X, Y, Z, B[3], dB[3][3], f = k->kind == 2 ? -kf : -zf;
      REAL xx[3];
      for ( m = 0 ; m < 3 ; m++ ) xx[m] = x[m];
      if ( sf.lattice )
        for ( m = 0 ; m < 2 ; m++ )
          xx[m] = c[m] + (x[m] - c[m]) - sf.L[m]*floor((x[m] - c[m])/sf.L[m] + 0.5);
      X = xx[0] - c[0]; Y = xx[1] - c[1]; Z = xx[2] - c[2];
      /* B = (x - c) x z-hat/2 = (Y/2, -X/2, 0) */
      B[0] = Y/2; B[1] = -X/2; B[2] = 0;
      memset(dB,0,sizeof(dB));
      dB[0][1] = 0.5; dB[1][0] = -0.5;
      for ( m = 0 ; m < 3 ; m++ )
      { v[m] += f*B[m];
        for ( i = 0 ; i < 3 ; i++ ) g[m][i] += f*dB[m][i];
      }
      if ( k->kind == 3 )
      { /* the sphere's own part (_flux_sphere): (Z/3 Y, -(c_z + Z/3) X, 0) */
        v[0] += Z/3.0*Y; g[0][1] += Z/3.0; g[0][2] += Y/3.0;
        v[1] += -(c[2] + Z/3.0)*X; g[1][0] += -(c[2] + Z/3.0); g[1][2] += -X/3.0;
        h[0][1][2] += 1.0/3.0; h[0][2][1] += 1.0/3.0;
        h[1][0][2] += -1.0/3.0; h[1][2][0] += -1.0/3.0;
      }
    }
  }
  return 1;
}

/* checks: the class fields at x by the regions and by Ewald's sum over all
   sources (out[0..3ncls) and [3ncls..6ncls)); the number of classes */
int pyse_slabflux_probe(const double *x, double *out)
{ REAL V[SF_MAXCLS][3], G[SF_MAXCLS][3][3], S[SF_MAXCLS][3][3][3], E[3];
  int cl, m, s;
  if ( !sf.on || !sf_fields(x,0,V,G,S) ) return 0;
  for ( cl = 0 ; cl < sf.ncls ; cl++ )
    for ( m = 0 ; m < 3 ; m++ ) { out[3*cl + m] = V[cl][m]; out[3*sf.ncls + 3*cl + m] = 0.0; }
  for ( s = 0 ; s < sf.nbase ; s++ )
  { REAL w[3] = { x[0] - sf.base[s].c[0], x[1] - sf.base[s].c[1], x[2] - sf.base[s].c[2] };
    sf_ewald(w,E);
    for ( cl = 0 ; cl < sf.ncls ; cl++ )
      for ( m = 0 ; m < 3 ; m++ ) out[3*sf.ncls + 3*cl + m] += sf.base[s].q[cl]*E[m];
  }
  return sf.ncls;
}
