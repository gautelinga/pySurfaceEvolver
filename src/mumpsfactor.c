/*
 * mumpsfactor.c -- factoring Newton-step Hessians with MUMPS.
 *
 * Not part of the original Surface Evolver.
 *
 * The same interface as Evolver's other factoring codes (xmd_factor(),
 * ysmp_factor(), mkl_factor()): factor the symmetric, possibly indefinite
 * system S (upper triangle, rows IA, columns JA, values A), report its
 * inertia in S->neg, S->zero, S->pos, and solve with the factors.
 *
 * MUMPS (sequential build, LDL^T with pivoting, null pivot detection) runs
 * as one shared instance. Its analysis (ordering and symbolic factorization)
 * is kept from one Newton step to the next and reused while the sparsity
 * pattern and the thread count are the same (each step builds a new linear
 * system, but the pattern changes only with the topology); freeing a system
 * frees the factors but keeps the analysis. The factors belong to one system
 * at a time (owner); a solve for another system refactors it first.
 * Null pivots, as Evolver's ZEROPIVOT, are pivots below hessian_epsilon
 * relative to the matrix norm; MUMPS fixes them so their solution
 * components are zero.
 */

#include "include.h"
#include "fastloops.h"

#ifdef PYSE_MUMPS

#include "dmumps_c.h"

#define ICNTL(I) icntl[(I)-1]
#define CNTL(I) cntl[(I)-1]
#define INFOG(I) infog[(I)-1]
#define USE_COMM_WORLD (-987654)

struct mumps_sys
{ DMUMPS_STRUC_C id;
  int n;
  int64_t nnz;
  MUMPS_INT *irn, *jcn;    /* pattern of the current analysis, 1-based */
  double *a;               /* values given to MUMPS (lambda subtracted) */
  int analysed;            /* analysis valid for irn, jcn, threads */
  int threads;             /* ICNTL(16) of the analysis */
  struct linsys *owner;    /* system the current factors are of, or NULL */
};

static struct mumps_sys *shared = NULL;

static void mumps_check(struct mumps_sys *m, const char *what)
{ if ( m->id.INFOG(1) < 0 )
  { sprintf(errmsg,"MUMPS %s failed: INFOG(1) = %d, INFOG(2) = %d.\n",
            what,(int)m->id.INFOG(1),(int)m->id.INFOG(2));
    kb_error(6360,errmsg,RECOVERABLE);
  }
}

/* OpenBLAS, if that is the BLAS: a pthreads build runs its own threads
   inside MUMPS's OpenMP threads (oversubscription: 1.2 s -> 2.2 s per
   Newton step at 4 threads), so make it single-threaded; an OpenMP build
   already runs serially inside parallel regions. Weak: other BLAS work. */
extern int openblas_get_parallel(void) __attribute__((weak));
extern void openblas_set_num_threads(int) __attribute__((weak));

static void blas_threads(void)
{ static int done = 0;
  if ( done ) return;
  done = 1;
  if ( openblas_get_parallel && openblas_set_num_threads
       && openblas_get_parallel() == 1 )
    openblas_set_num_threads(1);
}

static void mumps_controls(struct mumps_sys *m)
{ blas_threads(); m->id.ICNTL(1) = -1;    /* no error messages */
  m->id.ICNTL(2) = -1;    /* no diagnostics */
  m->id.ICNTL(3) = -1;    /* no global information */
  m->id.ICNTL(4) = 0;
  m->id.ICNTL(7) = 0;     /* AMD ordering: fastest analysis, good fill here */
  m->id.ICNTL(13) = 1;    /* no ScaLAPACK root: exact inertia */
  m->id.ICNTL(16) = fl_threads();   /* OpenMP threads */
  m->id.ICNTL(24) = 1;    /* null pivot detection */
  m->id.CNTL(3) = hessian_epsilon > 0.0 ? hessian_epsilon : 1e-8;  /* relative */
}

void mumps_factor(struct linsys *S, int mtype)
{ struct mumps_sys *m = shared;
  int64_t nnz, k;
  int i, same, job, tries;
  (void)mtype;

  S->neg = S->zero = S->pos = 0;
  if ( S->N == 0 ) return;

  if ( !m )
  { m = (struct mumps_sys *)calloc(1,sizeof(struct mumps_sys));
    if ( !m ) kb_error(6361,"Out of memory for MUMPS.\n",RECOVERABLE);
    m->id.comm_fortran = USE_COMM_WORLD;
    m->id.par = 1;
    m->id.sym = 2;        /* general symmetric (indefinite) */
    m->id.job = -1;
    fl_enter(); dmumps_c(&m->id); fl_leave();
    if ( m->id.INFOG(1) < 0 )
    { sprintf(errmsg,"MUMPS initialization failed: INFOG(1) = %d.\n",(int)m->id.INFOG(1));
      free(m);
      kb_error(6360,errmsg,RECOVERABLE);
    }
    shared = m;
  }
  S->mumps = m;
  m->owner = NULL;   /* until this factoring succeeds */

  /* pattern; same as the last analysis? */
  nnz = S->IA[S->N] - A_OFF;
  same = m->analysed && m->n == S->N && m->nnz == nnz && m->threads == fl_threads();
  if ( !same )
  { free(m->irn); free(m->jcn); free(m->a);
    m->irn = (MUMPS_INT *)malloc(nnz*sizeof(MUMPS_INT));
    m->jcn = (MUMPS_INT *)malloc(nnz*sizeof(MUMPS_INT));
    m->a = (double *)malloc(nnz*sizeof(double));
    if ( !m->irn || !m->jcn || !m->a )
      kb_error(6362,"Out of memory for MUMPS.\n",RECOVERABLE);
    m->analysed = 0;
  }
  for ( i = 0 ; i < S->N ; i++ )
    for ( k = S->IA[i]-A_OFF ; k < S->IA[i+1]-A_OFF ; k++ )
    { MUMPS_INT r = i + 1, c = S->JA[k] - A_OFF + 1;
      if ( same && (m->irn[k] != r || m->jcn[k] != c) ) same = 0;
      m->irn[k] = r;
      m->jcn[k] = c;
      m->a[k] = S->A[k];
      if ( (c == r) && (i < S->A_rows) ) m->a[k] -= S->lambda;
    }
  m->n = S->N;
  m->nnz = nnz;

  m->id.n = S->N;
  m->id.nnz = nnz;
  m->id.irn = m->irn;
  m->id.jcn = m->jcn;
  m->id.a = m->a;
  mumps_controls(m);
  job = same ? 2 : 4;     /* factor only, or analyse and factor */
  for ( tries = 0 ; ; tries++ )
  { m->id.job = job;
    fl_enter(); dmumps_c(&m->id); fl_leave();
    if ( m->id.INFOG(1) == -9 && tries < 5 )   /* workspace too small */
    { m->id.ICNTL(14) = m->id.ICNTL(14) > 0 ? 2*m->id.ICNTL(14) : 40;
      continue;
    }
    break;
  }
  m->analysed = m->id.INFOG(1) >= 0;
  m->threads = m->id.ICNTL(16);
  mumps_check(m,"factorization");
  m->owner = S;

  S->neg = m->id.INFOG(12);
  S->zero = m->id.INFOG(28);
  S->pos = S->N - S->neg - S->zero;
}

void mumps_solve(struct linsys *S, REAL *b, REAL *x, int mtype)
{ struct mumps_sys *m = shared;
  if ( S->N == 0 ) return;
  if ( !m || !S->mumps )
    kb_error(6363,"Internal error: MUMPS solve before factoring.\n",RECOVERABLE);
  if ( m->owner != S )   /* the factors are another system's now */
    mumps_factor(S,mtype);
  if ( x != b ) memcpy(x,b,S->N*sizeof(REAL));
  m->id.rhs = x;
  m->id.nrhs = 1;
  m->id.lrhs = S->N;
  m->id.ICNTL(20) = 0;    /* dense right side */
  m->id.ICNTL(21) = 0;    /* centralized solution, in rhs */
  m->id.job = 3;
  fl_enter(); dmumps_c(&m->id); fl_leave();
  mumps_check(m,"solve");
}

void mumps_solve_multi(struct linsys *S, REAL **b, REAL **x, int nrhs, int mtype)
{ int k;
  for ( k = 0 ; k < nrhs ; k++ )
    mumps_solve(S,b[k],x[k],mtype);
}

/* The system is going away: free the factors, keep the analysis. */
void mumps_free_system(struct linsys *S)
{ struct mumps_sys *m = shared;
  S->mumps = NULL;
  if ( !m || m->owner != S ) return;
  m->owner = NULL;
  m->id.job = -4;   /* free factors, keep the analysis */
  fl_enter(); dmumps_c(&m->id); fl_leave();
  if ( m->id.INFOG(1) < 0 ) m->analysed = 0;
}

int fl_have_mumps(void) { return 1; }

#else  /* no MUMPS in this build */

void mumps_factor(struct linsys *S, int mtype)
{ (void)S; (void)mtype;
  kb_error(6364,"This Evolver was built without MUMPS.\n",RECOVERABLE);
}
void mumps_solve(struct linsys *S, REAL *b, REAL *x, int mtype)
{ (void)S; (void)b; (void)x; (void)mtype;
  kb_error(6364,"This Evolver was built without MUMPS.\n",RECOVERABLE);
}
void mumps_solve_multi(struct linsys *S, REAL **b, REAL **x, int nrhs, int mtype)
{ (void)S; (void)b; (void)x; (void)nrhs; (void)mtype;
  kb_error(6364,"This Evolver was built without MUMPS.\n",RECOVERABLE);
}
void mumps_free_system(struct linsys *S) { (void)S; }
int fl_have_mumps(void) { return 0; }

#endif
