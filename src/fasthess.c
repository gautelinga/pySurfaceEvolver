/*
 * fasthess.c -- parallel assembly of named-quantity Hessians on facets.
 *
 * Not part of the original Surface Evolver.
 *
 * calc_quant_hess() visits every facet, computes each method's value,
 * gradient and Hessian, and adds them to the right side and the sparse
 * Hessian hash (fill_grad(), fill_mixed_entry(), sp_hash_search()). Here
 * the facets are taken in chunks: within a chunk the per-facet work (setup,
 * method Hessians, scaling, projections onto vertex freedoms) runs in
 * parallel, each facet recording the additions it would make in order;
 * then the additions are made serially in facet order. The matrix, right
 * side and method values are the same as calc_quant_hess() makes.
 *
 * Only for methods checked to be thread-safe (the Lagrange model's facet
 * area and volume methods); anything else runs calc_quant_hess() as is.
 */

#include "include.h"
#include "fastloops.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* one recorded addition */
enum { OP_ENTRY, OP_GRAD, OP_VALUE };
struct hop
{ int kind;
  int r, c;          /* OP_ENTRY: matrix row, column; OP_VALUE: method */
  vertex_id v;       /* OP_GRAD: vertex */
  REAL x[MAXCOORD];  /* OP_ENTRY: x[0]; OP_GRAD: gradient; OP_VALUE: x[0], x[1] */
};

struct hbuf { struct hop *ops; long n, size; int failed; };

static void put(struct hbuf *b, struct hop *op)
{ if ( b->failed ) return;
  if ( b->n == b->size )
  { long size = b->size ? 2*b->size : 4096;
    struct hop *ops = realloc(b->ops,size*sizeof(struct hop));
    if ( !ops ) { b->failed = 1; return; }
    b->ops = ops;
    b->size = size;
  }
  b->ops[b->n++] = *op;
}

static void put_entry(struct hbuf *b, int r, int c, REAL val)
{ struct hop op;
  op.kind = OP_ENTRY; op.r = r; op.c = c; op.x[0] = val;
  put(b,&op);
}

/* the sp_hash_search() calls of fill_self_entry() */
static void self_entry(struct hbuf *b, vertex_id v_id, REAL **self)
{ struct hess_verlist *v = get_vertex_vhead(v_id);
  int j, k;
  if ( v->proj )
  { MAT2D(temp_mat,MAXCOORD,MAXCOORD);
    MAT2D(temp_mat2,MAXCOORD,MAXCOORD);
    tr_mat_mul(v->proj,self,temp_mat,SDIM,v->freedom,SDIM);
    mat_mult(temp_mat,v->proj,temp_mat2,v->freedom,SDIM,v->freedom);
    for ( j = 0 ; j < v->freedom ; j++ )
      for ( k = 0 ; k <= j ; k++ )
        put_entry(b,v->rownum+k,v->rownum+j,temp_mat2[j][k]);
  }
  else
    for ( j = 0 ; j < v->freedom ; j++ )
      for ( k = 0 ; k <= j ; k++ )
        put_entry(b,v->rownum+k,v->rownum+j,self[j][k]);
}

/* the sp_hash_search() calls of fill_mixed_entry() */
static void mixed_entry(struct hbuf *b, vertex_id v_id1, vertex_id v_id2, REAL **mixed)
{ struct hess_verlist *v1, *v2;
  REAL **oo;
  int j, k;
  MAT2D(temp_mat,MAXCOORD,MAXCOORD);
  MAT2D(temp_mat2,MAXCOORD,MAXCOORD);
  if ( equal_id(v_id1,v_id2) ) { self_entry(b,v_id1,mixed); return; }
  v1 = get_vertex_vhead(v_id1);
  v2 = get_vertex_vhead(v_id2);
  if ( v1->proj )
  { tr_mat_mul(v1->proj,mixed,temp_mat,SDIM,v1->freedom,SDIM);
    oo = temp_mat;
  }
  else oo = mixed;
  if ( v2->proj )
  { mat_mult(oo,v2->proj,temp_mat2,v1->freedom,SDIM,v2->freedom);
    oo = temp_mat2;
  }
  if ( v1->rownum < v2->rownum )
    for ( j = 0 ; j < v1->freedom ; j++ )
      for ( k = 0 ; k < v2->freedom ; k++ )
        put_entry(b,v1->rownum+j,v2->rownum+k,oo[j][k]);
  else
    for ( j = 0 ; j < v1->freedom ; j++ )
      for ( k = 0 ; k < v2->freedom ; k++ )
        put_entry(b,v2->rownum+k,v1->rownum+j,oo[j][k]);
}

/* methods whose setup and Hessian are thread-safe */
static int method_ok(struct method_instance *mi)
{ struct gen_quant_method *gm = basic_gen_methods + mi->gen_method;
  if ( mi->flags & Q_COMPOUND ) return 0;
  /* for the Lagrange model (checked by the caller) these go to
     lagrange_facet_tension_hess() and lagrange_facet_volume_hess() */
  return gm->hessian == q_facet_tension_hessian
      || gm->hessian == q_facet_volume_hess
      || gm->hessian == lagrange_facet_volume_hess;
}

/* One facet: what calc_quant_hess() does for it (hess_mode 1), recorded. */
static void facet_ops(struct linsys *S, struct qinfo *q_info, facet_id f_id,
                      int global_needs, int meth_offset, int mode, struct hbuf *b)
{ struct element *e_ptr = elptr(f_id);
  int setup_flag = 0, needs = global_needs, flag, inum, k, i, ii, j, jj, m, n;
  struct method_instance *mi;
  struct gen_quant *q;
  struct hess_verlist *va, *vb;
  REAL g[MAXCOORD], *ggg;
  struct hop op;

  q_info->id = f_id;
  for ( k = 0 ; k < e_ptr->method_count ; k++ )
  { int mm = ((int*)((char*)e_ptr+meth_offset))[k];
    mi = METH_INSTANCE(abs(mm));
    if ( (mi->flags & Q_DOTHIS) && (mi->type == FACET) )
      needs |= basic_gen_methods[mi->gen_method].flags;
  }
  inum = global_meth_inst_count[FACET];
  for ( flag = 0 ; flag < 2 ; flag++, inum = e_ptr->method_count )
    for ( k = 0 ; k < inum ; k++ )
    { int sign = 1;
      REAL value, coeff = 0.0;
      struct gen_quant_method *gm;
      if ( flag )
      { int mm = ((int*)((char*)e_ptr+meth_offset))[k];
        q_info->method = abs(mm);
        if ( mm < 0 ) sign = -1;
      }
      else q_info->method = global_meth_inst[FACET][k];
      mi = METH_INSTANCE(q_info->method);
      if ( !(mi->flags & Q_DOTHIS) || (mi->type != FACET) ) continue;
      if ( !setup_flag ) { (*q_setup[FACET])(S,q_info,needs); setup_flag = 1; }
      for ( j = 0 ; j < MMAXQUANTS ; j++ )
      { if ( mi->quants[j] < 0 ) continue;
        q = GEN_QUANT(mi->quants[j]);
        if ( q->flags & (Q_FIXED|Q_CONSERVED) )
          coeff += -q->pressure*sign*q->modulus*mi->modulus;
        else
          coeff += sign*q->modulus*mi->modulus;
      }
      gm = basic_gen_methods + mi->gen_method;
      zerohess(q_info);
      value = (*gm->hessian)(q_info);
      if ( mi->flags & ELEMENT_MODULUS_FLAG )
      { REAL emdls = *(REAL*)get_extra(q_info->id,mi->elmodulus);
        value *= emdls;
        for ( i = 0 ; i < q_info->vcount ; i++ )
          for ( j = 0 ; j < SDIM ; j++ )
            q_info->grad[i][j] *= emdls;
        for ( i = 0 ; i < q_info->vcount ; i++ )
          for ( ii = 0 ; ii < q_info->vcount ; ii++ )
            for ( j = 0 ; j < SDIM ; j++ )
              for ( jj = 0 ; jj < SDIM ; jj++ )
                q_info->hess[i][ii][j][jj] *= emdls;
      }
      op.kind = OP_VALUE; op.r = q_info->method;
      op.x[0] = sign*value; op.x[1] = fabs(value);
      put(b,&op);

      for ( i = 0 ; i < q_info->vcount ; i++ )
      { op.kind = OP_GRAD; op.v = q_info->v[i];
        for ( j = 0 ; j < SDIM ; j++ )
          op.x[j] = coeff*q_info->grad[i][j];
        put(b,&op);
      }

      if ( !(mode & (Q_FIXED|Q_ENERGY|Q_CONSERVED)) ) continue;
      for ( i = 0 ; i < q_info->vcount ; i++ )
      { va = get_vertex_vhead(q_info->v[i]);
        if ( va->freedom == 0 ) continue;
        for ( j = i ; j < q_info->vcount ; j++ )
        { vb = get_vertex_vhead(q_info->v[j]);
          if ( vb->freedom == 0 ) continue;
          for ( n = 0 ; n < SDIM ; n++ )
            for ( m = 0 ; m < SDIM ; m++ )
              q_info->hess[i][j][m][n] *= coeff;
          mixed_entry(b,q_info->v[i],q_info->v[j],q_info->hess[i][j]);
          if ( (i != j) && (q_info->v[i] == q_info->v[j]) )
          { MAT2D(transpose,MAXCOORD,MAXCOORD);
            for ( n = 0 ; n < SDIM ; n++ )
              for ( m = 0 ; m < SDIM ; m++ )
                transpose[m][n] = q_info->hess[i][j][n][m];
            mixed_entry(b,q_info->v[i],q_info->v[j],transpose);
          }
        }
        /* fixed quantity gradients for left side */
        for ( j = 0 ; j < MMAXQUANTS ; j++ )
        { if ( mi->quants[j] < 0 ) continue;
          q = GEN_QUANT(mi->quants[j]);
          if ( q->flags & (Q_FIXED|Q_CONSERVED) )
          { int currentrow = S->quanrowstart + mi->quants[j];
            REAL ccoeff = sign*q->modulus*mi->modulus;
            if ( va->proj )
            { vec_mat_mul(q_info->grad[i],va->proj,g,SDIM,va->freedom);
              ggg = g;
            }
            else ggg = q_info->grad[i];
            for ( m = 0 ; m < va->freedom ; m++ )
              put_entry(b,va->rownum+m,currentrow,ccoeff*ggg[m]);
          }
        }
      }
    }
}

#define FH_MAXTHREADS 256

int fl_quant_hess_facets(struct linsys *S, int hess_mode, int mode, REAL *rhs,
                         int global_needs)
{ facet_id *list;
  long n, start, chunk, k;
  int threads, t, mk, meth_offset = get_meth_offset(FACET);
  static struct qinfo qi[FH_MAXTHREADS];
  static struct hbuf buf[FH_MAXTHREADS];
  long *fstart = NULL;   /* per chunk facet: start, end in its thread's buffer */
  int *fthread = NULL;
  int ok = 1;

  if ( fl_disabled() || !hess_mode || threadflag || nprocs > 1
       || web.modeltype != LAGRANGE || SDIM != 3 || web.dimension != 2
       || compound_quant_list_head >= 0 || (sym_flags & NEED_FORM_UNWRAPPING)
       || dirichlet_flag || sobolev_flag || web.torus_flag )
    return 0;
  for ( mk = LOW_INST ; mk < meth_inst_count ; mk++ )
  { struct method_instance *mi = METH_INSTANCE(mk);
    if ( mi->flags & Q_DELETED ) continue;
    if ( (mi->flags & Q_DOTHIS) && (mi->type == FACET) && !method_ok(mi) )
      return 0;
  }
  list = fl_facet_list(&n);
  if ( !list || n == 0 ) return 0;

  threads = fl_threads();
  if ( threads > FH_MAXTHREADS ) threads = FH_MAXTHREADS;
  if ( threads < 1 ) threads = 1;
  chunk = 64*threads;
  fstart = (long*)malloc(2*chunk*sizeof(long));
  fthread = (int*)malloc(chunk*sizeof(int));
  if ( !fstart || !fthread ) { free(fstart); free(fthread); return 0; }

  /* per-thread work space (Evolver's allocator is not thread-safe) */
  for ( t = 0 ; t < threads ; t++ )
  { q_info_init(&qi[t],METHOD_HESSIAN);
    qi[t].hess = dmatrix4(MAXVCOUNT,MAXVCOUNT,SDIM,SDIM);
  }
  /* first facet serially: lazy one-time setup (e.g. packed basis matrices) */
  qi[0].id = list[0];
  (*q_setup[FACET])(S,&qi[0],global_needs|NEED_SIDE);

  for ( start = 0 ; start < n && ok ; start += chunk )
  { long end = start + chunk < n ? start + chunk : n;
    for ( t = 0 ; t < threads ; t++ ) { buf[t].n = 0; buf[t].failed = 0; }

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic,4) num_threads(threads)
#endif
    for ( k = start ; k < end ; k++ )
    { int me = 0;
#ifdef _OPENMP
      me = omp_get_thread_num();
#endif
      fthread[k-start] = me;
      fstart[2*(k-start)] = buf[me].n;
      facet_ops(S,&qi[me],list[k],global_needs,meth_offset,mode,&buf[me]);
      fstart[2*(k-start)+1] = buf[me].n;
    }
    for ( t = 0 ; t < threads ; t++ )
      if ( buf[t].failed ) ok = 0;
    if ( !ok ) break;   /* out of memory: give up (nothing added this chunk) */

    /* the additions, serially in facet order */
    for ( k = start ; k < end ; k++ )
    { struct hbuf *b = buf + fthread[k-start];
      long e;
      for ( e = fstart[2*(k-start)] ; e < fstart[2*(k-start)+1] ; e++ )
      { struct hop *op = b->ops + e;
        switch ( op->kind )
        { case OP_ENTRY: sp_hash_search(S,op->r,op->c,op->x[0]); break;
          case OP_GRAD: fill_grad(S,get_vertex_vhead(op->v),op->x,rhs); break;
          case OP_VALUE:
          { struct method_instance *mi = METH_INSTANCE(op->r);
            mi->newvalue += op->x[0];
            mi->abstotal += op->x[1];
            break;
          }
        }
      }
    }
    comp_quant_stamp += (int)(end - start);
  }

  for ( t = 0 ; t < threads ; t++ )
  { free_matrix4(qi[t].hess);
    qi[t].hess = NULL;
    q_info_free(&qi[t]);
  }
  free(fstart);
  free(fthread);
  if ( !ok )
    kb_error(6350,"Out of memory in parallel Hessian assembly.\n",RECOVERABLE);
  return 1;
}

/**************************************************************************
 * Debugging and solver experiments: PYSE_DUMP_HESSIAN=path writes the
 * Newton-step matrix that is about to be factored (upper triangle, as
 * stored, in Matrix Market format) to path, overwriting it each time, and
 * reports the time of Evolver's factoring.
 */

static double fh_t0;

void fl_hessian_before_factor(struct linsys *S)
{ const char *path = getenv("PYSE_DUMP_HESSIAN");
  FILE *fd;
  int i, j;
  if ( !path ) return;
  fd = fopen(path,"w");
  if ( fd )
  { fprintf(fd,"%%%%MatrixMarket matrix coordinate real symmetric\n");
    fprintf(fd,"%% pySE Newton-step matrix: N %d, A_rows %d, quanrowstart %d\n",
            S->N,S->A_rows,S->quanrowstart);
    fprintf(fd,"%d %d %d\n",S->N,S->N,S->IA[S->N]-A_OFF);
    for ( i = 0 ; i < S->N ; i++ )
      for ( j = S->IA[i]-A_OFF ; j < S->IA[i+1]-A_OFF ; j++ )
        fprintf(fd,"%d %d %.17g\n",S->JA[j]-A_OFF+1,i+1,S->A[j]);
    fclose(fd);
  }
#ifdef _OPENMP
  fh_t0 = omp_get_wtime();
#endif
}

void fl_hessian_after_factor(struct linsys *S)
{ if ( !getenv("PYSE_DUMP_HESSIAN") ) return;
#ifdef _OPENMP
  fprintf(stderr,"PYSE_DUMP_HESSIAN: N %d, nnz %d, factor %.3f s, inertia %d neg %d zero %d pos\n",
          S->N,S->IA[S->N]-A_OFF,omp_get_wtime()-fh_t0,S->neg,S->zero,S->pos);
#endif
}
