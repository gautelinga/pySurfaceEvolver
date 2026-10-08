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
enum { OP_ENTRY, OP_GRAD, OP_VALUE, OP_QVALUE, OP_FORCE, OP_VGRAD, OP_AREA,
       OP_DONE /* an OP_ENTRY already added to the kept pattern */ };
struct hop
{ int kind;
  int r, c;          /* OP_ENTRY: matrix row, column; OP_VALUE, OP_QVALUE:
                        method; OP_VGRAD: fixnum, quantity */
  vertex_id v;       /* OP_GRAD, OP_FORCE, OP_VGRAD: vertex */
  body_id b;         /* OP_VGRAD: body */
  REAL s;            /* OP_FORCE, OP_VGRAD: coefficient */
  REAL x[MAXCOORD];  /* OP_ENTRY: x[0]; OP_GRAD: gradient; OP_VALUE: x[0], x[1];
                        OP_QVALUE: x[0]; OP_FORCE, OP_VGRAD: vector */
};

struct hbuf { struct hop *ops; long n, size; int failed; };

/* the buffer the current thread records into, while run_elements() runs */
static __thread struct hbuf *recording = NULL;

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

/* v1's projection transposed, times m, times v2's projection (either may be
   NULL: the identity): fill_mixed_entry()'s tr_mat_mul() and mat_mult(),
   the same arithmetic in the same order, without their generic overhead */
static void project_block(struct hess_verlist *v1, struct hess_verlist *v2, REAL **m,
                          REAL o[MAXCOORD][MAXCOORD])
{ REAL t[MAXCOORD][MAXCOORD];
  int f1 = v1->freedom, f2 = v2->freedom, a, bb, i, n, sd = SDIM;
  if ( v1->proj )
  { REAL **p = v1->proj;
    for ( a = 0 ; a < f1 ; a++ )
      for ( n = 0 ; n < sd ; n++ )
      { REAL sum = 0.0;
        for ( i = 0 ; i < sd ; i++ ) sum += p[i][a]*m[i][n];
        t[a][n] = sum;
      }
  }
  else
    for ( a = 0 ; a < f1 ; a++ )
      for ( n = 0 ; n < sd ; n++ ) t[a][n] = m[a][n];
  if ( v2->proj )
  { REAL **p = v2->proj;
    for ( a = 0 ; a < f1 ; a++ )
    { for ( bb = 0 ; bb < f2 ; bb++ ) o[a][bb] = 0.0;
      for ( n = 0 ; n < sd ; n++ )
      { REAL ta = t[a][n];
        if ( ta == 0.0 ) continue;
        for ( bb = 0 ; bb < f2 ; bb++ ) o[a][bb] += ta*p[n][bb];
      }
    }
  }
  else
    for ( a = 0 ; a < f1 ; a++ )
      for ( bb = 0 ; bb < f2 ; bb++ ) o[a][bb] = t[a][bb];
}

/* the sp_hash_search() calls of fill_self_entry() */
static void self_entry(struct hbuf *b, vertex_id v_id, REAL **self)
{ struct hess_verlist *v = get_vertex_vhead(v_id);
  REAL o[MAXCOORD][MAXCOORD];
  int j, k;
  project_block(v,v,self,o);
  for ( j = 0 ; j < v->freedom ; j++ )
    for ( k = 0 ; k <= j ; k++ )
      put_entry(b,v->rownum+k,v->rownum+j,o[j][k]);
}

/* the sp_hash_search() calls of fill_mixed_entry() */
static void mixed_entry(struct hbuf *b, vertex_id v_id1, vertex_id v_id2, REAL **mixed)
{ struct hess_verlist *v1, *v2;
  REAL o[MAXCOORD][MAXCOORD];
  int j, k;
  if ( equal_id(v_id1,v_id2) ) { self_entry(b,v_id1,mixed); return; }
  v1 = get_vertex_vhead(v_id1);
  v2 = get_vertex_vhead(v_id2);
  project_block(v1,v2,mixed,o);
  if ( v1->rownum < v2->rownum )
    for ( j = 0 ; j < v1->freedom ; j++ )
      for ( k = 0 ; k < v2->freedom ; k++ )
        put_entry(b,v1->rownum+j,v2->rownum+k,o[j][k]);
  else
    for ( j = 0 ; j < v1->freedom ; j++ )
      for ( k = 0 ; k < v2->freedom ; k++ )
        put_entry(b,v2->rownum+k,v1->rownum+j,o[j][k]);
}


/* ---- which methods may run in parallel ---- */

/* Expression node types that only read (constants, globals, parameters,
   coordinates, arithmetic, math, comparisons, conditionals, built-in user
   functions): the evaluator then touches only its per-thread stack. */
static int node_ok(int type)
{ switch ( type )
  { case SETUP_FRAME_NODE: case FINISHED_NODE:
    case PUSHCONST_NODE: case REPLACECONST_NODE: case PUSHDELTA_NODE:
    case PUSH_PARAM_SCALE_NODE: case PUSH_PARAM_FIXED_NODE:
    case PUSHGLOBAL_NODE: case PUSH_PERM_GLOBAL_NODE:
    case PUSHPI_NODE: case PUSHE_NODE: case PUSHG_NODE: case PUSHPARAM_NODE:
    case USERFUNC_NODE:
    case GT_NODE: case LT_NODE: case LE_NODE: case GE_NODE: case NE_NODE:
    case EQ_NODE: case AND_NODE: case CONJUNCTION_END_NODE: case OR_NODE:
    case NOT_NODE:
    case PLUS_NODE: case MINUS_NODE: case TIMES_NODE: case DIVIDE_NODE:
    case REALMOD_NODE: case IMOD_NODE: case IDIV_NODE: case INTPOW_NODE:
    case POW_NODE: case MAXIMUM_NODE: case MINIMUM_NODE: case ATAN2_NODE:
    case SQR_NODE: case SQRT_NODE: case CEIL_NODE: case FLOOR_NODE: case ABS_NODE:
    case SIN_NODE: case COS_NODE: case TAN_NODE: case EXP_NODE: case SINH_NODE:
    case COSH_NODE: case TANH_NODE: case ASINH_NODE: case ACOSH_NODE:
    case ATANH_NODE: case LOG_NODE: case ASIN_NODE: case ACOS_NODE:
    case ATAN_NODE: case ELLIPTICK_NODE: case ELLIPTICE_NODE:
    case INCOMPLETE_ELLIPTICF_NODE: case INCOMPLETE_ELLIPTICE_NODE:
    case CHS_NODE: case INV_NODE:
    case COORD_NODE: case INDEXED_COORD_NODE: case PARAM_NODE:
    case ID_NODE: case GET_ID_NODE: case GET_OID_NODE: case ORIGINAL_NODE:
    case GET_ORIGINAL_NODE:
    case IFTEST_NODE: case COND_TEST_NODE: case IF_NODE: case COND_EXPR_NODE:
      return 1;
  }
  return 0;
}

static int expr_ok(struct expnode *e)
{ struct treenode *node;
  if ( !e || !e->start || !e->root ) return 1;
  if ( e->locals && e->locals->totalsize > 0 ) return 0;
  for ( node = e->start + 1 ; node <= e->root ; node++ )
    if ( !node_ok(node->type) ) return 0;
  return 1;
}

/* Methods checked to be thread-safe for the linear and Lagrange models:
   their setup, value, gradient and Hessian write only their qinfo (the area
   methods' total area goes through fl_total_area_add()). Integral methods
   also need read-only integrands. */
static const struct
{ REAL (*value)(QINFO), (*grad)(QINFO), (*hess)(QINFO);
  int integrand;
} safe_methods[] = {
  { q_facet_tension_value, q_facet_tension_gradient, q_facet_tension_hessian, 0 },
  { q_facet_volume, q_facet_volume_grad, q_facet_volume_hess, 0 },
  { NULL, NULL, lagrange_facet_volume_hess, 0 },
  { facet_scalar_integral, facet_scalar_integral_grad, facet_scalar_integral_hess, 1 },
  { facet_vector_integral, facet_vector_integral_grad, facet_vector_integral_hess, 1 },
  { facet_2form_integral, facet_2form_integral_grad, facet_2form_integral_hess, 1 },
  { facet_general_value, facet_general_grad, facet_general_hess, 1 },
  { edge_scalar_integral, edge_scalar_integral_grad, edge_scalar_integral_hess, 1 },
  { edge_vector_integral, edge_vector_integral_grad, edge_vector_integral_hess, 1 },
  { edge_general_value, edge_general_grad, edge_general_hess, 1 },
  { vertex_scalar_integral, vertex_scalar_integral_grad, vertex_scalar_integral_hess, 1 },
};

enum { PASS_VALUE, PASS_GRAD, PASS_HESS };

static int method_safe(struct method_instance *mi, int pass)
{ struct gen_quant_method *gm = basic_gen_methods + mi->gen_method;
  int k, e;
  if ( mi->flags & Q_COMPOUND ) return 0;
  /* vertex_scalar_integral's value adjusts neighbouring facet areas */
  if ( gm->value == vertex_scalar_integral && pass == PASS_VALUE
       && (mi->flags & DEFAULT_INSTANCE) ) return 0;
  for ( k = 0 ; k < (int)(sizeof(safe_methods)/sizeof(safe_methods[0])) ; k++ )
  { REAL (*f)(QINFO) = pass == PASS_VALUE ? safe_methods[k].value
                     : pass == PASS_GRAD ? safe_methods[k].grad : safe_methods[k].hess;
    REAL (*g)(QINFO) = pass == PASS_VALUE ? gm->value
                     : pass == PASS_GRAD ? gm->gradient : gm->hessian;
    if ( !f || f != g ) continue;
    if ( safe_methods[k].integrand )
      for ( e = 0 ; e < MAXMEXPR ; e++ )
        if ( !expr_ok(mi->expr[e]) ) return 0;
    return 1;
  }
  return 0;
}

/* whether this pass over elements of `type` may run in parallel */
static int quant_ok(int type, int pass)
{ int mk;
  if ( fl_disabled() || threadflag || nprocs > 1
       || (web.modeltype != LAGRANGE && web.modeltype != LINEAR)
       || SDIM != 3 || web.dimension != 2 || web.representation != SOAPFILM
       || compound_quant_list_head >= 0 || (sym_flags & NEED_FORM_UNWRAPPING)
       || web.symmetry_flag || dirichlet_flag || sobolev_flag || web.torus_flag )
    return 0;
  if ( type != VERTEX && type != EDGE && type != FACET ) return 0;
  for ( mk = LOW_INST ; mk < meth_inst_count ; mk++ )
  { struct method_instance *mi = METH_INSTANCE(mk);
    if ( mi->flags & Q_DELETED ) continue;
    if ( (mi->flags & Q_DOTHIS) && (mi->type == type) && !method_safe(mi,pass) )
      return 0;
  }
  return 1;
}

/* the methods of an element in calc_quants() order: global ones, then the
   element's own (mm < 0: opposite sign) */
#define FOR_ELEMENT_METHODS(type,e_ptr,meth_offset,flag,k,inum,mm,sign) \
  for ( flag = 0, inum = global_meth_inst_count[type] ; flag < 2 ; \
        flag++, inum = (e_ptr)->method_count ) \
    for ( k = 0 ; k < inum ; k++ ) \
      if ( (mm = flag ? ((int*)((char*)(e_ptr)+(meth_offset)))[k] \
                      : global_meth_inst[type][k]), \
           (sign = (flag && mm < 0) ? -1 : 1), 0 ) ; else

static int element_needs(int type, struct element *e_ptr, int global_needs, int meth_offset)
{ int k, needs = global_needs;
  for ( k = 0 ; k < e_ptr->method_count ; k++ )
  { struct method_instance *mi =
        METH_INSTANCE(abs(((int*)((char*)e_ptr+meth_offset))[k]));
    if ( (mi->flags & Q_DOTHIS) && (mi->type == type) )
      needs |= basic_gen_methods[mi->gen_method].flags;
  }
  return needs;
}

/* ---- the element loop bodies, recorded ---- */

struct quant_ctx { struct qinfo *qi; int type, global_needs, meth_offset, mode; };

/* q_setup[type]() for an element, from the facet's cached corners where
   q_facet_setup() would walk the facet's facet-edges for them: linear
   soapfilm facets without symmetry wraps, needing at most sides, normal and
   Gauss points. The same vertices in the same (tail) order and the same
   arithmetic as q_facet_setup(). */
static void element_setup(struct linsys *S, struct qinfo *q, int type, vertex_id *c, int needs)
{ int i, j;
  if ( type != FACET || !c || web.modeltype != LINEAR || web.representation != SOAPFILM
       || web.symmetry_flag || inverted(q->id)
       || (needs & ALL_NEEDS & ~(NEED_SIDE|NEED_NORMAL|NEED_GAUSS)) )
  { (*q_setup[type])(S,q,needs);
    return;
  }
  q->S = S;
  q->vcount = FACET_VERTS;
  for ( i = 0 ; i < FACET_VERTS ; i++ )
  { REAL *p = get_coord(c[i]);
    q->v[i] = c[i];
    q->x[i] = q->xx[i];
    q->wraps[i] = 0;
    for ( j = 0 ; j < SDIM ; j++ ) q->xx[i][j] = p[j];
  }
  if ( needs & NEED_SIDE )
    for ( i = 0 ; i < web.dimension ; i++ )
      for ( j = 0 ; j < SDIM ; j++ )
        q->sides[0][i][j] = q->x[i+1][j] - q->x[0][j];
  if ( needs & NEED_NORMAL )
    cross_prod(q->sides[0][0],q->sides[0][1],q->normal);
  if ( needs & NEED_GAUSS )
    mat_mult(gpoly,q->x,q->gauss_pt,gauss2D_num,ctrl_num,SDIM);
}

/* per thread: an element's summed Hessian blocks, [vcount][vcount] blocks of
   MAXCOORD x MAXCOORD */
static __thread REAL *acc_buf = NULL;
static __thread size_t acc_size = 0;

static REAL *hess_acc(int vcount)
{ size_t need = (size_t)vcount*vcount*MAXCOORD*MAXCOORD;
  if ( need > acc_size )
  { REAL *buf = (REAL *)realloc(acc_buf,need*sizeof(REAL));
    if ( !buf ) return NULL;
    acc_buf = buf;
    acc_size = need;
  }
  return acc_buf;
}

/* calc_quant_hess()'s element loop body (hess_mode 1) */
static void quant_hess_element(struct linsys *S, element_id id, vertex_id *c, int me,
                               struct hbuf *b, void *ctx)
{ struct quant_ctx *q = (struct quant_ctx *)ctx;
  struct qinfo *q_info = q->qi + me;
  int type = q->type;
  struct element *e_ptr = elptr(id);
  int setup_flag = 0, needs, flag, inum, k, i, ii, j, jj, m, n, mm, sign;
  struct gen_quant *gq;
  struct hess_verlist *va, *vb;
  REAL g[MAXCOORD], *ggg, *acc = NULL;
  int have_hess = 0;
  struct hop op;
  (void)c;

  q_info->id = id;
  needs = element_needs(type,e_ptr,q->global_needs,q->meth_offset);
  FOR_ELEMENT_METHODS(type,e_ptr,q->meth_offset,flag,k,inum,mm,sign)
  { struct method_instance *mi = METH_INSTANCE(abs(mm));
    REAL value, coeff = 0.0;
    q_info->method = abs(mm);
    if ( !(mi->flags & Q_DOTHIS) || (mi->type != type) ) continue;
    if ( !setup_flag ) { element_setup(S,q_info,type,c,needs); setup_flag = 1; }
    for ( j = 0 ; j < MMAXQUANTS ; j++ )
    { if ( mi->quants[j] < 0 ) continue;
      gq = GEN_QUANT(mi->quants[j]);
      if ( gq->flags & (Q_FIXED|Q_CONSERVED) )
        coeff += -gq->pressure*sign*gq->modulus*mi->modulus;
      else
        coeff += sign*gq->modulus*mi->modulus;
    }
    zerohess(q_info);
    value = (*basic_gen_methods[mi->gen_method].hessian)(q_info);
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
    op.kind = OP_VALUE; op.r = abs(mm);
    op.x[0] = sign*value; op.x[1] = fabs(value);
    put(b,&op);

    for ( i = 0 ; i < q_info->vcount ; i++ )
    { op.kind = OP_GRAD; op.v = q_info->v[i];
      for ( j = 0 ; j < SDIM ; j++ )
        op.x[j] = coeff*q_info->grad[i][j];
      put(b,&op);
    }

    if ( !(q->mode & (Q_FIXED|Q_ENERGY|Q_CONSERVED)) ) continue;
    if ( !acc && !(acc = hess_acc(q_info->vcount)) ) { b->failed = 1; return; }
    for ( i = 0 ; i < q_info->vcount ; i++ )
    { va = get_vertex_vhead(q_info->v[i]);
      if ( va->freedom == 0 ) continue;
      for ( j = i ; j < q_info->vcount ; j++ )
      { REAL *a = acc + ((size_t)i*q_info->vcount + j)*MAXCOORD*MAXCOORD;
        vb = get_vertex_vhead(q_info->v[j]);
        if ( vb->freedom == 0 ) continue;
        /* the methods' coeff*hess summed; projected and entered below */
        for ( m = 0 ; m < SDIM ; m++ )
          for ( n = 0 ; n < SDIM ; n++ )
            if ( have_hess ) a[m*MAXCOORD+n] += coeff*q_info->hess[i][j][m][n];
            else a[m*MAXCOORD+n] = coeff*q_info->hess[i][j][m][n];
      }
      /* fixed quantity gradients for left side */
      for ( j = 0 ; j < MMAXQUANTS ; j++ )
      { if ( mi->quants[j] < 0 ) continue;
        gq = GEN_QUANT(mi->quants[j]);
        if ( gq->flags & (Q_FIXED|Q_CONSERVED) )
        { int currentrow = S->quanrowstart + mi->quants[j];
          REAL ccoeff = sign*gq->modulus*mi->modulus;
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
    have_hess = 1;
  }

  /* the element's Hessian blocks, all methods together, as fill_mixed_entry() */
  if ( have_hess )
    for ( i = 0 ; i < q_info->vcount ; i++ )
    { va = get_vertex_vhead(q_info->v[i]);
      if ( va->freedom == 0 ) continue;
      for ( j = i ; j < q_info->vcount ; j++ )
      { REAL *a = acc + ((size_t)i*q_info->vcount + j)*MAXCOORD*MAXCOORD;
        REAL *rows[MAXCOORD];
        vb = get_vertex_vhead(q_info->v[j]);
        if ( vb->freedom == 0 ) continue;
        for ( m = 0 ; m < SDIM ; m++ ) rows[m] = a + m*MAXCOORD;
        mixed_entry(b,q_info->v[i],q_info->v[j],rows);
        if ( (i != j) && (q_info->v[i] == q_info->v[j]) )
        { MAT2D(transpose,MAXCOORD,MAXCOORD);
          for ( n = 0 ; n < SDIM ; n++ )
            for ( m = 0 ; m < SDIM ; m++ )
              transpose[m][n] = rows[n][m];
          mixed_entry(b,q_info->v[i],q_info->v[j],transpose);
        }
      }
    }
}

/* calc_quants()'s element loop body */
static void quant_value_element(struct linsys *S, element_id id, vertex_id *c, int me,
                                struct hbuf *b, void *ctx)
{ struct quant_ctx *q = (struct quant_ctx *)ctx;
  struct qinfo *q_info = q->qi + me;
  int type = q->type;
  struct element *e_ptr = elptr(id);
  int needs, setup_flag = 0, flag, k, inum, mm, sign;
  struct hop op;
  (void)S; (void)c;

  q_info->id = id;
  needs = element_needs(type,e_ptr,q->global_needs,q->meth_offset);
  FOR_ELEMENT_METHODS(type,e_ptr,q->meth_offset,flag,k,inum,mm,sign)
  { struct method_instance *mi = METH_INSTANCE(abs(mm));
    REAL value;
    q_info->method = abs(mm);
    if ( !(mi->flags & Q_DOTHIS) || (mi->type != type) ) continue;
    if ( !setup_flag ) { element_setup(NULL,q_info,type,c,needs); setup_flag = 1; }
    value = (*basic_gen_methods[mi->gen_method].value)(q_info);
    if ( mi->flags & ELEMENT_MODULUS_FLAG )
      value *= *(REAL*)get_extra(q_info->id,mi->elmodulus);
    if ( sign < 0 ) value = -value;
    op.kind = OP_QVALUE; op.r = abs(mm); op.x[0] = value;
    put(b,&op);
  }
}

/* calc_quant_grads()'s element loop body (no compound quantities, no
   symmetry: no unwrapping) */
static void quant_grad_element(struct linsys *S, element_id id, vertex_id *c, int me,
                               struct hbuf *b, void *ctx)
{ struct quant_ctx *q = (struct quant_ctx *)ctx;
  struct qinfo *q_info = q->qi + me;
  int type = q->type;
  struct element *e_ptr = elptr(id);
  int needs, setup_flag = 0, flag, k, inum, mm, sign, i, j;
  struct hop op;
  (void)S; (void)c;

  q_info->id = id;
  needs = element_needs(type,e_ptr,q->global_needs,q->meth_offset);
  FOR_ELEMENT_METHODS(type,e_ptr,q->meth_offset,flag,k,inum,mm,sign)
  { struct method_instance *mi = METH_INSTANCE(abs(mm));
    REAL value;
    q_info->method = abs(mm);
    if ( !(mi->flags & Q_DOTHIS) || (mi->type != type) ) continue;
    if ( !setup_flag ) { element_setup(NULL,q_info,type,c,needs); setup_flag = 1; }
    for ( i = 0 ; i < q_info->vcount ; i++ )
      for ( j = 0 ; j < SDIM ; j++ )
        q_info->grad[i][j] = 0.0;
    value = (*basic_gen_methods[mi->gen_method].gradient)(q_info);
    if ( mi->flags & ELEMENT_MODULUS_FLAG )
    { REAL emdls = *(REAL*)get_extra(q_info->id,mi->elmodulus);
      value *= emdls;
      for ( i = 0 ; i < q_info->vcount ; i++ )
        for ( j = 0 ; j < SDIM ; j++ )
          q_info->grad[i][j] *= emdls;
    }
    op.kind = OP_VALUE; op.r = abs(mm);
    op.x[0] = sign*value; op.x[1] = fabs(value);
    put(b,&op);
    for ( i = 0 ; i < q_info->vcount ; i++ )
      for ( j = 0 ; j < MMAXQUANTS ; j++ )
      { struct gen_quant *gq;
        REAL cc;
        if ( mi->quants[j] < 0 ) continue;
        gq = GEN_QUANT(mi->quants[j]);
        cc = sign*mi->modulus*gq->modulus;
        if ( gq->flags & Q_ENERGY & q->mode )
        { op.kind = OP_FORCE; op.s = -cc; }
        else if ( gq->flags & (Q_FIXED|Q_CONSERVED) & q->mode )
        { op.kind = OP_VGRAD; op.s = cc;
          op.r = gq->fixnum; op.c = mi->quants[j]; op.b = gq->b_id;
        }
        else continue;
        op.v = q_info->v[i];
        memcpy(op.x,q_info->grad[i],SDIM*sizeof(REAL));
        put(b,&op);
      }
  }
}

#define FH_MAXTHREADS 256

/* One element's work: record its additions in b. c: the facet's corners
   (facets only). me: the thread number. */
typedef void (*element_fn)(struct linsys *S, element_id id, vertex_id *c, int me,
                           struct hbuf *b, void *ctx);

/* ---- Sparsity pattern kept across Newton steps --------------------------
 *
 * Each Newton step adds its matrix entries to a hash table, one at a time
 * on one thread, and sp_hash_end() sorts them into CSR arrays. While the
 * topology is unchanged the next step has (nearly) the same entries, so
 * the last CSR pattern is kept: hessian_init() starts the system with
 * fl_pattern_begin(), sp_hash_search() adds straight into the kept value
 * array (fl_pattern_add()), and run_elements() adds each element's entries
 * in parallel as soon as the element is done (atomically: the order of
 * the additions to an entry, and so its round-off, can vary from run to
 * run with several threads). Entries missing from the pattern go to the
 * hash table as before; sp_hash_end() then merges the two and keeps the
 * new pattern. Pattern entries that get no additions stay explicit zeros
 * (a merge drops them). PYSE_NO_PATTERN=1 turns this off.
 */

static struct linsys *pc_S = NULL;   /* the system being filled with it */
static int pc_live = 0;              /* the kept pattern applies to pc_S */
static int pc_rows = -1;             /* rows (= columns) of the kept pattern */
static long pc_stamp = -1;           /* top_timestamp it was made at */
static int *pc_IA = NULL;            /* 0-based CSR row starts */
static int *pc_JA = NULL;            /* columns, ascending within a row */
static REAL *pc_A = NULL;            /* values */
static long pc_nnz = 0;

static int pattern_disabled(void)
{ static int disabled = -1;
  if ( disabled < 0 ) disabled = getenv("PYSE_NO_PATTERN") != NULL;
  return disabled || fl_disabled();
}

void fl_pattern_begin(struct linsys *S)
{ pc_S = NULL;
  pc_live = 0;
  if ( pattern_disabled() ) return;
  pc_S = S;
  pc_live = pc_IA && pc_stamp == top_timestamp && pc_rows == S->total_rows;
  if ( pc_live ) memset(pc_A,0,pc_nnz*sizeof(REAL));
}

/* position of entry (row, col) in the kept pattern, or -1 */
static long pattern_find(int row, int col)
{ long lo, hi;
  if ( row < 0 || row >= pc_rows ) return -1;
  lo = pc_IA[row];
  hi = pc_IA[row+1];
  while ( lo < hi )
  { long mid = (lo + hi)/2;
    if ( pc_JA[mid] < col ) lo = mid + 1;
    else hi = mid;
  }
  return ( lo < pc_IA[row+1] && pc_JA[lo] == col ) ? lo : -1;
}

int fl_pattern_add(struct linsys *S, int row, int col, REAL value)
{ long pos;
  if ( S != pc_S || !pc_live ) return 0;
  pos = pattern_find(row,col);
  if ( pos < 0 ) return 0;
  pc_A[pos] += value;
  return 1;
}

/* After an element's work: add its matrix entries to the kept pattern
   (from any thread); those found are marked done for the serial replay. */
static void pattern_add_entries(struct hbuf *b, long from)
{ long e;
  for ( e = from ; e < b->n ; e++ )
  { struct hop *op = b->ops + e;
    long pos;
    if ( op->kind != OP_ENTRY ) continue;
    pos = pattern_find(op->r,op->c);
    if ( pos < 0 ) continue;   /* new entry: hash table, in the replay */
#ifdef _OPENMP
    #pragma omp atomic
#endif
    pc_A[pos] += op->x[0];
    op->kind = OP_DONE;
  }
}

int fl_pattern_end(struct linsys *S, int rows, int cols, int index_start)
{ long i;
  if ( S != pc_S ) return -1;
  if ( pc_live && rows == pc_rows && cols == pc_rows && S->hashcount == 0 )
  { /* every entry was in the pattern: hand out a copy of it */
    S->N = rows;
    S->maxN = rows;
    S->IA = (int *)temp_calloc(rows+1,sizeof(int));
    S->maxA = (int)pc_nnz + S->maxN;
    S->JA = (int *)temp_calloc(S->maxA,sizeof(int));
    S->A = (REAL *)temp_calloc(S->maxA,sizeof(REAL));
    for ( i = 0 ; i <= rows ; i++ ) S->IA[i] = pc_IA[i] + index_start;
    for ( i = 0 ; i < pc_nnz ; i++ ) S->JA[i] = pc_JA[i] + index_start;
    memcpy(S->A,pc_A,pc_nnz*sizeof(REAL));
    if ( !hessian_quiet_flag )
    { sprintf(msg,"Sparse entries: %ld (kept pattern)\n",pc_nnz);
      outstring(msg);
    }
    temp_free((char*)S->hashtable);
    S->hashtable = NULL;
    pc_S = NULL;
    return (int)(pc_nnz + pc_nnz/3);
  }
  if ( pc_live )
  { /* some new entries: merge the pattern's into the hash table, then
       sp_hash_end() goes on as usual and fl_pattern_store() keeps the
       result */
    int row;
    pc_live = 0;   /* so sp_hash_search() goes to the table */
    for ( row = 0 ; row < pc_rows ; row++ )
      for ( i = pc_IA[row] ; i < pc_IA[row+1] ; i++ )
        sp_hash_search(S,row,pc_JA[i],pc_A[i]);   /* zeros are dropped */
  }
  return -1;
}

void fl_pattern_store(struct linsys *S, int rows, int cols, int index_start)
{ long n, i;
  int *ia, *ja;
  REAL *a;
  if ( S != pc_S ) return;
  pc_S = NULL;
  pc_live = 0;
  pc_rows = -1;   /* nothing kept unless all goes well */
  if ( rows != cols ) return;
  n = S->IA[rows] - index_start;
  ia = (int *)realloc(pc_IA,(rows+1)*sizeof(int));
  if ( ia ) pc_IA = ia;
  ja = (int *)realloc(pc_JA,(n > 0 ? n : 1)*sizeof(int));
  if ( ja ) pc_JA = ja;
  a = (REAL *)realloc(pc_A,(n > 0 ? n : 1)*sizeof(REAL));
  if ( a ) pc_A = a;
  if ( !ia || !ja || !a ) return;
  for ( i = 0 ; i <= rows ; i++ ) pc_IA[i] = S->IA[i] - index_start;
  for ( i = 0 ; i < n ; i++ ) pc_JA[i] = S->JA[i] - index_start;
  pc_nnz = n;
  pc_rows = rows;
  pc_stamp = top_timestamp;
}

/* Run fn on all elements of `type` in chunks: in parallel within a chunk,
   then make the recorded additions serially in element order. An element
   whose work raises an error or warning (kb_error()) is redone serially
   then, so Evolver reports it as usual. Returns 0 (nothing done) if it
   can't start. */
static int run_elements(struct linsys *S, REAL *rhs, int type, element_fn fn,
                        void *ctx, int threads)
{ static struct hbuf buf[FH_MAXTHREADS];
  static struct hbuf serial;   /* elements redone serially */
  element_id *list;
  vertex_id *corners = NULL;
  long n, start, chunk, k;
  long *fstart;   /* per chunk element: start, end in its thread's buffer */
  int *fthread;
  unsigned char *ftrapped;     /* per chunk element: hit an error */
  int t, ok = 1, live;

  if ( type == FACET )
  { corners = fl_facet_corners();
    list = fl_facet_list(&n);
    if ( !corners ) return 0;
  }
  else list = fl_element_list(type,&n);
  if ( !list || n == 0 ) return 0;
  chunk = 64*threads;
  fstart = (long*)malloc(2*chunk*sizeof(long));
  fthread = (int*)malloc(chunk*sizeof(int));
  ftrapped = (unsigned char*)malloc(chunk);
  if ( !fstart || !fthread || !ftrapped )
  { free(fstart); free(fthread); free(ftrapped); return 0; }
  fl_prepare_threads(threads);
  live = S && S == pc_S && pc_live;

#define CORNERS(id) (corners ? corners + 3*ordinal(id) : NULL)
  for ( start = 0 ; start < n && ok ; start += chunk )
  { long end = start + chunk < n ? start + chunk : n;
    for ( t = 0 ; t < threads ; t++ ) { buf[t].n = 0; buf[t].failed = 0; }

    fl_enter();
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic,4) num_threads(threads)
#endif
    for ( k = start ; k < end ; k++ )
    { int me = 0;
      jmp_buf trap;
#ifdef _OPENMP
      me = omp_get_thread_num();
#endif
      fthread[k-start] = me;
      fstart[2*(k-start)] = buf[me].n;
      ftrapped[k-start] = 0;
      recording = &buf[me];
      if ( setjmp(trap) == 0 )
      { fl_trap = &trap;   /* kb_error() comes back here */
        (*fn)(S,list[k],CORNERS(list[k]),me,&buf[me],ctx);
        fl_trap = NULL;
        if ( live ) pattern_add_entries(&buf[me],fstart[2*(k-start)]);
      }
      else   /* error or warning: drop this element's records, redo it below */
      { buf[me].n = fstart[2*(k-start)];
        ftrapped[k-start] = 1;
      }
      recording = NULL;
      fstart[2*(k-start)+1] = buf[me].n;
    }
    fl_leave();   /* may abort (Ctrl-C): nothing of this chunk added yet */
    for ( t = 0 ; t < threads ; t++ )
      if ( buf[t].failed ) ok = 0;
    if ( !ok ) break;   /* out of memory: give up (nothing added this chunk) */

    /* the additions, serially in element order */
    for ( k = start ; k < end ; k++ )
    { struct hbuf *b = buf + fthread[k-start];
      long e, e0 = fstart[2*(k-start)], e1 = fstart[2*(k-start)+1];
      if ( ftrapped[k-start] )
      { /* redo serially: Evolver reports the error or warning as usual */
        serial.n = 0;
        serial.failed = 0;
        recording = &serial;
        (*fn)(S,list[k],CORNERS(list[k]),0,&serial,ctx);
        recording = NULL;
        if ( serial.failed ) { ok = 0; break; }
        b = &serial;
        e0 = 0;
        e1 = serial.n;
      }
      for ( e = e0 ; e < e1 ; e++ )
      { struct hop *op = b->ops + e;
        switch ( op->kind )
        { case OP_ENTRY: sp_hash_search(S,op->r,op->c,op->x[0]); break;
          case OP_GRAD: fill_grad(S,get_vertex_vhead(op->v),op->x,rhs); break;
          case OP_VALUE:    /* as calc_quant_grads(), calc_quant_hess() */
          { struct method_instance *mi = METH_INSTANCE(op->r);
            mi->newvalue += op->x[0];
            mi->abstotal += op->x[1];
            break;
          }
          case OP_QVALUE:   /* as calc_quants() */
          { struct method_instance *mi = METH_INSTANCE(op->r);
            binary_tree_add(mi->value_addends,op->x[0]);
            mi->abstotal += fabs(op->x[0]);
            break;
          }
          case OP_FORCE:    /* as calc_quant_grads() */
            vector_add_smul(get_force(op->v),op->x,op->s,SDIM);
            break;
          case OP_AREA:     /* methods' total area, see fl_total_area_add() */
            binary_tree_add(web.total_area_addends,op->x[0]);
            break;
          case OP_VGRAD:
          { volgrad *vgptr = get_bv_new_vgrad(op->r,op->v);
            vgptr->bb_id = op->b;
            vgptr->qnum = op->c;
            vector_add_smul(vgptr->grad,op->x,op->s,SDIM);
            break;
          }
        }
      }
    }
  }
#undef CORNERS
  free(fstart);
  free(fthread);
  free(ftrapped);
  if ( !ok )
    kb_error(6350,"Out of memory in a parallel element loop.\n",RECOVERABLE);
  return 1;
}

static int hess_threads(void)
{ int threads = fl_threads();
  if ( threads > FH_MAXTHREADS ) threads = FH_MAXTHREADS;
  return threads < 1 ? 1 : threads;
}

void fh_reset(void) { recording = NULL; pc_S = NULL; pc_live = 0; }

/* The area methods add each facet's area to the total area: recorded
   while run_elements() runs (the addition is made in element order),
   added right away otherwise. */
void fl_total_area_add(REAL area)
{ if ( recording )
  { struct hop op;
    op.kind = OP_AREA; op.x[0] = area;
    put(recording,&op);
  }
  else binary_tree_add(web.total_area_addends,area);
}

/* ---- entry points from calc_quants(), calc_quant_grads(), calc_quant_hess() ---- */

static int quant_elements(struct linsys *S, REAL *rhs, int type, element_fn fn,
                          int qmode, int mode, int global_needs)
{ static struct qinfo qi[FH_MAXTHREADS];
  struct quant_ctx ctx;
  element_id *list;
  long n;
  int threads, t, done;

  if ( type == FACET ) list = fl_facet_list(&n);
  else list = fl_element_list(type,&n);
  if ( !list || n == 0 ) return 0;
  threads = hess_threads();
  /* per-thread work space (Evolver's allocator is not thread-safe) */
  for ( t = 0 ; t < threads ; t++ )
  { q_info_init(&qi[t],qmode);
    if ( qmode == METHOD_HESSIAN )
      qi[t].hess = dmatrix4(MAXVCOUNT,MAXVCOUNT,SDIM,SDIM);
  }
  /* first element serially: lazy one-time setup (e.g. packed basis matrices) */
  qi[0].id = list[0];
  (*q_setup[type])(S,&qi[0],global_needs|NEED_SIDE);
  ctx.qi = qi;
  ctx.type = type;
  ctx.global_needs = global_needs;
  ctx.meth_offset = get_meth_offset(type);
  ctx.mode = mode;
  done = run_elements(S,rhs,type,fn,&ctx,threads);
  if ( done ) comp_quant_stamp += (int)n;
  for ( t = 0 ; t < threads ; t++ )
  { if ( qi[t].hess ) { free_matrix4(qi[t].hess); qi[t].hess = NULL; }
    q_info_free(&qi[t]);
  }
  return done;
}

int fl_quant_values(int type, int mode, int global_needs)
{ if ( !quant_ok(type,PASS_VALUE) ) return 0;
  return quant_elements(NULL,NULL,type,quant_value_element,METHOD_VALUE,mode,global_needs);
}

int fl_quant_grads(int type, int mode, int global_needs)
{ if ( !quant_ok(type,PASS_GRAD) ) return 0;
  return quant_elements(NULL,NULL,type,quant_grad_element,METHOD_GRADIENT,mode,global_needs);
}

int fl_quant_hess(struct linsys *S, int type, int hess_mode, int mode, REAL *rhs,
                  int global_needs)
{ if ( !hess_mode || !quant_ok(type,PASS_HESS) ) return 0;
  return quant_elements(S,rhs,type,quant_hess_element,METHOD_HESSIAN,mode,global_needs);
}

/* ---- linear model: area and body volume Hessians (hessian3.c) ---- */

static int linear_ok(void)
{ return !fl_disabled() && !threadflag && web.representation == SOAPFILM
      && web.modeltype == LINEAR && SDIM == 3 && !web.symmetry_flag
      && !web.torus_flag;
}

static void side_of(vertex_id *c, int i, REAL *side)
{ REAL *t = get_coord(c[i]), *h = get_coord(c[(i+1)%3]);
  int k;
  for ( k = 0 ; k < SDIM ; k++ ) side[k] = h[k] - t[k];   /* get_fe_side() */
}

/* area_hessian()'s loop body */
static void area_facet(struct linsys *S, facet_id f_id, vertex_id *c, int me,
                       struct hbuf *b, void *ctx)
{ REAL side[FACET_EDGES][MAXCOORD], ss[FACET_EDGES], sd[FACET_EDGES];
  REAL first[FACET_VERTS][MAXCOORD], two_area;
  struct hess_verlist *v[FACET_VERTS];
  REAL density = get_facet_density(f_id);
  MAT2D(self2,MAXCOORD,MAXCOORD);
  MAT2D(otherD,MAXCOORD,MAXCOORD);
  struct hop op;
  int i, j, k;
  (void)me; (void)ctx; (void)S;

  if ( density == 0.0 ) return;
  for ( i = 0 ; i < FACET_EDGES ; i++ )
  { v[i] = get_vertex_vhead(c[i]);
    side_of(c,i,side[i]);
  }
  for ( i = 0 ; i < FACET_EDGES ; i++ )
  { ss[i] = SDIM_dot(side[i],side[i]);
    sd[i] = -SDIM_dot(side[(i+1)%FACET_EDGES],side[(i+2)%FACET_EDGES]);
  }
  two_area = sqrt(ss[1]*ss[2] - sd[0]*sd[0]);

  for ( i = 0 ; i < FACET_EDGES ; i++ )
  { int ii = (i+2)%FACET_EDGES, jj = (i+1)%FACET_EDGES;
    if ( v[i]->freedom <= 0 ) continue;
    op.kind = OP_GRAD; op.v = c[i];
    for ( k = 0 ; k < SDIM ; k++ )
    { first[i][k] = (side[ii][k]*ss[jj] - sd[i]*(-side[jj][k]))/2/two_area;
      op.x[k] = density*first[i][k];
    }
    put(b,&op);
  }

  if ( hess_flag )
    for ( i = 0 ; i < FACET_EDGES ; i++ )
    { REAL self, other;
      int ii = (i+2)%FACET_EDGES, jj = (i+1)%FACET_EDGES;
      if ( v[i]->freedom <= 0 ) continue;
      for ( j = 0 ; j < SDIM ; j++ )
        for ( k = 0 ; k < SDIM ; k++ )
        { self = -(side[jj][j]*side[jj][k])/2;
          if ( j == k ) self += ss[jj]/2;
          self2[j][k] = density*(self - 2*first[i][j]*first[i][k])/two_area;
          if ( v[jj]->freedom <= 0 ) continue;
          other = -side[ii][j]*side[jj][k] - side[jj][j]*(-side[ii][k])/2;
          if ( j == k ) other -= sd[i]/2;
          otherD[j][k] = density*(other - 2*first[i][j]*first[jj][k])/two_area;
        }
      self_entry(b,c[i],self2);
      if ( v[jj]->freedom > 0 )
        mixed_entry(b,c[i],c[jj],otherD);
    }
}

int fl_area_hessian(struct linsys *S, REAL *rhs)
{ if ( !linear_ok() ) return 0;
  return run_elements(S,rhs,FACET,area_facet,NULL,hess_threads());
}

/* body_hessian()'s first facet loop: volume constraint gradients */
static void body_linear_facet(struct linsys *S, facet_id f_id, vertex_id *c, int me,
                              struct hbuf *b, void *ctx)
{ REAL *Z = (REAL *)ctx;
  REAL side[FACET_EDGES][MAXCOORD], *x[FACET_VERTS], coe, zsum, ssum;
  struct hess_verlist *v[FACET_VERTS];
  body_id b_id, bb_id;
  int do_b, do_bb, i, j;
  ATTR a;
  struct hop op;
  (void)me;

  if ( get_attr(f_id) & NONCONTENT ) return;
  b_id = get_facet_body(f_id);
  if ( valid_id(b_id) )
  { a = get_battr(b_id);
    do_b = (a & FIXEDVOL) && !(a & REDUNDANT_BIT);
  } else do_b = 0;
  bb_id = get_facet_body(inverse_id(f_id));
  if ( valid_id(bb_id) )
  { a = get_battr(bb_id);
    do_bb = (a & FIXEDVOL) && !(a & REDUNDANT_BIT);
  } else do_bb = 0;
  if ( !do_b && !do_bb ) return;

  for ( i = 0 ; i < FACET_EDGES ; i++ )
  { v[i] = get_vertex_vhead(c[i]);
    x[i] = get_coord(c[i]);
    side_of(c,i,side[i]);
  }
  coe = 0.0;
  if ( valid_id(b_id) ) coe += Z[loc_ordinal(b_id)];
  if ( valid_id(bb_id) ) coe -= Z[loc_ordinal(bb_id)];
  zsum = (x[0][2]+x[1][2]+x[2][2]);
  ssum = side[0][0]*side[1][1] - side[0][1]*side[1][0];

  for ( i = 0 ; i < FACET_VERTS ; i++ )
  { REAL g[MAXCOORD], gg[MAXCOORD], *ggg;
    int ii = (i+1)%3, iii = (i+2)%3;
    if ( web.symmetric_content )
      cross_prod(x[ii],x[iii],g);
    else
    { g[0] = zsum*(x[ii][1]-x[iii][1]);
      g[1] = zsum*(x[iii][0]-x[ii][0]);
      g[2] = ssum;
    }
    op.kind = OP_GRAD; op.v = c[i];
    for ( j = 0 ; j < SDIM ; j++ ) op.x[j] = -coe*g[j]/6;
    put(b,&op);
    if ( !hess_flag ) continue;
    if ( v[i]->proj )
    { vec_mat_mul(g,v[i]->proj,gg,SDIM,v[i]->freedom);
      ggg = gg;
    }
    else ggg = g;
    if ( do_b )
    { int currentrow = S->bodyrowstart + loc_ordinal(b_id);
      for ( j = 0 ; j < v[i]->freedom ; j++ )
        put_entry(b,v[i]->rownum+j,currentrow,ggg[j]/6);
    }
    if ( do_bb )
    { int currentrow = S->bodyrowstart + loc_ordinal(bb_id);
      for ( j = 0 ; j < v[i]->freedom ; j++ )
        put_entry(b,v[i]->rownum+j,currentrow,-ggg[j]/6);
    }
  }
}

int fl_body_hessian_linear(struct linsys *S, REAL *rhs, REAL *Z)
{ if ( !linear_ok() ) return 0;
  return run_elements(S,rhs,FACET,body_linear_facet,Z,hess_threads());
}

/* body_hessian()'s second facet loop: volume constraint Hessians */
static void body_quadratic_facet(struct linsys *S, facet_id f_id, vertex_id *c, int me,
                                 struct hbuf *b, void *ctx)
{ REAL *Z = (REAL *)ctx;
  REAL *x[FACET_VERTS], coe, zsum;
  struct hess_verlist *v[FACET_VERTS];
  body_id b_id;
  int i;
  MAT2D(otherD,MAXCOORD,MAXCOORD);
  MAT2D(self,MAXCOORD,MAXCOORD);
  (void)me; (void)S;

  otherD[0][0] = otherD[1][1] = otherD[2][2] = 0.0;
  self[0][0] = self[1][1] = self[2][2] = 0.0;
  self[0][1] = self[1][0] = 0.0;
  if ( get_attr(f_id) & NONCONTENT ) return;
  coe = 0.0;
  b_id = get_facet_body(f_id);
  if ( valid_id(b_id) ) coe += Z[loc_ordinal(b_id)];
  b_id = get_facet_body(inverse_id(f_id));
  if ( valid_id(b_id) ) coe -= Z[loc_ordinal(b_id)];
  coe /= 6;
  if ( coe == 0.0 ) return;
  for ( i = 0 ; i < FACET_EDGES ; i++ )
  { v[i] = get_vertex_vhead(c[i]);
    x[i] = get_coord(c[i]);
  }
  zsum = x[0][2] + x[1][2] + x[2][2];
  for ( i = 0 ; i < FACET_EDGES ; i++ )
  { int jj = (i+1)%FACET_EDGES, ii = (i+2)%FACET_EDGES;
    if ( v[i]->freedom <= 0 ) continue;
    if ( !web.symmetric_content )
    { self[0][2] = self[2][0] = coe*(x[jj][1]-x[ii][1]);
      self[1][2] = self[2][1] = coe*(x[ii][0]-x[jj][0]);
      self_entry(b,c[i],self);
    }
    if ( v[jj]->freedom <= 0 ) continue;
    if ( web.symmetric_content )
    { otherD[0][1] = -coe*x[ii][2];
      otherD[1][0] =  coe*x[ii][2];
      otherD[0][2] =  coe*x[ii][1];
      otherD[2][0] = -coe*x[ii][1];
      otherD[2][1] =  coe*x[ii][0];
      otherD[1][2] = -coe*x[ii][0];
    }
    else
    { otherD[0][1] = -coe*zsum;
      otherD[1][0] =  coe*zsum;
      otherD[0][2] = -coe*(x[jj][1]-x[ii][1]);
      otherD[2][0] = -coe*(x[ii][1]-x[i ][1]);
      otherD[1][2] = -coe*(x[ii][0]-x[jj][0]);
      otherD[2][1] = -coe*(x[i ][0]-x[ii][0]);
    }
    mixed_entry(b,c[i],c[jj],otherD);
  }
}

int fl_body_hessian_quadratic(struct linsys *S, REAL *Z)
{ if ( !linear_ok() ) return 0;
  return run_elements(S,NULL,FACET,body_quadratic_facet,Z,hess_threads());
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

/* ---- Lagrange facet area Hessian as matrix products ----------------------
 *
 * lagrange_facet_tension_hess() for 2D facets, regrouped. At each Gauss point
 * the Hessian of fudge*sqrt(det) over (node, coordinate) pairs is
 *     fudge [ -s s^T/det + (a+b)(a+b)^T - (a-b)(a-b)^T - (c+d)(c+d)^T ]
 *   + I (x) fudge G^T adj G,
 * with a = A (x) t0, b = B (x) t1, c = B (x) t0, d = A (x) t1 (A, B: the basis
 * polynomials' partials at the point, t0, t1: the tangents, adj: the adjugate
 * of the metric, s: the gradient vector). Summed over the Gauss points that is
 * two symmetric rank-k updates (BLAS dsyrk) instead of loops over node pairs;
 * the same arithmetic up to rounding order. PYSE_NO_FAST_LAGRANGE=1 turns it off.
 */
#ifdef PYSE_MUMPS          /* builds with MUMPS link a BLAS */
extern void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
                   const double *alpha, const double *a, const int *lda,
                   const double *beta, double *c, const int *ldc);

static __thread double *lt_buf = NULL;
static __thread size_t lt_size = 0;

static int fast_lagrange_disabled(void)
{ static int disabled = -1;
  if ( disabled < 0 ) disabled = getenv("PYSE_NO_FAST_LAGRANGE") != NULL;
  return disabled;
}

int fl_lagrange_tension_hess(struct qinfo *f_info, REAL density, REAL *energy)
{ struct gauss_lag *gl;
  int n, G, S = SDIM, N3, m, k, kk, j, jj, np = 0, nn = 0;
  double *P, *Q, *C, *M;
  size_t need;
  REAL value = 0.0;
  const double one = 1.0, minus = -1.0, zero = 0.0;

  if ( fast_lagrange_disabled() || web.dimension != 2 || sizeof(REAL) != sizeof(double) )
    return 0;
  gl = &gauss_lagrange[2][web.gauss2D_order];
  n = gl->lagpts;
  G = gl->gnumpts;
  N3 = n*S;
  need = (size_t)N3*G + (size_t)N3*3*G + (size_t)N3*N3 + (size_t)n*n;
  if ( need > lt_size )
  { double *buf = (double *)realloc(lt_buf,need*sizeof(double));
    if ( !buf ) return 0;
    lt_buf = buf;
    lt_size = need;
  }
  P = lt_buf;                 /* positive columns: (a+b) */
  Q = P + (size_t)N3*G;       /* negative columns: s/sqrt(det), (a-b), (c+d) */
  C = Q + (size_t)N3*3*G;     /* the sum, column-major, lower triangle */
  M = C + (size_t)N3*N3;      /* sum of fudge G^T adj G, lower triangle */
  memset(M,0,(size_t)n*n*sizeof(double));

  for ( m = 0 ; m < G ; m++ )
  { REAL **t = f_info->sides[m];
    REAL **gp = gl->gpolypart[m];
    REAL g00 = SDIM_dot(t[0],t[0]), g01 = SDIM_dot(t[0],t[1]), g11 = SDIM_dot(t[1],t[1]);
    REAL det = g00*g11 - g01*g01;
    REAL adj00 = g11, adj01 = -g01, adj11 = g00;
    REAL fudge, rf, rs;
    double *ab, *s, *am, *cd;
    if ( det <= 0.0 ) continue;
    value += gl->gausswt[m]*sqrt(det);
    fudge = density*gl->gausswt[m]/sqrt(det)/factorial[2];
    rf = sqrt(fudge);
    rs = sqrt(fudge/det);
    ab = P + (size_t)np*N3;
    s = Q + (size_t)nn*N3;
    am = s + N3;
    cd = am + N3;
    np += 1;
    nn += 3;
    for ( k = 0 ; k < n ; k++ )
    { REAL A = gp[0][k], B = gp[1][k];
      REAL w0 = A*adj00 + B*adj01, w1 = A*adj01 + B*adj11;
      for ( j = 0 ; j < S ; j++ )
      { REAL t0 = t[0][j], t1 = t[1][j];
        REAL sum = t0*w0 + t1*w1;               /* the gradient part */
        f_info->grad[k][j] += fudge*sum;
        s[k*S + j] = rs*sum;
        ab[k*S + j] = rf*(A*t0 + B*t1);
        am[k*S + j] = rf*(A*t0 - B*t1);
        cd[k*S + j] = rf*(B*t0 + A*t1);
      }
      for ( kk = 0 ; kk <= k ; kk++ )
      { REAL Ak = gp[0][kk], Bk = gp[1][kk];
        M[k*n + kk] += fudge*(A*(Ak*adj00 + Bk*adj01) + B*(Ak*adj01 + Bk*adj11));
      }
    }
  }
  if ( np )
  { dsyrk_("L","N",&N3,&np,&one,P,&N3,&zero,C,&N3);
    dsyrk_("L","N",&N3,&nn,&minus,Q,&N3,&one,C,&N3);
  }
  else memset(C,0,(size_t)N3*N3*sizeof(double));

  for ( k = 0 ; k < n ; k++ )
    for ( kk = 0 ; kk <= k ; kk++ )
      for ( j = 0 ; j < S ; j++ )
      { int jjend = (k == kk) ? j + 1 : S;
        for ( jj = 0 ; jj < jjend ; jj++ )
        { REAL h = C[(size_t)(kk*S + jj)*N3 + k*S + j];
          if ( j == jj ) h += M[k*n + kk];
          f_info->hess[k][kk][j][jj] += h;
          if ( kk != k || jj != j )
            f_info->hess[kk][k][jj][j] += h;
        }
      }
  *energy = density*value/factorial[2];
  return 1;
}

/* lagrange_facet_volume_all() in METHOD_HESSIAN mode for 2D facets, regrouped:
   over (node, node) pairs its Hessian blocks are sums over the Gauss points of
   outer products, so (x,y) = sum w z (A B^T - B A^T), (j,z) = sum w R_j p^T
   (R_j: the j-th row of adj(sides) times the partials, p: the basis
   polynomials) and their transposes: four small BLAS dgemm calls. */
extern void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
                   const int *k, const double *alpha, const double *a, const int *lda,
                   const double *b, const int *ldb, const double *beta, double *c,
                   const int *ldc);

static __thread double *lv_buf = NULL;
static __thread size_t lv_size = 0;

int fl_lagrange_volume_hess(struct qinfo *f_info, REAL *volume)
{ struct gauss_lag *gl;
  int n, G, m, k, kk, used = 0;
  double *Aw, *Bw, *A, *B, *R0, *R1, *Pg, *X, *Y0, *Y1;
  size_t need;
  REAL value = 0.0;
  const double one = 1.0, minus = -1.0, zero = 0.0;

  if ( fast_lagrange_disabled() || web.dimension != 2 || SDIM != 3
       || sizeof(REAL) != sizeof(double) )
    return 0;
  gl = &gauss_lagrange[2][web.gauss2D_order];
  n = gl->lagpts;
  G = gl->gnumpts;
  need = (size_t)7*n*G + (size_t)3*n*n;
  if ( need > lv_size )
  { double *buf = (double *)realloc(lv_buf,need*sizeof(double));
    if ( !buf ) return 0;
    lv_buf = buf;
    lv_size = need;
  }
  Aw = lv_buf; Bw = Aw + n*G; A = Bw + n*G; B = A + n*G;     /* n x G, column-major */
  R0 = B + n*G; R1 = R0 + n*G; Pg = R1 + n*G;
  X = Pg + n*G; Y0 = X + n*n; Y1 = Y0 + n*n;                /* n x n */

  for ( m = 0 ; m < G ; m++ )
  { REAL **sd = f_info->sides[m];
    REAL **gp = gl->gpolypart[m];
    REAL z = f_info->gauss_pt[m][2];
    REAL w = gl->gausswt[m]/factorial[2];
    /* adjugate of the 2x2 x,y part of the tangents */
    REAL a00 = sd[1][1], a01 = -sd[0][1], a10 = -sd[1][0], a11 = sd[0][0];
    REAL det = sd[0][0]*sd[1][1] - sd[0][1]*sd[1][0];
    value += w*det*z;
    for ( k = 0 ; k < n ; k++ )
    { REAL g0 = gp[0][k], g1 = gp[1][k];
      REAL r0 = g0*a00 + g1*a01, r1 = g0*a10 + g1*a11;   /* sum_i gp[i][k] mat[j][i] */
      f_info->grad[k][0] += w*z*r0;
      f_info->grad[k][1] += w*z*r1;
      f_info->grad[k][2] += w*gl->gpoly[m][k]*det;
      Aw[used*n + k] = w*z*g0;  Bw[used*n + k] = w*z*g1;
      A[used*n + k] = g0;       B[used*n + k] = g1;
      R0[used*n + k] = w*r0;    R1[used*n + k] = w*r1;
      Pg[used*n + k] = gl->gpoly[m][k];
    }
    used++;
  }
  dgemm_("N","T",&n,&n,&used,&one,Aw,&n,B,&n,&zero,X,&n);
  dgemm_("N","T",&n,&n,&used,&minus,Bw,&n,A,&n,&one,X,&n);
  dgemm_("N","T",&n,&n,&used,&one,R0,&n,Pg,&n,&zero,Y0,&n);
  dgemm_("N","T",&n,&n,&used,&one,R1,&n,Pg,&n,&zero,Y1,&n);
  for ( k = 0 ; k < n ; k++ )
    for ( kk = 0 ; kk < n ; kk++ )
    { REAL **h = f_info->hess[k][kk];
      REAL x = X[kk*n + k];
      h[0][1] += x;
      h[1][0] -= x;
      h[0][2] += Y0[kk*n + k];
      h[1][2] += Y1[kk*n + k];
      h[2][0] += Y0[k*n + kk];
      h[2][1] += Y1[k*n + kk];
    }
  *volume = value;
  return 1;
}
#else
int fl_lagrange_tension_hess(struct qinfo *f_info, REAL density, REAL *energy)
{ (void)f_info; (void)density; (void)energy; return 0; }
int fl_lagrange_volume_hess(struct qinfo *f_info, REAL *volume)
{ (void)f_info; (void)volume; return 0; }
#endif
