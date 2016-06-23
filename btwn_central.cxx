/** \addtogroup examples 
  * @{ 
  * \defgroup btwn_central betweenness centrality
  * @{ 
  * \brief betweenness centrality computation
  */

#include <float.h>
#include "btwn_central.h"
#include "generator/make_graph.h"

using namespace CTF;


//overwrite printfs to make it possible to print matrices of mpaths
namespace CTF {
  template <>  
  inline void Set<mpath>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(w=%d m=%d)",((mpath*)a)[0].w,((mpath*)a)[0].m);
  }
  template <>  
  inline void Set<cpath>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(w=%d m=%f c=%lf)",((cpath*)a)[0].w,((cpath*)a)[0].m,((cpath*)a)[0].c);
  }
}

void btwn_cnt_fast(Matrix<int> A, int64_t b, Vector<double> & v, int nbatches=0, bool sp_B=true, bool sp_C=true){
  assert(sp_B || !sp_C);
  World dw = *A.wrld;
  int64_t n = A.nrow;

  Semiring<mpath> p = get_mpath_semiring();
  Monoid<cpath> cp = get_cpath_monoid();


  Matrix<mpath> speye(n,n,SP,dw,p);
  Scalar<mpath> sm(mpath(0,1),dw,p);
  speye["ii"] = sm[""];
  ((Transform<int>)([=](int& w){ w = INT_MAX/2; }))(A["ii"]);
  for (int64_t ib=0; ib<n && (nbatches == 0 || ib/b<nbatches); ib+=b){
    int64_t k = std::min(b, n-ib);

    //initialize shortest mpath vectors from the next k sources to the corresponding columns of the adjacency matrices and loops with weight 0
    //((Transform<int>)([=](int& w){ w = 0; }))(A["ii"]);
    Tensor<int> iA = A.slice(ib*n, (ib+k-1)*n+n-1);

    //let shortest mpaths vectors be mpaths
    int atr_C = 0;
    if (sp_C) atr_C = atr_C | SP;
    Matrix<mpath> B(n, k, atr_C, dw, p, "B");
    Matrix<mpath> all_B(n, k, atr_C, dw, p, "all_B");
    B["ij"] = ((Function<int,mpath>)([](int w){ return mpath(w, 1); }))(iA["ij"]);

    Bivar_Function<int,mpath,mpath> * Bellman = get_Bellman_kernel();

     
    //compute Bellman Ford
    int nbl = 0;
#ifndef TEST_SUITE
    double sbl = MPI_Wtime();
#endif
    all_B["ij"] = B["ij"]; 

    Scalar<int> num_init(dw); 
    num_init[""] += ((Function<mpath,int>)([](mpath p){ return p.w<INT_MAX/2; }))(B["ij"]);
    int64_t nnz_last = num_init.get_val();
    double t_all_last = 0.0, t_bm_last = 0.0;
    int64_t nnz_out = 0;
    for (int i=0; i<n; i++, nbl++){
      double t_st = MPI_Wtime();
      Matrix<mpath> C(B);
      B.set_zero();
      if (sp_B || sp_C){
        C.sparsify([](mpath p){ return p.w < INT_MAX/2; });
//        if (dw.rank == 0) printf("Bellman nnz_tot = %ld\n",C.nnz_tot);
        if (!sp_C) nnz_out = C.nnz_tot;
        if (dw.rank == 0 && i!= 0){
          printf("Bellman [nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_out,A.nnz_tot,nnz_last,t_bm_last,t_all_last);
        }
        nnz_last = C.nnz_tot;
        if (C.nnz_tot == 0){ nbl--; break; }
      }
      CTF::Timer tbl("Bellman");
      tbl.start();
      double t_bm_st = MPI_Wtime();
      (*Bellman)(A["ik"],C["kj"],B["ij"]);
      double t_bm = MPI_Wtime() - t_bm_st;
      if (sp_C) nnz_out = B.nnz_tot;
      tbl.stop();
      CTF::Timer tblp1("Bellman_post_tform1");
      tblp1.start();
      ((Transform<mpath,mpath>)([](mpath p, mpath & q){ if (p.w<q.w || (p.w==q.w && q.m==0)) q.w = INT_MAX/2; } ))(all_B["ij"],B["ij"]);
      tblp1.stop();
      CTF::Timer tblp2("Bellman_post_tform2");
      tblp2.start();
      ((Transform<mpath,mpath>)([](mpath p, mpath & q){ if (p.w <= q.w){ if (p.w < q.w){ q=p; } else if (p.m > 0){ q.m+=p.m; } } }))(B["ij"],all_B["ij"]); 
      tblp2.stop();
      double t_all = MPI_Wtime() - t_st;
      if (!sp_B && !sp_C){
        Scalar<int> num_changed(dw); 
        num_changed[""] += ((Function<mpath,int>)([](mpath p){ return p.w<INT_MAX/2; }))(B["ij"]);
        int64_t nnz_new = num_changed.get_val();
        if (dw.rank == 0){
          printf("Bellman [nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_new,A.nnz_tot,nnz_last,t_bm,t_all);
          nnz_last = nnz_new;
        }
        if (nnz_new == 0) break;
      } else {
        t_all_last = t_all;
        t_bm_last = t_bm;
      }
    }
    Tensor<mpath> ispeye = speye.slice(ib*n, (ib+k-1)*n+n-1);
    all_B["ij"] += ispeye["ij"];
    
#ifndef TEST_SUITE
    double tbl = MPI_Wtime() - sbl;
#endif

    //transfer shortest mpath data to Matrix of cpaths to compute c centrality scores
    Matrix<cpath> cB(n, k, atr_C, dw, cp, "cB");
    ((Transform<mpath,cpath>)([](mpath p, cpath & cp){ cp = cpath(p.w, 1./p.m, 1.); }))(all_B["ij"],cB["ij"]);
    Matrix<cpath> all_cB(n, k, dw, cp, "all_cB");
    Bivar_Function<int,cpath,cpath> * Brandes = get_Brandes_kernel();
    //compute centrality scores by propagating them backwards from the furthest nodes (reverse Bellman Ford)
    int nbr = 0;
#ifndef TEST_SUITE
    double sbr = MPI_Wtime();
#endif
    ((Transform<mpath,cpath>)([](mpath p, cpath & cp){ cp = cpath(p.w, 1./p.m, 0.); }))(all_B["ij"],all_cB["ij"]);
    nnz_last = n*k-k;
    for (int i=0; i<n; i++, nbr++){
      double t_st = MPI_Wtime();
      Matrix<cpath> C(cB);
      if (sp_B || sp_C){
        if (!sp_C || i==0) C.sparsify([](cpath p){ return p.w >= 0 && p.w != INT_MAX/2 && p.c != 0.0; });
        if (!sp_C) nnz_out = C.nnz_tot;
        if (dw.rank == 0 && i!= 0){
          printf("Brandes [nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_out,A.nnz_tot,nnz_last,t_bm_last,t_all_last);
        }
        nnz_last = C.nnz_tot;
        if (C.nnz_tot == 0){ nbr--; break; }
      }
      cB.set_zero();
      CTF::Timer tbr("Brandes");
      tbr.start();
      double t_bm_st = MPI_Wtime();
      cB["ij"] += (*Brandes)(A["ki"],C["kj"]);
      double t_bm = MPI_Wtime() - t_bm_st;
      if (sp_C){
        nnz_out = cB.nnz_tot;
      }
      tbr.stop();
      CTF::Timer tbrp("Brandes_post_tform");
      tbrp.start();
      ((Transform<mpath,cpath>)([](mpath p, cpath & cp){ if (p.w == cp.w){ cp = cpath(p.w, 1./p.m, cp.c*p.m); } else { cp = cpath(p.w, 1./p.m, 0.0); } }))(all_B["ij"],cB["ij"]);
      if (sp_C){
        cB.sparsify([](cpath p){ return p.w >= 0 && p.c != 0.0; });
      }
      tbrp.stop();
      CTF::Timer tbra("Brandes_post_add");
      tbra.start();
      all_cB["ij"] += cB["ij"];
      tbra.stop();

      double t_all = MPI_Wtime() - t_st;
      if (!sp_B && !sp_C){
        Scalar<int> num_changed = Scalar<int>();
        num_changed[""] += ((Function<cpath,int>)([](cpath p){ return p.w >= 0 && p.c!=0.0; }))(cB["ij"]);
        int64_t nnz_new = num_changed.get_val();
        if (dw.rank == 0){
          printf("Brandes [nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_new,A.nnz_tot,nnz_last,t_bm,t_all);
          nnz_last = nnz_new;
        }
        if (nnz_new == 0) break;
      } else {
        t_all_last = t_all;
        t_bm_last = t_bm;
      }
    }
    ((Transform<mpath,cpath>)([](mpath p, cpath & cp){ if (p.w == cp.w){ cp = cpath(p.w, 1./p.m, cp.c); } else { cp = cpath(p.w, 1./p.m, 0.0); } }))(all_B["ij"],cB["ij"]);
#ifndef TEST_SUITE
    double tbr = MPI_Wtime() - sbr;
    if (dw.rank == 0)
      printf("(%d ,%d) iter (%lf, %lf) sec\n", nbl, nbr, tbl, tbr);
#endif
    //set self-centrality scores to zero
    //FIXME: assumes loops are zero edges and there are no others zero edges in A
    ((Transform<cpath>)([](cpath & p){ if (p.w == 0) p.c=0; }))(all_cB["ij"]);

    //accumulate centrality scores
    v["i"] += ((Function<cpath,double>)([](cpath a){ return a.c; }))(all_cB["ij"]);
  }
}

void btwn_cnt_naive(Matrix<int> & A, Vector<double> & v){
  World dw = *A.wrld;
  int n = A.nrow;

  Semiring<mpath> p = get_mpath_semiring();
  Monoid<cpath> cp = get_cpath_monoid();
  //mpath matrix to contain distance matrix
  Matrix<mpath> P(n, n, dw, p, "P");

  Function<int,mpath> setw([](int w){ return mpath(w, 1); });

  P["ij"] = setw(A["ij"]);
  
  ((Transform<mpath>)([=](mpath& w){ w = mpath(INT_MAX/2, 1); }))(P["ii"]);

  Matrix<mpath> Pi(n, n, dw, p);
  Pi["ij"] = P["ij"];
 
  //compute all shortest mpaths by Bellman Ford 
  for (int i=0; i<n; i++){
    ((Transform<mpath>)([=](mpath & p){ p = mpath(0,1); }))(P["ii"]);
    P["ij"] = Pi["ik"]*P["kj"];
  }
  ((Transform<mpath>)([=](mpath& p){ p = mpath(INT_MAX/2, 1); }))(P["ii"]);

  int lenn[3] = {n,n,n};
  Tensor<cpath> postv(3, lenn, dw, cp, "postv");

  //set postv_ijk = shortest mpath from i to k (d_ik)
  postv["ijk"] += ((Function<mpath,cpath>)([](mpath p){ return cpath(p.w, p.m, 0.0); }))(P["ik"]);

  //set postv_ijk = 
  //    for all nodes j on the shortest mpath from i to k (d_ik=d_ij+d_jk)
  //      let multiplicity of shortest mpaths from i to j is a, from j to k is b, and from i to k is c
  //        then postv_ijk = a*b/c
  ((Transform<mpath,mpath,cpath>)(
    [=](mpath a, mpath b, cpath & c){ 
      if (c.w<INT_MAX/2 && a.w+b.w == c.w){ c.c = ((double)a.m*b.m)/c.m; } 
      else { c.c = 0; }
    }
  ))(P["ij"],P["jk"],postv["ijk"]);

  //sum multiplicities v_j = sum(i,k) postv_ijk
  v["j"] += ((Function<cpath,double>)([](cpath p){ return p.c; }))(postv["ijk"]);
}

uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges) {

        uint64_t nedges;
        double   initiator[4] = {.57, .19, .19, .05};

        make_graph(scale, (((int64_t)1)<<scale)*edgef, seed, seed+1, initiator, (int64_t *)&nedges, (int64_t **)edges);

        return nedges;
}

uint64_t norm_graph(uint64_t *ed, uint64_t ned) {

	uint64_t l, n;

	if (ned == 0) return 0;

//	qsort(ed, ned, sizeof(uint64_t[2]), cmpedge);
	// record degrees considering multiple edges
	// and self-loop and remove them from edge list
	for(n = l = 1; n < ned; n++) {

		if (((ed[2*n]   != ed[2*(n-1)]  )  ||
		     (ed[2*n+1] != ed[2*(n-1)+1])) &&
		     (ed[2*n] != ed[2*n+1])) {

			ed[2*l]   = ed[2*n];
			ed[2*l+1] = ed[2*n+1];
			l++;
		}
	}
	return l;
}

