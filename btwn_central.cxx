/** \addtogroup examples 
  * @{ 
  * \defgroup btwn_central betweenness centrality
  * @{ 
  * \brief betweenness centrality computation
  */

#include <float.h>
#include "btwn_central.h"

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

void btwn_cnt_fast(Matrix<int> A, int b, Vector<double> & v, int nbatches){
  World dw = *A.wrld;
  int n = A.nrow;

  Semiring<mpath> p = get_mpath_semiring();
  Monoid<cpath> cp = get_cpath_monoid();

  for (int ib=0; ib<n && (nbatches == 0 || ib/b<nbatches); ib+=b){
    int k = std::min(b, n-ib);

    Timer_epoch tblmn("Bellman_ep");
    tblmn.begin();

    //initialize shortest mpath vectors from the next k sources to the corresponding columns of the adjacency matrices and loops with weight 0
    ((Transform<int>)([=](int& w){ w = 0; }))(A["ii"]);
    Tensor<int> iA = A.slice(ib*n, (ib+k-1)*n+n-1);
    ((Transform<int>)([=](int& w){ w = INT_MAX/2; }))(A["ii"]);

    //let shortest mpaths vectors be mpaths
    Matrix<mpath> B(n, k, dw, p, "B");
    B["ij"] = ((Function<int,mpath>)([](int w){ return mpath(w, 1); }))(iA["ij"]);

    Bivar_Function<int,mpath,mpath> * Bellman = get_Bellman_kernel();

     
    //compute Bellman Ford
    int nbl = 0;
    double sbl = MPI_Wtime();
    for (int i=0; i<n; i++, nbl++){
      Matrix<mpath> C(B);
      B.set_zero();
      CTF::Timer tbl("Bellman");
      tbl.start();
      (*Bellman)(A["ik"],C["kj"],B["ij"]);
      tbl.stop();
//      B["ij"] = ((Function<int,mpath,mpath>)([](int w, mpath p){ return mpath(p.w+w, p.m); }))(A["ik"],B["kj"]);
      B["ij"] += ((Function<int,mpath>)([](int w){ return mpath(w, 1); }))(iA["ij"]);
      Scalar<int> num_changed = Scalar<int>();
      num_changed[""] += ((Function<mpath,mpath,int>)([](mpath p, mpath q){ return (p.w!=q.w) | (p.m!=q.m); }))(C["ij"],B["ij"]);
      if (num_changed.get_val() == 0) break;
    }
    double tbl = MPI_Wtime() - sbl;

    tblmn.end();

    Timer_epoch tbrnd("Brandes_ep");
    tbrnd.begin();
    //transfer shortest mpath data to Matrix of cpaths to compute c centrality scores
    Matrix<cpath> cB(n, k, dw, cp, "cB");
    ((Transform<mpath,cpath>)([](mpath p, cpath & cp){ cp = cpath(p.w, 1./p.m, 0.); }))(B["ij"],cB["ij"]);
    Bivar_Function<int,cpath,cpath> * Brandes = get_Brandes_kernel();
    //compute centrality scores by propagating them backwards from the furthest nodes (reverse Bellman Ford)
    int nbr = 0;
    double sbr = MPI_Wtime();
    for (int i=0; i<n; i++, nbr++){
      Matrix<cpath> C(cB);
      cB.set_zero();
      CTF::Timer tbr("Brandes");
      tbr.start();
      cB["ij"] += (*Brandes)(A["ki"],C["kj"]);
      tbr.stop();
      ((Transform<mpath,cpath>)([](mpath p, cpath & cp){ 
        cp = (p.w <= cp.w) ? cpath(p.w, 1./p.m, cp.c*p.m) : cpath(p.w, 1./p.m, 0.); 
      }))(B["ij"],cB["ij"]);
      Scalar<int> num_changed = Scalar<int>();
      num_changed[""] += ((Function<cpath,cpath,int>)([](cpath p, cpath q){ return p.c!=q.c; }))(C["ij"],cB["ij"]);
      if (num_changed.get_val() == 0) break;
    }
    double tbr = MPI_Wtime() - sbr;
    tbrnd.end();
#ifndef TEST_SUITE
    if (dw.rank == 0)
      printf("(%d ,%d) iter (%lf, %lf) sec\n", nbl, nbr, tbl, tbr);
#endif
    //set self-centrality scores to zero
    //FIXME: assumes loops are zero edges and there are no others zero edges in A
    ((Transform<cpath>)([](cpath & p){ if (p.w == 0) p.c=0; }))(cB["ij"]);
    //((Transform<cpath>)([](cpath & p){ p.c=0; }))(cB["ii"]);

    //accumulate centrality scores
    v["i"] += ((Function<cpath,double>)([](cpath a){ return a.c; }))(cB["ij"]);
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

