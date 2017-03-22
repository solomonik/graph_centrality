// betweenness centrality computation test

#include <float.h>
#include "btwn_central.h"

using namespace CTF;
Matrix <wht> read_matrix(World  & dw,
                             int      n,
                             uint64_t edges,
                             const char *fpath,
                             bool     remove_singlets,
                             int *    n_nnz,
                             int64_t  max_ewht=1){
  uint64_t *edge=NULL;
  uint64_t nedges = 0;
  Semiring<wht> s(MAX_WHT, 
                  [](wht a, wht b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](wht a, wht b){ return a+b; });
  //random adjacency matrix
  Matrix<wht> A_pre(n, n, SP, dw, s, "A_rmat");
#ifdef MPIIO
    if (dw.rank == 0) printf("Running MPI-IO graph reader n = %d... ",n);
    char **leno;
    nedges = read_graph_mpiio(dw.rank, dw.np, fpath, &edge, &leno);
    processedges(leno, nedges, dw.rank, &edge);
#else
if (dw.rank == 0) printf("Running graph reader n = %d... ",n);
  nedges = read_graph(dw.rank, dw.np, fpath, &edge);
#endif
  if (dw.rank == 0) printf("donei (%d edges).\n", nedges);
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*nedges);
  wht * vals = (wht*)malloc(sizeof(wht)*nedges);

  srand(dw.rank+1);
  for (int64_t i=0; i<nedges; i++){
    inds[i] = edge[2*i]+edge[2*i+1]*n;
    vals[i] = (rand()%max_ewht) + 1;
  }
  if (dw.rank == 0) printf("filling CTF graph\n");
  A_pre.write(nedges,inds,vals);
  A_pre["ij"] += A_pre["ji"];
  free(inds);
  free(vals);
  
  A_pre["ii"] = 0;
 
  A_pre.sparsify([](int a){ return a>0; });
 
  if (dw.rank == 0) 
    printf("A contains %ld nonzeros, proc 0 generated %lu edges\n", A_pre.nnz_tot, nedges);

  if (remove_singlets){
    Vector<int> rc(n, dw);
    rc["i"] += ((Function<wht>)([](wht a){ return (int)(a>0); }))(A_pre["ij"]);
    rc["i"] += ((Function<wht>)([](wht a){ return (int)(a>0); }))(A_pre["ji"]);
    int * all_rc; // = (int*)malloc(sizeof(int)*n);
    int64_t nval;
    rc.read_all(&nval, &all_rc);
    int n_nnz_rc = 0;
    int n_single = 0;
    for (int i=0; i<nval; i++){
      if (all_rc[i] != 0){
        if (all_rc[i] == 2) n_single++;
        all_rc[i] = n_nnz_rc;
        n_nnz_rc++;
      } else {
        all_rc[i] = -1;
      }
    }
    if (dw.rank == 0) printf("n_nnz_rc = %d of %d vertices kept, %d are 0-degree, %d are 1-degree\n", n_nnz_rc, n,(n-n_nnz_rc),n_single);
    Matrix<wht> A(n_nnz_rc, n_nnz_rc, SP, dw, s, "A");
    int * pntrs[] = {all_rc, all_rc};
 
    A.permute(0, A_pre, pntrs, 0);
    free(all_rc);
    if (dw.rank == 0) printf("preprocessed matrix has %ld edges\n", A.nnz_tot); 
  
    A["ii"] = 0;
    *n_nnz = n_nnz_rc;
    return A;
  } else {
    *n_nnz= n;
    A_pre["ii"] = 0;
    return A_pre;
  }
//  return n_nnz_rc;

}

Matrix <wht> gen_rmat_matrix(World  & dw,
                             int      scale,
                             int      ef,
                             uint64_t gseed,
                             bool     remove_singlets,
                             int *    n_nnz,
                             int64_t  max_ewht=1){
  uint64_t *edge=NULL;
  uint64_t nedges = 0;
  Semiring<wht> s(MAX_WHT, 
                  [](wht a, wht b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](wht a, wht b){ return a+b; });
  //random adjacency matrix
  int n = pow(2,scale);
  Matrix<wht> A_pre(n, n, SP, dw, s, "A_rmat");
  if (dw.rank == 0) printf("Running graph generator n = %d... ",n);
  nedges = gen_graph(scale, ef, gseed, &edge);
  if (dw.rank == 0) printf("done.\n");
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*nedges);
  wht * vals = (wht*)malloc(sizeof(wht)*nedges);

  srand(dw.rank+1);
  for (int64_t i=0; i<nedges; i++){
    inds[i] = edge[2*i]+edge[2*i+1]*n;
    vals[i] = (rand()%max_ewht) + 1;
  }
  if (dw.rank == 0) printf("filling CTF graph\n");
  A_pre.write(nedges,inds,vals);
  A_pre["ij"] += A_pre["ji"];
  free(inds);
  free(vals);
  
  A_pre["ii"] = 0;
 
  A_pre.sparsify([](int a){ return a>0; });
 
  if (dw.rank == 0) 
    printf("A contains %ld nonzeros, proc 0 generated %lu edges\n", A_pre.nnz_tot, nedges);

  if (remove_singlets){
    Vector<int> rc(n, dw);
    rc["i"] += ((Function<wht>)([](wht a){ return (int)(a>0); }))(A_pre["ij"]);
    rc["i"] += ((Function<wht>)([](wht a){ return (int)(a>0); }))(A_pre["ji"]);
    int * all_rc; // = (int*)malloc(sizeof(int)*n);
    int64_t nval;
    rc.read_all(&nval, &all_rc);
    int n_nnz_rc = 0;
    int n_single = 0;
    for (int i=0; i<nval; i++){
      if (all_rc[i] != 0){
        if (all_rc[i] == 2) n_single++;
        all_rc[i] = n_nnz_rc;
        n_nnz_rc++;
      } else {
        all_rc[i] = -1;
      }
    }
    if (dw.rank == 0) printf("n_nnz_rc = %d of %d vertices kept, %d are 0-degree, %d are 1-degree\n", n_nnz_rc, n,(n-n_nnz_rc),n_single);
    Matrix<wht> A(n_nnz_rc, n_nnz_rc, SP, dw, s, "A");
    int * pntrs[] = {all_rc, all_rc};
 
    A.permute(0, A_pre, pntrs, 0);
    free(all_rc);
    if (dw.rank == 0) printf("preprocessed matrix has %ld edges\n", A.nnz_tot); 
  
    A["ii"] = 0;
    *n_nnz = n_nnz_rc;
    return A;
  } else {
    *n_nnz= n;
    A_pre["ii"] = 0;
    return A_pre;
  }
//  return n_nnz_rc;

}
Matrix <wht> gen_uniform_matrix(World & dw,
                                int64_t n,
                                double  sp=.20,
                                int64_t  max_ewht=1){
  Semiring<wht> s(MAX_WHT, 
                  [](wht a, wht b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](wht a, wht b){ return a+b; });

  //random adjacency matrix
  Matrix<wht> A(n, n, SP, dw, s, "A");

  //fill with values in the range of [1,min(n*n,100)]
  srand(dw.rank+1);
//  A.fill_random(1, std::min(n*n,100)); 
  int nmy = ((int)std::max((int)(n*sp),(int)1))*((int)((n+dw.np-1)/dw.np));
  int64_t inds[nmy];
  wht vals[nmy];
  int i=0;
  for (int64_t row=dw.rank*n/dw.np; row<(int)(dw.rank+1)*n/dw.np; row++){
    int64_t cols[std::max((int)(n*sp),1)];
    for (int64_t col=0; col<std::max((int)(n*sp),1); col++){
      bool is_rep;
      do {
        cols[col] = rand()%n;
        is_rep = 0;
        for (int c=0; c<col; c++){
          if (cols[c] == cols[col]) is_rep = 1;
        }
      } while (is_rep);
      inds[i] = cols[col]*n+row;
      vals[i] = (rand()%max_ewht)+1;
      i++;
    }
  }
  A.write(i,inds,vals);
  
  A["ii"] = 0;
  
  //keep only values smaller than 20 (about 20% sparsity)
  //A.sparsify([=](int a){ return a<sp*100; });
   return A; 
}
// calculate betweenness centrality a graph of n nodes distributed on World (communicator) dw
int btwn_cnt(Matrix <wht>A, 
             int     n,
             World & dw,
             bool    sp_B=true,
             bool    sp_C=true,
             int     bsize=2,
             int     nbatches=1,
             int     test=0,
             bool    adapt=1,
             int     c_rep=0){

  //tropical semiring, define additive identity to be MAX_WHT to prevent integer overflow

  Vector<real> v1(n,dw);
  Vector<real> v2(n,dw);

  if (test || n<= 20){
    btwn_cnt_naive(A, v1);
    //compute centrality scores by Bellman Ford with block size bsize
    btwn_cnt_fast(A, bsize, v2, nbatches, sp_B, sp_C, adapt, c_rep);
    ((Transform<real>)([](real a, real & b){ b= std::abs(b) >= 1 ? std::min(b-a, (b-a)/b) : b-a; }))(v2["i"], v1["i"]);
    double norm = v1.norm2();
    int pass = norm <= n*1.E-3;

    if (dw.rank == 0){
      printf("error norm is %E\n",norm);
      MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
      if (pass) 
        printf("{ betweenness centrality } passed \n");
      else
        printf("{ betweenness centrality } failed \n");
    } else 
      MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    return pass;
  } else {
    if (dw.rank == 0)
      printf("Executing warm-up batch\n");
    btwn_cnt_fast(A, bsize, v2, 1, sp_B, sp_C, adapt, c_rep);
    if (dw.rank == 0)
      printf("Starting benchmarking\n");
    Timer_epoch tbtwn("Betweenness centrality");
    tbtwn.begin();
    double st_time = MPI_Wtime();
    btwn_cnt_fast(A, bsize, v2, nbatches, sp_B, sp_C, adapt, c_rep);
    tbtwn.end();
    if (dw.rank == 0){
      if (nbatches == 0) printf("Completed all batches in time %lf sec, projected total %lf sec.\n", MPI_Wtime()-st_time, MPI_Wtime()-st_time);
      else printf("Completed %d batches in time %lf sec, projected total %lf sec. (rate: %lf verts/sec.)\n", nbatches, MPI_Wtime()-st_time, (n/(bsize*nbatches))*(MPI_Wtime()-st_time), (nbatches*bsize)/(MPI_Wtime()-st_time));
    }
    return 1;
  }
} 


#ifndef TEST_SUITE
char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, n, pass, bsize, nbatches, test, scale, ef, prep, adapt, c_rep;
  int64_t max_ewht;
  bool sp_B, sp_C;
  uint64_t myseed;
  double sp;
  uint64_t edges;
  char *gfile=NULL;
  int const in_num = argc;
  char ** input_str = argv;

  /*DEFAULT VALUE*/
  ef = 0, scale = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 27;
  } else n = 27;

  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0) sp = .2;
  } else sp = .2;
  if (getCmdOption(input_str, input_str+in_num, "-bsize")){
    bsize = atoi(getCmdOption(input_str, input_str+in_num, "-bsize"));
    if (bsize < 0) bsize = 8;
  } else bsize = 8;
  if (getCmdOption(input_str, input_str+in_num, "-nbatches")){
    nbatches = atoi(getCmdOption(input_str, input_str+in_num, "-nbatches"));
    if (nbatches < 0) nbatches = 0;
  } else nbatches = 0;
  if (getCmdOption(input_str, input_str+in_num, "-test")){
    test = atoi(getCmdOption(input_str, input_str+in_num, "-test"));
    if (test < 0) test = (nbatches == 0);
  } else test = (nbatches == 0);
  if (getCmdOption(input_str, input_str+in_num, "-S")){
    scale = atoi(getCmdOption(input_str, input_str+in_num, "-S"));
    if (scale < 0) scale=10;
  } else scale=0;
  if (getCmdOption(input_str, input_str+in_num, "-E")){
    ef = atoi(getCmdOption(input_str, input_str+in_num, "-E"));
    if (ef < 0) ef=16;
  } else ef=0;
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    myseed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    myseed=SEED;
  } else myseed=SEED;
  if (getCmdOption(input_str, input_str+in_num, "-prep")){
    prep = atoi(getCmdOption(input_str, input_str+in_num, "-prep"));
    if (prep < 0) prep = 0;
  } else prep = 0;
  if (getCmdOption(input_str, input_str+in_num, "-sp_B")){
    sp_B = atoi(getCmdOption(input_str, input_str+in_num, "-sp_B"));
    if (sp_B < 0 || sp_B > 1) sp_B = 1;
  } else sp_B = 1;
  if (getCmdOption(input_str, input_str+in_num, "-sp_C")){
    sp_C = atoi(getCmdOption(input_str, input_str+in_num, "-sp_C"));
    if (sp_C < 0 || sp_C > 1) sp_C = 1;
  } else sp_C = 1;
  if (getCmdOption(input_str, input_str+in_num, "-mwht")){
    max_ewht = atoi(getCmdOption(input_str, input_str+in_num, "-mwht"));
    if (max_ewht < 1) max_ewht = 1;
  } else max_ewht = std::min(n,20);

  if (getCmdOption(input_str, input_str+in_num, "-adapt")){
    adapt = atoi(getCmdOption(input_str, input_str+in_num, "-adapt"));
    if (adapt < 0 || adapt > 1) adapt = sp_B & sp_C;;
  } else adapt = sp_B & sp_C;
  if (getCmdOption(input_str, input_str+in_num, "-c")){
    c_rep = atoi(getCmdOption(input_str, input_str+in_num, "-c"));
    if (c_rep < 1) c_rep = 0;
  } else c_rep = 0;
  if (getCmdOption(input_str, input_str+in_num, "-edges")){
    edges = atoi(getCmdOption(input_str, input_str+in_num, "-edges"));
    if (edges < 1) edges = 1;
  } else edges=0;
  if (getCmdOption(input_str, input_str+in_num, "-f")){
    gfile = getCmdOption(input_str, input_str+in_num, "-f");
    if (edges < 1) edges = 1;
  }
  {
    World dw(argc, argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0){
      printf("Computing betweeness centrality of %d batches of size %d right operand sparsity set to %d output sparsity set to %d, verification set to %d, prep set to %d, max edge weight is %ld (n. of task %d)\n", nbatches, bsize, sp_B, sp_C, test, prep, max_ewht, world_size);
      if (test && (nbatches != 0 && nbatches != 1 && nbatches != n/bsize)){
        printf("Since testing, overriding nbatches to %d (all batches)\n",0);
      }
    }
    if (test) nbatches = 0;
    if (n > 0 && edges > 0){
      if (rank == 0)
        printf("READING REAL GRAPH n=%d edges=%d\n", n, edges);
        int n_nnz = 0;
        Matrix<wht> A = read_matrix(dw, n, edges, gfile, prep, &n_nnz, max_ewht);
        pass = btwn_cnt(A,n_nnz,dw,sp_B,sp_C, bsize, nbatches, test, adapt);

    }
    else if (scale > 0 && ef > 0){
      if (rank == 0)
        printf("R-MAT MODE ON scale=%d ef=%d seed=%lu\n", scale, ef, myseed);
      int n_nnz = 0;
      Matrix<wht> A = gen_rmat_matrix(dw, scale, ef, myseed, prep, &n_nnz, max_ewht);
      pass = btwn_cnt(A,n_nnz,dw,sp_B,sp_C, bsize, nbatches, test, adapt, c_rep);
    }
    else {
      if (rank == 0)
        printf("Uniform random graph with %d nodes, with %lf percent nonzeros\n",n,100*sp);
      Matrix<wht> A = gen_uniform_matrix(dw, n, sp, max_ewht);   
      pass = btwn_cnt(A,n,dw,sp_B,sp_C, bsize, nbatches, test, adapt, c_rep);
    }
  }
  //assert(pass);
  //pass = btwn_cnt_func(dw, bsize, nbatches, test, A);
  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
