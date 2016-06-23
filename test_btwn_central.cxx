// betweenness centrality computation test

#include <float.h>
#include "btwn_central.h"

using namespace CTF;

Matrix <int> gen_rmat_matrix(World  & dw,
                             int      scale,
                             int      ef,
                             uint64_t gseed,
                             bool     remove_singlets,
                             int *    n_nnz){
  uint64_t *edge=NULL;
  uint64_t nedges = 0;
  Semiring<int> s(INT_MAX/2, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });
  //random adjacency matrix
  int n = pow(2,scale);
  Matrix<int> A_pre(n, n, SP, dw, s, "A_rmat");
  if (dw.rank == 0) printf("Running graph generator n = %d... ",n);
  nedges = gen_graph(scale, ef, gseed, &edge);
  if (dw.rank == 0) printf("done.\n");
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*nedges);
  int * vals = (int*)malloc(sizeof(int)*nedges);

  srand(dw.rank+1);
  for (int64_t i=0; i<nedges; i++){
    inds[i] = edge[2*i]+edge[2*i+1]*n;
    vals[i] = rand()%20 + 1;
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
    rc["i"] += ((Function<int>)([](int a){ return (int)(a>0); }))(A_pre["ij"]);
    rc["i"] += ((Function<int>)([](int a){ return (int)(a>0); }))(A_pre["ji"]);
    int * all_rc; // = (int*)malloc(sizeof(int)*n);
    int64_t nval;
    rc.read_all(&nval, &all_rc);
    int n_nnz_rc = 0;
    int n_single = 0;
    int n_double = 0;
    for (int i=0; i<nval; i++){
      if (all_rc[i] != 0){
        if (all_rc[i] == 1) n_single++;
        else if (all_rc[i] == 2) n_double++;
        all_rc[n_nnz_rc] = i;
        n_nnz_rc++;
      } else {
        all_rc[i] = -1;
      }
    }
    if (dw.rank == 0) printf("n_nnz_rc = %d of %d vertices kept, %d are 0-degree, %d are 1-degree, %d are 2-degree\n", n_nnz_rc, n,(n-n_nnz_rc),n_single,n_double);
    Matrix<int> A(n_nnz_rc, n_nnz_rc, SP, dw, s, "A");
    int * pntrs[] = {all_rc, all_rc};
  
    A.permute(0, A_pre, pntrs, 0);
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
Matrix <int> gen_uniform_matrix(World  & dw,
                                int64_t  n,
                                double  sp=.20){
  Semiring<int> s(INT_MAX/2, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });

  //random adjacency matrix
  Matrix<int> A(n, n, SP, dw, s, "A");

  //fill with values in the range of [1,min(n*n,100)]
  srand(dw.rank+1);
//  A.fill_random(1, std::min(n*n,100)); 
  int nmy = ((int)std::max((int)(n*sp),(int)1))*((int)((n+dw.np-1)/dw.np));
  int64_t inds[nmy];
  int vals[nmy];
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
      vals[i] = (rand()%std::min(n,(int64_t)20))+1;
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
int btwn_cnt(Matrix <int>A, 
             int     n,
             World & dw,
             bool    sp_B=true,
             bool    sp_C=true,
             int     bsize=2,
             int     nbatches=1,
             int     test=0){

  //tropical semiring, define additive identity to be INT_MAX/2 to prevent integer overflow

  Vector<double> v1(n,dw);
  Vector<double> v2(n,dw);

  if (test || n<= 20){
    btwn_cnt_naive(A, v1);
    //compute centrality scores by Bellman Ford with block size bsize
    btwn_cnt_fast(A, bsize, v2, nbatches, sp_B, sp_C);
    //v2.print();
    v1["i"] -= v2["i"];
    double norm = v1.norm2();
    int pass = norm <= n*1.E-6;

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
    btwn_cnt_fast(A, bsize, v2, 1, sp_B, sp_C);
    if (dw.rank == 0)
      printf("Starting benchmarking\n");
    Timer_epoch tbtwn("Betweenness centrality");
    tbtwn.begin();
    double st_time = MPI_Wtime();
    btwn_cnt_fast(A, bsize, v2, nbatches, sp_B, sp_C);
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
  int rank, np, n, pass, bsize, nbatches, test, scale, ef, prep;
  bool sp_B, sp_C;
  uint64_t myseed;
  double sp;

  int const in_num = argc;
  char ** input_str = argv;

  /*DEFAULT VALUE*/
  ef = 0, scale = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0) sp = .2;
  } else sp = .2;
  if (getCmdOption(input_str, input_str+in_num, "-bsize")){
    bsize = atoi(getCmdOption(input_str, input_str+in_num, "-bsize"));
    if (bsize < 0) bsize = 2;
  } else bsize = 2;
  if (getCmdOption(input_str, input_str+in_num, "-nbatches")){
    nbatches = atoi(getCmdOption(input_str, input_str+in_num, "-nbatches"));
    if (nbatches < 0) nbatches = 1;
  } else nbatches = 1;
  if (getCmdOption(input_str, input_str+in_num, "-test")){
    test = atoi(getCmdOption(input_str, input_str+in_num, "-test"));
    if (test < 0) test = 0;
  } else test = 0;
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
    if (myseed < 0) myseed=SEED;
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


  {
    World dw(argc, argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0){
      printf("Computing betweeness centrality of %d batches of size %d right operand sparsity set to %d output sparsity set to %d, verification set to %d, prep set to %d (n. of task %d)\n", nbatches, bsize, sp_B, sp_C, test, prep, world_size);
      if (test && (nbatches != 0 && nbatches != 1 && nbatches != n/bsize)){
        printf("Since testing, overriding nbatches to %d (all batches)\n",0);
      }
    }
    if (test) nbatches = 0;


    if (scale > 0 && ef > 0){
      if (rank == 0)
        printf("R-MAT MODE ON scale=%d ef=%d seed=%lu\n", scale, ef, myseed);
      int n_nnz = 0;
      Matrix <int> A = gen_rmat_matrix(dw, scale, ef, myseed, prep, &n_nnz);
      pass = btwn_cnt(A,n_nnz,dw,sp_B,sp_C, bsize, nbatches, test);
    }
    else {
      if (rank == 0)
        printf("Uniform random graph with %d nodes, with %lf percent nonzeros\n",n,100*sp);
      Matrix <int>A = gen_uniform_matrix(dw, n, sp);   
      pass = btwn_cnt(A,n,dw,sp_B,sp_C, bsize, nbatches, test);
    }
     //assert(pass);
  }
  //pass = btwn_cnt_func(dw, bsize, nbatches, test, A);
  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
