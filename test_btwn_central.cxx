// betweenness centrality computation test

#include <float.h>
#include "btwn_central.h"

using namespace CTF;

int btwn_cnt_rmat(int     scale,
             World & dw,
             int  ef=10,
             int     bsize=2,
             int     nbatches=1,
             int     test=0,
	     uint64_t gseed=23  ){

  //tropical semiring, define additive identity to be INT_MAX/2 to prevent integer overflow
  Semiring<int> s(INT_MAX/2, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });

  //random adjacency matrix
  int n = pow (2, scale);
  Matrix<int> A(n, n, SP, dw, s, "A");

  //fill with values in the range of [1,min(n*n,100)]
  srand(dw.rank+1);
//  A.fill_random(1, std::min(n*n,100)); 
 // int nmy = ((int)std::max((int)(n*sp),(int)1))*((int)((n+dw.np-1)/dw.np));
  //int i=0;
  uint64_t *edge=NULL;
  uint64_t nedges = 0;
//  int scale = 10;
//  int edgef = 16;
  
  printf("Graph generator: ");
  nedges = gen_graph(scale, ef, gseed, &edge);
  int64_t inds[nedges];
  int vals[nedges];

  for (int i=0; i<nedges; i++){
    inds[i] = edge[2*i]+edge[2*i]*n;
    vals[i] = rand()%20 + 1;
  }
//  uint64_t gseed = SEED;
  printf("edges generated %lu\n", nedges);
  A.write(nedges,inds,vals);
  
  A["ii"] = 0;
  
  //keep only values smaller than 20 (about 20% sparsity)
  //A.sparsify([=](int a){ return a<sp*100; });


  Vector<double> v1(n,dw);
  Vector<double> v2(n,dw);

  double st_time = MPI_Wtime();


 // v1.print();
 // v2.print();

  if (test || n<= 20){
    btwn_cnt_naive(A, v1);
    //compute centrality scores by Bellman Ford with block size bsize
    btwn_cnt_fast(A, bsize, v2);
    //v1.print();
    //v2.print();
    v1["i"] -= v2["i"];
    int pass = v1.norm2() <= 1.E-6;

    if (dw.rank == 0){
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
    btwn_cnt_fast(A, bsize, v2, 1);
    if (dw.rank == 0)
      printf("Starting benchmarking\n");
    Timer_epoch tbtwn("Betweenness centrality");
    tbtwn.begin();
    btwn_cnt_fast(A, bsize, v2, nbatches);
    tbtwn.end();
    if (dw.rank == 0){
      if (nbatches == 0) printf("Completed all batches in time %lf sec, projected total %lf sec.\n", MPI_Wtime()-st_time, MPI_Wtime()-st_time);
      else printf("Completed %d batches in time %lf sec, projected total %lf sec.\n", nbatches, MPI_Wtime()-st_time, (n/(bsize*nbatches))*(MPI_Wtime()-st_time));
    }
    return 1;
  }
}
// calculate betweenness centrality a graph of n nodes distributed on World (communicator) dw
int btwn_cnt(int     n,
             World & dw,
             double  sp=.20,
             int     bsize=2,
             int     nbatches=1,
             int     test=0){

  //tropical semiring, define additive identity to be INT_MAX/2 to prevent integer overflow
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
  for (int row=dw.rank*n/dw.np; row<(int)(dw.rank+1)*n/dw.np; row++){
    int cols[std::max((int)(n*sp),1)];
    for (int col=0; col<std::max((int)(n*sp),1); col++){
      bool is_rep;
      do {
        cols[col] = rand()%n;
        is_rep = 0;
        for (int c=0; c<col; c++){
          if (cols[c] == cols[col]) is_rep = 1;
        }
      } while (is_rep);
      inds[i] = cols[col]*n+row;
      vals[i] = (rand()%std::min(n*n,20))+1;
      i++;
    }
  }
  A.write(i,inds,vals);
  
  A["ii"] = 0;
  
  //keep only values smaller than 20 (about 20% sparsity)
  //A.sparsify([=](int a){ return a<sp*100; });


  Vector<double> v1(n,dw);
  Vector<double> v2(n,dw);

  double st_time = MPI_Wtime();


 // v1.print();
 // v2.print();

  if (test || n<= 20){
    btwn_cnt_naive(A, v1);
    //compute centrality scores by Bellman Ford with block size bsize
    btwn_cnt_fast(A, bsize, v2);
    //v1.print();
    //v2.print();
    v1["i"] -= v2["i"];
    int pass = v1.norm2() <= 1.E-6;

    if (dw.rank == 0){
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
    btwn_cnt_fast(A, bsize, v2, 1);
    if (dw.rank == 0)
      printf("Starting benchmarking\n");
    Timer_epoch tbtwn("Betweenness centrality");
    tbtwn.begin();
    btwn_cnt_fast(A, bsize, v2, nbatches);
    tbtwn.end();
    if (dw.rank == 0){
      if (nbatches == 0) printf("Completed all batches in time %lf sec, projected total %lf sec.\n", MPI_Wtime()-st_time, MPI_Wtime()-st_time);
      else printf("Completed %d batches in time %lf sec, projected total %lf sec.\n", nbatches, MPI_Wtime()-st_time, (n/(bsize*nbatches))*(MPI_Wtime()-st_time));
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
  int rank, np, n, pass, bsize, nbatches, test, scale, ef;
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

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Computing betweenness centrality for graph with %d nodes, with %lf percent sparsity, and batch size %d\n",n,sp,bsize);
    }
    if (scale > 0 && ef > 0){
      printf("R-MAT MODE ON scale=%d ef=%d seed=%u\n", scale, ef, myseed);
    }
    //pass = btwn_cnt(n, dw, sp, bsize, nbatches, test);
    pass = btwn_cnt_rmat(scale, dw, ef, bsize, nbatches, test, myseed);
    //assert(pass);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
