#ifndef __BTWN_CENTRAL_H__
#define __BTWN_CENTRAL_H__

#include <ctf.hpp>

#ifdef __CUDACC__
#define DEVICE __device__
#define HOST __host__
#else
#define DEVICE
#define HOST
#endif

#define SEED 23
template <typename T>
struct pair {
  uint64_t k;
  double v;
} ;
//structure for regular path that keeps track of the multiplicity of paths
class mpath {
  public:
  int w; // weighted distance
  int m; // multiplictiy
  DEVICE HOST
  mpath(int w_, int m_){ w=w_; m=m_; }
  DEVICE HOST
  mpath(mpath const & p){ w=p.w; m=p.m; }
  DEVICE HOST
  mpath(){};
};

//path with a centrality score
class cpath {
  public:
  double c; // centrality score
  float m;
  int w;
  DEVICE HOST
  cpath(int w_, float m_, double c_){ w=w_; m=m_; c=c_;}
  DEVICE HOST
  cpath(cpath const & p){ w=p.w; m=p.m; c=p.c; }
  cpath(){};
};


// min Monoid for cpath structure
CTF::Monoid<cpath> get_cpath_monoid();

//(min, +) tropical semiring for mpath structure
CTF::Semiring<mpath> get_mpath_semiring();

CTF::Bivar_Function<int,mpath,mpath> * get_Bellman_kernel();

CTF::Bivar_Function<int,cpath,cpath> * get_Brandes_kernel();

/**
  * \brief fast algorithm for betweenness centrality using Bellman Ford
  * \param[in] A matrix on the tropical semiring containing edge weights
  * \param[in] b number of source vertices for which to compute Bellman Ford at a time
  * \param[out] v vector that will contain centrality scores for each vertex
  * \param[in] nbatches, number of batches (sets of nodes of size b) to compute on (0 means all)
  * \param[in] sp_B whether to store second operand as sparse
  * \param[in] sp_C whether to store output as sparse
  */
void btwn_cnt_fast(CTF::Matrix<int> A, int b, CTF::Vector<double> & v, int nbatches, bool sp_B, bool sp_C);

/**
  * \brief naive algorithm for betweenness centrality using 3D tensor of counts
  * \param[in] A matrix on the tropical semiring containing edge weights
  * \param[out] v vector that will contain centrality scores for each vertex
  */
void btwn_cnt_naive(CTF::Matrix<int> & A, CTF::Vector<double> & v);
uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges);
uint64_t norm_graph(uint64_t *ed, uint64_t ned);
#endif
