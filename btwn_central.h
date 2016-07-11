#ifndef __BTWN_CENTRAL_H__
#define __BTWN_CENTRAL_H__

#include <ctf.hpp>
#include <float.h>


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

typedef float mlt;
typedef float wht;
#define MAX_WHT (FLT_MAX/4.)
//typedef int mlt;
//typedef int wht;
//#define MAX_WHT (INT_MAX/2)

//structure for regular path that keeps track of the multiplicity of paths
class mpath {
  public:
  wht w; // weighted distance
  mlt m; // multiplictiy
  DEVICE HOST
  mpath(wht w_, mlt m_){ w=w_; m=m_; }
  DEVICE HOST
  mpath(mpath const & p){ w=p.w; m=p.m; }
  DEVICE HOST
  mpath(){};
};

//path with a centrality score and a counter (m)
class cmpath {
  public:
  wht w;
  int m;
  float c; // centrality score
  DEVICE HOST
  cmpath(wht w_, int m_, float c_){ w=w_; m=m_; c=c_;}
  DEVICE HOST
  cmpath(cmpath const & p){ w=p.w; m=p.m; c=p.c; }
  cmpath(){};
};

//path with a centrality score
class cpath {
  public:
  wht w;
  float c; // centrality score
  DEVICE HOST
  cpath(wht w_, float c_){ w=w_; c=c_;}
  DEVICE HOST
  cpath(cpath const & p){ w=p.w; c=p.c; }
  cpath(){};
};

// min Monoid for cpath structure
CTF::Monoid<cpath> get_cpath_monoid();

// min Monoid for cmpath structure
CTF::Monoid<cmpath> get_cmpath_monoid();

//(min, +) tropical semiring for mpath structure
CTF::Semiring<mpath> get_mpath_semiring();

CTF::Bivar_Function<wht,mpath,mpath> * get_Bellman_kernel();

CTF::Bivar_Function<wht,cpath,cmpath> * get_Brandes_kernel();

/**
  * \brief fast algorithm for betweenness centrality using Bellman Ford
  * \param[in] A matrix on the tropical semiring containing edge weights
  * \param[in] b number of source vertices for which to compute Bellman Ford at a time
  * \param[out] v vector that will contain centrality scores for each vertex
  * \param[in] nbatches, number of batches (sets of nodes of size b) to compute on (0 means all)
  * \param[in] sp_B whether to store second operand as sparse
  * \param[in] sp_C whether to store output as sparse
  */
void btwn_cnt_fast(CTF::Matrix<wht> A, int64_t b, CTF::Vector<float> & v, int nbatches, bool sp_B, bool sp_C);

/**
  * \brief naive algorithm for betweenness centrality using 3D tensor of counts
  * \param[in] A matrix on the tropical semiring containing edge weights
  * \param[out] v vector that will contain centrality scores for each vertex
  */
void btwn_cnt_naive(CTF::Matrix<wht> & A, CTF::Vector<float> & v);
uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges);
uint64_t norm_graph(uint64_t *ed, uint64_t ned);
#endif
