#include "btwn_central.h"
#include <ctf.hpp>
using namespace CTF;


DEVICE HOST 
void mfunc(mpath a, mpath & b){ 
  if (a.w<b.w){ b=a; }
  else if (b.w==a.w){ b.m+=a.m; }
}

DEVICE HOST 
mpath addw(wht w, mpath p){ return mpath(p.w+w,p.m); }

Bivar_Function<wht,mpath,mpath> * get_Bellman_kernel(){
  return new Bivar_Kernel<wht,mpath,mpath,addw,mfunc>();
}

DEVICE HOST 
void cfunc(cmpath a, cmpath & b){
  if (a.w>b.w){ b.c=a.c; b.w=a.w; b.m=a.m; }
  else if (b.w == a.w){ b.c+=a.c; b.m+=a.m; }
}

DEVICE HOST
cmpath subw(wht w, cpath p){
  return cmpath(p.w-w, -1, p.c);
}

Bivar_Function<wht,cpath,cmpath> * get_Brandes_kernel(){
  return new Bivar_Kernel<wht,cpath,cmpath,subw,cfunc>();
}


void mpath_red(mpath const * a,
               mpath * b,
               int n){
  #pragma omp parallel for
  for (int i=0; i<n; i++){
    if (a[i].w <  b[i].w){
      b[i].w  = a[i].w;
      b[i].m  = a[i].m;
    } else if (a[i].w == b[i].w) b[i].m += a[i].m;
  }
}

//(min, +) tropical semiring for mpath structure
Semiring<mpath> get_mpath_semiring(){
  //struct for mpath with w=mpath weight, h=#hops
  MPI_Op ompath;

  MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        mpath_red((mpath*)a, (mpath*)b, *n);
      },
      1, &ompath);

  //tropical semiring with hops carried by winner of min
  Semiring<mpath> p(mpath(MAX_WHT,0), 
                   [](mpath a, mpath b){ 
                     if (a.w<b.w){ return a; }
                     else if (b.w<a.w){ return b; }
                     else { return mpath(a.w, a.m+b.m); }
                   },
                   ompath,
                   mpath(0,1),
                   [](mpath a, mpath b){ return mpath(a.w+b.w, a.m*b.m); });

  return p;
}


void cmpath_red(cmpath const * a,
                cmpath * b,
                int n){
  #pragma omp parallel for
  for (int i=0; i<n; i++){
    if (a[i].w > b[i].w){
      b[i].w  = a[i].w;
      b[i].m  = a[i].m;
      b[i].c  = a[i].c;
    } else if (a[i].w == b[i].w){
      b[i].c += a[i].c;
    }
  }
}


// min Monoid for cmpath structure
Monoid<cmpath> get_cmpath_monoid(){
  //struct for cpath with w=cpath weight, h=#hops
  MPI_Op ocmpath;

  MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        cmpath_red((cmpath*)a, (cmpath*)b, *n);
      },
      1, &ocmpath);

  Monoid<cmpath> cmp(cmpath(-MAX_WHT,0,0.), 
                  [](cmpath a, cmpath b){
                    if (a.w>b.w){ return a; }
                    else if (b.w>a.w){ return b; }
                    else { return cmpath(a.w, a.m+b.m, a.c+b.c); }
                  }, ocmpath);

  return cmp;
}

void cpath_red(cpath const * a,
               cpath * b,
               int n){
  #pragma omp parallel for
  for (int i=0; i<n; i++){
    if (a[i].w > b[i].w){
      b[i].w  = a[i].w;
      b[i].c  = a[i].c;
    } else if (a[i].w == b[i].w){
      b[i].c += a[i].c;
    }
  }
}

// min Monoid for cpath structure
Monoid<cpath> get_cpath_monoid(){
  //struct for cpath with w=cpath weight, h=#hops
  MPI_Op ocpath;

  MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        cpath_red((cpath*)a, (cpath*)b, *n);
      },
      1, &ocpath);

  Monoid<cpath> cp(cpath(-MAX_WHT,0.), 
                  [](cpath a, cpath b){
                    if (a.w>b.w){ return a; }
                    else if (b.w>a.w){ return b; }
                    else { return cpath(a.w, a.c+b.c); }
                  }, ocpath);

  return cp;
}



