CTFDIR    = /home/edgar/work/ctf/debug
MPI_DIR   =
CXX       = mpicxx -cxx=clang++
OPTS      = -O0 -g
CXXFLAGS  = -std=c++0x -fopenmp $(OPTS) -Wall -fpermissive -fno-exceptions -lblas  -DPROFILE -DPMPI -DMPIIO
INCLUDES  = -I$(CTFDIR)/include
LIBS      = -L$(CTFDIR)/lib -lctf generator/libgraph_generator_mpi.a
DEFS      =
CUDA_ARCH = sm_37
NVCC      = $(CXX)
NVCCFLAGS = $(CXXFLAGS)


