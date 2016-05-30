CTFDIR    = /users/fvella/ctf/cpu-only
MPI_DIR   = $(MPICH_DIR)
CXX       = CC
OPTS      =  
CXXFLAGS  = -std=c++0x -openmp -O3 -ipo -Wall -mkl=parallel -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1  -DPROFILE -DPMPI
INCLUDES  = -I$(CTFDIR)/include -I$(MPI_DIR)/include/
LIBS      = -L$(CTFDIR)/lib -lctf  -L./generator/ -lgraph_generator_mpi
DEFS        = 
CUDA_ARCH  = sm_35
NVCC      = $(CXX)
NVCCFLAGS = $(CXXFLAGS)


