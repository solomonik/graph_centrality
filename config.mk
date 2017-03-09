CTFDIR    = /home.local/flavio/CTF/ctf
MPI_DIR   = /usr/mpi/gcc/openmpi-1.8.4
CXX       = /opt/gcc49/bin/gcc
OPTS      =  
CXXFLAGS  = -std=c++0x -fopenmp -O0  -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1  -DPROFILE -DPMPI
INCLUDES  = -I$(CTFDIR)/include -I$(MPI_DIR)/include/
LIBS      = -L$(CTFDIR)/lib -lctf  -L./generator/ -lgraph_generator_mpi -L$(MPI_DIR)/lib64 -lmpi -lmpi_cxx $(CTFDIR)/lib/libctf.a -lm -std=gnu++11 -lblas
DEFS        = 
CUDA_ARCH  = sm_35
NVCC      = $(CXX)
NVCCFLAGS = $(CXXFLAGS)


