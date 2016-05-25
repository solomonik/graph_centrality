CTFDIR    = /home/flavio/CTF/ctf
MPI_DIR   = /usr/mpi/gcc/openmpi-1.8.4
CXX       =  /opt/gcc49/bin/g++ 
OPTS      = -O2 -g  
CXXFLAGS  = -std=c++0x -fopenmp $(OPTS) -Wall -fpermissive -fno-exceptions
INCLUDES  = -I$(CTFDIR)/include -I$(MPI_DIR)/include/
LIBS      = -L$(CTFDIR)/lib -lctf -L/usr/local/cuda/lib64/ -lcuda -lcudart -lcublas -lblas -L$(MPI_DIR)/lib64 -lmpi -lmpi_cxx generator/libgraph_generator_mpi.a 
DEFS        = 
CUDA_ARCH  = sm_37
NVCC      = nvcc -ccbin $(CXX) -x cu -m64 -std=c++11 $(OPTS)
NVCCFLAGS =  -arch=$(CUDA_ARCH)


