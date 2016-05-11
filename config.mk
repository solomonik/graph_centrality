CTFDIR    = /home/edgar/work/ctf-cuda/cuda-g
CXX       = mpicxx
OPTS      = -O0 -g
CXXFLAGS  = -std=c++0x -fopenmp $(OPTS)
INCLUDES  = -I$(CTFDIR)/include
LIBS      = -L$(CTFDIR)/lib -lctf -L/usr/local/cuda/lib64/ -lcuda -lcudart -lcublas -lblas
NVCC      = nvcc -ccbin g++ -x cu -m64
NVCCFLAGS = -std=c++11 $(OPTS)


