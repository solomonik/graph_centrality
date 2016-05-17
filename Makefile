include config.mk

all: test_btwn_central

btwn_central_kernels.o: btwn_central.h btwn_central_kernels.cxx
	$(NVCC) $(NVCCFLAGS) -c btwn_central_kernels.cxx  $(DEFS) $(INCLUDES) generator/libgraph_generator_mpi.a

btwn_central.o: btwn_central.cxx btwn_central.h  generator/libgraph_generator_mpi.a 
	$(CXX) $(CXXFLAGS) -c btwn_central.cxx $(INCLUDES)

test_btwn_central: btwn_central.o btwn_central_kernels.o test_btwn_central.cxx
	$(CXX) $(CXXFLAGS) -o test_btwn_central test_btwn_central.cxx btwn_central.o btwn_central_kernels.o $(INCLUDES) $(LIBS) generator/libgraph_generator_mpi.a

clean:
	rm -f btwn_central_kernels.o btwn_central.o test_btwn_central
