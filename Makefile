include config.mk

all: test_btwn_central
btwn_central_kernels.o: btwn_central.h btwn_central_kernels.cxx $(CTFDIR)
	$(NVCC) $(NVCCFLAGS) -c btwn_central_kernels.cxx  $(DEFS) $(INCLUDES) 

btwn_central.o: btwn_central.cxx btwn_central.h   $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c btwn_central.cxx $(INCLUDES)

.PHONY: $(CTFDIR)
$(CTFDIR):
	$(MAKE) ctflib -C $(CTFDIR) 

test_btwn_central: btwn_central.o btwn_central_kernels.o test_btwn_central.cxx  generator/libgraph_generator_mpi.a $(CTFDIR) $(CTFDIR)/lib/libctf.a
	$(CXX) $(CXXFLAGS) -o test_btwn_central test_btwn_central.cxx btwn_central.o btwn_central_kernels.o $(INCLUDES) $(LIBS) generator/libgraph_generator_mpi.a

clean:
	rm -f btwn_central_kernels.o btwn_central.o test_btwn_central
