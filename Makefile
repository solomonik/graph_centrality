include config.mk

all: test_btwn_central

btwn_central_kernels.o: btwn_central.h btwn_central_kernels.cxx
	$(NVCC) $(NVCCFLAGS) -c btwn_central_kernels.cxx $(INCLUDES)

btwn_central.o: btwn_central.cxx btwn_central.h 
	$(CXX) $(CXXFLAGS) -c btwn_central.cxx $(INCLUDES)

test_btwn_central: btwn_central.o btwn_central_kernels.o test_btwn_central.cxx
	$(CXX) $(CXXFLAGS) -o test_btwn_central test_btwn_central.cxx btwn_central.o btwn_central_kernels.o $(INCLUDES) $(LIBS)

clean:
	rm -f btwn_central_kernels.o btwn_central.o test_btwn_central
