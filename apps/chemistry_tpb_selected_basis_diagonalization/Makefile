include Configuration
SBD_INCLUDE_DIR=$(SBD_PATH)/include
LIBFLAGS= $(SYSLIB) -lcudart
MAKEFILES= Makefile Configuration

# CUDA source and object files
CUDA_SOURCES = ../../include/sbd/chemistry/tpb_k/hamiltonian_kernel.cu
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
PROF_CUDA_O = $(CUDA_OBJECTS:.o=.prof.o)

# Regular source files
SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)
PROF_O = $(OBJECTS:.o=.prof.o)

.SUFFIXES: .h .cc .o .cu

# Compiler rules for supported file extensions
%.o: %.cc
	$(CCCOM) -c $(CCFLAGS) -I$(SBD_INCLUDE_DIR) -o $@ $<
%.o: %.cu
	$(CUDACOM) -arch=$(CUDA_ARCH) $(CUDAFLAGS) -o $@ $<
%.prof.o: %.cc
	$(PROF_CCCOM) -c $(CCFLAGS) -I$(SBD_INCLUDE_DIR) -o $@ $<
%.prof.o: %.cu
	$(PROF_CUDACOM) -arch=$(CUDA_ARCH) $(CUDAFLAGS) -o $@ $<

###############################################################
diag: main.o $(CUDA_OBJECTS) $(OBJECTS)
	@echo "Linking executable with CUDA support..."
	$(CCCOM) $(CCFLAGS) -o diag $(OBJECTS) $(CUDA_OBJECTS) $(LIBFLAGS)
	@echo "Build complete!"

profile: main.prof.o $(PROF_CUDA_O) $(PROF_O)
	@echo "Building profiling version with CUDA support..."
	$(PROF_CCCOM) $(CCFLAGS) -o diag $(PROF_O) $(PROF_CUDA_O) $(LIBFLAGS)

# Clean target
clean:
	rm -f $(OBJECTS) $(PROF_O) $(CUDA_OBJECTS) $(PROF_CUDA_O) diag
	@echo "Cleaned build artifacts"

# Clean only CUDA objects
clean-cuda:
	rm -f $(CUDA_OBJECTS)
	@echo "Cleaned CUDA objects"

# Rebuild CUDA files only
rebuild-cuda: clean-cuda $(CUDA_OBJECTS)
	@echo "CUDA files rebuilt"

# Check CUDA availability
check-cuda:
	@echo "Checking CUDA environment..."
	@which nvcc || (echo "ERROR: nvcc not found in PATH" && exit 1)
	@nvcc --version | head -n 1
	@echo "CUDA architecture: $(CUDA_ARCH)"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv | head -2

###############################################################
run-diag: diag
	@echo "Compiling and running optimized workflow"
	mpirun -np 16 -x OMP_NUM_THREADS=1 ./diag --fcidump fcidump_Fe4S4.txt --adetfile AlphaDets.txt \
       --method 0 --block 10 --iteration 2 --tolerance 1.0e-4 \
       --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2 \
       --init 0 --shuffle 0 --carryover_type 0 --rdm 0

run-profile: profile
	@echo "Set SCOREP_ENABLE_TRACING=true for tracing"
	mpirun -np 16 -x OMP_NUM_THREADS=1 ./diag --fcidump fcidump_Fe4S4.txt --adetfile AlphaDets.txt \
       --method 0 --block 10 --iteration 2 --tolerance 1.0e-4 \
       --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2 \
       --init 0 --shuffle 0 --carryover_type 0 --rdm 0

help:
	@echo "Available targets:"
	@echo "  make diag         - Build with CUDA support (default)"
	@echo "  make profile      - Build with Score-P instrumentation + CUDA"
	@echo "  make run-diag     - Build and run optimized workflow"
	@echo "  make run-profile  - Build and run profiling test"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make clean-cuda   - Remove only CUDA object files"
	@echo "  make rebuild-cuda - Rebuild CUDA files only"
	@echo "  make check-cuda   - Verify CUDA environment"
	@echo ""
	@echo "CUDA Configuration:"
	@echo "  Compiler: $(NVCC)"
	@echo "  Architecture: $(CUDA_ARCH)"
	@echo "  CUDA Sources: $(CUDA_SOURCES)"
	@echo ""
	@echo "For detailed profiling:"
	@echo "  export SCOREP_ENABLE_TRACING=true"
	@echo "  mpirun -np 16 ./diag [your arguments]"
