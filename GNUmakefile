# ==============================================================================
# DEPRECATED: This GNUmakefile is retained for backward compatibility.
# The project has migrated to CMake. Please use the CMake build system:
#
#   mkdir cmake_build && cd cmake_build
#   cmake ..
#   make -j$(nproc)
#   ctest
#
# This Makefile will be removed in a future release.
# ==============================================================================
#
# GNU MakeFile for OpenImpala Diffusion Application and Tests
# Corrected version 2: Uses Static Pattern Rules for compilation.

# ============================================================
# Environment Setup (Defaults set for the Singularity container)
# ============================================================
# Default paths matching the Singularity container build.
# Can still be overridden by setting environment variables before running make.
AMREX_HOME    ?= /opt/amrex/25.03
HYPRE_HOME    ?= /opt/hypre/v2.32.0
HDF5_HOME     ?= /opt/hdf5/1.12.3
H5CPP_HOME    ?= $(HDF5_HOME)
TIFF_HOME     ?= /opt/libtiff/4.6.0

# ============================================================
# Compilers and Flags
# ============================================================
CXX           := mpic++
F90           := mpif90

CXXFLAGS      := -Wextra -O3 -fopenmp -DOMPI_SKIP_MPICXX -g -std=c++17 -MMD -MP
F90FLAGS      := -g -O3 -fbounds-check

# Include project source dir and dependency includes
INCLUDE       := -Isrc \
		 -Isrc/io \
		 -Isrc/props \
                 -I$(AMREX_HOME)/include \
                 -I$(HYPRE_HOME)/include \
                 -I$(HDF5_HOME)/include \
                 -I$(TIFF_HOME)/include

# Linker Flags
LDFLAGS       := -L$(AMREX_HOME)/lib -lamrex \
                 -L$(HYPRE_HOME)/lib -lHYPRE \
                 -L$(HDF5_HOME)/lib -lhdf5 -lhdf5_cpp \
                 -L$(TIFF_HOME)/lib -ltiff \
                 -lm \
                 -lgfortran

# ============================================================
# Project Structure
# ============================================================
INC_DIR       := build/include# For Fortran modules
APP_DIR       := build/apps# For main executable
TST_DIR       := build/tests# For test executables
IO_DIR        := build/io# For IO object files   <<< REMOVE SPACES BEFORE #
PROPS_DIR     := build/props# For Props object files <<< REMOVE SPACES BEFORE #

MODULES       := io props
SRC_DIRS      := $(addprefix src/,$(MODULES))# src/io src/props
BUILD_DIRS    := $(addprefix build/,$(MODULES))# build/io build/props

# ============================================================
# Source and Object Files (Revised for Linking - v4 style)
# ============================================================
# --- Find source files ---
SOURCES_IO_ALL     := $(wildcard src/io/*.cpp)
SOURCES_PRP_ALL    := $(wildcard src/props/*.cpp)
SOURCES_F90_ALL    := $(wildcard src/props/*.F90) # Assuming Fortran only in props

# --- Test drivers (now in tests/ directory) ---
SOURCES_TESTS      := $(wildcard tests/t*.cpp)

# --- Identify App Driver ---
SOURCES_APP_DRIVER := src/props/Diffusion.cpp

# --- Identify Library Sources (excluding app driver and auxiliary files) ---
SOURCES_IO_LIB     := $(SOURCES_IO_ALL)
SOURCES_PRP_LIB    := $(filter-out $(SOURCES_APP_DRIVER) src/props/hypre_test.cpp, $(SOURCES_PRP_ALL))
SOURCES_F90_LIB    := $(SOURCES_F90_ALL) # Assuming all F90 are library code

# --- Define Object Files based on Categories ---
OBJECTS_APP_DRIVER := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_APP_DRIVER))
OBJECTS_TESTS      := $(patsubst tests/%.cpp,$(TST_DIR)/%.o,$(SOURCES_TESTS))
TEST_EXECS         := $(patsubst tests/%.cpp,$(TST_DIR)/%,$(SOURCES_TESTS))

OBJECTS_IO_LIB     := $(patsubst src/io/%.cpp,$(IO_DIR)/%.o,$(SOURCES_IO_LIB))
OBJECTS_PRP_LIB    := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_PRP_LIB))
OBJECTS_F90_LIB    := $(patsubst src/props/%.F90,$(PROPS_DIR)/%.o,$(SOURCES_F90_LIB))

# --- Consolidate Library Objects ---
OBJECTS_LIB_ALL    := $(OBJECTS_IO_LIB) $(OBJECTS_PRP_LIB) $(OBJECTS_F90_LIB)

# Let Make search for source files in relevant directories
VPATH := $(subst $(space),:,$(SRC_DIRS)):src

# ============================================================
# Main Targets
# ============================================================
.PHONY: all main tests test clean debug release tidy

all: main tests

# Main application executable (Diffusion)
main: $(APP_DIR)/Diffusion

# All test executables (sources now in tests/ directory)
tests: $(TEST_EXECS)

# Target to run the tests after they are built, linking specific input files
test: tests
	@echo ""
	@echo "--- Running Tests ---"
	@passed_all=true; \
	list_of_tests='$(TEST_EXECS)'; \
	rm -f ./inputs; \
	if [ -z "$$list_of_tests" ]; then \
	    echo "No test executables found to run."; \
	else \
	    for tst in $$list_of_tests; do \
	        echo "Running test $$tst..."; \
	        test_name=$$(basename $$tst); \
	        input_file="tests/inputs/$${test_name}.inputs"; \
	        if [ -f "$$input_file" ]; then \
	            echo "  Using input file: $$input_file"; \
	            ln -sf "$$input_file" ./inputs; \
	        else \
	            echo "  Warning: No specific input file found at $$input_file. Running without './inputs'."; \
	            rm -f ./inputs; \
	            # Optional: Fail if input file is missing (uncomment below) \
	            # echo "  ERROR: Required input file $$input_file not found for test $$test_name!"; \
	            # passed_all=false; \
	            # continue; \
	        fi; \
	        input_file_arg=""; \
	        if [ -f "./inputs" ]; then input_file_arg="./inputs"; fi; \
	        echo "  Executing: mpirun -np 1 --allow-run-as-root $$tst $$input_file_arg"; \
	        if mpirun -np 1 --allow-run-as-root $$tst $$input_file_arg amrex.verbose=0; then \
	            echo "  PASS: $$tst"; \
	        else \
	            echo "  FAIL: $$tst"; \
	            passed_all=false; \
	        fi; \
	        rm -f ./inputs; \
	    done; \
	fi; \
	echo "--- Test Summary ---"; \
	if $$passed_all; then \
	    echo "All tests passed."; \
	    exit 0; \
	else \
	    echo "ERROR: One or more tests failed."; \
	    exit 1; \
	fi

# ============================================================
# Compilation Rules (Using Structure Similar to Working v2)
# ============================================================

# Static Pattern Rule for IO Lib C++ objects
$(OBJECTS_IO_LIB): $(IO_DIR)/%.o: src/io/%.cpp
	@echo "Compiling (IO Lib) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for Test C++ objects (sources in tests/)
$(OBJECTS_TESTS): $(TST_DIR)/%.o: tests/%.cpp
	@echo "Compiling (Test) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for Props Lib C++ objects
$(OBJECTS_PRP_LIB): $(PROPS_DIR)/%.o: src/props/%.cpp
	@echo "Compiling (Props Lib) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for Props App Driver C++ object
$(OBJECTS_APP_DRIVER): $(PROPS_DIR)/%.o: src/props/%.cpp
	@echo "Compiling (App Driver) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for Props Lib F90 objects
$(OBJECTS_F90_LIB): $(PROPS_DIR)/%.o: src/props/%.F90
	@echo "Compiling (Props Fortran) $< ..."
	@mkdir -p $(@D) $(INC_DIR) # Ensure both obj and mod dirs exist
	$(F90) $(F90FLAGS) $(INCLUDE) -J$(INC_DIR) -c $< -o $@

# ============================================================
# Linking Executables (Revised Rules - v4 style)
# ============================================================

# Main application
$(APP_DIR)/Diffusion: $(OBJECTS_APP_DRIVER) $(OBJECTS_LIB_ALL)
	@echo "Linking Main App $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) # Use $^ for all prerequisites

# Test executables (sources in tests/)
$(TEST_EXECS): $(TST_DIR)/%: $(TST_DIR)/%.o $(OBJECTS_LIB_ALL)
	@echo "Linking Test $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) # Use $^ for all prerequisites

# ============================================================
# Supporting Targets
# ============================================================

# Build environments (simple examples)
debug: CXXFLAGS += -DDEBUG # Example C++ debug flag
debug: F90FLAGS += -fbounds-check # Example Fortran debug flag
debug: all

release: # Using defaults: -O3 defined initially
release: all

# Static analysis with clang-tidy (requires clang-tidy to be installed)
CLANG_TIDY    ?= clang-tidy
TIDY_FLAGS    := -- -std=c++17 -fopenmp -DOMPI_SKIP_MPICXX $(INCLUDE)
SOURCES_CPP_ALL := $(SOURCES_IO_ALL) $(SOURCES_PRP_ALL) $(SOURCES_TESTS)

tidy:
	@echo "--- Running clang-tidy ---"
	@for src in $(SOURCES_CPP_ALL); do \
	    echo "  Analyzing $$src..."; \
	    $(CLANG_TIDY) $$src $(TIDY_FLAGS) || exit 1; \
	done
	@echo "--- clang-tidy complete ---"

# Clean target (removes objects, module files, executables, dependency files)
clean:
	@echo "Cleaning build artifacts..."
	-@rm -rf build # Remove entire build directory is simplest
	# Clean dependency files potentially generated in source dirs if CXXFLAGS didn't handle output dir well
	-@find src -name '*.d' -delete
	-@find . -name '*.mod' -delete # Remove Fortran module files from source tree if accidentally created there

# ============================================================
# Include Auto-Generated Dependencies
# ============================================================
# Include the .d files generated by CXXFLAGS -MMD -MP for C++ files
# Includes both app and test objects dependencies now
-include $(OBJECTS_IO_LIB:.o=.d) $(OBJECTS_PRP_LIB:.o=.d) $(OBJECTS_APP_DRIVER:.o=.d) $(OBJECTS_TESTS:.o=.d)

# ============================================================
# Debugging Help (Uncomment to use)
# ============================================================
# print-%:
# 	@echo '$* = $($*)'
# Example: make print-OBJECTS_APP
