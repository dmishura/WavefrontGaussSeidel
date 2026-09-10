CXX ?= c++
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17 -Wall -Wextra -Wpedantic -fopenmp -Icompat -Isrc
HISTORY_SWEEPS ?= 20
AVX512_FLAGS ?= -mavx512f -mavx512vl

OPENFOAM_DIR ?= ../OpenFOAM-14
GS_DIR := $(OPENFOAM_DIR)/src/OpenFOAM/matrices/lduMatrix/smoothers/GaussSeidel

MESH_ARCHIVE := meshes/MTB_example_polyMesh.tgz
MESH_DIR := build/mesh-data/MTB_example_polyMesh
MESH_STAMP := $(MESH_DIR)/.extracted

SMOOTHER_TARGET := build/Test-GaussSeidel
READER_TARGET := build/Test-PolyMeshReader
COMMON_OBJECTS := build/src/PolyMeshReader.o
SMOOTHER_OBJECTS := build/src/GaussSeidelSmoother.o \
    build/src/WavefrontGaussSeidel.o \
    build/src/WavefrontGaussSeidelAVX512.o \
    build/tests/Test-GaussSeidel.o $(COMMON_OBJECTS)
READER_OBJECTS := build/tests/Test-PolyMeshReader.o $(COMMON_OBJECTS)
OBJECTS := $(sort $(SMOOTHER_OBJECTS) $(READER_OBJECTS))

.PHONY: all test test-reader test-smoother run clean sync

all: $(READER_TARGET) $(SMOOTHER_TARGET)

test: test-reader test-smoother

test-reader: $(READER_TARGET) $(MESH_STAMP)
	./$(READER_TARGET) $(MESH_DIR)

test-smoother run: $(SMOOTHER_TARGET) $(MESH_STAMP)
	./$(SMOOTHER_TARGET) $(MESH_DIR) --history-sweeps $(HISTORY_SWEEPS)

$(SMOOTHER_TARGET): $(SMOOTHER_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(READER_TARGET): $(READER_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(MESH_STAMP): $(MESH_ARCHIVE)
	mkdir -p build/mesh-data
	tar -xzf $< -C build/mesh-data
	touch $@

build/%.o: %.C
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

build/src/WavefrontGaussSeidelAVX512.o: src/WavefrontGaussSeidelAVX512.C
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(AVX512_FLAGS) -MMD -MP -c $< -o $@

# Replace both files with pristine OpenFOAM sources.  OPENFOAM_DIR may point
# at another checkout/version; the compatibility layer is intentionally local.
sync:
	cp $(GS_DIR)/GaussSeidelSmoother.C src/GaussSeidelSmoother.C
	cp $(GS_DIR)/GaussSeidelSmoother.H src/GaussSeidelSmoother.H

clean:
	rm -rf build

-include $(OBJECTS:.o=.d)
