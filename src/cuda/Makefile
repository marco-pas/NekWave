# ==============================================================================
# NekWave: Standalone CUDA Matrix Multiplication Subsystem Makefile
# Enables side compilation of the CUDA mxm kernel independently from CMake.
# ==============================================================================

NVCC ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda-11.7/bin/nvcc)
CUDA_ARCH ?= sm_80

NVCCFLAGS ?= -O3 -arch=$(CUDA_ARCH) -std=c++11 -Xcompiler -Wall,-Wextra,-fPIC --compiler-options -fPIC
AR ?= ar
ARFLAGS ?= rcs

TARGET_LIB = libnekwave_cuda.a
TARGET_TEST = test_mxm_cuda

SRCS = mxm_cuda.cu
OBJS = mxm_cuda.o

.PHONY: all clean run test

all: $(TARGET_LIB) $(TARGET_TEST)

$(TARGET_LIB): $(OBJS)
	@echo "[AR] Creating static archive $@"
	$(AR) $(ARFLAGS) $@ $^

mxm_cuda.o: mxm_cuda.cu mxm_cuda.hpp
	@echo "[NVCC] Compiling $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET_TEST): test_mxm_cuda.cu $(TARGET_LIB)
	@echo "[NVCC] Building verification test $@"
	$(NVCC) $(NVCCFLAGS) test_mxm_cuda.cu -L. -lnekwave_cuda -o $@

run: $(TARGET_TEST)
	./$(TARGET_TEST)

test: run

clean:
	@echo "[CLEAN] Removing CUDA artifacts..."
	rm -f $(OBJS) $(TARGET_LIB) $(TARGET_TEST)

