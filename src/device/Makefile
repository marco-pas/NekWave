# ==============================================================================
# NekWave: Standalone Device Kernel Subsystem Makefile
# Enables side compilation of device kernels independently from CMake.
# ==============================================================================

NVCC ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda-11.7/bin/nvcc)
CUDA_ARCH ?= sm_80

NVCCFLAGS ?= -O3 -arch=$(CUDA_ARCH) -std=c++14 -Xcompiler -Wall,-Wextra,-fPIC --compiler-options -fPIC
AR ?= ar
ARFLAGS ?= rcs

TARGET_LIB = libnekwave_device.a

OBJS = dg_kernels.o

.PHONY: all clean

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	@echo "[AR] Creating static archive $@"
	$(AR) $(ARFLAGS) $@ $^

dg_kernels.o: dg_kernels.cu dg_kernels.hpp
	@echo "[NVCC] Compiling $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	@echo "[CLEAN] Removing device artifacts..."
	rm -f $(OBJS) $(TARGET_LIB)
