.ONESHELL:
SHELL=/bin/bash


NVCC        = /usr/local/cuda/bin/nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include # what is this for?
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 # what is this for? 

KERNELS_DIR ?= kernels
KERNEL_NAME ?= vectorAdd
CU_FILE ?= $(KERNELS_DIR)/$(KERNEL_NAME).cu
OUTPUT_PATH = build/${KERNEL_NAME}





.PHONY: nvcc-compile
nvcc-compile:
	@echo "Compiling CUDA code"
	mkdir -p build
	$(NVCC) $(NVCC_FLAGS) $(CU_FILE) -o $(OUTPUT_PATH)
	

.PHONY: run-kernel
run-kernel:
	@echo "Running kernel"
	$(OUTPUT_PATH) 

.PHONY: build-run
build-run: nvcc-compile run-kernel

.PHONY: clean
clean:
	@echo "Cleaning up"
	rm -rf build

