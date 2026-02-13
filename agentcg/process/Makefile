# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
# Makefile for process monitor - adapted from agentsight/bpf/Makefile
# Uses third_party/bpftool's libbpf and generates vmlinux.h from running kernel

# Program name
PROG_NAME = process

# Compiler and tools
CLANG ?= clang
CC ?= gcc

# Project paths
PROJECT_ROOT := $(shell git rev-parse --show-toplevel)
BPFTOOL_ROOT := $(PROJECT_ROOT)/third_party/bpftool

# Local bpftool and libbpf paths
BPFTOOL := $(BPFTOOL_ROOT)/src/bpftool
LIBBPF_DIR := $(BPFTOOL_ROOT)/src/libbpf
LIBBPF_INCLUDE := $(LIBBPF_DIR)/include
LIBBPF_LIB := $(LIBBPF_DIR)/libbpf.a

# Architecture detection
ARCH := $(shell uname -m | sed 's/x86_64/x86/' \
			 | sed 's/arm.*/arm/' \
			 | sed 's/aarch64/arm64/' \
			 | sed 's/ppc64le/powerpc/' \
			 | sed 's/mips.*/mips/' \
			 | sed 's/riscv64/riscv/' \
			 | sed 's/loongarch64/loongarch/')

# Get Clang's default includes for BPF target
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

# Include paths
INCLUDES := -I. -I$(LIBBPF_INCLUDE) -I$(LIBBPF_INCLUDE)/uapi

# BPF compilation flags
BPF_CFLAGS = -g -O2 -Wall -Wno-compare-distinct-pointer-types \
    -D__TARGET_ARCH_$(ARCH) -target bpf \
    -I. -I$(LIBBPF_INCLUDE) -I$(LIBBPF_INCLUDE)/uapi \
    $(CLANG_BPF_SYS_INCLUDES)

# User space compilation flags
USER_CFLAGS = -O2 -g -Wall $(INCLUDES)

# Link against local libbpf.a
USER_LDFLAGS = $(LIBBPF_LIB) -lelf -lz -lzstd

# Build targets
BPF_SRC = $(PROG_NAME).bpf.c
BPF_OBJ = $(PROG_NAME).bpf.o
SKEL_H = $(PROG_NAME).skel.h
TARGET = $(PROG_NAME)

.PHONY: all clean help bpftool vmlinux

all: $(TARGET)

# Build bpftool if needed
$(BPFTOOL):
	@echo "  BUILD   bpftool"
	$(MAKE) -C $(BPFTOOL_ROOT)/src

# Ensure libbpf is built (built together with bpftool)
$(LIBBPF_LIB): $(BPFTOOL)
	@test -f $@ || (echo "libbpf not found, rebuilding bpftool..." && $(MAKE) -C $(BPFTOOL_ROOT)/src)

# Generate vmlinux.h from running kernel
vmlinux.h: $(BPFTOOL)
	@echo "  VMLINUX vmlinux.h"
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

# Compile BPF program
$(BPF_OBJ): $(BPF_SRC) $(PROG_NAME).h vmlinux.h $(BPFTOOL)
	@echo "  BPF     $@"
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@.tmp
	$(BPFTOOL) gen object $@ $@.tmp
	@rm -f $@.tmp

# Generate BPF skeleton
$(SKEL_H): $(BPF_OBJ) $(BPFTOOL)
	@echo "  SKEL    $@"
	$(BPFTOOL) gen skeleton $< > $@

# Compile user space program
$(TARGET): $(SKEL_H) $(PROG_NAME).c $(PROG_NAME).h process_filter.h process_utils.h $(LIBBPF_LIB)
	@echo "  CC      $@"
	$(CC) $(USER_CFLAGS) $(PROG_NAME).c -o $@ $(USER_LDFLAGS)

bpftool: $(BPFTOOL)

vmlinux: vmlinux.h

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(BPF_OBJ) $(BPF_OBJ).tmp $(SKEL_H) $(TARGET)

clean-all: clean
	rm -f vmlinux.h

help:
	@echo "process monitor build"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build process monitor"
	@echo "  make vmlinux  - Generate vmlinux.h from running kernel"
	@echo "  make bpftool  - Build bpftool only"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make clean-all - Clean everything including vmlinux.h"
	@echo ""
	@echo "Usage after build:"
	@echo "  sudo ./process -m 2 -c 'python,bash,pytest'"
