# SPDX-License-Identifier: GPL-2.0
# Makefile for memcg_priority BPF loader
#
# NOTE: This component requires a kernel with memcg_bpf_ops support.
# The vmlinux.h is generated from /sys/kernel/btf/vmlinux and must
# contain struct memcg_bpf_ops for the BPF program to compile.

# Program name
PROG_NAME = memcg_priority

# Compiler and tools
CLANG ?= clang
CC ?= gcc

# Project paths
PROJECT_ROOT := $(shell git rev-parse --show-toplevel)
BPFTOOL_ROOT := $(PROJECT_ROOT)/third_party/bpftool

# Use libbpf from bpftool (same as scheduler and process)
BPFTOOL := $(BPFTOOL_ROOT)/src/bpftool
LIBBPF_DIR := $(BPFTOOL_ROOT)/src/libbpf
LIBBPF_INCLUDE := $(LIBBPF_DIR)/include
LIBBPF_LIB := $(LIBBPF_DIR)/libbpf.a

# Get Clang's default includes for BPF target
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

# System includes for BPF
SYS_INCLUDES = \
    -I$(LIBBPF_INCLUDE) \
    -I. \
    $(CLANG_BPF_SYS_INCLUDES)

# BPF compilation flags
BPF_CFLAGS = -g -O2 -Wall -Wno-compare-distinct-pointer-types \
    -D__TARGET_ARCH_x86 -mcpu=v3 -mlittle-endian \
    $(SYS_INCLUDES)

# User space compilation flags
USER_CFLAGS = -O2 -g -Wall \
    -I$(LIBBPF_INCLUDE) \
    -I.

# Link against local libbpf.a
USER_LDFLAGS = $(LIBBPF_LIB) -lelf -lz -lzstd

# Build targets
BPF_SRC = $(PROG_NAME).bpf.c
BPF_OBJ = $(PROG_NAME).bpf.o
SKEL_H = $(PROG_NAME).bpf.skel.h
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
# NOTE: Requires kernel with memcg_bpf_ops support
vmlinux.h: $(BPFTOOL)
	@echo "  VMLINUX vmlinux.h"
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

# Compile BPF program
$(BPF_OBJ): $(BPF_SRC) $(PROG_NAME).h vmlinux.h
	@echo "  BPF     $@"
	$(CLANG) $(BPF_CFLAGS) -target bpf -c $< -o $@

# Generate BPF skeleton
$(SKEL_H): $(BPF_OBJ) $(BPFTOOL)
	@echo "  SKEL    $@"
	$(BPFTOOL) gen skeleton $< name $(PROG_NAME) > $@

# Compile user space program
$(TARGET): $(SKEL_H) $(PROG_NAME).c $(PROG_NAME).h $(LIBBPF_LIB)
	@echo "  CC      $@"
	$(CC) $(USER_CFLAGS) $(PROG_NAME).c -o $@ $(USER_LDFLAGS)

bpftool: $(BPFTOOL)

vmlinux: vmlinux.h

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(BPF_OBJ) $(SKEL_H) $(TARGET)

clean-all: clean
	rm -f vmlinux.h

help:
	@echo "memcg_priority BPF loader build"
	@echo ""
	@echo "NOTE: Requires a kernel with memcg_bpf_ops support."
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build memcg_priority"
	@echo "  make vmlinux  - Generate vmlinux.h from running kernel"
	@echo "  make bpftool  - Build bpftool only"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make clean-all - Clean everything including vmlinux.h"
	@echo ""
	@echo "Usage after build:"
	@echo "  sudo ./memcg_priority --high /path/to/high_cgroup \\"
	@echo "                        --low /path/to/low_cgroup1 \\"
	@echo "                        --low /path/to/low_cgroup2 \\"
	@echo "                        --delay-ms 2000 --below-low"
