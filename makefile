# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
   LION        = $(strip $(findstring 10.7, $(shell egrep "<string>10\.7" /System/Library/CoreServices/SystemVersion.plist)))
endif

# Define arch to use
ARCH := $(shell getconf LONG_BIT)

RPI_FLAGS :=
CPP_FLAGS_32 := -m32
CPP_FLAGS_64 := -m64
#use CUDA?
ifeq ($(CUDA),1) 

endif

# Encog Paths
LIB_IDIR =./encog-core/
CMD_IDIR =./encog-cmd/
LIB_ODIR=./obj-lib
CMD_ODIR=./obj-cmd
LDIR =./lib

# CUDA Paths
ifeq ($(CUDA),1) 
	CUDA_INSTALL_PATH ?= /usr/local/cuda
	ifdef cuda-install
		CUDA_INSTALL_PATH := $(cuda-install)
	endif
endif

_LIB_DEPS = encog.h
LIB_DEPS = $(patsubst %,$(LIB_IDIR)/%,$(_LIB_DEPS))

_LIB_OBJ = activation.o errorcalc.o network_io.o util.o util_str.o data.o errors.o network.o pso.o util_file.o vector.o encog.o nm.o object.o rprop.o hash.o train.o 
LIB_OBJ = $(patsubst %,$(LIB_ODIR)/%,$(_LIB_OBJ))

_CMD_DEPS = encog-cmd.h
CMD_DEPS = $(patsubst %,$(CMD_IDIR)/%,$(_CMD_DEPS))

_CMD_OBJ = encog-cmd.o cuda_test.o node_unix.o
CMD_OBJ = $(patsubst %,$(CMD_ODIR)/%,$(_CMD_OBJ))

ifeq ($(CUDA),1) 
	_CMD_CUOBJ = cuda_vecadd.cu.o
	_LIB_CUOBJ = encog_cuda.cu.o cuda_eval.cu.o
	CMD_CUOBJ = $(patsubst %,$(CMD_ODIR)/%,$(_CMD_CUOBJ))
	LIB_CUOBJ = $(patsubst %,$(LIB_ODIR)/%,$(_LIB_CUOBJ))
endif

ENCOG_LIB = $(LDIR)/encog.a

CC=gcc
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CFLAGS=-I$(LIB_IDIR) -fopenmp -std=gnu99 -pedantic -O3 -Wall $(CPP_FLAGS_$(ARCH))
NVCCFLAGS = -I$(LIB_IDIR) $(CPP_FLAGS_$(ARCH))
MKDIR_P = mkdir -p
LIBS=-lm

ifeq ($(CUDA),1) 
CFLAGS+= -DENCOG_CUDA=1
CFLAGS+= -I$(CUDA_INSTALL_PATH)/include
endif

# Libs
ifeq ($(CUDA),1) 
ifneq ($(DARWIN),)
    LIB       := -L$(CUDA_INSTALL_PATH)/lib 
else
  ifeq ($(ARCH),64)
       LIB       := -L$(CUDA_INSTALL_PATH)/lib64
  else
       LIB       := -L$(CUDA_INSTALL_PATH)/lib
  endif
endif

LIB+= -lcudart
endif

$(CMD_ODIR)/%.cu.o : ./encog-cmd/%.cu $(CMD_DEPS)
	${MKDIR_P} $(CMD_ODIR)
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

$(LIB_ODIR)/%.cu.o : ./encog-core/%.cu $(LIB_DEPS)
	${MKDIR_P} $(LIB_ODIR)
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

$(LIB_ODIR)/%.o: ./encog-core/%.c $(LIB_DEPS)
	${MKDIR_P} $(LIB_ODIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(CMD_ODIR)/%.o: ./encog-cmd/%.c $(CMD_DEPS)
	${MKDIR_P} $(CMD_ODIR)
	$(CC) -c -o $@ $< $(CFLAGS)

encog: $(CMD_OBJ) $(CMD_CUOBJ) $(ENCOG_LIB)
	$(CC) -o $@ $^ $(CFLAGS) -lm $(ENCOG_LIB) $(LIB)

$(ENCOG_LIB): $(LIB_OBJ) $(LIB_CUOBJ)
	${MKDIR_P} $(LDIR)
	ar rcs $(ENCOG_LIB) $(LIB_OBJ) $(LIB_CUOBJ)

.PHONY: clean
clean:
	rm -f $(LIB_ODIR)/*.o 
	rm -f $(CMD_ODIR)/*.o
	rm -f $(LDIR)/*.a
