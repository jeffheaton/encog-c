LIB_IDIR =./encog-core/
CMD_IDIR =./encog-cmd/
LIB_ODIR=./obj-lib
CMD_ODIR=./obj-cmd
LDIR =./lib

_LIB_DEPS = encog.h
LIB_DEPS = $(patsubst %,$(LIB_IDIR)/%,$(_LIB_DEPS))

_LIB_OBJ = activation.o errorcalc.o network_io.o util.o util_str.o data.o errors.o network.o pso.o util_file.o vector.o
LIB_OBJ = $(patsubst %,$(LIB_ODIR)/%,$(_LIB_OBJ))

_CMD_DEPS = encog-cmd.h
CMD_DEPS = $(patsubst %,$(CMD_IDIR)/%,$(_CMD_DEPS))

_CMD_OBJ = encog-cmd.o
CMD_OBJ = $(patsubst %,$(CMD_ODIR)/%,$(_CMD_OBJ))

ENCOG_LIB = $(LDIR)/encog.a

CC=gcc
CFLAGS=-I$(LIB_IDIR) -fopenmp -std=c99 -pedantic -Wall

LIBS=-lm

$(LIB_ODIR)/%.o: ./encog-core/%.c $(LIB_DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(CMD_ODIR)/%.o: ./encog-cmd/%.c $(CMD_DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

encog: $(CMD_OBJ) $(ENCOG_LIB)
	gcc -o $@ $^ $(CFLAGS) -lm $(ENCOG_LIB)

$(ENCOG_LIB): $(LIB_OBJ)
	ar rcs $(ENCOG_LIB) $(LIB_OBJ)

.PHONY: clean
clean:
	rm -f $(LIB_ODIR)/*.o *~ core $(INCDIR)/*~ 
	rm -f $(CMD_ODIR)/*.o *~ core $(INCDIR)/*~ 

