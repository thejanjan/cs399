CFLAGS=-I/usr/local/include -g

all: matmult

matmult: matmult.cu
	nvcc $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f matmult matmult.o
