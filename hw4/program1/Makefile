CFLAGS= -lcurand -I/usr/local/include -g

all: axpy

axpy: axpy.cu
	nvcc $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f axpy axpy.o
