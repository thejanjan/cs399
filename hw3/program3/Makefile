CFLAGS=-I/usr/local/include -g

all: mandelbrot

mandelbrot: mandelbrot.cu
	nvcc $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f mandelbrot mandelbrot.o
