CFLAGS=-W -Wall -I/usr/local/include -g

all: mandelbrot

mandelbrot: mandelbrot.c
	nvc -acc --gcc-toolchain=/usr/bin -o $@ $^

clean:
	rm -f mandelbrot mandelbrot.o
