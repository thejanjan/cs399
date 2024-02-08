CFLAGS=-W -Wall -I/usr/local/include -g

all: axpy

axpy: axpy.c
	nvc -acc --gcc-toolchain=/usr/bin -o $@ $^

clean:
	rm -f axpy axpy.o
