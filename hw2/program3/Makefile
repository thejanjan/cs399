CFLAGS=-W -Wall -I/usr/local/include -g

all: matmult

matmult: matmult.c
	nvc -acc --gcc-toolchain=/usr/bin -o $@ $^

clean:
	rm -f matmult matmult.o
