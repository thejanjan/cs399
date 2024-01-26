CFLAGS=-W -Wall -I/usr/local/include -g

all: stream

stream: stream.c
	nvc -acc --gcc-toolchain=/usr/bin -o $@ $^

clean:
	rm -f stream stream.o
