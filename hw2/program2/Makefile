CFLAGS=-W -Wall -I/usr/local/include -g

all: heat

heat: heat.c
	nvc -acc --gcc-toolchain=/usr/bin -o $@ $^

clean:
	rm -f heat heat.o
