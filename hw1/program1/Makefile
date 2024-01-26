CFLAGS=-W -Wall -I/usr/local/include -g

all: montecarlopi

montecarlopi: montecarlopi.c
	nvc -acc --gcc-toolchain=/usr/bin -o $@ $^

clean:
	rm -f montecarlopi montecarlopi.o
