CFLAGS=-W -Wall -I/usr/local/include -g

all: main

main: main.c
	nvc -fast -Minfo=accel -acc -o $@ $^

clean:
	rm -f main main.o
