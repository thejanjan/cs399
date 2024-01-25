CFLAGS=-W -Wall -I/usr/local/include -g

all: montecarlopi

montecarlopi: montecarlopi.c
	#nvc -fast -Minfo=accel -acc -o $@ $^

clean:
	rm -f montecarlopi montecarlopi.o
