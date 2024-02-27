CFLAGS= -lcurand -W -Wall -I/usr/local/include -g

all: montecarlopi

montecarlopi: montecarlopi.cu
	nvcc $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f montecarlopi montecarlopi.o
