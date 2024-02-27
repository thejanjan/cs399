CFLAGS=-W -Wall -I/usr/local/include -g

all: stream

stream: stream.cu
	nvcc $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f stream stream.o
