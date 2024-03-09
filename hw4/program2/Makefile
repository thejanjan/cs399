CFLAGS=-I/usr/local/include -g

all: heat

heat: heat.cu
	nvcc $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f heat heat.o
