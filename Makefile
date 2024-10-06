all: poisson

# -g outputs debugging information
# -Wall enables all warnings
# -pthread configures threading
CFLAGS = -g -Wall -lpthread -D_XOPEN_SOURCE=600
LDLIBS = -lm
CC = gcc 
SOURCE = poisson.c worker_thread.c utils.c poisson_iter.c flags.h

poisson: $(SOURCE)

.PHONY: disassembly
disassembly: poisson.s

poisson.s: poisson
	objdump -S --disassemble $< > $@

poisson-profile: $(SOURCE)
	$(CC) $(CFLAGS) -pg $^ $(LDLIBS) -o $@

poisson-profile-o: $(SOURCE)
	$(CC) $(CFLAGS) -pg $^ $(LDLIBS) -O3 -o $@
	
poisson-cuda: poisson_cuda.cu cuda_worker.cu
	nvcc $^ -o $@  

.PHONY: test
test: poisson
	./test.sh

.PHONY: clean
clean:
	rm -f poisson *.o *.s
