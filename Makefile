all: poisson

# -g outputs debugging information
# -Wall enables all warnings
# -pthread configures threading
CFLAGS = -g -Wall -lpthread -D_XOPEN_SOURCE=600 -mavx -O3
LDLIBS = -lm
CC = gcc 
SOURCE = poisson.c worker_thread.c utils.c poisson_iter.c flags.h

poisson: $(SOURCE)

.PHONY: disassembly
disassembly: poisson.s

poisson.s: poisson
	objdump -S --disassemble $< > $@

poisson-oz: $(SOURCE)
	$(CC) $(CFLAGS) -Oz $^ $(LDLIBS) -o $@

poisson-os: $(SOURCE)
	$(CC) $(CFLAGS) -Os $^ $(LDLIBS) -o $@

poisson-o3: $(SOURCE)
	$(CC) $(CFLAGS) -O3 $^ $(LDLIBS) -o $@

poisson-o2: $(SOURCE)
	$(CC) $(CFLAGS) -O2 $^ $(LDLIBS) -o $@

poisson-o1: $(SOURCE)
	$(CC) $(CFLAGS) -O1 $^ $(LDLIBS) -o $@

poisson-ofast: $(SOURCE)
	$(CC) $(CFLAGS) -Ofast $^ $(LDLIBS) -o $@


poisson-profile: $(SOURCE)
	$(CC) $(CFLAGS) -pg $^ $(LDLIBS) -o $@

poisson-profile-o: $(SOURCE)
	$(CC) $(CFLAGS) -pg $^ $(LDLIBS) -Og -o $@
	
poisson-cuda: poisson_cuda.cu cuda_worker.cu
	nvcc -O3 $^ -o $@  

.PHONY: test
test: poisson
	./test.sh

.PHONY: clean
clean:
	rm -f poisson *.o *.s
