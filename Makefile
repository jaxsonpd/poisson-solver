all: poisson

# -g outputs debugging information
# -Wall enables all warnings
# -pthread configures threading
CFLAGS = -g -Wall -pthread -D_XOPEN_SOURCE=600
CC = gcc 

poisson: poisson.c worker_thread.c

.PHONY: disassembly
disassembly: poisson.s

poisson.s: poisson
	objdump -S --disassemble $< > $@

.PHONY: profile
profile: poisson.c
	$(CC) $(CFLAGS) -pg $< -o poisson-profile

.PHONY: test
test: poisson
	./test.sh

.PHONY: clean
clean:
	rm -f poisson *.o *.s
