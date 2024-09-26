all: poisson

# -g outputs debugging information
# -Wall enables all warnings
# -pthread configures threading
CFLAGS = -g -Wall -lpthread -D_XOPEN_SOURCE=600
LDLIBS = -lm
CC = gcc 

poisson: poisson.c worker_thread.c utils.c

.PHONY: disassembly
disassembly: poisson.s

poisson.s: poisson
	objdump -S --disassemble $< > $@

.PHONY: profile
profile: poisson.c worker_thread.c utils.c
	$(CC) $(CFLAGS) -pg $^ $(LDLIBS) -o poisson-profile

.PHONY: test
test: poisson
	./test.sh

.PHONY: clean
clean:
	rm -f poisson *.o *.s
