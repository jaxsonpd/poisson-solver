#include <immintrin.h>
#include <stdio.h>

int main() {
    if (__builtin_cpu_supports("avx2")) {
        printf("AVX2 supported!\n");
    } else {
        printf("AVX2 not supported.\n");
    }
    return 0;
}