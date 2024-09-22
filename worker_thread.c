/**
 * @file worker_thread.c
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-09-23
 * @brief The implementation of the worker thread
 */


#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


#include "utils.h"

#include "worker_thread.h"

void* worker_thread(void* pargs) {
    workerThread_t* worker_info = (workerThread_t*)pargs;
    int N = worker_info->N;

    for (int n = 0; n < worker_info->iterations; n++) {
        for (int k = worker_info->k_start; k < worker_info->k_end; k++) {
            for (int j = worker_info->j_start; j < worker_info->j_end; j++) {
                for (int i = worker_info->i_start; i < worker_info->i_end; i++) {
                    if (i == 0 && j == 0) {
                        idx(worker_info->next, N, k, j, i) = (2 * idx(worker_info->curr, N, k, j, i + 1)
                            + 2 * idx(worker_info->curr, N, k, j + 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;
                    } else if (i == N - 1 && j == N - 1) {
                        idx(worker_info->next, N, k, j, i) = (2 * idx(worker_info->curr, N, k, j, i - 1)
                            + 2 * idx(worker_info->curr, N, k, j - 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;
                    } else if (i == 0 && j == N - 1) {
                        idx(worker_info->next, N, k, j, i) = (2 * idx(worker_info->curr, N, k, j, i + 1)
                            + 2 * idx(worker_info->curr, N, k, j - 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;
                    } else if (i == N - 1 && j == 0) {
                        idx(worker_info->next, N, k, j, i) = (2 * idx(worker_info->curr, N, k, j, i - 1)
                            + 2 * idx(worker_info->curr, N, k, j + 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;
                    } else if (i == 0) {
                        idx(worker_info->next, N, k, j, i) = (2 * idx(worker_info->curr, N, k, j, i + 1)
                            + idx(worker_info->curr, N, k, j + 1, i) + idx(worker_info->curr, N, k, j - 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;
                    } else if (i == N - 1) {
                        idx(worker_info->next, N, k, j, i) = (2 * idx(worker_info->curr, N, k, j, i - 1)
                            + idx(worker_info->curr, N, k, j + 1, i) + idx(worker_info->curr, N, k, j - 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;
                    } else if (j == 0) {
                        idx(worker_info->next, N, k, j, i) = (idx(worker_info->curr, N, k, j, i + 1) + idx(worker_info->curr, N, k, j, i - 1)
                            + 2 * idx(worker_info->curr, N, k, j + 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;
                    } else if (j == N - 1) {
                        idx(worker_info->next, N, k, j, i) = (idx(worker_info->curr, N, k, j, i + 1) + idx(worker_info->curr, N, k, j, i - 1)
                            + 2 * idx(worker_info->curr, N, k, j - 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;

                    } else {
                        idx(worker_info->next, N, k, j, i) = (idx(worker_info->curr, N, k, j, i + 1) + idx(worker_info->curr, N, k, j, i - 1)
                            + idx(worker_info->curr, N, k, j + 1, i) + idx(worker_info->curr, N, k, j - 1, i)
                            + idx(worker_info->curr, N, k + 1, j, i) + idx(worker_info->curr, N, k - 1, j, i)
                            - worker_info->delta * worker_info->delta * idx(worker_info->source, N, k, j, i)) / 6;;
                    }
                }
            }
        }
        // TODO move to custom function for worker indexes
        memcpy(worker_info->curr, worker_info->next, N * N * N * sizeof(double));
        // TODO do semaphore stuff
    }


    return NULL;
}
