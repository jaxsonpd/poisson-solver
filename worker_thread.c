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
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include "utils.h"
#include "poisson_iter.h"
#include "worker_thread_comms.h"

#include "worker_thread.h"

/**
 * @brief Wait before performing memcopy_3D
 *
 */
void wait_to_copy(workerThread_t* worker_info) {
    pthread_barrier_wait(worker_info->barrier);
}

/**
 * @brief Wait before starting new iteration
 *
 */
void wait_to_start_next(workerThread_t* worker_info) {
    pthread_barrier_wait(worker_info->barrier);
}

/**
 * @brief The worker thread function
 * @param pargs a WorkerThread_t pointer
 *
 * @return void*
 */
void* worker_thread(void* pargs) {
    workerThread_t* worker_info = (workerThread_t*)pargs;
    int N = worker_info->N;

    for (int n = 0; n < worker_info->iterations; n++) {
        if (worker_info->slice_3D.i_start == 0 || worker_info->slice_3D.i_end == N || worker_info->slice_3D.j_start == 0 || worker_info->slice_3D.j_end == N) {
            apply_von_neuman_boundary_slice(N, worker_info->source, worker_info->curr, worker_info->next, worker_info->delta, worker_info->slice_3D);
        }
        poisson_iteration_inner_slice(N, worker_info->source, worker_info->curr, worker_info->next, worker_info->delta, worker_info->slice_3D);

        wait_to_copy(worker_info);
        if (worker_info->thread_id == 0) {
            worker_comms_set(WORKERS_READY_TO_COPY);

            while (worker_comms_get() != COPY_COMPLETE) {
                usleep(10);
            }
        }
        // memcopy_3D(N, worker_info->curr, worker_info->next, worker_info->slice_3D);
        wait_to_start_next(worker_info);

    }

    return NULL;
}
