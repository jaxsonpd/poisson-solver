/**
 * @file worker_thread_comms.c
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-10-02
 * @brief Definitions of the functions that allow communication between worker
 * threads
 */


#include <stdint.h>
#include <stdbool.h>

#include "pthread.h"

#include "worker_thread_comms.h"

workerCommsFlag_t worker_comms_flag = 1;
pthread_mutex_t worker_comms_mutex;

void worker_comms_init(void) {
    pthread_mutex_init(&worker_comms_mutex, NULL);
}

void worker_comms_deinit(void) {
    pthread_mutex_destroy(&worker_comms_mutex);
}

void worker_comms_set(workerCommsFlag_t flag_value) {
    pthread_mutex_lock(&worker_comms_mutex);
    worker_comms_flag = flag_value;
    pthread_mutex_unlock(&worker_comms_mutex);
}

workerCommsFlag_t worker_comms_get(void) {
    pthread_mutex_lock(&worker_comms_mutex);
    workerCommsFlag_t flag_value = worker_comms_flag;
    pthread_mutex_unlock(&worker_comms_mutex);

    return flag_value;
}