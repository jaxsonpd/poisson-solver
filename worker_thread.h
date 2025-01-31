/** 
 * @file worker_thread.h
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-09-23
 * @brief Header file for the worker thread module
 */


#ifndef WORKER_THREAD_H
#define WORKER_THREAD_H


#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>


#include "utils.h"

typedef struct WorkerThread {
    int thread_id;
    
    int N;
    double* source;
    double* curr;
    double* next;
    float delta;

    int iterations;

    slice3D_t slice_3D;

    pthread_barrier_t* barrier;

} workerThread_t;

/**
 * @brief Entry point to a worker thread
 * @param pargs a WorkerThread_t pointer
 *
 * @return void* 
 */
void* worker_thread(void* pargs);

#endif // WORKER_THREAD_H