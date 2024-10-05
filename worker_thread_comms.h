/** 
 * @file worker_thread_comms.h
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-10-02
 * @brief Allow for communication between worker threads using mutexes
 */


#ifndef WORKER_THREAD_COMMS_H
#define WORKER_THREAD_COMMS_H


#include <stdint.h>
#include <stdbool.h>

typedef enum workerCommsFlag_e {
    WORKERS_READY_TO_COPY,
    COPY_COMPLETE
} workerCommsFlag_t;

/** 
 * @brief Init the comms
 * 
 */
void worker_comms_init(void);

/** 
 * @brief Deinit the comms
 * 
 */
void worker_comms_deinit(void);


/** 
 * @brief Set the worker thread comms flag
 * @param flag_value the value to set the flag to
 */
void worker_comms_set(workerCommsFlag_t flag_value);

/** 
 * @brief Get the worker thread comms flag
 * 
 * @return The current value of the comms flag
 */
workerCommsFlag_t worker_comms_get(void);


#endif // WORKER_THREAD_COMMS_H