#ifndef KERAS2CPP_IPC_HPP
#define KERAS2CPP_IPC_HPP

#include <fcntl.h>
#include <iostream>
#include <semaphore.h>
#include <unistd.h>
#include <sys/mman.h>
#include <vector>

#define SEM_READY_SIGNAL_NAME "/axc-sem-ready-signal"
#define SHARED_MEM_NAME "axc-shared-memory"
#define SHARED_MEM_OP1_NAME "axc-shared-memory-op1"
#define SHARED_MEM_OP2_NAME "axc-shared-memory-op2"
#define SHARED_MEM_PARAMS_NAME "axc-shared-memory-params"
#define SHARED_MEM_OUT_NAME "axc-shared-memory-out"

/**
 * @brief Represents the operations supported by the accelerator
 *
 */
typedef enum axc_op
{
    AXC_CONV2D,
    AXC_FULLY_CONNECTED,
    AXC_SOFTMAX,
} axc_op_t;

/**
 * @brief Current status of the driver request
 *
 */
typedef enum axc_driver_status
{
    AXC_SUCCESS,
    AXC_ERROR
} axc_driver_status_t;

/**
 * @brief Define the operations between the delegate and source driver
 *
 */
typedef enum axc_user_request
{
    /* No operation */
    AXC_IDLE,
    /* Allocate shared memory for buffers */
    AXC_ALLOCATE,
    /* Read buffers */
    AXC_READ,
    /* Execute operation (axc_op_t) */
    AXC_EXECUTE,
    /* Deallocate shared memory */
    AXC_DEALLOCATE,
} axc_user_request_t;

/**
 * @brief Shared memory section for IPC between delegate and source driver
 *
 */
typedef struct axc_shared_mem
{
    /* First operator */
    float *op1;
    /* Size of the first operator */
    uint32_t op1_size;
    /* Second operator */
    float *op2;
    /* Size of the second operator */
    uint32_t op2_size;
    /* Output */
    float *output;
    /* Size of the output */
    uint32_t out_size;
    /* Operation to perform */
    axc_op_t operation;
    /* Delegate request */
    axc_user_request_t request;
    /* Request status after finish */
    axc_driver_status_t status;
} axc_shared_mem_t;

/**
 * @brief Call back-end to execute softmax function
 *
 * @param input data to apply softmax
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float>
 */
std::vector<float> apply_softmax(std::vector<float> input, bool verbose);

/**
 * @brief Write in a specific buffer
 * 
 * @param data data to write in the buffer
 * @param shm_name shared memory name
 * @param verbose activate verbose mode to print out messages
 * @return bool true if the operation was successfull, otherwise false
 */
bool write_in_buffer(std::vector<float> data, const char* shm_name, bool verbose);

/**
 * @brief Print out shared memory data when verbose mode is on
 *
 * @param shmem shared memory to print
 */
void shared_memory_verbose(axc_shared_mem_t *shmem);

#endif /* KERAS2CPP_IPC_HPP */
