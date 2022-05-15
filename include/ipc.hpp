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
 * @brief Represents the convolution params passed by the delegate
 *
 */
typedef struct axc_delegate_conv_params
{
    /* Input height of the data */
    uint32_t input_height;
    /* Output height of the data */
    uint32_t output_height;
    /* Input width of the data */
    uint32_t input_width;
    /* Output width of the data */
    uint32_t output_width;
    /* Size of the kernel */
    uint8_t kernel_size;
    /* Number of kernels */
    uint16_t num_kernels;
    /* Type of padding to apply */
    uint8_t padding_type;
    /* Size of the stride {x, y} */
    uint8_t stride_x;
    uint8_t stride_y;
} axc_delegate_conv_params_t;

/**
 * @brief Represents the fully connected layer parameters passed by the
 * delegate
 *
 */
typedef struct axc_delegate_fully_connected_params
{
    /* Size of the input 1 */
    uint32_t input1_size;
    /* Input 2 height of the weights */
    uint32_t input2_height;
    /* Input 2 width of the weights */
    uint32_t input2_width;
} axc_delegate_fully_connected_params_t;

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
    /* Additional parameters */
    char *params;
    /* Additional parameters size */
    uint32_t params_size;
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
 * @brief Supported padding modes by the convolution driver
 *
 */
typedef enum conv_padding
{
    CONV_PADDING_VALID,
    CONV_PADDING_SAME
} conv_padding_t;

/**
 * @brief Call back-end to execute convolution 2D
 *
 * @param input input data to apply convolution
 * @param kernels kernels to use
 * @param params convolution parameters
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float> result of convolution 2D
 */
std::vector<float> apply_conv2d(
    std::vector<float> input,
    std::vector<float> kernels,
    axc_delegate_conv_params_t *params,
    bool verbose);

/**
 * @brief Call back-end to execute fully connected
 *
 * @param input input data to apply matrix-matrix multiplication
 * @param weights weights
 * @param params fully connected parameters
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float> result of matrix-matrix multiplication
 */
std::vector<float> apply_fully_connected(std::vector<float> input,
                                         std::vector<float> weights,
                                         axc_delegate_fully_connected_params_t *params,
                                         bool verbose);

/**
 * @brief Call back-end to execute softmax function
 *
 * @param input data to apply softmax
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float>
 */
std::vector<float> apply_softmax(std::vector<float> input, bool verbose);

/**
 * @brief Read data from a specific buffer
 *
 * @param shm_name shered memory name
 * @param size size of the buffer to read
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float> values read from the buffer
 */
std::vector<float> read_buffer(const char *shm_name, int size, bool verbose);

/**
 * @brief Write in a specific buffer
 *
 * @param data data to write in the buffer
 * @param shm_name shared memory name
 * @param size size of the memory to write
 * @param verbose activate verbose mode to print out messages
 * @return bool true if the operation was successfull, otherwise false
 */
template <typename I, typename T>
bool write_buffer(I data, size_t size, const char *shm_name, bool verbose);

/**
 * @brief Print out shared memory data when verbose mode is on
 *
 * @param shmem shared memory to print
 */
void shared_memory_verbose(axc_shared_mem_t *shmem);

#endif /* KERAS2CPP_IPC_HPP */
