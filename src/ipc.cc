#include <ipc.hpp>

IPC::IPC(bool verbose)
{
    fd_shm = 0;
    mutex_sem = nullptr;
    shared_mem = nullptr;
    this->verbose = verbose;
}

IPC::IPC()
{
    fd_shm = 0;
    mutex_sem = nullptr;
    shared_mem = nullptr;
    verbose = false;
}

/**
 * @brief Call back-end to execute convolution 2D
 *
 * @param input input data to apply convolution
 * @param kernels kernels to use
 * @param params convolution parameters
 * @return std::vector<float> result of convolution 2D
 */
std::vector<float> IPC::apply_conv2d(
    std::vector<float> input,
    std::vector<float> kernels,
    axc_delegate_conv_params_t *params)
{
    /* Open shared memory for IPC */
    open_shared_mem();
    /* Open semaphores for IPC synchronization */
    open_sem();

    /* Take control of the shared memory to allocate memory */
    sem_wait(mutex_sem);

    if (verbose)
    {
        std::cout << std::endl
                  << "Default shared memory" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    /* Ask for buffers */
    shared_mem->op1_size = input.size();
    shared_mem->op2_size = kernels.size();
    shared_mem->out_size = params->num_kernels * params->output_height * params->output_width;
    shared_mem->params_size = sizeof(axc_delegate_conv_params_t);
    shared_mem->request = AXC_ALLOCATE;
    wait_response();

    if (verbose)
    {
        std::cout << std::endl
                  << "Conv2D request: Allocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    if (AXC_ERROR == shared_mem->status)
    {
        /* Release semaphore */
        sem_post(mutex_sem);
        release_shared_mem();

        std::cout << std::endl
                  << "Buffer allocation was not successful" << std::endl;

        exit(-1);
    }

    if (verbose)
        std::cout << std::endl
                  << "Writting on the buffers" << std::endl;

    /* Copy data to the buffers */
    write_buffer<std::vector<float>, float>(input, input.size(), SHARED_MEM_OP1_NAME, verbose);
    write_buffer<std::vector<float>, float>(kernels, kernels.size(), SHARED_MEM_OP2_NAME, verbose);
    write_buffer<char *, char>((char *)params, sizeof(axc_delegate_conv_params_t), SHARED_MEM_PARAMS_NAME, verbose);

    if (verbose)
        std::cout << "Conv2D request: Execute" << std::endl;

    shared_mem->request = AXC_EXECUTE;
    shared_mem->operation = AXC_CONV2D;
    wait_response();

    /* Get the output of the conv2d function */
    std::vector<float> output = read_buffer(SHARED_MEM_OUT_NAME, shared_mem->out_size, verbose);

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Buffers were read" << std::endl;
        shared_memory_verbose(shared_mem);

        std::cout << std::endl
                  << "Conv2D request: Deallocate buffers" << std::endl;
    }

    shared_mem->request = AXC_DEALLOCATE;
    wait_response();

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Dellocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    /* Release semaphore */
    sem_post(mutex_sem);
    release_shared_mem();

    if (verbose)
        std::cout << std::endl
                  << "Shared memory was unmapped" << std::endl;

    return output;
}

/**
 * @brief Call back-end to execute fully connected
 *
 * @param input input data to apply matrix-matrix multiplication
 * @param weights weights
 * @param params fully connected parameters
 * @return std::vector<float> result of matrix-matrix multiplication
 */
std::vector<float> IPC::apply_fully_connected(
    std::vector<float> input,
    std::vector<float> weights,
    axc_delegate_fully_connected_params_t *params)
{
    /* Open IPC */
    open_shared_mem();
    open_sem();

    /* Result of the matrix-matrix multiplication */
    std::vector<float> output;

    /* Take control of the shared memory to allocate memory */
    sem_wait(mutex_sem);

    if (verbose)
    {
        std::cout << std::endl
                  << "Default shared memory" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    /* Ask for buffers */
    shared_mem->op1_size = input.size();
    shared_mem->op2_size = weights.size();
    shared_mem->out_size = params->input1_size * params->input2_width;
    shared_mem->params_size = sizeof(axc_delegate_fully_connected_params_t);
    shared_mem->request = AXC_ALLOCATE;
    wait_response();

    if (verbose)
    {
        std::cout << std::endl
                  << "Fully connected request: Allocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    if (AXC_ERROR == shared_mem->status)
    {
        /* Release semaphore */
        sem_post(mutex_sem);
        release_shared_mem();

        std::cout << std::endl
                  << "Buffer allocation was not successful" << std::endl;

        exit(-1);
    }

    if (verbose)
        std::cout << std::endl
                  << "Writing on the buffers" << std::endl;

    /* Copy data to the buffers */
    write_buffer<std::vector<float>, float>(input, input.size(), SHARED_MEM_OP1_NAME, verbose);
    write_buffer<std::vector<float>, float>(weights, weights.size(), SHARED_MEM_OP2_NAME, verbose);
    write_buffer<char *, char>((char *)params, sizeof(axc_delegate_fully_connected_params_t), SHARED_MEM_PARAMS_NAME, verbose);

    if (verbose)
        std::cout << "Fully connected request: Execute" << std::endl;

    shared_mem->request = AXC_EXECUTE;
    shared_mem->operation = AXC_CONV2D;
    wait_response();

    /* Get the output of the fully connected function */
    output = read_buffer(SHARED_MEM_OUT_NAME, shared_mem->out_size, verbose);

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Buffers were read" << std::endl;
        shared_memory_verbose(shared_mem);

        std::cout << std::endl
                  << "Fully connected request: Deallocate buffers" << std::endl;
    }

    shared_mem->request = AXC_DEALLOCATE;
    wait_response();

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Deallocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    /* Release semaphore */
    sem_post(mutex_sem);
    release_shared_mem();

    if (verbose)
        std::cout << std::endl
                  << "Shared memory was unmapped" << std::endl;

    return output;
}

/**
 * @brief Call back-end to execute Softmax function
 *
 * @param input data to apply softmax
 * @return std::vector<float> softmax result
 */
std::vector<float> IPC::apply_softmax(std::vector<float> input)
{
    open_shared_mem();
    open_sem();

    /* Take control of the shared memory to allocate memory */
    sem_wait(mutex_sem);

    if (verbose)
    {
        std::cout << std::endl
                  << "Default shared memory" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    /* Ask for buffer */
    shared_mem->op1_size = input.size();
    shared_mem->op2_size = 0;
    shared_mem->out_size = input.size();
    shared_mem->params_size = 0;
    shared_mem->request = AXC_ALLOCATE;

    if (verbose)
    {
        std::cout << std::endl
                  << "Softmax request: Allocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    wait_response();

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Allocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    if (AXC_ERROR == shared_mem->status)
    {
        /* Release semaphore */
        sem_post(mutex_sem);
        release_shared_mem();

        std::cout << std::endl
                  << "Buffer allocation was not successful" << std::endl;

        exit(-1);
    }

    if (verbose)
        std::cout << std::endl
                  << "Writing on the buffers" << std::endl;

    /* Copy data to the buffers */
    write_buffer<std::vector<float>, float>(input, input.size(), SHARED_MEM_OP1_NAME, verbose);

    if (verbose)
        std::cout << "Softmax request: Execute" << std::endl;

    shared_mem->request = AXC_EXECUTE;
    shared_mem->operation = AXC_SOFTMAX;
    wait_response();

    /* Get the output of the sofmax function */
    std::vector<float> output = read_buffer(SHARED_MEM_OUT_NAME, shared_mem->out_size, verbose);

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Buffers were read" << std::endl;
        shared_memory_verbose(shared_mem);

        std::cout << std::endl
                  << "Softmax request: Deallocate buffers" << std::endl;
    }

    shared_mem->request = AXC_DEALLOCATE;
    wait_response();

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Deallocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    /* Release semaphore */
    sem_post(mutex_sem);
    release_shared_mem();

    return output;
}

/**
 * @brief Check if there's an accelerator capable of executing the
 * operation
 *
 * @param op operation to execute
 * @return true if accelerator exists
 * @return false if accelerator doesn't exist
 */
bool IPC::exists_accel(axc_op_t op)
{
    bool exists;

    /* Open shared memory for IPC */
    open_shared_mem();
    /* Open semaphores for IPC synchronization */
    open_sem();

    shared_mem->request = AXC_EXISTS;
    shared_mem->operation = op;
    wait_response();

    exists = shared_mem->status == AXC_SUCCESS;

    /* Release semaphore */
    sem_post(mutex_sem);
    release_shared_mem();

    return exists;
}

/**
 * @brief Open shared memory section for IPC
 *
 */
void IPC::open_shared_mem()
{
    /* Get shared memory */
    fd_shm = shm_open(SHARED_MEM_NAME, O_RDWR, 0);

    if (fd_shm < 0)
    {
        std::cout << "The device '" << SHARED_MEM_NAME << "' couldn't be open" << std::endl;
        std::cout << errno << std::endl;
        exit(-1);
    }

    /* Ask for memory */
    shared_mem = (axc_shared_mem_t *)mmap(NULL, sizeof(axc_shared_mem_t),
                                          PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0);

    if (MAP_FAILED == shared_mem)
    {
        close(fd_shm);
        std::cout << "The mapping could not be done" << std::endl;
        exit(-1);
    }
}

/**
 * @brief Open semaphore for shared memory synchronization
 *
 */
void IPC::open_sem()
{
    /* Mutex sem for shared memory */
    mutex_sem = sem_open(SEM_READY_SIGNAL_NAME, 0, 0, 0);

    if (SEM_FAILED == mutex_sem)
    {
        munmap(shared_mem, sizeof(axc_shared_mem_t));
        close(fd_shm);
        std::cout << "'" << SEM_READY_SIGNAL_NAME << "' sem open failed" << std::endl;
        exit(-1);
    }

    if (verbose)
        std::cout << "Semaphores are ready!" << std::endl;
}

/**
 * @brief Read data from a specific buffer
 *
 * @param shm_name shared memory name
 * @param size size of the buffer to read
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float> values read from the buffer
 */
std::vector<float> IPC::read_buffer(const char *shm_name, int size, bool verbose)
{
    /* To store the read output */
    std::vector<float> output(size);

    /* File descriptor of the shared memory */
    int fd_buffer;
    /* Shared memory address for the buffer */
    float *buffer;

    /* Get shared memory */
    fd_buffer = shm_open(shm_name, O_RDWR, 0);

    if (fd_buffer < 0)
    {
        std::cout << "The device '" << shm_name << "' could not open" << std::endl;
        exit(-1);
    }

    /* Ask for memory */
    buffer = (float *)mmap(NULL, sizeof(float) * size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_buffer, 0);

    if (MAP_FAILED == buffer)
    {
        close(fd_buffer);
        std::cout << "The mapping of the '" << shm_name << "' could not be done" << std::endl;
        exit(-1);
    }

    /* Copy data from the buffers */
    for (int i = 0; i < size; ++i)
        output[i] = buffer[i];

    munmap(buffer, sizeof(float) * size);
    shm_unlink(shm_name);
    close(fd_buffer);

    return output;
}

/**
 * @brief Release shared memory section for IPC
 *
 */
void IPC::release_shared_mem()
{
    /* Release memory */
    munmap(shared_mem, sizeof(axc_shared_mem_t));
    close(fd_shm);

    shared_mem = nullptr;
    fd_shm = 0;
}

/**
 * @brief Print out shared memory data when verbose mode is on
 *
 * @param shmem shared memory to print
 */
void IPC::shared_memory_verbose(axc_shared_mem_t *shmem)
{
    std::cout << "Shared memory values: " << std::endl;
    std::cout << "operation: " << shmem->operation << std::endl;
    std::cout << "request: " << shmem->request << std::endl;
    std::cout << "request status: " << shmem->status << std::endl;
    std::cout << "op1: " << shmem->op1 << std::endl;
    std::cout << "op1 size: " << shmem->op1_size << std::endl;
    std::cout << "op2: " << shmem->op2 << std::endl;
    std::cout << "op2 size: " << shmem->op2_size << std::endl;
    std::cout << "out: " << shmem->output << std::endl;
    std::cout << "out size: " << shmem->out_size << std::endl;
    std::cout << "params size: " << shmem->params_size << std::endl;
}

/**
 * @brief Wait daemon response
 *
 */
void IPC::wait_response()
{
    while (AXC_IDLE != shared_mem->request)
    {
        sem_post(mutex_sem);
        /* Wait 1 us */
        usleep(1);
        sem_wait(mutex_sem);
    }
}

/**
 * @brief Write in a specific buffer
 *
 * @param data data to write in the buffer
 * @param shm_name shared memory name
 * @param size size of the memory to write
 * @param verbose activate verbose mode to print out messages
 * @return bool true if the operation was success, otherwise false
 */
template <typename I, typename T>
bool IPC::write_buffer(I data, size_t size, const char *shm_name, bool verbose)
{
    /* File descriptor of the shared memory */
    int fd_buffer;
    /* Shared memory address for the buffer */
    T *buffer;

    /* Get shared memory */
    fd_buffer = shm_open(shm_name, O_RDWR, 0);

    if (fd_buffer < 0)
    {
        std::cout << "The device '" << shm_name << "' could not open" << std::endl;

        return false;
    }

    /* Ask for memory */
    buffer = (T *)mmap(NULL, sizeof(T) * size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_buffer, 0);

    if (MAP_FAILED == buffer)
    {
        close(fd_buffer);
        std::cout << "The mapping of the '" << shm_name << "' could not be done" << std::endl;

        return false;
    }

    /* Copy data to the buffers */
    for (int i = 0; i < size; ++i)
    {
        buffer[i] = data[i];
    }

    munmap(buffer, sizeof(T) * size);
    shm_unlink(shm_name);
    close(fd_buffer);

    if (verbose)
        std::cout << "Shared memory for buffer '" << shm_name << "' was unmapped" << std::endl;

    return true;
}
