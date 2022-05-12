#include <ipc.hpp>

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
    bool verbose)
{
    /* Shared memory for IPC */
    axc_shared_mem_t *shared_mem;
    /* Semaphores to lock memory R/W operations */
    sem_t *mutex_sem;
    /* File descriptor of the shared memory */
    int fd_shm;

    /* Get shared memory */
    fd_shm = shm_open(SHARED_MEM_NAME, O_RDWR, 0);

    if (fd_shm < 0)
    {
        close(fd_shm);
        std::cout << "The device '" << SHARED_MEM_NAME << "' could not open" << std::endl;
        std::cout << errno << std::endl;
        exit(-1);
    }

    /* Ask for memory */
    shared_mem = (axc_shared_mem_t *)mmap(NULL, sizeof(axc_shared_mem_t),
                                          PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0);

    if (MAP_FAILED == shared_mem)
    {
        std::cout << "The mapping could not be done" << std::endl;
        exit(-1);
    }

    /* Mutex sem for shared memory */
    mutex_sem = sem_open(SEM_READY_SIGNAL_NAME, 0, 0, 0);

    if (SEM_FAILED == mutex_sem)
    {
        std::cout << "Ready sem open failed" << std::endl;
        exit(-1);
    }

    if (verbose)
        std::cout << "Semaphores are ready!" << std::endl;

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

    while (AXC_IDLE != shared_mem->request)
    {
        sem_post(mutex_sem);
        /* Wait 1 us */
        usleep(1);
        sem_wait(mutex_sem);
    }

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

        munmap(shared_mem, sizeof(axc_shared_mem_t));
        close(fd_shm);

        std::cout << std::endl
                  << "Buffer allocation was not successful" << std::endl;

        exit(-1);
    }

    if (verbose)
        std::cout << std::endl
                  << "Writting in the buffers" << std::endl;

    /* Copy data to the buffers */
    write_buffer<std::vector<float>, float>(input, input.size(), SHARED_MEM_OP1_NAME, verbose);
    write_buffer<std::vector<float>, float>(kernels, kernels.size(), SHARED_MEM_OP2_NAME, verbose);
    write_buffer<char *, char>((char *)params, sizeof(axc_delegate_conv_params_t), SHARED_MEM_PARAMS_NAME, verbose);

    if (verbose)
        std::cout << "Conv2D request: Execute" << std::endl;

    shared_mem->request = AXC_EXECUTE;
    shared_mem->operation = AXC_CONV2D;

    while (AXC_IDLE != shared_mem->request)
    {
        sem_post(mutex_sem);
        /* Wait 1 us */
        usleep(1);
        sem_wait(mutex_sem);
    }

    /* Get the output of the sofmax function */
    std::vector<float> output = read_buffer(SHARED_MEM_OUT_NAME, shared_mem->out_size, verbose);

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Buffers were read" << std::endl;
        shared_memory_verbose(shared_mem);

        std::cout << "Input/Output" << std::endl;

        for (int i = 0; i < shared_mem->op1_size; ++i)
        {
            std::cout << i << ": " << input[i] << "/" << output[i] << std::endl;
        }

        std::cout << std::endl
                  << "Softmax request: Deallocate buffers" << std::endl;
    }

    shared_mem->request = AXC_DEALLOCATE;

    while (AXC_IDLE != shared_mem->request)
    {
        sem_post(mutex_sem);
        /* Wait 1 us */
        usleep(1);
        sem_wait(mutex_sem);
    }

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Dellocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    /* Release semaphore */
    sem_post(mutex_sem);

    /* Release memory */
    munmap(shared_mem, sizeof(axc_shared_mem_t));
    close(fd_shm);

    if (verbose)
        std::cout << std::endl
                  << "Shared memory was unmapped" << std::endl;

    return output;
}

/**
 * @brief Call back-end to execute softmax function
 *
 * @param input data to apply softmax
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float>
 */
std::vector<float> apply_softmax(std::vector<float> input, bool verbose)
{
    /* Shared memory for IPC */
    axc_shared_mem_t *shared_mem;
    /* Semaphores to lock memory R/W operations */
    sem_t *mutex_sem;
    /* File descriptor of the shared memory */
    int fd_shm;

    /* Get shared memory */
    fd_shm = shm_open(SHARED_MEM_NAME, O_RDWR, 0);

    if (fd_shm < 0)
    {
        close(fd_shm);
        std::cout << "The device '" << SHARED_MEM_NAME << "' could not open" << std::endl;
        std::cout << errno << std::endl;
        exit(-1);
    }

    /* Ask for memory */
    shared_mem = (axc_shared_mem_t *)mmap(NULL, sizeof(axc_shared_mem_t),
                                          PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0);

    if (MAP_FAILED == shared_mem)
    {
        std::cout << "The mapping could not be done" << std::endl;
        exit(-1);
    }

    /* Mutex sem for shared memory */
    mutex_sem = sem_open(SEM_READY_SIGNAL_NAME, 0, 0, 0);

    if (SEM_FAILED == mutex_sem)
    {
        std::cout << "Ready sem open failed" << std::endl;
        exit(-1);
    }

    if (verbose)
        std::cout << "Semaphores are ready!" << std::endl;

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

    while (AXC_IDLE != shared_mem->request)
    {
        sem_post(mutex_sem);
        /* Wait 1 us */
        usleep(1);
        sem_wait(mutex_sem);
    }

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

        munmap(shared_mem, sizeof(axc_shared_mem_t));
        close(fd_shm);

        std::cout << std::endl
                  << "Buffer allocation was not successful" << std::endl;

        exit(-1);
    }

    if (verbose)
        std::cout << std::endl
                  << "Writting in the buffers" << std::endl;

    /* Copy data to the buffers */
    write_buffer<std::vector<float>, float>(input, input.size(), SHARED_MEM_OP1_NAME, verbose);

    if (verbose)
        std::cout << "Softmax request: Execute" << std::endl;

    shared_mem->request = AXC_EXECUTE;
    shared_mem->operation = AXC_SOFTMAX;

    while (AXC_IDLE != shared_mem->request)
    {
        sem_post(mutex_sem);
        /* Wait 1 us */
        usleep(1);
        sem_wait(mutex_sem);
    }

    /* Get the output of the sofmax function */
    std::vector<float> output = read_buffer(SHARED_MEM_OUT_NAME, shared_mem->out_size, verbose);

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Buffers were read" << std::endl;
        shared_memory_verbose(shared_mem);

        std::cout << "Input/Output" << std::endl;

        for (int i = 0; i < shared_mem->op1_size; ++i)
        {
            std::cout << i << ": " << input[i] << "/" << output[i] << std::endl;
        }

        std::cout << std::endl
                  << "Softmax request: Deallocate buffers" << std::endl;
    }

    shared_mem->request = AXC_DEALLOCATE;

    while (AXC_IDLE != shared_mem->request)
    {
        sem_post(mutex_sem);
        /* Wait 1 us */
        usleep(1);
        sem_wait(mutex_sem);
    }

    if (verbose)
    {
        std::cout << std::endl
                  << "Driver response: Dellocate memory for buffers" << std::endl;
        shared_memory_verbose(shared_mem);
    }

    /* Release semaphore */
    sem_post(mutex_sem);

    /* Release memory */
    munmap(shared_mem, sizeof(axc_shared_mem_t));
    close(fd_shm);

    return output;
}

/**
 * @brief Read data from a specific buffer
 *
 * @param shm_name shered memory name
 * @param size size of the buffer to read
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float> values read from the buffer
 */
std::vector<float> read_buffer(const char *shm_name, int size, bool verbose)
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
    {
        if (verbose)
            std::cout << "Value: " << i + 1 << "/" << size << std::endl;

        output[i] = buffer[i];
    }

    munmap(buffer, sizeof(float) * size);
    shm_unlink(shm_name);
    close(fd_buffer);

    return output;
}

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
bool write_buffer(I data, size_t size, const char *shm_name, bool verbose)
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
        if (verbose)
            std::cout << "Value[" << i + 1 << "/" << size << "] = " << data[i] << std::endl;

        buffer[i] = data[i];
    }

    munmap(buffer, sizeof(T) * size);
    shm_unlink(shm_name);
    close(fd_buffer);

    if (verbose)
        std::cout << "Shared memory for buffer '" << shm_name << "' was unmapped" << std::endl;

    return true;
}

/**
 * @brief Print out shared memory data when verbose mode is on
 *
 * @param shmem shared memory to print
 */
void shared_memory_verbose(axc_shared_mem_t *shmem)
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
