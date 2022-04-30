#include <delegates.hpp>

std::vector<float> keras::DelegateConv2D::eval(std::vector<float> input, std::vector<float> kernel, int k_size, int stride)
{
    std::vector<float> output;

    return output;
}

std::vector<float> keras::DelegateSoftmax::eval(std::vector<float> input)
{
    /** Check if it's in verbose mode **/
    if (m_verbose)
        std::cout << m_name << " delegate running..." << std::endl;

    /* Shared memory for IPC */
    axc_shared_mem_t* address;
    /* Semaphores to lock memory R/W operations */
    sem_t* mutex_sem;
    /* Number of available buffers */
    sem_t* buffer_count_sem;
    /* Data is ready to read */
    sem_t* ready_sem;
    /* File descriptor of the shared memory */
    int fd_shm;

    /* Mutual exclusion semaphore */
    mutex_sem = sem_open(SEM_MUTEX_NAME, 0, 0, 0);

    if (SEM_FAILED == mutex_sem)
    {
        std::cout << "Mutex semaphore open failed" << std::endl;
        exit(-1);
    }

    /* Get shared memory */
    fd_shm = shm_open(SHARED_MEM_NAME, O_RDWR, 0);

    if (fd_shm < 0)
    {
        std::cout << "The device '" << SHARED_MEM_NAME << "' could not open" << std::endl;
        std::cout << errno << std::endl;
        exit(-1);
    }

    /* Ask for memory */
    address = (axc_shared_mem_t*) mmap(NULL, sizeof(axc_shared_mem_t),
        PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0);

    if (MAP_FAILED == address)
    {
        std::cout << "The mapping could not be created" << std::endl;
        exit(-1);
    }

    std::cout << "Shared memory was mapped" << std::endl;

    /** Counting semaphore, indicating the number of buffers **/
    buffer_count_sem = sem_open(SEM_BUFFER_COUNT_NAME, 0, 0, 0);

    if (SEM_FAILED == buffer_count_sem)
    {
        std::cout << "Mutex buffer count open failed" << std::endl;
        exit(-1);
    }

    /** Counting semaphore, indicating the number of buffers **/
    ready_sem = sem_open(SEM_READY_SIGNAL_NAME, 0, 0, 0);

    if (SEM_FAILED == ready_sem)
    {
        std::cout << "Ready sem open failed" << std::endl;
        exit(-1);
    }

    std::cout << "Semaphores are ready!" << std::endl;

    if (-1 == sem_post(ready_sem))
    {
        std::cout << "Ready sem cannot be released" << std::endl;
        exit(-1);
    }

    std::cout << "Ready sem was released" << std::endl;

    munmap(address, sizeof(axc_shared_mem_t));
    std::cout << "Shared memory was unmapped" << std::endl;;


    /** To store the output **/
    std::vector<float> output(input.size());
    /** To store the sum **/
    /*float sum = 0.0f;

    /** Compute exp(x) and sum_reduce
    for (int i = 0; i < input.size(); ++i)
    {
        output[i] = exp(input[i]);
        sum += output[i];
    }

    /** Compute output
    for (int i = 0; i < input.size(); ++i)
    {
        output[i] /= sum;
    }*/

    return output;
}
