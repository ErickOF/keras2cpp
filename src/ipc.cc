#include <ipc.hpp>

/**
 * @brief Call back-end to execute softmax function
 *
 * @param input data to apply softmax
 * @return std::vector<float> 
 */
std::vector<float> apply_softmax(std::vector<float> input)
{
    std::vector<float> output(input.size());

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
        std::cout << "'" << SEM_MUTEX_NAME << "' semaphore open failed" << std::endl;
        exit(-1);
    }

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
    address = (axc_shared_mem_t*) mmap(NULL, sizeof(axc_shared_mem_t),
        PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0);

    if (MAP_FAILED == address)
    {
        std::cout << "The mapping could not be done" << std::endl;
        exit(-1);
    }

    std::cout << "Shared memory was mapped" << std::endl;
    std::cout << "Shared memory values:" << std::endl;
    std::cout << "op1:" << address->op1 << std::endl;
    std::cout << "op1 size:" << address->op1_size << std::endl;
    std::cout << "op2:" << address->op2 << std::endl;
    std::cout << "op2 size:" << address->op2_size << std::endl;
    std::cout << "out:" << address->output << std::endl;
    std::cout << "out size:" << address->out_size << std::endl;

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
    close(fd_shm);

    std::cout << "Shared memory was unmapped" << std::endl;

    return output;
}
