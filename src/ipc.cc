#include <ipc.hpp>


/**
 * @brief Call back-end to execute softmax function
 *
 * @param input data to apply softmax
 * @param verbose activate verbose mode to print out messages
 * @return std::vector<float> 
 */
std::vector<float> apply_softmax(std::vector<float> input, bool verbose)
{
    std::vector<float> output(input.size());

    /* Shared memory for IPC */
    axc_shared_mem_t* address;
    /* Semaphores to lock memory R/W operations */
    sem_t* mutex_sem;
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
    address = (axc_shared_mem_t*) mmap(NULL, sizeof(axc_shared_mem_t),
        PROT_READ | PROT_WRITE, MAP_SHARED, fd_shm, 0);

    if (MAP_FAILED == address)
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

    std::cout << "Semaphores are ready!" << std::endl;

    /* Take control of the shared memory to allocate memory */
    sem_wait(mutex_sem);

    if (verbose)
    {
        std::cout << "Default shared memory" << std::endl;
        shared_memory_verbose(address);
    }

    /* Ask for buffer */
    address->op1_size = input.size();
    address->op2_size = 0;
    address->out_size = input.size();
    address->request = AXC_ALLOCATE;

    if (verbose)
    {
        std::cout << "Softmax request: Allocate memory for buffers" << std::endl;
        shared_memory_verbose(address);
    }

    /* Release semaphore */
    sem_post(mutex_sem);

    /* Take control of the shared memory */
    sem_wait(mutex_sem);


    if (verbose)
    {
        std::cout << "Driver response: Allocate memory for buffers" << std::endl;
        shared_memory_verbose(address);
    }

    /* Release semaphore */
    sem_post(mutex_sem);

    munmap(address, sizeof(axc_shared_mem_t));
    close(fd_shm);

    std::cout << "Shared memory was unmapped" << std::endl;

    return output;
}

/**
 * @brief Print out shared memory data when verbose mode is on
 * 
 * @param shmem shared memory to print
 */
void shared_memory_verbose(axc_shared_mem_t* shmem)
{
    std::cout << "Shared memory values: " << std::endl;
    std::cout << "op1: " << shmem->op1 << std::endl;
    std::cout << "op1 size: " << shmem->op1_size << std::endl;
    std::cout << "op2: " << shmem->op2 << std::endl;
    std::cout << "op2 size: " << shmem->op2_size << std::endl;
    std::cout << "out: " << shmem->output << std::endl;
    std::cout << "out size: " << shmem->out_size << std::endl;
    std::cout << "request: " << shmem->request << std::endl;
    std::cout << "request status: " << shmem->status << std::endl;
}
