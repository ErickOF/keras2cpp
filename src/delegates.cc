#include <delegates.hpp>

/**
 * @brief Construct a new Delegate Conv2D object
 *
 * @param verbose enable or disable verbose mode
 */
keras::DelegateConv2D::DelegateConv2D(bool verbose) : Delegate("Conv2D", verbose)
{
    IPC ipc;
    m_accel = ipc.exists_accel(AXC_CONV2D);
}

/**
 * @brief Default constructor for a new Delegate Conv2D object
 *
 */
keras::DelegateConv2D::DelegateConv2D() : Delegate("Conv2D", false)
{
    IPC ipc;
    m_accel = ipc.exists_accel(AXC_CONV2D);
}

/**
 * @brief Construct a new Delegate FullyConnected object
 *
 * @param verbose enable or disable verbose mode
 */
keras::DelegateFullyConnected::DelegateFullyConnected(bool verbose) : Delegate("FullyConnected", verbose)
{
    IPC ipc;
    m_accel = ipc.exists_accel(AXC_FULLY_CONNECTED);
}

/**
 * @brief Default constructor for a new Delegate FullyConnected object
 *
 */
keras::DelegateFullyConnected::DelegateFullyConnected() : Delegate("FullyConnected", false)
{
    IPC ipc;
    m_accel = ipc.exists_accel(AXC_FULLY_CONNECTED);
}

/**
 * @brief Construct a new Delegate Softmax object
 *
 * @param verbose enable or disable verbose mode
 */
keras::DelegateSoftmax::DelegateSoftmax(bool verbose) : Delegate("Softmax", verbose)
{
    IPC ipc;
    m_accel = ipc.exists_accel(AXC_SOFTMAX);
}
/**
 * @brief Default constructor for a new Delegate Softmax object
 *
 */
keras::DelegateSoftmax::DelegateSoftmax() : Delegate("Softmax")
{
    IPC ipc;
    m_accel = ipc.exists_accel(AXC_SOFTMAX);
}

/**
 * @brief Evaluate convolution 2D
 *
 * @param input input data to apply convolution
 * @param kernels kernels to use
 * @param output result of convolution 2D
 * @param params convolution parameters
 */
void keras::DelegateConv2D::eval(
    const std::vector<std::vector<std::vector<float>>> input,
    std::vector<std::vector<std::vector<std::vector<float>>>> kernels,
    std::vector<std::vector<std::vector<float>>> output,
    axc_delegate_conv_params_t *params)
{
    /* Check if it's in verbose mode */
    if (m_verbose)
        std::cout << m_name << " delegate running..." << std::endl;

    std::vector<float> flat_input, flat_kernels, flat_output;

    /* Flat input values */
    /* Channels */
    for (int k = 0; k < input.size(); ++k)
        /* Rows */
        for (int i = 0; i < input[0].size(); ++i)
            /* Cols */
            for (int j = 0; j < input[0][0].size(); ++j)
                flat_input.push_back(input[k][i][j]);

    /* Flat kernels values */
    /* Number of kernels */
    for (int k = 0; k < kernels.size(); ++k)
        /* Channels */
        for (int l = 0; l < kernels[0].size(); ++l)
            /* Kernel size {x, y} */
            for (int i = 0; i < kernels[0][0].size(); ++i)
                for (int j = 0; j < kernels[0][0][0].size(); ++j)
                    flat_kernels.push_back(kernels[k][l][i][j]);

    /* Calling driver to get the output */
    IPC ipc(m_verbose);
    flat_output = ipc.apply_conv2d(flat_input, flat_kernels, params);
}

/**
 * @brief Evaluate fully connected layer
 *
 * @param input input data to apply convolution
 * @param weights weights to use
 * @param output result of the matrix-matrix multiplication
 * @param params fully connected params
 */
void keras::DelegateFullyConnected::eval(const std::vector<float> input,
                                         std::vector<std::vector<float>> weights,
                                         float *output,
                                         axc_delegate_fully_connected_params_t *params)
{
    /* Check if it's in verbose mode */
    if (m_verbose)
        std::cout << m_name << " delegate running..." << std::endl;

    /* Flat weights */
    std::vector<float> flat_weights;

    /* Rows */
    for (int i = 0; i < weights.size(); ++i)
        /* Cols */
        for (int j = 0; j < weights[0].size(); ++j)
            flat_weights.push_back(weights[i][j]);

    IPC ipc(m_verbose);
    std::vector<float> out_vector = ipc.apply_fully_connected(input, flat_weights, params);
}

/**
 * @brief Performs the softmax function delegated to the hardware
 * accelerator.
 *
 * y = exp(x) / sum_reduce(exp(x))
 *
 * @param input vector to perform softmax activation function
 * @return std::vector<float> result of softmax function
 */
std::vector<float> keras::DelegateSoftmax::eval(std::vector<float> input)
{
    /* Check if it's in verbose mode */
    if (m_verbose)
        std::cout << m_name << " delegate running..." << std::endl;

    /* Calling driver to get the output */
    IPC ipc(m_verbose);

    return ipc.apply_softmax(input);
}
