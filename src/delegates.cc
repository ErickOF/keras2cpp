#include <delegates.hpp>

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
    int k_size = kernels[0][0].size();

    /* Flat input values */
    /* Channels */
    for(int k = 0; k < input.size(); ++k)
        /* Rows */
        for(int i = 0; i < input[0].size(); ++i)
            /* Cols */
            for(int j = 0; j < input[0][0].size(); ++j)
                flat_input.push_back(input[k][i][j]);

    /* Flat kernels values */
    /* Number of kernels */
    for(int k = 0; k < kernels.size(); ++k)
        /* Channels */
        for(int l = 0; l < kernels[0].size(); ++l)
            /* Kernel size {x, y} */
            for(int i = 0; i < kernels[0][0].size(); ++i)
                for(int j = 0; j < kernels[0][0][0].size(); ++j)
                    flat_kernels.push_back(kernels[k][l][i][j]);

    /* Calling driver to get the output */
    flat_output = apply_conv2d(flat_input, flat_kernels, params, m_verbose);
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
    return apply_softmax(input, m_verbose);
}
