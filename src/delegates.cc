#include <delegates.hpp>

/**
 * @brief Evaluate convolution 2D
 *
 * @param input input data to apply convolution
 * @param kernels kernels to use
 * @param output result of convolution 2D
 * @param padding padding type
 * @param stride stride value
 */
void keras::DelegateConv2D::eval(
    const std::vector<std::vector<std::vector<float>>> input,
    std::vector<std::vector<std::vector<std::vector<float>>>> kernels,
    std::vector<std::vector<std::vector<float>>> output,
    conv_padding_t padding, int stride)
{
    /* Check if it's in verbose mode */
    if (m_verbose)
        std::cout << m_name << " delegate running..." << std::endl;

    std::vector<float> flat_input, flat_kernels, flat_output;
    int k_size = kernels[0][0].size();

    /* Flat input values */
    flat_input.push_back(0.0f);

    /* Flat kernels values */
    flat_kernels.push_back(0.0f);

    /* Calling driver to get the output */
    flat_output = apply_conv2d(flat_input, flat_kernels, k_size, padding, stride, m_verbose);
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
