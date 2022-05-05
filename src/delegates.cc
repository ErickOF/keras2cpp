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


    /** To store the output **/
    std::vector<float> output = apply_softmax(input);

    return output;
}
