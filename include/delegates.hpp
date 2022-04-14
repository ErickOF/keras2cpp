#ifndef KERAS2CPP_DELEGATES_HPP
#define KERAS2CPP_DELEGATES_HPP

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace keras
{
  typedef struct delegate_enabler
  {
    bool conv2d;
    bool fully_connected;
    bool softmax;
  } DelegateEnabler;

  class Delegate;
  // class DelegateConv2D;
  // class DelegateFullyConnected;
  class DelegateSoftmax;
}

class keras::Delegate
{
public:
  Delegate(std::string name, bool verbose) : m_name(name), m_verbose(verbose) {}
  Delegate(std::string name) : m_name(name), m_verbose(false) {}
  Delegate() : m_name("Default Delegate"), m_verbose(false) {}

  virtual std::vector<float> eval(std::vector<float> input) const = 0;

  std::string get_name()
  {
    return m_name;
  }

protected:
  std::string m_name;
  bool m_verbose;
};

class keras::DelegateSoftmax : public Delegate
{
public:
  DelegateSoftmax(bool verbose) : Delegate("Softmax", verbose) {}
  DelegateSoftmax() : Delegate("Softmax") {}
  ~DelegateSoftmax() {}

  std::vector<float> eval(std::vector<float> input) const override
  {
    if (m_verbose)
      std::cout << m_name << " delegate running..." << std::endl;

    std::vector<float> output(input.size());
    float sum = 0.0f;

    for (int i = 0; i < input.size(); ++i)
    {
      output[i] = exp(input[i]);
      sum += output[i];
    }

    for (int i = 0; i < input.size(); ++i)
    {
      output[i] /= sum;
    }

    return output;
  }
};

#endif /* KERAS2CPP_DELEGATES_HPP */
