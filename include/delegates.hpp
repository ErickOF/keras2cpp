#ifndef KERAS2CPP_DELEGATES_HPP
#define KERAS2CPP_DELEGATES_HPP

#include <cmath>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ipc.hpp>


namespace keras
{
  typedef struct delegate_enabler
  {
    bool conv2d;
    bool fully_connected;
    bool softmax;
  } DelegateEnabler;

  class Delegate;
  class DelegateConv2D;
  // class DelegateFullyConnected;
  class DelegateSoftmax;
}

class keras::Delegate
{
public:
  /**
   * @brief Construct a new Delegate object
   *
   * @param name delegate's name
   * @param verbose enable or disable verbose mode
   */
  Delegate(std::string name, bool verbose) : m_name(name), m_verbose(verbose) {}
  /**
   * @brief Construct a new Delegate object
   *
   * @param name delegate's name
   */
  Delegate(std::string name) : m_name(name), m_verbose(false) {}
  /**
   * @brief Default constructor for a new Delegate object
   *
   */
  Delegate() : m_name("Default Delegate"), m_verbose(false) {}

  /**
   * @brief Get the delegate's name
   *
   * @return std::string delegate's name
   */
  std::string get_name()
  {
    return m_name;
  }

protected:
  /**
   * @brief Delegate's name
   *
   */
  std::string m_name;
  /**
   * @brief Verbose mode
   *
   */
  bool m_verbose;
};


class keras::DelegateConv2D : public Delegate
{
public:
  /**
   * @brief Construct a new Delegate Conv2D object
   * 
   * @param verbose enable or disable verbose mode
   */
  DelegateConv2D(bool verbose) : Delegate("Conv2D", verbose) {}
  /**
   * @brief Default constructor for a new Delegate Conv2D object
   * 
   */
  DelegateConv2D() : Delegate("Conv2D", false) {}
  /**
   * @brief Destroy the Delegate Conv2D object
   * 
   */
  ~DelegateConv2D() {}

  std::vector<float> eval(std::vector<float> input, std::vector<float> kernel, int k_size, int stride);
};

class keras::DelegateSoftmax : public Delegate
{
public:
  /**
   * @brief Construct a new Delegate Softmax object
   *
   * @param verbose enable or disable verbose mode
   */
  DelegateSoftmax(bool verbose) : Delegate("Softmax", verbose) {}
  /**
   * @brief Default constructor for a new Delegate Softmax object
   *
   */
  DelegateSoftmax() : Delegate("Softmax") {}
  /**
   * @brief Destroy the Delegate Softmax object
   *
   */
  ~DelegateSoftmax() {}

  /**
   * @brief Performs the softmax function delegated to the hardware
   * accelerator.
   *
   * y = exp(x) / sum_reduce(exp(x))
   *
   * @param input vector to perform softmax activation function
   * @return std::vector<float> result of the softmax
   */
  std::vector<float> eval(std::vector<float> input);
};

#endif /* KERAS2CPP_DELEGATES_HPP */
