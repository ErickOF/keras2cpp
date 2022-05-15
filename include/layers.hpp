#ifndef KERAS2CPP_LAYERS_HPP
#define KERAS2CPP_LAYERS_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <datachunk.hpp>
#include <delegates.hpp>

namespace keras
{
  void missing_activation_impl(const std::string &act);
  std::vector<std::vector<float>> conv_single_depth_valid(std::vector<std::vector<float>> const &im, std::vector<std::vector<float>> const &k);
  std::vector<std::vector<float>> conv_single_depth_same(std::vector<std::vector<float>> const &im, std::vector<std::vector<float>> const &k);

  class Layer;
  class LayerActivation;
  class LayerFlatten;
  class LayerMaxPooling;
  class LayerConv2D;
  class LayerDense;
  class LayerInput;
}

class keras::Layer
{
public:
  virtual void load_weights(std::ifstream &fin, DelegateEnabler &enabler) = 0;
  virtual keras::DataChunk *compute_output(keras::DataChunk *) = 0;

  Layer(std::string name) : m_name(name), m_verbose(false) {}
  Layer(std::string name, bool verbose) : m_name(name), m_verbose(verbose) {}

  virtual unsigned int get_input_rows() const = 0;
  virtual unsigned int get_input_cols() const = 0;
  virtual unsigned int get_output_units() const = 0;

  std::string get_name()
  {
    return m_name;
  }

protected:
  std::string m_name;
  bool m_verbose;
};

class keras::LayerActivation : public Layer
{
public:
  LayerActivation(bool verbose) : Layer("Activation", verbose) {
    m_delegate = nullptr;
  }

  LayerActivation() : Layer("Activation") {
    m_delegate = nullptr;
  }

  ~LayerActivation();

  void load_weights(std::ifstream &fin, DelegateEnabler &enabler);
  keras::DataChunk *compute_output(keras::DataChunk *);

  unsigned int get_input_rows() const override
  {
    return 0;
  } // look for the value in the preceding layer

  unsigned int get_input_cols() const override
  {
    return 0;
  } // same as for rows

  unsigned int get_output_units() const override
  {
    return 0;
  }

private:
  std::string m_activation_type;
  DelegateSoftmax *m_delegate;
};

class keras::LayerFlatten : public Layer
{
public:
  LayerFlatten(bool verbose) : Layer("Flatten", verbose) {}
  LayerFlatten() : Layer("Flatten") {}

  void load_weights(std::ifstream &fin, DelegateEnabler &enabler){};
  keras::DataChunk *compute_output(keras::DataChunk *);

  unsigned int get_input_rows() const override
  {
    return 0;
  } // look for the value in the preceding layer

  unsigned int get_input_cols() const override
  {
    return 0;
  } // same as for rows

  unsigned int get_output_units() const override
  {
    return 0;
  }
};

class keras::LayerMaxPooling : public Layer
{
public:
  LayerMaxPooling(bool verbose) : Layer("MaxPooling2D", verbose) {}

  LayerMaxPooling() : Layer("MaxPooling2D") {}

  void load_weights(std::ifstream &fin, DelegateEnabler &enabler);
  keras::DataChunk *compute_output(keras::DataChunk *);

  virtual unsigned int get_input_rows() const
  {
    return 0;
  } // look for the value in the preceding layer

  virtual unsigned int get_input_cols() const
  {
    return 0;
  } // same as for rows

  virtual unsigned int get_output_units() const
  {
    return 0;
  }

  int m_pool_x;
  int m_pool_y;
};

class keras::LayerConv2D : public Layer
{
public:
  LayerConv2D(bool verbose) : Layer("Conv2D")
  {
    this->m_verbose = verbose;
    this->m_delegate = nullptr;
  }

  LayerConv2D() : Layer("Conv2D") {
    this->m_delegate = nullptr;
  }

  void load_weights(std::ifstream &fin, DelegateEnabler &enabler);
  keras::DataChunk *compute_output(keras::DataChunk *);
  std::vector<std::vector<std::vector<std::vector<float>>>> m_kernels; // kernel, depth, rows, cols
  std::vector<float> m_bias;                                           // kernel

  virtual unsigned int get_input_rows() const
  {
    return m_rows;
  }

  virtual unsigned int get_input_cols() const
  {
    return m_cols;
  }

  virtual unsigned int get_output_units() const
  {
    return m_kernels_cnt;
  }

  std::string m_border_mode;
  int m_kernels_cnt;
  int m_depth;
  int m_rows;
  int m_cols;
  DelegateConv2D *m_delegate;
};

class keras::LayerDense : public Layer
{
public:
  LayerDense(bool verbose) : Layer("Dense", verbose) {
    m_delegate = nullptr;
  }

  LayerDense() : Layer("Dense") {
    m_delegate = nullptr;
  }

  void load_weights(std::ifstream &fin, DelegateEnabler &enabler);
  keras::DataChunk *compute_output(keras::DataChunk *);
  std::vector<std::vector<float>> m_weights; // input, neuron
  std::vector<float> m_bias;                 // neuron

  virtual unsigned int get_input_rows() const
  {
    return 1;
  } // flat, just one row

  virtual unsigned int get_input_cols() const
  {
    return m_input_cnt;
  }

  virtual unsigned int get_output_units() const
  {
    return m_neurons;
  }

  int m_input_cnt;
  int m_neurons;
  DelegateFullyConnected *m_delegate;
};

class keras::LayerInput : public Layer
{
public:
  LayerInput() : Layer("Input") {}
  void load_weights(std::ifstream &fin, DelegateEnabler &enabler);
  keras::DataChunk *compute_output(keras::DataChunk *);
  std::vector<std::vector<float>> m_weights;

  virtual unsigned int get_input_rows() const
  {
    return 1;
  } // flat, just one row

  virtual unsigned int get_input_cols() const
  {
    return m_input_cnt;
  }

  virtual unsigned int get_output_units() const
  {
    return m_neurons;
  }

  int m_input_cnt;
  int m_neurons;
};

#endif /* KERAS2CPP_LAYERS_HPP */
