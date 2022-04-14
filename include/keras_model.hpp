#ifndef KERAS2CPP_KERAS_MODEL_H
#define KERAS2CPP_KERAS_MODEL_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <delegates.hpp>
#include <layers.hpp>

namespace keras
{
  class KerasModel;
}

class keras::KerasModel
{
public:
  KerasModel(const std::string &input_fname, bool verbose, keras::DelegateEnabler &enabler);
  KerasModel(const std::string &input_fname, bool verbose);
  ~KerasModel();
  std::vector<float> compute_output(keras::DataChunk *dc);

  unsigned int get_input_rows() const
  {
    return m_layers.front()->get_input_rows();
  }

  unsigned int get_input_cols() const
  {
    return m_layers.front()->get_input_cols();
  }

  int get_output_length() const;

private:
  void load_weights(const std::string &input_fname, keras::DelegateEnabler &enabler);
  int m_layers_cnt;              // number of layers
  std::vector<keras::Layer *> m_layers; // container with layers
  bool m_verbose;
};

#endif /* KERAS2CPP_KERAS_MODEL_H */
