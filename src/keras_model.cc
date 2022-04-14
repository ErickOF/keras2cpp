#include <keras_model.hpp>

keras::KerasModel::KerasModel(const std::string &input_fname, bool verbose, keras::DelegateEnabler &enabler)
    : m_verbose(verbose)
{
  load_weights(input_fname, enabler);
}

keras::KerasModel::KerasModel(const std::string &input_fname, bool verbose)
    : m_verbose(verbose)
{
  keras::DelegateEnabler enabler;
  load_weights(input_fname, enabler);
}

std::vector<float> keras::KerasModel::compute_output(keras::DataChunk *dc)
{
  if (this->m_verbose)
  {
    std::cout << std::endl << "KerasModel compute output" << std::endl;
    std::cout << "Input data size:" << std::endl;
    dc->show_name();
  }

  keras::DataChunk *inp = dc;
  keras::DataChunk *out = 0;

  for (int l = 0; l < (int)m_layers.size(); ++l)
  {
    if (this->m_verbose)
      std::cout << "Processing layer " << m_layers[l]->get_name() << std::endl;

    out = m_layers[l]->compute_output(inp);

    if (this->m_verbose)
    {
      std::cout << "Input" << std::endl;
      inp->show_name();

      std::cout << "Output" << std::endl;
      out->show_name();

      std::cout << std::endl;
    }

    if (inp != dc)
      delete inp;

    // delete inp;
    inp = 0L;
    inp = out;
  }

  std::vector<float> flat_out = out->get_1d();

  // if (this->m_verbose)
  //   out->show_values();

  delete out;

  return flat_out;
}

void keras::KerasModel::load_weights(const std::string &input_fname, keras::DelegateEnabler &enabler)
{
  std::cout << "Reading model from " << input_fname << std::endl;

  std::ifstream fin(input_fname.c_str());
  std::string layer_type = "";
  std::string tmp_str = "";
  int tmp_int = 0;

  fin >> tmp_str >> m_layers_cnt;

  bool reading = true;
  int layer = 0;

  while (reading)
  { // iterate over layers
    fin >> tmp_str >> tmp_int >> layer_type;

    std::cout << "Layer " << tmp_int << " " << layer_type << std::endl;

    Layer *l = 0L;

    if (layer_type == "Conv2D")
    {
      l = new LayerConv2D(m_verbose);
    }
    else if (layer_type == "Activation")
    {
      l = new LayerActivation(m_verbose);
    }
    else if (layer_type == "MaxPooling2D")
    {
      l = new LayerMaxPooling(m_verbose);
    }
    else if (layer_type == "Flatten")
    {
      l = new LayerFlatten(m_verbose);
    }
    else if (layer_type == "Dense")
    {
      l = new LayerDense(m_verbose);
    }
    else if (layer_type == "InputLayer")
    {
      continue;
    }
    else if (layer_type == "Dropout")
    {
      continue; // we dont need dropout layer in prediciton mode
    }
    else
    {
      reading = false;
      continue;
    }

    if (l == 0L)
    {
      std::cout << "Layer is empty, maybe it is not defined? Cannot define network." << std::endl;
      throw "Layer is empty, maybe it is not defined? Cannot define network.\n";
    }

    l->load_weights(fin, enabler);
    m_layers.push_back(l);
    layer++;

    std::cout << std::endl;
  }

  fin.close();
}

keras::KerasModel::~KerasModel()
{
  for (int i = 0; i < (int)m_layers.size(); ++i)
  {
    delete m_layers[i];
  }
}

int keras::KerasModel::get_output_length() const
{
  int i = m_layers.size() - 1;

  while ((i > 0) && (m_layers[i]->get_output_units() == 0))
    --i;

  return m_layers[i]->get_output_units();
}
