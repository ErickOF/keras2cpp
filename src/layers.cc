#include <layers.hpp>

void keras::missing_activation_impl(const std::string &act)
{
  std::cout << "Activation " << act << " not defined!" << std::endl;
  std::cout << "Please add its implementation before use." << std::endl;

  exit(1);
}

// with border mode = valid
std::vector<std::vector<float>> keras::conv_single_depth_valid(
    std::vector<std::vector<float>> const &im,
    std::vector<std::vector<float>> const &k)
{
  size_t k1_size = k.size(), k2_size = k[0].size();
  std::cout << "Kernel: " << k1_size << "x" << k2_size << std::endl;

  unsigned int st_x = (k1_size - 1) >> 1;
  unsigned int st_y = (k2_size - 1) >> 1;

  std::vector<std::vector<float>> y(im.size() - 2 * st_x, std::vector<float>(im[0].size() - 2 * st_y, 0));

  for (unsigned int i = st_x; i < im.size() - st_x; ++i)
  {
    for (unsigned int j = st_y; j < im[0].size() - st_y; ++j)
    {
      float sum = 0.0f;

      for (unsigned int k1 = 0; k1 < k.size(); ++k1)
      {
        // const float * k_data = k[k1_size-k1-1].data();
        // const float * im_data = im[i-st_x+k1].data();
        for (unsigned int k2 = 0; k2 < k[0].size(); ++k2)
        {
          sum += k[k1_size - k1 - 1][k2_size - k2 - 1] * im[i - st_x + k1][j - st_y + k2];
        }
      }
      y[i - st_x][j - st_y] = sum;
    }
  }

  return y;
}

// with border mode = same
std::vector<std::vector<float>> keras::conv_single_depth_same(
    std::vector<std::vector<float>> const &im,
    std::vector<std::vector<float>> const &k)
{
  size_t k1_size = k.size(), k2_size = k[0].size();
  unsigned int st_x = (k1_size - 1) >> 1;
  unsigned int st_y = (k2_size - 1) >> 1;

  size_t max_imc = im.size() - 1;
  size_t max_imr = im[0].size() - 1;
  std::vector<std::vector<float>> y(im.size(), std::vector<float>(im[0].size(), 0));

  for (unsigned int i = 0; i < im.size(); ++i)
  {
    for (unsigned int j = 0; j < im[0].size(); ++j)
    {
      float sum = 0;

      for (unsigned int k1 = 0; k1 < k.size(); ++k1)
      {
        // const float * k_data = k[k1_size-k1-1].data(); // it is not working ...
        // const float * im_data = im[i-st_x+k1].data();
        for (unsigned int k2 = 0; k2 < k[0].size(); ++k2)
        {
          if (i - st_x + k1 < 0)
            continue;
          if (i - st_x + k1 > max_imc)
            continue;
          if (j - st_y + k2 < 0)
            continue;
          if (j - st_y + k2 > max_imr)
            continue;

          sum += k[k1_size - k1 - 1][k2_size - k2 - 1] * im[i - st_x + k1][j - st_y + k2];
        }
      }

      y[i][j] = sum;
    }
  }

  return y;
}

/**
 * @brief Destroy the Activation Layer object
 *
 */
keras::LayerActivation::~LayerActivation()
{
  if (m_delegate != nullptr)
    delete m_delegate;
}

/**
 * @brief Load the weights from the input file
 *
 * @param fin input file to read
 * @param enabler delegate to enable
 */
void keras::LayerActivation::load_weights(std::ifstream &fin, DelegateEnabler &enabler)
{
  /** Read the activation type of the layer **/
  fin >> m_activation_type;

  /** Create the delegate if need it **/
  if (enabler.softmax && (m_activation_type == "softmax"))
    m_delegate = new DelegateSoftmax(m_verbose);

  if (m_verbose)
  {
    std::cout << "Activation type " << m_activation_type << std::endl;

    if (m_delegate != nullptr)
      std::cout << "Delegate was added" << std::endl;
  }
}

/**
 * @brief Compute the activation function
 *
 * @param dc data chunk to apply the activation function
 * @return keras::DataChunk*
 */
keras::DataChunk *keras::LayerActivation::compute_output(keras::DataChunk *dc)
{
  /** 3D Data chunk **/
  if (dc->get_data_dim() == 3)
  {
    /** Get the values from data chunk **/
    std::vector<std::vector<std::vector<float>>> y = dc->get_3d();

    /** Apply ReLU function **/
    if (m_activation_type == "relu")
    {
      for (unsigned int i = 0; i < y.size(); ++i)
      {
        for (unsigned int j = 0; j < y[0].size(); ++j)
        {
          for (unsigned int k = 0; k < y[0][0].size(); ++k)
          {
            if (y[i][j][k] < 0)
              y[i][j][k] = 0;
          }
        }
      }

      /** Set the output **/
      keras::DataChunk *out = new keras::DataChunk2D();
      out->set_data(y);

      return out;
    }
    /** No implemented activation function for 3D value **/
    else
    {
      keras::missing_activation_impl(m_activation_type);
    }
  }
  /** Activation function for flat data **/
  else if (dc->get_data_dim() == 1)
  {
    /** Get the data **/
    std::vector<float> y = dc->get_1d();

    /** Apply ReLU function **/
    if (m_activation_type == "relu")
    {
      for (unsigned int k = 0; k < y.size(); ++k)
      {
        if (y[k] < 0)
          y[k] = 0;
      }
    }
    /** Apply Softmax function **/
    else if (m_activation_type == "softmax")
    {
      if (m_delegate == nullptr)
      {
        float sum = 0.0;

        for (unsigned int k = 0; k < y.size(); ++k)
        {
          y[k] = exp(y[k]);
          sum += y[k];
        }

        for (unsigned int k = 0; k < y.size(); ++k)
        {
          y[k] /= sum;
        }
      }
      else
      {
        y = m_delegate->eval(y);
      }
    }
    /** Apply sigmoid function **/
    else if (m_activation_type == "sigmoid")
    {
      for (unsigned int k = 0; k < y.size(); ++k)
      {
        y[k] = 1 / (1 + exp(-y[k]));
      }
    }
    /** Apply tanh function **/
    else if (m_activation_type == "tanh")
    {
      for (unsigned int k = 0; k < y.size(); ++k)
      {
        y[k] = tanh(y[k]);
      }
    }
    /** No implemented activation function for flat data **/
    else
    {
      keras::missing_activation_impl(m_activation_type);
    }

    /** Set the flat output **/
    keras::DataChunk *out = new DataChunkFlat();
    out->set_data(y);

    return out;
  }
  /** Data dimensions are not supported **/
  else
  {
    throw "data dim not supported";
  }

  return dc;
}

void keras::LayerConv2D::load_weights(std::ifstream &fin, DelegateEnabler &enabler)
{
  char tmp_char = ' ';
  std::string tmp_str = "";
  float tmp_float;
  bool skip = false;

  fin >> m_kernels_cnt >> m_depth >> m_rows >> m_cols >> m_border_mode;

  if (m_border_mode == "[")
  {
    m_border_mode = "valid";
    skip = true;
  }

  /** Create the delegate if need it **/
  if (enabler.conv2d)
    m_delegate = new DelegateConv2D(m_verbose);

  if (m_verbose)
  {
    std::cout << "LayerConv2D " << m_kernels_cnt << "x" << m_depth << "x"
              << m_rows << "x" << m_cols << " border_mode " << m_border_mode << std::endl;

    if (nullptr != m_delegate)
      std::cout << "Delegate was added" << std::endl;
  }

  // Reading kernel weights
  for (int k = 0; k < m_kernels_cnt; ++k)
  {
    std::vector<std::vector<std::vector<float>>> tmp_depths;

    for (int d = 0; d < m_depth; ++d)
    {
      std::vector<std::vector<float>> tmp_single_depth;

      for (int r = 0; r < m_rows; ++r)
      {
        if (!skip)
        {
          fin >> tmp_char;
        } // for '['
        else
        {
          skip = false;
        }

        std::vector<float> tmp_row;

        for (int c = 0; c < m_cols; ++c)
        {
          fin >> tmp_float;
          tmp_row.push_back(tmp_float);
        }

        fin >> tmp_char; // for ']'
        tmp_single_depth.push_back(tmp_row);
      }

      tmp_depths.push_back(tmp_single_depth);
    }

    m_kernels.push_back(tmp_depths);
  }

  // reading kernel biases
  fin >> tmp_char; // for '['

  for (int k = 0; k < m_kernels_cnt; ++k)
  {
    fin >> tmp_float;
    m_bias.push_back(tmp_float);
  }

  fin >> tmp_char; // for ']'
}

void keras::LayerMaxPooling::load_weights(std::ifstream &fin, DelegateEnabler &enabler)
{
  fin >> m_pool_x >> m_pool_y;

  if (m_verbose)
    std::cout << "MaxPooling " << m_pool_x << "x" << m_pool_y << std::endl;
}

void keras::LayerDense::load_weights(std::ifstream &fin, DelegateEnabler &enabler)
{
  fin >> m_input_cnt >> m_neurons;
  float tmp_float;
  char tmp_char = ' ';

  if (m_verbose)
    std::cout << "Inputs: " << m_input_cnt << ", neurons: " << m_neurons << std::endl;

  for (int i = 0; i < m_input_cnt; ++i)
  {
    std::vector<float> tmp_n;
    fin >> tmp_char; // for '['

    for (int n = 0; n < m_neurons; ++n)
    {
      fin >> tmp_float;
      tmp_n.push_back(tmp_float);
    }

    fin >> tmp_char; // for ']'
    m_weights.push_back(tmp_n);
  }

  if (m_verbose)
    std::cout << "weights " << m_weights.size() << std::endl;

  fin >> tmp_char; // for '['

  for (int n = 0; n < m_neurons; ++n)
  {
    fin >> tmp_float;
    m_bias.push_back(tmp_float);
  }

  fin >> tmp_char; // for ']'

  if (m_verbose)
    std::cout << "bias " << m_bias.size() << std::endl;
}

keras::DataChunk *keras::LayerFlatten::compute_output(keras::DataChunk *dc)
{
  std::vector<std::vector<std::vector<float>>> im = dc->get_3d();

  size_t csize = im[0].size();
  size_t rsize = im[0][0].size();
  size_t size = im.size() * csize * rsize;
  keras::DataChunkFlat *out = new DataChunkFlat(size);
  float *y_ret = out->get_1d_rw().data();

  for (size_t i = 0, dst = 0; i < im.size(); ++i)
  {
    for (size_t j = 0; j < csize; ++j)
    {
      float *row = im[i][j].data();

      for (size_t k = 0; k < rsize; ++k)
      {
        y_ret[dst++] = row[k];
      }
    }
  }

  return out;
}

keras::DataChunk *keras::LayerMaxPooling::compute_output(keras::DataChunk *dc)
{
  std::vector<std::vector<std::vector<float>>> im = dc->get_3d();
  std::vector<std::vector<std::vector<float>>> y_ret;

  for (unsigned int i = 0; i < im.size(); ++i)
  {
    std::vector<std::vector<float>> tmp_y;

    for (unsigned int j = 0; j < (unsigned int)(im[0].size() / m_pool_x); ++j)
    {
      tmp_y.push_back(std::vector<float>((int)(im[0][0].size() / m_pool_y), 0.0));
    }

    y_ret.push_back(tmp_y);
  }

  for (unsigned int d = 0; d < y_ret.size(); ++d)
  {
    for (unsigned int x = 0; x < y_ret[0].size(); ++x)
    {
      unsigned int start_x = x * m_pool_x;
      unsigned int end_x = start_x + m_pool_x;

      for (unsigned int y = 0; y < y_ret[0][0].size(); ++y)
      {
        unsigned int start_y = y * m_pool_y;
        unsigned int end_y = start_y + m_pool_y;

        std::vector<float> values;

        for (unsigned int i = start_x; i < end_x; ++i)
        {
          for (unsigned int j = start_y; j < end_y; ++j)
          {
            values.push_back(im[d][i][j]);
          }
        }

        y_ret[d][x][y] = *max_element(values.begin(), values.end());
      }
    }
  }

  keras::DataChunk *out = new keras::DataChunk2D();
  out->set_data(y_ret);

  return out;
}

keras::DataChunk *keras::LayerConv2D::compute_output(keras::DataChunk *dc)
{
  if (m_verbose)
    std::cout << "Running: " << this->get_name() << std::endl;

  unsigned int st_x = (m_kernels[0][0].size() - 1) >> 1;
  unsigned int st_y = (m_kernels[0][0][0].size() - 1) >> 1;
  std::vector<std::vector<std::vector<float>>> y_ret;
  auto const &im = dc->get_3d();

  // Adding padding to the image
  size_t size_x = (m_border_mode == "valid") ? im[0].size() - 2 * st_x : im[0].size();
  size_t size_y = (m_border_mode == "valid") ? im[0][0].size() - 2 * st_y : im[0][0].size();

  // Reserve image memory
  for (unsigned int i = 0; i < m_kernels.size(); ++i)
  { // depth
    std::vector<std::vector<float>> tmp;
    tmp.reserve(size_x);

    for (unsigned int j = 0; j < size_x; ++j)
    { // rows
      tmp.emplace_back(std::vector<float>(size_y, 0.0));
    }

    y_ret.push_back(tmp);
  }

  if (m_verbose)
  {
    std::cout << "y_ret: " << y_ret.size() << "," << y_ret[0].size() << "," << y_ret[0][0].size() << std::endl;
    std::cout << "kernel: " << m_kernels.size() << "x" << m_kernels[0].size() << "x" << m_kernels[0][0].size() << "x" << m_kernels[0][0][0].size() << std::endl;
    std::cout << "img: " << im.size() << "x" << im[0].size() << "x" << im[0][0].size() << std::endl;
  }

  if (nullptr != m_delegate)
  {
    axc_delegate_conv_params_t params = {
        .input_height = (uint32_t)im[0].size(),
        .output_height = (uint32_t)size_x,
        .input_width = (uint32_t)im[0][0].size(),
        .output_width = (uint32_t)size_y,
        .kernel_size = (uint8_t)m_kernels[0][0][0].size(),
        .num_kernels = (uint16_t)m_kernels.size(),
        .padding_type = (m_border_mode == "valid") ? CONV_PADDING_VALID : CONV_PADDING_SAME,
        .stride_x = 0,
        .stride_y = 0
    };

    m_delegate->eval(im, m_kernels, y_ret, &params);
  }
  else
  {
    for (unsigned int j = 0; j < m_kernels.size(); ++j)
    { // loop over kernels
      for (unsigned int m = 0; m < im.size(); ++m)
      { // loop over image depth
        std::vector<std::vector<float>> tmp_w = (m_border_mode == "valid") ? keras::conv_single_depth_valid(im[m], m_kernels[j][m]) : keras::conv_single_depth_same(im[m], m_kernels[j][m]);

        for (unsigned int x = 0; x < tmp_w.size(); ++x)
        {
          for (unsigned int y = 0; y < tmp_w[0].size(); ++y)
          {
            y_ret[j][x][y] += tmp_w[x][y];
          }
        }
      }

      for (unsigned int x = 0; x < y_ret[0].size(); ++x)
      {
        for (unsigned int y = 0; y < y_ret[0][0].size(); ++y)
        {
          y_ret[j][x][y] += m_bias[j];
        }
      }
    }
  }

  keras::DataChunk *out = new keras::DataChunk2D();
  out->set_data(y_ret);

  return out;
}

keras::DataChunk *keras::LayerDense::compute_output(keras::DataChunk *dc)
{
  // cout << "weights: input size " << m_weights.size() << endl;
  // cout << "weights: neurons size " << m_weights[0].size() << endl;
  // cout << "bias " << m_bias.size() << endl;
  size_t size = m_weights[0].size();
  size_t size8 = size >> 3;
  keras::DataChunkFlat *out = new DataChunkFlat(size, 0);
  float *y_ret = out->get_1d_rw().data();

  auto const &im = dc->get_1d();

  for (size_t j = 0; j < m_weights.size(); ++j)
  { // iter over input
    const float *w = m_weights[j].data();
    float p = im[j];
    size_t k = 0;

    for (size_t i = 0; i < size8; ++i)
    {                       // iter over neurons
      y_ret[k] += w[k] * p; // vectorize if you can
      y_ret[k + 1] += w[k + 1] * p;
      y_ret[k + 2] += w[k + 2] * p;
      y_ret[k + 3] += w[k + 3] * p;
      y_ret[k + 4] += w[k + 4] * p;
      y_ret[k + 5] += w[k + 5] * p;
      y_ret[k + 6] += w[k + 6] * p;
      y_ret[k + 7] += w[k + 7] * p;
      k += 8;
    }

    while (k < size)
    {
      y_ret[k] += w[k] * p;
      ++k;
    }
  }

  for (size_t i = 0; i < size; ++i)
  { // add biases
    y_ret[i] += m_bias[i];
  }

  return out;
}
