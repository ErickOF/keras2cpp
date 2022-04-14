#include <datachunk.hpp>

std::vector<float> keras::read_1d_array(std::ifstream &fin, int cols)
{
  std::vector<float> arr;
  float tmp_float;
  char tmp_char;
  fin >> tmp_char; // for '['

  for (int n = 0; n < cols; ++n)
  {
    fin >> tmp_float;
    arr.push_back(tmp_float);
  }

  fin >> tmp_char; // for ']'

  return arr;
}

void keras::DataChunk2D::flatten()
{
  flat_out.clear();

  for (std::vector<std::vector<float>> &v1: data)
    for (std::vector<float> &v2: v1)
      for (float &value: v2)
        flat_out.push_back(value);
}

void keras::DataChunk2D::read_from_file(const std::string &fname)
{
  std::cout << "Loading data chunk2d from " << fname << std::endl;

  std::ifstream fin(fname.c_str());
  fin >> m_depth >> m_rows >> m_cols;

  if (this->verbose)
    std::cout << m_depth << "x" << m_rows << "x" << m_cols << std::endl;

  for (int d = 0; d < m_depth; ++d)
  {
    std::vector<std::vector<float>> tmp_single_depth;

    for (int r = 0; r < m_rows; ++r)
    {
      std::vector<float> tmp_row = keras::read_1d_array(fin, m_cols);
      tmp_single_depth.push_back(tmp_row);
    }

    data.push_back(tmp_single_depth);
  }

  fin.close();
}
