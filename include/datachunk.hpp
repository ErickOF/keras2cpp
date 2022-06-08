#ifndef KERAS2CPP_DATA_CHUNK_HPP
#define KERAS2CPP_DATA_CHUNK_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace keras
{
  std::vector<float> read_1d_array(std::ifstream &fin, int cols);

  class DataChunk;
  class DataChunk2D;
  class DataChunkFlat;
}

/**
 * @brief Template class for a DataChunk
 * 
 */
class keras::DataChunk
{
public:
  virtual ~DataChunk() {}

  /**
   * @brief Get the data dim object
   * 
   * @return size_t size of the chunk
   */
  virtual size_t get_data_dim(void) const
  {
    return 0;
  }

  /**
   * @brief Get the object as 1D vector
   * 
   * @return std::vector<float> const& 1D values
   */
  virtual std::vector<float> const &get_1d() const
  {
    throw "not implemented";
  };

  /**
   * @brief Get the object as 3D vector
   * 
   * @return std::vector<std::vector<std::vector<float>>> const& 3D values
   */
  virtual std::vector<std::vector<std::vector<float>>> const &get_3d() const
  {
    throw "not implemented";
  };

  /**
   * @brief Set the 3D data into the object
   * 
   */
  virtual void set_data(std::vector<std::vector<std::vector<float>>> const &) {};

  /**
   * @brief Set the 1D data into the object
   * 
   */
  virtual void set_data(std::vector<float> const &) {};
  /**
   * @brief Read the chunk data from a .dat file
   * 
   * @param fname filename to read
   */
  virtual void read_from_file(const std::string &fname) {};
  /**
   * @brief Name of the chunk
   * 
   */
  virtual void show_name() = 0;
  /**
   * @brief Prints out the values in the object
   * 
   */
  virtual void show_values() = 0;
};

/**
 * @brief Represents a 2D DataChunk
 * 
 */
class keras::DataChunk2D : public keras::DataChunk
{
public:
  /**
   * @brief Construct a new Data Chunk 2D object
   * 
   * @param verbose activate verbose mode
   */
  DataChunk2D(bool verbose) : DataChunk()
  {
    this->verbose = verbose;
  };

  /**
   * @brief Default constructor for a new Data Chunk 2D object
   * 
   */
  DataChunk2D() : DataChunk() {};

  /**
   * @brief Flatten data
   * 
   */
  void flatten();

  /**
   * @brief Get the data as 1D vector
   * 
   * @return std::vector<float> const& 1D values
   */
  std::vector<float> const &get_1d() const
  {
    return flat_out;
  };

  /**
   * @brief Get the data as 3D vector
   * 
   * @return std::vector<std::vector<std::vector<float>>> const& 3D values
   */
  std::vector<std::vector<std::vector<float>>> const &get_3d() const
  {
    return data;
  };

  /**
   * @brief Set the 1D data into the object
   * 
   */
  virtual void set_data(std::vector<std::vector<std::vector<float>>> const &d)
  {
    data = d;
  };

  /**
   * @brief Get the data dim object
   * 
   * @return size_t size of the chunk
   */
  size_t get_data_dim(void) const
  {
    return 3;
  }

  /**
   * @brief Name of the chunk
   * 
   */
  void show_name()
  {
    std::cout << "DataChunk2D " << data.size() << "x" << data[0].size() << "x" << data[0][0].size() << std::endl;
  }

  /**
   * @brief Prints out the values in the object
   * 
   */
  void show_values()
  {
    std::cout << "DataChunk2D values:" << std::endl;

    for (size_t i = 0; i < data.size(); ++i)
    {
      std::cout << "Kernel " << i << std::endl;

      for (size_t j = 0; j < data[0].size(); ++j)
      {
        for (size_t k = 0; k < data[0][0].size(); ++k)
        {
          std::cout << data[i][j][k] << " ";
        }

        std::cout << std::endl;
      }
    }
  }
  // unsigned int get_count()
  // {
  //   return data.size()*data[0].size()*data[0][0].size();
  // }

  /**
   * @brief Read the chunk data from a .dat file
   * 
   * @param fname filename to read
   */
  void read_from_file(const std::string &fname);

  /**
   * @brief Represents 3D data
   * 
   */
  std::vector<std::vector<std::vector<float>>> data; // depth, rows, cols
  /**
   * @brief Flatten output
   * 
   */
  std::vector<float> flat_out;

  /**
   * @brief Indicates if the verbose mode is activated
   * 
   */
  bool verbose;
  /**
   * @brief Number of channels
   * 
   */
  int m_depth;
  /**
   * @brief Number of rows
   * 
   */
  int m_rows;
  /**
   * @brief Number of columns
   * 
   */
  int m_cols;
};

/**
 * @brief Represents the Flatten DataChunk
 * 
 */
class keras::DataChunkFlat : public keras::DataChunk
{
public:
  /**
   * @brief Construct a new Data Chunk Flat object
   * 
   * @param size number of elements in the chunk
   */
  DataChunkFlat(size_t size) : f(size) {}
  /**
   * @brief Construct a new Data Chunk Flat object
   * 
   * @param size number of elements in the chunk
   * @param init initial values
   */
  DataChunkFlat(size_t size, float init) : f(size, init) {}
  /**
   * @brief Default constructor for a new Data Chunk Flat object
   * 
   */
  DataChunkFlat(void) {}

  /**
   * @brief Get the flatten data
   * 
   * @return std::vector<float>& 1D values
   */
  std::vector<float> &get_1d_rw()
  {
    return f;
  }

  /**
   * @brief Get the object as 1D vector
   * 
   * @return std::vector<float> const& 1D values
   */
  std::vector<float> const &get_1d() const
  {
    return f;
  }

  /**
   * @brief Set the 1D data into the object
   * 
   * @param d 1D values to set
   */
  void set_data(std::vector<float> const &d) {
    f = d;
  };

  /**
   * @brief Get the data dim object
   * 
   * @return size_t size of the chunk
   */
  size_t get_data_dim(void) const
  {
    return 1;
  }

  /**
   * @brief Name of the chunk
   * 
   */
  void show_name()
  {
    std::cout << "DataChunkFlat " << f.size() << std::endl;
  }

  /**
   * @brief Prints out the values in the object
   * 
   */
  void show_values()
  {
    std::cout << "DataChunkFlat values:" << std::endl;

    for (size_t i = 0; i < f.size(); ++i)
      std::cout << f[i] << " ";

    std::cout << std::endl;
  }

  /**
   * @brief Reads the chunk data from a .dat file
   * 
   * @param fname filename to read
   */
  void read_from_file(const std::string &fname) {};
  // unsigned int get_count() { return f.size(); }

  /**
   * @brief Flatten data in the chunk
   * 
   */
  std::vector<float> f;
};

#endif /* KERAS2CPP_DATA_CHUNK_HPP */
