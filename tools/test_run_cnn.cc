#include "keras_model.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
  if (argc != 4 && argc != 5)
  {
    std::cout << "Wrong input, going to exit." << std::endl;
    std::cout << "There should be arguments: dumped_cnn_file input_sample output_file." << std::endl;

    return -1;
  }

  std::string dumped_cnn = argv[1];
  std::string input_data = argv[2];
  std::string response_file = argv[3];
  bool verbose = (argc == 5 && argv[4][0] == '1');

  std::cout << "Testing network from " << dumped_cnn << " on data from " << input_data << std::endl;

  // Input data sample
  keras::DataChunk *sample = new keras::DataChunk2D(verbose);
  sample->read_from_file(input_data);

  // Add delegates
  keras::DelegateEnabler enabler;
  enabler.softmax = true;

  // Construct network
  keras::KerasModel m(dumped_cnn, verbose, enabler);
  std::vector<float> response = m.compute_output(sample);

  // clean sample
  delete sample;

  // save response into file
  std::ofstream fout(response_file);

  for (unsigned int i = 0; i < response.size(); i++)
  {
    std::cout << "Class " << i << ": " << response[i] << std::endl;
    fout << response[i] << " ";
  }

  fout.close();

  return 0;
}
