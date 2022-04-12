# Minimum CMake version
CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

PROJECT (keras2cpp-model)

# Add executable
ADD_EXECUTABLE (keras2cpp keras_model.cc test_run_cnn.cc)

# Install in default bin directory
INSTALL (TARGETS keras2cpp DESTINATION bin)
