# Minimum CMake version
CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

# Define the project
PROJECT (libkeras2cpp)

# Add lib subdirectory to build
ADD_SUBDIRECTORY (src)

# Include directories to building path
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/src)

# Add executable
ADD_EXECUTABLE (keras2cpp-model tools/test_run_cnn.cc)

# Link against libraries
TARGET_LINK_LIBRARIES (keras2cpp-model keras2cpp)

# Install the binary
INSTALL (TARGETS keras2cpp-model RUNTIME DESTINATION bin)

# Package source generation
SET (MAJOR "0")
SET (MINOR "1")
SET (PATCH "1")
SET (CPACK_SOURCE_GENERATOR "TGZ")
SET (CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${MAJOR}.${MINOR}.${PATCH}")
SET (CPACK_SOURCE_IGNORE_FILES "/build/;${CPACK_SOURCE_IGNORE_FILES}")

INCLUDE (CPack)
