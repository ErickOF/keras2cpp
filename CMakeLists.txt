# Minimum CMake version
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

# Define the project
PROJECT (keras2cpp)

# Include directories to building path
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/src)

# Add source and include files
SET (SRC_FILES
    src/datachunk.cc
    src/delegates.cc
    src/ipc.cc
    src/layers.cc
    src/keras_model.cc
    tools/test_run_cnn.cc
)
SET (INCLUDE_FILES
    include/datachunk.hpp
    include/delegates.hpp
    include/ipc.hpp
    include/layers.hpp
    include/keras_model.hpp
)

# Find glib-2.0 library
FIND_PACKAGE (PkgConfig REQUIRED)
PKG_CHECK_MODULES (GLIB REQUIRED glib-2.0)

# Add glib-2.0 source and library dirs
INCLUDE_DIRECTORIES (${GLIB_INCLUDE_DIRS})
LINK_DIRECTORIES (${GLIB_LIBRARY_DIRS})

# Add glib-2.0 flags
ADD_DEFINITIONS (${GLIB_CFLAGS_OTHER})

# Add executable
ADD_EXECUTABLE (keras2cpp-model ${SRC_FILES} ${INCLUDE_FILES})

# Add pthread
TARGET_LINK_LIBRARIES (keras2cpp-model pthread)
# Add rt
TARGET_LINK_LIBRARIES (keras2cpp-model rt)
# Add glib
TARGET_LINK_LIBRARIES (keras2cpp-model ${GLIB_LIBRARIES})

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
