# Add include to directories
INCLUDE_DIRECTORIES (${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Sec the proper macros
SET (LIBRARY_NAME keras2cpp)
SET (SRC_FILES
    datachunk.cc
    layers.cc 
    keras_model.cc
)
SET (INCLUDE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/datachunk.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/layers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/keras_model.hpp
)

# Add the library
ADD_LIBRARY (${LIBRARY_NAME} SHARED ${SRC_FILES} ${INCLUDE_FILES})

# Install library and include files
INSTALL (TARGETS ${LIBRARY_NAME} RUNTIME DESTINATION lib)
INSTALL (FILES ${INCLUDE_FILES} RUNTIME DESTINATION include)
