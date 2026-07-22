cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Maybe stop from CMAKEing in the wrong place
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Go use the /build directory.")
endif()

# List of .cpp files to be compiled
SET(IMPLICIT_MINIMAL_MATCHING_SRCS
        topology/product_mesh.h
        topology/product_mesh.cpp

        linebundle/connection_laplacian.h
        linebundle/connection_laplacian.cpp
        linebundle/singularity_locator.h
        linebundle/singularity_locator.cpp

        utilities/lbfgs.h
        utilities/blopex.h

        matching/minimal_matching.h
        matching/minimal_matching.cpp
)

# Create a single library for the project
add_library(implicit-minimal-matching
        ${IMPLICIT_MINIMAL_MATCHING_SRCS})
target_include_directories(implicit-minimal-matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(../deps/geometry-central deps/geometry-central)
add_subdirectory(../deps/polyscope deps/polyscope)

set(BLOPEX_ABSTRACT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/blopex/blopex_abstract")
add_library(blopex STATIC
        "${BLOPEX_ABSTRACT_DIR}/krylov/lobpcg.c"
        "${BLOPEX_ABSTRACT_DIR}/multivector/multivector.c"
        "${BLOPEX_ABSTRACT_DIR}/multivector/temp_multivector.c"
        "${BLOPEX_ABSTRACT_DIR}/utilities/fortran_matrix.c"
)
add_library(blopex::blopex ALIAS blopex)
find_package(LAPACK REQUIRED)
target_include_directories(blopex SYSTEM PUBLIC
        "${BLOPEX_ABSTRACT_DIR}/krylov"
        "${BLOPEX_ABSTRACT_DIR}/multivector"
        "${BLOPEX_ABSTRACT_DIR}/utilities"
)
set_target_properties(blopex PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(blopex PUBLIC ${MATH_LIBRARY})
endif()
target_link_libraries(blopex PUBLIC ${LAPACK_LIBRARIES})

# Add all includes and link libraries from dependencies
target_link_libraries(implicit-minimal-matching geometry-central blopex::blopex)

# Set compiler properties for the library
target_compile_features(implicit-minimal-matching PUBLIC cxx_std_20)
set_target_properties(implicit-minimal-matching PROPERTIES
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS OFF
)
target_compile_definitions(implicit-minimal-matching PUBLIC NOMINMAX _USE_MATH_DEFINES)

# Export symbols if DLL is requested
if(MSVC AND BUILD_SHARED_LIBS)
    set_target_properties(implicit-minimal-matching PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Create executables
add_executable(implicit-minimal-matcher implicit-minimal-matcher.cpp)
target_link_libraries(implicit-minimal-matcher polyscope geometry-central implicit-minimal-matching)
target_include_directories(implicit-minimal-matcher PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../deps/polyscope/deps/args")
