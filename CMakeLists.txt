cmake_minimum_required(VERSION 3.14.0)

project(implicit-minimal-matching)

# Print the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

### Configure the compiler
# This is a basic, decent setup that should do something sane on most compilers

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    # using Clang (linux or apple) or GCC
    message("Using clang/gcc compiler flags")
    SET(CMAKE_CXX_STANDARD 20)
    SET(BASE_CXX_FLAGS "-std=c++20 -Wall -Wextra")
    SET(DISABLED_WARNINGS " -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-braces -Wno-unused-label -Wno-register -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-result -Wno-narrowing -Wno-unused-but-set-variable -Wno-unused-lambda-capture -Wno-unused-local-typedefs")
    SET(TRACE_INCLUDES " -H -Wno-error=unused-command-line-argument")

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        message("Setting clang-specific options")
        SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=3 -Wfatal-errors -fcolor-diagnostics")
        SET(CMAKE_CXX_FLAGS_DEBUG          "-g3 -fsanitize=address -fno-limit-debug-info")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message("Setting gcc-specific options")
        SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
        SET(CMAKE_CXX_FLAGS_DEBUG          "-g3")
        SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable")
    endif()

    SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS} -std=c++20")

    include(CheckCXXCompilerFlag)
    unset(COMPILER_SUPPORTS_MARCH_NATIVE CACHE)
    CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG -std=c++20")
    else()
        SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -mcpu=apple-m1 -DNDEBUG -std=c++20") # Apple M1
    endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
    message("Using Visual Studio compiler flags")
    set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /W4")
    set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /MP") # parallel build
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4267\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4244\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4305\"")  # ignore truncation on initialization
    SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

    add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
    add_definitions(-DNOMINMAX)
    add_definitions(-D_USE_MATH_DEFINES)
else()
    # unrecognized
    message( FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]" )
endif()

### Policy settings
cmake_policy(SET CMP0054 NEW)   # don't implicitly dereference inside if()


### Process settings
option(BUILD_SHARED_LIBS "Build the shared library" FALSE)
if(BUILD_SHARED_LIBS)
    message("-- Building SHARED libraries")
else()
    message("-- Building STATIC libraries")
endif()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake") # look for stuff in the /cmake directory

# Work with non-standard homebrew installations
# (from ceres build system)
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    find_program(HOMEBREW_EXECUTABLE brew)
    mark_as_advanced(FORCE HOMEBREW_EXECUTABLE)
    if (HOMEBREW_EXECUTABLE)
        # Detected a Homebrew install, query for its install prefix.
        execute_process(COMMAND ${HOMEBREW_EXECUTABLE} --prefix
                OUTPUT_VARIABLE HOMEBREW_INSTALL_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "Detected Homebrew with install prefix: "
                "${HOMEBREW_INSTALL_PREFIX}, adding to CMake search paths.")
        list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_INSTALL_PREFIX}")
    endif()
endif()


### Handle windows-specific fixes
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions (-DNOMINMAX)  # don't use weird windows built-in min/max
    add_definitions (-D_USE_MATH_DEFINES)  # match unix behavior of constants in cmath
endif()

### Recurse to the source code
add_subdirectory(src)

# install
install(
        TARGETS implicit-minimal-matching
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
