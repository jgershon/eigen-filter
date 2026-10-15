cmake_minimum_required(VERSION 3.28)

project(eigen_filter
  VERSION 0.1.0
  DESCRIPTION "State estimation utilities in C++23 using Eigen"
  LANGUAGES CXX
)

option(EIGEN_FILTER_BUILD_TESTS "Build eigen_filter unit tests" ${PROJECT_IS_TOP_LEVEL})

# Only set global language settings when we're the top-level project; when
# consumed via FetchContent, the C++23 requirement propagates through the target.
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

include(FetchContent)

# --- Eigen ------------------------------------------------------------------
# Use a system Eigen 5.0+ if available, otherwise fetch it.  The content name
# matches the package name (Eigen3) so a parent project that declares Eigen
# first takes precedence.

# Let the normal variables below override Eigen's option() defaults.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_BLAS OFF)
set(EIGEN_BUILD_LAPACK OFF)
set(EIGEN_BUILD_DEMOS OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_CMAKE_PACKAGE OFF)

FetchContent_Declare(Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz
  URL_HASH SHA256=315c881e19e17542a7d428c5aa37d113c89b9500d350c433797b730cd449c056
  SYSTEM
  EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS 5.0 CONFIG
)
FetchContent_MakeAvailable(Eigen3)

# --- Library ----------------------------------------------------------------
add_library(eigen_filter INTERFACE)
add_library(eigen_filter::eigen_filter ALIAS eigen_filter)

target_include_directories(eigen_filter INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(eigen_filter INTERFACE cxx_std_23)
target_link_libraries(eigen_filter INTERFACE Eigen3::Eigen)

# --- Tests ------------------------------------------------------------------
if(EIGEN_FILTER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
