# Use a system GoogleTest if available, otherwise fetch it.
set(INSTALL_GTEST OFF)
# Match the default MSVC runtime (/MD) used by the rest of the build.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
  URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
  SYSTEM
  EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(eigen_filter_tests
  smoke.cpp
  moments.cpp
  gaussian.cpp
  gaussian_mixture.cpp
  resampling.cpp
  particle_set.cpp
  sampling.cpp
  kernel_density.cpp
  gaussian_subspace.cpp
  mixture_subspace.cpp
  particle_subspace.cpp
  fitting.cpp
  space.cpp
  gaussian_space.cpp
  space_distributions.cpp
)
target_link_libraries(eigen_filter_tests PRIVATE
  EigenFilter::EigenFilter
  GTest::gtest_main
)
target_compile_options(eigen_filter_tests PRIVATE
  $<IF:$<CXX_COMPILER_ID:MSVC>,/W4 /permissive- /Zc:__cplusplus /bigobj,-Wall -Wextra -Wpedantic>
)

gtest_discover_tests(eigen_filter_tests)
