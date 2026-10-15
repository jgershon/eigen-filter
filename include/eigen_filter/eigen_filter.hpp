#pragma once

#include <Eigen/Core>

static_assert(EIGEN_VERSION_AT_LEAST(5, 0, 0), "eigen_filter requires Eigen 5.0 or newer");

namespace eigen_filter {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 1;
inline constexpr int version_patch = 0;

}  // namespace eigen_filter
