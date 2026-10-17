#pragma once

#include <Eigen/Core>

static_assert(EIGEN_VERSION_AT_LEAST(5, 0, 0), "EigenFilter requires Eigen 5.0 or newer");

/** Assertion used for programmer errors (misuse of the API).
 *
 * Defaults to \c eigen_assert, so it is controlled by the same switches as Eigen's
 * own checks (\c NDEBUG, \c EIGEN_NO_DEBUG). Define it before including any
 * EigenFilter header to override.
 */
#ifndef EIGEN_FILTER_ASSERT
#define EIGEN_FILTER_ASSERT(x) eigen_assert(x)
#endif

/** Lets an empty member, such as the default Linear space, take no storage. */
#if defined(__has_cpp_attribute) && __has_cpp_attribute(msvc::no_unique_address)
#define EIGEN_FILTER_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#elif defined(__has_cpp_attribute) && __has_cpp_attribute(no_unique_address)
#define EIGEN_FILTER_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define EIGEN_FILTER_NO_UNIQUE_ADDRESS
#endif

namespace EigenFilter {

inline constexpr int versionMajor{0};
inline constexpr int versionMinor{1};
inline constexpr int versionPatch{0};

}  // namespace EigenFilter
