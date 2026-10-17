#pragma once

#include <algorithm>
#include <concepts>
#include <random>

#include "Gaussian.h"
#include "Macros.h"
#include "SampleSet.h"

namespace EigenFilter {

/** Samples together with the log of the density they were drawn from, evaluated at each.
 *
 * \sa drawSamplesWithLogPdf()
 */
template <typename Scalar, int Size, int MaxSize = Size>
struct SamplesWithLogPdf {
  SampleSet<Scalar, Size, MaxSize> samples;        ///< one sample per column
  Eigen::Array<Scalar, Eigen::Dynamic, 1> logPdf;  ///< \c logPdf(i) is the log-density at \c samples.col(i)
};

namespace internal {

/** \returns a \a rows by \a cols matrix of independent standard normal draws,
 * filled in storage order so results depend only on \a rng.
 */
template <typename Matrix, std::uniform_random_bit_generator G>
[[nodiscard]] Matrix standardNormal(Eigen::Index rows, Eigen::Index cols, G& rng) {
  using Scalar = typename Matrix::Scalar;
  static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
  Matrix z{rows, cols};
  std::normal_distribution<Scalar> normal;
  std::generate(z.data(), z.data() + z.size(), [&] { return normal(rng); });
  return z;
}

}  // namespace internal

/** \returns \a n independent draws from \a gaussian, one per column.
 *
 * Each draw is \f$ \mu + L z \f$ with \f$ z \sim \mathcal{N}(0, I) \f$ and \f$ L \f$
 * the Cholesky factor of the covariance.
 *
 * \sa drawSamplesWithLogPdf()
 */
template <typename Scalar, int Size, int MaxSize, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Scalar, Size, MaxSize> drawSamples(const Gaussian<Scalar, Size, MaxSize, Space>& gaussian,
                                                           Eigen::Index n, G& rng) {
  EIGEN_FILTER_ASSERT(n >= 0);
  using SampleSetType = SampleSet<Scalar, Size, MaxSize>;
  SampleSetType samples{gaussian.llt().matrixL() * internal::standardNormal<SampleSetType>(gaussian.size(), n, rng)};
  samples.colwise() += gaussian.mean();
  gaussian.space().normalizeValues(samples);
  return samples;
}

/** \returns \a n independent draws from \a gaussian, one per column, with the
 * log-density of \a gaussian at each.
 *
 * The log-densities cost almost nothing extra in a linear space: for
 * \f$ x = \mu + L z \f$ the squared Mahalanobis distance is \f$ \|z\|^2 \f$. In other
 * spaces each is Gaussian::logPdf() of the stored sample, so a draw that wraps far
 * around a periodic coordinate gets the nearest-image density, matching logPdf().
 *
 * \sa drawSamples()
 */
template <typename Scalar, int Size, int MaxSize, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Size, MaxSize> drawSamplesWithLogPdf(
    const Gaussian<Scalar, Size, MaxSize, Space>& gaussian, Eigen::Index n, G& rng) {
  EIGEN_FILTER_ASSERT(n >= 0);
  using SampleSetType = SampleSet<Scalar, Size, MaxSize>;
  const SampleSetType z{internal::standardNormal<SampleSetType>(gaussian.size(), n, rng)};
  SamplesWithLogPdf<Scalar, Size, MaxSize> result{gaussian.llt().matrixL() * z, {}};
  result.samples.colwise() += gaussian.mean();
  if constexpr (Space::IsLinear) {
    result.logPdf = gaussian.logPeak() - Scalar(0.5) * z.colwise().squaredNorm().transpose().array();
  } else {
    gaussian.space().normalizeValues(result.samples);
    result.logPdf.resize(n);
    for (Eigen::Index j{0}; j < n; ++j) result.logPdf(j) = gaussian.logPdf(result.samples.col(j));
  }
  return result;
}

}  // namespace EigenFilter
