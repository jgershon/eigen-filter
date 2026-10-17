#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

#include "Macros.h"

namespace EigenFilter {

/** \defgroup Resamplers Resamplers
 *
 * A resampler is a function object called as \c r(weights, ancestors, rng). It
 * fills every element of \a ancestors with an index into \a weights, drawn so
 * that index \c i is chosen in proportion to \c weights[i]. \a weights are
 * nonnegative linear weights with a positive sum; they need not be normalized.
 * Zero-weight indices are never chosen. \a ancestors may be any length.
 *
 * \sa ParticleSet::resample()
 */

/** Requirements on a resampler for scalar type \a Scalar and random generator \a G. */
template <typename R, typename Scalar, typename G>
concept Resampler = requires(const R& r, std::span<const Scalar> weights, std::span<Eigen::Index> ancestors, G& rng) {
  r(weights, ancestors, rng);
};

namespace internal {

/** \returns a uniform draw from [0, 1). Guards against std::uniform_real_distribution returning 1. */
template <std::floating_point Scalar, std::uniform_random_bit_generator G>
[[nodiscard]] Scalar uniform01(G& rng) {
  std::uniform_real_distribution<Scalar> uniform{Scalar(0), Scalar(1)};
  return std::min(uniform(rng), std::nextafter(Scalar(1), Scalar(0)));
}

/** Fills \a ancestors by inverting the cumulative weights at positions \c u(j),
 * which must be nondecreasing in [0, 1). Zero-weight indices are never chosen.
 */
template <typename Scalar, typename Positions>
void invertCumulative(std::span<const Scalar> weights, std::span<Eigen::Index> ancestors, Positions u) {
  EIGEN_FILTER_ASSERT(!weights.empty());
  Scalar total{0};
  std::size_t lastPositive{0};
  for (std::size_t i{0}; i < weights.size(); ++i) {
    EIGEN_FILTER_ASSERT(weights[i] >= Scalar(0) && "weights must be nonnegative");
    total += weights[i];
    if (weights[i] > Scalar(0)) lastPositive = i;
  }
  EIGEN_FILTER_ASSERT(total > Scalar(0) && "weights must have a positive sum");

  std::size_t i{0};
  Scalar cumulative{weights[0]};
  for (std::size_t j{0}; j < ancestors.size(); ++j) {
    const Scalar position{u(j) * total};
    while (i < lastPositive && position >= cumulative) cumulative += weights[++i];
    ancestors[j] = Eigen::Index(i);
  }
}

}  // namespace internal

/** \ingroup Resamplers
 * \brief Draws each ancestor independently. Highest variance; mainly a reference.
 */
struct MultinomialResampler {
  template <std::floating_point Scalar, std::uniform_random_bit_generator G>
  void operator()(std::span<const Scalar> weights, std::span<Eigen::Index> ancestors, G& rng) const {
    std::vector<Scalar> positions(ancestors.size());
    for (auto& p : positions) p = internal::uniform01<Scalar>(rng);
    std::ranges::sort(positions);
    internal::invertCumulative(weights, ancestors, [&](std::size_t j) { return positions[j]; });
  }
};

/** \ingroup Resamplers
 * \brief Draws one uniform in each of \a M equal strata of [0, 1).
 */
struct StratifiedResampler {
  template <std::floating_point Scalar, std::uniform_random_bit_generator G>
  void operator()(std::span<const Scalar> weights, std::span<Eigen::Index> ancestors, G& rng) const {
    const Scalar m{Scalar(ancestors.size())};
    internal::invertCumulative(weights, ancestors,
                               [&](std::size_t j) { return (Scalar(j) + internal::uniform01<Scalar>(rng)) / m; });
  }
};

/** \ingroup Resamplers
 * \brief Uses one uniform offset for \a M evenly spaced positions in [0, 1).
 *
 * Index \c i is chosen either floor or ceil of \f$ M w_i / \sum w \f$ times. Low
 * variance and cheap; the usual default.
 */
struct SystematicResampler {
  template <std::floating_point Scalar, std::uniform_random_bit_generator G>
  void operator()(std::span<const Scalar> weights, std::span<Eigen::Index> ancestors, G& rng) const {
    const Scalar m{Scalar(ancestors.size())};
    const Scalar offset{internal::uniform01<Scalar>(rng)};
    internal::invertCumulative(weights, ancestors, [&](std::size_t j) { return (Scalar(j) + offset) / m; });
  }
};

/** \ingroup Resamplers
 * \brief Chooses index \c i floor(\f$ M w_i / \sum w \f$) times deterministically,
 * then draws the remainder with \a ResidualResampler_ from the fractional parts.
 */
template <typename ResidualResampler_ = MultinomialResampler>
struct ResidualResampler {
  ResidualResampler_ residual{};

  template <std::floating_point Scalar, std::uniform_random_bit_generator G>
  void operator()(std::span<const Scalar> weights, std::span<Eigen::Index> ancestors, G& rng) const {
    Scalar total{0};
    for (const Scalar w : weights) total += w;
    EIGEN_FILTER_ASSERT(total > Scalar(0) && "weights must have a positive sum");

    const Scalar m{Scalar(ancestors.size())};
    std::vector<Scalar> fractions(weights.size());
    std::size_t out{0};
    for (std::size_t i{0}; i < weights.size(); ++i) {
      const Scalar expected{m * weights[i] / total};
      const Scalar copies{std::floor(expected)};
      fractions[i] = expected - copies;
      for (std::size_t k{0}; k < std::size_t(copies) && out < ancestors.size(); ++k) ancestors[out++] = Eigen::Index(i);
    }
    if (out < ancestors.size()) residual(std::span<const Scalar>{fractions}, ancestors.subspan(out), rng);
  }
};

}  // namespace EigenFilter
