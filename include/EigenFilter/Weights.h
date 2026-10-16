#pragma once

#include <expected>
#include <ranges>

#include "Macros.h"

namespace EigenFilter {

/** Reasons a collection of log-weights cannot be normalized.
 *
 * \sa MutableGaussianMixture::logTotalWeight(), MutableParticleSet::logTotalWeight()
 */
enum class WeightError {
  Empty,      ///< there are no weights
  ZeroSum,    ///< every weight is zero (every log-weight is -inf)
  NonFinite,  ///< a log-weight is NaN or +inf
};

namespace internal {

/** Streaming log-sum-exp: accumulates \f$ \log \sum_i e^{x_i} \f$ without overflow.
 * Inputs of -inf contribute nothing; an empty sum is -inf.
 */
template <typename Scalar>
class LogSumExp {
 public:
  void add(Scalar x) {
    if (x == -Eigen::NumTraits<Scalar>::infinity()) return;
    if (x <= m_max) {
      m_sum += Eigen::numext::exp(x - m_max);
    } else {
      m_sum = m_sum * Eigen::numext::exp(m_max - x) + Scalar(1);
      m_max = x;
    }
  }

  [[nodiscard]] Scalar value() const {
    return m_sum == Scalar(0) ? -Eigen::NumTraits<Scalar>::infinity() : m_max + Eigen::numext::log(m_sum);
  }

 private:
  Scalar m_max{-Eigen::NumTraits<Scalar>::infinity()};
  Scalar m_sum{0};
};

/** \returns the log of the sum of the weights whose logs are \a logWeights, or a
 *          WeightError if there are none, they sum to zero, or one is NaN or +inf.
 */
template <typename Scalar, std::ranges::input_range R>
[[nodiscard]] std::expected<Scalar, WeightError> logTotalWeight(R&& logWeights) {
  LogSumExp<Scalar> total;
  bool empty{true};
  for (const Scalar w : logWeights) {
    if (Eigen::numext::isnan(w) || w == Eigen::NumTraits<Scalar>::infinity()) {
      return std::unexpected{WeightError::NonFinite};
    }
    total.add(w);
    empty = false;
  }
  if (empty) return std::unexpected{WeightError::Empty};
  const Scalar result{total.value()};
  if (result == -Eigen::NumTraits<Scalar>::infinity()) return std::unexpected{WeightError::ZeroSum};
  return result;
}

}  // namespace internal

}  // namespace EigenFilter
