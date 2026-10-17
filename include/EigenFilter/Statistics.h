#pragma once

#include "Macros.h"
#include "SampleSet.h"
#include "Space.h"

namespace EigenFilter::internal {

/** Weighted covariance \f$ \sum_i w_i d_i d_i^T \f$ of the columns of \a deviations with
 * linear weights \a w. Exactly symmetric.
 */
template <typename MatrixType, typename DeviationsDerived, typename WeightsDerived>
[[nodiscard]] MatrixType weightedCovariance(const Eigen::MatrixBase<DeviationsDerived>& deviations,
                                            const Eigen::ArrayBase<WeightsDerived>& w) {
  using Scalar = typename MatrixType::Scalar;
  using Deviations = SampleSetN<Scalar, MatrixType::RowsAtCompileTime, MatrixType::MaxRowsAtCompileTime>;
  const Deviations scaled{deviations * w.sqrt().matrix().asDiagonal()};
  MatrixType cov{deviations.rows(), deviations.rows()};
  if constexpr (MatrixType::MaxRowsAtCompileTime == 1) {
    // rankUpdate() would treat the single-row matrix as a column vector.
    cov.setConstant(scaled.squaredNorm());
  } else {
    cov.setZero();
    cov.template selfadjointView<Eigen::Lower>().rankUpdate(scaled);
    for (Eigen::Index j{1}; j < cov.cols(); ++j) {
      for (Eigen::Index i{0}; i < j; ++i) cov(i, j) = cov(j, i);
    }
  }
  return cov;
}

/** Weighted mean and (biased) covariance of the columns of \a states, with weights
 * \f$ e^{\mathrm{logWeights}_i - \mathrm{logTotal}} \f$, in a linear space. The covariance
 * is exactly symmetric.
 */
template <typename MomentsType, typename StatesDerived, typename WeightsDerived>
[[nodiscard]] MomentsType weightedMoments(const Eigen::MatrixBase<StatesDerived>& states,
                                          const Eigen::ArrayBase<WeightsDerived>& logWeights,
                                          typename MomentsType::Scalar logTotal) {
  using Scalar = typename MomentsType::Scalar;
  const Eigen::Array<Scalar, Eigen::Dynamic, 1> w{(logWeights - logTotal).exp()};
  MomentsType result{states.rows()};
  result.mean() = states * w.matrix();
  result.cov() = weightedCovariance<typename MomentsType::MatrixType>((states.colwise() - result.mean()).eval(), w);
  return result;
}

/** weightedMoments() in \a space: a tangent-space mean and covariance.
 *
 * Differences are taken as \f$ x_i \boxminus r \f$ from a reference point \f$ r \f$ (the
 * heaviest point), averaged, and retracted onto \f$ r \f$; the step is repeated once
 * from the new mean, and the covariance is formed from the final centered differences.
 * For a linear space this is exactly weightedMoments(). For periodic coordinates it is
 * accurate while their spread is well under half the period.
 */
template <typename MomentsType, typename Space, typename StatesDerived, typename WeightsDerived>
[[nodiscard]] MomentsType weightedMomentsIn(const Space& space, const Eigen::MatrixBase<StatesDerived>& states,
                                            const Eigen::ArrayBase<WeightsDerived>& logWeights,
                                            typename MomentsType::Scalar logTotal) {
  if constexpr (Space::IsLinear) {
    return weightedMoments<MomentsType>(states, logWeights, logTotal);
  } else {
    using Scalar = typename MomentsType::Scalar;
    using VectorType = typename MomentsType::VectorType;
    using Deviations = SampleSet<Scalar, typename MomentsType::SpaceType>;
    const Eigen::Array<Scalar, Eigen::Dynamic, 1> w{(logWeights - logTotal).exp()};
    Eigen::Index reference{0};
    w.maxCoeff(&reference);

    MomentsType result{states.rows()};
    result.mean() = states.col(reference);
    Deviations deviations{states.rows(), states.cols()};
    for (int pass{0}; pass < 2; ++pass) {
      deviations = states.colwise() - result.mean();
      space.normalizeDifferences(deviations);
      const VectorType shift{deviations * w.matrix()};
      result.mean() += shift;
      space.normalizeValues(result.mean());
      if (pass == 1) deviations.colwise() -= shift;
    }
    result.cov() = weightedCovariance<typename MomentsType::MatrixType>(deviations, w);
    return result;
  }
}

/** Effective sample size \f$ 1 / \sum_i \hat w_i^2 \f$ of the given log-weights. */
template <typename WeightsDerived>
[[nodiscard]] typename WeightsDerived::Scalar effectiveSampleSize(const Eigen::ArrayBase<WeightsDerived>& logWeights,
                                                                  typename WeightsDerived::Scalar logTotal) {
  using Scalar = typename WeightsDerived::Scalar;
  return Scalar(1) / (logWeights - logTotal).exp().square().sum();
}

}  // namespace EigenFilter::internal
