#pragma once

#include "Macros.h"

namespace EigenFilter {

/** \brief First and second moments (mean and covariance) of a distribution.
 *
 * \tparam Scalar_   scalar type
 * \tparam Size_     dimension, or \c Eigen::Dynamic
 * \tparam MaxSize_  maximum dimension; a fixed value with \c Size_ = \c Eigen::Dynamic
 *                   keeps storage on the stack
 *
 * No invariants are enforced: the covariance is not required to be symmetric or
 * positive definite. Like Eigen matrices, fixed-size Moments are uninitialized
 * by default.
 */
template <typename Scalar_, int Size_, int MaxSize_ = Size_>
class Moments {
 public:
  using Scalar = Scalar_;
  using Index = Eigen::Index;
  static constexpr int SizeAtCompileTime = Size_;
  static constexpr int MaxSizeAtCompileTime = MaxSize_;

  using VectorType = Eigen::Matrix<Scalar, Size_, 1, Eigen::ColMajor, MaxSize_, 1>;
  using MatrixType = Eigen::Matrix<Scalar, Size_, Size_, Eigen::ColMajor, MaxSize_, MaxSize_>;

  Moments() = default;

  /** Constructs moments of dimension \a n with uninitialized contents. */
  explicit Moments(Index n) { resize(n); }

  template <typename MeanDerived, typename CovDerived>
  Moments(const Eigen::MatrixBase<MeanDerived>& mean, const Eigen::MatrixBase<CovDerived>& cov)
      : m_mean(mean), m_cov(cov) {
    EIGEN_FILTER_ASSERT(m_cov.rows() == m_mean.size() && m_cov.cols() == m_mean.size());
  }

  Index size() const { return m_mean.size(); }

  /** Resizes mean and covariance to dimension \a n. Contents are unspecified afterward. */
  void resize(Index n) {
    m_mean.resize(n);
    m_cov.resize(n, n);
  }

  const VectorType& mean() const { return m_mean; }
  const MatrixType& cov() const { return m_cov; }
  VectorType& mean() { return m_mean; }
  MatrixType& cov() { return m_cov; }

 private:
  VectorType m_mean;
  MatrixType m_cov;
};

}  // namespace EigenFilter
