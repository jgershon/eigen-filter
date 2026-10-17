#pragma once

#include "Macros.h"
#include "Space.h"

namespace EigenFilter {

/** Reasons a distribution's moments() can fail. */
enum class MomentsError {
  Infinite,  ///< the distribution has no finite mean or covariance, such as a Student-t with \f$ \nu \le 2 \f$
};

/** \brief First and second moments (mean and covariance) of a distribution.
 *
 * \tparam Scalar_  scalar type
 * \tparam Space_   space of the mean (see \ref Spaces), which sets the dimension and the
 *                  storage bound. The space is part of the type only: none is stored,
 *                  and the mean is not normalized.
 *
 * No invariants are enforced: the covariance is not required to be symmetric or
 * positive definite. Like Eigen matrices, fixed-size Moments are uninitialized
 * by default.
 *
 * \sa MomentsN
 */
template <typename Scalar_, typename Space_>
class Moments {
 public:
  using Scalar = Scalar_;
  using Index = Eigen::Index;
  using SpaceType = Space_;
  static constexpr int SizeAtCompileTime{spaceSize<Space_>};
  static constexpr int MaxSizeAtCompileTime{spaceMaxSize<Space_>};

  using VectorType = Eigen::Matrix<Scalar, SizeAtCompileTime, 1, Eigen::ColMajor, MaxSizeAtCompileTime, 1>;
  using MatrixType = Eigen::Matrix<Scalar, SizeAtCompileTime, SizeAtCompileTime, Eigen::ColMajor, MaxSizeAtCompileTime,
                                   MaxSizeAtCompileTime>;

  Moments() = default;

  /** Constructs moments of dimension \a n with uninitialized contents. */
  explicit Moments(Index n) { resize(n); }

  template <typename MeanDerived, typename CovDerived>
  Moments(const Eigen::MatrixBase<MeanDerived>& mean, const Eigen::MatrixBase<CovDerived>& cov)
      : m_mean{mean}, m_cov{cov} {
    EIGEN_FILTER_ASSERT(m_cov.rows() == m_mean.size() && m_cov.cols() == m_mean.size());
  }

  [[nodiscard]] Index size() const { return m_mean.size(); }

  /** Resizes mean and covariance to dimension \a n. Contents are unspecified afterward. */
  void resize(Index n) {
    m_mean.resize(n);
    m_cov.resize(n, n);
  }

  [[nodiscard]] const VectorType& mean() const { return m_mean; }
  [[nodiscard]] const MatrixType& cov() const { return m_cov; }
  [[nodiscard]] VectorType& mean() { return m_mean; }
  [[nodiscard]] MatrixType& cov() { return m_cov; }

 private:
  VectorType m_mean;
  MatrixType m_cov;
};

/** Moments of dimension \a Size in a linear space, with storage bound \a MaxSize. */
template <typename Scalar, int Size, int MaxSize = Size>
using MomentsN = Moments<Scalar, Linear<Size, MaxSize>>;

}  // namespace EigenFilter
