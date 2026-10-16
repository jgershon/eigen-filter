#pragma once

#include <expected>
#include <utility>

#include <Eigen/Cholesky>

#include "Macros.h"
#include "Moments.h"

namespace EigenFilter {

/** Reasons MutableGaussian::toGaussian() can fail.
 *
 * \sa MutableGaussian::toGaussian()
 */
enum class GaussianError {
  NotPositiveDefinite,  ///< Cholesky factorization of the covariance failed
  NonFinite,            ///< the covariance's lower triangle contains NaN or infinity
};

template <typename Scalar_, int Size_, int MaxSize_ = Size_>
class Gaussian;

/** \brief Mutable mean and covariance that can be viewed as a Gaussian.
 *
 * toGaussian() factorizes the covariance (if it has changed since the last
 * factorization) and returns a Gaussian view. While any Gaussian view is alive
 * this object is locked: mutating, resizing, moving, or destroying it is a
 * programmer error caught by \c EIGEN_FILTER_ASSERT.
 *
 * Only the lower triangle of the covariance is read by the factorization.
 * Changing the mean does not invalidate the factorization.
 *
 * \tparam Scalar_   scalar type
 * \tparam Size_     dimension, or \c Eigen::Dynamic
 * \tparam MaxSize_  maximum dimension
 *
 * \sa class Gaussian, class Moments
 */
template <typename Scalar_, int Size_, int MaxSize_ = Size_>
class MutableGaussian {
 public:
  using MomentsType = Moments<Scalar_, Size_, MaxSize_>;
  using Scalar = typename MomentsType::Scalar;
  using Index = typename MomentsType::Index;
  using VectorType = typename MomentsType::VectorType;
  using MatrixType = typename MomentsType::MatrixType;
  using GaussianType = Gaussian<Scalar_, Size_, MaxSize_>;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};

  MutableGaussian() = default;

  /** Constructs a Gaussian of dimension \a n with uninitialized contents. */
  explicit MutableGaussian(Index n) : m_moments{n} {}

  explicit MutableGaussian(MomentsType moments) : m_moments{std::move(moments)} {}

  template <typename MeanDerived, typename CovDerived>
  MutableGaussian(const Eigen::MatrixBase<MeanDerived>& mean, const Eigen::MatrixBase<CovDerived>& cov)
      : m_moments{mean, cov} {}

  /** Copies the data and any valid factorization. The copy starts unlocked. */
  MutableGaussian(const MutableGaussian& other)
      : m_moments{other.m_moments},
        m_llt{other.m_llt},
        m_logDeterminant{other.m_logDeterminant},
        m_factorValid{other.m_factorValid} {}

  MutableGaussian(MutableGaussian&& other)
      : m_moments{std::move(other.m_moments)},
        m_llt{std::move(other.m_llt)},
        m_logDeterminant{other.m_logDeterminant},
        m_factorValid{other.m_factorValid} {
    other.assertUnlocked();
    other.m_factorValid = false;
  }

  MutableGaussian& operator=(const MutableGaussian& other) {
    assertUnlocked();
    m_moments = other.m_moments;
    m_llt = other.m_llt;
    m_logDeterminant = other.m_logDeterminant;
    m_factorValid = other.m_factorValid;
    return *this;
  }

  MutableGaussian& operator=(MutableGaussian&& other) {
    assertUnlocked();
    other.assertUnlocked();
    m_moments = std::move(other.m_moments);
    m_llt = std::move(other.m_llt);
    m_logDeterminant = other.m_logDeterminant;
    m_factorValid = other.m_factorValid;
    other.m_factorValid = false;
    return *this;
  }

  ~MutableGaussian() { assertUnlocked(); }

  [[nodiscard]] Index size() const { return m_moments.size(); }

  /** Resizes mean, covariance, and factorization. Contents are unspecified afterward. */
  void resize(Index n) {
    assertUnlocked();
    m_moments.resize(n);
    m_factorValid = false;
  }

  [[nodiscard]] const MomentsType& moments() const { return m_moments; }
  [[nodiscard]] const VectorType& mean() const { return m_moments.mean(); }
  [[nodiscard]] const MatrixType& cov() const { return m_moments.cov(); }

  /** Sets the mean. Does not invalidate the factorization. */
  template <typename Derived>
  void setMean(const Eigen::MatrixBase<Derived>& mean) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(mean.size() == size());
    m_moments.mean() = mean;
  }

  /** Sets the covariance, invalidating the factorization. */
  template <typename Derived>
  void setCov(const Eigen::MatrixBase<Derived>& cov) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(cov.rows() == size() && cov.cols() == size());
    m_moments.cov() = cov;
    m_factorValid = false;
  }

  /** Factorizes the covariance if it has changed since the last factorization.
   *
   * Calling this ahead of time moves the cost of factorization out of a later
   * toGaussian() call. Does not lock this object.
   *
   * \returns nothing on success, or a GaussianError if the covariance is not
   *          positive definite or not finite.
   *
   * \sa toGaussian(), GaussianError
   */
  [[nodiscard]] std::expected<void, GaussianError> factorize() const {
    if (m_factorValid) return {};
    m_llt.compute(m_moments.cov());
    if (m_llt.info() != Eigen::Success) return std::unexpected{GaussianError::NotPositiveDefinite};
    // NaN slips through LLT's positivity test but always reaches the diagonal of
    // L, so checking the log-determinant catches any non-finite lower-triangle input.
    const Scalar logDet{Scalar(2) * m_llt.matrixLLT().diagonal().array().log().sum()};
    if (!Eigen::numext::isfinite(logDet)) return std::unexpected{GaussianError::NonFinite};
    m_logDeterminant = logDet;
    m_factorValid = true;
    return {};
  }

  /** Factorizes the covariance if needed (see factorize()).
   *
   * \returns a Gaussian view that locks this object for its lifetime, or a
   *          GaussianError if the covariance is not positive definite or not finite.
   *
   * \sa factorize(), class Gaussian, GaussianError
   */
  [[nodiscard("the Gaussian view holds the lock; discarding it releases the lock immediately")]]
  std::expected<GaussianType, GaussianError> toGaussian() const {
    if (auto result{factorize()}; !result) return std::unexpected{result.error()};
    return GaussianType{this};
  }

 private:
  friend GaussianType;

  void assertUnlocked() const {
    EIGEN_FILTER_ASSERT(m_lockCount == 0 && "MutableGaussian is locked by a live Gaussian view");
  }

  MomentsType m_moments;
  // Factorization cache, filled lazily by the const factorize().
  mutable Eigen::LLT<MatrixType> m_llt;
  mutable Scalar m_logDeterminant{};
  mutable bool m_factorValid{false};
  mutable int m_lockCount{0};
};

/** \brief Read-only view of a MutableGaussian whose covariance has been factorized.
 *
 * Represents the density
 * \f[ p(x) = (2\pi)^{-n/2} |\Sigma|^{-1/2} \exp\left(-\tfrac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right). \f]
 *
 * Obtained from MutableGaussian::toGaussian(). Holding a Gaussian locks the
 * source MutableGaussian; copies share the lock and the last one to be
 * destroyed releases it. A Gaussian must not outlive its source.
 *
 * \sa class MutableGaussian
 */
template <typename Scalar_, int Size_, int MaxSize_>
class Gaussian {
 public:
  using MutableType = MutableGaussian<Scalar_, Size_, MaxSize_>;
  using MomentsType = typename MutableType::MomentsType;
  using Scalar = typename MutableType::Scalar;
  using Index = typename MutableType::Index;
  using VectorType = typename MutableType::VectorType;
  using MatrixType = typename MutableType::MatrixType;
  using LLTType = Eigen::LLT<MatrixType>;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};

  Gaussian(const Gaussian& other) : m_source{other.m_source} { lock(); }
  Gaussian(Gaussian&& other) noexcept : m_source{std::exchange(other.m_source, nullptr)} {}

  Gaussian& operator=(const Gaussian& other) {
    if (this != &other) {
      unlock();
      m_source = other.m_source;
      lock();
    }
    return *this;
  }

  Gaussian& operator=(Gaussian&& other) noexcept {
    if (this != &other) {
      unlock();
      m_source = std::exchange(other.m_source, nullptr);
    }
    return *this;
  }

  ~Gaussian() { unlock(); }

  [[nodiscard]] Index size() const { return source().size(); }
  [[nodiscard]] const MomentsType& moments() const { return source().moments(); }
  [[nodiscard]] const VectorType& mean() const { return source().mean(); }
  [[nodiscard]] const MatrixType& cov() const { return source().cov(); }

  /** \returns the Cholesky factorization \f$ \Sigma = L L^T \f$ of the covariance */
  [[nodiscard]] const LLTType& llt() const { return source().m_llt; }

  /** \returns the log-determinant \f$ \log|\Sigma| \f$ of the covariance */
  [[nodiscard]] Scalar logDeterminant() const { return source().m_logDeterminant; }

  /** \returns the log of the density at the mean, \f$ -\frac{1}{2}(n \log 2\pi + \log|\Sigma|) \f$
   *
   * \sa logPdf()
   */
  [[nodiscard]] Scalar logPeak() const {
    const Scalar log2Pi{Eigen::numext::log(Scalar(2) * Scalar(EIGEN_PI))};
    return Scalar(-0.5) * (Scalar(size()) * log2Pi + logDeterminant());
  }

  /** \returns the weighted sum of squared errors (squared Mahalanobis distance)
   * \f$ (x-\mu)^T \Sigma^{-1} (x-\mu) \f$ of \a x
   *
   * \sa logPdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar wsse(const Eigen::MatrixBase<Derived>& x) const {
    static_assert(Derived::IsVectorAtCompileTime, "wsse() expects a vector");
    EIGEN_FILTER_ASSERT(x.size() == size());
    VectorType residual{x - mean()};
    llt().matrixL().solveInPlace(residual);
    return residual.squaredNorm();
  }

  /** \returns the log of the density at \a x, \f$ \log p(x) = \mathrm{logPeak} - \frac{1}{2}\mathrm{wsse}(x) \f$
   *
   * \sa logPeak(), wsse(), pdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar logPdf(const Eigen::MatrixBase<Derived>& x) const {
    return logPeak() - Scalar(0.5) * wsse(x);
  }

  /** \returns the density \f$ p(x) \f$ at \a x
   *
   * \warning Underflows easily in high dimensions; prefer logPdf().
   *
   * \sa logPdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar pdf(const Eigen::MatrixBase<Derived>& x) const {
    return Eigen::numext::exp(logPdf(x));
  }

 private:
  friend MutableType;

  explicit Gaussian(const MutableType* source) : m_source{source} { lock(); }

  [[nodiscard]] const MutableType& source() const {
    EIGEN_FILTER_ASSERT(m_source && "use of a moved-from Gaussian");
    return *m_source;
  }

  void lock() {
    if (m_source) ++m_source->m_lockCount;
  }

  void unlock() {
    if (m_source) --m_source->m_lockCount;
  }

  const MutableType* m_source;
};

}  // namespace EigenFilter
