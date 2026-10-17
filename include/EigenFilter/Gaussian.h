#pragma once

#include <expected>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>

#include "Indexing.h"
#include "Macros.h"
#include "Moments.h"
#include "Space.h"

namespace EigenFilter {

/** Reasons MutableGaussian::toGaussian() can fail.
 *
 * \sa MutableGaussian::toGaussian()
 */
enum class GaussianError {
  NotPositiveDefinite,  ///< Cholesky factorization of the covariance failed
  NonFinite,            ///< the covariance's lower triangle contains NaN or infinity
};

template <typename Scalar_, int Size_, int MaxSize_ = Size_, typename Space_ = Linear<Size_>>
class Gaussian;

template <typename Scalar_, int Size_, int MaxSize_, typename Space_>
class MutableGaussian;

template <typename Source, typename Indices,
          typename SubSpace = internal::ProjectedSpace<
              typename Source::SpaceType,
              internal::IndexedSize<Indices, Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>::size>>
class GaussianMarginal;

namespace internal {

template <template <typename, typename, typename> typename Marginal, typename... Path, typename Source>
[[nodiscard]] auto tagMarginal(const Source& source);

/** Shared implementation of MutableGaussian::conditional() and Gaussian::conditional(). */
template <typename Source, typename TargetIndices, typename GivenIndices, typename Derived>
[[nodiscard]] auto conditionalGaussian(const Source& source, const TargetIndices& targetIndices,
                                       const GivenIndices& givenIndices, const Eigen::MatrixBase<Derived>& values) {
  constexpr int SourceSize{Source::SizeAtCompileTime};
  constexpr int SourceMaxSize{Source::MaxSizeAtCompileTime};
  const auto target{resolveIndices<SourceSize, SourceMaxSize>(source.size(), targetIndices)};
  const auto given{resolveIndices<SourceSize, SourceMaxSize>(source.size(), givenIndices)};
  EIGEN_FILTER_ASSERT(distinctIndices(target, given) && "target and given indices must be distinct");
  using Target = std::remove_cvref_t<decltype(target)>;
  using Given = std::remove_cvref_t<decltype(given)>;
  const auto targetSpace{projectSpace<Target::RowsAtCompileTime>(source.space(), source.size(), target)};
  const auto givenSpace{projectSpace<Given::RowsAtCompileTime>(source.space(), source.size(), given)};
  auto result{conditionMoments(source.mean(), source.cov(), target, given, values, givenSpace, targetSpace)};
  return MutableGaussian<typename Source::Scalar, Target::RowsAtCompileTime, Target::MaxRowsAtCompileTime,
                         std::remove_cvref_t<decltype(targetSpace)>>{targetSpace, std::move(result.mean),
                                                                     std::move(result.cov)};
}

/** conditionalGaussian() with every entry not in \a givenIndices as the target. */
template <typename Source, typename GivenIndices, typename Derived>
[[nodiscard]] auto conditionalGaussian(const Source& source, const GivenIndices& givenIndices,
                                       const Eigen::MatrixBase<Derived>& values) {
  constexpr int SourceSize{Source::SizeAtCompileTime};
  constexpr int SourceMaxSize{Source::MaxSizeAtCompileTime};
  const auto given{resolveIndices<SourceSize, SourceMaxSize>(source.size(), givenIndices)};
  return conditionalGaussian(source, complementIndices<SourceSize, SourceMaxSize>(source.size(), given), given, values);
}

}  // namespace internal

/** \brief Mutable mean and covariance that can be viewed as a Gaussian.
 *
 * toGaussian() factorizes the covariance (if it has changed since the last
 * factorization) and returns a Gaussian view. While any Gaussian view is alive
 * this object is locked: mutating, resizing, moving, or destroying it is a
 * programmer error caught by \c EIGEN_FILTER_ASSERT.
 *
 * marginal() returns a read-only view of a subset of the entries. While any
 * marginal is alive this object is borrowed: its values may change, but resizing,
 * moving, or destroying it is a programmer error caught by \c EIGEN_FILTER_ASSERT.
 *
 * The space describes how each coordinate behaves (see \ref Spaces). The mean is
 * normalized whenever it is set, and residuals are normalized before they are used, so
 * a periodic coordinate's mean stays in range and its residuals take the short way
 * around. Densities use the nearest image only (see \ref Spaces).
 *
 * Only the lower triangle of the covariance is read by the factorization.
 * Changing the mean does not invalidate the factorization.
 *
 * \tparam Scalar_   scalar type
 * \tparam Size_     dimension, or \c Eigen::Dynamic
 * \tparam MaxSize_  maximum dimension
 * \tparam Space_    space of the coordinates; Linear by default
 *
 * \sa class Gaussian, class Moments, MutableGaussianIn
 */
template <typename Scalar_, int Size_, int MaxSize_ = Size_, typename Space_ = Linear<Size_>>
class MutableGaussian {
  static_assert(Size_ == Eigen::Dynamic || isValidSize<Space_>(Size_), "the size does not fit the space");

 public:
  using MomentsType = Moments<Scalar_, Size_, MaxSize_>;
  using Scalar = typename MomentsType::Scalar;
  using Index = typename MomentsType::Index;
  using VectorType = typename MomentsType::VectorType;
  using MatrixType = typename MomentsType::MatrixType;
  using SpaceType = Space_;
  using GaussianType = Gaussian<Scalar_, Size_, MaxSize_, Space_>;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};
  static constexpr bool IsMutable{true};

  MutableGaussian() = default;

  /** Constructs a Gaussian of dimension \a n with uninitialized contents. */
  explicit MutableGaussian(Index n) : m_moments{n} { adoptDefaultSpace(); }

  explicit MutableGaussian(MomentsType moments) : m_moments{std::move(moments)} { adoptDefaultSpace(); }

  template <typename MeanDerived, typename CovDerived>
  MutableGaussian(const Eigen::MatrixBase<MeanDerived>& mean, const Eigen::MatrixBase<CovDerived>& cov)
      : m_moments{mean, cov} {
    adoptDefaultSpace();
  }

  /** Constructs a Gaussian of dimension \a n in \a space, with uninitialized contents. */
  MutableGaussian(Space_ space, Index n) : m_moments{n}, m_space{std::move(space)} { checkSpace(); }

  MutableGaussian(Space_ space, MomentsType moments) : m_moments{std::move(moments)}, m_space{std::move(space)} {
    checkSpace();
  }

  template <typename MeanDerived, typename CovDerived>
  MutableGaussian(Space_ space, const Eigen::MatrixBase<MeanDerived>& mean, const Eigen::MatrixBase<CovDerived>& cov)
      : m_moments{mean, cov}, m_space{std::move(space)} {
    checkSpace();
  }

  /** Copies the data and any valid factorization. The copy starts unlocked. */
  MutableGaussian(const MutableGaussian& other)
      : m_moments{other.m_moments},
        m_space{other.m_space},
        m_llt{other.m_llt},
        m_logDeterminant{other.m_logDeterminant},
        m_factorValid{other.m_factorValid} {}

  /** Moving a locked object is a programmer error; if \c EIGEN_FILTER_ASSERT throws,
   * the exception reaches this \c noexcept boundary and calls \c std::terminate.
   */
  MutableGaussian(MutableGaussian&& other) noexcept
      : m_moments{std::move(other.m_moments)},
        m_space{std::move(other.m_space)},
        m_llt{std::move(other.m_llt)},
        m_logDeterminant{other.m_logDeterminant},
        m_factorValid{other.m_factorValid} {
    other.assertUnlocked();
    other.assertUnborrowed();
    other.m_factorValid = false;
  }

  MutableGaussian& operator=(const MutableGaussian& other) {
    assertUnlocked();
    assertUnborrowed();
    m_moments = other.m_moments;
    m_space = other.m_space;
    m_llt = other.m_llt;
    m_logDeterminant = other.m_logDeterminant;
    m_factorValid = other.m_factorValid;
    return *this;
  }

  MutableGaussian& operator=(MutableGaussian&& other) noexcept {
    assertUnlocked();
    assertUnborrowed();
    other.assertUnlocked();
    other.assertUnborrowed();
    m_moments = std::move(other.m_moments);
    m_space = std::move(other.m_space);
    m_llt = std::move(other.m_llt);
    m_logDeterminant = other.m_logDeterminant;
    m_factorValid = other.m_factorValid;
    other.m_factorValid = false;
    return *this;
  }

  ~MutableGaussian() {
    assertUnlocked();
    assertUnborrowed();
  }

  [[nodiscard]] Index size() const { return m_moments.size(); }

  /** Resizes mean, covariance, and factorization. \a n must fit the space. Contents are
   * unspecified afterward.
   */
  void resize(Index n) {
    assertUnlocked();
    assertUnborrowed();
    m_moments.resize(n);
    if constexpr (requires { m_space.resize(n); }) m_space.resize(n);
    EIGEN_FILTER_ASSERT(isValidSize<Space_>(n) && "the size does not fit the space");
    m_factorValid = false;
  }

  [[nodiscard]] const Space_& space() const { return m_space; }
  [[nodiscard]] const MomentsType& moments() const { return m_moments; }
  [[nodiscard]] const VectorType& mean() const { return m_moments.mean(); }
  [[nodiscard]] const MatrixType& cov() const { return m_moments.cov(); }

  /** Sets the mean, normalized by the space. Does not invalidate the factorization. */
  template <typename Derived>
  void setMean(const Eigen::MatrixBase<Derived>& mean) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(mean.size() == size());
    m_moments.mean() = mean;
    m_space.normalizeValues(m_moments.mean());
  }

  /** Sets the covariance, invalidating the factorization. */
  template <typename Derived>
  void setCov(const Eigen::MatrixBase<Derived>& cov) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(cov.rows() == size() && cov.cols() == size());
    m_moments.cov() = cov;
    m_factorValid = false;
  }

  /** \returns a read-only view of the entries \a indices, which may be any index set
   * Eigen accepts (\c seqN(0, 3), \c std::array, \c std::vector, \c lastN(2), ...).
   * The indices must be distinct. While the marginal is alive this object is borrowed.
   *
   * \sa class GaussianMarginal
   */
  template <typename Indices>
  [[nodiscard]] GaussianMarginal<MutableGaussian, Indices> marginal(const Indices& indices) const {
    return GaussianMarginal<MutableGaussian, Indices>{*this, indices};
  }

  /** \returns a read-only view of the subspace named by the tag path \a Path (see
   * index<>), whose space is that subspace with its parameters.
   */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    return internal::tagMarginal<GaussianMarginal, Path...>(*this);
  }

  /** \returns a copy of the distribution of the entries \a target given that the
   * entries \a given equal \a values. \a target and \a given must be distinct.
   *
   * \f$ \mu_{a|b} = \mu_a + K (v - \mu_b) \f$ and \f$ \Sigma_{a|b} = \Sigma_{aa} - K \Sigma_{ba} \f$,
   * with \f$ K = \Sigma_{ab} \Sigma_{bb}^{-1} \f$. Uses a pivoted LDLT, so it always produces
   * a result; for a numerically singular source the result may fail toGaussian().
   */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalGaussian(*this, target, given, values);
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalGaussian(*this, given, values);
  }

  /** conditional() given the subspace named by the tag path \a GivenPath (see index<>). */
  template <typename... GivenPath, typename Derived>
    requires(sizeof...(GivenPath) > 0)
  [[nodiscard]] auto conditional(const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalGaussian(*this, internal::asIndexSet(index<Space_, GivenPath...>), values);
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
  template <typename>
  friend class internal::Borrow;

  void assertUnlocked() const {
    EIGEN_FILTER_ASSERT(m_lockCount == 0 && "MutableGaussian is locked by a live Gaussian view");
  }

  void assertUnborrowed() const {
    EIGEN_FILTER_ASSERT(m_borrowCount == 0 && "MutableGaussian is borrowed by a live marginal");
  }

  /** Sizes a default-constructed space to the data and normalizes the mean. */
  void adoptDefaultSpace() {
    if constexpr (requires { m_space.resize(size()); }) m_space.resize(size());
    checkSpace();
  }

  /** Checks that the data fits the space and normalizes the mean. */
  void checkSpace() {
    if constexpr (requires { m_space.size(); }) {
      EIGEN_FILTER_ASSERT(m_space.size() == size() && "the space has a different number of coordinates");
    }
    EIGEN_FILTER_ASSERT(isValidSize<Space_>(size()) && "the size does not fit the space");
    m_space.normalizeValues(m_moments.mean());
  }

  MomentsType m_moments;
  EIGEN_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;
  // Factorization cache, filled lazily by the const factorize().
  mutable Eigen::LLT<MatrixType> m_llt;
  mutable Scalar m_logDeterminant{};
  mutable bool m_factorValid{false};
  mutable int m_lockCount{0};
  mutable int m_borrowCount{0};
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
template <typename Scalar_, int Size_, int MaxSize_, typename Space_>
class Gaussian {
 public:
  using MutableType = MutableGaussian<Scalar_, Size_, MaxSize_, Space_>;
  using SpaceType = Space_;
  using MomentsType = typename MutableType::MomentsType;
  using Scalar = typename MutableType::Scalar;
  using Index = typename MutableType::Index;
  using VectorType = typename MutableType::VectorType;
  using MatrixType = typename MutableType::MatrixType;
  using LLTType = Eigen::LLT<MatrixType>;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};
  static constexpr bool IsMutable{false};

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
  [[nodiscard]] const Space_& space() const { return source().space(); }
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
   * \f$ (x \boxminus \mu)^T \Sigma^{-1} (x \boxminus \mu) \f$ of \a x
   *
   * \sa logPdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar wsse(const Eigen::MatrixBase<Derived>& x) const {
    static_assert(Derived::IsVectorAtCompileTime, "wsse() expects a vector");
    EIGEN_FILTER_ASSERT(x.size() == size());
    VectorType residual{x - mean()};
    space().normalizeDifferences(residual);
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

  /** \returns a read-only view of the entries \a indices. It holds a copy of this
   * view, so the source stays locked while the marginal is alive.
   *
   * \sa MutableGaussian::marginal()
   */
  template <typename Indices>
  [[nodiscard]] GaussianMarginal<Gaussian, Indices> marginal(const Indices& indices) const {
    return GaussianMarginal<Gaussian, Indices>{*this, indices};
  }

  /** marginal() of the subspace named by the tag path \a Path; see MutableGaussian::marginal(). */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    return internal::tagMarginal<GaussianMarginal, Path...>(*this);
  }

  /** \returns a copy of the conditional distribution; see MutableGaussian::conditional(). */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalGaussian(*this, target, given, values);
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalGaussian(*this, given, values);
  }

  /** conditional() given the subspace named by the tag path \a GivenPath (see index<>). */
  template <typename... GivenPath, typename Derived>
    requires(sizeof...(GivenPath) > 0)
  [[nodiscard]] auto conditional(const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalGaussian(*this, internal::asIndexSet(index<Space_, GivenPath...>), values);
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

/** \brief Read-only view of a subset of the entries of a MutableGaussian or Gaussian.
 *
 * Obtained from MutableGaussian::marginal() or Gaussian::marginal(). From a
 * MutableGaussian it borrows the source (values may change underneath it, size may
 * not); from a Gaussian it holds a copy of the view, and so the lock.
 *
 * mean() and cov() return lazy Eigen expressions that must not outlive the marginal.
 * cov() reads only the source's lower triangle, so it is correct for indices in any
 * order. The dimension is known at compile time when the index set's size is.
 *
 * \tparam Source    MutableGaussian or Gaussian
 * \tparam Indices   the index type passed to marginal()
 * \tparam SubSpace  space of the selected coordinates: the projection of the source's
 *                   space, or a named subspace for marginal<Path...>()
 */
template <typename Source, typename Indices, typename SubSpace>
class GaussianMarginal {
  using Selection = internal::IndexedSize<Indices, Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>;

 public:
  using Scalar = typename Source::Scalar;
  using Index = Eigen::Index;
  using SpaceType = SubSpace;
  static constexpr int SizeAtCompileTime{Selection::size};
  static constexpr int MaxSizeAtCompileTime{Selection::maxSize};
  using MutableType = MutableGaussian<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime, SubSpace>;
  using MomentsType = typename MutableType::MomentsType;
  using IndexVectorType = typename Selection::type;

  /** Selects \a indices, projecting the source's space onto them. */
  GaussianMarginal(const Source& source, const Indices& indices)
      : m_holder{source},
        m_indices{
            internal::resolveIndices<Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>(source.size(), indices)},
        m_space{internal::projectSpace<SizeAtCompileTime>(source.space(), source.size(), m_indices)} {
    EIGEN_FILTER_ASSERT(internal::distinctIndices(m_indices) && "marginal indices must be distinct");
  }

  /** Selects \a indices, whose coordinates form \a space. */
  GaussianMarginal(const Source& source, const Indices& indices, SubSpace space)
      : m_holder{source},
        m_indices{
            internal::resolveIndices<Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>(source.size(), indices)},
        m_space{std::move(space)} {
    EIGEN_FILTER_ASSERT(internal::distinctIndices(m_indices) && "marginal indices must be distinct");
  }

  [[nodiscard]] Index size() const { return m_indices.size(); }
  [[nodiscard]] const SubSpace& space() const { return m_space; }

  /** \returns the source entries this marginal selects */
  [[nodiscard]] const IndexVectorType& indices() const { return m_indices; }

  [[nodiscard]] auto mean() const { return source().mean()(m_indices); }
  [[nodiscard]] auto cov() const { return internal::lowerSymmetricBlock(source().cov(), m_indices, m_indices); }
  [[nodiscard]] MomentsType moments() const { return MomentsType{mean(), cov()}; }

  /** \returns an independent MutableGaussian holding this marginal, in its space */
  [[nodiscard]] MutableType copy() const { return MutableType{m_space, moments()}; }

 private:
  [[nodiscard]] const Source& source() const { return m_holder.get(); }

  internal::HolderFor<Source> m_holder;
  IndexVectorType m_indices;
  SubSpace m_space;
};

namespace internal {

/** Shared implementation of marginal<Path...>(): the marginal over index<> of \a Path,
 * whose space is the named subspace.
 */
template <template <typename, typename, typename> typename Marginal, typename... Path, typename Source>
[[nodiscard]] auto tagMarginal(const Source& source) {
  using Space = typename Source::SpaceType;
  const auto indices{asIndexSet(index<Space, Path...>)};
  auto space{subspace<Space, Path...>(source.space())};
  return Marginal<Source, std::remove_cvref_t<decltype(indices)>, decltype(space)>{source, indices, std::move(space)};
}

}  // namespace internal

/** A MutableGaussian whose size comes from \a Space. */
template <typename Scalar, typename Space, int MaxSize = spaceSize<Space>>
using MutableGaussianIn = MutableGaussian<Scalar, spaceSize<Space>, MaxSize, Space>;

}  // namespace EigenFilter
