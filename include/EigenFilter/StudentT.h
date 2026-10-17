#pragma once

#include <cmath>
#include <concepts>
#include <expected>
#include <numbers>
#include <random>
#include <utility>

#include "Gaussian.h"
#include "Indexing.h"
#include "Macros.h"
#include "Moments.h"
#include "SampleSet.h"
#include "Sampling.h"
#include "Space.h"

namespace EigenFilter {

/** \brief The multivariate Student-t distribution: a Gaussian with heavier tails.
 *
 * With location \f$ \mu \f$, scale matrix \f$ \Sigma \f$, and \f$ \nu \f$ degrees of freedom in
 * \f$ d \f$ dimensions,
 * \f[ p(x) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) (\nu\pi)^{d/2}
 * |\Sigma|^{1/2}}
 *     \left(1 + \frac{\delta^2}{\nu}\right)^{-\frac{\nu + d}{2}}, \f]
 * where \f$ \delta^2 = (x \boxminus \mu)^T \Sigma^{-1} (x \boxminus \mu) \f$. It is a scale
 * mixture of Gaussians: \f$ x = \mu + L z / \sqrt{u / \nu} \f$ with \f$ z \sim \mathcal{N}(0, I) \f$,
 * \f$ u \sim \chi^2_\nu \f$, and \f$ \Sigma = L L^T \f$. As \f$ \nu \to \infty \f$ it approaches
 * \f$ \mathcal{N}(\mu, \Sigma) \f$; \f$ \nu = 1 \f$ is the Cauchy distribution.
 *
 * The scale matrix is not the covariance: the covariance is \f$ \frac{\nu}{\nu - 2} \Sigma \f$,
 * and only exists for \f$ \nu > 2 \f$ (see moments()). Densities use the nearest image, as
 * a GaussianDensity's do.
 *
 * Created by create(), which factorizes the scale matrix. A StudentTDensity is an immutable
 * value: it owns its parameters and holds no lock.
 *
 * \tparam Scalar_  scalar type
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 *
 * \sa drawSamples(const StudentTDensity&, Eigen::Index, G&), StudentTDensityN
 */
template <typename Scalar_, typename Space_>
class StudentTDensity {
 public:
  using ScaleGaussianType = Gaussian<Scalar_, Space_>;
  using MomentsType = typename ScaleGaussianType::MomentsType;
  using Scalar = Scalar_;
  using Index = Eigen::Index;
  using SpaceType = Space_;
  using VectorType = typename MomentsType::VectorType;
  using MatrixType = typename MomentsType::MatrixType;
  static constexpr int SizeAtCompileTime{MomentsType::SizeAtCompileTime};
  static constexpr int MaxSizeAtCompileTime{MomentsType::MaxSizeAtCompileTime};

  /** \returns a Student-t with location \a location, scale matrix \a scale, and \a dof degrees of
   * freedom in \a space, or a GaussianError if \a scale is not positive definite or not finite.
   * Only the lower triangle of \a scale is read. \a dof must be positive and finite.
   */
  template <typename LocationDerived, typename ScaleDerived>
  [[nodiscard]] static std::expected<StudentTDensity, GaussianError> create(
      Space_ space, const Eigen::MatrixBase<LocationDerived>& location, const Eigen::MatrixBase<ScaleDerived>& scale,
      Scalar dof) {
    EIGEN_FILTER_ASSERT(dof > Scalar(0) && Eigen::numext::isfinite(dof) && "the degrees of freedom must be positive");
    ScaleGaussianType gaussian{std::move(space), location, MatrixType{scale.template selfadjointView<Eigen::Lower>()}};
    if (auto result{gaussian.factorize()}; !result) return std::unexpected{result.error()};
    return StudentTDensity{std::move(gaussian), dof};
  }

  /** create() in the default space. */
  template <typename LocationDerived, typename ScaleDerived>
  [[nodiscard]] static std::expected<StudentTDensity, GaussianError> create(
      const Eigen::MatrixBase<LocationDerived>& location, const Eigen::MatrixBase<ScaleDerived>& scale, Scalar dof) {
    return create(Space_{}, location, scale, dof);
  }

  /** create() from the location and scale of \a gaussian, which keeps its space. */
  [[nodiscard]] static std::expected<StudentTDensity, GaussianError> create(const ScaleGaussianType& gaussian,
                                                                            Scalar dof) {
    return create(gaussian.space(), gaussian.mean(), gaussian.cov(), dof);
  }

  [[nodiscard]] Index size() const { return m_gaussian.size(); }
  [[nodiscard]] const Space_& space() const { return m_gaussian.space(); }

  /** \returns the location \f$ \mu \f$: the mode and median, and the mean when \f$ \nu > 1 \f$ */
  [[nodiscard]] const VectorType& location() const { return m_gaussian.mean(); }

  /** \returns the scale matrix \f$ \Sigma \f$, symmetric */
  [[nodiscard]] const MatrixType& scale() const { return m_gaussian.cov(); }

  /** \returns the degrees of freedom \f$ \nu \f$ */
  [[nodiscard]] Scalar dof() const { return m_dof; }

  /** \returns a view of \f$ \mathcal{N}(\mu, \Sigma) \f$, whose factorization this distribution
   * shares. It must not outlive this distribution.
   */
  [[nodiscard]] typename ScaleGaussianType::DensityType scaleDensity() const { return *m_gaussian.toDensity(); }

  /** \returns the log of the density at \a x
   *
   * \sa pdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar logPdf(const Eigen::MatrixBase<Derived>& x) const {
    return logPdfAtWsse(scaleDensity().wsse(x));
  }

  /** \returns the density at \a x
   *
   * \warning Underflows easily in high dimensions; prefer logPdf().
   */
  template <typename Derived>
  [[nodiscard]] Scalar pdf(const Eigen::MatrixBase<Derived>& x) const {
    return Eigen::numext::exp(logPdf(x));
  }

  /** \returns the mean \f$ \mu \f$ and covariance \f$ \frac{\nu}{\nu - 2} \Sigma \f$, or
   * MomentsError::Infinite if \f$ \nu \le 2 \f$, when the covariance is infinite
   */
  [[nodiscard]] std::expected<MomentsType, MomentsError> moments() const {
    if (!(m_dof > Scalar(2))) return std::unexpected{MomentsError::Infinite};
    return MomentsType{location(), MatrixType{(m_dof / (m_dof - Scalar(2))) * scale()}};
  }

  /** \returns the log-density at a point whose squared Mahalanobis distance under the scale
   * matrix is \a wsse
   */
  [[nodiscard]] Scalar logPdfAtWsse(Scalar wsse) const {
    using std::log1p;
    return m_logNormalizer - Scalar(0.5) * (m_dof + Scalar(size())) * log1p(wsse / m_dof);
  }

  /** \returns the distribution of the entries \a indices, in the projection of this space: a
   * Student-t with the same degrees of freedom and the selected location and block of the
   * scale. The indices must be distinct.
   */
  template <typename Indices>
  [[nodiscard]] auto marginal(const Indices& indices) const {
    return internal::marginalOver<SizeAtCompileTime, MaxSizeAtCompileTime>(
        space(), indices, [&](const auto& target, auto subspace) { return restricted(target, std::move(subspace)); });
  }

  /** marginal() of the subspace named by the tag path \a Path (see idx<>), in that subspace. */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    return internal::marginalOverPath<SizeAtCompileTime, MaxSizeAtCompileTime, Path...>(
        space(), [&](const auto& target, auto subspace) { return restricted(target, std::move(subspace)); });
  }

  /** \returns the distribution of the entries \a target given that the entries \a given equal
   * \a values: again a Student-t, or a GaussianError if its scale fails to factorize, as it
   * may for a numerically singular source.
   *
   * With \f$ d_b \f$ given entries at squared Mahalanobis distance \f$ \delta_b^2 \f$ from their
   * location, the conditional has \f$ \nu + d_b \f$ degrees of freedom, the Gaussian
   * conditional's location \f$ \mu_a + K (v - \mu_b) \f$, and the Gaussian conditional's scale
   * \f$ \Sigma_{aa} - K \Sigma_{ba} \f$ times \f$ (\nu + \delta_b^2) / (\nu + d_b) \f$, with
   * \f$ K = \Sigma_{ab} \Sigma_{bb}^{-1} \f$. So values far from the location widen it.
   */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalOver<SizeAtCompileTime, MaxSizeAtCompileTime>(
        space(), target, given,
        [&](const auto& t, const auto& g, auto subspace) { return conditioned(t, g, values, std::move(subspace)); });
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalOver<SizeAtCompileTime, MaxSizeAtCompileTime>(
        space(), given,
        [&](const auto& t, const auto& g, auto subspace) { return conditioned(t, g, values, std::move(subspace)); });
  }

  /** conditional() given the subspace named by the tag path \a GivenPath (see idx<>).
   *
   * When \a GivenPath names one top-level block of a Composite space, the result is in the
   * space of the remaining blocks; otherwise it is in the projection of this space.
   */
  template <typename... GivenPath, typename Derived>
    requires(sizeof...(GivenPath) > 0)
  [[nodiscard]] auto conditional(const Eigen::MatrixBase<Derived>& values) const {
    return internal::tagConditional<SizeAtCompileTime, MaxSizeAtCompileTime, GivenPath...>(
        space(), size(),
        [&](const auto& t, const auto& g, auto subspace) { return conditioned(t, g, values, std::move(subspace)); });
  }

 private:
  /** \returns the marginal on the coordinates \a target, which form \a subspace */
  template <typename Target, typename SubSpace>
  [[nodiscard]] StudentTDensity<Scalar, SubSpace> restricted(const Target& target, SubSpace subspace) const {
    using Result = StudentTDensity<Scalar, SubSpace>;
    auto result{Result::create(std::move(subspace), location()(target),
                               typename Result::MatrixType{internal::lowerSymmetricBlock(scale(), target, target)},
                               m_dof)};
    EIGEN_FILTER_ASSERT(result.has_value() && "a principal block of a positive definite scale should factorize");
    return std::move(*result);
  }

  /** \returns the conditional on \a given equal to \a values, for the coordinates \a target,
   * which form \a subspace
   */
  template <typename Target, typename Given, typename Derived, typename SubSpace>
  [[nodiscard]] std::expected<StudentTDensity<Scalar, SubSpace>, GaussianError> conditioned(
      const Target& target, const Given& given, const Eigen::MatrixBase<Derived>& values, SubSpace subspace) const {
    using Result = StudentTDensity<Scalar, SubSpace>;
    const auto givenSpace{
        internal::projectSpace<Given::RowsAtCompileTime, Given::MaxRowsAtCompileTime>(space(), size(), given)};
    const auto gaussian{internal::conditionMoments(location(), scale(), target, given, values, givenSpace, subspace)};
    const Scalar givenSize{Scalar(given.size())};
    const Scalar stretch{(m_dof + gaussian.wsse) / (m_dof + givenSize)};
    return Result::create(std::move(subspace), gaussian.mean, typename Result::MatrixType{stretch * gaussian.cov},
                          m_dof + givenSize);
  }

  StudentTDensity(ScaleGaussianType gaussian, Scalar dof) : m_gaussian{std::move(gaussian)}, m_dof{dof} {
    using std::lgamma;
    using std::log;
    const Scalar d{Scalar(size())};
    const Scalar logDeterminant{scaleDensity().logDeterminant()};
    m_logNormalizer = lgamma(Scalar(0.5) * (m_dof + d)) - lgamma(Scalar(0.5) * m_dof) -
                      Scalar(0.5) * d * log(m_dof * std::numbers::pi_v<Scalar>) - Scalar(0.5) * logDeterminant;
  }

  ScaleGaussianType m_gaussian;  ///< \f$ \mathcal{N}(\mu, \Sigma) \f$, factorized
  Scalar m_dof;
  Scalar m_logNormalizer{};
};

namespace internal {

/** Draws from a Student-t, not yet normalized, with each draw's squared Mahalanobis distance. */
template <typename Scalar, typename Space>
struct StudentTDraws {
  SampleSet<Scalar, Space> samples;
  Eigen::Array<Scalar, Eigen::Dynamic, 1> wsse;
};

template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] StudentTDraws<Scalar, Space> drawStudentT(const StudentTDensity<Scalar, Space>& t, Eigen::Index n,
                                                        G& rng) {
  EIGEN_FILTER_ASSERT(n >= 0);
  using SampleSetType = SampleSet<Scalar, Space>;
  const SampleSetType z{standardNormal<SampleSetType>(t.size(), n, rng)};
  std::chi_squared_distribution<Scalar> chiSquared{t.dof()};
  Eigen::Array<Scalar, Eigen::Dynamic, 1> stretch{n};  // sqrt(ν / u) for each draw
  for (Eigen::Index j{0}; j < n; ++j) stretch(j) = Eigen::numext::sqrt(t.dof() / chiSquared(rng));
  StudentTDraws<Scalar, Space> result{t.scaleDensity().llt().matrixL() * z,
                                      z.colwise().squaredNorm().transpose().array() * stretch.square()};
  for (Eigen::Index j{0}; j < n; ++j) result.samples.col(j) *= stretch(j);
  result.samples.colwise() += t.location();
  return result;
}

}  // namespace internal

/** \returns \a n independent draws from \a t, one per column: \f$ \mu + L z \sqrt{\nu / u} \f$
 * with \f$ z \sim \mathcal{N}(0, I) \f$ and \f$ u \sim \chi^2_\nu \f$, normalized in the space.
 *
 * \sa drawSamplesWithLogPdf(const StudentTDensity&, Eigen::Index, G&)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Scalar, Space> drawSamples(const StudentTDensity<Scalar, Space>& t, Eigen::Index n, G& rng) {
  auto draws{internal::drawStudentT(t, n, rng)};
  t.space().normalizeValues(draws.samples);
  return std::move(draws.samples);
}

/** \returns \a n independent draws from \a t, one per column, with the log-density at each.
 *
 * In a linear space the log-densities cost almost nothing extra: the squared Mahalanobis
 * distance of a draw is \f$ \|z\|^2 \nu / u \f$. In other spaces each is logPdf() of the
 * stored sample, the nearest-image density.
 *
 * \sa drawSamples(const StudentTDensity&, Eigen::Index, G&)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Space> drawSamplesWithLogPdf(const StudentTDensity<Scalar, Space>& t,
                                                                     Eigen::Index n, G& rng) {
  auto draws{internal::drawStudentT(t, n, rng)};
  SamplesWithLogPdf<Scalar, Space> result{std::move(draws.samples), {}};
  result.logPdf.resize(n);
  if constexpr (Space::IsLinear) {
    for (Eigen::Index j{0}; j < n; ++j) result.logPdf(j) = t.logPdfAtWsse(draws.wsse(j));
  } else {
    t.space().normalizeValues(result.samples);
    for (Eigen::Index j{0}; j < n; ++j) result.logPdf(j) = t.logPdf(result.samples.col(j));
  }
  return result;
}

/** A StudentTDensity of dimension \a Size in a linear space, with storage bound \a MaxSize. */
template <typename Scalar, int Size, int MaxSize = Size>
using StudentTDensityN = StudentTDensity<Scalar, Linear<Size, MaxSize>>;

}  // namespace EigenFilter
