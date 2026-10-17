#pragma once

#include <cmath>
#include <concepts>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "Indexing.h"
#include "Macros.h"
#include "Moments.h"
#include "SampleSet.h"
#include "Sampling.h"
#include "Space.h"

namespace EigenFilter {

/** \brief Independent Gamma distributions on the coordinates, for positive quantities such
 * as ranges, speeds, and rates.
 *
 * Coordinate \f$ i \f$ has shape \f$ k_i \f$ and scale \f$ \theta_i \f$:
 * \f[ p(x_i) = \frac{x_i^{k_i - 1} e^{-x_i / \theta_i}}{\Gamma(k_i)\, \theta_i^{k_i}}, \quad x_i \ge 0, \f]
 * with mean \f$ k_i \theta_i \f$ and variance \f$ k_i \theta_i^2 \f$. A shape of 1 is the
 * exponential distribution (see exponential()); shape \f$ \nu / 2 \f$ with scale 2 is
 * \f$ \chi^2_\nu \f$ (see chiSquared()). A shape above 1 puts the mode at
 * \f$ (k - 1) \theta \f$, away from zero; fromMeanAndStdDev() picks the shape and scale for a
 * given mean and spread.
 *
 * Every coordinate's space must contain all of <tt>[0, ∞)</tt> without folding it: Linear, or
 * Bounded below at or under zero and unbounded above, such as the default <tt>Bounded<></tt>.
 *
 * A GammaDensity is an immutable value: it owns its parameters and holds no lock.
 *
 * \tparam Scalar_  scalar type
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 *
 * \sa drawSamples(const GammaDensity&, Eigen::Index, G&), GammaDensityN
 */
template <typename Scalar_, typename Space_>
class GammaDensity {
 public:
  using Scalar = Scalar_;
  using Index = Eigen::Index;
  using SpaceType = Space_;
  using MomentsType = Moments<Scalar_, Space_>;
  using VectorType = typename MomentsType::VectorType;
  using MatrixType = typename MomentsType::MatrixType;
  static constexpr int SizeAtCompileTime{MomentsType::SizeAtCompileTime};
  static constexpr int MaxSizeAtCompileTime{MomentsType::MaxSizeAtCompileTime};

  /** Shape \a shape and scale \a scale in \a space. Each is a vector with one entry per
   * coordinate, or a scalar shared by every coordinate. A space with one resizable block is
   * resized to a vector parameter's size; with only scalars it keeps its own size.
   */
  template <typename Shape, typename Scale>
  GammaDensity(Space_ space, const Shape& shape, const Scale& scale) : m_space{std::move(space)} {
    const Index n{parameterSize(shape, scale, m_space.size())};
    if constexpr (requires { m_space.resize(n); }) {
      if (m_space.size() != n) m_space.resize(n);
    }
    EIGEN_FILTER_ASSERT(m_space.size() == n && "the size does not fit the space");
    m_shape = broadcast(shape, n);
    m_scale = broadcast(scale, n);
    EIGEN_FILTER_ASSERT((m_shape.array() > Scalar(0)).all() && m_shape.allFinite() &&
                        (m_scale.array() > Scalar(0)).all() && m_scale.allFinite() &&
                        "the shape and scale must be positive and finite");
    EIGEN_FILTER_ASSERT(spaceHoldsHalfLine(m_space) && "every coordinate must contain [0, inf) without folding it");
    using std::lgamma;
    using std::log;
    m_logNormalizer = Scalar(0);
    for (Index i{0}; i < n; ++i) m_logNormalizer -= lgamma(m_shape(i)) + m_shape(i) * log(m_scale(i));
  }

  /** Shape \a shape and scale \a scale in the default space; see GammaDensity(Space_, const Shape&, const Scale&). */
  template <typename Shape, typename Scale>
  GammaDensity(const Shape& shape, const Scale& scale) : GammaDensity{Space_{}, shape, scale} {}

  /** \returns the exponential distribution with mean \a scale (rate \f$ 1 / \theta \f$) in \a space */
  template <typename Scale>
  [[nodiscard]] static GammaDensity exponential(Space_ space, const Scale& scale) {
    return GammaDensity{std::move(space), Scalar(1), scale};
  }

  /** exponential() in the default space. */
  template <typename Scale>
  [[nodiscard]] static GammaDensity exponential(const Scale& scale) {
    return exponential(Space_{}, scale);
  }

  /** \returns \f$ \chi^2 \f$ with \a dof degrees of freedom in \a space: shape \f$ \nu / 2 \f$, scale 2 */
  template <typename Dof>
  [[nodiscard]] static GammaDensity chiSquared(Space_ space, const Dof& dof) {
    const Index n{parameterSize(dof, Scalar(2), space.size())};
    return GammaDensity{std::move(space), VectorType{Scalar(0.5) * broadcast(dof, n)}, Scalar(2)};
  }

  /** chiSquared() in the default space. */
  template <typename Dof>
  [[nodiscard]] static GammaDensity chiSquared(const Dof& dof) {
    return chiSquared(Space_{}, dof);
  }

  /** \returns the Gamma distribution with mean \a mean and standard deviation \a stdDev in
   * \a space: shape \f$ (\mu / \sigma)^2 \f$ and scale \f$ \sigma^2 / \mu \f$. Both must be positive.
   */
  template <typename Mean, typename StdDev>
  [[nodiscard]] static GammaDensity fromMeanAndStdDev(Space_ space, const Mean& mean, const StdDev& stdDev) {
    const Index n{parameterSize(mean, stdDev, space.size())};
    const VectorType mu{broadcast(mean, n)};
    const VectorType sigma{broadcast(stdDev, n)};
    return GammaDensity{std::move(space), VectorType{(mu.array() / sigma.array()).square().matrix()},
                        VectorType{(sigma.array().square() / mu.array()).matrix()}};
  }

  /** fromMeanAndStdDev() in the default space. */
  template <typename Mean, typename StdDev>
  [[nodiscard]] static GammaDensity fromMeanAndStdDev(const Mean& mean, const StdDev& stdDev) {
    return fromMeanAndStdDev(Space_{}, mean, stdDev);
  }

  [[nodiscard]] Index size() const { return m_shape.size(); }
  [[nodiscard]] const Space_& space() const { return m_space; }

  /** \returns the shape \f$ k \f$ of each coordinate */
  [[nodiscard]] const VectorType& shape() const { return m_shape; }

  /** \returns the scale \f$ \theta \f$ of each coordinate */
  [[nodiscard]] const VectorType& scale() const { return m_scale; }

  /** \returns the log of the density at \a x, \f$ -\infty \f$ if any coordinate is negative */
  template <typename Derived>
  [[nodiscard]] Scalar logPdf(const Eigen::MatrixBase<Derived>& x) const {
    static_assert(Derived::IsVectorAtCompileTime, "logPdf() expects a vector");
    EIGEN_FILTER_ASSERT(x.size() == size());
    using std::log;
    Scalar result{m_logNormalizer};
    for (Index i{0}; i < size(); ++i) {
      const Scalar xi{x(i)};
      if (xi < Scalar(0)) return -std::numeric_limits<Scalar>::infinity();
      // Skip the power term for shape 1, where it is 0 · log 0 at the origin.
      if (m_shape(i) != Scalar(1)) result += (m_shape(i) - Scalar(1)) * log(xi);
      result -= xi / m_scale(i);
    }
    return result;
  }

  /** \returns the density at \a x */
  template <typename Derived>
  [[nodiscard]] Scalar pdf(const Eigen::MatrixBase<Derived>& x) const {
    return Eigen::numext::exp(logPdf(x));
  }

  /** \returns the mean \f$ k \theta \f$ and the diagonal covariance \f$ k \theta^2 \f$ */
  [[nodiscard]] MomentsType moments() const {
    MomentsType result{size()};
    result.mean() = m_shape.cwiseProduct(m_scale);
    result.cov() = result.mean().cwiseProduct(m_scale).asDiagonal();
    return result;
  }

  /** \returns the distribution of the entries \a indices: their shapes and scales, in the
   * projection of this space. The indices must be distinct.
   */
  template <typename Indices>
  [[nodiscard]] auto marginal(const Indices& indices) const {
    return internal::marginalOver<SizeAtCompileTime, MaxSizeAtCompileTime>(
        m_space, indices, [&](const auto& target, auto space) { return restricted(target, std::move(space)); });
  }

  /** marginal() of the subspace named by the tag path \a Path (see idx<>), in that subspace. */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    return internal::marginalOverPath<SizeAtCompileTime, MaxSizeAtCompileTime, Path...>(
        m_space, [&](const auto& target, auto space) { return restricted(target, std::move(space)); });
  }

  /** \returns the distribution of the entries \a target given that the entries \a given equal
   * \a values: since the coordinates are independent, the marginal of \a target. Negative
   * values have zero density, so strictly the conditional is undefined there; this returns
   * the marginal anyway.
   */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalOver<SizeAtCompileTime, MaxSizeAtCompileTime>(
        m_space, target, given, [&](const auto& t, const auto& g, auto space) {
          EIGEN_FILTER_ASSERT(values.size() == g.size());
          return restricted(t, std::move(space));
        });
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalOver<SizeAtCompileTime, MaxSizeAtCompileTime>(
        m_space, given, [&](const auto& t, const auto& g, auto space) {
          EIGEN_FILTER_ASSERT(values.size() == g.size());
          return restricted(t, std::move(space));
        });
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
        m_space, size(), [&](const auto& t, const auto& g, auto space) {
          EIGEN_FILTER_ASSERT(values.size() == g.size());
          return restricted(t, std::move(space));
        });
  }

 private:
  /** \returns the shapes and scales of the coordinates \a target, which form \a space */
  template <typename Target, typename SubSpace>
  [[nodiscard]] GammaDensity<Scalar, SubSpace> restricted(const Target& target, SubSpace space) const {
    using ResultVector = typename GammaDensity<Scalar, SubSpace>::VectorType;
    return GammaDensity<Scalar, SubSpace>{std::move(space), ResultVector{m_shape(target)},
                                          ResultVector{m_scale(target)}};
  }

  template <typename P>
  static constexpr bool isScalar{std::is_arithmetic_v<P>};

  /** \returns the size of whichever parameter is a vector, or \a spaceSize if both are scalars */
  template <typename A, typename B>
  [[nodiscard]] static Index parameterSize(const A& a, const B& b, Index spaceSize) {
    if constexpr (!isScalar<A>) {
      if constexpr (!isScalar<B>) EIGEN_FILTER_ASSERT(a.size() == b.size() && "the parameters differ in size");
      return a.size();
    } else if constexpr (!isScalar<B>) {
      return b.size();
    } else {
      return spaceSize;
    }
  }

  template <typename P>
  [[nodiscard]] static VectorType broadcast(const P& p, Index n) {
    if constexpr (isScalar<P>) {
      return VectorType::Constant(n, Scalar(p));
    } else {
      EIGEN_FILTER_ASSERT(p.size() == n);
      return VectorType{p};
    }
  }

  [[nodiscard]] static bool spaceHoldsHalfLine(const Space_& space) {
    for (const Coordinate& c : internal::coordinatesOf(space)) {
      const bool holds{c.kind == CoordinateKind::Linear || (c.kind == CoordinateKind::Bounded && c.a <= 0 &&
                                                            c.b == std::numeric_limits<double>::infinity())};
      if (!holds) return false;
    }
    return true;
  }

  EIGEN_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;
  VectorType m_shape;
  VectorType m_scale;
  Scalar m_logNormalizer{};  ///< \f$ -\sum_i (\log\Gamma(k_i) + k_i \log\theta_i) \f$
};

/** \returns \a n independent draws from \a gamma, one per column.
 *
 * \sa drawSamplesWithLogPdf(const GammaDensity&, Eigen::Index, G&)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Scalar, Space> drawSamples(const GammaDensity<Scalar, Space>& gamma, Eigen::Index n, G& rng) {
  static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
  EIGEN_FILTER_ASSERT(n >= 0);
  const Eigen::Index d{gamma.size()};
  std::vector<std::gamma_distribution<Scalar>> distributions;
  distributions.reserve(std::size_t(d));
  for (Eigen::Index i{0}; i < d; ++i) distributions.emplace_back(gamma.shape()(i), gamma.scale()(i));
  SampleSet<Scalar, Space> samples{d, n};
  for (Eigen::Index j{0}; j < n; ++j) {
    for (Eigen::Index i{0}; i < d; ++i) samples(i, j) = distributions[std::size_t(i)](rng);
  }
  return samples;
}

/** \returns \a n independent draws from \a gamma, one per column, with the log-density at each.
 *
 * \sa drawSamples(const GammaDensity&, Eigen::Index, G&)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Space> drawSamplesWithLogPdf(const GammaDensity<Scalar, Space>& gamma,
                                                                     Eigen::Index n, G& rng) {
  SamplesWithLogPdf<Scalar, Space> result{drawSamples(gamma, n, rng), {}};
  result.logPdf.resize(n);
  for (Eigen::Index j{0}; j < n; ++j) result.logPdf(j) = gamma.logPdf(result.samples.col(j));
  return result;
}

/** A GammaDensity of dimension \a Size in a linear space, with storage bound \a MaxSize. */
template <typename Scalar, int Size, int MaxSize = Size>
using GammaDensityN = GammaDensity<Scalar, Linear<Size, MaxSize>>;

}  // namespace EigenFilter
