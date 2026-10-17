#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <utility>

#include "Indexing.h"
#include "Macros.h"
#include "Moments.h"
#include "SampleSet.h"
#include "Sampling.h"
#include "Space.h"

namespace EigenFilter {

/** \brief The uniform distribution over a box in a space.
 *
 * Each coordinate \f$ i \f$ is uniform on an interval of width \f$ w_i \f$ starting at
 * \f$ \ell_i \f$, independently, so the density is \f$ 1 / \prod_i w_i \f$ inside the box and
 * zero outside.
 *
 * The space supplies the default box: a Periodic coordinate is uniform over one whole
 * period, and a coordinate Bounded on both sides is uniform between its bounds. A Linear
 * coordinate, or one Bounded on one side only, has no default and needs an explicit box.
 * A Periodic coordinate's interval may cross the seam: <tt>[3, 3.5]</tt> on an angle covers
 * <tt>[3, π)</tt> and <tt>[-π, 3.5 - 2π]</tt>.
 *
 * A UniformDensity is an immutable value: it owns its box and holds no lock.
 *
 * \tparam Scalar_  scalar type
 * \tparam Space_   space of the coordinates (see \ref Spaces)
 *
 * \sa drawSamples(const UniformDensity&, Eigen::Index, G&), UniformDensityN
 */
template <typename Scalar_, typename Space_>
class UniformDensity {
 public:
  using Scalar = Scalar_;
  using Index = Eigen::Index;
  using SpaceType = Space_;
  using MomentsType = Moments<Scalar_, Space_>;
  using VectorType = typename MomentsType::VectorType;
  using MatrixType = typename MomentsType::MatrixType;
  static constexpr int SizeAtCompileTime{MomentsType::SizeAtCompileTime};
  static constexpr int MaxSizeAtCompileTime{MomentsType::MaxSizeAtCompileTime};

  /** Uniform over the whole of \a space. Every coordinate must be Periodic or Bounded on
   * both sides.
   */
  explicit UniformDensity(Space_ space = {})
    requires(!Space_::IsLinear)
      : m_space{std::move(space)} {
    const auto coordinates{internal::coordinatesOf(m_space)};
    const Index n{m_space.size()};
    m_lower.resize(n);
    m_width.resize(n);
    m_period.resize(n);
    for (Index i{0}; i < n; ++i) {
      const Coordinate& c{coordinates[std::size_t(i)]};
      if (c.kind == CoordinateKind::Periodic) {
        m_lower(i) = Scalar(c.a);
        m_width(i) = Scalar(c.b);
        m_period(i) = Scalar(c.b);
      } else {
        EIGEN_FILTER_ASSERT(c.kind == CoordinateKind::Bounded && std::isfinite(c.a) && std::isfinite(c.b) &&
                            "a coordinate has no default interval; give the box explicitly");
        m_lower(i) = Scalar(c.a);
        m_width(i) = Scalar(c.b - c.a);
        m_period(i) = Scalar(0);
      }
    }
    m_logVolume = m_width.array().log().sum();
  }

  /** Uniform over the box from \a lower to \a upper in \a space. A space with one resizable
   * block is resized to fit.
   *
   * Each upper bound must exceed its lower bound. On a Periodic coordinate the interval may
   * be at most one period wide; on a Bounded coordinate it must lie within the bounds.
   */
  template <typename LowerDerived, typename UpperDerived>
  UniformDensity(Space_ space, const Eigen::MatrixBase<LowerDerived>& lower,
                 const Eigen::MatrixBase<UpperDerived>& upper)
      : m_space{std::move(space)}, m_lower{lower}, m_width{upper - lower} {
    EIGEN_FILTER_ASSERT(upper.size() == lower.size());
    if constexpr (requires { m_space.resize(m_lower.size()); }) {
      if (m_space.size() != m_lower.size()) m_space.resize(m_lower.size());
    }
    EIGEN_FILTER_ASSERT(m_space.size() == m_lower.size() && "the size does not fit the space");
    const auto coordinates{internal::coordinatesOf(m_space)};
    m_period = VectorType::Zero(m_lower.size());
    for (Index i{0}; i < size(); ++i) {
      const Coordinate& c{coordinates[std::size_t(i)]};
      EIGEN_FILTER_ASSERT(Eigen::numext::isfinite(m_lower(i)) && Eigen::numext::isfinite(m_width(i)) &&
                          m_width(i) > Scalar(0) &&
                          "each interval must be finite with its upper bound above its lower");
      if (c.kind == CoordinateKind::Periodic) {
        EIGEN_FILTER_ASSERT(m_width(i) <= Scalar(c.b) && "an interval on a periodic coordinate spans at most a period");
        m_period(i) = Scalar(c.b);
      } else if (c.kind == CoordinateKind::Bounded) {
        EIGEN_FILTER_ASSERT(m_lower(i) >= Scalar(c.a) && Scalar(upper(i)) <= Scalar(c.b) &&
                            "an interval on a bounded coordinate must lie within its bounds");
      }
    }
    m_logVolume = m_width.array().log().sum();
  }

  /** Uniform over the box from \a lower to \a upper in the default space. */
  template <typename LowerDerived, typename UpperDerived>
  UniformDensity(const Eigen::MatrixBase<LowerDerived>& lower, const Eigen::MatrixBase<UpperDerived>& upper)
      : UniformDensity{Space_{}, lower, upper} {}

  [[nodiscard]] Index size() const { return m_lower.size(); }
  [[nodiscard]] const Space_& space() const { return m_space; }

  /** \returns the start \f$ \ell \f$ of each coordinate's interval, as given */
  [[nodiscard]] const VectorType& lower() const { return m_lower; }

  /** \returns the end \f$ \ell + w \f$ of each coordinate's interval, not normalized: on a
   * Periodic coordinate it may lie past the space's range
   */
  [[nodiscard]] VectorType upper() const { return m_lower + m_width; }

  /** \returns the width \f$ w \f$ of each coordinate's interval */
  [[nodiscard]] const VectorType& width() const { return m_width; }

  /** \returns the log of the box's volume, \f$ \sum_i \log w_i \f$ */
  [[nodiscard]] Scalar logVolume() const { return m_logVolume; }

  /** \returns whether \a x lies in the box, taking a Periodic coordinate's interval around the circle */
  template <typename Derived>
  [[nodiscard]] bool contains(const Eigen::MatrixBase<Derived>& x) const {
    static_assert(Derived::IsVectorAtCompileTime, "contains() expects a vector");
    EIGEN_FILTER_ASSERT(x.size() == size());
    for (Index i{0}; i < size(); ++i) {
      Scalar offset{Scalar(x(i)) - m_lower(i)};
      if (m_period(i) > Scalar(0)) offset = internal::wrapValue(offset, 0.0, double(m_period(i)));
      if (!(offset >= Scalar(0) && offset <= m_width(i))) return false;
    }
    return true;
  }

  /** \returns the log of the density at \a x: \f$ -\log \prod_i w_i \f$ inside the box, and
   * \f$ -\infty \f$ outside
   */
  template <typename Derived>
  [[nodiscard]] Scalar logPdf(const Eigen::MatrixBase<Derived>& x) const {
    return contains(x) ? -m_logVolume : -std::numeric_limits<Scalar>::infinity();
  }

  /** \returns the density at \a x */
  template <typename Derived>
  [[nodiscard]] Scalar pdf(const Eigen::MatrixBase<Derived>& x) const {
    return Eigen::numext::exp(logPdf(x));
  }

  /** \returns the mean and covariance: the box's center, normalized, and
   * \f$ \mathrm{diag}(w_i^2 / 12) \f$.
   *
   * For a Periodic coordinate these are tangent-space moments about the interval's center.
   * Over a whole period every point is equally a center, so the reported mean is arbitrary
   * and the variance \f$ p^2 / 12 \f$ only says the coordinate is unknown.
   */
  [[nodiscard]] MomentsType moments() const {
    MomentsType result{size()};
    result.mean() = m_lower + Scalar(0.5) * m_width;
    m_space.normalizeValues(result.mean());
    result.cov() = (m_width.array().square() / Scalar(12)).matrix().asDiagonal();
    return result;
  }

  /** \returns the uniform distribution of the entries \a indices: the box's intervals on them,
   * in the projection of this space. The indices must be distinct.
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
   * \a values: since the coordinates are independent, the marginal of \a target.
   *
   * Values outside the box have zero density, so strictly the conditional is undefined there;
   * this returns the marginal anyway.
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
  template <typename, typename>
  friend class UniformDensity;

  /** The box with the given intervals and periods, already validated. */
  UniformDensity(Space_ space, VectorType lower, VectorType width, VectorType period)
      : m_space{std::move(space)},
        m_lower{std::move(lower)},
        m_width{std::move(width)},
        m_period{std::move(period)},
        m_logVolume{m_width.array().log().sum()} {}

  /** \returns this box's intervals on the coordinates \a target, which form \a space */
  template <typename Target, typename SubSpace>
  [[nodiscard]] UniformDensity<Scalar, SubSpace> restricted(const Target& target, SubSpace space) const {
    using Result = UniformDensity<Scalar, SubSpace>;
    using ResultVector = typename Result::VectorType;
    return Result{std::move(space), ResultVector{m_lower(target)}, ResultVector{m_width(target)},
                  ResultVector{m_period(target)}};
  }

  EIGEN_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;
  VectorType m_lower;
  VectorType m_width;
  VectorType m_period;  ///< a Periodic coordinate's period, or zero
  Scalar m_logVolume{};
};

/** \returns \a n independent draws from \a uniform, one per column: \f$ \ell + w \circ u \f$
 * with \f$ u \f$ uniform on <tt>[0, 1)</tt>, normalized in the space.
 *
 * \sa drawSamplesWithLogPdf(const UniformDensity&, Eigen::Index, G&)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Scalar, Space> drawSamples(const UniformDensity<Scalar, Space>& uniform, Eigen::Index n,
                                                   G& rng) {
  EIGEN_FILTER_ASSERT(n >= 0);
  using SampleSetType = SampleSet<Scalar, Space>;
  SampleSetType samples{uniform.width().asDiagonal() *
                        internal::standardUniform<SampleSetType>(uniform.size(), n, rng)};
  samples.colwise() += uniform.lower();
  uniform.space().normalizeValues(samples);
  return samples;
}

/** \returns \a n independent draws from \a uniform, one per column, with the log-density at
 * each, which is the same everywhere in the box.
 *
 * \sa drawSamples(const UniformDensity&, Eigen::Index, G&)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Space> drawSamplesWithLogPdf(const UniformDensity<Scalar, Space>& uniform,
                                                                     Eigen::Index n, G& rng) {
  return {drawSamples(uniform, n, rng), Eigen::Array<Scalar, Eigen::Dynamic, 1>::Constant(n, -uniform.logVolume())};
}

/** A UniformDensity of dimension \a Size in a linear space, with storage bound \a MaxSize. */
template <typename Scalar, int Size, int MaxSize = Size>
using UniformDensityN = UniformDensity<Scalar, Linear<Size, MaxSize>>;

}  // namespace EigenFilter
