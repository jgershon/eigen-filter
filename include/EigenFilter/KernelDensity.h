#pragma once

#include <cstddef>
#include <expected>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "Gaussian.h"
#include "Macros.h"
#include "ParticleSet.h"
#include "Resampling.h"
#include "SampleSet.h"
#include "Sampling.h"
#include "Weights.h"

namespace EigenFilter {

/** \brief Gaussian kernel density estimate over a ParticleSet.
 *
 * Represents the density
 * \f[ \hat p(x) = \sum_i \hat w_i \, \mathcal{N}(x; x_i, H), \f]
 * where \f$ x_i \f$ are the particles, \f$ \hat w_i \f$ their normalized weights,
 * and \f$ H \f$ the bandwidth (kernel covariance), which is factorized once and
 * shared by every kernel.
 *
 * Created by kernelDensity(). Holds a copy of the ParticleSet view, so the
 * source MutableParticleSet stays locked for the lifetime of the estimate.
 *
 * \sa kernelDensity(), silvermanBandwidth(), class ParticleSet
 */
template <typename Scalar_, int Size_, int MaxSize_ = Size_, typename Space_ = Linear<Size_, MaxSize_>>
class KernelDensity {
 public:
  using SpaceType = Space_;
  using ParticleSetType = ParticleSet<Scalar_, Size_, MaxSize_, Space_>;
  using KernelType = MutableGaussian<Scalar_, Size_, MaxSize_>;
  using MomentsType = typename ParticleSetType::MomentsType;
  using Scalar = typename ParticleSetType::Scalar;
  using Index = typename ParticleSetType::Index;
  using VectorType = typename ParticleSetType::VectorType;
  using MatrixType = typename ParticleSetType::MatrixType;
  using SampleSetType = typename ParticleSetType::SampleSetType;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};

  /** \returns a kernel density estimate with bandwidth \a bandwidth, or a
   *          GaussianError if \a bandwidth is not positive definite or not finite.
   *          Only the lower triangle of \a bandwidth is read.
   */
  template <typename Derived>
  [[nodiscard]] static std::expected<KernelDensity, GaussianError> create(const ParticleSetType& particles,
                                                                          const Eigen::MatrixBase<Derived>& bandwidth) {
    KernelType kernel{VectorType::Zero(particles.particleSize()), bandwidth};
    if (auto result{kernel.factorize()}; !result) return std::unexpected{result.error()};
    return KernelDensity{particles, std::move(kernel)};
  }

  /** \returns the number of particles */
  [[nodiscard]] Index size() const { return m_particles.size(); }
  [[nodiscard]] Index particleSize() const { return m_particles.particleSize(); }
  [[nodiscard]] const Space_& space() const { return m_particles.space(); }
  [[nodiscard]] const ParticleSetType& particles() const { return m_particles; }

  /** \returns the kernel covariance \f$ H \f$ */
  [[nodiscard]] const MatrixType& bandwidth() const { return m_kernel.cov(); }

  /** \returns a view of the zero-mean kernel \f$ \mathcal{N}(0, H) \f$. It must not outlive this estimate. */
  [[nodiscard]] typename KernelType::GaussianType kernel() const { return *m_kernel.toGaussian(); }

  /** \returns the mean and covariance of the estimate: the particles' mean, and
   * their weighted covariance plus \f$ H \f$.
   */
  [[nodiscard]] MomentsType moments() const {
    MomentsType result{m_particles.moments()};
    result.cov() += m_kernel.cov().template selfadjointView<Eigen::Lower>().toDenseMatrix();
    return result;
  }

  /** \returns the log of the density at \a x
   *
   * Costs one triangular solve against all particles at once.
   *
   * \sa pdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar logPdf(const Eigen::MatrixBase<Derived>& x) const {
    static_assert(Derived::IsVectorAtCompileTime, "logPdf() expects a vector");
    EIGEN_FILTER_ASSERT(x.size() == particleSize());
    const auto kernel{this->kernel()};
    SampleSetType residuals{m_particles.states().colwise() - x.derived()};
    space().normalizeDifferences(residuals);
    kernel.llt().matrixL().solveInPlace(residuals);
    internal::LogSumExp<Scalar> sum;
    for (Index i{0}; i < size(); ++i) {
      sum.add(m_particles.logWeight(i) - Scalar(0.5) * residuals.col(i).squaredNorm());
    }
    return sum.value() - m_particles.logTotalWeight() + kernel.logPeak();
  }

  /** \returns the density \f$ \hat p(x) \f$ at \a x
   *
   * \warning Underflows easily in high dimensions; prefer logPdf().
   *
   * \sa logPdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar pdf(const Eigen::MatrixBase<Derived>& x) const {
    return Eigen::numext::exp(logPdf(x));
  }

  /** \returns a copy of the distribution of the entries \a target given that the
   * entries \a given equal \a values, as a MutableParticleSet storing the exact
   * conditional bandwidth.
   *
   * The conditional of a Gaussian KDE is again a Gaussian KDE: each center becomes
   * \f$ x_{a,i} + K (v - x_{b,i}) \f$, each weight gains \f$ \mathcal{N}(v; x_{b,i}, H_{bb}) \f$,
   * and the shared bandwidth becomes \f$ H_{aa} - K H_{ba} \f$, with \f$ K = H_{ab} H_{bb}^{-1} \f$.
   * The weights are not normalized. Building a KDE from the result with its stored
   * bandwidth reproduces this conditional exactly; Silverman's rule would re-smooth it.
   */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(m_particles, bandwidth(), target, given, values);
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(m_particles, bandwidth(), given, values);
  }

 private:
  KernelDensity(const ParticleSetType& particles, KernelType kernel)
      : m_particles{particles}, m_kernel{std::move(kernel)} {}

  ParticleSetType m_particles;
  KernelType m_kernel;
};

/** \returns a kernel density estimate over \a particles with bandwidth \a bandwidth,
 *          or a GaussianError if \a bandwidth is not positive definite or not finite.
 */
template <typename Scalar, int Size, int MaxSize, typename Space, typename Derived>
[[nodiscard]] std::expected<KernelDensity<Scalar, Size, MaxSize, Space>, GaussianError> kernelDensity(
    const ParticleSet<Scalar, Size, MaxSize, Space>& particles, const Eigen::MatrixBase<Derived>& bandwidth) {
  return KernelDensity<Scalar, Size, MaxSize, Space>::create(particles, bandwidth);
}

/** \returns a kernel density estimate over \a particles with their kdeBandwidth(): the
 *          stored bandwidth if there is one, otherwise silvermanBandwidth(). Fails
 *          with a GaussianError if that bandwidth is not positive definite (for
 *          example, Silverman's rule when every particle has the same state).
 */
template <typename Scalar, int Size, int MaxSize, typename Space>
[[nodiscard]] std::expected<KernelDensity<Scalar, Size, MaxSize, Space>, GaussianError> kernelDensity(
    const ParticleSet<Scalar, Size, MaxSize, Space>& particles) {
  return kernelDensity(particles, particles.kdeBandwidth());
}

/** \returns \a n draws from \a kde, one per column.
 *
 * Each draw picks a particle in proportion to its weight using \a resampler, then
 * adds kernel noise \f$ L z \f$ with \f$ z \sim \mathcal{N}(0, I) \f$ and
 * \f$ H = L L^T \f$. The default MultinomialResampler makes the draws independent;
 * a lower-variance resampler such as SystematicResampler spreads them more evenly
 * over the particles, as in a regularized particle filter.
 *
 * \sa drawSamplesWithLogPdf(const KernelDensity&, Eigen::Index, G&, const R&), \ref Resamplers
 */
template <typename Scalar, int Size, int MaxSize, typename Space, std::uniform_random_bit_generator G,
          typename R = MultinomialResampler>
  requires Resampler<R, Scalar, G>
[[nodiscard]] SampleSet<Scalar, Size, MaxSize> drawSamples(const KernelDensity<Scalar, Size, MaxSize, Space>& kde,
                                                           Eigen::Index n, G& rng, const R& resampler = {}) {
  EIGEN_FILTER_ASSERT(n >= 0);
  using SampleSetType = SampleSet<Scalar, Size, MaxSize>;
  const auto& particles{kde.particles()};
  std::vector<Scalar> weights(std::size_t(particles.size()));
  for (Eigen::Index i{0}; i < particles.size(); ++i) {
    weights[std::size_t(i)] = Eigen::numext::exp(particles.logWeight(i) - particles.logTotalWeight());
  }
  std::vector<Eigen::Index> ancestors(static_cast<std::size_t>(n));
  resampler(std::span<const Scalar>{weights}, std::span<Eigen::Index>{ancestors}, rng);

  SampleSetType samples{kde.kernel().llt().matrixL() *
                        internal::standardNormal<SampleSetType>(kde.particleSize(), n, rng)};
  for (Eigen::Index j{0}; j < n; ++j) samples.col(j) += particles.state(ancestors[std::size_t(j)]);
  kde.space().normalizeValues(samples);
  return samples;
}

/** \returns \a n draws from \a kde, one per column, with the log-density of \a kde at each.
 *
 * Draws exactly as drawSamples() does. Unlike the Gaussian case the log-densities
 * are not free: each costs a full KernelDensity::logPdf() evaluation, so the
 * total is \f$ O(n \cdot \mathrm{size}() \cdot d^2) \f$.
 *
 * \sa drawSamples(const KernelDensity&, Eigen::Index, G&, const R&)
 */
template <typename Scalar, int Size, int MaxSize, typename Space, std::uniform_random_bit_generator G,
          typename R = MultinomialResampler>
  requires Resampler<R, Scalar, G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Size, MaxSize> drawSamplesWithLogPdf(
    const KernelDensity<Scalar, Size, MaxSize, Space>& kde, Eigen::Index n, G& rng, const R& resampler = {}) {
  SamplesWithLogPdf<Scalar, Size, MaxSize> result{drawSamples(kde, n, rng, resampler), {}};
  result.logPdf.resize(n);
  for (Eigen::Index j{0}; j < n; ++j) result.logPdf(j) = kde.logPdf(result.samples.col(j));
  return result;
}

}  // namespace EigenFilter
