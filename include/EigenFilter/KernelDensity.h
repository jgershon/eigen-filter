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
template <typename Scalar_, int Size_, int MaxSize_ = Size_>
class KernelDensity {
 public:
  using ParticleSetType = ParticleSet<Scalar_, Size_, MaxSize_>;
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

 private:
  KernelDensity(const ParticleSetType& particles, KernelType kernel)
      : m_particles{particles}, m_kernel{std::move(kernel)} {}

  ParticleSetType m_particles;
  KernelType m_kernel;
};

/** \returns Silverman's rule-of-thumb bandwidth
 * \f[ H = \left(\frac{4}{N_\mathrm{eff}\,(d + 2)}\right)^{2/(d+4)} \Sigma, \f]
 * where \f$ \Sigma \f$ is the particles' weighted covariance, \f$ d \f$ their
 * dimension, and \f$ N_\mathrm{eff} \f$ their effective sample size.
 *
 * Near-optimal for roughly Gaussian particle clouds; scale it down for multimodal ones.
 */
template <typename Scalar, int Size, int MaxSize>
[[nodiscard]] typename ParticleSet<Scalar, Size, MaxSize>::MatrixType silvermanBandwidth(
    const ParticleSet<Scalar, Size, MaxSize>& particles) {
  const Scalar d{Scalar(particles.particleSize())};
  const Scalar factor{
      Eigen::numext::pow(Scalar(4) / (particles.effectiveSampleSize() * (d + Scalar(2))), Scalar(2) / (d + Scalar(4)))};
  return factor * particles.moments().cov();
}

/** \returns a kernel density estimate over \a particles with bandwidth \a bandwidth,
 *          or a GaussianError if \a bandwidth is not positive definite or not finite.
 */
template <typename Scalar, int Size, int MaxSize, typename Derived>
[[nodiscard]] std::expected<KernelDensity<Scalar, Size, MaxSize>, GaussianError> kernelDensity(
    const ParticleSet<Scalar, Size, MaxSize>& particles, const Eigen::MatrixBase<Derived>& bandwidth) {
  return KernelDensity<Scalar, Size, MaxSize>::create(particles, bandwidth);
}

/** \returns a kernel density estimate over \a particles with silvermanBandwidth(),
 *          or a GaussianError if that bandwidth is not positive definite (for
 *          example, when every particle has the same state).
 */
template <typename Scalar, int Size, int MaxSize>
[[nodiscard]] std::expected<KernelDensity<Scalar, Size, MaxSize>, GaussianError> kernelDensity(
    const ParticleSet<Scalar, Size, MaxSize>& particles) {
  return kernelDensity(particles, silvermanBandwidth(particles));
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
template <typename Scalar, int Size, int MaxSize, std::uniform_random_bit_generator G,
          typename R = MultinomialResampler>
  requires Resampler<R, Scalar, G>
[[nodiscard]] SampleSet<Scalar, Size, MaxSize> drawSamples(const KernelDensity<Scalar, Size, MaxSize>& kde,
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
template <typename Scalar, int Size, int MaxSize, std::uniform_random_bit_generator G,
          typename R = MultinomialResampler>
  requires Resampler<R, Scalar, G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Size, MaxSize> drawSamplesWithLogPdf(
    const KernelDensity<Scalar, Size, MaxSize>& kde, Eigen::Index n, G& rng, const R& resampler = {}) {
  SamplesWithLogPdf<Scalar, Size, MaxSize> result{drawSamples(kde, n, rng, resampler), {}};
  result.logPdf.resize(n);
  for (Eigen::Index j{0}; j < n; ++j) result.logPdf(j) = kde.logPdf(result.samples.col(j));
  return result;
}

}  // namespace EigenFilter
