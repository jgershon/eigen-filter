#pragma once

#include <cstddef>
#include <expected>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "Gaussian.h"
#include "Indexing.h"
#include "Macros.h"
#include "ParticleSet.h"
#include "Resampling.h"
#include "SampleSet.h"
#include "Sampling.h"
#include "Statistics.h"
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
 * Created by ParticleSet::toDensity(). Holding a KernelDensity locks the
 * source set; copies share the lock and the last one to be destroyed releases it. A
 * KernelDensity must not outlive its source.
 *
 * \sa ParticleSet::toDensity(), silvermanBandwidth(), KernelDensityN
 */
template <typename Scalar_, typename Space_>
class KernelDensity {
 public:
  using SpaceType = Space_;
  using ParticleSetType = ParticleSet<Scalar_, Space_>;
  static constexpr int SizeAtCompileTime{ParticleSetType::SizeAtCompileTime};
  static constexpr int MaxSizeAtCompileTime{ParticleSetType::MaxSizeAtCompileTime};
  /** The zero-mean kernel, in the linear tangent space of \a Space_. */
  using KernelType = GaussianN<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime>;
  using MomentsType = typename ParticleSetType::MomentsType;
  using Scalar = typename ParticleSetType::Scalar;
  using Index = typename ParticleSetType::Index;
  using VectorType = typename ParticleSetType::VectorType;
  using MatrixType = typename ParticleSetType::MatrixType;
  using SampleSetType = typename ParticleSetType::SampleSetType;

  /** \returns the number of particles */
  [[nodiscard]] Index size() const { return particles().size(); }
  [[nodiscard]] Index particleSize() const { return particles().particleSize(); }
  [[nodiscard]] const Space_& space() const { return particles().space(); }

  /** \returns the particles, which stay locked while this estimate is alive */
  [[nodiscard]] const ParticleSetType& particles() const { return m_particles.get(); }

  /** \returns the log of the sum of the (unnormalized) particle weights */
  [[nodiscard]] Scalar logTotalWeight() const { return m_logTotalWeight; }

  /** \returns the kernel covariance \f$ H \f$ */
  [[nodiscard]] const MatrixType& bandwidth() const { return m_kernel.cov(); }

  /** \returns a view of the zero-mean kernel \f$ \mathcal{N}(0, H) \f$. It must not outlive this estimate. */
  [[nodiscard]] typename KernelType::DensityType kernel() const { return *m_kernel.toDensity(); }

  /** \returns the mean and covariance of the estimate: the particles' mean, and
   * their weighted covariance plus \f$ H \f$.
   */
  [[nodiscard]] MomentsType moments() const {
    MomentsType result{internal::weightedMomentsIn<MomentsType>(space(), particles().states(), particles().logWeights(),
                                                                m_logTotalWeight)};
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
    SampleSetType residuals{particles().states().colwise() - x.derived()};
    space().normalizeDifferences(residuals);
    kernel.llt().matrixL().solveInPlace(residuals);
    internal::LogSumExp<Scalar> sum;
    for (Index i{0}; i < size(); ++i) {
      sum.add(particles().logWeight(i) - Scalar(0.5) * residuals.col(i).squaredNorm());
    }
    return sum.value() - m_logTotalWeight + kernel.logPeak();
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
   * entries \a given equal \a values, as a ParticleSet storing the exact
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
    return internal::conditionalParticles(particles(), bandwidth(), target, given, values);
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(particles(), bandwidth(), given, values);
  }

 private:
  friend ParticleSetType;

  /** \returns an estimate over \a particles, whose weights total \f$ e^{\mathrm{logTotalWeight}} \f$,
   *          or a GaussianError if \a bandwidth is not positive definite or not finite.
   *          Only the lower triangle of \a bandwidth is read.
   */
  template <typename Derived>
  [[nodiscard]] static std::expected<KernelDensity, GaussianError> create(const ParticleSetType& particles,
                                                                          Scalar logTotalWeight,
                                                                          const Eigen::MatrixBase<Derived>& bandwidth) {
    KernelType kernel{VectorType::Zero(particles.particleSize()), bandwidth};
    if (auto result{kernel.factorize()}; !result) return std::unexpected{result.error()};
    return KernelDensity{particles, logTotalWeight, std::move(kernel)};
  }

  KernelDensity(const ParticleSetType& particles, Scalar logTotalWeight, KernelType kernel)
      : m_particles{particles}, m_logTotalWeight{logTotalWeight}, m_kernel{std::move(kernel)} {}

  internal::Lock<ParticleSetType> m_particles;
  Scalar m_logTotalWeight;
  KernelType m_kernel;
};

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
template <typename Scalar, typename Space, std::uniform_random_bit_generator G, typename R = MultinomialResampler>
  requires Resampler<R, Scalar, G>
[[nodiscard]] SampleSet<Scalar, Space> drawSamples(const KernelDensity<Scalar, Space>& kde, Eigen::Index n, G& rng,
                                                   const R& resampler = {}) {
  EIGEN_FILTER_ASSERT(n >= 0);
  using SampleSetType = SampleSet<Scalar, Space>;
  const auto& particles{kde.particles()};
  std::vector<Scalar> weights(std::size_t(particles.size()));
  for (Eigen::Index i{0}; i < particles.size(); ++i) {
    weights[std::size_t(i)] = Eigen::numext::exp(particles.logWeight(i) - kde.logTotalWeight());
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
template <typename Scalar, typename Space, std::uniform_random_bit_generator G, typename R = MultinomialResampler>
  requires Resampler<R, Scalar, G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Space> drawSamplesWithLogPdf(const KernelDensity<Scalar, Space>& kde,
                                                                     Eigen::Index n, G& rng, const R& resampler = {}) {
  SamplesWithLogPdf<Scalar, Space> result{drawSamples(kde, n, rng, resampler), {}};
  result.logPdf.resize(n);
  for (Eigen::Index j{0}; j < n; ++j) result.logPdf(j) = kde.logPdf(result.samples.col(j));
  return result;
}

/** A KernelDensity of dimension \a Size in a linear space, with storage bound \a MaxSize. */
template <typename Scalar, int Size, int MaxSize = Size>
using KernelDensityN = KernelDensity<Scalar, Linear<Size, MaxSize>>;

}  // namespace EigenFilter
