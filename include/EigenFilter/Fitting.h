#pragma once

#include <array>
#include <cstddef>
#include <expected>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>

#include "GaussianMixture.h"
#include "Macros.h"
#include "Moments.h"
#include "ParticleSet.h"
#include "Resampling.h"
#include "SampleSet.h"
#include "Space.h"
#include "Statistics.h"
#include "Weights.h"

namespace EigenFilter {

/** Reasons fitMixture() can fail. */
enum class FitError {
  Degenerate,  ///< the source has no spread in some dimension, e.g. every particle has the same state
};

/** Options for fitMixture(). */
template <typename Scalar>
struct FitOptions {
  /** Maximum number of EM iterations (E-steps). */
  int maxIterations{100};
  /** Stop when an iteration improves the weighted log-likelihood by at most this. */
  Scalar tolerance{Scalar(1e-8)};
  /** Covariance floor, relative to the source's variance in each dimension: every
   * component covariance gets \f$ \epsilon \Sigma_{ii} \f$ added to its \f$ i \f$-th variance.
   * Scaling each dimension by its own variance keeps the floor consistent when the
   * dimensions have different units. Keeps components that capture duplicate particles
   * from collapsing.
   */
  Scalar regularization{Scalar(1e-6)};
  /** Components whose normalized log-weight falls below this are dropped. */
  Scalar minLogWeight{Scalar(-20)};
};

/** Result of fitMixture(), in \a Space.
 *
 * \sa MixtureFitN
 */
template <typename Scalar, typename Space>
struct MixtureFit {
  MutableGaussianMixture<Scalar, Space> mixture;  ///< fitted mixture, weights normalized to one
  Scalar logLikelihood;                           ///< weighted log-likelihood of the source under \c mixture
  int iterations;                                 ///< E-steps performed
  bool converged;                                 ///< whether \c tolerance was reached before \c maxIterations
};

/** MixtureFit of dimension \a Size in a linear space, with storage bound \a MaxSize. */
template <typename Scalar, int Size, int MaxSize = Size>
using MixtureFitN = MixtureFit<Scalar, Linear<Size, MaxSize>>;

namespace internal {

/** Weighted points to fit, each with an optional covariance (a particle has none). */
template <typename Scalar, typename Space>
struct FitSource {
  using MatrixType = typename Moments<Scalar, Space>::MatrixType;
  SampleSet<Scalar, Space> points;
  Eigen::Array<Scalar, Eigen::Dynamic, 1> weights;  ///< normalized linear weights
  std::vector<MatrixType> covs;                     ///< per-point covariances; empty for particles
  std::vector<MatrixType> covFactors;               ///< lower Cholesky factors of \c covs
  Space space;
};

template <typename Scalar, typename Space>
[[nodiscard]] FitSource<Scalar, Space> fitSource(const ParticleSet<Scalar, Space>& particles) {
  return {particles.states(), particles.normalizedWeights(), {}, {}, particles.space()};
}

template <typename Scalar, typename Space>
[[nodiscard]] FitSource<Scalar, Space> fitSource(const GaussianMixture<Scalar, Space>& mixture) {
  using MatrixType = typename FitSource<Scalar, Space>::MatrixType;
  FitSource<Scalar, Space> source{SampleSet<Scalar, Space>{mixture.componentSize(), mixture.size()},
                                  Eigen::Array<Scalar, Eigen::Dynamic, 1>{mixture.size()},
                                  {},
                                  {},
                                  mixture.space()};
  source.covs.reserve(std::size_t(mixture.size()));
  source.covFactors.reserve(std::size_t(mixture.size()));
  Eigen::Index i{0};
  for (const auto& component : mixture) {
    const auto gaussian{component.gaussian.toGaussian()};
    source.points.col(i) = gaussian->mean();
    source.weights(i) = Eigen::numext::exp(component.logWeight - mixture.logTotalWeight());
    const MatrixType factor{gaussian->llt().matrixL()};
    source.covs.push_back(factor * factor.transpose());
    source.covFactors.push_back(factor);
    ++i;
  }
  return source;
}

/** Weighted EM for a Gaussian mixture over weighted points that may carry covariances.
 *
 * For points with covariances \f$ S_i \f$ this is hierarchical EM: the E-step uses the
 * expected log-likelihood \f$ \log \mathcal{N}(x_i; \mu_k, \Sigma_k) - \tfrac12 \mathrm{tr}(\Sigma_k^{-1} S_i) \f$,
 * and the M-step moment-matches. For particles (\f$ S_i = 0 \f$) it is ordinary weighted EM.
 * Residuals are normalized by the source's space and means are tangent-space means.
 */
template <typename Scalar, typename Space>
class MixtureFitter {
 public:
  using MomentsType = Moments<Scalar, Space>;
  using VectorType = typename MomentsType::VectorType;
  using MatrixType = typename MomentsType::MatrixType;
  using Points = SampleSet<Scalar, Space>;
  using Weights = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
  using Responsibilities = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Source = FitSource<Scalar, Space>;
  using Result = MixtureFit<Scalar, Space>;

  MixtureFitter(const Source& source, const FitOptions<Scalar>& options) : m_source{source}, m_options{options} {
    EIGEN_FILTER_ASSERT(options.maxIterations >= 1);
  }

  /** Seeds up to \a k components by weighted k-means++ over \a candidates, then runs EM. */
  template <std::uniform_random_bit_generator G>
  [[nodiscard]] std::expected<Result, FitError> run(const Points& candidates, const Weights& candidateWeights,
                                                    Eigen::Index k, G& rng) {
    EIGEN_FILTER_ASSERT(k >= 1);
    if (!prepareWhitening()) return std::unexpected{FitError::Degenerate};
    Responsibilities responsibilities{initialAssignment(seed(candidates, candidateWeights, k, rng))};
    mStep(responsibilities);

    Scalar previous{-Eigen::NumTraits<Scalar>::infinity()};
    for (int iteration{1};; ++iteration) {
      const auto logLikelihood{eStep(responsibilities)};
      if (!logLikelihood) return std::unexpected{logLikelihood.error()};
      const bool converged{*logLikelihood - previous <= m_options.tolerance};
      if (converged || iteration >= m_options.maxIterations) return result(*logLikelihood, iteration, converged);
      previous = *logLikelihood;
      mStep(responsibilities);
    }
  }

 private:
  /** Factorizes the source's overall covariance, which defines the seeding metric and the floor. */
  [[nodiscard]] bool prepareWhitening() {
    const MomentsType moments{
        weightedMomentsIn<MomentsType>(m_source.space, m_source.points, m_source.weights.log(), Scalar(0))};
    MatrixType total{moments.cov()};
    for (std::size_t i{0}; i < m_source.covs.size(); ++i) total += m_source.weights(Eigen::Index(i)) * m_source.covs[i];
    const VectorType variances{total.diagonal()};
    // A dimension of identical points leaves only rounding noise, of order epsilon² times
    // its second moment; with no spread there is no scale to fit or to floor against.
    const auto secondMoments{(variances.array() + moments.mean().array().square()).eval()};
    if (!(variances.array() > Eigen::NumTraits<Scalar>::epsilon() * secondMoments).all() || !variances.allFinite()) {
      return false;
    }
    m_floor = m_options.regularization * variances;
    total.diagonal() += m_floor;
    m_whitening.compute(total);
    return m_whitening.info() == Eigen::Success;
  }

  /** \returns the squared Mahalanobis distance, under the source's overall covariance,
   * of each column of \a points from \a center, with differences normalized by the space.
   */
  template <typename Derived>
  [[nodiscard]] Weights distancesFrom(const Points& points, const Eigen::MatrixBase<Derived>& center) const {
    Points differences{points.colwise() - center.derived()};
    m_source.space.normalizeDifferences(differences);
    m_whitening.matrixL().solveInPlace(differences);
    return differences.colwise().squaredNorm().transpose().array();
  }

  /** Weighted k-means++: each seed is drawn with probability proportional to weight
   * times squared distance to the nearest seed so far, so points on top of a chosen seed
   * (stacked components, duplicate particles) are essentially never chosen again.
   * Stops early if no remaining candidate is at positive distance.
   *
   * \returns the seed points
   */
  template <typename G>
  [[nodiscard]] Points seed(const Points& candidates, const Weights& weights, Eigen::Index k, G& rng) const {
    const Eigen::Index m{candidates.cols()};
    Weights distances{Weights::Constant(m, Eigen::NumTraits<Scalar>::infinity())};
    std::vector<Scalar> probabilities(weights.begin(), weights.end());
    std::vector<Eigen::Index> chosen;
    while (true) {
      std::array<Eigen::Index, 1> pick{};
      const Scalar u{uniform01<Scalar>(rng)};
      invertCumulative(std::span<const Scalar>{probabilities}, std::span<Eigen::Index>{pick},
                       [u](std::size_t) { return u; });
      chosen.push_back(pick[0]);
      if (Eigen::Index(chosen.size()) == k) break;
      distances = distances.min(distancesFrom(candidates, candidates.col(pick[0])));
      Scalar total{0};
      for (Eigen::Index i{0}; i < m; ++i) {
        probabilities[std::size_t(i)] = weights(i) * distances(i);
        total += probabilities[std::size_t(i)];
      }
      if (!(total > Scalar(0))) break;
    }
    Points seeds{candidates.rows(), Eigen::Index(chosen.size())};
    for (std::size_t s{0}; s < chosen.size(); ++s) seeds.col(Eigen::Index(s)) = candidates.col(chosen[s]);
    return seeds;
  }

  /** \returns hard responsibilities assigning each source point to its nearest seed. */
  [[nodiscard]] Responsibilities initialAssignment(const Points& seeds) const {
    const Eigen::Index n{m_source.points.cols()};
    Weights nearestDistance{Weights::Constant(n, Eigen::NumTraits<Scalar>::infinity())};
    Eigen::Array<Eigen::Index, Eigen::Dynamic, 1> nearest{Eigen::Array<Eigen::Index, Eigen::Dynamic, 1>::Zero(n)};
    for (Eigen::Index s{0}; s < seeds.cols(); ++s) {
      const Weights distances{distancesFrom(m_source.points, seeds.col(s))};
      for (Eigen::Index i{0}; i < n; ++i) {
        if (distances(i) < nearestDistance(i)) {
          nearestDistance(i) = distances(i);
          nearest(i) = s;
        }
      }
    }
    Responsibilities responsibilities{Responsibilities::Zero(seeds.cols(), n)};
    for (Eigen::Index i{0}; i < n; ++i) responsibilities(nearest(i), i) = Scalar(1);
    return responsibilities;
  }

  /** Moment-matches each component to its responsibility-weighted points, dropping light ones. */
  void mStep(const Responsibilities& responsibilities) {
    m_logWeights.clear();
    m_means.clear();
    m_covs.clear();
    for (Eigen::Index k{0}; k < responsibilities.rows(); ++k) {
      const Weights w{m_source.weights * responsibilities.row(k).transpose()};
      const Scalar mass{w.sum()};
      if (!(mass > Scalar(0)) || Eigen::numext::log(mass) < m_options.minLogWeight) continue;
      MomentsType moments{
          weightedMomentsIn<MomentsType>(m_source.space, m_source.points, w.log(), Eigen::numext::log(mass))};
      for (std::size_t i{0}; i < m_source.covs.size(); ++i) {
        if (w(Eigen::Index(i)) > Scalar(0)) moments.cov() += (w(Eigen::Index(i)) / mass) * m_source.covs[i];
      }
      moments.cov().diagonal() += m_floor;
      m_logWeights.push_back(Eigen::numext::log(mass));
      m_means.push_back(moments.mean());
      m_covs.push_back(moments.cov());
    }
  }

  /** Computes responsibilities for the current components, dropping any whose covariance
   * does not factorize.
   *
   * \returns the weighted log-likelihood of the source, or FitError::Degenerate if no
   *          component survives
   */
  [[nodiscard]] std::expected<Scalar, FitError> eStep(Responsibilities& responsibilities) {
    std::vector<Eigen::LLT<MatrixType>> factors;
    std::vector<Scalar> logDeterminants;
    std::size_t out{0};
    for (std::size_t k{0}; k < m_covs.size(); ++k) {
      Eigen::LLT<MatrixType> llt{m_covs[k]};
      const Scalar logDeterminant{Scalar(2) * llt.matrixLLT().diagonal().array().log().sum()};
      if (llt.info() != Eigen::Success || !Eigen::numext::isfinite(logDeterminant)) continue;
      m_logWeights[out] = m_logWeights[k];
      m_means[out] = m_means[k];
      m_covs[out] = m_covs[k];
      factors.push_back(std::move(llt));
      logDeterminants.push_back(logDeterminant);
      ++out;
    }
    m_logWeights.resize(out);
    m_means.resize(out);
    m_covs.resize(out);
    if (out == 0) return std::unexpected{FitError::Degenerate};

    const Eigen::Index n{m_source.points.cols()};
    const Scalar log2Pi{Eigen::numext::log(Scalar(2) * Scalar(EIGEN_PI))};
    const Scalar dimension{Scalar(m_source.points.rows())};
    Responsibilities logTerms{Eigen::Index(out), n};
    for (std::size_t k{0}; k < out; ++k) {
      Points residuals{m_source.points.colwise() - m_means[k]};
      m_source.space.normalizeDifferences(residuals);
      factors[k].matrixL().solveInPlace(residuals);
      logTerms.row(Eigen::Index(k)) = m_logWeights[k] - Scalar(0.5) * (dimension * log2Pi + logDeterminants[k]) -
                                      Scalar(0.5) * residuals.colwise().squaredNorm().array();
      for (std::size_t i{0}; i < m_source.covFactors.size(); ++i) {
        const MatrixType solved{factors[k].matrixL().solve(m_source.covFactors[i])};
        logTerms(Eigen::Index(k), Eigen::Index(i)) -= Scalar(0.5) * solved.squaredNorm();
      }
    }

    responsibilities.resize(logTerms.rows(), n);
    Scalar logLikelihood{0};
    for (Eigen::Index i{0}; i < n; ++i) {
      LogSumExp<Scalar> sum;
      for (Eigen::Index k{0}; k < logTerms.rows(); ++k) sum.add(logTerms(k, i));
      const Scalar logNormalizer{sum.value()};
      // A point every component finds impossible (or with no weight) takes no part in the M-step.
      if (!Eigen::numext::isfinite(logNormalizer) || !(m_source.weights(i) > Scalar(0))) {
        responsibilities.col(i).setZero();
        continue;
      }
      responsibilities.col(i) = (logTerms.col(i) - logNormalizer).exp();
      logLikelihood += m_source.weights(i) * logNormalizer;
    }
    return logLikelihood;
  }

  [[nodiscard]] Result result(Scalar logLikelihood, int iterations, bool converged) const {
    MutableGaussianMixture<Scalar, Space> mixture{m_source.space, m_source.points.rows()};
    mixture.reserve(Eigen::Index(m_means.size()));
    for (std::size_t k{0}; k < m_means.size(); ++k) mixture.add(m_logWeights[k], m_means[k], m_covs[k]);
    return {std::move(mixture), logLikelihood, iterations, converged};
  }

  const Source& m_source;
  FitOptions<Scalar> m_options;
  Eigen::LLT<MatrixType> m_whitening;
  VectorType m_floor;
  std::vector<Scalar> m_logWeights;
  std::vector<VectorType> m_means;
  std::vector<MatrixType> m_covs;
};

}  // namespace internal

/** \returns a mixture of at most \a k Gaussians fitted to \a particles by weighted EM.
 *
 * Seeds by weighted k-means++ over the particles, using the Mahalanobis distance under
 * the particles' covariance, so duplicate particles are not seeded twice. Fewer than
 * \a k components result if fewer distinct particles exist or components die (see
 * FitOptions::minLogWeight). The result is in the particles' space. Costs
 * O(iterations · size() · k · d²).
 *
 * \returns the fit, or FitError::Degenerate if the particles have no spread in some dimension
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Scalar, Space>, FitError> fitMixture(const ParticleSet<Scalar, Space>& particles,
                                                                            Eigen::Index k, G& rng,
                                                                            const FitOptions<Scalar>& options = {}) {
  const auto source{internal::fitSource(particles)};
  internal::MixtureFitter<Scalar, Space> fitter{source, options};
  return fitter.run(source.points, source.weights, k, rng);
}

/** \returns a mixture of at most \a k Gaussians fitted to \a mixture by hierarchical EM.
 *
 * Reduces the mixture without sampling: each component is a weighted point carrying its
 * covariance. Seeds by weighted k-means++ over the component means, so stacked
 * components are not seeded twice. The result is in the mixture's space. Costs
 * O(iterations · size() · k · d³), with no pairwise pass over the components.
 *
 * \returns the fit, or FitError::Degenerate if the mixture has no spread in some dimension
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Scalar, Space>, FitError> fitMixture(
    const GaussianMixture<Scalar, Space>& mixture, Eigen::Index k, G& rng, const FitOptions<Scalar>& options = {}) {
  const auto source{internal::fitSource(mixture)};
  internal::MixtureFitter<Scalar, Space> fitter{source, options};
  return fitter.run(source.points, source.weights, k, rng);
}

/** \returns a mixture of at most \a k Gaussians fitted to \a particles by weighted EM,
 * seeded by weighted k-means++ over the component means of \a seedsFrom, which must be
 * in the same space.
 *
 * Useful when the particles were drawn from \a seedsFrom: its components summarize the
 * particles and are far fewer, so seeding is cheaper and starts EM closer to the answer.
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] std::expected<MixtureFit<Scalar, Space>, FitError> fitMixture(
    const ParticleSet<Scalar, Space>& particles, const GaussianMixture<Scalar, Space>& seedsFrom, Eigen::Index k,
    G& rng, const FitOptions<Scalar>& options = {}) {
  EIGEN_FILTER_ASSERT(particles.space() == seedsFrom.space() && "the particles and seeds must share a space");
  const auto source{internal::fitSource(particles)};
  const auto seeds{internal::fitSource(seedsFrom)};
  internal::MixtureFitter<Scalar, Space> fitter{source, options};
  return fitter.run(seeds.points, seeds.weights, k, rng);
}

}  // namespace EigenFilter
