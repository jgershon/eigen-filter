#include <EigenFilter/Fitting.h>
#include <EigenFilter/Sampling.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <vector>

using EigenFilter::FitError;
using EigenFilter::FitOptions;
using EigenFilter::MutableGaussian;
using EigenFilter::MutableGaussianMixture;
using EigenFilter::MutableParticleSet;

namespace {

using MGM2 = MutableGaussianMixture<double, 2>;
using MPS2 = MutableParticleSet<double, 2>;

const std::array<Eigen::Vector2d, 3> trueMeans{Eigen::Vector2d{-6.0, 0.0}, Eigen::Vector2d{0.0, 6.0},
                                               Eigen::Vector2d{6.0, 0.0}};
const std::array<Eigen::Matrix2d, 3> trueCovs{Eigen::Matrix2d{{1.0, 0.3}, {0.3, 0.5}},
                                              Eigen::Matrix2d{{0.6, -0.2}, {-0.2, 1.2}},
                                              Eigen::Matrix2d{{1.5, 0.0}, {0.0, 0.4}}};
const std::array<double, 3> trueWeights{0.2, 0.3, 0.5};

MGM2 makeTrueMixture() {
  MGM2 mixture;
  for (std::size_t c{0}; c < 3; ++c) mixture.add(std::log(trueWeights[c]), trueMeans[c], trueCovs[c]);
  return mixture;
}

// Expects \a fitted to have one component close to each true component.
void expectRecovers(const MGM2& fitted, double meanTolerance, double weightTolerance, double covTolerance) {
  ASSERT_EQ(fitted.size(), 3);
  for (std::size_t c{0}; c < 3; ++c) {
    auto nearest{fitted.begin()};
    for (auto it{fitted.begin()}; it != fitted.end(); ++it) {
      if ((it->gaussian.mean() - trueMeans[c]).norm() < (nearest->gaussian.mean() - trueMeans[c]).norm()) nearest = it;
    }
    EXPECT_LT((nearest->gaussian.mean() - trueMeans[c]).norm(), meanTolerance) << "component " << c;
    EXPECT_NEAR(std::exp(nearest->logWeight), trueWeights[c], weightTolerance) << "component " << c;
    EXPECT_LT((nearest->gaussian.cov() - trueCovs[c]).norm(), covTolerance * trueCovs[c].norm()) << "component " << c;
  }
}

}  // namespace

TEST(FitMixture, RecoversMixtureFromSamples) {
  const auto truth{makeTrueMixture()};
  std::mt19937_64 rng{41};
  const MPS2 particles{EigenFilter::drawSamples(*truth.toGaussianMixture(), 20000, rng)};
  const auto fit{EigenFilter::fitMixture(*particles.toParticleSet(), 3, rng)};
  ASSERT_TRUE(fit.has_value());
  EXPECT_TRUE(fit->converged);
  EXPECT_NEAR(*fit->mixture.logTotalWeight(), 0.0, 1e-12);
  expectRecovers(fit->mixture, 0.1, 0.02, 0.1);
}

TEST(FitMixture, UsesParticleWeights) {
  // Importance sampling from a broad proposal: the weights alone carry the target's shape.
  const auto truth{makeTrueMixture()};
  const auto target{truth.toGaussianMixture()};
  const MutableGaussian<double, 2> proposal{Eigen::Vector2d{0.0, 2.0}, 49.0 * Eigen::Matrix2d::Identity()};
  std::mt19937_64 rng{43};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(*proposal.toGaussian(), 60000, rng)};
  Eigen::ArrayXd logWeights{drawn.logPdf.size()};
  for (Eigen::Index i{0}; i < logWeights.size(); ++i) {
    logWeights(i) = target->logPdf(drawn.samples.col(i)) - drawn.logPdf(i);
  }
  const MPS2 particles{drawn.samples, logWeights};
  const auto fit{EigenFilter::fitMixture(*particles.toParticleSet(), 3, rng)};
  ASSERT_TRUE(fit.has_value());
  expectRecovers(fit->mixture, 0.25, 0.04, 0.3);
}

TEST(FitMixture, ReducesJitteredMixtureWithoutSampling) {
  // 999 children, each a narrowed copy of a true component jittered within it: the
  // moment-matched groups are the true components.
  const auto truth{makeTrueMixture()};
  std::mt19937_64 rng{47};
  MGM2 children;
  children.reserve(999);
  for (std::size_t c{0}; c < 3; ++c) {
    const MutableGaussian<double, 2> jitter{trueMeans[c], 0.5 * trueCovs[c]};
    const auto centers{EigenFilter::drawSamples(*jitter.toGaussian(), 333, rng)};
    for (Eigen::Index j{0}; j < centers.cols(); ++j) {
      children.add(std::log(trueWeights[c] / 333.0), centers.col(j), 0.5 * trueCovs[c]);
    }
  }
  const auto source{children.toGaussianMixture()};
  const auto fit{EigenFilter::fitMixture(*source, 3, rng)};
  ASSERT_TRUE(fit.has_value());
  EXPECT_TRUE(fit->converged);
  expectRecovers(fit->mixture, 0.2, 0.01, 0.2);

  // Moment matching preserves the overall mean exactly and the covariance up to the floor.
  const auto before{source->moments()};
  const auto after{fit->mixture.toGaussianMixture()->moments()};
  EXPECT_LT((after.mean() - before.mean()).norm(), 1e-9);
  EXPECT_LT((after.cov() - before.cov()).norm(), 1e-4);
}

TEST(FitMixture, StackedComponentsAreSeededOnce) {
  MGM2 stacked;
  for (int copy{0}; copy < 10; ++copy) {
    stacked.add(0.0, trueMeans[0], trueCovs[0]);
    stacked.add(0.0, trueMeans[2], trueCovs[2]);
  }
  std::mt19937_64 rng{53};
  const auto fit{EigenFilter::fitMixture(*stacked.toGaussianMixture(), 2, rng)};
  ASSERT_TRUE(fit.has_value());
  ASSERT_EQ(fit->mixture.size(), 2);
  const double a{fit->mixture.begin()->gaussian.mean()(0)};
  const double b{(fit->mixture.begin() + 1)->gaussian.mean()(0)};
  EXPECT_NEAR(std::min(a, b), -6.0, 1e-9);
  EXPECT_NEAR(std::max(a, b), 6.0, 1e-9);
}

TEST(FitMixture, DuplicateParticlesLimitTheComponentCount) {
  Eigen::Matrix<double, 2, 100> states;
  states.leftCols<50>().colwise() = trueMeans[0];
  states.rightCols<50>().colwise() = trueMeans[2];
  const MPS2 particles{states};
  std::mt19937_64 rng{59};
  const auto fit{EigenFilter::fitMixture(*particles.toParticleSet(), 5, rng)};
  ASSERT_TRUE(fit.has_value());
  EXPECT_EQ(fit->mixture.size(), 2);
}

TEST(FitMixture, SeedsFromAMixture) {
  const auto truth{makeTrueMixture()};
  const auto view{truth.toGaussianMixture()};
  std::mt19937_64 rng{61};
  const MPS2 particles{EigenFilter::drawSamples(*view, 20000, rng)};
  const auto fit{EigenFilter::fitMixture(*particles.toParticleSet(), *view, 3, rng)};
  ASSERT_TRUE(fit.has_value());
  EXPECT_TRUE(fit->converged);
  expectRecovers(fit->mixture, 0.1, 0.02, 0.1);
}

TEST(FitMixture, LogLikelihoodNeverDecreases) {
  const auto truth{makeTrueMixture()};
  std::mt19937_64 sampler{67};
  const MPS2 particles{EigenFilter::drawSamples(*truth.toGaussianMixture(), 2000, sampler)};
  const auto view{particles.toParticleSet()};
  double previous{-std::numeric_limits<double>::infinity()};
  for (int iterations{1}; iterations <= 12; ++iterations) {
    std::mt19937_64 rng{71};  // same seeding every time
    const auto fit{EigenFilter::fitMixture(*view, 4, rng, FitOptions<double>{.maxIterations = iterations})};
    ASSERT_TRUE(fit.has_value());
    EXPECT_GE(fit->logLikelihood, previous - 1e-9) << "iterations " << iterations;
    previous = fit->logLikelihood;
  }
}

TEST(FitMixture, DynamicSize) {
  const auto truth{makeTrueMixture()};
  std::mt19937_64 rng{73};
  const MutableParticleSet<double, Eigen::Dynamic> particles{
      Eigen::MatrixXd{EigenFilter::drawSamples(*truth.toGaussianMixture(), 5000, rng)}};
  const auto fit{EigenFilter::fitMixture(*particles.toParticleSet(), 3, rng)};
  ASSERT_TRUE(fit.has_value());
  static_assert(std::is_same_v<decltype(fit->mixture), MutableGaussianMixture<double, Eigen::Dynamic>>);
  EXPECT_EQ(fit->mixture.componentSize(), 2);
}

TEST(FitMixture, DegenerateSourceFails) {
  const MPS2 collapsed{Eigen::Matrix<double, 2, 10>::Ones()};
  std::mt19937_64 rng{79};
  const auto fit{EigenFilter::fitMixture(*collapsed.toParticleSet(), 2, rng)};
  ASSERT_FALSE(fit.has_value());
  EXPECT_EQ(fit.error(), FitError::Degenerate);
}
