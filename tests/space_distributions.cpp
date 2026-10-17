#include <EigenFilter/Fitting.h>
#include <EigenFilter/GaussianMixture.h>
#include <EigenFilter/KernelDensity.h>
#include <EigenFilter/ParticleSet.h>
#include <EigenFilter/Sampling.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <type_traits>

using EigenFilter::Composite;
using EigenFilter::GaussianMixture;
using EigenFilter::Linear;
using EigenFilter::ParticleSet;
using EigenFilter::Periodic;

namespace {

constexpr double pi{std::numbers::pi};

struct Position : Linear<1> {};
struct Heading : Periodic<> {
  using Periodic::Periodic;
};
using State = Composite<Position, Heading>;
using Particles = ParticleSet<double, State>;
using Mixture = GaussianMixture<double, State>;

// Distance around the circle between two headings.
double angularGap(double a, double b) { return std::abs(std::remainder(a - b, 2 * pi)); }

// Particles at headings π - 0.1 and -π + 0.1: a tight cloud straddling the seam.
Particles makeStraddlingParticles() {
  Eigen::Matrix<double, 2, 4> states{{1.0, 2.0, 1.0, 2.0}, {pi - 0.1, pi - 0.1, -pi + 0.1, -pi + 0.1}};
  return Particles{State{}, states};
}

}  // namespace

TEST(SpaceParticles, StatesAreNormalizedWhenStored) {
  Particles set{State{}, Eigen::Matrix<double, 2, 2>{{0.0, 0.0}, {4.0, -4.0}}};
  EXPECT_NEAR(set.state(0)(1), 4.0 - 2 * pi, 1e-15);
  EXPECT_NEAR(set.state(1)(1), -4.0 + 2 * pi, 1e-15);

  set.setState(0, Eigen::Vector2d{0.0, 3.5});
  EXPECT_NEAR(set.state(0)(1), 3.5 - 2 * pi, 1e-15);
  set.add(0.0, Eigen::Vector2d{0.0, 7.0});
  EXPECT_NEAR(set.state(2)(1), 7.0 - 2 * pi, 1e-15);
  set.forEach([](auto& p) { p.setState(Eigen::Vector2d{p.state()(0), p.state()(1) + 2 * pi}); });
  EXPECT_NEAR(set.state(2)(1), 7.0 - 2 * pi, 1e-12);
  set.setStates(Eigen::Matrix<double, 2, 3>::Constant(4.0));
  EXPECT_NEAR(set.state(1)(1), 4.0 - 2 * pi, 1e-15);
}

TEST(SpaceParticles, MomentsAreTangentSpaceMoments) {
  const auto set{makeStraddlingParticles()};
  const auto moments{*set.moments()};
  EXPECT_LT(angularGap(moments.mean()(1), pi), 1e-12);
  EXPECT_NEAR(moments.cov()(1, 1), 0.01, 1e-12);  // not ~π², as a linear average would give
  EXPECT_NEAR(moments.mean()(0), 1.5, 1e-12);
}

TEST(SpaceParticles, KernelDensityWrapsResiduals) {
  auto set{makeStraddlingParticles()};
  set.setBandwidth(Eigen::Matrix2d{{0.1, 0.0}, {0.0, 0.02}});
  const auto kde{set.toDensity()};
  ASSERT_TRUE(kde.has_value());
  // The cloud is symmetric about π, so points the same angular distance either side match.
  EXPECT_NEAR(kde->logPdf(Eigen::Vector2d{1.5, pi - 0.05}), kde->logPdf(Eigen::Vector2d{1.5, -pi + 0.05}), 1e-12);
  EXPECT_GT(kde->logPdf(Eigen::Vector2d{1.5, -pi + 0.05}), kde->logPdf(Eigen::Vector2d{1.5, 0.0}) + 10.0);

  std::mt19937_64 rng{1};
  const auto samples{EigenFilter::drawSamples(*kde, 500, rng)};
  EXPECT_TRUE((samples.row(1).array() >= -pi).all() && (samples.row(1).array() < pi).all());
}

TEST(SpaceParticles, ConditionalWrapsAndKeepsTheTargetSpace) {
  struct Swapped : Composite<Heading, Position> {};
  Eigen::Matrix<double, 2, 4> states{{pi - 0.1, pi - 0.1, -pi + 0.1, -pi + 0.1}, {1.0, 2.0, 1.0, 2.0}};
  ParticleSet<double, Swapped> set{Swapped{}, states};
  set.setBandwidth(Eigen::Matrix2d{{0.02, 0.0}, {0.0, 0.1}});
  // Conditioning on heading -π + 0.1 weights both particle groups the same way as π - 0.1 would.
  const auto a{set.conditional(std::array{1}, std::array{0}, Eigen::Matrix<double, 1, 1>{-pi + 0.1})};
  const auto b{set.conditional(std::array{1}, std::array{0}, Eigen::Matrix<double, 1, 1>{pi + 0.1})};
  EXPECT_TRUE((a.logWeights() - b.logWeights()).abs().maxCoeff() < 1e-12);

  const auto headingGivenPosition{set.conditional<Position>(Eigen::Matrix<double, 1, 1>{1.5})};
  static_assert(std::is_same_v<decltype(headingGivenPosition)::SpaceType, Heading>);
  EXPECT_TRUE((headingGivenPosition.states().array() >= -pi).all());
  EXPECT_TRUE((headingGivenPosition.states().array() < pi).all());
}

TEST(SpaceParticles, TagMarginalKeepsTheSubspace) {
  const auto set{makeStraddlingParticles()};
  const auto heading{set.marginal<Heading>().copy()};
  static_assert(std::is_same_v<decltype(heading)::SpaceType, Heading>);
  const auto moments{*heading.moments()};
  EXPECT_LT(angularGap(moments.mean()(0), pi), 1e-12);
}

TEST(SpaceMixture, ComponentMeansAreNormalized) {
  Mixture mixture;
  mixture.add(0.0, Eigen::Vector2d{0.0, 4.0}, Eigen::Matrix2d::Identity() * 0.01);
  EXPECT_NEAR(mixture.begin()->gaussian.mean()(1), 4.0 - 2 * pi, 1e-15);
  mixture.setMean(mixture.begin(), Eigen::Vector2d{0.0, -4.0});
  EXPECT_NEAR(mixture.begin()->gaussian.mean()(1), -4.0 + 2 * pi, 1e-15);
}

TEST(SpaceMixture, MomentsAndDensityWrap) {
  Mixture mixture;
  mixture.add(0.0, Eigen::Vector2d{1.0, pi - 0.1}, Eigen::Matrix2d::Identity() * 0.01);
  mixture.add(0.0, Eigen::Vector2d{2.0, -pi + 0.1}, Eigen::Matrix2d::Identity() * 0.01);
  const auto g{mixture.toDensity()};
  const auto moments{g->moments()};
  EXPECT_LT(angularGap(moments.mean()(1), pi), 1e-12);
  EXPECT_NEAR(moments.cov()(1, 1), 0.01 + 0.01, 1e-12);
  EXPECT_NEAR(g->logPdf(Eigen::Vector2d{1.5, pi - 0.05}), g->logPdf(Eigen::Vector2d{1.5, -pi + 0.05}), 1e-12);

  std::mt19937_64 rng{2};
  const auto samples{EigenFilter::drawSamples(*g, 500, rng)};
  EXPECT_TRUE((samples.row(1).array() >= -pi).all() && (samples.row(1).array() < pi).all());

  const auto conditional{mixture.conditional<Position>(Eigen::Matrix<double, 1, 1>{1.5})};
  static_assert(std::is_same_v<decltype(conditional)::SpaceType, Heading>);
  const auto heading{mixture.marginal<Heading>().copy()};
  static_assert(std::is_same_v<decltype(heading)::SpaceType, Heading>);
}

TEST(SpaceFitting, FitsAClusterAcrossTheSeam) {
  // One cluster centered at heading π, and one at 0; samples straddle the seam.
  Mixture truth;
  truth.add(std::log(0.6), Eigen::Vector2d{0.0, pi}, Eigen::Matrix2d{{1.0, 0.0}, {0.0, 0.04}});
  truth.add(std::log(0.4), Eigen::Vector2d{5.0, 0.0}, Eigen::Matrix2d{{1.0, 0.0}, {0.0, 0.04}});
  std::mt19937_64 rng{3};
  const Particles particles{State{}, EigenFilter::drawSamples(*truth.toDensity(), 20000, rng)};
  const auto fit{EigenFilter::fitMixture(particles, 2, rng)};
  ASSERT_TRUE(fit.has_value());
  static_assert(std::is_same_v<decltype(fit->mixture), Mixture>);
  ASSERT_EQ(fit->mixture.size(), 2);
  for (const auto& component : fit->mixture) {
    const bool seamCluster{std::abs(component.gaussian.mean()(0)) < 2.5};
    const double expectedHeading{seamCluster ? pi : 0.0};
    EXPECT_LT(angularGap(component.gaussian.mean()(1), expectedHeading), 0.01);
    EXPECT_NEAR(component.gaussian.cov()(1, 1), 0.04, 0.004);  // a linear fit would smear the seam cluster
  }
}

TEST(SpaceFitting, ReducesAMixtureAcrossTheSeam) {
  Mixture stacked;
  for (int i{0}; i < 20; ++i) {
    const double heading{i % 2 == 0 ? pi - 0.05 : -pi + 0.05};
    stacked.add(0.0, Eigen::Vector2d{1.0, heading}, Eigen::Matrix2d{{0.5, 0.0}, {0.0, 0.01}});
  }
  std::mt19937_64 rng{4};
  const auto fit{EigenFilter::fitMixture(*stacked.toDensity(), 1, rng)};
  ASSERT_TRUE(fit.has_value());
  const auto& component{*fit->mixture.begin()};
  EXPECT_LT(angularGap(component.gaussian.mean()(1), pi), 1e-9);
  EXPECT_NEAR(component.gaussian.cov()(1, 1), 0.01 + 0.0025, 1e-6);
}
