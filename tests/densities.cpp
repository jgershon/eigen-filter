#include <EigenFilter/Gamma.h>
#include <EigenFilter/StudentT.h>
#include <EigenFilter/Transform.h>
#include <EigenFilter/Uniform.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <type_traits>

using EigenFilter::Bounded;
using EigenFilter::Composite;
using EigenFilter::GammaDensity;
using EigenFilter::GammaDensityN;
using EigenFilter::Linear;
using EigenFilter::MomentsError;
using EigenFilter::Periodic;
using EigenFilter::StudentTDensity;
using EigenFilter::StudentTDensityN;
using EigenFilter::UniformDensity;
using EigenFilter::UniformDensityN;

namespace {

constexpr double pi{std::numbers::pi};
constexpr double inf{std::numeric_limits<double>::infinity()};

struct Heading : Periodic<> {
  using Periodic::Periodic;
};
struct Range : Bounded<> {
  using Bounded::Bounded;
};

// Non-Gaussian densities report moments() but not mean()/cov(), so transform() never mistakes them for Gaussians.
static_assert(!EigenFilter::UncertainInput<UniformDensityN<double, 2>>);
static_assert(!EigenFilter::UncertainInput<GammaDensityN<double, 2>>);
static_assert(!EigenFilter::UncertainInput<StudentTDensityN<double, 2>>);

}  // namespace

TEST(Uniform, DefaultIntervalComesFromTheSpace) {
  const UniformDensity<double, Heading> heading{};
  EXPECT_NEAR(heading.lower()(0), -pi, 1e-15);
  EXPECT_NEAR(heading.width()(0), 2 * pi, 1e-15);
  EXPECT_NEAR(heading.logPdf(Eigen::Matrix<double, 1, 1>{3.0}), -std::log(2 * pi), 1e-15);

  using Box = Composite<Heading, Bounded<1>>;
  const UniformDensity<double, Box> box{Box{Heading{}, Bounded<1>{2.0, 5.0}}};
  EXPECT_NEAR(box.logVolume(), std::log(2 * pi * 3.0), 1e-14);
  EXPECT_TRUE(box.contains(Eigen::Vector2d{0.0, 4.0}));
  EXPECT_FALSE(box.contains(Eigen::Vector2d{0.0, 5.5}));
  EXPECT_EQ(box.logPdf(Eigen::Vector2d{0.0, 1.0}), -inf);
}

TEST(Uniform, ExplicitBoxInALinearSpace) {
  const UniformDensityN<double, 2> uniform{Eigen::Vector2d{-1.0, 0.0}, Eigen::Vector2d{1.0, 4.0}};
  EXPECT_NEAR(uniform.logPdf(Eigen::Vector2d{0.5, 3.0}), -std::log(8.0), 1e-15);
  EXPECT_EQ(uniform.logPdf(Eigen::Vector2d{1.5, 3.0}), -inf);
  EXPECT_NEAR(uniform.pdf(Eigen::Vector2d{-1.0, 4.0}), 0.125, 1e-15);  // the box is closed

  const auto moments{uniform.moments()};
  EXPECT_TRUE(moments.mean().isApprox(Eigen::Vector2d{0.0, 2.0}));
  EXPECT_TRUE(moments.cov().isApprox(Eigen::Vector2d{4.0 / 12, 16.0 / 12}.asDiagonal().toDenseMatrix()));

  std::mt19937_64 rng{7};
  const auto samples{EigenFilter::drawSamples(uniform, 40000, rng)};
  EXPECT_TRUE((samples.row(0).array() >= -1.0).all() && (samples.row(0).array() < 1.0).all());
  EXPECT_TRUE((samples.row(1).array() >= 0.0).all() && (samples.row(1).array() < 4.0).all());
  const Eigen::Vector2d mean{samples.rowwise().mean()};
  EXPECT_NEAR(mean(0), 0.0, 0.02);
  EXPECT_NEAR(mean(1), 2.0, 0.04);
  const Eigen::Matrix<double, 2, Eigen::Dynamic> centered{samples.colwise() - mean};
  EXPECT_NEAR((centered.row(1).array().square().mean()), 16.0 / 12, 0.03);
}

TEST(Uniform, PeriodicIntervalCrossesTheSeam) {
  // [3, 3.5] on an angle wraps past π to 3.5 - 2π.
  const UniformDensity<double, Heading> arc{Heading{}, Eigen::Matrix<double, 1, 1>{3.0},
                                            Eigen::Matrix<double, 1, 1>{3.5}};
  EXPECT_TRUE(arc.contains(Eigen::Matrix<double, 1, 1>{3.1}));
  EXPECT_TRUE(arc.contains(Eigen::Matrix<double, 1, 1>{3.4 - 2 * pi}));
  EXPECT_FALSE(arc.contains(Eigen::Matrix<double, 1, 1>{0.0}));
  EXPECT_FALSE(arc.contains(Eigen::Matrix<double, 1, 1>{2.9}));
  EXPECT_NEAR(arc.moments().mean()(0), 3.25 - 2 * pi, 1e-14);  // the center, normalized

  std::mt19937_64 rng{3};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(arc, 2000, rng)};
  bool wrapped{false};
  for (Eigen::Index j{0}; j < drawn.samples.cols(); ++j) {
    const double x{drawn.samples(0, j)};
    EXPECT_TRUE(x >= -pi && x < pi);
    EXPECT_TRUE(arc.contains(drawn.samples.col(j)));
    EXPECT_NEAR(drawn.logPdf(j), -std::log(0.5), 1e-14);
    wrapped = wrapped || x < 0;
  }
  EXPECT_TRUE(wrapped);
}

TEST(Uniform, ResizableSpaceTakesTheBoxSize) {
  const UniformDensity<double, Linear<Eigen::Dynamic>> uniform{Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()};
  EXPECT_EQ(uniform.size(), 3);
  EXPECT_EQ(uniform.space().size(), 3);
  EXPECT_NEAR(uniform.logPdf(Eigen::Vector3d::Constant(0.5)), 0.0, 1e-15);
}

TEST(Gamma, LogPdfMatchesTheFormula) {
  const GammaDensity<double, Range> gamma{2.0, 3.0};
  // x^(k-1) e^(-x/θ) / (Γ(k) θ^k) with k = 2, θ = 3.
  EXPECT_NEAR(gamma.logPdf(Eigen::Matrix<double, 1, 1>{1.5}), std::log(1.5) - 0.5 - std::log(9.0), 1e-14);
  EXPECT_EQ(gamma.logPdf(Eigen::Matrix<double, 1, 1>{-0.1}), -inf);
  EXPECT_EQ(gamma.logPdf(Eigen::Matrix<double, 1, 1>{0.0}), -inf);

  // It integrates to one.
  const GammaDensityN<double, 1> skewed{2.5, 1.5};
  double integral{0};
  const double h{1e-3};
  for (double x{h / 2}; x < 60; x += h) integral += skewed.pdf(Eigen::Matrix<double, 1, 1>{x}) * h;
  EXPECT_NEAR(integral, 1.0, 1e-6);
}

TEST(Gamma, ExponentialAndChiSquared) {
  const auto exponential{GammaDensity<double, Range>::exponential(2.0)};
  EXPECT_NEAR(exponential.logPdf(Eigen::Matrix<double, 1, 1>{0.0}), -std::log(2.0), 1e-15);
  EXPECT_NEAR(exponential.logPdf(Eigen::Matrix<double, 1, 1>{3.0}), -std::log(2.0) - 1.5, 1e-15);
  EXPECT_NEAR(exponential.moments().mean()(0), 2.0, 1e-15);
  EXPECT_NEAR(exponential.moments().cov()(0, 0), 4.0, 1e-15);

  const auto chiSquared{GammaDensityN<double, 1>::chiSquared(4.0)};
  EXPECT_NEAR(chiSquared.moments().mean()(0), 4.0, 1e-15);
  EXPECT_NEAR(chiSquared.moments().cov()(0, 0), 8.0, 1e-15);
}

TEST(Gamma, FromMeanAndStdDevPerCoordinate) {
  const auto gamma{
      GammaDensity<double, Bounded<2>>::fromMeanAndStdDev(Eigen::Vector2d{100.0, 5.0}, Eigen::Vector2d{20.0, 5.0})};
  EXPECT_NEAR(gamma.shape()(0), 25.0, 1e-12);
  EXPECT_NEAR(gamma.shape()(1), 1.0, 1e-12);
  const auto moments{gamma.moments()};
  EXPECT_TRUE(moments.mean().isApprox(Eigen::Vector2d{100.0, 5.0}));
  EXPECT_TRUE(moments.cov().isApprox(Eigen::Vector2d{400.0, 25.0}.asDiagonal().toDenseMatrix()));
  EXPECT_EQ(moments.cov()(0, 1), 0.0);

  std::mt19937_64 rng{11};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(gamma, 40000, rng)};
  EXPECT_TRUE((drawn.samples.array() >= 0).all());
  const Eigen::Vector2d mean{drawn.samples.rowwise().mean()};
  EXPECT_NEAR(mean(0), 100.0, 0.5);
  EXPECT_NEAR(mean(1), 5.0, 0.1);
  for (Eigen::Index j{0}; j < 10; ++j) EXPECT_NEAR(drawn.logPdf(j), gamma.logPdf(drawn.samples.col(j)), 1e-12);
}

TEST(StudentT, LogPdfMatchesTheFormula) {
  const auto t{
      StudentTDensityN<double, 1>::create(Eigen::Matrix<double, 1, 1>{1.0}, Eigen::Matrix<double, 1, 1>{4.0}, 3.0)};
  ASSERT_TRUE(t.has_value());
  // Γ(2) / (Γ(3/2) √(3π) σ) (1 + ((x - μ)/σ)² / 3)^-2 with σ = 2.
  const double x{2.5};
  const double z{(x - 1.0) / 2.0};
  const double expected{-std::lgamma(1.5) - 0.5 * std::log(3 * pi) - std::log(2.0) - 2.0 * std::log1p(z * z / 3)};
  EXPECT_NEAR(t->logPdf(Eigen::Matrix<double, 1, 1>{x}), expected, 1e-14);
}

TEST(StudentT, ApproachesTheGaussianAndHasHeavierTails) {
  const Eigen::Vector2d mu{1.0, -1.0};
  const Eigen::Matrix2d sigma{{2.0, 0.5}, {0.5, 1.0}};
  const auto nearlyGaussian{StudentTDensityN<double, 2>::create(mu, sigma, 1e8)};
  const auto heavy{StudentTDensityN<double, 2>::create(mu, sigma, 2.0)};
  const EigenFilter::GaussianN<double, 2> gaussian{mu, sigma};
  const auto density{gaussian.toDensity()};
  ASSERT_TRUE(nearlyGaussian && heavy && density);
  const Eigen::Vector2d near{1.5, -0.5};
  const Eigen::Vector2d far{9.0, 6.0};
  EXPECT_NEAR(nearlyGaussian->logPdf(near), density->logPdf(near), 1e-6);
  EXPECT_GT(heavy->logPdf(far), density->logPdf(far) + 10);
}

TEST(StudentT, MomentsExistOnlyAboveTwoDegreesOfFreedom) {
  const Eigen::Matrix2d sigma{{2.0, 0.5}, {0.5, 1.0}};
  const auto t{StudentTDensityN<double, 2>::create(Eigen::Vector2d::Zero(), sigma, 5.0)};
  ASSERT_TRUE(t.has_value());
  const auto moments{t->moments()};
  ASSERT_TRUE(moments.has_value());
  EXPECT_TRUE(moments->cov().isApprox(sigma * 5.0 / 3.0));

  const auto cauchy{StudentTDensityN<double, 2>::create(Eigen::Vector2d::Zero(), sigma, 1.0)};
  ASSERT_TRUE(cauchy.has_value());
  EXPECT_EQ(cauchy->moments().error(), MomentsError::Infinite);
  const auto two{StudentTDensityN<double, 2>::create(Eigen::Vector2d::Zero(), sigma, 2.0)};
  ASSERT_TRUE(two.has_value());
  EXPECT_EQ(two->moments().error(), MomentsError::Infinite);
}

TEST(StudentT, RejectsAScaleThatIsNotPositiveDefinite) {
  const auto t{
      StudentTDensityN<double, 2>::create(Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 2.0}, {2.0, 1.0}}, 4.0)};
  ASSERT_FALSE(t.has_value());
  EXPECT_EQ(t.error(), EigenFilter::GaussianError::NotPositiveDefinite);
}

TEST(StudentT, SamplesMatchTheDistribution) {
  const Eigen::Matrix2d sigma{{2.0, 0.5}, {0.5, 1.0}};
  const auto t{StudentTDensityN<double, 2>::create(Eigen::Vector2d{1.0, 2.0}, sigma, 8.0)};
  ASSERT_TRUE(t.has_value());
  std::mt19937_64 rng{5};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(*t, 100000, rng)};
  for (Eigen::Index j{0}; j < 20; ++j) EXPECT_NEAR(drawn.logPdf(j), t->logPdf(drawn.samples.col(j)), 1e-10);
  const Eigen::Vector2d mean{drawn.samples.rowwise().mean()};
  EXPECT_NEAR(mean(0), 1.0, 0.02);
  EXPECT_NEAR(mean(1), 2.0, 0.02);
  const Eigen::Matrix<double, 2, Eigen::Dynamic> centered{drawn.samples.colwise() - mean};
  const Eigen::Matrix2d cov{centered * centered.transpose() / double(centered.cols())};
  EXPECT_TRUE(cov.isApprox(sigma * 8.0 / 6.0, 0.03)) << cov;
}

TEST(StudentT, WrapsInAPeriodicSpace) {
  const auto t{StudentTDensity<double, Heading>::create(Eigen::Matrix<double, 1, 1>{3.0},
                                                        Eigen::Matrix<double, 1, 1>{0.04}, 4.0)};
  ASSERT_TRUE(t.has_value());
  EXPECT_NEAR(t->logPdf(Eigen::Matrix<double, 1, 1>{3.1}), t->logPdf(Eigen::Matrix<double, 1, 1>{2.9}), 1e-12);
  std::mt19937_64 rng{9};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(*t, 2000, rng)};
  EXPECT_TRUE((drawn.samples.array() >= -pi).all() && (drawn.samples.array() < pi).all());
  for (Eigen::Index j{0}; j < 20; ++j) EXPECT_NEAR(drawn.logPdf(j), t->logPdf(drawn.samples.col(j)), 1e-12);
}

TEST(Uniform, MarginalsAndConditionalsKeepTheIntervals) {
  using Box = Composite<Heading, Bounded<1>>;
  const UniformDensity<double, Box> box{Box{Heading{}, Bounded<1>{2.0, 5.0}}};

  const auto heading{box.marginal<Heading>()};
  static_assert(std::is_same_v<decltype(heading), const UniformDensity<double, Heading>>);
  EXPECT_NEAR(heading.width()(0), 2 * pi, 1e-15);

  const auto bounded{box.marginal(std::array<Eigen::Index, 1>{1})};
  EXPECT_EQ(bounded.space().coordinate(0).kind, EigenFilter::CoordinateKind::Bounded);
  EXPECT_NEAR(bounded.logPdf(Eigen::Matrix<double, 1, 1>{3.0}), -std::log(3.0), 1e-15);

  // Independent coordinates: conditioning on the heading leaves the bounded interval.
  const auto given{box.conditional<Heading>(Eigen::Matrix<double, 1, 1>{1.0})};
  static_assert(std::is_same_v<decltype(given), const UniformDensity<double, Bounded<1>>>);
  EXPECT_NEAR(given.lower()(0), 2.0, 1e-15);
  EXPECT_NEAR(given.width()(0), 3.0, 1e-15);
  const auto byIndex{box.conditional(std::array<Eigen::Index, 1>{1}, Eigen::Matrix<double, 1, 1>{4.0})};
  EXPECT_NEAR(byIndex.logPdf(Eigen::Matrix<double, 1, 1>{-3.0}), -std::log(2 * pi), 1e-15);
}

TEST(Gamma, MarginalsAndConditionalsKeepTheParameters) {
  const GammaDensity<double, Bounded<3>> gamma{Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Vector3d{0.5, 1.0, 1.5}};
  const auto picked{gamma.marginal(std::array<Eigen::Index, 2>{2, 0})};
  EXPECT_TRUE(picked.shape().isApprox(Eigen::Vector2d{3.0, 1.0}));
  EXPECT_TRUE(picked.scale().isApprox(Eigen::Vector2d{1.5, 0.5}));
  const auto rest{gamma.conditional(std::array<Eigen::Index, 1>{1}, Eigen::Matrix<double, 1, 1>{4.0})};
  EXPECT_TRUE(rest.shape().isApprox(Eigen::Vector2d{1.0, 3.0}));
  const Eigen::Vector3d x{0.4, 4.0, 2.0};
  EXPECT_NEAR(rest.logPdf(Eigen::Vector2d{0.4, 2.0}),
              gamma.logPdf(x) - gamma.marginal(std::array<Eigen::Index, 1>{1}).logPdf(Eigen::Matrix<double, 1, 1>{4.0}),
              1e-13);
}

TEST(StudentT, MarginalKeepsTheDegreesOfFreedom) {
  const Eigen::Matrix3d sigma{{2.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 1.5}};
  const auto t{StudentTDensityN<double, 3>::create(Eigen::Vector3d{1.0, -1.0, 0.5}, sigma, 4.0)};
  ASSERT_TRUE(t.has_value());
  const auto marginal{t->marginal(std::array<Eigen::Index, 2>{2, 0})};
  EXPECT_EQ(marginal.dof(), 4.0);
  EXPECT_TRUE(marginal.location().isApprox(Eigen::Vector2d{0.5, 1.0}));
  EXPECT_TRUE(marginal.scale().isApprox(Eigen::Matrix2d{{1.5, 0.2}, {0.2, 2.0}}));
}

TEST(StudentT, ConditionalIsTheJointOverTheMarginal) {
  const Eigen::Matrix3d sigma{{2.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 1.5}};
  const auto t{StudentTDensityN<double, 3>::create(Eigen::Vector3d{1.0, -1.0, 0.5}, sigma, 4.0)};
  ASSERT_TRUE(t.has_value());
  const Eigen::Matrix<double, 1, 1> v{2.5};  // far from the location -1, so the conditional widens
  const auto conditional{t->conditional(std::array<Eigen::Index, 1>{1}, v)};
  ASSERT_TRUE(conditional.has_value());
  EXPECT_EQ(conditional->dof(), 5.0);
  const auto marginal{t->marginal(std::array<Eigen::Index, 1>{1})};
  for (const Eigen::Vector2d& a : {Eigen::Vector2d{1.0, 0.5}, Eigen::Vector2d{3.0, -2.0}}) {
    const Eigen::Vector3d x{a(0), v(0), a(1)};
    EXPECT_NEAR(conditional->logPdf(a), t->logPdf(x) - marginal.logPdf(v), 1e-12);
  }
  // The scale is the Gaussian conditional's, stretched by (ν + δ²) / (ν + 1) with δ² = 3.5² / 1.
  const EigenFilter::GaussianN<double, 3> gaussian{Eigen::Vector3d{1.0, -1.0, 0.5}, sigma};
  const auto gaussianConditional{gaussian.conditional(std::array<Eigen::Index, 1>{1}, v)};
  EXPECT_TRUE(conditional->scale().isApprox(gaussianConditional.cov() * (4.0 + 12.25) / 5.0));
  EXPECT_TRUE(conditional->location().isApprox(gaussianConditional.mean()));
}

TEST(StudentT, TagConditionalReturnsTheRemainingBlock) {
  struct Offset : Linear<2> {};
  using Pose = Composite<Offset, Heading>;
  const Eigen::Matrix3d sigma{{2.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 0.5}};
  const auto t{StudentTDensity<double, Pose>::create(Eigen::Vector3d{1.0, -1.0, 0.5}, sigma, 6.0)};
  ASSERT_TRUE(t.has_value());
  const auto offset{t->conditional<Heading>(Eigen::Matrix<double, 1, 1>{0.6})};
  ASSERT_TRUE(offset.has_value());
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(*offset)>, StudentTDensity<double, Offset>>);
  EXPECT_EQ(offset->dof(), 7.0);
  const auto heading{t->marginal<Heading>()};
  static_assert(std::is_same_v<decltype(heading), const StudentTDensity<double, Heading>>);
  const Eigen::Vector2d a{0.5, -0.5};
  EXPECT_NEAR(offset->logPdf(a),
              t->logPdf(Eigen::Vector3d{0.5, -0.5, 0.6}) - heading.logPdf(Eigen::Matrix<double, 1, 1>{0.6}), 1e-12);
}
