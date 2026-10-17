#include <EigenFilter/GaussianMixture.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <tuple>

using EigenFilter::MutableGaussianMixtureN;

namespace {

using MGM4 = MutableGaussianMixtureN<double, 4>;

Eigen::Matrix4d makeSpd(double phase) {
  const Eigen::Matrix4d a{Eigen::Matrix4d::NullaryExpr(
      [phase](Eigen::Index i, Eigen::Index j) { return std::sin(7.0 * double(i) + 3.0 * double(j) + phase); })};
  return a * a.transpose() + Eigen::Matrix4d::Identity();
}

Eigen::Vector4d makeMean(double phase) {
  return Eigen::Vector4d::NullaryExpr([phase](Eigen::Index i) { return 2.0 * std::sin(1.3 * double(i) + phase); });
}

// Three components with unnormalized weights 0.5, 2, 1.
MGM4 make() {
  MGM4 mixture;
  mixture.add(std::log(0.5), makeMean(0.0), makeSpd(1.0));
  mixture.add(std::log(2.0), makeMean(1.0), makeSpd(2.0));
  mixture.add(std::log(1.0), makeMean(2.0), makeSpd(3.0));
  return mixture;
}

}  // namespace

TEST(GaussianMixtureMarginal, MarginalizesEveryComponent) {
  const auto mixture{make()};
  const auto marginal{mixture.marginal(std::array{3, 0})};
  static_assert(decltype(marginal)::SizeAtCompileTime == 2);
  ASSERT_EQ(marginal.size(), 3);
  EXPECT_EQ(marginal.componentSize(), 2);

  for (Eigen::Index i{0}; i < marginal.size(); ++i) {
    const auto& component{*(mixture.begin() + i)};
    const auto expected{component.gaussian.marginal(std::array{3, 0}).moments()};
    EXPECT_EQ(marginal.logWeight(i), component.logWeight);
    EXPECT_EQ(marginal.mean(i), expected.mean());
    EXPECT_EQ(Eigen::Matrix2d{marginal.cov(i)}, expected.cov());
  }

  auto copy{marginal.copy()};
  static_assert(std::is_same_v<decltype(copy), MutableGaussianMixtureN<double, 2>>);
  EXPECT_EQ(copy.size(), 3);
  EXPECT_TRUE(copy.toGaussianMixture().has_value());
}

TEST(GaussianMixtureMarginal, ValuesMayChangeWhileBorrowed) {
  auto mixture{make()};
  const auto marginal{mixture.marginal(std::array{1})};
  mixture.setMean(mixture.begin(), Eigen::Vector4d::Zero());
  mixture.scaleWeights(1.0);
  EXPECT_EQ(marginal.mean(0)(0), 0.0);
  EXPECT_DOUBLE_EQ(marginal.logWeight(0), std::log(0.5) + 1.0);
}

TEST(GaussianMixtureConditional, ConditionsEachComponentAndReweights) {
  const auto mixture{make()};
  const Eigen::Vector2d v{0.4, -1.1};
  const std::array target{2, 0};
  const std::array given{1, 3};
  const auto conditional{mixture.conditional(target, given, v)};
  static_assert(std::is_same_v<decltype(conditional), const MutableGaussianMixtureN<double, 2>>);
  ASSERT_EQ(conditional.size(), 3);

  for (Eigen::Index i{0}; i < conditional.size(); ++i) {
    const auto& source{*(mixture.begin() + i)};
    const auto& result{*(conditional.begin() + i)};
    const auto expected{source.gaussian.conditional(target, given, v)};
    const double logLikelihood{source.gaussian.marginal(given).copy().toGaussian()->logPdf(v)};
    EXPECT_LT((result.gaussian.mean() - expected.mean()).norm(), 1e-12);
    EXPECT_LT((result.gaussian.cov() - expected.cov()).norm(), 1e-12);
    EXPECT_NEAR(result.logWeight, source.logWeight + logLikelihood, 1e-12);
  }
}

TEST(GaussianMixtureConditional, TotalWeightIsMarginalLikelihood) {
  const auto mixture{make()};
  const Eigen::Vector2d v{0.4, -1.1};
  const auto conditional{mixture.conditional(std::array{1, 3}, v)};
  static_assert(std::is_same_v<decltype(conditional), const MutableGaussianMixtureN<double, 2>>);

  const auto marginalAtGiven{mixture.marginal(std::array{1, 3}).copy()};
  const double expected{*mixture.logTotalWeight() + marginalAtGiven.toGaussianMixture()->logPdf(v)};
  EXPECT_NEAR(*conditional.logTotalWeight(), expected, 1e-12);
}

TEST(GaussianMixtureConditional, ViewMatchesMutable) {
  const auto mixture{make()};
  const Eigen::Matrix<double, 1, 1> v{0.2};
  const auto fromMutable{mixture.conditional(std::array{2}, v)};
  const auto fromView{mixture.toGaussianMixture()->conditional(std::array{2}, v)};
  ASSERT_EQ(fromMutable.size(), fromView.size());
  for (Eigen::Index i{0}; i < fromMutable.size(); ++i) {
    EXPECT_EQ((fromMutable.begin() + i)->logWeight, (fromView.begin() + i)->logWeight);
    EXPECT_EQ((fromMutable.begin() + i)->gaussian.mean(), (fromView.begin() + i)->gaussian.mean());
  }
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

TEST(GaussianMixtureMarginalDeathTest, StructuralChangeWhileBorrowed) {
  auto mixture{std::make_unique<MGM4>(make())};
  const auto marginal{mixture->marginal(std::array{0, 1})};
  EXPECT_DEATH(mixture->add(0.0, makeMean(3.0), makeSpd(4.0)), "borrowed");
  EXPECT_DEATH(mixture->remove(mixture->begin()), "borrowed");
  EXPECT_DEATH(mixture->truncate(1), "borrowed");
  EXPECT_DEATH(mixture->clear(), "borrowed");
  EXPECT_DEATH({ auto stolen{std::move(*mixture)}; }, "borrowed");
  EXPECT_DEATH(mixture.reset(), "borrowed");
}

TEST(GaussianMixtureMarginalDeathTest, MarginalOfViewKeepsLock) {
  auto mixture{make()};
  std::optional<EigenFilter::GaussianMixtureMarginal<MGM4::GaussianMixtureType, std::array<int, 1>>> marginal;
  marginal.emplace(mixture.toGaussianMixture()->marginal(std::array{0}));
  EXPECT_DEATH(mixture.scaleWeights(1.0), "locked");
}

#endif
