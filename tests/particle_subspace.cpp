#include <EigenFilter/ParticleSet.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <variant>

using EigenFilter::GaussianError;
using EigenFilter::KernelDensityError;
using EigenFilter::ParticleSetN;
using EigenFilter::WeightError;

namespace {

using MPS3 = ParticleSetN<double, 3>;

// Nine full-rank particles with unnormalized weights 1..9.
MPS3 make() {
  const Eigen::Matrix<double, 3, 9> states{Eigen::Matrix<double, 3, 9>::NullaryExpr(
      [](Eigen::Index i, Eigen::Index j) { return std::sin(1.3 * double(i * j) + 0.7 * double(j) + double(i)); })};
  const Eigen::Array<double, 9, 1> logWeights{
      Eigen::Array<double, 9, 1>::NullaryExpr([](Eigen::Index j) { return std::log(double(j + 1)); })};
  return MPS3{states, logWeights};
}

const Eigen::Matrix3d bandwidth3{{0.30, 0.05, 0.02}, {0.05, 0.20, -0.04}, {0.02, -0.04, 0.25}};

}  // namespace

TEST(ParticleSetMarginal, SelectsStateRowsAndKeepsWeights) {
  const auto set{make()};
  const auto marginal{set.marginal(std::array{2, 0})};
  static_assert(decltype(marginal)::SizeAtCompileTime == 2);
  EXPECT_EQ(marginal.size(), 9);
  EXPECT_EQ(marginal.particleSize(), 2);
  EXPECT_EQ(Eigen::MatrixXd{marginal.states()}.row(0), set.states().row(2));
  EXPECT_EQ(Eigen::MatrixXd{marginal.states()}.row(1), set.states().row(0));
  EXPECT_EQ(Eigen::Vector2d{marginal.state(4)}, Eigen::Vector2d(set.state(4)(2), set.state(4)(0)));
  EXPECT_TRUE((marginal.logWeights() == set.logWeights()).all());
  EXPECT_FALSE(marginal.bandwidth().has_value());
  EXPECT_FALSE(marginal.copy().bandwidth().has_value());
}

TEST(ParticleSetMarginal, MarginalizesStoredBandwidth) {
  auto set{make()};
  set.setBandwidth(bandwidth3);
  const auto marginal{set.marginal(std::array{2, 0})};
  const Eigen::Matrix2d expected{{0.25, 0.02}, {0.02, 0.30}};
  ASSERT_TRUE(marginal.bandwidth().has_value());
  EXPECT_EQ(*marginal.bandwidth(), expected);

  const auto copy{marginal.copy()};
  static_assert(std::is_same_v<decltype(copy), const ParticleSetN<double, 2>>);
  ASSERT_TRUE(copy.bandwidth().has_value());
  EXPECT_EQ(*copy.bandwidth(), expected);
}

TEST(ParticleSetMarginal, StatesMayChangeWhileBorrowed) {
  auto set{make()};
  const auto marginal{set.marginal(std::array{1})};
  set.setStates(Eigen::Matrix<double, 3, 9>::Zero());
  set.setBandwidth(bandwidth3);
  EXPECT_EQ(Eigen::MatrixXd{marginal.states()}, Eigen::MatrixXd::Zero(1, 9));
  EXPECT_TRUE(marginal.bandwidth().has_value());
}

TEST(ParticleSetBandwidth, StoredOrSilverman) {
  auto set{make()};
  const auto silverman{*EigenFilter::silvermanBandwidth(set)};
  EXPECT_EQ(set.kdeBandwidth(), silverman);
  EXPECT_EQ(set.toDensity()->bandwidth(), silverman);

  set.setBandwidth(bandwidth3);
  EXPECT_EQ(set.kdeBandwidth(), bandwidth3);
  EXPECT_EQ(set.toDensity()->bandwidth(), bandwidth3);

  std::mt19937_64 rng{1};
  ASSERT_TRUE(set.resample(rng).has_value());
  EXPECT_EQ(set.kdeBandwidth(), bandwidth3);  // kept through resampling

  set.clearBandwidth();
  EXPECT_FALSE(set.bandwidth().has_value());

  ParticleSetN<double, Eigen::Dynamic> dynamic{Eigen::MatrixXd::Random(2, 5)};
  dynamic.setBandwidth(Eigen::Matrix2d::Identity());
  dynamic.resizeParticles(3);
  EXPECT_FALSE(dynamic.bandwidth().has_value());
}

TEST(ParticleSetBandwidth, ToKernelDensityErrors) {
  EXPECT_EQ(MPS3{}.toDensity().error(), KernelDensityError{WeightError::Empty});
  const MPS3 collapsed{Eigen::Matrix<double, 3, 4>::Ones()};
  EXPECT_EQ(collapsed.toDensity().error(), KernelDensityError{GaussianError::NotPositiveDefinite});
}

TEST(KernelDensityConditional, IsJointOverMarginal) {
  auto set{make()};
  set.setBandwidth(bandwidth3);
  const Eigen::Matrix<double, 1, 1> v{0.35};
  const std::array given{1};

  auto conditional{set.conditional(given, v)};
  static_assert(std::is_same_v<decltype(conditional), ParticleSetN<double, 2>>);
  ASSERT_TRUE(conditional.bandwidth().has_value());
  const auto conditionalKde{conditional.toDensity()};
  ASSERT_TRUE(conditionalKde.has_value());

  const auto joint{set.toDensity()};
  const auto marginalOfGiven{set.marginal(given).copy()};
  const auto marginalKde{marginalOfGiven.toDensity()};
  for (const Eigen::Vector2d a : {Eigen::Vector2d{0.1, -0.2}, Eigen::Vector2d{-0.5, 0.6}, Eigen::Vector2d{0.9, 0.0}}) {
    const Eigen::Vector3d x{a(0), v(0), a(1)};
    EXPECT_NEAR(conditionalKde->logPdf(a), joint->logPdf(x) - marginalKde->logPdf(v), 1e-10);
  }
  EXPECT_NEAR(*conditional.logTotalWeight(), *set.logTotalWeight() + marginalKde->logPdf(v), 1e-10);
}

TEST(KernelDensityConditional, AllFlavorsAgree) {
  auto set{make()};
  set.setBandwidth(bandwidth3);
  const Eigen::Vector2d v{0.2, -0.3};
  const auto fromSet{set.conditional(std::array{0}, std::array{1, 2}, v)};
  const auto fromKde{set.toDensity()->conditional(std::array{0}, std::array{1, 2}, v)};
  EXPECT_EQ(fromSet.states(), fromKde.states());
  EXPECT_TRUE((fromSet.logWeights() == fromKde.logWeights()).all());
  EXPECT_EQ(*fromSet.bandwidth(), *fromKde.bandwidth());
}

TEST(KernelDensityConditional, SilvermanWhenNoBandwidthStored) {
  const auto set{make()};
  const Eigen::Matrix<double, 1, 1> v{0.0};
  const auto conditional{set.conditional(std::array{0}, v)};
  auto withSilverman{make()};
  withSilverman.setBandwidth(set.kdeBandwidth());
  EXPECT_EQ(conditional.states(), withSilverman.conditional(std::array{0}, v).states());
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

TEST(ParticleSetMarginalDeathTest, StructuralChangeWhileBorrowed) {
  auto set{std::make_unique<MPS3>(make())};
  const auto marginal{set->marginal(std::array{0, 1})};
  std::mt19937_64 rng{1};
  EXPECT_DEATH(set->add(0.0, Eigen::Vector3d::Zero()), "borrowed");
  EXPECT_DEATH(set->remove(0), "borrowed");
  EXPECT_DEATH(std::ignore = set->resample(rng), "borrowed");
  EXPECT_DEATH(set->clear(), "borrowed");
  EXPECT_DEATH({ auto stolen{std::move(*set)}; }, "borrowed");
  EXPECT_DEATH(set.reset(), "borrowed");
}

#endif
