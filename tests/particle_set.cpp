#include <EigenFilter/ParticleSet.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using EigenFilter::ParticleSet;
using EigenFilter::ParticleSetN;
using EigenFilter::StratifiedResampler;
using EigenFilter::WeightError;

namespace {

constexpr double inf{std::numeric_limits<double>::infinity()};

// Lets std::vector move rather than copy particle sets on reallocation.
static_assert(std::is_nothrow_move_constructible_v<ParticleSetN<double, Eigen::Dynamic>>);
static_assert(std::is_nothrow_move_constructible_v<ParticleSetN<double, 3>>);

// Each config names a ParticleSet type and the state dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config {
  using Type = ParticleSetN<Scalar, Size, MaxSize>;
  static constexpr Eigen::Index n{N};
};

using Configs = ::testing::Types<Config<double, 1, 1, 1>, Config<double, 3, 3, 3>,
                                 Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
                                 Config<double, Eigen::Dynamic, 6, 4>, Config<float, 2, 2, 2>>;

template <typename C>
class ParticleSetTest : public ::testing::Test {
 protected:
  using MPS = typename C::Type;
  using Scalar = typename MPS::Scalar;
  using SampleSetType = typename MPS::SampleSetType;
  using WeightArrayType = typename MPS::WeightArrayType;
  static constexpr Eigen::Index n{C::n};
  static constexpr Eigen::Index count{7};

  static Scalar tolerance() { return std::is_same_v<Scalar, float> ? Scalar(1e-5) : Scalar(1e-12); }

  static SampleSetType makeStates() {
    return SampleSetType::NullaryExpr(
        n, count, [](Eigen::Index i, Eigen::Index j) { return Scalar(std::sin(1.3 * double(i) + 0.7 * double(j))); });
  }

  // Unnormalized weights 1..count.
  static WeightArrayType makeLogWeights() {
    return WeightArrayType::NullaryExpr(count, [](Eigen::Index j) { return Scalar(std::log(double(j + 1))); });
  }

  static MPS make() { return MPS{makeStates(), makeLogWeights()}; }
};

}  // namespace

TYPED_TEST_SUITE(ParticleSetTest, Configs);

TYPED_TEST(ParticleSetTest, MomentsMatchReference) {
  const auto set{this->make()};
  const auto moments{set.moments()};
  ASSERT_TRUE(moments.has_value());

  const Eigen::MatrixXd x{set.states().template cast<double>()};
  const Eigen::VectorXd w{set.logWeights().template cast<double>().exp().matrix()};
  const Eigen::VectorXd wn{w / w.sum()};
  const Eigen::VectorXd mean{x * wn};
  Eigen::MatrixXd cov{Eigen::MatrixXd::Zero(this->n, this->n)};
  for (Eigen::Index j{0}; j < x.cols(); ++j) cov += wn(j) * (x.col(j) - mean) * (x.col(j) - mean).transpose();

  EXPECT_LT((moments->mean().template cast<double>() - mean).norm(), this->tolerance() * 10);
  EXPECT_LT((moments->cov().template cast<double>() - cov).norm(), this->tolerance() * 10);
  EXPECT_EQ(moments->cov(), moments->cov().transpose());
  EXPECT_NEAR(*set.logTotalWeight(), std::log(28.0), this->tolerance() * 10);
}

TYPED_TEST(ParticleSetTest, EffectiveSampleSize) {
  using Scalar = typename TestFixture::Scalar;
  auto set{this->make()};
  set.setLogWeights(TestFixture::WeightArrayType::Constant(this->count, Scalar(3)));
  EXPECT_NEAR(*set.effectiveSampleSize(), Scalar(this->count), this->tolerance() * 10);

  set.setLogWeights(TestFixture::WeightArrayType::Constant(this->count, -Eigen::NumTraits<Scalar>::infinity()));
  set.setLogWeight(2, Scalar(0));
  EXPECT_EQ(*set.effectiveSampleSize(), Scalar(1));
}

TYPED_TEST(ParticleSetTest, ResampleKeepsStatesAndNormalizes) {
  using Scalar = typename TestFixture::Scalar;
  auto set{this->make()};
  const auto original{set.states()};
  std::mt19937_64 rng{42};

  ASSERT_TRUE(set.resample(rng, 11).has_value());
  EXPECT_EQ(set.size(), 11);
  ASSERT_EQ(set.ancestors().size(), 11u);
  for (Eigen::Index j{0}; j < set.size(); ++j) {
    EXPECT_EQ(set.state(j), original.col(set.ancestors()[std::size_t(j)]));
    EXPECT_EQ(set.logWeight(j), -std::log(Scalar(11)));
  }
  EXPECT_NEAR(*set.logTotalWeight(), Scalar(0), this->tolerance() * 10);
}

namespace {

using MPS2 = ParticleSetN<double, 2>;

// A 2-D set with the given weights; particle i is at (i, -i).
MPS2 makeWeighted(const std::vector<double>& weights) {
  MPS2 set;
  for (std::size_t i{0}; i < weights.size(); ++i) set.add(std::log(weights[i]), Eigen::Vector2d{double(i), -double(i)});
  return set;
}

// Expects the set's weights, in any order, to match the ascending list \a expected.
void expectWeights(const MPS2& set, const std::vector<double>& expected) {
  std::vector<double> weights;
  for (Eigen::Index i{0}; i < set.size(); ++i) weights.push_back(std::exp(set.logWeight(i)));
  std::ranges::sort(weights);
  ASSERT_EQ(weights.size(), expected.size());
  for (std::size_t i{0}; i < weights.size(); ++i) EXPECT_NEAR(weights[i], expected[i], 1e-12) << "index " << i;
}

}  // namespace

TEST(ParticleSet, ConstructFromStatesHasEqualWeights) {
  const MPS2 set{Eigen::Matrix<double, 2, 4>::Random()};
  EXPECT_EQ(set.size(), 4);
  expectWeights(set, {0.25, 0.25, 0.25, 0.25});
}

TEST(ParticleSet, NormalizedWeights) {
  const auto set{makeWeighted({1.0, 3.0})};
  const auto weights{set.normalizedWeights()};
  ASSERT_TRUE(weights.has_value());
  EXPECT_NEAR((*weights)(0), 0.25, 1e-15);
  EXPECT_NEAR((*weights)(1), 0.75, 1e-15);
}

TEST(ParticleSet, WeightErrors) {
  const MPS2 empty;
  EXPECT_EQ(empty.moments().error(), WeightError::Empty);
  EXPECT_EQ(empty.effectiveSampleSize().error(), WeightError::Empty);
  EXPECT_EQ(empty.normalizedWeights().error(), WeightError::Empty);
  EXPECT_EQ(EigenFilter::silvermanBandwidth(empty).error(), WeightError::Empty);
  EXPECT_EQ(std::get<WeightError>(empty.toDensity().error()), WeightError::Empty);

  auto set{makeWeighted({1.0, 1.0})};
  set.scaleWeights(-inf);
  EXPECT_EQ(set.moments().error(), WeightError::ZeroSum);

  set.setLogWeight(0, std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(set.moments().error(), WeightError::NonFinite);
  set.setLogWeight(0, inf);
  EXPECT_EQ(set.moments().error(), WeightError::NonFinite);
}

TEST(ParticleSet, ResampleFailureLeavesSetUnchanged) {
  auto set{makeWeighted({1.0, 2.0})};
  set.scaleWeights(-inf);
  const auto before{set.states()};
  std::mt19937_64 rng{1};
  EXPECT_EQ(set.resample(rng).error(), WeightError::ZeroSum);
  EXPECT_EQ(set.states(), before);
}

TEST(ParticleSet, ResampleWithChosenResampler) {
  auto set{makeWeighted({0.0, 1.0, 0.0})};
  std::mt19937_64 rng{1};
  ASSERT_TRUE(set.resample(rng, StratifiedResampler{}).has_value());
  EXPECT_EQ(set.size(), 3);
  for (Eigen::Index j{0}; j < set.size(); ++j) EXPECT_EQ(set.state(j), Eigen::Vector2d(1.0, -1.0));
}

TEST(ParticleSet, RemoveMovesLastIntoPlace) {
  auto set{makeWeighted({1.0, 2.0, 3.0})};
  set.remove(0);
  EXPECT_EQ(set.size(), 2);
  EXPECT_EQ(set.state(0), Eigen::Vector2d(2.0, -2.0));
  set.remove(1);
  EXPECT_EQ(set.size(), 1);
  EXPECT_EQ(set.state(0), Eigen::Vector2d(2.0, -2.0));
}

TEST(ParticleSet, ForEachScalesWeightsAndModifiesParticles) {
  auto set{makeWeighted({1.0, 2.0})};
  set.forEach([](auto& p) {
    p.setState(2.0 * p.state());
    return std::log(3.0);
  });
  expectWeights(set, {3.0, 6.0});
  EXPECT_EQ(set.state(1), Eigen::Vector2d(2.0, -2.0));

  set.forEach([](auto& p) { p.setState(Eigen::Vector2d::Zero()); });
  expectWeights(set, {3.0, 6.0});
  EXPECT_EQ(set.states(), Eigen::Matrix2d::Zero());
}

TEST(ParticleSet, BulkStateAndWeightUpdates) {
  auto set{makeWeighted({1.0, 2.0, 4.0})};
  const Eigen::Matrix2d a{{0.0, 1.0}, {1.0, 0.0}};
  set.setStates(a * set.states());
  EXPECT_EQ(set.state(2), Eigen::Vector2d(-2.0, 2.0));

  set.scaleWeights(Eigen::Array3d{std::log(4.0), std::log(2.0), 0.0});
  expectWeights(set, {4.0, 4.0, 4.0});
}

TEST(ParticleSet, RemoveIf) {
  auto set{makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0})};
  const auto light{[](double logWeight, const auto&) { return logWeight < std::log(2.5); }};

  auto copy{set};
  copy.removeIf(light);
  expectWeights(copy, {3.0, 4.0, 5.0});

  const auto removed{set.removeIf<true>(light)};
  ASSERT_TRUE(removed.has_value());
  EXPECT_NEAR(*removed, std::log(3.0 / 15.0), 1e-12);
  expectWeights(set, {3.0, 4.0, 5.0});

  set.removeIf([](double, const auto& state) { return state(0) == 4.0; });
  expectWeights(set, {4.0, 5.0});
}

TEST(ParticleSet, RemoveIfReturningWeightFailsWithoutRemoving) {
  auto set{makeWeighted({1.0, 1.0})};
  set.scaleWeights(-inf);
  EXPECT_EQ(set.removeIf<true>([](double, const auto&) { return true; }).error(), WeightError::ZeroSum);
  EXPECT_EQ(set.size(), 2);
}

TEST(ParticleSet, NormalizeAndAppend) {
  auto set{makeWeighted({1.0, 3.0})};
  ASSERT_TRUE(set.normalize().has_value());
  expectWeights(set, {0.25, 0.75});

  set.append(set);
  EXPECT_EQ(set.size(), 4);
  EXPECT_EQ(set.state(3), Eigen::Vector2d(1.0, -1.0));
  expectWeights(set, {0.25, 0.25, 0.75, 0.75});
}

TEST(ParticleSet, ResizeParticlesDynamic) {
  ParticleSetN<double, Eigen::Dynamic> set{Eigen::MatrixXd::Zero(2, 3)};
  set.resizeParticles(4);
  EXPECT_EQ(set.particleSize(), 4);
  EXPECT_EQ(set.size(), 3);
  set.setStates(Eigen::MatrixXd::Ones(4, 3));
  EXPECT_EQ(set.moments()->mean(), Eigen::Vector4d::Ones());
}

TEST(ParticleSet, LockReleasedWhenKernelDensitiesDie) {
  auto set{makeWeighted({1.0, 2.0})};
  {
    auto kde{set.toDensity(Eigen::Matrix2d::Identity())};
    auto copy{*kde};
    auto moved{std::move(*kde)};
  }
  set.scaleWeights(1.0);  // would assert if any lock leaked
  set.forEach([](auto&) {});
  set.scaleWeights(1.0);  // would assert if forEach leaked its lock
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

TEST(ParticleSetDeathTest, MutateWhileLocked) {
  auto set{makeWeighted({1.0, 2.0})};
  const auto kde{set.toDensity(Eigen::Matrix2d::Identity())};
  std::mt19937_64 rng{1};
  EXPECT_DEATH(set.add(0.0, Eigen::Vector2d::Zero()), "locked");
  EXPECT_DEATH(set.remove(0), "locked");
  EXPECT_DEATH(set.setState(0, Eigen::Vector2d::Zero()), "locked");
  EXPECT_DEATH(set.scaleWeights(1.0), "locked");
  EXPECT_DEATH(set.forEach([](auto&) {}), "locked");
  EXPECT_DEATH(std::ignore = set.resample(rng), "locked");
  EXPECT_DEATH(set.clear(), "locked");
}

TEST(ParticleSetDeathTest, MutateSetInsideForEach) {
  auto set{makeWeighted({1.0, 2.0})};
  EXPECT_DEATH(set.forEach([&](auto&) { set.scaleWeights(1.0); }), "locked");
}

TEST(ParticleSetDeathTest, MoveWhileLocked) {
  auto set{makeWeighted({1.0, 2.0})};
  const auto kde{set.toDensity(Eigen::Matrix2d::Identity())};
  EXPECT_DEATH({ auto stolen{std::move(set)}; }, "locked");
}

TEST(ParticleSetDeathTest, DestroyWhileLocked) {
  EXPECT_DEATH(
      {
        auto set{std::make_unique<MPS2>(makeWeighted({1.0}))};
        auto kde{set->toDensity(Eigen::Matrix2d::Identity())};
        set.reset();
      },
      "locked");
}

#endif
