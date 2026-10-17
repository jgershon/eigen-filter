#include <EigenFilter/GaussianMixture.h>

#include <gtest/gtest.h>

#include <Eigen/LU>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

using EigenFilter::ComponentError;
using EigenFilter::GaussianError;
using EigenFilter::GaussianMixtureError;
using EigenFilter::MutableGaussianMixture;
using EigenFilter::MutableGaussianMixtureN;
using EigenFilter::WeightError;

namespace {

constexpr double inf{std::numeric_limits<double>::infinity()};

// Lets std::vector move rather than copy components on reallocation.
static_assert(std::is_nothrow_move_constructible_v<MutableGaussianMixtureN<double, Eigen::Dynamic>::Component>);
static_assert(std::is_nothrow_move_constructible_v<MutableGaussianMixtureN<double, Eigen::Dynamic>>);

// Each config names a MutableGaussianMixture type and the component dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config {
  using Type = MutableGaussianMixtureN<Scalar, Size, MaxSize>;
  static constexpr Eigen::Index n{N};
};

using Configs = ::testing::Types<Config<double, 1, 1, 1>, Config<double, 3, 3, 3>,
                                 Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
                                 Config<double, Eigen::Dynamic, 6, 4>, Config<float, 2, 2, 2>>;

template <typename C>
class GaussianMixtureTest : public ::testing::Test {
 protected:
  using MGM = typename C::Type;
  using Scalar = typename MGM::Scalar;
  using VectorType = typename MGM::VectorType;
  using MatrixType = typename MGM::MatrixType;
  static constexpr Eigen::Index n{C::n};

  static Scalar tolerance() { return std::is_same_v<Scalar, float> ? Scalar(1e-4) : Scalar(1e-10); }

  static VectorType makeVector(double phase) {
    return VectorType::NullaryExpr(n, [phase](Eigen::Index i) { return Scalar(std::sin(1.3 * i + phase)); });
  }

  // Deterministic symmetric positive definite matrix: A·Aᵀ + n·I.
  static MatrixType makeSpd(double phase) {
    const MatrixType a{MatrixType::NullaryExpr(
        n, n, [phase](Eigen::Index i, Eigen::Index j) { return Scalar(std::sin(7.0 * i + 3.0 * j + phase)); })};
    MatrixType p{a * a.transpose()};
    p.diagonal().array() += Scalar(n);
    return p;
  }

  // Three components with unnormalized weights 0.2, 1.5, 0.3.
  static MGM make() {
    MGM mixture{n};
    mixture.add(Scalar(std::log(0.2)), makeVector(0.0), makeSpd(1.0));
    mixture.add(Scalar(std::log(1.5)), makeVector(1.0), makeSpd(2.0));
    mixture.add(Scalar(std::log(0.3)), makeVector(2.0), makeSpd(3.0));
    return mixture;
  }

  // Direct evaluation through determinants and inverses, in double precision.
  static double referenceLogPdf(const MGM& mixture, const VectorType& x) {
    double density{0};
    double total{0};
    for (const auto& c : mixture) {
      const Eigen::MatrixXd p{c.gaussian.cov().template cast<double>()};
      const Eigen::VectorXd r{(x - c.gaussian.mean()).template cast<double>()};
      const double logN{
          -0.5 * (double(n) * std::log(2.0 * std::numbers::pi) + std::log(p.determinant()) + r.dot(p.inverse() * r))};
      const double w{std::exp(double(c.logWeight))};
      density += w * std::exp(logN);
      total += w;
    }
    return std::log(density / total);
  }

  static std::pair<Eigen::VectorXd, Eigen::MatrixXd> referenceMoments(const MGM& mixture) {
    double total{0};
    Eigen::VectorXd mean{Eigen::VectorXd::Zero(n)};
    for (const auto& c : mixture) {
      const double w{std::exp(double(c.logWeight))};
      total += w;
      mean += w * c.gaussian.mean().template cast<double>();
    }
    mean /= total;
    Eigen::MatrixXd cov{Eigen::MatrixXd::Zero(n, n)};
    for (const auto& c : mixture) {
      const double w{std::exp(double(c.logWeight))};
      const Eigen::VectorXd d{c.gaussian.mean().template cast<double>() - mean};
      cov += w * (c.gaussian.cov().template cast<double>() + d * d.transpose());
    }
    return {mean, cov / total};
  }
};

}  // namespace

TYPED_TEST_SUITE(GaussianMixtureTest, Configs);

TYPED_TEST(GaussianMixtureTest, LogPdfMatchesReference) {
  const auto mixture{this->make()};
  const auto g{mixture.toGaussianMixture()};
  ASSERT_TRUE(g.has_value());

  const auto x{this->makeVector(0.7)};
  const double expected{this->referenceLogPdf(mixture, x)};
  EXPECT_NEAR(g->logPdf(x), expected, this->tolerance() * std::abs(expected));
  EXPECT_NEAR(g->pdf(x), std::exp(g->logPdf(x)), this->tolerance());
  EXPECT_NEAR(g->logTotalWeight(), std::log(2.0), this->tolerance());
}

TYPED_TEST(GaussianMixtureTest, MomentsMatchReference) {
  const auto mixture{this->make()};
  const auto g{mixture.toGaussianMixture()};
  ASSERT_TRUE(g.has_value());

  const auto [mean, cov]{this->referenceMoments(mixture)};
  const auto moments{g->moments()};
  EXPECT_LT((moments.mean().template cast<double>() - mean).norm(), this->tolerance() * 10);
  EXPECT_LT((moments.cov().template cast<double>() - cov).norm(), this->tolerance() * 10 * cov.norm());
}

TYPED_TEST(GaussianMixtureTest, MomentsReadOnlyLowerTriangle) {
  auto mixture{this->make()};
  const auto expected{mixture.toGaussianMixture()->moments()};

  for (auto it{mixture.begin()}; it != mixture.end(); ++it) {
    auto cov{it->gaussian.cov()};
    cov.template triangularView<Eigen::StrictlyUpper>().setConstant(12345);
    mixture.setCov(it, cov);
  }
  const auto moments{mixture.toGaussianMixture()->moments()};
  EXPECT_EQ(moments.mean(), expected.mean());
  EXPECT_EQ(moments.cov(), expected.cov());
}

TYPED_TEST(GaussianMixtureTest, WeightScaleDoesNotChangeDensity) {
  using Scalar = typename TestFixture::Scalar;
  auto mixture{this->make()};
  const auto x{this->makeVector(0.7)};
  const auto logPdf{mixture.toGaussianMixture()->logPdf(x)};

  mixture.scaleWeights(Scalar(5));
  const auto g{mixture.toGaussianMixture()};
  EXPECT_NEAR(g->logPdf(x), logPdf, this->tolerance() * std::abs(logPdf));
  EXPECT_NEAR(g->logTotalWeight(), std::log(2.0) + 5.0, this->tolerance() * 10);
}

TYPED_TEST(GaussianMixtureTest, EffectiveSampleSize) {
  using Scalar = typename TestFixture::Scalar;
  auto mixture{this->make()};
  // Normalized weights 0.1, 0.75, 0.15.
  EXPECT_NEAR(mixture.toGaussianMixture()->effectiveSampleSize(), Scalar(1 / (0.01 + 0.5625 + 0.0225)),
              this->tolerance() * 10);

  for (auto it{mixture.begin()}; it != mixture.end(); ++it) mixture.setLogWeight(it, Scalar(2));
  EXPECT_NEAR(mixture.toGaussianMixture()->effectiveSampleSize(), Scalar(3), this->tolerance() * 10);

  mixture.setLogWeight(mixture.begin(), -Eigen::NumTraits<Scalar>::infinity());
  mixture.setLogWeight(mixture.begin() + 1, -Eigen::NumTraits<Scalar>::infinity());
  EXPECT_EQ(mixture.toGaussianMixture()->effectiveSampleSize(), Scalar(1));
}

namespace {

using MGM2 = MutableGaussianMixtureN<double, 2>;

// A 2-D mixture with the given weights, each component at a distinct mean.
MGM2 makeWeighted(const std::vector<double>& weights) {
  MGM2 mixture;
  for (std::size_t i{0}; i < weights.size(); ++i) {
    mixture.add(std::log(weights[i]), Eigen::Vector2d::Constant(double(i)), Eigen::Matrix2d::Identity());
  }
  return mixture;
}

// Expects the mixture's weights, in any order, to match the ascending list  expected.
void expectWeights(const MGM2& mixture, const std::vector<double>& expected) {
  std::vector<double> weights;
  for (const auto& c : mixture) weights.push_back(std::exp(c.logWeight));
  std::ranges::sort(weights);
  ASSERT_EQ(weights.size(), expected.size());
  for (std::size_t i{0}; i < weights.size(); ++i) EXPECT_NEAR(weights[i], expected[i], 1e-12) << "index " << i;
}

}  // namespace

TEST(GaussianMixture, EmptyFails) {
  const MGM2 mixture;
  EXPECT_EQ(mixture.logTotalWeight().error(), WeightError::Empty);
  EXPECT_EQ(mixture.toGaussianMixture().error(), GaussianMixtureError{WeightError::Empty});
}

TEST(GaussianMixture, ZeroWeightFails) {
  auto mixture{makeWeighted({1.0, 1.0})};
  mixture.scaleWeights(-inf);
  EXPECT_EQ(mixture.toGaussianMixture().error(), GaussianMixtureError{WeightError::ZeroSum});
}

TEST(GaussianMixture, NonFiniteWeightFails) {
  auto mixture{makeWeighted({1.0, 1.0})};
  mixture.setLogWeight(mixture.begin(), std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(mixture.toGaussianMixture().error(), GaussianMixtureError{WeightError::NonFinite});
  mixture.setLogWeight(mixture.begin(), inf);
  EXPECT_EQ(mixture.toGaussianMixture().error(), GaussianMixtureError{WeightError::NonFinite});
}

TEST(GaussianMixture, ComponentFailureNamesComponent) {
  auto mixture{makeWeighted({1.0, 1.0, 1.0})};
  mixture.setCov(mixture.begin() + 1, -Eigen::Matrix2d::Identity());
  const auto g{mixture.toGaussianMixture()};
  ASSERT_FALSE(g.has_value());
  EXPECT_EQ(g.error(), GaussianMixtureError{(ComponentError{1, GaussianError::NotPositiveDefinite})});
}

TEST(GaussianMixture, ZeroWeightComponentIsIgnored) {
  auto mixture{makeWeighted({1.0, 1.0})};
  mixture.setLogWeight(mixture.begin(), -inf);
  const auto g{mixture.toGaussianMixture()};
  ASSERT_TRUE(g.has_value());
  const auto moments{g->moments()};
  EXPECT_EQ(moments.mean(), Eigen::Vector2d::Constant(1.0));
  EXPECT_EQ(moments.cov(), Eigen::Matrix2d::Identity());
}

TEST(GaussianMixture, RemoveMovesLastIntoPlace) {
  auto mixture{makeWeighted({1.0, 2.0, 3.0})};
  const auto next{mixture.remove(mixture.begin())};
  EXPECT_EQ(mixture.size(), 2);
  EXPECT_EQ(next, mixture.begin());
  EXPECT_DOUBLE_EQ(next->logWeight, std::log(3.0));
  const auto last{mixture.remove(mixture.begin() + 1)};
  EXPECT_EQ(last, mixture.end());
}

TEST(GaussianMixture, ForEachScalesWeightsAndModifiesComponents) {
  auto mixture{makeWeighted({1.0, 2.0})};
  mixture.forEach([](auto& c) {
    c.setMean(Eigen::Vector2d::Zero());
    return std::log(2.0);
  });
  expectWeights(mixture, {2.0, 4.0});
  for (const auto& c : mixture) EXPECT_EQ(c.gaussian.mean(), Eigen::Vector2d::Zero());

  mixture.forEach([](auto& c) { c.setCov(2.0 * Eigen::Matrix2d::Identity()); });
  expectWeights(mixture, {2.0, 4.0});
  for (const auto& c : mixture) EXPECT_EQ(c.gaussian.cov(), 2.0 * Eigen::Matrix2d::Identity());
}

TEST(GaussianMixture, MergeSumsWeights) {
  auto mixture{makeWeighted({1.0, 3.0, 5.0})};
  const EigenFilter::MomentsN<double, 2> merged{Eigen::Vector2d{7.0, 8.0}, Eigen::Matrix2d::Identity()};
  const auto it{mixture.merge(mixture.begin(), mixture.begin() + 1, merged)};
  EXPECT_EQ(mixture.size(), 2);
  EXPECT_DOUBLE_EQ(it->logWeight, std::log(4.0));
  EXPECT_EQ(it->gaussian.mean(), merged.mean());
  expectWeights(mixture, {4.0, 5.0});
}

TEST(GaussianMixture, MergeIntoLastComponent) {
  auto mixture{makeWeighted({1.0, 3.0, 5.0})};
  const auto last{mixture.begin() + 2};
  const auto it{mixture.merge(last, mixture.begin(), last->gaussian.moments())};
  EXPECT_EQ(mixture.size(), 2);
  EXPECT_DOUBLE_EQ(it->logWeight, std::log(6.0));
  EXPECT_EQ(it->gaussian.mean(), Eigen::Vector2d::Constant(2.0));
}

TEST(GaussianMixture, SplitRedistributesWeight) {
  auto mixture{makeWeighted({6.0, 1.0})};
  const EigenFilter::MomentsN<double, 2> a{Eigen::Vector2d{-1.0, 0.0}, Eigen::Matrix2d::Identity()};
  const EigenFilter::MomentsN<double, 2> b{Eigen::Vector2d{1.0, 0.0}, Eigen::Matrix2d::Identity()};
  const std::array children{std::pair{std::log(1.0), a}, std::pair{std::log(2.0), b}};
  mixture.split(mixture.begin(), children);

  ASSERT_EQ(mixture.size(), 3);
  EXPECT_DOUBLE_EQ(mixture.begin()->logWeight, std::log(2.0));
  EXPECT_EQ(mixture.begin()->gaussian.mean(), a.mean());
  EXPECT_DOUBLE_EQ((mixture.begin() + 2)->logWeight, std::log(4.0));
  EXPECT_EQ((mixture.begin() + 2)->gaussian.mean(), b.mean());
  EXPECT_NEAR(*mixture.logTotalWeight(), std::log(7.0), 1e-12);
}

TEST(GaussianMixture, RemoveIf) {
  auto mixture{makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0})};
  const auto light{[](const MGM2::Component& c) { return c.logWeight < std::log(2.5); }};

  auto copy{mixture};
  copy.removeIf(light);
  expectWeights(copy, {3.0, 4.0, 5.0});

  const auto removed{mixture.removeIf<true>(light)};
  ASSERT_TRUE(removed.has_value());
  EXPECT_NEAR(*removed, std::log(3.0 / 15.0), 1e-12);
  expectWeights(mixture, {3.0, 4.0, 5.0});
}

TEST(GaussianMixture, RemoveIfReturningWeightFailsWithoutRemoving) {
  auto mixture{makeWeighted({1.0, 1.0})};
  mixture.scaleWeights(-inf);
  const auto removed{mixture.removeIf<true>([](const auto&) { return true; })};
  EXPECT_EQ(removed.error(), WeightError::ZeroSum);
  EXPECT_EQ(mixture.size(), 2);

  mixture.removeIf([](const auto&) { return true; });
  EXPECT_TRUE(mixture.empty());
}

TEST(GaussianMixture, TruncateKeepsHeaviest) {
  auto mixture{makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0})};
  auto copy{mixture};
  copy.truncate(3);
  expectWeights(copy, {3.0, 4.0, 5.0});

  const auto removed{mixture.truncate<true>(3)};
  ASSERT_TRUE(removed.has_value());
  EXPECT_NEAR(*removed, std::log(3.0 / 15.0), 1e-12);
  expectWeights(mixture, {3.0, 4.0, 5.0});

  const auto none{mixture.truncate<true>(3)};
  EXPECT_EQ(*none, -inf);
  EXPECT_EQ(mixture.size(), 3);
}

TEST(GaussianMixture, RemoveLightest) {
  auto mixture{makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0})};
  const auto removed{mixture.removeLightest(std::log(0.25))};
  ASSERT_TRUE(removed.has_value());
  EXPECT_NEAR(*removed, std::log(3.0 / 15.0), 1e-12);
  expectWeights(mixture, {3.0, 4.0, 5.0});

  EXPECT_EQ(MGM2{}.removeLightest(0.0).error(), WeightError::Empty);
}

TEST(GaussianMixture, Normalize) {
  auto mixture{makeWeighted({1.0, 3.0})};
  ASSERT_TRUE(mixture.normalize().has_value());
  EXPECT_NEAR(*mixture.logTotalWeight(), 0.0, 1e-15);
  EXPECT_NEAR(std::exp(mixture.begin()->logWeight), 0.25, 1e-15);
}

TEST(GaussianMixture, AssignAndAppend) {
  using Moments = EigenFilter::MomentsN<double, 2>;
  MGM2 mixture;
  const std::vector components{std::pair{0.0, Moments{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()}},
                               std::pair{1.0, Moments{Eigen::Vector2d::Ones(), Eigen::Matrix2d::Identity()}}};
  mixture.assign(components);
  EXPECT_EQ(mixture.size(), 2);
  EXPECT_EQ((mixture.begin() + 1)->gaussian.mean(), Eigen::Vector2d::Ones());

  mixture.append(mixture);
  EXPECT_EQ(mixture.size(), 4);
  EXPECT_EQ((mixture.begin() + 3)->gaussian.mean(), Eigen::Vector2d::Ones());
}

TEST(GaussianMixture, ResizeComponentsDynamic) {
  MutableGaussianMixtureN<double, Eigen::Dynamic> mixture{2};
  mixture.add(0.0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
  EXPECT_EQ(mixture.componentSize(), 2);

  mixture.resizeComponents(3);
  EXPECT_EQ(mixture.componentSize(), 3);
  mixture.setMean(mixture.begin(), Eigen::Vector3d::Zero());
  mixture.setCov(mixture.begin(), Eigen::Matrix3d::Identity());
  mixture.add(0.0, Eigen::Vector3d::Ones(), Eigen::Matrix3d::Identity());
  const auto g{mixture.toGaussianMixture()};
  ASSERT_TRUE(g.has_value());
  EXPECT_EQ(g->moments().mean(), Eigen::Vector3d::Constant(0.5));
}

TEST(GaussianMixture, LockReleasedWhenViewsDie) {
  auto mixture{makeWeighted({1.0, 2.0})};
  {
    auto g{mixture.toGaussianMixture()};
    auto copy{*g};
    auto moved{std::move(*g)};
  }
  mixture.scaleWeights(1.0);  // would assert if any lock leaked
  mixture.forEach([](auto&) {});
  mixture.scaleWeights(1.0);  // would assert if forEach leaked its lock
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

TEST(GaussianMixtureDeathTest, MutateWhileLocked) {
  auto mixture{makeWeighted({1.0, 2.0})};
  const auto g{mixture.toGaussianMixture()};
  EXPECT_DEATH(mixture.add(0.0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()), "locked");
  EXPECT_DEATH(mixture.remove(mixture.begin()), "locked");
  EXPECT_DEATH(mixture.setMean(mixture.begin(), Eigen::Vector2d::Zero()), "locked");
  EXPECT_DEATH(mixture.scaleWeights(1.0), "locked");
  EXPECT_DEATH(mixture.forEach([](auto&) {}), "locked");
  EXPECT_DEATH(mixture.truncate(1), "locked");
  EXPECT_DEATH(mixture.clear(), "locked");
}

TEST(GaussianMixtureDeathTest, MutateMixtureInsideForEach) {
  auto mixture{makeWeighted({1.0, 2.0})};
  EXPECT_DEATH(mixture.forEach([&](auto&) { mixture.scaleWeights(1.0); }), "locked");
}

TEST(GaussianMixtureDeathTest, MoveWhileLocked) {
  auto mixture{makeWeighted({1.0, 2.0})};
  const auto g{mixture.toGaussianMixture()};
  EXPECT_DEATH({ auto stolen{std::move(mixture)}; }, "locked");
}

TEST(GaussianMixtureDeathTest, DestroyWhileLocked) {
  EXPECT_DEATH(
      {
        auto mixture{std::make_unique<MGM2>(makeWeighted({1.0}))};
        auto g{mixture->toGaussianMixture()};
        mixture.reset();
      },
      "locked");
}

#endif
