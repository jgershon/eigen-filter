#include <EigenFilter/GaussianMixture.h>
#include <EigenFilter/ParticleSet.h>
#include <EigenFilter/Sampling.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <type_traits>

using EigenFilter::MutableGaussian;
using EigenFilter::MutableGaussianN;

namespace {

// Each config names a MutableGaussian type and the dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config {
  using Type = MutableGaussianN<Scalar, Size, MaxSize>;
  static constexpr Eigen::Index n{N};
};

using Configs = ::testing::Types<Config<double, 1, 1, 1>, Config<double, 3, 3, 3>,
                                 Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
                                 Config<double, Eigen::Dynamic, 6, 4>, Config<float, 2, 2, 2>>;

template <typename C>
class SamplingTest : public ::testing::Test {
 protected:
  using MG = typename C::Type;
  using Scalar = typename MG::Scalar;
  using VectorType = typename MG::VectorType;
  using MatrixType = typename MG::MatrixType;
  static constexpr Eigen::Index n{C::n};

  // Deterministic symmetric positive definite matrix: A·Aᵀ + I.
  static MatrixType makeSpd() {
    const MatrixType a{MatrixType::NullaryExpr(
        n, n, [](Eigen::Index i, Eigen::Index j) { return Scalar(std::sin(7.0 * i + 3.0 * j + 1.0)); })};
    MatrixType p{a * a.transpose()};
    p.diagonal().array() += Scalar(1);
    return p;
  }

  static MG make() {
    return MG{VectorType::NullaryExpr(n, [](Eigen::Index i) { return Scalar(1.0 + double(i)); }), makeSpd()};
  }
};

}  // namespace

TYPED_TEST_SUITE(SamplingTest, Configs);

TYPED_TEST(SamplingTest, SampleMomentsConverge) {
  const auto mg{this->make()};
  const auto g{mg.toGaussian()};
  ASSERT_TRUE(g.has_value());
  std::mt19937_64 rng{7};
  constexpr Eigen::Index count{40000};
  const auto samples{EigenFilter::drawSamples(*g, count, rng)};
  ASSERT_EQ(samples.rows(), this->n);
  ASSERT_EQ(samples.cols(), count);

  const Eigen::MatrixXd x{samples.template cast<double>()};
  const Eigen::VectorXd mean{x.rowwise().mean()};
  const Eigen::MatrixXd centered{x.colwise() - mean};
  const Eigen::MatrixXd cov{centered * centered.transpose() / double(count - 1)};
  const Eigen::MatrixXd p{mg.cov().template cast<double>()};
  for (Eigen::Index i{0}; i < this->n; ++i) {
    EXPECT_NEAR(mean(i), double(mg.mean()(i)), 5 * std::sqrt(p(i, i) / double(count))) << "mean " << i;
  }
  EXPECT_LT((cov - p).norm(), 0.05 * p.norm());
}

TYPED_TEST(SamplingTest, LogPdfMatchesGaussian) {
  using Scalar = typename TestFixture::Scalar;
  const auto mg{this->make()};
  const auto g{mg.toGaussian()};
  std::mt19937_64 rng{11};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(*g, 50, rng)};
  ASSERT_EQ(drawn.logPdf.size(), 50);
  const Scalar tolerance{std::is_same_v<Scalar, float> ? Scalar(1e-4) : Scalar(1e-10)};
  for (Eigen::Index i{0}; i < 50; ++i) {
    const Scalar expected{g->logPdf(drawn.samples.col(i))};
    EXPECT_NEAR(drawn.logPdf(i), expected, tolerance * std::abs(expected)) << "sample " << i;
  }
}

TYPED_TEST(SamplingTest, BothVersionsDrawTheSameSamples) {
  const auto mg{this->make()};
  const auto g{mg.toGaussian()};
  std::mt19937_64 a{3};
  std::mt19937_64 b{3};
  EXPECT_EQ(EigenFilter::drawSamples(*g, 20, a), EigenFilter::drawSamplesWithLogPdf(*g, 20, b).samples);
}

TEST(Sampling, FeedsParticleSet) {
  const MutableGaussianN<double, 2> mg{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
  std::mt19937_64 rng{5};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(*mg.toGaussian(), 10, rng)};
  // Importance weights for a uniform target, as a particle filter proposal would use.
  const EigenFilter::MutableParticleSetN<double, 2> set{drawn.samples, -drawn.logPdf};
  EXPECT_EQ(set.size(), 10);
  EXPECT_TRUE((set.logWeights() == -drawn.logPdf).all());
}

TEST(Sampling, ZeroSamples) {
  const MutableGaussianN<double, Eigen::Dynamic> mg{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};
  std::mt19937_64 rng{5};
  const auto samples{EigenFilter::drawSamples(*mg.toGaussian(), 0, rng)};
  EXPECT_EQ(samples.rows(), 3);
  EXPECT_EQ(samples.cols(), 0);
}

namespace {

// Three well-separated, narrow 2-D components with unnormalized weights 1, 2, 5.
EigenFilter::MutableGaussianMixtureN<double, 2> makeSeparatedMixture(double spread) {
  EigenFilter::MutableGaussianMixtureN<double, 2> mixture;
  const Eigen::Matrix2d cov{{spread, 0.3 * spread}, {0.3 * spread, 0.5 * spread}};
  mixture.add(std::log(1.0), Eigen::Vector2d{-10.0, 0.0}, cov);
  mixture.add(std::log(2.0), Eigen::Vector2d{0.0, 10.0}, cov);
  mixture.add(std::log(5.0), Eigen::Vector2d{10.0, 0.0}, cov);
  return mixture;
}

}  // namespace

TEST(MixtureSampling, SampleMomentsConverge) {
  const auto mixture{makeSeparatedMixture(4.0)};
  const auto g{mixture.toGaussianMixture()};
  ASSERT_TRUE(g.has_value());
  std::mt19937_64 rng{23};
  constexpr Eigen::Index count{40000};
  const auto samples{EigenFilter::drawSamples(*g, count, rng)};
  ASSERT_EQ(samples.cols(), count);

  const auto moments{g->moments()};
  const Eigen::Vector2d mean{samples.rowwise().mean()};
  const Eigen::Matrix2Xd centered{samples.colwise() - mean};
  const Eigen::Matrix2d cov{centered * centered.transpose() / double(count - 1)};
  for (Eigen::Index i{0}; i < 2; ++i) {
    EXPECT_NEAR(mean(i), moments.mean()(i), 5 * std::sqrt(moments.cov()(i, i) / double(count))) << "mean " << i;
  }
  EXPECT_LT((cov - moments.cov()).norm(), 0.05 * moments.cov().norm());
}

TEST(MixtureSampling, SystematicGivesEachComponentItsShare) {
  const auto mixture{makeSeparatedMixture(0.01)};
  const auto g{mixture.toGaussianMixture()};
  std::mt19937_64 rng{29};
  const auto samples{EigenFilter::drawSamples(*g, 80, rng, EigenFilter::SystematicResampler{})};
  std::array<int, 3> counts{};
  for (Eigen::Index j{0}; j < samples.cols(); ++j) {
    for (std::size_t c{0}; c < 3; ++c) {
      if ((samples.col(j) - (g->begin() + Eigen::Index(c))->gaussian.mean()).norm() < 1.0) ++counts[c];
    }
  }
  EXPECT_EQ(counts, (std::array<int, 3>{10, 20, 50}));
}

TEST(MixtureSampling, LogPdfMatchesMixture) {
  const auto mixture{makeSeparatedMixture(4.0)};
  const auto g{mixture.toGaussianMixture()};
  std::mt19937_64 a{31};
  std::mt19937_64 b{31};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(*g, 25, a)};
  EXPECT_EQ(drawn.samples, EigenFilter::drawSamples(*g, 25, b));
  for (Eigen::Index j{0}; j < 25; ++j) EXPECT_EQ(drawn.logPdf(j), g->logPdf(drawn.samples.col(j))) << "sample " << j;
}
