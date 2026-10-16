#include <EigenFilter/KernelDensity.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <optional>
#include <type_traits>
#include <utility>

using EigenFilter::GaussianError;
using EigenFilter::MutableGaussian;
using EigenFilter::MutableParticleSet;

namespace {

constexpr double inf{std::numeric_limits<double>::infinity()};

// Each config names a MutableParticleSet type and the particle dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config {
  using Type = MutableParticleSet<Scalar, Size, MaxSize>;
  static constexpr Eigen::Index n{N};
};

using Configs = ::testing::Types<Config<double, 1, 1, 1>, Config<double, 3, 3, 3>,
                                 Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
                                 Config<double, Eigen::Dynamic, 6, 4>, Config<float, 2, 2, 2>>;

template <typename C>
class KernelDensityTest : public ::testing::Test {
 protected:
  using MPS = typename C::Type;
  using Scalar = typename MPS::Scalar;
  using VectorType = typename MPS::VectorType;
  using MatrixType = typename MPS::MatrixType;
  using SampleSetType = typename MPS::SampleSetType;
  using WeightArrayType = typename MPS::WeightArrayType;
  static constexpr Eigen::Index n{C::n};
  static constexpr Eigen::Index count{9};

  static Scalar tolerance() { return std::is_same_v<Scalar, float> ? Scalar(1e-4) : Scalar(1e-10); }

  static MPS make() {
    const SampleSetType states{SampleSetType::NullaryExpr(n, count, [](Eigen::Index i, Eigen::Index j) {
      return Scalar(std::sin(1.3 * double(i * j) + 0.7 * double(j) + double(i)));
    })};
    const WeightArrayType logWeights{
        WeightArrayType::NullaryExpr(count, [](Eigen::Index j) { return Scalar(std::log(double(j + 1))); })};
    return MPS{states, logWeights};
  }

  static MatrixType makeBandwidth() {
    MatrixType h{MatrixType::Identity(n, n) * Scalar(0.3)};
    if (n > 1) h(1, 0) = h(0, 1) = Scalar(0.1);
    return h;
  }

  static VectorType makePoint() {
    return VectorType::NullaryExpr(n, [](Eigen::Index i) { return Scalar(0.2 * double(i) - 0.1); });
  }
};

}  // namespace

TYPED_TEST_SUITE(KernelDensityTest, Configs);

TYPED_TEST(KernelDensityTest, LogPdfMatchesSumOfKernels) {
  using Scalar = typename TestFixture::Scalar;
  const auto set{this->make()};
  const auto view{set.toParticleSet()};
  ASSERT_TRUE(view.has_value());
  const auto kde{EigenFilter::kernelDensity(*view, this->makeBandwidth())};
  ASSERT_TRUE(kde.has_value());

  const auto x{this->makePoint()};
  double density{0};
  double total{0};
  for (Eigen::Index i{0}; i < set.size(); ++i) {
    const typename TestFixture::VectorType center{set.state(i)};
    const MutableGaussian<Scalar, TestFixture::MPS::SizeAtCompileTime, TestFixture::MPS::MaxSizeAtCompileTime> kernel{
        center, this->makeBandwidth()};
    const double w{std::exp(double(set.logWeight(i)))};
    density += w * std::exp(double(kernel.toGaussian()->logPdf(x)));
    total += w;
  }
  const double expected{std::log(density / total)};
  EXPECT_NEAR(kde->logPdf(x), expected, this->tolerance() * std::abs(expected));
  EXPECT_NEAR(kde->pdf(x), std::exp(kde->logPdf(x)), this->tolerance());
}

TYPED_TEST(KernelDensityTest, MomentsAddBandwidth) {
  const auto set{this->make()};
  const auto view{set.toParticleSet()};
  const auto kde{EigenFilter::kernelDensity(*view, this->makeBandwidth())};
  ASSERT_TRUE(kde.has_value());
  const auto particleMoments{view->moments()};
  const auto moments{kde->moments()};
  EXPECT_EQ(moments.mean(), particleMoments.mean());
  EXPECT_LT((moments.cov() - particleMoments.cov() - this->makeBandwidth()).norm(), this->tolerance());
}

TYPED_TEST(KernelDensityTest, SilvermanBandwidth) {
  using Scalar = typename TestFixture::Scalar;
  const auto set{this->make()};
  const auto view{set.toParticleSet()};
  const double d{double(this->n)};
  const double factor{std::pow(4.0 / (double(view->effectiveSampleSize()) * (d + 2)), 2 / (d + 4))};
  const auto expected{(view->moments().cov() * Scalar(factor)).eval()};
  EXPECT_LT((EigenFilter::silvermanBandwidth(*view) - expected).norm(), this->tolerance() * expected.norm());

  const auto kde{EigenFilter::kernelDensity(*view)};
  ASSERT_TRUE(kde.has_value());
  EXPECT_EQ(kde->bandwidth(), EigenFilter::silvermanBandwidth(*view));
}

TEST(KernelDensity, SingleParticleIsGaussian) {
  const Eigen::Vector2d center{1.0, -2.0};
  const Eigen::Matrix2d h{{2.0, 0.5}, {0.5, 1.0}};
  const MutableParticleSet<double, 2> set{Eigen::Matrix<double, 2, 1>{center}};
  const auto view{set.toParticleSet()};
  const auto kde{EigenFilter::kernelDensity(*view, h)};
  ASSERT_TRUE(kde.has_value());

  const MutableGaussian<double, 2> gaussian{center, h};
  const Eigen::Vector2d x{0.3, 0.4};
  EXPECT_NEAR(kde->logPdf(x), gaussian.toGaussian()->logPdf(x), 1e-12);
}

TEST(KernelDensity, DegenerateParticlesFail) {
  const MutableParticleSet<double, 2> set{Eigen::Matrix<double, 2, 4>::Ones()};
  const auto view{set.toParticleSet()};
  const auto kde{EigenFilter::kernelDensity(*view)};
  ASSERT_FALSE(kde.has_value());
  EXPECT_EQ(kde.error(), GaussianError::NotPositiveDefinite);
}

TEST(KernelDensity, KeepsParticleSetLocked) {
  MutableParticleSet<double, 2> set{Eigen::Matrix<double, 2, 3>::Random()};
  {
    std::optional<EigenFilter::KernelDensity<double, 2>> kde;
    {
      const auto view{set.toParticleSet()};
      kde.emplace(*EigenFilter::kernelDensity(*view, Eigen::Matrix2d::Identity()));
    }
#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
    EXPECT_DEATH(set.scaleWeights(1.0), "locked");
#endif
  }
  set.scaleWeights(1.0);  // would assert if the estimate leaked its lock
}

TYPED_TEST(KernelDensityTest, SampleMomentsConverge) {
  const auto set{this->make()};
  const auto view{set.toParticleSet()};
  const auto kde{EigenFilter::kernelDensity(*view, this->makeBandwidth())};
  ASSERT_TRUE(kde.has_value());
  std::mt19937_64 rng{13};
  constexpr Eigen::Index count{40000};
  const auto samples{EigenFilter::drawSamples(*kde, count, rng)};
  ASSERT_EQ(samples.rows(), this->n);
  ASSERT_EQ(samples.cols(), count);

  const auto moments{kde->moments()};
  const Eigen::MatrixXd x{samples.template cast<double>()};
  const Eigen::VectorXd mean{x.rowwise().mean()};
  const Eigen::MatrixXd centered{x.colwise() - mean};
  const Eigen::MatrixXd cov{centered * centered.transpose() / double(count - 1)};
  const Eigen::MatrixXd p{moments.cov().template cast<double>()};
  for (Eigen::Index i{0}; i < this->n; ++i) {
    EXPECT_NEAR(mean(i), double(moments.mean()(i)), 5 * std::sqrt(p(i, i) / double(count))) << "mean " << i;
  }
  EXPECT_LT((cov - p).norm(), 0.05 * p.norm());
}

TYPED_TEST(KernelDensityTest, SampledLogPdfMatchesEstimate) {
  using Scalar = typename TestFixture::Scalar;
  const auto set{this->make()};
  const auto view{set.toParticleSet()};
  const auto kde{EigenFilter::kernelDensity(*view, this->makeBandwidth())};
  std::mt19937_64 a{17};
  std::mt19937_64 b{17};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(*kde, 20, a)};
  EXPECT_EQ(drawn.samples, EigenFilter::drawSamples(*kde, 20, b));
  ASSERT_EQ(drawn.logPdf.size(), 20);
  for (Eigen::Index j{0}; j < 20; ++j) {
    EXPECT_EQ(drawn.logPdf(j), kde->logPdf(drawn.samples.col(j))) << "sample " << j;
  }
  EXPECT_TRUE(drawn.logPdf.isFinite().all());
  static_assert(std::is_same_v<typename decltype(drawn.logPdf)::Scalar, Scalar>);
}

TEST(KernelDensity, SamplesStayNearWeightedParticles) {
  // Only the middle particle has weight, and the kernel is narrow.
  const Eigen::Matrix<double, 2, 3> states{{-10.0, 0.0, 10.0}, {0.0, 5.0, 0.0}};
  const MutableParticleSet<double, 2> set{states, Eigen::Array3d{-inf, 0.0, -inf}};
  const auto view{set.toParticleSet()};
  const auto kde{EigenFilter::kernelDensity(*view, Eigen::Matrix2d::Identity() * 1e-4)};
  std::mt19937_64 rng{19};
  const auto samples{EigenFilter::drawSamples(*kde, 100, rng, EigenFilter::SystematicResampler{})};
  EXPECT_LT((samples.colwise() - states.col(1)).cwiseAbs().maxCoeff(), 0.1);
}
