#include <EigenFilter/Gaussian.h>

#include <gtest/gtest.h>

#include <Eigen/LU>

#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>

using EigenFilter::GaussianError;
using EigenFilter::MutableGaussian;

namespace {

// Each config names a MutableGaussian type and the runtime dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config {
  using Type = MutableGaussian<Scalar, Size, MaxSize>;
  static constexpr Eigen::Index n = N;
};

using Configs = ::testing::Types<Config<double, 1, 1, 1>, Config<double, 3, 3, 3>,
                                 Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
                                 Config<double, Eigen::Dynamic, 6, 4>, Config<float, 2, 2, 2>>;

template <typename C>
class GaussianTest : public ::testing::Test {
 protected:
  using MG = typename C::Type;
  using Scalar = typename MG::Scalar;
  using VectorType = typename MG::VectorType;
  using MatrixType = typename MG::MatrixType;
  static constexpr Eigen::Index n = C::n;

  static Scalar tolerance() { return std::is_same_v<Scalar, float> ? Scalar(1e-4) : Scalar(1e-10); }

  static VectorType makeVector(double phase) {
    return VectorType::NullaryExpr(n, [phase](Eigen::Index i) { return Scalar(std::sin(1.3 * i + phase)); });
  }

  // Deterministic symmetric positive definite matrix: A·Aᵀ + n·I.
  static MatrixType makeSpd() {
    const MatrixType a = MatrixType::NullaryExpr(
        n, n, [](Eigen::Index i, Eigen::Index j) { return Scalar(std::sin(7.0 * i + 3.0 * j + 1.0)); });
    MatrixType p = a * a.transpose();
    p.diagonal().array() += Scalar(n);
    return p;
  }

  // Direct evaluation through determinant and inverse, in double precision.
  static double referenceLogPdf(const VectorType& mean, const MatrixType& cov, const VectorType& x) {
    const Eigen::MatrixXd p = cov.template cast<double>();
    const Eigen::VectorXd r = (x - mean).template cast<double>();
    const double wsse = r.dot(p.inverse() * r);
    return -0.5 * (double(n) * std::log(2.0 * std::numbers::pi) + std::log(p.determinant()) + wsse);
  }

  MG make() const { return MG(makeVector(0.0), makeSpd()); }
};

}  // namespace

TYPED_TEST_SUITE(GaussianTest, Configs);

TYPED_TEST(GaussianTest, LogPdfMatchesReference) {
  const auto mg = this->make();
  const auto g = mg.toGaussian();
  ASSERT_TRUE(g.has_value());

  const auto x = this->makeVector(0.7);
  const double expected = this->referenceLogPdf(mg.mean(), mg.cov(), x);
  EXPECT_NEAR(g->logPdf(x), expected, this->tolerance() * std::abs(expected));
  EXPECT_NEAR(g->pdf(x), std::exp(g->logPdf(x)), this->tolerance());
  EXPECT_NEAR(g->logDeterminant(), std::log(mg.cov().template cast<double>().determinant()), this->tolerance() * 10);
}

TYPED_TEST(GaussianTest, PeakIsAtMean) {
  const auto mg = this->make();
  const auto g = mg.toGaussian();
  ASSERT_TRUE(g.has_value());
  EXPECT_EQ(g->wsse(mg.mean()), 0);
  EXPECT_EQ(g->logPdf(mg.mean()), g->logPeak());
}

TYPED_TEST(GaussianTest, ReadsOnlyLowerTriangle) {
  auto mg = this->make();
  const auto x = this->makeVector(0.7);
  const auto expected = mg.toGaussian()->logPdf(x);

  mg.mutableCov().template triangularView<Eigen::StrictlyUpper>().setConstant(12345);
  EXPECT_EQ(mg.toGaussian()->logPdf(x), expected);
}

TYPED_TEST(GaussianTest, MeanChangeKeepsFactorization) {
  auto mg = this->make();
  const auto logPeak = mg.toGaussian()->logPeak();

  mg.setMean(this->makeVector(2.0));
  const auto g = mg.toGaussian();
  EXPECT_EQ(g->logPeak(), logPeak);
  EXPECT_EQ(g->logPdf(this->makeVector(2.0)), logPeak);
}

TYPED_TEST(GaussianTest, CovChangeRefactorizes) {
  using Scalar = typename TestFixture::Scalar;
  auto mg = this->make();
  const auto logDet = mg.toGaussian()->logDeterminant();

  mg.mutableCov() *= Scalar(4);
  const auto expected = logDet + Scalar(this->n) * std::log(Scalar(4));
  EXPECT_NEAR(mg.toGaussian()->logDeterminant(), expected, this->tolerance() * 10);
}

TYPED_TEST(GaussianTest, NotPositiveDefinite) {
  using MatrixType = typename TestFixture::MatrixType;
  auto mg = this->make();
  mg.setCov(-MatrixType::Identity(this->n, this->n));
  const auto g = mg.toGaussian();
  ASSERT_FALSE(g.has_value());
  EXPECT_EQ(g.error(), GaussianError::NotPositiveDefinite);
}

TYPED_TEST(GaussianTest, NonFinite) {
  using Scalar = typename TestFixture::Scalar;
  auto mg = this->make();
  mg.mutableCov()(this->n - 1, 0) = std::numeric_limits<Scalar>::quiet_NaN();
  const auto g = mg.toGaussian();
  ASSERT_FALSE(g.has_value());
  EXPECT_EQ(g.error(), GaussianError::NonFinite);
}

TYPED_TEST(GaussianTest, RecoversAfterFailure) {
  using MatrixType = typename TestFixture::MatrixType;
  auto mg = this->make();
  mg.setCov(-MatrixType::Identity(this->n, this->n));
  ASSERT_FALSE(mg.toGaussian().has_value());
  mg.setCov(this->makeSpd());
  EXPECT_TRUE(mg.toGaussian().has_value());
}

TYPED_TEST(GaussianTest, LockReleasedWhenViewsDie) {
  auto mg = this->make();
  {
    auto g = mg.toGaussian();
    auto copy = *g;
    auto moved = std::move(*g);
  }
  mg.mutableCov();  // would assert if any lock leaked
  mg.resize(this->n);
}

TYPED_TEST(GaussianTest, CopyOfLockedSourceIsUnlocked) {
  auto mg = this->make();
  const auto g = mg.toGaussian();
  auto copy = mg;
  copy.mutableCov();  // would assert if the lock were copied
  EXPECT_EQ(copy.toGaussian()->logPeak(), g->logPeak());
}

TEST(Gaussian, ResizeDynamic) {
  MutableGaussian<double, Eigen::Dynamic> mg(2);
  mg.setMean(Eigen::Vector2d::Zero());
  mg.setCov(Eigen::Matrix2d::Identity());
  ASSERT_TRUE(mg.toGaussian().has_value());

  mg.resize(3);
  EXPECT_EQ(mg.size(), 3);
  mg.setMean(Eigen::Vector3d::Zero());
  mg.setCov(Eigen::Matrix3d::Identity() * 2.0);
  const auto g = mg.toGaussian();
  ASSERT_TRUE(g.has_value());
  EXPECT_EQ(g->size(), 3);
  EXPECT_NEAR(g->logDeterminant(), 3.0 * std::log(2.0), 1e-12);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

class GaussianDeathTest : public ::testing::Test {
 protected:
  using MG = MutableGaussian<double, 2>;
  static MG make() { return MG(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()); }
};

TEST_F(GaussianDeathTest, MutateWhileLocked) {
  auto mg = make();
  const auto g = mg.toGaussian();
  EXPECT_DEATH(mg.mutableCov(), "locked");
  EXPECT_DEATH(mg.mutableMean(), "locked");
  EXPECT_DEATH(mg.resize(2), "locked");
}

TEST_F(GaussianDeathTest, CopyOfViewKeepsLock) {
  auto mg = make();
  std::optional<MG::GaussianType> copy;
  {
    const auto g = mg.toGaussian();
    copy.emplace(*g);
  }
  EXPECT_DEATH(mg.mutableCov(), "locked");
}

TEST_F(GaussianDeathTest, MoveWhileLocked) {
  auto mg = make();
  const auto g = mg.toGaussian();
  EXPECT_DEATH({ auto stolen = std::move(mg); }, "locked");
}

TEST_F(GaussianDeathTest, DestroyWhileLocked) {
  EXPECT_DEATH(
      {
        auto mg = std::make_unique<MG>(make());
        auto g = mg->toGaussian();
        mg.reset();
      },
      "locked");
}

#endif
