#include <EigenFilter/Gaussian.h>

#include <gtest/gtest.h>

#include <Eigen/LU>

#include <array>
#include <memory>
#include <optional>
#include <vector>

using EigenFilter::MutableGaussianN;

namespace {

using MG4 = MutableGaussianN<double, 4>;
using MGX = MutableGaussianN<double, Eigen::Dynamic>;

Eigen::Matrix4d makeSpd() {
  const Eigen::Matrix4d a{Eigen::Matrix4d::NullaryExpr(
      [](Eigen::Index i, Eigen::Index j) { return std::sin(7.0 * double(i) + 3.0 * double(j) + 1.0); })};
  return a * a.transpose() + Eigen::Matrix4d::Identity();
}

const Eigen::Vector4d mean4{1.0, -2.0, 3.0, 0.5};

// The covariance with its strict upper triangle overwritten, which only the lower triangle may be read past.
Eigen::Matrix4d withGarbageUpper(Eigen::Matrix4d cov) {
  cov.triangularView<Eigen::StrictlyUpper>().setConstant(12345.0);
  return cov;
}

}  // namespace

TEST(GaussianMarginal, SelectsEntriesInAnyOrder) {
  const MG4 mg{mean4, withGarbageUpper(makeSpd())};
  const auto marginal{mg.marginal(std::array{3, 1})};
  static_assert(decltype(marginal)::SizeAtCompileTime == 2);
  EXPECT_EQ(marginal.mean(), Eigen::Vector2d(0.5, -2.0));

  const Eigen::Matrix4d p{makeSpd()};
  const Eigen::Matrix2d expected{{p(3, 3), p(3, 1)}, {p(1, 3), p(1, 1)}};
  EXPECT_EQ(Eigen::Matrix2d{marginal.cov()}, expected);
}

TEST(GaussianMarginal, IndexTypes) {
  const MGX mg{mean4, makeSpd()};
  const auto fixed{mg.marginal(Eigen::seqN(Eigen::fix<1>, Eigen::fix<2>))};
  static_assert(decltype(fixed)::SizeAtCompileTime == 2);
  EXPECT_EQ(fixed.mean(), mean4.segment<2>(1));

  const auto dynamic{mg.marginal(std::vector<int>{0, 2, 3})};
  static_assert(decltype(dynamic)::SizeAtCompileTime == Eigen::Dynamic);
  EXPECT_EQ(dynamic.size(), 3);

  const auto symbolic{mg.marginal(Eigen::indexing::lastN(2))};
  EXPECT_EQ(symbolic.mean(), mean4.tail<2>());
}

TEST(GaussianMarginal, CopyIsIndependentAndViewIsLive) {
  MG4 mg{mean4, makeSpd()};
  const auto marginal{mg.marginal(std::array{0, 2})};
  auto copy{marginal.copy()};
  static_assert(std::is_same_v<decltype(copy), MutableGaussianN<double, 2>>);

  mg.setMean(Eigen::Vector4d::Zero());  // values may change while borrowed
  EXPECT_EQ(marginal.mean(), Eigen::Vector2d::Zero());
  EXPECT_EQ(copy.mean(), Eigen::Vector2d(1.0, 3.0));
  EXPECT_TRUE(copy.toGaussian().has_value());
}

TEST(GaussianMarginal, BorrowReleasedWhenMarginalsDie) {
  MGX mg{mean4, makeSpd()};
  {
    const auto marginal{mg.marginal(std::array{0, 1})};
    auto copy{marginal};
    auto moved{std::move(copy)};
  }
  mg.resize(3);  // would assert if any borrow leaked
}

TEST(GaussianMarginal, FromView) {
  const MG4 mg{mean4, makeSpd()};
  const auto marginal{mg.toGaussian()->marginal(std::array{2})};
  EXPECT_EQ(marginal.mean()(0), 3.0);
  EXPECT_EQ(marginal.copy().cov()(0, 0), makeSpd()(2, 2));
}

namespace {

// Reference conditional through the explicit formula, with indices a (target) and b (given).
std::pair<Eigen::VectorXd, Eigen::MatrixXd> referenceConditional(const std::vector<int>& a, const std::vector<int>& b,
                                                                 const Eigen::VectorXd& v) {
  const Eigen::Matrix4d p{makeSpd()};
  const Eigen::MatrixXd pbb{p(b, b)};
  const Eigen::MatrixXd gain{p(a, b) * pbb.inverse()};
  return {mean4(a) + gain * (v - mean4(b)), p(a, a) - gain * p(b, a)};
}

}  // namespace

TEST(GaussianConditional, MatchesFormula) {
  const MG4 mg{mean4, withGarbageUpper(makeSpd())};
  const Eigen::Vector2d v{0.3, -0.7};
  const auto conditional{mg.conditional(std::array{2, 0}, std::array{3, 1}, v)};
  static_assert(std::is_same_v<decltype(conditional), const MutableGaussianN<double, 2>>);

  const auto [mean, cov]{referenceConditional({2, 0}, {3, 1}, v)};
  EXPECT_LT((conditional.mean() - mean).norm(), 1e-12);
  EXPECT_LT((conditional.cov() - cov).norm(), 1e-12);
}

TEST(GaussianConditional, ComplementTarget) {
  const MG4 mg{mean4, makeSpd()};
  const Eigen::Matrix<double, 1, 1> v{0.25};
  const auto conditional{mg.conditional(std::array{1}, v)};
  static_assert(std::is_same_v<decltype(conditional), const MutableGaussianN<double, 3>>);
  const auto explicitTarget{mg.conditional(std::array{0, 2, 3}, std::array{1}, v)};
  EXPECT_EQ(conditional.mean(), explicitTarget.mean());
  EXPECT_EQ(conditional.cov(), explicitTarget.cov());

  const MGX dynamic{mean4, makeSpd()};
  const auto fromDynamic{dynamic.conditional(std::vector<int>{1}, Eigen::VectorXd{v})};
  EXPECT_EQ(fromDynamic.size(), 3);
  EXPECT_LT((fromDynamic.mean() - explicitTarget.mean()).norm(), 1e-12);
}

TEST(GaussianConditional, ViewMatchesMutable) {
  const MG4 mg{mean4, makeSpd()};
  const Eigen::Vector2d v{1.0, 2.0};
  const auto fromMutable{mg.conditional(std::array{0, 1}, std::array{2, 3}, v)};
  const auto fromView{mg.toGaussian()->conditional(std::array{0, 1}, std::array{2, 3}, v)};
  EXPECT_EQ(fromMutable.mean(), fromView.mean());
  EXPECT_EQ(fromMutable.cov(), fromView.cov());
  EXPECT_TRUE(fromView.toGaussian().has_value());
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

TEST(GaussianMarginalDeathTest, StructuralChangeWhileBorrowed) {
  auto mg{std::make_unique<MGX>(mean4, makeSpd())};
  const auto marginal{mg->marginal(std::array{0, 1})};
  EXPECT_DEATH(mg->resize(3), "borrowed");
  EXPECT_DEATH({ auto stolen{std::move(*mg)}; }, "borrowed");
  EXPECT_DEATH(*mg = MGX{3}, "borrowed");
  EXPECT_DEATH(mg.reset(), "borrowed");
}

TEST(GaussianMarginalDeathTest, MarginalOfViewKeepsLock) {
  MG4 mg{mean4, makeSpd()};
  std::optional<EigenFilter::GaussianMarginal<MG4::GaussianType, std::array<int, 1>>> marginal;
  marginal.emplace(mg.toGaussian()->marginal(std::array{0}));
  EXPECT_DEATH(mg.setMean(Eigen::Vector4d::Zero()), "locked");
}

TEST(GaussianMarginalDeathTest, RepeatedIndices) {
  const MG4 mg{mean4, makeSpd()};
  EXPECT_DEATH(std::ignore = mg.marginal(std::array{1, 1}), "distinct");
  EXPECT_DEATH(std::ignore = mg.conditional(std::array{0, 1}, std::array{1}, Eigen::Matrix<double, 1, 1>{0.0}),
               "distinct");
}

#endif
