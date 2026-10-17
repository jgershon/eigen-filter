#include <EigenFilter/Gaussian.h>
#include <EigenFilter/Sampling.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <type_traits>

using EigenFilter::Bounded;
using EigenFilter::Composite;
using EigenFilter::CoordinateKind;
using EigenFilter::idx;
using EigenFilter::Linear;
using EigenFilter::MutableGaussian;
using EigenFilter::MutableGaussianN;
using EigenFilter::Periodic;
using EigenFilter::Repeated;

namespace {

constexpr double pi{std::numbers::pi};

struct Position : Linear<2> {};
struct Heading : Periodic<> {
  using Periodic::Periodic;
};
struct Range : Bounded<> {
  using Bounded::Bounded;
};
using State = Composite<Position, Heading, Range>;
using StateGaussian = MutableGaussian<double, State>;

static_assert(std::is_same_v<MutableGaussianN<double, 4>::SpaceType, Linear<4>>);
static_assert(StateGaussian::SizeAtCompileTime == 4 && StateGaussian::MaxSizeAtCompileTime == 4);
static_assert(std::is_same_v<StateGaussian::VectorType, Eigen::Vector4d>);
static_assert(std::is_nothrow_move_constructible_v<StateGaussian>);

Eigen::Matrix4d makeCov() {
  return Eigen::Matrix4d{{1.0, 0.2, 0.1, 0.0}, {0.2, 2.0, -0.1, 0.1}, {0.1, -0.1, 0.04, 0.0}, {0.0, 0.1, 0.0, 0.5}};
}

}  // namespace

TEST(GaussianSpace, MeanIsNormalized) {
  StateGaussian g{State{}, Eigen::Vector4d{1.0, 2.0, 3.5, -2.0}, makeCov()};
  EXPECT_NEAR(g.mean()(2), 3.5 - 2 * pi, 1e-15);
  EXPECT_EQ(g.mean()(3), 2.0);

  g.setMean(Eigen::Vector4d{0.0, 0.0, -4.0, -1.0});
  EXPECT_NEAR(g.mean()(2), -4.0 + 2 * pi, 1e-15);
  EXPECT_EQ(g.mean()(3), 1.0);
}

TEST(GaussianSpace, ResidualsTakeTheShortWayAround) {
  const StateGaussian g{State{}, Eigen::Vector4d{0.0, 0.0, pi - 0.1, 3.0}, makeCov()};
  const MutableGaussianN<double, 4> linear{Eigen::Vector4d{0.0, 0.0, pi - 0.1, 3.0}, makeCov()};
  // -π + 0.1 is 0.2 past the mean across the wrap, not 2π - 0.2 behind it.
  const Eigen::Vector4d across{0.3, -0.2, -pi + 0.1, 3.4};
  const Eigen::Vector4d unwrapped{0.3, -0.2, pi + 0.1, 3.4};
  EXPECT_NEAR(g.toGaussian()->logPdf(across), linear.toGaussian()->logPdf(unwrapped), 1e-10);
}

TEST(GaussianSpace, SamplesStayInRange) {
  const StateGaussian g{State{}, Eigen::Vector4d{0.0, 0.0, pi - 0.05, 0.1}, makeCov()};
  const auto view{g.toGaussian()};
  std::mt19937_64 rng{3};
  const auto samples{EigenFilter::drawSamples(*view, 2000, rng)};
  EXPECT_TRUE((samples.row(2).array() >= -pi).all() && (samples.row(2).array() < pi).all());
  EXPECT_TRUE((samples.row(3).array() >= 0.0).all());
  EXPECT_GT((samples.row(2).array() < 0.0).count(), 100);  // some wrapped past π

  std::mt19937_64 again{3};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(*view, 50, again)};
  for (Eigen::Index j{0}; j < 50; ++j) {
    EXPECT_NEAR(drawn.logPdf(j), view->logPdf(drawn.samples.col(j)), 1e-12) << "sample " << j;
  }
}

TEST(GaussianSpace, TagMarginalKeepsTheSubspace) {
  const State state{Position{}, Heading{360.0, 0.0}, Range{}};
  const StateGaussian g{state, Eigen::Vector4d{1.0, 2.0, 359.0, 3.0}, makeCov()};
  const auto heading{g.marginal<Heading>()};
  static_assert(std::is_same_v<decltype(heading)::SpaceType, Heading>);
  EXPECT_EQ(heading.space().period(), 360.0);
  EXPECT_EQ(heading.mean()(0), 359.0);

  auto copy{heading.copy()};
  static_assert(std::is_same_v<decltype(copy), MutableGaussian<double, Heading>>);
  copy.setMean(Eigen::Matrix<double, 1, 1>{361.0});
  EXPECT_EQ(copy.mean()(0), 1.0);

  const auto position{g.marginal<Position>().copy()};
  static_assert(std::is_same_v<decltype(position), const MutableGaussian<double, Position>>);
  EXPECT_EQ(position.mean(), Eigen::Vector2d(1.0, 2.0));
}

TEST(GaussianSpace, TagMarginalThroughACappedRepeatedBlockKeepsTheCap) {
  struct Altitude : Linear<1> {};
  struct Vehicle : Composite<Linear<2>, Altitude> {};
  struct Fleet : Repeated<Vehicle, 3> {};
  using World = Composite<Linear<1>, Fleet>;
  const MutableGaussian<double, World> g{World{}, Eigen::VectorXd::LinSpaced(7, 0.0, 6.0),
                                         Eigen::MatrixXd::Identity(7, 7)};
  const auto altitudes{g.marginal<Fleet, Altitude>().copy()};
  static_assert(std::is_same_v<decltype(altitudes)::SpaceType, Repeated<Altitude, 3>>);
  static_assert(decltype(altitudes)::MaxSizeAtCompileTime == 3);
  EXPECT_EQ(altitudes.mean(), Eigen::Vector2d(3.0, 6.0));
}

TEST(GaussianSpace, IndexMarginalProjectsTheSpace) {
  const StateGaussian g{State{}, Eigen::Vector4d{1.0, 2.0, 0.5, 3.0}, makeCov()};
  const auto marginal{g.marginal(std::array{3, 2, 0})};
  EXPECT_EQ(marginal.space().coordinate(0).kind, CoordinateKind::Bounded);
  EXPECT_EQ(marginal.space().coordinate(1).kind, CoordinateKind::Periodic);
  EXPECT_EQ(marginal.space().coordinate(2).kind, CoordinateKind::Linear);

  auto copy{marginal.copy()};
  copy.setMean(Eigen::Vector3d{-1.0, 4.0, 7.0});
  EXPECT_EQ(copy.mean()(0), 1.0);
  EXPECT_NEAR(copy.mean()(1), 4.0 - 2 * pi, 1e-15);
  EXPECT_EQ(copy.mean()(2), 7.0);

  const auto linear{MutableGaussianN<double, 4>{Eigen::Vector4d::Zero(), makeCov()}.marginal(std::array{0, 1}).copy()};
  static_assert(std::is_same_v<decltype(linear), const MutableGaussianN<double, 2>>);
}

TEST(GaussianSpace, ConditionalWrapsTheGivenResidual) {
  const Eigen::Vector4d mean{1.0, 2.0, -pi + 0.1, 3.0};
  const StateGaussian g{State{}, mean, makeCov()};
  const MutableGaussianN<double, 4> linear{mean, makeCov()};
  // Given heading π - 0.1 is 0.2 behind the mean across the wrap.
  const Eigen::Matrix<double, 1, 1> given{pi - 0.1};
  const Eigen::Matrix<double, 1, 1> unwrapped{-pi - 0.1};
  const auto conditional{g.conditional<Heading>(given)};
  const auto expected{linear.conditional(std::array{2}, unwrapped)};
  EXPECT_LT((conditional.mean() - expected.mean()).norm(), 1e-12);
  EXPECT_LT((conditional.cov() - expected.cov()).norm(), 1e-12);
  EXPECT_EQ(conditional.space().coordinate(2).kind, CoordinateKind::Bounded);
}

TEST(GaussianSpace, ConditionalNormalizesTheTargetMean) {
  // Strong position-heading correlation pushes the conditional heading past π.
  const Eigen::Matrix2d cov{{1.0, 0.9}, {0.9, 1.0}};
  const MutableGaussian<double, Composite<Linear<1>, Periodic<>>> g{{}, Eigen::Vector2d{0.0, 3.0}, cov};
  const auto conditional{g.conditional(std::array{1}, std::array{0}, Eigen::Matrix<double, 1, 1>{1.0})};
  EXPECT_NEAR(conditional.mean()(0), 3.0 + 0.9 - 2 * pi, 1e-12);
}

TEST(GaussianSpace, ResizableSpace) {
  struct Vehicle : Composite<Linear<2>, Periodic<>> {};
  using World = Composite<Linear<1>, Repeated<Vehicle>>;
  MutableGaussian<double, World> g{World{}, Eigen::Index{7}};
  EXPECT_EQ(g.size(), 7);
  g.resize(10);
  EXPECT_EQ(g.size(), 10);
  g.setMean(Eigen::VectorXd::Constant(10, 4.0));
  EXPECT_NEAR(g.mean()(3), 4.0 - 2 * pi, 1e-15);
  EXPECT_NEAR(g.mean()(9), 4.0 - 2 * pi, 1e-15);
  EXPECT_EQ(g.mean()(8), 4.0);
}

TEST(GaussianSpace, CappedSpaceGivesFixedCapacityStorage) {
  struct Vehicle : Composite<Linear<2>, Periodic<>> {};
  using Fleet = Composite<Linear<1>, Repeated<Vehicle, 3>>;
  using G = MutableGaussian<double, Fleet>;
  static_assert(G::SizeAtCompileTime == Eigen::Dynamic && G::MaxSizeAtCompileTime == 10);
  static_assert(G::VectorType::MaxRowsAtCompileTime == 10);
  G g{Fleet{}, Eigen::Index{10}};
  EXPECT_EQ(g.size(), 10);
  static_assert(std::is_same_v<MutableGaussianN<double, Eigen::Dynamic, 6>::SpaceType, Linear<Eigen::Dynamic, 6>>);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

TEST(GaussianSpaceDeathTest, SizeMustFitTheSpace) {
  struct Vehicle : Composite<Linear<2>, Periodic<>> {};
  using World = Composite<Linear<1>, Repeated<Vehicle>>;
  MutableGaussian<double, World> g{World{}, Eigen::Index{4}};
  EXPECT_DEATH(g.resize(5), "fit the space");
  EXPECT_DEATH((MutableGaussian<double, World>{World{}, Eigen::Index{3}}), "fit the space");

  using Capped = Composite<Linear<1>, Repeated<Vehicle, 2>>;
  MutableGaussian<double, Capped> capped{Capped{}, Eigen::Index{7}};
  EXPECT_DEATH(capped.resize(10), "fit the space");
}

#endif
