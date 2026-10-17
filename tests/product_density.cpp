#include <EigenFilter/Gamma.h>
#include <EigenFilter/Gaussian.h>
#include <EigenFilter/KernelDensity.h>
#include <EigenFilter/ParticleSet.h>
#include <EigenFilter/ProductDensity.h>
#include <EigenFilter/StudentT.h>
#include <EigenFilter/Transform.h>
#include <EigenFilter/Uniform.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <expected>
#include <numbers>
#include <random>
#include <type_traits>

using EigenFilter::Bounded;
using EigenFilter::Composite;
using EigenFilter::GammaDensity;
using EigenFilter::Gaussian;
using EigenFilter::Linear;
using EigenFilter::MomentsError;
using EigenFilter::on;
using EigenFilter::Periodic;
using EigenFilter::ProductDensity;
using EigenFilter::UniformDensity;

namespace {

constexpr double pi{std::numbers::pi};

struct Position : Linear<2> {};
struct Heading : Periodic<> {
  using Periodic::Periodic;
};
struct Speed : Bounded<> {
  using Bounded::Bounded;
};
using State = Composite<Position, Heading, Speed>;

Gaussian<double, Position> makePosition() {
  return Gaussian<double, Position>{Eigen::Vector2d{10.0, -5.0}, Eigen::Matrix2d{{4.0, 1.0}, {1.0, 2.0}}};
}

}  // namespace

TEST(ProductDensity, LogPdfAddsTheFactors) {
  const auto position{makePosition()};
  const auto positionDensity{position.toDensity()};
  ASSERT_TRUE(positionDensity.has_value());
  const UniformDensity<double, Heading> heading{};
  const GammaDensity<double, Speed> speed{4.0, 0.5};
  const ProductDensity prior{State{}, on<Position>(*positionDensity), on<Heading>(heading), on<Speed>(speed)};
  static_assert(std::is_same_v<std::remove_const_t<decltype(prior)>,
                               ProductDensity<double, State, EigenFilter::GaussianDensity<double, Position>,
                                              UniformDensity<double, Heading>, GammaDensity<double, Speed>>>);
  static_assert(!EigenFilter::UncertainInput<decltype(prior)>);
  EXPECT_EQ(prior.size(), 4);

  const Eigen::Vector4d x{11.0, -4.0, 2.0, 1.5};
  const double expected{positionDensity->logPdf(x.head<2>()) - std::log(2 * pi) +
                        speed.logPdf(Eigen::Matrix<double, 1, 1>{1.5})};
  EXPECT_NEAR(prior.logPdf(x), expected, 1e-13);
  EXPECT_EQ(prior.logPdf(Eigen::Vector4d{11.0, -4.0, 2.0, -1.0}), -std::numeric_limits<double>::infinity());
}

TEST(ProductDensity, SamplesFillEachFactorsRows) {
  const auto position{makePosition()};
  const auto positionDensity{position.toDensity()};
  ASSERT_TRUE(positionDensity.has_value());
  const ProductDensity prior{State{}, on<Position>(*positionDensity), on<Heading>(UniformDensity<double, Heading>{}),
                             on<Speed>(GammaDensity<double, Speed>{4.0, 0.5})};

  std::mt19937_64 rng{17};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(prior, 50000, rng)};
  ASSERT_EQ(drawn.samples.rows(), 4);
  for (Eigen::Index j{0}; j < 20; ++j) EXPECT_NEAR(drawn.logPdf(j), prior.logPdf(drawn.samples.col(j)), 1e-10);
  EXPECT_TRUE((drawn.samples.row(2).array() >= -pi).all() && (drawn.samples.row(2).array() < pi).all());
  EXPECT_TRUE((drawn.samples.row(3).array() >= 0).all());

  const Eigen::Vector4d mean{drawn.samples.rowwise().mean()};
  EXPECT_NEAR(mean(0), 10.0, 0.05);
  EXPECT_NEAR(mean(1), -5.0, 0.05);
  EXPECT_NEAR(mean(2), 0.0, 0.05);
  EXPECT_NEAR(mean(3), 2.0, 0.02);
  const Eigen::Matrix<double, 4, Eigen::Dynamic> centered{drawn.samples.colwise() - mean};
  const Eigen::Matrix4d cov{centered * centered.transpose() / double(centered.cols())};
  EXPECT_NEAR(cov(0, 1), 1.0, 0.1);
  EXPECT_NEAR(cov(0, 3), 0.0, 0.05);  // independent factors
}

TEST(ProductDensity, MomentsAreBlockDiagonal) {
  const auto position{makePosition()};
  const auto positionDensity{position.toDensity()};
  ASSERT_TRUE(positionDensity.has_value());
  const ProductDensity prior{State{}, on<Position>(*positionDensity), on<Heading>(UniformDensity<double, Heading>{}),
                             on<Speed>(GammaDensity<double, Speed>{4.0, 0.5})};
  const auto moments{prior.moments()};
  static_assert(std::is_same_v<std::remove_const_t<decltype(moments)>, EigenFilter::Moments<double, State>>);
  EXPECT_TRUE(moments.mean().isApprox(Eigen::Vector4d{10.0, -5.0, 0.0, 2.0}));
  Eigen::Matrix4d expected{Eigen::Matrix4d::Zero()};
  expected.topLeftCorner<2, 2>() = Eigen::Matrix2d{{4.0, 1.0}, {1.0, 2.0}};
  expected(2, 2) = 4 * pi * pi / 12;
  expected(3, 3) = 1.0;
  EXPECT_TRUE(moments.cov().isApprox(expected));
}

TEST(ProductDensity, MomentsCanFailWithAStudentTFactor) {
  using Plane = Composite<Position, Heading>;
  const auto cauchy{EigenFilter::StudentTDensity<double, Position>::create(Eigen::Vector2d::Zero(),
                                                                           Eigen::Matrix2d::Identity(), 1.0)};
  const auto t{EigenFilter::StudentTDensity<double, Position>::create(Eigen::Vector2d::Zero(),
                                                                      Eigen::Matrix2d::Identity(), 4.0)};
  ASSERT_TRUE(cauchy && t);
  const ProductDensity heavy{Plane{}, on<Position>(*cauchy), on<Heading>(UniformDensity<double, Heading>{})};
  const ProductDensity finite{Plane{}, on<Position>(*t), on<Heading>(UniformDensity<double, Heading>{})};
  static_assert(
      std::is_same_v<decltype(heavy.moments()), std::expected<EigenFilter::Moments<double, Plane>, MomentsError>>);
  EXPECT_EQ(heavy.moments().error(), MomentsError::Infinite);
  ASSERT_TRUE(finite.moments().has_value());
  EXPECT_NEAR(finite.moments()->cov()(0, 0), 2.0, 1e-14);
}

TEST(ProductDensity, ExplicitIndicesNeedNotBeContiguous) {
  // A uniform on coordinates 2 and 0, in that order, and an exponential on coordinate 1.
  const EigenFilter::UniformDensityN<double, 2> box{Eigen::Vector2d{0.0, -1.0}, Eigen::Vector2d{4.0, 1.0}};
  const auto exponential{EigenFilter::GammaDensityN<double, 1>::exponential(2.0)};
  const ProductDensity product{Linear<3>{}, on(std::array<Eigen::Index, 2>{2, 0}, box),
                               on(Eigen::Index{1}, exponential)};

  const Eigen::Vector3d x{0.5, 3.0, 2.0};  // coordinate 2 is the box's first, coordinate 0 its second
  EXPECT_NEAR(product.logPdf(x), -std::log(8.0) - std::log(2.0) - 1.5, 1e-14);
  EXPECT_EQ(product.logPdf(Eigen::Vector3d{2.0, 3.0, 0.5}), -std::numeric_limits<double>::infinity());

  std::mt19937_64 rng{23};
  const auto samples{EigenFilter::drawSamples(product, 1000, rng)};
  EXPECT_TRUE((samples.row(2).array() >= 0.0).all() && (samples.row(2).array() < 4.0).all());
  EXPECT_TRUE((samples.row(0).array() >= -1.0).all() && (samples.row(0).array() < 1.0).all());
  EXPECT_TRUE((samples.row(1).array() >= 0.0).all());

  const auto moments{product.moments()};
  EXPECT_TRUE(moments.mean().isApprox(Eigen::Vector3d{0.0, 2.0, 2.0}));
  EXPECT_NEAR(moments.cov()(2, 2), 16.0 / 12, 1e-14);
  EXPECT_NEAR(moments.cov()(0, 0), 4.0 / 12, 1e-14);
  EXPECT_NEAR(moments.cov()(1, 1), 4.0, 1e-14);
}

TEST(ProductDensity, TakesAKernelDensityFactor) {
  using Plane = Composite<Position, Heading>;
  EigenFilter::ParticleSet<double, Position> particles{Position{},
                                                       Eigen::Matrix<double, 2, 3>{{0.0, 1.0, 2.0}, {0.0, 1.0, 0.0}}};
  particles.setBandwidth(Eigen::Matrix2d::Identity() * 0.1);
  const auto kde{particles.toDensity()};
  ASSERT_TRUE(kde.has_value());
  const ProductDensity product{Plane{}, on<Position>(*kde), on<Heading>(UniformDensity<double, Heading>{})};
  const Eigen::Vector3d x{1.0, 0.5, 1.0};
  EXPECT_NEAR(product.logPdf(x), kde->logPdf(x.head<2>()) - std::log(2 * pi), 1e-13);
  std::mt19937_64 rng{29};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(product, 50, rng)};
  for (Eigen::Index j{0}; j < 10; ++j) EXPECT_NEAR(drawn.logPdf(j), product.logPdf(drawn.samples.col(j)), 1e-10);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
TEST(ProductDensityDeathTest, FactorsMustPartitionTheSpaceWithMatchingCoordinates) {
  const UniformDensity<double, Heading> heading{};
  const EigenFilter::UniformDensityN<double, 1> line{Eigen::Matrix<double, 1, 1>{0.0},
                                                     Eigen::Matrix<double, 1, 1>{1.0}};
  using Two = Composite<Heading, Linear<1>>;
  EXPECT_DEATH((ProductDensity{Two{}, on(Eigen::Index{0}, heading), on(Eigen::Index{0}, line)}), "partition");
  EXPECT_DEATH((ProductDensity{Two{}, on(Eigen::Index{1}, heading), on(Eigen::Index{0}, line)}), "match");
}
#endif

namespace {

struct East : Linear<1> {};
struct North : Linear<1> {};
struct Place : Composite<East, North> {
  using Composite::Composite;
};
struct Pose : Composite<Place, Heading> {
  using Composite::Composite;
};
using Vehicle = Composite<Pose, Speed>;

Gaussian<double, Place> makePlace() {
  return Gaussian<double, Place>{Eigen::Vector2d{10.0, -5.0}, Eigen::Matrix2d{{4.0, 1.0}, {1.0, 2.0}}};
}

double speedLogPdf(double speed) {
  return GammaDensity<double, Speed>{4.0, 0.5}.logPdf(Eigen::Matrix<double, 1, 1>{speed});
}

// A Gaussian place, a uniform heading, and a Gamma speed, with the Gaussian held by value.
auto makeVehiclePrior() {
  return ProductDensity{Vehicle{}, on<Place>(makePlace()), on<Heading>(UniformDensity<double, Heading>{}),
                        on<Speed>(GammaDensity<double, Speed>{4.0, 0.5})};
}

}  // namespace

TEST(ProductDensity, HoldsAnOwnerByValue) {
  const auto prior{makeVehiclePrior()};
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(prior.factor<0>())>, Gaussian<double, Place>>);
  const auto placeGaussian{makePlace()};
  const auto place{placeGaussian.toDensity()};
  ASSERT_TRUE(place.has_value());
  const Eigen::Vector4d x{11.0, -4.0, 2.0, 1.5};
  EXPECT_NEAR(prior.logPdf(x), place->logPdf(x.head<2>()) - std::log(2 * pi) + speedLogPdf(1.5), 1e-13);
  std::mt19937_64 rng{31};
  const auto drawn{EigenFilter::drawSamplesWithLogPdf(prior, 20, rng)};
  for (Eigen::Index j{0}; j < 20; ++j) EXPECT_NEAR(drawn.logPdf(j), prior.logPdf(drawn.samples.col(j)), 1e-10);
}

TEST(ProductDensity, MarginalOfAFactorIsThatFactor) {
  const auto prior{makeVehiclePrior()};
  const auto place{prior.marginal<Place>()};
  static_assert(std::is_same_v<decltype(place), const Gaussian<double, Place>>);
  EXPECT_TRUE(place.mean().isApprox(Eigen::Vector2d{10.0, -5.0}));
  const auto qualified{prior.marginal<Pose, Heading>()};
  static_assert(std::is_same_v<decltype(qualified), const UniformDensity<double, Heading>>);
}

TEST(ProductDensity, MarginalInsideAFactorIsTheFactorsMarginal) {
  const auto prior{makeVehiclePrior()};
  const auto north{prior.marginal<North>()};
  static_assert(std::is_same_v<decltype(north), const Gaussian<double, North>>);  // an owner, not a view
  EXPECT_NEAR(north.mean()(0), -5.0, 1e-15);
  EXPECT_NEAR(north.cov()(0, 0), 2.0, 1e-15);
  const auto east{prior.marginal<Pose, Place, East>()};
  EXPECT_NEAR(east.mean()(0), 10.0, 1e-15);
}

TEST(ProductDensity, MarginalOfWholeFactorsIsTheirProduct) {
  const auto prior{makeVehiclePrior()};
  const auto pose{prior.marginal<Pose>()};
  static_assert(
      std::is_same_v<decltype(pose),
                     const ProductDensity<double, Pose, Gaussian<double, Place>, UniformDensity<double, Heading>>>);
  const Eigen::Vector3d x{11.0, -4.0, 2.0};
  EXPECT_NEAR(pose.logPdf(x), prior.marginal<Place>().toDensity()->logPdf(x.head<2>()) - std::log(2 * pi), 1e-13);
}

TEST(ProductDensity, ConditionalOnAWholeFactorDropsIt) {
  const auto prior{makeVehiclePrior()};
  const auto rest{prior.conditional<Speed>(Eigen::Matrix<double, 1, 1>{3.0})};
  static_assert(
      std::is_same_v<decltype(rest),
                     const ProductDensity<double, Pose, Gaussian<double, Place>, UniformDensity<double, Heading>>>);
  const auto heading{prior.conditional<Pose, Place>(Eigen::Vector2d{1.0, 2.0})};
  static_assert(std::is_same_v<std::remove_const_t<decltype(heading)>,
                               ProductDensity<double, EigenFilter::CoordinateSpace<2>, UniformDensity<double, Heading>,
                                              GammaDensity<double, Speed>>>);
  EXPECT_NEAR(heading.logPdf(Eigen::Vector2d{0.3, 1.5}), -std::log(2 * pi) + speedLogPdf(1.5), 1e-13);
  const auto speed{prior.conditional<Pose>(Eigen::Vector3d{1.0, 2.0, 0.0})};
  static_assert(std::is_same_v<decltype(speed), const GammaDensity<double, Speed>>);
}

TEST(ProductDensity, ConditionalInsideAFactorReplacesIt) {
  const auto prior{makeVehiclePrior()};
  const auto given{prior.conditional<East>(Eigen::Matrix<double, 1, 1>{12.0})};
  ASSERT_TRUE(given.has_value());  // a Gaussian conditional must factorize, so this can fail
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(*given)>,
                               ProductDensity<double, EigenFilter::CoordinateSpace<3>, Gaussian<double, North>,
                                              UniformDensity<double, Heading>, GammaDensity<double, Speed>>>);
  const auto northGaussian{makePlace().conditional<East>(Eigen::Matrix<double, 1, 1>{12.0})};
  const auto north{northGaussian.toDensity()};
  ASSERT_TRUE(north.has_value());
  const Eigen::Vector3d x{-3.0, 0.5, 2.5};
  EXPECT_NEAR(given->logPdf(x), north->logPdf(x.head<1>()) - std::log(2 * pi) + speedLogPdf(2.5), 1e-13);
}

TEST(ProductDensity, ConditionalInsideAStudentTFactor) {
  using Plane = Composite<Place, Heading>;
  const auto t{EigenFilter::StudentTDensity<double, Place>::create(Eigen::Vector2d::Zero(),
                                                                   Eigen::Matrix2d{{1.0, 0.5}, {0.5, 1.0}}, 3.0)};
  ASSERT_TRUE(t.has_value());
  const ProductDensity product{Plane{}, on<Place>(*t), on<Heading>(UniformDensity<double, Heading>{})};
  const auto given{product.conditional<North>(Eigen::Matrix<double, 1, 1>{2.0})};
  ASSERT_TRUE(given.has_value());
  const auto east{t->conditional<North>(Eigen::Matrix<double, 1, 1>{2.0})};
  ASSERT_TRUE(east.has_value());
  EXPECT_EQ(given->factor<0>().dof(), 4.0);
  EXPECT_NEAR(given->logPdf(Eigen::Vector2d{1.0, 0.2}),
              east->logPdf(Eigen::Matrix<double, 1, 1>{1.0}) - std::log(2 * pi), 1e-13);
}
