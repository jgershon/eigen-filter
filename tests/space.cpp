#include <EigenFilter/Gaussian.h>
#include <EigenFilter/Space.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <type_traits>
#include <vector>

using EigenFilter::Bounded;
using EigenFilter::Composite;
using EigenFilter::Coordinate;
using EigenFilter::CoordinateKind;
using EigenFilter::idx;
using EigenFilter::idxOf;
using EigenFilter::isValidSize;
using EigenFilter::Linear;
using EigenFilter::Periodic;
using EigenFilter::Repeated;
using EigenFilter::spaceSize;

namespace {

constexpr double pi{std::numbers::pi};

struct Position : Linear<3> {};
struct Heading : Periodic<> {
  using Periodic::Periodic;
};
struct Range : Bounded<> {
  using Bounded::Bounded;
};
using State = Composite<Position, Heading, Range>;

struct Ownship : Linear<6> {};
struct Latitude : Linear<1> {};
struct Longitude : Periodic<> {
  using Periodic::Periodic;
};
struct Altitude : Linear<1> {};
struct Vehicle : Composite<Latitude, Longitude, Altitude> {};
struct Vehicles : Repeated<Vehicle> {};
using World = Composite<Ownship, Vehicles>;

// Two groups that both contain an X, so a bare X is ambiguous.
struct X : Linear<1> {};
struct Y : Linear<1> {};
struct Left : Composite<X, Y> {};
struct Right : Composite<X> {};
using Sides = Composite<Left, Right>;

// The default space takes no storage.
struct WithEmptySpace {
  EIGEN_FILTER_NO_UNIQUE_ADDRESS Linear<4> space;
  double value;
};
static_assert(sizeof(WithEmptySpace) == sizeof(double));

Eigen::VectorXd iota(Eigen::Index n) { return Eigen::VectorXd::LinSpaced(n, 0.0, double(n - 1)); }

}  // namespace

TEST(Space, SizesAndStrides) {
  static_assert(Linear<3>::FixedSize == 3 && Linear<3>::Stride == 0);
  static_assert(Linear<Eigen::Dynamic>::FixedSize == 0 && Linear<Eigen::Dynamic>::Stride == 1);
  static_assert(State::FixedSize == 5 && State::Stride == 0 && spaceSize<State> == 5);
  static_assert(World::FixedSize == 6 && World::Stride == 3 && spaceSize<World> == Eigen::Dynamic);
  static_assert(State::IsLinear == false && Composite<Position, Altitude>::IsLinear);

  EXPECT_TRUE(isValidSize<State>(5));
  EXPECT_FALSE(isValidSize<State>(6));
  EXPECT_TRUE(isValidSize<World>(6));
  EXPECT_TRUE(isValidSize<World>(12));
  EXPECT_FALSE(isValidSize<World>(10));
  EXPECT_FALSE(isValidSize<World>(3));
}

TEST(Space, MaximumSizes) {
  struct Fleet : Repeated<Vehicle, 4> {};
  using CappedWorld = Composite<Ownship, Fleet>;
  static_assert(World::MaxSize == Eigen::Dynamic && EigenFilter::spaceMaxSize<World> == Eigen::Dynamic);
  static_assert(CappedWorld::MaxSize == 6 + 3 * 4 && EigenFilter::spaceMaxSize<CappedWorld> == 18);
  static_assert(EigenFilter::spaceMaxSize<State> == 5);
  static_assert(Linear<Eigen::Dynamic, 8>::MaxSize == 8);
  EXPECT_TRUE(isValidSize<CappedWorld>(18));
  EXPECT_FALSE(isValidSize<CappedWorld>(21));
  EXPECT_TRUE((isValidSize<Linear<Eigen::Dynamic, 8>>(8)));
  EXPECT_FALSE((isValidSize<Linear<Eigen::Dynamic, 8>>(9)));
}

TEST(Space, PeriodicWrapsValuesAndDifferences) {
  const Periodic<3> angles;
  Eigen::Vector3d values{3.5, -3.5, 7.0};
  angles.normalizeValues(values);
  EXPECT_NEAR(values(0), 3.5 - 2 * pi, 1e-15);
  EXPECT_NEAR(values(1), -3.5 + 2 * pi, 1e-15);
  EXPECT_NEAR(values(2), 7.0 - 2 * pi, 1e-15);

  Eigen::Vector3d differences{6.0, -6.0, 0.5};
  angles.normalizeDifferences(differences);
  EXPECT_NEAR(differences(0), 6.0 - 2 * pi, 1e-15);
  EXPECT_NEAR(differences(1), -6.0 + 2 * pi, 1e-15);
  EXPECT_EQ(differences(2), 0.5);

  const Periodic<> bins{256.0, 0.0};
  Eigen::Matrix<double, 1, 3> bin{-1.0, 256.0, 300.0};
  bins.normalizeValues(bin);
  EXPECT_EQ(bin, (Eigen::Matrix<double, 1, 3>{255.0, 0.0, 44.0}));
}

TEST(Space, BoundedReflects) {
  const Bounded<> range;
  Eigen::Vector2d values{-2.0, 3.0};
  range.normalizeValues(values);
  EXPECT_EQ(values, Eigen::Vector2d(2.0, 3.0));

  const Bounded<> unit{0.0, 1.0};
  Eigen::Vector4d folded{1.3, -0.4, 2.5, 0.25};
  unit.normalizeValues(folded);
  EXPECT_NEAR(folded(0), 0.7, 1e-15);
  EXPECT_NEAR(folded(1), 0.4, 1e-15);
  EXPECT_NEAR(folded(2), 0.5, 1e-15);
  EXPECT_EQ(folded(3), 0.25);

  Eigen::Vector2d differences{-5.0, 5.0};
  range.normalizeDifferences(differences);
  EXPECT_EQ(differences, Eigen::Vector2d(-5.0, 5.0));
}

TEST(Space, CompositeNormalizesEachBlockAndEverySample) {
  const State state;
  Eigen::Matrix<double, 5, 2> samples;
  samples.col(0) << 10.0, 20.0, 30.0, 4.0, -1.0;
  samples.col(1) << -10.0, 0.0, 0.0, -4.0, 2.0;
  state.normalizeValues(samples);
  EXPECT_EQ(samples.col(0).head<3>(), Eigen::Vector3d(10.0, 20.0, 30.0));
  EXPECT_NEAR(samples(3, 0), 4.0 - 2 * pi, 1e-15);
  EXPECT_EQ(samples(4, 0), 1.0);
  EXPECT_NEAR(samples(3, 1), -4.0 + 2 * pi, 1e-15);
  EXPECT_EQ(samples(4, 1), 2.0);
}

TEST(Space, RepeatedNormalizesEveryRepetition) {
  const World world;
  Eigen::VectorXd x{Eigen::VectorXd::Zero(12)};
  x(7) = 4.0;    // vehicle 0 longitude
  x(10) = -4.0;  // vehicle 1 longitude
  x(8) = 4.0;    // vehicle 0 altitude, linear
  world.normalizeValues(x);
  EXPECT_NEAR(x(7), 4.0 - 2 * pi, 1e-15);
  EXPECT_NEAR(x(10), -4.0 + 2 * pi, 1e-15);
  EXPECT_EQ(x(8), 4.0);
}

TEST(SpaceIndex, FixedTags) {
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(idx<State, Heading>)>, Eigen::Index>);
  static_assert(idx<State, Heading> == 3);  // usable in constant expressions
  static_assert(idx<State, Range> == 4);

  const Eigen::Matrix<double, 5, 1> x{iota(5)};
  const auto position{x(idx<State, Position>)};
  static_assert(decltype(position)::RowsAtCompileTime == 3);
  EXPECT_EQ(Eigen::Vector3d{position}, Eigen::Vector3d(0.0, 1.0, 2.0));
}

TEST(SpaceIndex, RepeatedTail) {
  const Eigen::VectorXd x{iota(12)};
  EXPECT_EQ(Eigen::VectorXd{x(idx<World, Ownship>)}, iota(6));
  EXPECT_EQ(Eigen::VectorXd{x(idx<World, Vehicles>)}, (iota(6).array() + 6.0).matrix());
  EXPECT_EQ(Eigen::VectorXd{x(idx<World, Vehicles, Altitude>)}, Eigen::Vector2d(8.0, 11.0));
  EXPECT_EQ(Eigen::VectorXd{x(idx<World, Vehicles, Longitude>)}, Eigen::Vector2d(7.0, 10.0));

  EXPECT_EQ(Eigen::Vector3d{x(idxOf<World, Vehicles>(1))}, Eigen::Vector3d(9.0, 10.0, 11.0));
  EXPECT_EQ((idxOf<World, Vehicles, Longitude>(1)), 10);
  EXPECT_EQ((idxOf<World, Vehicles, Altitude>(0)), 8);
}

TEST(SpaceIndex, QualifiedPaths) {
  static_assert(idx<Sides, Left, X> == 0);
  static_assert(idx<Sides, Right, X> == 2);
  static_assert(idx<Sides, Y> == 1);  // unique, so no qualification needed
}

TEST(SpaceProjection, PicksCoordinates) {
  const State state{Position{}, Heading{100.0}, Range{1.0, 5.0}};
  const auto projected{EigenFilter::internal::projectSpace<2, 2>(state, 5, Eigen::Vector2<Eigen::Index>{4, 3})};
  static_assert(std::is_same_v<decltype(projected), const EigenFilter::CoordinateSpace<2>>);
  EXPECT_EQ(projected.coordinate(0), (Coordinate{CoordinateKind::Bounded, 1.0, 5.0}));
  EXPECT_EQ(projected.coordinate(1), (Coordinate{CoordinateKind::Periodic, -50.0, 100.0}));

  const auto linear{EigenFilter::internal::projectSpace<2, 2>(Linear<5>{}, 5, Eigen::Vector2<Eigen::Index>{4, 3})};
  static_assert(std::is_same_v<decltype(linear), const Linear<2>>);
}
