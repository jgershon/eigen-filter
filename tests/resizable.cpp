#include <EigenFilter/GaussianMixture.h>
#include <EigenFilter/ParticleSet.h>
#include <EigenFilter/Transform.h>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <type_traits>

using EigenFilter::Composite;
using EigenFilter::Gaussian;
using EigenFilter::GaussianMixture;
using EigenFilter::idxIn;
using EigenFilter::idxOf;
using EigenFilter::Linear;
using EigenFilter::ParticleSet;
using EigenFilter::Periodic;
using EigenFilter::Repeated;
using EigenFilter::ScalarOf;
using EigenFilter::withCross;

namespace {

constexpr double pi{std::numbers::pi};

// A pose and a variable number of landmarks, each a position and an orientation.
struct Pose : Linear<3> {};
struct Orientation : Periodic<> {};
struct Landmark : Composite<Linear<2>, Orientation> {};
struct Landmarks : Repeated<Landmark> {
  using Repeated::Repeated;
};
using Map = Composite<Pose, Landmarks>;

Map makeMap(Eigen::Index landmarks) { return Map{Pose{}, Landmarks{Landmark{}, landmarks}}; }

// A small mean and a covariance with every pair correlated, so misplaced blocks show.
Eigen::VectorXd iota(Eigen::Index n) { return Eigen::VectorXd::LinSpaced(n, 0.0, double(n - 1)) * 0.1; }

Eigen::MatrixXd makeCov(Eigen::Index n) {
  Eigen::MatrixXd cov{Eigen::MatrixXd::Constant(n, n, 0.01)};
  for (Eigen::Index i{0}; i < n; ++i) {
    for (Eigen::Index j{0}; j < n; ++j) cov(i, j) += 0.001 * double(i + j);
    cov(i, i) += 1.0 + double(i);
  }
  return cov;
}

Eigen::Vector3d makeLandmark() { return {5.0, -5.0, 4.0}; }  // the orientation needs wrapping

}  // namespace

TEST(ResizableSpace, CarriesItsSize) {
  static_assert(EigenFilter::spaceSize<Map> == Eigen::Dynamic && Map::Stride == 3);
  Map map{makeMap(2)};
  EXPECT_EQ(map.size(), 9);
  EXPECT_EQ(map.offset<1>(), 3);
  map.resize(12);
  EXPECT_EQ(map.block<1>().count(), 3);

  Eigen::VectorXd x{Eigen::VectorXd::Constant(12, 4.0)};
  map.normalizeValues(x);
  for (const Eigen::Index row : {5, 8, 11}) EXPECT_NEAR(x(row), 4.0 - 2 * pi, 1e-15) << "row " << row;
  EXPECT_EQ(x(4), 4.0);
}

TEST(ResizableSpace, RuntimeIndices) {
  const Map map{makeMap(2)};
  const Eigen::VectorXd x{Eigen::VectorXd::LinSpaced(9, 0.0, 8.0)};
  EXPECT_EQ(Eigen::VectorXd{x(idxIn<Landmarks>(map))}, Eigen::VectorXd::LinSpaced(6, 3.0, 8.0));
  EXPECT_EQ(Eigen::VectorXd{x(idxIn<Landmarks, Orientation>(map))}, Eigen::Vector2d(5.0, 8.0));
  EXPECT_EQ(Eigen::Vector3d{x(idxOf<Landmarks>(map, 1))}, Eigen::Vector3d(6.0, 7.0, 8.0));
  EXPECT_EQ((idxOf<Landmarks, Orientation>(map, 0)), 5);

  // A fixed block keeps its compile-time size, and the constant idx<> agrees.
  const auto pose{x(idxIn<Pose>(map))};
  static_assert(decltype(pose)::RowsAtCompileTime == 3);
  EXPECT_EQ(Eigen::VectorXd{x(EigenFilter::idx<Map, Landmarks>)}, Eigen::VectorXd{x(idxIn<Landmarks>(map))});
}

TEST(ResizableSpace, GaussianInsertAndErase) {
  const Eigen::VectorXd mean{iota(9)};
  const Eigen::MatrixXd cov{makeCov(9)};
  Gaussian<double, Map> g{makeMap(2), mean, cov};

  // A new landmark between the two, correlated with the pose's first coordinate only.
  const Eigen::Matrix3d landmarkCov{{0.5, 0.1, 0.0}, {0.1, 0.4, 0.0}, {0.0, 0.0, 0.2}};
  Eigen::MatrixXd cross{Eigen::MatrixXd::Zero(3, 9)};
  cross(0, 0) = 0.2;
  g.insert<Landmarks>(1, makeLandmark(), landmarkCov, cross);
  EXPECT_EQ(g.size(), 12);
  EXPECT_EQ(g.space().block<1>().count(), 3);
  EXPECT_EQ(g.mean().segment<2>(6), makeLandmark().head<2>());
  EXPECT_NEAR(g.mean()(8), 4.0 - 2 * pi, 1e-15);  // wrapped like any stored orientation
  EXPECT_EQ(g.mean().head<6>(), mean.head<6>());
  EXPECT_EQ(g.mean().tail<3>(), mean.tail<3>());
  EXPECT_EQ((g.cov().block<3, 3>(6, 6)), landmarkCov);
  EXPECT_EQ(g.cov()(6, 0), 0.2);
  EXPECT_EQ(g.cov()(0, 6), 0.2);
  EXPECT_EQ((g.cov().block<6, 6>(0, 0)), (cov.block<6, 6>(0, 0)));
  EXPECT_EQ((g.cov().block<3, 3>(9, 9)), (cov.block<3, 3>(6, 6)));
  EXPECT_EQ((g.cov().block<6, 3>(0, 9)), (cov.block<6, 3>(0, 6)));

  // Erasing it restores the original exactly: the marginal of the rest.
  g.erase<Landmarks>(1);
  EXPECT_EQ(g.space(), makeMap(2));
  EXPECT_EQ(g.mean(), mean);
  EXPECT_EQ(g.cov(), cov);

  // Dropping the first landmark.
  g.erase<Landmarks>(0);
  EXPECT_EQ(g.mean().tail<3>(), mean.tail<3>());
  EXPECT_EQ((g.cov().block<3, 3>(3, 3)), (cov.block<3, 3>(6, 6)));
  EXPECT_TRUE(g.toDensity().has_value());
}

TEST(ResizableSpace, MarginalsAndConditionalsFollowTheSize) {
  const Gaussian<double, Map> g{makeMap(2), iota(9), makeCov(9)};
  const auto landmarks{g.marginal<Landmarks>().copy()};
  static_assert(std::is_same_v<decltype(landmarks)::SpaceType, Landmarks>);
  EXPECT_EQ(landmarks.space().count(), 2);
  EXPECT_EQ(landmarks.mean(), iota(9).tail<6>());

  const auto orientations{g.marginal<Landmarks, Orientation>().copy()};
  EXPECT_EQ(orientations.space().count(), 2);

  const auto pose{g.conditional<Landmarks>(iota(9).tail<6>())};
  static_assert(std::is_same_v<decltype(pose)::SpaceType, Pose>);
  EXPECT_LT((pose.mean() - iota(3)).norm(), 1e-12);
}

TEST(ResizableSpace, MixtureInsertAndErase) {
  GaussianMixture<double, Map> mixture{makeMap(1)};
  mixture.add(0.0, Eigen::VectorXd::Zero(6), Eigen::MatrixXd::Identity(6, 6));
  mixture.add(-1.0, Eigen::VectorXd::Ones(6), 2.0 * Eigen::MatrixXd::Identity(6, 6));
  mixture.insert<Landmarks>(0, makeLandmark(), Eigen::Matrix3d::Identity());
  EXPECT_EQ(mixture.componentSize(), 9);
  for (const auto& component : mixture) {
    EXPECT_EQ(component.gaussian.space(), mixture.space());
    EXPECT_EQ(component.gaussian.mean().segment<2>(3), makeLandmark().head<2>());
  }
  mixture.erase<Landmarks>(1);  // the original landmark; the inserted one remains
  EXPECT_EQ(mixture.componentSize(), 6);
  EXPECT_EQ(mixture.begin()->gaussian.mean().segment<2>(3), makeLandmark().head<2>());
  EXPECT_TRUE(mixture.toDensity().has_value());
}

TEST(ResizableSpace, ParticleInsertAndErase) {
  const Eigen::MatrixXd states{Eigen::MatrixXd::Random(6, 30)};
  ParticleSet<double, Map> set{makeMap(1), states};
  set.setBandwidth(Eigen::MatrixXd::Identity(6, 6));
  set.erase<Landmarks>(0);
  EXPECT_EQ(set.particleSize(), 3);
  EXPECT_EQ(set.bandwidth()->rows(), 3);  // marginalized
  EXPECT_EQ(set.states(), states.topRows<3>());

  // Spread across particles, so Silverman's rule has something to smooth.
  const Eigen::MatrixXd landmarks{Eigen::MatrixXd::Random(3, 30)};
  set.insert<Landmarks>(0, landmarks);
  EXPECT_EQ(set.particleSize(), 6);
  EXPECT_FALSE(set.bandwidth().has_value());  // no longer fits, so cleared
  EXPECT_EQ(set.states().bottomRows<3>(), landmarks);
  EXPECT_TRUE(set.toDensity().has_value());
}

TEST(ResizableSpace, TransformAndUpdateAResizableState) {
  // Each landmark's position is observed relative to the pose; there is a reading per landmark.
  struct Reading : Linear<2> {};
  struct Readings : Repeated<Reading> {
    using Repeated::Repeated;
  };
  const Map map{makeMap(2)};
  const Gaussian<double, Map> x{map, iota(9), makeCov(9)};
  const Eigen::Vector4d z{0.4, 0.4, 0.7, 0.7};
  const auto innovation{[](const auto& state, const Eigen::Vector4d& readings) {
    using T = ScalarOf<decltype(state)>;
    Eigen::Matrix<T, Eigen::Dynamic, 1> result{readings.cast<T>()};
    for (Eigen::Index k{0}; k < 2; ++k) result.segment(2 * k, 2) -= state.segment(3 + 3 * k, 2) - state.head(2);
    return result;
  }};
  const auto predicted{EigenFilter::linearized(innovation, Readings{Reading{}, 2}, withCross(x), z)};
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(predicted.cross)>, Eigen::MatrixXd>);
  EXPECT_EQ(predicted.cross.rows(), 9);
  EXPECT_EQ(predicted.cross.cols(), 4);
  EXPECT_EQ(predicted.output.space().count(), 2);

  const auto [posterior, logLikelihood]{EigenFilter::update(x, predicted)};
  static_assert(std::is_same_v<typename decltype(posterior)::SpaceType, Map>);
  EXPECT_EQ(posterior.space(), map);
  EXPECT_TRUE(std::isfinite(logLikelihood));
  // With no measurement noise, the relative positions now match the readings exactly.
  for (Eigen::Index k{0}; k < 2; ++k) {
    EXPECT_LT(
        (posterior.mean().template segment<2>(3 + 3 * k) - posterior.mean().template head<2>() - z.segment<2>(2 * k))
            .norm(),
        1e-9);
  }

  // With no output space given, a dynamic return is Linear and sized by f.
  const auto unnamed{EigenFilter::linearized(innovation, x, z)};
  static_assert(std::is_same_v<decltype(unnamed)::SpaceType, Linear<Eigen::Dynamic>>);
  EXPECT_EQ(unnamed.space().size(), 4);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

TEST(ResizableSpaceDeathTest, SizesMustMatchTheSpace) {
  Gaussian<double, Map> g{makeMap(2), iota(9), makeCov(9)};
  EXPECT_DEATH(g.resize(10), "fit the space");
  EXPECT_DEATH(g.insert<Landmarks>(3, makeLandmark(), Eigen::Matrix3d::Identity()), "out of range");
  EXPECT_DEATH(g.insert<Landmarks>(0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()), "whole elements");
  EXPECT_DEATH(g.erase<Landmarks>(1, 2), "out of range");

  struct Capped : Repeated<Landmark, 2> {
    using Repeated::Repeated;
  };
  Gaussian<double, Composite<Pose, Capped>> capped{Composite<Pose, Capped>{Pose{}, Capped{Landmark{}, 2}}, iota(9),
                                                   makeCov(9)};
  EXPECT_DEATH(capped.insert<Capped>(0, makeLandmark(), Eigen::Matrix3d::Identity()), "fit the space");
}

#endif
