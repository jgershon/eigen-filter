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

// A pose, a variable number of landmarks, and a variable number of sensor bias angles:
// two resizable blocks, so offsets after the landmarks depend on their count.
struct Pose : Linear<3> {};
struct Landmark : Linear<2> {};
struct Landmarks : Repeated<Landmark> {
  using Repeated::Repeated;
};
struct Bias : Periodic<> {};
struct Biases : Repeated<Bias> {
  using Repeated::Repeated;
};
using Map = Composite<Pose, Landmarks, Biases>;

Map makeMap(Eigen::Index landmarks, Eigen::Index biases) {
  return Map{Pose{}, Landmarks{Landmark{}, landmarks}, Biases{Bias{}, biases}};
}

// Mean 0..n-1 and a covariance with every pair correlated, so misplaced blocks show.
Eigen::VectorXd iota(Eigen::Index n) { return Eigen::VectorXd::LinSpaced(n, 0.0, double(n - 1)) * 0.1; }

Eigen::MatrixXd makeCov(Eigen::Index n) {
  Eigen::MatrixXd cov{Eigen::MatrixXd::Constant(n, n, 0.01)};
  for (Eigen::Index i{0}; i < n; ++i) {
    for (Eigen::Index j{0}; j < n; ++j) cov(i, j) += 0.001 * double(i + j);
    cov(i, i) += 1.0 + double(i);
  }
  return cov;
}

template <typename T>
concept ResizableAsAWhole = requires(T space) { space.resize(Eigen::Index{0}); };

}  // namespace

TEST(MultipleResizable, SeveralResizableBlocks) {
  static_assert(Map::Stride == 1 && EigenFilter::spaceSize<Map> == Eigen::Dynamic);
  static_assert(Map::MaxSize == Eigen::Dynamic);
  static_assert(!ResizableAsAWhole<Map>, "with two resizable blocks, the split would be ambiguous");
  static_assert(ResizableAsAWhole<Composite<Pose, Landmarks>>);

  const Map map{makeMap(2, 3)};
  EXPECT_EQ(map.size(), 3 + 4 + 3);
  EXPECT_EQ(map.offset<1>(), 3);
  EXPECT_EQ(map.offset<2>(), 7);

  Eigen::VectorXd x{Eigen::VectorXd::Constant(10, 4.0)};
  map.normalizeValues(x);
  EXPECT_EQ(x.head<7>(), Eigen::VectorXd::Constant(7, 4.0));
  EXPECT_LT((x.tail<3>().array() - (4.0 - 2 * pi)).abs().maxCoeff(), 1e-15);
}

TEST(MultipleResizable, RuntimeIndices) {
  const Map map{makeMap(2, 3)};
  const Eigen::VectorXd x{Eigen::VectorXd::LinSpaced(10, 0.0, 9.0)};
  EXPECT_EQ(Eigen::VectorXd{x(idxIn<Landmarks>(map))}, Eigen::VectorXd::LinSpaced(4, 3.0, 6.0));
  EXPECT_EQ(Eigen::VectorXd{x(idxIn<Biases>(map))}, Eigen::Vector3d(7.0, 8.0, 9.0));
  EXPECT_EQ(idxOf<Biases>(map, 1), 8);
  EXPECT_EQ(Eigen::Vector2d{x(idxOf<Landmarks>(map, 1))}, Eigen::Vector2d(5.0, 6.0));

  // A fixed block keeps its compile-time size, and a static path is still a constant.
  const auto pose{x(idxIn<Pose>(map))};
  static_assert(decltype(pose)::RowsAtCompileTime == 3);
  static_assert(EigenFilter::idx<Map, Pose>.size() == 3);
}

TEST(MultipleResizable, GaussianInsertAndErase) {
  const Eigen::VectorXd mean{iota(10)};
  const Eigen::MatrixXd cov{makeCov(10)};
  Gaussian<double, Map> g{makeMap(2, 3), mean, cov};

  // A new landmark between the two, correlated with the pose only.
  const Eigen::Vector2d landmark{5.0, -5.0};
  const Eigen::Matrix2d landmarkCov{{0.5, 0.1}, {0.1, 0.4}};
  Eigen::MatrixXd cross{Eigen::MatrixXd::Zero(2, 10)};
  cross(0, 0) = 0.2;
  g.insert<Landmarks>(1, landmark, landmarkCov, cross);
  EXPECT_EQ(g.size(), 12);
  EXPECT_EQ(g.space().block<1>().count(), 3);
  EXPECT_EQ(g.mean().segment<2>(5), landmark);
  EXPECT_EQ(g.mean().head<5>(), mean.head<5>());
  EXPECT_EQ(g.mean().tail<5>(), mean.tail<5>());
  EXPECT_EQ((g.cov().block<2, 2>(5, 5)), landmarkCov);
  EXPECT_EQ(g.cov()(5, 0), 0.2);
  EXPECT_EQ(g.cov()(0, 5), 0.2);
  EXPECT_EQ((g.cov().block<5, 5>(0, 0)), (cov.block<5, 5>(0, 0)));
  EXPECT_EQ((g.cov().block<5, 5>(7, 7)), (cov.block<5, 5>(5, 5)));
  EXPECT_EQ((g.cov().block<5, 5>(0, 7)), (cov.block<5, 5>(0, 5)));

  // Bias offsets moved with the insertion, and a new bias is wrapped like the others.
  EXPECT_EQ(Eigen::VectorXd{g.mean()(idxIn<Biases>(g.space()))}, mean.tail<3>());
  g.insert<Biases>(3, Eigen::Matrix<double, 1, 1>{4.0}, Eigen::Matrix<double, 1, 1>{0.3});
  EXPECT_NEAR(g.mean()(12), 4.0 - 2 * pi, 1e-15);

  // Erasing both restores the original exactly: the marginal of the rest.
  g.erase<Biases>(3);
  g.erase<Landmarks>(1);
  EXPECT_EQ(g.space(), makeMap(2, 3));
  EXPECT_EQ(g.mean(), mean);
  EXPECT_EQ(g.cov(), cov);

  // Dropping a landmark from the middle.
  g.erase<Landmarks>(0);
  EXPECT_EQ(g.mean().segment<2>(3), mean.segment<2>(5));
  EXPECT_EQ((g.cov().block<2, 2>(3, 3)), (cov.block<2, 2>(5, 5)));
  EXPECT_TRUE(g.toDensity().has_value());
}

TEST(MultipleResizable, MarginalsAndConditionalsUseRuntimeOffsets) {
  const Gaussian<double, Map> g{makeMap(2, 3), iota(10), makeCov(10)};
  const auto biases{g.marginal<Biases>().copy()};
  static_assert(std::is_same_v<decltype(biases)::SpaceType, Biases>);
  EXPECT_EQ(biases.space().count(), 3);
  EXPECT_EQ(biases.mean(), iota(10).tail<3>());

  const auto givenLandmarks{g.conditional<Landmarks>(iota(10).segment<4>(3))};
  static_assert(std::is_same_v<decltype(givenLandmarks)::SpaceType, Composite<Pose, Biases>>);
  EXPECT_EQ(givenLandmarks.space().block<1>().count(), 3);
  EXPECT_LT((givenLandmarks.mean() - (Eigen::VectorXd(6) << iota(3), iota(10).tail<3>()).finished()).norm(), 1e-12);
}

TEST(MultipleResizable, MixtureInsertAndErase) {
  GaussianMixture<double, Map> mixture{makeMap(1, 1)};
  mixture.add(0.0, Eigen::VectorXd::Zero(6), Eigen::MatrixXd::Identity(6, 6));
  mixture.add(-1.0, Eigen::VectorXd::Ones(6), 2.0 * Eigen::MatrixXd::Identity(6, 6));
  mixture.insert<Landmarks>(1, Eigen::Vector2d{3.0, 4.0}, Eigen::Matrix2d::Identity());
  EXPECT_EQ(mixture.componentSize(), 8);
  for (const auto& component : mixture) {
    EXPECT_EQ(component.gaussian.space(), mixture.space());
    EXPECT_EQ(component.gaussian.mean().segment<2>(5), Eigen::Vector2d(3.0, 4.0));
  }
  mixture.erase<Landmarks>(0);
  EXPECT_EQ(mixture.componentSize(), 6);
  EXPECT_EQ(mixture.begin()->gaussian.mean().segment<2>(3), Eigen::Vector2d(3.0, 4.0));
  EXPECT_TRUE(mixture.toDensity().has_value());
}

TEST(MultipleResizable, ParticleInsertAndErase) {
  const Eigen::MatrixXd states{Eigen::MatrixXd::Random(6, 20)};
  ParticleSet<double, Map> set{makeMap(1, 1), states};
  set.setBandwidth(Eigen::MatrixXd::Identity(6, 6));
  set.erase<Landmarks>(0);
  EXPECT_EQ(set.particleSize(), 4);
  EXPECT_EQ(set.bandwidth()->rows(), 4);  // marginalized
  EXPECT_EQ(set.states().bottomRows<1>(), states.bottomRows<1>());

  // Spread across particles, so Silverman's rule has something to smooth.
  const Eigen::MatrixXd landmarks{Eigen::MatrixXd::Random(2, 20)};
  set.insert<Landmarks>(0, landmarks);
  EXPECT_EQ(set.particleSize(), 6);
  EXPECT_FALSE(set.bandwidth().has_value());  // no longer fits, so cleared
  EXPECT_EQ(set.states().middleRows<2>(3), landmarks);
  EXPECT_TRUE(set.toDensity().has_value());
}

TEST(MultipleResizable, TransformAndUpdateAResizableState) {
  // Each bias is read directly; the readings are as many as the biases.
  struct Reading : Linear<1> {};
  struct Readings : Repeated<Reading> {
    using Repeated::Repeated;
  };
  const Map map{makeMap(2, 3)};
  const Gaussian<double, Map> x{map, iota(10), makeCov(10)};
  const Eigen::Vector3d z{0.75, 0.85, 0.95};
  const auto innovation{[&map](const auto& state, const Eigen::Vector3d& readings) {
    using T = ScalarOf<decltype(state)>;
    return Eigen::Matrix<T, Eigen::Dynamic, 1>{readings.cast<T>() - state(idxIn<Biases>(map))};
  }};
  const auto predicted{EigenFilter::affine(innovation, Readings{Reading{}, 3}, withCross(x), z)};
  EXPECT_EQ(predicted.cross.rows(), 10);
  EXPECT_EQ(predicted.cross.cols(), 3);

  const auto [posterior, logLikelihood]{EigenFilter::update(x, predicted)};
  static_assert(std::is_same_v<typename decltype(posterior)::SpaceType, Map>);
  EXPECT_EQ(posterior.space(), map);
  EXPECT_TRUE(std::isfinite(logLikelihood));
  // Observing the biases exactly would pin them; with no noise, the update does.
  EXPECT_LT((posterior.mean().template tail<3>() - z).norm(), 1e-10);

  // With no output space given, a dynamic return is Linear and sized by f.
  const auto unnamed{EigenFilter::affine(innovation, x, z)};
  static_assert(std::is_same_v<decltype(unnamed)::SpaceType, Linear<Eigen::Dynamic>>);
  EXPECT_EQ(unnamed.space().size(), 3);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

TEST(MultipleResizableDeathTest, SizesMustMatchTheSpace) {
  // With two resizable blocks, the space cannot guess how to split new rows.
  Gaussian<double, Map> g{makeMap(2, 3), iota(10), makeCov(10)};
  EXPECT_DEATH(g.resize(12), "fit the space");
  EXPECT_DEATH((Gaussian<double, Map>{makeMap(1, 1), iota(10), makeCov(10)}), "fit the space");

  EXPECT_DEATH(g.insert<Landmarks>(3, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()), "out of range");
  EXPECT_DEATH(g.insert<Landmarks>(0, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()), "whole elements");
  EXPECT_DEATH(g.erase<Biases>(2, 2), "out of range");

  struct Capped : Repeated<Landmark, 2> {
    using Repeated::Repeated;
  };
  Gaussian<double, Composite<Pose, Capped>> capped{Composite<Pose, Capped>{Pose{}, Capped{Landmark{}, 2}}, iota(7),
                                                   makeCov(7)};
  EXPECT_DEATH(capped.insert<Capped>(0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()), "fit the space");
}

#endif
