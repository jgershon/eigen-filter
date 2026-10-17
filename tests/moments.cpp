#include <EigenFilter/Moments.h>
#include <EigenFilter/SampleSet.h>

#include <gtest/gtest.h>

#include <type_traits>

using EigenFilter::Moments;
using EigenFilter::MomentsN;

namespace {

struct Heading : EigenFilter::Periodic<> {};
using State = EigenFilter::Composite<EigenFilter::Linear<2>, Heading>;

}  // namespace

TEST(Moments, SpaceSetsTheSizeAndIsPartOfTheType) {
  using StateMoments = Moments<double, State>;
  static_assert(std::is_same_v<StateMoments::SpaceType, State>);
  static_assert(std::is_same_v<StateMoments::VectorType, Eigen::Vector3d>);
  static_assert(std::is_same_v<StateMoments::MatrixType, Eigen::Matrix3d>);
  // Same storage, different space: not interchangeable.
  static_assert(!std::is_convertible_v<StateMoments, MomentsN<double, 3>>);
  static_assert(
      std::is_same_v<MomentsN<double, Eigen::Dynamic, 6>, Moments<double, EigenFilter::Linear<Eigen::Dynamic, 6>>>);
}

TEST(Moments, SampleSetTakesItsSizeFromTheSpace) {
  static_assert(std::is_same_v<EigenFilter::SampleSet<double, State>, Eigen::Matrix<double, 3, Eigen::Dynamic>>);
  static_assert(std::is_same_v<EigenFilter::SampleSet<double, EigenFilter::Linear<Eigen::Dynamic, 5>>,
                               EigenFilter::SampleSetN<double, Eigen::Dynamic, 5>>);
}

TEST(Moments, FixedSize) {
  MomentsN<double, 3> m;
  EXPECT_EQ(m.size(), 3);
  EXPECT_EQ(m.cov().rows(), 3);
  EXPECT_EQ(m.cov().cols(), 3);
}

TEST(Moments, DynamicSizeAndResize) {
  MomentsN<double, Eigen::Dynamic> m;
  EXPECT_EQ(m.size(), 0);
  m.resize(4);
  EXPECT_EQ(m.size(), 4);
  EXPECT_EQ(m.mean().size(), 4);
  EXPECT_EQ(m.cov().rows(), 4);
  EXPECT_EQ(m.cov().cols(), 4);
}

TEST(Moments, DynamicWithMaxSize) {
  MomentsN<double, Eigen::Dynamic, 6> m{2};
  EXPECT_EQ(m.size(), 2);
  m.resize(6);
  EXPECT_EQ(m.size(), 6);
}

TEST(Moments, FromMeanAndCov) {
  const Eigen::Vector2d mean{1.0, 2.0};
  const Eigen::Matrix2d cov{{2.0, 0.5}, {0.5, 1.0}};
  MomentsN<double, Eigen::Dynamic> m{mean, cov};
  EXPECT_EQ(m.mean(), mean);
  EXPECT_EQ(m.cov(), cov);
}

TEST(Moments, MutableAccess) {
  MomentsN<double, 2> m;
  m.mean().setConstant(3.0);
  m.cov().setIdentity();
  EXPECT_EQ(m.mean(), Eigen::Vector2d(3.0, 3.0));
  EXPECT_EQ(m.cov(), Eigen::Matrix2d::Identity());
}
