#include <EigenFilter/Moments.h>

#include <gtest/gtest.h>

using EigenFilter::Moments;

TEST(Moments, FixedSize) {
  Moments<double, 3> m;
  EXPECT_EQ(m.size(), 3);
  EXPECT_EQ(m.cov().rows(), 3);
  EXPECT_EQ(m.cov().cols(), 3);
}

TEST(Moments, DynamicSizeAndResize) {
  Moments<double, Eigen::Dynamic> m;
  EXPECT_EQ(m.size(), 0);
  m.resize(4);
  EXPECT_EQ(m.size(), 4);
  EXPECT_EQ(m.mean().size(), 4);
  EXPECT_EQ(m.cov().rows(), 4);
  EXPECT_EQ(m.cov().cols(), 4);
}

TEST(Moments, DynamicWithMaxSize) {
  Moments<double, Eigen::Dynamic, 6> m(2);
  EXPECT_EQ(m.size(), 2);
  m.resize(6);
  EXPECT_EQ(m.size(), 6);
}

TEST(Moments, FromMeanAndCov) {
  const Eigen::Vector2d mean(1.0, 2.0);
  const Eigen::Matrix2d cov{{2.0, 0.5}, {0.5, 1.0}};
  Moments<double, Eigen::Dynamic> m(mean, cov);
  EXPECT_EQ(m.mean(), mean);
  EXPECT_EQ(m.cov(), cov);
}

TEST(Moments, MutableAccess) {
  Moments<double, 2> m;
  m.mean().setConstant(3.0);
  m.cov().setIdentity();
  EXPECT_EQ(m.mean(), Eigen::Vector2d(3.0, 3.0));
  EXPECT_EQ(m.cov(), Eigen::Matrix2d::Identity());
}
