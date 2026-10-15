#include <eigen_filter/eigen_filter.hpp>

#include <gtest/gtest.h>

TEST(Smoke, CompilesAsCxx23) {
  EXPECT_GE(__cplusplus, 202302L);
}

TEST(Smoke, EigenIsVersion5) {
  // Eigen 5 keeps EIGEN_WORLD_VERSION at 3; the release number is the major.
  EXPECT_GE(EIGEN_MAJOR_VERSION, 5);
}

TEST(Smoke, EigenWorks) {
  const Eigen::Matrix2d a = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d x(1.0, 2.0);
  EXPECT_TRUE((a * x).isApprox(x));
}
