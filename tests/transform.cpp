#include <EigenFilter/Transform.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <tuple>
#include <type_traits>

using EigenFilter::Affine;
using EigenFilter::Composite;
using EigenFilter::Gaussian;
using EigenFilter::GaussianN;
using EigenFilter::Linear;
using EigenFilter::Linearized;
using EigenFilter::Periodic;
using EigenFilter::ScalarOf;
using EigenFilter::Unscented;
using EigenFilter::withCross;

namespace {

constexpr double pi{std::numbers::pi};

struct Position : Linear<2> {};
struct Velocity : Linear<2> {};
using State = Composite<Position, Velocity>;
struct Innovation : Linear<2> {};

Eigen::Matrix4d makeCov() {
  return Eigen::Matrix4d{{1.0, 0.2, 0.1, 0.0}, {0.2, 2.0, -0.1, 0.3}, {0.1, -0.1, 0.5, 0.05}, {0.0, 0.3, 0.05, 0.8}};
}

Gaussian<double, State> makeState() { return {State{}, Eigen::Vector4d{1.0, -2.0, 0.5, 0.25}, makeCov()}; }

// A constant-velocity step with a control input and process noise on the velocity.
const Eigen::Matrix4d F{{1.0, 0.0, 0.5, 0.0}, {0.0, 1.0, 0.0, 0.5}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};
const Eigen::Matrix<double, 4, 2> G{{0.125, 0.0}, {0.0, 0.125}, {0.5, 0.0}, {0.0, 0.5}};

const auto process{[](const auto& x, const auto& w, const Eigen::Vector4d& u) {
  using T = ScalarOf<decltype(x)>;
  return Eigen::Matrix<T, 4, 1>{F * x + G * w + u.cast<T>()};
}};

// Position measurement; the innovation is z - (Hx + v).
const Eigen::Matrix<double, 2, 4> H{{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}};

const auto innovation{[](const auto& x, const auto& v, const Eigen::Vector2d& z) {
  using T = ScalarOf<decltype(x)>;
  return Eigen::Matrix<T, 2, 1>{z.cast<T>() - H * x - v};
}};

GaussianN<double, 2> makeNoise() { return {Eigen::Vector2d::Zero(), Eigen::Matrix2d{{0.3, 0.1}, {0.1, 0.2}}}; }

template <typename Method>
class TransformMethodTest : public ::testing::Test {};

using Methods = ::testing::Types<Affine, Linearized<>, Unscented>;

}  // namespace

TYPED_TEST_SUITE(TransformMethodTest, Methods);

TYPED_TEST(TransformMethodTest, LinearProcessMatchesClosedForm) {
  const auto x{makeState()};
  const auto w{makeNoise()};
  const Eigen::Vector4d u{0.1, 0.2, 0.0, -0.1};
  const auto predicted{EigenFilter::transform(TypeParam{}, process, State{}, x, w, u)};
  static_assert(std::is_same_v<decltype(predicted), const Gaussian<double, State>>);
  EXPECT_LT((predicted.mean() - (F * x.mean() + u)).norm(), 1e-12);
  const Eigen::Matrix4d expected{F * makeCov() * F.transpose() + G * w.cov() * G.transpose()};
  EXPECT_LT((predicted.cov() - expected).norm(), 1e-12);
}

TYPED_TEST(TransformMethodTest, CrossWithTheMarkedInput) {
  const auto x{makeState()};
  const auto w{makeNoise()};
  const Eigen::Vector4d u{Eigen::Vector4d::Zero()};
  const auto predicted{EigenFilter::transform(TypeParam{}, process, State{}, withCross(x), w, u)};
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(predicted.cross)>, Eigen::Matrix4d>);
  static_assert(std::is_same_v<typename decltype(predicted.output)::SpaceType, State>);
  EXPECT_LT((predicted.cross - makeCov() * F.transpose()).norm(), 1e-12);
  const Eigen::Matrix4d expected{F * makeCov() * F.transpose() + G * w.cov() * G.transpose()};
  EXPECT_LT((predicted.output.cov() - expected).norm(), 1e-12);
}

TYPED_TEST(TransformMethodTest, UpdateIsTheKalmanUpdate) {
  const auto x{makeState()};
  const auto v{makeNoise()};
  const Eigen::Vector2d z{1.4, -1.5};
  const auto predicted{EigenFilter::transform(TypeParam{}, innovation, Innovation{}, withCross(x), v, z)};
  const auto [posterior, logLikelihood]{EigenFilter::update(x, predicted)};
  static_assert(std::is_same_v<typename decltype(posterior)::SpaceType, State>);

  const Eigen::Matrix4d P{makeCov()};
  const Eigen::Matrix2d S{H * P * H.transpose() + v.cov()};
  const Eigen::Matrix<double, 4, 2> K{P * H.transpose() * S.inverse()};
  const Eigen::Vector2d r{z - H * x.mean()};
  EXPECT_LT((posterior.mean() - (x.mean() + K * r)).norm(), 1e-10);
  EXPECT_LT((posterior.cov() - (Eigen::Matrix4d::Identity() - K * H) * P).norm(), 1e-10);
  const double expected{-0.5 * (2 * std::log(2 * pi) + std::log(S.determinant()) + r.dot(S.inverse() * r))};
  EXPECT_NEAR(logLikelihood, expected, 1e-10);
}

TEST(Transform, UpdateTakesAnObservation) {
  // A predicted measurement (not an innovation) updated with the observation itself.
  const auto x{makeState()};
  const auto v{makeNoise()};
  const Eigen::Vector2d z{1.4, -1.5};
  const auto measurement{
      [](const auto& s, const auto& noise) { return Eigen::Matrix<ScalarOf<decltype(s)>, 2, 1>{H * s + noise}; }};
  const auto byObservation{EigenFilter::update(x, EigenFilter::affine(measurement, withCross(x), v), z)};
  const auto byInnovation{EigenFilter::update(x, EigenFilter::affine(innovation, withCross(x), v, z))};
  EXPECT_LT((byObservation.posterior.mean() - byInnovation.posterior.mean()).norm(), 1e-12);
  EXPECT_LT((byObservation.posterior.cov() - byInnovation.posterior.cov()).norm(), 1e-12);
  EXPECT_NEAR(byObservation.logLikelihood, byInnovation.logLikelihood, 1e-12);
}

TEST(Transform, AutoDiffMatchesASuppliedJacobian) {
  // Range and bearing to the state's position from an uncertain sensor position, which is not kept.
  const auto rangeBearing{[](const auto& x, const auto& sensor) {
    using T = ScalarOf<decltype(x)>;
    using std::atan2;
    using std::sqrt;
    const Eigen::Matrix<T, 2, 1> d{x.template head<2>() - sensor};
    return Eigen::Matrix<T, 2, 1>{sqrt(d.squaredNorm()), atan2(d(1), d(0))};
  }};
  const auto jacobian{[](const Eigen::Vector4d& x, const Eigen::Vector2d& sensor) {
    const Eigen::Vector2d d{x.head<2>() - sensor};
    const double r2{d.squaredNorm()};
    const double r{std::sqrt(r2)};
    Eigen::Matrix<double, 2, 4> jx{Eigen::Matrix<double, 2, 4>::Zero()};
    jx.leftCols<2>() << d(0) / r, d(1) / r, -d(1) / r2, d(0) / r2;
    return std::tuple{jx, Eigen::Matrix2d{-jx.leftCols<2>()}};
  }};
  const auto x{makeState()};
  const GaussianN<double, 2> sensor{Eigen::Vector2d{-1.0, 1.0}, 0.04 * Eigen::Matrix2d::Identity()};
  const auto automatic{EigenFilter::transform(Linearized{}, rangeBearing, withCross(x), sensor)};
  const auto supplied{EigenFilter::transform(Linearized{jacobian}, rangeBearing, withCross(x), sensor)};
  EXPECT_LT((automatic.output.mean() - supplied.output.mean()).norm(), 1e-14);
  EXPECT_LT((automatic.output.cov() - supplied.output.cov()).norm(), 1e-12);
  EXPECT_LT((automatic.cross - supplied.cross).norm(), 1e-12);

  // The sensor's uncertainty widens the output; only the state's cross-covariance is returned.
  const auto [jx, js]{jacobian(x.mean(), sensor.mean())};
  const Eigen::Matrix2d expected{jx * makeCov() * jx.transpose() + js * sensor.cov() * js.transpose()};
  EXPECT_LT((automatic.output.cov() - expected).norm(), 1e-12);
  EXPECT_LT((automatic.cross - makeCov() * jx.transpose()).norm(), 1e-12);
}

TEST(Transform, UnscentedIsExactForASquare) {
  // For x ~ N(μ, σ²), E[x²] = μ² + σ² and Var[x²] = 4μ²σ² + 2σ⁴; linearizing misses the σ terms.
  const double mu{1.5};
  const double variance{0.25};
  const GaussianN<double, 1> x{Eigen::Matrix<double, 1, 1>{mu}, Eigen::Matrix<double, 1, 1>{variance}};
  const auto square{[](const auto& s) { return Eigen::Matrix<ScalarOf<decltype(s)>, 1, 1>{s(0) * s(0)}; }};

  const auto sigma{EigenFilter::unscented(square, x)};
  EXPECT_NEAR(sigma.mean()(0), mu * mu + variance, 1e-12);
  EXPECT_NEAR(sigma.cov()(0, 0), 4 * mu * mu * variance + 2 * variance * variance, 1e-12);

  const auto tangent{EigenFilter::linearized(square, x)};
  EXPECT_NEAR(tangent.mean()(0), mu * mu, 1e-12);
  EXPECT_NEAR(tangent.cov()(0, 0), 4 * mu * mu * variance, 1e-12);
}

TEST(Transform, UnscentedAveragesAcrossTheSeam) {
  // The function wraps its own output, so its sigma points straddle ±π.
  const GaussianN<double, 1> heading{Eigen::Matrix<double, 1, 1>{pi - 0.05}, Eigen::Matrix<double, 1, 1>{0.01}};
  const auto turn{[](const auto& theta) {
    using T = ScalarOf<decltype(theta)>;
    using std::atan2;
    using std::cos;
    using std::sin;
    return Eigen::Matrix<T, 1, 1>{atan2(sin(theta(0) + T(0.1)), cos(theta(0) + T(0.1)))};
  }};
  const auto turned{EigenFilter::unscented(turn, Periodic<>{}, heading)};
  EXPECT_NEAR(turned.mean()(0), -pi + 0.05, 1e-12);
  EXPECT_NEAR(turned.cov()(0, 0), 0.01, 1e-12);

  const auto unwrapped{EigenFilter::unscented(turn, heading)};
  EXPECT_GT(std::abs(unwrapped.cov()(0, 0) - 0.01), 1.0);  // averaging across the seam in a linear space
}

TEST(Transform, OutputSpaceDefaultsToLinear) {
  const auto x{makeState()};
  const auto positionAndSpeed{[](const auto& s) {
    using T = ScalarOf<decltype(s)>;
    return Eigen::Matrix<T, 3, 1>{s(0), s(1), s.template tail<2>().norm()};
  }};
  const auto result{EigenFilter::linearized(positionAndSpeed, x)};
  static_assert(std::is_same_v<decltype(result)::SpaceType, Linear<3>>);
  EXPECT_EQ(result.mean().head<2>(), x.mean().head<2>());
}

TEST(Transform, CrossWithAnyArgument) {
  // The marked input need not be first; its cross-covariance is with it alone.
  const auto a{makeNoise()};
  const GaussianN<double, 2> b{Eigen::Vector2d{1.0, 2.0}, 3.0 * Eigen::Matrix2d::Identity()};
  const auto sum{[](const auto& p, const auto& q) { return Eigen::Matrix<ScalarOf<decltype(p)>, 2, 1>{p + q}; }};
  const auto predicted{EigenFilter::affine(sum, a, withCross(b))};
  EXPECT_EQ(predicted.cross, b.cov());
  EXPECT_EQ(predicted.output.mean(), a.mean() + b.mean());
  EXPECT_EQ(predicted.output.cov(), a.cov() + b.cov());
}

TEST(Transform, AcceptsDensitiesAndMarginals) {
  const auto x{makeState()};
  const auto density{x.toDensity()};
  const auto speed{
      [](const auto& velocity) { return Eigen::Matrix<ScalarOf<decltype(velocity)>, 1, 1>{velocity.norm()}; }};
  const auto fromMarginal{EigenFilter::linearized(speed, x.marginal<Velocity>())};
  const auto fromDensity{EigenFilter::linearized(
      [&](const auto& s) { return speed(Eigen::Matrix<ScalarOf<decltype(s)>, 2, 1>{s.template tail<2>()}); },
      *density)};
  EXPECT_NEAR(fromMarginal.mean()(0), fromDensity.mean()(0), 1e-15);
  EXPECT_NEAR(fromMarginal.cov()(0, 0), fromDensity.cov()(0, 0), 1e-15);
}

TEST(Transform, UnscentedHandlesSemidefiniteNoise) {
  // Noise only on the first coordinate: its covariance has no Cholesky factor.
  const GaussianN<double, 2> w{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{0.5, 0.0}, {0.0, 0.0}}};
  const auto x{makeState()};
  const Eigen::Vector4d u{Eigen::Vector4d::Zero()};
  const auto sigma{EigenFilter::unscented(process, State{}, x, w, u)};
  const auto exact{EigenFilter::affine(process, State{}, x, w, u)};
  EXPECT_LT((sigma.mean() - exact.mean()).norm(), 1e-12);
  EXPECT_LT((sigma.cov() - exact.cov()).norm(), 1e-12);
}

namespace {

template <typename T>
concept Markable = requires(const T& input) { withCross(input); };

static_assert(Markable<GaussianN<double, 2>>);
static_assert(!Markable<Eigen::Vector2d>, "a known input has no cross-covariance");

// Declared only: the concept below never calls it.
struct Sum {
  template <typename P, typename Q>
  Eigen::Matrix<ScalarOf<P>, 2, 1> operator()(const P& p, const Q& q) const;
};

template <typename T>
concept TwoMarkable =
    requires(const T& a, const T& b) { EigenFilter::transform(Affine{}, Sum{}, withCross(a), withCross(b)); };

static_assert(!TwoMarkable<GaussianN<double, 2>>, "at most one input is marked");

}  // namespace
