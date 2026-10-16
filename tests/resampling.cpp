#include <EigenFilter/Resampling.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

using EigenFilter::MultinomialResampler;
using EigenFilter::ResidualResampler;
using EigenFilter::StratifiedResampler;
using EigenFilter::SystematicResampler;

namespace {

static_assert(EigenFilter::Resampler<SystematicResampler, double, std::mt19937_64>);
static_assert(EigenFilter::Resampler<ResidualResampler<SystematicResampler>, float, std::mt19937>);

// Unnormalized on purpose; index 2 and the last index have zero weight.
const std::vector<double> weights{0.8, 0.4, 0.0, 1.6, 0.2, 0.0};
constexpr double totalWeight{3.0};

template <typename R>
std::vector<std::size_t> countOffspring(const R& resampler, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 rng{seed};
  std::vector<Eigen::Index> ancestors(m);
  resampler(std::span<const double>{weights}, std::span<Eigen::Index>{ancestors}, rng);
  std::vector<std::size_t> counts(weights.size(), 0);
  for (const auto a : ancestors) {
    EXPECT_GE(a, 0);
    EXPECT_LT(a, Eigen::Index(weights.size()));
    if (a >= 0 && a < Eigen::Index(weights.size())) ++counts[std::size_t(a)];
  }
  return counts;
}

template <typename R>
class ResamplerTest : public ::testing::Test {};

using Resamplers = ::testing::Types<MultinomialResampler, StratifiedResampler, SystematicResampler, ResidualResampler<>,
                                    ResidualResampler<SystematicResampler>>;

}  // namespace

TYPED_TEST_SUITE(ResamplerTest, Resamplers);

TYPED_TEST(ResamplerTest, NeverChoosesZeroWeight) {
  for (std::uint64_t seed{0}; seed < 50; ++seed) {
    const auto counts{countOffspring(TypeParam{}, 7, seed)};
    EXPECT_EQ(counts[2], 0u);
    EXPECT_EQ(counts[5], 0u);
  }
}

TYPED_TEST(ResamplerTest, OffspringProportionalToWeight) {
  constexpr std::size_t m{30000};
  const auto counts{countOffspring(TypeParam{}, m, 1)};
  for (std::size_t i{0}; i < weights.size(); ++i) {
    const double p{weights[i] / totalWeight};
    const double sigma{std::sqrt(double(m) * p * (1 - p))};
    EXPECT_NEAR(double(counts[i]), double(m) * p, 5 * sigma + 1) << "index " << i;
  }
}

TYPED_TEST(ResamplerTest, SingleNonzeroWeight) {
  const std::vector<double> oneHot{0.0, 0.0, 2.0};
  std::mt19937_64 rng{3};
  std::vector<Eigen::Index> ancestors(5);
  TypeParam{}(std::span<const double>{oneHot}, std::span<Eigen::Index>{ancestors}, rng);
  for (const auto a : ancestors) EXPECT_EQ(a, 2);
}

TEST(SystematicResampler, CountsAreFloorOrCeil) {
  constexpr std::size_t m{17};
  for (std::uint64_t seed{0}; seed < 50; ++seed) {
    const auto counts{countOffspring(SystematicResampler{}, m, seed)};
    for (std::size_t i{0}; i < weights.size(); ++i) {
      const double expected{double(m) * weights[i] / totalWeight};
      EXPECT_GE(double(counts[i]), std::floor(expected)) << "index " << i;
      EXPECT_LE(double(counts[i]), std::ceil(expected)) << "index " << i;
    }
  }
}

TEST(ResidualResampler, CountsAtLeastFloor) {
  constexpr std::size_t m{17};
  for (std::uint64_t seed{0}; seed < 50; ++seed) {
    const auto counts{countOffspring(ResidualResampler<>{}, m, seed)};
    for (std::size_t i{0}; i < weights.size(); ++i) {
      EXPECT_GE(double(counts[i]), std::floor(double(m) * weights[i] / totalWeight)) << "index " << i;
    }
  }
}
