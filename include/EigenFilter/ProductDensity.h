#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "Indexing.h"
#include "Macros.h"
#include "Moments.h"
#include "SampleSet.h"
#include "Sampling.h"
#include "Space.h"

namespace EigenFilter {

/** A distribution that evaluates a log-density in its space, such as a GaussianDensity, a
 * KernelDensity, or a UniformDensity. Its dimension is its space's size.
 */
template <typename T>
concept Density = requires(
    const T& density,
    const Eigen::Matrix<typename T::Scalar, T::SizeAtCompileTime, 1, Eigen::ColMajor, T::MaxSizeAtCompileTime, 1>& x) {
  typename T::SpaceType;
  { density.space().size() } -> std::convertible_to<Eigen::Index>;
  { density.logPdf(x) } -> std::convertible_to<typename T::Scalar>;
};

namespace internal {

/** An owner that factorizes once and then hands out cheap density views, such as a Gaussian or
 * a GaussianMixture.
 */
template <typename T>
concept FactorizableOwner = requires(const T& owner) {
  { owner.factorize().has_value() } -> std::convertible_to<bool>;
  requires Density<std::remove_cvref_t<decltype(*owner.toDensity())>>;
};

}  // namespace internal

/** What a ProductDensity takes as a factor: a Density, or a factorizable owner such as a
 * Gaussian, which the product holds and factorizes itself.
 */
template <typename T>
concept ProductFactor = Density<T> || internal::FactorizableOwner<T>;

namespace internal {

/** A factor for a ProductDensity on the subspace named by the tag path \a Path. */
template <ProductFactor F, typename... Path>
struct FactorOnPath {
  using FactorType = F;
  F factor;

  template <typename Space>
  [[nodiscard]] auto indicesIn(const Space& space) const {
    return asIndexSet(idxIn<Path...>(space));
  }
};

/** A factor for a ProductDensity on the coordinates \a indices. */
template <ProductFactor F, typename Indices>
struct FactorAt {
  using FactorType = F;
  F factor;
  Indices indices;

  template <typename Space>
  [[nodiscard]] auto indicesIn(const Space&) const {
    return asIndexSet(indices);
  }
};

/** \returns the density of \a factor: itself, or a view of a factorized owner */
template <typename F>
[[nodiscard]] decltype(auto) densityOf(const F& factor) {
  if constexpr (Density<F>) {
    return (factor);
  } else {
    auto view{*factor.toDensity()};
    return view;
  }
}

template <typename T>
inline constexpr bool isExpected{false};

template <typename T, typename E>
inline constexpr bool isExpected<std::expected<T, E>>{true};

/** Whether the moments() of the density of \a F returns a std::expected. */
template <typename F>
inline constexpr bool momentsMayFail{
    isExpected<std::remove_cvref_t<decltype(densityOf(std::declval<const F&>()).moments())>>};

template <typename... T>
struct TypeList {};

/** The types \a T after the first \a N. */
template <std::size_t N, typename... T>
struct DropFront {
  using type = TypeList<T...>;
};

template <std::size_t N, typename First, typename... Rest>
  requires(N > 0)
struct DropFront<N, First, Rest...> : DropFront<N - 1, Rest...> {};

/** The first \a N of the types \a T. */
template <std::size_t N, typename Done, typename... T>
struct TakeFrontImpl {
  using type = Done;
};

template <std::size_t N, typename... Done, typename First, typename... Rest>
  requires(N > 0)
struct TakeFrontImpl<N, TypeList<Done...>, First, Rest...> : TakeFrontImpl<N - 1, TypeList<Done..., First>, Rest...> {};

template <std::size_t N, typename... T>
using TakeFront = typename TakeFrontImpl<N, TypeList<>, T...>::type;

/** \returns the position within \a target of each entry of \a indices, as an index vector of
 * the given size; every entry must be in \a target, whose entries are below \a n
 */
template <int Size, int MaxSize, typename Indices, typename Target>
[[nodiscard]] IndexVector<Size, MaxSize> positionsIn(const Indices& indices, const Target& target, Eigen::Index n) {
  std::vector<Eigen::Index> position(std::size_t(n), -1);
  for (Eigen::Index j{0}; j < Eigen::Index(target.size()); ++j) position[std::size_t(target[j])] = j;
  IndexVector<Size, MaxSize> result;
  result.resize(Eigen::Index(indices.size()));
  for (Eigen::Index i{0}; i < result.size(); ++i) {
    result(i) = position[std::size_t(indices[i])];
    EIGEN_FILTER_ASSERT(result(i) >= 0 && "a factor's coordinate is missing from the result");
  }
  return result;
}

/** \returns whether \a a and \a b hold the same indices, in any order */
template <typename A, typename B>
[[nodiscard]] bool sameIndexSet(const A& a, const B& b) {
  std::vector<Eigen::Index> left(a.begin(), a.end());
  std::vector<Eigen::Index> right(b.begin(), b.end());
  std::ranges::sort(left);
  std::ranges::sort(right);
  return left == right;
}

/** The value inside a factor's conditional, unwrapped from a std::expected if need be. */
template <typename R>
struct ConditionalValue {
  using type = R;
};

template <typename T, typename E>
struct ConditionalValue<std::expected<T, E>> {
  using type = T;
};

/** The error a factor's conditional can report: from its std::expected, or from factorizing the
 * owner it returns; \c void if it cannot fail.
 */
template <typename R>
struct ConditionalError {
  using type = void;
};

template <typename T, typename E>
struct ConditionalError<std::expected<T, E>> {
  using type = E;
};

template <FactorizableOwner R>
struct ConditionalError<R> {
  using type = typename decltype(std::declval<const R&>().factorize())::error_type;
};

}  // namespace internal

template <typename Scalar_, typename Space_, ProductFactor... Factors>
class ProductDensity;

namespace internal {

/** The ProductDensity in \a Space of \a Factors with factor \a K replaced by \a Replacement. */
template <typename Scalar, typename Space, std::size_t K, typename Replacement, typename Factors,
          typename Sequence = std::make_index_sequence<std::tuple_size_v<Factors>>>
struct ReplacedProduct;

template <typename Scalar, typename Space, std::size_t K, typename Replacement, typename... Factors, std::size_t... J>
struct ReplacedProduct<Scalar, Space, K, Replacement, std::tuple<Factors...>, std::index_sequence<J...>> {
  using type = ProductDensity<Scalar, Space, std::conditional_t<J == K, Replacement, Factors>...>;
};

}  // namespace internal

/** \returns \a factor as a factor of a ProductDensity on the subspace named by the tag path
 * \a Path, such as <tt>on<Heading>(uniform)</tt>.
 */
template <typename... Path, ProductFactor F>
  requires(sizeof...(Path) > 0)
[[nodiscard]] internal::FactorOnPath<F, Path...> on(F factor) {
  return {std::move(factor)};
}

/** \returns \a factor as a factor of a ProductDensity on the coordinates \a indices: a single
 * index or any index set Eigen accepts.
 */
template <typename Indices, ProductFactor F>
[[nodiscard]] internal::FactorAt<F, Indices> on(Indices indices, F factor) {
  return {std::move(factor), std::move(indices)};
}

/** \brief The product of independent densities, each on its own coordinates of a space.
 *
 * Represents \f$ p(x) = \prod_k p_k(x_{S_k}) \f$, where the index sets \f$ S_k \f$ partition
 * the coordinates. A prior that is Gaussian in position, uniform in heading, and Gamma in
 * speed is one product:
 * \code
 * const EigenFilter::ProductDensity prior{State{}, on<Position>(position),  // a Gaussian<double, Position>
 *                                         on<Heading>(UniformDensity<double, Heading>{}),
 *                                         on<Speed>(GammaDensity<double, Speed>{4.0, 0.5})};
 * \endcode
 * Each factor's space must have the same coordinates (kinds, periods, and bounds) as the
 * coordinates it covers, so a factor on a named subspace is usually built in that tag's
 * space. Sampling fills each factor's rows; log-densities add.
 *
 * A factor is a Density or a factorizable owner (see ProductFactor), held by value. The
 * product factorizes an owner when it is built, and asserts that this succeeds; call the
 * owner's factorize() first to handle a failure. A factor that is a view, such as a
 * GaussianDensity, keeps its source locked while the product is alive, so prefer passing
 * the owner.
 *
 * marginal<Path...>() and conditional<GivenPath...>() work on the factors' blocks, matched by
 * the factors' space types:
 *  - a path through a factor's space type, or to a block inside it, is that factor's own
 *    marginal or conditional, and the other factors are unchanged;
 *  - a path to a block made of whole factors selects those factors, or drops them when given.
 * A marginal or conditional with one factor left returns that factor alone. Marginals return
 * owners or values, never views of this product. A conditional that may fail, such as a
 * Student-t's or a Gaussian's that must factorize, returns a std::expected.
 *
 * \tparam Scalar_   scalar type, shared by every factor
 * \tparam Space_    space of the whole vector (see \ref Spaces)
 * \tparam Factors   the factors' types, in order
 *
 * \sa on(), drawSamples(const ProductDensity&, Eigen::Index, G&)
 */
template <typename Scalar_, typename Space_, ProductFactor... Factors>
class ProductDensity {
  static_assert(sizeof...(Factors) > 0, "a product needs at least one factor");
  static_assert((std::is_same_v<typename Factors::Scalar, Scalar_> && ...), "every factor must have the same scalar");

  static constexpr bool MomentsMayFail{(internal::momentsMayFail<Factors> || ...)};

 public:
  using Scalar = Scalar_;
  using Index = Eigen::Index;
  using SpaceType = Space_;
  using MomentsType = Moments<Scalar_, Space_>;
  using VectorType = typename MomentsType::VectorType;
  using MatrixType = typename MomentsType::MatrixType;
  static constexpr int SizeAtCompileTime{MomentsType::SizeAtCompileTime};
  static constexpr int MaxSizeAtCompileTime{MomentsType::MaxSizeAtCompileTime};
  static constexpr std::size_t FactorCount{sizeof...(Factors)};

  template <std::size_t K>
  using FactorType = std::tuple_element_t<K, std::tuple<Factors...>>;

  /** The product of the factors \a placed, made by on(), in \a space. Their indices must
   * partition the space, and each factor's space must match the coordinates it covers.
   */
  template <typename... Placed>
    requires(sizeof...(Placed) == sizeof...(Factors) && (std::is_same_v<typename Placed::FactorType, Factors> && ...))
  explicit ProductDensity(Space_ space, Placed... placed)
      : m_space{std::move(space)}, m_indices{resolve<Factors>(placed)...}, m_factors{std::move(placed.factor)...} {
    EIGEN_FILTER_ASSERT(sizesMatch() && "a factor's dimension must match the number of coordinates it covers");
    EIGEN_FILTER_ASSERT(partitionsSpace() && "the factors' indices must partition the space");
    EIGEN_FILTER_ASSERT(spacesMatch() && "a factor's space must match the coordinates it covers");
    std::apply([](const auto&... factors) { (factorizeOwner(factors), ...); }, m_factors);
  }

  [[nodiscard]] Index size() const { return m_space.size(); }
  [[nodiscard]] const Space_& space() const { return m_space; }

  /** \returns factor \a K, as stored */
  template <std::size_t K>
  [[nodiscard]] const FactorType<K>& factor() const {
    return std::get<K>(m_factors);
  }

  /** \returns the coordinates factor \a K covers */
  template <std::size_t K>
  [[nodiscard]] const auto& indices() const {
    return std::get<K>(m_indices);
  }

  /** Calls \a fn(density, indices) for each factor in order, with its density (a view, for an
   * owner) and the coordinates it covers.
   */
  template <typename Fn>
  void forEachFactor(Fn&& fn) const {
    [&]<std::size_t... K>(std::index_sequence<K...>) {
      (fn(internal::densityOf(std::get<K>(m_factors)), std::get<K>(m_indices)), ...);
    }(std::index_sequence_for<Factors...>{});
  }

  /** \returns the log of the density at \a x: the sum of the factors' log-densities */
  template <typename Derived>
  [[nodiscard]] Scalar logPdf(const Eigen::MatrixBase<Derived>& x) const {
    static_assert(Derived::IsVectorAtCompileTime, "logPdf() expects a vector");
    EIGEN_FILTER_ASSERT(x.size() == size());
    Scalar result{0};
    forEachFactor(
        [&](const auto& density, const auto& indices) { result += Scalar(density.logPdf(x.derived()(indices))); });
    return result;
  }

  /** \returns the density at \a x
   *
   * \warning Underflows easily in high dimensions; prefer logPdf().
   */
  template <typename Derived>
  [[nodiscard]] Scalar pdf(const Eigen::MatrixBase<Derived>& x) const {
    return Eigen::numext::exp(logPdf(x));
  }

  /** \returns the mean and covariance: each factor's moments in its coordinates, and zero
   * covariance between factors. If any factor's moments() can fail, so can this, with the
   * first factor's error.
   */
  [[nodiscard]] auto moments() const
      -> std::conditional_t<MomentsMayFail, std::expected<MomentsType, MomentsError>, MomentsType> {
    MomentsType result{size()};
    result.mean().setZero();
    result.cov().setZero();
    std::optional<MomentsError> error;
    const auto place{[&](const auto& moments, const auto& indices) {
      using FactorMatrix = std::remove_cvref_t<decltype(moments.cov())>;
      result.mean()(indices) = moments.mean();
      result.cov()(indices, indices) = FactorMatrix{moments.cov().template selfadjointView<Eigen::Lower>()};
    }};
    forEachFactor([&](const auto& density, const auto& indices) {
      if (error) return;
      const auto factorMoments{density.moments()};
      if constexpr (internal::isExpected<std::remove_cvref_t<decltype(factorMoments)>>) {
        if (!factorMoments) {
          error = factorMoments.error();
        } else {
          place(*factorMoments, indices);
        }
      } else {
        place(factorMoments, indices);
      }
    });
    if constexpr (MomentsMayFail) {
      if (error) return std::unexpected{*error};
    }
    return result;
  }

  /** \returns the distribution of the subspace named by the tag path \a Path (see idx<>):
   *  - for a path through a factor's space type, or to a block inside it, that factor's
   *    marginal, as an owner or value (or the factor itself, for the factor's whole block);
   *  - for a block made of whole factors, the product of those factors in that subspace, or
   *    the factor itself if there is one.
   */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    using Match = MatchType<Path...>;
    if constexpr (Match::value.factor != None) {
      assertOnOwnBlock<Match::value.factor>(OwnBlockPath<Match::value, Path...>{});
      const auto& factor{std::get<Match::value.factor>(m_factors)};
      if constexpr (Match::value.whole) {
        return factor;
      } else {
        return ownedMarginal(factor, typename internal::DropFront<Match::value.start, Path...>::type{});
      }
    } else {
      static_assert(std::ranges::count(withinOf<Path...>(), true) > 0,
                    "the path neither reaches into a factor nor contains one");
      const auto target{internal::resolveIndices<SizeAtCompileTime, MaxSizeAtCompileTime>(
          size(), internal::asIndexSet(idxIn<Path...>(m_space)))};
      EIGEN_FILTER_ASSERT(internal::sameIndexSet(coveredBy(withinOf<Path...>()), target) &&
                          "the path's block must be made of whole factors");
      return keepOnly<withinOf<Path...>()>(target, internal::subspace<Space_, Path...>(m_space));
    }
  }

  /** \returns the distribution of the other coordinates given that the subspace named by the
   * tag path \a GivenPath (see idx<>) equals \a values:
   *  - for a path to a block inside a factor's space, that factor's conditional in place of
   *    the factor, and the other factors unchanged;
   *  - for a factor's whole block, or a block made of whole factors, the remaining factors,
   *    since the factors are independent.
   *
   * The result is in the space of the remaining blocks when \a GivenPath names one top-level
   * block of a Composite space, and otherwise in the projection of this space; with one
   * factor left, it is that factor (or its conditional) alone. When the conditioned factor's
   * conditional can fail, the result is a std::expected with its error, including a failure to
   * factorize a Gaussian it returns.
   */
  template <typename... GivenPath, typename Derived>
    requires(sizeof...(GivenPath) > 0)
  [[nodiscard]] auto conditional(const Eigen::MatrixBase<Derived>& values) const {
    return internal::tagConditional<SizeAtCompileTime, MaxSizeAtCompileTime, GivenPath...>(
        m_space, size(), [&](const auto& target, const auto& given, auto targetSpace) {
          EIGEN_FILTER_ASSERT(values.size() == given.size());
          using Match = MatchType<GivenPath...>;
          if constexpr (Match::value.factor != None && !Match::value.whole) {
            assertOnOwnBlock<Match::value.factor>(OwnBlockPath<Match::value, GivenPath...>{});
            return replaceFactor<Match::value.factor>(
                target, given, std::move(targetSpace),
                typename internal::DropFront<Match::value.start, GivenPath...>::type{}, values);
          } else {
            static_assert(std::ranges::count(droppedOf<GivenPath...>(), true) > 0,
                          "the path neither reaches into a factor nor contains one");
            static_assert(std::ranges::count(droppedOf<GivenPath...>(), true) < std::ptrdiff_t(FactorCount),
                          "conditioning on every factor leaves nothing");
            EIGEN_FILTER_ASSERT(internal::sameIndexSet(coveredBy(droppedOf<GivenPath...>()), given) &&
                                "the given block must be made of whole factors");
            return keepOnly<complement(droppedOf<GivenPath...>())>(target, std::move(targetSpace));
          }
        });
  }

 private:
  static constexpr std::size_t None{FactorCount};

  /** The factor a tag path reaches into, if any: the path from \c start on lies within it, and
   * \c whole means the path names exactly the factor's block.
   */
  struct PathMatch {
    std::size_t factor{None};
    std::size_t start{0};
    bool whole{false};
    bool named{false};  ///< the path names the factor's space type at \c start
  };

  /** The tag path of the block of the factor that \a Match found in \a Path: the path up to the
   * factor's space type, or that type alone if the path only reaches inside it.
   */
  template <PathMatch Match, typename... Path>
  using OwnBlockPath = std::conditional_t<Match.named, internal::TakeFront<Match.start + 1, Path...>,
                                          internal::TypeList<typename FactorType<Match.factor>::SpaceType>>;

  /** \returns the factor whose space type is \a T, or None */
  template <typename T>
  static consteval std::size_t factorNamed() {
    constexpr std::array<bool, FactorCount> same{std::is_same_v<T, typename Factors::SpaceType>...};
    static_assert(std::ranges::count(same, true) <= 1, "two factors share this space type, so it names neither");
    const auto found{std::ranges::find(same, true)};
    return found == same.end() ? None : std::size_t(found - same.begin());
  }

  /** \returns the factor whose space contains the tag \a T, or None */
  template <typename T>
  static consteval std::size_t factorContaining() {
    constexpr std::array<bool, FactorCount> inside{(internal::countTag<T, typename Factors::SpaceType>() > 0)...};
    static_assert(std::ranges::count(inside, true) <= 1, "this tag lies in more than one factor's space");
    const auto found{std::ranges::find(inside, true)};
    return found == inside.end() ? None : std::size_t(found - inside.begin());
  }

  /** \returns which factors' spaces lie within the block named by the tag \a T */
  template <typename T>
  static consteval std::array<bool, FactorCount> factorsWithin() {
    return {(internal::countTag<typename Factors::SpaceType, T>() > 0)...};
  }

  template <typename... Path>
  static consteval PathMatch matchPath() {
    constexpr std::array<std::size_t, sizeof...(Path)> named{factorNamed<Path>()...};
    for (std::size_t j{0}; j < named.size(); ++j) {
      if (named[j] != None) return {named[j], j, j + 1 == named.size(), true};
    }
    return {factorContaining<std::tuple_element_t<0, std::tuple<Path...>>>(), 0, false};
  }

  /** \returns the factors that conditioning on a path ending in the tag \a Last drops, when the
   * path does not reach inside a factor: the factor it names, or those within its block
   */
  template <PathMatch Match, typename Last>
  static consteval std::array<bool, FactorCount> droppedBy() {
    if constexpr (Match.factor != None) {
      std::array<bool, FactorCount> result{};
      result[Match.factor] = true;
      return result;
    } else {
      return factorsWithin<Last>();
    }
  }

  // The path analyses are types and consteval functions, not local constexpr variables: GCC can
  // conflate a local used as a template argument across instantiations of a generic lambda.

  /** The PathMatch of \a Path, as a type. */
  template <typename... Path>
  using MatchType = std::integral_constant<PathMatch, matchPath<Path...>()>;

  template <typename... Path>
  using LastTag = std::tuple_element_t<sizeof...(Path) - 1, std::tuple<Path...>>;

  /** The factors within the block named by \a Path. */
  template <typename... Path>
  static consteval std::array<bool, FactorCount> withinOf() {
    return factorsWithin<LastTag<Path...>>();
  }

  /** The factors that conditioning on \a Path drops, when it does not reach inside a factor. */
  template <typename... Path>
  static consteval std::array<bool, FactorCount> droppedOf() {
    return droppedBy<MatchType<Path...>::value, LastTag<Path...>>();
  }

  static consteval std::array<bool, FactorCount> complement(std::array<bool, FactorCount> mask) {
    for (bool& b : mask) b = !b;
    return mask;
  }

  template <std::array<bool, FactorCount> Mask>
  static consteval auto maskedFactors() {
    std::array<std::size_t, std::size_t(std::ranges::count(Mask, true))> result{};
    std::size_t out{0};
    for (std::size_t k{0}; k < FactorCount; ++k) {
      if (Mask[k]) result[out++] = k;
    }
    return result;
  }

  template <typename Factor>
  static void factorizeOwner(const Factor& factor) {
    if constexpr (!Density<Factor>) {
      [[maybe_unused]] const bool factorized{factor.factorize().has_value()};
      EIGEN_FILTER_ASSERT(factorized && "a factor failed to factorize; call its factorize() first to handle this");
    }
  }

  template <typename Factor, typename Placed>
  [[nodiscard]] internal::IndexVector<Factor::SizeAtCompileTime, Factor::MaxSizeAtCompileTime> resolve(
      const Placed& placed) const {
    return internal::IndexVector<Factor::SizeAtCompileTime, Factor::MaxSizeAtCompileTime>{
        internal::resolveIndices<SizeAtCompileTime, MaxSizeAtCompileTime>(size(), placed.indicesIn(m_space))};
  }

  /** Calls \a fn(factor, indices) for each stored factor, owner or density. */
  template <typename Fn>
  void forEachStored(Fn&& fn) const {
    [&]<std::size_t... K>(std::index_sequence<K...>) {
      (fn(std::get<K>(m_factors), std::get<K>(m_indices)), ...);
    }(std::index_sequence_for<Factors...>{});
  }

  [[nodiscard]] bool sizesMatch() const {
    bool result{true};
    forEachStored(
        [&](const auto& factor, const auto& indices) { result = result && factor.space().size() == indices.size(); });
    return result;
  }

  [[nodiscard]] bool partitionsSpace() const {
    Index total{0};
    forEachStored([&](const auto&, const auto& indices) { total += indices.size(); });
    const bool distinct{
        std::apply([](const auto&... indices) { return internal::distinctIndices(indices...); }, m_indices)};
    return total == size() && distinct;
  }

  [[nodiscard]] bool spacesMatch() const {
    const std::vector<Coordinate> all{internal::coordinatesOf(m_space)};
    bool result{true};
    forEachStored([&](const auto& factor, const auto& indices) {
      const std::vector<Coordinate> covered{internal::coordinatesOf(factor.space())};
      for (Index k{0}; k < indices.size(); ++k) {
        result = result && covered[std::size_t(k)] == all[std::size_t(indices(k))];
      }
    });
    return result;
  }

  /** Asserts that factor \a K covers the block named by \a Path, in order, as a factor placed
   * by on<Path...>() does, so paths within it line up with paths within the product.
   */
  template <std::size_t K, typename... Path>
  void assertOnOwnBlock(internal::TypeList<Path...>) const {
    [[maybe_unused]] const auto block{internal::resolveIndices<SizeAtCompileTime, MaxSizeAtCompileTime>(
        size(), internal::asIndexSet(idxIn<Path...>(m_space)))};
    EIGEN_FILTER_ASSERT(
        std::equal(block.begin(), block.end(), std::get<K>(m_indices).begin(), std::get<K>(m_indices).end()) &&
        "a factor reached by a tag path must be placed on that tag's block");
  }

  /** \returns the coordinates covered by the factors in \a mask */
  [[nodiscard]] std::vector<Index> coveredBy(const std::array<bool, FactorCount>& mask) const {
    std::vector<Index> result;
    std::size_t k{0};
    forEachStored([&](const auto&, const auto& indices) {
      if (mask[k++]) result.insert(result.end(), indices.begin(), indices.end());
    });
    return result;
  }

  /** \returns the marginal of \a factor on \a Path, which lies within it, as an owner or value */
  template <typename Factor, typename... Path>
  [[nodiscard]] static auto ownedMarginal(const Factor& factor, internal::TypeList<Path...>) {
    static_assert(requires { factor.template marginal<Path...>(); }, "this factor has no marginal<Path...>()");
    auto result{factor.template marginal<Path...>()};
    if constexpr (requires { result.copy(); }) {
      return result.copy();
    } else {
      return result;
    }
  }

  /** \returns the factors in \a Keep, on their coordinates' positions within \a target, which
   * form \a space: their product, or the one factor alone
   */
  template <std::array<bool, FactorCount> Keep, typename Target, typename SubSpace>
  [[nodiscard]] auto keepOnly(const Target& target, SubSpace space) const {
    constexpr auto kept{maskedFactors<Keep>()};
    if constexpr (kept.size() == 1) {
      return std::get<kept[0]>(m_factors);
    } else {
      return [&]<std::size_t... J>(std::index_sequence<J...>) {
        return ProductDensity<Scalar, SubSpace, FactorType<kept[J]>...>{
            std::move(space), on(positionsOf<kept[J]>(target), std::get<kept[J]>(m_factors))...};
      }(std::make_index_sequence<kept.size()>{});
    }
  }

  /** \returns the positions within \a target of factor \a K's coordinates */
  template <std::size_t K, typename Target>
  [[nodiscard]] auto positionsOf(const Target& target) const {
    using Factor = FactorType<K>;
    return internal::positionsIn<Factor::SizeAtCompileTime, Factor::MaxSizeAtCompileTime>(std::get<K>(m_indices),
                                                                                          target, size());
  }

  /** \returns this product with factor \a K replaced by its conditional on the relative path
   * \a Path equal to \a values, in \a space, the space of the coordinates \a target
   */
  template <std::size_t K, typename Target, typename Given, typename SubSpace, typename... Path, typename Derived>
  [[nodiscard]] auto replaceFactor(const Target& target, const Given& given, SubSpace space,
                                   internal::TypeList<Path...>, const Eigen::MatrixBase<Derived>& values) const {
    const auto& factor{std::get<K>(m_factors)};
    static_assert(
        requires { factor.template conditional<Path...>(values); }, "this factor has no conditional<Path...>()");
    if constexpr (FactorCount == 1) {
      return factor.template conditional<Path...>(values);
    } else {
      using Raw = decltype(factor.template conditional<Path...>(values));
      using Replacement = typename internal::ConditionalValue<Raw>::type;
      using Error = typename internal::ConditionalError<Raw>::type;
      static_assert(ProductFactor<Replacement>, "this factor's conditional cannot be a factor of a product");
      using Result = typename internal::ReplacedProduct<Scalar, SubSpace, K, Replacement, std::tuple<Factors...>>::type;

      const auto build{[&](Replacement replacement) {
        // The replacement covers factor K's coordinates that were not given, in order.
        std::vector<Index> remaining;
        for (const Index i : std::get<K>(m_indices)) {
          if (std::find(given.begin(), given.end(), i) == given.end()) remaining.push_back(i);
        }
        auto positions{internal::positionsIn<Replacement::SizeAtCompileTime, Replacement::MaxSizeAtCompileTime>(
            remaining, target, size())};
        return [&]<std::size_t... J>(std::index_sequence<J...>) {
          return Result{std::move(space), placeFactor<J, K>(target, positions, replacement)...};
        }(std::index_sequence_for<Factors...>{});
      }};

      auto raw{factor.template conditional<Path...>(values)};
      if constexpr (std::is_void_v<Error>) {
        return build(std::move(raw));
      } else {
        using Return = std::expected<Result, Error>;
        if constexpr (internal::isExpected<Raw>) {
          if (!raw) return Return{std::unexpect, raw.error()};
          return Return{build(std::move(*raw))};
        } else {
          if (auto factorized{raw.factorize()}; !factorized) return Return{std::unexpect, factorized.error()};
          return Return{build(std::move(raw))};
        }
      }
    }
  }

  /** \returns factor \a J placed on its positions within \a target, or for \a J equal to \a K,
   * \a replacement placed on \a positions
   */
  template <std::size_t J, std::size_t K, typename Target, typename Positions, typename Replacement>
  [[nodiscard]] auto placeFactor(const Target& target, const Positions& positions, Replacement& replacement) const {
    if constexpr (J == K) {
      return on(positions, std::move(replacement));
    } else {
      return on(positionsOf<J>(target), std::get<J>(m_factors));
    }
  }

  EIGEN_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;
  std::tuple<internal::IndexVector<Factors::SizeAtCompileTime, Factors::MaxSizeAtCompileTime>...> m_indices;
  std::tuple<Factors...> m_factors;
};

template <typename Space, typename First, typename... Rest>
ProductDensity(Space, First, Rest...) -> ProductDensity<typename First::FactorType::Scalar, Space,
                                                        typename First::FactorType, typename Rest::FactorType...>;

/** \returns \a n independent draws from \a product, one per column: each factor's draws in
 * its coordinates, factor by factor.
 *
 * \sa drawSamplesWithLogPdf(const ProductDensity&, Eigen::Index, G&)
 */
template <typename Scalar, typename Space, typename... Factors, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Scalar, Space> drawSamples(const ProductDensity<Scalar, Space, Factors...>& product,
                                                   Eigen::Index n, G& rng) {
  EIGEN_FILTER_ASSERT(n >= 0);
  SampleSet<Scalar, Space> samples{product.size(), n};
  product.forEachFactor([&](const auto& density, const auto& indices) {
    samples(indices, Eigen::placeholders::all) = drawSamples(density, n, rng);
  });
  return samples;
}

/** \returns \a n independent draws from \a product, one per column, with the log-density at
 * each: the sum of the factors' log-densities at their draws.
 *
 * \sa drawSamples(const ProductDensity&, Eigen::Index, G&)
 */
template <typename Scalar, typename Space, typename... Factors, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Space> drawSamplesWithLogPdf(
    const ProductDensity<Scalar, Space, Factors...>& product, Eigen::Index n, G& rng) {
  EIGEN_FILTER_ASSERT(n >= 0);
  SamplesWithLogPdf<Scalar, Space> result{SampleSet<Scalar, Space>{product.size(), n},
                                          Eigen::Array<Scalar, Eigen::Dynamic, 1>::Zero(n)};
  product.forEachFactor([&](const auto& density, const auto& indices) {
    auto draws{drawSamplesWithLogPdf(density, n, rng)};
    result.samples(indices, Eigen::placeholders::all) = draws.samples;
    result.logPdf += draws.logPdf;
  });
  return result;
}

}  // namespace EigenFilter
