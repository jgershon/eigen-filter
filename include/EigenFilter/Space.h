#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "Macros.h"

namespace EigenFilter {

/** \defgroup Spaces Spaces
 *
 * A space describes how the coordinates of a vector behave: whether they wrap,
 * fold at a bound, or are plain linear values, and how many there are. Every
 * distribution type, and Moments and SampleSet, is parameterized by a scalar and a
 * space, and takes its size and storage bound from the space; Linear costs nothing.
 * Each also has an Eigen-style alias taking sizes instead, such as
 * <tt>GaussianN<double, 3></tt> for <tt>Gaussian<double, Linear<3>></tt>.
 *
 * A space provides two operations, applied to blocks whose rows are its coordinates
 * and whose columns are samples:
 *  - normalizeValues() brings stored values into canonical form, so applying an
 *    update \f$ x \boxplus \delta \f$ is \c normalizeValues(x + δ);
 *  - normalizeDifferences() brings differences into canonical form, so a residual
 *    \f$ a \boxminus b \f$ is \c normalizeDifferences(a - b).
 *
 * Spaces compose: Composite concatenates them, and Repeated repeats a fixed-size block
 * a runtime number of times. Only the last block of a Composite may be resizable, so
 * every coordinate's offset is known at compile time. Named subspaces are tag types
 * derived from a space (<tt>struct Heading : Periodic<> { using Periodic::Periodic; };</tt>),
 * and idx<> and idxOf<>() turn them into Eigen index sets.
 *
 * Every space exposes \c FixedSize (the size of its fixed part), \c Stride (the size of
 * the block its resizable tail repeats, or 0 if it is fully fixed), \c MaxSize (the
 * largest valid size, or \c Eigen::Dynamic if unbounded), and \c IsLinear. Like Eigen's
 * matrices, the coordinate types take a size and a maximum size, and Repeated takes a
 * maximum number of repetitions. Distributions use the space's \c MaxSize as their
 * storage bound, so a capped space gets fixed-capacity storage.
 */

/** How one coordinate of a CoordinateSpace behaves. */
enum class CoordinateKind : unsigned char {
  Linear,    ///< plain values
  Periodic,  ///< wraps with period \c b into <tt>[a, a + b)</tt>
  Bounded,   ///< reflects at the bounds <tt>[a, b]</tt> (either may be infinite)
};

/** One coordinate of a CoordinateSpace. */
struct Coordinate {
  CoordinateKind kind{CoordinateKind::Linear};
  double a{0};  ///< Periodic: lowest value. Bounded: lower bound.
  double b{0};  ///< Periodic: period. Bounded: upper bound.

  friend bool operator==(const Coordinate&, const Coordinate&) = default;
};

namespace internal {

template <typename Scalar>
[[nodiscard]] Scalar wrapValue(Scalar x, double lowest, double period) {
  using std::floor;
  const Scalar p{Scalar(period)};
  return x - p * floor((x - Scalar(lowest)) / p);
}

template <typename Scalar>
[[nodiscard]] Scalar wrapDifference(Scalar difference, double period) {
  using std::floor;
  const Scalar p{Scalar(period)};
  return difference - p * floor(difference / p + Scalar(0.5));
}

template <typename Scalar>
[[nodiscard]] Scalar foldValue(Scalar x, double lower, double upper) {
  const bool hasLower{std::isfinite(lower)};
  const bool hasUpper{std::isfinite(upper)};
  if (hasLower && hasUpper) {
    // Reflecting at both bounds is a triangle wave with period 2 (upper - lower).
    const Scalar width{Scalar(upper - lower)};
    const Scalar y{wrapValue(x, lower, 2 * (upper - lower)) - Scalar(lower)};
    return Scalar(lower) + (y > width ? Scalar(2) * width - y : y);
  }
  if (hasLower && x < Scalar(lower)) return Scalar(2 * lower) - x;
  if (hasUpper && x > Scalar(upper)) return Scalar(2 * upper) - x;
  return x;
}

template <typename Derived>
[[nodiscard]] Eigen::MatrixBase<Derived>& writable(const Eigen::MatrixBase<Derived>& block) {
  // Eigen's idiom for functions that write through an expression passed as const&.
  return const_cast<Eigen::MatrixBase<Derived>&>(block);
}

template <typename Derived, typename F>
void applyToCoefficients(const Eigen::MatrixBase<Derived>& block, F f) {
  auto& m{writable(block)};
  for (Eigen::Index j{0}; j < m.cols(); ++j) {
    for (Eigen::Index i{0}; i < m.rows(); ++i) m.coeffRef(i, j) = f(m.coeff(i, j));
  }
}

}  // namespace internal

/** \ingroup Spaces
 * \brief \a Size_ plain coordinates, or a resizable run of at most \a MaxSize_ of them if
 * \a Size_ is \c Eigen::Dynamic.
 *
 * The default space everywhere: an empty type whose operations do nothing.
 */
template <int Size_ = 1, int MaxSize_ = Size_>
struct Linear {
  static_assert(Size_ == Eigen::Dynamic || Size_ >= 0);
  static_assert(Size_ == Eigen::Dynamic || MaxSize_ == Size_, "a fixed size is its own maximum");
  static constexpr int FixedSize{Size_ == Eigen::Dynamic ? 0 : Size_};
  static constexpr int Stride{Size_ == Eigen::Dynamic ? 1 : 0};
  static constexpr int MaxSize{MaxSize_};
  static constexpr bool IsLinear{true};

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>&) const {}
  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>&) const {}

  void appendCoordinates(Eigen::Index n, std::vector<Coordinate>& out) const {
    out.insert(out.end(), std::size_t(n), Coordinate{});
  }

  friend bool operator==(const Linear&, const Linear&) = default;
};

/** \ingroup Spaces
 * \brief \a Size_ coordinates that wrap with a common period, such as angles or DFT bins.
 *
 * Values are kept in <tt>[lowest, lowest + period)</tt> and differences in
 * <tt>[-period/2, period/2)</tt>. The default is an angle in <tt>[-π, π)</tt>.
 */
template <int Size_ = 1, int MaxSize_ = Size_>
class Periodic {
 public:
  static_assert(Size_ == Eigen::Dynamic || Size_ >= 1);
  static_assert(Size_ == Eigen::Dynamic || MaxSize_ == Size_, "a fixed size is its own maximum");
  static constexpr int FixedSize{Size_ == Eigen::Dynamic ? 0 : Size_};
  static constexpr int Stride{Size_ == Eigen::Dynamic ? 1 : 0};
  static constexpr int MaxSize{MaxSize_};
  static constexpr bool IsLinear{false};

  /** An angle: period 2π, values in <tt>[-π, π)</tt>. */
  Periodic() = default;

  /** Period \a period, values in <tt>[-period/2, period/2)</tt>. */
  explicit Periodic(double period) : Periodic{period, -period / 2} {}

  /** Period \a period, values in <tt>[lowest, lowest + period)</tt>. */
  Periodic(double period, double lowest) : m_period{period}, m_lowest{lowest} {
    EIGEN_FILTER_ASSERT(period > 0 && "the period must be positive");
  }

  [[nodiscard]] double period() const { return m_period; }
  [[nodiscard]] double lowest() const { return m_lowest; }

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>& block) const {
    internal::applyToCoefficients(block, [this](auto x) { return internal::wrapValue(x, m_lowest, m_period); });
  }

  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>& block) const {
    internal::applyToCoefficients(block, [this](auto d) { return internal::wrapDifference(d, m_period); });
  }

  void appendCoordinates(Eigen::Index n, std::vector<Coordinate>& out) const {
    out.insert(out.end(), std::size_t(n), Coordinate{CoordinateKind::Periodic, m_lowest, m_period});
  }

  friend bool operator==(const Periodic&, const Periodic&) = default;

 private:
  double m_period{2 * std::numbers::pi};
  double m_lowest{-std::numbers::pi};
};

/** \ingroup Spaces
 * \brief \a Size_ coordinates that reflect at a lower bound, an upper bound, or both.
 *
 * An update that crosses a bound is folded back: below a lower bound \f$ a \f$, a value
 * \f$ x \f$ becomes \f$ 2a - x \f$. For a range this is usually the physical answer, since
 * a range is a norm. Differences are plain. The default is <tt>[0, ∞)</tt>.
 */
template <int Size_ = 1, int MaxSize_ = Size_>
class Bounded {
 public:
  static_assert(Size_ == Eigen::Dynamic || Size_ >= 1);
  static_assert(Size_ == Eigen::Dynamic || MaxSize_ == Size_, "a fixed size is its own maximum");
  static constexpr int FixedSize{Size_ == Eigen::Dynamic ? 0 : Size_};
  static constexpr int Stride{Size_ == Eigen::Dynamic ? 1 : 0};
  static constexpr int MaxSize{MaxSize_};
  static constexpr bool IsLinear{false};

  /** Nonnegative values: <tt>[0, ∞)</tt>. */
  Bounded() = default;

  /** Values in <tt>[lower, upper]</tt>; either bound may be infinite. */
  explicit Bounded(double lower, double upper = std::numeric_limits<double>::infinity())
      : m_lower{lower}, m_upper{upper} {
    EIGEN_FILTER_ASSERT(lower < upper && "the lower bound must be below the upper bound");
  }

  [[nodiscard]] double lower() const { return m_lower; }
  [[nodiscard]] double upper() const { return m_upper; }

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>& block) const {
    internal::applyToCoefficients(block, [this](auto x) { return internal::foldValue(x, m_lower, m_upper); });
  }

  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>&) const {}

  void appendCoordinates(Eigen::Index n, std::vector<Coordinate>& out) const {
    out.insert(out.end(), std::size_t(n), Coordinate{CoordinateKind::Bounded, m_lower, m_upper});
  }

  friend bool operator==(const Bounded&, const Bounded&) = default;

 private:
  double m_lower{0};
  double m_upper{std::numeric_limits<double>::infinity()};
};

/** \ingroup Spaces
 * \brief The concatenation of \a Blocks. Only the last block may be resizable.
 *
 * Nesting a named composite names a range of coordinates; storage is always the flat
 * concatenation.
 */
template <typename... Blocks>
class Composite {
  static_assert(sizeof...(Blocks) > 0, "a composite needs at least one block");
  static constexpr std::size_t BlockCount{sizeof...(Blocks)};
  static constexpr std::array<int, BlockCount> Sizes{Blocks::FixedSize...};
  static constexpr std::array<int, BlockCount> Strides{Blocks::Stride...};
  static constexpr std::array<int, BlockCount> MaxSizes{Blocks::MaxSize...};

  static constexpr bool onlyLastIsResizable() {
    for (std::size_t i{0}; i + 1 < BlockCount; ++i) {
      if (Strides[i] != 0) return false;
    }
    return true;
  }
  static_assert(onlyLastIsResizable(), "only the last block of a composite may be resizable");

  static constexpr std::array<int, BlockCount> offsets() {
    std::array<int, BlockCount> result{};
    for (std::size_t i{1}; i < BlockCount; ++i) result[i] = result[i - 1] + Sizes[i - 1];
    return result;
  }

 public:
  static constexpr int FixedSize{(Blocks::FixedSize + ...)};
  static constexpr int Stride{Strides[BlockCount - 1]};
  static constexpr int MaxSize{MaxSizes[BlockCount - 1] == Eigen::Dynamic
                                   ? Eigen::Dynamic
                                   : FixedSize - Sizes[BlockCount - 1] + MaxSizes[BlockCount - 1]};
  static constexpr bool IsLinear{(Blocks::IsLinear && ...)};
  /** Offset of each block's first coordinate. */
  static constexpr std::array<int, BlockCount> Offsets{offsets()};

  Composite() = default;
  explicit Composite(Blocks... blocks) : m_blocks{std::move(blocks)...} {}

  /** \returns block \a I */
  template <std::size_t I>
  [[nodiscard]] const auto& block() const {
    return std::get<I>(m_blocks);
  }

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>& rows) const {
    forEachBlock(rows, [](const auto& block, auto&& sub) { block.normalizeValues(sub); });
  }

  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>& rows) const {
    forEachBlock(rows, [](const auto& block, auto&& sub) { block.normalizeDifferences(sub); });
  }

  void appendCoordinates(Eigen::Index n, std::vector<Coordinate>& out) const {
    appendEach(n, out, std::make_index_sequence<BlockCount>{});
  }

  friend bool operator==(const Composite&, const Composite&) = default;

 private:
  template <typename Derived, typename F>
  void forEachBlock(const Eigen::MatrixBase<Derived>& rows, F f) const {
    forEachBlockImpl(internal::writable(rows), f, std::make_index_sequence<BlockCount>{});
  }

  template <typename Derived, typename F, std::size_t... I>
  void forEachBlockImpl(Eigen::MatrixBase<Derived>& rows, F f, std::index_sequence<I...>) const {
    (f(std::get<I>(m_blocks), rows.middleRows(Offsets[I], blockRows<I>(rows.rows()))), ...);
  }

  template <std::size_t I>
  [[nodiscard]] static Eigen::Index blockRows(Eigen::Index total) {
    return I + 1 < BlockCount ? Eigen::Index{Sizes[I]} : total - Offsets[I];
  }

  template <std::size_t... I>
  void appendEach(Eigen::Index n, std::vector<Coordinate>& out, std::index_sequence<I...>) const {
    (std::get<I>(m_blocks).appendCoordinates(blockRows<I>(n), out), ...);
  }

  std::tuple<Blocks...> m_blocks;
};

/** \ingroup Spaces
 * \brief A fixed-size \a Block repeated any number of times, up to \a MaxCount, such as one
 * block per vehicle.
 *
 * Its size is always a multiple of the block's size. Every repetition shares the
 * block's parameters (periods and bounds).
 */
template <typename Block, int MaxCount_ = Eigen::Dynamic>
class Repeated {
  static_assert(Block::Stride == 0 && Block::FixedSize > 0, "a repeated block must have a fixed, nonzero size");
  static_assert(MaxCount_ == Eigen::Dynamic || MaxCount_ >= 0);

 public:
  using BlockType = Block;
  /** Maximum number of repetitions, or \c Eigen::Dynamic if unbounded. */
  static constexpr int MaxCount{MaxCount_};
  static constexpr int FixedSize{0};
  static constexpr int Stride{Block::FixedSize};
  static constexpr int MaxSize{MaxCount == Eigen::Dynamic ? Eigen::Dynamic : MaxCount * Block::FixedSize};
  static constexpr bool IsLinear{Block::IsLinear};

  Repeated() = default;
  explicit Repeated(Block block) : m_block{std::move(block)} {}

  [[nodiscard]] const Block& block() const { return m_block; }

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>& rows) const {
    auto& m{internal::writable(rows)};
    for (Eigen::Index k{0}; k + Stride <= m.rows(); k += Stride) m_block.normalizeValues(m.middleRows(k, Stride));
  }

  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>& rows) const {
    auto& m{internal::writable(rows)};
    for (Eigen::Index k{0}; k + Stride <= m.rows(); k += Stride) m_block.normalizeDifferences(m.middleRows(k, Stride));
  }

  void appendCoordinates(Eigen::Index n, std::vector<Coordinate>& out) const {
    for (Eigen::Index k{0}; k + Stride <= n; k += Stride) m_block.appendCoordinates(Stride, out);
  }

  friend bool operator==(const Repeated&, const Repeated&) = default;

 private:
  Block m_block;
};

/** \ingroup Spaces
 * \brief A space described coordinate by coordinate at runtime.
 *
 * The space of a marginal or conditional taken by arbitrary indices from a non-linear
 * space. Resizing appends Linear coordinates.
 */
template <int Size_ = Eigen::Dynamic, int MaxSize_ = Size_>
class CoordinateSpace {
 public:
  static constexpr int FixedSize{Size_ == Eigen::Dynamic ? 0 : Size_};
  static constexpr int Stride{Size_ == Eigen::Dynamic ? 1 : 0};
  static constexpr int MaxSize{MaxSize_};
  static constexpr bool IsLinear{false};

  CoordinateSpace() : m_coordinates(std::size_t(FixedSize)) {}

  explicit CoordinateSpace(std::vector<Coordinate> coordinates) : m_coordinates{std::move(coordinates)} {
    EIGEN_FILTER_ASSERT((Size_ == Eigen::Dynamic || m_coordinates.size() == std::size_t(Size_)) &&
                        "wrong number of coordinates for a fixed-size space");
  }

  [[nodiscard]] Eigen::Index size() const { return Eigen::Index(m_coordinates.size()); }
  [[nodiscard]] const Coordinate& coordinate(Eigen::Index i) const { return m_coordinates[std::size_t(i)]; }

  /** Resizes to \a n coordinates; new ones are Linear. */
  void resize(Eigen::Index n) { m_coordinates.resize(std::size_t(n)); }

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>& rows) const {
    auto& m{internal::writable(rows)};
    EIGEN_FILTER_ASSERT(m.rows() == size());
    for (Eigen::Index i{0}; i < m.rows(); ++i) {
      const Coordinate& c{coordinate(i)};
      if (c.kind == CoordinateKind::Periodic) {
        internal::applyToCoefficients(m.row(i), [&c](auto x) { return internal::wrapValue(x, c.a, c.b); });
      } else if (c.kind == CoordinateKind::Bounded) {
        internal::applyToCoefficients(m.row(i), [&c](auto x) { return internal::foldValue(x, c.a, c.b); });
      }
    }
  }

  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>& rows) const {
    auto& m{internal::writable(rows)};
    EIGEN_FILTER_ASSERT(m.rows() == size());
    for (Eigen::Index i{0}; i < m.rows(); ++i) {
      const Coordinate& c{coordinate(i)};
      if (c.kind == CoordinateKind::Periodic) {
        internal::applyToCoefficients(m.row(i), [&c](auto d) { return internal::wrapDifference(d, c.b); });
      }
    }
  }

  void appendCoordinates(Eigen::Index n, std::vector<Coordinate>& out) const {
    EIGEN_FILTER_ASSERT(n == size());
    out.insert(out.end(), m_coordinates.begin(), m_coordinates.end());
  }

  friend bool operator==(const CoordinateSpace&, const CoordinateSpace&) = default;

 private:
  std::vector<Coordinate> m_coordinates;
};

/** \returns whether a vector of size \a n fits \a Space: at least its fixed size, a
 * whole number of repetitions beyond it, and at most its maximum size.
 */
template <typename Space>
[[nodiscard]] constexpr bool isValidSize(Eigen::Index n) {
  if constexpr (Space::Stride == 0) {
    return n == Space::FixedSize;
  } else {
    return n >= Space::FixedSize && (n - Space::FixedSize) % Space::Stride == 0 &&
           (Space::MaxSize == Eigen::Dynamic || n <= Space::MaxSize);
  }
}

/** Compile-time size of \a Space, or \c Eigen::Dynamic if it is resizable. */
template <typename Space>
inline constexpr int spaceSize{Space::Stride == 0 ? Space::FixedSize : Eigen::Dynamic};

/** Largest valid size of \a Space, or \c Eigen::Dynamic if it is unbounded. */
template <typename Space>
inline constexpr int spaceMaxSize{Space::Stride == 0 ? Space::FixedSize : Space::MaxSize};

namespace internal {

template <typename... Blocks>
std::tuple<Blocks...>* compositeBlocks(const Composite<Blocks...>*);
template <typename T>
concept IsComposite = requires(const T* t) { compositeBlocks(t); };
template <typename T>
using CompositeBlocks = std::remove_pointer_t<decltype(compositeBlocks(std::declval<const T*>()))>;

template <typename Block, int MaxCount>
Block* repeatedBlock(const Repeated<Block, MaxCount>*);
template <typename T>
concept IsRepeated = requires(const T* t) { repeatedBlock(t); };
template <typename T>
using RepeatedBlock = std::remove_pointer_t<decltype(repeatedBlock(std::declval<const T*>()))>;

/** Number of times \a Tag appears in \a Space, not looking inside repeated blocks. */
template <typename Tag, typename Space>
consteval int countTag();

template <typename Tag, typename... Blocks>
consteval int countTagIn(std::tuple<Blocks...>*) {
  return (countTag<Tag, Blocks>() + ... + 0);
}

template <typename Tag, typename Space>
consteval int countTag() {
  if constexpr (std::is_same_v<Tag, Space>) {
    return 1;
  } else if constexpr (IsComposite<Space>) {
    return countTagIn<Tag>(static_cast<CompositeBlocks<Space>*>(nullptr));
  } else {
    return 0;
  }
}

template <typename Tag, typename Space>
consteval int tagOffset();

template <typename Tag, typename... Blocks>
consteval int tagOffsetIn(std::tuple<Blocks...>*) {
  int offset{0};
  int result{-1};
  ((result < 0 && countTag<Tag, Blocks>() > 0 ? (result = offset + tagOffset<Tag, Blocks>()) : 0,
    offset += Blocks::FixedSize),
   ...);
  return result;
}

/** Offset of the unique \a Tag within \a Space. */
template <typename Tag, typename Space>
consteval int tagOffset() {
  if constexpr (std::is_same_v<Tag, Space>) {
    return 0;
  } else if constexpr (IsComposite<Space>) {
    return tagOffsetIn<Tag>(static_cast<CompositeBlocks<Space>*>(nullptr));
  } else {
    return -1;  // not here; only consulted when countTag() says the tag is present
  }
}

/** Where a path of tags resolves to. */
struct ResolvedIndex {
  int offset{0};          ///< first coordinate
  int size{0};            ///< fixed size of the resolved block (0 if resizable)
  bool resizable{false};  ///< the block is the space's resizable tail
  int stride{0};          ///< nonzero: one coordinate of every repetition of a repeated block
};

template <typename Space, typename Tag, typename... Rest>
consteval ResolvedIndex resolveTags() {
  static_assert(countTag<Tag, Space>() > 0, "the space does not contain this tag");
  static_assert(countTag<Tag, Space>() < 2, "the tag appears more than once; qualify it with an enclosing tag");
  constexpr int offset{tagOffset<Tag, Space>()};
  if constexpr (sizeof...(Rest) == 0) {
    return {offset, Tag::FixedSize, Tag::Stride != 0, 0};
  } else if constexpr (IsRepeated<Tag>) {
    constexpr ResolvedIndex inner{resolveTags<RepeatedBlock<Tag>, Rest...>()};
    static_assert(inner.size == 1, "a path through a repeated block must end at a single coordinate");
    return {offset + inner.offset, 1, false, Tag::Stride};
  } else {
    constexpr ResolvedIndex inner{resolveTags<Tag, Rest...>()};
    return {offset + inner.offset, inner.size, inner.resizable, inner.stride};
  }
}

template <ResolvedIndex R>
constexpr auto makeIndex() {
  if constexpr (R.stride != 0) {
    return Eigen::seq(Eigen::fix<R.offset>, Eigen::placeholders::last, Eigen::fix<R.stride>);
  } else if constexpr (R.resizable) {
    return Eigen::seq(Eigen::fix<R.offset>, Eigen::placeholders::last);
  } else if constexpr (R.size == 1) {
    return Eigen::Index{R.offset};
  } else {
    return Eigen::seqN(Eigen::fix<R.offset>, Eigen::fix<R.size>);
  }
}

}  // namespace internal

/** \ingroup Spaces
 * Index of the subspace named by the tag path \a Path within \a Space, usable directly
 * in Eigen indexing and in marginal() and conditional():
 *  - a single fixed coordinate: an \c Eigen::Index (a constant expression);
 *  - a fixed block: \c seqN(fix<offset>, fix<size>);
 *  - the resizable tail: \c seq(fix<offset>, last);
 *  - a path through a repeated block to one of its coordinates, such as
 *    <tt>idx<World, Vehicles, Altitude></tt>: that coordinate of every repetition,
 *    \c seq(fix<offset>, last, fix<stride>).
 *
 * A tag that appears more than once must be qualified by an enclosing tag.
 *
 * \sa idxOf()
 */
template <typename Space, typename... Path>
inline const auto idx{internal::makeIndex<internal::resolveTags<Space, Path...>()>()};

/** \ingroup Spaces
 * \returns the index of repetition \a element of the repeated block \a RepeatedTag, or of
 * the coordinates \a Inner within it: an \c Eigen::Index for one coordinate, otherwise
 * \c seqN(start, fix<size>).
 */
template <typename Space, typename RepeatedTag, typename... Inner>
[[nodiscard]] auto idxOf(Eigen::Index element) {
  static_assert(internal::IsRepeated<RepeatedTag>, "idxOf() takes a repeated block");
  constexpr internal::ResolvedIndex outer{internal::resolveTags<Space, RepeatedTag>()};
  constexpr internal::ResolvedIndex inner{[] {
    if constexpr (sizeof...(Inner) == 0) {
      return internal::ResolvedIndex{0, RepeatedTag::Stride, false, 0};
    } else {
      return internal::resolveTags<internal::RepeatedBlock<RepeatedTag>, Inner...>();
    }
  }()};
  const Eigen::Index start{outer.offset + element * RepeatedTag::Stride + inner.offset};
  if constexpr (inner.size == 1) {
    return start;
  } else {
    return Eigen::seqN(start, Eigen::fix<inner.size>);
  }
}

namespace internal {

template <typename Tag, typename Space>
[[nodiscard]] const Tag& findTag(const Space& space);

template <typename Tag, std::size_t I, typename Space>
void findTagAt(const Space& space, const Tag*& found) {
  using Block = std::tuple_element_t<I, CompositeBlocks<Space>>;
  if constexpr (countTag<Tag, Block>() > 0) found = &findTag<Tag>(space.template block<I>());
}

template <typename Tag, typename Space, std::size_t... I>
[[nodiscard]] const Tag& findTagIn(const Space& space, std::index_sequence<I...>) {
  const Tag* found{nullptr};
  (findTagAt<Tag, I>(space, found), ...);
  return *found;
}

/** \returns the unique block of type \a Tag within \a space, with its parameters. */
template <typename Tag, typename Space>
[[nodiscard]] const Tag& findTag(const Space& space) {
  if constexpr (std::is_same_v<Tag, Space>) {
    return space;
  } else {
    static_assert(countTag<Tag, Space>() == 1, "the tag must appear exactly once");
    return findTagIn<Tag>(space, std::make_index_sequence<std::tuple_size_v<CompositeBlocks<Space>>>{});
  }
}

/** \returns the space of the subspace named by \a Tag and \a Rest within \a space: the
 * tag's own block, or for a path through a repeated block, the selected coordinate
 * repeated up to the same maximum count.
 */
template <typename Space, typename Tag, typename... Rest>
[[nodiscard]] auto subspace(const Space& space) {
  const Tag& tag{findTag<Tag>(space)};
  if constexpr (sizeof...(Rest) == 0) {
    return tag;
  } else if constexpr (IsRepeated<Tag>) {
    auto inner{subspace<RepeatedBlock<Tag>, Rest...>(tag.block())};
    return Repeated<decltype(inner), Tag::MaxCount>{std::move(inner)};
  } else {
    return subspace<Tag, Rest...>(tag);
  }
}

/** An index set for marginal() and conditional(): a single index becomes a one-element set. */
template <typename Indices>
[[nodiscard]] auto asIndexSet(const Indices& indices) {
  if constexpr (std::is_integral_v<Indices>) {
    return std::array<Eigen::Index, 1>{indices};
  } else {
    return indices;
  }
}

/** Space of the coordinates \a indices selected from \a Space: Linear if it is linear,
 * otherwise a CoordinateSpace.
 */
template <typename Space, int SubSize, int SubMaxSize = SubSize>
using ProjectedSpace =
    std::conditional_t<Space::IsLinear, Linear<SubSize, SubMaxSize>, CoordinateSpace<SubSize, SubMaxSize>>;

template <int SubSize, int SubMaxSize, typename Space, typename IndexVector>
[[nodiscard]] ProjectedSpace<Space, SubSize, SubMaxSize> projectSpace(const Space& space, Eigen::Index n,
                                                                      const IndexVector& indices) {
  if constexpr (Space::IsLinear) {
    return {};
  } else {
    std::vector<Coordinate> all;
    all.reserve(std::size_t(n));
    space.appendCoordinates(n, all);
    std::vector<Coordinate> picked;
    picked.reserve(std::size_t(indices.size()));
    for (const Eigen::Index i : indices) picked.push_back(all[std::size_t(i)]);
    return ProjectedSpace<Space, SubSize, SubMaxSize>{std::move(picked)};
  }
}

}  // namespace internal

}  // namespace EigenFilter
