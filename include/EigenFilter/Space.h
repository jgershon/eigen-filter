#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
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
 * a runtime number of times. Named subspaces are tag types derived from a space
 * (<tt>struct Heading : Periodic<> { using Periodic::Periodic; };</tt>).
 *
 * A space knows its own size(). A fixed-size space is an empty type whose size is a
 * compile-time constant. A resizable space (a Repeated block, a coordinate type with
 * \c Eigen::Dynamic size, or a Composite ending in one) stores its size. Only the last
 * block of a Composite may be resizable. Distributions keep their space's size in step
 * with their data, and grow or shrink their resizable block in place with insert() and
 * erase().
 *
 * idxIn<Path...>(space) turns a tag path into an Eigen index set from the space's sizes.
 * idx<> gives the same index as a compile-time constant.
 *
 * Every space exposes \c FixedSize (the size of its fixed part), \c Stride (nonzero if it
 * is resizable: the size of one element of its resizable part), \c MaxSize (the largest
 * valid size, or \c Eigen::Dynamic if unbounded), and \c IsLinear. Like Eigen's matrices,
 * the coordinate types take a size and a maximum size, and Repeated takes a maximum number
 * of repetitions. Distributions use the space's \c MaxSize as their storage bound, so a
 * capped space gets fixed-capacity storage.
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

/** The size of a coordinate type: a compile-time constant when fixed, which takes no storage. */
template <int Size, int MaxSize>
class LeafExtent {
 public:
  [[nodiscard]] static constexpr Eigen::Index size() { return Size; }

  friend bool operator==(const LeafExtent&, const LeafExtent&) = default;
};

/** The size of a resizable coordinate type: stored, and at most \a MaxSize. */
template <int MaxSize>
class LeafExtent<Eigen::Dynamic, MaxSize> {
 public:
  [[nodiscard]] Eigen::Index size() const { return m_size; }

  /** Resizes to \a n coordinates. */
  void resize(Eigen::Index n) {
    EIGEN_FILTER_ASSERT(n >= 0 && (MaxSize == Eigen::Dynamic || n <= MaxSize) && "the size does not fit the space");
    m_size = n;
  }

  friend bool operator==(const LeafExtent&, const LeafExtent&) = default;

 private:
  Eigen::Index m_size{0};
};

}  // namespace internal

/** \ingroup Spaces
 * \brief \a Size_ plain coordinates, or a resizable run of at most \a MaxSize_ of them if
 * \a Size_ is \c Eigen::Dynamic.
 *
 * The default space everywhere: when fixed, an empty type whose operations do nothing.
 */
template <int Size_ = 1, int MaxSize_ = Size_>
struct Linear : internal::LeafExtent<Size_, MaxSize_> {
  static_assert(Size_ == Eigen::Dynamic || Size_ >= 0);
  static_assert(Size_ == Eigen::Dynamic || MaxSize_ == Size_, "a fixed size is its own maximum");
  static constexpr int FixedSize{Size_ == Eigen::Dynamic ? 0 : Size_};
  static constexpr int Stride{Size_ == Eigen::Dynamic ? 1 : 0};
  static constexpr int MaxSize{MaxSize_};
  static constexpr bool IsLinear{true};

  Linear() = default;

  /** \a n coordinates of a resizable Linear. */
  explicit Linear(Eigen::Index n)
    requires(Size_ == Eigen::Dynamic)
  {
    this->resize(n);
  }

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>&) const {}
  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>&) const {}

  void appendCoordinates(std::vector<Coordinate>& out) const {
    out.insert(out.end(), std::size_t(this->size()), Coordinate{});
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
class Periodic : public internal::LeafExtent<Size_, MaxSize_> {
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

  void appendCoordinates(std::vector<Coordinate>& out) const {
    out.insert(out.end(), std::size_t(this->size()), Coordinate{CoordinateKind::Periodic, m_lowest, m_period});
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
class Bounded : public internal::LeafExtent<Size_, MaxSize_> {
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

  void appendCoordinates(std::vector<Coordinate>& out) const {
    out.insert(out.end(), std::size_t(this->size()), Coordinate{CoordinateKind::Bounded, m_lower, m_upper});
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
 * concatenation. Each block's offset is the total size of the blocks before it.
 */
template <typename... Blocks>
class Composite {
  static_assert(sizeof...(Blocks) > 0, "a composite needs at least one block");
  static constexpr std::size_t BlockCount{sizeof...(Blocks)};
  static constexpr std::array<bool, BlockCount> Resizable{(Blocks::Stride != 0)...};
  static constexpr std::size_t ResizableCount{(std::size_t{Blocks::Stride != 0} + ...)};

  static constexpr int strides() {
    int result{0};
    ((result = std::gcd(result, Blocks::Stride)), ...);
    return result;
  }

  static constexpr std::size_t firstResizable() {
    for (std::size_t i{0}; i < BlockCount; ++i) {
      if (Resizable[i]) return i;
    }
    return BlockCount;
  }
  static_assert(ResizableCount == 0 || firstResizable() == BlockCount - 1,
                "only the last block of a composite may be resizable");

 public:
  static constexpr int FixedSize{(Blocks::FixedSize + ...)};
  /** Zero if fixed; otherwise the stride of its resizable block. */
  static constexpr int Stride{strides()};
  static constexpr int MaxSize{((Blocks::MaxSize == Eigen::Dynamic) || ...) ? Eigen::Dynamic : (Blocks::MaxSize + ...)};
  static constexpr bool IsLinear{(Blocks::IsLinear && ...)};

  Composite() = default;
  explicit Composite(Blocks... blocks) : m_blocks{std::move(blocks)...} {}

  /** \returns block \a I */
  template <std::size_t I>
  [[nodiscard]] const auto& block() const {
    return std::get<I>(m_blocks);
  }

  /** \returns block \a I, to resize it */
  template <std::size_t I>
  [[nodiscard]] auto& block() {
    return std::get<I>(m_blocks);
  }

  [[nodiscard]] Eigen::Index size() const {
    return std::apply([](const auto&... blocks) { return (Eigen::Index{0} + ... + blocks.size()); }, m_blocks);
  }

  /** \returns the offset of block \a I's first coordinate */
  template <std::size_t I>
  [[nodiscard]] Eigen::Index offset() const {
    return [&]<std::size_t... J>(std::index_sequence<J...>) {
      return (Eigen::Index{0} + ... + block<J>().size());
    }(std::make_index_sequence<I>{});
  }

  /** Resizes the composite's only resizable block so the composite has \a n coordinates. */
  void resize(Eigen::Index n)
    requires(ResizableCount == 1)
  {
    auto& resizable{block<firstResizable()>()};
    const Eigen::Index rest{size() - resizable.size()};
    EIGEN_FILTER_ASSERT(n >= rest && "the size does not fit the space");
    resizable.resize(n - rest);
  }

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>& rows) const {
    forEachBlock(rows, [](const auto& block, auto&& sub) { block.normalizeValues(sub); });
  }

  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>& rows) const {
    forEachBlock(rows, [](const auto& block, auto&& sub) { block.normalizeDifferences(sub); });
  }

  void appendCoordinates(std::vector<Coordinate>& out) const {
    std::apply([&out](const auto&... blocks) { (blocks.appendCoordinates(out), ...); }, m_blocks);
  }

  friend bool operator==(const Composite&, const Composite&) = default;

 private:
  template <typename Derived, typename F>
  void forEachBlock(const Eigen::MatrixBase<Derived>& rows, F f) const {
    auto& m{internal::writable(rows)};
    EIGEN_FILTER_ASSERT(m.rows() == size() && "the rows do not match the space");
    Eigen::Index offset{0};
    std::apply(
        [&](const auto&... blocks) {
          ((f(blocks, m.middleRows(offset, blocks.size())), offset += blocks.size()), ...);
        },
        m_blocks);
  }

  std::tuple<Blocks...> m_blocks;
};

/** \ingroup Spaces
 * \brief A fixed-size \a Block repeated count() times, up to \a MaxCount_, such as one
 * block per vehicle.
 *
 * Its size is count() times the block's size. Every repetition shares the block's
 * parameters (periods and bounds). A tag derived from it can inherit its constructors to
 * start with a count: <tt>struct Vehicles : Repeated<Vehicle> { using Repeated::Repeated; };</tt>,
 * then <tt>Vehicles{Vehicle{}, 3}</tt>.
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

  /** \a count repetitions of \a block. */
  explicit Repeated(Block block, Eigen::Index count = 0) : m_block{std::move(block)} { setCount(count); }

  [[nodiscard]] const Block& block() const { return m_block; }

  /** \returns the number of repetitions */
  [[nodiscard]] Eigen::Index count() const { return m_count; }

  [[nodiscard]] Eigen::Index size() const { return m_count * Stride; }

  void setCount(Eigen::Index count) {
    EIGEN_FILTER_ASSERT(count >= 0 && (MaxCount == Eigen::Dynamic || count <= MaxCount) &&
                        "the size does not fit the space");
    m_count = count;
  }

  /** Resizes to \a n coordinates, a whole number of repetitions. */
  void resize(Eigen::Index n) {
    EIGEN_FILTER_ASSERT(n >= 0 && n % Stride == 0 && "the size does not fit the space");
    setCount(n / Stride);
  }

  template <typename Derived>
  void normalizeValues(const Eigen::MatrixBase<Derived>& rows) const {
    forEachRepetition(rows, [this](auto&& sub) { m_block.normalizeValues(sub); });
  }

  template <typename Derived>
  void normalizeDifferences(const Eigen::MatrixBase<Derived>& rows) const {
    forEachRepetition(rows, [this](auto&& sub) { m_block.normalizeDifferences(sub); });
  }

  void appendCoordinates(std::vector<Coordinate>& out) const {
    for (Eigen::Index k{0}; k < m_count; ++k) m_block.appendCoordinates(out);
  }

  friend bool operator==(const Repeated&, const Repeated&) = default;

 private:
  template <typename Derived, typename F>
  void forEachRepetition(const Eigen::MatrixBase<Derived>& rows, F f) const {
    auto& m{internal::writable(rows)};
    EIGEN_FILTER_ASSERT(m.rows() == size() && "the rows do not match the space");
    for (Eigen::Index k{0}; k < m_count; ++k) f(m.middleRows(k * Stride, Stride));
  }

  Block m_block;
  Eigen::Index m_count{0};
};

/** \ingroup Spaces
 * \brief A space described coordinate by coordinate at runtime.
 *
 * The space of a marginal or conditional taken by arbitrary indices from a non-linear
 * space. Resizing a resizable one appends Linear coordinates.
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
  void resize(Eigen::Index n)
    requires(Size_ == Eigen::Dynamic)
  {
    EIGEN_FILTER_ASSERT(n >= 0 && (MaxSize_ == Eigen::Dynamic || n <= MaxSize_) && "the size does not fit the space");
    m_coordinates.resize(std::size_t(n));
  }

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

  void appendCoordinates(std::vector<Coordinate>& out) const {
    out.insert(out.end(), m_coordinates.begin(), m_coordinates.end());
  }

  friend bool operator==(const CoordinateSpace&, const CoordinateSpace&) = default;

 private:
  std::vector<Coordinate> m_coordinates;
};

/** \returns whether some instance of \a Space has size \a n: at least its fixed size, a
 * whole number of strides beyond it, and at most its maximum size. An instance's own size
 * is its size().
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

template <int Size, int MaxSize>
std::true_type leafExtent(const LeafExtent<Size, MaxSize>*);
std::false_type leafExtent(const void*);
/** A Linear, Periodic, or Bounded space (or a tag derived from one): every coordinate alike. */
template <typename T>
concept IsUniformLeaf = decltype(leafExtent(std::declval<const T*>()))::value;

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

/** Where a tag sits in a space, as far as is known at compile time. */
struct TagLocation {
  int offset{-1};            ///< offset, counting fixed sizes only
  bool staticOffset{false};  ///< no resizable block precedes the tag, so \c offset is exact
  bool tail{false};          ///< the tag's block ends the space
};

template <typename Tag, typename Space>
consteval TagLocation locateTag();

template <typename Tag, typename... Blocks>
consteval TagLocation locateTagIn(std::tuple<Blocks...>*) {
  constexpr std::size_t count{sizeof...(Blocks)};
  constexpr std::array<int, count> counts{countTag<Tag, Blocks>()...};
  constexpr std::array<TagLocation, count> inner{locateTag<Tag, Blocks>()...};
  constexpr std::array<int, count> sizes{Blocks::FixedSize...};
  constexpr std::array<bool, count> fixed{(Blocks::Stride == 0)...};
  int offset{0};
  bool staticOffset{true};
  for (std::size_t i{0}; i < count; ++i) {
    if (counts[i] > 0) {
      return {offset + inner[i].offset, staticOffset && inner[i].staticOffset, inner[i].tail && i + 1 == count};
    }
    offset += sizes[i];
    staticOffset = staticOffset && fixed[i];
  }
  return {};
}

/** Location of the first \a Tag within \a Space; only consulted when countTag() finds it. */
template <typename Tag, typename Space>
consteval TagLocation locateTag() {
  if constexpr (std::is_same_v<Tag, Space>) {
    return {0, true, true};
  } else if constexpr (IsComposite<Space>) {
    return locateTagIn<Tag>(static_cast<CompositeBlocks<Space>*>(nullptr));
  } else {
    return {};
  }
}

/** Where a path of tags resolves to, as far as is known at compile time. */
struct ResolvedIndex {
  int offset{0};            ///< first coordinate, if \c staticOffset
  int size{0};              ///< fixed size of the resolved block (0 if resizable)
  bool resizable{false};    ///< the resolved block is resizable
  int stride{0};            ///< nonzero: one coordinate of every repetition of a repeated block
  bool staticOffset{true};  ///< no resizable block precedes it
  bool tail{true};          ///< it ends the space
};

template <typename Space, typename Tag, typename... Rest>
consteval ResolvedIndex resolveTags() {
  static_assert(countTag<Tag, Space>() > 0, "the space does not contain this tag");
  static_assert(countTag<Tag, Space>() < 2, "the tag appears more than once; qualify it with an enclosing tag");
  constexpr TagLocation location{locateTag<Tag, Space>()};
  if constexpr (sizeof...(Rest) == 0) {
    return {location.offset, Tag::FixedSize, Tag::Stride != 0, 0, location.staticOffset, location.tail};
  } else if constexpr (IsRepeated<Tag>) {
    constexpr ResolvedIndex inner{resolveTags<RepeatedBlock<Tag>, Rest...>()};
    static_assert(inner.size == 1, "a path through a repeated block must end at a single coordinate");
    return {location.offset + inner.offset, 1, false, Tag::Stride, location.staticOffset, location.tail};
  } else {
    constexpr ResolvedIndex inner{resolveTags<Tag, Rest...>()};
    return {location.offset + inner.offset,
            inner.size,
            inner.resizable,
            inner.stride,
            location.staticOffset && inner.staticOffset,
            location.tail && inner.tail};
  }
}

template <ResolvedIndex R>
constexpr auto makeIndex() {
  static_assert(R.staticOffset, "this path's offset depends on the size of a resizable block; use idxIn(space)");
  if constexpr (R.stride != 0) {
    static_assert(R.tail, "this repeated block is not at the end of the space; use idxIn(space)");
    return Eigen::seq(Eigen::fix<R.offset>, Eigen::placeholders::last, Eigen::fix<R.stride>);
  } else if constexpr (R.resizable) {
    static_assert(R.tail, "this resizable block is not at the end of the space; use idxIn(space)");
    return Eigen::seq(Eigen::fix<R.offset>, Eigen::placeholders::last);
  } else if constexpr (R.size == 1) {
    return Eigen::Index{R.offset};
  } else {
    return Eigen::seqN(Eigen::fix<R.offset>, Eigen::fix<R.size>);
  }
}

/** \returns the offset of the unique block of type \a Tag within \a space, and a pointer to
 * that block (const if \a Space is).
 */
template <typename Tag, typename Space>
[[nodiscard]] auto locate(Space& space) {
  using Plain = std::remove_const_t<Space>;
  using TagPointer = std::conditional_t<std::is_const_v<Space>, const Tag*, Tag*>;
  if constexpr (std::is_same_v<Tag, Plain>) {
    return std::pair<Eigen::Index, TagPointer>{0, &space};
  } else {
    static_assert(countTag<Tag, Plain>() > 0, "the space does not contain this tag");
    static_assert(countTag<Tag, Plain>() < 2, "the tag appears more than once; qualify it with an enclosing tag");
    std::pair<Eigen::Index, TagPointer> found{0, nullptr};
    Eigen::Index offset{0};
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (
          [&] {
            using Block = std::tuple_element_t<I, CompositeBlocks<Plain>>;
            auto& block{space.template block<I>()};
            if constexpr (countTag<Tag, Block>() > 0) {
              const auto inner{locate<Tag>(block)};
              found = {offset + inner.first, inner.second};
            }
            offset += block.size();
          }(),
          ...);
    }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<Plain>>>{});
    return found;
  }
}

/** locate() along a path of tags, none of them inside a repeated block. */
template <typename Tag, typename... Rest, typename Space>
[[nodiscard]] auto locatePath(Space& space) {
  const auto [offset, tag]{locate<Tag>(space)};
  if constexpr (sizeof...(Rest) == 0) {
    return std::pair{offset, tag};
  } else {
    static_assert(!IsRepeated<Tag>, "this path continues inside a repeated block");
    const auto [innerOffset, inner]{locatePath<Rest...>(*tag)};
    return std::pair{offset + innerOffset, inner};
  }
}

/** \returns \a index shifted by \a offset, keeping any compile-time size and increment. */
template <typename Index>
[[nodiscard]] auto shiftIndex(const Index& index, Eigen::Index offset) {
  if constexpr (std::is_integral_v<Index>) {
    return Eigen::Index{index + offset};
  } else {
    return Eigen::seqN(index.first() + offset, index.sizeObject(), index.incrObject());
  }
}

}  // namespace internal

/** \ingroup Spaces
 * Index of the subspace named by the tag path \a Path within \a Space, as a compile-time
 * constant usable directly in Eigen indexing:
 *  - a single fixed coordinate: an \c Eigen::Index (a constant expression);
 *  - a fixed block: \c seqN(fix<offset>, fix<size>);
 *  - a resizable block at the end of the space: \c seq(fix<offset>, last);
 *  - a path through a repeated block at the end of the space to one of its coordinates,
 *    such as <tt>idx<World, Vehicles, Altitude></tt>: that coordinate of every repetition,
 *    \c seq(fix<offset>, last, fix<stride>).
 *
 * Only available when no resizable block precedes the path's block, and a resizable or
 * repeated block ends the space; otherwise use idxIn(), which works for every path. A tag
 * that appears more than once must be qualified by an enclosing tag.
 *
 * \sa idxIn(), idxOf()
 */
template <typename Space, typename... Path>
inline const auto idx{internal::makeIndex<internal::resolveTags<Space, Path...>()>()};

/** \ingroup Spaces
 * \returns the index of the subspace named by the tag path \a Path within \a space, from its
 * runtime sizes: an \c Eigen::Index for a single fixed coordinate; \c seqN(offset, fix<size>)
 * for a fixed block; \c seqN(offset, size) for a resizable one; and for a path through a
 * repeated block to one of its coordinates, \c seqN(offset, count, fix<stride>).
 *
 * \sa idx<>
 */
template <typename Tag, typename... Rest, typename Space>
[[nodiscard]] auto idxIn(const Space& space) {
  const auto [offset, tag]{internal::locate<Tag>(space)};
  if constexpr (sizeof...(Rest) == 0) {
    if constexpr (Tag::Stride != 0) {
      return Eigen::seqN(offset, tag->size());
    } else if constexpr (Tag::FixedSize == 1) {
      return Eigen::Index{offset};
    } else {
      return Eigen::seqN(offset, Eigen::fix<Tag::FixedSize>);
    }
  } else if constexpr (internal::IsRepeated<Tag>) {
    constexpr internal::ResolvedIndex inner{internal::resolveTags<internal::RepeatedBlock<Tag>, Rest...>()};
    static_assert(inner.size == 1, "a path through a repeated block must end at a single coordinate");
    return Eigen::seqN(offset + inner.offset, tag->count(), Eigen::fix<Tag::Stride>);
  } else {
    return internal::shiftIndex(idxIn<Rest...>(*tag), offset);
  }
}

namespace internal {

/** Index within one repetition of \a RepeatedTag of the coordinates \a Inner (all of it if none). */
template <typename RepeatedTag, typename... Inner>
consteval ResolvedIndex repetitionIndex() {
  static_assert(IsRepeated<RepeatedTag>, "idxOf() takes a repeated block");
  if constexpr (sizeof...(Inner) == 0) {
    return ResolvedIndex{0, RepeatedTag::Stride, false, 0};
  } else {
    return resolveTags<RepeatedBlock<RepeatedTag>, Inner...>();
  }
}

template <ResolvedIndex Inner>
[[nodiscard]] auto repetitionIndexAt(Eigen::Index start) {
  if constexpr (Inner.size == 1) {
    return start;
  } else {
    return Eigen::seqN(start, Eigen::fix<Inner.size>);
  }
}

}  // namespace internal

/** \ingroup Spaces
 * \returns the index of repetition \a element of the repeated block \a RepeatedTag, or of
 * the coordinates \a Inner within it: an \c Eigen::Index for one coordinate, otherwise
 * \c seqN(start, fix<size>). Only available when no resizable block precedes \a RepeatedTag;
 * otherwise use idxOf(space, element).
 */
template <typename Space, typename RepeatedTag, typename... Inner>
[[nodiscard]] auto idxOf(Eigen::Index element) {
  constexpr internal::ResolvedIndex outer{internal::resolveTags<Space, RepeatedTag>()};
  static_assert(outer.staticOffset, "this block's offset depends on the size of a resizable block; use idxOf(space)");
  constexpr internal::ResolvedIndex inner{internal::repetitionIndex<RepeatedTag, Inner...>()};
  return internal::repetitionIndexAt<inner>(outer.offset + element * RepeatedTag::Stride + inner.offset);
}

/** \ingroup Spaces
 * idxOf() for any space, from its runtime sizes. \a element must be one of its repetitions.
 */
template <typename RepeatedTag, typename... Inner, typename Space>
[[nodiscard]] auto idxOf(const Space& space, Eigen::Index element) {
  const auto [offset, tag]{internal::locate<RepeatedTag>(space)};
  EIGEN_FILTER_ASSERT(element >= 0 && element < tag->count() && "repetition out of range");
  constexpr internal::ResolvedIndex inner{internal::repetitionIndex<RepeatedTag, Inner...>()};
  return internal::repetitionIndexAt<inner>(offset + element * RepeatedTag::Stride + inner.offset);
}

namespace internal {

/** \returns the unique block of type \a Tag within \a space, with its parameters and size. */
template <typename Tag, typename Space>
[[nodiscard]] const Tag& findTag(const Space& space) {
  return *locate<Tag>(space).second;
}

/** \returns the space of the subspace named by \a Tag and \a Rest within \a space: the
 * tag's own block, or for a path through a repeated block, the selected coordinate
 * repeated as many times, up to the same maximum count.
 */
template <typename Space, typename Tag, typename... Rest>
[[nodiscard]] auto subspace(const Space& space) {
  const Tag& tag{findTag<Tag>(space)};
  if constexpr (sizeof...(Rest) == 0) {
    return tag;
  } else if constexpr (IsRepeated<Tag>) {
    auto inner{subspace<RepeatedBlock<Tag>, Rest...>(tag.block())};
    return Repeated<decltype(inner), Tag::MaxCount>{std::move(inner), tag.count()};
  } else {
    return subspace<Tag, Rest...>(tag);
  }
}

/** Whether \a Given names exactly one top-level block of the Composite \a Space, so that the
 * rest of \a Space is whole blocks too.
 */
template <typename Space, typename... GivenPath>
inline constexpr bool isTopLevelBlock{false};

template <typename Space, typename Given>
  requires IsComposite<Space>
inline constexpr bool isTopLevelBlock<Space, Given>{[]<typename... Blocks>(std::tuple<Blocks...>*) {
  return (std::is_same_v<Given, Blocks> + ...) == 1 && countTag<Given, Space>() == 1;
}(static_cast<CompositeBlocks<Space>*>(nullptr))};

template <typename Given, std::size_t I, typename Space>
[[nodiscard]] auto blockUnlessGiven(const Space& space) {
  using Block = std::tuple_element_t<I, CompositeBlocks<Space>>;
  if constexpr (std::is_same_v<Block, Given>) {
    return std::tuple<>{};
  } else {
    return std::tuple<Block>{space.template block<I>()};
  }
}

/** \returns the blocks of the Composite \a space other than the top-level block \a Given, in
 * order: that block itself when one remains, otherwise their Composite. Its coordinates are
 * the complement of \a Given's, in ascending order.
 */
template <typename Given, typename Space>
  requires isTopLevelBlock<Space, Given>
[[nodiscard]] auto complementSubspace(const Space& space) {
  constexpr std::size_t count{std::tuple_size_v<CompositeBlocks<Space>>};
  auto rest{[&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::tuple_cat(blockUnlessGiven<Given, I>(space)...);
  }(std::make_index_sequence<count>{})};
  if constexpr (std::tuple_size_v<decltype(rest)> == 1) {
    return std::get<0>(std::move(rest));
  } else {
    return std::apply(
        [](auto&&... blocks) { return Composite<std::remove_cvref_t<decltype(blocks)>...>{std::move(blocks)...}; },
        std::move(rest));
  }
}

/** A block that insert() and erase() can grow and shrink element by element: a Repeated
 * block (elements are repetitions) or a resizable Linear, Periodic, or Bounded block
 * (elements are coordinates).
 */
template <typename Block>
concept IsElementwise = IsRepeated<Block> || (IsUniformLeaf<Block> && Block::Stride != 0);

template <IsElementwise Block>
[[nodiscard]] Eigen::Index elementCount(const Block& block) {
  if constexpr (IsRepeated<Block>) {
    return block.count();
  } else {
    return block.size();
  }
}

template <IsElementwise Block>
void setElementCount(Block& block, Eigen::Index count) {
  if constexpr (IsRepeated<Block>) {
    block.setCount(count);
  } else {
    block.resize(count);
  }
}

/** Rows of a block of elements in the full space. */
struct ElementRows {
  Eigen::Index start;  ///< first row
  Eigen::Index count;  ///< number of rows
};

/** Grows the elementwise block named by \a Path by \a rows rows, inserted before its element
 * \a element. \returns the rows they occupy.
 */
template <typename... Path, typename Space>
[[nodiscard]] ElementRows growBlock(Space& space, Eigen::Index element, Eigen::Index rows) {
  auto [offset, block]{locatePath<Path...>(space)};
  using Block = std::remove_pointer_t<decltype(block)>;
  static_assert(IsElementwise<Block>,
                "insert() and erase() take a Repeated block or a resizable Linear, Periodic, or Bounded block");
  constexpr Eigen::Index elementSize{Block::Stride};
  const Eigen::Index count{elementCount(*block)};
  EIGEN_FILTER_ASSERT(element >= 0 && element <= count && "insertion point out of range");
  EIGEN_FILTER_ASSERT(rows % elementSize == 0 && "insert whole elements");
  setElementCount(*block, count + rows / elementSize);
  return {offset + element * elementSize, rows};
}

/** Shrinks the elementwise block named by \a Path by \a elements elements starting at its
 * element \a element. \returns the rows they occupied.
 */
template <typename... Path, typename Space>
[[nodiscard]] ElementRows shrinkBlock(Space& space, Eigen::Index element, Eigen::Index elements) {
  auto [offset, block]{locatePath<Path...>(space)};
  using Block = std::remove_pointer_t<decltype(block)>;
  static_assert(IsElementwise<Block>,
                "insert() and erase() take a Repeated block or a resizable Linear, Periodic, or Bounded block");
  constexpr Eigen::Index elementSize{Block::Stride};
  const Eigen::Index count{elementCount(*block)};
  EIGEN_FILTER_ASSERT(element >= 0 && elements >= 0 && element + elements <= count && "erased elements out of range");
  setElementCount(*block, count - elements);
  return {offset + element * elementSize, elements * elementSize};
}

/** \returns the coordinates of \a space, one per entry, in order. */
template <typename Space>
[[nodiscard]] std::vector<Coordinate> coordinatesOf(const Space& space) {
  std::vector<Coordinate> result;
  result.reserve(std::size_t(space.size()));
  space.appendCoordinates(result);
  return result;
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

/** \returns the space of the coordinates \a indices of \a space, which has \a n coordinates. */
template <int SubSize, int SubMaxSize, typename Space, typename IndexVector>
[[nodiscard]] ProjectedSpace<Space, SubSize, SubMaxSize> projectSpace(const Space& space, Eigen::Index n,
                                                                      const IndexVector& indices) {
  EIGEN_FILTER_ASSERT(space.size() == n && "the space has a different number of coordinates");
  if constexpr (Space::IsLinear) {
    ProjectedSpace<Space, SubSize, SubMaxSize> result;
    if constexpr (SubSize == Eigen::Dynamic) result.resize(Eigen::Index(indices.size()));
    return result;
  } else {
    std::vector<Coordinate> all;
    all.reserve(std::size_t(n));
    space.appendCoordinates(all);
    std::vector<Coordinate> picked;
    picked.reserve(std::size_t(indices.size()));
    for (const Eigen::Index i : indices) picked.push_back(all[std::size_t(i)]);
    return ProjectedSpace<Space, SubSize, SubMaxSize>{std::move(picked)};
  }
}

}  // namespace internal

}  // namespace EigenFilter
