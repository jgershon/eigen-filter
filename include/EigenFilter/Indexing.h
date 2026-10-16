#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>

#include "Macros.h"

namespace EigenFilter::internal {

/** Holds a borrow on a mutable object: while any Borrow is alive, the object
 * may change values but not size or structure, and must not be moved or destroyed.
 * \a Source must befriend Borrow and have a \c mutable \c int \c m_borrowCount.
 */
template <typename Source>
class Borrow {
 public:
  explicit Borrow(const Source& source) : m_source{&source} { ++m_source->m_borrowCount; }
  Borrow(const Borrow& other) : m_source{other.m_source} { ++m_source->m_borrowCount; }

  Borrow& operator=(const Borrow& other) {
    ++other.m_source->m_borrowCount;
    --m_source->m_borrowCount;
    m_source = other.m_source;
    return *this;
  }

  ~Borrow() { --m_source->m_borrowCount; }

  [[nodiscard]] const Source& get() const { return *m_source; }

 private:
  const Source* m_source;
};

/** Holds a copy of a read-only view, and with it the view's lock. */
template <typename View>
class ViewHolder {
 public:
  explicit ViewHolder(const View& view) : m_view{view} {}

  [[nodiscard]] const View& get() const { return m_view; }

 private:
  View m_view;
};

/** A Borrow for mutable sources, a ViewHolder for read-only views. */
template <typename Source>
using HolderFor = std::conditional_t<Source::IsMutable, Borrow<Source>, ViewHolder<Source>>;

/** A plain Eigen matrix with the storage order Eigen requires for single-row matrices. */
template <typename Scalar, int Rows, int Cols, int MaxRows, int MaxCols>
using MatrixOf = Eigen::Matrix<Scalar, Rows, Cols, (MaxRows == 1 && MaxCols != 1 ? Eigen::RowMajor : Eigen::ColMajor),
                               MaxRows, MaxCols>;

template <int Size, int MaxSize>
using IndexVector = Eigen::Matrix<Eigen::Index, Size, 1, Eigen::ColMajor, MaxSize, 1>;

/** Compile-time size of the subset that \a Indices selects from a vector of size \a SourceSize. */
template <typename Indices, int SourceSize, int SourceMaxSize>
struct IndexedSize {
 private:
  using Probe = Eigen::Matrix<int, SourceSize, 1, Eigen::ColMajor, SourceMaxSize, 1>;
  using Expr = std::remove_cvref_t<decltype(std::declval<const Probe&>()(std::declval<const Indices&>()))>;
  static_assert(Expr::ColsAtCompileTime == 1, "marginal and conditional indices must select a set, not one index");

 public:
  static constexpr int size{Expr::RowsAtCompileTime};
  static constexpr int maxSize{size != Eigen::Dynamic                         ? size
                               : Expr::MaxRowsAtCompileTime != Eigen::Dynamic ? int(Expr::MaxRowsAtCompileTime)
                                                                              : SourceMaxSize};
  using type = IndexVector<size, maxSize>;
};

/** \returns the concrete indices that \a indices selects from \c 0..n-1. Any index
 * type Eigen accepts works, including symbolic ones such as \c lastN(2). Eigen
 * asserts that every index is in range.
 */
template <int SourceSize, int SourceMaxSize, typename Indices>
[[nodiscard]] typename IndexedSize<Indices, SourceSize, SourceMaxSize>::type resolveIndices(Eigen::Index n,
                                                                                            const Indices& indices) {
  using Identity = Eigen::Matrix<Eigen::Index, SourceSize, 1, Eigen::ColMajor, SourceMaxSize, 1>;
  const Identity identity{Identity::LinSpaced(n, 0, n - 1)};
  return identity(indices);
}

/** \returns whether the concatenation of \a indices contains no repeats. */
template <typename... Vectors>
[[nodiscard]] bool distinctIndices(const Vectors&... indices) {
  std::vector<Eigen::Index> all;
  (all.insert(all.end(), indices.begin(), indices.end()), ...);
  std::ranges::sort(all);
  return std::ranges::adjacent_find(all) == all.end();
}

/** \returns the indices in \c 0..n-1 not in \a given, ascending. Fixed-size when
 * both \a SourceSize and the size of \a given are known at compile time.
 */
template <int SourceSize, int SourceMaxSize, int GivenSize, int GivenMaxSize>
[[nodiscard]] auto complementIndices(Eigen::Index n, const IndexVector<GivenSize, GivenMaxSize>& given) {
  std::vector<bool> taken(static_cast<std::size_t>(n), false);
  for (const Eigen::Index i : given) taken[std::size_t(i)] = true;
  if constexpr (SourceSize != Eigen::Dynamic && GivenSize != Eigen::Dynamic) {
    std::array<Eigen::Index, std::size_t(SourceSize - GivenSize)> result{};
    std::size_t out{0};
    for (Eigen::Index i{0}; i < n; ++i) {
      if (!taken[std::size_t(i)]) result[out++] = i;
    }
    return result;
  } else {
    std::vector<Eigen::Index> result;
    for (Eigen::Index i{0}; i < n; ++i) {
      if (!taken[std::size_t(i)]) result.push_back(i);
    }
    return result;
  }
}

/** \returns a lazy expression for the block of the symmetric matrix whose lower
 * triangle is \a matrix, at rows \a rows and columns \a cols. Only lower-triangle
 * entries of \a matrix are read, whatever order the indices are in. The expression
 * refers to its arguments and must not outlive them.
 */
template <int Rows, int MaxRows, int Cols, int MaxCols, typename Derived>
[[nodiscard]] auto lowerSymmetricBlock(const Eigen::MatrixBase<Derived>& matrix, const IndexVector<Rows, MaxRows>& rows,
                                       const IndexVector<Cols, MaxCols>& cols) {
  using Scalar = typename Derived::Scalar;
  using Result = MatrixOf<Scalar, Rows, Cols, MaxRows, MaxCols>;
  return Result::NullaryExpr(rows.size(), cols.size(),
                             [&matrix = matrix.derived(), &rows, &cols](Eigen::Index i, Eigen::Index j) {
                               const Eigen::Index r{rows(i)};
                               const Eigen::Index c{cols(j)};
                               return r >= c ? matrix(r, c) : matrix(c, r);
                             });
}

/** Moments of a Gaussian conditional, and the log-likelihood of the conditioning values. */
template <typename Scalar, int Size, int MaxSize>
struct ConditionedMoments {
  Eigen::Matrix<Scalar, Size, 1, Eigen::ColMajor, MaxSize, 1> mean;
  MatrixOf<Scalar, Size, Size, MaxSize, MaxSize> cov;
  Scalar logLikelihood;
};

/** Conditions the Gaussian with mean \a mean and (lower-triangle) covariance \a cov
 * on the entries \a given equal to \a values, for the entries \a target.
 *
 * Solves against \f$ \Sigma_{bb} \f$ with a pivoted LDLT, which always produces a
 * result; if the source is numerically singular, the conditional covariance may not
 * be numerically positive definite.
 */
template <typename MeanDerived, typename CovDerived, typename ValuesDerived, int TargetSize, int TargetMaxSize,
          int GivenSize, int GivenMaxSize>
[[nodiscard]] ConditionedMoments<typename MeanDerived::Scalar, TargetSize, TargetMaxSize> conditionMoments(
    const Eigen::MatrixBase<MeanDerived>& mean, const Eigen::MatrixBase<CovDerived>& cov,
    const IndexVector<TargetSize, TargetMaxSize>& target, const IndexVector<GivenSize, GivenMaxSize>& given,
    const Eigen::MatrixBase<ValuesDerived>& values) {
  using Scalar = typename MeanDerived::Scalar;
  using GivenVector = Eigen::Matrix<Scalar, GivenSize, 1, Eigen::ColMajor, GivenMaxSize, 1>;
  using GivenMatrix = MatrixOf<Scalar, GivenSize, GivenSize, GivenMaxSize, GivenMaxSize>;
  using CrossMatrix = MatrixOf<Scalar, TargetSize, GivenSize, TargetMaxSize, GivenMaxSize>;
  EIGEN_FILTER_ASSERT(values.size() == given.size());

  const GivenVector residual{values - mean.derived()(given)};
  const Eigen::LDLT<GivenMatrix> ldlt{GivenMatrix{lowerSymmetricBlock(cov, given, given)}};
  const CrossMatrix cross{lowerSymmetricBlock(cov, target, given)};
  const CrossMatrix gain{ldlt.solve(cross.transpose()).transpose()};

  const Scalar log2Pi{Eigen::numext::log(Scalar(2) * Scalar(EIGEN_PI))};
  const Scalar logDeterminant{ldlt.vectorD().array().log().sum()};
  const Scalar wsse{residual.dot(ldlt.solve(residual))};
  return {mean.derived()(target) + gain * residual, lowerSymmetricBlock(cov, target, target) - gain * cross.transpose(),
          Scalar(-0.5) * (Scalar(given.size()) * log2Pi + logDeterminant + wsse)};
}

}  // namespace EigenFilter::internal
