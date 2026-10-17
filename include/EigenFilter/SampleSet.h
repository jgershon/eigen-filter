#pragma once

#include "Macros.h"
#include "Space.h"

namespace EigenFilter {

/** A matrix whose columns are samples of dimension \a Size.
 *
 * \tparam Scalar   scalar type
 * \tparam Size     sample dimension, or \c Eigen::Dynamic
 * \tparam MaxSize  maximum sample dimension
 *
 * Eigen requires a single-row matrix to be row-major, so one-dimensional sample
 * sets are stored row-major; all others are column-major.
 *
 * \sa SampleSet
 */
template <typename Scalar, int Size, int MaxSize = Size>
using SampleSetN = Eigen::Matrix<Scalar, Size, Eigen::Dynamic, (MaxSize == 1 ? Eigen::RowMajor : Eigen::ColMajor),
                                 MaxSize, Eigen::Dynamic>;

/** A matrix whose columns are samples in \a Space (see \ref Spaces): a SampleSetN with the
 * space's size and storage bound. It is a plain Eigen matrix, so the space is not part
 * of its type and the samples are not normalized by it.
 */
template <typename Scalar, typename Space>
using SampleSet = SampleSetN<Scalar, spaceSize<Space>, spaceMaxSize<Space>>;

}  // namespace EigenFilter
