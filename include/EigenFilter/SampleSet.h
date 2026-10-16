#pragma once

#include "Macros.h"

namespace EigenFilter {

/** A matrix whose columns are samples of dimension \a Size.
 *
 * \tparam Scalar   scalar type
 * \tparam Size     sample dimension, or \c Eigen::Dynamic
 * \tparam MaxSize  maximum sample dimension
 *
 * Eigen requires a single-row matrix to be row-major, so one-dimensional sample
 * sets are stored row-major; all others are column-major.
 */
template <typename Scalar, int Size, int MaxSize = Size>
using SampleSet = Eigen::Matrix<Scalar, Size, Eigen::Dynamic, (MaxSize == 1 ? Eigen::RowMajor : Eigen::ColMajor),
                                MaxSize, Eigen::Dynamic>;

}  // namespace EigenFilter
