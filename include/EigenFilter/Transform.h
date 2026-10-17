#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>
#include <unsupported/Eigen/AutoDiff>

#include "Gaussian.h"
#include "Indexing.h"
#include "Macros.h"
#include "SampleSet.h"
#include "Space.h"

namespace EigenFilter {

/** \defgroup Transforms Transforms
 *
 * transform() pushes independent uncertain inputs through a function
 * \f$ y = f(a_1, \ldots, a_k, \mathrm{known}\ldots) \f$ and returns the distribution of
 * \f$ y \f$ as a Gaussian. A process model maps a state to a later state; a measurement
 * model maps a state to an innovation, owning its subtraction and wrapping.
 *
 * The arguments after \a f (and the optional output space) reach \a f in order:
 *  - an uncertain input (see UncertainInput: a Gaussian, a GaussianDensity, or a marginal
 *    of one) reaches \a f as a vector. Uncertain inputs are independent of each other, so
 *    their joint space is the concatenation of their spaces;
 *  - anything else is a known input, such as a control input or a measurement, and reaches
 *    \a f unchanged.
 *
 * The result is a Gaussian in the output space. Marking one uncertain input with
 * withCross(), usually the state, also returns its cross-covariance with \a y (see
 * OutputWithCross), and update() then conditions that input on an observation of \a y: the
 * Kalman update. The other uncertain inputs are nuisance parameters, such as noise or an
 * uncertain sensor position, whose correlations are not tracked; anything worth tracking
 * belongs in the state.
 *
 * The output space is the argument after \a f if that is a space (see \ref Spaces), and
 * otherwise Linear, sized by \a f's return type.
 *
 * The method is Affine, Linearized, or Unscented. To be differentiated by Linearized, \a f
 * must be generic in its scalar type: take uncertain inputs as <tt>const auto&</tt> and
 * build outputs with ScalarOf. Known inputs may stay \c double.
 *
 * \code
 * const auto rangeBearing{[](const auto& state, const Eigen::Vector2d& z) {
 *   using T = EigenFilter::ScalarOf<decltype(state)>;
 *   using std::atan2;
 *   using std::sqrt;
 *   const Eigen::Matrix<T, 2, 1> r{sqrt(state.template head<2>().squaredNorm()), atan2(state(1), state(0))};
 *   return Eigen::Matrix<T, 2, 1>{z.cast<T>() - r};
 * }};
 * const auto predicted{EigenFilter::linearized(rangeBearing, Innovation{}, withCross(state), z)};
 * const auto [posterior, logLikelihood]{EigenFilter::update(state, predicted)};
 * \endcode
 */

/** \ingroup Transforms
 * The scalar type of the Eigen object \a X, for writing functions generic in it.
 */
template <typename X>
using ScalarOf = typename std::remove_cvref_t<X>::Scalar;

/** \ingroup Transforms
 * An argument of transform() that is uncertain: something with a mean, a covariance, and a
 * space, such as a Gaussian, a GaussianDensity, or a GaussianMarginal.
 */
template <typename T>
concept UncertainInput = requires(const T& input) {
  typename T::Scalar;
  typename T::SpaceType;
  input.mean();
  input.cov();
  input.space();
  input.size();
};

/** \ingroup Transforms
 * The result of transform() with an input marked by withCross(): the output's Gaussian, and
 * the cross-covariance of that input with the output. update() conditions the input on an
 * observation of the output.
 *
 * \tparam InputSpace   space of the marked input
 * \tparam OutputSpace  space of the output
 */
template <typename Scalar, typename InputSpace, typename OutputSpace>
struct OutputWithCross {
  using CrossMatrix = internal::MatrixOf<Scalar, spaceSize<InputSpace>, spaceSize<OutputSpace>,
                                         spaceMaxSize<InputSpace>, spaceMaxSize<OutputSpace>>;

  Gaussian<Scalar, OutputSpace> output;  ///< distribution of the output
  CrossMatrix cross;  ///< \f$ \Sigma_{xy} \f$: a row per input coordinate, a column per output coordinate
};

/** \ingroup Transforms
 * The result of update(): the posterior, and the log-likelihood of the observation.
 */
template <typename Scalar, typename Space>
struct Updated {
  Gaussian<Scalar, Space> posterior;
  Scalar logLikelihood;  ///< \f$ \log \mathcal{N}(y; \bar y, \Sigma_{yy}) \f$
};

/** \ingroup Transforms
 * Affine method: \a f must be affine. Its Jacobian is the output difference over a unit
 * step in each input coordinate, which is exact for an affine \a f, so \a f need not be
 * generic in its scalar type. Steps are not normalized; output differences are.
 */
struct Affine {};

/** \ingroup Transforms
 * Linearized's default: differentiate \a f with forward-mode automatic differentiation.
 */
struct AutoDiff {};

/** \ingroup Transforms
 * Linearized method: \f$ \bar y = f(\mu) \f$ and \f$ \Sigma_{yy} = \sum_i J_i \Sigma_i J_i^T \f$,
 * with the Jacobians \f$ J_i \f$ of \a f at the input means.
 *
 * By default \a f is differentiated automatically (AutoDiff). Otherwise \a jacobian is
 * called with the same arguments as \a f, at the means, and returns a \c std::tuple of
 * Jacobian blocks, one per uncertain input in order; with one uncertain input it may return
 * the matrix itself.
 */
template <typename Jacobian = AutoDiff>
struct Linearized {
  Jacobian jacobian{};
};

Linearized() -> Linearized<>;

template <typename Jacobian>
Linearized(Jacobian) -> Linearized<Jacobian>;

/** \ingroup Transforms
 * Unscented method: \f$ 2L + 1 \f$ sigma points over the joint input space of dimension
 * \f$ L \f$, spread by \f$ \lambda = \alpha^2 (L + \kappa) - L \f$.
 *
 * The defaults \f$ \alpha = 1, \beta = 2, \kappa = 0 \f$ give a zero center mean weight and
 * positive weights otherwise; \f$ \beta = 2 \f$ is optimal for Gaussian inputs. Sigma points are
 * normalized in their inputs' spaces before \a f sees them, and the output mean is a
 * tangent-space mean in the output space.
 */
struct Unscented {
  double alpha{1};
  double beta{2};
  double kappa{0};
};

namespace internal {

/** An uncertain input marked by withCross(). Valid until the end of the full expression. */
template <typename Input>
struct WithCross {
  const Input& input;
};

template <typename T>
concept IsSpace = requires {
  T::FixedSize;
  T::Stride;
  T::MaxSize;
  T::IsLinear;
};

/** How transform() treats an argument: known by default. */
template <typename Arg>
struct ArgTraits {
  using Input = Arg;
  using Scalar = void;
  static constexpr bool isUncertain{false};
  static constexpr bool hasCross{false};
  static constexpr int size{0};
  static constexpr int maxSize{0};
};

template <UncertainInput Arg>
struct ArgTraits<Arg> {
  using Input = Arg;
  using Scalar = typename Arg::Scalar;
  static constexpr bool isUncertain{true};
  static constexpr bool hasCross{false};
  static constexpr int size{Arg::SizeAtCompileTime};
  static constexpr int maxSize{Arg::MaxSizeAtCompileTime};
};

template <typename In>
struct ArgTraits<WithCross<In>> : ArgTraits<In> {
  static constexpr bool hasCross{true};
};

/** Number of arguments marked by withCross(). */
template <typename... Args>
inline constexpr std::size_t crossCount{(std::size_t{ArgTraits<Args>::hasCross} + ... + 0)};

/** \returns the input an argument carries: itself, or the input inside withCross(). */
template <typename Arg>
[[nodiscard]] const auto& argInput(const Arg& arg) {
  if constexpr (ArgTraits<Arg>::hasCross) {
    return arg.input;
  } else {
    return arg;
  }
}

template <typename First, typename... Rest>
struct FirstNonVoid {
  using type = std::conditional_t<std::is_void_v<First>, typename FirstNonVoid<Rest...>::type, First>;
};

template <typename First>
struct FirstNonVoid<First> {
  using type = First;
};

/** Compile-time facts about the arguments of transform(). */
template <typename... Args>
struct TransformArgs {
  static constexpr std::size_t count{sizeof...(Args)};
  static constexpr std::array<bool, count> uncertain{ArgTraits<Args>::isUncertain...};
  static constexpr std::size_t uncertainCount{(std::size_t{ArgTraits<Args>::isUncertain} + ... + 0)};
  static_assert(uncertainCount > 0, "transform() needs at least one uncertain input");
  static_assert(crossCount<Args...> <= 1, "withCross() marks at most one input");

  using Scalar = typename FirstNonVoid<typename ArgTraits<Args>::Scalar...>::type;
  static_assert(((std::is_void_v<typename ArgTraits<Args>::Scalar> ||
                  std::is_same_v<typename ArgTraits<Args>::Scalar, Scalar>) &&
                 ...),
                "every uncertain input must have the same scalar type");

  /** Dimension of the joint uncertain input, or \c Eigen::Dynamic. */
  static constexpr int size{((ArgTraits<Args>::size == Eigen::Dynamic) || ...) ? Eigen::Dynamic
                                                                               : (ArgTraits<Args>::size + ... + 0)};
  static constexpr int maxSize{
      ((ArgTraits<Args>::maxSize == Eigen::Dynamic) || ...) ? Eigen::Dynamic : (ArgTraits<Args>::maxSize + ... + 0)};

  /** Position among the uncertain inputs of each argument (meaningful for uncertain ones). */
  static constexpr std::array<std::size_t, count> ordinal{[] {
    std::array<std::size_t, count> result{};
    std::size_t next{0};
    for (std::size_t i{0}; i < count; ++i) result[i] = uncertain[i] ? next++ : 0;
    return result;
  }()};

  /** Argument position of the input marked by withCross(), or \c count if none is. */
  static constexpr std::size_t crossPosition{[] {
    constexpr std::array<bool, count> marked{ArgTraits<Args>::hasCross...};
    for (std::size_t i{0}; i < count; ++i) {
      if (marked[i]) return i;
    }
    return count;
  }()};
};

/** A function of the arguments of transform(), evaluated at a point of the joint uncertain input. */
template <typename F, typename OutputSpace, typename... Args>
class Transform {
 public:
  using ArgInfo = TransformArgs<Args...>;
  using Scalar = typename ArgInfo::Scalar;
  using Index = Eigen::Index;
  static constexpr int InputSize{ArgInfo::size};
  static constexpr int InputMaxSize{ArgInfo::maxSize};
  static constexpr int OutputSize{spaceSize<OutputSpace>};
  static constexpr int OutputMaxSize{spaceMaxSize<OutputSpace>};

  template <typename T>
  using InputVector = Eigen::Matrix<T, InputSize, 1, Eigen::ColMajor, InputMaxSize, 1>;
  template <typename T>
  using OutputVector = Eigen::Matrix<T, OutputSize, 1, Eigen::ColMajor, OutputMaxSize, 1>;
  using InputMatrix = Eigen::Matrix<Scalar, InputSize, InputSize, Eigen::ColMajor, InputMaxSize, InputMaxSize>;
  using OutputMatrix = Eigen::Matrix<Scalar, OutputSize, OutputSize, Eigen::ColMajor, OutputMaxSize, OutputMaxSize>;
  using JacobianMatrix = MatrixOf<Scalar, OutputSize, InputSize, OutputMaxSize, InputMaxSize>;
  using CrossMatrix = MatrixOf<Scalar, InputSize, OutputSize, InputMaxSize, OutputMaxSize>;
  using OutputSamples = SampleSetN<Scalar, OutputSize, OutputMaxSize>;

  /** Output moments, and the cross-covariance of every uncertain input with the output. */
  struct Result {
    OutputVector<Scalar> mean;
    OutputMatrix cov;
    CrossMatrix cross;  ///< rows in joint input order
  };

  Transform(F& f, const OutputSpace& outputSpace, const Args&... args)
      : m_f{f}, m_outputSpace{outputSpace}, m_args{args...} {
    forEachUncertain([this]<std::size_t I>(const auto& input) {
      m_offsets[I] = m_inputSize;
      m_inputSize += input.size();
    });
    // A resizable output space with one resizable block takes the size f returns.
    if constexpr (OutputSize == Eigen::Dynamic && requires(Index n) { m_outputSpace.resize(n); }) {
      const Index outputSize{OutputVector<Scalar>{call<Scalar>(m_f, mean())}.size()};
      if (m_outputSpace.size() != outputSize) m_outputSpace.resize(outputSize);
    }
  }

  [[nodiscard]] const OutputSpace& outputSpace() const { return m_outputSpace; }

  /** \returns the dimension of the joint uncertain input */
  [[nodiscard]] Index inputSize() const { return m_inputSize; }

  /** \returns the concatenated means of the uncertain inputs */
  [[nodiscard]] InputVector<Scalar> mean() const {
    InputVector<Scalar> result{InputVector<Scalar>::Zero(m_inputSize)};
    forEachUncertain(
        [&]<std::size_t I>(const auto& input) { result.segment(m_offsets[I], input.size()) = input.mean(); });
    return result;
  }

  /** \returns the block-diagonal joint covariance of the independent uncertain inputs */
  [[nodiscard]] InputMatrix cov() const {
    InputMatrix result{InputMatrix::Zero(m_inputSize, m_inputSize)};
    forEachUncertain([&]<std::size_t I>(const auto& input) {
      result.block(m_offsets[I], m_offsets[I], input.size(), input.size()) = symmetricCov(input);
    });
    return result;
  }

  /** \returns a square root \f$ S \f$ of cov(), \f$ S S^T = \Sigma \f$, block by block. Each block
   * comes from a pivoted LDLT with negative pivots clamped to zero, so a positive
   * semidefinite input has a root too.
   */
  [[nodiscard]] InputMatrix root() const {
    InputMatrix result{InputMatrix::Zero(m_inputSize, m_inputSize)};
    forEachUncertain([&]<std::size_t I>(const auto& input) {
      using Block = std::remove_cvref_t<decltype(symmetricCov(input))>;
      const Eigen::LDLT<Block> ldlt{symmetricCov(input)};
      const Block lower{ldlt.matrixL()};
      const Block scaled{lower * ldlt.vectorD().cwiseMax(Scalar(0)).cwiseSqrt().asDiagonal()};
      result.block(m_offsets[I], m_offsets[I], input.size(), input.size()) =
          ldlt.transpositionsP().transpose() * scaled;
    });
    return result;
  }

  /** Normalizes each uncertain input's block of \a x in that input's space. */
  void normalizeInputs(InputVector<Scalar>& x) const {
    forEachUncertain([&]<std::size_t I>(const auto& input) {
      input.space().normalizeValues(x.segment(m_offsets[I], input.size()));
    });
  }

  /** \returns \a f at the joint uncertain input \a x, with scalar type \a T */
  template <typename T>
  [[nodiscard]] OutputVector<T> evaluate(const InputVector<T>& x) const {
    OutputVector<T> y{call<T>(m_f, x)};
    EIGEN_FILTER_ASSERT(y.size() == m_outputSpace.size() && "the output does not fit the output space");
    return y;
  }

  /** \returns \a g called with the same arguments as \a f, at the joint uncertain input \a x */
  template <typename T, typename G>
  [[nodiscard]] decltype(auto) call(G& g, const InputVector<T>& x) const {
    return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto) {
      return g(argument<T, I>(x)...);
    }(std::index_sequence_for<Args...>{});
  }

  /** Calls \a fn.template operator()<I>(input) for each uncertain argument \a I, in order. */
  template <typename Fn>
  void forEachUncertain(Fn&& fn) const {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (visitIfUncertain<I>(fn), ...);
    }(std::index_sequence_for<Args...>{});
  }

  /** \returns the offset in the joint input of uncertain argument \a i */
  [[nodiscard]] Index offset(std::size_t i) const { return m_offsets[i]; }

  /** \returns the distribution of the output: a Gaussian in the output space, and with an
   * input marked by withCross(), that input's cross-covariance with it
   */
  [[nodiscard]] auto assemble(const Result& result) const {
    Gaussian<Scalar, OutputSpace> output{m_outputSpace, result.mean, result.cov};
    if constexpr (ArgInfo::crossPosition == ArgInfo::count) {
      return output;
    } else {
      constexpr std::size_t position{ArgInfo::crossPosition};
      using InputSpace = typename ArgTraits<ArgAt<position>>::Input::SpaceType;
      using WithCrossResult = OutputWithCross<Scalar, InputSpace, OutputSpace>;
      const Index inputSize{argInput(std::get<position>(m_args)).size()};
      return WithCrossResult{std::move(output), typename WithCrossResult::CrossMatrix{
                                                    result.cross.middleRows(m_offsets[position], inputSize)}};
    }
  }

 private:
  template <std::size_t I>
  using ArgAt = std::tuple_element_t<I, std::tuple<Args...>>;

  template <typename Input>
  [[nodiscard]] static auto symmetricCov(const Input& input) {
    using Block = Eigen::Matrix<Scalar, Input::SizeAtCompileTime, Input::SizeAtCompileTime, Eigen::ColMajor,
                                Input::MaxSizeAtCompileTime, Input::MaxSizeAtCompileTime>;
    const Block cov{input.cov()};
    return Block{cov.template selfadjointView<Eigen::Lower>()};
  }

  template <std::size_t I, typename Fn>
  void visitIfUncertain(Fn& fn) const {
    if constexpr (ArgInfo::uncertain[I]) fn.template operator()<I>(argInput(std::get<I>(m_args)));
  }

  /** \returns argument \a I for \a f: its block of \a x if uncertain, otherwise itself */
  template <typename T, std::size_t I>
  [[nodiscard]] decltype(auto) argument(const InputVector<T>& x) const {
    const auto& arg{std::get<I>(m_args)};
    if constexpr (ArgInfo::uncertain[I]) {
      using Input = typename ArgTraits<ArgAt<I>>::Input;
      using Block = Eigen::Matrix<T, Input::SizeAtCompileTime, 1, Eigen::ColMajor, Input::MaxSizeAtCompileTime, 1>;
      return Block{x.segment(m_offsets[I], argInput(arg).size())};
    } else {
      return arg;
    }
  }

  F& m_f;
  OutputSpace m_outputSpace;
  std::tuple<const Args&...> m_args;
  std::array<Index, sizeof...(Args)> m_offsets{};
  Index m_inputSize{0};
};

/** The linear core: output mean \a mean (normalized here) and Jacobian \a jacobian over the
 * joint input give \f$ \Sigma_{yy} = J \Sigma J^T \f$ and \f$ \Sigma_{xy} = \Sigma J^T \f$.
 */
template <typename Problem>
[[nodiscard]] typename Problem::Result linearResult(
    const Problem& problem, typename Problem::template OutputVector<typename Problem::Scalar> mean,
    const typename Problem::JacobianMatrix& jacobian) {
  problem.outputSpace().normalizeValues(mean);
  typename Problem::CrossMatrix cross{problem.cov() * jacobian.transpose()};
  typename Problem::OutputMatrix cov{jacobian * cross};
  cov = (typename Problem::Scalar(0.5) * (cov + cov.transpose())).eval();
  return {std::move(mean), std::move(cov), std::move(cross)};
}

template <typename Problem>
[[nodiscard]] typename Problem::Result transformResult(const Affine&, const Problem& problem) {
  using Scalar = typename Problem::Scalar;
  using OutputVector = typename Problem::template OutputVector<Scalar>;
  const Eigen::Index n{problem.inputSize()};
  const typename Problem::template InputVector<Scalar> mu{problem.mean()};
  const OutputVector center{problem.template evaluate<Scalar>(mu)};
  typename Problem::JacobianMatrix jacobian{Problem::JacobianMatrix::Zero(center.size(), n)};
  auto x{mu};
  for (Eigen::Index j{0}; j < n; ++j) {
    x(j) += Scalar(1);
    OutputVector step{problem.template evaluate<Scalar>(x) - center};
    problem.outputSpace().normalizeDifferences(step);
    jacobian.col(j) = step;
    x(j) = mu(j);
  }
  return linearResult(problem, center, jacobian);
}

template <typename Problem>
[[nodiscard]] typename Problem::Result transformResult(const Linearized<AutoDiff>&, const Problem& problem) {
  using Scalar = typename Problem::Scalar;
  using Derivatives = Eigen::Matrix<Scalar, Problem::InputSize, 1>;
  using Dual = Eigen::AutoDiffScalar<Derivatives>;
  const Eigen::Index n{problem.inputSize()};
  const typename Problem::template InputVector<Scalar> mu{problem.mean()};
  typename Problem::template InputVector<Dual> x;
  x.resize(n);
  for (Eigen::Index i{0}; i < n; ++i) x(i) = Dual{mu(i), int(n), int(i)};
  const typename Problem::template OutputVector<Dual> y{problem.template evaluate<Dual>(x)};

  typename Problem::template OutputVector<Scalar> mean{Problem::template OutputVector<Scalar>::Zero(y.size())};
  typename Problem::JacobianMatrix jacobian{Problem::JacobianMatrix::Zero(y.size(), n)};
  for (Eigen::Index r{0}; r < y.size(); ++r) {
    mean(r) = y(r).value();
    // An output that does not depend on the inputs may carry no derivatives at all.
    if (y(r).derivatives().size() > 0) jacobian.row(r) = y(r).derivatives().transpose();
  }
  return linearResult(problem, std::move(mean), jacobian);
}

template <typename Problem, typename Jacobian>
[[nodiscard]] typename Problem::Result transformResult(const Linearized<Jacobian>& method, const Problem& problem) {
  using Scalar = typename Problem::Scalar;
  const typename Problem::template InputVector<Scalar> mu{problem.mean()};
  typename Problem::template OutputVector<Scalar> mean{problem.template evaluate<Scalar>(mu)};
  typename Problem::JacobianMatrix jacobian{Problem::JacobianMatrix::Zero(mean.size(), problem.inputSize())};
  Jacobian jacobianFn{method.jacobian};
  const auto blocks{problem.template call<Scalar>(jacobianFn, mu)};
  problem.forEachUncertain([&]<std::size_t I>(const auto& input) {
    const auto block{[&]() -> decltype(auto) {
      if constexpr (requires { std::tuple_size<std::remove_cvref_t<decltype(blocks)>>::value; }) {
        return std::get<Problem::ArgInfo::ordinal[I]>(blocks);
      } else {
        static_assert(Problem::ArgInfo::uncertainCount == 1,
                      "with more than one uncertain input, the Jacobian must return a tuple of blocks");
        return (blocks);
      }
    }()};
    EIGEN_FILTER_ASSERT(block.rows() == mean.size() && block.cols() == input.size() &&
                        "a Jacobian block must be output size by input size");
    jacobian.middleCols(problem.offset(I), input.size()) = block;
  });
  return linearResult(problem, std::move(mean), jacobian);
}

template <typename Problem>
[[nodiscard]] typename Problem::Result transformResult(const Unscented& method, const Problem& problem) {
  using Scalar = typename Problem::Scalar;
  using OutputVector = typename Problem::template OutputVector<Scalar>;
  using OutputSamples = typename Problem::OutputSamples;
  const Eigen::Index n{problem.inputSize()};
  const Scalar alpha{Scalar(method.alpha)};
  const Scalar lambda{alpha * alpha * (Scalar(n) + Scalar(method.kappa)) - Scalar(n)};
  EIGEN_FILTER_ASSERT(Scalar(n) + lambda > Scalar(0) && "the sigma-point spread must be positive");
  const Scalar centerWeight{lambda / (Scalar(n) + lambda)};
  const Scalar centerCovWeight{centerWeight + Scalar(1) - alpha * alpha + Scalar(method.beta)};
  const Scalar pointWeight{Scalar(1) / (Scalar(2) * (Scalar(n) + lambda))};

  const typename Problem::template InputVector<Scalar> mu{problem.mean()};
  const typename Problem::InputMatrix spread{Eigen::numext::sqrt(Scalar(n) + lambda) * problem.root()};
  const OutputVector center{problem.template evaluate<Scalar>(mu)};
  OutputSamples plus{center.size(), n};
  OutputSamples minus{center.size(), n};
  for (Eigen::Index j{0}; j < n; ++j) {
    typename Problem::template InputVector<Scalar> x{mu + spread.col(j)};
    problem.normalizeInputs(x);
    plus.col(j) = problem.template evaluate<Scalar>(x);
    x = mu - spread.col(j);
    problem.normalizeInputs(x);
    minus.col(j) = problem.template evaluate<Scalar>(x);
  }

  // Tangent-space mean about the center point, then deviations about that mean.
  const auto& space{problem.outputSpace()};
  OutputSamples plusDeviations{plus.colwise() - center};
  OutputSamples minusDeviations{minus.colwise() - center};
  space.normalizeDifferences(plusDeviations);
  space.normalizeDifferences(minusDeviations);
  OutputVector mean{center + pointWeight * (plusDeviations.rowwise().sum() + minusDeviations.rowwise().sum())};
  space.normalizeValues(mean);

  OutputVector centerDeviation{center - mean};
  space.normalizeDifferences(centerDeviation);
  plusDeviations = plus.colwise() - mean;
  minusDeviations = minus.colwise() - mean;
  space.normalizeDifferences(plusDeviations);
  space.normalizeDifferences(minusDeviations);

  typename Problem::OutputMatrix cov{
      centerCovWeight * centerDeviation * centerDeviation.transpose() +
      pointWeight * (plusDeviations * plusDeviations.transpose() + minusDeviations * minusDeviations.transpose())};
  cov = (Scalar(0.5) * (cov + cov.transpose())).eval();
  typename Problem::CrossMatrix cross{pointWeight * spread * (plusDeviations - minusDeviations).transpose()};
  return {std::move(mean), std::move(cov), std::move(cross)};
}

/** The type \a f receives for argument \a Arg: a plain vector if uncertain, the argument otherwise. */
template <typename Arg, typename Scalar>
struct ArgumentFor {
  using type = const Arg&;
};

template <typename Arg, typename Scalar>
  requires ArgTraits<Arg>::isUncertain
struct ArgumentFor<Arg, Scalar> {
  using Input = typename ArgTraits<Arg>::Input;
  using type = Eigen::Matrix<Scalar, Input::SizeAtCompileTime, 1, Eigen::ColMajor, Input::MaxSizeAtCompileTime, 1>;
};

/** Linear, sized by the return type of \a F called with plain vectors for the uncertain inputs. */
template <typename F, typename... Args>
struct DefaultOutputSpace {
  using Scalar = typename TransformArgs<Args...>::Scalar;
  using Returned = decltype(std::declval<F&>()(std::declval<typename ArgumentFor<Args, Scalar>::type>()...));
  using Evaluated = std::remove_cvref_t<decltype(std::declval<Returned>().eval())>;
  static_assert(Evaluated::ColsAtCompileTime == 1, "a transformed function must return a column vector");
  using type = Linear<Evaluated::RowsAtCompileTime, Evaluated::MaxRowsAtCompileTime>;
};

template <typename... Args>
inline constexpr bool firstIsSpace{false};

template <typename First, typename... Rest>
inline constexpr bool firstIsSpace<First, Rest...>{IsSpace<First>};

}  // namespace internal

/** \ingroup Transforms
 * Marks the uncertain \a input, usually the state, whose cross-covariance with the output
 * transform() also returns (see OutputWithCross). At most one input may be marked.
 */
template <UncertainInput Input>
[[nodiscard]] internal::WithCross<Input> withCross(const Input& input) {
  return {input};
}

/** \ingroup Transforms
 * \returns the distribution of \f$ y = f(\ldots) \f$ in \a outputSpace, with \a args passed to
 * \a f in order, computed by \a method (Affine, Linearized, or Unscented): a Gaussian in
 * \a outputSpace, or with an input marked by withCross(), an OutputWithCross.
 *
 * \sa \ref Transforms, update()
 */
template <typename Method, typename F, typename OutputSpace, typename... Args>
  requires(internal::IsSpace<OutputSpace> && internal::crossCount<Args...> <= 1)
[[nodiscard]] auto transform(const Method& method, F&& f, const OutputSpace& outputSpace, const Args&... args) {
  using Problem = internal::Transform<std::remove_reference_t<F>, OutputSpace, Args...>;
  const Problem problem{f, outputSpace, args...};
  return problem.assemble(internal::transformResult(method, problem));
}

/** \ingroup Transforms
 * transform() into the Linear space sized by \a f's return type.
 */
template <typename Method, typename F, typename... Args>
  requires(!internal::firstIsSpace<Args...> && internal::crossCount<Args...> <= 1)
[[nodiscard]] auto transform(const Method& method, F&& f, const Args&... args) {
  using OutputSpace = typename internal::DefaultOutputSpace<std::remove_reference_t<F>, Args...>::type;
  return transform(method, std::forward<F>(f), OutputSpace{}, args...);
}

/** \ingroup Transforms
 * transform() with the Affine method.
 */
template <typename F, typename... Args>
[[nodiscard]] auto affine(F&& f, const Args&... args) {
  return transform(Affine{}, std::forward<F>(f), args...);
}

/** \ingroup Transforms
 * transform() with the Linearized method, differentiating \a f automatically.
 */
template <typename F, typename... Args>
[[nodiscard]] auto linearized(F&& f, const Args&... args) {
  return transform(Linearized{}, std::forward<F>(f), args...);
}

/** \ingroup Transforms
 * transform() with the Unscented method and its default parameters.
 */
template <typename F, typename... Args>
[[nodiscard]] auto unscented(F&& f, const Args&... args) {
  return transform(Unscented{}, std::forward<F>(f), args...);
}

/** \ingroup Transforms
 * \returns \a prior conditioned on an observation \a observed of the output of \a predicted,
 * which came from transform() with \a prior marked by withCross(), and the log-likelihood of
 * the observation.
 *
 * The Kalman update: with gain \f$ K = \Sigma_{xy} \Sigma_{yy}^{-1} \f$ and residual
 * \f$ r = y \boxminus \bar y \f$, normalized in the output space, the posterior mean is
 * \f$ \mu \boxplus K r \f$, normalized in the prior's space, and its covariance is
 * \f$ \Sigma - K \Sigma_{xy}^T \f$. Solves with a pivoted LDLT, so it always produces a result.
 */
template <UncertainInput Prior, typename OutputSpace, typename Derived>
[[nodiscard]] auto update(
    const Prior& prior,
    const OutputWithCross<typename Prior::Scalar, typename Prior::SpaceType, OutputSpace>& predicted,
    const Eigen::MatrixBase<Derived>& observed) {
  using Scalar = typename Prior::Scalar;
  using InputSpace = typename Prior::SpaceType;
  constexpr int InputSize{spaceSize<InputSpace>};
  constexpr int InputMaxSize{spaceMaxSize<InputSpace>};
  constexpr int OutputSize{spaceSize<OutputSpace>};
  constexpr int OutputMaxSize{spaceMaxSize<OutputSpace>};
  constexpr int JointSize{InputSize == Eigen::Dynamic || OutputSize == Eigen::Dynamic ? Eigen::Dynamic
                                                                                      : InputSize + OutputSize};
  constexpr int JointMaxSize{InputMaxSize == Eigen::Dynamic || OutputMaxSize == Eigen::Dynamic
                                 ? Eigen::Dynamic
                                 : InputMaxSize + OutputMaxSize};
  const Eigen::Index n{prior.size()};
  const Eigen::Index m{predicted.output.size()};
  EIGEN_FILTER_ASSERT(predicted.cross.rows() == n && predicted.cross.cols() == m &&
                      "the prediction was not made from this prior");
  EIGEN_FILTER_ASSERT(observed.size() == m);

  // The joint of the prior and the output, conditioned on the output.
  Eigen::Matrix<Scalar, JointSize, 1, Eigen::ColMajor, JointMaxSize, 1> mean;
  mean.resize(n + m);
  mean << prior.mean(), predicted.output.mean();
  internal::MatrixOf<Scalar, JointSize, JointSize, JointMaxSize, JointMaxSize> cov;
  cov.resize(n + m, n + m);
  cov.topLeftCorner(n, n) = prior.cov();
  cov.topRightCorner(n, m) = predicted.cross;
  cov.bottomLeftCorner(m, n) = predicted.cross.transpose();
  cov.bottomRightCorner(m, m) = predicted.output.cov();
  using Target = internal::IndexVector<InputSize, InputMaxSize>;
  using Given = internal::IndexVector<OutputSize, OutputMaxSize>;
  const Target target{Target::LinSpaced(n, 0, n - 1)};
  const Given given{Given::LinSpaced(m, n, n + m - 1)};
  auto conditioned{
      internal::conditionMoments(mean, cov, target, given, observed, predicted.output.space(), prior.space())};
  return Updated<Scalar, InputSpace>{
      Gaussian<Scalar, InputSpace>{prior.space(), std::move(conditioned.mean), std::move(conditioned.cov)},
      conditioned.logLikelihood};
}

/** \ingroup Transforms
 * update() with an observation of zero, as for a measurement model that returns an innovation.
 */
template <UncertainInput Prior, typename OutputSpace>
[[nodiscard]] auto update(
    const Prior& prior,
    const OutputWithCross<typename Prior::Scalar, typename Prior::SpaceType, OutputSpace>& predicted) {
  using Vector = typename Gaussian<typename Prior::Scalar, OutputSpace>::VectorType;
  return update(prior, predicted, Vector::Zero(predicted.output.size()));
}

}  // namespace EigenFilter
