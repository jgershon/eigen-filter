#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Cholesky>

#include "Gaussian.h"
#include "Indexing.h"
#include "Macros.h"
#include "Moments.h"
#include "Resampling.h"
#include "SampleSet.h"
#include "Space.h"
#include "Statistics.h"
#include "Weights.h"

namespace EigenFilter {

template <typename Scalar_, int Size_, int MaxSize_ = Size_, typename Space_ = Linear<Size_, MaxSize_>>
class ParticleSet;

template <typename Scalar_, int Size_, int MaxSize_, typename Space_>
class MutableParticleSet;

template <typename Scalar_, int Size_, int MaxSize_, typename Space_>
class KernelDensity;

template <typename Source, typename Indices,
          typename SubSpace = internal::ProjectedSpace<
              typename Source::SpaceType,
              internal::IndexedSize<Indices, Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>::size,
              internal::IndexedSize<Indices, Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>::maxSize>>
class ParticleSetMarginal;

/** Reasons MutableParticleSet::toKernelDensity() can fail: the weights cannot be
 * normalized, or the bandwidth is not positive definite or not finite.
 */
using KernelDensityError = std::variant<WeightError, GaussianError>;

namespace internal {

/** Silverman's rule-of-thumb bandwidth for the columns of \a states, from their
 * covariance in \a space. Does not validate the weights; invalid weights give a
 * non-finite result.
 */
template <typename MomentsType, typename Space, typename StatesDerived, typename WeightsDerived>
[[nodiscard]] typename MomentsType::MatrixType silvermanBandwidth(const Space& space,
                                                                  const Eigen::MatrixBase<StatesDerived>& states,
                                                                  const Eigen::ArrayBase<WeightsDerived>& logWeights) {
  using Scalar = typename MomentsType::Scalar;
  LogSumExp<Scalar> total;
  for (const Scalar w : logWeights.derived()) total.add(w);
  const Scalar logTotal{total.value()};
  const Scalar d{Scalar(states.rows())};
  const Scalar factor{Eigen::numext::pow(Scalar(4) / (effectiveSampleSize(logWeights, logTotal) * (d + Scalar(2))),
                                         Scalar(2) / (d + Scalar(4)))};
  return factor * weightedMomentsIn<MomentsType>(space, states, logWeights, logTotal).cov();
}

/** Shared implementation of the particle set and KernelDensity conditionals.
 *
 * Each kernel \f$ \mathcal{N}(x_i, H) \f$ is conditioned on the entries \a given
 * equal to \a values: its center becomes \f$ x_{a,i} + K (v - x_{b,i}) \f$ with
 * \f$ K = H_{ab} H_{bb}^{-1} \f$, its weight gains \f$ \log \mathcal{N}(v; x_{b,i}, H_{bb}) \f$,
 * and the shared bandwidth becomes \f$ H_{aa} - K H_{ba} \f$.
 */
template <typename Source, typename BandwidthDerived, typename TargetIndices, typename GivenIndices, typename Derived>
[[nodiscard]] auto conditionalParticles(const Source& source, const Eigen::MatrixBase<BandwidthDerived>& bandwidth,
                                        const TargetIndices& targetIndices, const GivenIndices& givenIndices,
                                        const Eigen::MatrixBase<Derived>& values) {
  using Scalar = typename Source::Scalar;
  constexpr int SourceSize{Source::SizeAtCompileTime};
  constexpr int SourceMaxSize{Source::MaxSizeAtCompileTime};
  const auto target{resolveIndices<SourceSize, SourceMaxSize>(source.particleSize(), targetIndices)};
  const auto given{resolveIndices<SourceSize, SourceMaxSize>(source.particleSize(), givenIndices)};
  EIGEN_FILTER_ASSERT(distinctIndices(target, given) && "target and given indices must be distinct");
  EIGEN_FILTER_ASSERT(values.size() == given.size());
  using Target = std::remove_cvref_t<decltype(target)>;
  using Given = std::remove_cvref_t<decltype(given)>;
  constexpr int TargetSize{Target::RowsAtCompileTime};
  constexpr int TargetMaxSize{Target::MaxRowsAtCompileTime};
  constexpr int GivenSize{Given::RowsAtCompileTime};
  constexpr int GivenMaxSize{Given::MaxRowsAtCompileTime};
  using GivenMatrix = MatrixOf<Scalar, GivenSize, GivenSize, GivenMaxSize, GivenMaxSize>;
  using CrossMatrix = MatrixOf<Scalar, TargetSize, GivenSize, TargetMaxSize, GivenMaxSize>;
  using GivenStates = SampleSet<Scalar, GivenSize, GivenMaxSize>;
  using TargetStates = SampleSet<Scalar, TargetSize, TargetMaxSize>;
  auto targetSpace{projectSpace<TargetSize, TargetMaxSize>(source.space(), source.particleSize(), target)};
  const auto givenSpace{projectSpace<GivenSize, GivenMaxSize>(source.space(), source.particleSize(), given)};

  const Eigen::LDLT<GivenMatrix> ldlt{GivenMatrix{lowerSymmetricBlock(bandwidth, given, given)}};
  const CrossMatrix cross{lowerSymmetricBlock(bandwidth, target, given)};
  const CrossMatrix gain{ldlt.solve(cross.transpose()).transpose()};

  GivenStates offsets{source.states()(given, Eigen::placeholders::all).colwise() - values.derived()};
  givenSpace.normalizeDifferences(offsets);
  const GivenStates solved{ldlt.solve(offsets)};
  const Eigen::Array<Scalar, 1, Eigen::Dynamic> wsse{(offsets.array() * solved.array()).colwise().sum()};
  const Scalar log2Pi{Eigen::numext::log(Scalar(2) * Scalar(EIGEN_PI))};
  const Scalar logNormalizer{Scalar(-0.5) * (Scalar(given.size()) * log2Pi + ldlt.vectorD().array().log().sum())};

  const TargetStates centers{source.states()(target, Eigen::placeholders::all) - gain * offsets};
  MutableParticleSet<Scalar, TargetSize, TargetMaxSize, decltype(targetSpace)> result{
      targetSpace, centers,
      Eigen::Array<Scalar, Eigen::Dynamic, 1>{source.logWeights() + logNormalizer - Scalar(0.5) * wsse.transpose()}};
  result.setBandwidth(lowerSymmetricBlock(bandwidth, target, target) - gain * cross.transpose());
  return result;
}

/** conditionalParticles() with every entry not in \a givenIndices as the target. */
template <typename Source, typename BandwidthDerived, typename GivenIndices, typename Derived>
[[nodiscard]] auto conditionalParticles(const Source& source, const Eigen::MatrixBase<BandwidthDerived>& bandwidth,
                                        const GivenIndices& givenIndices, const Eigen::MatrixBase<Derived>& values) {
  constexpr int SourceSize{Source::SizeAtCompileTime};
  constexpr int SourceMaxSize{Source::MaxSizeAtCompileTime};
  const auto given{resolveIndices<SourceSize, SourceMaxSize>(source.particleSize(), givenIndices)};
  return conditionalParticles(
      source, bandwidth, complementIndices<SourceSize, SourceMaxSize>(source.particleSize(), given), given, values);
}

}  // namespace internal

/** \brief Mutable set of weighted samples that can be viewed as a ParticleSet.
 *
 * Particle states are the columns of states(); their log-weights are
 * logWeights(). Weights need not be normalized; a log-weight of -inf is a zero
 * weight. Particles are addressed by index. Removal may reorder the remaining
 * particles.
 *
 * A set may carry a kernel bandwidth: the shared covariance \f$ H \f$ of the
 * Gaussian kernels in its kernel density estimate. kdeBandwidth() is the stored
 * bandwidth if there is one, and Silverman's rule otherwise. Conditionals store
 * their exact bandwidth this way. A stored bandwidth is kept as the states change,
 * so it can go stale; call clearBandwidth() to return to Silverman's rule.
 *
 * toParticleSet() validates the weights and returns a ParticleSet view, and
 * toKernelDensity() goes straight to a KernelDensity. While any view is alive this
 * object is locked: mutating, moving, or destroying it is a programmer error caught
 * by \c EIGEN_FILTER_ASSERT. It is also locked while forEach() and removeIf() call
 * back into user code.
 *
 * marginal() returns a read-only view of a subset of the state entries. While any
 * marginal is alive this object is borrowed: states, weights, and bandwidth may
 * change, but adding, removing, resampling, or resizing particles, or moving or
 * destroying this object, is a programmer error.
 *
 * \tparam Scalar_   scalar type
 * \tparam Size_     state dimension, or \c Eigen::Dynamic
 * \tparam MaxSize_  maximum state dimension
 * \tparam Space_    space of the states (see \ref Spaces): states are normalized when
 *                   stored, and moments are tangent-space moments. Linear by default.
 *
 * \sa class ParticleSet, class KernelDensity, \ref Resamplers
 */
template <typename Scalar_, int Size_, int MaxSize_ = Size_, typename Space_ = Linear<Size_, MaxSize_>>
class MutableParticleSet {
  static_assert(Size_ == Eigen::Dynamic || isValidSize<Space_>(Size_), "the size does not fit the space");

 public:
  using MomentsType = Moments<Scalar_, Size_, MaxSize_>;
  using Scalar = typename MomentsType::Scalar;
  using Index = typename MomentsType::Index;
  using VectorType = typename MomentsType::VectorType;
  using MatrixType = typename MomentsType::MatrixType;
  /** Particle states as columns. */
  using SampleSetType = SampleSet<Scalar_, Size_, MaxSize_>;
  using ConstStateType = typename SampleSetType::ConstColXpr;
  using WeightArrayType = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
  using SpaceType = Space_;
  using ParticleSetType = ParticleSet<Scalar_, Size_, MaxSize_, Space_>;
  using KernelDensityType = KernelDensity<Scalar_, Size_, MaxSize_, Space_>;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};
  static constexpr bool IsMutable{true};

  /** Result of removeIf(): the log of the removed fraction of the total weight,
   * or nothing when \a ReturnRemovedWeight is false.
   */
  template <bool ReturnRemovedWeight>
  using RemovalResult = std::conditional_t<ReturnRemovedWeight, std::expected<Scalar, WeightError>, void>;

  /** \brief Writable access to one particle during forEach().
   *
   * Exists only for the duration of the callback; it cannot be copied or moved.
   */
  class MutableParticle {
   public:
    MutableParticle(const MutableParticle&) = delete;
    MutableParticle& operator=(const MutableParticle&) = delete;

    [[nodiscard]] Index index() const { return m_index; }
    [[nodiscard]] Scalar logWeight() const { return m_set.m_logWeights(m_index); }
    [[nodiscard]] ConstStateType state() const { return std::as_const(m_set.m_states).col(m_index); }

    template <typename Derived>
    void setState(const Eigen::MatrixBase<Derived>& state) {
      EIGEN_FILTER_ASSERT(state.size() == m_set.particleSize());
      m_set.m_states.col(m_index) = state;
      m_set.m_space.normalizeValues(m_set.m_states.col(m_index));
    }

   private:
    friend MutableParticleSet;

    MutableParticle(MutableParticleSet& set, Index index) : m_set{set}, m_index{index} {}

    MutableParticleSet& m_set;
    Index m_index;
  };

  MutableParticleSet() = default;

  /** Constructs an empty set whose particles will have dimension \a particleSize. */
  explicit MutableParticleSet(Index particleSize) : m_states{particleSize, 0} { adoptSize(); }

  /** Constructs a set from the columns of \a states, with equal weights summing to one. */
  template <typename StatesDerived>
  explicit MutableParticleSet(const Eigen::MatrixBase<StatesDerived>& states) {
    assign(states);
  }

  template <typename StatesDerived, typename WeightsDerived>
  MutableParticleSet(const Eigen::MatrixBase<StatesDerived>& states,
                     const Eigen::DenseBase<WeightsDerived>& logWeights) {
    assign(states, logWeights);
  }

  /** Constructs an empty set in \a space whose particles will have dimension \a particleSize. */
  MutableParticleSet(Space_ space, Index particleSize) : m_states{particleSize, 0}, m_space{std::move(space)} {
    checkSpace();
  }

  /** Constructs a set in \a space from the columns of \a states, with equal weights summing to one. */
  template <typename StatesDerived>
  MutableParticleSet(Space_ space, const Eigen::MatrixBase<StatesDerived>& states)
      : m_states{states},
        m_logWeights{WeightArrayType::Constant(states.cols(), -Eigen::numext::log(Scalar(states.cols())))},
        m_space{std::move(space)} {
    checkSpace();
  }

  template <typename StatesDerived, typename WeightsDerived>
  MutableParticleSet(Space_ space, const Eigen::MatrixBase<StatesDerived>& states,
                     const Eigen::DenseBase<WeightsDerived>& logWeights)
      : m_states{states}, m_logWeights{logWeights.derived().array()}, m_space{std::move(space)} {
    EIGEN_FILTER_ASSERT(m_logWeights.size() == m_states.cols());
    checkSpace();
  }

  /** Copies the particles and any stored bandwidth. The copy starts unlocked and unborrowed. */
  MutableParticleSet(const MutableParticleSet& other)
      : m_states{other.m_states},
        m_logWeights{other.m_logWeights},
        m_space{other.m_space},
        m_bandwidth{other.m_bandwidth} {}

  /** Moving a locked or borrowed object is a programmer error; if \c EIGEN_FILTER_ASSERT
   * throws, the exception reaches this \c noexcept boundary and calls \c std::terminate.
   */
  MutableParticleSet(MutableParticleSet&& other) noexcept
      : m_states{std::move(other.m_states)},
        m_logWeights{std::move(other.m_logWeights)},
        m_space{std::move(other.m_space)},
        m_bandwidth{std::move(other.m_bandwidth)},
        m_stateScratch{std::move(other.m_stateScratch)},
        m_weightScratch{std::move(other.m_weightScratch)},
        m_ancestors{std::move(other.m_ancestors)} {
    other.assertUnlocked();
    other.assertUnborrowed();
  }

  MutableParticleSet& operator=(const MutableParticleSet& other) {
    assertUnlocked();
    assertUnborrowed();
    m_states = other.m_states;
    m_logWeights = other.m_logWeights;
    m_space = other.m_space;
    m_bandwidth = other.m_bandwidth;
    return *this;
  }

  MutableParticleSet& operator=(MutableParticleSet&& other) noexcept {
    assertUnlocked();
    assertUnborrowed();
    other.assertUnlocked();
    other.assertUnborrowed();
    m_states = std::move(other.m_states);
    m_logWeights = std::move(other.m_logWeights);
    m_space = std::move(other.m_space);
    m_bandwidth = std::move(other.m_bandwidth);
    return *this;
  }

  ~MutableParticleSet() {
    assertUnlocked();
    assertUnborrowed();
  }

  /** \returns the number of particles */
  [[nodiscard]] Index size() const { return m_states.cols(); }
  [[nodiscard]] bool empty() const { return size() == 0; }

  /** \returns the dimension of every particle state */
  [[nodiscard]] Index particleSize() const { return m_states.rows(); }

  [[nodiscard]] const Space_& space() const { return m_space; }
  [[nodiscard]] const SampleSetType& states() const { return m_states; }
  [[nodiscard]] const WeightArrayType& logWeights() const { return m_logWeights; }
  [[nodiscard]] ConstStateType state(Index i) const { return m_states.col(i); }
  [[nodiscard]] Scalar logWeight(Index i) const { return m_logWeights(i); }

  /** \returns the stored kernel bandwidth, if any */
  [[nodiscard]] const std::optional<MatrixType>& bandwidth() const { return m_bandwidth; }

  /** \returns the bandwidth a kernel density estimate uses: the stored bandwidth, or
   * Silverman's rule if none is stored (non-finite if the weights are invalid)
   */
  [[nodiscard]] MatrixType kdeBandwidth() const {
    return m_bandwidth ? *m_bandwidth : internal::silvermanBandwidth<MomentsType>(m_space, m_states, m_logWeights);
  }

  /** Stores \a bandwidth as the kernel covariance. Only its lower triangle is read. */
  template <typename Derived>
  void setBandwidth(const Eigen::MatrixBase<Derived>& bandwidth) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(bandwidth.rows() == particleSize() && bandwidth.cols() == particleSize());
    m_bandwidth = MatrixType{bandwidth};
  }

  /** Forgets the stored bandwidth, so kdeBandwidth() returns to Silverman's rule. */
  void clearBandwidth() {
    assertUnlocked();
    m_bandwidth.reset();
  }

  /** Resizes every particle state to dimension \a n and clears the bandwidth.
   * States are unspecified afterward.
   */
  void resizeParticles(Index n) {
    assertUnlocked();
    assertUnborrowed();
    m_states.resize(n, size());
    m_bandwidth.reset();
    adoptSize();
  }

  void clear() {
    assertUnlocked();
    assertUnborrowed();
    m_states.resize(particleSize(), 0);
    m_logWeights.resize(0);
  }

  /** Replaces the particles with the columns of \a states, with equal weights summing
   * to one. Clears the bandwidth if the dimension changes.
   */
  template <typename StatesDerived>
  void assign(const Eigen::MatrixBase<StatesDerived>& states) {
    assertUnlocked();
    assertUnborrowed();
    if (states.rows() != particleSize()) m_bandwidth.reset();
    m_states = states;
    m_logWeights.setConstant(size(), -Eigen::numext::log(Scalar(size())));
    adoptSize();
  }

  /** Replaces the particles and weights. Clears the bandwidth if the dimension changes. */
  template <typename StatesDerived, typename WeightsDerived>
  void assign(const Eigen::MatrixBase<StatesDerived>& states, const Eigen::DenseBase<WeightsDerived>& logWeights) {
    assertUnlocked();
    assertUnborrowed();
    EIGEN_FILTER_ASSERT(logWeights.size() == states.cols());
    if (states.rows() != particleSize()) m_bandwidth.reset();
    m_states = states;
    m_logWeights = logWeights.derived().array();
    adoptSize();
  }

  /** Appends a particle. Costs O(size()) because the storage is contiguous.
   *
   * \returns the index of the new particle
   */
  template <typename Derived>
  Index add(Scalar logWeight, const Eigen::MatrixBase<Derived>& state) {
    assertUnlocked();
    assertUnborrowed();
    EIGEN_FILTER_ASSERT(state.size() == particleSize());
    const Index i{size()};
    m_states.conservativeResize(Eigen::NoChange, i + 1);
    m_logWeights.conservativeResize(i + 1);
    m_states.col(i) = state;
    m_space.normalizeValues(m_states.col(i));
    m_logWeights(i) = logWeight;
    return i;
  }

  /** Appends copies of \a other's particles, which must have the same dimension. */
  void append(const MutableParticleSet& other) {
    assertUnlocked();
    assertUnborrowed();
    EIGEN_FILTER_ASSERT(other.particleSize() == particleSize());
    EIGEN_FILTER_ASSERT(other.m_space == m_space && "cannot append a set in a different space");
    const Index n{size()};
    const Index m{other.size()};
    m_states.conservativeResize(Eigen::NoChange, n + m);
    m_logWeights.conservativeResize(n + m);
    m_states.rightCols(m) = other.m_states.leftCols(m);
    m_logWeights.tail(m) = other.m_logWeights.head(m);
  }

  /** Removes particle \a i by moving the last particle into its place. */
  void remove(Index i) {
    assertUnlocked();
    assertUnborrowed();
    assertIndex(i);
    const Index last{size() - 1};
    if (i != last) {
      m_states.col(i) = m_states.col(last);
      m_logWeights(i) = m_logWeights(last);
    }
    m_states.conservativeResize(Eigen::NoChange, last);
    m_logWeights.conservativeResize(last);
  }

  template <typename Derived>
  void setState(Index i, const Eigen::MatrixBase<Derived>& state) {
    assertUnlocked();
    assertIndex(i);
    EIGEN_FILTER_ASSERT(state.size() == particleSize());
    m_states.col(i) = state;
    m_space.normalizeValues(m_states.col(i));
  }

  /** Sets every state at once. \a states must have the current dimensions. */
  template <typename Derived>
  void setStates(const Eigen::MatrixBase<Derived>& states) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(states.rows() == particleSize() && states.cols() == size());
    m_states = states;
    m_space.normalizeValues(m_states);
  }

  void setLogWeight(Index i, Scalar logWeight) {
    assertUnlocked();
    assertIndex(i);
    m_logWeights(i) = logWeight;
  }

  template <typename Derived>
  void setLogWeights(const Eigen::DenseBase<Derived>& logWeights) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(logWeights.size() == size());
    m_logWeights = logWeights.derived().array();
  }

  /** Multiplies the weight of particle \a i by \f$ e^{\mathrm{logScale}} \f$. */
  void scaleWeight(Index i, Scalar logScale) {
    assertUnlocked();
    assertIndex(i);
    m_logWeights(i) += logScale;
  }

  /** Multiplies every weight by \f$ e^{\mathrm{logScale}} \f$. */
  void scaleWeights(Scalar logScale) {
    assertUnlocked();
    m_logWeights += logScale;
  }

  /** Multiplies weight \c i by \f$ e^{\mathrm{logScales}_i} \f$, e.g. by per-particle likelihoods. */
  template <typename Derived>
  void scaleWeights(const Eigen::DenseBase<Derived>& logScales) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(logScales.size() == size());
    m_logWeights += logScales.derived().array();
  }

  /** \returns the log of the sum of the weights, or a WeightError if the set is
   *          empty, the weights sum to zero, or a log-weight is NaN or +inf.
   */
  [[nodiscard]] std::expected<Scalar, WeightError> logTotalWeight() const {
    return internal::logTotalWeight<Scalar>(std::span<const Scalar>{m_logWeights.data(), std::size_t(size())});
  }

  /** Scales the weights so they sum to one. On failure the weights are unchanged.
   *
   * \sa logTotalWeight()
   */
  [[nodiscard]] std::expected<void, WeightError> normalize() {
    assertUnlocked();
    const auto total{logTotalWeight()};
    if (!total) return std::unexpected{total.error()};
    scaleWeights(-*total);
    return {};
  }

  /** Calls \a f on each particle as a \c MutableParticle&.
   *
   * If \a f returns a value, it is a log scale factor added to that particle's
   * log-weight. This set is locked while \a f runs, so \a f may modify the
   * particle it is given but not the set.
   */
  template <typename F>
  void forEach(F&& f) {
    assertUnlocked();
    const ScopedLock lock{*this};
    for (Index i{0}; i < size(); ++i) {
      MutableParticle particle{*this, i};
      if constexpr (std::is_void_v<std::invoke_result_t<F&, MutableParticle&>>) {
        std::invoke(f, particle);
      } else {
        m_logWeights(i) += std::invoke(f, particle);
      }
    }
  }

  /** Removes every particle for which \a pred(logWeight, state) is true.
   *
   * \tparam ReturnRemovedWeight  if true, returns the log of the removed fraction
   *         of the total weight. This requires normalizing the weights; if that
   *         fails, nothing is removed and the WeightError is returned.
   */
  template <bool ReturnRemovedWeight = false, typename Pred>
  RemovalResult<ReturnRemovedWeight> removeIf(Pred pred) {
    assertUnlocked();
    assertUnborrowed();
    const auto removeAtIndex{
        [&](Index i) { return static_cast<bool>(std::invoke(pred, m_logWeights(i), std::as_const(m_states).col(i))); }};
    if constexpr (ReturnRemovedWeight) {
      const auto total{logTotalWeight()};
      if (!total) return std::unexpected{total.error()};
      const ScopedLock lock{*this};
      return compact(removeAtIndex) - *total;
    } else {
      const ScopedLock lock{*this};
      compact(removeAtIndex);
    }
  }

  /** Replaces the particles with \a newSize draws chosen by \a resampler in
   * proportion to their weights, then sets every weight to \f$ 1/\mathrm{newSize} \f$.
   * A stored bandwidth is kept.
   *
   * \a rng is used only during this call. Scratch storage is kept between calls,
   * so repeated resampling at a steady size does not allocate.
   *
   * \returns nothing on success, or a WeightError (with the set unchanged) if
   *          the weights cannot be normalized.
   *
   * \sa \ref Resamplers
   */
  template <std::uniform_random_bit_generator G, typename R = SystematicResampler>
    requires Resampler<R, Scalar, G>
  [[nodiscard]] std::expected<void, WeightError> resample(G& rng, Index newSize, const R& resampler = {}) {
    assertUnlocked();
    assertUnborrowed();
    EIGEN_FILTER_ASSERT(newSize > 0);
    const auto total{logTotalWeight()};
    if (!total) return std::unexpected{total.error()};

    m_weightScratch.resize(std::size_t(size()));
    for (Index i{0}; i < size(); ++i) m_weightScratch[std::size_t(i)] = Eigen::numext::exp(m_logWeights(i) - *total);
    m_ancestors.resize(std::size_t(newSize));
    resampler(std::span<const Scalar>{m_weightScratch}, std::span<Index>{m_ancestors}, rng);

    m_stateScratch.resize(particleSize(), newSize);
    for (Index j{0}; j < newSize; ++j) {
      const Index ancestor{m_ancestors[std::size_t(j)]};
      EIGEN_FILTER_ASSERT(ancestor >= 0 && ancestor < size() && "resampler produced an out-of-range ancestor");
      m_stateScratch.col(j) = m_states.col(ancestor);
    }
    m_states.swap(m_stateScratch);
    m_logWeights.setConstant(newSize, -Eigen::numext::log(Scalar(newSize)));
    return {};
  }

  /** Resamples to the current size(). */
  template <std::uniform_random_bit_generator G, typename R = SystematicResampler>
    requires Resampler<R, Scalar, G>
  [[nodiscard]] std::expected<void, WeightError> resample(G& rng, const R& resampler = {}) {
    return resample(rng, size(), resampler);
  }

  /** \returns the ancestor indices chosen by the last successful resample() */
  [[nodiscard]] std::span<const Index> ancestors() const { return m_ancestors; }

  /** \returns a read-only view of the state entries \a indices; see
   * MutableGaussian::marginal() for the index types accepted. While the marginal is
   * alive this object is borrowed.
   *
   * \sa class ParticleSetMarginal
   */
  template <typename Indices>
  [[nodiscard]] ParticleSetMarginal<MutableParticleSet, Indices> marginal(const Indices& indices) const {
    return ParticleSetMarginal<MutableParticleSet, Indices>{*this, indices};
  }

  /** marginal() of the subspace named by the tag path \a Path (see idx<>), whose space is
   * that subspace with its parameters.
   */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    return internal::tagMarginal<ParticleSetMarginal, Path...>(*this);
  }

  /** \returns a copy of the kernel density estimate's distribution of the entries
   * \a target given that the entries \a given equal \a values.
   *
   * Uses kdeBandwidth(). The result's particles are the conditioned kernel centers,
   * its weights are unnormalized (they gain each kernel's likelihood of \a values),
   * and its stored bandwidth is the exact conditional bandwidth. Always produces a result.
   */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(*this, kdeBandwidth(), target, given, values);
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(*this, kdeBandwidth(), given, values);
  }

  /** conditional() given the subspace named by the tag path \a GivenPath (see idx<>). */
  template <typename... GivenPath, typename Derived>
    requires(sizeof...(GivenPath) > 0)
  [[nodiscard]] auto conditional(const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(*this, kdeBandwidth(), internal::asIndexSet(idx<Space_, GivenPath...>),
                                          values);
  }

  /** Validates the weights (see logTotalWeight()).
   *
   * \returns a ParticleSet view that locks this object for its lifetime, or a WeightError.
   */
  [[nodiscard("the ParticleSet view holds the lock; discarding it releases the lock immediately")]]
  std::expected<ParticleSetType, WeightError> toParticleSet() const {
    const auto total{logTotalWeight()};
    if (!total) return std::unexpected{total.error()};
    m_logTotalWeight = *total;
    return ParticleSetType{this};
  }

  /** Validates the weights and builds a kernel density estimate with kdeBandwidth().
   *
   * \returns a KernelDensity that locks this object for its lifetime, or a
   *          KernelDensityError if the weights cannot be normalized or the bandwidth
   *          is not positive definite.
   */
  [[nodiscard("the KernelDensity holds the lock; discarding it releases the lock immediately")]]
  std::expected<KernelDensityType, KernelDensityError> toKernelDensity() const {
    const auto view{toParticleSet()};
    if (!view) return std::unexpected{KernelDensityError{view.error()}};
    auto kde{kernelDensity(*view)};
    if (!kde) return std::unexpected{KernelDensityError{kde.error()}};
    return std::move(*kde);
  }

 private:
  friend ParticleSetType;
  template <typename>
  friend class internal::Borrow;

  class ScopedLock {
   public:
    explicit ScopedLock(const MutableParticleSet& set) : m_set{set} { ++m_set.m_lockCount; }
    ~ScopedLock() { --m_set.m_lockCount; }
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;

   private:
    const MutableParticleSet& m_set;
  };

  void assertUnlocked() const { EIGEN_FILTER_ASSERT(m_lockCount == 0 && "MutableParticleSet is locked"); }

  void assertUnborrowed() const {
    EIGEN_FILTER_ASSERT(m_borrowCount == 0 && "MutableParticleSet is borrowed by a live marginal");
  }

  void assertIndex([[maybe_unused]] Index i) const {
    EIGEN_FILTER_ASSERT(i >= 0 && i < size() && "particle index out of range");
  }

  /** Sizes a runtime space to the states, checks the size fits, and normalizes the states. */
  void adoptSize() {
    if constexpr (requires { m_space.resize(particleSize()); }) {
      if (m_space.size() != particleSize()) m_space.resize(particleSize());
    }
    checkSpace();
  }

  /** Checks that the states fit the space and normalizes them. */
  void checkSpace() {
    if constexpr (requires { m_space.size(); }) {
      EIGEN_FILTER_ASSERT(m_space.size() == particleSize() && "the space has a different number of coordinates");
    }
    EIGEN_FILTER_ASSERT(isValidSize<Space_>(particleSize()) && "the size does not fit the space");
    m_space.normalizeValues(m_states);
  }

  /** Removes particles whose index satisfies \a removeAtIndex, keeping the rest in order.
   *
   * \returns the log of the sum of the removed weights
   */
  template <typename RemoveAtIndex>
  Scalar compact(RemoveAtIndex removeAtIndex) {
    internal::LogSumExp<Scalar> removed;
    Index out{0};
    for (Index i{0}; i < size(); ++i) {
      if (removeAtIndex(i)) {
        removed.add(m_logWeights(i));
        continue;
      }
      if (out != i) {
        m_states.col(out) = m_states.col(i);
        m_logWeights(out) = m_logWeights(i);
      }
      ++out;
    }
    m_states.conservativeResize(Eigen::NoChange, out);
    m_logWeights.conservativeResize(out);
    return removed.value();
  }

  SampleSetType m_states{Size_ == Eigen::Dynamic ? 0 : Size_, 0};
  WeightArrayType m_logWeights;
  EIGEN_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;
  std::optional<MatrixType> m_bandwidth;
  // Resampling scratch, kept to avoid reallocating on every resample.
  SampleSetType m_stateScratch;
  std::vector<Scalar> m_weightScratch;
  std::vector<Index> m_ancestors;
  // Filled by the const toParticleSet() for use by views.
  mutable Scalar m_logTotalWeight{};
  mutable int m_lockCount{0};
  mutable int m_borrowCount{0};
};

/** \brief Read-only view of a MutableParticleSet whose weights have been validated.
 *
 * Represents the empirical distribution
 * \f[ \hat p(x) = \sum_i \hat w_i \, \delta(x - x_i), \qquad \hat w_i = w_i / \textstyle\sum_j w_j. \f]
 *
 * Obtained from MutableParticleSet::toParticleSet(). Holding a ParticleSet
 * locks the source; copies share the lock and the last one to be destroyed
 * releases it. A ParticleSet must not outlive its source.
 *
 * \sa class MutableParticleSet, class KernelDensity
 */
template <typename Scalar_, int Size_, int MaxSize_, typename Space_>
class ParticleSet {
 public:
  using MutableType = MutableParticleSet<Scalar_, Size_, MaxSize_, Space_>;
  using SpaceType = Space_;
  using MomentsType = typename MutableType::MomentsType;
  using Scalar = typename MutableType::Scalar;
  using Index = typename MutableType::Index;
  using VectorType = typename MutableType::VectorType;
  using MatrixType = typename MutableType::MatrixType;
  using SampleSetType = typename MutableType::SampleSetType;
  using ConstStateType = typename MutableType::ConstStateType;
  using WeightArrayType = typename MutableType::WeightArrayType;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};
  static constexpr bool IsMutable{false};

  ParticleSet(const ParticleSet& other) : m_source{other.m_source} { lock(); }
  ParticleSet(ParticleSet&& other) noexcept : m_source{std::exchange(other.m_source, nullptr)} {}

  ParticleSet& operator=(const ParticleSet& other) {
    if (this != &other) {
      unlock();
      m_source = other.m_source;
      lock();
    }
    return *this;
  }

  ParticleSet& operator=(ParticleSet&& other) noexcept {
    if (this != &other) {
      unlock();
      m_source = std::exchange(other.m_source, nullptr);
    }
    return *this;
  }

  ~ParticleSet() { unlock(); }

  /** \returns the number of particles */
  [[nodiscard]] Index size() const { return source().size(); }
  [[nodiscard]] Index particleSize() const { return source().particleSize(); }
  [[nodiscard]] const Space_& space() const { return source().space(); }
  [[nodiscard]] const SampleSetType& states() const { return source().states(); }
  [[nodiscard]] const WeightArrayType& logWeights() const { return source().logWeights(); }
  [[nodiscard]] ConstStateType state(Index i) const { return source().state(i); }
  [[nodiscard]] Scalar logWeight(Index i) const { return source().logWeight(i); }
  [[nodiscard]] const std::optional<MatrixType>& bandwidth() const { return source().bandwidth(); }
  [[nodiscard]] MatrixType kdeBandwidth() const { return source().kdeBandwidth(); }

  /** \returns the log of the sum of the (unnormalized) weights */
  [[nodiscard]] Scalar logTotalWeight() const { return source().m_logTotalWeight; }

  /** \returns an expression for the normalized linear weights \f$ \hat w_i \f$ */
  [[nodiscard]] auto normalizedWeights() const { return (logWeights() - logTotalWeight()).exp(); }

  /** \returns the effective sample size \f$ 1 / \sum_i \hat w_i^2 \f$, between 1 and size() */
  [[nodiscard]] Scalar effectiveSampleSize() const {
    return internal::effectiveSampleSize(logWeights(), logTotalWeight());
  }

  /** \returns the weighted mean and covariance,
   * \f$ \mu = \sum_i \hat w_i x_i \f$ and \f$ \Sigma = \sum_i \hat w_i (x_i - \mu)(x_i - \mu)^T \f$.
   *
   * The covariance is the plain weighted (biased) estimate and is exactly symmetric.
   * It excludes any kernel bandwidth; see KernelDensity::moments() for that. In a
   * non-linear space these are tangent-space moments.
   */
  [[nodiscard]] MomentsType moments() const {
    return internal::weightedMomentsIn<MomentsType>(space(), states(), logWeights(), logTotalWeight());
  }

  /** \returns a read-only view of the state entries \a indices. It holds a copy of
   * this view, so the source stays locked while the marginal is alive.
   */
  template <typename Indices>
  [[nodiscard]] ParticleSetMarginal<ParticleSet, Indices> marginal(const Indices& indices) const {
    return ParticleSetMarginal<ParticleSet, Indices>{*this, indices};
  }

  /** marginal() of the subspace named by the tag path \a Path; see MutableParticleSet::marginal(). */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    return internal::tagMarginal<ParticleSetMarginal, Path...>(*this);
  }

  /** \returns a copy of the conditional distribution; see MutableParticleSet::conditional(). */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(*this, kdeBandwidth(), target, given, values);
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(*this, kdeBandwidth(), given, values);
  }

  /** conditional() given the subspace named by the tag path \a GivenPath (see idx<>). */
  template <typename... GivenPath, typename Derived>
    requires(sizeof...(GivenPath) > 0)
  [[nodiscard]] auto conditional(const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalParticles(*this, kdeBandwidth(), internal::asIndexSet(idx<Space_, GivenPath...>),
                                          values);
  }

 private:
  friend MutableType;

  explicit ParticleSet(const MutableType* source) : m_source{source} { lock(); }

  [[nodiscard]] const MutableType& source() const {
    EIGEN_FILTER_ASSERT(m_source && "use of a moved-from ParticleSet");
    return *m_source;
  }

  void lock() {
    if (m_source) ++m_source->m_lockCount;
  }

  void unlock() {
    if (m_source) --m_source->m_lockCount;
  }

  const MutableType* m_source;
};

/** \brief Read-only view of a subset of the state entries of a MutableParticleSet
 * or ParticleSet.
 *
 * Obtained from marginal(). From a MutableParticleSet it borrows the source; from a
 * ParticleSet it holds a copy of the view, and so the lock. Weights are those of
 * the source. A stored bandwidth is marginalized too, which keeps the kernel density
 * estimate exact: the marginal of a KDE is the KDE of the marginal particles with
 * bandwidth \f$ H_{aa} \f$.
 *
 * \tparam Source    MutableParticleSet or ParticleSet
 * \tparam Indices   the index type passed to marginal()
 * \tparam SubSpace  space of the selected coordinates
 */
template <typename Source, typename Indices, typename SubSpace>
class ParticleSetMarginal {
  using Selection = internal::IndexedSize<Indices, Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>;

 public:
  using Scalar = typename Source::Scalar;
  using Index = Eigen::Index;
  static constexpr int SizeAtCompileTime{Selection::size};
  static constexpr int MaxSizeAtCompileTime{Selection::maxSize};
  using SpaceType = SubSpace;
  using MutableType = MutableParticleSet<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime, SubSpace>;
  using MatrixType = typename MutableType::MatrixType;
  using WeightArrayType = typename MutableType::WeightArrayType;
  using IndexVectorType = typename Selection::type;

  /** Selects \a indices, projecting the source's space onto them. */
  ParticleSetMarginal(const Source& source, const Indices& indices)
      : m_holder{source},
        m_indices{internal::resolveIndices<Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>(
            source.particleSize(), indices)},
        m_space{internal::projectSpace<SizeAtCompileTime, MaxSizeAtCompileTime>(source.space(), source.particleSize(),
                                                                                m_indices)} {
    EIGEN_FILTER_ASSERT(internal::distinctIndices(m_indices) && "marginal indices must be distinct");
  }

  /** Selects \a indices, whose coordinates form \a space. */
  ParticleSetMarginal(const Source& source, const Indices& indices, SubSpace space)
      : m_holder{source},
        m_indices{internal::resolveIndices<Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>(
            source.particleSize(), indices)},
        m_space{std::move(space)} {
    EIGEN_FILTER_ASSERT(internal::distinctIndices(m_indices) && "marginal indices must be distinct");
  }

  [[nodiscard]] const SubSpace& space() const { return m_space; }

  /** \returns the number of particles */
  [[nodiscard]] Index size() const { return source().size(); }

  /** \returns the dimension of every marginal particle */
  [[nodiscard]] Index particleSize() const { return m_indices.size(); }

  /** \returns the source entries this marginal selects */
  [[nodiscard]] const IndexVectorType& indices() const { return m_indices; }

  [[nodiscard]] auto states() const { return source().states()(m_indices, Eigen::placeholders::all); }
  [[nodiscard]] auto state(Index i) const { return source().state(i)(m_indices); }
  [[nodiscard]] const WeightArrayType& logWeights() const { return source().logWeights(); }
  [[nodiscard]] Scalar logWeight(Index i) const { return source().logWeight(i); }

  /** \returns the marginal of the stored bandwidth, if the source has one */
  [[nodiscard]] std::optional<MatrixType> bandwidth() const {
    if (!source().bandwidth()) return std::nullopt;
    return MatrixType{internal::lowerSymmetricBlock(*source().bandwidth(), m_indices, m_indices)};
  }

  /** \returns an independent MutableParticleSet holding this marginal, with the
   * source's weights and the marginal of any stored bandwidth
   */
  [[nodiscard]] MutableType copy() const {
    MutableType result{m_space, states(), logWeights()};
    if (const auto h{bandwidth()}) result.setBandwidth(*h);
    return result;
  }

 private:
  [[nodiscard]] const Source& source() const { return m_holder.get(); }

  internal::HolderFor<Source> m_holder;
  IndexVectorType m_indices;
  SubSpace m_space;
};

/** \returns Silverman's rule-of-thumb bandwidth
 * \f[ H = \left(\frac{4}{N_\mathrm{eff}\,(d + 2)}\right)^{2/(d+4)} \Sigma, \f]
 * where \f$ \Sigma \f$ is the particles' weighted covariance, \f$ d \f$ their
 * dimension, and \f$ N_\mathrm{eff} \f$ their effective sample size.
 *
 * Near-optimal for roughly Gaussian particle clouds; scale it down for multimodal ones.
 */
template <typename Scalar, int Size, int MaxSize, typename Space>
[[nodiscard]] typename ParticleSet<Scalar, Size, MaxSize, Space>::MatrixType silvermanBandwidth(
    const ParticleSet<Scalar, Size, MaxSize, Space>& particles) {
  return internal::silvermanBandwidth<typename ParticleSet<Scalar, Size, MaxSize, Space>::MomentsType>(
      particles.space(), particles.states(), particles.logWeights());
}

}  // namespace EigenFilter

// KernelDensity completes MutableParticleSet::toKernelDensity(); it includes this header first.
#include "KernelDensity.h"
