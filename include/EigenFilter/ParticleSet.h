#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "Macros.h"
#include "Moments.h"
#include "Resampling.h"
#include "SampleSet.h"
#include "Weights.h"

namespace EigenFilter {

template <typename Scalar_, int Size_, int MaxSize_ = Size_>
class ParticleSet;

/** \brief Mutable set of weighted samples that can be viewed as a ParticleSet.
 *
 * Particle states are the columns of states(); their log-weights are
 * logWeights(). Weights need not be normalized; a log-weight of -inf is a zero
 * weight. Particles are addressed by index. Removal may reorder the remaining
 * particles.
 *
 * toParticleSet() validates the weights and returns a ParticleSet view. While
 * any view is alive this object is locked: mutating, moving, or destroying it
 * is a programmer error caught by \c EIGEN_FILTER_ASSERT. It is also locked
 * while forEach() and removeIf() call back into user code.
 *
 * \tparam Scalar_   scalar type
 * \tparam Size_     state dimension, or \c Eigen::Dynamic
 * \tparam MaxSize_  maximum state dimension
 *
 * \sa class ParticleSet, \ref Resamplers
 */
template <typename Scalar_, int Size_, int MaxSize_ = Size_>
class MutableParticleSet {
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
  using ParticleSetType = ParticleSet<Scalar_, Size_, MaxSize_>;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};

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
    }

   private:
    friend MutableParticleSet;

    MutableParticle(MutableParticleSet& set, Index index) : m_set{set}, m_index{index} {}

    MutableParticleSet& m_set;
    Index m_index;
  };

  MutableParticleSet() = default;

  /** Constructs an empty set whose particles will have dimension \a particleSize. */
  explicit MutableParticleSet(Index particleSize) : m_states{particleSize, 0} {}

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

  /** Copies the particles. The copy starts unlocked. */
  MutableParticleSet(const MutableParticleSet& other) : m_states{other.m_states}, m_logWeights{other.m_logWeights} {}

  /** Moving a locked object is a programmer error; if \c EIGEN_FILTER_ASSERT throws,
   * the exception reaches this \c noexcept boundary and calls \c std::terminate.
   */
  MutableParticleSet(MutableParticleSet&& other) noexcept
      : m_states{std::move(other.m_states)},
        m_logWeights{std::move(other.m_logWeights)},
        m_stateScratch{std::move(other.m_stateScratch)},
        m_weightScratch{std::move(other.m_weightScratch)},
        m_ancestors{std::move(other.m_ancestors)} {
    other.assertUnlocked();
  }

  MutableParticleSet& operator=(const MutableParticleSet& other) {
    assertUnlocked();
    m_states = other.m_states;
    m_logWeights = other.m_logWeights;
    return *this;
  }

  MutableParticleSet& operator=(MutableParticleSet&& other) noexcept {
    assertUnlocked();
    other.assertUnlocked();
    m_states = std::move(other.m_states);
    m_logWeights = std::move(other.m_logWeights);
    return *this;
  }

  ~MutableParticleSet() { assertUnlocked(); }

  /** \returns the number of particles */
  [[nodiscard]] Index size() const { return m_states.cols(); }
  [[nodiscard]] bool empty() const { return size() == 0; }

  /** \returns the dimension of every particle state */
  [[nodiscard]] Index particleSize() const { return m_states.rows(); }

  [[nodiscard]] const SampleSetType& states() const { return m_states; }
  [[nodiscard]] const WeightArrayType& logWeights() const { return m_logWeights; }
  [[nodiscard]] ConstStateType state(Index i) const { return m_states.col(i); }
  [[nodiscard]] Scalar logWeight(Index i) const { return m_logWeights(i); }

  /** Resizes every particle state to dimension \a n. States are unspecified afterward. */
  void resizeParticles(Index n) {
    assertUnlocked();
    m_states.resize(n, size());
  }

  void clear() {
    assertUnlocked();
    m_states.resize(particleSize(), 0);
    m_logWeights.resize(0);
  }

  /** Replaces the particles with the columns of \a states, with equal weights summing to one. */
  template <typename StatesDerived>
  void assign(const Eigen::MatrixBase<StatesDerived>& states) {
    assertUnlocked();
    m_states = states;
    m_logWeights.setConstant(size(), -Eigen::numext::log(Scalar(size())));
  }

  template <typename StatesDerived, typename WeightsDerived>
  void assign(const Eigen::MatrixBase<StatesDerived>& states, const Eigen::DenseBase<WeightsDerived>& logWeights) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(logWeights.size() == states.cols());
    m_states = states;
    m_logWeights = logWeights.derived().array();
  }

  /** Appends a particle. Costs O(size()) because the storage is contiguous.
   *
   * \returns the index of the new particle
   */
  template <typename Derived>
  Index add(Scalar logWeight, const Eigen::MatrixBase<Derived>& state) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(state.size() == particleSize());
    const Index i{size()};
    m_states.conservativeResize(Eigen::NoChange, i + 1);
    m_logWeights.conservativeResize(i + 1);
    m_states.col(i) = state;
    m_logWeights(i) = logWeight;
    return i;
  }

  /** Appends copies of \a other's particles, which must have the same dimension. */
  void append(const MutableParticleSet& other) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(other.particleSize() == particleSize());
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
  }

  /** Sets every state at once. \a states must have the current dimensions. */
  template <typename Derived>
  void setStates(const Eigen::MatrixBase<Derived>& states) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(states.rows() == particleSize() && states.cols() == size());
    m_states = states;
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

 private:
  friend ParticleSetType;

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

  void assertIndex([[maybe_unused]] Index i) const {
    EIGEN_FILTER_ASSERT(i >= 0 && i < size() && "particle index out of range");
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
  // Resampling scratch, kept to avoid reallocating on every resample.
  SampleSetType m_stateScratch;
  std::vector<Scalar> m_weightScratch;
  std::vector<Index> m_ancestors;
  // Filled by the const toParticleSet() for use by views.
  mutable Scalar m_logTotalWeight{};
  mutable int m_lockCount{0};
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
 * \sa class MutableParticleSet
 */
template <typename Scalar_, int Size_, int MaxSize_>
class ParticleSet {
 public:
  using MutableType = MutableParticleSet<Scalar_, Size_, MaxSize_>;
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
  [[nodiscard]] const SampleSetType& states() const { return source().states(); }
  [[nodiscard]] const WeightArrayType& logWeights() const { return source().logWeights(); }
  [[nodiscard]] ConstStateType state(Index i) const { return source().state(i); }
  [[nodiscard]] Scalar logWeight(Index i) const { return source().logWeight(i); }

  /** \returns the log of the sum of the (unnormalized) weights */
  [[nodiscard]] Scalar logTotalWeight() const { return source().m_logTotalWeight; }

  /** \returns an expression for the normalized linear weights \f$ \hat w_i \f$ */
  [[nodiscard]] auto normalizedWeights() const { return (logWeights() - logTotalWeight()).exp(); }

  /** \returns the effective sample size \f$ 1 / \sum_i \hat w_i^2 \f$, between 1 and size() */
  [[nodiscard]] Scalar effectiveSampleSize() const { return Scalar(1) / normalizedWeights().square().sum(); }

  /** \returns the weighted mean and covariance,
   * \f$ \mu = \sum_i \hat w_i x_i \f$ and \f$ \Sigma = \sum_i \hat w_i (x_i - \mu)(x_i - \mu)^T \f$.
   *
   * The covariance is the plain weighted (biased) estimate and is exactly symmetric.
   */
  [[nodiscard]] MomentsType moments() const {
    const WeightArrayType w{normalizedWeights()};
    MomentsType result{particleSize()};
    result.mean() = states() * w.matrix();
    const SampleSetType scaled{(states().colwise() - result.mean()) * w.sqrt().matrix().asDiagonal()};
    auto& cov{result.cov()};
    if constexpr (MaxSize_ == 1) {
      // rankUpdate() would treat the single-row matrix as a column vector.
      cov.setConstant(scaled.squaredNorm());
    } else {
      cov.setZero();
      cov.template selfadjointView<Eigen::Lower>().rankUpdate(scaled);
      for (Index j{1}; j < cov.cols(); ++j) {
        for (Index i{0}; i < j; ++i) cov(i, j) = cov(j, i);
      }
    }
    return result;
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

}  // namespace EigenFilter
