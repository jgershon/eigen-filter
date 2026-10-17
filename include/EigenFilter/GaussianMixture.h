#pragma once

#include <algorithm>
#include <expected>
#include <functional>
#include <iterator>
#include <random>
#include <span>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "Gaussian.h"
#include "Indexing.h"
#include "Macros.h"
#include "Moments.h"
#include "Resampling.h"
#include "SampleSet.h"
#include "Sampling.h"
#include "Space.h"
#include "Statistics.h"
#include "Weights.h"

namespace EigenFilter {

/** A mixture component whose covariance could not be factorized. */
struct ComponentError {
  Eigen::Index index;   ///< position of the component in the mixture
  GaussianError error;  ///< why factorization failed

  friend bool operator==(const ComponentError&, const ComponentError&) = default;
};

/** Reasons GaussianMixture::toDensity() can fail. */
using GaussianMixtureError = std::variant<WeightError, ComponentError>;

template <typename Scalar_, typename Space_>
class GaussianMixtureDensity;

template <typename Scalar_, typename Space_>
class GaussianMixture;

template <typename Source, typename Indices,
          typename SubSpace = internal::ProjectedSpace<
              typename Source::SpaceType,
              internal::IndexedSize<Indices, Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>::size,
              internal::IndexedSize<Indices, Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>::maxSize>>
class GaussianMixtureMarginal;

namespace internal {

/** Shared implementation of GaussianMixture::conditional() and GaussianMixtureDensity::conditional(). */
template <typename Source, typename TargetIndices, typename GivenIndices, typename Derived>
[[nodiscard]] auto conditionalMixture(const Source& source, const TargetIndices& targetIndices,
                                      const GivenIndices& givenIndices, const Eigen::MatrixBase<Derived>& values) {
  constexpr int SourceSize{Source::SizeAtCompileTime};
  constexpr int SourceMaxSize{Source::MaxSizeAtCompileTime};
  const auto target{resolveIndices<SourceSize, SourceMaxSize>(source.componentSize(), targetIndices)};
  const auto given{resolveIndices<SourceSize, SourceMaxSize>(source.componentSize(), givenIndices)};
  EIGEN_FILTER_ASSERT(distinctIndices(target, given) && "target and given indices must be distinct");
  using Target = std::remove_cvref_t<decltype(target)>;
  using Given = std::remove_cvref_t<decltype(given)>;
  auto targetSpace{projectSpace<Target::RowsAtCompileTime, Target::MaxRowsAtCompileTime>(
      source.space(), source.componentSize(), target)};
  const auto givenSpace{projectSpace<Given::RowsAtCompileTime, Given::MaxRowsAtCompileTime>(
      source.space(), source.componentSize(), given)};
  GaussianMixture<typename Source::Scalar, decltype(targetSpace)> result{targetSpace, target.size()};
  result.reserve(source.size());
  for (const auto& component : source) {
    auto conditioned{conditionMoments(component.gaussian.mean(), component.gaussian.cov(), target, given, values,
                                      givenSpace, targetSpace)};
    result.add(component.logWeight + conditioned.logLikelihood, std::move(conditioned.mean),
               std::move(conditioned.cov));
  }
  return result;
}

/** conditionalMixture() with every entry not in \a givenIndices as the target. */
template <typename Source, typename GivenIndices, typename Derived>
[[nodiscard]] auto conditionalMixture(const Source& source, const GivenIndices& givenIndices,
                                      const Eigen::MatrixBase<Derived>& values) {
  constexpr int SourceSize{Source::SizeAtCompileTime};
  constexpr int SourceMaxSize{Source::MaxSizeAtCompileTime};
  const auto given{resolveIndices<SourceSize, SourceMaxSize>(source.componentSize(), givenIndices)};
  return conditionalMixture(source, complementIndices<SourceSize, SourceMaxSize>(source.componentSize(), given), given,
                            values);
}

}  // namespace internal

/** \brief A weighted sum of Gaussians, all in one space.
 *
 * Each component has a log-weight and a Gaussian. Weights need not be
 * normalized; a log-weight of -inf is a zero weight. All components share the
 * dimension componentSize().
 *
 * Components are read through const iterators and changed only through this
 * object's methods. Adding or removing components invalidates all iterators,
 * and removal may reorder the remaining components.
 *
 * toDensity() validates the weights, factorizes every component whose covariance has
 * changed, and returns a GaussianMixtureDensity. While any GaussianMixtureDensity is
 * alive this object is locked: mutating, moving, or destroying it is a programmer
 * error caught by \c EIGEN_FILTER_ASSERT. It is also locked while forEach() and
 * removeIf() call back into user code.
 *
 * marginal() returns a read-only view of a subset of every component's entries.
 * While any marginal is alive this object is borrowed: values and weights may
 * change, but adding, removing, or resizing components, or moving or destroying
 * this object, is a programmer error.
 *
 * Every component shares the mixture's space (see \ref Spaces): component means are
 * normalized when set, and moments() averages in the space's tangent space.
 *
 * \tparam Scalar_  scalar type
 * \tparam Space_   space of the components, which sets their dimension and storage
 *                  bound (see \ref Spaces)
 *
 * \sa class GaussianMixtureDensity, class Gaussian, GaussianMixtureN
 */
template <typename Scalar_, typename Space_>
class GaussianMixture {
 public:
  using SpaceType = Space_;
  using ComponentGaussianType = Gaussian<Scalar_, Space_>;
  using MomentsType = typename ComponentGaussianType::MomentsType;
  using Scalar = typename ComponentGaussianType::Scalar;
  using Index = typename ComponentGaussianType::Index;
  using VectorType = typename ComponentGaussianType::VectorType;
  using MatrixType = typename ComponentGaussianType::MatrixType;
  using DensityType = GaussianMixtureDensity<Scalar_, Space_>;
  static constexpr int SizeAtCompileTime{ComponentGaussianType::SizeAtCompileTime};
  static constexpr int MaxSizeAtCompileTime{ComponentGaussianType::MaxSizeAtCompileTime};
  static constexpr bool IsOwner{true};

  /** One weighted component. Only ever exposed as const outside the mixture. */
  struct Component {
    Scalar logWeight;
    ComponentGaussianType gaussian;
  };

  using const_iterator = typename std::vector<Component>::const_iterator;

  /** Result of removeIf() and truncate(): the log of the removed fraction of the
   * total weight, or nothing when \a ReturnRemovedWeight is false.
   */
  template <bool ReturnRemovedWeight>
  using RemovalResult = std::conditional_t<ReturnRemovedWeight, std::expected<Scalar, WeightError>, void>;

  /** \brief Writable access to one component during forEach().
   *
   * Exists only for the duration of the callback; it cannot be copied or moved.
   * The component's dimension cannot be changed through it.
   */
  class MutableComponent {
   public:
    MutableComponent(const MutableComponent&) = delete;
    MutableComponent& operator=(const MutableComponent&) = delete;

    [[nodiscard]] Scalar logWeight() const { return m_component.logWeight; }
    [[nodiscard]] const ComponentGaussianType& gaussian() const { return m_component.gaussian; }
    [[nodiscard]] const MomentsType& moments() const { return gaussian().moments(); }
    [[nodiscard]] const VectorType& mean() const { return gaussian().mean(); }
    [[nodiscard]] const MatrixType& cov() const { return gaussian().cov(); }

    template <typename Derived>
    void setMean(const Eigen::MatrixBase<Derived>& mean) {
      m_component.gaussian.setMean(mean);
    }

    template <typename Derived>
    void setCov(const Eigen::MatrixBase<Derived>& cov) {
      m_component.gaussian.setCov(cov);
    }

    void setMoments(const MomentsType& moments) { GaussianMixture::assignMoments(m_component, moments); }

   private:
    friend GaussianMixture;

    explicit MutableComponent(Component& component) : m_component{component} {}

    Component& m_component;
  };

  GaussianMixture() = default;

  /** Constructs an empty mixture whose components will have dimension \a componentSize. */
  explicit GaussianMixture(Index componentSize) : m_componentSize{componentSize} {
    if constexpr (requires { m_space.resize(componentSize); }) m_space.resize(componentSize);
    assertValidComponentSize(componentSize);
  }

  /** Constructs an empty mixture in \a space whose components will have dimension \a componentSize. */
  GaussianMixture(Space_ space, Index componentSize) : m_componentSize{componentSize}, m_space{std::move(space)} {
    assertValidComponentSize(componentSize);
  }

  /** Copies the components and their factorizations. The copy starts unlocked. */
  GaussianMixture(const GaussianMixture& other)
      : m_components{other.m_components}, m_componentSize{other.m_componentSize}, m_space{other.m_space} {}

  /** Moving a locked object is a programmer error; if \c EIGEN_FILTER_ASSERT throws,
   * the exception reaches this \c noexcept boundary and calls \c std::terminate.
   */
  GaussianMixture(GaussianMixture&& other) noexcept
      : m_components{std::move(other.m_components)},
        m_componentSize{other.m_componentSize},
        m_space{std::move(other.m_space)} {
    other.assertUnlocked();
    other.assertUnborrowed();
  }

  GaussianMixture& operator=(const GaussianMixture& other) {
    assertUnlocked();
    assertUnborrowed();
    m_components = other.m_components;
    m_componentSize = other.m_componentSize;
    m_space = other.m_space;
    return *this;
  }

  GaussianMixture& operator=(GaussianMixture&& other) noexcept {
    assertUnlocked();
    assertUnborrowed();
    other.assertUnlocked();
    other.assertUnborrowed();
    m_components = std::move(other.m_components);
    m_componentSize = other.m_componentSize;
    m_space = std::move(other.m_space);
    return *this;
  }

  ~GaussianMixture() {
    assertUnlocked();
    assertUnborrowed();
  }

  /** \returns the number of components */
  [[nodiscard]] Index size() const { return Index(m_components.size()); }
  [[nodiscard]] bool empty() const { return m_components.empty(); }

  /** \returns the dimension of every component */
  [[nodiscard]] Index componentSize() const { return m_componentSize; }

  [[nodiscard]] const Space_& space() const { return m_space; }

  /** Resizes every component to dimension \a n, which must fit the space. Component
   * contents are unspecified afterward.
   */
  void resizeComponents(Index n) {
    assertUnlocked();
    assertUnborrowed();
    if constexpr (requires { m_space.resize(n); }) m_space.resize(n);
    assertValidComponentSize(n);
    m_componentSize = n;
    for (auto& component : m_components) component.gaussian.resize(n);
  }

  void reserve(Index n) {
    assertUnlocked();
    m_components.reserve(std::size_t(n));
  }

  void clear() {
    assertUnlocked();
    assertUnborrowed();
    m_components.clear();
  }

  [[nodiscard]] const_iterator begin() const { return m_components.cbegin(); }
  [[nodiscard]] const_iterator end() const { return m_components.cend(); }

  /** Appends a component. \returns an iterator to it. */
  const_iterator add(Scalar logWeight, MomentsType moments) {
    assertUnlocked();
    assertUnborrowed();
    EIGEN_FILTER_ASSERT(moments.size() == m_componentSize);
    m_components.push_back(Component{logWeight, ComponentGaussianType{m_space, std::move(moments)}});
    return std::prev(m_components.cend());
  }

  template <typename MeanDerived, typename CovDerived>
  const_iterator add(Scalar logWeight, const Eigen::MatrixBase<MeanDerived>& mean,
                     const Eigen::MatrixBase<CovDerived>& cov) {
    return add(logWeight, MomentsType{mean, cov});
  }

  /** Removes the component at \a pos by moving the last component into its place.
   *
   * \returns an iterator to the component now at \a pos's position, or end()
   */
  const_iterator remove(const_iterator pos) {
    assertUnlocked();
    assertUnborrowed();
    const Index i{indexOf(pos)};
    removeAt(i);
    return m_components.cbegin() + i;
  }

  void setMoments(const_iterator pos, const MomentsType& moments) { assignMoments(mutableAt(pos), moments); }

  template <typename Derived>
  void setMean(const_iterator pos, const Eigen::MatrixBase<Derived>& mean) {
    mutableAt(pos).gaussian.setMean(mean);
  }

  template <typename Derived>
  void setCov(const_iterator pos, const Eigen::MatrixBase<Derived>& cov) {
    mutableAt(pos).gaussian.setCov(cov);
  }

  void setLogWeight(const_iterator pos, Scalar logWeight) { mutableAt(pos).logWeight = logWeight; }

  /** Multiplies the weight at \a pos by \f$ e^{\mathrm{logScale}} \f$. */
  void scaleWeight(const_iterator pos, Scalar logScale) { mutableAt(pos).logWeight += logScale; }

  /** Multiplies every weight by \f$ e^{\mathrm{logScale}} \f$. */
  void scaleWeights(Scalar logScale) {
    assertUnlocked();
    for (auto& component : m_components) component.logWeight += logScale;
  }

  /** \returns the log of the sum of the weights, or a WeightError if the mixture
   *          is empty, the weights sum to zero, or a log-weight is NaN or +inf.
   */
  [[nodiscard]] std::expected<Scalar, WeightError> logTotalWeight() const {
    return internal::logTotalWeight<Scalar>(m_components | std::views::transform(&Component::logWeight));
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

  /** Replaces the components with those in \a components, a range of
   * (log-weight, moments) pairs or other structured-bindable pairs.
   */
  template <std::ranges::input_range R>
  void assign(R&& components) {
    assertUnlocked();
    assertUnborrowed();
    m_components.clear();
    if constexpr (std::ranges::sized_range<R>) m_components.reserve(std::ranges::size(components));
    for (auto&& [logWeight, moments] : components) add(logWeight, moments);
  }

  /** Appends copies of \a other's components, which must have the same dimension. */
  void append(const GaussianMixture& other) {
    assertUnlocked();
    assertUnborrowed();
    EIGEN_FILTER_ASSERT(other.m_componentSize == m_componentSize);
    EIGEN_FILTER_ASSERT(other.m_space == m_space && "cannot append a mixture in a different space");
    const std::size_t n{other.m_components.size()};
    m_components.reserve(m_components.size() + n);
    for (std::size_t i{0}; i < n; ++i) m_components.push_back(other.m_components[i]);
  }

  /** Calls \a f on each component as a \c MutableComponent&.
   *
   * If \a f returns a value, it is a log scale factor added to that component's
   * log-weight. This mixture is locked while \a f runs, so \a f may modify the
   * component it is given but not the mixture.
   */
  template <typename F>
  void forEach(F&& f) {
    assertUnlocked();
    const ScopedLock lock{*this};
    for (auto& component : m_components) {
      MutableComponent handle{component};
      if constexpr (std::is_void_v<std::invoke_result_t<F&, MutableComponent&>>) {
        std::invoke(f, handle);
      } else {
        component.logWeight += std::invoke(f, handle);
      }
    }
  }

  /** Merges \a absorb into \a keep.
   *
   * \a keep's moments become \a moments and its weight becomes the sum of both
   * weights; \a absorb is then removed. \a moments may refer to either component.
   *
   * \returns an iterator to the merged component
   */
  const_iterator merge(const_iterator keep, const_iterator absorb, const MomentsType& moments) {
    assertUnlocked();
    assertUnborrowed();
    const Index k{indexOf(keep)};
    const Index a{indexOf(absorb)};
    EIGEN_FILTER_ASSERT(k != a && "cannot merge a component with itself");
    internal::LogSumExp<Scalar> weight;
    weight.add(at(k).logWeight);
    weight.add(at(a).logWeight);
    at(k).logWeight = weight.value();
    assignMoments(at(k), moments);
    const bool keepWasLast{k == size() - 1};
    removeAt(a);
    return m_components.cbegin() + (keepWasLast ? a : k);
  }

  /** Replaces the component at \a pos with children.
   *
   * \a children is a non-empty range of (log relative weight, moments) pairs. The
   * parent's weight is divided among the children in proportion to their relative
   * weights. The first child takes the parent's position; the rest are appended.
   * \a children must not refer to this mixture's components.
   */
  template <std::ranges::forward_range R>
  void split(const_iterator pos, R&& children) {
    assertUnlocked();
    assertUnborrowed();
    const Index i{indexOf(pos)};
    EIGEN_FILTER_ASSERT(!std::ranges::empty(children));
    internal::LogSumExp<Scalar> relativeTotal;
    for ([[maybe_unused]] auto&& [logRelativeWeight, moments] : children) relativeTotal.add(logRelativeWeight);
    const Scalar logTotal{relativeTotal.value()};
    EIGEN_FILTER_ASSERT(Eigen::numext::isfinite(logTotal) && "relative weights must have a finite, nonzero sum");
    const Scalar logBase{at(i).logWeight - logTotal};
    bool first{true};
    for (auto&& [logRelativeWeight, moments] : children) {
      if (first) {
        at(i).logWeight = logBase + logRelativeWeight;
        assignMoments(at(i), moments);
        first = false;
      } else {
        add(logBase + logRelativeWeight, moments);
      }
    }
  }

  /** Removes every component for which \a pred(const Component&) is true.
   *
   * \tparam ReturnRemovedWeight  if true, returns the log of the removed fraction
   *         of the total weight. This requires normalizing the weights; if that
   *         fails, nothing is removed and the WeightError is returned.
   */
  template <bool ReturnRemovedWeight = false, typename Pred>
  RemovalResult<ReturnRemovedWeight> removeIf(Pred pred) {
    assertUnlocked();
    assertUnborrowed();
    const auto removeAtIndex{[&](Index i) { return static_cast<bool>(std::invoke(pred, std::as_const(at(i)))); }};
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

  /** Removes the lowest-weighted components until at most \a maxSize remain.
   *
   * \tparam ReturnRemovedWeight  as for removeIf()
   */
  template <bool ReturnRemovedWeight = false>
  RemovalResult<ReturnRemovedWeight> truncate(Index maxSize) {
    assertUnlocked();
    assertUnborrowed();
    EIGEN_FILTER_ASSERT(maxSize >= 0);
    [[maybe_unused]] Scalar logTotal{0};
    if constexpr (ReturnRemovedWeight) {
      const auto total{logTotalWeight()};
      if (!total) return std::unexpected{total.error()};
      logTotal = *total;
    }
    [[maybe_unused]] Scalar logRemoved{-Eigen::NumTraits<Scalar>::infinity()};
    if (size() > maxSize) {
      std::vector<Index> order{indicesByWeight()};
      const auto keepEnd{order.begin() + maxSize};
      std::nth_element(order.begin(), keepEnd, order.end(), heavierFirst());
      std::vector<bool> drop(m_components.size(), false);
      for (auto it{keepEnd}; it != order.end(); ++it) drop[std::size_t(*it)] = true;
      logRemoved = compact([&](Index i) { return bool{drop[std::size_t(i)]}; });
    }
    if constexpr (ReturnRemovedWeight) return logRemoved - logTotal;
  }

  /** Removes the lowest-weighted components whose combined weight, as a fraction
   * of the total, is at most \f$ e^{\mathrm{maxLogFraction}} \f$.
   *
   * \returns the log of the removed fraction of the total weight, or a WeightError
   *          (with nothing removed) if the weights cannot be normalized.
   */
  [[nodiscard]] std::expected<Scalar, WeightError> removeLightest(Scalar maxLogFraction) {
    assertUnlocked();
    assertUnborrowed();
    const auto total{logTotalWeight()};
    if (!total) return std::unexpected{total.error()};
    std::vector<Index> order{indicesByWeight()};
    std::ranges::sort(order, [cmp = heavierFirst()](Index a, Index b) { return cmp(b, a); });
    std::vector<bool> drop(m_components.size(), false);
    internal::LogSumExp<Scalar> removed;
    for (const Index i : order) {
      auto next{removed};
      next.add(at(i).logWeight);
      if (next.value() - *total > maxLogFraction) break;
      removed = next;
      drop[std::size_t(i)] = true;
    }
    compact([&](Index i) { return bool{drop[std::size_t(i)]}; });
    return removed.value() - *total;
  }

  /** \returns a read-only view of the entries \a indices of every component; see
   * Gaussian::marginal() for the index types accepted. While the marginal is
   * alive this object is borrowed.
   *
   * \sa class GaussianMixtureMarginal
   */
  template <typename Indices>
  [[nodiscard]] GaussianMixtureMarginal<GaussianMixture, Indices> marginal(const Indices& indices) const {
    return GaussianMixtureMarginal<GaussianMixture, Indices>{*this, indices};
  }

  /** marginal() of the subspace named by the tag path \a Path (see idx<>), whose space is
   * that subspace with its parameters.
   */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    return internal::tagMarginal<GaussianMixtureMarginal, Path...>(*this);
  }

  /** \returns a copy of the distribution of the entries \a target given that the
   * entries \a given equal \a values.
   *
   * Each component is conditioned as in Gaussian::conditional(), and its
   * log-weight gains the log-likelihood of \a values under that component's marginal.
   * The weights are not normalized, so the result's total weight is this mixture's
   * total weight times the marginal density of \a values. Always produces a result.
   */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalMixture(*this, target, given, values);
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalMixture(*this, given, values);
  }

  /** conditional() given the subspace named by the tag path \a GivenPath (see idx<>). */
  template <typename... GivenPath, typename Derived>
    requires(sizeof...(GivenPath) > 0)
  [[nodiscard]] auto conditional(const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalMixture(*this, internal::asIndexSet(idx<Space_, GivenPath...>), values);
  }

  /** Validates the weights and factorizes every component whose covariance has changed.
   *
   * Does not lock this object.
   *
   * \returns nothing on success; a WeightError if the mixture is empty or its
   *          weights cannot be normalized; or a ComponentError naming the first
   *          component that failed to factorize.
   *
   * \sa toDensity()
   */
  [[nodiscard]] std::expected<void, GaussianMixtureError> factorize() const {
    const auto total{logTotalWeight()};
    if (!total) return std::unexpected{GaussianMixtureError{total.error()}};
    for (Index i{0}; i < size(); ++i) {
      if (auto result{at(i).gaussian.factorize()}; !result) {
        return std::unexpected{GaussianMixtureError{ComponentError{i, result.error()}}};
      }
    }
    m_logTotalWeight = *total;
    return {};
  }

  /** Factorizes as needed (see factorize()).
   *
   * \returns a GaussianMixtureDensity that locks this object for its lifetime, or
   *          the GaussianMixtureError from factorize().
   */
  [[nodiscard("the GaussianMixtureDensity holds the lock; discarding it releases the lock immediately")]]
  std::expected<DensityType, GaussianMixtureError> toDensity() const {
    if (auto result{factorize()}; !result) return std::unexpected{result.error()};
    return DensityType{this};
  }

 private:
  friend DensityType;
  template <typename>
  friend class internal::Borrow;

  class ScopedLock {
   public:
    explicit ScopedLock(const GaussianMixture& mixture) : m_mixture{mixture} { ++m_mixture.m_lockCount; }
    ~ScopedLock() { --m_mixture.m_lockCount; }
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;

   private:
    const GaussianMixture& m_mixture;
  };

  static void assertValidComponentSize(Index n) {
    EIGEN_FILTER_ASSERT(n >= 0 && (SizeAtCompileTime == Eigen::Dynamic || n == SizeAtCompileTime) &&
                        (MaxSizeAtCompileTime == Eigen::Dynamic || n <= MaxSizeAtCompileTime));
    EIGEN_FILTER_ASSERT(isValidSize<Space_>(n) && "the size does not fit the space");
  }

  static void assignMoments(Component& component, const MomentsType& moments) {
    component.gaussian.setMean(moments.mean());
    component.gaussian.setCov(moments.cov());
  }

  void assertUnlocked() const { EIGEN_FILTER_ASSERT(m_lockCount == 0 && "GaussianMixture is locked"); }

  void assertUnborrowed() const {
    EIGEN_FILTER_ASSERT(m_borrowCount == 0 && "GaussianMixture is borrowed by a live marginal");
  }

  [[nodiscard]] Index indexOf(const_iterator pos) const {
    const Index i{pos - m_components.cbegin()};
    EIGEN_FILTER_ASSERT(i >= 0 && i < size() && "iterator does not refer to a component of this mixture");
    return i;
  }

  [[nodiscard]] Component& at(Index i) { return m_components[std::size_t(i)]; }
  [[nodiscard]] const Component& at(Index i) const { return m_components[std::size_t(i)]; }

  [[nodiscard]] Component& mutableAt(const_iterator pos) {
    assertUnlocked();
    return at(indexOf(pos));
  }

  void removeAt(Index i) {
    if (i != size() - 1) at(i) = std::move(m_components.back());
    m_components.pop_back();
  }

  [[nodiscard]] std::vector<Index> indicesByWeight() const {
    std::vector<Index> order(m_components.size());
    std::iota(order.begin(), order.end(), Index{0});
    return order;
  }

  /** Orders component indices by descending weight, treating NaN as lowest. */
  [[nodiscard]] auto heavierFirst() const {
    return [this](Index a, Index b) {
      const auto key{[this](Index i) {
        const Scalar w{at(i).logWeight};
        return Eigen::numext::isnan(w) ? -Eigen::NumTraits<Scalar>::infinity() : w;
      }};
      return key(a) > key(b);
    };
  }

  /** Removes components whose index satisfies \a removeAtIndex, keeping the rest in order.
   *
   * \returns the log of the sum of the removed weights
   */
  template <typename RemoveAtIndex>
  Scalar compact(RemoveAtIndex removeAtIndex) {
    internal::LogSumExp<Scalar> removed;
    Index out{0};
    for (Index i{0}; i < size(); ++i) {
      if (removeAtIndex(i)) {
        removed.add(at(i).logWeight);
        continue;
      }
      if (out != i) at(out) = std::move(at(i));
      ++out;
    }
    m_components.erase(m_components.begin() + out, m_components.end());
    return removed.value();
  }

  std::vector<Component> m_components;
  Index m_componentSize{SizeAtCompileTime == Eigen::Dynamic ? 0 : SizeAtCompileTime};
  EIGEN_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;
  // Filled by the const factorize() for use by views.
  mutable Scalar m_logTotalWeight{};
  mutable int m_lockCount{0};
  mutable int m_borrowCount{0};
};

/** \brief The density of a GaussianMixture whose components have been factorized: a
 * read-only view.
 *
 * Represents the normalized density
 * \f[ p(x) = \frac{\sum_i w_i \, \mathcal{N}(x; \mu_i, \Sigma_i)}{\sum_i w_i}. \f]
 *
 * Obtained from GaussianMixture::toDensity(). Holding a GaussianMixtureDensity locks
 * the source; copies share the lock and the last one to be destroyed releases it. A
 * GaussianMixtureDensity must not outlive its source.
 *
 * \sa class GaussianMixture, GaussianMixtureDensityN
 */
template <typename Scalar_, typename Space_>
class GaussianMixtureDensity {
 public:
  using OwnerType = GaussianMixture<Scalar_, Space_>;
  using SpaceType = Space_;
  using Component = typename OwnerType::Component;
  using const_iterator = typename OwnerType::const_iterator;
  using MomentsType = typename OwnerType::MomentsType;
  using Scalar = typename OwnerType::Scalar;
  using Index = typename OwnerType::Index;
  using VectorType = typename OwnerType::VectorType;
  using MatrixType = typename OwnerType::MatrixType;
  static constexpr int SizeAtCompileTime{OwnerType::SizeAtCompileTime};
  static constexpr int MaxSizeAtCompileTime{OwnerType::MaxSizeAtCompileTime};
  static constexpr bool IsOwner{false};

  GaussianMixtureDensity(const GaussianMixtureDensity& other) : m_source{other.m_source} { lock(); }
  GaussianMixtureDensity(GaussianMixtureDensity&& other) noexcept : m_source{std::exchange(other.m_source, nullptr)} {}

  GaussianMixtureDensity& operator=(const GaussianMixtureDensity& other) {
    if (this != &other) {
      unlock();
      m_source = other.m_source;
      lock();
    }
    return *this;
  }

  GaussianMixtureDensity& operator=(GaussianMixtureDensity&& other) noexcept {
    if (this != &other) {
      unlock();
      m_source = std::exchange(other.m_source, nullptr);
    }
    return *this;
  }

  ~GaussianMixtureDensity() { unlock(); }

  /** \returns the number of components */
  [[nodiscard]] Index size() const { return source().size(); }
  [[nodiscard]] Index componentSize() const { return source().componentSize(); }
  [[nodiscard]] const Space_& space() const { return source().space(); }
  [[nodiscard]] const_iterator begin() const { return source().begin(); }
  [[nodiscard]] const_iterator end() const { return source().end(); }

  /** \returns the log of the sum of the (unnormalized) component weights */
  [[nodiscard]] Scalar logTotalWeight() const { return source().m_logTotalWeight; }

  /** \returns the effective sample size \f$ 1 / \sum_i \hat w_i^2 \f$ of the normalized
   * weights, between 1 and size()
   */
  [[nodiscard]] Scalar effectiveSampleSize() const {
    Scalar sumOfSquares{0};
    for (const auto& component : *this) sumOfSquares += Eigen::numext::abs2(normalizedWeight(component));
    return Scalar(1) / sumOfSquares;
  }

  /** \returns the mean and covariance of the mixture,
   * \f$ \mu = \sum_i \hat w_i \mu_i \f$ and
   * \f$ \Sigma = \sum_i \hat w_i \left(\Sigma_i + (\mu_i - \mu)(\mu_i - \mu)^T\right) \f$,
   * where \f$ \hat w_i \f$ are the normalized weights. In a non-linear space the mean is
   * a tangent-space mean and the deviations are \f$ \mu_i \boxminus \mu \f$.
   *
   * Only the lower triangle of each component covariance is read; the result is symmetric.
   */
  [[nodiscard]] MomentsType moments() const {
    SampleSet<Scalar, Space_> means{componentSize(), size()};
    Eigen::Array<Scalar, Eigen::Dynamic, 1> logWeights{size()};
    Index i{0};
    for (const auto& component : *this) {
      means.col(i) = component.gaussian.mean();
      logWeights(i) = component.logWeight;
      ++i;
    }
    MomentsType result{internal::weightedMomentsIn<MomentsType>(space(), means, logWeights, logTotalWeight())};
    auto& cov{result.cov()};
    for (const auto& component : *this) {
      cov.template triangularView<Eigen::Lower>() += normalizedWeight(component) * component.gaussian.cov();
    }
    for (Index j{1}; j < cov.cols(); ++j) {
      for (Index r{0}; r < j; ++r) cov(r, j) = cov(j, r);
    }
    return result;
  }

  /** \returns the log of the density at \a x
   *
   * \sa pdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar logPdf(const Eigen::MatrixBase<Derived>& x) const {
    static_assert(Derived::IsVectorAtCompileTime, "logPdf() expects a vector");
    EIGEN_FILTER_ASSERT(x.size() == componentSize());
    internal::LogSumExp<Scalar> sum;
    for (const auto& component : *this) {
      if (component.logWeight == -Eigen::NumTraits<Scalar>::infinity()) continue;
      sum.add(component.logWeight + component.gaussian.toDensity()->logPdf(x));
    }
    return sum.value() - logTotalWeight();
  }

  /** \returns the density \f$ p(x) \f$ at \a x
   *
   * \warning Underflows easily in high dimensions; prefer logPdf().
   *
   * \sa logPdf()
   */
  template <typename Derived>
  [[nodiscard]] Scalar pdf(const Eigen::MatrixBase<Derived>& x) const {
    return Eigen::numext::exp(logPdf(x));
  }

  /** \returns a read-only view of the entries \a indices of every component. It holds
   * a copy of this view, so the source stays locked while the marginal is alive.
   */
  template <typename Indices>
  [[nodiscard]] GaussianMixtureMarginal<GaussianMixtureDensity, Indices> marginal(const Indices& indices) const {
    return GaussianMixtureMarginal<GaussianMixtureDensity, Indices>{*this, indices};
  }

  /** marginal() of the subspace named by the tag path \a Path; see GaussianMixture::marginal(). */
  template <typename... Path>
    requires(sizeof...(Path) > 0)
  [[nodiscard]] auto marginal() const {
    return internal::tagMarginal<GaussianMixtureMarginal, Path...>(*this);
  }

  /** \returns a copy of the conditional distribution; see GaussianMixture::conditional(). */
  template <typename TargetIndices, typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const TargetIndices& target, const GivenIndices& given,
                                 const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalMixture(*this, target, given, values);
  }

  /** conditional() with every entry not in \a given as the target. */
  template <typename GivenIndices, typename Derived>
  [[nodiscard]] auto conditional(const GivenIndices& given, const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalMixture(*this, given, values);
  }

  /** conditional() given the subspace named by the tag path \a GivenPath (see idx<>). */
  template <typename... GivenPath, typename Derived>
    requires(sizeof...(GivenPath) > 0)
  [[nodiscard]] auto conditional(const Eigen::MatrixBase<Derived>& values) const {
    return internal::conditionalMixture(*this, internal::asIndexSet(idx<Space_, GivenPath...>), values);
  }

 private:
  friend OwnerType;

  explicit GaussianMixtureDensity(const OwnerType* source) : m_source{source} { lock(); }

  [[nodiscard]] const OwnerType& source() const {
    EIGEN_FILTER_ASSERT(m_source && "use of a moved-from GaussianMixtureDensity");
    return *m_source;
  }

  [[nodiscard]] Scalar normalizedWeight(const Component& component) const {
    return Eigen::numext::exp(component.logWeight - logTotalWeight());
  }

  void lock() {
    if (m_source) ++m_source->m_lockCount;
  }

  void unlock() {
    if (m_source) --m_source->m_lockCount;
  }

  const OwnerType* m_source;
};

/** \brief Read-only view of a subset of the entries of every component of a
 * GaussianMixture or GaussianMixtureDensity.
 *
 * Obtained from marginal(). From a GaussianMixture it borrows the source; from a
 * GaussianMixtureDensity it holds a copy of the view, and so the lock. Weights are
 * those of the source. Component covariances read only the source's lower triangle.
 *
 * \tparam Source    GaussianMixture or GaussianMixtureDensity
 * \tparam Indices   the index type passed to marginal()
 * \tparam SubSpace  space of the selected coordinates
 */
template <typename Source, typename Indices, typename SubSpace>
class GaussianMixtureMarginal {
  using Selection = internal::IndexedSize<Indices, Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>;

 public:
  using Scalar = typename Source::Scalar;
  using Index = Eigen::Index;
  static constexpr int SizeAtCompileTime{spaceSize<SubSpace>};
  static constexpr int MaxSizeAtCompileTime{spaceMaxSize<SubSpace>};
  using SpaceType = SubSpace;
  using OwnerType = GaussianMixture<Scalar, SubSpace>;
  using MomentsType = typename OwnerType::MomentsType;
  using IndexVectorType = typename Selection::type;

  /** Selects \a indices, projecting the source's space onto them. */
  GaussianMixtureMarginal(const Source& source, const Indices& indices)
      : m_holder{source},
        m_indices{internal::resolveIndices<Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>(
            source.componentSize(), indices)},
        m_space{internal::projectSpace<SizeAtCompileTime, MaxSizeAtCompileTime>(source.space(), source.componentSize(),
                                                                                m_indices)} {
    EIGEN_FILTER_ASSERT(internal::distinctIndices(m_indices) && "marginal indices must be distinct");
  }

  /** Selects \a indices, whose coordinates form \a space. */
  GaussianMixtureMarginal(const Source& source, const Indices& indices, SubSpace space)
      : m_holder{source},
        m_indices{internal::resolveIndices<Source::SizeAtCompileTime, Source::MaxSizeAtCompileTime>(
            source.componentSize(), indices)},
        m_space{std::move(space)} {
    EIGEN_FILTER_ASSERT(internal::distinctIndices(m_indices) && "marginal indices must be distinct");
  }

  [[nodiscard]] const SubSpace& space() const { return m_space; }

  /** \returns the number of components */
  [[nodiscard]] Index size() const { return source().size(); }

  /** \returns the dimension of every marginal component */
  [[nodiscard]] Index componentSize() const { return m_indices.size(); }

  /** \returns the source entries this marginal selects */
  [[nodiscard]] const IndexVectorType& indices() const { return m_indices; }

  [[nodiscard]] Scalar logWeight(Index i) const { return component(i).logWeight; }
  [[nodiscard]] auto mean(Index i) const { return component(i).gaussian.mean()(m_indices); }
  [[nodiscard]] auto cov(Index i) const {
    return internal::lowerSymmetricBlock(component(i).gaussian.cov(), m_indices, m_indices);
  }
  [[nodiscard]] MomentsType moments(Index i) const { return MomentsType{mean(i), cov(i)}; }

  /** \returns an independent GaussianMixture holding this marginal, with the source's weights */
  [[nodiscard]] OwnerType copy() const {
    OwnerType result{m_space, componentSize()};
    result.reserve(size());
    for (Index i{0}; i < size(); ++i) result.add(logWeight(i), moments(i));
    return result;
  }

 private:
  [[nodiscard]] const Source& source() const { return m_holder.get(); }

  [[nodiscard]] const auto& component(Index i) const {
    EIGEN_FILTER_ASSERT(i >= 0 && i < size() && "component index out of range");
    return *(source().begin() + i);
  }

  internal::HolderFor<Source> m_holder;
  IndexVectorType m_indices;
  SubSpace m_space;
};

/** \returns \a n draws from \a mixture, one per column.
 *
 * Each draw picks a component in proportion to its weight using \a resampler, then
 * draws \f$ \mu_c + L_c z \f$ from it, with \f$ z \sim \mathcal{N}(0, I) \f$ and
 * \f$ \Sigma_c = L_c L_c^T \f$. The default MultinomialResampler makes the draws
 * independent; a lower-variance resampler such as SystematicResampler gives each
 * component a count closer to its expected share.
 *
 * \sa drawSamplesWithLogPdf(const GaussianMixture&, Eigen::Index, G&, const R&), \ref Resamplers
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G, typename R = MultinomialResampler>
  requires Resampler<R, Scalar, G>
[[nodiscard]] SampleSet<Scalar, Space> drawSamples(const GaussianMixtureDensity<Scalar, Space>& mixture, Eigen::Index n,
                                                   G& rng, const R& resampler = {}) {
  EIGEN_FILTER_ASSERT(n >= 0);
  using SampleSetType = SampleSet<Scalar, Space>;
  std::vector<Scalar> weights;
  weights.reserve(std::size_t(mixture.size()));
  for (const auto& component : mixture) {
    weights.push_back(Eigen::numext::exp(component.logWeight - mixture.logTotalWeight()));
  }
  std::vector<Eigen::Index> ancestors(static_cast<std::size_t>(n));
  resampler(std::span<const Scalar>{weights}, std::span<Eigen::Index>{ancestors}, rng);

  const SampleSetType noise{internal::standardNormal<SampleSetType>(mixture.componentSize(), n, rng)};
  SampleSetType samples{mixture.componentSize(), n};
  for (Eigen::Index j{0}; j < n; ++j) {
    const auto gaussian{(mixture.begin() + ancestors[std::size_t(j)])->gaussian.toDensity()};
    samples.col(j) = gaussian->mean() + gaussian->llt().matrixL() * noise.col(j);
  }
  mixture.space().normalizeValues(samples);
  return samples;
}

/** \returns \a n draws from \a mixture, one per column, with the log-density of
 * \a mixture at each.
 *
 * Draws exactly as drawSamples() does. Each log-density is a full
 * GaussianMixtureDensity::logPdf() evaluation, so the total is
 * \f$ O(n \cdot \mathrm{size}() \cdot d^2) \f$.
 *
 * \sa drawSamples(const GaussianMixture&, Eigen::Index, G&, const R&)
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G, typename R = MultinomialResampler>
  requires Resampler<R, Scalar, G>
[[nodiscard]] SamplesWithLogPdf<Scalar, Space> drawSamplesWithLogPdf(
    const GaussianMixtureDensity<Scalar, Space>& mixture, Eigen::Index n, G& rng, const R& resampler = {}) {
  SamplesWithLogPdf<Scalar, Space> result{drawSamples(mixture, n, rng, resampler), {}};
  result.logPdf.resize(n);
  for (Eigen::Index j{0}; j < n; ++j) result.logPdf(j) = mixture.logPdf(result.samples.col(j));
  return result;
}

/** A GaussianMixture of dimension \a Size in a linear space, with storage bound \a MaxSize. */
template <typename Scalar, int Size, int MaxSize = Size>
using GaussianMixtureN = GaussianMixture<Scalar, Linear<Size, MaxSize>>;

/** A GaussianMixtureDensity of dimension \a Size in a linear space, with storage bound \a MaxSize. */
template <typename Scalar, int Size, int MaxSize = Size>
using GaussianMixtureDensityN = GaussianMixtureDensity<Scalar, Linear<Size, MaxSize>>;

}  // namespace EigenFilter
