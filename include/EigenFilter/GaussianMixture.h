#pragma once

#include <algorithm>
#include <expected>
#include <functional>
#include <iterator>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "Gaussian.h"
#include "Macros.h"
#include "Moments.h"

namespace EigenFilter {

/** Reasons the weights of a MutableGaussianMixture cannot be normalized.
 *
 * \sa MutableGaussianMixture::logTotalWeight()
 */
enum class WeightError {
  Empty,      ///< the mixture has no components
  ZeroSum,    ///< every weight is zero (every log-weight is -inf)
  NonFinite,  ///< a log-weight is NaN or +inf
};

/** A mixture component whose covariance could not be factorized. */
struct ComponentError {
  Eigen::Index index;   ///< position of the component in the mixture
  GaussianError error;  ///< why factorization failed

  friend bool operator==(const ComponentError&, const ComponentError&) = default;
};

/** Reasons MutableGaussianMixture::toGaussianMixture() can fail. */
using GaussianMixtureError = std::variant<WeightError, ComponentError>;

namespace internal {

/** Streaming log-sum-exp: accumulates \f$ \log \sum_i e^{x_i} \f$ without overflow.
 * Inputs of -inf contribute nothing; an empty sum is -inf.
 */
template <typename Scalar>
class LogSumExp {
 public:
  void add(Scalar x) {
    if (x == -Eigen::NumTraits<Scalar>::infinity()) return;
    if (x <= m_max) {
      m_sum += Eigen::numext::exp(x - m_max);
    } else {
      m_sum = m_sum * Eigen::numext::exp(m_max - x) + Scalar(1);
      m_max = x;
    }
  }

  [[nodiscard]] Scalar value() const {
    return m_sum == Scalar(0) ? -Eigen::NumTraits<Scalar>::infinity() : m_max + Eigen::numext::log(m_sum);
  }

 private:
  Scalar m_max{-Eigen::NumTraits<Scalar>::infinity()};
  Scalar m_sum{0};
};

}  // namespace internal

template <typename Scalar_, int Size_, int MaxSize_ = Size_>
class GaussianMixture;

/** \brief Mutable weighted sum of Gaussians that can be viewed as a GaussianMixture.
 *
 * Each component has a log-weight and a MutableGaussian. Weights need not be
 * normalized; a log-weight of -inf is a zero weight. All components share the
 * dimension componentSize().
 *
 * Components are read through const iterators and changed only through this
 * object's methods. Adding or removing components invalidates all iterators,
 * and removal may reorder the remaining components.
 *
 * toGaussianMixture() factorizes every component whose covariance has changed
 * and returns a GaussianMixture view. While any view is alive this object is
 * locked: mutating, moving, or destroying it is a programmer error caught by
 * \c EIGEN_FILTER_ASSERT. It is also locked while forEach() and removeIf() call
 * back into user code.
 *
 * \tparam Scalar_   scalar type
 * \tparam Size_     component dimension, or \c Eigen::Dynamic
 * \tparam MaxSize_  maximum component dimension
 *
 * \sa class GaussianMixture, class MutableGaussian
 */
template <typename Scalar_, int Size_, int MaxSize_ = Size_>
class MutableGaussianMixture {
 public:
  using ComponentGaussianType = MutableGaussian<Scalar_, Size_, MaxSize_>;
  using MomentsType = typename ComponentGaussianType::MomentsType;
  using Scalar = typename ComponentGaussianType::Scalar;
  using Index = typename ComponentGaussianType::Index;
  using VectorType = typename ComponentGaussianType::VectorType;
  using MatrixType = typename ComponentGaussianType::MatrixType;
  using GaussianMixtureType = GaussianMixture<Scalar_, Size_, MaxSize_>;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};

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

    void setMoments(const MomentsType& moments) { MutableGaussianMixture::assignMoments(m_component, moments); }

   private:
    friend MutableGaussianMixture;

    explicit MutableComponent(Component& component) : m_component{component} {}

    Component& m_component;
  };

  MutableGaussianMixture() = default;

  /** Constructs an empty mixture whose components will have dimension \a componentSize. */
  explicit MutableGaussianMixture(Index componentSize) : m_componentSize{componentSize} {
    assertValidComponentSize(componentSize);
  }

  /** Copies the components and their factorizations. The copy starts unlocked. */
  MutableGaussianMixture(const MutableGaussianMixture& other)
      : m_components{other.m_components}, m_componentSize{other.m_componentSize} {}

  /** Moving a locked object is a programmer error; if \c EIGEN_FILTER_ASSERT throws,
   * the exception reaches this \c noexcept boundary and calls \c std::terminate.
   */
  MutableGaussianMixture(MutableGaussianMixture&& other) noexcept
      : m_components{std::move(other.m_components)}, m_componentSize{other.m_componentSize} {
    other.assertUnlocked();
  }

  MutableGaussianMixture& operator=(const MutableGaussianMixture& other) {
    assertUnlocked();
    m_components = other.m_components;
    m_componentSize = other.m_componentSize;
    return *this;
  }

  MutableGaussianMixture& operator=(MutableGaussianMixture&& other) noexcept {
    assertUnlocked();
    other.assertUnlocked();
    m_components = std::move(other.m_components);
    m_componentSize = other.m_componentSize;
    return *this;
  }

  ~MutableGaussianMixture() { assertUnlocked(); }

  /** \returns the number of components */
  [[nodiscard]] Index size() const { return Index(m_components.size()); }
  [[nodiscard]] bool empty() const { return m_components.empty(); }

  /** \returns the dimension of every component */
  [[nodiscard]] Index componentSize() const { return m_componentSize; }

  /** Resizes every component to dimension \a n. Component contents are unspecified afterward. */
  void resizeComponents(Index n) {
    assertUnlocked();
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
    m_components.clear();
  }

  [[nodiscard]] const_iterator begin() const { return m_components.cbegin(); }
  [[nodiscard]] const_iterator end() const { return m_components.cend(); }

  /** Appends a component. \returns an iterator to it. */
  const_iterator add(Scalar logWeight, MomentsType moments) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(moments.size() == m_componentSize);
    m_components.push_back(Component{logWeight, ComponentGaussianType{std::move(moments)}});
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
    if (m_components.empty()) return std::unexpected{WeightError::Empty};
    internal::LogSumExp<Scalar> total;
    for (const auto& component : m_components) {
      const Scalar w{component.logWeight};
      if (Eigen::numext::isnan(w) || w == Eigen::NumTraits<Scalar>::infinity()) {
        return std::unexpected{WeightError::NonFinite};
      }
      total.add(w);
    }
    const Scalar result{total.value()};
    if (result == -Eigen::NumTraits<Scalar>::infinity()) return std::unexpected{WeightError::ZeroSum};
    return result;
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
    m_components.clear();
    if constexpr (std::ranges::sized_range<R>) m_components.reserve(std::ranges::size(components));
    for (auto&& [logWeight, moments] : components) add(logWeight, moments);
  }

  /** Appends copies of \a other's components, which must have the same dimension. */
  void append(const MutableGaussianMixture& other) {
    assertUnlocked();
    EIGEN_FILTER_ASSERT(other.m_componentSize == m_componentSize);
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

  /** Validates the weights and factorizes every component whose covariance has changed.
   *
   * Does not lock this object.
   *
   * \returns nothing on success; a WeightError if the mixture is empty or its
   *          weights cannot be normalized; or a ComponentError naming the first
   *          component that failed to factorize.
   *
   * \sa toGaussianMixture()
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
   * \returns a GaussianMixture view that locks this object for its lifetime, or
   *          the GaussianMixtureError from factorize().
   */
  [[nodiscard("the GaussianMixture view holds the lock; discarding it releases the lock immediately")]]
  std::expected<GaussianMixtureType, GaussianMixtureError> toGaussianMixture() const {
    if (auto result{factorize()}; !result) return std::unexpected{result.error()};
    return GaussianMixtureType{this};
  }

 private:
  friend GaussianMixtureType;

  class ScopedLock {
   public:
    explicit ScopedLock(const MutableGaussianMixture& mixture) : m_mixture{mixture} { ++m_mixture.m_lockCount; }
    ~ScopedLock() { --m_mixture.m_lockCount; }
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;

   private:
    const MutableGaussianMixture& m_mixture;
  };

  static void assertValidComponentSize(Index n) {
    EIGEN_FILTER_ASSERT(n >= 0 && (Size_ == Eigen::Dynamic || n == Size_) &&
                        (MaxSize_ == Eigen::Dynamic || n <= MaxSize_));
  }

  static void assignMoments(Component& component, const MomentsType& moments) {
    component.gaussian.setMean(moments.mean());
    component.gaussian.setCov(moments.cov());
  }

  void assertUnlocked() const {
    EIGEN_FILTER_ASSERT(m_lockCount == 0 && "MutableGaussianMixture is locked");
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
  Index m_componentSize{Size_ == Eigen::Dynamic ? 0 : Size_};
  // Filled by the const factorize() for use by views.
  mutable Scalar m_logTotalWeight{};
  mutable int m_lockCount{0};
};

/** \brief Read-only view of a MutableGaussianMixture whose components have been factorized.
 *
 * Represents the normalized density
 * \f[ p(x) = \frac{\sum_i w_i \, \mathcal{N}(x; \mu_i, \Sigma_i)}{\sum_i w_i}. \f]
 *
 * Obtained from MutableGaussianMixture::toGaussianMixture(). Holding a
 * GaussianMixture locks the source; copies share the lock and the last one to be
 * destroyed releases it. A GaussianMixture must not outlive its source.
 *
 * \sa class MutableGaussianMixture
 */
template <typename Scalar_, int Size_, int MaxSize_>
class GaussianMixture {
 public:
  using MutableType = MutableGaussianMixture<Scalar_, Size_, MaxSize_>;
  using Component = typename MutableType::Component;
  using const_iterator = typename MutableType::const_iterator;
  using MomentsType = typename MutableType::MomentsType;
  using Scalar = typename MutableType::Scalar;
  using Index = typename MutableType::Index;
  using VectorType = typename MutableType::VectorType;
  using MatrixType = typename MutableType::MatrixType;
  static constexpr int SizeAtCompileTime{Size_};
  static constexpr int MaxSizeAtCompileTime{MaxSize_};

  GaussianMixture(const GaussianMixture& other) : m_source{other.m_source} { lock(); }
  GaussianMixture(GaussianMixture&& other) noexcept : m_source{std::exchange(other.m_source, nullptr)} {}

  GaussianMixture& operator=(const GaussianMixture& other) {
    if (this != &other) {
      unlock();
      m_source = other.m_source;
      lock();
    }
    return *this;
  }

  GaussianMixture& operator=(GaussianMixture&& other) noexcept {
    if (this != &other) {
      unlock();
      m_source = std::exchange(other.m_source, nullptr);
    }
    return *this;
  }

  ~GaussianMixture() { unlock(); }

  /** \returns the number of components */
  [[nodiscard]] Index size() const { return source().size(); }
  [[nodiscard]] Index componentSize() const { return source().componentSize(); }
  [[nodiscard]] const_iterator begin() const { return source().begin(); }
  [[nodiscard]] const_iterator end() const { return source().end(); }

  /** \returns the log of the sum of the (unnormalized) component weights */
  [[nodiscard]] Scalar logTotalWeight() const { return source().m_logTotalWeight; }

  /** \returns the mean and covariance of the mixture,
   * \f$ \mu = \sum_i \hat w_i \mu_i \f$ and
   * \f$ \Sigma = \sum_i \hat w_i \left(\Sigma_i + (\mu_i - \mu)(\mu_i - \mu)^T\right) \f$,
   * where \f$ \hat w_i \f$ are the normalized weights.
   *
   * Only the lower triangle of each component covariance is read; the result is symmetric.
   */
  [[nodiscard]] MomentsType moments() const {
    MomentsType result{componentSize()};
    auto& mean{result.mean()};
    auto& cov{result.cov()};
    mean.setZero();
    cov.setZero();
    for (const auto& component : *this) mean += normalizedWeight(component) * component.gaussian.mean();
    for (const auto& component : *this) {
      const Scalar w{normalizedWeight(component)};
      const VectorType deviation{component.gaussian.mean() - mean};
      cov.template triangularView<Eigen::Lower>() += w * component.gaussian.cov();
      cov.template selfadjointView<Eigen::Lower>().rankUpdate(deviation, w);
    }
    for (Index j{1}; j < cov.cols(); ++j) {
      for (Index i{0}; i < j; ++i) cov(i, j) = cov(j, i);
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
      sum.add(component.logWeight + component.gaussian.toGaussian()->logPdf(x));
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

 private:
  friend MutableType;

  explicit GaussianMixture(const MutableType* source) : m_source{source} { lock(); }

  [[nodiscard]] const MutableType& source() const {
    EIGEN_FILTER_ASSERT(m_source && "use of a moved-from GaussianMixture");
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

  const MutableType* m_source;
};

}  // namespace EigenFilter
