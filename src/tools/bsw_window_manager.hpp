#ifndef MRHYDE_BSW_WINDOW_MANAGER_H
#define MRHYDE_BSW_WINDOW_MANAGER_H

#include "sketch_window.hpp"
#include "svd_window.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace MrHyDE {

  /// Configuration for one manager; every quantity in state-equivalents (SE)
  /// counts M doubles as one state.
  struct BswManagerOptions {
    double total_budget_se = 0.0;   ///< the one storage pool windows and checkpoints share
    int min_checkpoints = 1;        ///< slots always reserved for the classic axis
    int base_rank = 2;              ///< opening rank for the first window
    double tol = 1.0e-8;            ///< per-column relative reconstruction tolerance
    int rho = 5;                    ///< monitor cadence and rolling-buffer depth
    int min_commit = 4;             ///< smallest prefix worth committing
    int rank_max = 15;              ///< hard rank cap
    uint64_t master_seed = 0;
    /// Planned [a,b] spans instead of greedy discovery; empty means greedy.
    std::vector<std::pair<int,int> > planned_windows;
  };

  /// One committed window as the scheduler sees it: real steps a+1..b_eff.
  struct BswEffectiveWindow {
    int a = 0;
    int b_eff = 0;
    int index = -1;   ///< manager-internal window index
  };

  /**
   * @class BswWindowManagerBase
   * @brief Host-facing interface: the driver streams states through one
   *        forward sweep, then serves the reverse sweep from the committed
   *        windows.  Host-neutral -- raw arrays and integers only.
   */
  class BswWindowManagerBase {

  public:

    virtual ~BswWindowManagerBase() {}

    /// Start a fresh sweep; forgets every window from the previous one.
    virtual void beginSweep(const int & Mdim, const int & num_steps,
                            const double & initial_time) = 0;

    /// Feed the accepted state at step k (1-based) and the time reached.
    virtual void recordState(const int & k, const double* u, const double & time) = 0;

    /// Close the sweep: fix the effective windows and the slot flow-back.
    virtual void endSweep() = 0;

    /// Committed windows with b_eff - a >= 2, sorted, for BswRevolve.
    virtual const std::vector<BswEffectiveWindow> & effectiveWindows() const = 0;

    /// Checkpoint slots for the classic axis: the floor plus every unspent SE.
    virtual int numCheckpointSlots() const = 0;

    /// Reconstruct the state at real step k from effective window widx.
    virtual void reconstruct(const int & widx, const int & k, double* out) const = 0;

    /// The window's right-edge state u_{b_eff} (exact anchor, or reconstruction).
    virtual void rightEdge(const int & widx, double* out) const = 0;

    /// Time reached after step k, as recorded during the forward sweep.
    virtual double getTime(const int & k) const = 0;

    /// SE held by committed payloads and anchors.
    virtual double getSpentSE() const = 0;

    /// Largest transient SE any window held while recording (honest count).
    virtual double getPeakSE() const = 0;

    /// One line per window: edges, status, rank, error.  For the layout log.
    virtual std::vector<std::string> layoutReport() const = 0;
  };

  /**
   * @class BswWindowManager
   * @brief Window lifecycle and budget governor over one compressor type
   *        (SketchWindow or SvdWindow).
   *
   * The budget is one pool: windows spend what their committed payloads and
   * anchors hold, and everything unspent at the end of the sweep flows back
   * into checkpoint slots for the classic revolve axis -- more slots mean
   * less recomputation, and revolve places any count optimally, so the
   * flow-back is a monotone win.
   *
   * Greedy mode opens a rolling window at the first step that fits: the
   * streaming-transient model A*m + C (honest count, resident test matrices
   * included) against the unspent pool decides the horizon.  A window that
   * cannot verify its tolerance cuts or falls back on its own; those steps
   * simply stay on the classic axis.  The rank hint a struggling window
   * leaves behind seeds the next window's rank.
   */
  template<class WindowT>
  class BswWindowManager : public BswWindowManagerBase {

  public:

    BswWindowManager(const BswManagerOptions & options)
      : options_(options) {
      if (!options_.planned_windows.empty()) {
        // validated properly by BswRevolve later; catch cheap mistakes here
        for (size_t i=0; i<options_.planned_windows.size(); ++i) {
          if (options_.planned_windows[i].second - options_.planned_windows[i].first < 2) {
            throw std::invalid_argument(
              "BswWindowManager: a planned window must cover at least 2 steps.");
          }
        }
      }
    }

    void beginSweep(const int & Mdim, const int & num_steps,
                    const double & initial_time) override {
      M_ = Mdim;
      num_steps_ = num_steps;
      windows_.clear();
      effective_.clear();
      open_index_ = -1;
      next_planned_ = 0;
      spent_se_ = 0.0;
      peak_se_ = 0.0;
      current_rank_ = options_.base_rank;
      slots_ = options_.min_checkpoints;
      swept_ = false;
      times_.assign(num_steps + 1, initial_time);
    }

    void recordState(const int & k, const double* u, const double & time) override {

      if (k < 1 || k > num_steps_) {
        throw std::runtime_error("BswWindowManager: step outside the sweep.");
      }
      times_[k] = time;

      if (open_index_ < 0) {
        maybeOpenWindow(k);
      }
      if (open_index_ >= 0) {
        WindowT & w = *windows_[open_index_];
        if (k > w.getA() && k <= w.getB()) {
          w.record(u, k);
          if (!w.isOpen()) {
            closeWindow();
          }
        }
      }
    }

    void endSweep() override {

      // a still-open window ran past the sweep end; nothing more will arrive
      if (open_index_ >= 0) {
        closeWindow();
      }

      effective_.clear();
      for (size_t i=0; i<windows_.size(); ++i) {
        const WindowT & w = *windows_[i];
        if ((w.getStatus() == WindowStatus::Accepted ||
             w.getStatus() == WindowStatus::Prefix) &&
            w.getBEff() - w.getA() >= 2) {
          BswEffectiveWindow e;
          e.a = w.getA();
          e.b_eff = w.getBEff();
          e.index = static_cast<int>(i);
          effective_.push_back(e);
        }
      }

      // flow-back: every state-equivalent the windows did not keep becomes a
      // checkpoint slot for the classic axis
      const double unspent = windowBudget() - spent_se_;
      slots_ = options_.min_checkpoints + std::max(0, static_cast<int>(unspent));
      swept_ = true;
    }

    const std::vector<BswEffectiveWindow> & effectiveWindows() const override {
      requireSwept();
      return effective_;
    }

    int numCheckpointSlots() const override {
      requireSwept();
      return slots_;
    }

    void reconstruct(const int & widx, const int & k, double* out) const override {
      requireSwept();
      windows_[effective_[widx].index]->reconstruct(k, out);
    }

    void rightEdge(const int & widx, double* out) const override {
      requireSwept();
      windows_[effective_[widx].index]->rightEdgeRaw(out);
    }

    double getTime(const int & k) const override {
      return times_[k];
    }

    double getSpentSE() const override { return spent_se_; }
    double getPeakSE()  const override { return peak_se_; }

    std::vector<std::string> layoutReport() const override {
      std::vector<std::string> lines;
      for (size_t i=0; i<windows_.size(); ++i) {
        const WindowT & w = *windows_[i];
        std::string status = "planned";
        if (w.getStatus() == WindowStatus::Accepted) { status = "accepted"; }
        else if (w.getStatus() == WindowStatus::Prefix) { status = "prefix"; }
        else if (w.getStatus() == WindowStatus::Fallback) { status = "fallback"; }
        lines.push_back("window " + std::to_string(i)
          + ": steps " + std::to_string(w.getA()+1) + ".." + std::to_string(w.getB())
          + " -> " + status
          + " (served through " + std::to_string(w.getBEff())
          + ", rank " + std::to_string(w.getPayloadColumns())
          + ", errMaxCol " + std::to_string(w.getErrMaxCol()) + ")");
      }
      return lines;
    }

  private:

    /// The share of the pool windows may hold (10 SE of work vectors and the
    /// checkpoint floor are always kept out of their reach).
    double windowBudget() const {
      return std::max(0.0, options_.total_budget_se - 10.0 - options_.min_checkpoints);
    }

    // Open a window whose first recorded column is step k, if one is due
    // (planned mode) or if the transient cost model fits the pool (greedy).
    void maybeOpenWindow(const int & k) {

      const int a = k - 1;
      int b = 0;

      if (!options_.planned_windows.empty()) {
        if (next_planned_ >= options_.planned_windows.size()) { return; }
        if (a != options_.planned_windows[next_planned_].first) { return; }
        b = std::min(options_.planned_windows[next_planned_].second, num_steps_);
        ++next_planned_;
        if (b - a < 2) { return; }
      }
      else {
        double A = 0.0, C = 0.0;
        WindowT::rollingStreamCoeffs(M_, current_rank_, options_.rho, A, C);
        const double room = windowBudget() - spent_se_ - C;
        if (room <= 0.0) { return; }
        const int horizon = static_cast<int>(room/A);
        b = std::min(a + horizon, num_steps_);
        if (b - a < WindowT::minCommitForRank(current_rank_, options_.min_commit)) {
          return;
        }
      }

      WindowPolicy policy = WindowT::fillPolicy();
      policy.tol = options_.tol;
      policy.rho = options_.rho;
      policy.min_commit = options_.min_commit;
      policy.rank_max = options_.rank_max;
      policy.rolling = true;

      windows_.push_back(std::unique_ptr<WindowT>(
        new WindowT(a, b, current_rank_, static_cast<int>(windows_.size()),
                    options_.master_seed)));
      open_index_ = static_cast<int>(windows_.size()) - 1;
      windows_[open_index_]->beginRecord(M_, policy);
      peak_se_ = std::max(peak_se_, spent_se_ + windows_[open_index_]->peakSE());
    }

    // A window closed itself (commit, cut, or fallback): account for what it
    // keeps, and let its rank hint seed the next window.
    void closeWindow() {
      const WindowT & w = *windows_[open_index_];
      spent_se_ += w.storageSE();
      current_rank_ = std::min(std::max(current_rank_, w.getNextRankHint()),
                               options_.rank_max);
      open_index_ = -1;
    }

    void requireSwept() const {
      if (!swept_) {
        throw std::runtime_error("BswWindowManager: endSweep has not run.");
      }
    }

    BswManagerOptions options_;
    int M_ = 0;
    int num_steps_ = 0;
    std::vector<std::unique_ptr<WindowT> > windows_;
    std::vector<BswEffectiveWindow> effective_;
    int open_index_ = -1;
    size_t next_planned_ = 0;
    double spent_se_ = 0.0;
    double peak_se_ = 0.0;
    int current_rank_ = 2;
    int slots_ = 1;
    bool swept_ = false;
    std::vector<double> times_;
  };

  /// The two compressors by name: 'sketch' or 'rm'.
  inline std::unique_ptr<BswWindowManagerBase>
  makeBswWindowManager(const std::string & compressor, const BswManagerOptions & options) {
    if (compressor == "sketch") {
      return std::unique_ptr<BswWindowManagerBase>(new BswWindowManager<SketchWindow>(options));
    }
    if (compressor == "rm") {
      return std::unique_ptr<BswWindowManagerBase>(new BswWindowManager<SvdWindow>(options));
    }
    throw std::invalid_argument("BswWindowManager: unknown compressor '" + compressor
                                + "' (use 'sketch' or 'rm').");
  }

}

#endif
