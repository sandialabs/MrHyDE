#ifndef MRHYDE_BSW_REVOLVE_H
#define MRHYDE_BSW_REVOLVE_H

#include "revolve.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <memory>
#include <stdexcept>
#include <vector>

namespace MrHyDE {

  /**
   * @brief The primitive directives a BswRevolve schedule can emit.
   */
  enum class BswAction {
    SolveRange,     ///< run the forward model for real steps k1..k2
    JumpAnchor,     ///< set the state from window win's stored right-edge anchor u_b
    Store,          ///< copy the current state (= u_step) into checkpoint slot
    Restore,        ///< load checkpoint slot back (state becomes u_step)
    ReverseStep,    ///< classic reverse of real step k (is_first: terminal adjoint step)
    ReverseWindow,  ///< adjoint sweep k = b..a+1 of window win from reconstructions;
                    ///< on entry the state is u_a EXACT (the classic invariant)
    Terminate,      ///< the reversal is complete
    Error           ///< the inner schedule is inconsistent; see getStatus()
  };

  /**
   * @brief One schedule directive.  Only the fields for its action are set;
   *        the rest stay at their defaults.
   */
  struct BswDirective {
    BswAction action = BswAction::Error;
    int k1 = -1;            ///< SolveRange: first real step
    int k2 = -1;            ///< SolveRange: last real step
    int win = -1;           ///< JumpAnchor/ReverseWindow: 0-based window index
    int step = -1;          ///< JumpAnchor/Store/Restore: real position after the action
    int slot = -1;          ///< Store/Restore: 0-based checkpoint slot
    int k = -1;             ///< ReverseStep: real step to reverse
    int a = -1;             ///< ReverseWindow: window left edge (state on entry is u_a)
    int b = -1;             ///< ReverseWindow: window right edge
    bool is_first = false;  ///< ReverseStep/ReverseWindow: terminal adjoint step
  };

  /**
   * @class BswRevolve
   * @brief Block-Sketched-Windows scheduler on a condensed revolve axis.
   *
   * Wraps the certified classic Revolve (Algorithm 799): each sketched window
   * of consecutive time steps becomes ONE macro-step of a condensed time axis;
   * Revolve runs UNMODIFIED on that axis and this class translates its actions
   * into primitive directives on the real axis.  Pure integers: no physics, no
   * vectors, no storage.
   *
   * Real axis: steps 1..N (1-based, driver convention).  Window i covers the
   * consecutive real steps a_i+1 .. b_i (windows[i] = {a_i, b_i}, with
   * m_i = b_i - a_i >= 2).  Windows are disjoint, sorted, may be adjacent.
   * Crossing a window forward is free (the driver stores the window's exact
   * right-edge anchor u_b); reversing it is free (interior states come from
   * sketch reconstructions).
   *
   * Condensed axis: the N_eff = N - sum(m_i) outside steps, in order, plus one
   * macro-step per window: N_c = N_eff + K_s, run by Revolve(N_c, num_checkpoints).
   *
   * With no windows this reduces action-for-action to classic Revolve (the unit
   * test replays both side by side).  Rank/compression profitability guards
   * live with the sketch payload, not here.  Known cosmetic waste: a condensed
   * Store can land right after a JumpAnchor, duplicating that window's anchor
   * into a checkpoint slot -- correct, but the memory model should either
   * charge or waive it explicitly.
   *
   * Differences from the MATLAB reference (BswRevolve.m): window indices and
   * checkpoint slots are 0-based here; the directive stream is otherwise
   * identical.
   */
  class BswRevolve {

  public:

    /**
     * @brief Build a schedule for N real steps with the given sketched windows
     *        and checkpoint budget.  windows[i] = {a, b} covers steps a+1..b.
     */
    BswRevolve(const int & num_steps, const std::vector<std::array<int,2> > & windows,
               const int & num_checkpoints)
      : windows_(windows) {

      if (num_checkpoints < 1) {
        throw std::invalid_argument("BswRevolve: num_checkpoints must be at least 1.");
      }
      validateWindows(num_steps, windows);

      // condensed maps: walk the real axis, collapsing windows
      const int num_windows = static_cast<int>(windows.size());
      int wi = 0;
      int k = 1;
      while (k <= num_steps) {
        if (wi < num_windows && k == windows[wi][0] + 1) {
          step_of_condensed_.push_back(windows[wi][1]);   // macro-step ends at b
          window_of_condensed_.push_back(wi);
          k = windows[wi][1] + 1;
          ++wi;
        }
        else {
          step_of_condensed_.push_back(k);
          window_of_condensed_.push_back(-1);             // plain real step
          k = k + 1;
        }
      }

      num_condensed_ = static_cast<int>(step_of_condensed_.size());
      num_effective_ = num_steps;
      for (size_t i=0; i<windows.size(); ++i) {
        num_effective_ -= windows[i][1] - windows[i][0];
      }

      inner_ = std::unique_ptr<Revolve>(new Revolve(num_condensed_, num_checkpoints));
    }

    /**
     * @brief The next primitive directive.  One inner Revolve action can expand
     *        to several directives; they are queued and handed out one at a time.
     */
    BswDirective next() {
      while (queue_.empty()) {
        translate();
      }
      BswDirective d = queue_.front();
      queue_.pop_front();
      return d;
    }

    int getNumCondensedSteps()  const { return num_condensed_; }
    int getNumEffectiveSteps()  const { return num_effective_; }
    int getStatus()             const { return inner_->getStatus(); }

    /**
     * @brief Reject malformed window tables: out of range, shorter than 2 steps
     *        (that is a plain checkpoint), unsorted, or overlapping.
     */
    static void validateWindows(const int & num_steps,
                                const std::vector<std::array<int,2> > & windows) {
      for (size_t i=0; i<windows.size(); ++i) {
        const int a = windows[i][0];
        const int b = windows[i][1];
        if (a < 0 || b > num_steps) {
          throw std::invalid_argument("BswRevolve: window out of range.");
        }
        if (b - a < 2) {
          throw std::invalid_argument(
            "BswRevolve: window shorter than 2 steps is a plain checkpoint -- reject.");
        }
        if (i > 0 && (windows[i][0] <= windows[i-1][0] || windows[i][0] < windows[i-1][1])) {
          throw std::invalid_argument(
            "BswRevolve: windows must be sorted and disjoint (adjacent is fine).");
        }
      }
    }

    /**
     * @brief Windows = the given blocks of the uniform block_size-partition of
     *        1..num_steps: block j (1-based) covers steps (j-1)*block_size+1 .. j*block_size.
     */
    static std::vector<std::array<int,2> > uniformWindows(const int & num_steps,
                                                          const int & block_size,
                                                          std::vector<int> block_ids) {
      const int num_blocks = num_steps/block_size;
      for (size_t i=0; i<block_ids.size(); ++i) {
        if (block_ids[i] < 1 || block_ids[i] > num_blocks) {
          throw std::invalid_argument("BswRevolve: block id out of range.");
        }
      }
      std::sort(block_ids.begin(), block_ids.end());
      std::vector<std::array<int,2> > windows;
      for (size_t i=0; i<block_ids.size(); ++i) {
        const int j = block_ids[i];
        windows.push_back({(j-1)*block_size, j*block_size});
      }
      return windows;
    }

    /// Solve counts for one gradient, by kind.
    struct SolveCounts {
      long advance_solves = 0;    ///< forward solves outside reversals
      long reverse_solves = 0;    ///< one recompute per real-step reverse
      long anchor_jumps = 0;      ///< free forward window crossings
      long window_reverses = 0;   ///< free window reversals
      long total = 0;             ///< advance_solves + reverse_solves
    };

    /**
     * @brief Integer simulation of one gradient: con.solve counts by kind.
     *        Sketches and anchors are built during the objective sweep, so a
     *        gradient costs only advance solves plus one recompute per
     *        real-step reverse.
     */
    static SolveCounts predictGradientSolves(const int & num_steps,
                                             const std::vector<std::array<int,2> > & windows,
                                             const int & num_checkpoints) {
      BswRevolve bsw(num_steps, windows, num_checkpoints);
      SolveCounts s;
      while (true) {
        BswDirective d = bsw.next();
        if (d.action == BswAction::SolveRange)         { s.advance_solves += d.k2 - d.k1 + 1; }
        else if (d.action == BswAction::ReverseStep)   { s.reverse_solves += 1; }
        else if (d.action == BswAction::JumpAnchor)    { s.anchor_jumps += 1; }
        else if (d.action == BswAction::ReverseWindow) { s.window_reverses += 1; }
        else if (d.action == BswAction::Terminate)     { break; }
        else if (d.action == BswAction::Error) {
          throw std::runtime_error("BswRevolve: inner Revolve returned an error.");
        }
      }
      s.total = s.advance_solves + s.reverse_solves;
      return s;
    }

  private:

    /// Run one inner Revolve action and queue its real-axis directives.
    void translate() {

      const int prev = inner_->getRangeStart();   // condensed position BEFORE
      RevolveAction action = inner_->next();

      if (action == RevolveAction::Advance) {

        // expand condensed steps prev+1..range_start on the real axis, merging
        // contiguous real steps into one SolveRange
        int run_start = 0;
        for (int jc=prev+1; jc<=inner_->getRangeStart(); ++jc) {
          if (windowOf(jc) >= 0) {
            if (run_start > 0) {
              BswDirective d;
              d.action = BswAction::SolveRange;
              d.k1 = run_start;
              d.k2 = stepOf(jc-1);
              queue_.push_back(d);
              run_start = 0;
            }
            BswDirective d;
            d.action = BswAction::JumpAnchor;
            d.win = windowOf(jc);
            d.step = stepOf(jc);
            queue_.push_back(d);
          }
          else if (run_start == 0) {
            run_start = stepOf(jc);
          }
        }
        if (run_start > 0) {
          BswDirective d;
          d.action = BswAction::SolveRange;
          d.k1 = run_start;
          d.k2 = stepOf(inner_->getRangeStart());
          queue_.push_back(d);
        }
      }
      else if (action == RevolveAction::Store) {
        BswDirective d;
        d.action = BswAction::Store;
        d.slot = inner_->getNumCheckpointsStored() - 1;
        d.step = realPosition(inner_->getRangeStart());
        queue_.push_back(d);
      }
      else if (action == RevolveAction::Restore) {
        BswDirective d;
        d.action = BswAction::Restore;
        d.slot = inner_->getNumCheckpointsStored() - 1;
        d.step = realPosition(inner_->getRangeStart());
        queue_.push_back(d);
      }
      else if (action == RevolveAction::FirstReverse || action == RevolveAction::Reverse) {

        const int jc = inner_->getRangeStart() + 1;   // condensed step reversed
        const int w = windowOf(jc);
        BswDirective d;
        d.is_first = (action == RevolveAction::FirstReverse);
        if (w >= 0) {
          d.action = BswAction::ReverseWindow;
          d.win = w;
          d.a = windows_[w][0];
          d.b = windows_[w][1];
        }
        else {
          d.action = BswAction::ReverseStep;
          d.k = stepOf(jc);
        }
        queue_.push_back(d);
      }
      else if (action == RevolveAction::Terminate) {
        BswDirective d;
        d.action = BswAction::Terminate;
        queue_.push_back(d);
      }
      else {
        BswDirective d;
        d.action = BswAction::Error;
        queue_.push_back(d);
      }
    }

    /// Real position reached after condensed step jc (1-based jc, like the maps).
    int stepOf(const int & jc)   const { return step_of_condensed_[jc-1]; }
    /// Window id of condensed step jc, or -1 for a plain real step.
    int windowOf(const int & jc) const { return window_of_condensed_[jc-1]; }
    /// Real position after pc condensed steps (pc = 0 -> position 0).
    int realPosition(const int & pc) const { return (pc == 0) ? 0 : stepOf(pc); }

    std::vector<std::array<int,2> > windows_;    ///< {a,b}: window covers steps a+1..b
    int num_condensed_;                          ///< condensed steps = num_effective_ + #windows
    int num_effective_;                          ///< real steps outside windows

    std::vector<int> step_of_condensed_;         ///< real position reached after condensed step j
    std::vector<int> window_of_condensed_;       ///< window id of macro-step j (-1 = real step)

    std::unique_ptr<Revolve> inner_;             ///< classic Revolve on the condensed axis
    std::deque<BswDirective> queue_;             ///< pending primitive directives
  };

}

#endif
