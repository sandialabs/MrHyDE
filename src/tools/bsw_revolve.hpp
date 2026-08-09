#ifndef MRHYDE_BSW_REVOLVE_H
#define MRHYDE_BSW_REVOLVE_H

#include "revolve.hpp"

#include <algorithm>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace MrHyDE {

  /// The seven primitive directives a BSW schedule can issue.
  enum class BswAction {
    SolveRange,     ///< run the forward model for real steps k1..k2
    JumpAnchor,     ///< set the state to window win's exact right-edge anchor u_b
    Store,          ///< copy the current state (= u_step) into checkpoint slot
    Restore,        ///< load slot back; the current state becomes u_step
    ReverseStep,    ///< classic reverse of real step k (is_first: terminal adjoint)
    ReverseWindow,  ///< adjoint sweep k = b..a+1 from reconstructions; entry state u_a EXACT
    Terminate       ///< the schedule is complete
  };

  /// One directive; only the fields for its action are meaningful.
  struct BswDirective {
    BswAction action = BswAction::Terminate;
    int k1 = 0;             ///< SolveRange: first step of the run
    int k2 = 0;             ///< SolveRange: last step of the run
    int win = -1;           ///< JumpAnchor/ReverseWindow: 0-based window index
    int step = 0;           ///< JumpAnchor/Store/Restore: the state's step; ReverseStep: the step reversed
    int slot = -1;          ///< Store/Restore: 0-based checkpoint slot
    int a = 0;              ///< ReverseWindow: left edge (state u_a is exact on entry)
    int b = 0;              ///< ReverseWindow: right edge
    bool is_first = false;  ///< ReverseStep/ReverseWindow: terminal adjoint path
  };

  /**
   * @class BswRevolve
   * @brief Block-Sketched-Windows scheduler on a condensed revolve axis
   *        (port of BswRevolve.m).
   *
   * Wraps the certified classic Revolve (Algorithm 799): each sketched window
   * of consecutive time steps becomes ONE macro-step of a condensed time axis;
   * Revolve runs UNMODIFIED on that axis and this class translates its actions
   * into primitive directives on the real axis.  Pure integers: no physics, no
   * vectors, no storage.
   *
   * Real axis: steps 1..N.  Window i covers the consecutive real steps
   * a_i+1..b_i (m_i = b_i - a_i >= 2); windows are disjoint, sorted, may be
   * adjacent.  Crossing a window forward is free (the driver holds the
   * window's exact right-edge anchor u_b); reversing it is free (interior
   * states come from reconstructions).
   *
   * Condensed axis: the N_eff = N - sum(m_i) outside steps, in order, plus one
   * macro-step per window: N_c = N_eff + num_windows.
   *
   * With no windows this reduces action-for-action to classic Revolve.
   * Rank/compression profitability guards live with the window payloads, not
   * here.  Known cosmetic waste, carried over from the MATLAB: a condensed
   * Store can land right after a JumpAnchor, duplicating that window's anchor
   * into a checkpoint slot -- correct, but the memory model should either
   * charge or waive it explicitly.
   */
  class BswRevolve {

  public:

    BswRevolve(const int & num_steps,
               const std::vector<std::pair<int,int> > & windows,
               const int & num_checkpoints)
      : N_(num_steps), windows_(windows) {

      if (num_checkpoints < 1) {
        throw std::invalid_argument("BswRevolve: num_checkpoints must be at least 1.");
      }
      validateWindows(num_steps, windows);

      // condensed maps: walk the real axis, collapsing windows.  Entry jc
      // (1-based condensed step) records the real position reached after it
      // and which window it collapses (-1 = a plain real step).
      const int num_windows = static_cast<int>(windows_.size());
      int wi = 0;
      int k = 1;
      while (k <= N_) {
        if (wi < num_windows && k == windows_[wi].first + 1) {
          step_of_c_.push_back(windows_[wi].second);
          win_of_c_.push_back(wi);
          k = windows_[wi].second + 1;
          ++wi;
        }
        else {
          step_of_c_.push_back(k);
          win_of_c_.push_back(-1);
          ++k;
        }
      }

      Nc_ = static_cast<int>(step_of_c_.size());
      Neff_ = N_;
      for (size_t i=0; i<windows_.size(); ++i) {
        Neff_ -= windows_[i].second - windows_[i].first;
      }

      inner_.reset(new Revolve(Nc_, num_checkpoints));
    }

    /// The next primitive directive; translates inner Revolve actions on demand.
    BswDirective next() {
      while (queue_.empty()) {
        translate();
      }
      BswDirective d = queue_.front();
      queue_.pop_front();
      return d;
    }

    int getNumSteps()          const { return N_; }
    int getNumCondensedSteps() const { return Nc_; }
    int getNumOutsideSteps()   const { return Neff_; }
    int getNumWindows()        const { return static_cast<int>(windows_.size()); }

    /**
     * @brief Reject malformed window tables: out of range, shorter than 2
     *        steps (that is a plain checkpoint), unsorted or overlapping.
     */
    static void validateWindows(const int & num_steps,
                                const std::vector<std::pair<int,int> > & windows) {
      for (size_t i=0; i<windows.size(); ++i) {
        const int a = windows[i].first;
        const int b = windows[i].second;
        if (a < 0 || b > num_steps) {
          throw std::invalid_argument("BswRevolve: window out of range [0,"
                                      + std::to_string(num_steps) + "].");
        }
        if (b - a < 2) {
          throw std::invalid_argument(
            "BswRevolve: a window shorter than 2 steps is a plain checkpoint -- reject.");
        }
        if (i > 0 && (windows[i].first <= windows[i-1].first ||
                      windows[i].first < windows[i-1].second)) {
          throw std::invalid_argument(
            "BswRevolve: windows must be sorted and disjoint (adjacent is fine).");
        }
      }
    }

    /**
     * @brief Blocks of the uniform m-partition of 1..N: block j (1-based)
     *        covers steps (j-1)*m+1 .. j*m, i.e. the window [(j-1)*m, j*m].
     */
    static std::vector<std::pair<int,int> > uniformWindows(const int & num_steps,
                                                           const int & block_size,
                                                           const std::vector<int> & block_ids) {
      const int num_blocks = num_steps/block_size;
      std::vector<int> ids = block_ids;
      std::sort(ids.begin(), ids.end());
      std::vector<std::pair<int,int> > windows;
      for (size_t i=0; i<ids.size(); ++i) {
        if (ids[i] < 1 || ids[i] > num_blocks) {
          throw std::invalid_argument("BswRevolve: block ids must lie in 1.."
                                      + std::to_string(num_blocks) + ".");
        }
        windows.push_back(std::make_pair((ids[i]-1)*block_size, ids[i]*block_size));
      }
      return windows;
    }

    /// Solve counts for one gradient, by kind.
    struct GradientSolves {
      long advance_solves = 0;   ///< forward solves inside SolveRange runs
      long reverse_solves = 0;   ///< one recompute per ReverseStep
      long anchor_jumps = 0;
      long window_reverses = 0;
      long total = 0;            ///< advance + reverse (window interiors are free)
    };

    /**
     * @brief Integer simulation of one gradient.  Sketches and anchors are
     *        built during the forward sweep, so a gradient costs only the
     *        advance solves plus one recompute per real-step reverse.
     */
    static GradientSolves predictGradientSolves(const int & num_steps,
                                                const std::vector<std::pair<int,int> > & windows,
                                                const int & num_checkpoints) {
      BswRevolve schedule(num_steps, windows, num_checkpoints);
      GradientSolves s;
      while (true) {
        BswDirective d = schedule.next();
        if (d.action == BswAction::SolveRange)         { s.advance_solves += d.k2 - d.k1 + 1; }
        else if (d.action == BswAction::ReverseStep)   { s.reverse_solves += 1; }
        else if (d.action == BswAction::JumpAnchor)    { s.anchor_jumps += 1; }
        else if (d.action == BswAction::ReverseWindow) { s.window_reverses += 1; }
        else if (d.action == BswAction::Terminate)     { break; }
      }
      s.total = s.advance_solves + s.reverse_solves;
      return s;
    }

  private:

    // Expand one inner Revolve action into primitive directives on the real axis.
    void translate() {

      const int prev = inner_->getRangeStart();
      const RevolveAction action = inner_->next();

      if (action == RevolveAction::Advance) {

        // expand condensed steps prev+1..rangeStart, merging contiguous real
        // steps into one SolveRange and crossing windows by anchor jump
        int run_start = 0;
        for (int jc=prev+1; jc<=inner_->getRangeStart(); ++jc) {
          if (win_of_c_[jc-1] >= 0) {
            if (run_start > 0) {
              BswDirective d;
              d.action = BswAction::SolveRange;
              d.k1 = run_start;
              d.k2 = step_of_c_[jc-2];
              queue_.push_back(d);
              run_start = 0;
            }
            BswDirective d;
            d.action = BswAction::JumpAnchor;
            d.win = win_of_c_[jc-1];
            d.step = step_of_c_[jc-1];
            queue_.push_back(d);
          }
          else if (run_start == 0) {
            run_start = step_of_c_[jc-1];
          }
        }
        if (run_start > 0) {
          BswDirective d;
          d.action = BswAction::SolveRange;
          d.k1 = run_start;
          d.k2 = step_of_c_[inner_->getRangeStart()-1];
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

        const int jc = inner_->getRangeStart() + 1;    // condensed step being reversed
        const int w = win_of_c_[jc-1];
        BswDirective d;
        d.is_first = (action == RevolveAction::FirstReverse);
        if (w >= 0) {
          d.action = BswAction::ReverseWindow;
          d.win = w;
          d.a = windows_[w].first;
          d.b = windows_[w].second;
        }
        else {
          d.action = BswAction::ReverseStep;
          d.step = step_of_c_[jc-1];
        }
        queue_.push_back(d);
      }
      else if (action == RevolveAction::Terminate) {
        BswDirective d;
        d.action = BswAction::Terminate;
        queue_.push_back(d);
      }
      else {
        throw std::runtime_error("BswRevolve: inner Revolve reported an inconsistent "
                                 "state (status " + std::to_string(inner_->getStatus()) + ").");
      }
    }

    /// Real position after pc condensed steps (pc = 0 means position 0 = u_0).
    int realPosition(const int & pc) const {
      return (pc == 0) ? 0 : step_of_c_[pc-1];
    }

    int N_;                                      ///< real time steps
    std::vector<std::pair<int,int> > windows_;   ///< [a,b] rows, sorted, disjoint
    int Nc_ = 0;                                 ///< condensed steps = Neff + num windows
    int Neff_ = 0;                               ///< real steps outside windows
    std::vector<int> step_of_c_;                 ///< real position reached after condensed step
    std::vector<int> win_of_c_;                  ///< window collapsed by that step (-1 = none)
    std::unique_ptr<Revolve> inner_;             ///< classic Revolve on the condensed axis
    std::deque<BswDirective> queue_;             ///< pending primitive directives
  };

}

#endif
