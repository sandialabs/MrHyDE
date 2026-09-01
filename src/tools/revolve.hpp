#ifndef MRHYDE_REVOLVE_H
#define MRHYDE_REVOLVE_H

#include <cmath>
#include <stdexcept>
#include <vector>

namespace MrHyDE {
    
    /**
   * @brief The six things a Revolve schedule can ask the driver to do.
   */
   enum class RevolveAction{
    Advance,
    Store,
    Restore,
    FirstReverse,
    Reverse,
    Terminate,
    Error
   };

  class Revolve {

  public:

    /**
     * @brief Build a schedule for reversing num_steps time steps using at
     *        most num_checkpoints stored states.
     */
    Revolve(const int & num_steps, const int & num_checkpoints) {

      if (num_checkpoints < 1) {
        throw std::invalid_argument("Revolve: num_checkpoints must be at least 1.");
      }
      if (num_steps < 1) {
        throw std::invalid_argument("Revolve: num_steps must be at least 1.");
      }

      max_checkpoints_        = num_checkpoints;
      num_checkpoints_stored_ = 0;
      range_start_            = 0;
      range_end_              = num_steps;
      adjoint_started_        = false;
      status_                 = 0;

      checkpoint_step_.assign(num_checkpoints, -1);
    }

    int getNumCheckpointsStored() const { return num_checkpoints_stored_; }
    int getRangeStart()           const { return range_start_; }
    int getRangeEnd()             const { return range_end_; }
    int getStatus()               const { return status_; }

    /**
     * @brief Decide what the driver should do next, and advance the schedule.
     *        Griewank & Walther, Algorithm 799, p.30.
     */
    RevolveAction next() {

      // 1. sanity: the bookkeeping must never get into these states
      if (num_checkpoints_stored_ < 0 || range_start_ > range_end_) {
        status_ = 10;
        return RevolveAction::Error;
      }

      // nothing stored yet: start a fresh sweep over this subrange
      if (num_checkpoints_stored_ == 0 && range_start_ < range_end_) {
        adjoint_started_    = false;
        checkpoint_step_[0] = range_start_ - 1;   // sentinel: no real checkpoint here
      }

      const int range_length = range_end_ - range_start_;

      // 2. subrange exhausted: finished, or pop back to a checkpoint
      if (range_length == 0) {
        if (num_checkpoints_stored_ == 0 || range_start_ == checkpoint_step_[0]) {
          status_ = 13;
          return RevolveAction::Terminate;
        }
        range_start_ = checkpoint_step_[num_checkpoints_stored_ - 1];
        return RevolveAction::Restore;
      }

      // 3. exactly one step left: reverse it
      if (range_length == 1) {
        range_end_ = range_end_ - 1;

        // if a checkpoint sits right here, it is now spent
        if (num_checkpoints_stored_ >= 1 &&
            checkpoint_step_[num_checkpoints_stored_ - 1] == range_start_) {
          num_checkpoints_stored_ = num_checkpoints_stored_ - 1;
        }

        if (!adjoint_started_) {
          adjoint_started_ = true;
          return RevolveAction::FirstReverse;
        }
        return RevolveAction::Reverse;
      }

      // 4. no checkpoint at the current position: put one here
      if (num_checkpoints_stored_ == 0 ||
          checkpoint_step_[num_checkpoints_stored_ - 1] != range_start_) {

        num_checkpoints_stored_ = num_checkpoints_stored_ + 1;

        if (num_checkpoints_stored_ > max_checkpoints_) {
          status_ = 14;
          return RevolveAction::Error;
        }

        checkpoint_step_[num_checkpoints_stored_ - 1] = range_start_;
        return RevolveAction::Store;
      }

      // 5. otherwise: run forward to the next checkpoint position
      const int free_checkpoints = max_checkpoints_ - num_checkpoints_stored_ + 1;

      if (free_checkpoints < 1) {
        status_ = 11;
        return RevolveAction::Error;
      }

      int depth = 0;
      double reachable = 1.0;
      while (reachable < range_length) {
        depth = depth + 1;
        reachable = reachable*(free_checkpoints + depth)/depth;
      }

      range_start_ = range_start_ + placementOffset(range_length, free_checkpoints, depth);

      return RevolveAction::Advance;
    }

    /**
     * @brief beta(s,t) = C(s+t,s): the most time steps reversible with
     *        num_checkpoints checkpoints and num_sweeps forward sweeps.
     *        Griewank & Walther, Algorithm 799, eq. (1), p.23.
     *
     * Returns 0 for a negative argument and 1 for a zero sweep count; both
     * conventions are relied on by placementOffset.  Accumulated in double, so
     * the result is exact while C(s+t,s) stays below 2^53 -- far beyond the
     * subrange lengths next() evaluates it at.
     */
    static long maxReversibleSteps(const int & num_checkpoints, const int & num_sweeps) {

      if (num_checkpoints < 0 || num_sweeps < 0) {
        return 0;
      }

      double steps = 1.0;
      for (int i = 1; i <= num_sweeps; i++) {
        steps = steps*(num_checkpoints + i)/i;
      }

      return std::llround(steps);
    }

    /**
     * @brief p(m,s): the fewest extra forward steps needed to reverse
     *        num_steps steps with num_checkpoints checkpoints.  Total
     *        forward solves = num_steps + p.  Alg. 799, eq. (3), p.25.
     */
    static long minExtraForwardSteps(const int & num_steps, const int & num_checkpoints) {

      if (num_checkpoints < 1) {
        throw std::invalid_argument("Revolve: num_checkpoints must be at least 1.");
      }

      int depth = 0;
      double reachable = 1.0;
      while (reachable < num_steps) {
        depth = depth + 1;
        reachable = reachable*(depth + num_checkpoints)/depth;
      }

      // widen before multiplying: depth*num_steps is quadratic for a small
      // budget and overflows int at 46342 steps
      const long long total_steps = static_cast<long long>(depth)*num_steps;
      const long long saved_steps = std::llround(reachable*depth/(num_checkpoints + 1.0));

      return static_cast<long>(total_steps - saved_steps);
    }

    /**
     * @brief How far ahead to place the next checkpoint, given a subrange of
     *        range_length steps, free_checkpoints free slots, and depth sweeps.
     *        Griewank & Walther, Algorithm 799, p.34.
     */
     static int placementOffset(const int & range_length, const int & free_checkpoints, const int & depth){
        const int m = range_length;
        const int s = free_checkpoints;
        const int t = depth;

        long offset = 0;

        if (m <= maxReversibleSteps(s, t-1) + maxReversibleSteps(s-2, t -1)){
            offset = maxReversibleSteps(s, t - 2);
        }
        else if (m >= maxReversibleSteps(s,t) - maxReversibleSteps(s - 3, t)){
            offset = maxReversibleSteps(s,t - 1);
        }
        else {
            offset = m - maxReversibleSteps(s - 1, t - 1) - maxReversibleSteps(s - 2, t - 1);
        }
        if (offset < 1){
            offset = 1;
        }
        if (offset > m - 1){
            offset = m - 1;
        }
        return static_cast<int>(offset);
     }

  private:

    int max_checkpoints_;         ///< storage budget (never changes)
    int num_checkpoints_stored_;  ///< how many are held; the newest is in slot num_checkpoints_stored_-1
    int range_start_;             ///< first step of the subrange left to reverse (0-based)
    int range_end_;               ///< last step of that subrange (0-based)
    int status_;                  ///< 0 = OK; >0 = irregular termination code
    bool adjoint_started_;        ///< true once the first reverse step has been taken

    std::vector<int> checkpoint_step_;  ///< checkpoint_step_[slot] = step number held there

  };

}

#endif
