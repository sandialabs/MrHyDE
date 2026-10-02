#include "revolve.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace MrHyDE;

/// What one complete schedule did.
struct ScheduleStats {
    std::vector<int> stores;          ///< step number of each checkpoint, in order stored
    std::vector<int> restores;        ///< step number each Restore landed on
    std::vector<int> reversed_steps;  ///< time step reversed, in the order reversed
    int forward_solves = 0;           ///< advance steps, plus one recompute per reversal
    int reversals      = 0;
    int first_reverses = 0;           ///< how many FirstReverse actions were emitted
    int max_stored     = 0;
    bool terminated    = false;
    bool slots_consistent = true;     ///< the driver's slot bookkeeping held throughout
};

/// Drive a Revolve schedule to completion with no PDE solves, just schedule
ScheduleStats runSchedule(const int & num_steps, const int & num_checkpoints) {

    ScheduleStats stats;
    Revolve r(num_steps, num_checkpoints);
    int guard = 0;

    // Mirror what solverManager_checkpointing.hpp does with the schedule: keep the
    // carried state's step number and the contents of each storage slot, indexing
    // by getNumCheckpointsStored()-1 exactly as the driver does.  Without this the
    // test would pass on a next() that got every count right but handed the driver
    // the wrong slot.
    int carried_step = 0;
    std::vector<int> slot(num_checkpoints, -1);

    while (guard < 100000) {

        const int prev = r.getRangeStart();
        RevolveAction action = r.next();

        if (r.getNumCheckpointsStored() > stats.max_stored) {
            stats.max_stored = r.getNumCheckpointsStored();
        }

        if (action == RevolveAction::Advance) {
            stats.forward_solves += r.getRangeStart() - prev;
            carried_step = r.getRangeStart();
        }
        else if (action == RevolveAction::Store) {
            stats.stores.push_back(r.getRangeStart());
            const int index = r.getNumCheckpointsStored() - 1;
            if (index < 0 || index >= num_checkpoints || carried_step != r.getRangeStart()) {
                stats.slots_consistent = false;
            }
            else {
                slot[index] = carried_step;
            }
        }
        else if (action == RevolveAction::Restore) {
            stats.restores.push_back(r.getRangeStart());
            const int index = r.getNumCheckpointsStored() - 1;
            if (index < 0 || index >= num_checkpoints || slot[index] != r.getRangeStart()) {
                stats.slots_consistent = false;
            }
            else {
                carried_step = slot[index];
            }
        }
        else if (action == RevolveAction::Reverse || action == RevolveAction::FirstReverse) {
            if (action == RevolveAction::FirstReverse) {
                ++stats.first_reverses;
            }
            // a reversal at range_start handles time step range_start+1, and the
            // driver enters it holding the state at range_start
            if (carried_step != r.getRangeStart()) {
                stats.slots_consistent = false;
            }
            stats.reversed_steps.push_back(r.getRangeStart() + 1);
            ++stats.reversals;
            ++stats.forward_solves;   // each reversal recomputes one state
        }
        else if (action == RevolveAction::Terminate) {
            stats.terminated = true;
            break;
        }
        else if (action == RevolveAction::Error) {
            break;
        }

        ++guard;
    }

    return stats;
}

int main(){

    int num_failures = 0;

    long beta_3_1 = Revolve::maxReversibleSteps(3,1);
    if (beta_3_1 != 4){
        std::cout << "Fail: beta(3,1) should be 4, got " << beta_3_1 << std::endl;
        ++num_failures;
    }
    long beta_3_2 = Revolve::maxReversibleSteps(3,2);
    if (beta_3_2 != 10){
        std::cout << "Fail: beta(3,2) should be 10, got " << beta_3_2 << std::endl;
        ++num_failures;
    }
    long beta_10_10 = Revolve::maxReversibleSteps(10,10);
    if (beta_10_10 != 184756){
        std::cout << "Fail: beta(10,10) should be 184756, got " << beta_10_10 << std::endl;
        ++num_failures;
    }
    // placementOffset leans on these conventions: beta is 1 for zero sweeps and
    // 0 for a negative argument, which is how the paper's special cases fall out
    if (Revolve::maxReversibleSteps(3,0) != 1 || Revolve::maxReversibleSteps(0,5) != 1) {
        std::cout << "FAIL: beta with a zero argument should be 1" << std::endl;
        ++num_failures;
    }
    if (Revolve::maxReversibleSteps(3,-1) != 0 || Revolve::maxReversibleSteps(-1,3) != 0) {
        std::cout << "FAIL: beta with a negative argument should be 0" << std::endl;
        ++num_failures;
    }

    long p_10_3 = Revolve::minExtraForwardSteps(10,3);
    if (p_10_3 != 15) {
        std::cout << "FAIL: p(10,3) should be 15, got " << p_10_3 << std::endl;
        ++num_failures;
    }
    if (Revolve::placementOffset(10,3,2) != 4) {
        std::cout << "FAIL: placementOffset(10,3,2) should be 4, got "
                << Revolve::placementOffset(10,3,2) << std::endl;
        ++num_failures;
    }
    if (Revolve::placementOffset(6,2,2) != 3) {
        std::cout << "FAIL: placementOffset(6,2,2) should be 3, got "
                << Revolve::placementOffset(6,2,2) << std::endl;
        ++num_failures;
    }
    // the two above both land in the second case; pin the other two as well
    if (Revolve::placementOffset(4,2,2) != 1) {
        std::cout << "FAIL: placementOffset(4,2,2) should be 1, got "
                << Revolve::placementOffset(4,2,2) << std::endl;
        ++num_failures;
    }
    if (Revolve::placementOffset(5,2,2) != 2) {
        std::cout << "FAIL: placementOffset(5,2,2) should be 2, got "
                << Revolve::placementOffset(5,2,2) << std::endl;
        ++num_failures;
    }

    // a freshly built schedule: whole range still to reverse, nothing stored yet
    Revolve r(10, 3);

    if (r.getRangeStart() != 0) {
        std::cout << "FAIL: new Revolve should start at step 0, got "
                << r.getRangeStart() << std::endl;
        ++num_failures;
    }
    if (r.getRangeEnd() != 10) {
        std::cout << "FAIL: new Revolve should end at step 10, got "
                << r.getRangeEnd() << std::endl;
        ++num_failures;
    }
    if (r.getNumCheckpointsStored() != 0) {
        std::cout << "FAIL: new Revolve should hold 0 checkpoints, got "
                << r.getNumCheckpointsStored() << std::endl;
        ++num_failures;
    }
    // --- Figure 1, Algorithm 799 p.24: 10 steps, 3 checkpoints -> checkpoints at 0, 4, 7
    ScheduleStats fig1 = runSchedule(10, 3);

    if (fig1.stores.size() < 3 ||
        fig1.stores[0] != 0 || fig1.stores[1] != 4 || fig1.stores[2] != 7) {
        std::cout << "FAIL: Figure 1 checkpoints should be 0 4 7; got";
        for (size_t i = 0; i < fig1.stores.size() && i < 3; ++i) {
            std::cout << " " << fig1.stores[i];
        }
        std::cout << std::endl;
        ++num_failures;
    }

    // --- cost and budget across a sweep of problem sizes
    for (int n : {3, 5, 10, 20, 50, 100}) {
        for (int c = 1; c <= 6; ++c) {

            ScheduleStats s = runSchedule(n, c);
            const long want = n + Revolve::minExtraForwardSteps(n, c);

            if (!s.terminated) {
                std::cout << "FAIL: schedule (" << n << "," << c << ") did not terminate" << std::endl;
                ++num_failures;
            }
            if (s.reversals != n) {
                std::cout << "FAIL: schedule (" << n << "," << c << ") reversed "
                        << s.reversals << " steps, want " << n << std::endl;
                ++num_failures;
            }
            if (s.max_stored > c) {
                std::cout << "FAIL: schedule (" << n << "," << c << ") stored "
                        << s.max_stored << ", budget " << c << std::endl;
                ++num_failures;
            }
            if (s.forward_solves != want) {
                std::cout << "FAIL: schedule (" << n << "," << c << ") cost "
                        << s.forward_solves << " solves, want " << want << std::endl;
                ++num_failures;
            }

            // FirstReverse is what tells the driver to seed the terminal adjoint
            // condition, so there must be exactly one
            if (s.first_reverses != 1) {
                std::cout << "FAIL: schedule (" << n << "," << c << ") emitted "
                        << s.first_reverses << " FirstReverse actions, want 1" << std::endl;
                ++num_failures;
            }

            // steps must come out n, n-1, ..., 1 with none skipped or repeated
            bool order_ok = (s.reversed_steps.size() == static_cast<size_t>(n));
            for (size_t i = 0; order_ok && i < s.reversed_steps.size(); ++i) {
                if (s.reversed_steps[i] != n - static_cast<int>(i)) {
                    order_ok = false;
                }
            }
            if (!order_ok) {
                std::cout << "FAIL: schedule (" << n << "," << c
                        << ") did not reverse steps in strict order n..1" << std::endl;
                ++num_failures;
            }

            if (!s.slots_consistent) {
                std::cout << "FAIL: schedule (" << n << "," << c
                        << ") broke the slot contract the driver relies on" << std::endl;
                ++num_failures;
            }

            // every Restore must land on a step that was checkpointed earlier
            for (size_t i = 0; i < s.restores.size(); ++i) {
                bool was_stored = false;
                for (size_t j = 0; j < s.stores.size(); ++j) {
                    if (s.stores[j] == s.restores[i]) {
                        was_stored = true;
                    }
                }
                if (!was_stored) {
                    std::cout << "FAIL: schedule (" << n << "," << c << ") restored step "
                            << s.restores[i] << ", which was never stored" << std::endl;
                    ++num_failures;
                }
            }
        }
    }

    // --- Figure 1 must actually use both FirstReverse and Restore
    if (fig1.reversed_steps.empty() || fig1.first_reverses != 1) {
        std::cout << "FAIL: Figure 1 should emit exactly one FirstReverse" << std::endl;
        ++num_failures;
    }
    if (fig1.restores.empty()) {
        std::cout << "FAIL: Figure 1 must use Restore to rewind to checkpoints" << std::endl;
        ++num_failures;
    }

    // --- the constructor must reject budgets it cannot honour
    bool threw_on_zero_checkpoints = false;
    try {
        Revolve bad(10, 0);
    }
    catch (const std::invalid_argument &) {
        threw_on_zero_checkpoints = true;
    }
    if (!threw_on_zero_checkpoints) {
        std::cout << "FAIL: Revolve(10,0) should throw -- a schedule needs at least one checkpoint"
                << std::endl;
        ++num_failures;
    }

    bool threw_on_zero_steps = false;
    try {
        Revolve bad(0, 3);
    }
    catch (const std::invalid_argument &) {
        threw_on_zero_steps = true;
    }
    if (!threw_on_zero_steps) {
        std::cout << "FAIL: Revolve(0,3) should throw -- there is nothing to reverse" << std::endl;
        ++num_failures;
    }

    // --- budgets where Sierra's heuristic version misbehaves
    ScheduleStats tight_budget = runSchedule(5, 2);
    if (tight_budget.forward_solves != 11) {
        std::cout << "FAIL: (5,2) should cost 11 solves (Sierra pays 12), got "
                << tight_budget.forward_solves << std::endl;
        ++num_failures;
    }

    ScheduleStats tiny_range = runSchedule(2, 3);      // more checkpoints than steps
    if (!tiny_range.terminated || tiny_range.reversals != 2) {
        std::cout << "FAIL: (2,3) should terminate with 2 reversals (Sierra throws here)"
                << std::endl;
        ++num_failures;
    }
    
    

    

    if (num_failures == 0){
        std::cout << "All revolve tests PASSED" << std::endl;
    }
    

    return num_failures == 0 ? 0 : 1;
}