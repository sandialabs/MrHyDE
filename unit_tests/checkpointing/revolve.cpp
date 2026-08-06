#include "revolve.hpp"
#include <iostream>

using namespace MrHyDE;
#include <vector>

struct ScheduleStats {
    std::vector<int> stores;
    int forward_solves = 0;
    int reversals      = 0;
    int max_stored     = 0;
    bool terminated    = false;
};
/// Drive a Revolve schedule to completion with no PDE solves, just schedule 
ScheduleStats runSchedule(const int & num_steps, const int & num_checkpoints) {

    ScheduleStats stats;
    Revolve r(num_steps, num_checkpoints);
    int guard = 0;

    while (guard < 100000) {

        const int prev = r.getRangeStart();
        RevolveAction action = r.next();

        if (r.getNumCheckpointsStored() > stats.max_stored) {
            stats.max_stored = r.getNumCheckpointsStored();
        }

        if (action == RevolveAction::Advance) {
            stats.forward_solves += r.getRangeStart() - prev;
        }
        else if (action == RevolveAction::Store) {
            stats.stores.push_back(r.getRangeStart());
        }
        else if (action == RevolveAction::Reverse || action == RevolveAction::FirstReverse) {
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
    // run a whole schedule and check it completes correctly
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
        }
    }

    // --- two special cases
    ScheduleStats sierra_gap = runSchedule(5, 2);
    if (sierra_gap.forward_solves != 11) {
        std::cout << "FAIL: (5,2) should cost 11 solves (Sierra pays 12), got "
                << sierra_gap.forward_solves << std::endl;
        ++num_failures;
    }

    ScheduleStats tiny = runSchedule(2, 3);      // more checkpoints than steps
    if (!tiny.terminated || tiny.reversals != 2) {
        std::cout << "FAIL: (2,3) should terminate with 2 reversals (Sierra throws here)"
                << std::endl;
        ++num_failures;
    }
    
    

    

    if (num_failures == 0){
        std::cout << "All revolve tests PASSED" << std::endl;
    }
    

    return num_failures;
}