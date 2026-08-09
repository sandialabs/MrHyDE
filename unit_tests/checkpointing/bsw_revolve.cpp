#include "bsw_revolve.hpp"
#include "revolve.hpp"

#include <iostream>
#include <utility>
#include <vector>

using namespace MrHyDE;

typedef std::vector<std::pair<int,int> > WindowTable;

/// What one complete BSW schedule did, as seen by a virtual driver.
struct BswStats {
    long advance_solves = 0;
    long reverse_solves = 0;
    long anchor_jumps = 0;
    long window_reverses = 0;
    int max_slot_used = 0;          ///< 1-based count of the highest slot touched
    int first_reversals = 0;        ///< directives with is_first set
    bool first_was_first = false;   ///< the is_first directive led the reversals
    bool invariants_ok = true;
    bool terminated = false;
    std::vector<int> reversed;      ///< every real step, in the order reversed
};

/**
 * Drive a schedule against a virtual driver whose whole state is the current
 * real position and the slot contents -- the same bookkeeping the PDE driver
 * will do, with the physics stripped out.  Any directive that asks for a state
 * the driver does not hold is an invariant failure.
 */
BswStats runSchedule(const int & num_steps, const WindowTable & windows,
                     const int & num_checkpoints) {

    BswStats stats;
    BswRevolve schedule(num_steps, windows, num_checkpoints);

    int pos = 0;                                        // current state is u_pos
    std::vector<int> slot(num_checkpoints, -1);
    long guard = 0;
    const long guard_max = 100000;

    while (guard++ < guard_max) {
        BswDirective d = schedule.next();

        if (d.action == BswAction::SolveRange) {
            if (d.k1 != pos + 1 || d.k2 < d.k1 || d.k2 > num_steps) {
                stats.invariants_ok = false;
            }
            stats.advance_solves += d.k2 - d.k1 + 1;
            pos = d.k2;
        }
        else if (d.action == BswAction::JumpAnchor) {
            // anchors only launch from the window's exact left edge
            if (d.win < 0 || pos != windows[d.win].first ||
                d.step != windows[d.win].second) {
                stats.invariants_ok = false;
            }
            ++stats.anchor_jumps;
            pos = d.step;
        }
        else if (d.action == BswAction::Store) {
            if (d.slot < 0 || d.slot >= num_checkpoints || d.step != pos) {
                stats.invariants_ok = false;
            }
            else {
                slot[d.slot] = pos;
                stats.max_slot_used = std::max(stats.max_slot_used, d.slot + 1);
            }
        }
        else if (d.action == BswAction::Restore) {
            if (d.slot < 0 || d.slot >= num_checkpoints || slot[d.slot] != d.step) {
                stats.invariants_ok = false;
            }
            pos = d.step;
        }
        else if (d.action == BswAction::ReverseStep) {
            // the classic invariant: the driver holds u_{k-1} on entry
            if (pos != d.step - 1) {
                stats.invariants_ok = false;
            }
            if (d.is_first) {
                ++stats.first_reversals;
                if (stats.reversed.empty()) { stats.first_was_first = true; }
            }
            stats.reversed.push_back(d.step);
            ++stats.reverse_solves;
        }
        else if (d.action == BswAction::ReverseWindow) {
            // window reversal enters with the exact left-edge state u_a
            if (d.win < 0 || pos != d.a ||
                d.a != windows[d.win].first || d.b != windows[d.win].second) {
                stats.invariants_ok = false;
            }
            if (d.is_first) {
                ++stats.first_reversals;
                if (stats.reversed.empty()) { stats.first_was_first = true; }
            }
            for (int k=d.b; k>d.a; --k) {
                stats.reversed.push_back(k);
            }
            ++stats.window_reverses;
        }
        else if (d.action == BswAction::Terminate) {
            stats.terminated = true;
            break;
        }
    }

    // every real step reversed exactly once, in strictly decreasing order
    if (stats.reversed.size() != static_cast<size_t>(num_steps)) {
        stats.invariants_ok = false;
    }
    for (size_t i=0; i<stats.reversed.size(); ++i) {
        if (stats.reversed[i] != num_steps - static_cast<int>(i)) {
            stats.invariants_ok = false;
        }
    }
    if (stats.first_reversals != 1 || !stats.first_was_first) {
        stats.invariants_ok = false;
    }

    return stats;
}

int main(){

    int num_failures = 0;

    // --- no windows: the schedule must be classic Revolve, action for action
    for (int n : {8, 33, 100}) {
        for (int c : {2, 3, 5, 8}) {

            BswStats s = runSchedule(n, WindowTable(), c);

            if (!s.terminated || !s.invariants_ok) {
                std::cout << "FAIL: windowless schedule (" << n << "," << c
                        << ") broke an invariant" << std::endl;
                ++num_failures;
            }
            if (s.anchor_jumps != 0 || s.window_reverses != 0) {
                std::cout << "FAIL: windowless schedule (" << n << "," << c
                        << ") issued window directives" << std::endl;
                ++num_failures;
            }
            if (s.advance_solves != Revolve::minExtraForwardSteps(n, c)) {
                std::cout << "FAIL: windowless schedule (" << n << "," << c
                        << ") cost " << s.advance_solves << " advances, want "
                        << Revolve::minExtraForwardSteps(n, c) << std::endl;
                ++num_failures;
            }
            if (s.max_slot_used > c) {
                std::cout << "FAIL: windowless schedule (" << n << "," << c
                        << ") exceeded the slot budget" << std::endl;
                ++num_failures;
            }
        }
    }

    // --- windowed layouts: interior blocks, a window at each end, adjacent
    // windows, and a long sketched prefix
    {
        const int n = 40;
        std::vector<WindowTable> layouts;
        layouts.push_back(BswRevolve::uniformWindows(n, 10, {2, 3}));
        layouts.push_back(WindowTable{{0, 5}});
        layouts.push_back(WindowTable{{35, 40}});
        layouts.push_back(WindowTable{{5, 10}, {10, 15}});
        layouts.push_back(WindowTable{{0, 12}, {12, 24}});

        for (size_t l=0; l<layouts.size(); ++l) {
            for (int c : {2, 3}) {

                const WindowTable & w = layouts[l];
                BswStats s = runSchedule(n, w, c);

                int outside = n;
                for (size_t i=0; i<w.size(); ++i) {
                    outside -= w[i].second - w[i].first;
                }

                if (!s.terminated || !s.invariants_ok) {
                    std::cout << "FAIL: layout " << l << " (c=" << c
                            << ") broke a directive invariant" << std::endl;
                    ++num_failures;
                }
                if (s.reverse_solves != outside) {
                    std::cout << "FAIL: layout " << l << " (c=" << c << ") paid "
                            << s.reverse_solves << " reverse solves, want " << outside
                            << std::endl;
                    ++num_failures;
                }
                if (s.window_reverses != static_cast<long>(w.size())) {
                    std::cout << "FAIL: layout " << l << " (c=" << c
                            << ") reversed " << s.window_reverses << " windows, want "
                            << w.size() << std::endl;
                    ++num_failures;
                }
                if (s.max_slot_used > c) {
                    std::cout << "FAIL: layout " << l << " (c=" << c
                            << ") exceeded the slot budget" << std::endl;
                    ++num_failures;
                }
            }
        }
    }

    // --- malformed window tables must be rejected outright
    {
        const int n = 40;
        std::vector<WindowTable> bad;
        bad.push_back(WindowTable{{0, 1}});             // shorter than 2 steps
        bad.push_back(WindowTable{{5, 3}});             // reversed edges
        bad.push_back(WindowTable{{0, 10}, {5, 15}});   // overlapping
        bad.push_back(WindowTable{{20, 30}, {0, 10}});  // unsorted
        bad.push_back(WindowTable{{-1, 5}});            // negative edge
        bad.push_back(WindowTable{{35, 45}});           // past the last step

        for (size_t i=0; i<bad.size(); ++i) {
            bool threw = false;
            try { BswRevolve schedule(n, bad[i], 3); }
            catch (const std::invalid_argument &) { threw = true; }
            if (!threw) {
                std::cout << "FAIL: malformed window table " << i
                        << " was accepted" << std::endl;
                ++num_failures;
            }
        }
    }

    // --- proposal-scale accounting: N=500 in blocks of 50 with a 125-SE pool
    // and window cost m/zeta, zeta=3.  Sketching K_s blocks trades checkpoint
    // slots for free window reversals; the K_s=0 row is the classic 873.
    {
        const int n = 500, block = 50, pool = 125, zeta = 3;

        for (int Ks=0; Ks<=7; ++Ks) {

            std::vector<int> ids;
            for (int j=1; j<=Ks; ++j) { ids.push_back(j); }
            const WindowTable w = BswRevolve::uniformWindows(n, block, ids);

            const int slots = pool - (Ks*block)/zeta;
            const int outside = n - Ks*block;

            BswStats s = runSchedule(n, w, slots);
            BswRevolve::GradientSolves p = BswRevolve::predictGradientSolves(n, w, slots);

            if (!s.terminated || !s.invariants_ok) {
                std::cout << "FAIL: proposal schedule Ks=" << Ks
                        << " broke an invariant" << std::endl;
                ++num_failures;
            }
            if (s.advance_solves != p.advance_solves || s.reverse_solves != p.reverse_solves) {
                std::cout << "FAIL: predictGradientSolves disagrees with the schedule at Ks="
                        << Ks << std::endl;
                ++num_failures;
            }
            if (Ks == 0 && s.advance_solves != 873) {
                std::cout << "FAIL: the classic row should cost exactly 873 advances, got "
                        << s.advance_solves << std::endl;
                ++num_failures;
            }
            if (s.reverse_solves != outside || s.window_reverses != Ks) {
                std::cout << "FAIL: Ks=" << Ks << " reversed "
                        << s.reverse_solves << "+" << s.window_reverses
                        << ", want " << outside << "+" << Ks << std::endl;
                ++num_failures;
            }
            // the condensed axis bounds the advance cost
            const long bound = Revolve::minExtraForwardSteps(outside + Ks, slots);
            if (s.advance_solves > bound) {
                std::cout << "FAIL: Ks=" << Ks << " advances "
                        << s.advance_solves << " exceed the condensed bound "
                        << bound << std::endl;
                ++num_failures;
            }
        }
    }

    if (num_failures == 0) {
        std::cout << "All BswRevolve tests PASSED" << std::endl;
    }

    return num_failures == 0 ? 0 : 1;
}
