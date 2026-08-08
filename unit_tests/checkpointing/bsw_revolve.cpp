#include "bsw_revolve.hpp"
#include "revolve.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace MrHyDE;

/// What one complete BSW schedule did, plus whether the driver invariants held.
struct BswStats {
    long advance_solves = 0;
    long reverse_solves = 0;
    long anchor_jumps = 0;
    long window_reverses = 0;
    int max_slot = -1;
    bool terminated = false;
    bool invariants_ok = true;
    std::vector<std::string> violations;
};

/// Drive a schedule against a virtual position, checking every directive
/// invariant the real driver will rely on.  No physics, just integers.
BswStats runSchedule(const int & num_steps,
                     const std::vector<std::array<int,2> > & windows,
                     const int & num_checkpoints) {

    BswStats stats;
    BswRevolve bsw(num_steps, windows, num_checkpoints);

    int pos = 0;                                        // current state is u_pos
    std::vector<int> slot(num_checkpoints, -1);         // slot contents, by step number
    std::vector<int> times_reversed(num_steps+1, 0);    // per real step
    int last_reversed = num_steps + 1;                  // reversals must be decreasing
    long guard = 0;

    auto fail = [&](const std::string & why) {
        stats.invariants_ok = false;
        stats.violations.push_back(why);
    };

    while (guard++ < 1000000) {

        BswDirective d = bsw.next();

        if (d.action == BswAction::SolveRange) {
            if (d.k1 != pos + 1) { fail("solveRange does not continue from the current position"); }
            if (d.k2 < d.k1)     { fail("solveRange with empty range"); }
            stats.advance_solves += d.k2 - d.k1 + 1;
            pos = d.k2;
        }
        else if (d.action == BswAction::JumpAnchor) {
            if (d.win < 0 || d.win >= static_cast<int>(windows.size())) { fail("jumpAnchor bad window id"); }
            else {
                if (pos != windows[d.win][0])  { fail("jumpAnchor fired away from the window's left edge"); }
                if (d.step != windows[d.win][1]) { fail("jumpAnchor does not land on the window's right edge"); }
            }
            ++stats.anchor_jumps;
            pos = d.step;
        }
        else if (d.action == BswAction::Store) {
            if (d.slot < 0 || d.slot >= num_checkpoints) { fail("store slot out of range"); }
            else {
                if (d.step != pos) { fail("store step disagrees with the current position"); }
                slot[d.slot] = pos;
                if (d.slot > stats.max_slot) { stats.max_slot = d.slot; }
            }
        }
        else if (d.action == BswAction::Restore) {
            if (d.slot < 0 || d.slot >= num_checkpoints) { fail("restore slot out of range"); }
            else if (slot[d.slot] != d.step) { fail("restore from a slot that does not hold that step"); }
            pos = d.step;
        }
        else if (d.action == BswAction::ReverseStep) {
            if (pos != d.k - 1) { fail("reverseStep entered without u_{k-1} in hand"); }
            if (d.k < 1 || d.k > num_steps) { fail("reverseStep step out of range"); }
            else {
                ++times_reversed[d.k];
                if (d.k >= last_reversed) { fail("reversals not strictly decreasing"); }
                last_reversed = d.k;
            }
            ++stats.reverse_solves;
        }
        else if (d.action == BswAction::ReverseWindow) {
            if (d.win < 0 || d.win >= static_cast<int>(windows.size())) { fail("reverseWindow bad window id"); }
            else {
                if (pos != d.a) { fail("reverseWindow entered away from the exact bottom edge u_a"); }
                for (int k=d.b; k>d.a; --k) {
                    ++times_reversed[k];
                    if (k >= last_reversed) { fail("reversals not strictly decreasing"); }
                    last_reversed = k;
                }
            }
            ++stats.window_reverses;
        }
        else if (d.action == BswAction::Terminate) {
            stats.terminated = true;
            break;
        }
        else {
            fail("schedule returned Error");
            break;
        }
    }

    for (int k=1; k<=num_steps; ++k) {
        if (times_reversed[k] != 1) {
            fail("a real step was not reversed exactly once");
            break;
        }
    }

    return stats;
}

/// With no windows the directive stream must replay classic Revolve exactly:
/// same stores, same restores, same reversal order, same advance expansion.
bool matchesClassicRevolve(const int & num_steps, const int & num_checkpoints) {

    Revolve classic(num_steps, num_checkpoints);
    BswRevolve bsw(num_steps, std::vector<std::array<int,2> >(), num_checkpoints);
    long guard = 0;

    while (guard++ < 1000000) {

        const int prev = classic.getRangeStart();
        RevolveAction a = classic.next();
        BswDirective d;

        if (a == RevolveAction::Advance) {
            d = bsw.next();
            if (d.action != BswAction::SolveRange) { return false; }
            if (d.k1 != prev + 1 || d.k2 != classic.getRangeStart()) { return false; }
        }
        else if (a == RevolveAction::Store) {
            d = bsw.next();
            if (d.action != BswAction::Store) { return false; }
            if (d.slot != classic.getNumCheckpointsStored() - 1) { return false; }
            if (d.step != classic.getRangeStart()) { return false; }
        }
        else if (a == RevolveAction::Restore) {
            d = bsw.next();
            if (d.action != BswAction::Restore) { return false; }
            if (d.slot != classic.getNumCheckpointsStored() - 1) { return false; }
            if (d.step != classic.getRangeStart()) { return false; }
        }
        else if (a == RevolveAction::FirstReverse || a == RevolveAction::Reverse) {
            d = bsw.next();
            if (d.action != BswAction::ReverseStep) { return false; }
            if (d.k != classic.getRangeStart() + 1) { return false; }
            if (d.is_first != (a == RevolveAction::FirstReverse)) { return false; }
        }
        else if (a == RevolveAction::Terminate) {
            d = bsw.next();
            return d.action == BswAction::Terminate;
        }
        else {
            return false;
        }
    }
    return false;
}

int main(){

    int num_failures = 0;

    // --- S1: with no windows, replay classic Revolve action-for-action
    for (int n : {8, 33, 100}) {
        for (int c : {2, 3, 5, 8}) {
            if (!matchesClassicRevolve(n, c)) {
                std::cout << "FAIL: (" << n << "," << c
                        << ") with no windows does not replay classic Revolve" << std::endl;
                ++num_failures;
            }
        }
    }

    // --- S2: window layouts, every directive invariant checked
    struct Layout {
        int n;
        std::vector<std::array<int,2> > windows;
        const char * name;
    };
    std::vector<Layout> layouts = {
        {40, BswRevolve::uniformWindows(40, 5, {3, 6}), "interior blocks"},
        {40, {{{0, 5}}},                                "window at the start"},
        {40, {{{35, 40}}},                              "window at the end"},
        {40, {{{5, 10}, {10, 15}}},                     "adjacent windows"},
        {24, {{{0, 12}, {12, 24}}},                     "everything sketched"},
    };

    for (size_t i=0; i<layouts.size(); ++i) {
        const Layout & L = layouts[i];
        const int num_windows = static_cast<int>(L.windows.size());
        int windowed_steps = 0;
        for (int w=0; w<num_windows; ++w) { windowed_steps += L.windows[w][1] - L.windows[w][0]; }
        const int n_eff = L.n - windowed_steps;

        BswStats s = runSchedule(L.n, L.windows, 3);

        if (!s.terminated || !s.invariants_ok) {
            std::cout << "FAIL: layout '" << L.name << "'";
            if (!s.violations.empty()) { std::cout << " -- " << s.violations[0]; }
            std::cout << std::endl;
            ++num_failures;
        }
        if (s.reverse_solves != n_eff) {
            std::cout << "FAIL: layout '" << L.name << "' used " << s.reverse_solves
                    << " reverse solves, want " << n_eff << std::endl;
            ++num_failures;
        }
        if (s.window_reverses != num_windows) {
            std::cout << "FAIL: layout '" << L.name << "' reversed " << s.window_reverses
                    << " windows, want " << num_windows << std::endl;
            ++num_failures;
        }
        if (s.max_slot >= 3) {
            std::cout << "FAIL: layout '" << L.name << "' exceeded the checkpoint budget" << std::endl;
            ++num_failures;
        }
    }

    // --- S2b: malformed window tables must be rejected
    struct BadTable {
        std::vector<std::array<int,2> > windows;
        const char * why;
    };
    std::vector<BadTable> bad = {
        {{{{0, 1}}},           "shorter than 2 steps"},
        {{{{5, 3}}},           "reversed edges"},
        {{{{0, 5}, {3, 8}}},   "overlapping"},
        {{{{10, 15}, {0, 5}}}, "unsorted"},
        {{{{-1, 5}}},          "negative edge"},
        {{{{0, 99}}},          "out of range"},
    };
    for (size_t i=0; i<bad.size(); ++i) {
        bool threw = false;
        try {
            BswRevolve reject(40, bad[i].windows, 3);
        }
        catch (const std::invalid_argument &) {
            threw = true;
        }
        if (!threw) {
            std::cout << "FAIL: window table (" << bad[i].why << ") was accepted" << std::endl;
            ++num_failures;
        }
    }

    // --- S3: proposal-scale accounting.  N=500 in blocks of m=50; a storage
    // budget of K=125 state-equivalents is split between sketched windows
    // (each costs m/zeta state-equivalents at compression factor zeta=3) and
    // classic checkpoints (one state-equivalent each).
    {
        const int n = 500;
        const int m = 50;
        const int budget = 125;
        const int zeta = 3;

        for (int num_windows=0; num_windows<=7; ++num_windows) {

            std::vector<int> block_ids;
            for (int j=1; j<=num_windows; ++j) { block_ids.push_back(j); }
            std::vector<std::array<int,2> > windows =
                BswRevolve::uniformWindows(n, m, block_ids);

            const int num_checkpoints = budget - num_windows*m/zeta;
            const int n_eff = n - num_windows*m;
            const int n_condensed = n_eff + num_windows;

            BswStats s = runSchedule(n, windows, num_checkpoints);

            if (!s.terminated || !s.invariants_ok) {
                std::cout << "FAIL: accounting sweep broke at " << num_windows << " windows";
                if (!s.violations.empty()) { std::cout << " -- " << s.violations[0]; }
                std::cout << std::endl;
                ++num_failures;
            }
            if (s.reverse_solves != n_eff) {
                std::cout << "FAIL: " << num_windows << " windows used " << s.reverse_solves
                        << " reverse solves, want " << n_eff << std::endl;
                ++num_failures;
            }

            // each condensed advance costs at most one real solve (macro-steps
            // are free anchor jumps), so the classic count is an upper bound
            const long classic_advances =
                Revolve::minExtraForwardSteps(n_condensed, num_checkpoints);
            if (s.advance_solves > classic_advances) {
                std::cout << "FAIL: " << num_windows << " windows cost " << s.advance_solves
                        << " advance solves, bound " << classic_advances << std::endl;
                ++num_failures;
            }

            // the no-window row is exactly classic Revolve on the full axis
            if (num_windows == 0 && s.advance_solves != 873) {
                std::cout << "FAIL: no-window advance solves should be 873 = p(500,125), got "
                        << s.advance_solves << std::endl;
                ++num_failures;
            }
        }
    }

    // --- predictGradientSolves must agree with the measured schedule
    {
        std::vector<std::array<int,2> > windows = BswRevolve::uniformWindows(40, 5, {3, 6});
        BswStats s = runSchedule(40, windows, 3);
        BswRevolve::SolveCounts p = BswRevolve::predictGradientSolves(40, windows, 3);
        if (p.advance_solves != s.advance_solves || p.reverse_solves != s.reverse_solves ||
            p.anchor_jumps != s.anchor_jumps || p.window_reverses != s.window_reverses) {
            std::cout << "FAIL: predictGradientSolves disagrees with the measured schedule" << std::endl;
            ++num_failures;
        }
    }

    if (num_failures == 0) {
        std::cout << "All BswRevolve tests PASSED" << std::endl;
    }

    return num_failures == 0 ? 0 : 1;
}
