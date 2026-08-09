#include "bsw_window_manager.hpp"
#include "bsw_revolve.hpp"
#include "dense_kernels.hpp"
#include "threefry.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace MrHyDE;

/// Synthetic trajectory: rank-r inside [lo, hi], novel directions elsewhere.
std::vector<double> trajectory(const int & M, const int & n, const int & r,
                               const int & lo, const int & hi) {
    Threefry rng(55, 1);
    std::vector<double> L(static_cast<size_t>(M)*r);
    rng.fillNormal(0, 0, M*r, L.data());
    std::vector<double> U(static_cast<size_t>(M)*n, 0.0);
    Threefry coeffs(55, 2);
    for (int k=1; k<=n; ++k) {
        double* col = U.data() + static_cast<size_t>(k-1)*M;
        for (int c=0; c<r; ++c) {
            const double w = coeffs.normal(k, c);
            dense::axpy(M, w/std::sqrt(static_cast<double>(M)), L.data() + static_cast<size_t>(c)*M, col);
        }
        if (k < lo || k > hi) {
            col[(37*k)%M] += 5.0;      // novel direction outside the window band
        }
    }
    return U;
}

int main(){

    int num_failures = 0;
    const int M = 300;
    const int n = 60;

    // rank-3 band over steps 15..45, full-rank noise outside it
    std::vector<double> U = trajectory(M, n, 3, 15, 45);

    // --- planned mode: one window over the band commits; its steps leave the
    // classic axis and the unspent pool flows back into slots
    {
        BswManagerOptions opt;
        opt.total_budget_se = 40.0;
        opt.min_checkpoints = 2;
        opt.base_rank = 3;
        opt.tol = 1.0e-8;
        opt.master_seed = 7;
        opt.planned_windows.push_back(std::make_pair(15, 45));

        BswWindowManager<SketchWindow> manager(opt);
        manager.beginSweep(M, n, 0.0);
        for (int k=1; k<=n; ++k) {
            manager.recordState(k, U.data() + static_cast<size_t>(k-1)*M, 0.1*k);
        }
        manager.endSweep();

        if (manager.effectiveWindows().size() != 1 ||
            manager.effectiveWindows()[0].a != 15 ||
            manager.effectiveWindows()[0].b_eff != 45) {
            std::cout << "FAIL: planned window over the band should commit in full" << std::endl;
            ++num_failures;
        }

        // reconstructions serve the band; the right edge is the exact state
        std::vector<double> uhat(M), d(M);
        double emax = 0.0;
        for (int k=16; k<=45; ++k) {
            manager.reconstruct(0, k, uhat.data());
            const double* truth = U.data() + static_cast<size_t>(k-1)*M;
            for (int i=0; i<M; ++i) { d[i] = uhat[i] - truth[i]; }
            emax = std::max(emax, dense::norm2(M, d.data())/dense::norm2(M, truth));
        }
        if (emax > 1.0e-8) {
            std::cout << "FAIL: band reconstruction error " << emax << std::endl;
            ++num_failures;
        }
        manager.rightEdge(0, uhat.data());
        bool exact = true;
        for (int i=0; i<M; ++i) {
            if (uhat[i] != U[static_cast<size_t>(44)*M + i]) { exact = false; }
        }
        if (!exact) {
            std::cout << "FAIL: right edge should be the exact recorded state" << std::endl;
            ++num_failures;
        }

        // budget honesty: slots = floor + floor(unspent), spent > 0, and the
        // recorded times come back verbatim
        if (manager.getSpentSE() <= 0.0 ||
            manager.numCheckpointSlots() < opt.min_checkpoints) {
            std::cout << "FAIL: budget accounting is inconsistent" << std::endl;
            ++num_failures;
        }
        const double expected_slots = opt.min_checkpoints
            + std::floor(opt.total_budget_se - 10.0 - opt.min_checkpoints
                         - manager.getSpentSE());
        if (manager.numCheckpointSlots() != static_cast<int>(expected_slots)) {
            std::cout << "FAIL: flow-back slots " << manager.numCheckpointSlots()
                    << ", want " << expected_slots << std::endl;
            ++num_failures;
        }
        if (manager.getTime(30) != 3.0) {
            std::cout << "FAIL: recorded time not returned verbatim" << std::endl;
            ++num_failures;
        }

        // the effective windows must form a valid BSW schedule
        std::vector<std::pair<int,int> > spans;
        for (size_t i=0; i<manager.effectiveWindows().size(); ++i) {
            spans.push_back(std::make_pair(manager.effectiveWindows()[i].a,
                                           manager.effectiveWindows()[i].b_eff));
        }
        BswRevolve::GradientSolves p =
            BswRevolve::predictGradientSolves(n, spans, manager.numCheckpointSlots());
        if (p.window_reverses != 1 || p.reverse_solves != n - 30) {
            std::cout << "FAIL: schedule accounting over the effective windows" << std::endl;
            ++num_failures;
        }
    }

    // --- greedy mode: discovery opens windows only where the budget model
    // fits, commits inside the band, and never overspends the pool.  The
    // honest count makes a rank-3 rolling window cost about 60 SE while
    // recording, so the pool must actually afford one.
    {
        BswManagerOptions opt;
        opt.total_budget_se = 120.0;
        opt.min_checkpoints = 2;
        opt.base_rank = 3;
        opt.tol = 1.0e-8;
        opt.master_seed = 11;

        BswWindowManager<SketchWindow> manager(opt);
        manager.beginSweep(M, n, 0.0);
        for (int k=1; k<=n; ++k) {
            manager.recordState(k, U.data() + static_cast<size_t>(k-1)*M, 0.1*k);
        }
        manager.endSweep();

        if (manager.effectiveWindows().empty()) {
            std::cout << "FAIL: greedy should commit at least one window on the band" << std::endl;
            ++num_failures;
        }
        if (manager.getSpentSE() > 120.0 - 10.0 - opt.min_checkpoints) {
            std::cout << "FAIL: greedy overspent the window pool" << std::endl;
            ++num_failures;
        }
        if (manager.getPeakSE() <= 0.0) {
            std::cout << "FAIL: peak transient SE should be tracked" << std::endl;
            ++num_failures;
        }

        // every committed span reconstructs within tolerance
        std::vector<double> uhat(M), d(M);
        for (size_t wi=0; wi<manager.effectiveWindows().size(); ++wi) {
            const BswEffectiveWindow & e = manager.effectiveWindows()[wi];
            for (int k=e.a+1; k<=e.b_eff; ++k) {
                manager.reconstruct(static_cast<int>(wi), k, uhat.data());
                const double* truth = U.data() + static_cast<size_t>(k-1)*M;
                for (int i=0; i<M; ++i) { d[i] = uhat[i] - truth[i]; }
                const double e_col = dense::norm2(M, d.data())/dense::norm2(M, truth);
                if (e_col > 1.0e-7) {
                    std::cout << "FAIL: greedy window " << wi << " column " << k
                            << " error " << e_col << std::endl;
                    ++num_failures;
                    break;
                }
            }
        }

        std::vector<std::string> report = manager.layoutReport();
        if (report.empty()) {
            std::cout << "FAIL: the layout report should list every window" << std::endl;
            ++num_failures;
        }
    }

    // --- zero budget: greedy must open nothing and hand every slot back
    {
        BswManagerOptions opt;
        opt.total_budget_se = 0.0;
        opt.min_checkpoints = 3;
        BswWindowManager<SketchWindow> manager(opt);
        manager.beginSweep(M, n, 0.0);
        for (int k=1; k<=n; ++k) {
            manager.recordState(k, U.data() + static_cast<size_t>(k-1)*M, 0.1*k);
        }
        manager.endSweep();

        if (!manager.effectiveWindows().empty() ||
            manager.numCheckpointSlots() != opt.min_checkpoints) {
            std::cout << "FAIL: zero budget should reduce to classic checkpointing" << std::endl;
            ++num_failures;
        }
    }

    // --- the 'rm' compressor drops in through the same interface
    {
        BswManagerOptions opt;
        opt.total_budget_se = 60.0;
        opt.min_checkpoints = 2;
        opt.base_rank = 6;
        opt.tol = 1.0e-8;
        opt.planned_windows.push_back(std::make_pair(15, 45));

        std::unique_ptr<BswWindowManagerBase> manager = makeBswWindowManager("rm", opt);
        manager->beginSweep(M, n, 0.0);
        for (int k=1; k<=n; ++k) {
            manager->recordState(k, U.data() + static_cast<size_t>(k-1)*M, 0.1*k);
        }
        manager->endSweep();

        if (manager->effectiveWindows().size() != 1) {
            std::cout << "FAIL: rm compressor should commit the planned band" << std::endl;
            ++num_failures;
        }

        bool threw = false;
        try { makeBswWindowManager("qtt", opt); }
        catch (const std::invalid_argument &) { threw = true; }
        if (!threw) {
            std::cout << "FAIL: unknown compressor name should be rejected" << std::endl;
            ++num_failures;
        }
    }

    if (num_failures == 0) {
        std::cout << "All BswWindowManager tests PASSED" << std::endl;
    }

    return num_failures == 0 ? 0 : 1;
}
