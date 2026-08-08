#include "sketch_window.hpp"
#include "dense_kernels.hpp"
#include "threefry.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace MrHyDE;

/// Column-major M x n matrix with deterministic normal entries.
std::vector<double> randomMatrix(const int & M, const int & n, const uint64_t & stream) {
    Threefry rng(99, stream);
    std::vector<double> A(static_cast<size_t>(M)*n);
    rng.fillNormal(0, 0, M*n, A.data());
    return A;
}

/// Exactly rank-r column block: U = L * R', scaled so columns are O(1).
std::vector<double> lowRankBlock(const int & M, const int & m, const int & r,
                                 const uint64_t & stream) {
    std::vector<double> L = randomMatrix(M, r, stream);
    std::vector<double> R = randomMatrix(m, r, stream + 50);
    std::vector<double> U(static_cast<size_t>(M)*m, 0.0);
    dense::gemm('N', 'T', M, m, r, 1.0/std::sqrt(static_cast<double>(M)), L.data(), M,
                R.data(), m, 0.0, U.data(), M);
    return U;
}

double relColErr(const int & M, const double* approx, const double* truth) {
    std::vector<double> d(M);
    for (int i=0; i<M; ++i) { d[i] = approx[i] - truth[i]; }
    return dense::norm2(M, d.data())/dense::norm2(M, truth);
}

int main(){

    int num_failures = 0;
    const int M = 200;
    const int m = 40;

    // --- exactly rank-3 data, full-adaptive at rank 3: accepted, no retries
    {
        std::vector<double> U = lowRankBlock(M, m, 3, 1);
        WindowPolicy p = SketchWindow::fillPolicy();
        SketchWindow w(0, m, 3, 1, 42);
        w.beginRecord(M, p);
        for (int k=1; k<=m; ++k) {
            w.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }

        if (w.getStatus() != WindowStatus::Accepted || w.getBEff() != m ||
            w.getRetryAttempt() != 0 || w.getRank() != 3) {
            std::cout << "FAIL: rank-3 data at rank 3 should be accepted untouched" << std::endl;
            ++num_failures;
        }
        double emax = 0.0;
        std::vector<double> uhat(M);
        for (int k=1; k<=m; ++k) {
            w.reconstruct(k, uhat.data());
            emax = std::max(emax, relColErr(M, uhat.data(), U.data() + static_cast<size_t>(k-1)*M));
        }
        if (emax > 1.0e-10) {
            std::cout << "FAIL: exact-rank reconstruction error " << emax << std::endl;
            ++num_failures;
        }
        if (w.getErrMaxCol() > 1.0e-10) {
            std::cout << "FAIL: reported errMaxCol " << w.getErrMaxCol() << std::endl;
            ++num_failures;
        }
    }

    // --- rank-8 data started at rank 3: the retry ladder must rescue it,
    // with no PDE solves (re-streams from the buffer)
    {
        std::vector<double> U = lowRankBlock(M, m, 8, 2);
        WindowPolicy p = SketchWindow::fillPolicy();
        SketchWindow w(0, m, 3, 2, 42);
        w.beginRecord(M, p);
        for (int k=1; k<=m; ++k) {
            w.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }

        if (w.getStatus() != WindowStatus::Accepted) {
            std::cout << "FAIL: rank-8 data should be rescued by retries" << std::endl;
            ++num_failures;
        }
        if (w.getRetryAttempt() < 1 || w.getRank() <= 3) {
            std::cout << "FAIL: retry should have raised the rank (attempts "
                    << w.getRetryAttempt() << ", rank " << w.getRank() << ")" << std::endl;
            ++num_failures;
        }
        if (w.getErrMaxCol() > 1.0e-8) {
            std::cout << "FAIL: rescued window errMaxCol " << w.getErrMaxCol() << std::endl;
            ++num_failures;
        }
    }

    // --- rolling mode, moving front from column 26: one novel direction per
    // column, so the width-9 sketch (rank 4) keeps up until the accumulated
    // rank passes 9.  The j=30 monitor still verifies (rank 7); the j=35
    // monitor fails (rank 12) and cuts to the verified prefix at 30, with the
    // exact state u_30 as anchor.  No retries are possible: the states are gone.
    {
        std::vector<double> U = lowRankBlock(M, m, 2, 3);
        for (int k=26; k<=m; ++k) {                 // novel direction per column
            U[static_cast<size_t>(k-1)*M + 100 + k] += 5.0;
        }
        WindowPolicy p = SketchWindow::fillPolicy();
        p.rolling = true;
        SketchWindow w(0, m, 4, 3, 42);
        w.beginRecord(M, p);
        for (int k=1; k<=m && w.isOpen(); ++k) {
            w.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }

        if (w.getStatus() != WindowStatus::Prefix || w.getBEff() != 30) {
            std::cout << "FAIL: rolling front should cut at the verified prefix 30, got bEff "
                    << w.getBEff() << std::endl;
            ++num_failures;
        }
        std::vector<double> anchor(M);
        w.anchorRaw(anchor.data());
        bool anchor_exact = true;
        for (int i=0; i<M; ++i) {
            if (anchor[i] != U[static_cast<size_t>(29)*M + i]) { anchor_exact = false; }
        }
        if (!anchor_exact) {
            std::cout << "FAIL: prefix anchor is not the exact state u_30" << std::endl;
            ++num_failures;
        }
        double emax = 0.0;
        std::vector<double> uhat(M);
        for (int k=1; k<=30; ++k) {
            w.reconstruct(k, uhat.data());
            emax = std::max(emax, relColErr(M, uhat.data(), U.data() + static_cast<size_t>(k-1)*M));
        }
        if (emax > 1.0e-8) {
            std::cout << "FAIL: committed prefix error " << emax << std::endl;
            ++num_failures;
        }
    }

    // --- full-rank data at rank 1, rolling: nothing verifiable, fall back
    // visibly -- never a silently wrong payload
    {
        std::vector<double> U = randomMatrix(M, m, 4);
        WindowPolicy p = SketchWindow::fillPolicy();
        p.rolling = true;
        SketchWindow w(0, m, 1, 4, 42);
        w.beginRecord(M, p);
        for (int k=1; k<=m && w.isOpen(); ++k) {
            w.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }

        if (w.getStatus() != WindowStatus::Fallback || w.getBEff() != 0 ||
            w.getPayloadColumns() != 0) {
            std::cout << "FAIL: hopeless window should fall back with no payload" << std::endl;
            ++num_failures;
        }
    }

    // --- static mode: finalize commits, and identical seeds give identical
    // payloads (the seed-rematerialized design in one assertion)
    {
        std::vector<double> U = lowRankBlock(M, m, 4, 5);

        SketchWindow w1(0, m, 4, 7, 123);
        SketchWindow w2(0, m, 4, 7, 123);
        w1.beginRecord(M);
        w2.beginRecord(M);
        for (int k=1; k<=m; ++k) {
            w1.record(U.data() + static_cast<size_t>(k-1)*M, k);
            w2.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }
        w1.finalize(1.0e-8);
        w2.finalize(1.0e-8);

        if (w1.getStatus() != WindowStatus::Accepted) {
            std::cout << "FAIL: static exact-rank window should be accepted" << std::endl;
            ++num_failures;
        }
        std::vector<double> r1(M), r2(M);
        bool identical = true;
        for (int k=1; k<=m; ++k) {
            w1.reconstruct(k, r1.data());
            w2.reconstruct(k, r2.data());
            for (int i=0; i<M; ++i) {
                if (r1[i] != r2[i]) { identical = false; }
            }
        }
        if (!identical) {
            std::cout << "FAIL: identical seeds must give bit-identical reconstructions" << std::endl;
            ++num_failures;
        }

        // bookkeeping sanity
        if (w1.storageSE() <= w1.payloadSE()) {
            std::cout << "FAIL: storage should exceed payload while the anchor is held" << std::endl;
            ++num_failures;
        }
        w1.releaseAnchor();
        if (w1.storageSE() != w1.payloadSE()) {
            std::cout << "FAIL: releasing the anchor should equalize storage and payload" << std::endl;
            ++num_failures;
        }
        w1.rightEdgeRaw(r1.data());     // must fall back to reconstruction
        w1.discardPayload();
        if (w1.getStatus() != WindowStatus::Fallback || w1.storageSE() != 0.0) {
            std::cout << "FAIL: discardPayload should empty the window" << std::endl;
            ++num_failures;
        }
    }

    // --- a window shorter than the sketch width: every cadence monitor is a
    // young auto-pass, so the never-skipped j == m monitor is the only real
    // verification -- it must still run and commit
    {
        const int m_short = 6;
        std::vector<double> U = lowRankBlock(M, m_short, 2, 6);
        WindowPolicy p = SketchWindow::fillPolicy();
        p.rolling = true;
        SketchWindow w(0, m_short, 2, 8, 42);
        w.beginRecord(M, p);
        for (int k=1; k<=m_short; ++k) {
            w.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }
        if (w.getStatus() != WindowStatus::Accepted || w.getBEff() != m_short) {
            std::cout << "FAIL: short window should be verified and accepted at j == m" << std::endl;
            ++num_failures;
        }
    }

    // --- misuse must throw, not corrupt
    {
        SketchWindow w(0, m, 3, 9, 42);
        w.beginRecord(M);
        std::vector<double> u(M, 1.0);
        w.record(u.data(), 1);

        bool threw = false;
        try { w.record(u.data(), 3); }              // skipped column 2
        catch (const std::runtime_error &) { threw = true; }
        if (!threw) {
            std::cout << "FAIL: out-of-order record should throw" << std::endl;
            ++num_failures;
        }

        threw = false;
        try { std::vector<double> out(M); w.reconstruct(1, out.data()); }
        catch (const std::runtime_error &) { threw = true; }
        if (!threw) {
            std::cout << "FAIL: reconstruct before commit should throw" << std::endl;
            ++num_failures;
        }
    }

    if (num_failures == 0) {
        std::cout << "All SketchWindow tests PASSED" << std::endl;
    }

    return num_failures == 0 ? 0 : 1;
}
