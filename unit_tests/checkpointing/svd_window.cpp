#include "svd_window.hpp"
#include "dense_kernels.hpp"
#include "threefry.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace MrHyDE;

// same helpers as sketch_window.cpp, different seed family
std::vector<double> randomMatrix(const int & M, const int & n, const uint64_t & stream) {
    Threefry rng(77, stream);
    std::vector<double> A(static_cast<size_t>(M)*n);
    rng.fillNormal(0, 0, M*n, A.data());
    return A;
}

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

    // --- static, exactly rank-3 data with cap 10: the working rank settles at
    // the data rank, and the payload carries the TRUE rank, not the cap
    {
        std::vector<double> U = lowRankBlock(M, m, 3, 1);
        SvdWindow w(0, m, 10, 1, 42);
        w.beginRecord(M);
        for (int k=1; k<=m; ++k) {
            w.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }
        w.finalize(1.0e-8);

        if (w.getStatus() != WindowStatus::Accepted || w.getBEff() != m) {
            std::cout << "FAIL: static exact-rank window should be accepted" << std::endl;
            ++num_failures;
        }
        if (w.getPayloadColumns() != 3) {
            std::cout << "FAIL: payload should carry the true rank 3, got "
                    << w.getPayloadColumns() << std::endl;
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
    }

    // --- rolling with a rank cap: a moving front grows the basis to the cap,
    // then the overflow pre-check cuts BEFORE the unrepresentable column is
    // applied -- the committed factors are exact for every served column
    {
        std::vector<double> U = lowRankBlock(M, m, 2, 2);
        for (int k=21; k<=m; ++k) {                 // novel direction per column
            U[static_cast<size_t>(k-1)*M + 100 + k] += 5.0;
        }
        WindowPolicy p = SvdWindow::fillPolicy();
        p.rolling = true;
        SvdWindow w(0, m, 4, 2, 42);
        w.beginRecord(M, p);
        for (int k=1; k<=m && w.isOpen(); ++k) {
            w.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }

        // columns 1..20 need rank 2; columns 21 and 22 grow the basis to the
        // cap of 4; column 23 cannot be represented and cuts at 22
        if (w.getStatus() != WindowStatus::Prefix || w.getBEff() != 22) {
            std::cout << "FAIL: capped front should cut at 22, got bEff "
                    << w.getBEff() << std::endl;
            ++num_failures;
        }
        std::vector<double> anchor(M);
        w.anchorRaw(anchor.data());
        bool anchor_exact = true;
        for (int i=0; i<M; ++i) {
            if (anchor[i] != U[static_cast<size_t>(21)*M + i]) { anchor_exact = false; }
        }
        if (!anchor_exact) {
            std::cout << "FAIL: prefix anchor is not the exact state u_22" << std::endl;
            ++num_failures;
        }
        double emax = 0.0;
        std::vector<double> uhat(M);
        for (int k=1; k<=22; ++k) {
            w.reconstruct(k, uhat.data());
            emax = std::max(emax, relColErr(M, uhat.data(), U.data() + static_cast<size_t>(k-1)*M));
        }
        if (emax > 1.0e-8) {
            std::cout << "FAIL: committed prefix error " << emax << std::endl;
            ++num_failures;
        }
        if (w.getNextRankHint() < 5) {
            std::cout << "FAIL: the hint should ask for more rank than the cap, got "
                    << w.getNextRankHint() << std::endl;
            ++num_failures;
        }
    }

    // --- repeated columns take the guard-row path: the basis stays rank 1
    {
        std::vector<double> u0 = randomMatrix(M, 1, 3);
        SvdWindow w(0, 10, 5, 3, 42);
        w.beginRecord(M);
        for (int k=1; k<=10; ++k) {
            w.record(u0.data(), k);
        }
        w.finalize(1.0e-10);

        if (w.getStatus() != WindowStatus::Accepted || w.getPayloadColumns() != 1) {
            std::cout << "FAIL: identical columns should commit at rank 1, got "
                    << w.getPayloadColumns() << std::endl;
            ++num_failures;
        }
        std::vector<double> uhat(M);
        w.reconstruct(7, uhat.data());
        if (relColErr(M, uhat.data(), u0.data()) > 1.0e-12) {
            std::cout << "FAIL: rank-1 reconstruction error" << std::endl;
            ++num_failures;
        }
    }

    // --- a long window crosses the every-32-columns re-orthonormalization;
    // the factors must stay accurate through it
    {
        const int m_long = 70;
        std::vector<double> U = lowRankBlock(M, m_long, 5, 4);
        SvdWindow w(0, m_long, 12, 4, 42);
        w.beginRecord(M);
        for (int k=1; k<=m_long; ++k) {
            w.record(U.data() + static_cast<size_t>(k-1)*M, k);
        }
        w.finalize(1.0e-8);

        if (w.getStatus() != WindowStatus::Accepted) {
            std::cout << "FAIL: long window should be accepted" << std::endl;
            ++num_failures;
        }
        double emax = 0.0;
        std::vector<double> uhat(M);
        for (int k=1; k<=m_long; ++k) {
            w.reconstruct(k, uhat.data());
            emax = std::max(emax, relColErr(M, uhat.data(), U.data() + static_cast<size_t>(k-1)*M));
        }
        if (emax > 1.0e-9) {
            std::cout << "FAIL: long-window reconstruction error " << emax << std::endl;
            ++num_failures;
        }
    }

    // --- adaptive non-rolling has no monitor to commit it: reject outright
    {
        SvdWindow w(0, m, 4, 5, 42);
        WindowPolicy p = SvdWindow::fillPolicy();     // rolling = false
        bool threw = false;
        try { w.beginRecord(M, p); }
        catch (const std::invalid_argument &) { threw = true; }
        if (!threw) {
            std::cout << "FAIL: adaptive non-rolling SvdWindow should be rejected" << std::endl;
            ++num_failures;
        }
    }

    if (num_failures == 0) {
        std::cout << "All SvdWindow tests PASSED" << std::endl;
    }

    return num_failures == 0 ? 0 : 1;
}
