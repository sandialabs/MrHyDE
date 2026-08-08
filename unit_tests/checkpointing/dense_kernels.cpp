#include "dense_kernels.hpp"
#include "threefry.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace MrHyDE;

/// Max-magnitude entry of x - y.
double maxDiff(const std::vector<double> & x, const std::vector<double> & y) {
    double d = 0.0;
    for (size_t i=0; i<x.size(); ++i) {
        const double e = std::abs(x[i] - y[i]);
        if (e > d) { d = e; }
    }
    return d;
}

/// Column-major m x n matrix with deterministic normal entries.
std::vector<double> randomMatrix(const int & m, const int & n, const uint64_t & stream) {
    Threefry rng(2026, stream);
    std::vector<double> A(static_cast<size_t>(m)*n);
    rng.fillNormal(0, 0, m*n, A.data());
    return A;
}

int main(){

    int num_failures = 0;
    const double tol = 1.0e-12;

    // --- gemm and gemv against hand-computed values
    {
        // A = [1 2; 3 4; 5 6], column-major
        const std::vector<double> A = {1, 3, 5, 2, 4, 6};
        const std::vector<double> B = {7, 9, 8, 10};      // [7 8; 9 10]
        std::vector<double> C(6, 0.0);
        dense::gemm('N', 'N', 3, 2, 2, 1.0, A.data(), 3, B.data(), 2, 0.0, C.data(), 3);
        const std::vector<double> want_C = {25, 57, 89, 28, 64, 100};
        if (maxDiff(C, want_C) > tol) {
            std::cout << "FAIL: gemm hand-computed product" << std::endl;
            ++num_failures;
        }

        const std::vector<double> x = {1, 1};
        std::vector<double> y(3, 0.0);
        dense::gemv('N', 3, 2, 1.0, A.data(), 3, x.data(), 0.0, y.data());
        const std::vector<double> want_y = {3, 7, 11};
        if (maxDiff(y, want_y) > tol) {
            std::cout << "FAIL: gemv hand-computed product" << std::endl;
            ++num_failures;
        }

        // transposed: A^T * [1;1;1] = [9; 12]
        std::vector<double> yt(2, 0.0);
        const std::vector<double> ones = {1, 1, 1};
        dense::gemv('T', 3, 2, 1.0, A.data(), 3, ones.data(), 0.0, yt.data());
        if (std::abs(yt[0] - 9.0) > tol || std::abs(yt[1] - 12.0) > tol) {
            std::cout << "FAIL: gemv transposed product" << std::endl;
            ++num_failures;
        }
    }

    // --- QR: orthonormal factor, exact reconstruction
    {
        const int m = 50, n = 8;
        std::vector<double> A = randomMatrix(m, n, 1);
        const std::vector<double> A_orig = A;

        std::vector<double> R;
        dense::qr(m, n, A, R);          // A now holds Q

        // Q^T Q = I
        std::vector<double> QtQ(static_cast<size_t>(n)*n, 0.0);
        dense::gemm('T', 'N', n, n, m, 1.0, A.data(), m, A.data(), m, 0.0, QtQ.data(), n);
        double orth = 0.0;
        for (int j=0; j<n; ++j) {
            for (int i=0; i<n; ++i) {
                const double want = (i == j) ? 1.0 : 0.0;
                orth = std::max(orth, std::abs(QtQ[j*n + i] - want));
            }
        }
        if (orth > 1.0e-13) {
            std::cout << "FAIL: QR orthonormality error " << orth << std::endl;
            ++num_failures;
        }

        // Q*R = A
        std::vector<double> QR(static_cast<size_t>(m)*n, 0.0);
        dense::gemm('N', 'N', m, n, n, 1.0, A.data(), m, R.data(), n, 0.0, QR.data(), m);
        if (maxDiff(QR, A_orig) > 1.0e-12) {
            std::cout << "FAIL: QR reconstruction error " << maxDiff(QR, A_orig) << std::endl;
            ++num_failures;
        }
    }

    // --- SVD: orthonormal factors, exact reconstruction, known rank
    {
        const int m = 30, n = 10;
        const std::vector<double> A = randomMatrix(m, n, 2);

        std::vector<double> U, S, VT;
        dense::svd(m, n, A, U, S, VT);

        // singular values sorted and nonnegative
        for (int i=1; i<n; ++i) {
            if (S[i] > S[i-1] || S[i] < 0.0) {
                std::cout << "FAIL: singular values not sorted nonnegative" << std::endl;
                ++num_failures;
                break;
            }
        }

        // U*diag(S)*VT = A
        std::vector<double> SVT(static_cast<size_t>(n)*n, 0.0);
        for (int j=0; j<n; ++j) {
            for (int i=0; i<n; ++i) {
                SVT[j*n + i] = S[i]*VT[j*n + i];
            }
        }
        std::vector<double> USVT(static_cast<size_t>(m)*n, 0.0);
        dense::gemm('N', 'N', m, n, n, 1.0, U.data(), m, SVT.data(), n, 0.0, USVT.data(), m);
        if (maxDiff(USVT, A) > 1.0e-12) {
            std::cout << "FAIL: SVD reconstruction error " << maxDiff(USVT, A) << std::endl;
            ++num_failures;
        }

        // a rank-3 matrix must come back with exactly 3 significant values
        const int r = 3;
        std::vector<double> L = randomMatrix(m, r, 3);
        std::vector<double> Rt = randomMatrix(n, r, 4);
        std::vector<double> low(static_cast<size_t>(m)*n, 0.0);
        dense::gemm('N', 'T', m, n, r, 1.0, L.data(), m, Rt.data(), n, 0.0, low.data(), m);

        std::vector<double> U3, S3, VT3;
        dense::svd(m, n, low, U3, S3, VT3);
        if (S3[r]/S3[0] > 1.0e-13) {
            std::cout << "FAIL: rank-3 matrix has significant 4th singular value "
                    << S3[r]/S3[0] << std::endl;
            ++num_failures;
        }
    }

    // --- triangular solve against a known solution
    {
        const int n = 8, nrhs = 3;
        std::vector<double> A = randomMatrix(20, n, 5);
        std::vector<double> R;
        dense::qr(20, n, A, R);          // random R, well conditioned

        const std::vector<double> X0 = randomMatrix(n, nrhs, 6);
        std::vector<double> B(static_cast<size_t>(n)*nrhs, 0.0);
        dense::gemm('N', 'N', n, nrhs, n, 1.0, R.data(), n, X0.data(), n, 0.0, B.data(), n);

        dense::solveUpper(n, nrhs, R.data(), B.data());   // B <- R^{-1} B = X0
        if (maxDiff(B, X0) > 1.0e-11) {
            std::cout << "FAIL: triangular solve error " << maxDiff(B, X0) << std::endl;
            ++num_failures;
        }
    }

    // --- least squares recovers the generator of a consistent system
    {
        const int m = 20, n = 5;
        const std::vector<double> A = randomMatrix(m, n, 7);
        const std::vector<double> x0 = randomMatrix(n, 1, 8);

        std::vector<double> b(m, 0.0);
        dense::gemv('N', m, n, 1.0, A.data(), m, x0.data(), 0.0, b.data());

        dense::leastSquares(m, n, 1, A, b);   // first n rows of b = solution
        double err = 0.0;
        for (int i=0; i<n; ++i) {
            err = std::max(err, std::abs(b[i] - x0[i]));
        }
        if (err > 1.0e-11) {
            std::cout << "FAIL: least squares error " << err << std::endl;
            ++num_failures;
        }
    }

    // --- BLAS-1 wrappers
    {
        const std::vector<double> x = {3.0, 4.0};
        if (std::abs(dense::norm2(2, x.data()) - 5.0) > tol) {
            std::cout << "FAIL: norm2" << std::endl;
            ++num_failures;
        }
        std::vector<double> y = {1.0, 1.0};
        dense::axpy(2, 2.0, x.data(), y.data());
        if (std::abs(y[0] - 7.0) > tol || std::abs(y[1] - 9.0) > tol) {
            std::cout << "FAIL: axpy" << std::endl;
            ++num_failures;
        }
        if (std::abs(dense::dot(2, x.data(), y.data()) - 57.0) > tol) {
            std::cout << "FAIL: dot" << std::endl;
            ++num_failures;
        }
    }

    if (num_failures == 0) {
        std::cout << "All dense kernel tests PASSED" << std::endl;
    }

    return num_failures == 0 ? 0 : 1;
}
