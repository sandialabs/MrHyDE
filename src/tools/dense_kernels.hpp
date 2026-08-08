#ifndef MRHYDE_DENSE_KERNELS_H
#define MRHYDE_DENSE_KERNELS_H

#include <stdexcept>
#include <string>
#include <vector>

// Fortran name mangling: trailing underscore by default, overridable to match
// the host toolchain (compare F77_SIERRA_LAPACK in Sierra's sierra_blas_lapack.h).
#ifndef MRHYDE_FORTRAN_NAME
#define MRHYDE_FORTRAN_NAME(name) name##_
#endif

// Every routine declared here also appears in Sierra's sierra_blas_lapack.h,
// so this layer adds no dependency in either host.  Note dgesvd, not dgesdd:
// Sierra does not declare the divide-and-conquer variant.
extern "C" {
  void MRHYDE_FORTRAN_NAME(dgemm)(const char* transa, const char* transb,
    const int* m, const int* n, const int* k, const double* alpha,
    const double* a, const int* lda, const double* b, const int* ldb,
    const double* beta, double* c, const int* ldc);
  void MRHYDE_FORTRAN_NAME(dgemv)(const char* trans, const int* m, const int* n,
    const double* alpha, const double* a, const int* lda, const double* x,
    const int* incx, const double* beta, double* y, const int* incy);
  void MRHYDE_FORTRAN_NAME(dtrsm)(const char* side, const char* uplo,
    const char* transa, const char* diag, const int* m, const int* n,
    const double* alpha, const double* a, const int* lda, double* b, const int* ldb);
  double MRHYDE_FORTRAN_NAME(dnrm2)(const int* n, const double* x, const int* incx);
  double MRHYDE_FORTRAN_NAME(ddot)(const int* n, const double* x, const int* incx,
    const double* y, const int* incy);
  void MRHYDE_FORTRAN_NAME(daxpy)(const int* n, const double* alpha,
    const double* x, const int* incx, double* y, const int* incy);
  void MRHYDE_FORTRAN_NAME(dscal)(const int* n, const double* alpha,
    double* x, const int* incx);
  void MRHYDE_FORTRAN_NAME(dgeqrf)(const int* m, const int* n, double* a,
    const int* lda, double* tau, double* work, const int* lwork, int* info);
  void MRHYDE_FORTRAN_NAME(dorgqr)(const int* m, const int* n, const int* k,
    double* a, const int* lda, const double* tau, double* work,
    const int* lwork, int* info);
  void MRHYDE_FORTRAN_NAME(dgesvd)(const char* jobu, const char* jobvt,
    const int* m, const int* n, double* a, const int* lda, double* s,
    double* u, const int* ldu, double* vt, const int* ldvt, double* work,
    const int* lwork, int* info);
  void MRHYDE_FORTRAN_NAME(dgels)(const char* trans, const int* m, const int* n,
    const int* nrhs, double* a, const int* lda, double* b, const int* ldb,
    double* work, const int* lwork, int* info);
}

namespace MrHyDE {
namespace dense {

  // Dense kernels for the sketching windows.  Everything is column-major
  // (Fortran/MATLAB layout), matching the math being ported: an M x n matrix
  // is a std::vector<double> of length M*n with column j at [j*M, (j+1)*M).

  inline void check(const int & info, const std::string & routine) {
    if (info != 0) {
      throw std::runtime_error("dense::" + routine + " failed (info = "
                               + std::to_string(info) + ").");
    }
  }

  /// C = alpha*op(A)*op(B) + beta*C
  inline void gemm(const char & transa, const char & transb, const int & m,
                   const int & n, const int & k, const double & alpha,
                   const double* A, const int & lda, const double* B,
                   const int & ldb, const double & beta, double* C, const int & ldc) {
    MRHYDE_FORTRAN_NAME(dgemm)(&transa, &transb, &m, &n, &k, &alpha, A, &lda,
                               B, &ldb, &beta, C, &ldc);
  }

  /// y = alpha*op(A)*x + beta*y
  inline void gemv(const char & trans, const int & m, const int & n,
                   const double & alpha, const double* A, const int & lda,
                   const double* x, const double & beta, double* y) {
    const int one = 1;
    MRHYDE_FORTRAN_NAME(dgemv)(&trans, &m, &n, &alpha, A, &lda, x, &one, &beta, y, &one);
  }

  inline double norm2(const int & n, const double* x) {
    const int one = 1;
    return MRHYDE_FORTRAN_NAME(dnrm2)(&n, x, &one);
  }

  inline double dot(const int & n, const double* x, const double* y) {
    const int one = 1;
    return MRHYDE_FORTRAN_NAME(ddot)(&n, x, &one, y, &one);
  }

  inline void axpy(const int & n, const double & alpha, const double* x, double* y) {
    const int one = 1;
    MRHYDE_FORTRAN_NAME(daxpy)(&n, &alpha, x, &one, y, &one);
  }

  inline void scal(const int & n, const double & alpha, double* x) {
    const int one = 1;
    MRHYDE_FORTRAN_NAME(dscal)(&n, &alpha, x, &one);
  }

  /**
   * @brief Economy QR of an m x n matrix, m >= n.  On exit A holds the
   *        orthonormal factor Q (m x n) and R is n x n upper triangular.
   */
  inline void qr(const int & m, const int & n, std::vector<double> & A,
                 std::vector<double> & R) {

    if (m < n) {
      throw std::invalid_argument("dense::qr expects m >= n.");
    }

    std::vector<double> tau(n);
    int info = 0;

    // workspace query, then factor
    double wsize = 0.0;
    int lwork = -1;
    MRHYDE_FORTRAN_NAME(dgeqrf)(&m, &n, A.data(), &m, tau.data(), &wsize, &lwork, &info);
    check(info, "qr(dgeqrf query)");
    lwork = static_cast<int>(wsize);
    std::vector<double> work(lwork);
    MRHYDE_FORTRAN_NAME(dgeqrf)(&m, &n, A.data(), &m, tau.data(), work.data(), &lwork, &info);
    check(info, "qr(dgeqrf)");

    // copy R out before the reflectors are overwritten with Q
    R.assign(static_cast<size_t>(n)*n, 0.0);
    for (int j=0; j<n; ++j) {
      for (int i=0; i<=j; ++i) {
        R[j*n + i] = A[j*m + i];
      }
    }

    lwork = -1;
    MRHYDE_FORTRAN_NAME(dorgqr)(&m, &n, &n, A.data(), &m, tau.data(), &wsize, &lwork, &info);
    check(info, "qr(dorgqr query)");
    lwork = static_cast<int>(wsize);
    work.resize(lwork);
    MRHYDE_FORTRAN_NAME(dorgqr)(&m, &n, &n, A.data(), &m, tau.data(), work.data(), &lwork, &info);
    check(info, "qr(dorgqr)");
  }

  /**
   * @brief Economy SVD, A = U*diag(S)*VT with q = min(m,n) columns.  A is
   *        taken by value because dgesvd destroys its input.
   */
  inline void svd(const int & m, const int & n, std::vector<double> A,
                  std::vector<double> & U, std::vector<double> & S,
                  std::vector<double> & VT) {

    const int q = (m < n) ? m : n;
    U.assign(static_cast<size_t>(m)*q, 0.0);
    S.assign(q, 0.0);
    VT.assign(static_cast<size_t>(q)*n, 0.0);

    const char job = 'S';
    int info = 0;
    double wsize = 0.0;
    int lwork = -1;
    MRHYDE_FORTRAN_NAME(dgesvd)(&job, &job, &m, &n, A.data(), &m, S.data(),
                                U.data(), &m, VT.data(), &q, &wsize, &lwork, &info);
    check(info, "svd(dgesvd query)");
    lwork = static_cast<int>(wsize);
    std::vector<double> work(lwork);
    MRHYDE_FORTRAN_NAME(dgesvd)(&job, &job, &m, &n, A.data(), &m, S.data(),
                                U.data(), &m, VT.data(), &q, work.data(), &lwork, &info);
    check(info, "svd(dgesvd)");
  }

  /**
   * @brief Solve R*X = B in place, R upper triangular n x n, B n x nrhs.
   */
  inline void solveUpper(const int & n, const int & nrhs, const double* R,
                         double* B) {
    const char side = 'L', uplo = 'U', trans = 'N', diag = 'N';
    const double one = 1.0;
    MRHYDE_FORTRAN_NAME(dtrsm)(&side, &uplo, &trans, &diag, &n, &nrhs, &one, R, &n, B, &n);
  }

  /**
   * @brief Least squares min ||A*X - B||.  A is m x n with m >= n, destroyed;
   *        B is m x nrhs in, and its first n rows hold the solution on exit.
   */
  inline void leastSquares(const int & m, const int & n, const int & nrhs,
                           std::vector<double> A, std::vector<double> & B) {

    if (m < n) {
      throw std::invalid_argument("dense::leastSquares expects m >= n.");
    }

    const char trans = 'N';
    int info = 0;
    double wsize = 0.0;
    int lwork = -1;
    MRHYDE_FORTRAN_NAME(dgels)(&trans, &m, &n, &nrhs, A.data(), &m, B.data(), &m,
                               &wsize, &lwork, &info);
    check(info, "leastSquares(dgels query)");
    lwork = static_cast<int>(wsize);
    std::vector<double> work(lwork);
    MRHYDE_FORTRAN_NAME(dgels)(&trans, &m, &n, &nrhs, A.data(), &m, B.data(), &m,
                               work.data(), &lwork, &info);
    check(info, "leastSquares(dgels)");
  }

}
}

#endif
