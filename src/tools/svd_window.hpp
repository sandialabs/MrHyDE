#ifndef MRHYDE_SVD_WINDOW_H
#define MRHYDE_SVD_WINDOW_H

#include "dense_kernels.hpp"
#include "sketch_window.hpp"   // WindowStatus, WindowPolicy

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace MrHyDE {

  /**
   * @class SvdWindow
   * @brief Deterministic incremental-SVD window (port of SvdWindow2.m); the
   *        alternative compressor, selected by comp='rm'.
   *
   * Same interface, payload shape, reconstruct formula and status vocabulary
   * as SketchWindow, but fully deterministic: the seed is stored and unused,
   * and there is no retry logic.  'rank' is a CAP: the working rank starts at
   * zero and grows in place one direction at a time, up to the cap.
   *
   * The growth tolerance is the design pivot: 1e-12 in static mode (grow
   * for anything above machine noise), policy.tol in rolling mode (grow only
   * for directions that matter at the requested accuracy).  In rolling mode
   * the overflow pre-check uses the same threshold as the in-span test, so a
   * rolling window never truncates -- no drift, no snapshot needed.  At the
   * cap, an unrepresentable column cuts the window at the previous step, whose
   * factors are exact.
   *
   * Factor invariant: u_j ~= Qf * (Sv .* Vf(j,:)') for every applied column.
   * Committed payload: B1 = Qf*diag(Sv) carries the TRUE rank, not a padded
   * sketch width.
   */
  class SvdWindow {

  public:

    SvdWindow(const int & a, const int & b, const int & rank,
              const int & window_id, const uint64_t & master_seed)
      : a_(a), b_(b), m_(b - a), rank_(rank), window_id_(window_id),
        master_seed_(master_seed) {
      if (m_ < 2) {
        throw std::invalid_argument("SvdWindow: window must cover >= 2 steps.");
      }
      if (rank_ < 1) {
        throw std::invalid_argument("SvdWindow: rank must be at least 1.");
      }
      (void)master_seed_;   // interface parity with SketchWindow; deterministic here
    }

    /// Class defaults: a larger bump and cap than the sketch, same tolerances.
    static WindowPolicy fillPolicy() {
      WindowPolicy p;
      p.rank_bump = 4;
      p.rank_max = 30;
      return p;
    }

    /// Smallest prefix worth committing; the factors are exact, so two
    /// columns already pay for themselves.
    static int minCommitForRank(const int & rank, const int & min_commit) {
      (void)rank;
      return std::max(min_commit, 2);
    }

    /// Streaming-transient cost model for the greedy planner.
    static void rollingStreamCoeffs(const int & M, const int & q, const int & rho,
                                    double & A, double & C) {
      A = 2.0*q/M;
      C = rho + 2.0 + 2.0*q + 2.0*q/M;
    }

    /// Static mode: record everything, commit via finalize(tol).
    void beginRecord(const int & Mdim) {
      adaptive_ = false;
      policy_ = WindowPolicy();
      startRecording(Mdim);
    }

    /**
     * @brief Adaptive mode.  Only rolling is supported: a non-rolling adaptive
     *        SVD window has no monitor to commit it, so reject the
     *        configuration outright.
     */
    void beginRecord(const int & Mdim, const WindowPolicy & policy) {
      if (!policy.rolling) {
        throw std::invalid_argument(
          "SvdWindow: adaptive mode requires policy.rolling; use static mode instead.");
      }
      adaptive_ = true;
      policy_ = policy;
      startRecording(Mdim);
    }

    bool isOpen() const { return open_; }

    /**
     * @brief Add the state at real step k.  In rolling mode, a column that a
     *        capped basis cannot represent cuts the window at the previous
     *        step BEFORE being applied -- the factors then serve columns
     *        1..lastGood exactly.
     */
    void record(const double* u, const int & k) {

      if (!open_) {
        throw std::runtime_error("SvdWindow: record on a closed window.");
      }
      const int j = k - a_;
      if (j < 1 || j > m_) {
        throw std::runtime_error("SvdWindow: step outside the window.");
      }
      if (j != j_cur_ + 1) {
        throw std::runtime_error("SvdWindow: columns must arrive in order.");
      }

      const double g_tol = policy_.rolling ? policy_.tol : 1.0e-12;

      // overflow pre-check (rolling, basis at the cap): would this column
      // need a new direction?  If so, cut now -- the column is never applied.
      if (policy_.rolling && q_ >= rank_ && q_ > 0) {
        std::vector<double> proj(q_), res(M_);
        dense::gemv('T', M_, q_, 1.0, Qf_.data(), M_, u, 0.0, proj.data());
        std::copy(u, u + M_, res.begin());
        dense::gemv('N', M_, q_, -1.0, Qf_.data(), M_, proj.data(), 1.0, res.data());
        const double rn = dense::norm2(M_, res.data());
        if (rn > g_tol*Sv_[0]) {
          overflowCut();
          return;
        }
      }

      j_cur_ = j;
      isvdStep(u, rank_, g_tol);

      if (policy_.rolling) {
        std::copy(u, u + M_, buf_roll_.begin() + static_cast<size_t>((j-1)%policy_.rho)*M_);
        last_good_ = j;                        // every applied column is exact
        cand_anchor_.assign(u, u + M_);
        if (j%policy_.rho == 0 || j == m_) {
          monitorRolling(j);
        }
      }
      else {
        seen_[j-1] = true;
        std::copy(u, u + M_, buf_.begin() + static_cast<size_t>(j-1)*M_);
      }
    }

    /// Static-mode commit; phase-1 semantics on failure (payload kept, flagged).
    void finalize(const double & tol) {
      if (adaptive_) {
        throw std::runtime_error("SvdWindow: finalize is for static mode only.");
      }
      for (int j=0; j<m_; ++j) {
        if (!seen_[j]) {
          throw std::runtime_error("SvdWindow: finalize with missing columns.");
        }
      }
      double emax = 0.0, fro = 0.0;
      columnErrors(1, m_, false, emax, fro);
      err_max_col_ = err_run_ = emax;
      err_fro_ = fro;
      commit(m_, (emax <= tol) ? WindowStatus::Accepted : WindowStatus::Fallback,
             buf_.data() + static_cast<size_t>(m_-1)*M_);
    }

    /// Reconstruct the state at real step k into out (M doubles).
    void reconstruct(const int & k, double* out) const {
      requirePayload();
      const int j = k - a_;
      if (j < 1 || j > b_eff_ - a_) {
        throw std::runtime_error("SvdWindow: reconstruct outside the committed range.");
      }
      const int q = payload_cols_;
      std::vector<double> row(q);
      for (int c=0; c<q; ++c) {
        row[c] = Q2_[static_cast<size_t>(c)*(b_eff_ - a_) + (j-1)];
      }
      dense::gemv('N', M_, q, 1.0, B1_.data(), M_, row.data(), 0.0, out);
    }

    void rightEdgeRaw(double* out) const {
      if (has_anchor_) {
        std::copy(anchor_.begin(), anchor_.end(), out);
      }
      else {
        reconstruct(b_eff_, out);
      }
    }

    void anchorRaw(double* out) const {
      if (!has_anchor_) {
        throw std::runtime_error("SvdWindow: no anchor held.");
      }
      std::copy(anchor_.begin(), anchor_.end(), out);
    }

    bool hasAnchor() const { return has_anchor_; }

    void releaseAnchor() {
      anchor_.clear();
      has_anchor_ = false;
    }

    void discardPayload() {
      B1_.clear();
      Q2_.clear();
      anchor_.clear();
      has_anchor_ = false;
      payload_cols_ = 0;
      b_eff_ = a_;
      status_ = WindowStatus::Fallback;
    }

    double storageSE() const {
      if (M_ == 0) { return 0.0; }
      return static_cast<double>(B1_.size() + Q2_.size() + anchor_.size())/M_;
    }

    double payloadSE() const {
      if (M_ == 0) { return 0.0; }
      return static_cast<double>(B1_.size() + Q2_.size())/M_;
    }

    double peakSE() const { return peak_se_; }

    /// No retry logic here; kept so the manager treats both compressors alike.
    void bumpRetry() { ++retry_attempt_; }

    int getA()            const { return a_; }
    int getB()            const { return b_; }
    int getM()            const { return m_; }
    int getRank()         const { return rank_; }
    int getWindowId()     const { return window_id_; }
    int getRetryAttempt() const { return retry_attempt_; }
    int getBEff()         const { return b_eff_; }
    int getLastGood()     const { return last_good_; }
    int getNextRankHint() const { return next_rank_hint_; }
    int getPayloadColumns() const { return payload_cols_; }
    WindowStatus getStatus() const { return status_; }
    double getErrMaxCol() const { return err_max_col_; }
    double getErrFro()    const { return err_fro_; }

  private:

    void startRecording(const int & Mdim) {
      M_ = Mdim;
      Qf_.clear();
      Sv_.clear();
      Vf_.clear();
      q_ = 0;
      rows_v_ = 0;

      if (policy_.rolling) {
        buf_roll_.assign(static_cast<size_t>(M_)*policy_.rho, 0.0);
        buf_.clear();
        seen_.clear();
      }
      else {
        buf_.assign(static_cast<size_t>(M_)*m_, 0.0);
        seen_.assign(m_, false);
        buf_roll_.clear();
      }

      cand_anchor_.clear();
      j_cur_ = 0;
      err_run_ = 0.0;
      B1_.clear();
      Q2_.clear();
      anchor_.clear();
      has_anchor_ = false;
      payload_cols_ = 0;
      b_eff_ = a_;
      err_max_col_ = err_fro_ = std::numeric_limits<double>::infinity();
      status_ = WindowStatus::Planned;
      last_good_ = 0;
      next_rank_hint_ = 0;
      open_ = true;
      peak_se_ = transientSE(rank_);
    }

    double transientSE(const int & q) const {
      if (policy_.rolling) {
        double A = 0.0, C = 0.0;
        rollingStreamCoeffs(M_, q, policy_.rho, A, C);
        return A*m_ + C;
      }
      const double doubles = static_cast<double>(M_)*m_ + static_cast<double>(M_)*q
                           + static_cast<double>(m_)*q + q
                           + static_cast<double>(M_)*q + M_;
      return doubles/M_;
    }

    // One Brand-style incremental SVD step: grow by the residual direction, or
    // append a guard row when the column is already in span to g_tol.
    void isvdStep(const double* u, const int & cap, const double & g_tol) {

      if (q_ == 0) {
        const double nrm = dense::norm2(M_, u);
        Qf_.assign(u, u + M_);
        dense::scal(M_, 1.0/std::max(nrm, DBL_MIN), Qf_.data());
        Sv_.assign(1, nrm);
        Vf_.assign(1, 1.0);
        q_ = 1;
        rows_v_ = 1;
        return;
      }

      std::vector<double> proj(q_), res(M_);
      dense::gemv('T', M_, q_, 1.0, Qf_.data(), M_, u, 0.0, proj.data());
      std::copy(u, u + M_, res.begin());
      dense::gemv('N', M_, q_, -1.0, Qf_.data(), M_, proj.data(), 1.0, res.data());
      const double rn = dense::norm2(M_, res.data());

      if (rn <= g_tol*std::max(Sv_[0], DBL_MIN)) {
        // guard row: represent the column in the existing basis.  Vf is
        // row-major (one row per applied column), so the append is contiguous.
        Vf_.resize(static_cast<size_t>(q_)*(rows_v_ + 1));
        for (int c=0; c<q_; ++c) {
          Vf_[static_cast<size_t>(rows_v_)*q_ + c] = proj[c]/std::max(Sv_[c], DBL_MIN);
        }
        ++rows_v_;
        return;
      }

      // growth: extend the basis by the residual direction
      const int q1 = q_ + 1;
      std::vector<double> K(static_cast<size_t>(q1)*q1, 0.0);
      for (int c=0; c<q_; ++c) {
        K[static_cast<size_t>(c)*q1 + c] = Sv_[c];
      }
      for (int r=0; r<q_; ++r) {
        K[static_cast<size_t>(q_)*q1 + r] = proj[r];
      }
      K[static_cast<size_t>(q_)*q1 + q_] = rn;

      std::vector<double> Uk, Sk, VkT;
      dense::svd(q1, q1, K, Uk, Sk, VkT);

      const int q_new = std::min(q1, cap);

      // Q <- [Q, dir] * Uk(:, 1:q_new)
      std::vector<double> Qa(static_cast<size_t>(M_)*q1);
      std::copy(Qf_.begin(), Qf_.end(), Qa.begin());
      for (int i=0; i<M_; ++i) {
        Qa[static_cast<size_t>(q_)*M_ + i] = res[i]/rn;
      }
      Qf_.assign(static_cast<size_t>(M_)*q_new, 0.0);
      dense::gemm('N', 'N', M_, q_new, q1, 1.0, Qa.data(), M_, Uk.data(), q1,
                  0.0, Qf_.data(), M_);

      // V <- [V, 0; 0, 1] * Vk(:, 1:q_new), rows stay row-major
      const int rows1 = rows_v_ + 1;
      std::vector<double> Vnew(static_cast<size_t>(rows1)*q_new, 0.0);
      for (int r=0; r<rows1; ++r) {
        for (int c=0; c<q_new; ++c) {
          double acc = 0.0;
          for (int t=0; t<q1; ++t) {
            // old Va(r, t): rows 0..rows_v_-1 hold Vf, the last row is e_{q1}
            double va;
            if (r < rows_v_) {
              va = (t < q_) ? Vf_[static_cast<size_t>(r)*q_ + t] : 0.0;
            }
            else {
              va = (t == q_) ? 1.0 : 0.0;
            }
            // Vk(t, c) = VkT(c, t); VkT is q1 x q1 column-major
            acc += va*VkT[static_cast<size_t>(t)*q1 + c];
          }
          Vnew[static_cast<size_t>(r)*q_new + c] = acc;
        }
      }
      Vf_.swap(Vnew);
      rows_v_ = rows1;
      Sv_.assign(Sk.begin(), Sk.begin() + q_new);
      q_ = q_new;

      // periodic re-orthonormalization keeps drift out of long windows
      if (rows_v_%32 == 0) {
        std::vector<double> Qo = Qf_;
        std::vector<double> R;
        dense::qr(M_, q_, Qo, R);
        std::vector<double> RS(static_cast<size_t>(q_)*q_, 0.0);
        for (int c=0; c<q_; ++c) {
          for (int r=0; r<q_; ++r) {
            RS[static_cast<size_t>(c)*q_ + r] = R[static_cast<size_t>(c)*q_ + r]*Sv_[c];
          }
        }
        std::vector<double> ur, sr, vrT;
        dense::svd(q_, q_, RS, ur, sr, vrT);
        Qf_.assign(static_cast<size_t>(M_)*q_, 0.0);
        dense::gemm('N', 'N', M_, q_, q_, 1.0, Qo.data(), M_, ur.data(), q_,
                    0.0, Qf_.data(), M_);
        Sv_ = sr;
        std::vector<double> Vr(static_cast<size_t>(rows_v_)*q_, 0.0);
        for (int r=0; r<rows_v_; ++r) {
          for (int c=0; c<q_; ++c) {
            double acc = 0.0;
            for (int t=0; t<q_; ++t) {
              acc += Vf_[static_cast<size_t>(r)*q_ + t]*vrT[static_cast<size_t>(t)*q_ + c];
            }
            Vr[static_cast<size_t>(r)*q_ + c] = acc;
          }
        }
        Vf_.swap(Vr);
      }
    }

    // The rolling monitor never cuts on error: error control lives entirely in
    // the growth threshold.  It accumulates the diagnostic and the rank hint.
    void monitorRolling(const int & j) {
      double emax = 0.0, fro = 0.0;
      const int first = std::max(1, j - policy_.rho + 1);
      columnErrors(first, j, true, emax, fro);
      err_run_ = std::max(err_run_, emax);
      setHint();
      if (j == m_) {
        err_max_col_ = err_fro_ = err_run_;
        commit(m_, WindowStatus::Accepted, cand_anchor_.data());
      }
    }

    // A capped basis met a column it cannot represent.  The factors are exact
    // for every applied column, so the cut needs no snapshot.
    void overflowCut() {
      setHint();
      if (last_good_ >= std::max(policy_.min_commit, 2) && q_ > 0) {
        err_max_col_ = err_fro_ = err_run_;
        commit(last_good_, WindowStatus::Prefix, cand_anchor_.data());
      }
      else {
        fallback();
      }
    }

    /// Exact rank from the singular values, plus one direction of headroom.
    void setHint() {
      if (Sv_.empty()) { return; }
      int significant = 0;
      for (size_t i=0; i<Sv_.size(); ++i) {
        if (Sv_[i] > policy_.tol*Sv_[0]) { ++significant; }
      }
      next_rank_hint_ = std::max(next_rank_hint_, significant + 1);
    }

    /// Per-column errors against the raw buffer (circular in rolling mode).
    void columnErrors(const int & first, const int & last, const bool & circular,
                      double & emax, double & fro) const {
      emax = 0.0;
      double num = 0.0, den = 0.0;
      std::vector<double> uhat(M_), d(M_);
      for (int j=first; j<=last; ++j) {
        const double* truth = circular
          ? buf_roll_.data() + static_cast<size_t>((j-1)%policy_.rho)*M_
          : buf_.data() + static_cast<size_t>(j-1)*M_;
        factorColumn(j, uhat.data());
        for (int i=0; i<M_; ++i) { d[i] = uhat[i] - truth[i]; }
        const double nd = dense::norm2(M_, d.data());
        const double nt = dense::norm2(M_, truth);
        emax = std::max(emax, nd/std::max(nt, DBL_EPSILON));
        num += nd*nd;
        den += nt*nt;
      }
      fro = std::sqrt(num/std::max(den, DBL_EPSILON));
    }

    /// u_j from the working factors: Qf * (Sv .* Vf(j,:)').
    void factorColumn(const int & j, double* out) const {
      std::vector<double> w(q_);
      for (int c=0; c<q_; ++c) {
        w[c] = Sv_[c]*Vf_[static_cast<size_t>(j-1)*q_ + c];
      }
      dense::gemv('N', M_, q_, 1.0, Qf_.data(), M_, w.data(), 0.0, out);
    }

    // Payload from the working factors: B1 = Qf*diag(Sv) carries the true rank.
    void commit(const int & j, const WindowStatus & status, const double* anchor_vec) {

      B1_.assign(static_cast<size_t>(M_)*q_, 0.0);
      for (int c=0; c<q_; ++c) {
        for (int i=0; i<M_; ++i) {
          B1_[static_cast<size_t>(c)*M_ + i] = Qf_[static_cast<size_t>(c)*M_ + i]*Sv_[c];
        }
      }
      Q2_.assign(static_cast<size_t>(j)*q_, 0.0);
      for (int r=0; r<j; ++r) {
        for (int c=0; c<q_; ++c) {
          Q2_[static_cast<size_t>(c)*j + r] = Vf_[static_cast<size_t>(r)*q_ + c];
        }
      }
      payload_cols_ = q_;
      anchor_.assign(anchor_vec, anchor_vec + M_);
      has_anchor_ = true;
      b_eff_ = a_ + j;
      status_ = status;
      freeTransients();
    }

    void fallback() {
      status_ = WindowStatus::Fallback;
      b_eff_ = a_;
      B1_.clear();
      Q2_.clear();
      anchor_.clear();
      has_anchor_ = false;
      payload_cols_ = 0;
      freeTransients();
    }

    void freeTransients() {
      buf_.clear();
      seen_.clear();
      buf_roll_.clear();
      cand_anchor_.clear();
      Qf_.clear();
      Sv_.clear();
      Vf_.clear();
      q_ = 0;
      rows_v_ = 0;
      open_ = false;
    }

    void requirePayload() const {
      if (B1_.empty()) {
        throw std::runtime_error("SvdWindow: no committed payload.");
      }
    }

    // ---- configuration ----
    int a_, b_, m_;
    int rank_;                       ///< the cap on the working rank
    int window_id_;
    uint64_t master_seed_;           ///< unused (deterministic)
    int retry_attempt_ = 0;
    int M_ = 0;
    bool adaptive_ = false;
    WindowPolicy policy_;

    // ---- results ----
    WindowStatus status_ = WindowStatus::Planned;
    double err_max_col_ = std::numeric_limits<double>::infinity();
    double err_fro_ = std::numeric_limits<double>::infinity();
    int b_eff_ = 0;
    int last_good_ = 0;
    int next_rank_hint_ = 0;
    double peak_se_ = 0.0;

    // ---- committed payload ----
    std::vector<double> B1_;         ///< M x q
    std::vector<double> Q2_;         ///< (b_eff - a) x q
    int payload_cols_ = 0;
    std::vector<double> anchor_;
    bool has_anchor_ = false;

    // ---- working factors (live only while recording) ----
    std::vector<double> Qf_;         ///< M x q orthonormal, column-major
    std::vector<double> Sv_;         ///< q singular values, descending
    std::vector<double> Vf_;         ///< rows_v x q, ROW-major (written/read by row)
    int q_ = 0;
    int rows_v_ = 0;
    std::vector<double> buf_;
    std::vector<bool> seen_;
    std::vector<double> buf_roll_;
    std::vector<double> cand_anchor_;
    int j_cur_ = 0;
    double err_run_ = 0.0;
    bool open_ = false;
  };

}

#endif
