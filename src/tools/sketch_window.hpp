#ifndef MRHYDE_SKETCH_WINDOW_H
#define MRHYDE_SKETCH_WINDOW_H

#include "dense_kernels.hpp"
#include "threefry.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace MrHyDE {

  /// Terminal states of a window payload.
  enum class WindowStatus {
    Planned,    ///< recording, not yet committed
    Accepted,   ///< full window committed
    Prefix,     ///< cut: only columns up to b_eff are served
    Fallback    ///< nothing served; the classic revolve axis covers these steps
  };

  /**
   * @brief Adaptive-window policy.  Defaults are per-class
   *        (SketchWindow::fillPolicy / SvdWindow::fillPolicy).
   */
  struct WindowPolicy {
    double tol = 1.0e-8;    ///< per-column relative reconstruction tolerance
    int rho = 5;            ///< monitor cadence and rolling-buffer depth
    int rank_bump = 2;      ///< additive rank raise per retry
    int rank_max = 15;      ///< hard rank cap
    int max_retries = 2;    ///< full-mode retry budget
    int min_commit = 4;     ///< smallest prefix worth committing
    bool rolling = false;   ///< keep only the last rho states (no retries)
  };

  /**
   * @class SketchWindow
   * @brief Streaming two-sided randomized sketch of one window of consecutive
   *        time steps (port of SketchWindow2.m).
   *
   * The window covers real steps a+1..b and owns the column block
   * U = [u_{a+1} ... u_b], M x m with m = b - a.  Columns arrive strictly in
   * order through record(); the sketch accumulates rank-1 updates and never
   * stores its test matrices -- they are regenerated bit-identically from
   * Threefry seeds, which is what makes the caches evictable and retries
   * deterministic.  Incremental projections G1 = Phi1'*V1 and G2 = Phi2'*V2
   * are maintained alongside, so trial factorizations and commits never need
   * a test matrix either.
   *
   * Modes: static (no policy; commit via finalize), full adaptive (buffers all
   * columns, monitors at cadence, retries at higher rank by re-streaming the
   * buffer), rolling (keeps only the last rho columns; a failure cuts to the
   * last verified prefix or falls back -- no retries, the states are gone).
   *
   * Committed payload: B1 (M x q) and Q2 (m_eff x q) with
   * reconstruct(k) = B1 * Q2(k-a, :)', plus the exact right-edge anchor.
   * Failure is visible, never silent: a window that cannot meet its tolerance
   * ends Prefix or Fallback and the objective reroutes those steps through
   * classic checkpointing.
   */
  class SketchWindow {

  public:

    SketchWindow(const int & a, const int & b, const int & rank,
                 const int & window_id, const uint64_t & master_seed)
      : a_(a), b_(b), m_(b - a), rank_(rank), window_id_(window_id),
        master_seed_(master_seed) {
      if (m_ < 2) {
        throw std::invalid_argument("SketchWindow: window must cover >= 2 steps.");
      }
      if (rank_ < 1) {
        throw std::invalid_argument("SketchWindow: rank must be at least 1.");
      }
    }

    /// Class defaults; any adaptive window starts from these.
    static WindowPolicy fillPolicy() {
      return WindowPolicy();
    }

    /**
     * @brief Streaming-transient cost model for the greedy planner: holding a
     *        rolling window of length m costs about A*m + C state-equivalents.
     */
    static void rollingStreamCoeffs(const int & M, const int & r, const int & rho,
                                    double & A, double & C) {
      const int k1 = (2*r + 1 < M) ? 2*r + 1 : M;
      const int k2 = 2*r + 1;
      const int s1 = 2*k1 + 1;
      const int s2 = 2*k2 + 1;
      A = 2.0*k2/M;
      C = rho + 2.0 + 4.0*k1
        + (2.0*s1*s2 + 2.0*s1*k1 + 2.0*s2*k2 + 1.0*k1*k2)/M;
    }

    /// Static mode: record everything, commit via finalize(tol).
    void beginRecord(const int & Mdim) {
      adaptive_ = false;
      policy_ = WindowPolicy();
      startRecording(Mdim);
    }

    /// Adaptive mode: monitor at cadence, self-commit or self-cut.
    void beginRecord(const int & Mdim, const WindowPolicy & policy) {
      adaptive_ = true;
      policy_ = policy;
      startRecording(Mdim);
    }

    bool isOpen() const { return open_; }

    /**
     * @brief Add the state at real step k (an M-vector).  Columns must arrive
     *        strictly in order a+1, a+2, ...  May self-close the window in
     *        adaptive modes (commit, cut, or fallback).
     */
    void record(const double* u, const int & k) {

      if (!open_) {
        throw std::runtime_error("SketchWindow: record on a closed window.");
      }
      const int j = k - a_;
      if (j < 1 || j > m_) {
        throw std::runtime_error("SketchWindow: step outside the window.");
      }
      if (j != j_cur_ + 1) {
        throw std::runtime_error("SketchWindow: columns must arrive in order.");
      }
      j_cur_ = j;

      ingest(u, j);

      if (policy_.rolling) {
        std::copy(u, u + M_, buf_roll_.begin() + static_cast<size_t>((j-1)%policy_.rho)*M_);
        if (j%policy_.rho == 0 || j == m_) {
          monitorRolling(j, u);
        }
      }
      else {
        seen_[j-1] = true;
        std::copy(u, u + M_, buf_.begin() + static_cast<size_t>(j-1)*M_);
        if (adaptive_ && (j%policy_.rho == 0 || j == m_)) {
          monitorFull(j);
        }
      }
    }

    /**
     * @brief Static-mode commit: requires every column recorded.  Phase-1
     *        semantics on failure: the payload is committed and served even
     *        when the tolerance is missed; only the status flags it.
     */
    void finalize(const double & tol) {
      if (adaptive_) {
        throw std::runtime_error("SketchWindow: finalize is for static mode only.");
      }
      for (int j=0; j<m_; ++j) {
        if (!seen_[j]) {
          throw std::runtime_error("SketchWindow: finalize with missing columns.");
        }
      }
      peek(m_);
      finalErrs(m_);
      const double* last = buf_.data() + static_cast<size_t>(m_-1)*M_;
      commitPeek(m_, (err_max_col_ <= tol) ? WindowStatus::Accepted : WindowStatus::Fallback,
                 last);
    }

    /// Reconstruct the state at real step k into out (M doubles).
    void reconstruct(const int & k, double* out) const {
      requirePayload();
      const int j = k - a_;
      if (j < 1 || j > b_eff_ - a_) {
        throw std::runtime_error("SketchWindow: reconstruct outside the committed range.");
      }
      const int q = payload_cols_;
      std::vector<double> row(q);
      for (int c=0; c<q; ++c) {
        row[c] = Q2_[static_cast<size_t>(c)*(b_eff_ - a_) + (j-1)];
      }
      dense::gemv('N', M_, q, 1.0, B1_.data(), M_, row.data(), 0.0, out);
    }

    /// The exact right-edge anchor u_{b_eff}, or its reconstruction if released.
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
        throw std::runtime_error("SketchWindow: no anchor held.");
      }
      std::copy(anchor_.begin(), anchor_.end(), out);
    }

    bool hasAnchor() const { return has_anchor_; }

    /// Drop the anchor (saves one state-equivalent; rightEdgeRaw then reconstructs).
    void releaseAnchor() {
      anchor_.clear();
      has_anchor_ = false;
    }

    /// External demotion by the budget governor: forget the payload entirely.
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

    // ---- setup ----

    void startRecording(const int & Mdim) {
      M_ = Mdim;
      sizes();
      streams();

      V1_.assign(static_cast<size_t>(M_)*k1_, 0.0);
      V2_.assign(static_cast<size_t>(m_)*k2_, 0.0);
      H_.assign(static_cast<size_t>(s1_)*s2_, 0.0);
      G1_.assign(static_cast<size_t>(s1_)*k1_, 0.0);
      G2_.assign(static_cast<size_t>(s2_)*k2_, 0.0);

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

      has_snapshot_ = false;
      cand_anchor_.clear();
      clearPeek();
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

    void sizes() {
      k1_ = (2*rank_ + 1 < M_) ? 2*rank_ + 1 : M_;
      k2_ = 2*rank_ + 1;
      s1_ = 2*k1_ + 1;
      s2_ = 2*k2_ + 1;
    }

    // The four streams: Om1 rows, the fixed Om2, the fixed Ph1, Ph2 rows.
    // Fresh seeds per retry, so every retry draws entirely new test matrices.
    void streams() {
      seed0_ = master_seed_ + 1000ULL*window_id_ + 10ULL*retry_attempt_;
      caches_valid_ = false;
      om2_cache_.clear();
      ph1t_cache_.clear();
    }

    // Regenerate the fixed test matrices from their seeds.  Evictable at will:
    // holding them is a wall-clock optimization, never a storage requirement.
    void ensureCaches() {
      if (caches_valid_) { return; }

      om2_cache_.resize(static_cast<size_t>(M_)*k2_);
      Threefry om2(seed0_ + 2, 0);
      om2.fillNormal(0, 0, M_*k2_, om2_cache_.data());

      // Ph1 is drawn M x s1 and held transposed so hcol = Ph1'*u is contiguous
      std::vector<double> ph1(static_cast<size_t>(M_)*s1_);
      Threefry ph1s(seed0_ + 3, 0);
      ph1s.fillNormal(0, 0, M_*s1_, ph1.data());
      ph1t_cache_.resize(static_cast<size_t>(s1_)*M_);
      for (int c=0; c<s1_; ++c) {
        for (int r=0; r<M_; ++r) {
          ph1t_cache_[static_cast<size_t>(r)*s1_ + c] = ph1[static_cast<size_t>(c)*M_ + r];
        }
      }
      caches_valid_ = true;
    }

    double transientSE(const int & r) const {
      if (policy_.rolling) {
        double A = 0.0, C = 0.0;
        rollingStreamCoeffs(M_, r, policy_.rho, A, C);
        return A*m_ + C;
      }
      const double k = 2.0*r + 1.0;
      const double doubles = static_cast<double>(M_)*m_ + M_*k + m_*k + M_*k + m_*k + M_;
      return doubles/M_;
    }

    // ---- streaming update ----

    // One column update: V1 += u*om1, V2 row j = u'*Om2, and the incremental
    // projections H, G1, G2.  O(M*(k1+k2+s1)) once the caches exist.
    void ingest(const double* u, const int & j) {

      ensureCaches();

      std::vector<double> om1(k1_), ph2(s2_), v2j(k2_), hcol(s1_);

      Threefry om1s(seed0_ + 1, 0);
      om1s.fillNormal(static_cast<uint64_t>(j), 0, k1_, om1.data());
      Threefry ph2s(seed0_ + 4, 0);
      ph2s.fillNormal(static_cast<uint64_t>(j), 0, s2_, ph2.data());

      // rank-1 updates as gemm with an inner dimension of one
      dense::gemm('N', 'N', M_, k1_, 1, 1.0, u, M_, om1.data(), 1, 1.0, V1_.data(), M_);

      dense::gemv('T', M_, k2_, 1.0, om2_cache_.data(), M_, u, 0.0, v2j.data());
      for (int c=0; c<k2_; ++c) {
        V2_[static_cast<size_t>(c)*m_ + (j-1)] = v2j[c];      // row write, once, immutable
      }

      dense::gemv('N', s1_, M_, 1.0, ph1t_cache_.data(), s1_, u, 0.0, hcol.data());

      dense::gemm('N', 'N', s1_, s2_, 1, 1.0, hcol.data(), s1_, ph2.data(), 1, 1.0, H_.data(), s1_);
      dense::gemm('N', 'N', s1_, k1_, 1, 1.0, hcol.data(), s1_, om1.data(), 1, 1.0, G1_.data(), s1_);
      dense::gemm('N', 'N', s2_, k2_, 1, 1.0, ph2.data(), s2_, v2j.data(), 1, 1.0, G2_.data(), s2_);
    }

    // ---- trial factorization ----

    // Build trial factors from the accumulators using columns 1..j, read-only.
    // No test matrices needed: Phi'*Q = G/R by the incremental projections.
    void peek(const int & j) {

      // column-pivoted QR of V1; rank cut at 1e-12 of the leading diagonal
      std::vector<double> F1 = V1_;
      std::vector<double> tau1;
      std::vector<int> e1;
      dense::qrPivotedFactor(M_, k1_, F1, tau1, e1);

      const int q1 = (M_ < k1_) ? M_ : k1_;
      double dmax1 = 0.0;
      for (int c=0; c<q1; ++c) {
        dmax1 = std::max(dmax1, std::abs(F1[static_cast<size_t>(c)*M_ + c]));
      }
      int rk1 = 0;
      for (int c=0; c<q1; ++c) {
        if (std::abs(F1[static_cast<size_t>(c)*M_ + c]) > 1.0e-12*dmax1) { ++rk1; }
      }
      rk1 = std::max(1, rk1);

      std::vector<double> R1k(static_cast<size_t>(rk1)*rk1, 0.0);
      for (int c=0; c<rk1; ++c) {
        for (int r=0; r<=c; ++r) {
          R1k[static_cast<size_t>(c)*rk1 + r] = F1[static_cast<size_t>(c)*M_ + r];
        }
      }
      dense::formQ(M_, q1, q1, F1, tau1);
      pQ1_.assign(F1.begin(), F1.begin() + static_cast<size_t>(M_)*rk1);
      p_rk1_ = rk1;

      // PhiQ1 = G1(:, e1)(:, 1:rk1) / R1k  = Phi1' * Q1k, exactly
      std::vector<double> PhiQ1(static_cast<size_t>(s1_)*rk1);
      for (int c=0; c<rk1; ++c) {
        const double* src = G1_.data() + static_cast<size_t>(e1[c])*s1_;
        std::copy(src, src + s1_, PhiQ1.begin() + static_cast<size_t>(c)*s1_);
      }
      dense::trsm('R', 'U', 'N', s1_, rk1, R1k.data(), rk1, PhiQ1.data(), s1_);

      // same on the time side, first j rows of V2; the Q factor is never formed
      std::vector<double> Vtop(static_cast<size_t>(j)*k2_);
      for (int c=0; c<k2_; ++c) {
        for (int r=0; r<j; ++r) {
          Vtop[static_cast<size_t>(c)*j + r] = V2_[static_cast<size_t>(c)*m_ + r];
        }
      }
      std::vector<double> tau2;
      std::vector<int> e2;
      dense::qrPivotedFactor(j, k2_, Vtop, tau2, e2);

      const int q2 = (j < k2_) ? j : k2_;
      double dmax2 = 0.0;
      for (int c=0; c<q2; ++c) {
        dmax2 = std::max(dmax2, std::abs(Vtop[static_cast<size_t>(c)*j + c]));
      }
      int rk2 = 0;
      for (int c=0; c<q2; ++c) {
        if (std::abs(Vtop[static_cast<size_t>(c)*j + c]) > 1.0e-12*dmax2) { ++rk2; }
      }
      rk2 = std::max(1, rk2);

      pR2_.assign(static_cast<size_t>(rk2)*rk2, 0.0);
      for (int c=0; c<rk2; ++c) {
        for (int r=0; r<=c; ++r) {
          pR2_[static_cast<size_t>(c)*rk2 + r] = Vtop[static_cast<size_t>(c)*j + r];
        }
      }
      pE2_.assign(e2.begin(), e2.begin() + rk2);
      p_rk2_ = rk2;

      std::vector<double> PhiQ2(static_cast<size_t>(s2_)*rk2);
      for (int c=0; c<rk2; ++c) {
        const double* src = G2_.data() + static_cast<size_t>(e2[c])*s2_;
        std::copy(src, src + s2_, PhiQ2.begin() + static_cast<size_t>(c)*s2_);
      }
      dense::trsm('R', 'U', 'N', s2_, rk2, pR2_.data(), rk2, PhiQ2.data(), s2_);

      // A1 ~ pinv(PhiQ1), A2 ~ pinv(PhiQ2): tall least-squares against I
      std::vector<double> I1(static_cast<size_t>(s1_)*s1_, 0.0);
      for (int i=0; i<s1_; ++i) { I1[static_cast<size_t>(i)*s1_ + i] = 1.0; }
      dense::leastSquares(s1_, rk1, s1_, PhiQ1, I1);
      std::vector<double> A1(static_cast<size_t>(rk1)*s1_);
      for (int c=0; c<s1_; ++c) {
        for (int r=0; r<rk1; ++r) {
          A1[static_cast<size_t>(c)*rk1 + r] = I1[static_cast<size_t>(c)*s1_ + r];
        }
      }

      std::vector<double> I2(static_cast<size_t>(s2_)*s2_, 0.0);
      for (int i=0; i<s2_; ++i) { I2[static_cast<size_t>(i)*s2_ + i] = 1.0; }
      dense::leastSquares(s2_, rk2, s2_, PhiQ2, I2);
      std::vector<double> A2(static_cast<size_t>(rk2)*s2_);
      for (int c=0; c<s2_; ++c) {
        for (int r=0; r<rk2; ++r) {
          A2[static_cast<size_t>(c)*rk2 + r] = I2[static_cast<size_t>(c)*s2_ + r];
        }
      }

      // core pW = A1 * H * A2'
      std::vector<double> T(static_cast<size_t>(rk1)*s2_);
      dense::gemm('N', 'N', rk1, s2_, s1_, 1.0, A1.data(), rk1, H_.data(), s1_, 0.0, T.data(), rk1);
      pW_.assign(static_cast<size_t>(rk1)*rk2, 0.0);
      dense::gemm('N', 'T', rk1, rk2, s2_, 1.0, T.data(), rk1, A2.data(), rk2, 0.0, pW_.data(), rk1);

      std::vector<double> U, VT;
      dense::svd(rk1, rk2, pW_, U, pSv_, VT);
      peek_valid_ = true;
    }

    // Reconstruct column jj from the current peek factors.
    void recon(const int & jj, double* out) const {

      std::vector<double> q2(p_rk2_);
      for (int c=0; c<p_rk2_; ++c) {
        q2[c] = V2_[static_cast<size_t>(pE2_[c])*m_ + (jj-1)];
      }
      dense::trsm('L', 'U', 'T', p_rk2_, 1, pR2_.data(), p_rk2_, q2.data(), p_rk2_);

      std::vector<double> t(p_rk1_);
      dense::gemv('N', p_rk1_, p_rk2_, 1.0, pW_.data(), p_rk1_, q2.data(), 0.0, t.data());
      dense::gemv('N', M_, p_rk1_, 1.0, pQ1_.data(), M_, t.data(), 0.0, out);
    }

    // ---- commits ----

    void commitPeek(const int & j, const WindowStatus & status, const double* anchor_vec) {

      B1_.assign(static_cast<size_t>(M_)*p_rk2_, 0.0);
      dense::gemm('N', 'N', M_, p_rk2_, p_rk1_, 1.0, pQ1_.data(), M_, pW_.data(), p_rk1_,
                  0.0, B1_.data(), M_);

      // Q2 = V2(1:j, e2) / R2: one transpose-triangular solve with j right-hand sides
      std::vector<double> X(static_cast<size_t>(p_rk2_)*j);
      for (int r=0; r<j; ++r) {
        for (int c=0; c<p_rk2_; ++c) {
          X[static_cast<size_t>(r)*p_rk2_ + c] = V2_[static_cast<size_t>(pE2_[c])*m_ + r];
        }
      }
      dense::trsm('L', 'U', 'T', p_rk2_, j, pR2_.data(), p_rk2_, X.data(), p_rk2_);
      Q2_.assign(static_cast<size_t>(j)*p_rk2_, 0.0);
      for (int r=0; r<j; ++r) {
        for (int c=0; c<p_rk2_; ++c) {
          Q2_[static_cast<size_t>(c)*j + r] = X[static_cast<size_t>(r)*p_rk2_ + c];
        }
      }

      payload_cols_ = p_rk2_;
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
      V1_.clear();
      V2_.clear();
      H_.clear();
      G1_.clear();
      G2_.clear();
      snapV1_.clear();
      snapH_.clear();
      snapG1_.clear();
      snapG2_.clear();
      has_snapshot_ = false;
      om2_cache_.clear();
      ph1t_cache_.clear();
      caches_valid_ = false;
      clearPeek();
      open_ = false;
    }

    void clearPeek() {
      pQ1_.clear();
      pW_.clear();
      pR2_.clear();
      pE2_.clear();
      pSv_.clear();
      p_rk1_ = p_rk2_ = 0;
      peek_valid_ = false;
    }

    // O(k) state-equivalents: V2 rows are immutable so V2 needs no snapshot,
    // and the raw states are not snapshotted at all.
    void snapshot() {
      snapV1_ = V1_;
      snapH_ = H_;
      snapG1_ = G1_;
      snapG2_ = G2_;
      has_snapshot_ = true;
    }

    void restoreSnapshot() {
      V1_ = snapV1_;
      H_ = snapH_;
      G1_ = snapG1_;
      G2_ = snapG2_;
    }

    // ---- monitors and the retry/cut/fallback machine ----

    int minCommitEff() const {
      return std::max(policy_.min_commit, k2_ + 2);
    }

    // With fewer than k2+2 columns the sketch trivially spans the data, so
    // checking wastes work.  The j == m case is never skipped.
    bool youngPass(const int & j) const {
      return j < k2_ + 2 && j < m_;
    }

    double relColErr(const double* approx, const double* truth) const {
      std::vector<double> d(M_);
      for (int i=0; i<M_; ++i) { d[i] = approx[i] - truth[i]; }
      const double nt = dense::norm2(M_, truth);
      return dense::norm2(M_, d.data())/std::max(nt, DBL_EPSILON);
    }

    void monitorRolling(const int & j, const double* u_last) {

      if (youngPass(j)) {
        last_good_ = j;
        cand_anchor_.assign(u_last, u_last + M_);
        snapshot();
        return;
      }

      peek(j);
      double e = 0.0;
      std::vector<double> uhat(M_);
      for (int jj=last_good_+1; jj<=j; ++jj) {
        recon(jj, uhat.data());
        const double* truth = buf_roll_.data() + static_cast<size_t>((jj-1)%policy_.rho)*M_;
        e = std::max(e, relColErr(uhat.data(), truth));
      }

      if (e <= policy_.tol) {
        last_good_ = j;
        err_run_ = std::max(err_run_, e);
        cand_anchor_.assign(u_last, u_last + M_);
        snapshot();
        if (j == m_) {
          err_max_col_ = err_fro_ = err_run_;
          commitPeek(m_, WindowStatus::Accepted, cand_anchor_.data());
        }
      }
      else {
        fitRankHint();
        cutRolling();
      }
    }

    // Rolling failure resolution.  No retries here: the raw states are gone.
    // The failing columns are simply lost; the classic revolve axis recomputes them.
    void cutRolling() {
      if (last_good_ >= minCommitEff() && has_snapshot_) {
        restoreSnapshot();
        peek(last_good_);
        err_max_col_ = err_fro_ = err_run_;
        commitPeek(last_good_, WindowStatus::Prefix, cand_anchor_.data());
      }
      else {
        fallback();
      }
    }

    void monitorFull(const int & j) {

      if (youngPass(j)) {
        last_good_ = j;
        return;
      }

      peek(j);
      double e = 0.0;
      std::vector<double> uhat(M_);
      for (int jj=last_good_+1; jj<=j; ++jj) {
        recon(jj, uhat.data());
        e = std::max(e, relColErr(uhat.data(), buf_.data() + static_cast<size_t>(jj-1)*M_));
      }

      if (e <= policy_.tol) {
        last_good_ = j;
        if (j == m_) {
          finalErrs(j);
          commitPeek(j, WindowStatus::Accepted, buf_.data() + static_cast<size_t>(j-1)*M_);
        }
      }
      else {
        fitRankHint();
        retryOrCut(j);
      }
    }

    // Full-mode retries: new seeds, higher rank, re-stream the buffer -- no
    // PDE solves.  A mid-window success leaves the window open and recording
    // at the higher rank.
    void retryOrCut(const int & j) {

      while (retry_attempt_ < policy_.max_retries && rank_ < policy_.rank_max) {

        bumpRetry();
        rank_ = std::min(std::max(next_rank_hint_, rank_ + policy_.rank_bump),
                         policy_.rank_max);
        restream(j);
        peak_se_ = std::max(peak_se_, transientSE(rank_));

        peek(j);
        double e = 0.0;
        std::vector<double> uhat(M_);
        for (int jj=1; jj<=j; ++jj) {       // full recheck: new sketch, old verification void
          recon(jj, uhat.data());
          e = std::max(e, relColErr(uhat.data(), buf_.data() + static_cast<size_t>(jj-1)*M_));
        }

        if (e <= policy_.tol) {
          last_good_ = j;
          if (j == m_) {
            finalErrs(j);
            commitPeek(j, WindowStatus::Accepted, buf_.data() + static_cast<size_t>(m_-1)*M_);
          }
          return;
        }
        fitRankHint();
      }

      // retries exhausted or rank at the cap
      if (last_good_ >= minCommitEff()) {
        restream(last_good_);
        peek(last_good_);
        finalErrs(last_good_);
        if (err_max_col_ <= policy_.tol) {
          commitPeek(last_good_, WindowStatus::Prefix,
                     buf_.data() + static_cast<size_t>(last_good_-1)*M_);
        }
        else {
          fallback();
        }
      }
      else {
        fallback();
      }
    }

    // Rebuild the accumulators at the current rank/seeds from the buffer.
    void restream(const int & j) {
      sizes();
      streams();
      V1_.assign(static_cast<size_t>(M_)*k1_, 0.0);
      V2_.assign(static_cast<size_t>(m_)*k2_, 0.0);
      H_.assign(static_cast<size_t>(s1_)*s2_, 0.0);
      G1_.assign(static_cast<size_t>(s1_)*k1_, 0.0);
      G2_.assign(static_cast<size_t>(s2_)*k2_, 0.0);
      for (int jj=1; jj<=j; ++jj) {
        ingest(buf_.data() + static_cast<size_t>(jj-1)*M_, jj);
      }
    }

    // All-column error against the buffer (full/static modes).
    void finalErrs(const int & j) {
      double emax = 0.0, num = 0.0, den = 0.0;
      std::vector<double> uhat(M_), d(M_);
      for (int jj=1; jj<=j; ++jj) {
        const double* truth = buf_.data() + static_cast<size_t>(jj-1)*M_;
        recon(jj, uhat.data());
        for (int i=0; i<M_; ++i) { d[i] = uhat[i] - truth[i]; }
        const double nd = dense::norm2(M_, d.data());
        const double nt = dense::norm2(M_, truth);
        emax = std::max(emax, nd/std::max(nt, DBL_EPSILON));
        num += nd*nd;
        den += nt*nt;
      }
      err_max_col_ = err_run_ = emax;
      err_fro_ = std::sqrt(num/std::max(den, DBL_EPSILON));
    }

    // Spectral fit on the trial core's singular values: how much rank would
    // this data need?  Feeds in-window retries and the greedy planner's
    // next-window inheritance.
    void fitRankHint() {

      const int nsv = static_cast<int>(pSv_.size());
      if (nsv < 1) { return; }

      double tot2 = 0.0;
      for (int i=0; i<nsv; ++i) { tot2 += pSv_[i]*pSv_[i]; }
      const double tot = std::sqrt(tot2);

      // relative tails: tail[r-1] = sqrt(sum_{i>r} sv_i^2)/tot for 1-based rank r
      std::vector<double> tail(nsv);
      double run = 0.0;
      for (int r=nsv-1; r>=0; --r) {
        tail[r] = std::sqrt(run)/std::max(tot, DBL_MIN);
        run += pSv_[r]*pSv_[r];
      }

      const double target = policy_.tol/10.0;

      for (int r=0; r<nsv; ++r) {
        if (tail[r] <= target) {
          const int idx = r + 1;              // 1-based data rank
          next_rank_hint_ = std::max(next_rank_hint_, (idx + 2)/2);   // ceil((idx+1)/2)
          return;
        }
      }

      // log-linear extrapolation of the tail decay
      const int lo = std::max(2, nsv/2);
      const int hi = nsv - 1;
      if (hi - lo + 1 < 2) { return; }
      if (tail[hi-1] >= tail[lo-1]*0.999) { return; }     // no decay

      double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
      const int npts = hi - lo + 1;
      for (int r=lo; r<=hi; ++r) {
        const double x = r;
        const double y = std::log(std::max(tail[r-1], DBL_MIN));
        sx += x; sy += y; sxx += x*x; sxy += x*y;
      }
      const double denom = npts*sxx - sx*sx;
      if (denom == 0.0) { return; }
      const double c2 = (npts*sxy - sx*sy)/denom;
      const double c1 = (sy - c2*sx)/npts;
      if (c2 >= -1.0e-12) { return; }                     // non-decreasing fit

      const double k_star = std::ceil((std::log(target) - c1)/c2);
      const int hint = static_cast<int>(std::ceil((k_star + 1.0)/2.0));
      next_rank_hint_ = std::max(next_rank_hint_, std::min(hint, 60));
    }

    void requirePayload() const {
      if (B1_.empty()) {
        throw std::runtime_error("SketchWindow: no committed payload.");
      }
    }

    // ---- configuration ----
    int a_, b_, m_;
    int rank_;
    int window_id_;
    uint64_t master_seed_;
    int retry_attempt_ = 0;
    int M_ = 0;
    bool adaptive_ = false;
    WindowPolicy policy_;
    uint64_t seed0_ = 0;

    // ---- results ----
    WindowStatus status_ = WindowStatus::Planned;
    double err_max_col_ = std::numeric_limits<double>::infinity();
    double err_fro_ = std::numeric_limits<double>::infinity();
    int b_eff_ = 0;
    int last_good_ = 0;
    int next_rank_hint_ = 0;
    double peak_se_ = 0.0;

    // ---- committed payload ----
    std::vector<double> B1_;        ///< M x q
    std::vector<double> Q2_;        ///< (b_eff - a) x q
    int payload_cols_ = 0;
    std::vector<double> anchor_;    ///< exact u_{b_eff}
    bool has_anchor_ = false;

    // ---- transients (live only while recording) ----
    int k1_ = 0, k2_ = 0, s1_ = 0, s2_ = 0;
    std::vector<double> V1_, V2_, H_, G1_, G2_;
    std::vector<double> snapV1_, snapH_, snapG1_, snapG2_;
    bool has_snapshot_ = false;
    std::vector<double> om2_cache_, ph1t_cache_;
    bool caches_valid_ = false;
    std::vector<double> buf_;
    std::vector<bool> seen_;
    std::vector<double> buf_roll_;
    std::vector<double> cand_anchor_;
    int j_cur_ = 0;
    double err_run_ = 0.0;
    bool open_ = false;

    // ---- last trial factors ----
    std::vector<double> pQ1_, pW_, pR2_, pSv_;
    std::vector<int> pE2_;
    int p_rk1_ = 0, p_rk2_ = 0;
    bool peek_valid_ = false;
  };

}

#endif
