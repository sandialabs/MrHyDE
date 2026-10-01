/***********************************************************************
 MrHyDE - Diagonal-inverse strategy for the block-triangular preconditioner.

 Supplies Teko with one approximate inverse per field split: split 0 is inverted as
 given, every split after it on a Schur approximation. Teko's Gauss-Seidel factory calls
 getInvD once per build and composes the triangular solve from the result.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_SCHUR_INV_DIAG_STRATEGY_HPP
#define MRHYDE_BLOCK_PREC_SCHUR_INV_DIAG_STRATEGY_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/BlockVerify.hpp"
#include "block_prec/SplitInverse.hpp"
#include "block_prec/SchurApproximation.hpp"
#include "block_prec/TekoAdapter.hpp"

#include <Teko_BlockInvDiagonalStrategy.hpp>

#include <sstream>
#include <vector>

namespace MrHyDE {
namespace block_prec {

template<class Node>
class SchurInvDiagStrategy : public Teko::BlockInvDiagonalStrategy {
public:
  using Types = BlockTypes<Node>;
  using matrix_rcp = typename Types::CrsMatrixRCP;

  SchurInvDiagStrategy(const matrix_rcp & J,
                       const BlockSystem<Node> & blocks,
                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                       const int verbosity)
    : J_(J), blocks_(blocks), cntxt_(cntxt), verbosity_(verbosity),
      comm_(J->getComm()) {}

  // invDiag[k] inverts S_k = J_kk - damping * sum_{j<k} J_kj diag(J_jj)^-1 J_jk ('diag'),
  // J_kk ('base'), or J_kk + mass scale * M_k on the target ('mass').
  void getInvD(const Teko::BlockedLinearOp & /* A */,
               Teko::BlockPreconditionerState & /* state */,
               std::vector<Teko::LinearOp> & invDiag) const override {
    const size_t nb = blocks_.numBlocks();
    const SchurVariant variant = parseSchurVariant(cntxt_->schur.approximation_type);
    const bool useDiag = (variant == SchurVariant::Diag);
    if (useDiag && verbosity_ >= 5 && comm_->getRank() == 0 &&
        cntxt_->schur.damping == Teuchos::ScalarTraits<ScalarT>::zero()) {
      std::cout << "[BlockTri] damping is 0; the diagonal correction is off." << std::endl;
    }
    // Only a split that weights a later split in its own group needs a diagonal inverse.
    std::vector<Teuchos::RCP<typename Types::Vector> > weightInverse(nb);
    if (useDiag) {
      for (size_t j = 0; j + 1 < nb; ++j) {
        if (!this->eliminatesWithin(j, j + 1)) continue;
        weightInverse[j] = this->weightDiagonalInverse(j);
      }
    }

    invDiag.clear();
    for (size_t k = 0; k < nb; ++k) {
      matrix_rcp S = blocks_.blocks[k][k];
      matrix_rcp corr;
      if (useDiag) {
        for (size_t j = 0; j < k; ++j) {
          if (!this->eliminatesWithin(j, k)) continue;
          SchurAssemblyInputs<Node> inputs;
          inputs.base = S;
          inputs.left = blocks_.blocks[k][j];
          inputs.weight = blocks_.blocks[j][j];
          inputs.right = blocks_.blocks[j][k];
          inputs.scale = -cntxt_->schur.damping;
          validateSchurAssemblyInputs<Node>(inputs, "Schur assembly 'diag'");
          corr = buildCorrectionMatrix<Node>(inputs, weightInverse[j]);
          S = addCorrection<Node>(S, corr);
        }
      }
      if (k == cntxt_->schur_target_index && variant == SchurVariant::Mass) {
        S = addCorrection<Node>(S, this->targetMassMatrix(), cntxt_->schur.mass_scale);
      }
      if (k == cntxt_->schur_target_index) {
        // beta is read off the 2x2 correction, so it stays undefined above two splits.
        this->recordSchurAddonBeta(nb == 2 ? corr : Teuchos::null);
        verifyBlockSystem<Node>(blocks_, J_, S,
          cntxt_->refMaxwell.D0_matrix, cntxt_->refMaxwell.M1_matrix,
          cntxt_->refMaxwell.nodal_coords, cntxt_->refMaxwell.nodal_lumped_mass,
          cntxt_->schur.damping, cntxt_->schur.correction_use_lumped_weight,
          useDiag, this->blockVerifyRequested(), verbosity_);
      }
      invDiag.push_back(this->buildSplitInverse(k, S));
    }
  }

private:
  // Split j weights split k only inside one pivot/target group.
  bool eliminatesWithin(const size_t j, const size_t k) const {
    const size_t g = cntxt_->splitGroupOf(k);
    return g == cntxt_->splitGroupOf(j) && !cntxt_->splitGroupIsJacobi(g);
  }

  // 'verify: true' asks for the identity checks without raising verbosity, which would drag
  // every MueLu list along with it.
  bool blockVerifyRequested() const {
    const Teuchos::ParameterList & target = cntxt_->splitSettings(cntxt_->schur_target_index);
    bool wanted = false;
    for (const char * sub : {"RefMaxwell Settings", "Maxwell1 Settings"}) {
      if (!target.isSublist(sub)) continue;
      wanted = wanted || detail::readFilterOpts(target.sublist(sub)).verifyBlocks;
    }
    return wanted;
  }

  // A zero row here would only warn and zero the inverse, leaving a singular S_k.
  Teuchos::RCP<typename Types::Vector> weightDiagonalInverse(const size_t split) const {
    using LA_CrsMatrix = typename Types::CrsMatrix;
    detail::InverseDiagonalCounts counts;
    Teuchos::RCP<typename Types::Vector> dinv = detail::buildInverseDiagonal<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(blocks_.blocks[split][split]),
      cntxt_->schur.correction_use_lumped_weight, counts);
    detail::reportInverseDiagonal<Node>(counts, "Schur weight diag inverse", comm_, verbosity_);
    GO missing = 0;
    Teuchos::reduceAll<int, GO>(*comm_, Teuchos::REDUCE_SUM, 1, &counts.missing, &missing);
    TEUCHOS_TEST_FOR_EXCEPTION(missing > 0, std::runtime_error,
      "Block-triangular 'diag' Schur: split " << split << " has " << missing
      << " rows with no usable diagonal, so it cannot weight the elimination.");
    return dinv;
  }

  // S = J_kk + scale * M_k; scale is 1/nu for constant viscosity.
  matrix_rcp targetMassMatrix() const {
    const matrix_rcp mass =
      massMatrixOnMap<Node>(cntxt_->block.mass_matrices,
                            blocks_.maps[cntxt_->schur_target_index]);
    TEUCHOS_TEST_FOR_EXCEPTION(mass.is_null(), std::runtime_error,
      "Schur 'approximation type: mass' needs the block mass matrix on the target split's "
      "map, which was not assembled.");
    return mass;
  }

  Teko::LinearOp buildSplitInverse(const size_t split, const matrix_rcp & mat) const {
    const Teuchos::ParameterList & blockList = cntxt_->splitSettings(split);
    const BlockPrecType type = parseBlockPrecType(cntxt_->splitPrecType(split));
    std::ostringstream label;
    label << "BlockTri split " << split;
    const std::string name = label.str();
    Teuchos::ParameterList params = splitMueLuParams<Node>(cntxt_, split, mat, verbosity_);
    return buildBlockOperator<Node>(mat, cntxt_, blockList, type, split, name, verbosity_,
      [&] {
        return cntxt_->inverseLibrary(verbosity_, comm_->getRank()).build(
          "MueLu", params, name + " MueLu", mat);
      });
  }

  void recordSchurAddonBeta(const matrix_rcp & schurCorr) const {
    RefMaxwellData<Node> & refMaxwell = cntxt_->refMaxwell;
    const bool wanted = cntxt_->schurAddonWanted(*J_->getComm());
    const matrix_rcp pivotMass = wanted
      ? massMatrixOnMap<Node>(cntxt_->block.mass_matrices, blocks_.maps[0]) : Teuchos::null;
    const bool haveInputs = !schurCorr.is_null() && !refMaxwell.nodal_lumped_mass.is_null() &&
      !pivotMass.is_null();
    if (!wanted || !haveInputs) {
      refMaxwell.schur_addon_beta = 0.0;
      refMaxwell.schur_addon_beta_valid = false;
      return;
    }
    // Cached per stage_alpha_u and recomputed only when it changes.
    if (refMaxwell.schur_addon_beta_valid &&
        refMaxwell.schur_addon_beta_alpha_u == cntxt_->stage_alpha_u) {
      return;
    }
    refMaxwell.schur_addon_beta = addonBeta<Node>(
      blocks_, schurCorr, pivotMass,
      cntxt_->schur.correction_use_lumped_weight,
      cntxt_->stage_alpha_u, verbosity_);
    refMaxwell.schur_addon_beta_alpha_u = cntxt_->stage_alpha_u;
    refMaxwell.schur_addon_beta_valid = true;
  }

  matrix_rcp J_;
  BlockSystem<Node> blocks_;
  Teuchos::RCP<LinearSolverContext<Node> > cntxt_;
  int verbosity_;
  Teuchos::RCP<const Teuchos::Comm<int> > comm_;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
