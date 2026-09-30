/***********************************************************************
 MrHyDE - Diagonal-inverse strategy for the block-triangular preconditioner.

 Supplies Teko with one approximate inverse per block row: the pivot inverse for
 role 0 and a Schur inverse for every role after it. Teko's Gauss-Seidel factory
 calls getInvD once per build and composes the triangular solve from the result.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_SCHUR_INV_DIAG_STRATEGY_HPP
#define MRHYDE_BLOCK_PREC_SCHUR_INV_DIAG_STRATEGY_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/BlockVerify.hpp"
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

  SchurInvDiagStrategy(LinearAlgebraInterface<Node> & interface,
                       const matrix_rcp & J,
                       const BlockSystem<Node> & blocks,
                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt)
    : interface_(interface), J_(J), blocks_(blocks), cntxt_(cntxt) {}

  // invDiag[b] is the inverse for block row b, so the pivot comes first.
  void getInvD(const Teko::BlockedLinearOp & /* A */,
               Teko::BlockPreconditionerState & /* state */,
               std::vector<Teko::LinearOp> & invDiag) const override {
    if (blocks_.numBlocks() != 2) {
      this->getInvDNBlock(invDiag);
      return;
    }
    matrix_rcp schurCorr;
    matrix_rcp schurApprox =
      interface_.buildBlockTriangularSchurApproximation(blocks_, cntxt_, &schurCorr);
    if (interface_.verbosity >= 5 && interface_.comm->getRank() == 0 &&
        cntxt_->schur.damping == Teuchos::ScalarTraits<ScalarT>::zero()) {
      std::cout << "[BlockTri] damping is 0; the diagonal correction is off." << std::endl;
    }
    this->recordSchurAddonBeta(schurCorr);

    verifyBlockSystem<Node>(blocks_, J_, schurApprox,
      cntxt_->refMaxwell.D0_matrix, cntxt_->refMaxwell.M1_matrix,
      cntxt_->refMaxwell.nodal_coords, cntxt_->refMaxwell.nodal_lumped_mass,
      cntxt_->schur.damping,
      cntxt_->schur.diag_use_lumped_pivot_diagonal,
      parseSchurVariant(cntxt_->schur.approximation_type) == SchurVariant::Diag,
      interface_.verbosity);

    Teuchos::ParameterList schurMueLuParams =
      interface_.getBlockTriangularMueLuParams(cntxt_, schurApprox);
    Teuchos::ParameterList pivotParams = this->pivotMueLuParams();

    invDiag.clear();
    invDiag.push_back(buildPivotBlockPrec<Node>(interface_, blocks_.J00, cntxt_, pivotParams));
    invDiag.push_back(buildSchurBlockPrec<Node>(interface_, schurApprox, cntxt_, schurMueLuParams));
  }

private:
  // S_k = J_kk - damping * sum_{j<k} J_kj diag(J_jj)^-1 J_jk, SIMPLE's approximation.
  void getInvDNBlock(std::vector<Teko::LinearOp> & invDiag) const {
    const size_t nb = blocks_.numBlocks();
    // No 2x2 role pair, so the RefMaxwell addon has nothing to read.
    this->recordSchurAddonBeta(Teuchos::null);
    const bool useDiag =
      parseSchurVariant(cntxt_->schur.approximation_type) == SchurVariant::Diag;
    // Every role except the last is used as an elimination weight.
    if (useDiag) {
      for (size_t j = 0; j + 1 < nb; ++j) this->requireUsableDiagonal(j);
    }
    invDiag.clear();
    for (size_t k = 0; k < nb; ++k) {
      matrix_rcp S = blocks_.blocks[k][k];
      if (useDiag) {
        for (size_t j = 0; j < k; ++j) {
          SchurAssemblyInputs<Node> inputs;
          inputs.base = S;
          inputs.left = blocks_.blocks[k][j];
          inputs.weight = blocks_.blocks[j][j];
          inputs.right = blocks_.blocks[j][k];
          inputs.scale = -cntxt_->schur.damping;
          inputs.useLumpedWeightDiagonal = cntxt_->schur.diag_use_lumped_pivot_diagonal;
          validateSchurAssemblyInputs<Node>(inputs, "Schur assembly 'diag'");
          S = addCorrection<Node>(
            S, buildCorrectionMatrix<Node>(inputs, interface_.verbosity));
        }
      }
      invDiag.push_back(this->buildRoleInverse(k, S));
    }
  }

  // buildCorrectionMatrix would only warn and zero the inverse, leaving a singular S_k.
  void requireUsableDiagonal(const size_t role) const {
    using LA_CrsMatrix = typename Types::CrsMatrix;
    detail::InverseDiagonalCounts counts;
    detail::buildInverseDiagonal<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(blocks_.blocks[role][role]),
      cntxt_->schur.diag_use_lumped_pivot_diagonal, counts);
    GO missing = 0;
    Teuchos::reduceAll<int, GO>(*interface_.comm, Teuchos::REDUCE_SUM, 1, &counts.missing,
                                &missing);
    TEUCHOS_TEST_FOR_EXCEPTION(missing > 0, std::runtime_error,
      "Block-triangular 'diag' Schur: role " << role << " has " << missing
      << " rows with no usable diagonal, so it cannot weight the elimination.");
  }

  Teko::LinearOp buildRoleInverse(const size_t role, const matrix_rcp & mat) const {
    const bool isPivot = (role == 0);
    // The context caches one pivot/Schur hierarchy pair, so two Schur roles would collide.
    TEUCHOS_TEST_FOR_EXCEPTION(
      !isPivot && blocks_.numBlocks() > 2 &&
      (parseBlockPrecType(cntxt_->schur.schur_block_preconditioner_type) == BlockPrecType::RefMaxwell ||
       parseBlockPrecType(cntxt_->schur.schur_block_preconditioner_type) == BlockPrecType::Maxwell1),
      std::runtime_error,
      "Schur block preconditioner '" << cntxt_->schur.schur_block_preconditioner_type
      << "' supports exactly two roles, but this set has " << blocks_.numBlocks()
      << "; use AMG or Direct.");
    std::ostringstream label;
    label << "BlockTri role " << role;
    const Teuchos::ParameterList & blockList = cntxt_->roleSettings(role);
    const BlockPrecType type = parseBlockPrecType(cntxt_->rolePrecType(role));
    Teuchos::ParameterList params =
      isPivot ? this->pivotMueLuParams()
              : interface_.getBlockTriangularMueLuParams(cntxt_, mat);
    const std::string name = label.str();
    return buildBlockOperator<Node>(interface_, mat, cntxt_, blockList, type, !isPivot, name,
      [&] {
        return interface_.inverseLibrary(cntxt_).build("MueLu", params, name + " MueLu", mat);
      });
  }

  void recordSchurAddonBeta(const matrix_rcp & schurCorr) const {
    RefMaxwellData<Node> & refMaxwell = cntxt_->refMaxwell;
    // By map, not variable index: a fused or renamed role has no single index.
    const matrix_rcp pivotMass =
      massMatrixOnMap<Node>(cntxt_->block.mass_matrices, blocks_.pivotMap);
    const bool wanted = cntxt_->schurAddonWanted(*J_->getComm());
    const bool haveInputs = !schurCorr.is_null() && !refMaxwell.nodal_lumped_mass.is_null() &&
      !pivotMass.is_null();
    if (!wanted || !haveInputs) {
      refMaxwell.schur_addon_beta = 0.0;
      refMaxwell.schur_addon_beta_valid = false;
      return;
    }
    // beta depends on the DIRK stage scaling, not on J.
    if (refMaxwell.schur_addon_beta_valid &&
        refMaxwell.schur_addon_beta_alpha_u == cntxt_->stage_alpha_u) {
      return;
    }
    refMaxwell.schur_addon_beta = addonBeta<Node>(
      blocks_, schurCorr, pivotMass,
      cntxt_->schur.diag_use_lumped_pivot_diagonal,
      cntxt_->stage_alpha_u, interface_.verbosity);
    refMaxwell.schur_addon_beta_alpha_u = cntxt_->stage_alpha_u;
    refMaxwell.schur_addon_beta_valid = true;
  }

  // Never inherit the Schur list: it is tuned for curl-curl, not the mass pivot.
  Teuchos::ParameterList pivotMueLuParams() const {
    Teuchos::ParameterList params = defaultMueLuParams();
    const Teuchos::ParameterList & pivotList = cntxt_->roleSettings(0);
    if (pivotList.isSublist("AMG Settings")) {
      const Teuchos::ParameterList & amg = pivotList.sublist("AMG Settings");
      Teuchos::ParameterList xmlLoaded;
      if (loadMueLuXmlIfPresent(amg, xmlLoaded, "pivot block", J_->getComm())) {
        params = xmlLoaded;
      }
      else {
        Teuchos::ParameterList filtered(amg);
        removeMrHyDEOwnedKeys(filtered);
        removeIfpack2OnlyKeys(filtered);
        params.setParameters(filtered);
        if (amg.isSublist("smoother: params")) {
          params.sublist("smoother: params").setParameters(amg.sublist("smoother: params"));
        }
      }
    }
    normalizeMueLuVerbosity(params, interface_.verbosity);
    return params;
  }

  LinearAlgebraInterface<Node> & interface_;
  matrix_rcp J_;
  BlockSystem<Node> blocks_;
  Teuchos::RCP<LinearSolverContext<Node> > cntxt_;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
