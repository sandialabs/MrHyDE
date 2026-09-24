#ifndef MRHYDE_BLOCK_PREC_TRIANGULAR_FACTORY_HPP
#define MRHYDE_BLOCK_PREC_TRIANGULAR_FACTORY_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/BlockVerify.hpp"
#include "block_prec/TekoAdapter.hpp"

#include <Teko_GaussSeidelPreconditionerFactory.hpp>

#include <vector>

namespace MrHyDE {
namespace block_prec {

// 2x2 block-triangular preconditioner. Not a Teko::BlockPreconditionerFactory:
// it builds a GaussSeidelPreconditionerFactory and applies it here.
template<class Node>
class BlockTriangularFactory {
public:
  using Types = BlockTypes<Node>;
  using matrix_rcp = typename Types::CrsMatrixRCP;
  using map_rcp = typename Types::MapRCP;

  BlockTriangularFactory(LinearAlgebraInterface<Node> & interface,
                         const matrix_rcp & J,
                         const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                         const size_t set)
    : interface_(interface), J_(J), cntxt_(cntxt), set_(set) {}

  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > build() {
    blocks_ = buildBlockSystemForSet<Node>(interface_, J_, cntxt_, set_);
    Teko::BlockedLinearOp blocked =
      buildThyraBlocked2x2<Node>(blocks_.J00, blocks_.J01, blocks_.J10, blocks_.J11);
    Teko::LinearOp prec = this->buildPreconditionerOperator(blocked);
    const std::vector<map_rcp> triMaps = {blocks_.pivotMap, blocks_.targetMap};
    return Teuchos::rcp(new TekoTpetraAdapter<Node>(J_->getRowMap(), triMaps, prec));
  }

  Teko::LinearOp buildPreconditionerOperator(Teko::BlockedLinearOp & blocked) const {
    matrix_rcp schurCorr;
    matrix_rcp schurApprox =
      interface_.buildBlockTriangularSchurApproximation(blocks_, cntxt_, &schurCorr);
    this->recordSchurAddonBeta(schurCorr);

    verifyBlockSystem<Node>(blocks_, J_, schurApprox,
      cntxt_->refMaxwell.D0_matrix, cntxt_->refMaxwell.M1_matrix,
      cntxt_->refMaxwell.nodal_coords, cntxt_->refMaxwell.nodal_lumped_mass,
      cntxt_->schur.damping,
      cntxt_->schur.diag_use_lumped_pivot_diagonal,
      parseSchurVariant(cntxt_->schur.approximation_type) == SchurVariant::Diag,
      interface_.verbosity);

    Teuchos::ParameterList schurMueLuParams = interface_.getBlockTriangularMueLuParams(cntxt_, schurApprox);
    Teuchos::ParameterList pivotParams = this->pivotMueLuParams();
    Teko::LinearOp pivotPrec =
      buildPivotBlockPrec<Node>(interface_, blocks_.J00, cntxt_, pivotParams);
    Teko::LinearOp schurPrec =
      buildSchurBlockPrec<Node>(interface_, schurApprox, cntxt_, schurMueLuParams);

    const Teko::TriSolveType triType = this->useUpperTriangular() ? Teko::GS_UseUpperTriangle
                                                                 : Teko::GS_UseLowerTriangle;
    return detail::composeTekoBlockOp(blocked, {pivotPrec, schurPrec},
      [triType](const Teuchos::RCP<Teko::BlockInvDiagonalStrategy> & s) {
        return Teuchos::rcp(new Teko::GaussSeidelPreconditionerFactory(triType, s));
      });
  }

private:
  void recordSchurAddonBeta(const matrix_rcp & schurCorr) const {
    RefMaxwellData<Node> & refMaxwell = cntxt_->refMaxwell;
    const size_t pivot = static_cast<size_t>(cntxt_->schur.pivot_block);
    // The Schur XML is read during the RefMaxwell build, so the first pass assumes the addon.
    const bool wanted = refMaxwell.schur_addon_wanted || !cntxt_->have_preconditioner;
    const bool haveInputs = !schurCorr.is_null() && !refMaxwell.nodal_lumped_mass.is_null() &&
      pivot < cntxt_->block.mass_matrices.size() &&
      !cntxt_->block.mass_matrices[pivot].is_null();
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
      blocks_, schurCorr, cntxt_->block.mass_matrices[pivot],
      cntxt_->schur.diag_use_lumped_pivot_diagonal,
      cntxt_->stage_alpha_u, interface_.verbosity);
    refMaxwell.schur_addon_beta_alpha_u = cntxt_->stage_alpha_u;
    refMaxwell.schur_addon_beta_valid = true;
  }

  // Never inherit the Schur list: it is tuned for curl-curl, not the mass pivot.
  Teuchos::ParameterList pivotMueLuParams() const {
    Teuchos::ParameterList params = defaultMueLuParams();
    if (cntxt_->pivot_block_sublist.isSublist("AMG Settings")) {
      const Teuchos::ParameterList & amg = cntxt_->pivot_block_sublist.sublist("AMG Settings");
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

  bool useUpperTriangular() const {
    const TriangleSide triangle = parseTriangleSide(cntxt_->schur.triangle);
    if (interface_.verbosity >= 5 && interface_.comm->getRank() == 0 &&
        cntxt_->schur.damping == Teuchos::ScalarTraits<ScalarT>::zero()) {
      std::cout << "Schur damping is 0; diagonal correction disabled." << std::endl;
    }
    return (triangle == TriangleSide::Auto) ? cntxt_->right_preconditioner
                                            : (triangle == TriangleSide::Upper);
  }

  LinearAlgebraInterface<Node> & interface_;
  matrix_rcp J_;
  Teuchos::RCP<LinearSolverContext<Node> > cntxt_;
  size_t set_;
  BlockSystem<Node> blocks_;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
