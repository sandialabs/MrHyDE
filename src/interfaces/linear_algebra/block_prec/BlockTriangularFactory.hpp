/***********************************************************************
 MrHyDE - Assembly of the block-triangular preconditioner.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_TRIANGULAR_FACTORY_HPP
#define MRHYDE_BLOCK_PREC_TRIANGULAR_FACTORY_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/SchurInvDiagStrategy.hpp"
#include "block_prec/TekoAdapter.hpp"

#include <Teko_BlockPreconditionerFactory.hpp>
#include <Teko_GaussSeidelPreconditionerFactory.hpp>
#include <Thyra_DefaultIdentityLinearOp.hpp>
#include <Thyra_DefaultMultipliedLinearOp.hpp>
#include <Thyra_DefaultScaledAdjointLinearOp.hpp>

namespace MrHyDE {
namespace block_prec {

// Block-triangular preconditioner over the field splits in 'variable groups' order; Teko
// composes the triangle except under exact-curl.
template<class Node>
class BlockTriangularFactory : public Teko::BlockPreconditionerFactory {
public:
  using Types = BlockTypes<Node>;
  using matrix_rcp = typename Types::CrsMatrixRCP;

  BlockTriangularFactory(const matrix_rcp & J,
                         const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                         const int verbosity)
    : J_(J), cntxt_(cntxt), verbosity_(verbosity) {}

  Teko::LinearOp buildPreconditionerOperator(Teko::BlockedLinearOp & blocked,
                                             Teko::BlockPreconditionerState & state) const override {
    BlockSystem<Node> blocks = blockSystemFromBlockedOp<Node>(blocked);
    const Teko::TriSolveType triType = this->useUpperTriangular() ? Teko::GS_UseUpperTriangle
                                                                 : Teko::GS_UseLowerTriangle;
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    const ScalarT s =
      (parseSchurVariant(cntxt_->schur.approximation_type) == SchurVariant::ExactCurl)
      ? exactCurlScale<Node>(blocks, cntxt_->block.curl, verbosity_) : zero;
    Teuchos::RCP<SchurInvDiagStrategy<Node> > strategy =
      Teuchos::rcp(new SchurInvDiagStrategy<Node>(J_, blocks, cntxt_, verbosity_, s));
    if (s != zero) {
      std::vector<Teko::LinearOp> invDiag;
      strategy->getInvD(blocked, state, invDiag);
      return this->exactCurlInverse(s, invDiag);
    }
    return detail::tekoBuildInverse(
      Teuchos::rcp(new Teko::GaussSeidelPreconditionerFactory(triType, strategy)), blocked);
  }

private:
  // [I, -s C; 0, I] diag(P_0^-1, P_1^-1): back-substitutes with J00^-1 J01 = s C exactly.
  Teko::LinearOp exactCurlInverse(const ScalarT s,
                                  const std::vector<Teko::LinearOp> & invDiag) const {
    const Teko::LinearOp U = Thyra::block2x2<ScalarT>(
      Thyra::identity<ScalarT>(invDiag[0]->range()),
      Thyra::scale<ScalarT>(-s, tpetraToThyraConst<Node>(cntxt_->block.curl)),
      Teuchos::null, Thyra::identity<ScalarT>(invDiag[1]->range()));
    const Teko::LinearOp D = Thyra::block2x2<ScalarT>(invDiag[0], Teuchos::null,
                                                      Teuchos::null, invDiag[1]);
    return Thyra::multiply<ScalarT>(U, D);
  }

  bool useUpperTriangular() const {
    const bool upper = resolveUpperTriangle(cntxt_->schur.triangle,
                                           cntxt_->right_preconditioner);
    if (verbosity_ >= 5 && J_->getComm()->getRank() == 0) {
      std::cout << "[BlockTri] triangle = " << (upper ? "upper" : "lower")
                << (parseTriangleSide(cntxt_->schur.triangle) == TriangleSide::Auto
                    ? " (auto)" : "") << std::endl;
    }
    return upper;
  }

  matrix_rcp J_;
  Teuchos::RCP<LinearSolverContext<Node> > cntxt_;
  int verbosity_;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
