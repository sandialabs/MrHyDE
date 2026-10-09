/***********************************************************************
 MrHyDE - Assembly of the block-triangular preconditioner.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_TRIANGULAR_FACTORY_HPP
#define MRHYDE_BLOCK_PREC_TRIANGULAR_FACTORY_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/SchurInvDiagStrategy.hpp"
#include "block_prec/TekoAdapter.hpp"

#include <Teko_BlockPreconditionerFactory.hpp>
#include <Teko_GaussSeidelPreconditionerFactory.hpp>

namespace MrHyDE {
namespace block_prec {

// Block-triangular preconditioner: Teko inverts the triangle over the field splits in
// the order 'variable groups' declares them.
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
                                             Teko::BlockPreconditionerState & /* state */) const override {
    BlockSystem<Node> blocks = blockSystemFromBlockedOp<Node>(blocked);
    Teuchos::RCP<Teko::BlockInvDiagonalStrategy> strategy =
      Teuchos::rcp(new SchurInvDiagStrategy<Node>(J_, blocks, cntxt_, verbosity_));
    const Teko::TriSolveType triType = this->useUpperTriangular() ? Teko::GS_UseUpperTriangle
                                                                 : Teko::GS_UseLowerTriangle;
    return detail::tekoBuildInverse(
      Teuchos::rcp(new Teko::GaussSeidelPreconditionerFactory(triType, strategy)), blocked);
  }

private:
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
