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

// Block-triangular preconditioner: Teko inverts the triangle over role-ordered blocks.
template<class Node>
class BlockTriangularFactory : public Teko::BlockPreconditionerFactory {
public:
  using Types = BlockTypes<Node>;
  using matrix_rcp = typename Types::CrsMatrixRCP;

  BlockTriangularFactory(LinearAlgebraInterface<Node> & interface,
                         const matrix_rcp & J,
                         const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                         const size_t targetBlock)
    : interface_(interface), J_(J), cntxt_(cntxt), target_block_(targetBlock) {}

  // Teko calls this. targetBlock is the one thing not recoverable from the operator.
  Teko::LinearOp buildPreconditionerOperator(Teko::BlockedLinearOp & blocked,
                                             Teko::BlockPreconditionerState & /* state */) const override {
    BlockSystem<Node> blocks = blockSystemFromBlockedOp<Node>(blocked);
    blocks.targetBlock = target_block_;
    Teuchos::RCP<Teko::BlockInvDiagonalStrategy> strategy =
      Teuchos::rcp(new SchurInvDiagStrategy<Node>(interface_, J_, blocks, cntxt_));
    const Teko::TriSolveType triType = this->useUpperTriangular() ? Teko::GS_UseUpperTriangle
                                                                 : Teko::GS_UseLowerTriangle;
    return detail::tekoBuildInverse(
      Teuchos::rcp(new Teko::GaussSeidelPreconditionerFactory(triType, strategy)), blocked);
  }

private:
  bool useUpperTriangular() const {
    const TriangleSide triangle = parseTriangleSide(cntxt_->schur.triangle);
    const bool upper = (triangle == TriangleSide::Auto) ? cntxt_->right_preconditioner
                                                        : (triangle == TriangleSide::Upper);
    if (interface_.verbosity >= 5 && interface_.comm->getRank() == 0) {
      std::cout << "[BlockTri] triangle = " << (upper ? "upper" : "lower")
                << (triangle == TriangleSide::Auto ? " (auto)" : "") << std::endl;
    }
    return upper;
  }

  LinearAlgebraInterface<Node> & interface_;
  matrix_rcp J_;
  Teuchos::RCP<LinearSolverContext<Node> > cntxt_;
  size_t target_block_;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
