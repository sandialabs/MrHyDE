/***********************************************************************
 MrHyDE - Gauss-Seidel over groups of field splits instead of over single splits.

 The flat BlockTriangularFactory sweeps every split in sequence, so split k is
 corrected by all of 0..k-1. Grouping lets a set of splits be applied together as one
 sub-problem, with the sweep running between groups:

   split groups:
     bulk: 'v1, v2'          lower triangle
     target: 'p'               z_bulk = M_bulk^-1 r_bulk, block Jacobi over v1, v2
                               z_p    = S_p^-1 (r_p - A_p,bulk z_bulk)

 Teko::NestedBlockGS sweeps between groups; within a group the splits are applied by
 Jacobi or by Gauss-Seidel.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_HIERARCHICAL_SPLIT_FACTORY_HPP
#define MRHYDE_BLOCK_PREC_HIERARCHICAL_SPLIT_FACTORY_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/SchurInvDiagStrategy.hpp"
#include "block_prec/TekoAdapter.hpp"

#include <Teko_BlockPreconditionerFactory.hpp>
#include <Teko_GaussSeidelPreconditionerFactory.hpp>
#include <Teko_HierarchicalGaussSeidelPreconditionerFactory.hpp>
#include <Teko_JacobiPreconditionerFactory.hpp>

#include <map>
#include <vector>

namespace MrHyDE {
namespace block_prec {

template<class Node>
class HierarchicalSplitFactory : public Teko::BlockPreconditionerFactory {
public:
  using Types = BlockTypes<Node>;
  using matrix_rcp = typename Types::CrsMatrixRCP;

  HierarchicalSplitFactory(const matrix_rcp & J,
                           const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                           const int verbosity)
    : J_(J), cntxt_(cntxt), verbosity_(verbosity) {}

  Teko::LinearOp buildPreconditionerOperator(
      Teko::BlockedLinearOp & blocked,
      Teko::BlockPreconditionerState & state) const override {
    BlockSystem<Node> blocks = blockSystemFromBlockedOp<Node>(blocked);
    // The split inverses are the same ones the flat factory would use; only the
    // composition changes.
    SchurInvDiagStrategy<Node> splitStrategy(J_, blocks, cntxt_, verbosity_);
    std::vector<Teko::LinearOp> splitInverse;
    splitStrategy.getInvD(blocked, state, splitInverse);

    const std::vector<std::vector<size_t> > & groups = cntxt_->split_groups;
    TEUCHOS_TEST_FOR_EXCEPTION(groups.empty(), std::runtime_error,
      "'split groups' names no groups.");
    const bool upper = resolveUpperTriangle(cntxt_->schur.triangle,
                                            cntxt_->right_preconditioner);
    std::map<int, std::vector<int> > groupToRow;
    std::map<int, Teko::LinearOp> groupToInverse;
    for (size_t g = 0; g < groups.size(); ++g) {
      std::vector<int> rows;
      for (size_t m = 0; m < groups[g].size(); ++m) {
        rows.push_back(static_cast<int>(groups[g][m]));
      }
      groupToRow[static_cast<int>(g)] = rows;
      groupToInverse[static_cast<int>(g)] =
        this->groupInverse(blocks, groups[g], splitInverse, upper,
                           cntxt_->splitGroupIsJacobi(g));
    }
    const bool lower = !upper;
    if (verbosity_ >= 5 && J_->getComm()->getRank() == 0) {
      std::cout << "[BlockTri] " << groups.size() << " split groups, "
                << (lower ? "lower" : "upper") << " sweep between them" << std::endl;
      for (size_t g = 0; g < groups.size(); ++g) {
        std::cout << "[BlockTri]   group '" << cntxt_->split_group_names[g] << "' holds "
                  << groups[g].size() << " split" << (groups[g].size() == 1 ? "" : "s")
                  << ", "
                  << (cntxt_->splitGroupIsJacobi(g) ? "block diagonal over them"
                                                    : "pivot/target chain through them")
                  << std::endl;
      }
    }
    return Teuchos::rcp(new Teko::NestedBlockGS(groupToRow, groupToInverse, blocked, lower));
  }

private:
  // Gauss-Seidel keeps the pivot/target relation between a group's splits; Jacobi drops it,
  // making the group a block-diagonal sub-problem.
  Teko::LinearOp groupInverse(const BlockSystem<Node> & blocks,
                              const std::vector<size_t> & members,
                              const std::vector<Teko::LinearOp> & splitInverse,
                              const bool upper,
                              const bool jacobi) const {
    if (members.size() == 1) return splitInverse[members[0]];
    const size_t n = members.size();
    std::vector<std::vector<matrix_rcp> > sub(n, std::vector<matrix_rcp>(n, Teuchos::null));
    std::vector<Teko::LinearOp> invs;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) sub[i][j] = blocks.blocks[members[i]][members[j]];
      invs.push_back(splitInverse[members[i]]);
    }
    Teko::BlockedLinearOp groupOp = buildThyraBlockedFromSplits<Node>(sub);
    if (jacobi) {
      return detail::composeTekoBlockOp(groupOp, invs, detail::makeJacobiFactory);
    }
    const Teko::TriSolveType tri = upper ? Teko::GS_UseUpperTriangle
                                         : Teko::GS_UseLowerTriangle;
    return detail::composeTekoBlockOp(groupOp, invs,
      [tri](const Teuchos::RCP<Teko::BlockInvDiagonalStrategy> & s) {
        return Teuchos::rcp(new Teko::GaussSeidelPreconditionerFactory(tri, s));
      });
  }

  matrix_rcp J_;
  Teuchos::RCP<LinearSolverContext<Node> > cntxt_;
  int verbosity_;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
