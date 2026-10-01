/***********************************************************************
 MrHyDE - One approximate inverse per block split, for both block schemes.

 buildBlockOperator dispatches on the split's 'preconditioner' key; the
 block-diagonal helpers below merge the deck's split sublist first.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_SPLIT_INVERSE_HPP
#define MRHYDE_BLOCK_PREC_SPLIT_INVERSE_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/MaxwellInverse.hpp"
#include "block_prec/ParamUtils.hpp"
#include "block_prec/TekoAdapter.hpp"
#include "linearSolverContext.hpp"

#include <BelosLinearProblem.hpp>
#include <BelosTpetraOperator.hpp>
#include <Ifpack2_Factory.hpp>

namespace MrHyDE {
namespace block_prec {

// Applies diag(M)^-1. Thyra::diagonal would do the same through RTOps, which zero-fill
// Y and drop to a serial loop; elementWiseMultiply is one fused kernel.
template<class Node>
class DiagonalInverseOperator : public Tpetra::Operator<ScalarT,LO,GO,Node> {
public:
  using LA_Map = typename BlockTypes<Node>::Map;
  using LA_MultiVector = typename BlockTypes<Node>::MultiVector;
  using LA_Vector = typename BlockTypes<Node>::Vector;

  explicit DiagonalInverseOperator(const Teuchos::RCP<LA_Vector> & invDiag) : invDiag_(invDiag) {}

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return invDiag_->getMap(); }
  Teuchos::RCP<const LA_Map> getRangeMap() const override { return invDiag_->getMap(); }
  bool hasTransposeApply() const override { return true; }

  void apply(const LA_MultiVector & X, LA_MultiVector & Y,
             Teuchos::ETransp = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    // Symmetric, so the transpose mode needs no special case.
    Y.elementWiseMultiply(alpha, *invDiag_, X, beta);
  }

private:
  Teuchos::RCP<LA_Vector> invDiag_;
};

template<class Node>
Teko::LinearOp
buildDiagonalBlockInverse(const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                          const bool useLumpedDiagonal,
                          const Teuchos::RCP<const Teuchos::Comm<int> > & comm,
                          const int verbosity) {
  using Types = BlockTypes<Node>;
  detail::InverseDiagonalCounts counts;
  Teuchos::RCP<typename Types::Vector> invDiag =
    detail::buildInverseDiagonal<Node>(
      Teuchos::rcp_implicit_cast<const typename Types::CrsMatrix>(J00), useLumpedDiagonal, counts);
  detail::reportInverseDiagonal<Node>(counts, "Pivot-block diag inverse", comm, verbosity);
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > op =
    Teuchos::rcp(new DiagonalInverseOperator<Node>(invDiag));
  return tpetraToThyra<Node>(op, J00->getRangeMap(), J00->getDomainMap());
}

template<class Node>
Teko::LinearOp
buildDirectBlockInverse(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                        const typename BlockTypes<Node>::CrsMatrixRCP & A,
                        const std::string & label,
                        const int verbosity) {
  const std::string & solverName = cntxt->amesos_type;
  Teuchos::ParameterList entry;
  entry.set("Solver Type", solverName);
  // Block row maps are extracted from the monolithic map, so they are not contiguous.
  entry.sublist("Amesos2 Settings").sublist(solverName).set("IsContiguous", false);
  return cntxt->inverseLibrary(verbosity, A->getComm()->getRank()).build(
    "Amesos2", entry, label + " Amesos2", A);
}


template<class Node>
Teko::LinearOp
maybeWrapInInnerKrylov(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                       const typename BlockTypes<Node>::CrsMatrixRCP & blockMat,
                       const Teko::LinearOp & innerPrec,
                       const Teuchos::ParameterList & blockList,
                       const std::string & label,
                       const int verbosity) {
  if (!blockList.isParameter("inner krylov solver")) return innerPrec;
  // Inner Krylov gives a different operator on every apply, so the outer solver
  // has to be right-preconditioned flexible GMRES.
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt.is_null() ||
                             toUpperAsciiCopy(cntxt->belos_type) != "BLOCK GMRES" ||
                             !cntxt->flexible_gmres || !cntxt->right_preconditioner,
    std::runtime_error,
    "[" << label << "] 'inner krylov solver' requires Block GMRES with "
    "'Flexible Gmres: true' and 'right preconditioner: true'; deck has '"
    << (cntxt.is_null() ? std::string("none") : cntxt->belos_type) << "'.");
  const std::string innerSolver = blockList.get<std::string>("inner krylov solver");
  const int innerMaxIters = blockList.isParameter("inner krylov max iters")
    ? blockList.get<int>("inner krylov max iters") : 5;
  const double innerTol = blockList.isParameter("inner krylov tol")
    ? blockList.get<double>("inner krylov tol") : 1.0e-2;
  Teuchos::ParameterList belosList;
  belosList.set("Solver Type", innerSolver);
  Teuchos::ParameterList & solverList =
    belosList.sublist("Solver Types").sublist(innerSolver);
  solverList.set("Maximum Iterations", innerMaxIters);
  solverList.set("Num Blocks", innerMaxIters);
  solverList.set("Convergence Tolerance", innerTol);
  solverList.set("Verbosity", static_cast<int>(Belos::Errors));
  solverList.set("Output Frequency", 0);
  solverList.set("Output Style", static_cast<int>(Belos::Brief));
  solverList.set("Implicit Residual Scaling", std::string("Norm of Initial Residual"));
  if (verbosity >= 10 && blockMat->getComm()->getRank() == 0) {
    std::cout << "[" << label << "] wrapping block preconditioner in inner Belos '"
              << innerSolver << "' (max iters=" << innerMaxIters
              << ", tol=" << innerTol << ")" << std::endl;
  }
  return cntxt->inverseLibrary(verbosity, blockMat->getComm()->getRank()).build(
    "Belos", belosList, label + " inner", blockMat, innerPrec);
}

template<class Node, class GenericFn>
Teko::LinearOp
buildBlockOperator(const typename BlockTypes<Node>::CrsMatrixRCP & mat,
                   const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                   const Teuchos::ParameterList & blockList,
                   const BlockPrecType type,
                   const size_t split,
                   const std::string & label,
                   const int verbosity,
                   GenericFn && buildGeneric) {
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > tpetraPrec;
  Teko::LinearOp innerPrec;
  switch (type) {
    case BlockPrecType::RefMaxwell:
      validateRefMaxwellBlockInputs<Node>(mat, cntxt, verbosity);
      tpetraPrec = buildRefMaxwellPreconditioner<Node>(mat, cntxt, split, verbosity);
      break;
    case BlockPrecType::Maxwell1:
      validateRefMaxwellBlockInputs<Node>(mat, cntxt, verbosity);
      tpetraPrec = buildMaxwell1Preconditioner<Node>(mat, cntxt, split, verbosity);
      break;
    case BlockPrecType::Direct:
      innerPrec = buildDirectBlockInverse<Node>(cntxt, mat, label, verbosity);
      break;
    case BlockPrecType::Diagonal:
      // S is formed from J00, so inverting its diagonal is not an approximation of it.
      TEUCHOS_TEST_FOR_EXCEPTION(split != 0, std::runtime_error,
        "Only the leading field split supports Diagonal.");
      innerPrec = buildDiagonalBlockInverse<Node>(
        mat, cntxt->schur.diagonal_prec_use_lumped, mat->getComm(), verbosity);
      break;
    default:
      innerPrec = buildGeneric();
      break;
  }
  if (innerPrec.is_null()) {
    innerPrec = tpetraToThyra<Node>(tpetraPrec, mat->getRangeMap(), mat->getDomainMap());
  }
  return maybeWrapInInnerKrylov<Node>(cntxt, mat, innerPrec, blockList, label, verbosity);
}

namespace detail {

template<class Node>
Teuchos::ParameterList mergeBlockSettings(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                          const size_t blockIndex) {
  Teuchos::ParameterList list;
  list.set("relaxation: type", "Jacobi");
  if (cntxt != Teuchos::null && cntxt->prec_sublist.name() != "empty") {
    list.setParameters(cntxt->prec_sublist);
  }
  if (cntxt != Teuchos::null) {
    list.setParameters(cntxt->splitSettings(blockIndex));
  }
  return list;
}

inline void ensureRelaxationDampingDouble(Teuchos::ParameterList & list) {
  if (!list.isParameter("relaxation: damping factor")) return;
  const Teuchos::ParameterEntry & e = list.getEntry("relaxation: damping factor");
  if (e.isType<double>()) return;
  const double val = e.isType<int>() ? static_cast<double>(list.get<int>("relaxation: damping factor")) : 1.0;
  list.remove("relaxation: damping factor", false);
  list.set("relaxation: damping factor", val);
}

inline std::string resolveBlockMethod(Teuchos::ParameterList & blockList) {
  // 'preconditioner' is the unified key; 'preconditioner variant' is the older spelling.
  std::string method = blockList.isParameter("preconditioner")
    ? blockList.get<std::string>("preconditioner")
    : blockList.get<std::string>("preconditioner variant", "RELAXATION");
  if (toUpperAsciiCopy(method) != "AMG" &&
      blockList.isParameter("smoother: type") &&
      toUpperAsciiCopy(blockList.get<std::string>("smoother: type")) == "CHEBYSHEV") {
    method = "Chebyshev";
  }
  const std::string methodUpper = toUpperAsciiCopy(method);
  if (methodUpper == "CHEBYSHEV" && !blockList.isParameter("chebyshev: degree") &&
      !(blockList.isSublist("smoother: params") &&
        blockList.sublist("smoother: params").isParameter("chebyshev: degree"))) {
    blockList.set("chebyshev: degree", 2);
  }
  return method;
}

inline bool mueluParamsWantCoordinates(const Teuchos::ParameterList & pl) {
  const auto contains = [&](const std::string & key) {
    return pl.isParameter(key) &&
           pl.get<std::string>(key).find("distance laplacian") != std::string::npos;
  };
  return contains("aggregation: drop scheme") ||
         contains("aggregation: strength-of-connection: matrix");
}

template<class Node>
Teko::LinearOp
buildAmgBlockOperator(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                      const typename BlockTypes<Node>::CrsMatrixRCP & blockMat,
                      const Teuchos::ParameterList & blockList,
                      const size_t blockIndex,
                      const Teuchos::RCP<Tpetra::MultiVector<
                          typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node> > & dofCoords,
                      const typename BlockTypes<Node>::CrsMatrixRCP & D0_matrix,
                      const int verbosity) {
  Teuchos::ParameterList mueluList;

  if (!loadMueLuXmlIfPresent(blockList, mueluList, "block-diag AMG", blockMat->getComm())) {
    mueluList = defaultMueLuParams();
    applyDeckMueLuOverrides(mueluList, blockList);
  }

  if (mueluParamsWantCoordinates(mueluList)) {
    TEUCHOS_TEST_FOR_EXCEPTION(dofCoords.is_null(), std::runtime_error,
      "MueLu params request distance-laplacian aggregation but no per-DOF coordinates were "
      "supplied for this block. Set 'hgrad basis name' and 'hcurl basis name' in this "
      "split's sublist so setupBlockTriangularAuxiliary can build the coords.");
    TEUCHOS_TEST_FOR_EXCEPTION(!dofCoords->getMap()->isSameAs(*blockMat->getRowMap()), std::runtime_error,
      "Per-DOF coordinate MultiVector map does not match block matrix row map "
      "(coords length=" << dofCoords->getGlobalLength() << ", block rows=" << blockMat->getGlobalNumRows() << ").");
    mueluList.sublist("user data").set("Coordinates", dofCoords);
  }

  addHiptmairUserData<Node>(mueluList, blockMat, D0_matrix, "the split's settings", verbosity);

  return cntxt->inverseLibrary(verbosity, blockMat->getComm()->getRank()).build(
    "MueLu", mueluList, "BlockDiag block " + std::to_string(blockIndex) + " MueLu", blockMat);
}

template<class Node>
Teko::LinearOp
buildIfpack2BlockOperator(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                          const typename BlockTypes<Node>::CrsMatrixRCP & blockMat,
                          const Teuchos::ParameterList & blockListIn,
                          const std::string & method,
                          const size_t blockIndex,
                          const int verbosity) {
  Teuchos::ParameterList blockList(blockListIn);
  if (verbosity >= 15 && blockMat->getComm()->getRank() == 0) {
    std::cout << "Preconditioner parameters (block diagonal, block " << blockIndex
              << ", method " << method << "):" << std::endl;
    blockList.print(std::cout);
  }

  const std::string methodUpper = toUpperAsciiCopy(method);
  Teuchos::ParameterList ifpackList(blockList);
  removeMrHyDEOwnedKeys(ifpackList);
  if (methodUpper == "CHEBYSHEV") {
    ifpackList.remove("smoother: type", false);
    promoteSublistToTopLevel(ifpackList, "smoother: params");
  }
  else {
    ifpackList.remove("smoother: type", false);
    ifpackList.remove("smoother: params", false);
    ensureRelaxationDampingDouble(ifpackList);
  }

  Teuchos::ParameterList entry;
  entry.set("Prec Type", method);
  entry.sublist("Ifpack2 Settings").setParameters(ifpackList);
  return cntxt->inverseLibrary(verbosity, blockMat->getComm()->getRank()).build(
    "Ifpack2", entry, "BlockDiag block " + std::to_string(blockIndex) + " Ifpack2", blockMat);
}

template<class Node>
Teko::LinearOp
buildSingleBlockPreconditioner(const typename BlockTypes<Node>::CrsMatrixRCP & blockMat,
                               const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                               const size_t blockIndex,
                               const bool useRefMaxwellOnBlock0,
                               const int verbosity) {
  const std::string label = "BlockDiag block " + std::to_string(blockIndex);
  if (blockIndex == 0 && useRefMaxwellOnBlock0) {
    return buildBlockOperator<Node>(blockMat, cntxt, cntxt->splitSettings(0),
      BlockPrecType::RefMaxwell, 0, label, verbosity,
      [] { return Teko::LinearOp(); });
  }

  Teuchos::ParameterList blockList = mergeBlockSettings<Node>(cntxt, blockIndex);

  // 'use mass matrix' swaps the Jacobian block for its assembled mass (M1 HCURL / M2 HDIV).
  typename BlockTypes<Node>::CrsMatrixRCP preconditioner_matrix = blockMat;
  const bool useMassMatrix = blockList.isParameter("use mass matrix") &&
                             blockList.get<bool>("use mass matrix");
  if (useMassMatrix) {
    // Prefer the index; fall back to matching by map, the only option for a fused split.
    typename BlockTypes<Node>::CrsMatrixRCP mass = Teuchos::null;
    if (cntxt != Teuchos::null) {
      if (blockIndex < cntxt->block.mass_matrices.size() &&
          !cntxt->block.mass_matrices[blockIndex].is_null() &&
          cntxt->block.mass_matrices[blockIndex]->getRowMap()->isSameAs(*blockMat->getRowMap())) {
        mass = cntxt->block.mass_matrices[blockIndex];
      }
      else {
        mass = block_prec::massMatrixOnMap<Node>(cntxt->block.mass_matrices,
                                                blockMat->getRowMap());
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(mass.is_null(), std::runtime_error,
      "'use mass matrix: true' on split " << blockIndex << " but no assembled mass matrix "
      "lives on that split's map; a fused split has none.");
    preconditioner_matrix = mass;
    if (verbosity >= 10 && blockMat->getComm()->getRank() == 0) {
      std::cout << "[BlockDiag] Block " << blockIndex
                << ": substituting mass matrix for extracted Jacobian block" << std::endl;
    }
  }

  // Block-diagonal blocks pick AMG or an Ifpack2 smoother by name, so they all
  // take buildBlockOperator's generic branch.
  const std::string method = resolveBlockMethod(blockList);
  return buildBlockOperator<Node>(preconditioner_matrix, cntxt, blockList,
    BlockPrecType::AMG, blockIndex, label, verbosity,
    [&] () -> Teko::LinearOp {
      if (toUpperAsciiCopy(method) != "AMG") {
        return buildIfpack2BlockOperator<Node>(cntxt, preconditioner_matrix, blockList,
                                               method, blockIndex, verbosity);
      }
      if (verbosity >= 15 && preconditioner_matrix->getComm()->getRank() == 0) {
        std::cout << "Preconditioner parameters (block diagonal, block " << blockIndex
                  << ", method AMG):" << std::endl;
        blockList.print(std::cout);
      }
      typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
      Teuchos::RCP<Tpetra::MultiVector<CoordScalar,LO,GO,Node> > dofCoords;
      if (!cntxt.is_null() && blockIndex < cntxt->block.dof_coords.size()) {
        dofCoords = cntxt->block.dof_coords[blockIndex];
      }
      // Associate D0 with the HCURL block by matching its range map.
      return buildAmgBlockOperator<Node>(cntxt, preconditioner_matrix, blockList, blockIndex,
                                         dofCoords, splitD0<Node>(cntxt, preconditioner_matrix),
                                         verbosity);
    });
}

} // namespace detail
} // namespace block_prec
} // namespace MrHyDE

#endif
