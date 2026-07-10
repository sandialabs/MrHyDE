#ifndef MRHYDE_BLOCK_PREC_TYPES_HPP
#define MRHYDE_BLOCK_PREC_TYPES_HPP

#include "trilinos.hpp"
#include "preferences.hpp"

#include <Ifpack2_Preconditioner.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_TestForException.hpp>

#include <cctype>
#include <stdexcept>
#include <string>

namespace MrHyDE {
namespace block_prec {

// BlockTypes centralizes Tpetra aliases used across block-preconditioner code.
// Includes Map/Vector/MultiVector/CrsMatrix/Operator aliases plus import/export
// and host row-view types reused by extraction, assembly, and apply helpers.
template<class Node>
struct BlockTypes {
  using Map = Tpetra::Map<LO,GO,Node>;
  using MapRCP = Teuchos::RCP<const Map>;
  using Vector = Tpetra::Vector<ScalarT,LO,GO,Node>;
  using IntVector = Tpetra::Vector<int,LO,GO,Node>;
  using IntVectorRCP = Teuchos::RCP<IntVector>;
  using MultiVector = Tpetra::MultiVector<ScalarT,LO,GO,Node>;
  using MultiVecRCP = Teuchos::RCP<MultiVector>;
  using CrsMatrix = Tpetra::CrsMatrix<ScalarT,LO,GO,Node>;
  using CrsMatrixRCP = Teuchos::RCP<CrsMatrix>;
  using Operator = Tpetra::Operator<ScalarT,LO,GO,Node>;
  using Import = Tpetra::Import<LO,GO,Node>;
  using Export = Tpetra::Export<LO,GO,Node>;
  using ImportRCP = Teuchos::RCP<Import>;
  using ExportRCP = Teuchos::RCP<Export>;
  using Preconditioner = Ifpack2::Preconditioner<ScalarT,LO,GO,Node>;
  using MueLuOperator = MueLu::TpetraOperator<ScalarT,LO,GO,Node>;
  using HostInds = typename CrsMatrix::nonconst_local_inds_host_view_type;
  using HostVals = typename CrsMatrix::nonconst_values_host_view_type;
};

} // namespace block_prec

// Enums and string utilities below are at MrHyDE scope (not block_prec)
// because LinearSolverContext and ParamUtils reference them directly.
enum class SchurVariant { Base, Diag };

inline std::string schurVariantName(const SchurVariant variant) {
  if (variant == SchurVariant::Base) return "base";
  if (variant == SchurVariant::Diag) return "diag";
  return "base";
}

inline void toUpperAscii(std::string & value) {
  for (size_t i = 0; i < value.size(); ++i) {
    value[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(value[i])));
  }
}

inline std::string toUpperAsciiCopy(std::string value) {
  toUpperAscii(value);
  return value;
}

inline SchurVariant parseSchurVariant(const std::string & canonical, const std::string & fallbackType) {
  std::string raw = canonical.empty() ? fallbackType : canonical;
  std::string forError = raw;
  toUpperAscii(raw);
  if (raw == "BASE") return SchurVariant::Base;
  if (raw == "DIAG") return SchurVariant::Diag;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported Schur approximation type '" << forError
    << "'. Supported canonical values are: base, diag.");
  return SchurVariant::Base;
}

enum class BlockPrecType { AMG, RefMaxwell, Direct, Diagonal };

inline BlockPrecType parseBlockPrecType(const std::string & raw) {
  std::string u = raw.empty() ? std::string("AMG") : raw;
  toUpperAscii(u);
  if (u == "AMG" || u == "MUELU") return BlockPrecType::AMG;
  if (u == "REFMAXWELL") return BlockPrecType::RefMaxwell;
  if (u == "DIRECT") return BlockPrecType::Direct;
  if (u == "DIAG" || u == "DIAGONAL") return BlockPrecType::Diagonal;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported block preconditioner type '" << raw << "'. Supported: AMG, RefMaxwell, Direct, Diagonal.");
  return BlockPrecType::AMG;
}
} // namespace MrHyDE

#endif
