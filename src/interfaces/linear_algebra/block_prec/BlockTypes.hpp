/***********************************************************************
 MrHyDE - Tpetra aliases and option enums shared across block_prec/.

 Everything in block_prec/ is in MrHyDE::block_prec; ::detail is module-private.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_TYPES_HPP
#define MRHYDE_BLOCK_PREC_TYPES_HPP

#include "trilinos.hpp"
#include "preferences.hpp"

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

// Tpetra aliases
template<class Node>
struct BlockTypes {
  using Map = Tpetra::Map<LO,GO,Node>;
  using MapRCP = Teuchos::RCP<const Map>;
  using Vector = Tpetra::Vector<ScalarT,LO,GO,Node>;
  using IntVector = Tpetra::Vector<int,LO,GO,Node>;
  using MultiVector = Tpetra::MultiVector<ScalarT,LO,GO,Node>;
  using CrsMatrix = Tpetra::CrsMatrix<ScalarT,LO,GO,Node>;
  using CrsMatrixRCP = Teuchos::RCP<CrsMatrix>;
  using Import = Tpetra::Import<LO,GO,Node>;
  using HostInds = typename CrsMatrix::nonconst_local_inds_host_view_type;
  using HostVals = typename CrsMatrix::nonconst_values_host_view_type;
};

enum class SchurVariant { Base, Diag, Mass };

inline std::string schurVariantName(const SchurVariant variant) {
  if (variant == SchurVariant::Base) return "base";
  if (variant == SchurVariant::Diag) return "diag";
  if (variant == SchurVariant::Mass) return "mass";
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

// Comma-separated YAML list to trimmed, non-empty entries.
inline std::vector<std::string> splitCommaList(const std::string & spec) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= spec.size()) {
    const size_t comma = spec.find(',', pos);
    const std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos
                                                                        : comma - pos);
    const size_t b = item.find_first_not_of(" \t");
    const size_t e = item.find_last_not_of(" \t");
    if (b != std::string::npos) out.push_back(item.substr(b, e - b + 1));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

inline SchurVariant parseSchurVariant(const std::string & canonical) {
  const std::string up = toUpperAsciiCopy(canonical);
  if (up == "BASE") return SchurVariant::Base;
  if (up == "DIAG") return SchurVariant::Diag;
  if (up == "MASS") return SchurVariant::Mass;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported Schur approximation type '" << canonical
    << "'. Supported canonical values are: base, diag, mass.");
  return SchurVariant::Base;
}

enum class TriangleSide { Auto, Upper, Lower };

inline std::string triangleSideName(const TriangleSide side) {
  if (side == TriangleSide::Upper) return "upper";
  if (side == TriangleSide::Lower) return "lower";
  return "auto";
}

inline TriangleSide parseTriangleSide(const std::string & raw) {
  std::string u = raw.empty() ? std::string("AUTO") : raw;
  toUpperAscii(u);
  if (u == "AUTO") return TriangleSide::Auto;
  if (u == "UPPER") return TriangleSide::Upper;
  if (u == "LOWER") return TriangleSide::Lower;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported Schur triangle '" << raw << "'. Supported: auto, upper, lower.");
  return TriangleSide::Auto;
}

// 'auto' follows the Krylov side: a right preconditioner wants the upper triangle.
inline bool resolveUpperTriangle(const std::string & triangle, const bool rightPreconditioner) {
  const TriangleSide side = parseTriangleSide(triangle);
  return (side == TriangleSide::Auto) ? rightPreconditioner : (side == TriangleSide::Upper);
}

enum class BlockPrecType { AMG, RefMaxwell, Maxwell1, Direct, Diagonal };

inline BlockPrecType parseBlockPrecType(const std::string & raw) {
  std::string u = raw.empty() ? std::string("AMG") : raw;
  toUpperAscii(u);
  if (u == "AMG" || u == "MUELU") return BlockPrecType::AMG;
  if (u == "REFMAXWELL") return BlockPrecType::RefMaxwell;
  if (u == "MAXWELL1") return BlockPrecType::Maxwell1;
  if (u == "DIRECT") return BlockPrecType::Direct;
  if (u == "DIAG" || u == "DIAGONAL") return BlockPrecType::Diagonal;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported block preconditioner type '" << raw << "'. Supported: AMG, RefMaxwell, Maxwell1, Direct, Diagonal.");
  return BlockPrecType::AMG;
}

} // namespace block_prec
} // namespace MrHyDE

#endif
