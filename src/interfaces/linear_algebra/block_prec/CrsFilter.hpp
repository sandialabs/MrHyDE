/***********************************************************************
 MrHyDE - One filtered-copy kernel for the row and entry filters the block
 preconditioners need.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_CRS_FILTER_HPP
#define MRHYDE_BLOCK_PREC_CRS_FILTER_HPP

#include "block_prec/BlockTypes.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace MrHyDE {
namespace block_prec {
namespace detail {

// rowSource(i) gives the source row for output row i, entryMap(srcRow, col, value) the output
// column; either returns -1 to drop, and entryMap may rewrite value.
template<class Node, class RowSource, class EntryMap>
typename BlockTypes<Node>::CrsMatrixRCP
filterCopyCrs(const typename BlockTypes<Node>::CrsMatrix & src,
              const typename BlockTypes<Node>::MapRCP & rowMap,
              const typename BlockTypes<Node>::MapRCP & colMap,
              const typename BlockTypes<Node>::MapRCP & domainMap,
              const typename BlockTypes<Node>::MapRCP & rangeMap,
              const RowSource rowSource,
              const EntryMap entryMap) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using LocalMatrix  = typename LA_CrsMatrix::local_matrix_device_type;
  using exec_space   = typename Node::execution_space;
  using RowPtr    = typename LocalMatrix::row_map_type::non_const_type;
  using Entries   = typename LocalMatrix::index_type::non_const_type;
  using Values    = typename LocalMatrix::values_type::non_const_type;
  using size_type = typename LocalMatrix::size_type;

  const LocalMatrix lcl = src.getLocalMatrixDevice();
  const LO nrows = static_cast<LO>(rowMap->getLocalNumElements());
  RowPtr rowptr(Kokkos::view_alloc("crs_filter_rowptr", Kokkos::WithoutInitializing),
                static_cast<size_t>(nrows) + 1);

  size_type nnz = 0;
  Kokkos::parallel_scan("filterCopyCrs::size",
    Kokkos::RangePolicy<exec_space, LO>(0, nrows),
    KOKKOS_LAMBDA(const LO i, size_type & offset, const bool finalPass) {
      if (finalPass) rowptr(i) = offset;
      const LO srcRow = rowSource(i);
      if (srcRow >= 0) {
        const auto row = lcl.row(srcRow);
        for (LO k = 0; k < row.length; ++k) {
          ScalarT value = row.value(k);
          if (entryMap(srcRow, row.colidx(k), value) >= 0) ++offset;
        }
      }
    }, nnz);
  Kokkos::deep_copy(Kokkos::subview(rowptr, nrows), nnz);

  Entries entries(Kokkos::view_alloc("crs_filter_entries", Kokkos::WithoutInitializing), nnz);
  Values values(Kokkos::view_alloc("crs_filter_values", Kokkos::WithoutInitializing), nnz);
  Kokkos::parallel_for("filterCopyCrs::fill",
    Kokkos::RangePolicy<exec_space, LO>(0, nrows),
    KOKKOS_LAMBDA(const LO i) {
      const LO srcRow = rowSource(i);
      if (srcRow < 0) return;
      size_type at = rowptr(i);
      const auto row = lcl.row(srcRow);
      for (LO k = 0; k < row.length; ++k) {
        ScalarT value = row.value(k);
        const LO outCol = entryMap(srcRow, row.colidx(k), value);
        if (outCol < 0) continue;
        entries(at) = outCol;
        values(at) = value;
        ++at;
      }
      // Tpetra's constructor below needs sorted columns, and entryMap may permute them.
      for (size_type p = rowptr(i) + 1; p < at; ++p) {
        const LO col = entries(p);
        const ScalarT val = values(p);
        size_type q = p;
        while (q > rowptr(i) && entries(q - 1) > col) {
          entries(q) = entries(q - 1);
          values(q) = values(q - 1);
          --q;
        }
        entries(q) = col;
        values(q) = val;
      }
    });

  const LocalMatrix out("crs_filter", nrows,
                        static_cast<LO>(colMap->getLocalNumElements()), nnz,
                        values, rowptr, entries);
  return Teuchos::rcp(new LA_CrsMatrix(out, rowMap, colMap, domainMap, rangeMap));
}

struct KeepAllRows {
  KOKKOS_INLINE_FUNCTION LO operator()(const LO i) const { return i; }
};

struct KeepAllEntries {
  KOKKOS_INLINE_FUNCTION LO operator()(const LO, const LO col, ScalarT &) const { return col; }
};

template<class Node>
struct KeepUnflaggedRows {
  Kokkos::View<const bool*, typename Node::device_type::memory_space> flagged;
  KOKKOS_INLINE_FUNCTION LO operator()(const LO i) const { return flagged(i) ? -1 : i; }
};

// Output row i reads source row srcRow(i); -1 means the source has no such row.
template<class Node>
struct RowsFromTable {
  Kokkos::View<const LO*, typename Node::device_type> srcRow;
  KOKKOS_INLINE_FUNCTION LO operator()(const LO i) const { return srcRow(i); }
};

// Source column LID to output column LID; -1 drops the entry.
template<class Node>
struct ColumnsFromTable {
  Kokkos::View<const LO*, typename Node::device_type> outCol;
  KOKKOS_INLINE_FUNCTION LO operator()(const LO, const LO col, ScalarT &) const {
    return outCol(col);
  }
};

// Snaps every surviving entry to its sign and drops what is numerically zero.
struct SnapEntrySigns {
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  MagT tol;
  KOKKOS_INLINE_FUNCTION LO operator()(const LO, const LO col, ScalarT & value) const {
    const ScalarT v = value;
    const MagT m = (v < ScalarT(0)) ? static_cast<MagT>(-v) : static_cast<MagT>(v);
    if (m <= tol) return -1;
    value = (v > ScalarT(0)) ? ScalarT(1) : ScalarT(-1);
    return col;
  }
};

// Drops off-diagonal |A(i,j)| < tol*sqrt(|A(i,i)|*|A(j,j)|). The diagonal always stays.
template<class Node>
struct DropSmallOffDiagonals {
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  using LocalMap = typename Tpetra::Map<LO,GO,Node>::local_map_type;
  Kokkos::View<const ScalarT*, typename Node::device_type> rowDiag, colDiag;
  LocalMap rowMap, colMap;
  MagT tol;

  KOKKOS_INLINE_FUNCTION static MagT mag(const ScalarT v) {
    return (v < ScalarT(0)) ? static_cast<MagT>(-v) : static_cast<MagT>(v);
  }

  KOKKOS_INLINE_FUNCTION LO operator()(const LO row, const LO col, ScalarT & value) const {
    if (colMap.getGlobalElement(col) == rowMap.getGlobalElement(row)) return col;
    const bool keep = mag(value) >= tol * Kokkos::sqrt(mag(rowDiag(row)) * mag(colDiag(col)));
    return keep ? col : LO(-1);
  }
};

} // namespace detail
} // namespace block_prec
} // namespace MrHyDE

#endif
