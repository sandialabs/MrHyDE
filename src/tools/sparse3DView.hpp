/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_SPARSE3DVIEW_H
#define MRHYDE_SPARSE3DVIEW_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // New data structure to store sparse data
  // =================================================================
  
  class Sparse3DView {

  private:
    View_Sc3 values_;
    Kokkos::View<size_type***,AssemblyDevice> columns_, local_columns_;
    Kokkos::View<size_type**,AssemblyDevice> nnz_row_;
    ScalarT tol_;
    size_type maxent_;
    bool have_local_columns_ = false;

  public:
    Sparse3DView(View_Sc3 denseview, ScalarT & tol)
    : tol_(tol)
    {
      ScalarT maxval = 0.0;
      size_type numelem = denseview.extent(0);
      size_type numrows = denseview.extent(1);

      auto denseview_host = create_mirror_view(denseview);
      deep_copy(denseview_host,denseview);

      // Determine the maximum value for compression
      // Could have also used Kokkos::parallel_reduce
      for (size_type elem=0; elem<denseview_host.extent(0); elem++ ) {
        for (size_type i=0; i<denseview_host.extent(1); i++ ) {
          for (size_type j=0; j<denseview_host.extent(2); j++ ) {
            if (std::abs(denseview_host(elem,i,j))>maxval) {
              maxval = std::abs(denseview_host(elem,i,j));
            }
          }
        }
      }

      // Figure out how many entries will be retained per row
      nnz_row_ = Kokkos::View<size_type**,AssemblyDevice>("num nonzeros per elem/row",numelem,numrows);
      auto nnz_host = create_mirror_view(nnz_row_);
      maxent_ = 0;
      for (size_type elem=0; elem<denseview_host.extent(0); elem++ ) {
        for (size_type i=0; i<denseview_host.extent(1); i++ ) {
          size_type nnz = 0;
          for (size_type j=0; j<denseview_host.extent(2); j++ ) {
            if (std::abs(denseview_host(elem,i,j))/maxval>tol_) {
              nnz++;
            }
          }
          nnz_host(elem,i) = nnz;
          maxent_ = std::max(maxent_,nnz);
        }
      }
      deep_copy(nnz_host,nnz_row_);

      // Allocate and fill in values and columns
      values_ = View_Sc3("values",numelem,numrows,maxent_);
      columns_ = Kokkos::View<size_type***,AssemblyDevice>("columns",numelem,numrows,maxent_);
      local_columns_ = Kokkos::View<size_type***,AssemblyDevice>("columns",numelem,numrows,maxent_);

      auto values_host = create_mirror_view(values_);
      auto columns_host = create_mirror_view(columns_);

      for (size_type elem=0; elem<denseview_host.extent(0); elem++ ) {
        for (size_type i=0; i<denseview_host.extent(1); i++ ) {
          size_type prog = 0;
          for (size_type j=0; j<denseview_host.extent(2); j++ ) {
            if (std::abs(denseview_host(elem,i,j))/maxval>tol_) {
              columns_host(elem,i,prog) = j;
              values_host(elem,i,prog) = denseview_host(elem,i,j);
              ++prog;
            }
          }
        }
      }
      deep_copy(values_,values_host);
      deep_copy(columns_,columns_host);
    }
      
    KOKKOS_INLINE_FUNCTION    
    Sparse3DView() {};
    
    KOKKOS_INLINE_FUNCTION
    ~Sparse3DView() {};
    
    KOKKOS_INLINE_FUNCTION
    size_type size() const {
      auto nnz_host = create_mirror_view(nnz_row_);
      deep_copy(nnz_host,nnz_row_);
      size_type total = 0;
      for (size_type e=0; e<nnz_host.extent(0); ++e) {
        for (size_type j=0; j<nnz_host.extent(1); ++j) {
          total += nnz_host(e,j);
        }
      }
      return total;
    }

    KOKKOS_INLINE_FUNCTION
    View_Sc3 getValues() const {
      return values_;
    }
    
    KOKKOS_INLINE_FUNCTION
    bool getStatus() const {
      return have_local_columns_;
    }
    
    KOKKOS_INLINE_FUNCTION
    Kokkos::View<size_type***,AssemblyDevice> getColumns() const {
      return columns_;
    }
    
    KOKKOS_INLINE_FUNCTION
    Kokkos::View<size_type***,AssemblyDevice> getLocalColumns() const {
      return local_columns_;
    }
    
    void setLocalColumns(Kokkos::View<int**,AssemblyDevice> offsets, Kokkos::View<int*,AssemblyDevice> numDOF) {
      parallel_for("get mass",
                    RangePolicy<AssemblyExec>(0,columns_.extent(0)),
                    KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type var=0; var<numDOF.extent(0); var++) {
          for (int i=0; i<numDOF(var); i++ ) {
            LO localrow = offsets(var,i);
            for (size_type k=0; k<nnz_row_(elem,localrow); ++k ) {
              for (int j=0; j<numDOF(var); j++ ) {                    
                size_type localcol = offsets(var,j);
                if (columns_(elem,localrow,k) == localcol) {
                  local_columns_(elem,localrow,k) = j;
                }
              }
            }
          }
        }
      });   
      have_local_columns_ = true;  
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::View<size_type**,AssemblyDevice> getNNZPerRow() const {
      return nnz_row_;
    }
    
  };
  
}

#endif

