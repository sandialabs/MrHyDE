/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_KOKKOS_TOOLS_H
#define MRHYDE_KOKKOS_TOOLS_H

#include "trilinos.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "Sacado.hpp"
#include "Shards_CellTopology.hpp"
#include "Intrepid2_Utils.hpp"

#include "Kokkos_Core.hpp"
#include "preferences.hpp"
#include "Teuchos_FancyOStream.hpp"

#include "vista.hpp"

namespace MrHyDE {
  
  class KokkosTools {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SolverNode>   LA_CrsMatrix;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector> vector_RCP;
    typedef Teuchos::RCP<LA_CrsMatrix>   matrix_RCP;
    
  public:
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    template<class T>
    static void print(Kokkos::View<T*,AssemblyDevice> V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for View: " << V.label() << std::endl;
      
      std::cout << "  i  " << "  value  " << std::endl;
      std::cout << "--------------------" << std::endl;
      
      auto V_host = Kokkos::create_mirror_view(V);
      Kokkos::deep_copy(V_host,V);
      
      for (size_type i=0; i<V_host.extent(0); i++) {
        //printf("   %i      %f\n",i,V(i));
        std::cout << "  " << i << "  " << "  " << "  " << V_host(i) << "  " << std::endl;
      }
      std::cout << "--------------------" << std::endl;
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    template<class T>
    static void print(const std::vector<T> & V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for std::vector: " << std::endl;
      
      std::cout << "  i  " << "  value  " << std::endl;
      std::cout << "--------------------" << std::endl;
      
      for (size_t i=0; i<V.size(); i++) {
        //printf("   %i      %f\n",i,V[i]);
        std::cout << "  " << i << "  " << "  " << "  " << V[i] << "  " << std::endl;
      }
      std::cout << "--------------------" << std::endl;
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    template<class T>
    static void print(Kokkos::View<T**,AssemblyDevice> V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for View: " << V.label() << std::endl;
      
      std::cout << "  i  " << "  j  " << "  value  " << std::endl;
      std::cout << "-------------------------------" << std::endl;
      
      auto V_host = Kokkos::create_mirror_view(V);
      Kokkos::deep_copy(V_host,V);
      
      for (size_type i=0; i<V_host.extent(0); i++) {
        for (size_type j=0; j<V_host.extent(1); j++) {
          //printf("   %i      %i      %f\n", i, j, V(i,j));
          std::cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << "  " << V_host(i,j) << "  " << std::endl;
        }
      }
      std::cout << "-------------------------------" << std::endl;
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // The following can only be called on the host
    ////////////////////////////////////////////////////////////////////////////////
    /*
    static void print(Teuchos::RCP<MpiComm> & Comm, vector_RCP & V, const string & message="") {
      auto V_kv = V->getLocalView<HostDevice>(Tpetra::Access::ReadWrite);
      
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for View: " << V_kv.label() << std::endl;
      
      std::cout << " PID " << "  i  " << "  j  " << "  value  " << std::endl;
      std::cout << "------------------------------------------" << std::endl;
      
      for (size_type i=0; i<V_kv.extent(0); i++) {
        for (size_type j=0; j<V_kv.extent(1); j++) {
          //printf("   %i      %i      %f\n", i, j, V_kv(i,j));
          std::cout << "  " << Comm->getRank() <<  "  " << i << "  " << "  " << j << "  " <<
          "  " << "  " << V_kv(i,j) << "  " << std::endl;
        }
      }
      std::cout << "------------------------------------------" << std::endl;
      
    }*/
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    static void print(matrix_RCP & M, const string & message="") {
      std::cout << message << std::endl;
      Teuchos::EVerbosityLevel vl = Teuchos::VERB_EXTREME;
      auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
      M->describe(*out,vl);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    static void print(const vector_RCP & V, const string & message="") {
      std::cout << message << std::endl;
      Teuchos::EVerbosityLevel vl = Teuchos::VERB_EXTREME;
      auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
      V->describe(*out,vl);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /*
    static void print(View_AD2 V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for View: " << V.label() << std::endl;
      
      std::cout << "  i  " << "  j  " << "  value  " << std::endl;
      std::cout << "-------------------------------" << std::endl;
      
      auto V_host = Kokkos::create_mirror_view(V);
      Kokkos::deep_copy(V_host,V);
      
      for (size_type i=0; i<V_host.extent(0); i++) {
        for (size_type j=0; j<V_host.extent(1); j++) {
      //    //printf("   %i      %i      %f\n", i, j, V(i,j));
          std::cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << "  " << V(i,j) << "  " << std::endl;
        }
      }
      std::cout << "-------------------------------" << std::endl;
      
    }
    */
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    static void print(Vista V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      if (V.isView) {
        if (V.isAD) {
          std::cout << "Printing data for View: " << V.viewdata.label() << std::endl;
        }
        else {
          std::cout << "Printing data for View: " << V.viewdata_Sc.label() << std::endl;
        }
      }
      
      
      std::cout << "  i  " << "  j  " << "  value  " << std::endl;
      std::cout << "-------------------------------" << std::endl;
            
      size_type ext0 = 1, ext1 = 1;
      if (V.isView) {
        if (V.isAD) {
          ext0 = V.viewdata.extent(0);
          ext1 = V.viewdata.extent(1);
        }
        else {
          ext0 = V.viewdata_Sc.extent(0);
          ext1 = V.viewdata_Sc.extent(1);
        }
      }
      for (size_type i=0; i<ext0; i++) {
        for (size_type j=0; j<ext1; j++) {
      //    //printf("   %i      %i      %f\n", i, j, V(i,j));
          std::cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << "  " << V(i,j) << "  " << std::endl;
        }
      }
      std::cout << "-------------------------------" << std::endl;
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    template<class T>
    static void print(Kokkos::View<T***,AssemblyDevice> V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for View: " << V.label() << std::endl;
      
      std::cout << "  i  " << "  j  " << "  k  " << "  value  " << std::endl;
      std::cout << "------------------------------------------" << std::endl;
      
      auto V_host = Kokkos::create_mirror_view(V);
      Kokkos::deep_copy(V_host,V);
      
      for (size_type i=0; i<V_host.extent(0); i++) {
        for (size_type j=0; j<V_host.extent(1); j++) {
          for (size_type k=0; k<V_host.extent(2); k++) {
            //    printf("   %i      %i      %i      %f\n", i, j, k, V(i,j,k));
            std::cout << "  " << i << "  " << "  " << j << "  " <<
            "  " << k << "  " << "  " << V_host(i,j,k) << "  " << std::endl;
          }
        }
      }
      std::cout << "------------------------------------------" << std::endl;
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    template<class T>
    static void print(Kokkos::View<T****,AssemblyDevice> V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for View: " << V.label() << std::endl;
      std::cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << std::endl;
      std::cout << "-----------------------------------------------------" << std::endl;
      
      auto V_host = Kokkos::create_mirror_view(V);
      Kokkos::deep_copy(V_host,V);
      
      for (size_type i=0; i<V_host.extent(0); i++) {
        for (size_type j=0; j<V_host.extent(1); j++) {
          for (size_type k=0; k<V_host.extent(2); k++) {
            for (size_type n=0; n<V_host.extent(3); n++) {
              //printf("   %i      %i      %i      %i      %f\n", i, j, k, n, V(i,j,k,n));
              std::cout << "  " << i << "  " << "  " << j << "  " <<
              "  " << k << "  " << "  " << n << "  " << "  " << V_host(i,j,k,n) << "  " << std::endl;
            }
          }
        }
      }
      std::cout << "-----------------------------------------------------" << std::endl;
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    template<class T>
    static void print(Kokkos::View<T*****,AssemblyDevice> V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for View: " << V.label() << std::endl;
      std::cout << "  i  " << "  j  " << "  k  " << "  n  " << "  m  " << "  value  " << std::endl;
      std::cout << "----------------------------------------------------------------" << std::endl;
      
      auto V_host = Kokkos::create_mirror_view(V);
      Kokkos::deep_copy(V_host,V);
      
      for (size_type i=0; i<V_host.extent(0); i++) {
        for (size_type j=0; j<V_host.extent(1); j++) {
          for (size_type k=0; k<V_host.extent(2); k++) {
            for (size_type n=0; n<V_host.extent(3); n++) {
              for (size_type m=0; m<V_host.extent(4); m++) {
                //printf("   %i      %i      %i      %i      %i      %f\n", i, j, k, n, m, V(i,j,k,n,m));
                std::cout << "  " << i << "  " << "  " << j << "  " <<
                "  " << k << "  " << "  " << n << "  " << "  " << m
                << "  " << "  " << V_host(i,j,k,n,m) << "  " << std::endl;
              }
            }
          }
        }
      }
      std::cout << "----------------------------------------------------------------" << std::endl;
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    static void print(DRV V, const string & message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      std::cout << "Printing data for DynRankView: " << V.label() << std::endl;
      
      auto V_host = Kokkos::create_mirror_view(V);
      Kokkos::deep_copy(V_host,V);
      
      if (V_host.rank() == 1) {
        std::cout << "  i  " << "  value  " << std::endl;
        std::cout << "-------------------------------" << std::endl;
        
        for (size_type i=0; i<V_host.extent(0); i++) {
          //printf("   %i      %f\n", i, V(i));
          std::cout << "  " << i << "  " <<
          "  " << "  " << V_host(i) << "  " << std::endl;
        }
        std::cout << "-------------------------------" << std::endl;
        
      }
      else if (V_host.rank() == 2) {
        std::cout << "  i  " << "  j  " << "  value  " << std::endl;
        std::cout << "-------------------------------" << std::endl;
        
        for (size_type i=0; i<V_host.extent(0); i++) {
          for (size_type j=0; j<V_host.extent(1); j++) {
            //printf("   %i      %i      %f\n", i, j, V(i,j));
            std::cout << "  " << i << "  " << "  " << j << "  " <<
            "  " << "  " << V_host(i,j) << "  " << std::endl;
          }
        }
        std::cout << "-------------------------------" << std::endl;
        
      }
      else if (V_host.rank() == 3) {
        std::cout << "  i  " << "  j  " << "  k  " << "  value  " << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        
        for (size_type i=0; i<V_host.extent(0); i++) {
          for (size_type j=0; j<V_host.extent(1); j++) {
            for (size_type k=0; k<V_host.extent(2); k++) {
              //printf("   %i      %i      %i      %f\n", i, j, k, V(i,j,k));
              std::cout << "  " << i << "  " << "  " << j << "  " <<
              "  " << k << "  " << "  " << V_host(i,j,k) << "  " << std::endl;
            }
          }
        }
        std::cout << "------------------------------------------" << std::endl;
        
      }
      else if (V_host.rank() == 4) {
        std::cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl;
        
        for (size_type i=0; i<V_host.extent(0); i++) {
          for (size_type j=0; j<V_host.extent(1); j++) {
            for (size_type k=0; k<V_host.extent(2); k++) {
              for (size_type n=0; n<V_host.extent(3); n++) {
                //printf("   %i      %i      %i      %i      %f\n", i, j, k, n, V(i,j,k,n));
                std::cout << "  " << i << "  " << "  " << j << "  " <<
                "  " << k << "  " << "  " << n << "  " << "  " << V_host(i,j,k,n) << "  " << std::endl;
              }
            }
          }
        }
        std::cout << "-----------------------------------------------------" << std::endl;
        
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void printExtents(Kokkos::View<T*,AssemblyDevice> V, const string message = "") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      for (size_type k=0; k<V.rank(); k++) {
        std::cout << "extent(" << k << ") = " << V.extent(k) << std::endl;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void printExtents(Kokkos::View<T**,AssemblyDevice> V, const string message = "") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      for (size_type k=0; k<V.rank(); k++) {
        std::cout << "extent(" << k << ") = " << V.extent(k) << std::endl;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void printExtents(Kokkos::View<T***,AssemblyDevice> V, const string message = "") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      for (size_type k=0; k<V.rank(); k++) {
        std::cout << "extent(" << k << ") = " << V.extent(k) << std::endl;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void printExtents(Kokkos::View<T****,AssemblyDevice> V, const string message = "") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      for (size_type k=0; k<V.rank(); k++) {
        std::cout << "extent(" << k << ") = " << V.extent(k) << std::endl;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void checkSizes(Kokkos::View<T*,AssemblyDevice> V, vector<int> & sizes, const string message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      if (V.rank() != sizes.size()) {
        std::cout << "ERROR ---" << std::endl;
        std::cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << std::endl;
        std::cout << "---------" << std::endl;
      }
      else {
        for (size_type k=0; k<V.rank(); k++) {
          if (V.extent(k) != sizes[k]) {
            std::cout << "ERROR ---" << std::endl;
            std::cout << "Size of dimension(" << k << ") = " << V.extent(k) << "    Expected size = " << sizes[k] << std::endl;
            std::cout << "---------" << std::endl;
          }
        }
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void checkSizes(Kokkos::View<T**,AssemblyDevice> V, vector<int> & sizes, const string message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      if (V.rank() != sizes.size()) {
        std::cout << "ERROR ---" << std::endl;
        std::cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << std::endl;
        std::cout << "---------" << std::endl;
      }
      else {
        for (size_t k=0; k<V.rank(); k++) {
          if (V.extent(k) != sizes[k]) {
            std::cout << "ERROR ---" << std::endl;
            std::cout << "Size of dimension(" << k << ") = " << V.extent(k) << "    Expected size = " << sizes[k] << std::endl;
            std::cout << "---------" << std::endl;
          }
        }
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void checkSizes(Kokkos::View<T***,AssemblyDevice> V, vector<int> & sizes, const string message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      if (V.rank() != sizes.size()) {
        std::cout << "ERROR ---" << std::endl;
        std::cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << std::endl;
        std::cout << "---------" << std::endl;
      }
      else {
        for (size_t k=0; k<V.rank(); k++) {
          if (V.extent(k) != sizes[k]) {
            std::cout << "ERROR ---" << std::endl;
            std::cout << "Size of dimension(" << k << ") = " << V.extent(k) << "    Expected size = " << sizes[k] << std::endl;
            std::cout << "---------" << std::endl;
          }
        }
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void checkSizes(Kokkos::View<T****,AssemblyDevice> V, vector<int> & sizes, const string message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      if (V.rank() != sizes.size()) {
        std::cout << "ERROR ---" << std::endl;
        std::cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << std::endl;
        std::cout << "---------" << std::endl;
      }
      else {
        for (size_t k=0; k<V.rank(); k++) {
          if (V.extent(k) != sizes[k]) {
            std::cout << "ERROR ---" << std::endl;
            std::cout << "Size of dimension(" << k << ") = " << V.extent(k) << "    Expected size = " << sizes[k] << std::endl;
            std::cout << "---------" << std::endl;
          }
        }
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static void checkSizes(Kokkos::View<T*****,AssemblyDevice> V, vector<int> & sizes, const string message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      if (V.rank() != sizes.size()) {
        std::cout << "ERROR ---" << std::endl;
        std::cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << std::endl;
        std::cout << "---------" << std::endl;
      }
      else {
        for (size_t k=0; k<V.rank(); k++) {
          if (V.extent(k) != sizes[k]) {
            std::cout << "ERROR ---" << std::endl;
            std::cout << "Size of dimension(" << k << ") = " << V.extent(k) << "    Expected size = " << sizes[k] << std::endl;
            std::cout << "---------" << std::endl;
          }
        }
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    static void checkSizes(DRV V, vector<size_type> & sizes, const string message="") {
      std::cout << std::endl;
      std::cout << message << std::endl;
      
      if (V.rank() != sizes.size()) {
        std::cout << "ERROR ---" << std::endl;
        std::cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << std::endl;
        std::cout << "---------" << std::endl;
      }
      else {
        for (size_t k=0; k<V.rank(); k++) {
          if (V.extent(k) != sizes[k]) {
            std::cout << "ERROR ---" << std::endl;
            std::cout << "Size of dimension(" << k << ") = " << V.extent(k) << "    Expected size = " << sizes[k] << std::endl;
            std::cout << "---------" << std::endl;
          }
        }
      }
    }
        
  };
  
}

#endif

