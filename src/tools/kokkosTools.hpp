/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef KKTOOLS_H
#define KKTOOLS_H

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

//typedef Kokkos::DynRankView<ScalarT,AssemblyDevice> DRV;
//typedef Kokkos::DynRankView<int,AssemblyDevice> DRVint;

class KokkosTools {
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
    
    parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
      //printf("   %i      %f\n",i,V(i));
      //std::cout << "  " << i << "  " << "  " << "  " << V(i) << "  " << std::endl;
    });
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
    
    for (int i=0; i<V.size(); i++) {
      //printf("   %i      %f\n",i,V[i]);
      //std::cout << "  " << i << "  " << "  " << "  " << V[i] << "  " << std::endl;
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
    
    parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
      for (size_t j=0; j<V.extent(1); j++) {
        //printf("   %i      %i      %f\n", i, j, V(i,j));
        std::cout << "  " << i << "  " << "  " << j << "  " <<
        "  " << "  " << V(i,j) << "  " << std::endl;
      }
    });
    std::cout << "-------------------------------" << std::endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // The following can only be called on the host
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(Teuchos::RCP<MpiComm> & Comm, vector_RCP & V, const string & message="") {
    auto V_kv = V->getLocalView<HostDevice>();
    
    std::cout << std::endl;
    std::cout << message << std::endl;
    std::cout << "Printing data for View: " << V_kv.label() << std::endl;
    
    std::cout << " PID " << "  i  " << "  j  " << "  value  " << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    for (unsigned int i=0; i<V_kv.extent(0); i++) {
      for (unsigned int j=0; j<V_kv.extent(1); j++) {
        //printf("   %i      %i      %f\n", i, j, V_kv(i,j));
        //std::cout << "  " << Comm->getRank() <<  "  " << i << "  " << "  " << j << "  " <<
        //"  " << "  " << V_kv(i,j) << "  " << std::endl;
      }
    }
    std::cout << "------------------------------------------" << std::endl;
    
  }
  
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
  
  static void print(FDATA V, const string & message="") {
    std::cout << std::endl;
    std::cout << message << std::endl;
    std::cout << "Printing data for View: " << V.label() << std::endl;
    
    std::cout << "  i  " << "  j  " << "  value  " << std::endl;
    std::cout << "-------------------------------" << std::endl;
    
    parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
      for (unsigned int j=0; j<V.extent(1); j++) {
        //printf("   %i      %i      %f\n", i, j, V(i,j));
        //std::cout << "  " << i << "  " << "  " << j << "  " << // GH: std::std::cout is illegal in device
        //"  " << "  " << V(i,j) << "  " << std::endl;
      }
    });
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
    
    parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
      for (unsigned int j=0; j<V.extent(1); j++) {
        for (unsigned int k=0; k<V.extent(2); k++) {
      //    printf("   %i      %i      %i      %f\n", i, j, k, V(i,j,k));
          std::cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << k << "  " << "  " << V(i,j,k) << "  " << std::endl;
        }
      }
    });
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
    
    parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
      
      for (unsigned int j=0; j<V.extent(1); j++) {
        for (unsigned int k=0; k<V.extent(2); k++) {
          for (unsigned int n=0; n<V.extent(3); n++) {
            //printf("   %i      %i      %i      %i      %f\n", i, j, k, n, V(i,j,k,n));
            //std::cout << "  " << i << "  " << "  " << j << "  " <<
            //"  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << std::endl;
          }
        }
      }
    });
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
    
    parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
      for (unsigned int j=0; j<V.extent(1); j++) {
        for (unsigned int k=0; k<V.extent(2); k++) {
          for (unsigned int n=0; n<V.extent(3); n++) {
            for (unsigned int m=0; m<V.extent(4); m++) {
              //printf("   %i      %i      %i      %i      %i      %f\n", i, j, k, n, m, V(i,j,k,n,m));
              //std::cout << "  " << i << "  " << "  " << j << "  " <<
              //"  " << k << "  " << "  " << n << "  " << "  " << m
              //<< "  " << "  " << V(i,j,k,n,m) << "  " << std::endl;
            }
          }
        }
      }
    });
    std::cout << "----------------------------------------------------------------" << std::endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(DRV V, const string & message="") {
    std::cout << std::endl;
    std::cout << message << std::endl;
    std::cout << "Printing data for DynRankView: " << V.label() << std::endl;
  
    if (V.rank() == 1) {
      std::cout << "  i  " << "  value  " << std::endl;
      std::cout << "-------------------------------" << std::endl;
      
      parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
        //printf("   %i      %f\n", i, V(i));
        //std::cout << "  " << i << "  " << // GH: std::std::cout is illegal in device
        //"  " << "  " << V(i) << "  " << std::endl;
      });
      std::cout << "-------------------------------" << std::endl;
      
    }
    else if (V.rank() == 2) {
      std::cout << "  i  " << "  j  " << "  value  " << std::endl;
      std::cout << "-------------------------------" << std::endl;
      
      parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
        for (unsigned int j=0; j<V.extent(1); j++) {
          //printf("   %i      %i      %f\n", i, j, V(i,j));
          //std::cout << "  " << i << "  " << "  " << j << "  " << // GH: std::std::cout is illegal in device
          //"  " << "  " << V(i,j) << "  " << std::endl;
        }
      });
      std::cout << "-------------------------------" << std::endl;
      
    }
    else if (V.rank() == 3) {
      std::cout << "  i  " << "  j  " << "  k  " << "  value  " << std::endl;
      std::cout << "------------------------------------------" << std::endl;
      
      parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
        for (unsigned int j=0; j<V.extent(1); j++) {
          for (unsigned int k=0; k<V.extent(2); k++) {
            //printf("   %i      %i      %i      %f\n", i, j, k, V(i,j,k));
            //std::cout << "  " << i << "  " << "  " << j << "  " << // GH: std::std::cout is illegal in device
            //"  " << k << "  " << "  " << V(i,j,k) << "  " << std::endl;
          }
        }
      });
      std::cout << "------------------------------------------" << std::endl;
      
    }
    else if (V.rank() == 4) {
      std::cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << std::endl;
      std::cout << "-----------------------------------------------------" << std::endl;
      
      parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
        for (unsigned int j=0; j<V.extent(1); j++) {
          for (unsigned int k=0; k<V.extent(2); k++) {
            for (unsigned int n=0; n<V.extent(3); n++) {
              //printf("   %i      %i      %i      %i      %f\n", i, j, k, n, V(i,j,k,n));
              //std::cout << "  " << i << "  " << "  " << j << "  " << // GH: std::std::cout is illegal in device
              //"  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << std::endl;
            }
          }
        }
      });
      std::cout << "-----------------------------------------------------" << std::endl;
      
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(DRVint V, const string & message="") {
    std::cout << std::endl;
    std::cout << message << std::endl;
    std::cout << "Printing data for DynRankView: " << V.label() << std::endl;
    
    if (V.rank() == 2) {
      std::cout << "  i  " << "  j  " << "  value  " << std::endl;
      std::cout << "-------------------------------" << std::endl;
      
      parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
        for (unsigned int j=0; j<V.extent(1); j++) {
          //printf("   %i      %i      %i\n", i, j, V(i,j));
          //std::cout << "  " << i << "  " << "  " << j << "  " << // GH: std::std::cout is illegal in device
          //"  " << "  " << V(i,j) << "  " << std::endl;
        }
      });
      std::cout << "-------------------------------" << std::endl;
      
    }
    else if (V.rank() == 3) {
      std::cout << "  i  " << "  j  " << "  k  " << "  value  " << std::endl;
      std::cout << "------------------------------------------" << std::endl;
      
      parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
        for (unsigned int j=0; j<V.extent(1); j++) {
          for (unsigned int k=0; k<V.extent(2); k++) {
            //printf("   %i      %i      %i      %i\n", i, j, k, V(i,j,k));
            //std::cout << "  " << i << "  " << "  " << j << "  " << // GH: std::std::cout is illegal in device
            //"  " << k << "  " << "  " << V(i,j,k) << "  " << std::endl;
          }
        }
      });
      std::cout << "------------------------------------------" << std::endl;
      
    }
    else if (V.rank() == 4) {
      std::cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << std::endl;
      std::cout << "-----------------------------------------------------" << std::endl;
      
      parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int i ) {
        for (unsigned int j=0; j<V.extent(1); j++) {
          for (unsigned int k=0; k<V.extent(2); k++) {
            for (unsigned int n=0; n<V.extent(3); n++) {
              //printf("   %i      %i      %i      %i      %i\n", i, j, k, n, V(i,j,k,n));
              //std::cout << "  " << i << "  " << "  " << j << "  " << // GH: std::std::cout is illegal in device
              //"  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << std::endl;
            }
          }
        }
      });
      std::cout << "-----------------------------------------------------" << std::endl;
      
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
  
  static void checkSizes(DRV V, vector<int> & sizes, const string message="") {
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
  
  static void checkSizes(DRVint V, vector<int> & sizes, const string message="") {
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
  
  static void reset(Kokkos::View<AD****, AssemblyDevice> & V, AD & value) {
    parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<V.extent(1); k++) {
        for (int i=0; i<V.extent(2); i++) {
          for (int s=0; s<V.extent(3); s++) {
            V(e,k,i,s) = value;
          }
        }
      }
    });
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void reset(Kokkos::View<AD***, AssemblyDevice> & V, AD & value) {
    parallel_for(RangePolicy<AssemblyExec>(0,V.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<V.extent(1); k++) {
        for (int i=0; i<V.extent(2); i++) {
          V(e,k,i) = value;
        }
      }
    });
  }
};
#endif

