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
#include "preferences.hpp"


class KokkosTools {
public:
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T*,AssemblyDevice> V) {
    cout << endl;
    cout << "Printing data for View: " << V.label() << endl;
    
    cout << "  i  " << "  value  " << endl;
    cout << "--------------------" << endl;
    
    for (int i=0; i<V.dimension(0); i++) {
      for (int j=0; j<V.dimension(1); j++) {
        cout << "  " << i << "  " << "  " << "  " << V(i,j) << "  " << endl;
      }
    }
    cout << "--------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T**,AssemblyDevice> V) {
    cout << endl;
    cout << "Printing data for View: " << V.label() << endl;
    
    cout << "  i  " << "  j  " << "  value  " << endl;
    cout << "-------------------------------" << endl;
    
    for (int i=0; i<V.dimension(0); i++) {
      for (int j=0; j<V.dimension(1); j++) {
        cout << "  " << i << "  " << "  " << j << "  " <<
        "  " << "  " << V(i,j) << "  " << endl;
      }
    }
    cout << "-------------------------------" << endl;
    
  }
  
  ///Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice>
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(FDATA V) {
    cout << endl;
    cout << "Printing data for View: " << V.label() << endl;
    
    cout << "  i  " << "  j  " << "  value  " << endl;
    cout << "-------------------------------" << endl;
    
    for (int i=0; i<V.dimension(0); i++) {
      for (int j=0; j<V.dimension(1); j++) {
        cout << "  " << i << "  " << "  " << j << "  " <<
        "  " << "  " << V(i,j) << "  " << endl;
      }
    }
    cout << "-------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T***,AssemblyDevice> V) {
    cout << endl;
    cout << "Printing data for View: " << V.label() << endl;
    
    cout << "  i  " << "  j  " << "  k  " << "  value  " << endl;
    cout << "------------------------------------------" << endl;
    
    for (int i=0; i<V.dimension(0); i++) {
      for (int j=0; j<V.dimension(1); j++) {
        for (int k=0; k<V.dimension(2); k++) {
          cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << k << "  " << "  " << V(i,j,k) << "  " << endl;
        }
      }
    }
    cout << "------------------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T****,AssemblyDevice> V) {
    cout << endl;
    cout << "Printing data for View: " << V.label() << endl;
    cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << endl;
    cout << "-----------------------------------------------------" << endl;
    
    for (int i=0; i<V.dimension(0); i++) {
      for (int j=0; j<V.dimension(1); j++) {
        for (int k=0; k<V.dimension(2); k++) {
          for (int n=0; n<V.dimension(3); n++) {
            cout << "  " << i << "  " << "  " << j << "  " <<
            "  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << endl;
          }
        }
      }
    }
    cout << "-----------------------------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T*****,AssemblyDevice> V) {
    cout << endl;
    cout << "Printing data for View: " << V.label() << endl;
    cout << "  i  " << "  j  " << "  k  " << "  n  " << "  m  " << "  value  " << endl;
    cout << "----------------------------------------------------------------" << endl;
    
    for (int i=0; i<V.dimension(0); i++) {
      for (int j=0; j<V.dimension(1); j++) {
        for (int k=0; k<V.dimension(2); k++) {
          for (int n=0; n<V.dimension(3); n++) {
            for (int m=0; m<V.dimension(4); m++) {
              cout << "  " << i << "  " << "  " << j << "  " <<
              "  " << k << "  " << "  " << n << "  " << "  " << m
              << "  " << "  " << V(i,j,k,n,m) << "  " << endl;
            }
          }
        }
      }
    }
    cout << "----------------------------------------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(DRV V) {
    cout << endl;
    cout << "Printing data for DynRankView: " << V.label() << endl;
  
    if (V.rank() == 2) {
      cout << "  i  " << "  j  " << "  value  " << endl;
      cout << "-------------------------------" << endl;
      
      for (int i=0; i<V.dimension(0); i++) {
        for (int j=0; j<V.dimension(1); j++) {
          cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << "  " << V(i,j) << "  " << endl;
        }
      }
      cout << "-------------------------------" << endl;
      
    }
    else if (V.rank() == 3) {
      cout << "  i  " << "  j  " << "  k  " << "  value  " << endl;
      cout << "------------------------------------------" << endl;
      
      for (int i=0; i<V.dimension(0); i++) {
        for (int j=0; j<V.dimension(1); j++) {
          for (int k=0; k<V.dimension(2); k++) {
            cout << "  " << i << "  " << "  " << j << "  " <<
            "  " << k << "  " << "  " << V(i,j,k) << "  " << endl;
          }
        }
      }
      cout << "------------------------------------------" << endl;
      
    }
    else if (V.rank() == 4) {
      cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << endl;
      cout << "-----------------------------------------------------" << endl;
      
      for (int i=0; i<V.dimension(0); i++) {
        for (int j=0; j<V.dimension(1); j++) {
          for (int k=0; k<V.dimension(2); k++) {
            for (int n=0; n<V.dimension(3); n++) {
              cout << "  " << i << "  " << "  " << j << "  " <<
              "  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << endl;
            }
          }
        }
      }
      cout << "-----------------------------------------------------" << endl;
      
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(DRVint V) {
    cout << endl;
    cout << "Printing data for DynRankView: " << V.label() << endl;
    
    if (V.rank() == 2) {
      cout << "  i  " << "  j  " << "  value  " << endl;
      cout << "-------------------------------" << endl;
      
      for (int i=0; i<V.dimension(0); i++) {
        for (int j=0; j<V.dimension(1); j++) {
          cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << "  " << V(i,j) << "  " << endl;
        }
      }
      cout << "-------------------------------" << endl;
      
    }
    else if (V.rank() == 3) {
      cout << "  i  " << "  j  " << "  k  " << "  value  " << endl;
      cout << "------------------------------------------" << endl;
      
      for (int i=0; i<V.dimension(0); i++) {
        for (int j=0; j<V.dimension(1); j++) {
          for (int k=0; k<V.dimension(2); k++) {
            cout << "  " << i << "  " << "  " << j << "  " <<
            "  " << k << "  " << "  " << V(i,j,k) << "  " << endl;
          }
        }
      }
      cout << "------------------------------------------" << endl;
      
    }
    else if (V.rank() == 4) {
      cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << endl;
      cout << "-----------------------------------------------------" << endl;
      
      for (int i=0; i<V.dimension(0); i++) {
        for (int j=0; j<V.dimension(1); j++) {
          for (int k=0; k<V.dimension(2); k++) {
            for (int n=0; n<V.dimension(3); n++) {
              cout << "  " << i << "  " << "  " << j << "  " <<
              "  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << endl;
            }
          }
        }
      }
      cout << "-----------------------------------------------------" << endl;
      
    }
  }


};
#endif

