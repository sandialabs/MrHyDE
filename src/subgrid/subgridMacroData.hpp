/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
 ************************************************************************/

#ifndef MRHYDE_SUBGRIDMACRODATA_H
#define MRHYDE_SUBGRIDMACRODATA_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  class SubGridMacroData {
  public:
    
    SubGridMacroData() {} ;
    
    ~SubGridMacroData() {} ;
    
    SubGridMacroData(DRV & macronodes_, Kokkos::View<int****,HostDevice> & macrosideinfo_,
                     LIDView macroLIDs_,
                     Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & macroorientation_)
    : macronodes(macronodes_), macrosideinfo(macrosideinfo_), macroLIDs(macroLIDs_),
    macroorientation(macroorientation_) {
      
    }
    
    void setMacroIDs(const size_t & numElem) {
      
      macroIDs = Kokkos::View<int*,AssemblyDevice>("macro elem IDs",numElem);
      auto host_macroIDs = Kokkos::create_mirror_view(macroIDs);
      
      size_t numMacro = macronodes.extent(0);
      size_t numEperM = numElem/numMacro;
      size_t prog = 0;
      for (size_t i=0; i<numMacro; i++) {
        for (size_t j=0; j<numEperM; j++) {
          host_macroIDs(prog) = i;
          prog++;
        }
      }
      Kokkos::deep_copy(macroIDs, host_macroIDs);
      
    }
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    DRV macronodes;
    Kokkos::View<int****,HostDevice> macrosideinfo;
    LIDView macroLIDs;
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> macroorientation;
    Kokkos::View<string**,HostDevice> bcs;
    Kokkos::View<int*,AssemblyDevice> macroIDs;
  };
  
}

#endif

