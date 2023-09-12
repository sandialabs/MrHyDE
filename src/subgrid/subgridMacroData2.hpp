/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_SUBGRIDMACRODATA2_H
#define MRHYDE_SUBGRIDMACRODATA2_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  class SubGridMacroData2 {
  public:
    
    SubGridMacroData2() {} ;
    
    ~SubGridMacroData2() {} ;
    
    SubGridMacroData2(DRV & macronodes_,
                     LIDView macroLIDs_,
                     Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & macroorientation_)
    : macronodes(macronodes_), macroLIDs(macroLIDs_),
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
    LIDView macroLIDs;
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> macroorientation;
    //Kokkos::View<string**,HostDevice> bcs;
    Kokkos::View<int*,AssemblyDevice> macroIDs;
  };
  
}

#endif

