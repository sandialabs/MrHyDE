/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_WELLS_H
#define MRHYDE_WELLS_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {

  /** @struct well
   * Contains the source name and type for each well defined in the
   * physics sublist
   */
  struct well {
    string name;
    string type;
  };

  /**
   * @brief A helper class to store Peaceman well sources and evalute them
   */
  
  template<class EvalT>
  class wells {

    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
      
  public:
    
    wells() {};
    
    wells(Teuchos::ParameterList & physSettings) {

      // get list of wells for the physics set from input file
      // and store their name and type
      if (physSettings.isSublist("Wells")) {
        Teuchos::ParameterList plWells = physSettings.sublist("Wells");
        Teuchos::ParameterList::ConstIterator itr = plWells.begin();

        // param list entry should be (well name) : (well type)
        // function list should contain a function of the same name
        // which specifies either the flow rate or the pressure 
        // throughout the domain

        // these can be localized by exponential kernels, etc.

        while(itr != plWells.end()) {
          well currwell;
          currwell.name = itr->first;
          currwell.type = plWells.get<string>(currwell.name);

          myWells.push_back(currwell);
          
          itr++;
        }
      }
      else {
        // TODO throw an error
      }

    }

    /**
     * @brief Add Peaceman well sources to the other volumetric sources
     * 
     * @param[in] source  Other volumetric sources, should be already evaluated by the function manager
     * @param[in] h  Element size
     * @param[in] functionManager The function manager with which to evaluate the sources
     * 
     * @returns  The finalized source term
     */

    Vista<EvalT> addWellSources(Vista<EvalT> & source, View_Sc1 & h, 
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager,
                         int numElem, int numIP) {

      auto Kinv_xx = functionManager->evaluate("Kinv_xx","ip");
      auto Kinv_yy = functionManager->evaluate("Kinv_yy","ip");
      auto Kinv_zz = functionManager->evaluate("Kinv_zz","ip");

      View_EvalT2 source_kv("KV for source", numElem, numIP);

      for (size_t wellnum=0; wellnum<myWells.size(); wellnum++) {

        auto wellfun = functionManager->evaluate(myWells[wellnum].name,"ip");

        parallel_for("update volumetric sources",
                     RangePolicy<AssemblyExec>(0,numElem),
                     KOKKOS_CLASS_LAMBDA (const int elem) {
          ScalarT C = std::log(0.25*std::exp(-0.5772)*h(elem)/2.0);

          for (int pt=0; pt<numIP; pt++) {
            if (wellnum == 0) source_kv(elem,pt) = source(elem,pt); // initialize properly
            // this allows for an additional volumetric source
            // do not name any of your wells the same as the default volumetric source

            // flow rate specified directly
            if (myWells[wellnum].type == "flow rate") {
              source_kv(elem,pt) += wellfun(elem,pt);
            }
            // need to compute the flow rate from the model
            else {
#ifndef MrHyDE_NO_AD
              EvalT Kxval = 1.0/Kinv_xx(elem,pt);
              EvalT Kyval = 1.0/Kinv_yy(elem,pt);
              EvalT Kzval = 1.0/Kinv_zz(elem,pt);
#else
              ScalarT Kxval = 1.0/Kinv_xx(elem,pt);
              ScalarT Kyval = 1.0/Kinv_yy(elem,pt);
              ScalarT Kzval = 1.0/Kinv_zz(elem,pt);
#endif
              // TODO revisit this, especially for 3-D
              EvalT Kval = sqrt(Kxval*Kxval + Kyval*Kyval + Kzval*Kzval);
              source_kv(elem,pt) += 2.0*PI/C*Kval*wellfun(elem,pt);
            }
          }
        });
      }

      return Vista<EvalT>(source_kv);

    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
  private:
    std::vector<well> myWells;

  };
  
}

template class MrHyDE::wells<ScalarT>;

#ifndef MrHyDE_NO_AD
template class MrHyDE::wells<AD>;
template class MrHyDE::wells<AD2>;
template class MrHyDE::wells<AD4>;
template class MrHyDE::wells<AD8>;
template class MrHyDE::wells<AD16>;
template class MrHyDE::wells<AD18>;
template class MrHyDE::wells<AD24>;
template class MrHyDE::wells<AD32>;
#endif

#endif
