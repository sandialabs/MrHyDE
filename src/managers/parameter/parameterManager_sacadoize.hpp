/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::sacadoizeParams(const bool & seed_active) {
  
  if (paramvals.size() > 0) {
    if (paramvals[0].size() > 0) {
      
      size_t maxlength = paramvals_KV.extent(1);
      
      Kokkos::View<int*,AssemblyDevice> ptypes("parameter types",paramtypes.size());
      auto host_ptypes = Kokkos::create_mirror_view(ptypes);
      for (size_t i=0; i<paramtypes.size(); i++) {
        host_ptypes(i) = paramtypes[i];
      }
      Kokkos::deep_copy(ptypes, host_ptypes);
      
      Kokkos::View<size_t*,AssemblyDevice> plengths("parameter lengths",paramvals[0].size());
      auto host_plengths = Kokkos::create_mirror_view(plengths);
      for (size_t i=0; i<paramvals[0].size(); i++) {
        host_plengths(i) = paramvals[0][i].size();
      }
      Kokkos::deep_copy(plengths, host_plengths);
      
      size_t prog = 0;
      Kokkos::View<size_t**,AssemblyDevice> pseed("parameter seed index",paramvals[0].size(),maxlength);
      auto host_pseed = Kokkos::create_mirror_view(pseed);
      for (size_t i=0; i<paramvals[0].size(); i++) {
        if (paramtypes[i] == 1) {
          for (size_t j=0; j<paramvals[0][i].size(); j++) {
            host_pseed(i,j) = prog;
            prog++;
          }
        }
      }
      Kokkos::deep_copy(pseed,host_pseed);
      //KokkosTools::print(pseed);
      
      Kokkos::View<ScalarT***,AssemblyDevice> pvals("parameter values",paramvals.size(), paramvals[0].size(), maxlength);
      auto host_pvals = Kokkos::create_mirror_view(pvals);
      for (size_t k=0; k<paramvals.size(); k++) {
        for (size_t i=0; i<paramvals[k].size(); i++) {
          for (size_t j=0; j<paramvals[k][i].size(); j++) {
            host_pvals(k,i,j) = paramvals[k][i][j];
          }
        }
      }
      Kokkos::deep_copy(pvals, host_pvals);
      
      this->sacadoizeParamsSc(seed_active, ptypes, plengths, pseed, pvals, paramvals_KV_ALL);
#ifndef MrHyDE_NO_AD
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD2_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD4_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD8_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD16_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD18_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD24_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD32_ALL);
#endif
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::sacadoizeParamsSc(const bool & seed_active,
                                             Kokkos::View<int*,AssemblyDevice> ptypes,
                                             Kokkos::View<size_t*,AssemblyDevice> plengths,
                                             Kokkos::View<size_t**,AssemblyDevice> pseed,
                                             Kokkos::View<ScalarT***,AssemblyDevice> pvals,
                                             Kokkos::View<ScalarT***,AssemblyDevice> kv_pvals) {
  
  parallel_for("parameter manager sacadoize - no seeding",
               RangePolicy<AssemblyExec>(0,pvals.extent(0)),
               MRHYDE_LAMBDA (const size_type i ) {
    for (size_t j=0; j<pvals.extent(1); j++) {
      for (size_t k=0; k<pvals.extent(2); k++) {
        kv_pvals(i,j,k) = pvals(i,j,k);
      }
    }
  });
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
template<class EvalT>
void ParameterManager<Node>::sacadoizeParams(const bool & seed_active,
                                             Kokkos::View<int*,AssemblyDevice> ptypes,
                                             Kokkos::View<size_t*,AssemblyDevice> plengths,
                                             Kokkos::View<size_t**,AssemblyDevice> pseed,
                                             Kokkos::View<ScalarT***,AssemblyDevice> pvals,
                                             Kokkos::View<EvalT***,AssemblyDevice> kv_pvals) {
  
  
  if (paramvals.size() > 0) {
    if (seed_active) {
      
      parallel_for("parameter manager sacadoize - seed active",
                   RangePolicy<AssemblyExec>(0,pvals.extent(0)),
                   MRHYDE_LAMBDA (const size_type k ) {
        for (size_t i=0; i<plengths.extent(0); i++) {
          if (ptypes(i) == 1) { // active params
            for (size_t j=0; j<plengths(i); j++) {
              EvalT dummyval = 0.0;
              if (dummyval.size() > pseed(i,j)) {
                kv_pvals(k,i,j) = EvalT(dummyval.size(), pseed(i,j), pvals(k,i,j));
              }
              else {
                kv_pvals(k,i,j) = EvalT(pvals(k,i,j));
              }
            }
          }
          else {
            for (size_t j=0; j<plengths(i); j++) {
              kv_pvals(k,i,j) = EvalT(pvals(k,i,j));
            }
          }
        }
      });
    }
    else {
      parallel_for("parameter manager sacadoize - no seeding",
                   RangePolicy<AssemblyExec>(0,pvals.extent(0)),
                   MRHYDE_LAMBDA (const size_type index ) {
        for (size_t i=0; i<plengths.extent(0); i++) {
          for (size_t j=0; j<plengths(i); j++) {
            kv_pvals(index,i,j) = EvalT(pvals(index,i,j));
          }
        }
      });
    }
  }
}
