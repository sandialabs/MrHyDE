/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// ========================================================================================
#if defined(MrHyDE_ENABLE_HDSA)
template<class Node>
void ParameterManager<Node>::updateParams(const vector_RCP & newparams) {

        // only for steady state
        discretized_params[0]->assign(*newparams);
        discretized_params_over[0]->putScalar(0.0);
        discretized_params_over[0]->doImport(*newparams, *param_importer, Tpetra::ADD);
}
#endif
// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateParams(MrHyDE_OptVector & newparams) {
  
  Teuchos::TimeMonitor localtimer(*updatetimer);

  if (newparams.haveScalar()) {
    auto scalar_params = newparams.getParameter();
    
    for (size_t i=0; i<paramvals.size(); i++) {
      size_t pprog = 0;
      for (size_t k=0; k<paramvals[i].size(); k++) {
        if (paramtypes[k] == 1) {
          auto cparams = scalar_params[i];
          for (size_t j=0; j<paramvals[i][k].size(); j++) {
            if (Comm->getRank() == 0 && verbosity > 0) {
              cout << "Updated Params: " << paramvals[i][k][j] << " (old value)   " << (*cparams)[pprog] << " (new value)" << endl;
            }
            paramvals[i][k][j] = (*cparams)[pprog];
            pprog++;
          }
        }
      }
    }
    
  }
  
  if (newparams.haveField()) {
    auto disc_params = newparams.getField();
    
    for (size_t i=0; i<disc_params.size(); ++i) {
      auto owned_vec = disc_params[i]->getVector();
      discretized_params[i]->assign(*owned_vec);
      discretized_params_over[i]->putScalar(0.0);
      discretized_params_over[i]->doImport(*owned_vec, *param_importer, Tpetra::ADD);
    }
    
  }

}


// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateParams(const vector<ScalarT> & newparams, const int & type) {
  size_t pprog = 0;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  
  for (size_t i=0; i<paramvals[0].size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[0][i].size(); j++) {
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << "Updated Params: " << paramvals[0][i][j] << " (old value)   " << newparams[pprog] << " (new value)" << endl;
        }
        paramvals[0][i][j] = newparams[pprog];
        pprog++;
      }
    }
  }
  
  /*
  if ((type == 4) && (globalParamUnknowns > 0)) {
    int numClassicParams = this->getNumParams(1); // offset for ROL param vector
    for (size_t i = 0; i < paramOwnedAndShared.size(); i++) {
      int gid = paramOwnedAndShared[i];
      Psol->replaceGlobalValue(gid,0,newparams[gid+numClassicParams]);
    }
  }
  if ((type == 2) && (globalParamUnknowns > 0)) {
    int numClassicParams = this->getNumParams(2); // offset for ROL param vector
    for (size_t i=0; i<paramOwnedAndShared.size(); i++) {
      int gid = paramOwnedAndShared[i];
      Psol->replaceGlobalValue(gid,0,newparams[i+numClassicParams]);
    }
  }
  */
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateDynamicParams(const int & timestep) {
  
  dynamic_timeindex = timestep;
  size_t index = 0;
  if (have_dynamic_scalar) {
    index = dynamic_timeindex;
  }
  
  auto pslice = subview(paramvals_KV_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KV, pslice);
  
#ifndef MrHyDE_NO_AD
  auto pslice_AD = subview(paramvals_KVAD_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD, pslice_AD);
  auto pslice_AD2 = subview(paramvals_KVAD2_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD2, pslice_AD2);
  auto pslice_AD4 = subview(paramvals_KVAD4_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD4, pslice_AD4);
  auto pslice_AD8 = subview(paramvals_KVAD8_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD8, pslice_AD8);
  auto pslice_AD16 = subview(paramvals_KVAD16_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD16, pslice_AD16);
  auto pslice_AD18 = subview(paramvals_KVAD18_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD18, pslice_AD18);
  auto pslice_AD24 = subview(paramvals_KVAD24_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD24, pslice_AD24);
  auto pslice_AD32 = subview(paramvals_KVAD32_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD32, pslice_AD32);
#endif
  
  if (index == 0) {
    parallel_for("paramman copy zero",
                 RangePolicy<AssemblyExec>(0,paramdot_KV.extent(0)),
                 KOKKOS_CLASS_LAMBDA (const int c ) {
      for (size_type j=0; j<paramdot_KV.extent(1); j++) {
        paramdot_KV(c,j) = 0.0;
#ifndef MrHyDE_NO_AD
        paramdot_KVAD(c,j) = 0.0;
        paramdot_KVAD2(c,j) = 0.0;
        paramdot_KVAD4(c,j) = 0.0;
        paramdot_KVAD8(c,j) = 0.0;
        paramdot_KVAD16(c,j) = 0.0;
        paramdot_KVAD18(c,j) = 0.0;
        paramdot_KVAD24(c,j) = 0.0;
        paramdot_KVAD32(c,j) = 0.0;
#endif
      }
    });
  }
  else {
    parallel_for("paramman copy zero",
                 RangePolicy<AssemblyExec>(0,paramdot_KV.extent(0)),
                 KOKKOS_CLASS_LAMBDA (const int c ) {
      for (size_type j=0; j<paramdot_KV.extent(1); j++) {
        paramdot_KV(c,j) = (paramvals_KV_ALL(index,c,j) - paramvals_KV_ALL(index-1,c,j))/dynamic_dt;
#ifndef MrHyDE_NO_AD
        paramdot_KVAD(c,j) = (paramvals_KVAD_ALL(index,c,j) - paramvals_KVAD_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD2(c,j) = (paramvals_KVAD2_ALL(index,c,j) - paramvals_KVAD2_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD4(c,j) = (paramvals_KVAD4_ALL(index,c,j) - paramvals_KVAD4_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD8(c,j) = (paramvals_KVAD8_ALL(index,c,j) - paramvals_KVAD8_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD16(c,j) = (paramvals_KVAD16_ALL(index,c,j) - paramvals_KVAD16_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD18(c,j) = (paramvals_KVAD18_ALL(index,c,j) - paramvals_KVAD18_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD24(c,j) = (paramvals_KVAD24_ALL(index,c,j) - paramvals_KVAD24_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD32(c,j) = (paramvals_KVAD32_ALL(index,c,j) - paramvals_KVAD32_ALL(index-1,c,j).val())/dynamic_dt;
#endif
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateParams(const vector<ScalarT> & newparams, const std::string & stype) {
  size_t pprog = 0;
  int type = -1;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  if (stype == "inactive") { type = 0;}
  else if (stype == "active") { type = 1;}
  else if (stype == "stochastic") { type = 2;}
  else if (stype == "discrete") { type = 3;}
  else {
    //complain
  }
  
  int index = 0;
  if (have_dynamic_scalar) {
    index = dynamic_timeindex;
  }
  
  if (paramvals.size() > index) {
    for (size_t i=0; i<paramvals[index].size(); i++) {
      if (paramtypes[i] == type) {
        for (size_t j=0; j<paramvals[index][i].size(); j++) {
          paramvals[index][i][j] = newparams[pprog];
          pprog++;
        }
      }
    }
  }
}
