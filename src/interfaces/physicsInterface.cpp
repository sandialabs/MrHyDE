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

#include "physicsInterface.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SetupUtilities.hpp"

#include "rectPeriodicMatcher.hpp"

// Enabled physics modules:
#include "porous.hpp"
#include "porousHDIV.hpp"
#include "porousHDIV_hybridized.hpp"
#include "porousHDIV_weakGalerkin.hpp"
//#include "twophasePoNo.hpp"
//#include "twophasePoPw.hpp"
#include "cdr.hpp"
#include "thermal.hpp"
//#include "thermal_enthalpy.hpp"
#include "msphasefield.hpp"
#include "stokes.hpp"
#include "navierstokes.hpp"
#include "linearelasticity.hpp"
#include "helmholtz.hpp"
#include "maxwells_fp.hpp"
#include "shallowwater.hpp"
#include "maxwell.hpp"
#include "maxwell_hybridized.hpp"
#include "ode.hpp"

// Disabled/out-of-date physics modules
//#include "msconvdiff.hpp"
//#include "phasesolidification.hpp"
//#include "mwhelmholtz.hpp"
//#include "peridynamics.hpp"
//#include "euler.hpp"
//#include "burgers.hpp"
//#include "phasefield.hpp"
//#include "thermal_fr.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

physics::physics(Teuchos::RCP<Teuchos::ParameterList> & settings_, Teuchos::RCP<MpiComm> & Comm_,
                 vector<topo_RCP> & cellTopo, vector<topo_RCP> & sideTopo,
                 Teuchos::RCP<panzer_stk::STK_Interface> & mesh) :
settings(settings_), Commptr(Comm_){
  
  milo_debug_level = settings->get<int>("debug level",0);
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics constructor ..." << endl;
    }
  }
  
  mesh->getElementBlockNames(blocknames);
  mesh->getSidesetNames(sideNames);
  
  numBlocks = blocknames.size();
  spaceDim = settings->sublist("Mesh").get<int>("dim");
  cellfield_reduction = settings->sublist("Postprocess").get<string>("extra cell field reduction","mean");
  
  for (size_t b=0; b<blocknames.size(); b++) {
    if (settings->sublist("Physics").isSublist(blocknames[b])) { // adding block overwrites the default
      blockPhysSettings.push_back(settings->sublist("Physics").sublist(blocknames[b]));
    }
    else { // default
      blockPhysSettings.push_back(settings->sublist("Physics"));
    }
    
    if (settings->sublist("Discretization").isSublist(blocknames[b])) { // adding block overwrites default
      blockDiscSettings.push_back(settings->sublist("Discretization").sublist(blocknames[b]));
    }
    else { // default
      blockDiscSettings.push_back(settings->sublist("Discretization"));
    }
  }
  
  this->importPhysics();
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics constructor" << endl;
    }
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Add the functions to the function managers
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::defineFunctions(vector<Teuchos::RCP<FunctionManager> > & functionManagers_) {
  
  functionManagers = functionManagers_;
  
  for (size_t b=0; b<blocknames.size(); b++) {
    Teuchos::ParameterList fs;
    if (settings->sublist("Functions").isSublist(blocknames[b])) {
      fs = settings->sublist("Functions").sublist(blocknames[b]);
    }
    else {
      fs = settings->sublist("Functions");
    }
    
    for (size_t n=0; n<modules[b].size(); n++) {
      modules[b][n]->defineFunctions(fs, functionManagers[b]);
    }
    
    // Add initial conditions
    Teuchos::ParameterList initial_conds = blockPhysSettings[b].sublist("Initial conditions");
    for (size_t j=0; j<varlist[b].size(); j++) {
      string expression;
      if (initial_conds.isType<string>(varlist[b][j])) {
        expression = initial_conds.get<string>(varlist[b][j]);
      }
      else if (initial_conds.isType<double>(varlist[b][j])) {
        double value = initial_conds.get<double>(varlist[b][j]);
        expression = std::to_string(value);
      }
      else {
        expression = "0.0";
      }
      functionManagers[b]->addFunction("initial "+varlist[b][j],expression,"ip");
      functionManagers[b]->addFunction("initial "+varlist[b][j],expression,"point");
    }
    
    // Dirichlet conditions
    Teuchos::ParameterList dbcs = blockPhysSettings[b].sublist("Dirichlet conditions");
    for (size_t j=0; j<varlist[b].size(); j++) {
      if (dbcs.isSublist(varlist[b][j])) {
        if (dbcs.sublist(varlist[b][j]).isType<string>("all boundaries")) {
          string entry = dbcs.sublist(varlist[b][j]).get<string>("all boundaries");
          for (size_t s=0; s<sideNames.size(); s++) {
            string label = "Dirichlet " + varlist[b][j] + " " + sideNames[s];
            functionManagers[b]->addFunction(label,entry,"side ip");
          }
        }
        else if (dbcs.sublist(varlist[b][j]).isType<double>("all boundaries")) {
          double value = dbcs.sublist(varlist[b][j]).get<double>("all boundaries");
          string entry = std::to_string(value);
          for (size_t s=0; s<sideNames.size(); s++) {
            string label = "Dirichlet " + varlist[b][j] + " " + sideNames[s];
            functionManagers[b]->addFunction(label,entry,"side ip");
          }
        }
        else {
          Teuchos::ParameterList currdbcs = dbcs.sublist(varlist[b][j]);
          Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
          while (d_itr != currdbcs.end()) {
            if (currdbcs.isType<string>(d_itr->first)) {
              string entry = currdbcs.get<string>(d_itr->first);
              string label = "Dirichlet " + varlist[b][j] + " " + d_itr->first;
              functionManagers[b]->addFunction(label,entry,"side ip");
            }
            else if (currdbcs.isType<double>(d_itr->first)) {
              double value = currdbcs.get<double>(d_itr->first);
              string entry = std::to_string(value);
              string label = "Dirichlet " + varlist[b][j] + " " + d_itr->first;
              functionManagers[b]->addFunction(label,entry,"side ip");
            }
            d_itr++;
          }
        }
      }
    }
    
    // Neumann/robin conditions
    Teuchos::ParameterList nbcs = blockPhysSettings[b].sublist("Neumann conditions");
    for (size_t j=0; j<varlist[b].size(); j++) {
      if (nbcs.isSublist(varlist[b][j])) {
        if (nbcs.sublist(varlist[b][j]).isParameter("all boundaries")) {
          string entry = nbcs.sublist(varlist[b][j]).get<string>("all boundaries");
          for (size_t s=0; s<sideNames.size(); s++) {
            string label = "Neumann " + varlist[b][j] + " " + sideNames[s];
            functionManagers[b]->addFunction(label,entry,"side ip");
          }
        }
        else {
          Teuchos::ParameterList currnbcs = nbcs.sublist(varlist[b][j]);
          Teuchos::ParameterList::ConstIterator n_itr = currnbcs.begin();
          while (n_itr != currnbcs.end()) {
            string entry = currnbcs.get<string>(n_itr->first);
            string label = "Neumann " + varlist[b][j] + " " + n_itr->first;
            functionManagers[b]->addFunction(label,entry,"side ip");
            n_itr++;
          }
        }
      }
    }
    
    vector<string> block_ef;
    Teuchos::ParameterList efields = blockPhysSettings[b].sublist("Extra fields");
    Teuchos::ParameterList::ConstIterator ef_itr = efields.begin();
    while (ef_itr != efields.end()) {
      string entry = efields.get<string>(ef_itr->first);
      block_ef.push_back(ef_itr->first);
      functionManagers[b]->addFunction(ef_itr->first,entry,"ip");
      functionManagers[b]->addFunction(ef_itr->first,entry,"point");
      ef_itr++;
    }
    extrafields_list.push_back(block_ef);
    
    vector<string> block_ecf;
    Teuchos::ParameterList ecfields = blockPhysSettings[b].sublist("Extra cell fields");
    Teuchos::ParameterList::ConstIterator ecf_itr = ecfields.begin();
    while (ecf_itr != ecfields.end()) {
      string entry = ecfields.get<string>(ecf_itr->first);
      block_ecf.push_back(ecf_itr->first);
      functionManagers[b]->addFunction(ecf_itr->first,entry,"ip");
      ecf_itr++;
    }
    extracellfields_list.push_back(block_ecf);
    
    vector<string> block_resp;
    Teuchos::ParameterList rfields = blockPhysSettings[b].sublist("Responses");
    Teuchos::ParameterList::ConstIterator r_itr = rfields.begin();
    while (r_itr != rfields.end()) {
      string entry = rfields.get<string>(r_itr->first);
      block_resp.push_back(r_itr->first);
      functionManagers[b]->addFunction(r_itr->first,entry,"point");
      functionManagers[b]->addFunction(r_itr->first,entry,"ip");
      r_itr++;
    }
    response_list.push_back(block_resp);
    
    vector<string> block_targ;
    Teuchos::ParameterList tfields = blockPhysSettings[b].sublist("Targets");
    Teuchos::ParameterList::ConstIterator t_itr = tfields.begin();
    while (t_itr != tfields.end()) {
      string entry = tfields.get<string>(t_itr->first);
      block_targ.push_back(t_itr->first);
      functionManagers[b]->addFunction(t_itr->first,entry,"point");
      functionManagers[b]->addFunction(t_itr->first,entry,"ip");
      t_itr++;
    }
    target_list.push_back(block_targ);
    
    vector<string> block_wts;
    Teuchos::ParameterList wfields = blockPhysSettings[b].sublist("Weights");
    Teuchos::ParameterList::ConstIterator w_itr = wfields.begin();
    while (w_itr != wfields.end()) {
      string entry = wfields.get<string>(w_itr->first);
      block_wts.push_back(w_itr->first);
      functionManagers[b]->addFunction(w_itr->first,entry,"point");
      functionManagers[b]->addFunction(w_itr->first,entry,"ip");
      w_itr++;
    }
    weight_list.push_back(block_wts);
    
  }
  
  for (size_t b=0; b<blocknames.size(); b++) {
    Teuchos::ParameterList functions;
    if (settings->sublist("Functions").isSublist(blocknames[b])) {
      functions = settings->sublist("Functions").sublist(blocknames[b]);
    }
    else {
      functions = settings->sublist("Functions");
    }
    Teuchos::ParameterList::ConstIterator fnc_itr = functions.begin();
    while (fnc_itr != functions.end()) {
      string entry = functions.get<string>(fnc_itr->first);
      functionManagers[b]->addFunction(fnc_itr->first,entry,"ip");
      functionManagers[b]->addFunction(fnc_itr->first,entry,"side ip");
      functionManagers[b]->addFunction(fnc_itr->first,entry,"point");
      fnc_itr++;
    }
  }
  
  /*
  if (functions.isSublist("Side")) {
    Teuchos::ParameterList side_functions = functions.sublist("Side");
    
    for (size_t b=0; b<blocknames.size(); b++) {
      Teuchos::ParameterList::ConstIterator fnc_itr = side_functions.begin();
      while (fnc_itr != side_functions.end()) {
        string entry = side_functions.get<string>(fnc_itr->first);
        functionManagers[b]->addFunction(fnc_itr->first,entry,"side ip");
        fnc_itr++;
      }
    }
  }
  */
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Add the requested physics modules, variables, discretization types
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::importPhysics() {
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics::importPhysics ..." << endl;
    }
  }
  
  for (size_t b=0; b<blocknames.size(); b++) {
    vector<int> currorders;
    vector<string> currtypes;
    vector<string> currvarlist;
    vector<int> currvarowned;
    
    vector<Teuchos::RCP<physicsbase> > currmodules;
    vector<bool> currSubgrid, curruseDG;
    std::string var;
    int default_order = 1;
    std::string default_type = "HGRAD";
    string module_list = blockPhysSettings[b].get<string>("modules","");
    vector<string> enabled_modules;
    // Script to break delimited list into pieces
    {
      string delimiter = ", ";
      size_t pos = 0;
      if (module_list.find(delimiter) == string::npos) {
        enabled_modules.push_back(module_list);
      }
      else {
        string token;
        while ((pos = module_list.find(delimiter)) != string::npos) {
          token = module_list.substr(0, pos);
          enabled_modules.push_back(token);
          module_list.erase(0, pos + delimiter.length());
        }
        enabled_modules.push_back(module_list);
      }
    }
    
    for (size_t mod=0; mod<enabled_modules.size(); mod++) {
      string modname = enabled_modules[mod];
      
      // Porous media (single phase slightly compressible)
      if (modname == "porous") {
        Teuchos::RCP<porous> porous_RCP = Teuchos::rcp(new porous(settings) );
        currmodules.push_back(porous_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_porous",false));
      }
    
      // Porous media with HDIV basis
      if (modname == "porousHDIV") {
        Teuchos::RCP<porousHDIV> porousHDIV_RCP = Teuchos::rcp(new porousHDIV(settings) );
        currmodules.push_back(porousHDIV_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_porousHDIV",false));
      }
      
      // Hybridized porous media with HDIV basis
      if (modname == "porousHDIV_hybrid") {
        Teuchos::RCP<porousHDIV_HYBRID> porousHDIV_HYBRID_RCP = Teuchos::rcp(new porousHDIV_HYBRID(settings) );
        currmodules.push_back(porousHDIV_HYBRID_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_porousHDIV_HYBRID",false));
      }
      
      // weak Galerkin porous media with HDIV basis
      if (modname == "porousHDIV_weakGalerkin") {
        Teuchos::RCP<porousHDIV_WG> porousHDIV_WG_RCP = Teuchos::rcp(new porousHDIV_WG(settings) );
        currmodules.push_back(porousHDIV_WG_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_porousHDIV_WG",false));
      }
      
      /*
      // Two phase porous media
      if (modname == "twophase") {
        string formulation = blockPhysSettings[b].get<string>("formulation","PoNo");
        if (formulation == "PoPw"){
          //Teuchos::RCP<twophasePoPw> twophase_RCP = Teuchos::rcp(new twophasePoPw(settings) );
          //currmodules.push_back(twophase_RCP);
        }
        else if (formulation == "PoNo"){
          Teuchos::RCP<twophasePoNo> twophase_RCP = Teuchos::rcp(new twophasePoNo(settings) );
          currmodules.push_back(twophase_RCP);
          currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_twophase",false));
        }
        else if (formulation == "PoPw"){
          Teuchos::RCP<twophasePoPw> twophase_RCP = Teuchos::rcp(new twophasePoPw(settings) );
          currmodules.push_back(twophase_RCP);
          currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_twophase",false));
        }
        
      }
      */
      // Convection diffusion
      if (modname == "cdr" || modname == "CDR") {
        Teuchos::RCP<cdr> cdr_RCP = Teuchos::rcp(new cdr(settings) );
        currmodules.push_back(cdr_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_cdr",false));
      }
      
      /* not setting up correctly
       // Multiple Species convection diffusion reaction
       if (blockPhysSettings[b].get<bool>("solve_msconvdiff",false)) {
       //currmodules.push_back(msconvdiff_RCP);
       }
       */
      
      // Thermal
      if (modname == "thermal") {
        Teuchos::RCP<thermal> thermal_RCP = Teuchos::rcp(new thermal(settings) );
        currmodules.push_back(thermal_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_thermal",false));
      }
      
      /*
       // Thermal with fractional operator
       if (blockPhysSettings[b].get<bool>("solve_thermal_fr",false)) {
       Teuchos::RCP<thermal_fr> thermal_fr_RCP = Teuchos::rcp(new thermal_fr(settings, numip, numip_side) );
       currmodules.push_back(thermal_fr_RCP);
       }
       */
      /*
      // Thermal with enthalpy variable
      if (modname == "thermal enthalpy") {
        Teuchos::RCP<thermal_enthalpy> thermal_enthalpy_RCP = Teuchos::rcp(new thermal_enthalpy(settings) );
        currmodules.push_back(thermal_enthalpy_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_thermal_enthalpy",false));
      }
      */
      // Shallow Water
      if (modname == "shallow water") {
        Teuchos::RCP<shallowwater> shallowwater_RCP = Teuchos::rcp(new shallowwater(settings) );
        currmodules.push_back(shallowwater_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_shallowwater",false));
      }
      
      // Maxwell
      if (modname == "maxwell") {
        Teuchos::RCP<maxwell> maxwell_RCP = Teuchos::rcp(new maxwell(settings) );
        currmodules.push_back(maxwell_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_maxwell",false));
      }
      
      // Maxwell hybridized
      if (modname == "maxwell hybrid") {
        Teuchos::RCP<maxwell_HYBRID> maxwell_HYBRID_RCP = Teuchos::rcp(new maxwell_HYBRID(settings) );
        currmodules.push_back(maxwell_HYBRID_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_maxwell_hybrid",false));
      }
      
      /* not setting up correctly
       // Burgers (entropy viscosity)
       if (blockPhysSettings[b].get<bool>("solve_burgers",false)) {
       currmodules.push_back(burgers_RCP);
       }
       */
      
      /*
       // PhaseField
       if (blockPhysSettings[b].get<bool>("solve_phasefield",false)) {
       Teuchos::RCP<phasefield> phasefield_RCP = Teuchos::rcp(new phasefield(settings, numip, numip_side) );
       currmodules.push_back(phasefield_RCP);
       }
       
       */
      
      // Multiple Species PhaseField
      if (modname == "msphasefield") {
        Teuchos::RCP<msphasefield> msphasefield_RCP = Teuchos::rcp(new msphasefield(settings, Commptr) );
        currmodules.push_back(msphasefield_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_msphasefield",false));
      }
      
      // Stokes
      if (modname == "stokes" || modname == "Stokes") {
        Teuchos::RCP<stokes> stokes_RCP = Teuchos::rcp(new stokes(settings) );
        
        currmodules.push_back(stokes_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_stokes",false));
      }
      
      // Navier Stokes
      if (modname == "navier stokes" || modname == "Navier Stokes") {
        Teuchos::RCP<navierstokes> navierstokes_RCP = Teuchos::rcp(new navierstokes(settings) );
        
        currmodules.push_back(navierstokes_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_navierstokes",false));
      }
      
      /* not setting up correctly
       // Euler
       if (blockPhysSettings[b].get<bool>("solve_euler",false)) {
       currmodules.push_back(euler_RCP);
       }
       */
      
      // Linear Elasticity
      if (modname == "linearelasticity" || modname == "linear elasticity") {
        Teuchos::RCP<linearelasticity> linearelasticity_RCP = Teuchos::rcp(new linearelasticity(settings) );
        currmodules.push_back(linearelasticity_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_linearelasticity",false));
      }
      
      /* not setting up correctly
       // Peridynamics
       if (blockPhysSettings[b].get<bool>("solve_peridynamics",false)) {
       currmodules.push_back(peridynamics_RCP);
       }
       */
      
      
      // Helmholtz
      if (modname == "helmholtz") {
        Teuchos::RCP<helmholtz> helmholtz_RCP = Teuchos::rcp(new helmholtz(settings) );
        currmodules.push_back(helmholtz_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_helmholtz",false));
      }
      
      
      /* not setting up correctly
       // Helmholtz with multiple wavenumbers
       if (blocksettings.get<bool>("solve_mwhelmholtz",false)){
       currmodules.push_back(mwhelmholtz_RCP);
       }
       */
      
      // Maxwell's (potential of electric field, curl-curl frequency domain (Boyse et al (1992))
      if (modname == "maxwells_freq_pot"){
        Teuchos::RCP<maxwells_fp> maxwells_fp_RCP = Teuchos::rcp(new maxwells_fp(settings) );
        currmodules.push_back(maxwells_fp_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_maxwells_freq_pot",false));
      }
      
      // Scalar ODE for testing time integrators independent of spatial discretizations
      if (modname == "ODE"){
        Teuchos::RCP<ODE> ODE_RCP = Teuchos::rcp(new ODE(settings) );
        currmodules.push_back(ODE_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_ODE",false));
      }
      
      /*
       // PhaseField Solidification
       if (blockPhysSettings[b].get<bool>("solve_phasesolidification",false)) {
       Teuchos::RCP<phasesolidification> phasesolid_RCP = Teuchos::rcp(new phasesolidification(settings, Commptr, numip, numip_side) );
       currmodules.push_back(phasesolid_RCP);
       }
       */
    }
    
    modules.push_back(currmodules);
    useSubgrid.push_back(currSubgrid);
    
    for (size_t m=0; m<currmodules.size(); m++) {
      vector<string> cvars = currmodules[m]->myvars;
      vector<string> ctypes = currmodules[m]->mybasistypes;
      for (size_t v=0; v<cvars.size(); v++) {
        currvarlist.push_back(cvars[v]);
        
        if (ctypes[v] == "HGRAD-DG") {
          currtypes.push_back("HGRAD");
          curruseDG.push_back(true);
        }
        else if (ctypes[v] == "HDIV-DG") {
          currtypes.push_back("HDIV");
          curruseDG.push_back(true);
        }
        else if (ctypes[v] == "HCURL-DG") {
          currtypes.push_back("HCURL");
          curruseDG.push_back(true);
        }
        else {
          currtypes.push_back(ctypes[v]);
          curruseDG.push_back(false);
        }
        currvarowned.push_back(m);
        currorders.push_back(blockDiscSettings[b].sublist("order").get<int>(cvars[v],default_order));
      }
    }
    useDG.push_back(curruseDG);
    
    int currnumVars = currvarlist.size();
    //activeModules.push_back(block_activeModules);
    TEUCHOS_TEST_FOR_EXCEPTION(currnumVars==0,std::runtime_error,"Error: no physics were enabled on block: " + blocknames[b]);
    
    
    std::vector<int> currunique_orders;
    std::vector<string> currunique_types;
    std::vector<int> currunique_index;
    
    for (size_t j=0; j<currorders.size(); j++) {
      bool is_unique = true;
      for (size_t k=0; k<currunique_orders.size(); k++) {
        if (currunique_orders[k] == currorders[j] && currunique_types[k] == currtypes[j]) {
          is_unique = false;
          currunique_index.push_back(k);
        }
      }
      if (is_unique) {
        currunique_orders.push_back(currorders[j]);
        currunique_types.push_back(currtypes[j]);
        currunique_index.push_back(currunique_orders.size()-1);
      }
    }
    
    vector<string> discretized_param_basis_types;
    vector<int> discretized_param_basis_orders;
    if (settings->isSublist("Parameters")) {
      Teuchos::ParameterList parameters = settings->sublist("Parameters");
      Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
      while (pl_itr != parameters.end()) {
        Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
        if (newparam.get<string>("usage") == "discretized") {
          discretized_param_basis_types.push_back(newparam.get<string>("type","HGRAD"));
          discretized_param_basis_orders.push_back(newparam.get<int>("order",1));
        }
        pl_itr++;
      }
    }
    
    for (size_t j=0; j<discretized_param_basis_orders.size(); j++) {
      bool is_unique = true;
      for (size_t k=0; k<currunique_orders.size(); k++) {
        if (currunique_orders[k] == discretized_param_basis_orders[j] && currunique_types[k] == discretized_param_basis_types[j]) {
          is_unique = false;
          //currunique_index.push_back(k);
        }
      }
      if (is_unique) {
        currunique_orders.push_back(discretized_param_basis_orders[j]);
        currunique_types.push_back(discretized_param_basis_types[j]);
      //  currunique_index.push_back(currunique_orders.size()-1);
      }
    }
    orders.push_back(currorders);
    types.push_back(currtypes);
    varlist.push_back(currvarlist);
    varowned.push_back(currvarowned);
    numVars.push_back(currnumVars);
    unique_orders.push_back(currunique_orders);
    unique_types.push_back(currunique_types);
    unique_index.push_back(currunique_index);
  }
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics::importPhysics ..." << endl;
    }
  }
  
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int physics::getvarOwner(const int & block, const string & var) {
  int owner = 0;
  for (size_t k=0; k<varlist[block].size(); k++) {
    if (varlist[block][k] == var) {
      owner = varowned[block][k];
    }
  }
  return owner;
  
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

// TMW: this function is probably never used

AD physics::getDirichletValue(const int & block, const ScalarT & x, const ScalarT & y,
                              const ScalarT & z, const ScalarT & t, const string & var,
                              const string & gside, const bool & useadjoint,
                              Teuchos::RCP<workset> & wkset) {
  
  // update point in wkset
  wkset->point(0,0,0) = x;
  wkset->point(0,0,1) = y;
  if(spaceDim == 3)
    wkset->point(0,0,2) = z;
  wkset->time_KV(0) = t;
  
  // evaluate the response
  FDATA ddata = functionManagers[block]->evaluate("Dirichlet " + var + " " + gside,"point");
  AD val = 0.0;
  return ddata(0,0);
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

ScalarT physics::getInitialValue(const int & block, const ScalarT & x, const ScalarT & y,
                                const ScalarT & z, const string & var, const bool & useadjoint) {
  
  /*
  // update point in wkset
  wkset->point_KV(0,0,0) = x;
  wkset->point_KV(0,0,1) = y;
  wkset->point_KV(0,0,2) = z;
  
  // evaluate the response
  FDATA idata = functionManager->evaluate("initial " + var,"point",block);
  return idata(0,0).val();
  */
  return 0.0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// TMW: the following function will soon be removed
/////////////////////////////////////////////////////////////////////////////////////////////

int physics::getNumResponses(const int & block, const string & var) {
  return response_list[block].size();
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int physics::getNumResponses(const int & block) {
  return response_list[block].size();
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Really designed for sensor responses, but can be used for ip responses (global)
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> physics::getPointResponse(const int & block,
                                                             Kokkos::View<AD****,AssemblyDevice> u_ip,
                                                             Kokkos::View<AD****,AssemblyDevice> ugrad_ip,
                                                             Kokkos::View<AD****,AssemblyDevice> p_ip,
                                                             Kokkos::View<AD****,AssemblyDevice> pgrad_ip,
                                                             const DRV ip, const ScalarT & time,
                                                             Teuchos::RCP<workset> & wkset) {
  
  size_t numElem = u_ip.extent(0);
  size_t numip = ip.extent(1);
  size_t numResponses = response_list[block].size();
  
  Kokkos::View<AD***,AssemblyDevice> responsetotal("responses",numElem,numResponses,numip);
  
  auto point = Kokkos::subview(wkset->point, 0, 0, Kokkos::ALL());
  auto sol = Kokkos::subview(wkset->local_soln_point, 0, Kokkos::ALL(), 0, Kokkos::ALL());
  auto sol_grad = Kokkos::subview(wkset->local_soln_grad_point, 0, Kokkos::ALL(), 0, Kokkos::ALL());
  
  // This is very clumsy
  Kokkos::View<size_t*, AssemblyDevice> indices("view to hold indices",3);
  auto host_indices = Kokkos::create_mirror_view(indices);
  for (size_t e=0; e<numElem; e++) {
    host_indices(0) = e;
    for (size_t k=0; k<numip; k++) {
      host_indices(1) = k;
      auto ip_sv = Kokkos::subview(ip, e, k, Kokkos::ALL());
      auto u_sv = Kokkos::subview(u_ip, e, Kokkos::ALL(), k, Kokkos::ALL());
      auto ugrad_sv = Kokkos::subview(ugrad_ip, e, Kokkos::ALL(), k, Kokkos::ALL());
      
      Kokkos::deep_copy(point, ip_sv);
      Kokkos::deep_copy(sol, u_sv);
      Kokkos::deep_copy(sol_grad, ugrad_sv);
      
      if (p_ip.extent(0) > 0) {
        auto param = Kokkos::subview(wkset->local_param_point, 0, Kokkos::ALL(), 0, 0);
        auto param_grad = Kokkos::subview(wkset->local_param_grad_point, 0, Kokkos::ALL(), 0, Kokkos::ALL());
        auto p_sv = Kokkos::subview(p_ip, e, Kokkos::ALL(), k, 0);
        auto pgrad_sv = Kokkos::subview(pgrad_ip, e, Kokkos::ALL(), k, Kokkos::ALL());
        Kokkos::deep_copy(param, p_sv);
        Kokkos::deep_copy(param_grad, pgrad_sv);
      }
      
      for (size_t r=0; r<numResponses; r++) {
        host_indices(2) = r;
        Kokkos::deep_copy(indices,host_indices);
        // evaluate the response
        FDATA rdata = functionManagers[block]->evaluate(response_list[block][r],"point");
        // copy data into responsetotal
        // again clumsy
        parallel_for("physics point response",RangePolicy<AssemblyExec>(0,1), KOKKOS_LAMBDA (const int elem ) {
          responsetotal(indices(0),indices(2),indices(1)) = rdata(0,0);
        });
      }
    }
  }
  
  return responsetotal;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> physics::getResponse(const int & block,
                                                        Kokkos::View<AD****,AssemblyDevice> u_ip,
                                                        Kokkos::View<AD****,AssemblyDevice> ugrad_ip,
                                                        Kokkos::View<AD****,AssemblyDevice> p_ip,
                                                        Kokkos::View<AD****,AssemblyDevice> pgrad_ip,
                                                        const Kokkos::View<ScalarT***,AssemblyDevice> ip,
                                                        const ScalarT & time,
                                                        Teuchos::RCP<workset> & wkset) {
  size_t numElem = u_ip.extent(0);
  size_t numip = ip.extent(1);
  size_t numResponses = response_list[block].size();
  
  Kokkos::View<AD***,AssemblyDevice> responsetotal("responses",numElem,numResponses,numip);
  
  //wkset->ip_KV = ip;
  Kokkos::deep_copy(wkset->ip,ip);
  Kokkos::deep_copy(wkset->local_soln,u_ip);
  if (wkset->vars_HGRAD.size() > 0) {
    Kokkos::deep_copy(wkset->local_soln_grad, ugrad_ip);
  }
  if (p_ip.extent(0) > 0) {
    Kokkos::deep_copy(wkset->local_param,p_ip);
  }
  for (size_t r=0; r<numResponses; r++) {
    
    // evaluate the response
    FDATA rdata = functionManagers[block]->evaluate(response_list[block][r],"ip");
    
    auto cresp = Kokkos::subview(responsetotal,Kokkos::ALL(), r, Kokkos::ALL());
    Kokkos::deep_copy(cresp,rdata);
    
  }
  
  return responsetotal;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// TMW: following function may be removed
/////////////////////////////////////////////////////////////////////////////////////////////

AD physics::computeTopoResp(const size_t & block){
  AD topoResp = 0.0;
  for (size_t i=0; i<modules[block].size(); i++) {
    // needs to be updated
    //topoResp += udfunc->penaltyTopo();
  }
  
  return topoResp;
}

/////////////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////////////

bool physics::checkFace(const size_t & block){
  bool include_face = false;
  for (size_t i=0; i<modules[block].size(); i++) {
    bool cuseef = modules[block][i]->include_face;
    if (cuseef) {
      include_face = true;
    }
  }
  
  return include_face;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> physics::target(const int & block,
                                                   const Kokkos::View<ScalarT***,AssemblyDevice> ip,
                                                   const ScalarT & current_time,
                                                   Teuchos::RCP<workset> & wkset) {
  
  Kokkos::View<AD***,AssemblyDevice> targettotal("target",ip.extent(0),
                                                 target_list[block].size(),ip.extent(1));
  
  for (size_t t=0; t<target_list[block].size(); t++) {
    FDATA tdata = functionManagers[block]->evaluate(target_list[block][t],"ip");
    auto ctarg = Kokkos::subview(targettotal,Kokkos::ALL(), t, Kokkos::ALL());
    Kokkos::deep_copy(ctarg,tdata);
  }
  return targettotal;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> physics::weight(const int & block,
                                                   const Kokkos::View<ScalarT***,AssemblyDevice> ip,
                                                   const ScalarT & current_time,
                                                   Teuchos::RCP<workset> & wkset) {
  
  Kokkos::View<AD***,AssemblyDevice> weighttotal("weight",ip.extent(0),
                                                 weight_list[block].size(),ip.extent(1));
  
  for (size_t t=0; t<weight_list[block].size(); t++) {
    FDATA wdata = functionManagers[block]->evaluate(weight_list[block][t],"ip");
    auto cwt = Kokkos::subview(weighttotal,Kokkos::ALL(), t, Kokkos::ALL());
    Kokkos::deep_copy(cwt,wdata);
  }
  return weighttotal;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> physics::getInitial(const Kokkos::View<ScalarT***,AssemblyDevice> ip,
                                                            const int & block,
                                                            const bool & project,
                                                            Teuchos::RCP<workset> & wkset) {
  
  
  size_t numElem = ip.extent(0);
  size_t numVars = varlist[block].size();
  size_t numip = ip.extent(1);
  
  Kokkos::View<ScalarT***,AssemblyDevice> ivals("temp invals", numElem, numVars, numip);
  
  if (project) {
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<varlist[block].size(); n++) {
  
      FDATA ivals_AD = functionManagers[block]->evaluate("initial " + varlist[block][n],"ip");
      auto cvals = Kokkos::subview( ivals, Kokkos::ALL(), n, Kokkos::ALL());
      //copy
      parallel_for("physics fill initial values",RangePolicy<AssemblyExec>(0,cvals.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t i=0; i<cvals.extent(1); i++) {
          cvals(e,i) = ivals_AD(e,i).val();
        }
      });
    }
  }
  else {
    // TMW: will not work on device yet
    Kokkos::View<ScalarT***,AssemblyDevice> point_KV = wkset->point;
    auto host_ivals = Kokkos::create_mirror_view(ivals);
    for (size_t e=0; e<numElem; e++) {
      for (size_t i=0; i<numip; i++) {
        // set the node in wkset
        auto node = Kokkos::subview( ip, e, i, Kokkos::ALL());
        
        parallel_for("physics initial set point",RangePolicy<AssemblyExec>(0,node.extent(0)), KOKKOS_LAMBDA (const int s ) {
          point_KV(0,0,s) = node(s);
        });
        
        for (size_t n=0; n<varlist[block].size(); n++) {
          // evaluate
          FDATA ivals_AD = functionManagers[block]->evaluate("initial " + varlist[block][n],"point");
          
          ivals(e,n,i) = ivals_AD(0,0).val();
          // copy
          //auto iv = Kokkos::subview( ivals, e, n, i);
          //parallel_for("physics initial set point",RangePolicy<AssemblyExec>(0,1), KOKKOS_LAMBDA (const int s ) {
          //  iv(0) = ivals_AD(0,0).val();
          //});
        }
      }
    }
  }
  //KokkosTools::print(ivals);
  return ivals;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,AssemblyDevice> physics::getDirichlet(const Kokkos::View<ScalarT***,AssemblyDevice> ip, const int & var,
                                                              const int & block,
                                                              const std::string & sidename,
                                                              Teuchos::RCP<workset> & wkset) {
  
  
  size_t numElem = ip.extent(0);
  size_t numip = ip.extent(1);
  
  Kokkos::View<ScalarT**,AssemblyDevice> dvals("temp dnvals", numElem, numip);
  
  // evaluate
  FDATA dvals_AD = functionManagers[block]->evaluate("Dirichlet " + varlist[block][var] + " " + sidename,"side ip");
  
  // copy values
  parallel_for("physics fill Dirichlet values",RangePolicy<AssemblyExec>(0,dvals.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (size_t i=0; i<dvals.extent(1); i++) {
      dvals(e,i) = dvals_AD(e,i).val();
    }
  });
  return dvals;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::setVars(size_t & block, vector<string> & vars) {
  for (size_t i=0; i<modules[block].size(); i++) {
    modules[block][i]->setVars(vars);
  }
}

void physics::setAuxVars(size_t & block, vector<string> & vars) {
  for (size_t i=0; i<modules[block].size(); i++) {
    modules[block][i]->setAuxVars(vars);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params,
                               const vector<string> & paramnames) {
  
  //needs to be deprecated
  //udfunc->updateParameters(params,paramnames);
  
  for (size_t b=0; b<modules.size(); b++) {
    for (size_t i=0; i<modules[b].size(); i++) {
      modules[b][i]->updateParameters(params, paramnames);
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

std::vector<string> physics::getResponseFieldNames(const int & block) {
  vector<string> fields;
  vector<vector<string> > rfields;
  /*
  for (size_t i=0; i<modules[block].size(); i++) {
    rfields.push_back(modules[block][i]->ResponseFieldNames());
  }
  for (size_t i=0; i<rfields.size(); i++) {
    for (size_t j=0; j<rfields[i].size(); j++) {
      fields.push_back(rfields[i][j]);
    }
  }
   */
  return fields;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

std::vector<string> physics::getExtraFieldNames(const int & block) {
  return extrafields_list[block];
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<string> physics::getExtraCellFieldNames(const int & block) {
  return extracellfields_list[block];
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,AssemblyDevice> physics::getExtraFields(const int & block,
                                                                const int & fnum,
                                                                const DRV & ip,
                                                                const ScalarT & time,
                                                                Teuchos::RCP<workset> & wkset) {
  
  Kokkos::View<ScalarT**,AssemblyDevice> fields("field data",ip.extent(0),ip.extent(1));
  
  for (size_type e=0; e<ip.extent(0); e++) {
    for (size_type j=0; j<ip.extent(1); j++) {
      for (int s=0; s<spaceDim; s++) {
        wkset->point(0,0,s) = ip(e,j,s);
      }
      FDATA efdata = functionManagers[block]->evaluate(extrafields_list[block][fnum],"point");
      parallel_for("physics get extra fields",RangePolicy<AssemblyExec>(0,1), KOKKOS_LAMBDA (const int elem ) {
        fields(e,j) = efdata(0,0).val();
      });
    }
  }
  return fields;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT*,AssemblyDevice> physics::getExtraCellFields(const int & block,
                                                                  const int & fnum,
                                                                  Kokkos::View<ScalarT**,AssemblyDevice> wts) {
  
  int numElem = wts.extent(0);
  Kokkos::View<ScalarT*,AssemblyDevice> fields("cell field data",numElem);
  
  FDATA efdata = functionManagers[block]->evaluate(extracellfields_list[block][fnum],"ip");
  
  if (cellfield_reduction == "mean") { // default
    parallel_for("physics get extra cell fields",RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
      ScalarT cellmeas = 0.0;
      for (size_t pt=0; pt<wts.extent(1); pt++) {
        cellmeas += wts(e,pt);
      }
      for (size_t j=0; j<wts.extent(1); j++) {
        ScalarT val = efdata(e,j).val();
        fields(e) += val*wts(e,j)/cellmeas;
      }
    });
  }
  else if (cellfield_reduction == "max") {
    parallel_for("physics get extra cell fields",RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<wts.extent(1); j++) {
        ScalarT val = efdata(e,j).val();
        if (val>fields(e)) {
          fields(e) = val;
        }
      }
    });
  }
  if (cellfield_reduction == "min") {
    parallel_for("physics get extra cell fields",RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<wts.extent(1); j++) {
        ScalarT val = efdata(e,j).val();
        if (val<fields(e)) {
          fields(e) = val;
        }
      }
    });
  }
  
  return fields;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int physics::getUniqueIndex(const int & block, const std::string & var) {
  int index = 0;
  for (int j=0; j<numVars[block]; j++) {
    if (varlist[block][j] == var)
    index = unique_index[block][j];
  }
  return index;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::setBCData(Teuchos::RCP<Teuchos::ParameterList> & settings,
                        Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                        Teuchos::RCP<panzer::DOFManager> & DOF,
                        std::vector<std::vector<int> > cards) {
  
  Teuchos::TimeMonitor localtimer(*bctimer);
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics::setBCData ..." << endl;
    }
  }
  
  int maxvars = 0;
  for (size_t b=0; b<blocknames.size(); b++) {
    for (size_t j=0; j<varlist[b].size(); j++) {
      string var = varlist[b][j];
      int num = DOF->getFieldNum(var);
      maxvars = std::max(num,maxvars);
    }
  }
  
  for (size_t b=0; b<blocknames.size(); b++) {
    
    mesh->getSidesetNames(sideSets);
    mesh->getNodesetNames(nodeSets);
    
    Kokkos::View<int**,HostDevice> currbcs("boundary conditions",varlist[b].size(),sideSets.size());
    topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
    if (spaceDim == 1) {
      numSidesPerElem = 2;
    }
    if (spaceDim == 2) {
      numSidesPerElem = cellTopo->getEdgeCount();
    }
    if (spaceDim == 3) {
      numSidesPerElem = cellTopo->getFaceCount();
    }
    
    numNodesPerElem = cellTopo->getNodeCount();
    
    std::string blockID = blocknames[b];
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->getMyElements(blockID, stk_meshElems);
    size_t maxElemLID = 0;
    for (size_t i=0; i<stk_meshElems.size(); i++) {
      size_t lid = mesh->elementLocalId(stk_meshElems[i]);
      maxElemLID = std::max(lid,maxElemLID);
    }
    std::vector<size_t> localelemmap(maxElemLID+1);
    for (size_t i=0; i<stk_meshElems.size(); i++) {
      size_t lid = mesh->elementLocalId(stk_meshElems[i]);
      localelemmap[lid] = i;
    }
    
    numElem.push_back(stk_meshElems.size());
    
    Teuchos::ParameterList blocksettings;
    if (settings->sublist("Physics").isSublist(blockID)) {
      blocksettings = settings->sublist("Physics").sublist(blockID);
    }
    else {
      blocksettings = settings->sublist("Physics");
    }
    
    Teuchos::ParameterList dbc_settings = blocksettings.sublist("Dirichlet conditions");
    Teuchos::ParameterList nbc_settings = blocksettings.sublist("Neumann conditions");
    bool use_weak_dbcs = dbc_settings.get<bool>("use weak Dirichlet",false);
    int maxcard = 0;
    for (size_t j=0; j<cards[b].size(); j++) {
      if (cards[b][j] > maxcard)
      maxcard = cards[b][j];
    }
    
    vector<vector<int> > celloffsets;
    Kokkos::View<int****,HostDevice> currside_info("side info",numElem[b],numVars[b],numSidesPerElem,2);
    
    
    //std::vector<std::vector<size_t> > block_SideIDs, block_GlobalSideIDs;
    //std::vector<std::vector<size_t> > block_ElemIDs;
    std::vector<int> block_dbc_dofs;
    
    std::string perBCs = settings->sublist("Mesh").get<string>("Periodic Boundaries","");
    
    for (size_t j=0; j<varlist[b].size(); j++) {
      string var = varlist[b][j];
      int num = DOF->getFieldNum(var);
      vector<int> var_offsets = DOF->getGIDFieldOffsets(blockID,num);
      
      celloffsets.push_back(var_offsets);
      
      //vector<size_t> curr_SideIDs;
      //vector<size_t> curr_GlobalSideIDs;
      //vector<size_t> curr_ElemIDs;
      
      for( size_t side=0; side<sideSets.size(); side++ ) {
        string sideName = sideSets[side];
        
        vector<stk::mesh::Entity> sideEntities;
        mesh->getMySides(sideName, blockID, sideEntities);
        
        bool isDiri = false;
        //bool isPeri = false;
        bool isNeum = false;
        if (dbc_settings.sublist(var).isParameter("all boundaries") || dbc_settings.sublist(var).isParameter(sideName)) {
          isDiri = true;
          if (use_weak_dbcs) {
            currbcs(j,side) = 4;
          }
          else {
            currbcs(j,side) = 1;
          }
        }
        if (nbc_settings.sublist(var).isParameter("all boundaries") || nbc_settings.sublist(var).isParameter(sideName)) {
          isNeum = true;
          currbcs(j,side) = 2;
        }
        
        vector<size_t>             local_side_Ids;
        vector<stk::mesh::Entity> side_output;
        vector<size_t>             local_elem_Ids;
        panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities, local_side_Ids, side_output);
        
        for( size_t i=0; i<side_output.size(); i++ ) {
          local_elem_Ids.push_back(mesh->elementLocalId(side_output[i]));
          size_t localid = localelemmap[local_elem_Ids[i]];
          if( isDiri ) {
            //curr_SideIDs.push_back(local_side_Ids[i]);
            //curr_GlobalSideIDs.push_back(side);
            //curr_ElemIDs.push_back(localid);
            //curr_ElemIDs.push_back(local_elem_Ids[i]);
            if (use_weak_dbcs) {
              currside_info(localid, j, local_side_Ids[i], 0) = 4;
            }
            else {
              currside_info(localid, j, local_side_Ids[i], 0) = 1;
            }
            currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
          }
          else if (isNeum) { // Neumann or Robin
            currside_info(localid, j, local_side_Ids[i], 0) = 2;
            currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
          }
        }
      }
      //block_SideIDs.push_back(curr_SideIDs);
      //block_GlobalSideIDs.push_back(curr_GlobalSideIDs);
      
      //block_ElemIDs.push_back(curr_ElemIDs);
     
      
      // nodeset loop
      string point_DBCs = blocksettings.get<std::string>(var+"_point_DBCs","");
      
      vector<int> dbc_nodes;
      for( size_t node=0; node<nodeSets.size(); node++ ) {
        string nodeName = nodeSets[node];
        std::size_t found = point_DBCs.find(nodeName);
        bool isDiri = false;
        if (found!=std::string::npos) {
          isDiri = true;
        }
        
        if (isDiri) {
          //int num = DOF->getFieldNum(var);
          //vector<int> var_offsets = DOF->getGIDFieldOffsets(blockID,num);
          vector<stk::mesh::Entity> nodeEntities;
          mesh->getMyNodes(nodeName, blockID, nodeEntities);
          vector<GO> elemGIDs;
          
          vector<size_t> local_elem_Ids;
          vector<size_t> local_node_Ids;
          vector<stk::mesh::Entity> side_output;
          panzer_stk::workset_utils::getNodeElements(*mesh,blockID,nodeEntities,local_node_Ids,side_output);
          
          for( size_t i=0; i<side_output.size(); i++ ) {
            local_elem_Ids.push_back(mesh->elementLocalId(side_output[i]));
            size_t localid = localelemmap[local_elem_Ids[i]];
            DOF->getElementGIDs(localid,elemGIDs,blockID);
            block_dbc_dofs.push_back(elemGIDs[var_offsets[local_node_Ids[i]]]);
          }
        }
        
      }
    }
    
    offsets.push_back(celloffsets);
    var_bcs.push_back(currbcs);
    
    side_info.push_back(currside_info);
    //localDirichletSideIDs.push_back(block_SideIDs);
    //globalDirichletSideIDs.push_back(block_GlobalSideIDs);
    //boundDirichletElemIDs.push_back(block_ElemIDs);
    
    std::sort(block_dbc_dofs.begin(), block_dbc_dofs.end());
    block_dbc_dofs.erase(std::unique(block_dbc_dofs.begin(),
                                     block_dbc_dofs.end()), block_dbc_dofs.end());
    
    int localsize = (int)block_dbc_dofs.size();
    int globalsize = 0;
    
    //Teuchos::reduceAll<int, int>(*Commptr, Teuchos::REDUCE_SUM, localsize, Teuchos::outArg(globalsize));
    Teuchos::reduceAll<int,int>(*Commptr,Teuchos::REDUCE_SUM,1,&localsize,&globalsize);
    //Commptr->SumAll(&localsize, &globalsize, 1);
    int gathersize = Commptr->getSize()*globalsize;
    int *block_dbc_dofs_local = new int [globalsize];
    int *block_dbc_dofs_global = new int [gathersize];
    
    int mxdof = (int) block_dbc_dofs.size();
    for (int i = 0; i < globalsize; i++) {
      if ( i < mxdof) {
        block_dbc_dofs_local[i] = (int) block_dbc_dofs[i];
      }
      else {
        block_dbc_dofs_local[i] = -1;
      }
    }
    
    //Commptr->GatherAll(block_dbc_dofs_local, block_dbc_dofs_global, globalsize);
    Teuchos::gatherAll(*Commptr, globalsize, &block_dbc_dofs_local[0], gathersize, &block_dbc_dofs_global[0]);
    vector<GO> all_dbcs;
    
    for (int i = 0; i < gathersize; i++) {
      all_dbcs.push_back(block_dbc_dofs_global[i]);
    }
    delete [] block_dbc_dofs_local;
    delete [] block_dbc_dofs_global;
    
    vector<GO> dbc_final;
    vector<GO> ownedAndShared;
    DOF->getOwnedAndGhostedIndices(ownedAndShared);
    
    sort(all_dbcs.begin(),all_dbcs.end());
    sort(ownedAndShared.begin(),ownedAndShared.end());
    set_intersection(all_dbcs.begin(),all_dbcs.end(),
                     ownedAndShared.begin(),ownedAndShared.end(),
                     back_inserter(dbc_final));
    
    
    point_dofs.push_back(dbc_final);
    //offsets.push_back(curroffsets);
    
  }
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics::setBCData" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::setDirichletData(Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                               Teuchos::RCP<panzer::DOFManager> & DOF) {
  
  Teuchos::TimeMonitor localtimer(*dbctimer);
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics::setDirichletData ..." << endl;
    }
  }
  
  haveDirichlet = false;
  for (size_t b=0; b<blocknames.size(); b++) {
    
    std::string blockID = blocknames[b];
    
    Teuchos::ParameterList dbc_settings;
    if (settings->sublist("Physics").isSublist(blockID)) {
      dbc_settings = settings->sublist("Physics").sublist(blockID).sublist("Dirichlet conditions");
    }
    else {
      dbc_settings = settings->sublist("Physics").sublist("Dirichlet conditions");
    }
    
    //std::vector<Kokkos::View<LO*,AssemblyDevice> > block_dbc_dofs;
    std::vector<std::vector<LO> > block_dbc_dofs;
    
    for (size_t j=0; j<varlist[b].size(); j++) {
      std::string var = varlist[b][j];
      int fieldnum = DOF->getFieldNum(var);
      std::vector<LO> var_dofs;
      for (size_t side=0; side<sideNames.size(); side++ ) {
        std::string sideName = sideNames[side];
        vector<stk::mesh::Entity> sideEntities;
        mesh->getMySides(sideName, blockID, sideEntities);
        
        bool isDiri = false;
        if (dbc_settings.sublist(var).isParameter("all boundaries") || dbc_settings.sublist(var).isParameter(sideName)) {
          isDiri = true;
          haveDirichlet = true;
        }
        
        if (isDiri) {
          
          vector<size_t>             local_side_Ids;
          vector<stk::mesh::Entity>  side_output;
          vector<size_t>             local_elem_Ids;
          panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities,
                                                     local_side_Ids, side_output);
        
          for( size_t i=0; i<side_output.size(); i++ ) {
            LO local_EID = mesh->elementLocalId(side_output[i]);
            auto elemLIDs = DOF->getElementLIDs(local_EID);
            const std::pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldnum,
                                                                                                  spaceDim-1,
                                                                                                  local_side_Ids[i]);
            const vector<int> sideOffset = SideIndex.first;
            
            for( size_t i=0; i<sideOffset.size(); i++ ) { // for each node
              var_dofs.push_back(elemLIDs(sideOffset[i]));
            }
          }
        }
      }
      std::sort(var_dofs.begin(), var_dofs.end());
      var_dofs.erase(std::unique(var_dofs.begin(), var_dofs.end()), var_dofs.end());
      
      //Kokkos::View<LO*,AssemblyDevice> var_dofs_kv("dbc dofs on block",var_dofs.size());
      //auto var_dofs_host = Kokkos::create_mirror_view(var_dofs_kv);
      //for (size_t k=0; k<var_dofs.size(); k++) {
      //  var_dofs_host(k) = var_dofs[k];
      //}
      //Kokkos::deep_copy(var_dofs_kv, var_dofs_host);
      block_dbc_dofs.push_back(var_dofs);
    }
    
    dbc_dofs.push_back(block_dbc_dofs);
    
  }
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics::setDirichletData" << endl;
    }
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<int****,HostDevice> physics::getSideInfo(const size_t & block,
                                                     Kokkos::View<int*,HostDevice> elem) {
  
  Teuchos::TimeMonitor localtimer(*sideinfotimer);
  
  size_type nelem = elem.extent(0);
  size_type nvars = side_info[block].extent(1);
  size_type nelemsides = side_info[block].extent(2);
  //size_type nglobalsides = side_info[block].extent(3);
  Kokkos::View<int****,HostDevice> currsi("side info for cell",nelem,nvars,nelemsides, 2);
  for (size_type e=0; e<nelem; e++) {
    for (size_type j=0; j<nelemsides; j++) {
      for (size_type i=0; i<nvars; i++) {
        int sidetype = side_info[block](elem(e),i,j,0);
        if (sidetype > 0) { // TMW: why is this here?
          currsi(e,i,j,0) = sidetype;
          currsi(e,i,j,1) = side_info[block](elem(e),i,j,1);
        }
        else {
          currsi(e,i,j,0) = sidetype;
          currsi(e,i,j,1) = 0;
        }
      }
    }
  }
  return currsi;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<vector<int> > physics::getOffsets(const int & block, Teuchos::RCP<panzer::DOFManager> & DOF) {
  return offsets[block];
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<int**,HostDevice> physics::getSideInfo(const int & block, int & num, size_t & e) {
  Kokkos::View<int**,HostDevice> local_side_info = Kokkos::subview(side_info[block],e,num,Kokkos::ALL(),Kokkos::ALL());
  /*
  for (int j=0; j<numSidesPerElem; j++) {
    for (int k=0; k<2; k++) {
      Teuchos::Array<int> fcindex(4);
      local_side_info(j,k) = side_info[block](e,num,j,k);//(e,num,j,k);
    }
  }
  */
  return local_side_info;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::volumeResidual(const size_t block) {
  for (size_t i=0; i<modules[block].size(); i++) {
    if (useSubgrid[block][i] == false) {
      modules[block][i]->volumeResidual();
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::boundaryResidual(const size_t block) {
  for (size_t i=0; i<modules[block].size(); i++) {
    modules[block][i]->boundaryResidual();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::computeFlux(const size_t block) {
  for (size_t i=0; i<modules[block].size(); i++) {
    modules[block][i]->computeFlux();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::setWorkset(vector<Teuchos::RCP<workset> > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++){
    for (size_t i=0; i<modules[block].size(); i++) {
      modules[block][i]->setWorkset(wkset[block]);//setWorkset(wkset[block]);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::faceResidual(const size_t block) {
  for (size_t i=0; i<modules[block].size(); i++) {
    if (useSubgrid[block][i] == false) {
      modules[block][i]->faceResidual();
    }
  }
}
