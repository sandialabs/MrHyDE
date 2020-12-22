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
  
  this->importPhysics(false);
  
  if (settings->isSublist("Aux Physics") && settings->isSublist("Aux Discretization")) {
    have_aux = true;
    for (size_t b=0; b<blocknames.size(); b++) {
      if (settings->sublist("Aux Physics").isSublist(blocknames[b])) { // adding block overwrites the default
        aux_blockPhysSettings.push_back(settings->sublist("Aux Physics").sublist(blocknames[b]));
      }
      else { // default
        aux_blockPhysSettings.push_back(settings->sublist("Aux Physics"));
      }
      
      if (settings->sublist("Aux Discretization").isSublist(blocknames[b])) { // adding block overwrites default
        aux_blockDiscSettings.push_back(settings->sublist("Aux Discretization").sublist(blocknames[b]));
      }
      else { // default
        aux_blockDiscSettings.push_back(settings->sublist("Aux Discretization"));
      }
    }
    this->importPhysics(true);
  }
  else {
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<string> avars;
      aux_varlist.push_back(avars);
    }
  }

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

void physics::importPhysics(const bool & isaux) {
  
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
    string module_list;
    if (isaux) {
      module_list = aux_blockPhysSettings[b].get<string>("modules","");
    }
    else {
      module_list = blockPhysSettings[b].get<string>("modules","");
    }
    
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
        Teuchos::RCP<porous> porous_RCP = Teuchos::rcp(new porous(settings, isaux) );
        currmodules.push_back(porous_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_porous",false));
      }
    
      // Porous media with HDIV basis
      if (modname == "porousHDIV") {
        Teuchos::RCP<porousHDIV> porousHDIV_RCP = Teuchos::rcp(new porousHDIV(settings, isaux) );
        currmodules.push_back(porousHDIV_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_porousHDIV",false));
      }
      
      // Hybridized porous media with HDIV basis
      if (modname == "porousHDIV_hybrid") {
        Teuchos::RCP<porousHDIV_HYBRID> porousHDIV_HYBRID_RCP = Teuchos::rcp(new porousHDIV_HYBRID(settings, isaux) );
        currmodules.push_back(porousHDIV_HYBRID_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_porousHDIV_HYBRID",false));
      }
      
      // weak Galerkin porous media with HDIV basis
      if (modname == "porousHDIV_weakGalerkin") {
        Teuchos::RCP<porousHDIV_WG> porousHDIV_WG_RCP = Teuchos::rcp(new porousHDIV_WG(settings, isaux) );
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
        Teuchos::RCP<cdr> cdr_RCP = Teuchos::rcp(new cdr(settings, isaux) );
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
        Teuchos::RCP<thermal> thermal_RCP = Teuchos::rcp(new thermal(settings, isaux) );
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
        Teuchos::RCP<shallowwater> shallowwater_RCP = Teuchos::rcp(new shallowwater(settings, isaux) );
        currmodules.push_back(shallowwater_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_shallowwater",false));
      }
      
      // Maxwell
      if (modname == "maxwell") {
        Teuchos::RCP<maxwell> maxwell_RCP = Teuchos::rcp(new maxwell(settings, isaux) );
        currmodules.push_back(maxwell_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_maxwell",false));
      }
      
      // Maxwell hybridized
      if (modname == "maxwell hybrid") {
        Teuchos::RCP<maxwell_HYBRID> maxwell_HYBRID_RCP = Teuchos::rcp(new maxwell_HYBRID(settings, isaux) );
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
        Teuchos::RCP<msphasefield> msphasefield_RCP = Teuchos::rcp(new msphasefield(settings, isaux, Commptr) );
        currmodules.push_back(msphasefield_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_msphasefield",false));
      }
      
      // Stokes
      if (modname == "stokes" || modname == "Stokes") {
        Teuchos::RCP<stokes> stokes_RCP = Teuchos::rcp(new stokes(settings, isaux) );
        
        currmodules.push_back(stokes_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_stokes",false));
      }
      
      // Navier Stokes
      if (modname == "navier stokes" || modname == "Navier Stokes") {
        Teuchos::RCP<navierstokes> navierstokes_RCP = Teuchos::rcp(new navierstokes(settings, isaux) );
        
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
        Teuchos::RCP<linearelasticity> linearelasticity_RCP = Teuchos::rcp(new linearelasticity(settings, isaux) );
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
        Teuchos::RCP<helmholtz> helmholtz_RCP = Teuchos::rcp(new helmholtz(settings, isaux) );
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
        Teuchos::RCP<maxwells_fp> maxwells_fp_RCP = Teuchos::rcp(new maxwells_fp(settings, isaux) );
        currmodules.push_back(maxwells_fp_RCP);
        currSubgrid.push_back(blockPhysSettings[b].get<bool>("subgrid_maxwells_freq_pot",false));
      }
      
      // Scalar ODE for testing time integrators independent of spatial discretizations
      if (modname == "ODE"){
        Teuchos::RCP<ODE> ODE_RCP = Teuchos::rcp(new ODE(settings, isaux) );
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
    
    if (isaux) {
      aux_modules.push_back(currmodules);
    }
    else {
      modules.push_back(currmodules);
      useSubgrid.push_back(currSubgrid);
    }
    
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
    if (isaux) {
      aux_orders.push_back(currorders);
      aux_types.push_back(currtypes);
      aux_varlist.push_back(currvarlist);
      aux_varowned.push_back(currvarowned);
      aux_numVars.push_back(currnumVars);
      aux_useDG.push_back(curruseDG);
      aux_unique_orders.push_back(currunique_orders);
      aux_unique_types.push_back(currunique_types);
      aux_unique_index.push_back(currunique_index);
    }
    else {
      orders.push_back(currorders);
      types.push_back(currtypes);
      varlist.push_back(currvarlist);
      varowned.push_back(currvarowned);
      numVars.push_back(currnumVars);
      useDG.push_back(curruseDG);
      unique_orders.push_back(currunique_orders);
      unique_types.push_back(currunique_types);
      unique_index.push_back(currunique_index);
    }
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
  auto ddata = functionManagers[block]->evaluate("Dirichlet " + var + " " + gside,"point");
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
  View_AD2_sv idata = functionManager->evaluate("initial " + var,"point",block);
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

View_AD3 physics::getPointResponse(const int & block, View_AD4 u_ip, View_AD4 ugrad_ip,
                                   View_AD4 p_ip, View_AD4 pgrad_ip,
                                   const DRV ip, const ScalarT & time,
                                   Teuchos::RCP<workset> & wkset) {
  
  size_t numElem = u_ip.extent(0);
  size_t numip = ip.extent(1);
  size_t numResponses = response_list[block].size();
  
  View_AD3 responsetotal("responses",numElem,numResponses,numip);
  
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
        auto rdata = functionManagers[block]->evaluate(response_list[block][r],"point");
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

View_AD3 physics::getResponse(const int & block, View_AD4 u_ip, View_AD4 ugrad_ip,
                              View_AD4 p_ip, View_AD4 pgrad_ip,
                              const View_Sc3 ip,
                              const ScalarT & time,
                              Teuchos::RCP<workset> & wkset) {
  
  size_t numElem = u_ip.extent(0);
  size_t numip = ip.extent(1);
  size_t numResponses = response_list[block].size();
  
  View_AD3 responsetotal("responses",numElem,numResponses,numip);
  
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
    auto rdata = functionManagers[block]->evaluate(response_list[block][r],"ip");
    
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

View_AD3 physics::target(const int & block, const View_Sc3 ip,
                         const ScalarT & current_time,
                         Teuchos::RCP<workset> & wkset) {
  
  View_AD3 targettotal("target",ip.extent(0), target_list[block].size(),ip.extent(1));
  
  for (size_t t=0; t<target_list[block].size(); t++) {
    auto tdata = functionManagers[block]->evaluate(target_list[block][t],"ip");
    auto ctarg = Kokkos::subview(targettotal,Kokkos::ALL(), t, Kokkos::ALL());
    Kokkos::deep_copy(ctarg,tdata);
  }
  return targettotal;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_AD3 physics::weight(const int & block, const View_Sc3 ip,
                         const ScalarT & current_time,
                         Teuchos::RCP<workset> & wkset) {
  
  View_AD3 weighttotal("weight",ip.extent(0), weight_list[block].size(),ip.extent(1));
  
  for (size_t t=0; t<weight_list[block].size(); t++) {
    auto wdata = functionManagers[block]->evaluate(weight_list[block][t],"ip");
    auto cwt = Kokkos::subview(weighttotal,Kokkos::ALL(), t, Kokkos::ALL());
    Kokkos::deep_copy(cwt,wdata);
  }
  return weighttotal;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc3 physics::getInitial(const View_Sc3 ip, const int & block,
                             const bool & project, Teuchos::RCP<workset> & wkset) {
  
  
  size_t numElem = ip.extent(0);
  size_t numVars = varlist[block].size();
  size_t numip = ip.extent(1);
  
  View_Sc3 ivals("temp invals", numElem, numVars, numip);
  
  if (project) {
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<varlist[block].size(); n++) {
  
      auto ivals_AD = functionManagers[block]->evaluate("initial " + varlist[block][n],"ip");
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
    auto point_KV = wkset->point;
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
          auto ivals_AD = functionManagers[block]->evaluate("initial " + varlist[block][n],"point");
          
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

View_Sc2 physics::getDirichlet(const View_Sc3 ip, const int & var,
                               const int & block,
                               const std::string & sidename,
                               Teuchos::RCP<workset> & wkset) {
  
  
  size_t numElem = ip.extent(0);
  size_t numip = ip.extent(1);
  
  View_Sc2 dvals("temp dnvals", numElem, numip);
  
  // evaluate
  auto dvals_AD = functionManagers[block]->evaluate("Dirichlet " + varlist[block][var] + " " + sidename,"side ip");
  
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

void physics::setVars() {
  for (size_t block=0; block<modules.size(); ++block) {
    for (size_t i=0; i<modules[block].size(); ++i) {
      if (varlist[block].size() > 0){
        modules[block][i]->setVars(varlist[block]);
      }
    }
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

View_Sc2 physics::getExtraFields(const int & block, const int & fnum,
                                 const DRV & ip, const ScalarT & time,
                                 Teuchos::RCP<workset> & wkset) {
  
  View_Sc2 fields("field data",ip.extent(0),ip.extent(1));
  
  for (size_type e=0; e<ip.extent(0); e++) {
    for (size_type j=0; j<ip.extent(1); j++) {
      for (int s=0; s<spaceDim; s++) {
        wkset->point(0,0,s) = ip(e,j,s);
      }
      auto eView_AD2_sv = functionManagers[block]->evaluate(extrafields_list[block][fnum],"point");
      parallel_for("physics get extra fields",RangePolicy<AssemblyExec>(0,1), KOKKOS_LAMBDA (const int elem ) {
        fields(e,j) = eView_AD2_sv(0,0).val();
      });
    }
  }
  return fields;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc1 physics::getExtraCellFields(const int & block, const int & fnum, View_Sc2 wts) {
  
  int numElem = wts.extent(0);
  View_Sc1 fields("cell field data",numElem);
  
  auto ecf = functionManagers[block]->evaluate(extracellfields_list[block][fnum],"ip");
  
  if (cellfield_reduction == "mean") { // default
    parallel_for("physics get extra cell fields",RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
      ScalarT cellmeas = 0.0;
      for (size_t pt=0; pt<wts.extent(1); pt++) {
        cellmeas += wts(e,pt);
      }
      for (size_t j=0; j<wts.extent(1); j++) {
        ScalarT val = ecf(e,j).val();
        fields(e) += val*wts(e,j)/cellmeas;
      }
    });
  }
  else if (cellfield_reduction == "max") {
    parallel_for("physics get extra cell fields",RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<wts.extent(1); j++) {
        ScalarT val = ecf(e,j).val();
        if (val>fields(e)) {
          fields(e) = val;
        }
      }
    });
  }
  if (cellfield_reduction == "min") {
    parallel_for("physics get extra cell fields",RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<wts.extent(1); j++) {
        ScalarT val = ecf(e,j).val();
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

//vector<vector<int> > physics::getOffsets(const int & block, Teuchos::RCP<panzer::DOFManager> & DOF) {
//  return offsets[block];
//}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

//Kokkos::View<int**,HostDevice> physics::getSideInfo(const int & block, int & num, size_t & e) {
//  Kokkos::View<int**,HostDevice> local_side_info = Kokkos::subview(side_info[block],e,num,Kokkos::ALL(),Kokkos::ALL());
  /*
  for (int j=0; j<numSidesPerElem; j++) {
    for (int k=0; k<2; k++) {
      Teuchos::Array<int> fcindex(4);
      local_side_info(j,k) = side_info[block](e,num,j,k);//(e,num,j,k);
    }
  }
  */
//  return local_side_info;
//}

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
