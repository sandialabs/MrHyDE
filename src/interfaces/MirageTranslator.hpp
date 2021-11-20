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

#ifndef MIRAGETRANSLATOR_H
#define MIRAGETRANSLATOR_H

#include "trilinos.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

#include "preferences.hpp"

namespace MrHyDE {
  
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  // Standard constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  void MirageTranslator(Teuchos::RCP<Teuchos::ParameterList> & settings, const std::string & filename) {
    
    using Teuchos::RCP;
    using Teuchos::rcp;
        
    //////////////////////////////////////////////////////////////////////////////////////////
    // Import the main input.xml file
    //////////////////////////////////////////////////////////////////////////////////////////
    
    RCP<Teuchos::ParameterList> mirage_settings = rcp(new Teuchos::ParameterList("FEM3"));
    
    std::ifstream fnmast(filename.c_str());
    if (fnmast.good()) {
      Teuchos::RCP<Teuchos::ParameterList> main_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
      Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*main_parlist) );
      mirage_settings->setParameters( *main_parlist );
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: MrHyDE could not find the FEM3 input file: " + filename);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Import the mirage mesh settings
    //////////////////////////////////////////////////////////////////////////////////////////
    
    if (mirage_settings->isParameter("Mesh xml")) {
      std::string filename = mirage_settings->get<std::string>("Mesh xml");
      std::ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> mesh_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*mesh_parlist) );
        mirage_settings->sublist("Mesh").setParameters( *mesh_parlist );
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the mesh settings file: " + filename);
      }
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the FEM3.xml needs to contain a path to a mesh settings file!");
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Import the mirage closure model settings
    //////////////////////////////////////////////////////////////////////////////////////////
    
    if (mirage_settings->isParameter("Closure model xml")) {
      std::string filename = mirage_settings->get<std::string>("Closure model xml");
      std::ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> cm_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*cm_parlist) );
        mirage_settings->sublist("Closure Models").setParameters( *cm_parlist );
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the closure model file: " + filename);
      }
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the FEM3.xml needs to contain a path to a closure model file!");
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Import the mirage solver settings (not currently used though)
    //////////////////////////////////////////////////////////////////////////////////////////
    
    /*
    if (mirage_settings->isParameter("Linear solver xml")) {
      std::string filename = mirage_settings->get<std::string>("Linear solver xml");
      std::ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> ls_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*ls_parlist) );
        mirage_settings->setParameters( *ls_parlist );
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the linear solver file: " + filename);
      }
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the FEM3.xml needs to contain a path to a linear solver file!");
    }
    */
    
    mirage_settings->print();
    
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // We need to translate the settings in the mirage plist into the appropriate settings for MrHyDE
    // This involves defining the following blocks:
    //   1. Mesh
    //   2. Physics (including multi-block problems and closure models)
    //   3. Discretization
    //   4. Function
    //   5. Parameters (optional)
    //   6. Solver (mostly ignores the mirage settings)
    //   7. Analysis
    //   8. Postprocess
    //////////////////////////////////////////////////////////////////////////////////////////
    
    bool use_explicit = mirage_settings->sublist("MrHyDE Options").get<bool>("Use explicit integration",true);
    
    //----------------------------------------
    // Mesh block
    //----------------------------------------
    
    Teuchos::RCP<Teuchos::ParameterList> mesh_list = Teuchos::rcp( new Teuchos::ParameterList() );
    
    int mirage_dim = mirage_settings->sublist("Mesh").get<int>("spatialDim",3);
    
    mesh_list->set("dimension",mirage_dim);
    if (mirage_settings->sublist("Mesh").get<string>("filename","") != "") { // read in mesh if provided
      mesh_list->set("mesh file", mirage_settings->sublist("Mesh").get<string>("filename",""));
      mesh_list->set("source", "Exodus");
    }
    else { // otherwise build panzer inline mesh
      mesh_list->set("xmin",mirage_settings->sublist("Mesh").get<double>("x-min",0.0));
      mesh_list->set("xmax",mirage_settings->sublist("Mesh").get<double>("x-max",1.0));
      mesh_list->set("Xblocks",mirage_settings->sublist("Mesh").get<int>("x-blocks",1));
      mesh_list->set("Xprocs",mirage_settings->sublist("Mesh").get<int>("x-procs",1));
      mesh_list->set("NX",mirage_settings->sublist("Mesh").get<int>("x-elements",2));
      if (mirage_dim == 1) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: neither code runs Maxwells in 1D.");
      }
      if (mirage_dim > 1) {
        mesh_list->set("ymin",mirage_settings->sublist("Mesh").get<double>("y-min",0.0));
        mesh_list->set("ymax",mirage_settings->sublist("Mesh").get<double>("y-max",1.0));
        mesh_list->set("Yblocks",mirage_settings->sublist("Mesh").get<int>("y-blocks",1));
        mesh_list->set("Yprocs",mirage_settings->sublist("Mesh").get<int>("y-procs",1));
        mesh_list->set("NY",mirage_settings->sublist("Mesh").get<int>("y-elements",2));
      }
      if (mirage_dim > 2) {
        mesh_list->set("zmin",mirage_settings->sublist("Mesh").get<double>("z-min",0.0));
        mesh_list->set("zmax",mirage_settings->sublist("Mesh").get<double>("z-max",1.0));
        mesh_list->set("Zblocks",mirage_settings->sublist("Mesh").get<int>("z-blocks",1));
        mesh_list->set("Zprocs",mirage_settings->sublist("Mesh").get<int>("z-procs",1));
        mesh_list->set("NZ",mirage_settings->sublist("Mesh").get<int>("z-elements",2));
      }
    }
    
    if (mirage_settings->sublist("Mesh").get<bool>("build-tet-mesh",false)) {
      
      if (mirage_dim == 2) {
        mesh_list->set("shape","tri");
      }
      else {
        mesh_list->set("shape","tet");
      }
    }
    else {
      if (mirage_dim == 2) {
        mesh_list->set("shape","quad");
      }
      else {
        mesh_list->set("shape","hex");
      }
    }
    
    if (mirage_settings->sublist("Mesh").isSublist("Periodic BCs")) {
      Teuchos::ParameterList pbcs = mirage_settings->sublist("Mesh").sublist("Periodic BCs");
      mesh_list->sublist("Periodic BCs").setParameters(pbcs);
    }
    
    settings->sublist("Mesh").setParameters(*mesh_list);
    
    // Need to determine the list of block names
    // An empty indicates that all settings apply to all blocks
    vector<string> blocknames;
    if (mirage_settings->sublist("Closure Models").isSublist("Mapping to Blocks")) {
      Teuchos::ParameterList mappings = mirage_settings->sublist("Closure Models").sublist("Mapping to Blocks");
      Teuchos::ParameterList::ConstIterator m_itr = mappings.begin();
      while (m_itr != mappings.end()) {
        blocknames.push_back(m_itr->first);
        m_itr++;
      }
    }
    
    //----------------------------------------
    // Physics block
    //----------------------------------------
    
    Teuchos::RCP<Teuchos::ParameterList> physics_list = Teuchos::rcp( new Teuchos::ParameterList() );
    
    double maxperm = -1.0;
    double minperm = -1.0;
    
    if (blocknames.size() == 0) {
      physics_list->set("modules","mirage");
      physics_list->set("use fully explicit",use_explicit);
      if (mirage_settings->sublist("MrHyDE Options").get<string>("Butcher tableau","leap-frog") == "leap-frog") {
        physics_list->set("use leap-frog",true);
      }
      else {
        physics_list->set("use leap-frog",false);
      }
      if (mirage_settings->sublist("Closure Models").isSublist("electromagnetics")) {
        physics_list->sublist("Mirage settings").setParameters(mirage_settings->sublist("Closure Models").sublist("electromagnetics"));
      }
      else if (mirage_settings->sublist("Closure Models").isSublist("electromagnetics0")) {
        physics_list->sublist("Mirage settings").setParameters(mirage_settings->sublist("Closure Models").sublist("electromagnetics0"));
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: could not find the closure model settings");
      }
      physics_list->sublist("Initial Conditions").set<bool>("scalar data",true);
      physics_list->sublist("Initial Conditions").set<double>("E",0.0);
      physics_list->sublist("Initial Conditions").set<double>("B",0.0);
      
      double epsval = 1.0e-11;
      if (physics_list->sublist("Mirage settings").sublist("PERMITTIVITY").isParameter("Value") ) {
        epsval = physics_list->sublist("Mirage settings").sublist("PERMITTIVITY").get<double>("Value");
      }
      else if (physics_list->sublist("Mirage settings").sublist("PERMITTIVITY").isParameter("epsilon") ) {
        epsval = physics_list->sublist("Mirage settings").sublist("PERMITTIVITY").get<double>("epsilon");
      }
      
      if (minperm < 0 || epsval<minperm) {
        minperm = epsval;
      }
      if (maxperm < 0 || epsval>maxperm) {
        maxperm = epsval;
      }
      
      double rival = 1.0;
      if (physics_list->sublist("Mirage settings").isSublist("REFRACTIVE_INDEX")) {
        if (physics_list->sublist("Mirage settings").sublist("REFRACTIVE_INDEX").isParameter("Value") ) {
          rival = physics_list->sublist("Mirage settings").sublist("REFRACTIVE_INDEX").get<double>("Value");
        }
      }
      
      physics_list->sublist("Mass weights").set<double>("E",rival*rival);
      physics_list->sublist("Mass weights").set<double>("B",1.0);
      
      double invmuval = physics_list->sublist("Mirage settings").sublist("INVERSE_PERMEABILITY").get<double>("Value");
      physics_list->sublist("Norm weights").set<double>("E",epsval);
      physics_list->sublist("Norm weights").set<double>("B",invmuval);
    }
    else {
      for (size_t b=0; b<blocknames.size(); ++b) {
        string cmname = mirage_settings->sublist("Closure Models").sublist("Mapping to Blocks").get<string>(blocknames[b]);
        physics_list->sublist(blocknames[b]).set("modules","mirage");
        physics_list->sublist(blocknames[b]).set("use fully explicit",use_explicit);
        if (mirage_settings->sublist("MrHyDE Options").get<string>("Butcher tableau","leap-frog") == "leap-frog") {
          physics_list->sublist(blocknames[b]).set("use leap-frog",true);
        }
        else {
          physics_list->sublist(blocknames[b]).set("use leap-frog",false);
        }
        
        physics_list->sublist(blocknames[b]).sublist("Mirage settings").setParameters(mirage_settings->sublist("Closure Models").sublist(cmname));
        physics_list->sublist(blocknames[b]).sublist("Initial Conditions").set<bool>("scalar data",true);
        physics_list->sublist(blocknames[b]).sublist("Initial Conditions").set<double>("E",0.0);
        physics_list->sublist(blocknames[b]).sublist("Initial Conditions").set<double>("B",0.0);
        
        double epsval = 1.0e-11;
        if (physics_list->sublist(blocknames[b]).sublist("Mirage settings").sublist("PERMITTIVITY").isParameter("Value") ) {
          epsval = physics_list->sublist(blocknames[b]).sublist("Mirage settings").sublist("PERMITTIVITY").get<double>("Value");
        }
        else if (physics_list->sublist(blocknames[b]).sublist("Mirage settings").sublist("PERMITTIVITY").isParameter("epsilon") ) {
          epsval = physics_list->sublist(blocknames[b]).sublist("Mirage settings").sublist("PERMITTIVITY").get<double>("epsilon");
        }
        
        if (minperm < 0 || epsval<minperm) {
          minperm = epsval;
        }
        if (maxperm < 0 || epsval>maxperm) {
          maxperm = epsval;
        }
        
        double rival = 1.0;
        if (physics_list->sublist(blocknames[b]).sublist("Mirage settings").isSublist("REFRACTIVE_INDEX")) {
          if (physics_list->sublist(blocknames[b]).sublist("Mirage settings").sublist("REFRACTIVE_INDEX").isParameter("Value") ) {
            rival = physics_list->sublist(blocknames[b]).sublist("Mirage settings").sublist("REFRACTIVE_INDEX").get<double>("Value");
          }
        }
        
        physics_list->sublist(blocknames[b]).sublist("Mass weights").set<double>("E",rival*rival);
        physics_list->sublist(blocknames[b]).sublist("Mass weights").set<double>("B",1.0);
        
        double invmuval = physics_list->sublist(blocknames[b]).sublist("Mirage settings").sublist("INVERSE_PERMEABILITY").get<double>("Value");
        physics_list->sublist(blocknames[b]).sublist("Norm weights").set<double>("E",epsval);
        physics_list->sublist(blocknames[b]).sublist("Norm weights").set<double>("B",invmuval);
      }
    }
    
    // Safeguard against using discontinuous permittivity
    if (maxperm/minperm > 1.1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: discontinuous permittivities detected.  Please use REFRACTIVE_INDEX to define multiple materials.");
    }
    
    settings->sublist("Physics").setParameters(*physics_list);
    
    //----------------------------------------
    // Discretization block
    //----------------------------------------
    
    Teuchos::RCP<Teuchos::ParameterList> disc_list = Teuchos::rcp( new Teuchos::ParameterList() );
    
    if (blocknames.size() == 0) {
      disc_list->set<int>("quadrature",2);
      disc_list->sublist("order").set<int>("E",mirage_settings->sublist("Solver Options").get<int>("Basis order",1));
      if (mirage_dim == 2) {
        disc_list->sublist("order").set<int>("B",0);
      }
      else {
        disc_list->sublist("order").set<int>("B",mirage_settings->sublist("Solver Options").get<int>("Basis order",1));
      }
    }
    else {
      for (size_t b=0; b<blocknames.size(); ++b) {
        disc_list->sublist(blocknames[b]).set<int>("quadrature",2);
        disc_list->sublist(blocknames[b]).sublist("order").set<int>("E",mirage_settings->sublist("Solver Options").get<int>("Basis order",1));
        if (mirage_dim == 2) {
          disc_list->sublist("order").set<int>("B",0);
        }
        else {
          disc_list->sublist(blocknames[b]).sublist("order").set<int>("B",mirage_settings->sublist("Solver Options").get<int>("Basis order",1));
        }
      }
    }
    
    settings->sublist("Discretization").setParameters(*disc_list);
    
    //----------------------------------------
    // Functions block
    //----------------------------------------
    
    Teuchos::RCP<Teuchos::ParameterList> functions_list = Teuchos::rcp( new Teuchos::ParameterList() );
    settings->sublist("Functions").setParameters(*functions_list);
        
    //----------------------------------------
    // Parameter block
    //----------------------------------------
    
    Teuchos::RCP<Teuchos::ParameterList> params_list = Teuchos::rcp( new Teuchos::ParameterList() );
    settings->sublist("Parameters").setParameters(*params_list);
    
    //----------------------------------------
    // Solver block
    // For now, we are assuming that explicit integration has been requested somehow
    //----------------------------------------
    settings->sublist("Solver").set<string>("solver","transient");
    settings->sublist("Solver").set<int>("transient BDF order",1);
    settings->sublist("Solver").set<int>("workset size",mirage_settings->sublist("Discretization Options").get<int>("Workset size",100));
    settings->sublist("Solver").set<double>("final time",mirage_settings->sublist("Discretization Options").get<double>("Final time",0.0));
    settings->sublist("Solver").set<int>("number of steps",mirage_settings->sublist("Discretization Options").get<int>("Num time steps",1));
  
    // The explicit solver may use a mass solve, so we check these parameters for both implicit and explicit
    settings->sublist("Solver").set<double>("linear TOL",
                                            mirage_settings->sublist("MrHyDE Options").get<double>("linear TOL",1.0e-7));
    settings->sublist("Solver").set<int>("max linear iters",
                                         mirage_settings->sublist("MrHyDE Options").get<int>("max linear iters",50));
    settings->sublist("Solver").set<bool>("use direct solver",
                                          mirage_settings->sublist("MrHyDE Options").get<bool>("use direct solver",false));
    settings->sublist("Solver").set<bool>("reuse preconditioner",
                                          mirage_settings->sublist("MrHyDE Options").get<bool>("reuse preconditioner",true));
    settings->sublist("Solver").set<bool>("reuse Jacobian",
                                          mirage_settings->sublist("MrHyDE Options").get<bool>("reuse Jacobian",true));
    settings->sublist("Solver").set<string>("preconditioner reuse type",mirage_settings->sublist("MrHyDE Options").get<string>("preconditioner reuse type","full"));
    settings->sublist("Solver").set<string>("Belos implicit residual scaling","Norm of Initial Residual");
    
    if (use_explicit) {
      settings->sublist("Solver").set<string>("transient Butcher tableau",mirage_settings->sublist("MrHyDE Options").get<string>("Butcher tableau","leap-frog"));
      settings->sublist("Solver").set<bool>("lump mass",mirage_settings->sublist("MrHyDE Options").get<bool>("lump mass",true));
      settings->sublist("Solver").set<string>("Belos solver",
                                              mirage_settings->sublist("MrHyDE Options").get<string>("Belos solver","Block CG"));
      settings->sublist("Solver").set<bool>("use preconditioner",
                                            mirage_settings->sublist("MrHyDE Options").get<bool>("use preconditioner",true));
      settings->sublist("Solver").set<bool>("use custom PCG",mirage_settings->sublist("MrHyDE Options").get<bool>("use custom PCG",false));
      settings->sublist("Solver").set<string>("preconditioner type",
                                              mirage_settings->sublist("MrHyDE Options").get<string>("preconditioner type","Ifpack2"));
      settings->sublist("Solver").set<bool>("fully explicit",true);
      settings->sublist("Solver").set<bool>("store all cell data",mirage_settings->sublist("MrHyDE Options").get<bool>("store basis functions",true));
    }
    else {
      settings->sublist("Solver").set<string>("transient Butcher tableau",mirage_settings->sublist("MrHyDE Options").get<string>("Butcher tableau","DIRK-1,2"));
      settings->sublist("Solver").set<double>("nonlinear TOL",1.0e-07);
      settings->sublist("Solver").set<int>("max nonlinear iters",1);
      settings->sublist("Solver").set<string>("Belos solver",
                                              mirage_settings->sublist("MrHyDE Options").get<string>("Belos solver","Block GMRES"));
      settings->sublist("Solver").set<bool>("use preconditioner",
                                            mirage_settings->sublist("MrHyDE Options").get<bool>("use preconditioner",true));
      settings->sublist("Solver").set<string>("preconditioner type",
                                              mirage_settings->sublist("MrHyDE Options").get<string>("preconditioner type","domain decomposition"));
      
    }
    
    //----------------------------------------
    // Analysis block
    //----------------------------------------
    
    settings->sublist("Analysis").set<string>("analysis type",mirage_settings->sublist("MrHyDE Options").get<string>("analysis mode","forward"));
    
    //----------------------------------------
    // Postprocess block
    //----------------------------------------
    
    bool write_solution = mirage_settings->sublist("Postprocess Options").get<bool>("Exodus output",true);
    settings->sublist("Postprocess").set<bool>("write solution",write_solution);
    settings->sublist("Postprocess").set<int>("write frequency",mirage_settings->sublist("Postprocess Options").get("Exodus output frequency",1));
    
    if (mirage_settings->sublist("Postprocess Options").get<bool>("Print timers",true)) {
      settings->set<int>("verbosity",10);
      settings->set<int>("debug level",mirage_settings->sublist("MrHyDE Options").get<int>("debug level",0));
    }
    settings->sublist("Postprocess").sublist("Objective functions").sublist("EM Energy").set<string>("type","integrated response");
    string energy = "0.5*epsilon*(rindex^2)*(E[x]^2+E[y]^2+E[z]^2) + 0.5/mu*(B[x]^2+B[y]^2+B[z]^2)";
    if (mirage_dim == 2) {
      energy = "0.5*epsilon*(rindex^2)*(E[x]^2+E[y]^2) + 0.5/mu*(B^2)";
    }
    settings->sublist("Postprocess").sublist("Objective functions").sublist("EM Energy").set<string>("response",energy);
    settings->sublist("Postprocess").sublist("Objective functions").sublist("EM Energy").set<double>("target",0.0);
    settings->sublist("Postprocess").sublist("Objective functions").sublist("EM Energy").set<double>("weight",1.0);
    settings->sublist("Postprocess").sublist("Objective functions").sublist("EM Energy").set<bool>("save response data",true);
    settings->sublist("Postprocess").sublist("Objective functions").sublist("EM Energy").set<string>("response file","EM_Energy");
    settings->sublist("Postprocess").set<bool>("compute responses",mirage_settings->sublist("MrHyDE Options").get<bool>("compute energy",true));
    settings->sublist("Postprocess").set<bool>("compute weighted norm",mirage_settings->sublist("MrHyDE Options").get<bool>("compute discrete energy",false));
    
    settings->print();
    
  }
  
}

#endif
