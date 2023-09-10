/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** \file   physicsInterface.hpp
 \brief  Contains the user interface to MrHyDE.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_USERINTERFACE_H
#define MRHYDE_USERINTERFACE_H

#include "trilinos.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

#include "preferences.hpp"

#if defined(MrHyDE_ENABLE_MIRAGE)
#include "MirageTranslator.hpp"
#endif

namespace MrHyDE {
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  // Figure out if a file is .xml or .yaml (default)
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  int getFileType(const std::string & filename)
  {
    int type = -1;
    if(filename.find_last_of(".") != std::string::npos) {
      std::string extension = filename.substr(filename.find_last_of(".")+1);
      if (extension == "yaml") {
        type = 0;
      }
      else if (extension == "xml") {
        type = 1;
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized file extension: " + filename);
      }
    }
    return type;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  // Function to print out help information
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  void userHelp(const std::string & details) {
    cout << "********** Help and Documentation for the User Interface **********" << endl;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  // Standard constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Teuchos::ParameterList> UserInterface(const std::string & filename) {
    
    using Teuchos::RCP;
    using Teuchos::rcp;
    
    RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::UserInterface - constructor");
    Teuchos::TimeMonitor constructortimer(*constructortime);
    
    RCP<Teuchos::ParameterList> settings = rcp(new Teuchos::ParameterList("MrHyDE"));
        
    // If called from Mirage, then the input file will be called FEM3.xml
    // This required a special interpreter
    
    if (filename == "FEM3.xml") {
      #if defined(MrHyDE_ENABLE_MIRAGE)
      MirageTranslator(settings, filename);
      #else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Mirage extensions were not enabled!");
      #endif
    }
    else {
      
      // MrHyDE uses a set of input files ... one for each interface: mesh, physics, solver, analysis, postprocessing, parameters
      
      bool have_mesh = false;
      bool have_phys = false;
      bool have_disc = false;
      bool have_solver = false;
      bool have_analysis = false;
      bool have_pp = false; // optional
      bool have_params = false; //optional
      bool have_subgrid = false; //optional
      bool have_functions = false; //optional
      bool have_aux_phys = false; //optional
      bool have_aux_disc = false; //optional
      
      //////////////////////////////////////////////////////////////////////////////////////////
      // Import the main input.xml file
      //////////////////////////////////////////////////////////////////////////////////////////
      
      std::ifstream fnmast(filename.c_str());
      if (fnmast.good()) {
        Teuchos::RCP<Teuchos::ParameterList> main_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*main_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*main_parlist) );
        
        settings->setParameters( *main_parlist );
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: MrHyDE could not find the main input file: " + filename);
      }
      
      if (settings->isSublist("Mesh"))
        have_mesh = true;
      if (settings->isSublist("Physics"))
        have_phys = true;
      if (settings->isSublist("Discretization"))
        have_disc = true;
      if (settings->isSublist("Solver"))
        have_solver = true;
      if (settings->isSublist("Analysis"))
        have_analysis = true;
      if (settings->isSublist("Postprocess"))
        have_pp = true;
      if (settings->isSublist("Parameters"))
        have_params = true;
      if (settings->isSublist("Subgrid"))
        have_subgrid = true;
      if (settings->isSublist("Functions"))
        have_functions = true;
      if (settings->isSublist("Aux Physics"))
        have_aux_phys = true;
      if (settings->isSublist("Aux Discretization"))
        have_aux_disc = true;
      
      //////////////////////////////////////////////////////////////////////////////////////////
      // Some of the sublists are required (mesh, physics, solver, analysis)
      // If they do not appear in input.xml, then a file needs to be provided
      // This allows input.xml to be rather clean and easily point to different xml files
      //////////////////////////////////////////////////////////////////////////////////////////
      
      if (!have_mesh) {
        if (settings->isParameter("Mesh input file")) {
          std::string filename = settings->get<std::string>("Mesh input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> mesh_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*mesh_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*mesh_parlist) );
            
            settings->setParameters( *mesh_parlist );
          }
          else {
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the mesh settings file: " + filename);
          }
        }
        else
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either a Mesh sublist or a path to a mesh settings file!");
      }
      
      if (!have_phys) {
        if (settings->isParameter("Physics input file")) {
          std::string filename = settings->get<std::string>("Physics input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> phys_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*phys_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*phys_parlist) );
            
            settings->setParameters( *phys_parlist );
          }
          else
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the physics settings file: " + filename);
        }
        else
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either a Physics sublist or a path to a physics settings file!");
      }
      
      if (!have_disc) {
        if (settings->isParameter("Discretization input file")) {
          std::string filename = settings->get<std::string>("Discretization input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> disc_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*disc_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*disc_parlist) );
            
            settings->setParameters( *disc_parlist );
          }
          else
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the discretization settings file: " + filename);
        }
        else
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either a Discretization sublist or a path to a Discretization settings file!");
      }
      
      if (!have_solver) {
        if (settings->isParameter("Solver input file")) {
          std::string filename = settings->get<std::string>("Solver input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> solver_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*solver_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*solver_parlist) );
            
            settings->setParameters( *solver_parlist );
          }
          else
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the solver settings file:" + filename);
        }
        else
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either a Solver sublist or a path to a solver settings file!");
      }
      
      if (!have_analysis) {
        if (settings->isParameter("Analysis input file")) {
          std::string filename = settings->get<std::string>("Analysis input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> analysis_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*analysis_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*analysis_parlist) );
            
            settings->setParameters( *analysis_parlist );
          }
          else
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the analysis settings file: " + filename);
        }
        else
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either an Analysis sublist or a path to an analysis settings file!");
      }
      
      if (!have_pp) { // this is optional (but recommended!)
        if (settings->isParameter("Postprocess input file")) {
          std::string filename = settings->get<std::string>("Postprocess input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> pp_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*pp_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*pp_parlist) );
            
            settings->setParameters( *pp_parlist );
          }
          else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the postprocess settings file: " + filename);
        }
        else
          settings->sublist("Postprocess",false,"Empty sublist for postprocessing.");
      }
      
      if (!have_params) { // this is optional
        if (settings->isParameter("Parameters input file")) {
          std::string filename = settings->get<std::string>("Parameters input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> param_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*param_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*param_parlist) );
            
            settings->setParameters( *param_parlist );
          }
          else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the parameters settings file: " + filename);
        }
        else
          settings->sublist("Parameters",false,"Empty sublist for parameters.");
      }
      
      if (!have_subgrid) { // this is optional
        if (settings->isParameter("Subgrid input file")) {
          std::string filename = settings->get<std::string>("Subgrid input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> subgrid_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*subgrid_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*subgrid_parlist) );
            
            settings->setParameters( *subgrid_parlist );
          }
          else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the subgrid settings file: " + filename);
        }
      }
      
      if (!have_functions) { // this is optional
        if (settings->isParameter("Functions input file")) {
          std::string filename = settings->get<std::string>("Functions input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> functions_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*functions_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*functions_parlist) );
            
            settings->setParameters( *functions_parlist );
          }
          else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the functions settings file: " + filename);
        }
      }
      
      if (!have_aux_phys) { // this is optional (unless Aux Discretization is defined)
        if (settings->isParameter("Aux Physics input file")) {
          std::string filename = settings->get<std::string>("Aux Physics input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> phys_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*phys_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*phys_parlist) );
            
            settings->setParameters( *phys_parlist );
          }
          else
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the aux physics settings file: " + filename);
        }
      }
      
      if (!have_aux_disc) { // this is optional (unless Aux Physics is defined)
        if (settings->isParameter("Aux Discretization input file")) {
          std::string filename = settings->get<std::string>("Aux Discretization input file");
          std::ifstream fn(filename.c_str());
          if (fn.good()) {
            Teuchos::RCP<Teuchos::ParameterList> disc_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
            int type = getFileType(filename);
            if (type == 0)
              Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*disc_parlist) );
            else if (type == 1)
              Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*disc_parlist) );
            
            settings->setParameters( *disc_parlist );
          }
          else
            TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the aux discretization settings file: " + filename);
        }
      }
      
      if (have_aux_disc && !have_aux_phys) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: an aux discretization was defined, but not an aux physics");
      }
      else if (!have_aux_disc && have_aux_phys) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: an aux physics was defined, but not an aux discretization");
      }
      
    }
    
    return settings;
    
  }
  
}

#endif
