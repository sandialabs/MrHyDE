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

#ifndef SUBGRID_GEN_H
#define SUBGRID_GEN_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "subgridModel.hpp"
#include "subgridFEM.hpp"

using namespace std;

namespace MrHyDE {
  
  vector<Teuchos::RCP<SubGridModel> > subgridGenerator(const Teuchos::RCP<MpiComm> & Comm,
                                                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                                                       Teuchos::RCP<panzer_stk::STK_Interface> & macromesh ) {
    
    vector<Teuchos::RCP<SubGridModel> > subgridModels;
    
    if (settings->isSublist("Subgrid")) {
      
      ////////////////////////////////////////////////////////////////////////////////
      // Define the subgrid models specified in the input file
      ////////////////////////////////////////////////////////////////////////////////
      
      int nummodels = settings->sublist("Subgrid").get<int>("number of models",1);
      int  num_macro_time_steps = settings->sublist("Solver").get("number of steps",1);
      ScalarT finaltime = settings->sublist("Solver").get<ScalarT>("finaltime",1.0);
      ScalarT macro_deltat = finaltime/num_macro_time_steps;
      if (nummodels == 1) {
        Teuchos::RCP<Teuchos::ParameterList> subgrid_pl = rcp(new Teuchos::ParameterList("Subgrid"));
        subgrid_pl->setParameters(settings->sublist("Subgrid"));
        string subgrid_model_type = subgrid_pl->get<string>("subgrid model","FEM");
        string macro_block_name = subgrid_pl->get<string>("macro block","eblock-0_0_0");
        std::vector<string> macro_blocknames;
        macromesh->getElementBlockNames(macro_blocknames);
        int macro_block = 0; // default to single block case
        for (size_t m=0; m<macro_blocknames.size(); ++m) {
          if (macro_blocknames[m] == macro_block_name) {
            macro_block = m;
          }
        }
        topo_RCP macro_topo = macromesh->getCellTopology(macro_blocknames[macro_block]);
        if (subgrid_model_type == "FEM") {
          subgridModels.push_back(Teuchos::rcp( new SubGridFEM(Comm, subgrid_pl, macro_topo,
                                                               num_macro_time_steps,
                                                               macro_deltat) ) );
        }
        else if (subgrid_model_type == "Explicit FEM") {
          // subgridModels.push_back(Teuchos::rcp( new SubGridExpFEM(Comm, subgrid_pl, macro_topo,
          //                                                         num_macro_time_steps,
          //                                                         macro_deltat) ) );
        }
        else if (subgrid_model_type == "FEM2") {
          //subgridModels.push_back(Teuchos::rcp( new SubGridFEM2(Comm, subgrid_pl, macro_topo, num_macro_time_steps, macro_deltat) ) );
        }
        subgridModels[subgridModels.size()-1]->macro_block = macro_block;
        subgridModels[subgridModels.size()-1]->usage = "1.0";
      }
      else {
        for (int j=0; j<nummodels; j++) {
          stringstream ss;
          ss << j;
          if (settings->sublist("Subgrid").isSublist("Model" + ss.str())) {
            Teuchos::RCP<Teuchos::ParameterList> subgrid_pl = rcp(new Teuchos::ParameterList("Subgrid"));
            subgrid_pl->setParameters(settings->sublist("Subgrid").sublist("Model" + ss.str()));
            string subgrid_model_type = subgrid_pl->get<string>("subgrid model","FEM");
            string macro_block_name = subgrid_pl->get<string>("macro block","eblock-0_0_0");
            std::vector<string> macro_blocknames;
            macromesh->getElementBlockNames(macro_blocknames);
            int macro_block = 0; // default to single block case
            for (size_t m=0; m<macro_blocknames.size(); ++m) {
              if (macro_blocknames[m] == macro_block_name) {
                macro_block = m;
              }
            }
            topo_RCP macro_topo = macromesh->getCellTopology(macro_blocknames[macro_block]);
            
            if (subgrid_model_type == "FEM") {
              subgridModels.push_back(Teuchos::rcp( new SubGridFEM(Comm, subgrid_pl, macro_topo,
                                                                   num_macro_time_steps,
                                                                   macro_deltat ) ) );
            }
            else if (subgrid_model_type == "Explicit FEM") {
              // subgridModels.push_back(Teuchos::rcp( new SubGridExpFEM(Comm, subgrid_pl, macro_topo,
              //                                                          num_macro_time_steps,
              //                                                          macro_deltat ) ) );
            }
            else if (subgrid_model_type == "FEM2") {
              //subgridModels.push_back(Teuchos::rcp( new SubGridFEM2(Comm, subgrid_pl, macro_topo, num_macro_time_steps, macro_deltat ) ) );
            }
            subgridModels[subgridModels.size()-1]->macro_block = macro_block;
            string usage;
            if (j==0) {// to enable default behavior
              usage = subgrid_pl->get<string>("usage","1.0");
            }
            else {
              usage = subgrid_pl->get<string>("usage","0.0");
            }
            subgridModels[subgridModels.size()-1]->usage = usage;
          }
        }
        
      }
    }
    
    return subgridModels;
  }
  
}
#endif
