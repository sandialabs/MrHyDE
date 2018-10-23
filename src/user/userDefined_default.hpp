/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef USERDEF_H
#define USERDEF_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "userDefined_base.hpp"
#include "workset.hpp"

class UserDefined : public UserDefinedBase {
public:
  
  UserDefined() {} ;
  
  ~UserDefined() {};
  
  UserDefined(Teuchos::RCP<Teuchos::ParameterList> & settings) {
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    finalTime = settings->sublist("Solver").get<double>("finaltime",1);
    numSteps = settings->sublist("Solver").get<int>("numSteps",1);
    delT = finalTime/numSteps;
    
    if (settings->sublist("Physics").get<int>("solver",0) == 1)
      isTD = true;
    else
      isTD = false;
    
    multiscale = settings->isSublist("Subgrid");
    
    test = settings->sublist("Physics").get<int>("test",0);
    simNum = settings->sublist("Physics").get<int>("simulation_number",0);
    simName = settings->sublist("Physics").get<string>("simulation_name","mySim");
    
    Lx = settings->sublist("Physics").get<double>("Lx",1.0);
    Ly = settings->sublist("Physics").get<double>("Ly",1.0);
    Px = settings->sublist("Physics").get<double>("Px",0.0);
    Py = settings->sublist("Physics").get<double>("Py",0.0);
    
    //specific to particular test cases
    cMY = settings->sublist("Physics").get<double>("cMY",10.0);
    sinkwidth = settings->sublist("Physics").get<double>("sink width",0.1);
    
    // parameters for AM thermal model
    T_ambient = settings->sublist("Physics").get<double>("T_ambient",0.0);
    T_rad = settings->sublist("Physics").get<double>("T_ambient",0.0);
    emiss = settings->sublist("Physics").get<double>("emiss",0.25);
    //rho = settings->sublist("Physics").get<double>("rho",7609.0);
    rho = settings->sublist("Physics").get<double>("rho",1.0);
    eta = settings->sublist("Physics").get<double>("eta",0.6);
    //hconv = settings->sublist("Physics").get<double>("hconv",20.0);
    hconv = settings->sublist("Physics").get<double>("hconv",1.0);
    e_a = settings->sublist("Physics").get<double>("e_a",6.35e-3);
    e_b = settings->sublist("Physics").get<double>("e_b",3.81e-3);
    e_c = settings->sublist("Physics").get<double>("e_c",6.35e-3);
    sb_constant = 5.6704e-8;
    laser_xpos = settings->sublist("Physics").get<double>("laser_xpos",0.0);
    laser_ypos = settings->sublist("Physics").get<double>("laser_ypos",0.0);
    laser_zpos = settings->sublist("Physics").get<double>("laser_zpos",0.0);
    rthres1 = settings->sublist("Physics").get<double>("rthres1",8000.0);
    rthres2 = settings->sublist("Physics").get<double>("rthres2",8000.0);
    rthres3 = settings->sublist("Physics").get<double>("rthres3",8000.0);
    cparamx = settings->sublist("Physics").get<double>("cparamx",1.0);
    cparamz = settings->sublist("Physics").get<double>("cparamz",1.0);
    melt_pool_temp = settings->sublist("Physics").get<double>("melt pool temperature",0.0);
    weld_length = settings->sublist("Physics").get<double>("weld length",0.0);
    laser_off = settings->sublist("Physics").get<double>("laser off",0.0);
    laser_on = settings->sublist("Physics").get<double>("laser on",0.0);
    
    toff = settings->sublist("Physics").get<double>("toff",1.000);
    sensor_refine = settings->sublist("Physics").get<bool>("sensor_refine",false);
    
    // parameters for linear elasticity
    incplanestress = settings->sublist("Physics").get<bool>("incplanestress",false);
    ledisp_response_type = settings->sublist("Physics").get<bool>("disp_response_type",true);
    E_ref = settings->sublist("Physics").get<double>("E_ref",1.0);
    t_ref = settings->sublist("Physics").get<double>("t_ref",1.0);
    alpha_T = settings->sublist("Physics").get<double>("alpha_T",1.0e-6);
    tbase = settings->sublist("Physics").get<double>("tbase",1.0e6);
    tscale = settings->sublist("Physics").get<double>("tscale",1.0e6);
    use_log_E = settings->sublist("Physics").get<bool>("use log E",false);
    two_well_random = settings->sublist("Physics").get<bool>("two well random",false);
    one_well_random = settings->sublist("Physics").get<bool>("one well random",false);
    forward_data_gen = settings->sublist("Physics").get<bool>("generate forward data",false);
    
    //parameters/toggles for Helmholtz
    usePML = settings->sublist("Physics").get<bool>("use PML",false);
    coreSize = settings->sublist("Physics").get<double>("core size",0.5);
    xSize = settings->sublist("Mesh").get<double>("xmax",1.0);
    ySize = settings->sublist("Mesh").get<double>("ymax",1.0);
    sigma0 = settings->sublist("Physics").get<double>("sigma0",10.0);
    
    responseTarget = settings->sublist("Postprocess").get("response target", 0.0);
    regCoeff = settings->sublist("Postprocess").get("regularization coefficient", 0.0);
    
    s_param = settings->sublist("Physics").get<double>("s_param",1.0);
    
    // ceramic burnoff
    stddevx = settings->sublist("Physics").get<double>("stddevx",1.0);
    stddevy = settings->sublist("Physics").get<double>("stddevy",1.0);
    stddevz = settings->sublist("Physics").get<double>("stddevz",1.0);
    xloc = settings->sublist("Physics").get<double>("xloc",0.5);
    yloc = settings->sublist("Physics").get<double>("yloc",0.5);
    zloc = settings->sublist("Physics").get<double>("zloc",0.5);
    mag = settings->sublist("Physics").get<double>("mag",1.0);
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Set the list of variable names
  //////////////////////////////////////////////////////////////////////////////////////
  
  void setVarlist(const vector<vector<string> > & varlist_) {
    varlist = varlist_;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the index for a specific variable
  //////////////////////////////////////////////////////////////////////////////////////
  
  size_t getVariableIndex(const int & block, const string & var) const {
    size_t index=0;
    for (size_t i=0; i<varlist[block].size(); i++) {
      if (var == varlist[block][i]) {
        index = i;
      }
    }
    return index;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the index for the subgrid model
  //////////////////////////////////////////////////////////////////////////////////////
  
  vector<size_t> getSubgridModel(const DRV & nodes, Teuchos::RCP<workset> & wkset,
                                 Kokkos::View<double***,AssemblyDevice> localsol, const int & nummodels) const {
    int numElem = nodes.dimension(0);
    vector<size_t> index(numElem);
    
    for (int p=0; p<numElem; p++) {
      if (nummodels > 1) {
        if (test == 1) {
          if (nodes(p,0,0)<0.5 && nodes(p,0,1)>0.5) {
            index[p] = 1;
          }
        }
        else if (test == 2) {
          size_t c_num = this->getVariableIndex(wkset->block, "c");
          double avgc = 0.0;
          size_t cdof = localsol.dimension(2);
          for (size_t i=0; i<cdof; i++) {
            avgc += localsol(p,c_num,i)/(double)cdof;
          }
          double cvar = 0.0;
          for (size_t i=0; i<cdof; i++) {
            cvar += (localsol(p,c_num,i)-avgc)*(localsol(p,c_num,i)-avgc)/(double)cdof;
          }
          //cout << cvar << endl;
          
          if (cvar > 1.0e-4)
            index[p] = 1;
          if (cvar > 1.0e-3)
            index[p] = 2;
        }
        else if (test == 11) {
          double midx = 0.0;
          double midy = 0.0;
          int nnodes = nodes.dimension(1);
          for (size_t i=0; i<nnodes; i++){
            midx += nodes(p,i,0)/(double)nnodes;
            midy += nodes(p,i,1)/(double)nnodes;
          }
          double rad = midx*midx+midy*midy;
          double time = wkset->time;
          if (rad<time*time) {
            index[p] = 1;
          }
          if (rad<(time-0.2)*(time-0.2)) {
            index[p] = 0;
          }
        }
        else if (test == 12) {
          size_t e_num = this->getVariableIndex(wkset->block,"e");
          double avge = 0.0;
          size_t edof = localsol.dimension(2);
          for (size_t i=0; i<edof; i++) {
            avge += localsol(p,e_num,i)/(double)edof;
          }
          if (avge > 0.1) {
            index[p] = 1;
          }
          if (avge > 0.5) {
            index[p] = 2;
          }
        }
        else if (test == 902) {
          size_t e_num = this->getVariableIndex(wkset->block,"e");
          double avge = 0.0;
          size_t edof = localsol.dimension(2);
          for (size_t i=0; i<edof; i++) {
            avge += localsol(p,e_num,i)/(double)edof;
          }
          if (abs(avge) > 1.0) {
            index[p] = 1;
          }
          if (abs(avge) > 3.0) {
            index[p] = 2;
          }
        }
        else if (simNum == 115 && sensor_refine) {
          fstream rfile;
          vector<vector<double> > rDat;
          vector<double> rowVec(3);
          int row = 0;
          rfile.open("refinement.dat", ios::in);
          if (rfile.is_open()) {
            while (rfile.good()) {
              rDat.push_back(rowVec);
              for (int col=0; col<3; col++) {
                rfile >> rDat[row][col];
              }
              row++;
            }
          }
          rfile.close();
          
          double xc = 0;
          double yc = 0;
          for (int i = 0; i < 4; i++) {
            xc += nodes(p,i,0)/4.0;
            yc += nodes(p,i,1)/4.0;
          }
          
          double dist, xd, yd;
          index[p] = 0;
          for (int i = 0; i < row; i++) {
            xd = rDat[i][0] - xc;
            yd = rDat[i][1] - yc;
            dist = sqrt(xd*xd + yd*yd);
            if (dist < 1.0e-6)
              index[p] = (int)rDat[i][2];
          }
        }
        else if (simNum == 404 || simNum == 45 || simNum == 44) {
          size_t e_num = this->getVariableIndex(wkset->block,"e");
          size_t dx_num = this->getVariableIndex(wkset->block,"dx");
          size_t dy_num = this->getVariableIndex(wkset->block,"dy");
          size_t dz_num = this->getVariableIndex(wkset->block,"dz");
          double avge = 0.0;
          size_t edof = localsol.dimension(2);
          for (size_t i=0; i<edof; i++) {
            avge = std::max(localsol(p,e_num,i),avge);///(double)edof;
          }
          double mxd = 0.0;
          for (size_t i=0; i<edof; i++) {
            double dmag = abs(localsol(p,dx_num,i)) + abs(localsol(p,dy_num,i)) + abs(localsol(p,dz_num,i));
            mxd = std::max(mxd,dmag);
          }
          if (abs(mxd) > rthres1) {
            index[p] = 1;
          }
          if (abs(mxd) > rthres2) {
            index[p] = 2;
          }
          if (abs(mxd) > rthres3) {
            index[p] = 3;
          }
          
        }
        else if (simNum == 1077) {
          double midy = 0.0;
          int nnodes = nodes.dimension(1);
          for (size_t i=0; i<nnodes; i++){
            midy += nodes(p,i,1)/(double)nnodes;
          }
          if (midy>=-0.075 && midy<=0.075) {
            index[p]=1;
          }
          else {
            index[p]=0;
          }
        }
      }
    }
    return index;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Compute the Neumann source term
  //////////////////////////////////////////////////////////////////////////////////////
  // TMW: remove this!!
  
  Kokkos::View<AD*,AssemblyDevice> boundaryNeumannSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset) const {
    
    DRV ip = wkset->ip_side;
    
    int numip = ip.dimension(1);
    string side = wkset->sidename;
    
    Kokkos::View<AD*,AssemblyDevice> vals("neumann source values",numip); //defaults to zeros
    double x,y,z;
    double time  = wkset->time;
    
    
    // Specialize for physics, var and test
    
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy" ) {
      if (var == "e") {
        for (int i=0; i<numip; i++) {
          x = ip(0,i,0);
          if (spaceDim > 1)
            y = ip(0,i,1);
          if (spaceDim > 2)
            z = ip(0,i,2);
          if (test == 2) { // mixed bcs verification test u = sin(2\pi x)sin(2\pi y)
            if (side == "top"){
              vals(i) = 2.0*PI*sin(2.0*PI*x)*cos(2.0*PI*y);
            }
            else if (side == "bottom"){
              vals(i) = -2.0*PI*sin(2.0*PI*x)*cos(2.0*PI*y);
            }
          }
          if (test == 27 && side == "front") {
            vals(i) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                              laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
            
          }
          else if (simNum == 3 && side == "top") {
            // assumes 2D
            vals(i) = laser_intensity[0]*(exp(-laser_width[0]*(x-1.0)*(x-1.0))) +
            laser_intensity[1]*(exp(-laser_width[0]*(x-2.0)*(x-2.0))) +
            laser_intensity[2]*(exp(-laser_width[0]*(x-3.0)*(x-3.0)));
            
          }
          
          else if (simNum == 4 && side == "top") {
            // 2D gaussian stationary, known variance beams
            double width_factor = 2*pow(0.2,2);
            
            vals(i) =  laser_intensity[0]*exp(-pow(x-0.1,2.0)/width_factor) +
            laser_intensity[1]*exp(-pow(x-0.3,2.0)/width_factor) +
            laser_intensity[2]*exp(-pow(x-0.5,2.0)/width_factor) +
            laser_intensity[3]*exp(-pow(x-0.7,2.0)/width_factor) +
            laser_intensity[4]*exp(-pow(x-0.9,2.0)/width_factor);
          }
          
          else if (simNum == 5 && side == "top") {
            // 2D gaussian stationary, known variance beams
            double width_factor = 2*pow(0.2,2);
            
            vals(i) =  laser_intensity[0]*exp(-pow(x-0.1,2.0)/width_factor) +
            laser_intensity[1]*exp(-pow(x-0.3,2.0)/width_factor) +
            laser_intensity[2]*exp(-pow(x-0.5,2.0)/width_factor) +
            laser_intensity[3]*exp(-pow(x-0.7,2.0)/width_factor) +
            laser_intensity[4]*exp(-pow(x-0.8,2.0)/width_factor) +
            laser_intensity[5]*exp(-pow(x-0.9,2.0)/width_factor);
            
          }
          
          
          else if (simName == "thermo-boundary-ms" && side == "top") {
            double width_factor = 2*pow(0.02,2);
            
            vals(i) =  laser_intensity[0]*exp(-pow(x-2.1,2.0)/width_factor) +
            laser_intensity[1]*exp(-pow(x-2.3,2.0)/width_factor) +
            laser_intensity[2]*exp(-pow(x-2.5,2.0)/width_factor) +
            laser_intensity[3]*exp(-pow(x-2.7,2.0)/width_factor) +
            laser_intensity[4]*exp(-pow(x-2.9,2.0)/width_factor);
          }
          
          else if (simNum == 12 && side == "top") {
            vals(i) = 1.0;
            //val = 0.0;
          }
          
          else if (simNum == 13 && side == "top")
            vals(i) = boundary_params[0];
          
          else if ((simNum == 44) && (side != "bottom")) {
            // qrad + qconv
            size_t e_num = this->getVariableIndex(wkset->block,"e");
            AD e = wkset->local_soln_side(0,e_num,i,0);
            vals(i) = -emiss*sb_constant*(e*e*e*e - pow(T_rad,4.0)) - hconv*(e - T_ambient);
            
          }
          else if (simNum == 144 || simNum == 145) {
            // qrad + qconv
            size_t e_num = this->getVariableIndex(wkset->block,"e");
            AD e = wkset->local_soln_side(0,e_num,i,0);
            vals(i) = -emiss*sb_constant*(e*e*e*e - pow(T_rad,4.0)) - hconv*(e - T_ambient);
          }
          
          else if ((simNum == 45) && ((side == "top") || (side == "bottom"))) {
            // qrad + qconv
            size_t e_num = this->getVariableIndex(wkset->block,"e"); //TMW: needs to be updated
            AD e = wkset->local_soln_side(0,e_num,i,0);
            vals(i) = -emiss*sb_constant*(e*e*e*e - pow(T_rad,4.0)) - hconv*(e - T_ambient);
            
          }
          
          else if ((simNum == 112) && ((side == "top") || (side == "bottom"))) {
            // qrad + qconv
            size_t e_num = this->getVariableIndex(wkset->block,"e"); //TMW: needs to be updated
            AD e = wkset->local_soln_side(0,e_num,i,0);
            vals(i) = -emiss*sb_constant*(e*e*e*e - pow(T_rad,4.0)) - hconv*(e - T_ambient);
            
          }
          
          else if (simNum == 62) {
            if (side == "top")
              vals(i) = 1.0;
          }
          else if (simNum == 778) {
            if (side == "top") {
              vals(i) = 100.0*exp(-100.0*(x-0.5)*(x-0.5) - 100.0*(z-0.5)*(z-0.5));
            }
          }
          /*
           else if (simNum == 115) {
           if (side == "top") {
           vals(i) = boundary_params[0];
           }
           }
           */
          else if (simNum == 779) {
            if (side == "top") {
              for (int jj=0;jj<stddev1.size();jj++) {
                vals(i) +=  sourceControl[jj]*(1/sqrt(2.0*PI*stddev1[jj]*stddev1[jj]))
                *exp(-pow(x-mean[jj],2.0)/(2.0*stddev1[jj]*stddev1[jj]) +
                     -pow(y-mean[jj],2.0)/(2.0*stddev1[jj]*stddev1[jj]) +
                     -pow(z-mean[jj],2.0)/(2.0*stddev1[jj]*stddev1[jj]));
              }
            }
            for (int kk=0;kk<stddev2.size();kk++) {
              vals(i) +=  expMag[kk]*(1/sqrt(2.0*PI*stddev2[kk]*stddev2[kk]))
              *exp(-pow(x-smean[kk],2.0)/(2.0*stddev2[kk]*stddev2[kk]) +
                   -pow(y-smean[kk],2.0)/(2.0*stddev2[kk]*stddev2[kk]) +
                   -pow(z-smean[kk],2.0)/(2.0*stddev2[kk]*stddev2[kk]));
            }
          }
        }
      }
    }
    else if (physics == "porous") {
      // default to zero
    }
    else if (physics == "porous2p") {
      // default to zero
    }
    else if (physics == "linearelasticity") {
      
      for (int i=0; i<numip; i++) {
        x = ip(0,i,0);
        if (spaceDim > 1)
          y = ip(0,i,1);
        if (spaceDim > 2)
          z = ip(0,i,2);
        
        AD val = 0.0;
        if (simNum == 444) { // used to be 44, conflicted with weld problem
          if (side == "right") {
            if (var == "dx") {
              val = nbcx[0];
            }
          }
        }
        else if (simNum == 7) {
          if (side == "top") {
            if (var == "dy") {
              val = nbcy[0];
            }
          }
        }
        else if ((simNum == 87) || (simNum == 88)) {
          if (side == "left") {
            if (var == "dx") {
              val = nbcx[0];
            }
          }
          else if (side == "right") {
            if (var == "dx") {
              val = nbcx[1];
            }
          }
          else if (side == "top") {
            if (var == "dy") {
              val = nbcy[0];
            }
          }
        }
        else if ((simNum == 8) || (simNum == 9)) {
          if (side == "top") {
            if (var == "dy") {
              val = nbcy[0];
            }
          }
          else if (side == "bottom") {
            if (var == "dy") {
              val = -nbcy[0];
            }
          }
        }
        else if ((simNum == 5) || (simNum == 6) || (simNum == 22)) {
          if (side == "top") {
            if (var == "dy") {
              val = wkset->local_param_side(0,1,i); // discretized traction y component
              //val = nbcy[0];
            }
          }
        }
        else if (simNum == 200) {
          if (side == "top") {
            if (var == "dy") {
              val = nbcy[0];
            }
          }
        }
        else if (simNum == 36 || simNum == 26) {
          if (side == "surface_3") {
            if (var == "dx") {
              val = nbcx[0];
            }
          }
        }
        else if (simNum == 49) {
          if (side == "top") {
            if (var == "dy") {
              AD pp = nbcy[0];
              val = 0.45*pp + 0.55*pp*pp + 1.0*pp*pp*pp;
            }
          }
        }
        else if (simNum == 2) {
          if (side == "top") {
            if (var == "dx") {
              val = ttop[0];
            }
            if (var == "dy") {
              val = ttop[1];
            }
            if (var == "dz") {
              val = ttop[2];
            }
          }
          if (side == "bottom") {
            if (var == "dx") {
              val = tbottom[0];
            }
            if (var == "dy") {
              val = tbottom[1];
            }
            if (var == "dz") {
              val = tbottom[2];
            }
          }
          if (side == "left") {
            if (var == "dx") {
              val = tleft[0];
            }
            if (var == "dy") {
              val = tleft[1];
            }
            if (var == "dz") {
              val = tleft[2];
            }
          }
          if (side == "right") {
            if (var == "dx") {
              val = tright[0];
            }
            if (var == "dy") {
              val = tright[1];
            }
            if (var == "dz") {
              val = tright[2];
            }
          }
        }
        else if (simNum == 3) {
          if (side == "front") {
            if (var == "dz") {
              val = nbcz[0];
            }
          }
          
          if (side == "top") {
            if (var == "dy") {
              if (x < 0.2) {
                val = nbcy[0];
              }
              else if (x >= 0.2 && x < 0.4) {
                val = nbcy[1];
              }
              else if (x >= 0.4 && x < 0.6) {
                val = nbcy[2];
              }
              else {
                val = nbcy[3];
              }
            }
          }
          
          if (side == "left") {
            if (var == "dx") {
              if (y < 0.2) {
                val = -1.0*nbcx[0];
              }
              else if (y >= 0.2 && y < 0.4) {
                val = -1.0*nbcx[1];
              }
              else if (y >= 0.4 && y < 0.6) {
                val = -1.0*nbcx[2];
              }
              else {
                val = -1.0*nbcx[3];
              }
            }
          }
          
          if (side == "right") {
            if (var == "dx") {
              if (y < 0.2) {
                val = nbcx[4];
              } else if (y >= 0.2 && y < 0.4) {
                val = nbcx[5];
              } else if (y >= 0.4 && y < 0.6) {
                val = nbcx[6];
              } else {
                val = nbcx[7];
              }
            }
          }
        }
        else if (simNum == 32) {
          if (side == "top") {
            if (var == "dy") {
              if (x < 0.5) {
                val = nbcy[0];
              } else {
                val = nbcy[1];
              }
            }
          }
          if (side == "left") {
            if (var == "dx")
              if (y < 0.5) {
                val = nbcx[0];
              } else {
                val = nbcx[1];
                
              }
          }
          
          if (side == "right") {
            if (var == "dx") {
              if (y < 0.5) {
                val = -1.0*nbcx[0];
              }
              else {
                val = -1.0*nbcx[1];
              }
            }
          }
        }
        
        else if (simNum == 366) {
          double a = 2.0e3;
          double b = 1.0e3;
          double c = 1.0e3;
          double r = 6.0;
          double xc = 7.5e3;
          double yc = 7.5e3;
          double zc = 7.5e3;
          double scale = 5.0e2;
          double discx = r - (y-yc)*(y-yc)/(b*b) - (z-zc)*(z-zc)/(c*c);
          double discz = r - (y-yc)*(y-yc)/(b*b) - (x-xc)*(x-xc)/(a*a);
          if (var == "dx") {
            if (side == "left") {
              if (discx > 0.0)
                val = 10.0e6*(1.0 - y/15.0e3) + scale*sqrt(a*a*discx);
              else
                val = 10.0e6*(1.0 - y/15.0e3);
            }
            if (side == "right") {
              if (discx > 0.0)
                val = -10.0e6*(1.0 - y/15.0e3) - scale*sqrt(a*a*discx);
              else
                val = -10.0e6*(1.0 - y/15.0e3);
            }
          }
          if (var == "dz") {
            if (side == "front") {
              if (discz > 0.0)
                val = -10.0e6*(1.0 - y/15.0e3) - scale*sqrt(c*c*discz);
              else
                val = -10.0e6*(1.0 - y/15.0e3);
            }
            if (side == "back") {
              if (discz > 0.0)
                val = 10.0e6*(1.0 - y/15.0e3) + scale*sqrt(c*c*discz);
              else
                val = 10.0e6*(1.0 - y/15.0e3);
            }
          }
        }
        else if (simNum == 367) {
          if ((side == "left") || (side == "right")) {
            if (var == "dx") {
              val = t_ref*wkset->local_param_side(0,0,i); // discretized traction y component
            }
          }
          if ((side == "front") || (side == "back")) {
            if (var == "dz") {
              val = t_ref*wkset->local_param_side(0,1,i); // discretized traction y component
            }
          }
        }
        else if (simNum == 540) {
          if ((side == "left") || (side == "right")) {
            if (var == "dx")
              val = wkset->local_param_side(0,0,i); // discretized traction y component
          }
        }
        else if (simNum == 541) {
          if (var == "dx") {
            if (side == "left")
              val = -1.0;
            if (side == "right")
              val = 1.0;
          }
        }
        else if (simNum == 205) {
          if ((side == "top") && (var == "dy")) {
            val = x*(1-x);
          }
        }
        else if (simNum == 206) {
          if ((side == "top") && (var == "dy")) {
            val = wkset->local_param_side(0,0,i); // discretized traction y component
          }
        }
        
        vals(i) = val;
      }
      
    }
    
    return vals;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Compute the Neumann source term
  //////////////////////////////////////////////////////////////////////////////////////
  
  void boundaryNeumannSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset,
                             Kokkos::View<AD**> vals) {
    
    int numElem = vals.dimension(0);
    int numip = wkset->ip_side.dimension(1);
    string side = wkset->sidename;
    
    double x,y,z;
    double time  = wkset->time;
    
    
    // Specialize for physics, var and test
    
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      if (var == "e") {
        for (int e=0; e<numElem; e++) {
          for (int i=0; i<numip; i++) {
            x = wkset->ip_side(e,i,0);
            if (spaceDim > 1)
              y = wkset->ip_side(e,i,1);
            if (spaceDim > 2)
              z = wkset->ip_side(e,i,2);
            if (test == 2) { // mixed bcs verification test u = sin(2\pi x)sin(2\pi y)
              if (side == "top"){
                vals(e,i) = 2.0*PI*sin(2.0*PI*x)*cos(2.0*PI*y);
              }
              else if (side == "bottom"){
                vals(e,i) = -2.0*PI*sin(2.0*PI*x)*cos(2.0*PI*y);
              }
            }
            if (test == 27 && side == "front") {
              vals(e,i) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                  laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
              
            }
            else if (simNum == 3 && side == "top") {
              // assumes 2D
              vals(e,i) = laser_intensity[0]*(exp(-laser_width[0]*(x-1.0)*(x-1.0))) +
              laser_intensity[1]*(exp(-laser_width[0]*(x-2.0)*(x-2.0))) +
              laser_intensity[2]*(exp(-laser_width[0]*(x-3.0)*(x-3.0)));
              
            }
            
            else if (simNum == 4 && side == "top") {
              // 2D gaussian stationary, known variance beams
              double width_factor = 2*pow(0.2,2);
              
              vals(e,i) =  laser_intensity[0]*exp(-pow(x-0.1,2.0)/width_factor) +
              laser_intensity[1]*exp(-pow(x-0.3,2.0)/width_factor) +
              laser_intensity[2]*exp(-pow(x-0.5,2.0)/width_factor) +
              laser_intensity[3]*exp(-pow(x-0.7,2.0)/width_factor) +
              laser_intensity[4]*exp(-pow(x-0.9,2.0)/width_factor);
            }
            
            else if (simNum == 5 && side == "top") {
              // 2D gaussian stationary, known variance beams
              double width_factor = 2*pow(0.2,2);
              
              vals(e,i) =  laser_intensity[0]*exp(-pow(x-0.1,2.0)/width_factor) +
              laser_intensity[1]*exp(-pow(x-0.3,2.0)/width_factor) +
              laser_intensity[2]*exp(-pow(x-0.5,2.0)/width_factor) +
              laser_intensity[3]*exp(-pow(x-0.7,2.0)/width_factor) +
              laser_intensity[4]*exp(-pow(x-0.8,2.0)/width_factor) +
              laser_intensity[5]*exp(-pow(x-0.9,2.0)/width_factor);
              
            }
            
            
            else if (simName == "thermo-boundary-ms" && side == "top") {
              double width_factor = 2*pow(0.02,2);
              
              vals(e,i) =  laser_intensity[0]*exp(-pow(x-2.1,2.0)/width_factor) +
              laser_intensity[1]*exp(-pow(x-2.3,2.0)/width_factor) +
              laser_intensity[2]*exp(-pow(x-2.5,2.0)/width_factor) +
              laser_intensity[3]*exp(-pow(x-2.7,2.0)/width_factor) +
              laser_intensity[4]*exp(-pow(x-2.9,2.0)/width_factor);
            }
            
            else if (simNum == 12 && side == "top") {
              vals(e,i) = 1.0;
              //val = 0.0;
            }
            
            else if (simNum == 13 && side == "top")
              vals(e,i) = boundary_params[0];
          
            else if (simNum == 144 || simNum == 145) {
              // qrad + qconv
              size_t e_num = this->getVariableIndex(wkset->block,"e");
              AD eval = wkset->local_soln_side(e,e_num,i,0);
              vals(e,i) = -emiss*sb_constant*(eval*eval*eval*eval - pow(T_rad,4.0)) - hconv*(eval - T_ambient);
            }
            
            else if ((simNum == 44) && (side != "bottom")) {
              // qrad + qconv
              size_t e_num = this->getVariableIndex(wkset->block,"e");
              AD eval = wkset->local_soln_side(e,e_num,i,0);
              vals(e,i) = -emiss*sb_constant*(eval*eval*eval*eval - pow(T_rad,4.0)) - hconv*(eval - T_ambient);
              
            }
            
            else if ((simNum == 45) && ((side == "top") || (side == "bottom"))) {
              // qrad + qconv
              size_t e_num = this->getVariableIndex(wkset->block,"e"); //TMW: needs to be updated
              AD eval = wkset->local_soln_side(e,e_num,i,0);
              vals(e,i) = -emiss*sb_constant*(eval*eval*eval*eval - pow(T_rad,4.0)) - hconv*(eval - T_ambient);
              
            }
            
            else if ((simNum == 112) && ((side == "top") || (side == "bottom"))) {
              // qrad + qconv
              size_t e_num = this->getVariableIndex(wkset->block,"e"); //TMW: needs to be updated
              AD eval = wkset->local_soln_side(e,e_num,i,0);
              vals(e,i) = -emiss*sb_constant*(eval*eval*eval*eval - pow(T_rad,4.0)) - hconv*(eval - T_ambient);
              
            }
            
            else if (simNum == 62) {
              if (side == "top")
                vals(e,i) = 1.0;
            }
            else if (simNum == 778) {
              if (side == "top") {
                vals(e,i) = 100.0*exp(-10.0*(x-0.5)*(x-0.5) - 20.0*(y-0.5)*(y-0.5));
              }
            }
            /*
             else if (simNum == 115) {
             if (side == "top") {
             vals(i) = boundary_params[0];
             }
             }
             */
            
          }
        }
      }
    }
    else if (physics == "porous") {
      // default to zero
    }
    else if (physics == "porous2p") {
      // default to zero
    }
    else if (physics == "linearelasticity") {
      
      for (int e=0; e<numElem; e++) {
        for (int i=0; i<numip; i++) {
          x = wkset->ip_side(e,i,0);
          if (spaceDim > 1)
            y = wkset->ip_side(e,i,1);
          if (spaceDim > 2)
            z = wkset->ip_side(e,i,2);
          
          AD val = 0.0;
          if (simNum == 444) { // used to be 44, conflicted with weld problem
            if (side == "right") {
              if (var == "dx") {
                val = nbcx[0];
              }
            }
          }
          else if (simNum == 7 || simNum == 776 || simNum == 777 || simNum == 985) {
            if (side == "top") {
              if (var == "dy") {
                val = nbcy[0];
              }
            }
          }
          else if (simNum == 1077) {
            if (side == "upper_grip_left") {
              if (var == "dy" || var == "dx") {
                val = 10.0;
              }
            }
            if (side == "upper_grip_right") {
              if (var == "dy") {
                val = 10.0;
              }
              else if (var == "dx") {
                val = -10.0;
              }
            }
          }
          else if (simNum == 780) {
            double r = sqrt(y*y+z*z);
            double theta = atan2(y,z);
            
            if (side == "surface_1") {
              if (var == "dx") {
                val = 0.01;
              }
              else if (var == "dy") {
                val = -1.0/300.0*r*cos(theta);
              }
              else if (var == "dz") {
                val = -1.0/300.0*r*sin(theta);
              }
            }
            else if (side == "surface_2") {
              if (var == "dx") {
                val = 0.01;
              }
              else if (var == "dy") {
                val = 0.0/300.0*r*cos(theta);
              }
              else if (var == "dz") {
                val = 0.0/300.0*r*sin(theta);
              }
            }
          }
          else if ((simNum == 87) || (simNum == 88)) {
            if (side == "left") {
              if (var == "dx") {
                val = nbcx[0];
              }
            }
            else if (side == "right") {
              if (var == "dx") {
                val = nbcx[1];
              }
            }
            else if (side == "top") {
              if (var == "dy") {
                val = nbcy[0];
              }
            }
          }
          else if (simNum == 993 || simNum == 994) {
            if (var == "dx") {
              if (side == "right") {
                //val = t_ref*nbcx[0];
                val = t_ref*(nbcx[0]*y*y + nbcx[1]*y + nbcx[2]);
              }
            }
          }
          else if (simNum == 93) {
            if ((side == "top") && (var == "dx")) {
              val = wkset->local_param_side(e,1,i); // discretized traction x component
            }
            if ((side == "top") && (var == "dy")) {
              val = wkset->local_param_side(e,2,i); // discretized traction y component
            }
          }
          else if ((simNum == 8) || (simNum == 9)) {
            if (side == "top") {
              if (var == "dy") {
                val = nbcy[0];
              }
            }
            else if (side == "bottom") {
              if (var == "dy") {
                val = -nbcy[0];
              }
            }
          }
          else if ((simNum == 5) || (simNum == 6) || (simNum == 22)) {
            if (side == "top") {
              if (var == "dy") {
                val = wkset->local_param_side(e,1,i); // discretized traction y component
                //val = nbcy[0];
              }
            }
          }
          else if (simNum == 200) {
            if (side == "top") {
              if (var == "dy") {
                val = nbcy[0];
              }
            }
          }
          else if (simNum == 36 || simNum == 26) {
            if (side == "surface_3") {
              if (var == "dx") {
                val = nbcx[0];
              }
            }
          }
          else if (simNum == 49) {
            if (side == "top") {
              if (var == "dy") {
                AD pp = nbcy[0];
                val = 0.45*pp + 0.55*pp*pp + 1.0*pp*pp*pp;
              }
            }
          }
          else if (simNum == 2) {
            if (side == "top") {
              if (var == "dx") {
                val = ttop[0];
              }
              if (var == "dy") {
                val = ttop[1];
              }
              if (var == "dz") {
                val = ttop[2];
              }
            }
            if (side == "bottom") {
              if (var == "dx") {
                val = tbottom[0];
              }
              if (var == "dy") {
                val = tbottom[1];
              }
              if (var == "dz") {
                val = tbottom[2];
              }
            }
            if (side == "left") {
              if (var == "dx") {
                val = tleft[0];
              }
              if (var == "dy") {
                val = tleft[1];
              }
              if (var == "dz") {
                val = tleft[2];
              }
            }
            if (side == "right") {
              if (var == "dx") {
                val = tright[0];
              }
              if (var == "dy") {
                val = tright[1];
              }
              if (var == "dz") {
                val = tright[2];
              }
            }
          }
          else if (simNum == 3) {
            if (side == "front") {
              if (var == "dz") {
                val = nbcz[0];
              }
            }
            
            if (side == "top") {
              if (var == "dy") {
                if (x < 0.2) {
                  val = nbcy[0];
                }
                else if (x >= 0.2 && x < 0.4) {
                  val = nbcy[1];
                }
                else if (x >= 0.4 && x < 0.6) {
                  val = nbcy[2];
                }
                else {
                  val = nbcy[3];
                }
              }
            }
            
            if (side == "left") {
              if (var == "dx") {
                if (y < 0.2) {
                  val = -1.0*nbcx[0];
                }
                else if (y >= 0.2 && y < 0.4) {
                  val = -1.0*nbcx[1];
                }
                else if (y >= 0.4 && y < 0.6) {
                  val = -1.0*nbcx[2];
                }
                else {
                  val = -1.0*nbcx[3];
                }
              }
            }
            
            if (side == "right") {
              if (var == "dx") {
                if (y < 0.2) {
                  val = nbcx[4];
                } else if (y >= 0.2 && y < 0.4) {
                  val = nbcx[5];
                } else if (y >= 0.4 && y < 0.6) {
                  val = nbcx[6];
                } else {
                  val = nbcx[7];
                }
              }
            }
          }
          else if (simNum == 32) {
            if (side == "top") {
              if (var == "dy") {
                if (x < 0.5) {
                  val = nbcy[0];
                } else {
                  val = nbcy[1];
                }
              }
            }
            if (side == "left") {
              if (var == "dx")
                if (y < 0.5) {
                  val = nbcx[0];
                } else {
                  val = nbcx[1];
                  
                }
            }
            
            if (side == "right") {
              if (var == "dx") {
                if (y < 0.5) {
                  val = -1.0*nbcx[0];
                }
                else {
                  val = -1.0*nbcx[1];
                }
              }
            }
          }
          
          else if (simNum == 366) {
            double a = 2.0e3;
            double b = 1.0e3;
            double c = 1.0e3;
            double r = 6.0;
            double xc = 7.5e3;
            double yc = 7.5e3;
            double zc = 7.5e3;
            double scale = 5.0e2;
            double discx = r - (y-yc)*(y-yc)/(b*b) - (z-zc)*(z-zc)/(c*c);
            double discz = r - (y-yc)*(y-yc)/(b*b) - (x-xc)*(x-xc)/(a*a);
            if (var == "dx") {
              if (side == "left") {
                if (discx > 0.0)
                  val = 10.0e6*(1.0 - y/15.0e3) + scale*sqrt(a*a*discx);
                else
                  val = 10.0e6*(1.0 - y/15.0e3);
              }
              if (side == "right") {
                if (discx > 0.0)
                  val = -10.0e6*(1.0 - y/15.0e3) - scale*sqrt(a*a*discx);
                else
                  val = -10.0e6*(1.0 - y/15.0e3);
              }
            }
            if (var == "dz") {
              if (side == "front") {
                if (discz > 0.0)
                  val = -10.0e6*(1.0 - y/15.0e3) - scale*sqrt(c*c*discz);
                else
                  val = -10.0e6*(1.0 - y/15.0e3);
              }
              if (side == "back") {
                if (discz > 0.0)
                  val = 10.0e6*(1.0 - y/15.0e3) + scale*sqrt(c*c*discz);
                else
                  val = 10.0e6*(1.0 - y/15.0e3);
              }
            }
          }
          else if (simNum == 367) {
            if ((side == "left") || (side == "right")) {
              if (var == "dx") {
                val = t_ref*wkset->local_param_side(e,0,i); // discretized traction y component
              }
            }
            if ((side == "front") || (side == "back")) {
              if (var == "dz") {
                val = t_ref*wkset->local_param_side(e,1,i); // discretized traction y component
              }
            }
          }
          else if (simNum == 540) {
            if ((side == "left") || (side == "right")) {
              if (var == "dx") {
                val = t_ref*wkset->local_param_side(e,0,i); // discretized traction y component
              }
            }
          }
          else if (simNum == 205) {
            if ((side == "top") && (var == "dy")) {
              val = x*(1-x);
            }
          }
          else if (simNum == 206) {
            if ((side == "top") && (var == "dy")) {
              val = wkset->local_param_side(e,0,i); // discretized traction y component
            }
          }
          else if (simNum == 500) {
            if (forward_data_gen) {
              if (((side == "top") || (side == "right") || (side == "left") || (side == "bottom"))
                  && ((var == "dx" || var == "dy"))) {
                double lim = 18.75e3/2.0;
                double zlim = 18.75e3;
                
                double tpert;
                double z_scale = 15.0e3;
                double y_scale = 18.75e3/2.0;
                double x_scale = 18.75e3/2.0;
                
                double f, grad, lvlset;
                double r, step, bigr;
                
                if (z > 15.0e3)
                  tpert = 0.0;
                else {
                  
                  double xs = x/x_scale;
                  double ys = y/y_scale;
                  double zs = z/z_scale;
                  
                  lvlset = (1.0e0-2.0758*zs + 6.7273*zs*zs - 5.1515*zs*zs*zs)*(1.0e0-sqrt(xs*xs+ys*ys));
                  grad = -(1.0e0-2.0758*zs + 6.7273*zs*zs - 5.1515*zs*zs*zs);
                  
                  size_t numpert = salt_thetavals.size();
                  double decay = 10.0;
                  double currtheta = atan2(y,x);
                  double pert = 0.0;
                  for (size_t j=0; j<numpert; j++) {
                    pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta,2));
                    pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta+2.0*PI,2));
                    pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta-2.0*PI,2));
                  }
                  
                  lvlset += 1.0/3.0*pert*(1.0-pow(zs,2));
                  f = lvlset - 0.7;
                  bigr = sqrt(x*x + y*y);
                  r = bigr/x_scale;
                  step = -f/grad;
                  r = (r + step)*x_scale;
                  tpert = tscale*1.0/abs(bigr - r);
                }
                
                if (var == "dx") {
                  if (abs(x - -lim) < 1.0) {
                    val = tbase*(1.0 - z/zlim) + tpert;
                  }
                  if (abs(x - lim) < 1.0) {
                    val = -tbase*(1.0 - z/zlim) - tpert;
                  }
                }
                else if (var == "dy") {
                  if (abs(y - -lim) < 1.0) {
                    val = tbase*(1.0 - z/zlim) + tpert;
                  }
                  if (abs(y - lim) < 1.0) {
                    val = -tbase*(1.0 - z/zlim) - tpert;
                  }
                }
              }
            }
            else {
              if ((side == "left") || (side == "right")) {
                if (var == "dx") {
                  val = t_ref*wkset->local_param_side(e,0,i); // discretized traction y component
                }
              }
              if ((side == "top") || (side == "bottom")) {
                if (var == "dy") {
                  val = t_ref*wkset->local_param_side(e,1,i); // discretized traction y component
                }
              }
            }
          }
          else if (simNum == 202) {
            if ((var == "dx") && ((side == "left") || (side == "right")))
              val = wkset->local_param_side(e,0,i);
          }
          
          vals(e,i) = val;
        }
      }
      
    }
    else if(physics == "helmholtz"){
      AD val = 0.0; //zero by default (Sommerfeld radiation condition)
      for (int e=0; e<numElem; e++) {
        for (int i=0; i<numip; i++) {
          x = wkset->ip_side(e,i,0);
          if (spaceDim > 1){
            y = wkset->ip_side(e,i,1);
          }
          if (spaceDim > 2){
            z = wkset->ip_side(e,i,2);
          }
          if(test == 4){
            if(side == "right"){
              val = 2*PI*cos(2*PI*x);
              if(spaceDim > 1){
                val *= sin(2*PI*y);
              }
              if(spaceDim > 2){
                val *= sin(2*PI*z);
              }
            }
          }
          vals(e,i) = val;
        }
      }
      
    }
    
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD**> boundaryRobinSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset) const {
    
    int numip = wkset->ip_side.dimension(1);
    int numElem = wkset->numElem;
    Kokkos::View<AD**> vals("vals",numElem,numip); //defaults to zeros
    double x,y,z;
    double time  = wkset->time;
    
    // Specialize for physics, var and test
    if(physics == "helmholtz"){
      if(var == "ureal"){
        if(test == 4 || test == 2){
          //zero default
        }else{ //default; Sommerfeld radiation condition (assuming scalar speed and freq)
          Kokkos::View<AD**> c2r_side_x("c2r_side_x",numElem,numip);
          Kokkos::View<AD**> omega2r("omega2r",numElem,numip);
          this->coefficient("helmholtz_square_speed_real_x",wkset,true,c2r_side_x);
          this->coefficient("helmholtz_square_freq_real",wkset,true,omega2r);
          for (int e=0; e<numElem; e++) {
            for (int i=0; i<numip; i++) {
              vals(e,i) = -sqrt(omega2r(e,i)/c2r_side_x(e,i));
            }
          }
        }
      }else if(var == "uimag"){
        if(test == 4){
          for (int e=0; e<numElem; e++) {
            for (int i=0; i<numip; i++) {
              vals(e,i) = 0.0; //default; Sommerfeld radiation condition (assuming scalar speed and freq)
            }
          }
        } else if(test == 2) {
          Kokkos::View<AD**> c2r_side_x("c2r_side_x",numElem,numip);
          Kokkos::View<AD**> omega2r("omega2r",numElem,numip);
          this->coefficient("helmholtz_square_speed_real_x",wkset,true,c2r_side_x);
          this->coefficient("helmholtz_square_freq_real",wkset,true,omega2r);
          for (int e=0; e<numElem; e++) {
            for (int i=0; i<numip; i++) {
              vals(e,i) = -sqrt(omega2r(e,i)/c2r_side_x(e,i));
            }
          }
        }
      }
    }
    
    return vals;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the Dirichlet source (mostly for weak BCs)
  //////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD*,AssemblyDevice> boundaryDirichletSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset) const {
    
    int numip = wkset->ip_side.dimension(1);
    Kokkos::View<AD*,AssemblyDevice> vals("dirichlet values",numip); //defaults to zeros
    
    // Specialize for physics, var and test
    
    return vals;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the Dirichlet value
  //////////////////////////////////////////////////////////////////////////////////////
  
  AD boundaryDirichletValue(const string & physics, const string & var, const double & x,
                            const double & y, const double & z, const double & t,
                            const string & gside, const bool & useadjoint) const {
    
    AD val = 0.0;
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      if (simNum == 2) {
        if (gside == "top") {
          val = 1.0;
        }
      }
      
      if (simNum == 44) {
        if (gside == "bottom") {
          val = T_ambient;
        }
      }
      
      if (simNum == 45) {
        //if ((gside == "left") || (gside == "right")){
        if ((gside == "bottom") || (gside == "top")){
          val = T_ambient;
        }
      }
      
      if (simNum == 778) {
        if (gside == "top") {
          val = 0.0;
        }
        else if (gside == "bottom") {
          val = T_ambient;
        }
      }
      
      else if (test == 222) {
        if (gside == "top") {
          val = 0.0;
        }
        else if (gside == "bottom") {
          if (t>=1e-2) {
            val = 1.0;
          }
          else if (t<=1.0e-12) { //steady-state hack
            val = 1.0;
          }
          else {
            val = 1e2*t;
          }
          //val *= exp(-10.0*(x-0.5)*(x-0.5));
        }
      }
    }
    else if (physics == "linearelasticity") {
      if (simNum == 28) {
        if (gside == "surface_5") {
          val = 0.0;
        }
        else if (gside == "surface_6") {
          if (var == "dz") {
            val = 0.1;
          }
          else
            val = 0.0;
        }
      }
      else if (simNum == 59) {
        if (gside == "top") {
          if (var == "dy") {
            AD pp = nbcy[0];
            val = 0.45*pp + 0.55*pp*pp + 1.0*pp*pp*pp;
          }
        }
      }
      else if (simNum == 776) {
        if (gside == "top") {
          if (var == "dy") {
            val = 0.0;
          }
        }
        if (gside == "bottom") {
          if (var == "dy") {
            val = 0.0;
          }
        }
      }
      else if (simNum == 1077) {
        if (gside == "lower_grip_bottom") {
          //if (var == "dy") {
            val = 0.0;
          //}
        }
        if (gside == "upper_grip") {
          if (var == "dy") {
            val = 0.0;
          }
        }
      }

      else if (simNum == 780) {
        double r = sqrt(y*y+z*z);
        double theta = atan2(y,z);
        
        if (gside == "surface_1") {
          if (var == "dx") {
            val = 0.0;
          }
          else if (var == "dy") {
            val = 0.0/3.0*r*cos(theta);
          }
          else if (var == "dz") {
            val = 0.0/3.0*r*sin(theta);
          }
        }
        else if (gside == "surface_2") {
          if (var == "dx") {
            val = 0.001;
          }
          else if (var == "dy") {
            val = 0.0/3.0*r*cos(theta);
          }
          else if (var == "dz") {
            val = 0.0/3.0*r*sin(theta);
          }
        }
      }
      
      
    }
    else if (physics == "navierstokes") {
      if(test == 10){
        if(var == "ux" && gside == "top")
          val = bc_params[0]*(x-x*x);
        else
          val = 0.0;
      }else if(test == 3){
        if(var == "ux" && gside == "top")
          val = bc_params[0]*(x-x*x);
        else if(var == "ux" && gside == "bottom")
          val = bc_params[0]*(x-x*x);
        else
          val = 0.0;
      }else if(test == 30){
        if(var == "ux"){
          if (gside == "top")
            val = bc_params[0]*(x-x*x);
          else if (gside == "bottom")
            val = bc_params[1]*(x*(x-1)*(x-(1.1-cos(t+0.3))));
          else
            val = 0.0;
        }else
          val = 0.0;
      }else{ //"default" behavior
        if(bc_params.size() > 0){
          if(var == "ux")
            val = bc_params[0];
          else if(var == "pr")
            val = bc_params[1];
          else if(var == "uy")
            val = bc_params[2];
          else if(var == "uz")
            val = bc_params[3];
        }else
          val = 0.0;
      }
    }
    else if (physics == "porous") {
      if (simNum == 2) {
        if (gside == "left") {
          val = 1.0;
        }
      }
    }
    else if (physics == "porous2p") {
      // default to zero
    }
    else if (physics == "helmholtz"){
      if(!useadjoint){
        val = this->trueSolution(physics,var,x,y,z,t);
      }
    }
    
    return val;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the initial value
  //////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<double**,AssemblyDevice> getInitial(const string & physics, const DRV & ip,
                                                   const string & var, const double & time,
                                                   const bool & isAdjoint) const {
    
    int numip = ip.dimension(1);
    int numElem = ip.dimension(0);
    Kokkos::View<double**,AssemblyDevice> initial("initial",numElem,numip);
    
    
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      if ((simNum == 44) || (simNum == 111) || (simNum == 45) || (simNum == 112) || (simNum == 144) || (simNum == 145) || (simNum == 222)) {
        if (physics == "thermal_enthalpy") {
          if (var == "e") {
            for (int e=0; e<numElem; e++) {
              for (int j=0; j<numip; j++) {
                initial(e,j) = T_ambient;
              }
            }
          }
          if (var == "H") {
            for (int e=0; e<numElem; e++) {
              for (int j=0; j<numip; j++) {
                initial(e,j) = 0.0;
              }
            }
          }
        }
        else {
          for (int e=0; e<numElem; e++) {
            for (int j=0; j<numip; j++) {
              initial(e,j) = T_ambient;
            }
          }
        }
      }
    }
    else if (physics == "convdiff" ) {
      if (test == 2) {
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            double x = ip(e,j,0);
            if (x>=-1.0) {
              if (x<=0.0)
                initial(e,j) = x+1.0;
              else if (x<=1.0)
                initial(e,j) = 1.0-x;
              
            }
          }
        }
      }
    }
    else if (physics == "linearelasticity") {
      if (simNum == 11 && spaceDim == 3) {
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            double x = ip(e,j,0);
            double y = ip(e,j,1);
            double z = ip(e,j,2);
            
            initial(e,j) = sin(4*PI*x)*sin(4*PI*y)*sin(4*PI*z);
            
          }
        }
      }
      else {
        if (var == "dx") {
          for (int e=0; e<numElem; e++) {
            for (int j=0; j<numip; j++) {
              initial(e,j) = 0.0;
            }
          }
        }
        else if (var == "dy") {
          for (int e=0; e<numElem; e++) {
            for (int j=0; j<numip; j++) {
              initial(e,j) = 0.0;
            }
          }
        }
        else if (var == "dz") {
          for (int e=0; e<numElem; e++) {
            for (int j=0; j<numip; j++) {
              initial(e,j) = 0.0;
            }
          }
        }
      }
    }
    else if (physics == "shallowwater") {
      if (simNum == 1) {
        if (var == "H") {
          for (int e=0; e<numElem; e++) {
            for (int j=0; j<numip; j++) {
              double x = ip(e,j,0);
              double y = ip(e,j,1);
              initial(e,j) = 1.0+0.1*exp(-100.0*(x-0.5)*(x-0.5) - 100.0*(y-0.5)*(y-0.5));
            }
          }
        }
      }
    }
    else if (physics == "navierstokes") {
      if (test == 10) {
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            
            if (var == "ux")
              initial(e,j) = init_params[0].val();
            if (var == "uy")
              initial(e,j) = init_params[2].val();
            if (var == "uz")
              initial(e,j) = init_params[3].val();
            if (var == "pr")
              initial(e,j) = init_params[1].val();
          }
        }
      }
    }
    else if (physics == "porous") {
      // nothing added yet
    }
    else if (physics == "porous2p") {
      // nothing added yet
    }
    return initial;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the response
  //////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> response(const string & physics,
                                              Kokkos::View<AD****,AssemblyDevice> local_soln,
                                              Kokkos::View<AD****,AssemblyDevice> local_soln_grad,
                                              Kokkos::View<AD***,AssemblyDevice> local_psoln,
                                              Kokkos::View<AD****,AssemblyDevice> local_psoln_grad,
                                              const DRV & ip, const double & time) const {
    
    int numip = local_soln.dimension(2);
    int numElem = local_soln.dimension(0);
    Kokkos::View<AD***,AssemblyDevice> resp;
    int block = 0;
    
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
      size_t e_num = this->getVariableIndex(block,"e");
      for (int e=0; e<numElem; e++) {
        for (int j=0; j<numip; j++) {
          if (test == 40 || test == 41) {
            resp(e,0,j) = -0.01*local_soln(e,e_num,j,0);
          }
          else if (simNum == 112) {
            resp(e,0,j) = 0.0;
          }
          else if (simNum == 779) {
            resp(e,0,j) += 0.0*local_soln(e,e_num,j,0);;     //bvbw += to ensure that resp is not overwritten resp from le
          }
          else if (simNum == 900 || simNum == 901 || simNum == 902) {
            double epsilon = 1E-9;
            int numParams = laser_intensity_stoch.size();
            int tindx = int((time+epsilon)/delT)-1;
            
            AD regPenalty = 0.0;
            double x = 0.0;
            double y = 0.0;
            x = ip(e,j,0);
            y = ip(e,j,1);
            if(sqrt((x-0.5)*(x-0.5)+(y-0.75)*(y-0.75)) < 0.25 || sqrt((x-0.5)*(x-0.5)+(y-0.25)*(y-0.25)) < 0.25){ // circular target at top of trajectory
              resp(e,0,j) = 0.5*pow((local_soln(e,e_num,j,0)- responseTarget),2.0) + regCoeff*laser_intensity[tindx]*laser_intensity[tindx];
            }
            //	 if(sqrt((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)) < 0.55){ // circular target at top of trajectory
            //        resp(0,j) = 0.5*pow((local_soln(e_num,j)- responseTarget),2.0);
            //     }
          }
          else if (simNum == 903) {
            double x = 0.0;
            double y = 0.0;
            x = ip(e,j,0);
            y = ip(e,j,1);
            if(sqrt((x-0.5)*(x-0.5)+(y-0.75)*(y-0.75)) < 0.25){ // circular target
              resp(e,0,j) = 0.5*pow((local_soln(e,e_num,j,0)- responseTarget),2.0);
            }
          }
          else // "default"
            resp(e,0,j) = local_soln(e,e_num,j,0);
        }
      }
      
    }
    else if (physics == "linearelasticity") {
      
      int dx_num = this->getVariableIndex(block,"dx");
      int dy_num = this->getVariableIndex(block,"dy");
      int dz_num = this->getVariableIndex(block,"dz");
      size_t e_num = this->getVariableIndex(block,"e");
      if ((simNum == 45) || (simNum == 112)) {
        resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
        double x, y, z;
        AD dx, ddx_dx, ddx_dy, ddx_dz;
        AD dy, ddy_dx, ddy_dy, ddy_dz;
        AD dz, ddz_dx, ddz_dy, ddz_dz;
        AD delta_e;
        Kokkos::View<AD**,AssemblyDevice> stress;
        AD mu, lambda;
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            lambda = this->MaterialProperty("lambda", x, y, z, time);
            mu = this->MaterialProperty("mu", x, y, z, time);
            x = ip(e,j,0);
            y = ip(e,j,1);
            z = 0.0;
            dx = local_soln(e,dx_num,j,0);
            dy = local_soln(e,dy_num,j,0);
            ddx_dx = local_soln_grad(e,dx_num,j,0);
            ddx_dy = local_soln_grad(e,dx_num,j,1);
            ddy_dx = local_soln_grad(e,dy_num,j,0);
            ddy_dy = local_soln_grad(e,dy_num,j,1);
            delta_e = local_soln(e,e_num,j,0) - T_ambient;
            
            stress = this->computeStress(x, y, z, time, dx, ddx_dx, ddx_dy, ddx_dz,
                                         dy, ddy_dx, ddy_dy, ddy_dz, dz, ddz_dx, ddz_dy, ddz_dz,
                                         mu, lambda, delta_e);
            
            AD pratio = lambda / (2.0*(lambda + mu));
            AD s11 = stress(0,0);
            AD s22 = stress(1,1);
            AD s33 = pratio*(s11 + s22);
            AD s12 = stress(0,1);
            if(sqrt((x-0.5)*(x-0.5)+(y-0.25)*(y-0.25)) < 0.25) {
              resp(e,0,j) = 0.5*pow(sqrt(0.5*( (s11 - s22)*(s11 - s22) + (s22 - s33)*(s22 - s33)
                                              + (s33 - s11)*(s33 - s11) + 6.0*s12*s12)) - responseTarget,2);
            }
            //resp(0,j) = dx;
          }
        }
      }
      else if (simNum == 779) {
        resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
        double x, y, z;
        AD dx, ddx_dx, ddx_dy, ddx_dz;
        AD dy, ddy_dx, ddy_dy, ddy_dz;
        AD dz, ddz_dx, ddz_dy, ddz_dz;
        AD delta_e;
        Kokkos::View<AD**,AssemblyDevice> stress;
        AD mu, lambda;
        
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            lambda = this->MaterialProperty("lambda", x, y, z, time);
            mu = this->MaterialProperty("mu", x, y, z, time);
            x = ip(e,j,0);
            y = ip(e,j,1);
            z = 0.0;
            dx = local_soln(e,dx_num,j,0);
            dy = local_soln(e,dy_num,j,0);
            ddx_dx = local_soln_grad(e,dx_num,j,0);
            ddx_dy = local_soln_grad(e,dx_num,j,1);
            ddy_dx = local_soln_grad(e,dy_num,j,0);
            ddy_dy = local_soln_grad(e,dy_num,j,1);
            delta_e = local_soln(e,e_num,j,0) - T_ambient;
            
            stress = this->computeStress(x, y, z, time, dx, ddx_dx, ddx_dy, ddx_dz,
                                         dy, ddy_dx, ddy_dy, ddy_dz, dz, ddz_dx, ddz_dy, ddz_dz,
                                         mu, lambda, delta_e);
            
            AD pratio = lambda / (2.0*(lambda + mu));
            AD s11 = stress(0,0);
            AD s22 = stress(1,1);
            AD s33 = pratio*(s11 + s22);
            AD s12 = stress(0,1);
            
            
            if(x > 0.0 && x < 1.0 && y > 0.3 && y < 0.4 && z> 0.0 && z <1.0 ) { // rectangular target
              resp(e,0,j) = 0.5*pow(sqrt(0.5*( (s11 - s22)*(s11 - s22) + (s22 - s33)*(s22 - s33)
                                            + (s33 - s11)*(s33 - s11) + 6.0*s12*s12)) - responseTarget,2);
            }
            //resp(0,j) = dx;
            
          }
        }
      }
      else if (simNum == 985) {
        resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,2,numip);
        double x, y, z;
        AD dx, ddx_dx, ddx_dy, ddx_dz;
        AD dy, ddy_dx, ddy_dy, ddy_dz;
        AD dz, ddz_dx, ddz_dy, ddz_dz;
        AD delta_e;
        Kokkos::View<AD**,AssemblyDevice> stress;
        AD mu, lambda;
        
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            lambda = this->MaterialProperty("lambda", x, y, z, time);
            mu = this->MaterialProperty("mu", x, y, z, time);
            x = ip(e,j,0);
            y = ip(e,j,1);
            z = 0.0;
            dx = local_soln(e,dx_num,j,0);
            dy = local_soln(e,dy_num,j,0);
            ddx_dx = local_soln_grad(e,dx_num,j,0);
            ddx_dy = local_soln_grad(e,dx_num,j,1);
            ddy_dx = local_soln_grad(e,dy_num,j,0);
            ddy_dy = local_soln_grad(e,dy_num,j,1);
            
            stress = this->computeStress(x, y, z, time, dx, ddx_dx, ddx_dy, ddx_dz,
                                         dy, ddy_dx, ddy_dy, ddy_dz, dz, ddz_dx, ddz_dy, ddz_dz,
                                         mu, lambda, delta_e);
            
            AD pratio = lambda / (2.0*(lambda + mu));
            AD s11 = stress(0,0);
            AD s22 = stress(1,1);
            AD s33 = pratio*(s11 + s22);
            AD s12 = stress(0,1);
            
            resp(e,0,j) = sqrt(0.5*( (s11 - s22)*(s11 - s22) + (s22 - s33)*(s22 - s33)
                          + (s33 - s11)*(s33 - s11) + 6.0*s12*s12));
            resp(e,1,j) = dx;
            
          }
        }
      }
      else if (simNum == 1077) {
        resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
        double y = 0.0;
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            y = ip(e,j,1);
            if (y >= 0.0 && y <= 0.0254) {
              resp(e,0,j) = local_soln(e,dy_num,j,0)/(0.0254*0.0254/2.0*0.0254/2.0);
            }
          }
        }
      }
      else if (ledisp_response_type) {
        resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,spaceDim,numip);
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            resp(e,0,j) = local_soln(e,dx_num,j,0);
            if (spaceDim > 1)
              resp(e,1,j) = local_soln(e,dy_num,j,0);
            if (spaceDim > 2)
              resp(e,2,j) = local_soln(e,dz_num,j,0);
          }
        }
      }
      
      else {
        if (spaceDim == 2) {
          resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,4,numip);
        }
        else if (spaceDim == 3) {
          resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,9,numip);
        }
        
        double x, y, z;
        AD dx, ddx_dx, ddx_dy, ddx_dz;
        AD dy, ddy_dx, ddy_dy, ddy_dz;
        AD dz, ddz_dx, ddz_dy, ddz_dz;
        AD delta_e;
        AD mu;
        AD lambda;
        
        // implement stress computation here (3 entires to return in 2D)
        Kokkos::View<AD**,AssemblyDevice> stress;
        
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            //x = ip(e,j,0);
            //y = ip(e,j,1);
            z = 0.0;
            dx = local_soln(e,dx_num,j,0);
            dy = local_soln(e,dy_num,j,0);
            ddx_dx = local_soln_grad(e,dx_num,j,0);
            ddx_dy = local_soln_grad(e,dx_num,j,1);
            ddy_dx = local_soln_grad(e,dy_num,j,0);
            ddy_dy = local_soln_grad(e,dy_num,j,1);
            if (spaceDim > 2) {
              z = ip(e,j,2);
              dz = local_soln(e,dz_num,j,0);
              ddx_dz = local_soln_grad(e,dx_num,j,2);
              ddy_dz = local_soln_grad(e,dy_num,j,2);
              ddz_dx = local_soln_grad(e,dz_num,j,0);
              ddz_dy = local_soln_grad(e,dz_num,j,1);
              ddz_dz = local_soln_grad(e,dz_num,j,2);
            }
            lambda = this->MaterialProperty("lambda", x, y, z, time);
            if (simNum != 7)
              mu = this->MaterialProperty("mu", x, y, z, time);
            else
              mu = local_psoln(e,0,j);
            
            stress = computeStress(x, y, z, time, dx, ddx_dx, ddx_dy, ddx_dz,
                                   dy, ddy_dx, ddy_dy, ddy_dz, dz, ddz_dx, ddz_dy, ddz_dz,
                                   mu, lambda, delta_e);
            
            if (spaceDim == 2) {
              resp(e,0,j) = stress(0,0);
              resp(e,1,j) = stress(0,1);
              resp(e,2,j) = stress(1,0);
              resp(e,3,j) = stress(1,1);
            }
            else if (spaceDim == 3) {
              resp(e,0,j) = stress(0,0);
              resp(e,1,j) = stress(0,1);
              resp(e,2,j) = stress(0,2);
              resp(e,3,j) = stress(1,0);
              resp(e,4,j) = stress(1,1);
              resp(e,5,j) = stress(1,2);
              resp(e,6,j) = stress(2,0);
              resp(e,7,j) = stress(2,1);
              resp(e,8,j) = stress(2,2);
            }
          }
        }
      }
      
    }
    else if (physics == "navierstokes") {
      if (simNum == 779) {
        resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
        size_t ux_num = this->getVariableIndex(block,"ux");
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            resp(e,0,j) += 0.0*local_soln(e,ux_num,j,0);;     //bvbw += to ensure that resp is not overwritten resp from le
          }
        }
      }
      if (test == 222) {
        resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
        size_t ux_num = this->getVariableIndex(block,"ux");
        size_t uy_num = this->getVariableIndex(block,"uy");
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            AD diff = (local_soln_grad(e,uy_num,j,0) - local_soln_grad(e,ux_num,j,1));
            resp(e,0,j) += diff*diff;     //bvbw += to ensure that resp is not overwritten resp from le
          }
        }
      }
    }
    else if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
      size_t e_num = this->getVariableIndex(block,"e");
      for (int e=0; e<numElem; e++) {
        for (int j=0; j<numip; j++) {
          if (test == 40 || test == 41) {
            resp(e,0,j) = -0.01*local_soln(e,e_num,j,0);
          }
          else {//"default"
            resp(e,0,j) = local_soln(e,e_num,j,0);
            //resp(0,j) = 0.0;
          }
        }
      }
    }
    else if (physics == "porous") {
      resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
      size_t p_num = this->getVariableIndex(block,"p");
      for (int e=0; e<numElem; e++) {
        for (int j=0; j<numip; j++) {
          resp(e,0,j) = local_soln(e,p_num,j,0);
        }
      }
    }
    else if (physics == "porous2p") {
      resp = Kokkos::View<AD***,AssemblyDevice>("response",numElem,1,numip);
      size_t po_num = this->getVariableIndex(block,"po");
      size_t no_num = this->getVariableIndex(block,"no");
      for (int e=0; e<numElem; e++) {
        for (int j=0; j<numip; j++) {
          resp(e,0,j) = local_soln(e,po_num,j,0);
          resp(e,1,j) = local_soln(e,no_num,j,0);
        }
      }
    }
    
    return resp;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get penalty for topological optimization (maybe generalize later...)
  //////////////////////////////////////////////////////////////////////////////////////
  
  AD penaltyTopo(){
    AD penalty_topo = 0.0;
    return penalty_topo;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // return the stress (duplicated from linear elasticity ... needed by response function)
  // Not happy with this duplication.  Need to come up with better strategy.
  //////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD**,AssemblyDevice> computeStress(const double & x, const double & y,
                                                  const double & z, const double & t,
                                                  const AD & dx, const AD & ddx_dx, const AD & ddx_dy, const AD & ddx_dz,
                                                  const AD & dy, const AD & ddy_dx, const AD & ddy_dy, const AD & ddy_dz,
                                                  const AD & dz, const AD & ddz_dx, const AD & ddz_dy, const AD & ddz_dz,
                                                  const AD & mu_val, const AD & lambda_val, const AD & delta_e) const {
    
    Kokkos::View<AD**,AssemblyDevice> stress("stress",3,3);
    
    //AD lambda_val = this->MaterialProperty("lambda", x, y, z, t);
    
    int block = 0;
    
    AD lambda;
    if (incplanestress)
      lambda = 2.0*mu_val;
    else
      lambda = lambda_val;
    
    stress(0,0) = (2.0*mu_val+lambda)*ddx_dx + lambda*(ddy_dy+ddz_dz);
    stress(0,1) = mu_val*(ddx_dy+ddy_dx);
    stress(0,2) = mu_val*(ddx_dz+ddz_dx);
    
    stress(1,0) = mu_val*(ddx_dy+ddy_dx);
    stress(1,1) = (2.0*mu_val+lambda)*ddy_dy + lambda*(ddx_dx+ddz_dz);
    stress(1,2) = mu_val*(ddy_dz+ddz_dy);
    
    stress(2,0) = mu_val*(ddx_dz+ddz_dx);
    stress(2,1) = mu_val*(ddy_dz+ddz_dy);
    stress(2,2) = (2.0*mu_val+lambda)*ddz_dz + lambda*(ddx_dx+ddy_dy);
    
    int e_num = getVariableIndex(block, "e");
    if (e_num >= 0) { // if we are running thermoelasticity
      
      stress(0,0) += -alpha_T*delta_e*(3.0*lambda + 2.0*mu_val);
      stress(1,1) += -alpha_T*delta_e*(3.0*lambda + 2.0*mu_val);
      stress(2,2) += -alpha_T*delta_e*(3.0*lambda + 2.0*mu_val);
    }
    
    return stress;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the weighting function for the objective function
  //////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> weight(const string & physics, const DRV & ip,
                                           const double & time) const {
    int numip = ip.dimension(1);
    int numElem = ip.dimension(0);
    Kokkos::View<AD***,AssemblyDevice> weight;
    
    if (physics == "thermal") {
      weight = Kokkos::View<AD***,AssemblyDevice>("weight",numElem,1,numip);
      double width = 0.25;
      for (int e=0; e<numElem; e++) {
        for (int j=0; j<numip; j++) {
          double x = ip(e,j,0);
          double y = ip(e,j,1);
          if (simNum == 145) {
            double yloc = laser_ypos;
            double xloc = laser_xpos;
            AD path_length;
            AD time_ramp;
            path_length = laser_speed[0]*time;
            
            if (path_length.val() < laser_on) {
              //time_ramp = 1.0;
              time_ramp = 1.0 + 1.0e-5*path_length;
            }
            else if (path_length.val() > laser_off) {
              //time_ramp = 0.0;
              time_ramp = 1.0e-5*path_length;
            }
            else {
              time_ramp = 1.0 - (path_length - laser_on)/(laser_off - laser_on) + 1.0e-5*path_length;
            }
            weight(e,0,j) = time_ramp*exp(-1.0/2.0*( 6.0*(y-yloc - laser_speed[0]*time)*(y-yloc - laser_speed[0]*time)/(e_b*e_b)
                                                    + 6.0*(x-xloc )*(x-xloc)/(e_a*e_a) ));
          }
          else {
            weight(e,0,j) = 1.0;
          }
        }
      }
    }
    
    else {
      // equal weight
      weight = Kokkos::View<AD***,AssemblyDevice>("weight",numElem,1,numip);
      for (int e=0; e<numElem; e++) {
        for (int j=0; j<numip; j++) {
          weight(e,0,j) = 1.0;
        }
      }
    }
    return weight;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the target
  //////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> target(const string & physics, const DRV & ip,
                                           const double & time) const {
    int numip = ip.dimension(1);
    int numElem = ip.dimension(0);
    Kokkos::View<AD***,AssemblyDevice> targ;
    
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,1,numip);
      double width = 0.25;
      for (int e=0; e< numElem; e++) {
        for (int j=0; j<numip; j++) {
          double x = ip(e,j,0);
          double y = ip(e,j,1);
          //targ(0,j) = 0.0*sin(2*PI*ip(0,j,0))*sin(2*PI*ip(0,j,1));
          //targ(0,j) = exp(-5.0*(1.0 - ip(0,j,1))); //gaussian boundary control
          if (simNum == 4) {
            targ(e,0,j) = exp(-4.0*(1.0 - y))*exp(-(x-0.5)*(x-0.5)/(2.0*width*width));
          }
          else if (simNum == 111) {
            targ(e,0,j) = sin(PI*x)*sin(PI*y);
          }
          else if (simNum == 113 || simNum == 114) {
            targ(e,0,j) = 2.0*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*time)
            -PI*sin(PI*x)*sin(PI*y)*cos(PI*time);
          }
          else if (simNum == 112) {
            targ(e,0,j) = 0.0;
          }
          else if (simNum == 77) {
            double p1 = PI;
            double ddiff = 0;
            for ( int k=1 ; k<4; k++) {
              ddiff += diff_params[k].val()*sin(1.02*p1*x)*sin(1.04*p1*y);
              p1 *= 4.0;
            }
            targ(e,0,j) = diff_params[0]+ ddiff;
          }
          /*
           else if (simNum == 115 || simNum == 116) {
           AD addiff = 0.0;
           double freq = 2.0*PI;
           for (int k=0; k<3 ; k++) {
           addiff += diff_params[k]*sin(1.02*freq*x)*sin(1.04*freq*y);
           freq *= 3.0;
           }
           targ(0,j) =   2.0 + diff_params[3]*sin(PI*x)*sin(PI*y)
           + diff_params[4]*x*x*x
           + diff_params[5]*(1.0-y)*(1.0-y) + addiff;
           }
           */
          else if (simNum == 115 || simNum == 116) {
            AD addiff = 0.0;
            double freq1 = 4.0*PI;
            double freq2 = 8.0*PI;
            addiff = exp(diff_params[0])*sin(freq1*x)*sin(freq1*y) +
            exp(diff_params[1])*sin(freq2*x)*sin(freq2*y);
            targ(e,0,j) =   2.0 + exp(diff_params[2])*sin(PI*x)*sin(PI*y)
            + exp(diff_params[3])*x*x*x
            + exp(diff_params[4]*(1.0-y)*(1.0-y)) + addiff;
          }
          
          else if (simNum == 145) {
            double yloc = laser_ypos;
            double xloc = laser_xpos;
            AD path_length;
            AD time_ramp;
            /*
             path_length = laser_speed[0]*time;
             if (path_length.val() < laser_on) {
             //time_ramp = 1.0;
             time_ramp = 1.0 + 1.0e-5*path_length;
             }
             else if (path_length.val() > laser_off) {
             //time_ramp = 0.0;
             time_ramp = 0.0 + 1.0e-5*path_length;
             }
             else {
             time_ramp = 1.0 - (path_length - laser_on)/(laser_off - laser_on) + 1.0e-5*path_length;
             }
             */
            //path_length = laser_speed[0].val()*time;
            /*
             targ(0,j) = time_ramp*(6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
             * exp(-( 3.0*(y-yloc - laser_speed[0]*time)*(y-yloc - laser_speed[0]*time)/(e_b*e_b)
             + 3.0*(x-xloc )*(x-xloc)/(e_b*e_b) ));
             */
            
            targ(e,0,j) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
            * exp(-( 3.0*(y-yloc - laser_speed[0]*time)*(y-yloc - laser_speed[0]*time)/(e_b*e_b)
                    + 3.0*(x-xloc )*(x-xloc)/(e_b*e_b) ));
            
            
          }
          else {
            targ(e,0,j) = 0.0; // zero for testing objective function computation
          }
        }
      }
    }
    else if (physics == "linearelasticity") {
      if (simNum == 22) { // inclusion problem
        
        targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,1,numip);
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            double x = ip(e,j,0);
            double y = ip(e,j,1);
            double th = -25.0*PI/180.0;
            double xc = 0.5;
            double yc = 0.5;
            double rxc = xc*cos(th) + yc*sin(th);
            double ryc = -xc*sin(th) + yc*cos(th);
            double rx = x*cos(th) + y*sin(th);
            double ry = -x*sin(th) + y*cos(th);
            double a = 2;
            double b = 1;
            double erad = 0.04;
            double rad = pow(rx-rxc,2.0)/pow(a,2.0) + pow(ry-ryc,2.0)/pow(b,2.0);
            // E salt = 30e9, E rock = 5e9, nu both = 0.3
            
            if (rad <= erad) {
              targ(e,0,j) = mu[0];
            }
            else {
              targ(e,0,j) = mu[1];
            }
          }
        }
      }
      else if (simNum == 202) {
        targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,2,numip);
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            double x = ip(e,j,0);
            double y = ip(e,j,1);
            double xc = 0.5;
            double yc = 0.5;
            double erad = 0.2;
            double rad = sqrt(pow(x-xc,2.0) + pow(y-yc,2.0));
            targ(e,0,j) = poisson_ratio[0];
            
            if (rad <= erad) {
              if (use_log_E)
                targ(e,1,j) = exp(youngs_mod_in[0]);
              else
                targ(e,1,j) = youngs_mod_in[0];
            }
            else {
              if (use_log_E)
                targ(e,1,j) = exp(youngs_mod_out[0]);
              else
                targ(e,1,j) = youngs_mod_out[0];
            }
          }
        }
      }
      else if (simNum == 500) { // xom stress matching inclusion
        targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,4,numip);
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            double x = ip(e,j,0);
            double y = ip(e,j,1);
            double z = ip(e,j,2);
            AD E1, E2;
            if (use_log_E) {
              E1 = E_ref*exp(youngs_mod_in[0]);
              E2 = E_ref*exp(youngs_mod_out[0]);
            }
            else {
              E1 = E_ref*youngs_mod_in[0];
              E2 = E_ref*youngs_mod_out[0];
            }
            AD nu1 = poisson_ratio[0];
            AD nu2 = poisson_ratio[1];
            
            double lim = 18.75e3/2.0;
            double zlim = 18.75e3;
            
            double tpert;
            double z_scale = 15.0e3;
            double y_scale = 18.75e3/2.0;
            double x_scale = 18.75e3/2.0;
            
            double f, grad, lvlset;
            double r, step, bigr;
            
            if (z > 15.0e3)
              tpert = 0.0;
            else {
              
              double xs = x/x_scale;
              double ys = y/y_scale;
              double zs = z/z_scale;
              
              lvlset = (1.0e0-2.0758*zs + 6.7273*zs*zs - 5.1515*zs*zs*zs)*(1.0e0-sqrt(xs*xs+ys*ys));
              grad = -(1.0e0-2.0758*zs + 6.7273*zs*zs - 5.1515*zs*zs*zs);
              
              size_t numpert = salt_thetavals.size();
              double decay = 10.0;
              double currtheta = atan2(y,x);
              double pert = 0.0;
              for (size_t j=0; j<numpert; j++) {
                pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta,2));
                pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta+2.0*PI,2));
                pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta-2.0*PI,2));
              }
              
              lvlset += 1.0/3.0*pert*(1.0-pow(zs,2));
              f = lvlset - 0.7;
              bigr = sqrt(x*x + y*y);
              r = bigr/x_scale;
              step = -f/grad;
              r = (r + step)*x_scale;
              tpert = tscale*1.0/abs(bigr - r);
            }
            
            bool inside = this->insideSalt(x,y,z);
            
            if (inside) {
              targ(e,0,j) = nu1;
              targ(e,1,j) = E1;
            }
            else {
              targ(e,0,j) = nu2;
              targ(e,1,j) = E2;
            }
            
            if (abs(x - -lim) < 1.0) {
              targ(e,2,j) = tbase*(1.0 - z/zlim) + tpert;
            }
            if (abs(x - lim) < 1.0) {
              targ(e,2,j) = -tbase*(1.0 - z/zlim) - tpert;
            }
            if (abs(y - -lim) < 1.0) {
              targ(e,3,j) = tbase*(1.0 - z/zlim) + tpert;
            }
            if (abs(y - lim) < 1.0) {
              targ(e,3,j) = -tbase*(1.0 - z/zlim) - tpert;
            }
          }
        }
      }
      
      else if ((simNum == 366) || (simNum == 367)) { // xom stress matching inclusion
        targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,4,numip);
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            double x = ip(e,j,0);
            double y = ip(e,j,1);
            double z = ip(e,j,2);
            double xc = 7.5e3;
            double yc = 7.5e3;
            double zc = 7.5e3;
            double a = 2e3;
            double b = 1e3;
            double c = 1e3;
            double erad = 6;
            double rad = pow(x-xc,2.0)/pow(a,2.0) + pow(y-yc,2.0)/pow(b,2.0) + pow(z-zc,2.0)/pow(c,2.0);
            AD E1 = E_ref*youngs_mod_in[0];
            AD E2 = E_ref*youngs_mod_out[0];
            AD nu1 = poisson_ratio[0];
            AD nu2 = poisson_ratio[1];
            
            if (rad <= erad) {
              targ(e,0,j) = nu1;
              targ(e,1,j) = E1;
            }
            else {
              targ(e,0,j) = nu2;
              targ(e,1,j) = E2;
            }
            
            // plot x trac
            double scale = 5.0e2;
            double discx = erad - (y-yc)*(y-yc)/(b*b) - (z-zc)*(z-zc)/(c*c);
            double discz = erad - (y-yc)*(y-yc)/(b*b) - (x-xc)*(x-xc)/(a*a);
            // left face
            if (abs(x - 0) < 1.0) {
              if (discx > 0)
                targ(e,2,j) = 10.0e6*(1.0 - y/15.0e3) + scale*sqrt(a*a*discx);
              //targ(2,j) = scale*sqrt(a*a*discx);
              else
                targ(e,2,j) = 10.0e6*(1.0 - y/15.0e3);
              //targ(2,j) = 0.0;
            }
            if (abs(x - 15.0e3) < 1.0) {
              if (discx > 0)
                targ(e,2,j) = -10.0e6*(1.0 - y/15.0e3) - scale*sqrt(a*a*discx);
              else
                targ(e,2,j) = -10.0e6*(1.0 - y/15.0e3);
            }
            if (abs(z - 0) < 1.0) {
              if (discz > 0)
                targ(e,3,j) = 10.0e6*(1.0 - y/15.0e3) + scale*sqrt(c*c*discz);
              else
                targ(e,3,j) = 10.0e6*(1.0 - y/15.0e3);
            }
            if (abs(z - 15.0e3) < 1.0) {
              if (discz > 0)
                targ(e,3,j) = -10.0e6*(1.0 - y/15.0e3) - scale*sqrt(c*c*discz);
              else
                targ(e,3,j) = -10.0e6*(1.0 - y/15.0e3);
            }
          }
        }
      }
      else if (simNum == 112) {
        targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,1,numip);
      }
      else if (simNum == 779) {
        targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,1,numip);
        for (int e=0; e<numElem; e++) {
          for (int j=0; j<numip; j++) {
            double x = ip(e,j,0);
            double y = ip(e,j,1);
            double z = ip(e,j,2);
            
            if(x > 0.0 && x < 1.0 && y > 0.3 && y < 0.4 && z> 0.0 && z <1.0 ) { // rectangular target
              targ(e,0,j) = 1E-6;
            }
          }
        }
      }
      else {
        targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,2,numip);
      }
    }
    else if (physics == "navierstokes") {
      targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,spaceDim+1,numip);
    }
    else if (physics == "porous") {
      targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,1,numip);
    }
    else if (physics == "porous2p") {
      targ = Kokkos::View<AD***,AssemblyDevice>("target",numElem,2,numip);
    }
    
    return targ;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the volumetric source
  //////////////////////////////////////////////////////////////////////////////////////
  // TMW: remove this!!
  
  Kokkos::View<AD*,AssemblyDevice> volumetricSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset) const {
    
    int numip = wkset->ip.dimension(1);
    double time = wkset->time;
    
    Kokkos::View<AD*,AssemblyDevice> source("source",numip); //defaults to zeros
    Kokkos::View<AD*,AssemblyDevice> source_stoch("stochastic source",numip); //defaults to zeros
    double x,y,z;
    int block = wkset->block;
    
    // Specialize for physics, var and test
    
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      if (test == 21) {
        double xlocs[10] = { 2.05, 2.15, 2.25, 2.35, 2.45,
          2.55, 2.65, 2.75, 2.85, 2.95 };
        double ylocs[2] = { 0.45, 0.55};
        double width_factor = 2*pow(0.01,2);
        size_t laser_counter = 0;
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          for (int i = 0;i < 10;i++) {
            for (int j = 0;j < 2; j++) {
              source(k) += laser_intensity[laser_counter]*exp(-((x - xlocs[i])*(x - xlocs[i]) +
                                                                (y - ylocs[j])*(y - ylocs[j])) / width_factor);
              laser_counter += 1;
            }
          }
        }
      }
      else if (test == 27 || test == 12) {
        // assumes 2D
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                              laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
        }
      }
      else if (test == 900) {  //bvbw this case does not acount for laser movement - 901 has a time dependence; keeping code for now
        int numParams = laser_intensity.size();
        // assumes 2D
        double theta = 2*PI/numParams;
        std::vector<double> xloc(numParams+1), yloc(numParams+1);
        std::vector<double> angle(numParams+1);
        for (int i=0; i<=numParams; i++) {
          angle[i] = atan(theta*i);
          xloc[i] = 0.5 + 0.25 * cos(theta*i);
          yloc[i] = 0.5 + 0.25 * sin(theta*i);
          //cout << angle[i]  << endl;
        }
        
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          if (numParams == 1) {
            source(k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
          }
          if (numParams == 2) {
            if(y >= yloc[0]) {
              source(k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                  laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
            } else {
              source(k) = laser_intensity[1]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                  laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
            }
          }
          if (numParams >= 4) {
            
            for(int i= 0; i < numParams; i++) {
              double angleLocal = atan((y+0.5)/(x+0.5));
              if(angleLocal <= angle[i+1] && angleLocal >= angle[i]) {
                source(k) = laser_intensity[i]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                    laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
                //bvbw
                //	cout << k << " " << i << " " << angleLocal << " " << angle[i+1] << " " << angle[i] << endl;
                //		cout << k << " " << i << " " << angleLocal << " " << angle[i+1] << " " << angle[i] << "  " << laser_intensity[i] << endl;
              }
            }
          }
        }
      }
      else if (test == 901 || test == 902) {
        double epsilon = 1E-9;
        int numParams = laser_intensity_stoch.size();
        int tindx = int((time+epsilon)/delT)-1;
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = laser_intensity[tindx]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                  laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
          
          for (int i =0; i < numParams; i++) {
            source_stoch(k) += laser_intensity_stoch[i]*(exp(-1000.0*(x-0.5)*(x-0.5) -1000.0*(y-0.75)*(y-0.75)));
          }
          
          // if(numParams==1) {
          //   source_stoch(k) = laser_intensity_stoch[0]*(exp(-1000.0*(x-0.5)*(x-0.5) -1000.0*(y-0.75)*(y-0.75)));
          // } else if(numParams == 2) {
          //   source_stoch(k) = laser_intensity_stoch[0]*(exp(-1000.0*(x-0.5)*(x-0.5) -1000.0*(y-0.75)*(y-0.75))) +
          //    	              laser_intensity_stoch[1]*(exp(-1000.0*(x-0.3)*(x-0.3) -1000.0*(y-0.3)*(y-0.3)));
          // }
          
          source(k) = source(k) + source_stoch(k);
          //bvbw
          //  cout << time << "  " << tindx << "  " << laser_intensity[tindx] << "  " << source(k) << endl;
        }
      }
      else if (test == 903) {  // one laser intensity and one stochastic laser intensity parameter
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                              laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
          source_stoch(k) += laser_intensity_stoch[0]*(exp(-1000.0*(x-0.5)*(x-0.5) -1000.0*(y-0.75)*(y-0.75)));
          source(k) = source(k) + source_stoch(k);
        }
      }
      else if (test == 28) {
        double x0 = 20.0;
        double y0 = 60.0;
        double sigmax = 1.0;
        double sigmay = 1.0;
        double magnitude = 1.0;
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = magnitude*exp(-((x-x0)*(x-x0)/(2.0*sigmax*sigmax) + (y-y0)*(y-y0)/(2.0*sigmay*sigmay)));
        }
      }
      else if (test == 1 || test == 2) {
        if (spaceDim == 1) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(0,k,0);
            source(k) = 4*PI*PI*sin(2*PI*x);
          }
        }
        if (spaceDim == 2){
          for (int k=0; k<numip; k++) {
            x = wkset->ip(0,k,0);
            y = wkset->ip(0,k,1);
            source(k) = 8*PI*PI*sin(2*PI*x)*sin(2*PI*y);
          }
        }
        if (spaceDim == 3) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(0,k,0);
            y = wkset->ip(0,k,1);
            z = wkset->ip(0,k,2);
            source(k) = 12*PI*PI*sin(2.0*PI*x)*sin(2.0*PI*y)*sin(2.0*PI*z);
          }
        }
        
      }
      else if (test == 3) {
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = (8*PI*PI*sin(2*PI*time)+2*PI*cos(2*PI*time))*sin(2*PI*x)*sin(2*PI*y);
        }
      }
      else if (test == 33) {
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = time*abs(sin(10.0*PI*time));
        }
      }
      else if (simNum == 44) {
        //double yloc = 15e-3; // top of domain
        double yloc = 2.5e-3; // top of domain
        double zloc = 0.0; // start at end of plate ? delay somehow?
        double xloc = 5e-3; // middle of domain
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          z = wkset->ip(0,k,2);
          source(k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*e_c*PI*sqrt(PI))
          * exp(-( (3.0*pow((x-xloc),2.0))/pow(e_a,2.0)
                  + (3.0*pow((y-yloc),2.0))/pow(e_b,2.0)
                  + (3.0*pow((z-zloc) - laser_speed[0]*time,2.0)/pow(e_c,2.0)) ));
        }
      }
      
      else if (simNum == 45) {
        double yloc = 1e-1; // top of domain
        double zloc = 0.0; // start at end of plate ? delay somehow?
        double xloc = 50e-3; // middle of domain
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          z = wkset->ip(0,k,2);
          source(k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
          * exp(-( 3.0*(x-xloc)*(x-xloc)/(e_a*e_a)
                  + 3.0*(y-yloc + laser_speed[0]*time)*(y-yloc + laser_speed[0]*time)/(e_b*e_b) ));
        }
      }
      else if (simNum == 404) {
        double yloc = laser_ypos; // top of domain
        double zloc = 0.0; // start at end of plate ? delay somehow?
        double xloc = laser_xpos; // middle of domain
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          //z = wkset->ip(0,k,2);
          source(k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
          * exp(-( 3.0*(x-xloc)*(x-xloc)/(e_a*e_a)
                  + 3.0*(y-yloc)*(y-yloc)/(e_b*e_b) ));
        }
      }
      else if (simNum == 112) {
        double yloc = 1e-0; // top of domain
        double zloc = 0.0; // start at end of plate ? delay somehow?
        double xloc = 5e-1; // middle of domain
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          //bvbw - this causes FC error in debug mode when running a 2D
          //       problem
          // z = wkset->ip(0,k,2);
          
          source(k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
          * exp(-( (3.0*pow((x-xloc),2.0))/pow(e_a,2.0)
                  + (3.0*pow((y-yloc) + laser_speed[0]*time,2.0))/pow(e_b,2.0) ));
          
          // source_stoch(k) = (6.0*sqrt(3.0)*laser_intensity_stoch[0]*eta)/(e_a*e_b*PI*sqrt(PI))
          // * exp(-( (3.0*pow((x-0.5),2.0))/pow(e_a,2.0)
          //         + (3.0*pow((y-0.25),2.0))/pow(e_b,2.0) ));
          // source(k) = source(k) + source_stoch(k);
          
        }
      }
      else if (simNum == 111) {
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = tsource[0]*2.0*PI*PI*sin(PI*x)*sin(PI*y);
        }
      }
      else if (simNum == 113 || simNum == 114) {
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = tsource[0]*(2.0*PI*PI*sin(PI*time) + PI*cos(PI*time))
          *sin(PI*x)*sin(PI*y);
        }
      }
      else if (test == 2048){ //debugging SOL interface
        for (int k=0; k<numip; k++) {
          source(k) = tsource[0] + tsource_stoch[0];
        }
      }
      else if (simNum == 62) {
        size_t e_num = this->getVariableIndex(block,"e");
        for (int k =0; k< numip; k++) {
          AD e = wkset->local_soln(0,e_num,k,0);
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) = (y-1)*y*(x-1)*x*e*e*e*e;
        }
      }
      else if (simNum == 48) {
        // assumes 2D
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          for (int i=0; i<laser_intensity.size(); i++) {
            source(k) += laser_intensity[i]*
            exp(-((x - laser_xlocs[i])*(x - laser_xlocs[i]) + (y - laser_ylocs[i])*(y - laser_ylocs[i])) / laser_width[0]);
          }
        }
      }
      else if (simNum == 666) { // verification source for thermal_fr
        // assumes 2D
        for (int k=0; k<numip; k++) {
          x = wkset->ip(0,k,0);
          y = wkset->ip(0,k,1);
          source(k) += sin(2.0*PI*x)*sin(2.0*PI*y);
        }
      }
      else {
        for (int k=0; k<numip; k++) {
          source(k) = tsource[0];
        }
      }
      
    }
    else if (physics == "linearelasticity") {
      if (simNum == 11) {
        if (var == "dx") {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(0,k,0);
            y = wkset->ip(0,k,1);
            source(k) = 20.0*PI*PI*(4.0*x + y + 2.0)*sin(2.0*PI*x)*sin(4.0*PI*y) - 4.0*PI*sin(2.0*PI*x)*cos(4.0*PI*y)
            - 4.0*PI*sin(4.0*PI*x)*cos(2.0*PI*y) - 4.0*PI*sin(2.0*PI*y)*cos(4.0*PI*x) - 20.0*PI*sin(4.0*PI*y)*cos(2.0*PI*x)
            - (-4.0*PI*PI*sin(2.0*PI*x)*sin(4.0*PI*y) + 8.0*PI*PI*cos(4.0*PI*x)*cos(2.0*PI*y))*(6.0*x + 4.0*y + 3);
          }
        }
        else if (var == "dy") {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(0,k,0);
            y = wkset->ip(0,k,1);
            source(k) = 20.0*PI*PI*(4.0*x + y + 2.0)*sin(4.0*PI*x)*sin(2.0*PI*y) - 16.0*PI*sin(2.0*PI*x)*cos(4.0*PI*y) -
            10.0*PI*sin(4.0*PI*x)*cos(2.0*PI*y) - 16.0*PI*sin(2.0*PI*y)*cos(4.0*PI*x) - 6.0*PI*sin(4.0*PI*y)*cos(2.0*PI*x)
            - (-4.0*PI*PI*sin(4.0*PI*x)*sin(2.0*PI*y) + 8.0*PI*PI*cos(2.0*PI*x)*cos(4.0*PI*y))*(6.0*x + 4.0*y + 3.0);
          }
        }
      }
    }
    else if (physics == "navierstokes") {
      if (test == 1) {
        if (var == "ux") {
          for (int k=0; k<numip; k++) {
            source(k) = 1.0;
          }
        }
      }
      if (test == 222) {
        if (var == "uy") {
          for (int k=0; k<numip; k++) {
            source(k) = -1.0;
          }
        }
      }
      else if(source_params.size() > 0){
        for (int k=0; k<numip; k++) {
          if(var == "ux")
            source(k) = source_params[0];
          else if(var == "pr")
            source(k) = source_params[1];
          else if(var == "uy")
            source(k) = source_params[2];
          else if(var == "uz")
            source(k) = source_params[3];
        }
      }
      else {
        for (int k=0; k<numip; k++) {
          source(k) = 0.0;
        }
      }
    }
    else if (physics == "porousHDIV") {
      if (test == 1) {
        if (var == "p") {
          for (int k=0; k<numip; k++) {
            source(k) = 1.0;
          }
        }
      }
    }
    
    
    return source;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the volumetric source
  //////////////////////////////////////////////////////////////////////////////////////
  
  void volumetricSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset,
                        Kokkos::View<AD**> source) const {
    
    int numElem = source.dimension(0);
    int numip = wkset->ip.dimension(1);
    double time = wkset->time;
    
    double x,y,z;
    int block = wkset->block;
    
    // Specialize for physics, var and test
    
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      if (test == 21) {
        double xlocs[10] = { 2.05, 2.15, 2.25, 2.35, 2.45,
          2.55, 2.65, 2.75, 2.85, 2.95 };
        double ylocs[2] = { 0.45, 0.55};
        double width_factor = 2*pow(0.01,2);
        size_t laser_counter = 0;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            for (int i = 0;i < 10;i++) {
              for (int j = 0;j < 2; j++) {
                source(e,k) += laser_intensity[laser_counter]*exp(-((x - xlocs[i])*(x - xlocs[i]) +
                                                                    (y - ylocs[j])*(y - ylocs[j])) / width_factor);
                laser_counter += 1;
              }
            }
          }
        }
      }
      else if (test == 27 || test == 12 ) {
        // assumes 2D
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                  laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
          }
        }
      }
      else if (simNum == 115 || simNum == 116) {
        double ptime = 2.0*PI*time;
        // assumes 2D
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*ptime))*(x-0.5-0.25*cos(laser_speed[0]*ptime)) -
                                                  laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*ptime))*(y-0.5-0.25*sin(laser_speed[0]*ptime)) ));
          }
        }
      }
      else if (test == 900) {  //bvbw this case does not acount for laser movement - 901 has a time dependence; keeping code for now
        int numParams = laser_intensity.size();
        // assumes 2D
        double theta = 2*PI/numParams;
        std::vector<double> xloc(numParams+1), yloc(numParams+1);
        std::vector<double> angle(numParams+1);
        for (int i=0; i<=numParams; i++) {
          angle[i] = atan(theta*i);
          xloc[i] = 0.5 + 0.25 * cos(theta*i);
          yloc[i] = 0.5 + 0.25 * sin(theta*i);
          //cout << angle[i]  << endl;
        }
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            if (numParams == 1) {
              source(e,k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                    laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
            }
            if (numParams == 2) {
              if(y >= yloc[0]) {
                source(e,k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                      laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
              } else {
                source(e,k) = laser_intensity[1]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                      laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
              }
            }
            if (numParams >= 4) {
              
              for(int i= 0; i < numParams; i++) {
                double angleLocal = atan((y+0.5)/(x+0.5));
                if(angleLocal <= angle[i+1] && angleLocal >= angle[i]) {
                  source(e,k) = laser_intensity[i]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                        laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
                  //bvbw
                  //	cout << k << " " << i << " " << angleLocal << " " << angle[i+1] << " " << angle[i] << endl;
                  //		cout << k << " " << i << " " << angleLocal << " " << angle[i+1] << " " << angle[i] << "  " << laser_intensity[i] << endl;
                }
              }
            }
          }
        }
      }
      else if (test == 901 || test == 902) {
        double epsilon = 1E-9;
        int numParams = laser_intensity_stoch.size();
        int tindx = int((time+epsilon)/delT)-1;
        Kokkos::View<AD*,AssemblyDevice> source_stoch("stochastic source",numip);
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = laser_intensity[tindx]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                      laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
            
            for (int i =0; i < numParams; i++) {
              source_stoch(k) += laser_intensity_stoch[i]*(exp(-1000.0*(x-0.5)*(x-0.5) -1000.0*(y-0.75)*(y-0.75)));
            }
            
            // if(numParams==1) {
            //   source_stoch(k) = laser_intensity_stoch[0]*(exp(-1000.0*(x-0.5)*(x-0.5) -1000.0*(y-0.75)*(y-0.75)));
            // } else if(numParams == 2) {
            //   source_stoch(k) = laser_intensity_stoch[0]*(exp(-1000.0*(x-0.5)*(x-0.5) -1000.0*(y-0.75)*(y-0.75))) +
            //    	              laser_intensity_stoch[1]*(exp(-1000.0*(x-0.3)*(x-0.3) -1000.0*(y-0.3)*(y-0.3)));
            // }
            
            source(e,k) = source(e,k) + source_stoch(k);
            //bvbw
            //  cout << time << "  " << tindx << "  " << laser_intensity[tindx] << "  " << source(k) << endl;
          }
        }
      }
      else if (test == 903) {  // one laser intensity and one stochastic laser intensity parameter
        Kokkos::View<AD*,AssemblyDevice> source_stoch("stochastic source",numip);
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = laser_intensity[0]*(exp(-laser_width[0]*(x-0.5-0.25*cos(laser_speed[0]*time))*(x-0.5-0.25*cos(laser_speed[0]*time)) -
                                                  laser_width[0]*(y-0.5-0.25*sin(laser_speed[0]*time))*(y-0.5-0.25*sin(laser_speed[0]*time)) ));
            source_stoch(k) += laser_intensity_stoch[0]*(exp(-1000.0*(x-0.5)*(x-0.5) -1000.0*(y-0.75)*(y-0.75)));
            source(e,k) = source(e,k) + source_stoch(k);
          }
        }
      }
      else if (test == 28) {
        double x0 = 20.0;
        double y0 = 60.0;
        double sigmax = 1.0;
        double sigmay = 1.0;
        double magnitude = 1.0;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = magnitude*exp(-((x-x0)*(x-x0)/(2.0*sigmax*sigmax) + (y-y0)*(y-y0)/(2.0*sigmay*sigmay)));
          }
        }
      }
      else if (test == 1 || test == 2) {
        if (spaceDim == 1) {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = wkset->ip(e,k,0);
              source(e,k) = 4*PI*PI*sin(2*PI*x);
            }
          }
        }
        if (spaceDim == 2){
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = wkset->ip(e,k,0);
              y = wkset->ip(e,k,1);
              source(e,k) = 8*PI*PI*sin(2*PI*x)*sin(2*PI*y);
            }
          }
        }
        if (spaceDim == 3) {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = wkset->ip(e,k,0);
              y = wkset->ip(e,k,1);
              z = wkset->ip(e,k,2);
              source(e,k) = 12*PI*PI*sin(2.0*PI*x)*sin(2.0*PI*y)*sin(2.0*PI*z);
            }
          }
        }
        
      }
      else if (test == 3) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = (8*PI*PI*sin(2*PI*time)+2*PI*cos(2*PI*time))*sin(2*PI*x)*sin(2*PI*y);
          }
        }
      }
      else if (test == 33) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = time*abs(sin(10.0*PI*time));
          }
        }
      }
      else if (test == 333) {      //ceramic burnoff
        AD value;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            // z = wkset->ip(0,k,2);
            if(time >= 0.0 && time <= 0.1) {
              //if(time >= 0.1 && time <= 0.2) {
              for (int i=0; i < surf_xlocs.size(); i++) {
                //	      for (int j=0; j < surf_ylocs.size(); j++) {
                //		for (int l=0; k < surf_zlocs.size(); k++) {
                value += mag*exp(-((x-surf_xlocs[i])*(x-surf_xlocs[i])/2.0*stddevx*stddevx
                                   + (y-surf_ylocs[i])*(y-surf_ylocs[i])/2.0*stddevy*stddevy));
                // value =  mag*exp(-((x-12.5)*(x-12.5)/2.0*stddevx*stddevx
                // 			 + (y-12.5)*(y-12.5)/2.0*stddevy*stddevy))
                //   + mag*exp(-((x-37.5)*(x-37.5)/2.0*stddevx*stddevx
                // 		+ (y-37.5)*(y-37.5)/2.0*stddevy*stddevy));
              }
              source(e,k) = value;
              //		       + (z-surf_zlocs[l])*(z-surf_zlocs[l])/2.0*stddevz*stddevz));
              // if(source(k) > 0.0001) {
              //   cout << x << " " << y << " " << z << endl;
              //   cout << surf_xlocs[i] << " " << surf_ylocs[j] << " " << surf_zlocs[k] << endl;
              //   cout << source(k) << endl;
              // }
              //	      }
            }
          }
        }
      }
      else if (simNum == 44) {
        double xloc = laser_xpos;
        double yloc = laser_ypos;
        double zloc = laser_zpos;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            if (time < toff) {
              x = wkset->ip(e,k,0);
              y = wkset->ip(e,k,1);
              z = wkset->ip(e,k,2);
              //source(k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*e_c*PI*sqrt(PI))
              //* exp(-( (3.0*pow((x-xloc) - cparamx*(cos(laser_speed[0]*time)),2.0))/pow(e_a,2.0)
              //        + (3.0*pow((y-yloc),2.0))/pow(e_b,2.0)
              //        + (3.0*pow((z-zloc) - cparamz*sin(laser_speed[0]*time),2.0)/pow(e_c,2.0)) ));
              source(e,k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*e_c*PI*sqrt(PI))
              * exp(-( (3.0*(x-xloc)*(x-xloc)/(e_a*e_a))
                      + (3.0*(y-yloc)*(y-yloc)/(e_b*e_b))
                      + (3.0*(z - zloc - laser_speed[0]*time)*(z-zloc - laser_speed[0]*time)/(e_c*e_c)) ) );
            }
            else {
              source(e,k) = 0.0;
            }
          }
        }
      }
      
      else if (simNum == 45) {
        double yloc = laser_ypos;
        double xloc = laser_xpos;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            if (time < toff) {
              x = wkset->ip(e,k,0);
              y = wkset->ip(e,k,1);
              source(e,k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
              * exp(-( 3.0*(y-yloc)*(y-yloc)/(e_b*e_b)
                      + 3.0*(x-xloc + laser_speed[0]*time)*(x-xloc + laser_speed[0]*time)/(e_b*e_b) ));
            }
            else {
              source(e,k) = 0.0;
            }
          }
        }
      }
      else if (simNum == 404) {
        double yloc = laser_ypos; // top of domain
        double zloc = 0.0; // start at end of plate ? delay somehow?
        double xloc = laser_xpos; // middle of domain
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            //z = wkset->ip(0,k,2);
            source(e,k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
            * exp(-( 3.0*(x-xloc)*(x-xloc)/(e_a*e_a)
                    + 3.0*(y-yloc)*(y-yloc)/(e_b*e_b) ));
          }
        }
      }
      else if (simNum == 112) {
        double yloc = 1e-0; // top of domain
        double zloc = 0.0; // start at end of plate ? delay somehow?
        double xloc = 5e-1; // middle of domain
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            //bvbw - this causes FC error in debug mode when running a 2D
            //       problem
            // z = wkset->ip(0,k,2);
            
            source(e,k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
            * exp(-( (3.0*pow((x-xloc),2.0))/pow(e_a,2.0)
                    + (3.0*pow((y-yloc) + laser_speed[0]*time,2.0))/pow(e_b,2.0) ));
            
            // source_stoch(k) = (6.0*sqrt(3.0)*laser_intensity_stoch[0]*eta)/(e_a*e_b*PI*sqrt(PI))
            // * exp(-( (3.0*pow((x-0.5),2.0))/pow(e_a,2.0)
            //         + (3.0*pow((y-0.25),2.0))/pow(e_b,2.0) ));
            // source(k) = source(k) + source_stoch(k);
            
          }
        }
      }
      else if (simNum == 111) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = tsource[0]*2.0*PI*PI*sin(PI*x)*sin(PI*y);
          }
        }
      }
      else if (simNum == 113 || simNum == 114) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            //AD eval = wkset->local_soln(e,0,k,0);
            source(e,k) = tsource[0]*(2.0*PI*PI*sin(PI*time) + PI*cos(PI*time))*sin(PI*x)*sin(PI*y);
          }
        }
      }
      else if (test == 2048){ //debugging SOL interface
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            source(e,k) = tsource[0] + tsource_stoch[0];
          }
        }
      }
      else if (simNum == 62) {
        size_t e_num = this->getVariableIndex(block,"e");
        for (int e=0; e<numElem; e++) {
          for (int k =0; k< numip; k++) {
            AD eval = wkset->local_soln(e,e_num,k,0);
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            source(e,k) = (y-1)*y*(x-1)*x*eval*eval*eval*eval;
          }
        }
      }
      else if (simNum == 144 || simNum == 145) {
        double yloc = laser_ypos;
        double xloc = laser_xpos;
        /*
         const double factr = 1.0/(1673.0 - 1648.0);
         double latent_heat = 2.7e5;
         double ad_temp;
         size_t e_num = this->getVariableIndex(block,"e");
         AD e;
         */
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            AD path_length;
            AD time_ramp;
            /*
             path_length = laser_speed[0]*time;
             if (path_length.val() < laser_on) {
             //time_ramp = 1.0;
             time_ramp = 1.0 + 1.0e-5*path_length;
             }
             else if (path_length.val() > laser_off) {
             //time_ramp = 0.0;
             time_ramp = 0.0 + 1.0e-5*path_length;
             }
             else {
             time_ramp = 1.0 - (path_length - laser_on)/(laser_off - laser_on) + 1.0e-5*path_length;
             }
             */
            //path_length = laser_speed[0].val()*time;
            //source(k) = time_ramp*(6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
            source(e,k) = (6.0*sqrt(3.0)*laser_intensity[0]*eta)/(e_a*e_b*PI*sqrt(PI))
            * exp(-( 3.0*(y-yloc - laser_speed[0]*time)*(y-yloc - laser_speed[0]*time)/(e_b*e_b)
                    + 3.0*(x-xloc )*(x-xloc)/(e_b*e_b) ));
            
            /*
             e = wkset->local_soln(e_num,k,0);
             ad_temp = e.val();
             if ((ad_temp > 1648.0) && (ad_temp < 1673.0))
             source(k) -= factr*rho*latent_heat;
             */
          }
        }
      }
      else if (simNum == 48) {
        // assumes 2D
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = wkset->ip(e,k,0);
            y = wkset->ip(e,k,1);
            for (int i=0; i<laser_intensity.size(); i++) {
              source(e,k) += laser_intensity[i]*
              exp(-((x - laser_xlocs[i])*(x - laser_xlocs[i]) + (y - laser_ylocs[i])*(y - laser_ylocs[i])) / laser_width[0]);
            }
          }
        }
      }
      else {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            source(e,k) = tsource[0];
          }
        }
      }
      
    }
    else if (physics == "linearelasticity") {
      if (simNum == 11) {
        if (spaceDim == 2) {
          if (var == "dx") {
            for (int e=0; e<numElem; e++) {
              for (int k=0; k<numip; k++) {
                x = wkset->ip(e,k,0);
                y = wkset->ip(e,k,1);
                source(e,k) = 20.0*PI*PI*(4.0*x + y + 2.0)*sin(2.0*PI*x)*sin(4.0*PI*y) - 4.0*PI*sin(2.0*PI*x)*cos(4.0*PI*y)
                - 4.0*PI*sin(4.0*PI*x)*cos(2.0*PI*y) - 4.0*PI*sin(2.0*PI*y)*cos(4.0*PI*x) - 20.0*PI*sin(4.0*PI*y)*cos(2.0*PI*x)
                - (-4.0*PI*PI*sin(2.0*PI*x)*sin(4.0*PI*y) + 8.0*PI*PI*cos(4.0*PI*x)*cos(2.0*PI*y))*(6.0*x + 4.0*y + 3);
              }
            }
          }
          else if (var == "dy") {
            for (int e=0; e<numElem; e++) {
              for (int k=0; k<numip; k++) {
                x = wkset->ip(e,k,0);
                y = wkset->ip(e,k,1);
                source(e,k) = 20.0*PI*PI*(4.0*x + y + 2.0)*sin(4.0*PI*x)*sin(2.0*PI*y) - 16.0*PI*sin(2.0*PI*x)*cos(4.0*PI*y) -
                10.0*PI*sin(4.0*PI*x)*cos(2.0*PI*y) - 16.0*PI*sin(2.0*PI*y)*cos(4.0*PI*x) - 6.0*PI*sin(4.0*PI*y)*cos(2.0*PI*x)
                - (-4.0*PI*PI*sin(4.0*PI*x)*sin(2.0*PI*y) + 8.0*PI*PI*cos(2.0*PI*x)*cos(4.0*PI*y))*(6.0*x + 4.0*y + 3.0);
              }
            }
          }
        }
        else if (spaceDim == 3) {
          if (var == "dx") {
            for (int e=0; e<numElem; e++) {
              for (int k=0; k<numip; k++) {
                x = wkset->ip(e,k,0);
                y = wkset->ip(e,k,1);
                z = wkset->ip(e,k,2);
                source(e,k) = 16*PI*PI*(3*sin(4*PI*x)*sin(4*PI*y)*sin(4*PI*z) - 1.5*cos(4*PI*x)*cos(4*PI*y)*sin(4*PI*z) - 1.5*cos(4*PI*x)*sin(4*PI*y)*cos(4*PI*z));
              }
            }
          }
          else if (var == "dy") {
            for (int e=0; e<numElem; e++) {
              for (int k=0; k<numip; k++) {
                x = wkset->ip(e,k,0);
                y = wkset->ip(e,k,1);
                z = wkset->ip(e,k,2);
                source(e,k) = 16*PI*PI*(3*sin(4*PI*x)*sin(4*PI*y)*sin(4*PI*z) - 1.5*cos(4*PI*x)*cos(4*PI*y)*sin(4*PI*z) - 1.5*sin(4*PI*x)*cos(4*PI*y)*cos(4*PI*z));
              }
            }
          }
          else if (var == "dz") {
            for (int e=0; e<numElem; e++) {
              for (int k=0; k<numip; k++) {
                x = wkset->ip(e,k,0);
                y = wkset->ip(e,k,1);
                z = wkset->ip(e,k,2);
                source(e,k) = 16*PI*PI*(3*sin(4*PI*x)*sin(4*PI*y)*sin(4*PI*z) - 1.5*cos(4*PI*x)*sin(4*PI*y)*cos(4*PI*z) - 1.5*sin(4*PI*x)*cos(4*PI*y)*cos(4*PI*z));
              }
            }
          }
        }
      }
    }
    else if (physics == "navierstokes") {
      if (test == 1) {
        if (var == "ux") {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              source(e,k) = 1.0;
            }
          }
        }
      }
      if (test == 222) {
        if (var == "uy") {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              source(e,k) = 10.0;
            }
          }
        }
      }
      else if(source_params.size() > 0){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            if(var == "ux")
              source(e,k) = source_params[0];
            else if(var == "pr")
              source(e,k) = source_params[1];
            else if(var == "uy")
              source(e,k) = source_params[2];
            else if(var == "uz")
              source(e,k) = source_params[3];
          }
        }
      }
    }
    else if (physics == "porousHDIV") {
      if (test == 1) {
        if (var == "p") {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              source(e,k) = 1.0;
            }
          }
        }
      }
    }
    else if (physics == "shallowwater") {
      if (test == 2) {
        if (var == "Hu") {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = wkset->ip(e,k,0);
              y = wkset->ip(e,k,1);
              source(e,k) = -20.0*(x-0.75)*0.5*exp(-10.0*(x-0.75)*(x-0.75) - 10.0*(y-0.5)*(y-0.5));
            }
          }
        }
        if (var == "Hv") {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = wkset->ip(e,k,0);
              y = wkset->ip(e,k,1);
              source(e,k) = -20.0*(y-0.5)*0.5*exp(-10.0*(x-0.75)*(x-0.75) - 10.0*(y-0.5)*(y-0.5));
            }
          }
        }
      }
    }
    else if (physics == "helmholtz"){
      if (test == 4){
        AD omega2 = freq_params[0]*freq_params[0];
        if (var == "ureal"){
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = wkset->ip(e,k,0);
              if(spaceDim == 1){
                source(e,k) = (4*PI*PI*(x*x-2.0*x-1.0)-omega2)*sin(2*PI*x) + (2.0-2.0*x)*(2*PI*cos(2*PI*x));
              }else if(spaceDim == 2){
                y = wkset->ip(e,k,1);
                source(e,k) = (8*PI*PI*(x*x-2.0*x-1.0)-omega2)*sin(2*PI*x)*sin(2*PI*y) + (2.0-2.0*x)*(2*PI*cos(2*PI*x)*sin(2*PI*y));
              }else if(spaceDim == 3){
                y = wkset->ip(e,k,1);
                z = wkset->ip(e,k,2);
                source(e,k) = (12*PI*PI*(x*x-2.0*x-1.0)-omega2)*sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z) + (2.0-2.0*x)*(2*PI*cos(2*PI*x)*sin(2*PI*y)*sin(2*PI*z));
              }
            }
          }
        }
        else if (var == "uimag"){
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = wkset->ip(e,k,0);
              if(spaceDim == 1){
                source(e,k) = (4*PI*PI*(x*x+2.0*x-1.0)-omega2)*sin(2*PI*x) + (-2.0-2.0*x)*(2*PI*cos(2*PI*x));
              }else if(spaceDim == 2){
                y = wkset->ip(e,k,1);
                source(e,k) = (8*PI*PI*(x*x+2.0*x-1.0)-omega2)*sin(2*PI*x)*sin(2*PI*y) + (-2.0-2.0*x)*(2*PI*cos(2*PI*x)*sin(2*PI*y));
              }else if(spaceDim == 3){
                y = wkset->ip(e,k,1);
                z = wkset->ip(e,k,2);
                source(e,k) = (12*PI*PI*(x*x+2.0*x-1.0)-omega2)*sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z) + (-2.0-2.0*x)*(2*PI*cos(2*PI*x)*sin(2*PI*y)*sin(2*PI*z));
              }
            }
          }
        }
      } else if (test ==2) {
        double source_size = 0.5;
        double source_shift = 2.0;
        if(var == "ureal") {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = wkset->ip(e,k,0);
              y = wkset->ip(e,k,1);
              source(e,k) = std::max(pow(source_size,2.0)-x*x-pow(y-source_shift,2.0),0.0)
              + std::max(pow(source_size,2.0)-x*x-pow(y+source_shift,2.0),0.0);
            }
          }
        } else {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              source(e,k) = 0.0;
            }
          }
        }
      }
    }
  }
  
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  //TMW remove this!
  
  Kokkos::View<AD*,AssemblyDevice> coefficient(const string & name, const Teuchos::RCP<workset> & wkset, const bool & onSide) const {
    
    DRV ip = wkset->ip;
    if (onSide)
      ip = wkset->ip_side;
    
    int numip = ip.dimension(1);
    
    Kokkos::View<AD*,AssemblyDevice> vals("coefficient values",numip); //defaults to zeros
    
    double x,y,z;
    int block= wkset->block;
    // Specialize for physics, var and test
    
    if (name == "thermal_diffusion") {
      AD ddiff;
      for (int k=0; k<numip; k++) {
        x = ip(0,k,0);
        if (spaceDim > 1)
          y = ip(0,k,1);
        if (spaceDim > 2)
          z = ip(0,k,2);
        
        if(test == 40 || test == 41){
          int n = round(sqrt(diff_params.size()));
          if(n*n != diff_params.size())
            cout << "AAAHHH need square number of parameters...number of parameters: " << diff_params.size() << endl;
          
          double dx = 1.0/(n-1);
          double dy = 1.0/(n-1);
          double qmin = 1.e-3;
          
          
          //[0,1]x[0,1] domain
          int indTopRight = std::max(ceil(x/dx),1.0)*n + std::max(ceil(y/dy),1.0);
          int indTopLeft = indTopRight - n;
          int indBotRight = indTopRight - 1;
          int indBotLeft = indBotRight - n;
          
          double xi = (x-floor(x/dx)*dx)/dx;
          double eta = (y-floor(y/dy)*dy)/dy;
          
          AD q = diff_params[indBotLeft]*(1-xi-eta+xi*eta)
          + diff_params[indTopLeft]*(eta-xi*eta)
          + diff_params[indTopRight]*(xi*eta)
          + diff_params[indBotRight]*(xi-xi*eta);
          
          if(maxdiff.size() > 0)
            ddiff = qmin + (maxdiff[0]-qmin)*(3.0*q*q - 2.0*q*q*q);
          else
            ddiff = qmin + (1.0-qmin)*(3.0*q*q - 2.0*q*q*q);
        }
        else if (test == 7) {
          // double ddiff = 1.0*(2.0+cos(1.0*x)*cos(3.0*y)*cos(3.0*z)) +
          //        0.5*(2.0+cos(2.0*x)*cos(1.1*y)*cos(4.0*z)) +
          //        0.25*(2.0+cos(4.0*x)*cos(2.1*y)*cos(6.0*z)) +
          //        0.125*(2.0+cos(8.0*x)*cos(4.1*y)*cos(8.0*z));
          
          // diff = ddiff;
          //double denom = diff_params[0];
          AD p1 = diff_params[1];
          for ( int k=0 ; k<8; k++) {
            ddiff += diff_params[k+2]*(1.25+cos(1.0*p1*x)*sin(4.0*p1*y)*cos(3.0*p1*z));
            //denom *= 0.5;
            p1 *= 2.0;
          }
        }
        else if (simNum == 77) {
          double p1 = PI;
          for ( int k=1 ; k<4; k++) {
            ddiff += diff_params[k]*sin(1.02*p1*x)*sin(1.04*p1*y);
            p1 *= 4.0;
          }
          ddiff = ddiff + diff_params[0];
        }
        //else if ((simNum == 44) || (simNum == 45)) {
        else if ((simNum == 44) || (simNum == 45) || (simNum == 112)) {
          size_t e_num = this->getVariableIndex(block,"e");
          AD e;
          if (onSide) {
            e = wkset->local_soln_side(0,e_num,k,0);
          }
          else {
            e = wkset->local_soln(0,e_num,k,0);
          }
          double temps[] = {0.0,100.0,200.0,300.0,400.0,500.0,600.0,
            700.0,800.0,900.0,1000.0,1100.0,1200.0,
            1300.0,1400.0,1500.0,1600.0,1700.0,1800.0, 24000.0};
          double vals[] = {7.9318,10.1727,12.2853,14.2696,16.1255,17.8531,19.4524,
            20.9234,22.266,23.4803,24.5662,25.5238,26.3531,27.054,
            27.6267,28.0709,28.3869, 28.5, 28.7, 30.0};
          for (int i=0; i < 19; i++ ) {
            if ((e.val() >= temps[i]) && (e.val() < temps[i+1]))
              ddiff = (vals[i+1] - vals[i])*(e - temps[i])/(temps[i+1] - temps[i]) + vals[i];
          }
        }
        else if (simNum == 900 ) {
          int numParams = diff_params.size();
          if(numParams == 1 ) {
            ddiff = abs(diff_stoch_params[0]);
            //bvbw
            //	    cout << ddiff << endl;
          } else {
            if (x <= 0.5) {
              ddiff = diff_stoch_params[0];
            } else {
              ddiff = diff_stoch_params[1];
            }
          }
        }
        else if (simNum == 901) {
          double epsilon = 1E-9;
          int numParams = diff_stoch_params.size();
          int index = int((x-epsilon)/(1.0/numParams));  // assumes xmax-xmin = 1.0 - hence the 1.0 in the calc in case it changes
          ddiff = abs(diff_stoch_params[index]);
        }
        else if (simNum == 902) {     // deterministic case
          int numParams = diff_params.size();
          double epsilon = 1E-9;
          int index = int((x-epsilon)/(1.0/numParams));  // assumes xmax-xmin = 1.0 - hence the 1.0 in the calc in case it changes
          ddiff = diff_params[index];
        }
        else if (simNum == 903) {     // simple case
          int numParams = diff_params.size();
          double epsilon = 1E-9;
          int index = int((x-epsilon)/(1.0/numParams));  // assumes xmax-xmin = 1.0 - hence the 1.0 in the calc in case it changes
          ddiff = diff_params[index];
        }
        else if (simNum == 63) {
          size_t e_num = this->getVariableIndex(block,"e");
          AD e;
          if (onSide) {
            e = wkset->local_soln_side(0,e_num,k,0);
          }
          else {
            e = wkset->local_soln(0,e_num,k,0);
          }
          ddiff = 4.0 + 1.0*e;
        }
        else { //"default"
          //diff = 1.0;
          ddiff = diff_params[0];
        }
        vals(k) = ddiff;
      }
    }
    else if (name == "heat_capacity") {
      for (int k=0; k<numip; k++) {
        x = ip(0,k,0);
        if (spaceDim > 1)
          y = ip(0,k,1);
        if (spaceDim > 2)
          z = ip(0,k,2);
        
        size_t e_num = this->getVariableIndex(block,"e");
        AD e = wkset->local_soln(0,e_num,k,0);
        
        AD hc = 1.0;
        if ((simNum == 44) || (simNum == 45) || (simNum == 112)) {
          //double temps[] = {200.0,300.0,600.0,1400.0,1550.0,1700.0};
          //double vals[] = {425.0,480.0,550.0,675.0,693.0,725.0};
          //double temps[] = {0.0, 200.0,300.0,600.0,1400.0,1550.0,1700.0};
          //double vals[] = {380.0, 425.0,480.0,550.0,675.0,693.0,725.0};
          //for (int i=0; i < 5; i++ ) {
          /*
           for (int i=0; i < 6; i++ ) {
           if ((e.val() >= temps[i]) && (e.val() < temps[i+1]))
           hc = (vals[i+1] - vals[i])*(e - temps[i])/(temps[i+1] - temps[i]) + vals[i];
           }
           */
          double temps[] = {200.0, 300.0,600.0,1400.0,1550.0,1650.0,1673.0,1723.0,1750.0, 1800.0, 24000.0};
          double vals[] = {425.0,480.0,550.0,675.0,693.0,714.0,6419.24,6430.0,736.0, 744.0, 750.0};
          for (int i=0; i < 10; i++ ) {
            if ((e.val() >= temps[i]) && (e.val() < temps[i+1]))
              hc = (vals[i+1] - vals[i])*(e - temps[i])/(temps[i+1] - temps[i]) + vals[i];
          }
        }
        vals(k) = hc;
        //vals(k) = cp_params[0];
      }
    }
    else if (name == "density") {
      if ((simNum == 44) || (simNum == 45) || (simNum == 112)) {
        for (int k=0; k<numip; k++) {
          vals(k) = rho;
        }
      }
      else {
        for (int k=0; k<numip; k++) {
          vals(k) = 1.0;
        }
      }
    }
    else if (name == "viscosity") {
      for (int k=0; k<numip; k++) {
        vals(k) = 1.0;
      }
    }
    else if (name == "permeability") {
      for (int k=0; k<numip; k++) {
        vals(k) = 1.0;
      }
    }
    else if (name == "porosity") {
      for (int k=0; k<numip; k++) {
        vals(k) = 0.1;
      }
    }
    else if (name == "lambda") {
      for (int k=0; k<numip; k++) {
        x = ip(0,k,0);
        if (spaceDim > 1)
          y = ip(0,k,1);
        if (spaceDim > 2)
          z = ip(0,k,2);
        
        if (simNum == 10) {
          if (onSide)
            vals(k) = wkset->local_param_side(0,0,k);
          else
            vals(k) = wkset->local_param(0,0,k);
        }
        else if (simNum == 25) {
          AD rval = 1.0;
          vals(k) = lambda1[0]+rval;
        }
        else if (simNum == 26 || simNum == 28 || simNum == 29) {
          AD E = 0.0;
          if (onSide)
            E = wkset->local_param_side(0,0,k);
          else
            E = wkset->local_param(0,0,k);
          
          AD nu = poisson_ratio[0];
          vals(k) = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
        }
        else if ((simNum == 36) || (simNum == 200) || (simNum == 44) || (simNum == 45) || (simNum == 48)) {
          AD E = youngs_mod[0];
          AD nu = poisson_ratio[0];
          vals(k) = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
        }
        else if (simNum == 11) {
          if (spaceDim ==2) {
            vals(k) = 2.0*x + 3.0*y + 1.0;
          }
          else if (spaceDim == 3) {
            vals(k) = 1.0;
          }
        }
        else if ((simNum == 366) || (simNum == 367)) { // xom stress matching inclusion
          double xc = 7.5e3;
          double yc = 7.5e3;
          double zc = 7.5e3;
          double a = 2e3;
          double b = 1e3;
          double c = 1e3;
          double erad = 6;
          double rad = pow(x-xc,2.0)/pow(a,2.0) + pow(y-yc,2.0)/pow(b,2.0) + pow(z-zc,2.0)/pow(c,2.0);
          AD E1 = E_ref*youngs_mod_in[0];
          AD E2 = E_ref*youngs_mod_out[0];
          AD nu1 = poisson_ratio[0];
          AD nu2 = poisson_ratio[1];
          if (rad <= erad)
            vals(k) = (E1*nu1)/((1.0+nu1)*(1.0-2.0*nu1));
          else
            vals(k) = (E2*nu2)/((1.0+nu2)*(1.0-2.0*nu2));
        }
        else if (simNum == 22) { // xom salt body problem
          double th = -25.0*PI/180.0;
          double xc = 0.5;
          double yc = 0.5;
          double rxc = xc*cos(th) + yc*sin(th);
          double ryc = -xc*sin(th) + yc*cos(th);
          double rx = x*cos(th) + y*sin(th);
          double ry = -x*sin(th) + y*cos(th);
          double a = 2;
          double b = 1;
          double erad = 0.04;
          double rad = pow(rx-rxc,2.0)/pow(a,2.0) + pow(ry-ryc,2.0)/pow(b,2.0);
          // E salt = 30e9, E rock = 5e9, nu both = 0.3
          
          if (rad <= erad) {
            vals(k) = lambda1[0];
          }
          else {
            vals(k) = lambda1[1];
          }
          
        }
        else
          vals(k) = lambda1[0];
        
      }
      
    }
    else if (name == "mu") {
      for (int k=0; k<numip; k++) {
        x = ip(0,k,0);
        if (spaceDim > 1)
          y = ip(0,k,1);
        if (spaceDim > 2)
          z = ip(0,k,2);
        
        if (simNum == 10 || simNum == 2) {
          vals(k) = mu[0];
        }
        else if (simNum == 87) {
          if (y < 0.25)
            vals(k) = mu[0];
          else if (y >= 0.25 && y < 0.5)
            vals(k) = mu[1];
          else if (y >= 0.5 && y < 0.75)
            vals(k) = mu[2];
          else
            vals(k) = mu[3];
        }
        else if (simNum == 88) {
          if (y < 0.25)
            vals(k) = exp(mu[0]);
          else if (y >= 0.25 && y < 0.5)
            vals(k) = exp(mu[1]);
          else if (y >= 0.5 && y < 0.75)
            vals(k) = exp(mu[2]);
          else
            vals(k) = exp(mu[3]);
        }
        else if (simNum == 89) {
          if (y < 0.25)
            vals(k) = exp(mu_opt[0]);
          else if (y >= 0.25 && y < 0.5)
            vals(k) = exp(mu[0]);
          else if (y >= 0.5 && y < 0.75)
            vals(k) = exp(mu[1]);
          else
            vals(k) = exp(mu[2]);
        }
        else if (simNum == 25) {
          AD rval = 1.0;
          vals(k) = mu[0]+rval;
        }
        else if (simNum == 26 || simNum == 28 || simNum == 29) {
          AD E = 0.0;
          if (onSide)
            E = wkset->local_param_side(0,0,k);
          else
            E = wkset->local_param(0,0,k);
          
          AD nu = poisson_ratio[0];
          vals(k) = E/(2.0*(1.0+nu));
          
        }
        else if ((simNum == 366) || (simNum == 367)) { // xom stress matching inclusion
          double xc = 7.5e3;
          double yc = 7.5e3;
          double zc = 7.5e3;
          double a = 2e3;
          double b = 1e3;
          double c = 1e3;
          double erad = 6;
          double rad = pow(x-xc,2.0)/pow(a,2.0) + pow(y-yc,2.0)/pow(b,2.0) + pow(z-zc,2.0)/pow(c,2.0);
          AD E1 = E_ref*youngs_mod_in[0];
          AD E2 = E_ref*youngs_mod_out[0];
          AD nu1 = poisson_ratio[0];
          AD nu2 = poisson_ratio[1];
          if (rad <= erad)
            vals(k) = E1/(2.0*(1.0+nu1));
          else
            vals(k) = E2/(2.0*(1.0+nu2));
        }
        
        else if (simNum == 7) {
          AD mu = 0.0;
          if (onSide)
            mu = wkset->local_param_side(0,0,k);
          else
            mu = wkset->local_param(0,0,k);
          
          vals(k) = mu;
          
        }
        else if (simNum == 6) {
          AD mu = 0.0;
          if (onSide)
            mu = wkset->local_param_side(0,0,k);
          else
            mu = wkset->local_param(0,0,k);
          vals(k) = mu;
          
        }
        else if ((simNum == 36) || (simNum == 200) || (simNum == 44) || (simNum == 45) || (simNum == 48)) {
          AD E = youngs_mod[0];
          AD nu = poisson_ratio[0];
          vals(k) = E/(2.0*(1.0+nu));
        }
        else if (simNum == 11) {
          if (spaceDim ==2) {
            vals(k) = 4.0*x + y + 2.0;
          }
          else if (spaceDim ==3) {
            vals(k) = 0.5;
          }
        }
        else if (simNum == 22) { // xom salt body problem
          double th = -25.0*PI/180.0;
          double xc = 0.5;
          double yc = 0.5;
          double rxc = xc*cos(th) + yc*sin(th);
          double ryc = -xc*sin(th) + yc*cos(th);
          double rx = x*cos(th) + y*sin(th);
          double ry = -x*sin(th) + y*cos(th);
          double a = 2;
          double b = 1;
          double erad = 0.04;
          double rad = pow(rx-rxc,2.0)/pow(a,2.0) + pow(ry-ryc,2.0)/pow(b,2.0);
          // E salt = 30e9, E rock = 5e9, nu both = 0.3
          
          if (rad <= erad) {
            vals(k) = mu[0];
          }
          else {
            vals(k) = mu[1];
          }
          
          
        }
        else
          vals(k) = mu[0];
        
        
      }
    }
    return vals;
    
  }
  
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  void coefficient(const string & name, const Teuchos::RCP<workset> & wkset, const bool & onSide,
                   Kokkos::View<AD**> vals) const {
    
    DRV ip = wkset->ip;
    if (onSide)
      ip = wkset->ip_side;
    
    int numElem = vals.dimension(0);
    int numip = ip.dimension(1);
    
    double x,y,z;
    int block = wkset->block;
    // Specialize for physics, var and test
    
    if (name == "thermal_diffusion") {
      
      if(test == 40 || test == 41){
        int n = round(sqrt(diff_params.size()));
        if(n*n != diff_params.size())
          cout << "AAAHHH need square number of parameters...number of parameters: " << diff_params.size() << endl;
        
        double dx = 1.0/(n-1);
        double dy = 1.0/(n-1);
        double qmin = 1.e-3;
        
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            if (spaceDim > 1)
              y = ip(e,k,1);
            if (spaceDim > 2)
              z = ip(e,k,2);
            
            //[0,1]x[0,1] domain
            int indTopRight = std::max(ceil(x/dx),1.0)*n + std::max(ceil(y/dy),1.0);
            int indTopLeft = indTopRight - n;
            int indBotRight = indTopRight - 1;
            int indBotLeft = indBotRight - n;
            
            double xi = (x-floor(x/dx)*dx)/dx;
            double eta = (y-floor(y/dy)*dy)/dy;
            
            AD q = diff_params[indBotLeft]*(1-xi-eta+xi*eta)
            + diff_params[indTopLeft]*(eta-xi*eta)
            + diff_params[indTopRight]*(xi*eta)
            + diff_params[indBotRight]*(xi-xi*eta);
            
            if(maxdiff.size() > 0)
              vals(e,k) = qmin + (maxdiff[0]-qmin)*(3.0*q*q - 2.0*q*q*q);
            else
              vals(e,k) = qmin + (1.0-qmin)*(3.0*q*q - 2.0*q*q*q);
          }
        }
      }
      else if (test == 7) {
        // double ddiff = 1.0*(2.0+cos(1.0*x)*cos(3.0*y)*cos(3.0*z)) +
        //        0.5*(2.0+cos(2.0*x)*cos(1.1*y)*cos(4.0*z)) +
        //        0.25*(2.0+cos(4.0*x)*cos(2.1*y)*cos(6.0*z)) +
        //        0.125*(2.0+cos(8.0*x)*cos(4.1*y)*cos(8.0*z));
        
        // diff = ddiff;
        //double denom = diff_params[0];
        AD p1 = diff_params[1];
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            if (spaceDim > 1)
              y = ip(e,k,1);
            if (spaceDim > 2)
              z = ip(e,k,2);
            
            for ( int j=0 ; j<8; j++) {
              vals(e,k) += diff_params[j+2]*(1.25+cos(1.0*p1*x)*sin(4.0*p1*y)*cos(3.0*p1*z));
              //denom *= 0.5;
              p1 *= 2.0;
            }
          }
        }
      }
      else if (simNum == 77) {
        double p1 = PI;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            if (spaceDim > 1)
              y = ip(e,k,1);
            if (spaceDim > 2)
              z = ip(e,k,2);
            
            for ( int j=1 ; j<4; j++) {
              vals(e,k) += diff_params[j]*sin(1.02*p1*x)*sin(1.04*p1*y);
              p1 *= 4.0;
            }
            vals(e,k) += diff_params[0];
          }
        }
      }
      //else if ((simNum == 44) || (simNum == 45) || (simNum == 112)) {
      else if (simNum == 112) {
        size_t e_num = this->getVariableIndex(block,"e");
        AD eval;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            if (spaceDim > 1)
              y = ip(e,k,1);
            if (spaceDim > 2)
              z = ip(e,k,2);
            
            if (onSide) {
              eval = wkset->local_soln_side(e,e_num,k,0);
            }
            else {
              eval = wkset->local_soln(e,e_num,k,0);
            }
            double temps[] = {0.0,100.0,200.0,300.0,400.0,500.0,600.0,
              700.0,800.0,900.0,1000.0,1100.0,1200.0,
              1300.0,1400.0,1500.0,1600.0,1700.0,1800.0, 24000.0};
            double vvals[] = {7.9318,10.1727,12.2853,14.2696,16.1255,17.8531,19.4524,
              20.9234,22.266,23.4803,24.5662,25.5238,26.3531,27.054,
              27.6267,28.0709,28.3869, 28.5, 28.7, 30.0};
            for (int i=0; i < 19; i++ ) {
              if ((eval.val() >= temps[i]) && (eval.val() < temps[i+1]))
                vals(e,k) = (vvals[i+1] - vvals[i])*(eval - temps[i])/(temps[i+1] - temps[i]) + vvals[i];
            }
          }
        }
      }
      else if (simNum == 900 ) {
        int numParams = diff_params.size();
        
        if(numParams == 1 ) {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              vals(e,k) = abs(diff_stoch_params[0]);
            }
          }
        }
        else {
          for (int e=0; e<numElem; e++) {
            for (int k=0; k<numip; k++) {
              x = ip(e,k,0);
              
              if (x <= 0.5) {
                vals(e,k) = diff_stoch_params[0];
              } else {
                vals(e,k) = diff_stoch_params[1];
              }
            }
          }
        }
      }
      else if (simNum == 901) {
        double epsilon = 1E-9;
        int numParams = diff_stoch_params.size();
        int index = int((x-epsilon)/(1.0/numParams));  // assumes xmax-xmin = 1.0 - hence the 1.0 in the calc in case it changes
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = abs(diff_stoch_params[index]);
          }
        }
      }
      else if (simNum == 902) {     // deterministic case
        int numParams = diff_params.size();
        double epsilon = 1E-9;
        int index = int((x-epsilon)/(1.0/numParams));  // assumes xmax-xmin = 1.0 - hence the 1.0 in the calc in case it changes
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = diff_params[index];
          }
        }
      }
      else if (simNum == 903) {     // simple case
        int numParams = diff_params.size();
        double epsilon = 1E-9;
        int index = int((x-epsilon)/(1.0/numParams));  // assumes xmax-xmin = 1.0 - hence the 1.0 in the calc in case it changes
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = diff_params[index];
          }
        }
        
      }
      else if (simNum == 63) {
        size_t e_num = this->getVariableIndex(block,"e");
        AD eval;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            
            if (onSide) {
              eval = wkset->local_soln_side(e,e_num,k,0);
            }
            else {
              eval = wkset->local_soln(e,e_num,k,0);
            }
            vals(e,k) = 4.0 + 1.0*eval;
          }
        }
      }
      else if (simNum == 115 || simNum == 116) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            y = ip(e,k,1);
            AD addiff = 0.0;
            double freq1 = 5.0*PI;
            double freq2 = 10.0*PI;
            addiff = exp(diff_params[0])*sin(freq1*x)*sin(freq1*y) +
            exp(diff_params[1])*sin(freq2*x)*sin(freq2*y);
            vals(e,k) =   2.0 + exp(diff_params[2])*sin(PI*x)*sin(PI*y)
            + exp(diff_params[3])*x*x*x
            + exp(diff_params[4])*(1.0-y)*(1.0-y) + addiff;
          }
        }
      }
      else { //"default"
        //diff = 1.0;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = diff_params[0];
          }
        }
      }
    }
    else if (name == "heat_capacity") {
      //if (simNum == 44 || simNum == 45 || simNum == 112 || simNum == 144) {
      if (simNum == 44 || simNum == 45 || simNum == 112) {
        
        double temps[] = {200.0, 300.0,600.0,1400.0,1550.0,1650.0,1673.0,1723.0,1750.0, 10000.0};
        double vvals[] = {425.0,480.0,550.0,675.0,693.0,714.0,6419.24,6430.0,736.0, 736.05};
        size_t e_num = this->getVariableIndex(block,"e");
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            AD eval = wkset->local_soln(e,e_num,k,0);
            for (int i=0; i < 9; i++ ) {
              if ((eval.val() >= temps[i]) && (eval.val() < temps[i+1]))
                vals(e,k) = (vvals[i+1] - vvals[i])*(eval - temps[i])/(temps[i+1] - temps[i]) + vvals[i];
            }
          }
        }
      }
      else if (simNum == 114) {
        size_t e_num = this->getVariableIndex(block,"e");
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            AD eval = wkset->local_soln(e,e_num,k,0);
            vals(e,k) = 2.0 + eval*eval;
          }
        }
      }
      else if (simNum == 115) {
        size_t e_num = this->getVariableIndex(block,"e");
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            AD eval = wkset->local_soln(e,e_num,k,0);
            vals(e,k) = 2.0 + 10.0*eval*eval;
          }
        }
      }
      else if (simNum == 144 || simNum == 145) {
        size_t e_num = this->getVariableIndex(block,"e");
        /*
         const double factr = 1.0/(1673.0 - 1648.0);
         double latent_heat = 2.7e5;
         double ad_temp;
         */
        AD eval;
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            eval = wkset->local_soln(e,e_num,k,0);
            vals(e,k) = 320.3 + 0.379*eval;
            /*
             ad_temp = e.val();
             if ((ad_temp > 1648.0) && (ad_temp < 1673.0)) {
             vals(k) += factr*latent_heat*(1.0 + 1.0e-10*e);
             }
             */
          }
        }
      }
      else {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 1.0;
          }
        }
      }
    }
    else if (name == "density") {
      if ((simNum == 44) || (simNum == 45) || (simNum == 112) || (simNum == 144) || (simNum == 145)) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = rho;
          }
        }
      }
      else {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 1.0;
          }
        }
      }
    }
    else if (name == "viscosity") {
      if (test == 222) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = viscosity_params[0];
          }
        }

      }
      else {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 1.0;
          }
        }
      }
    }
    else if (name == "permeability") {
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          vals(e,k) = 1.0;
        }
      }
    }
    else if (name == "porosity") {
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          vals(e,k) = 0.1;
        }
      }
    }
    else if (name == "lambda") {
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          x = ip(e,k,0);
          if (spaceDim > 1)
            y = ip(e,k,1);
          if (spaceDim > 2)
            z = ip(e,k,2);
          
          if (simNum == 10) {
            if (onSide)
              vals(e,k) = wkset->local_param_side(e,0,k);
            else
              vals(e,k) = wkset->local_param(e,0,k);
          }
          else if (simNum == 25) {
            AD rval = 1.0;
            vals(e,k) = lambda1[0]+rval;
          }
          else if (simNum == 775) {
            AD rval = 1.0;
            vals(e,k) = rval;
          }
          else if (simNum == 985) {
            double xc1 = 2.0;
            double yc1 = 0.5;
            double xc2 = 4.0;
            double yc2 = 0.5;
            double erad = 0.2;
            double rad1 = sqrt(0.5*pow(x-xc1,2.0)+ pow(y-yc1,2.0));
            double rad2 = sqrt(0.5*pow(x-xc2,2.0)+ pow(y-yc2,2.0));
            AD E0 = youngs_mod_out[0];
            AD E1 = youngs_mod_in[0];
            AD E2 = youngs_mod_in[1];
            
            AD nu0 = poisson_ratio_out[0];
            AD nu1 = poisson_ratio_in[0];
            AD nu2 = poisson_ratio_in[1];
            
            if (rad1 <= erad) {
              vals(e,k) = (E1*nu1)/((1.0+nu1)*(1.0-2.0*nu1));
            }
            else if (rad2 <= erad) {
              vals(e,k) = (E2*nu2)/((1.0+nu2)*(1.0-2.0*nu2));
            }
            else {
              vals(e,k) = (E0*nu0)/((1.0+nu0)*(1.0-2.0*nu0));
            }
            
          }
          
          else if (simNum == 26 || simNum == 28 || simNum == 29) {
            AD E = 0.0;
            if (onSide)
              E = wkset->local_param_side(e,0,k);
            else
              E = wkset->local_param(e,0,k);
            
            AD nu = poisson_ratio[0];
            vals(e,k) = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
          }
          else if ((simNum == 36) || (simNum == 200) || (simNum == 44) || (simNum == 45) || (simNum == 48) || (simNum == 781) || simNum == 1077) {
            AD E = youngs_mod[0];
            AD nu = poisson_ratio[0];
            vals(e,k) = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
          }
          else if (simNum == 993) {
            AD E = E_ref*youngs_mod[0];
            AD nu = poisson_ratio[0];
            AD E_tmp = (E*(1.0+2.0*nu))/((1.0 + nu)*(1.0+nu));
            AD nu_tmp = nu/(1.0 + nu);
            vals(e,k) = (E_tmp*nu_tmp)/((1.0+nu_tmp)*(1.0-2.0*nu_tmp));
          }
          else if (simNum == 994) {
            AD E;
            if (onSide)
              E = E_ref*wkset->local_param_side(e,0,k);
            else
              E = E_ref*wkset->local_param(e,0,k);
            AD nu = poisson_ratio[0];
            AD E_tmp = (E*(1.0+2.0*nu))/((1.0 + nu)*(1.0+nu));
            AD nu_tmp = nu/(1.0 + nu);
            vals(e,k) = (E_tmp*nu_tmp)/((1.0+nu_tmp)*(1.0-2.0*nu_tmp));
          }
          else if (simNum == 11) {
            if (spaceDim ==2) {
              vals(e,k) = 2.0*x + 3.0*y + 1.0;
            }
            else if (spaceDim ==3) {
              vals(e,k) = 1.0;
            }
          }
          else if ((simNum == 366) || (simNum == 367)) { // xom stress matching inclusion
            double xc = 7.5e3;
            double yc = 7.5e3;
            double zc = 7.5e3;
            double a = 2e3;
            double b = 1e3;
            double c = 1e3;
            double erad = 6;
            double rad = pow(x-xc,2.0)/pow(a,2.0) + pow(y-yc,2.0)/pow(b,2.0) + pow(z-zc,2.0)/pow(c,2.0);
            AD E1 = E_ref*youngs_mod_in[0];
            AD E2 = E_ref*youngs_mod_out[0];
            AD nu1 = poisson_ratio[0];
            AD nu2 = poisson_ratio[1];
            if (rad <= erad)
              vals(e,k) = (E1*nu1)/((1.0+nu1)*(1.0-2.0*nu1));
            else
              vals(e,k) = (E2*nu2)/((1.0+nu2)*(1.0-2.0*nu2));
          }
          else if (simNum == 202) {
            double xc = 0.5;
            double yc = 0.5;
            double erad = 0.2;
            double rad = sqrt(pow(x-xc,2.0)+ pow(y-yc,2.0));
            AD E1, E2;
            if (use_log_E) {
              E1 = exp(youngs_mod_in[0]);
              E2 = exp(youngs_mod_out[0]);
            }
            else {
              E1 = youngs_mod_in[0];
              E2 = youngs_mod_out[0];
            }
            //AD E1 = youngs_mod_in[0];
            //AD E2 = youngs_mod_out[0];
            AD nu = poisson_ratio[0];
            if (rad <= erad)
              vals(e,k) = (E1*nu)/((1.0+nu)*(1.0-2.0*nu));
            else
              vals(e,k) = (E2*nu)/((1.0+nu)*(1.0-2.0*nu));
          }
          else if (simNum == 22) { // xom salt body problem
            double th = -25.0*PI/180.0;
            double xc = 0.5;
            double yc = 0.5;
            double rxc = xc*cos(th) + yc*sin(th);
            double ryc = -xc*sin(th) + yc*cos(th);
            double rx = x*cos(th) + y*sin(th);
            double ry = -x*sin(th) + y*cos(th);
            double a = 2;
            double b = 1;
            double erad = 0.04;
            double rad = pow(rx-rxc,2.0)/pow(a,2.0) + pow(ry-ryc,2.0)/pow(b,2.0);
            // E salt = 30e9, E rock = 5e9, nu both = 0.3
            
            if (rad <= erad) {
              vals(e,k) = lambda1[0];
            }
            else {
              vals(e,k) = lambda1[1];
            }
            
          }
          else if (simNum == 500) { // xom stress matching inclusion
            bool inside = this->insideSalt(x,y,z);
            AD E1, E2;
            if (use_log_E) {
              E1 = E_ref*exp(youngs_mod_in[0]);
              E2 = E_ref*exp(youngs_mod_out[0]);
            }
            else {
              E1 = E_ref*youngs_mod_in[0];
              E2 = E_ref*youngs_mod_out[0];
            }
            AD nu1 = poisson_ratio[0];
            AD nu2 = poisson_ratio[1];
            if (inside)
              vals(e,k) = (E1*nu1)/((1.0+nu1)*(1.0-2.0*nu1));
            else
              vals(e,k) = (E2*nu2)/((1.0+nu2)*(1.0-2.0*nu2));
          }
          else {
            vals(e,k) = lambda1[0];
          }
        }
      }
      
    }
    else if (name == "mu") {
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          x = ip(e,k,0);
          if (spaceDim > 1)
            y = ip(e,k,1);
          if (spaceDim > 2)
            z = ip(e,k,2);
          
          if (simNum == 10 || simNum == 2) {
            vals(e,k) = mu[0];
          }
          else if (simNum == 87) {
            if (y < 0.25)
              vals(e,k) = mu[0];
            else if (y >= 0.25 && y < 0.5)
              vals(e,k) = mu[1];
            else if (y >= 0.5 && y < 0.75)
              vals(e,k) = mu[2];
            else
              vals(e,k) = mu[3];
          }
          else if (simNum == 88) {
            if (y < 0.25)
              vals(e,k) = exp(mu[0]);
            else if (y >= 0.25 && y < 0.5)
              vals(e,k) = exp(mu[1]);
            else if (y >= 0.5 && y < 0.75)
              vals(e,k) = exp(mu[2]);
            else
              vals(e,k) = exp(mu[3]);
          }
          else if (simNum == 89) {
            if (y < 0.25)
              vals(e,k) = exp(mu_opt[0]);
            else if (y >= 0.25 && y < 0.5)
              vals(e,k) = exp(mu[0]);
            else if (y >= 0.5 && y < 0.75)
              vals(e,k) = exp(mu[1]);
            else
              vals(e,k) = exp(mu[2]);
          }
          else if (simNum == 25) {
            AD rval = 1.0;
            vals(e,k) = mu[0]+rval;
          }
          else if (simNum == 775) {
            AD rval = 0.5;
            vals(e,k) = rval;
          }
          else if (simNum == 985) {
            double xc1 = 2.0;
            double yc1 = 0.5;
            double xc2 = 4.0;
            double yc2 = 0.5;
            double erad = 0.2;
            double rad1 = sqrt(0.5*pow(x-xc1,2.0)+ pow(y-yc1,2.0));
            double rad2 = sqrt(0.5*pow(x-xc2,2.0)+ pow(y-yc2,2.0));
            AD E0 = youngs_mod_out[0];
            AD E1 = youngs_mod_in[0];
            AD E2 = youngs_mod_in[1];
            
            AD nu0 = poisson_ratio_out[0];
            AD nu1 = poisson_ratio_in[0];
            AD nu2 = poisson_ratio_in[1];
            
            if (rad1 <= erad) {
              vals(e,k) = E1/(2.0*(1.0+nu1));
            }
            else if (rad2 <= erad) {
              vals(e,k) = E2/(2.0*(1.0+nu2));
            }
            else {
              vals(e,k) = E0/(2.0*(1.0+nu0));
            }
            
          }
          
          else if (simNum == 26 || simNum == 28 || simNum == 29) {
            AD E = 0.0;
            if (onSide)
              E = wkset->local_param_side(e,0,k);
            else
              E = wkset->local_param(e,0,k);
            
            AD nu = poisson_ratio[0];
            vals(e,k) = E/(2.0*(1.0+nu));
            
          }
          else if ((simNum == 366) || (simNum == 367)) { // xom stress matching inclusion
            double xc = 7.5e3;
            double yc = 7.5e3;
            double zc = 7.5e3;
            double a = 2e3;
            double b = 1e3;
            double c = 1e3;
            double erad = 6;
            double rad = pow(x-xc,2.0)/pow(a,2.0) + pow(y-yc,2.0)/pow(b,2.0) + pow(z-zc,2.0)/pow(c,2.0);
            AD E1 = E_ref*youngs_mod_in[0];
            AD E2 = E_ref*youngs_mod_out[0];
            AD nu1 = poisson_ratio[0];
            AD nu2 = poisson_ratio[1];
            if (rad <= erad)
              vals(e,k) = E1/(2.0*(1.0+nu1));
            else
              vals(e,k) = E2/(2.0*(1.0+nu2));
          }
          else if (simNum == 202) {
            double xc = 0.5;
            double yc = 0.5;
            double erad = 0.2;
            double rad = sqrt(pow(x-xc,2.0)+ pow(y-yc,2.0));
            AD E1, E2;
            if (use_log_E) {
              E1 = exp(youngs_mod_in[0]);
              E2 = exp(youngs_mod_out[0]);
            }
            else {
              E1 = youngs_mod_in[0];
              E2 = youngs_mod_out[0];
            }
            //AD E1 = youngs_mod_in[0];
            //AD E2 = youngs_mod_out[0];
            AD nu = poisson_ratio[0];
            if (rad <= erad)
              vals(e,k) = E1/(2.0*(1.0+nu));
            else
              vals(e,k) = E2/(2.0*(1.0+nu));
          }
          
          else if (simNum == 7) {
            AD mu = 0.0;
            if (onSide)
              mu = wkset->local_param_side(e,0,k);
            else
              mu = wkset->local_param(e,0,k);
            
            vals(e,k) = mu;
            
          }
          else if (simNum == 6) {
            AD mu = 0.0;
            if (onSide)
              mu = wkset->local_param_side(e,0,k);
            else
              mu = wkset->local_param(e,0,k);
            
            vals(e,k) = mu;
            
          }
          else if ((simNum == 36) || (simNum == 200) || (simNum == 44) || (simNum == 45) || (simNum == 48) || (simNum == 781) || (simNum == 1077)) {
            AD E = youngs_mod[0];
            AD nu = poisson_ratio[0];
            vals(e,k) = E/(2.0*(1.0+nu));
          }
          else if (simNum == 993) {
            AD E = E_ref*youngs_mod[0];
            AD nu = poisson_ratio[0];
            AD nu_tmp = nu/(1.0 + nu);
            AD E_tmp = (E*(1.0+2.0*nu))/((1.0 + nu)*(1.0+nu));
            vals(e,k) = E_tmp/(2.0*(1.0+nu_tmp));
          }
          else if (simNum == 994) {
            AD E;
            if (onSide)
              E = E_ref*wkset->local_param_side(0,0,k);
            else
              E = E_ref*wkset->local_param(0,0,k);
            AD nu = poisson_ratio[0];
            AD nu_tmp = nu/(1.0 + nu);
            AD E_tmp = (E*(1.0+2.0*nu))/((1.0 + nu)*(1.0+nu));
            vals(e,k) = E_tmp/(2.0*(1.0+nu_tmp));
          }
          else if (simNum == 11) {
            if (spaceDim == 2){
              vals(e,k) = 4.0*x + y + 2.0;
            }
            else if (spaceDim ==3 ){
              vals(e,k) = 0.5;
            }
          }
          else if (simNum == 22) { // xom salt body problem
            double th = -25.0*PI/180.0;
            double xc = 0.5;
            double yc = 0.5;
            double rxc = xc*cos(th) + yc*sin(th);
            double ryc = -xc*sin(th) + yc*cos(th);
            double rx = x*cos(th) + y*sin(th);
            double ry = -x*sin(th) + y*cos(th);
            double a = 2;
            double b = 1;
            double erad = 0.04;
            double rad = pow(rx-rxc,2.0)/pow(a,2.0) + pow(ry-ryc,2.0)/pow(b,2.0);
            // E salt = 30e9, E rock = 5e9, nu both = 0.3
            
            if (rad <= erad) {
              vals(e,k) = mu[0];
            }
            else {
              vals(e,k) = mu[1];
            }
            
            
          }
          else if (simNum == 500) { // xom stress matching inclusion
            bool inside = this->insideSalt(x,y,z);
            AD E1, E2;
            if (use_log_E) {
              E1 = E_ref*exp(youngs_mod_in[0]);
              E2 = E_ref*exp(youngs_mod_out[0]);
            }
            else {
              E1 = E_ref*youngs_mod_in[0];
              E2 = E_ref*youngs_mod_out[0];
            }
            AD nu1 = poisson_ratio[0];
            AD nu2 = poisson_ratio[1];
            if (inside)
              vals(e,k) = E1/(2.0*(1.0+nu1));
            else
              vals(e,k) = E2/(2.0*(1.0+nu2));
          }
          else if (simNum == 93) {
            if (use_log_E) {
              if (onSide)
                vals(e,k) = exp(wkset->local_param_side(e,0,k));
              else
                vals(e,k) = exp(wkset->local_param(e,0,k));
            }
            else {
              if (onSide)
                vals(e,k) = wkset->local_param_side(e,0,k);
              else
                vals(e,k) = wkset->local_param(e,0,k);
            }
          }
          else
            vals(e,k) = mu[0];
        }
        
      }
    }
    else if(name == "helmholtz_square_speed_real_x"){
      if(test == 4){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            vals(e,k) = x*x-1.0;
          }
        }
      }else if (test == 2){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 1.0;
          }
        }
      }else{
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            y = ip(e,k,1);
            if(!usePML || (usePML && abs(x) <= coreSize & abs(y) <= coreSize)){
              vals(e,k) = 1.0;
            }else{
              double omega = freq_params[0].val();
              double deltax = xSize - coreSize; //thickness in x
              double deltay = ySize - coreSize; //thickness in y
              double sigma1 = std::max(0.,std::max(x-xSize+deltax,-xSize+deltax-x));
              double sigma2 = std::max(0.,std::max(y-ySize+deltay,-ySize+deltay-y));
              double gamma1r = 1.;// - sigma0*sigma1/omega;
              double gamma1i = sigma0*sigma1/omega;
              double gamma2r = 1.;// - sigma0*sigma2/omega;
              double gamma2i = sigma0*sigma2/omega;
              
              AD D11 = (gamma1r*gamma2r + gamma1i*gamma2i)/(gamma1r*gamma1r + gamma1i*gamma1i);
              vals(e,k) = D11;
            }
          }
        }
      }
    }
    else if(name == "helmholtz_square_speed_imag_x"){
      if(test == 4){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            vals(e,k) = 2.0*x;
          }
        }
      }else if (test == 2){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 0.0;
          }
        }
      }else{
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            y = ip(e,k,1);
            if(!usePML || (usePML && abs(x) <= coreSize & abs(y) <= coreSize)){
              vals(e,k) = 0.0;
            }else{
              double omega = freq_params[0].val();
              double deltax = xSize - coreSize; //thickness in x
              double deltay = ySize - coreSize; //thickness in y
              double sigma1 = std::max(0.,std::max(x-xSize+deltax,-xSize+deltax-x));
              double sigma2 = std::max(0.,std::max(y-ySize+deltay,-ySize+deltay-y));
              double gamma1r = 1.;// - sigma0*sigma1/omega;
              double gamma1i = sigma0*sigma1/omega;
              double gamma2r = 1.;// - sigma0*sigma2/omega;
              double gamma2i = sigma0*sigma2/omega;
              
              AD D11 = (gamma1r*gamma2i - gamma1i*gamma2r)/(gamma1r*gamma1r + gamma1i*gamma1i);
              vals(e,k) = D11;
            }
          }
        }
      }
    }
    else if(name == "helmholtz_square_speed_real_y"){
      if(test == 4){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            vals(e,k) = x*x-1.0;
          }
        }
      }else if (test == 2){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 1.0;
          }
        }
      }else{
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            y = ip(e,k,1);
            if(!usePML || (usePML && abs(x) <= coreSize & abs(y) <= coreSize)){
              vals(e,k) = 1.0;
            }else{
              double omega = freq_params[0].val();
              double deltax = xSize - coreSize; //thickness in x
              double deltay = ySize - coreSize; //thickness in y
              double sigma1 = std::max(0.,std::max(x-xSize+deltax,-xSize+deltax-x));
              double sigma2 = std::max(0.,std::max(y-ySize+deltay,-ySize+deltay-y));
              double gamma1r = 1.;// - sigma0*sigma1/omega;
              double gamma1i = sigma0*sigma1/omega;
              double gamma2r = 1.;// - sigma0*sigma2/omega;
              double gamma2i = sigma0*sigma2/omega;
              
              AD D22 = (gamma1r*gamma2r + gamma1i*gamma2i)/(gamma2r*gamma2r + gamma2i*gamma2i);
              vals(e,k) = D22;
            }
          }
        }
      }
    }
    else if(name == "helmholtz_square_speed_imag_y"){
      if(test == 4){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            vals(e,k) = 2.0*x;
          }
        }
      }else if (test == 2){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 0.0;
          }
        }
      }else{
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            y = ip(e,k,1);
            if(!usePML || (usePML && abs(x) <= coreSize & abs(y) <= coreSize)){
              vals(e,k) = 0.0;
            }else{
              double omega = freq_params[0].val();
              double deltax = xSize - coreSize; //thickness in x
              double deltay = ySize - coreSize; //thickness in y
              double sigma1 = std::max(0.,std::max(x-xSize+deltax,-xSize+deltax-x));
              double sigma2 = std::max(0.,std::max(y-ySize+deltay,-ySize+deltay-y));
              double gamma1r = 1.;// - sigma0*sigma1/omega;
              double gamma1i = sigma0*sigma1/omega;
              double gamma2r = 1.;// - sigma0*sigma2/omega;
              double gamma2i = sigma0*sigma2/omega;
              
              AD D22 = (gamma1i*gamma2r - gamma1r*gamma2i)/(gamma2r*gamma2r + gamma2i*gamma2i);
              vals(e,k) = D22;
            }
          }
        }
      }
    }
    else if(name == "helmholtz_square_speed_real_z"){
      if(test == 4){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            vals(e,k) = x*x-1.0;
          }
        }
      }else if (test == 2){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 1.0;
          }
        }
      }else{
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            if(usePML){
              std::cout << "PML NOT READY IN 3D" << std::endl;
            }
            vals(e,k) = 1.0;
          }
        }
      }
    }
    else if(name == "helmholtz_square_speed_imag_z"){
      if(test == 4){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            x = ip(e,k,0);
            vals(e,k) = 2.0*x;
          }
        }
      }else if (test == 2){
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            vals(e,k) = 0.0;
          }
        }
      }else{
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            if(usePML){
              std::cout << "PML NOT READY IN 3D" << std::endl;
            }
            vals(e,k) = 0.0;
          }
        }
      }
    }
    else if(name == "helmholtz_square_freq_real"){
      double gammar = 1.0;
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          x = ip(e,k,0);
          y = ip(e,k,1);
          AD omega2 = freq_params[0]*freq_params[0];
          if(usePML){
            if(spaceDim > 2){
              double deltax = xSize - coreSize; //thickness in x
              double deltay = ySize - coreSize; //thickness in y
              double sigma1 = std::max(0.,std::max(x-xSize+deltax,-xSize+deltax-x));
              double sigma2 = std::max(0.,std::max(y-ySize+deltay,-ySize+deltay-y));
              double omega = sqrt(omega2.val());
              double gamma1r = 1.;// - sigma0*sigma1/omega;
              double gamma1i = sigma0*sigma1/omega;
              double gamma2r = 1.;// - sigma0*sigma2/omega;
              double gamma2i = sigma0*sigma2/omega;
              gammar = gamma1r*gamma2r - gamma1i*gamma2i;
            }
          }
          omega2 *= gammar;
          vals(e,k) = omega2;
        }
      }
    }
    else if(name == "helmholtz_square_freq_imag"){
      double gammai = 0.0;
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          x = ip(e,k,0);
          y = ip(e,k,1);
          AD omega2 = freq_params[0]*freq_params[0];
          if(usePML){
            if(spaceDim > 2){
              double deltax = xSize - coreSize; //thickness in x
              double deltay = ySize - coreSize; //thickness in y
              double sigma1 = std::max(0.,std::max(x-xSize+deltax,-xSize+deltax-x));
              double sigma2 = std::max(0.,std::max(y-ySize+deltay,-ySize+deltay-y));
              double omega = sqrt(omega2.val());
              double gamma1r = 1.;// - sigma0*sigma1/omega;
              double gamma1i = sigma0*sigma1/omega;
              double gamma2r = 1.;// - sigma0*sigma2/omega;
              double gamma2i = sigma0*sigma2/omega;
              gammai = gamma1r*gamma2i + gamma1i*gamma2r;
            }
          }
          omega2 *= gammai;
          vals(e,k) = omega2;
        }
      }
    }
    // else if(name == "helmholtz_fractional_real"){
    //   for (int k=0; k<numip; k++) {
    // 	vals(k) = ud_alphar[0];
    //   }
    // }
    // else if(name == "helmholtz_fractional_imag"){
    //   for (int k=0; k<numip; k++) {
    // 	vals(k) = ud_alphai[0];
    //   }
    // }
    else if(name == "helmholtz_fractional_alphaHr"){
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          vals(e,k) = ud_alphaHr[0];
        }
      }
    }
    else if(name == "helmholtz_fractional_alphaHi"){
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          vals(e,k) = ud_alphaHi[0];
        }
      }
    }
    else if(name == "helmholtz_fractional_alphaTr"){
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          vals(e,k) = ud_alphaTr[0];
        }
      }
    }
    else if(name == "helmholtz_fractional_alphaTi"){
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          vals(e,k) = ud_alphaTi[0];
        }
      }
    }
    else if(name == "helmholtz_fractional_freqExp"){
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          vals(e,k) = ud_freqExp[0];
        }
      }
    }
    //else if(name == "helmholtz_robin_alpha_real"){
    //  for (int k=0; k<numip; k++) {
    //    vals(k) = 0.0;
    //  }
    //}
    //else if(name == "helmholtz_robin_alpha_imag"){
    //  for (int k=0; k<numip; k++) {
    //    vals(k) = 0.0;
    //  }
    //}
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the extra field names for plotting at the nodes
  //////////////////////////////////////////////////////////////////////////////////////
  
  vector<string> extraFieldNames(const string & physics) const {
    std::vector<string> ef;
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      if (simNum == 145) {
        //ef.push_back("constant_target");
        ef.push_back("source_term");
      }
      else
        ef.push_back("target_therm");
    }
    else if (physics == "linearelasticity") {
      if (test == 303) {
        ef.push_back("body_func");
      }
      if ((simNum == 366) || (simNum == 367) || (simNum == 499) || (simNum == 500)) {
        ef.push_back("Poisson's ratio");
        ef.push_back("Young's modulus");
        if ((simNum == 366) || (simNum == 367)) {
          ef.push_back("xtrac");
          ef.push_back("ztrac");
        }
        else if (simNum == 500) {
          ef.push_back("xtrac");
          ef.push_back("ytrac");
        }
      }
      else if (simNum == 202) {
        ef.push_back("Poisson's ratio");
        ef.push_back("Young's modulus");
      }
      else if (simNum == 112) {
        ef.push_back("target_le");
      }
      else {
        ef.push_back("target_le");
        ef.push_back("target_le2");
      }
    }
    else if (physics == "navierstokes") {
      if (test == 222) {
        ef.push_back("vorticity");
      }
    }
    else if (physics == "shallowwater") {
      ef.push_back("bathymetry");
    }
    return ef;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the extra fields for plotting at the nodes
  //////////////////////////////////////////////////////////////////////////////////////
  
  vector<Kokkos::View<double***,AssemblyDevice> > extraFields(const string & physics, const DRV & ip, const double & time,
                         const Teuchos::RCP<workset> & wkset) {
    
    vector<Kokkos::View<double***,AssemblyDevice> > ef;
    //Teuchos::RCP<workset> wkset;
    wkset->ip = ip;
    wkset->time = time;
    int numElem = ip.dimension(0);
    if (physics == "thermal") {
      
      Kokkos::View<AD***,AssemblyDevice> targ_AD = this->target("thermal",ip, time);
      Kokkos::View<double***,AssemblyDevice> targ("target",numElem,targ_AD.dimension(1), targ_AD.dimension(2));
      for (int e=0; e<numElem; e++) {
        for (size_t i=0; i<targ_AD.dimension(1); i++) {
          for (size_t j=0; j<targ_AD.dimension(2); j++) {
            targ(e,i,j) = targ_AD(e,i,j).val();
          }
        }
      }
      ef.push_back(targ);
    }
    if (physics == "thermal_fr" || physics == "thermal_enthalpy") {
      
      Kokkos::View<AD***,AssemblyDevice> targ_AD = this->target("thermal_fr",ip, time);
      Kokkos::View<double***,AssemblyDevice> targ("target",numElem,targ_AD.dimension(0), targ_AD.dimension(1));
      for (int e=0; e<numElem; e++) {
        for (size_t i=0; i<targ_AD.dimension(1); i++) {
          for (size_t j=0; j<targ_AD.dimension(2); j++) {
            targ(e,i,j) = targ_AD(e,i,j).val();
          }
        }
      }
      ef.push_back(targ);
    }
    else if (physics == "navierstokes") {
      /*Kokkos::View<double***,AssemblyDevice> targ("vorticity",numElem,targ_AD.dimension(0), targ_AD.dimension(1));
      for (int e=0; e<numElem; e++) {
        for (size_t i=0; i<targ_AD.dimension(1); i++) {
          for (size_t j=0; j<targ_AD.dimension(2); j++) {
            targ(e,i,j) = wkset->(e,i,j).val();
          }
        }
      }
      ef.push_back(targ);
       */
    }
    else if (physics == "linearelasticity") {
      if (test == 303) {
        size_t numip = ip.dimension(1);
        Kokkos::View<double***,AssemblyDevice> bodyfunc("bodyfunc",numElem,1,numip);
        double z_scale = 15.0e3;
        double y_scale = 18.75e3/2.0;
        double x_scale = 18.75e3/2.0;
        for (int e=0; e<numElem; e++) {
          for (size_t i=0; i<numip; i++){
            double x = ip(e,i,0);
            double y = ip(e,i,1);
            double z = ip(e,i,2);
            
            double xs = x/x_scale;
            double ys = y/y_scale;
            double zs = z/z_scale;
            
            //bodyfunc(0,i) = (1.0-2.0758*z + 6.7273*z*z - 5.1515*z*z*z)*(1.0-sqrt(x*x+y*y));
            bodyfunc(e,0,i) = (1.0e0-2.0758*zs + 6.7273*zs*zs - 5.1515*zs*zs*zs)*(1.0e0-sqrt(xs*xs+ys*ys));
            
            size_t numpert = salt_thetavals.size();
            double decay = 10.0;
            double currtheta = atan2(y,x);
            double pert = 0.0;
            for (size_t j=0; j<numpert; j++) {
              pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta,2));
              pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta+2.0*PI,2));
              pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta-2.0*PI,2));
            }
            
            //bodyfunc(0,i) += 1.0/3.0*pert*(1.0-pow(z,2));
            bodyfunc(e,0,i) += 1.0/3.0*pert*(1.0-pow(zs,2));
          }
        }
        ef.push_back(bodyfunc);
        if (simNum == 500) { // xom stress matching inclusion
          Kokkos::View<AD***,AssemblyDevice> targ_AD = this->target("linearelasticity",ip, time);
          Kokkos::View<double***,AssemblyDevice> targ("target",numElem,targ_AD.dimension(0), targ_AD.dimension(1));
          for (int e=0; e<numElem; e++) {
            for (size_t i=0; i<targ_AD.dimension(1); i++) {
              for (size_t j=0; j<targ_AD.dimension(2); j++) {
                targ(e,i,j) = targ_AD(e,i,j).val();
              }
            }
          }
          ef.push_back(targ);
        }
      }
      else {
        Kokkos::View<AD***,AssemblyDevice> targ_AD = this->target("linearelasticity",ip, time);
        for (int i=0; i<2; i++) {
          Kokkos::View<double***,AssemblyDevice> targ("target",numElem,1, targ_AD.dimension(2));
          for (int e=0; e<numElem; e++) {
            for (size_t i=0; i<targ_AD.dimension(0); i++) {
              for (size_t j=0; j<targ_AD.dimension(1); j++) {
                targ(e,0,j) = targ_AD(e,i,j).val();
              }
            }
          }
          ef.push_back(targ);
        }
      }
    }
    else if (physics == "shallowwater") {
      size_t numip = ip.dimension(1);
      Kokkos::View<double***,AssemblyDevice> bathymetry("bathymetry",numElem,1,numip);
      for (int e=0; e<numElem; e++) {
        for (size_t i=0; i<numip; i++){
          double x = ip(e,i,0);
          double y = ip(e,i,1);
          
          if (test == 2) {
            bathymetry(e,0,i) = 0.5*exp(-10.0*(x-0.75)*(x-0.75) - 10.0*(y-0.5)*(y-0.5));
          }
        }
      }
      ef.push_back(bathymetry);
    }
    return ef;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the extra field names for plotting as piecewise constants
  //////////////////////////////////////////////////////////////////////////////////////
  
  vector<string> extraCellFieldNames(const string & physics) const {
    std::vector<string> ef;
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      ef.push_back("grain");
    }
    else if (physics == "linearelasticity") {
      // stress calculations moved to linearelasticity.hpp since they depend on physics
      // can still add extra cell fields here
      if (simNum == 500) {
        ef.push_back("wells");
      }
    }
    return ef;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the extra fields for plotting as piecewise constants
  //////////////////////////////////////////////////////////////////////////////////////
  
  vector<Kokkos::View<double***,AssemblyDevice> > extraCellFields(const string & physics, const Teuchos::RCP<workset> & wkset) const {
    vector<Kokkos::View<double***,AssemblyDevice> > ef;
    int numElem = wkset->ip.dimension(0);
    if (physics == "thermal" || physics == "thermal_fr" || physics == "thermal_enthalpy") {
      Kokkos::View<double***,AssemblyDevice> dmod("dmod",numElem,1,1);
      ef.push_back(dmod);
    }
    else if (physics == "linearelasticity") {
      // stress calculations moved to linearelasticity.hpp since they depend on physics
      // can still add extra cell fields here
      if (simNum == 500) {
        int numip = wkset->ip.dimension(1);
        DRV ip = wkset->ip;
        double x_av = 0.0;
        double y_av = 0.0;
        //double c1 = 4.6875e3;
        //double c2 = 1.875e2;
        double c0 = -6937.5;
        double c1 = 7312.5;
        double c2 = 187.5;
        double dist1, dist2, dist3, dist4;
        double tol = 100.0;
        Kokkos::View<double***,AssemblyDevice> well("well",numElem,1,1);
        for (int e=0; e<numElem; e++) {
          for (int i=0; i<numip; i++) {
            x_av += ip(e,i,0);
            y_av += ip(e,i,1);
          }
          x_av = x_av/numip;
          y_av = y_av/numip;
          
          /*
           dist1 = sqrt(pow(x_av + c1,2.0) + pow(y_av - c2,2.0));
           dist2 = sqrt(pow(x_av - c1,2.0) + pow(y_av - c2,2.0));
           dist3 = sqrt(pow(x_av - c2,2.0) + pow(y_av + c1,2.0));
           dist4 = sqrt(pow(x_av - c2,2.0) + pow(y_av - c1,2.0));
           */
          if (one_well_random) {
            double w1x, w1y;
            w1x = -5.81250000e+03;
            w1y = -3.56250000e+03;
            
            dist1 = sqrt(pow(x_av - w1x,2.0) + pow(y_av - w1y,2.0));
            
            if (dist1 < tol)
              well(e,0,0) = 1.0;
          }
          else if (two_well_random) {
            double w1x, w1y, w2x, w2y;
            w1x = -5.81250000e+03;
            w1y = -3.56250000e+03;
            w2x = 2.43750000e+03;
            w2y = 6.18750000e+03;
            
            dist1 = sqrt(pow(x_av - w1x,2.0) + pow(y_av - w1y,2.0));
            dist2 = sqrt(pow(x_av - w2x,2.0) + pow(y_av - w2y,2.0));
            
            if (dist1 < tol)
              well(e,0,0) = 1.0;
            else if (dist2 < tol)
              well(e,0,0) = 1.0;
          }
          else {
            
            dist1 = sqrt(pow(x_av - c0,2.0) + pow(y_av - c2,2.0));
            dist2 = sqrt(pow(x_av - c1,2.0) + pow(y_av - c2,2.0));
            dist3 = sqrt(pow(x_av - c2,2.0) + pow(y_av - c0,2.0));
            dist4 = sqrt(pow(x_av - c2,2.0) + pow(y_av - c1,2.0));
            
            
            if (dist1 < tol)
              well(e,0,0) = 1.0;
            else if (dist2 < tol)
              well(e,0,0) = 1.0;
            else if (dist3 < tol)
              well(e,0,0) = 1.0;
            else if (dist4 < tol)
              well(e,0,0) = 1.0;
            
          }
        }
        
        ef.push_back(well);
      }
    }
    
    return ef;
  }
  
  // ========================================================================================
  /* return the material property value at an integration point */
  // ========================================================================================
  
  AD MaterialProperty(const string & var, const double & x, const double & y, const double & z,
                      const double & t) const {
    
    AD val = 0.0;
    
    if ((simNum == 366) || (simNum == 367)) { // xom stress matching inclusion
      double xc = 7.5e3;
      double yc = 7.5e3;
      double zc = 7.5e3;
      double a = 2e3;
      double b = 1e3;
      double c = 1e3;
      double erad = 6;
      
      double rad = pow(x-xc,2.0)/pow(a,2.0) + pow(y-yc,2.0)/pow(b,2.0) + pow(z-zc,2.0)/pow(c,2.0);
      AD E1 = E_ref*youngs_mod_in[0];
      AD E2 = E_ref*youngs_mod_out[0];
      AD nu1 = poisson_ratio[0];
      AD nu2 = poisson_ratio[1];
      if (rad <= erad) {
        if (var == "mu") {
          val = E1/(2.0*(1.0+nu1));
        }
        else if (var == "lambda") {
          val = (E1*nu1)/((1.0+nu1)*(1.0-2.0*nu1));
        }
      }
      else {
        if (var == "mu") {
          val = E2/(2.0*(1.0+nu2));
        }
        else if (var == "lambda") {
          val = (E2*nu2)/((1.0+nu2)*(1.0-2.0*nu2));
        }
      }
    }
    else if (simNum == 202) {
      double xc = 0.5;
      double yc = 0.5;
      double erad = 0.2;
      double rad = sqrt(pow(x-xc,2.0)+ pow(y-yc,2.0));
      AD E1 = youngs_mod_in[0];
      AD E2 = youngs_mod_out[0];
      AD nu = poisson_ratio[0];
      if (var == "lambda") {
        if (rad <= erad)
          val = (E1*nu)/((1.0+nu)*(1.0-2.0*nu));
        else
          val = (E2*nu)/((1.0+nu)*(1.0-2.0*nu));
      }
      else if (var == "mu") {
        if (rad <= erad)
          val = E1/(2.0*(1.0+nu));
        else
          val = E2/(2.0*(1.0+nu));
      }
    }
    else if (simNum == 985) {
      double xc1 = 2.0;
      double yc1 = 0.5;
      double xc2 = 4.0;
      double yc2 = 0.5;
      double erad = 0.2;
      double rad1 = sqrt(0.5*pow(x-xc1,2.0)+ pow(y-yc1,2.0));
      double rad2 = sqrt(0.5*pow(x-xc2,2.0)+ pow(y-yc2,2.0));
      AD E0 = youngs_mod_out[0];
      AD E1 = youngs_mod_in[0];
      AD E2 = youngs_mod_in[1];
      
      AD nu0 = poisson_ratio_out[0];
      AD nu1 = poisson_ratio_in[0];
      AD nu2 = poisson_ratio_in[1];
      
      if (var == "lambda") {
        if (rad1 <= erad) {
          val = (E1*nu1)/((1.0+nu1)*(1.0-2.0*nu1));
        }
        else if (rad2 <= erad) {
          val = (E2*nu2)/((1.0+nu2)*(1.0-2.0*nu2));
        }
        else {
          val = (E0*nu0)/((1.0+nu0)*(1.0-2.0*nu0));
        }
      }
      else if (var == "mu") {
        if (rad1 <= erad) {
          val = E1/(2.0*(1.0+nu1));
        }
        else if (rad2 <= erad) {
          val = E2/(2.0*(1.0+nu2));
        }
        else {
          val = E0/(2.0*(1.0+nu0));
        }
      }
    }
    else if ((simNum == 320) || (simNum == 322)) { // xom stress matching inclusion
      double th = 0;
      double xc = 7.5e3;
      double yc = 7.5e3;
      double rxc = xc*cos(th) + yc*sin(th);
      double ryc = -xc*sin(th) + yc*cos(th);
      double rx = x*cos(th) + y*sin(th);
      double ry = -x*sin(th) + y*cos(th);
      double a = 2e3;
      double b = 1e3;
      double erad = 6;
      double rad = pow(rx-rxc,2.0)/pow(a,2.0) + pow(ry-ryc,2.0)/pow(b,2.0);
      AD E1 = E_ref*youngs_mod_in[0];
      AD E2 = E_ref*youngs_mod_out[0];
      AD nu1 = poisson_ratio[0];
      AD nu2 = poisson_ratio[1];
      
      if (rad <= erad) {
        if (var == "mu") {
          val = E1/(2.0*(1.0+nu1));
        }
        else if (var == "lambda") {
          val = (E1*nu1)/((1.0+nu1)*(1.0-2.0*nu1));
        }
      }
      else {
        if (var == "mu") {
          val = E2/(2.0*(1.0+nu2));
        }
        else if (var == "lambda") {
          val = (E2*nu2)/((1.0+nu2)*(1.0-2.0*nu2));
        }
      }
    }
    else if (simNum == 500) { // xom stress matching inclusion
      AD E1, E2;
      if (use_log_E) {
        E1 = E_ref*exp(youngs_mod_in[0]);
        E2 = E_ref*exp(youngs_mod_out[0]);
      }
      else {
        E1 = E_ref*youngs_mod_in[0];
        E2 = E_ref*youngs_mod_out[0];
      }
      AD nu1 = poisson_ratio[0];
      AD nu2 = poisson_ratio[1];
      bool inside = this->insideSalt(x,y,z);
      if (inside) {
        if (var == "mu") {
          val = E1/(2.0*(1.0+nu1));
        }
        else if (var == "lambda") {
          val = (E1*nu1)/((1.0+nu1)*(1.0-2.0*nu1));
        }
      }
      else {
        if (var == "mu") {
          val = E2/(2.0*(1.0+nu2));
        }
        else if (var == "lambda") {
          val = (E2*nu2)/((1.0+nu2)*(1.0-2.0*nu2));
        }
      }
    }
    else if (simNum == 45) {
      AD E = youngs_mod[0];
      AD nu = poisson_ratio[0];
      if (var == "mu") {
        val = E/(2.0*(1.0+nu));
      }
      else if (var == "lambda") {
        val = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
      }
    }
    else {
      if (var == "mu")
        val = mu[0];
      else if (var == "lambda")
        val = lambda1[0];
    }
    return val;
  }
  
  bool insideSalt(const double & x, const double & y, const double & z) const {
    
    double lvlset;
    double z_scale = 15.0e3;
    double y_scale = 18.75e3/2.0;
    double x_scale = 18.75e3/2.0;
    
    if (z > 15.0e3)
      return false;
    else {
      
      double xs = x/x_scale;
      double ys = y/y_scale;
      double zs = z/z_scale;
      
      lvlset = (1.0e0-2.0758*zs + 6.7273*zs*zs - 5.1515*zs*zs*zs)*(1.0e0-sqrt(xs*xs+ys*ys));
      
      size_t numpert = salt_thetavals.size();
      double decay = 10.0;
      double currtheta = atan2(y,x);
      double pert = 0.0;
      for (size_t j=0; j<numpert; j++) {
        pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta,2));
        pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta+2.0*PI,2));
        pert += salt_thetamags[j].val()*exp(-decay*pow(salt_thetavals[j].val()-currtheta-2.0*PI,2));
      }
      
      lvlset += 1.0/3.0*pert*(1.0-pow(zs,2));
      if (lvlset > 0.70)
        return true;
      else
        return false;
    }
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                        const vector<string> & paramnames) {
    
    for (size_t p=0; p<paramnames.size(); p++) {
      
      if (paramnames[p] == "thermal_diff")
        diff_params = *(params[p]);
      else if (paramnames[p] == "thermal_diff_stoch")
        diff_stoch_params = *(params[p]);
      else if (paramnames[p] == "thermal_source")
        tsource = *(params[p]);
      else if (paramnames[p] == "thermal_source_stoch") //for debugging SOL interface
        tsource_stoch = *(params[p]);
      else if (paramnames[p] == "thermal_boundary")
        boundary_params = *(params[p]);
      else if (paramnames[p] == "thermal_init")
        init_params = *(params[p]);
      else if (paramnames[p] == "thermal_topoopt_maxdiff")
        maxdiff = *(params[p]);
      else if (paramnames[p] == "laser_width")
        laser_width = *(params[p]);
      else if (paramnames[p] == "laser_speed")
        laser_speed = *(params[p]);
      else if (paramnames[p] == "laser_xlocs")
        laser_xlocs= *(params[p]);
      else if (paramnames[p] == "laser_ylocs")
        laser_ylocs= *(params[p]);
      else if (paramnames[p] == "laser_intensity")
        laser_intensity = *(params[p]);
      else if (paramnames[p] == "laser_intensity_stoch")
        laser_intensity_stoch = *(params[p]);
      else if (paramnames[p] == "laser_beam_1")
        laser_beam_1 = *(params[p]);
      else if (paramnames[p] == "laser_beam_2")
        laser_beam_2 = *(params[p]);
      else if (paramnames[p] == "laser_beam_3")
        laser_beam_3 = *(params[p]);
      else if (paramnames[p] == "laser_beam_4")
        laser_beam_4 = *(params[p]);
      else if (paramnames[p] == "laser_beam_5")
        laser_beam_5 = *(params[p]);
      else if (paramnames[p] == "cp_params")
        cp_params = *(params[p]);
      else if (paramnames[p] == "lambda1")
        lambda1 = *(params[p]);
      else if (paramnames[p] == "lambda2")
        lambda2 = *(params[p]);
      else if (paramnames[p] == "mu")
        mu = *(params[p]);
      else if (paramnames[p] == "mu_opt")
        mu_opt = *(params[p]);
      else if (paramnames[p] == "source_x")
        source_x = *(params[p]);
      else if (paramnames[p] == "source_y")
        source_y = *(params[p]);
      else if (paramnames[p] == "source_z")
        source_z = *(params[p]);
      else if (paramnames[p] == "nbcx")
        nbcx = *(params[p]);
      else if (paramnames[p] == "nbcy")
        nbcy = *(params[p]);
      else if (paramnames[p] == "nbcz")
        nbcz = *(params[p]);
      else if (paramnames[p] == "ttop")
        ttop = *(params[p]);
      else if (paramnames[p] == "tbottom")
        tbottom = *(params[p]);
      else if (paramnames[p] == "tleft")
        tleft = *(params[p]);
      else if (paramnames[p] == "tright")
        tright = *(params[p]);
      else if (paramnames[p] == "KLrvals")
        KLrvals = *(params[p]);
      else if (paramnames[p] == "youngs_mod")
        youngs_mod = *(params[p]);
      else if (paramnames[p] == "youngs_mod_in")
        youngs_mod_in = *(params[p]);
      else if (paramnames[p] == "youngs_mod_out")
        youngs_mod_out = *(params[p]);
      else if (paramnames[p] == "poisson_ratio")
        poisson_ratio = *(params[p]);
      else if (paramnames[p] == "poisson_ratio_in")
        poisson_ratio_in = *(params[p]);
      else if (paramnames[p] == "poisson_ratio_out")
        poisson_ratio_out = *(params[p]);
      else if (paramnames[p] == "density")
        density_params = *(params[p]);
      else if (paramnames[p] == "viscosity")
        viscosity_params = *(params[p]);
      else if (paramnames[p] == "lid_parameters")
        bc_params = *(params[p]);
      else if (paramnames[p] == "ns_source")
        source_params = *(params[p]);
      else if (paramnames[p] == "ns_init")
        init_params = *(params[p]);
      else if (paramnames[p] == "helmholtz_freq")
        freq_params = *(params[p]);
      else if (paramnames[p] == "salt_thetavals")
        salt_thetavals = *(params[p]);
      else if (paramnames[p] == "salt_thetamags")
        salt_thetamags = *(params[p]);
      else if (paramnames[p] == "mag_stoch_exp")
        expMag = *(params[p]);
      else if (paramnames[p] == "sourceControl")
        sourceControl = *(params[p]);
      else if (paramnames[p] == "mean")
        mean = *(params[p]);
      else if (paramnames[p] == "smean")
        smean = *(params[p]);
      else if (paramnames[p] == "stddev1")
        stddev1 = *(params[p]);
      else if (paramnames[p] == "stddev2")
        stddev2 = *(params[p]);
      // else if (paramnames[p] == "alpha_real")
      //   ud_alphar = *(params[p]);
      // else if (paramnames[p] == "alpha_imag")
      //   ud_alphai = *(params[p]);
      else if (paramnames[p] == "helmholtz_frac_alphaHr")
        ud_alphaHr = *(params[p]);
      else if (paramnames[p] == "helmholtz_frac_alphaHi")
        ud_alphaHi = *(params[p]);
      else if (paramnames[p] == "helmholtz_frac_alphaTr")
        ud_alphaTr = *(params[p]);
      else if (paramnames[p] == "helmholtz_frac_alphaTi")
        ud_alphaTi = *(params[p]);
      else if (paramnames[p] == "helmholtz_frac_freqExp")
        ud_freqExp = *(params[p]);
      else if (paramnames[p] == "surf_xlocs")
        surf_xlocs= *(params[p]);
      else if (paramnames[p] == "surf_ylocs")
        surf_ylocs= *(params[p]);
      else if (paramnames[p] == "surf_zlocs")
        surf_zlocs= *(params[p]);
      //else
      
      //else
      //cout << "Parameter not used: " << paramnames[p] << endl;
      
    }
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Set a custom initial guess for the discretized parameters
  //////////////////////////////////////////////////////////////////////////////////////
  vector<vector<vector<double> > > setInitialParams(const DRV & nodes,
                                                    const vector<vector<vector<int> > > & indices) {
    
    vector<vector<vector<double > > > param_initial_vals;
    double x,y,z;
    double val;
    for (int p=0; p<indices.size(); p++) {
      
      vector<vector<double > > curr_initial_vals;
      
      for (int n = 0; n < indices[p].size(); n++) {
        curr_initial_vals.push_back(vector<double>(indices[p][n].size()));
        for (int i = 0; i < indices[p][n].size(); i++) {
          x = nodes(p,i,0);
          if (spaceDim > 1)
            y = nodes(p,i,1);
          if (spaceDim > 2)
            z = nodes(p,i,2);
          if (simNum == 202) {
            if (abs(x) < 0.02)
              val = y*(1.0 - y);
            else if (abs(x - 1.0) < 0.02)
              val = -y*(1.0 - y);
            else
              val = 0.0;
          }
          if (simNum == 500) {
            double lim = 18.75e3/2.0;
            double zlim = 18.75e3;
            if (n == 0) {
              if (abs(x - -lim) < 0.05) {
                val = tbase*(1.0 - z/zlim)/t_ref;
              }
              else if (abs(x - lim) < 0.05) {
                val = -tbase*(1.0 - z/zlim)/t_ref;
              }
              else
                val = 0.0;
            }
            else if (n == 1) {
              if (abs(y - -lim) < 0.05) {
                val = tbase*(1.0 - z/zlim)/t_ref;
              }
              else if (abs(y - lim) < 0.05) {
                val = -tbase*(1.0 - z/zlim)/t_ref;
              }
              else
                val = 0.0;
            }
          }
          curr_initial_vals[n][i] = val;
        }
      }
      param_initial_vals.push_back(curr_initial_vals);
    }
    return param_initial_vals;
  }
  
  
protected:
  
  bool isTD, multiscale;
  int test, simNum;
  int spaceDim;
  string simName;
  vector<vector<string> > varlist;
  
  //double PI = 3.141592653589793238463;
  
  // Scalar parameters- derivatives not available
  // Thermal:
  double Lx, Ly, Px, Py, cMY, sinkwidth, formparam, T_ambient, T_rad, emiss, rho;
  double eta, hconv, e_a, e_b, e_c, sb_constant;
  double laser_xpos, laser_ypos, laser_zpos, rthres1, rthres2, rthres3, toff, cparamx, cparamz;
  double sensor_refine;
  double weld_length;
  double laser_off, laser_on;
  AD melt_pool_temp;
  
  // Linear elasticity:
  bool incplanestress, ledisp_response_type, use_log_E;
  double E_ref, t_ref, alpha_T;
  double tbase, tscale;
  bool two_well_random, forward_data_gen, one_well_random;
  
  // Helmholtz
  bool usePML; //whether to use PML
  double coreSize; //region included in PML is origin-centered square of edge-length 2*coreSize
  double sigma0; //decay parameter for PML
  double xSize, ySize; //for PML; assume entire domain is [-xSize,xSize] x [-ySize,ySize]
  
  // Optimization parameters - stored as vectors of AD types
  //Thermal:
  vector<AD> diff_params, diff_stoch_params, tsource, boundary_params, init_params, cp_params;
  vector<AD> tsource_stoch; //for SOL debugging
  vector<AD> maxdiff, laser_width, laser_speed, laser_intensity, laser_intensity_stoch;
  vector<AD> laser_beam_1, laser_beam_2, laser_beam_3, laser_beam_4, laser_beam_5;
  vector<AD> laser_xlocs, laser_ylocs;
  
  //thermal_fr
  double     s_param;
  
  // Linear elasticity:
  vector<AD> lambda1, lambda2, mu, source_x, source_y, source_z, nbcx, nbcy, nbcz, ttop, tbottom, tleft, tright;
  vector<AD> KLrvals, youngs_mod, poisson_ratio, poisson_ratio_in, poisson_ratio_out, youngs_mod_in, youngs_mod_out, mu_opt;
  vector<AD> salt_thetavals, salt_thetamags;
  
  // Navier Stokes
  vector<AD> density_params, viscosity_params, bc_params, source_params;
  
  // Helmholtz
  vector<AD> freq_params;
  
  // Post processing
  double responseTarget;
  double regCoeff;
  double finalTime, delT;
  int numSteps;
  vector<AD> stddev1;
  vector<AD> stddev2;
  vector<AD> mean, smean;
  vector<AD> sourceControl,expMag;
  vector<AD> ud_alphaHr, ud_alphaHi, ud_alphaTr, ud_alphaTi, ud_freqExp;
  
  //ceramic burnoff
  double stddevx;
  double stddevy;
  double stddevz;
  double xloc;
  double yloc;
  double zloc;
  double mag;
  vector<AD> surf_xlocs, surf_ylocs, surf_zlocs;
};
#endif

