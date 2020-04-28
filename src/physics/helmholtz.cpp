/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "helmholtz.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

helmholtz::helmholtz(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
          const size_t & numip_side_, const int & numElem_,
          Teuchos::RCP<FunctionManager> & functionManager_) :
numip(numip_), numip_side(numip_side_), numElem(numElem_) {
  
  
  label = "helmholtz";
  functionManager = functionManager_;
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  verbosity = settings->sublist("Physics").get<int>("Verbosity",0);
  fractional = settings->sublist("Physics").get<bool>("fractional",false);
  
  useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (helmholtz)",false);
  
  myvars.push_back("ureal");
  myvars.push_back("uimag");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  
  // Functions
  Teuchos::ParameterList fs = settings->sublist("Functions");
  functionManager->addFunction("c2r_x",fs.get<string>("c2r_x","0.0"),numElem,numip,"ip");
  functionManager->addFunction("c2i_x",fs.get<string>("c2i_x","0.0"),numElem,numip,"ip");
  functionManager->addFunction("c2r_y",fs.get<string>("c2r_y","0.0"),numElem,numip,"ip");
  functionManager->addFunction("c2i_y",fs.get<string>("c2i_y","0.0"),numElem,numip,"ip");
  functionManager->addFunction("c2r_z",fs.get<string>("c2r_z","0.0"),numElem,numip,"ip");
  functionManager->addFunction("c2i_z",fs.get<string>("c2i_z","0.0"),numElem,numip,"ip");
  functionManager->addFunction("omega2r",fs.get<string>("omega2r","0.0"),numElem,numip,"ip");
  functionManager->addFunction("omega2i",fs.get<string>("omega2i","0.0"),numElem,numip,"ip");
  functionManager->addFunction("omega2r",fs.get<string>("omega2r","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("omega2i",fs.get<string>("omega2i","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("omegar",fs.get<string>("omegar","0.0"),numElem,numip,"ip");
  functionManager->addFunction("omegai",fs.get<string>("omegai","0.0"),numElem,numip,"ip");
  functionManager->addFunction("source_r",fs.get<string>("source_r","0.0"),numElem,numip,"ip");
  functionManager->addFunction("source_i",fs.get<string>("source_i","0.0"),numElem,numip,"ip");
  functionManager->addFunction("source_r_side",fs.get<string>("source_r_side","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("source_i_side",fs.get<string>("source_i_side","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("robin_alpha_r",fs.get<string>("robin_alpha_r","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("robin_alpha_i",fs.get<string>("robin_alpha_i","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("c2r_x",fs.get<string>("c2r_x","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("c2i_x",fs.get<string>("c2i_x","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("c2r_y",fs.get<string>("c2r_y","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("c2i_y",fs.get<string>("c2i_y","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("c2r_z",fs.get<string>("c2r_z","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("c2i_z",fs.get<string>("c2i_z","0.0"),numElem,numip_side,"side ip");
  functionManager->addFunction("alphaHr",fs.get<string>("alphaHr","0.0"),numElem,numip,"ip");
  functionManager->addFunction("alphaHi",fs.get<string>("alphaHi","0.0"),numElem,numip,"ip");
  functionManager->addFunction("alphaTr",fs.get<string>("alphaTr","0.0"),numElem,numip,"ip");
  functionManager->addFunction("alphaTi",fs.get<string>("alphaTi","0.0"),numElem,numip,"ip");
  functionManager->addFunction("freqExp",fs.get<string>("freqExp","0.0"),numElem,numip,"ip");
  functionManager->addFunction("freqExp",fs.get<string>("freqExp","0.0"),numElem,numip_side,"side ip");
}

// ========================================================================================
// ========================================================================================

void helmholtz::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int ur_basis_num = wkset->usebasis[ur_num];
  int ui_basis_num = wkset->usebasis[ui_num];
  
  c2r_x = functionManager->evaluate("c2r_x","ip");
  c2i_x = functionManager->evaluate("c2i_x","ip");
  c2r_y = functionManager->evaluate("c2r_y","ip");
  c2i_y = functionManager->evaluate("c2i_y","ip");
  c2r_z = functionManager->evaluate("c2r_z","ip");
  c2i_z = functionManager->evaluate("c2i_z","ip");
  omega2r = functionManager->evaluate("omega2r","ip");
  omega2i = functionManager->evaluate("omega2i","ip");
  
  if (fractional) {
    alphaHr = functionManager->evaluate("alphaHr","ip");
    alphaHi = functionManager->evaluate("alphaHi","ip");
    alphaTr = functionManager->evaluate("alphaTr","ip");
    alphaTi = functionManager->evaluate("alphaTi","ip");
    freqExp = functionManager->evaluate("freqExp","ip");
  }
  source_r = functionManager->evaluate("source_r","ip");
  source_i = functionManager->evaluate("source_i","ip");
  
  sol = wkset->local_soln;
  sol_dot = wkset->local_soln_dot;
  sol_grad = wkset->local_soln_grad;
  
  urbasis = wkset->basis[ur_basis_num];
  urbasis_grad = wkset->basis_grad[ur_basis_num];
  uibasis = wkset->basis[ui_basis_num];
  uibasis_grad = wkset->basis_grad[ui_basis_num];
  
  offsets = wkset->offsets;
  
  res = wkset->res;
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int k=0; k<sol.extent(2); k++ ) {
      AD ur = sol(e,ur_num,k,0);
      AD durdx = sol_grad(e,ur_num,k,0);
      AD ui = sol(e,ui_num,k,0);
      AD duidx = sol_grad(e,ui_num,k,0);
      
      AD durdy, duidy, durdz, duidz;
      if (spaceDim > 1) {
        durdy = sol_grad(e,ur_num,k,1);
        duidy = sol_grad(e,ui_num,k,1);
      }
      if (spaceDim > 2) {
        durdz = sol_grad(e,ur_num,k,2);
        duidz = sol_grad(e,ui_num,k,2);
      }
      
      
      //TMW: this residual makes no sense to me
      for (int i=0; i<urbasis.extent(1); i++ ) { // what if ui uses a different basis?
        ScalarT vr = urbasis(e,i,k);
        ScalarT vi = uibasis(e,i,k);  //bvbw check to make sure first index  = 0
        ScalarT dvrdx = urbasis_grad(e,i,k,0);
        ScalarT dvidx = uibasis_grad(e,i,k,0);
        ScalarT dvrdy = 0.0;
        ScalarT dvidy = 0.0;
        if (spaceDim > 1) {
          dvrdy = urbasis_grad(e,i,k,1);
          dvidy = uibasis_grad(e,i,k,1);
        }
        ScalarT dvrdz = 0.0;
        ScalarT dvidz = 0.0;
        if (spaceDim > 2) {
          dvrdz = urbasis_grad(e,i,k,2);
          dvidz = uibasis_grad(e,i,k,2);
        }
        
        if(!fractional) {       // fractional exponent on time operator or i_omega in frequency mode
          int resindex = offsets(ur_num,i);
          res(e,resindex) += -omega2r(e,k)*(ur*vr + ui*vi) + omega2i(e,k)*(ui*vr - ur*vi)
          + (c2r_x(e,k)*(durdx*dvrdx + duidx*dvidx)
             + c2r_y(e,k)*(durdy*dvrdy + duidy*dvidy)
             + c2r_z(e,k)*(durdz*dvrdz + duidz*dvidz)
             - c2i_x(e,k)*(duidx*dvrdx - durdx*dvidx)
             - c2i_y(e,k)*(duidy*dvrdy - durdy*dvidy)
             - c2i_z(e,k)*(duidz*dvrdz - durdz*dvidz))
          - (source_r(e,k)*vr + source_i(e,k)*vi); // TMW: how can both vr and vi appear in this equation?
          
          resindex = offsets(ui_num,i);
          
          res(e,resindex) += -omega2r(e,k)*(ui*vr - ur*vi) - omega2i(e,k)*(ur*vr + ui*vi)
          + (c2r_x(e,k)*(duidx*dvrdx - durdx*dvidx)
             + c2r_y(e,k)*(duidy*dvrdy - durdy*dvidy)
             + c2r_z(e,k)*(duidz*dvrdz - durdz*dvidz)
             + c2i_x(e,k)*(durdx*dvrdx + duidx*dvidx)
             + c2i_y(e,k)*(durdy*dvrdy + duidy*dvidy)
             + c2i_z(e,k)*(durdz*dvrdz + duidz*dvidz))
          - (source_i(e,k)*vr - source_r(e,k)*vi);
        }
        else {
          omegar(e,k) = sqrt(omega2r(e,k));
          omegai(e,k) = sqrt(omega2i(e,k));
          int resindex = offsets(ur_num,i);
          
          res(e,resindex) += alphaHr(e,k)*pow(omegar(e,k),2.0*freqExp(e,k))*(ur*vr + ui*vi)
          + alphaHr(e,k)*pow(omegar(e,k),2.0*freqExp(e,k))*(-ui*vr + ur*vi)
          + alphaHi(e,k)*pow(omegai(e,k),2.0*freqExp(e,k))*(-ui*vr + ur*vi)
          + alphaHi(e,k)*pow(omegai(e,k),2.0*freqExp(e,k))*(-ur*vr - ui*vi)
          + (c2r_x(e,k)*(durdx*dvrdx + duidx*dvidx)
             + c2r_y(e,k)*(durdy*dvrdy + duidy*dvidy)
             + c2r_z(e,k)*(durdz*dvrdz + duidz*dvidz)
             - c2i_x(e,k)*(duidx*dvrdx - durdx*dvidx)
             - c2i_y(e,k)*(duidy*dvrdy - durdy*dvidy)
             - c2i_z(e,k)*(duidz*dvrdz - durdz*dvidz))
          - (source_r(e,k)*vr + source_i(e,k)*vi);
          
          resindex = offsets(ui_num,i);
          
          res(e,resindex) += alphaHr(e,k)*pow(omegar(e,k),2.0*freqExp(e,k))*(ui*vr - ur*vi)
          + alphaHr(e,k)*pow(omegar(e,k),2.0*freqExp(e,k))*(ur*vr + ui*vi)
          + alphaHi(e,k)*pow(omegai(e,k),2.0*freqExp(e,k))*(ur*vr + ui*vi)
          + alphaHi(e,k)*pow(omegai(e,k),2.0*freqExp(e,k))*(-ui*vr + ur*vi)
          + (c2r_x(e,k)*(duidx*dvrdx - durdx*dvidx)
             + c2r_y(e,k)*(duidy*dvrdy - durdy*dvidy)
             + c2r_z(e,k)*(duidz*dvrdz - durdz*dvidz)
             + c2i_x(e,k)*(durdx*dvrdx + duidx*dvidx)
             + c2i_y(e,k)*(durdy*dvrdy + duidy*dvidy)
             + c2i_z(e,k)*(durdz*dvrdz + duidz*dvidz))
          - (source_i(e,k)*vr - source_r(e,k)*vi);
          
          // ScalarT c = 1.0; // bvbw need to move c and omega to input_params
          // ScalarT omega = 1.0;
          // wkset->res(resindex) +=
          //   -c*c*(durdx*dvrdx - duidx*dvidx) +
          //   -c*c*(durdy*dvrdy - duidy*dvidy) +
          //   -c*c*(durdz*dvrdz - duidz*dvidz) +
          //   alphar(k)*omega*(ur*vr - ui*vi) - alphai(k) * omega*(ur*vi + ui*vr) -
          //   (source_r(k)*vr - source_i(k)*vi);
          
          // resindex = wkset->offsets[ui_num][i];
          // //imaginary
          // wkset->res(resindex) +=
          //   -c*c*(durdx*dvrdx + duidx*dvidx) +
          //   -c*c*(durdy*dvrdy + duidy*dvidy) +
          //   -c*c*(durdz*dvrdz + duidz*dvidz) +
          //   alphar(k)*omega*(ur*vi + ui*vr) + alphai(k) * omega*(ur*vr - ui*vi) -
          //   (source_r(k)*vi + source_i(k)*vr);
        }
      }
    }
  });
  
}

// ========================================================================================
// ========================================================================================

void helmholtz::boundaryResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  Kokkos::View<int**,AssemblyDevice> bcs = wkset->var_bcs;
  int cside = wkset->currentside;
  
  int ur_basis_num = wkset->usebasis[ur_num];
  int ui_basis_num = wkset->usebasis[ui_num];
  
  // Set the parameters
  
  c2r_side_x = functionManager->evaluate("c2r_x","side ip");
  c2i_side_x = functionManager->evaluate("c2i_x","side ip");
  c2r_side_y = functionManager->evaluate("c2r_y","side ip");
  c2i_side_y = functionManager->evaluate("c2i_y","side ip");
  c2r_side_z = functionManager->evaluate("c2r_z","side ip");
  c2i_side_z = functionManager->evaluate("c2i_z","side ip");
  
  robin_alpha_r = functionManager->evaluate("robin_alpha_r","side ip");
  robin_alpha_i = functionManager->evaluate("robin_alpha_i","side ip");
  
  source_r_side = functionManager->evaluate("source_r_side","side ip");
  source_i_side = functionManager->evaluate("source_i_side","side ip");
  
  omega2r = functionManager->evaluate("omega2r","side ip");
  omega2i = functionManager->evaluate("omega2i","side ip");
  freqExp = functionManager->evaluate("freqExp","side ip");
  
  //sideinfo = wkset->sideinfo;
  sol = wkset->local_soln_side;
  sol_grad = wkset->local_soln_grad_side;
  offsets = wkset->offsets;
  DRV normals = wkset->normals;
  res = wkset->res;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  urbasis = wkset->basis_side[ur_basis_num];
  urbasis_grad = wkset->basis_grad_side[ur_basis_num];
  uibasis = wkset->basis_side[ui_basis_num];
  uibasis_grad = wkset->basis_grad_side[ui_basis_num];
  
  //Robin boundary condition of form alpha*u + dudn - source = 0, where u is the state and dudn is its normal derivative
  if (bcs(ur_num,cside) == 2) {
    for (int e=0; e<urbasis.extent(0); e++) { // not parallelized yet
      for( int k=0; k<urbasis.extent(2); k++ ) {
        
        
        AD ur = sol(e,ur_num,k,0);
        AD ui = sol(e,ui_num,k,0);
        AD durdx = sol_grad(e,ur_num,k,0);
        AD duidx = sol_grad(e,ui_num,k,0);
        AD durdn = durdx*normals(e,k,0);
        AD duidn = duidx*normals(e,k,0);
        
        AD durdy, duidy;
        if (spaceDim > 1){
          durdy = sol_grad(e,ur_num,k,1);
          duidy = sol_grad(e,ui_num,k,1);
          durdn += durdy*normals(e,k,1);
          duidn += duidy*normals(e,k,1);
        }
        AD durdz, duidz;
        if (spaceDim > 2) {
          durdz = sol_grad(e,ur_num,k,2);
          duidz = sol_grad(e,ui_num,k,2);
          durdn += durdz*normals(e,k,2);
          duidn += duidz*normals(e,k,2);
        }
        
        AD c2durdn = (c2r_side_x(e,k)*durdx - c2i_side_x(e,k)*duidx)*normals(e,k,0)
        + (c2r_side_y(e,k)*durdy - c2i_side_y(e,k)*duidy)*normals(e,k,1);
        
        AD c2duidn = (c2r_side_x(e,k)*duidx + c2i_side_x(e,k)*durdx)*normals(e,k,0)
        + (c2r_side_y(e,k)*duidy + c2i_side_y(e,k)*durdy)*normals(e,k,1);
        
        if (spaceDim > 2) {
          c2durdn +=(c2r_side_z(e,k)*durdz - c2i_side_z(e,k)*duidz)*normals(e,k,2);
          c2duidn +=(c2r_side_z(e,k)*duidz + c2i_side_z(e,k)*durdz)*normals(e,k,2);
        }
        
        if(!fractional) {       // fractional exponent on time operator or i_omega in frequency mode
          for (int i=0; i<urbasis.extent(1); i++ ) {
            int resindex = offsets(ur_num,i);
            ScalarT vr = urbasis(e,i,k);
            ScalarT vi = uibasis(e,i,k);
            
            res(e,resindex) += ((robin_alpha_r(e,k)*(ur*vr + ui*vi) - robin_alpha_i(e,k)*(ui*vr - ur*vi))
                                + (durdn*vr + duidn*vi)
                                - (source_r_side(e,k)*vr + source_i_side(e,k)*vi))
            - (c2durdn*vr + c2duidn*vi);
            
            resindex = offsets(ui_num,i);
            
            res(e,resindex) += ((robin_alpha_r(e,k)*(ui*vr - ur*vi) + robin_alpha_i(e,k)*(ur*vr + ui*vi))
                                + (duidn*vr - durdn*vi)
                                - (source_i_side(e,k)*vr - source_r_side(e,k)*vi))
            - (c2duidn*vr - c2durdn*vi);
          }
        }
        else {
          
          AD omegar = sqrt(omega2r(e,k));
          AD omegai = sqrt(omega2i(e,k));
          
          for (int i=0; i<urbasis.extent(1); i++ ) {
            int resindex = offsets(ur_num,i);
            ScalarT vr = urbasis(e,i,k);
            ScalarT vi = uibasis(e,i,k);
            
            res(e,resindex) +=  alphaTr(e,k)*pow(omegar,freqExp(e,k))*(-ur*vr - ui*vi)
            +  alphaTi(e,k)*pow(omegai,freqExp(e,k))*( ui*vr - ur*vi)
            + (durdn*vr + duidn*vi)
            - (source_r_side(e,k)*vr + source_i_side(e,k)*vi)
            - (c2durdn*vr + c2duidn*vi);
            
            resindex = offsets(ui_num,i);
            
            res(e,resindex) +=  alphaTr(e,k)*pow(omegar,freqExp(e,k))*(-ui*vr + ur*vi)
            +  alphaTi(e,k)*pow(omegai,freqExp(e,k))*(-ui*vr - ur*vi)
            + (duidn*vr - durdn*vi)
            - (source_i_side(e,k)*vr - source_r_side(e,k)*vi)
            - (c2duidn*vr - c2durdn*vi);
          }
          
        }
        // ScalarT c = 1.0; // bvbw need to move c and omega to input_params
        // omega(k) = sqrt(omega2r(k));
        // for (int i=0; i<numBasis; i++ ) {
        //   resindex = wkset->offsets[ur_num][i];
        //   wkset->res(resindex) += -1.0*(alphar(k)*omega(k)*(ur*vr - ui*vi) + alphai(k)*omega(k)*(k)*(ui*vr + ur*vi))
        //     + (c*durdn*vr - c*duidn*vi);
        
        //   // cout << alpha_r << "  " << omega << "  " << ur << "  " << alpha_i << "  " << ui << "  "<< c << endl;
        
        //   resindex = wkset->offsets[ui_num][i];
        //   wkset->res(resindex) += -1.0*(alphar(k)*omega(k)*(ur*vr - ui*vi) + alphai(k)*omega(k)*(k)*(ui*vr + ur*vi))
        //     + (c*durdn*vr - c*duidn*vi);
        // }
      }
    }
  }
  
}


void helmholtz::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void helmholtz::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void helmholtz::setVars(std::vector<string> & varlist_) {
  varlist = varlist_;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "ureal"){
      ur_num = i;
    }if (varlist[i] == "uimag"){
      ui_num = i;
    }
  }
}

