/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef THERMAL_FR_H
#define THERMAL_FR_H

#include "physics_base.hpp"

class thermal_fr : public physicsbase {
public:
  
  thermal_fr() {} ;
  
  ~thermal_fr() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  thermal_fr(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
             const size_t & numip_side_) :
  numip(numip_), numip_side(numip_side_) {
    label = "thermal_fr";
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    
    if (settings->sublist("Physics").get<int>("solver",0) == 1) 
      isTD = true;
    else
      isTD = false;
   
    multiscale = settings->isSublist("Subgrid"); 
    analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
    
    numResponses = settings->sublist("Physics").get<int>("numResp_thermal_fr",1); 
    useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (thermal_fr)",false);
    
    //test = settings->sublist("Physics").get<int>("test",0);
    //test == 40: topological optimization (Dirichlet on entire left side)
    //test == 41: topological optimization (Dirichlet on part of left side, enforced weakly)
    // test == 27; additive manufacturing thermal example
    //simNum = settings->sublist("Physics").get<int>("simulation_number",0);
    //simName = settings->sublist("Physics").get<string>("simulation_name","mySim");
    formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
    s_param = settings->sublist("Physics").get<ScalarT>("s_param",1.0);
    
    x = 0.0;
    y = 0.0;
    z = 0.0;

    //if (test == 28) {
    //  string ptsfile = "grains_pts.dat";
    //  string datafile = "grains_ids.dat";
    //  grains = data("grains", spaceDim, ptsfile, datafile, false);
    //}

    have_nsvel = false;
    dbgtimer1 = Teuchos::rcp(new Teuchos::Time("debug1",false)); 
    dbgtimer2 = Teuchos::rcp(new Teuchos::Time("debug2",false)); 
    dbgtimer3 = Teuchos::rcp(new Teuchos::Time("debug3",false)); 

    source = FCAD(numip);
    diff = FCAD(numip);
    cp = FCAD(numip);
    rho = FCAD(numip);
    
    nsource = FCAD(numip_side);
    diff_side = FCAD(numip_side);
    lvals = FCAD(numip_side);
    robin_alpha = FCAD(numip_side);
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    ScalarT s =0.2;
    ScalarT y = wkset->y;

    e_basis = wkset->usebasis[e_num];
    numBasis = wkset->basis[e_basis].dimension(1);
    
    udfunc->volumetricSource(label,"e",wkset,source);
    udfunc->coefficient("thermal_diffusion",wkset,false,diff);
    udfunc->coefficient("heat_capacity",wkset,false,cp);
    udfunc->coefficient("density",wkset,false,rho);
    
    for (int k=0; k<numip; k++ ) {
      e = wkset->local_soln(e_num,k,0);
      e_dot = wkset->local_soln_dot(e_num,k,0);
      dedx = wkset->local_soln_grad(e_num,k,0);
      if (spaceDim > 1) {
        dedy = wkset->local_soln_grad(e_num,k,1);
      }
      if (spaceDim > 2) {
        dedz = wkset->local_soln_grad(e_num,k,2);
      }
      if (have_nsvel) {
        ux = wkset->local_soln(ux_num,k,0);
        if (spaceDim > 1) {
          uy = wkset->local_soln(uy_num,k,1);
        }
        if (spaceDim > 2) {
          uz = wkset->local_soln(uz_num,k,2);
        }
      }
      
      for (int i=0; i<numBasis; i++ ) {
        resindex = wkset->offsets[e_num][i];
        v = wkset->basis[e_basis](0,i,k);
        dvdx = wkset->basis_grad[e_basis](0,i,k,0);
        if (spaceDim > 1) {
          dvdy = wkset->basis_grad[e_basis](0,i,k,1);
        }
        if (spaceDim > 2) {
          dvdz = wkset->basis_grad[e_basis](0,i,k,2);
        }

        wkset->res(resindex) += rho(k)*cp(k)*e_dot*v + diff(k)*(dedx*dvdx + dedy*dvdy + dedz*dvdz) 
	                      + exp((1-s_param)*y)*exp(y)*e*v - source(k)*v;
	//        wkset->res(resindex) += rho(k)*cp(k)*e_dot*v + diff(k)*(dedx*dvdx + dedy*dvdy + dedz*dvdz) - source(k)*v;
        if (have_nsvel)
          wkset->res(resindex) += (ux*dvdx + uy*dvdy + uz*dvdz);
        
      }
      
    }
  }  
  
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
   
    
    e_basis = wkset->usebasis[e_num];
    numBasis = wkset->basis_side[e_basis].dimension(1);
  
    if (wkset->sidetype > 1) { // Neumann BC
      nsource = udfunc->boundaryNeumannSource(label,"e",wkset);
      for (int k=0; k<numip_side; k++ ) {
        for (int i=0; i<numBasis; i++ ) {
          resindex = wkset->offsets[e_num][i];
          v = wkset->basis_side[e_basis](0,i,k);
          wkset->res(resindex) += -nsource(k)*v;
        }
      }
    }
    else { // Weak Dirichlet
      udfunc->coefficient("thermal_diffusion",wkset,true,diff_side);
      lvals = udfunc->boundaryNeumannSource(label,"e",wkset);
      udfunc->coefficient("robinAlpha", wkset,true,robin_alpha);
      
      ScalarT sf = formparam;
      if (wkset->isAdjoint) {
        sf = 1.0;
      }
      
      for (int k=0; k<numip_side; k++ ) {
        e = wkset->local_soln_side(e_num,k,0);
        dedx = wkset->local_soln_grad_side(e_num,k,0);
        if (spaceDim > 1) {
          dedy = wkset->local_soln_grad_side(e_num,k,1);
        }
        if (spaceDim > 2) {
          dedz = wkset->local_soln_grad_side(e_num,k,2);
        }

        if (wkset->sidetype == -1) 
          lambda = wkset->local_aux_side(e_num,k);
        else {
          lambda = this->getDirichletValue("e",x, y, z, wkset->time, wkset->sidename, wkset->isAdjoint);
        }
        
        for (int i=0; i<numBasis; i++ ) {
          resindex = wkset->offsets[e_num][i];
          v = wkset->basis_side[e_basis](0,i,k);
          dvdx = wkset->basis_grad_side[e_basis](0,i,k,0);
          if (spaceDim > 1)
            dvdy = wkset->basis_grad_side[e_basis](0,i,k,1);
          if (spaceDim > 2)
            dvdz = wkset->basis_grad_side[e_basis](0,i,k,2);
  
          weakDiriScale = 10.0*diff_side(k)/wkset->h;
          wkset->res(resindex) += -diff_side(k)*dedx*wkset->normals(0,k,0)*v - sf*diff_side(k)*dvdx*wkset->normals(0,k,0)*(e-lambda) + weakDiriScale*(e-lambda)*v;
          if (spaceDim > 1) {
            wkset->res(resindex) += -diff_side(k)*dedy*wkset->normals(0,k,1)*v - sf*diff_side(k)*dvdy*wkset->normals(0,k,1)*(e-lambda);
          }
          if (spaceDim > 2) 
            wkset->res(resindex) += -diff_side(k)*dedz*wkset->normals(0,k,2)*v - sf*diff_side(k)*dvdz*wkset->normals(0,k,2)*(e-lambda);

          if (wkset->isAdjoint) {
            wkset->adjrhs(resindex) += sf*diff_side(k)*dvdx*wkset->normals(0,k,0)*lambda - weakDiriScale*lambda*v;
            if (spaceDim > 1)
              wkset->adjrhs(resindex) += sf*diff_side(k)*dvdy*wkset->normals(0,k,1)*lambda;
            if (spaceDim > 2) 
              wkset->adjrhs(resindex) += sf*diff_side(k)*dvdz*wkset->normals(0,k,2)*lambda;
          }
        }
      }
    }
  }  
  
  // ========================================================================================
  // ========================================================================================

  void edgeResidual() {}
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux() {

    ScalarT sf = 1.0;
    if (wkset->isAdjoint) {
      sf = formparam;
    }

    udfunc->coefficient("thermal_diffusion",wkset,true,diff_side);
    for (size_t i=0; i<wkset->ip_side.dimension(1); i++) {
      penalty = 10.0*diff_side(i)/wkset->h;
      wkset->flux(e_num,i) += sf*diff_side(i)*wkset->local_soln_grad_side(e_num,i,0)*wkset->normals(0,i,0) + penalty*(wkset->local_aux_side(e_num,i)-wkset->local_soln_side(e_num,i,0));
      if (spaceDim > 1)
        wkset->flux(e_num,i) += sf*diff_side(i)*wkset->local_soln_grad_side(e_num,i,1)*wkset->normals(0,i,1);
      if (spaceDim > 2)
        wkset->flux(e_num,i) += sf*diff_side(i)*wkset->local_soln_grad_side(e_num,i,2)*wkset->normals(0,i,2);

    }

  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses, const bool & justDeriv) const {
    return integralResponses;
  }
  
  bool useScalarRespFunc() const {
    return useScalarRespFx;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    ux_num = -1;
    uy_num = -1;
    uz_num = -1;
    
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "e")
        e_num = i;
      if (varlist[i] == "ux")
        ux_num = i;
      if (varlist[i] == "uy")
        uy_num = i;
      if (varlist[i] == "uz")
        uz_num = i;
    }
    if (ux_num >=0)
      have_nsvel = true;
  }
  
  // ========================================================================================
  // return the coefficient of robin term
  // ========================================================================================
  
  // TMW: could not find where this was used. NEEDS TO BE MOVED TO UDFUNC
  /*AD robinAlpha(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                const string & side, const AD & weakDiriScale) const {
    AD val = 0.0;
    if((test == 41) && ((side == "left") && (abs(y-0.5) < 0.5*sinkwidth)))
      val = weakDiriScale;
    return 0.0;
  }*/
  
  // ========================================================================================
  // ========================================================================================
  
  int getNumResponses() {
    return numResponses;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<string> ResponseFieldNames() const {
    std::vector<string> rf; 
    return rf;
  }

  // ========================================================================================
  // ========================================================================================
  
  vector<FC> extraFields() const {
    vector<FC> ef;
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setExtraFields(const size_t & numElem_) {
    numElem = numElem_;
  }
  
  // ========================================================================================
  // ========================================================================================

  void printTimers() {
    cout << "Thermal Debug timer 1: " << dbgtimer1->totalElapsedTime() << endl;
    cout << "Thermal Debug timer 2: " << dbgtimer2->totalElapsedTime() << endl;
    cout << "Thermal Debug timer 3: " << dbgtimer3->totalElapsedTime() << endl;
  }

private:
  
  data grains;
 
  size_t numip, numip_side;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int e_num, e_basis, numBasis, ux_num, uy_num, uz_num;
  ScalarT alpha;
  bool isTD;

  //int test, simNum;
  //string simName;
  
  ScalarT v, dvdx, dvdy, dvdz, x, y, z;
  AD e, e_dot, dedx, dedy, dedz, reax, weakDiriScale, lambda, penalty;
  AD ux, uy, uz;
  
  int resindex;
  
  FCAD diff, rho, cp, source, nsource, diff_side, lvals, robin_alpha;
  
  string analysis_type; //to know when parameter is a sample that needs to be transformed
  
  bool useScalarRespFx;
  bool multiscale, have_nsvel;
  ScalarT formparam;
  ScalarT s_param;      //fractional exponent
  Teuchos::RCP<Teuchos::Time> dbgtimer1;
  Teuchos::RCP<Teuchos::Time> dbgtimer2;
  Teuchos::RCP<Teuchos::Time> dbgtimer3;
  
};

#endif
