/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/


#ifndef MWHELMHOLTZ_H
#define MWHELMHOLTZ_H

#include "physics_base.hpp"

class mwhelmholtz : public physicsbase {
public:
  
  mwhelmholtz() {} ;
  
  ~mwhelmholtz() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  mwhelmholtz(Teuchos::RCP<Teuchos::ParameterList> & settings) { //this version treats the real and imaginary parts as seperate variables  
  
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    PI = 3.141592653589793238463;
    
    verbosity = settings->sublist("Physics").get<int>("Verbosity",0);
    
    if (settings->sublist("Physics").get<int>("solver",0) == 1){
      cout << "Helmholtz equation doesn't have transience...?" << endl;
      isTD = true; 
    }
    else
      isTD = false;
      
    numwavenum = settings->sublist("Physics").get<int>("number_wavenumbers",1);
      
    regParam = settings->sublist("Analysis").sublist("ROL").get<double>("regularization parameter",1.e-4);
    cMY = settings->sublist("Physics").get<double>("cMY",1.0);
    pixelate = settings->sublist("Physics").get<bool>("topo opt pixelate",false);
    linInterp = settings->sublist("Physics").get<bool>("topo opt linear interp",false);
    binPenalty = settings->sublist("Physics").get<double>("topo opt binary penalty",0.0); 
    
    test = settings->sublist("Physics").get<int>("test",0);
    //test == 4: for convergence study with manufactured solution (spatially varying c)
    //test == 7: topo opt (two targets for two frequencies)
    
    numResponses = 2*numwavenum;
    if(test == 7)
      numResponses = 1;
      
    if(test == 7 && numwavenum != 2)
      cout << "AAAHHH test 7 is meant for exactly 2 wavenumbers.." << endl;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual(const double & h, const double & current_time, const FCAD & local_soln, 
                      const FCAD & local_solngrad, const FCAD & local_soln_dot, 
                      const FCAD & local_param, const FCAD & local_param_grad, 
                      const FCAD & local_aux, const FCAD & local_aux_grad, 
                      const FC & ip, const vector<FC> & basis, const vector<FC> & basis_grad, 
                      const vector<int> & usebasis, const vector<vector<int> > & offsets,
                      const bool & onlyTransient, FCAD & local_resid) {
  
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    size_t numCC = h.size();
    int ur_basis = usebasis[ur_num];
    int ui_basis = usebasis[ui_num];
    
    int numBasis = basis.dimension(2);
    int numCubPoints = ip.dimension(1);
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    vector<AD> ur, ur_dot, durdx, durdy, durdz;
    vector<AD> ui, ui_dot, duidx, duidy, duidz;
    double vr, dvrdx, dvrdy, dvrdz;
    double vi, dvidx, dvidy, dvidz;
    vector<AD> source_r, source_i; 
    vector<AD> omega2;
    vector<AD> c2r, c2i;
    
    for (size_t ee=0; ee<numCC; ee++) {
      for( int i=0; i<numBasis; i++ ) {
        for( int nPt=0; nPt<numCubPoints; nPt++ ) {
          
          ur.clear(); ur_dot.clear(); durdx.clear(); durdy.clear(); durdz.clear();
          ui.clear(); ui_dot.clear(); duidx.clear(); duidy.clear(); duidz.clear();
          
          vr = basis(ee,0,i,nPt);
          vi = basis(ee,1,i,nPt);
          dvrdx = basis_grad(ee,0,i,nPt,0);
          dvidx = basis_grad(ee,1,i,nPt,0);
          
          x = ip(ee,nPt,0);
          
          for(int j=0; j<numwavenum; j++){
            ur.push_back(local_soln(ee,ur_num[j],nPt));
            ur_dot.push_back(local_soln_dot(ee,ur_num[j],nPt));
            durdx.push_back(local_solngrad(ee,ur_num[j],nPt,0));
            ui.push_back(local_soln(ee,ui_num[j],nPt));
            ui_dot.push_back(local_soln_dot(ee,ui_num[j],nPt));
            duidx.push_back(local_solngrad(ee,ui_num[j],nPt,0));
          }
          if (spaceDim > 1) {
            y = ip(ee,nPt,1);
            for(int j=0; j<numwavenum; j++){
              durdy.push_back(local_solngrad(ee,ur_num[j],nPt,1));
              duidy.push_back(local_solngrad(ee,ui_num[j],nPt,1));
            }
            dvrdy = basis_grad(ee,0,i,nPt,1);
            dvidy = basis_grad(ee,1,i,nPt,1);
          }
          
          if (spaceDim > 2) {
            z = ip(ee,nPt,2);
            for(int j=0; j<numwavenum; j++){
              durdz.push_back(local_solngrad(ee,ur_num[j],nPt,2));
              duidz.push_back(local_solngrad(ee,ui_num[j],nPt,2));
            }
            dvrdz = basis_grad(ee,0,i,nPt,2);
            dvidz = basis_grad(ee,1,i,nPt,2);
          }
          
          vector<vector<AD> > c2 = this->velSquared(x, y, z, current_time);
          c2r = c2[0];
          c2i = c2[1];
          
          omega2 = this->freqSquared(x, y, z, current_time);
          
          vector<vector<AD> > source = this->sourceTerm(x, y, z, current_time);
          source_r = source[0]; 
          source_i = source[1]; 
          
          for(int j=0; j<numwavenum; j++){
            if(isTD){
              local_resid(ee,2*j,i) += ur_dot[j]*vr + ui_dot[j]*vi;
              local_resid(ee,2*j+1) += ui_dot[j]*vr - ur_dot[j]*vi;
            }
            
            
            if(spaceDim == 1){
              local_resid(ee,2*j,i) += -omega2[j]*(ur[j]*vr + ui[j]*vi) //indefinite version
                          + (c2r[j]*(durdx[j]*dvrdx + duidx[j]*dvidx) 
                           - c2i[j]*(duidx[j]*dvrdx - durdx[j]*dvidx))
                          - (source_r[j]*vr + source_i[j]*vi);
              local_resid(ee,2*j+1,i) += -omega2[j]*(ui[j]*vr - ur[j]*vi) //indefinite version
                          + (c2r[j]*(duidx[j]*dvrdx - durdx[j]*dvidx)
                           + c2i[j]*(durdx[j]*dvrdx + duidx[j]*dvidx))
                          - (source_i[j]*vr - source_r[j]*vi);
            }else if(spaceDim == 2){
              local_resid(ee,2*j,i) += -omega2[j]*(ur[j]*vr + ui[j]*vi) //indefinite version
                          + (c2r[j]*(durdx[j]*dvrdx + duidx[j]*dvidx + durdy[j]*dvrdy + duidy[j]*dvidy) 
                           - c2i[j]*(duidx[j]*dvrdx - durdx[j]*dvidx + duidy[j]*dvrdy - durdy[j]*dvidy))
                          - (source_r[j]*vr + source_i[j]*vi);
              local_resid(ee,2*j+1,i) += -omega2[j]*(ui[j]*vr - ur[j]*vi) //indefinite version
                          + (c2r[j]*(duidx[j]*dvrdx - durdx[j]*dvidx + duidy[j]*dvrdy - durdy[j]*dvidy)
                           + c2i[j]*(durdx[j]*dvrdx + duidx[j]*dvidx + durdy[j]*dvrdy + duidy[j]*dvidy))
                          - (source_i[j]*vr - source_r[j]*vi);
                          
            }else if(spaceDim == 3){
              local_resid(ee,2*j,i) += -omega2[j]*(ur[j]*vr + ui[j]*vi) //indefinite version
                          + (c2r[j]*(durdx[j]*dvrdx + duidx[j]*dvidx + durdy[j]*dvrdy + duidy[j]*dvidy + durdz[j]*dvrdz + duidz[j]*dvidz) 
                           - c2i[j]*(duidx[j]*dvrdx - durdx[j]*dvidx + duidy[j]*dvrdy - durdy[j]*dvidy + duidz[j]*dvrdz - durdz[j]*dvidz))
                          - (source_r[j]*vr + source_i[j]*vi);
              local_resid(ee,2*j+1,i) += -omega2[j]*(ui[j]*vr - ur[j]*vi) //indefinite version
                          + (c2r[j]*(duidx[j]*dvrdx - durdx[j]*dvidx + duidy[j]*dvrdy - durdy[j]*dvidy + duidz[j]*dvrdz - durdz[j]*dvidz)
                           + c2i[j]*(durdx[j]*dvrdx + duidx[j]*dvidx + durdy[j]*dvrdy + duidy[j]*dvidy + durdz[j]*dvrdz + duidz[j]*dvidz))
                          - (source_i[j]*vr - source_r[j]*vi);
            }
          }
        }
      }
    }
    
    return local_resid;
  
  }  
  
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD boundaryResidual(const std::vector<double> & h, 
                    const double & current_time, const double & deltat, const FCAD & local_soln, 
                    const FCAD & local_solngrad, 
                    const FCAD & local_solndot, 
                                      const FCAD & local_aux, const FCAD & local_aux_grad, 
                                      const FC & ip, const FC & normals, 
                    const FC & basis, const FC & basis_grad, 
                    const int & numDOF, const string & side_name) const {
  
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    size_t numCC = h.size();
    int numBasis = basis.dimension(2);
    int numSideCubPoints = ip.dimension(1);
    
    FCAD local_resid(numCC, 2*numwavenum, numBasis);
    
    // Set the parameters
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    vector<AD> ur, durdn;
    vector<AD> ui, duidn;
    double vr, vi;
    vector<AD> source_r, source_i;
    vector<AD> robin_alpha_r, robin_alpha_i;
    vector<AD> c2r, c2i;
    
    for (size_t ee=0; ee<numCC; ee++) {
      for( int i=0; i<numBasis; i++ ) {
        for( int nPt=0; nPt<numSideCubPoints; nPt++ ) {
          ur.clear(); durdn.clear();
          ui.clear(); duidn.clear();
        
          vr = basis(ee,0,i,nPt);
          vi = basis(ee,1,i,nPt);
          x = ip(ee,nPt,0);
          
          for(int j=0; j<numwavenum; j++){
            ur.push_back(local_soln(ee,ur_num[j],nPt));
            ui.push_back(local_soln(ee,ui_num[j],nPt));
            durdn.push_back(local_solngrad(ee,ur_num[j],nPt,0)*normals(ee,nPt,0));
            duidn.push_back(local_solngrad(ee,ui_num[j],nPt,0)*normals(ee,nPt,0));    
          } 

          if (spaceDim > 1) {
            y = ip(ee,nPt,1);
            for(int j=0; j<numwavenum; j++){
              durdn[j] += local_solngrad(ee,ur_num[j],nPt,1)*normals(ee,nPt,1);
              duidn[j] += local_solngrad(ee,ui_num[j],nPt,1)*normals(ee,nPt,1);
            }
          }
          if (spaceDim > 2) {
            z = ip(ee,nPt,2);
            for(int j=0; j<numwavenum; j++){
              durdn[j] += local_solngrad(ee,ur_num[j],nPt,2)*normals(ee,nPt,2);
              duidn[j] += local_solngrad(ee,ui_num[j],nPt,2)*normals(ee,nPt,2);
            }
          }
          
          //Robin boundary condition of form alpha*u + dudn - source = 0, where u is the state and dudn is its normal derivative
          
          vector<vector<AD> > source = this->boundarySource(x, y, z, current_time, side_name);
          source_r = source[0];
          source_i = source[1];
          
          vector<vector<AD> > robin_alpha = this->robinAlpha(x, y, z, current_time, side_name);
          robin_alpha_r = robin_alpha[0]; 
          robin_alpha_i = robin_alpha[1];
        
          vector<vector<AD> > c2 = this->velSquared(x, y, z, current_time);
          c2r = c2[0];
          c2i = c2[1];
          
          for(int j=0; j<numwavenum; j++){
            local_resid(ee,2*j,i) += ((robin_alpha_r[j]*(ur[j]*vr + ui[j]*vi) - robin_alpha_i[j]*(ui[j]*vr - ur[j]*vi)) 
                      + (durdn[j]*vr + duidn[j]*vi) - (source_r[j]*vr + source_i[j]*vi)) 
                      - (c2r[j]*(durdn[j]*vr + duidn[j]*vi) - c2i[j]*(duidn[j]*vr - durdn[j]*vi)); 
            local_resid(ee,2*j+1,i) += ((robin_alpha_r[j]*(ui[j]*vr - ur[j]*vi) + robin_alpha_i[j]*(ur[j]*vr + ui[j]*vi)) 
                      + (duidn[j]*vr - durdn[j]*vi) - (source_i[j]*vr - source_r[j]*vi))
                      - (c2r[j]*(duidn[j]*vr - durdn[j]*vi) + c2i[j]*(durdn[j]*vr + duidn[j]*vi));
          }
        }
      }
    }
    
    return local_resid;
  
  }  
  
  // ========================================================================================
  // error calculation
  // ========================================================================================
  
  double trueSolution(const string & var, const double & x, const double & y, const double & z, const double & time) const {
    double e = 0.0;
    
    if(test == 4){
      e = sin(2*PI*x);
      if (spaceDim > 1)
        e *= sin(2*PI*y);
      if (spaceDim > 2)
        e *= sin(2*PI*z);
    }
    else
      std::cout << "AAHHH NO TRUE SOLUTION KNOWN WHAT ARE YOU DOING HERE" << std::endl;
    
    return e;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  AD getDirichletValue(const string & var, const double & x, const double & y, const double & z, const double & t, const string & gside, const bool & useadjoint) const {
    AD val = 0.0;
    
    if(!useadjoint){
      if(test == 4)
        val = this->trueSolution(var,x,y,z,t); 
    }
      
    return val;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  AD getInitialValue(const string & var, const double & x, const double & y, const double & z, const bool & useadjoint) const {
    AD val = 0.0;
    return val;
  }
  
  // ========================================================================================
  // response calculation
  // ========================================================================================
  
  FCAD response(const FCAD & local_soln, const FCAD & local_soln_grad,
                 const DRV & ip, const double & time) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD resp(numCC,numResponses,numip);
    
    double x = 0.0;
    double y = 0.0;
    
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        if(test == 7){
          x = ip(i,j,0);
          y = ip(i,j,1);
          if(x > 0.5 && x < 1.0 && y > 0.25 && y < 0.5){ //focus energy from first frequency here
            resp(i,0,j) += -0.5*(pow(local_soln(i,ur_num[0],j),2.0)+pow(local_soln(i,ui_num[0],j),2.0)) 
                         + 0.5*(pow(local_soln(i,ur_num[1],j),2.0)+pow(local_soln(i,ui_num[1],j),2.0));
          }else if(x > 0.5 && x < 1.0 && y < -0.25 && y > -0.5){ //focus energy from second frequency here
            resp(i,0,j) += 0.5*(pow(local_soln(i,ur_num[0],j),2.0)+pow(local_soln(i,ui_num[0],j),2.0)) 
                         - 0.5*(pow(local_soln(i,ur_num[1],j),2.0)+pow(local_soln(i,ui_num[1],j),2.0));
          }else if(x > 0.5 && x < 1.0 && y >= -0.25 && y <= 0.25){ //miminize energy in between...
            resp(i,0,j) += 0.5*(pow(local_soln(i,ur_num[0],j),2.0)+pow(local_soln(i,ui_num[0],j),2.0)) 
                         + 0.5*(pow(local_soln(i,ur_num[1],j),2.0)+pow(local_soln(i,ui_num[1],j),2.0));
          }
        }else{
          for(int k=0; k<numwavenum; k++){
            resp(i,2*k,j) = local_soln(i,ur_num[k],j); 
            resp(i,2*k+1,j) = local_soln(i,ui_num[k],j); 
          }
        }   
      }
    }
    
    return resp;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses, const bool & justDeriv) const {
    if(test == 7){
      FCAD resp(integralResponses.dimension(0));
      resp(0) = integralResponses(0);
      if(!justDeriv){
        if(pixelate){
          AD volFrac = 0.0;
          double cVF = 0.1;
          AD binPen = 0.0;
          int n = round(sqrt(vel_params.size()));    
          double dx = 1.0/n;
          double dy = 1.0/n;
          
          for(int i=0; i<vel_params.size(); i++){
            volFrac += vel_params[i]*dx*dy;
            binPen += binPenalty*vel_params[i]*(1.0-vel_params[i])*dx*dy;
          }
          
          resp(0) += (cMY/3.0)*pow(max(volFrac-cVF,0.0),3.0) + binPen;
        }else{
          AD volFrac = 0.0;
          double cVF = 0.1; //desired max volume fraction
          int n = round(sqrt(vel_params.size()));    
          double dx = 1.0/(n-1);
          double dy = 1.0/(n-1);
          
          for(int i=0; i<vel_params.size(); i++){
            if((i == 0) || (i == n-1) || (i == n*n-1) || (i == n*n-n)) //corner points
              volFrac += vel_params[i]*0.25*dx*dy;
            else if((i > 0 && i < n) || (i%n == 0) || ((i+1)%n == 0) || (i < n*n-1 && i > n*n-n)) //edge points
              volFrac += vel_params[i]*0.5*dx*dy;
            else //interior points
              volFrac += vel_params[i]*dx*dy;
          }
          resp(0) += (cMY/3.0)*pow(max(volFrac-cVF,0.0),3.0);
        }
      }
      return resp;
    }
    else{
      cout << "?!?!?!?" << endl;
      return integralResponses;
    }
  }
  
  bool useScalarRespFunc() const {
    if(test == 7)
      return true;
    else
      return false;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD target(const FC & ip, const double & time) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD targ(numCC,numResponses,numip);

    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        targ(i,0,j) = 1.0;
        if(numResponses > 1)
          targ(i,1,j) = 1.0;
      }
    }
    
    return targ;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    ur_num.clear();
    ui_num.clear();
    for (size_t i=0; i<varlist.size(); i++) {
      if ((varlist[i].find("ureal") != string::npos) && (varlist[i] != "ureal"))
        ur_num.push_back(i);
      if ((varlist[i].find("uimag") != string::npos) && (varlist[i] != "uimag"))
        ui_num.push_back(i);
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setUserDefined(Teuchos::RCP<userDefined> & udfunc_) {
    udfunc = udfunc_;
  }
  
  // ========================================================================================
  /* return the source term (to be multiplied by test_function) */
  // ========================================================================================  
  
  vector<vector<AD> > sourceTerm(const double & x, const double & y, const double & z, const double & time) const {
    vector<vector<AD> > source(2,vector<AD>(numwavenum,0.0));
    
    if(test == 4){
      vector<AD> omega2 = this->freqSquared(x, y, z, time);
      if(spaceDim == 1){
        for(int j=0; j<numwavenum; j++){
          source[0][j] = (4*PI*PI*(x*x-2.0*x-1.0)-omega2[j])*sin(2*PI*x) + (2.0-2.0*x)*(2*PI*cos(2*PI*x));
          source[1][j] = (4*PI*PI*(x*x+2.0*x-1.0)-omega2[j])*sin(2*PI*x) + (-2.0-2.0*x)*(2*PI*cos(2*PI*x));
        }
      }else if(spaceDim == 2){
        for(int j=0; j<numwavenum; j++){
          source[0][j] = (8*PI*PI*(x*x-2.0*x-1.0)-omega2[j])*sin(2*PI*x)*sin(2*PI*y) + (2.0-2.0*x)*(2*PI*cos(2*PI*x)*sin(2*PI*y));
          source[1][j] = (8*PI*PI*(x*x+2.0*x-1.0)-omega2[j])*sin(2*PI*x)*sin(2*PI*y) + (-2.0-2.0*x)*(2*PI*cos(2*PI*x)*sin(2*PI*y));
        }
      }else if(spaceDim == 3){
        for(int j=0; j<numwavenum; j++){
          source[0][j] = (12*PI*PI*(x*x-2.0*x-1.0)-omega2[j])*sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z) + (2.0-2.0*x)*(2*PI*cos(2*PI*x)*sin(2*PI*y)*sin(2*PI*z));
          source[1][j] = (12*PI*PI*(x*x+2.0*x-1.0)-omega2[j])*sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z) + (-2.0-2.0*x)*(2*PI*cos(2*PI*x)*sin(2*PI*y)*sin(2*PI*z));
        }
      }
    }else if(test == 7){
      double source_size = 0.2;
      double source_shift = -1.0;
      for(int j=0; j<numwavenum; j++){
        source[0][j] = source_params[0]*max(pow(source_size,2.0)-y*y-pow(x-source_shift,2.0),0.0);
        source[1][j] = 0.0;
      }
    }
    return source;
  }

  // ========================================================================================
  /* return the (nominal) squared velocity */
  // ========================================================================================
  
  vector<vector<AD> > velSquared(const double & x, const double & y, const double & z, const double & time) const {
  
    vector<vector<AD> > c2(2,vector<AD>(numwavenum,0.0));
    
    if(test == 4){
      for(int j=0; j<numwavenum; j++){
        c2[0][j] = x*x-1.0;
        c2[1][j] = 2.0*x;
      }
    }else if(test == 7){
      if(abs(x) < 0.5 && abs(y) < 0.5){
        if(pixelate){
          int n = round(sqrt(vel_params.size()));
          if(n*n != vel_params.size())
            cout << "AAAHHH need square number of parameters...number of parameters: " << vel_params.size() << endl;
            
          double dx = 1.0/n;
          double dy = 1.0/n;
          
          //shift x and y
          double xs = x + 0.5;
          double ys = y + 0.5;
          
          int ind = floor(ys/dy) + floor(xs/dx)*n;
          AD q = vel_params[ind];
          
          //cubic-y interpolation (like with thermal topo opt)
          double c2minr = 1.0;
          double c2mini = 0.0;
          if(!linInterp){
            for(int j=0; j<numwavenum; j++){
              c2[0][j] = c2minr + (-5./25.04 - c2minr)*(3.0*q*q - 2.0*q*q*q);
              c2[1][j] = c2mini + (-0.2/25.04 - c2mini)*(3.0*q*q - 2.0*q*q*q);
            }
          }else{
            //linear interpolation (like in Wadbro and Engstrom (2015))
            for(int j=0; j<numwavenum; j++){
              c2[0][j] = c2minr + q*(-5./25.04 - c2minr); 
              c2[1][j] = c2mini + q*(-0.2/25.04 - c2mini);
            }
          }
        }else{
          int n = round(sqrt(vel_params.size()));
          if(n*n != vel_params.size())
            cout << "AAAHHH need square number of parameters...number of parameters: " << vel_params.size() << endl;
            
          double dx = 1.0/(n-1);
          double dy = 1.0/(n-1);
          
          //shift x and y
          double xs = x + 0.5;
          double ys = y + 0.5;
          
          //[0,1]x[0,1]
          int indTopRight = max(ceil(xs/dx),1.0)*n + max(ceil(ys/dy),1.0);
          int indTopLeft = indTopRight - n;
          int indBotRight = indTopRight - 1;
          int indBotLeft = indBotRight - n;
          
          double xi = (xs-floor(xs/dx)*dx)/dx;
          double eta = (ys-floor(ys/dy)*dy)/dy;
          
          AD q = vel_params[indBotLeft]*(1-xi-eta+xi*eta)
              + vel_params[indTopLeft]*(eta-xi*eta)
              + vel_params[indTopRight]*(xi*eta)
              + vel_params[indBotRight]*(xi-xi*eta);
          
          //cubic-y interpolation (like with thermal topo opt)
          double c2minr = 1.0;
          double c2mini = 0.0;
          if(!linInterp){
            for(int j=0; j<numwavenum; j++){
              c2[0][j] = c2minr + (-5./25.04 - c2minr)*(3.0*q*q - 2.0*q*q*q);
              c2[1][j] = c2mini + (-0.2/25.04 - c2mini)*(3.0*q*q - 2.0*q*q*q);
            }
          }else{
            //linear interpolation (like in Wadbro and Engstrom (2015))
            for(int j=0; j<numwavenum; j++){
              c2[0][j] = c2minr + q*(-5./25.04 - c2minr);
              c2[1][j] = c2mini + q*(-0.2/25.04 - c2mini);
            }
          }
        }
      }else{
        for(int j=0; j<numwavenum; j++){
          c2[0][j] = 1.0;
          c2[1][j] = 0.0;
        }
      }
    }
    
    return c2;
  
  }

  // ========================================================================================
  /* return the (nominal) frequency squared */
  // ========================================================================================
  
  vector<AD> freqSquared(const double & x, const double & y, const double & z, const double & time) const {
  
    vector<AD> omega2(numwavenum,0.0);
    if( test == 4 || test == 7){
      for(int j=0; j<numwavenum; j++)
        omega2[j] = freq_params[j]*freq_params[j];
    }
    return omega2;
  
  }
  
  // ========================================================================================
  /* return the boundary source term */
  // ========================================================================================
  
  vector<vector<AD> > boundarySource(const double & x, const double & y, const double & z, const double & t, const string & side) const {
  
    vector<vector<AD> > bsource(2,vector<AD>(numwavenum,0.0));
    
    if(test == 4){
      if(side == "right"){
        for(int j=0; j<numwavenum; j++){
          bsource[0][j] = 2*PI*cos(2*PI*x); 
          bsource[1][j] = 2*PI*cos(2*PI*x);
          if(spaceDim > 1){
            bsource[0][j] *= sin(2*PI*y); 
            bsource[1][j] *= sin(2*PI*y); 
          }
          if(spaceDim > 2){
            bsource[0][j] *= sin(2*PI*z); 
            bsource[1][j] *= sin(2*PI*z); 
          }
        }
      }
    }else if(test == 7){
      for(int j=0; j<numwavenum; j++){
        bsource[0][j] = 0.0; 
        bsource[1][j] = 0.0; 
      }
    }
    
    return bsource;
  }
  
  // ========================================================================================
  /* return the coefficient for Robin/Neumann boundary conditions*/
  // ========================================================================================
  
  vector<vector<AD> > robinAlpha(const double & x, const double & y, const double & z, const double & t, const string & side) const {
    vector<vector<AD> > alpha(2,vector<AD>(numwavenum,0.0));
    
    if(test == 4){
      for(int j=0; j<numwavenum; j++){
        alpha[0][j] = 0.0; 
        alpha[1][j] = 0.0; 
      }
    }else if(test == 7){
      vector<vector<AD> > c2 = this->velSquared(x,y,z,t);
      vector<AD> c2r = c2[0];
      vector<AD> c2i = c2[1];
      
      vector<AD> omega2 = this->freqSquared(x,y,z,t);
      
      for(int j=0; j<numwavenum; j++){
        AD k2r = omega2[j]*(c2r[j]/(c2r[j]*c2r[j]+c2i[j]*c2i[j]));
        AD k2i = omega2[j]*(-c2i[j]/(c2r[j]*c2r[j]+c2i[j]*c2i[j]));
        
        //get principal square root of complex k^2
        AD k2mag = sqrt(k2r*k2r+k2i*k2i); //|k^2|
        AD k2pk2magmag = sqrt((k2mag+k2r)*(k2mag+k2r)+k2i*k2i); //|k^2+|k^2||
        AD sqrtk2r = sqrt(k2mag)*(k2mag+k2r)/k2pk2magmag;
        AD sqrtk2i = sqrt(k2mag)*k2i/k2pk2magmag;
        
        //alpha = i*omega/c
        alpha[0][j] = -sqrtk2i; //exp(iwt) convention
        alpha[1][j] = sqrtk2r; //exp(iwt) convention
        //alpha[0][j] = sqrtk2i; //exp(-iwt) convention
        //alpha[1][j] = -sqrtk2r; //exp(-iwt) convention
      }
    }
  
    return alpha;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  int getNumResponses() {
    return numResponses;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  std::vector<string> extraFieldNames() const {
    std::vector<string> ef;
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  std::vector<FC > extraFields() const {
    std::vector<FC > ef;
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setExtraFields(const size_t & numElem_) {
    numElem = numElem_;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void updateParameters(const std::vector<std::vector<AD> > & params, const std::vector<string> & paramnames) {
  
    for (size_t p=0; p<paramnames.size(); p++) {
      if (paramnames[p] == "helmholtz_freq") 
        freq_params = params[p];
      else if (paramnames[p] == "helmholtz_source") 
        source_params = params[p];
      else if (paramnames[p] == "helmholtz_velocity") 
        vel_params = params[p];
      else if(verbosity > 0) //false alarms if multiple physics modules used...
        cout << "Parameter not used in msconvdiff: " << paramnames[p] << endl;  
    }
  }
  
  
private:
  
  Teuchos::RCP<userDefined> udfunc;
  
  std::vector<AD> freq_params, source_params, vel_params;
  
  int spaceDim, numElem, numResponses;
  vector<string> varlist;
  vector<int> ur_num, ui_num;
  double PI;
  bool isTD;
  int test;
  int verbosity;
  
  int numwavenum; //number of different wavenumbers
  
  double regParam; //regularization/control penalty parameter
  double cMY; //scaling for how strictly to enforce volume fraction limit
  bool pixelate; //whether optimizing field or actual pixels
  bool linInterp; //whether to use linear or cubic interpolation
  double binPenalty; //penalize non-binary-ness; only with pixelated version
};

#endif
