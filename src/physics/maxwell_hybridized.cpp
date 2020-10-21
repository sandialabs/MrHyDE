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

#include "maxwell_hybridized.hpp"
using namespace MrHyDE;

maxwell_HYBRID::maxwell_HYBRID(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "maxwell_hybrid";
  spaceDim = settings->sublist("Mesh").get<int>("dim",3);
  
  // GH Note: it's likely none of this will make sense in the 2D case... should it require 3D?
  myvars.push_back("Ex");
  mybasistypes.push_back("HGRAD-DG");
  if (spaceDim > 1) {
    myvars.push_back("Ey");
    mybasistypes.push_back("HGRAD-DG");
  }
  if (spaceDim > 2) {
    myvars.push_back("Ez");
    mybasistypes.push_back("HGRAD-DG");
  }

  myvars.push_back("Hx");
  mybasistypes.push_back("HGRAD-DG");
  if (spaceDim > 1) {
    myvars.push_back("Hy");
    mybasistypes.push_back("HGRAD-DG");
  }
  if (spaceDim > 2) {
    myvars.push_back("Hz");
    mybasistypes.push_back("HGRAD-DG");
  }

  myvars.push_back("lambdax");
  mybasistypes.push_back("HFACE");
  if (spaceDim > 1) {
    myvars.push_back("lambday");
    mybasistypes.push_back("HFACE");
  }
  if (spaceDim > 2) {
    myvars.push_back("lambdaz");
    mybasistypes.push_back("HFACE");
  }
}

// ========================================================================================
// ========================================================================================

void maxwell_HYBRID::defineFunctions(Teuchos::ParameterList & fs,
                                     Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("current x",fs.get<string>("current x","0.0"),"ip");
  functionManager->addFunction("current y",fs.get<string>("current y","0.0"),"ip");
  functionManager->addFunction("current z",fs.get<string>("current z","0.0"),"ip");
  functionManager->addFunction("mu",fs.get<string>("mu","1.0"),"ip");
  functionManager->addFunction("epsilon",fs.get<string>("epsilon","1.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void maxwell_HYBRID::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int Ex_basis_num = wkset->usebasis[Ex_num];
  int Hx_basis_num = wkset->usebasis[Hx_num];
  
  FDATA mu, epsilon;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    mu = functionManager->evaluate("mu","ip");
    epsilon = functionManager->evaluate("epsilon","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);


  // (\varepsilon \partial_t E_h, v)_{T_h} = \varepsilon (dEx_dt * vx + dEy_dt * vy + dEz_dt * vz)
  // using the basis for v as the same in each component
  auto basis = wkset->basis[Ex_basis_num];
  auto wts = wkset->wts;
  auto res = wkset->res;
  auto sol = wkset->local_soln;
  auto sol_dot = wkset->local_soln_dot;
  auto offsets = wkset->offsets;
  
  parallel_for("Maxwells hybrid E volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e) {

    ScalarT v = 0.0;

    for (int k=0; k<basis.extent(2); k++ ) {

      AD dEx_dt = sol_dot(e,Ex_num,k,0);
      AD dEy_dt, dEz_dt;

      if(spaceDim > 1) {
        AD dEy_dt = sol_dot(e,Ey_num,k,0);
      }

      if(spaceDim > 2) {
        AD dEz_dt = sol_dot(e,Ez_num,k,0);
      }

      for(int i=0; i<basis.extent(1); i++) {

        int resindex_x = offsets(Ex_num,i);
        int resindex_y = offsets(Ey_num,i);
        int resindex_z = offsets(Ez_num,i);
        v = basis(e,i,k);

        // using the basis for v as the same in each component
        res(e, resindex_x) += epsilon(e,k) * (dEx_dt * v);
        res(e, resindex_y) += epsilon(e,k) * (dEy_dt * v);
        res(e, resindex_z) += epsilon(e,k) * (dEz_dt * v);
      }
    }
  });


  // (\mu \partial_t H_h, v)_{T_h} = \mu (dHx_dt * vx + dHy_dt * vy + dHz_dt * vz)
  // using the basis for v as the same in each component
  basis = wkset->basis[Hx_basis_num];

  parallel_for("Maxwells hybrid H volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e) {

    ScalarT v = 0.0;

    for (int k=0; k<basis.extent(2); k++ ) {

      AD dHx_dt = sol_dot(e,Hx_num,k,0);
      AD dHy_dt, dHz_dt;

      if(spaceDim > 1) {
        AD dHy_dt = sol_dot(e,Hy_num,k,0);
      }

      if(spaceDim > 2) {
        AD dHz_dt = sol_dot(e,Hz_num,k,0);
      }

      for(int i=0; i<basis.extent(1); i++) {

        int resindex_x = offsets(Hx_num,i);
        int resindex_y = offsets(Hy_num,i);
        int resindex_z = offsets(Hz_num,i);
        v = basis(e,i,k);

        // using the basis for v as the same in each component
        res(e, resindex_x) += mu(e,k) * (dHx_dt * v);
        res(e, resindex_y) += mu(e,k) * (dHy_dt * v);
        res(e, resindex_z) += mu(e,k) * (dHz_dt * v);
      }
    }
  });


  // - (H_h, curl(v))_{T_h} = - (Hx * (dvz_dy - dvy_dz) + Hy * (dvx_dz - dvz_dx) + Hz * (dvy_dx - dvx_dy))
  // to use only a single basis, we will assemble each into different components by using indices in a smart way
  // Hy*dvdz - Hz*dvdy                                         into resindex for Hx
  //                   + Hz*dvdx - Hx*dvdz                     into resindex for Hy
  //                                       + Hx*dvdy - Hy*dvdx into resindex for Hz
  // this avoids needing dvx_dx, dvx_dy, dvx_dz, etc., which make it more complicated
  basis = wkset->basis[Hx_basis_num];
  auto basis_grad = wkset->basis_grad[Hx_basis_num];

  parallel_for("Maxwells hybrid extra volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e) {

    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0;

    for (int k=0; k<basis.extent(2); k++ ) {

      AD Hx = sol(e,Hx_num,k,0);
      AD Hy, Hz;

      if(spaceDim > 1) {
        AD Hy = sol(e,Hy_num,k,0);
      }

      if(spaceDim > 2) {
        AD Hz = sol(e,Hz_num,k,0);
      }

      for(int i=0; i<basis.extent(1); i++) {

        dvdx = basis_grad(e,i,k,0);

        if(spaceDim > 1) {
          dvdy = basis_grad(e,i,k,1);
        }

        if(spaceDim > 2) {
          dvdz = basis_grad(e,i,k,2);
        }

        int resindex_x = offsets(Hx_num,i);
        int resindex_y = offsets(Hy_num,i);
        int resindex_z = offsets(Hz_num,i);

        // using the basis for v as the same in each component
        res(e, resindex_x) -= (Hy*dvdz - Hz*dvdy);
        res(e, resindex_y) -= (Hz*dvdx - Hx*dvdz);
        res(e, resindex_z) -= (Hx*dvdy - Hy*dvdx);
      }
    }
  });


  // (curl(E_h), v)_{T_h} = - (vx * (dEz_dy - dEy_dz) + vy * (dEx_dz - dEz_dx) + vz * (dEy_dx - dEx_dy))
  basis = wkset->basis[Ex_basis_num];

  parallel_for("Maxwells hybrid E volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e) {

    ScalarT v = 0.0;

    for (int k=0; k<basis.extent(2); k++ ) {

      AD dEx_dx = sol_grad(e,Ex_num,k,0);
      AD dEy_dx = sol_grad(e,Ey_num,k,0);
      AD dEz_dx = sol_grad(e,Ez_num,k,0);

      AD dEx_dy, dEx_dz, dEy_dy, dEy_dz, dEz_dy, dEz_dz;

      if(spaceDim > 1) {
        dEx_dy = sol_grad(e,Ex_num,k,1);
        dEy_dy = sol_grad(e,Ey_num,k,1);
        dEz_dy = sol_grad(e,Ez_num,k,1);
      }

      if(spaceDim > 2) {
        dEx_dz = sol_grad(e,Ex_num,k,2);
        dEy_dz = sol_grad(e,Ey_num,k,2);
        dEz_dz = sol_grad(e,Ez_num,k,2);
      }

      for(int i=0; i<basis.extent(1); i++) {

        v = basis(e,i,k,0);

        int resindex_x = offsets(Ex_num,i);
        int resindex_y = offsets(Ey_num,i);
        int resindex_z = offsets(Ez_num,i);

        // using the basis for v as the same in each component
        res(e, resindex_x) += (v * (dEz_dy - dEy_dz));
        res(e, resindex_y) += (v * (dEx_dz - dEz_dx));
        res(e, resindex_z) += (v * (dEy_dx - dEx_dy));
      }
    }
  });
}


// ========================================================================================
// ========================================================================================

void maxwell_HYBRID::boundaryResidual() {

  Kokkos::View<int**,HostDevice> bcs = wkset->var_bcs;

  int cside = wkset->currentside;
  int sidetype = bcs(lambdax_num,cside);

  int lambdax_basis = wkset->usebasis[lambdax_num];
  auto basis = wkset->basis_side[lambdax_basis];

  FDATA bsourcex, bsourcey, bsourcez, current_x, current_y, current_z;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);

    if (sidetype == 1 ) {
      bsourcex = functionManager->evaluate("Dirichlet lambdax " + wkset->sidename,"side ip");
      bsourcey = functionManager->evaluate("Dirichlet lambday " + wkset->sidename,"side ip");
      bsourcez = functionManager->evaluate("Dirichlet lambdaz " + wkset->sidename,"side ip");
    }

    current_x = functionManager->evaluate("current x","ip");
    current_y = functionManager->evaluate("current y","ip");
    current_z = functionManager->evaluate("current z","ip");
  }

  // Since normals get recomputed often, this needs to be reset
  auto normals = wkset->normals;
  auto res = wkset->res;
  auto offsets = wkset->offsets;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);

  // - (\lambda_h, \eta)_{\Gamma_a} = - (lambdax * etax + lambday * etay + lambdaz * etaz)
  ScalarT eta = 0.0;
  ScalarT nx = 0.0, ny = 0.0, nz = 0.0;
  for (int e=0; e<basis.extent(0); e++) {
    if (bcs(lambdax_num,cside) == 1) {
      for (int k=0; k<basis.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          eta = basis(e,i,k,0);
          nx = normals(e,k,0);

          if (spaceDim>1) {
            ny = normals(e,k,1);
          }

          if (spaceDim>2) {
            nz = normals(e,k,2);
          }

          int resindex_x = offsets(lambdax_num,i);
          int resindex_y = offsets(lambday_num,i);
          int resindex_z = offsets(lambdaz_num,i);

          res(e,resindex_x) -= bsourcex(e,k)*eta;
          res(e,resindex_y) -= bsourcey(e,k)*eta;
          res(e,resindex_z) -= bsourcez(e,k)*eta;
        }
      }
    }
  }
  // - (g^{inc}, \eta)_{\Gamma_a} = - (current_x * etax + current_y * etay + current_z * etaz)


}

// ========================================================================================
// The edge (2D) and face (3D) contributions to the residual
// ========================================================================================

void maxwell_HYBRID::faceResidual() {

  int lambdax_basis = wkset->usebasis[lambdax_num];
  int Ex_basis = wkset->usebasis[Ex_num];

  // Since normals get recomputed often, this needs to be reset
  auto normals = wkset->normals;
  auto res = wkset->res;
  
  FDATA mu, epsilon;
  {
    // It should still be possible to evaluate and use these, right?
    // TODO: add a timer for this
    mu = functionManager->evaluate("mu","ip");
    epsilon = functionManager->evaluate("epsilon","ip");
  }

  Teuchos::TimeMonitor localtime(*boundaryResidualFill);

  ScalarT vx = 0.0, vy = 0.0, vz = 0.0;
  ScalarT nx = 0.0, ny = 0.0, nz = 0.0;

  // (lambda_h, n x v)_{\partial T_h} = lambdax*(ny*vz - nz*vy) + lambday*(nz*vx - nx*vz) + lambdaz*(nx*vy - vx*ny)
  auto basis = wkset->basis_face[Ex_basis];

  AD lambdax, lambday, lambdaz;
  for (int e=0; e<basis.extent(0); e++) {
    for (int k=0; k<basis.extent(2); k++ ) {
      for (int i=0; i<basis.extent(1); i++ ) {
        lambdax = sol_face(e,lambdax_num,k,0);

        vx = basis(e,i,k,0);
        nx = normals(e,k,0);
        if (spaceDim>1) {
          vy = basis(e,i,k,1);
          ny = normals(e,k,1);
          lambday = sol_face(e,lambday_num,k,0);
        }
        if (spaceDim>2) {
          vz = basis(e,i,k,2);
          nz = normals(e,k,2);
          lambdaz = sol_face(e,lambdaz_num,k,0);
        }

        int resindex_x = offsets(lambdax_num,i);
        int resindex_y = offsets(lambday_num,i);
        int resindex_z = offsets(lambdaz_num,i);

        res(e,resindex_x) += lambdax*(ny*vz - nz*vy);
        res(e,resindex_y) += lambday*(nz*vx - nx*vz);
        res(e,resindex_z) += lambdaz*(nx*vy - vx*ny);
      }
    }
  }

  // - (\tau n x (H_h - lambda_h), n x v)_{\partial T_h}
  // = - \tau ( n x (Hdiff), n x v)_{\partial T_h}
  // = - \tau ( (Hdiff_y*nz - Hdiff_z*ny)*(vy*nz - vz*ny)
  //                     + (Hdiffx_*nz - Hdiff_z*nx)*(vx*nz - vz*nx)
  //                             + (Hdiff_y*nx - Hdiff_x*ny)*(vy*nx - vx*ny))
  AD Hx, Hy, Hz;
  AD Hdiff_x, Hdiff_y, Hdiff_z;
  AD tau;
  for (int e=0; e<basis.extent(0); e++) {
    for (int k=0; k<basis.extent(2); k++ ) {
      for (int i=0; i<basis.extent(1); i++ ) {
        tau = sqrt(mu(e,k)/epsilon(e,k));

        lambdax = sol_face(e,lambdax_num,k,0);
        Hx = sol_face(e, Hx_num, k, 0);
        Hdiff_x = Hx - lambdax;

        vx = basis(e,i,k,0);
        nx = normals(e,k,0);
        if (spaceDim>1) {
          vy = basis(e,i,k,1);
          ny = normals(e,k,1);
          lambday = sol_face(e,lambday_num,k,0);
          Hy = sol_face(e, Hy_num, k, 0);
          Hdiff_y = Hy - lambday;
        }
        if (spaceDim>2) {
          vz = basis(e,i,k,2);
          nz = normals(e,k,2);
          lambdaz = sol_face(e,lambdaz_num,k,0);
          Hz = sol_face(e, Hz_num, k, 0);
          Hdiff_z = Hz - lambdaz;
        }

        int resindex_x = offsets(lambdax_num,i);
        int resindex_y = offsets(lambday_num,i);
        int resindex_z = offsets(lambdaz_num,i);

        res(e,resindex_x) -= tau * (Hdiff_y*nz - Hdiff_z*ny)*(vy*nz - vz*ny);
        res(e,resindex_y) -= tau * (Hdiff_x*nz - Hdiff_z*nx)*(vx*nz - vz*nx);
        res(e,resindex_z) -= tau * (Hdiff_y*nx - Hdiff_x*ny)*(vy*nx - vx*ny);
      }
    }
  }


  // (n x E_h, eta)_{\partial T_h} = (Ez*ny-Ey*nz)*etax + (Ex*nz-Ez*nx)*etay + (Ey*nx-Ex*ny)*etaz
  // assemble into different indices instead of using multiple bases for eta
  AD Ex = 0.0, Ey = 0.0, Ez = 0.0;
  basis = wkset->basis_face[lambdax_basis];
  ScalarT eta;
  for (int e=0; e<basis.extent(0); e++) {
    for (int k=0; k<basis.extent(2); k++ ) {
      for (int i=0; i<basis.extent(1); i++ ) {
        Ex = sol_face(e,Ex_num,k,0);
        nx = normals(e,k,0);

        if (spaceDim>1) {
          Ey = sol_face(e,Ey_num,k,1);
          ny = normals(e,k,1);
        }
        if (spaceDim>2) {
          Ez = sol_face(e,Ez_num,k,2);
          nz = normals(e,k,2);
        }
        eta = basis(e,i,k);

        int resindex_x = offsets(lambdax_num,i);
        int resindex_y = offsets(lambday_num,i);
        int resindex_z = offsets(lambdaz_num,i);

        res(e,resindex_x) += (Ez*ny - Ey*nz)*eta;
        res(e,resindex_y) += (Ex*nz - Ez*nx)*eta;
        res(e,resindex_z) += (Ey*nx - Ex*ny)*eta;
      }
    }
  }

  // (\tau(H_h - lambda_h), eta)_{\partial T_h}
  // = \tau (Hdiff_x*etax + Hdiff_y*etay + Hdiff_z*etaz)
  for (int e=0; e<basis.extent(0); e++) {
    for (int k=0; k<basis.extent(2); k++ ) {
      for (int i=0; i<basis.extent(1); i++ ) {
        tau = sqrt(mu(e,k)/epsilon(e,k));

        lambdax = sol_face(e,lambdax_num,k,0);
        Hx = sol_face(e, Hx_num, k, 0);
        Hdiff_x = Hx - lambdax;

        if (spaceDim>1) {
          lambday = sol_face(e,lambday_num,k,0);
          Hy = sol_face(e, Hy_num, k, 0);
          Hdiff_y = Hy - lambday;
        }
        if (spaceDim>2) {
          lambdaz = sol_face(e,lambdaz_num,k,0);
          Hz = sol_face(e, Hz_num, k, 0);
          Hdiff_z = Hz - lambdaz;
        }

        eta = basis(e,i,k);

        int resindex_x = offsets(lambdax_num,i);
        int resindex_y = offsets(lambday_num,i);
        int resindex_z = offsets(lambdaz_num,i);

        res(e,resindex_x) += tau*Hdiff_x*eta;
        res(e,resindex_y) += tau*Hdiff_y*eta;
        res(e,resindex_z) += tau*Hdiff_z*eta;
      }
    }
  }

}
// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void maxwell_HYBRID::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void maxwell_HYBRID::setVars(std::vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "Ex")
      Ex_num = i;
    if (varlist[i] == "Ey")
      Ey_num = i;
    if (varlist[i] == "Ez")
      Ez_num = i;
    if (varlist[i] == "Hx")
      Hx_num = i;
    if (varlist[i] == "Hy")
      Hx_num = i;
    if (varlist[i] == "Hz")
      Hx_num = i;
    if (varlist[i] == "lambdax")
      lambdax_num = i;
    if (varlist[i] == "lambday")
      lambday_num = i;
    if (varlist[i] == "lambdaz")
      lambdaz_num = i;
  }
}
