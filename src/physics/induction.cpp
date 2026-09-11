/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "induction.hpp"
using namespace MrHyDE;

// TODO BWR -- rho is both on the convective part but we have nu and rho*source showing up too
// this is inconsistent and needs fixing!

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
induction<EvalT>::induction(Teuchos::ParameterList & settings, const int & dimension_)
: PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "induction";
  int spaceDim = dimension_;
  
  if (spaceDim < 3) {
    // throw an error -- just 3D for now
    // 2D will be faked using periodic BCs and one element in z as in Shadid paper
  }
  
  
  myvars.push_back("Bx");
  myvars.push_back("By");
  myvars.push_back("Bz");
  myvars.push_back("psi"); // Lagrange multiplier for the involution contraint
  
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  use_stabilization = settings.get<bool>("use stabilization",true);
  include_resistive = settings.get<bool>("include resistive",false);
  include_Hall = settings.get<bool>("include Hall",false);
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void induction<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                 Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("eta",fs.get<string>("resistivity","1.0"),"ip");
  functionManager->addFunction("mu0",fs.get<string>("magnetic permeability","1.0"),"ip");
  functionManager->addFunction("ndens",fs.get<string>("number density","1.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void induction<EvalT>::volumeResidual() {
  
  
  
  //ScalarT dt = wkset->deltat;
  //bool isTransient = wkset->isTransient;
  Vista<EvalT> eta, mu0, ndens;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    eta = functionManager->evaluate("eta","ip");
    mu0 = functionManager->evaluate("mu0","ip");
    ndens = functionManager->evaluate("ndens","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  auto ux = wkset->getSolutionField("ux");
  auto uy = wkset->getSolutionField("uy");
  auto uz = wkset->getSolutionField("uz");
  auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
  auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
  auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
  auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
  auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
  auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
  auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
  auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
  auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
  
  auto press = wkset->getSolutionField("pr");
  
  auto Bx = wkset->getSolutionField("Bx");
  auto By = wkset->getSolutionField("By");
  auto Bz = wkset->getSolutionField("Bz");
  
  auto dBx_dx = wkset->getSolutionField("grad(Bx)[x]");
  auto dBx_dy = wkset->getSolutionField("grad(Bx)[y]");
  auto dBx_dz = wkset->getSolutionField("grad(Bx)[z]");
  auto dBy_dx = wkset->getSolutionField("grad(By)[x]");
  auto dBy_dy = wkset->getSolutionField("grad(By)[y]");
  auto dBy_dz = wkset->getSolutionField("grad(By)[z]");
  auto dBz_dx = wkset->getSolutionField("grad(Bz)[x]");
  auto dBz_dy = wkset->getSolutionField("grad(Bz)[y]");
  auto dBz_dz = wkset->getSolutionField("grad(Bz)[z]");
  auto dBx_dt = wkset->getSolutionField("Bx_t");
  auto dBy_dt = wkset->getSolutionField("By_t");
  auto dBz_dt = wkset->getSolutionField("Bz_t");
  
  auto psi = wkset->getSolutionField("psi");
  auto dpsi_dx = wkset->getSolutionField("grad(psi)[x]");
  auto dpsi_dy = wkset->getSolutionField("grad(psi)[y]");
  auto dpsi_dz = wkset->getSolutionField("grad(psi)[z]");
  
  bool use_stab = use_stabilization; // seems redundant but necessary to capture properly in lambdas
  auto hsize = wkset->getElementSize();
  ScalarT dt = wkset->deltat;
  
  
  // Magnetic field equation
  // dB/dt + \nabla \cdot (u ox B - B ox u + \psi I) = 0
  {
    // Let's assume all components use the same basis
    int basis_num = wkset->usebasis[Bx_num];
    auto basis = wkset->basis[basis_num];
    auto basis_grad = wkset->basis_grad[basis_num];
    auto off_Bx = subview(wkset->offsets,Bx_num,ALL());
    auto off_By = subview(wkset->offsets,By_num,ALL());
    auto off_Bz = subview(wkset->offsets,Bz_num,ALL());
    auto off_psi = subview(wkset->offsets,psi_num,ALL());
    
    parallel_for("MHD Bx volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        
        { // Bx equation
          EvalT Fx = 0.0;
          EvalT Fy = (uy(elem,pt)*Bx(elem,pt) - ux(elem,pt)*By(elem,pt))*wts(elem,pt);
          EvalT Fz = (uz(elem,pt)*Bx(elem,pt) - ux(elem,pt)*Bz(elem,pt))*wts(elem,pt);
          EvalT F = (dBx_dt(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off_Bx(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
        { // By equation
          EvalT Fx = (ux(elem,pt)*By(elem,pt) - uy(elem,pt)*Bx(elem,pt))*wts(elem,pt);
          EvalT Fy = 0.0;
          EvalT Fz = (uz(elem,pt)*By(elem,pt) - uy(elem,pt)*Bz(elem,pt))*wts(elem,pt);
          EvalT F = (dBy_dt(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off_By(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
        { // Bz equation
          EvalT Fx = (ux(elem,pt)*Bz(elem,pt) - uz(elem,pt)*Bx(elem,pt))*wts(elem,pt);
          EvalT Fy = (uy(elem,pt)*Bz(elem,pt) - uz(elem,pt)*By(elem,pt))*wts(elem,pt);
          EvalT Fz = 0.0;
          EvalT F = (dBz_dt(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off_Bz(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
        { // psi equation
          EvalT Fx = -1.0*(psi(elem,pt))*wts(elem,pt);
          EvalT Fy = -1.0*(psi(elem,pt))*wts(elem,pt);
          EvalT Fz = -1.0*(psi(elem,pt))*wts(elem,pt);
          EvalT F = 0.0;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off_psi(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
        
        if (include_resistive) {
          { // Bx equation
            EvalT Fx = eta(elem,pt)/mu0(elem,pt)*dBx_dx(elem,pt)*wts(elem,pt);
            EvalT Fy = eta(elem,pt)/mu0(elem,pt)*dBx_dy(elem,pt)*wts(elem,pt);
            EvalT Fz = eta(elem,pt)/mu0(elem,pt)*dBx_dz(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++) {
              res(elem,off_Bx(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
            }
          }
          { // By equation
            EvalT Fx = eta(elem,pt)/mu0(elem,pt)*dBy_dx(elem,pt)*wts(elem,pt);
            EvalT Fy = eta(elem,pt)/mu0(elem,pt)*dBy_dy(elem,pt)*wts(elem,pt);
            EvalT Fz = eta(elem,pt)/mu0(elem,pt)*dBy_dz(elem,pt)*wts(elem,pt);
            
            for (size_type dof=0; dof<basis.extent(1); dof++) {
              res(elem,off_By(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
            }
          }
          { // Bz equation
            EvalT Fx = eta(elem,pt)/mu0(elem,pt)*dBz_dx(elem,pt)*wts(elem,pt);
            EvalT Fy = eta(elem,pt)/mu0(elem,pt)*dBz_dy(elem,pt)*wts(elem,pt);
            EvalT Fz = eta(elem,pt)/mu0(elem,pt)*dBz_dz(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++) {
              res(elem,off_Bz(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
            }
          }
          
        }
        
        if (include_Hall) {
          EvalT Jx = 1/mu0(elem,pt)*(dBz_dy(elem,pt) - dBy_dz(elem,pt));
          EvalT Jy = 1/mu0(elem,pt)*(dBx_dz(elem,pt) - dBz_dx(elem,pt));
          EvalT Jz = 1/mu0(elem,pt)*(dBy_dx(elem,pt) - dBx_dy(elem,pt));
          EvalT Hx = 1/ndens(elem,pt)*(Jy*Bz(elem,pt) - Jz*By(elem,pt));
          EvalT Hy = 1/ndens(elem,pt)*(Jz*Bx(elem,pt) - Jx*Bz(elem,pt));
          EvalT Hz = 1/ndens(elem,pt)*(Jx*By(elem,pt) - Jy*Bx(elem,pt));
          
          { // Bx equation
            EvalT Fx = 0.0;
            EvalT Fy = Hz*wts(elem,pt);
            EvalT Fz = -Hy*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++) {
              res(elem,off_Bx(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
            }
          }
          { // By equation
            EvalT Fx = -Hz*wts(elem,pt);
            EvalT Fy = 0.0;
            EvalT Fz = Hx*wts(elem,pt);
            
            for (size_type dof=0; dof<basis.extent(1); dof++) {
              res(elem,off_By(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
            }
          }
          { // Bz equation
            EvalT Fx = Hy*wts(elem,pt);
            EvalT Fy = -Hz*wts(elem,pt);
            EvalT Fz = 0.0;
            for (size_type dof=0; dof<basis.extent(1); dof++) {
              res(elem,off_Bz(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
            }
          }
        }
        
        bool localflag = true;
        if (use_stab && localflag) {
          /*
          EvalT tauB = this->computeTauB(ux(elem,pt), uy(elem,pt), uz(elem,pt), Bx(elem,pt), By(elem,pt), Bz(elem,pt), mu0(elem,pt),
                                         1.0, hsize(elem), dt);
          EvalT taupsi = this->computeTauPsi(tauB, hsize(elem));
          
          EvalT sBxres = this->computeStrongResidualBx(dBx_dt(elem,pt), Bx(elem,pt), dBx_dx(elem,pt),
                                                       dBx_dy(elem,pt), dBx_dz(elem,pt), By(elem,pt), Bz(elem,pt),
                                                       dBy_dy(elem,pt), dBz_dz(elem,pt), ux(elem,pt), uy(elem,pt), uz(elem,pt),
                                                       dux_dx(elem,pt), dux_dy(elem,pt), dux_dz(elem,pt), duy_dy(elem,pt), duz_dz(elem,pt),
                                                       1.0, dpsi_dx(elem,pt));
          EvalT sByres = this->computeStrongResidualBy(dBy_dt(elem,pt), By(elem,pt), dBy_dx(elem,pt),
                                                       dBy_dy(elem,pt), dBy_dz(elem,pt),
                                                       Bx(elem,pt), Bz(elem,pt),
                                                       dBx_dx(elem,pt), dBz_dz(elem,pt), ux(elem,pt), uy(elem,pt), uz(elem,pt),
                                                       dux_dx(elem,pt), duy_dx(elem,pt), duy_dy(elem,pt), duy_dz(elem,pt), duz_dz(elem,pt),
                                                       1.0, dpsi_dy(elem,pt));
          EvalT sBzres = this->computeStrongResidualBz(dBz_dt(elem,pt), Bz(elem,pt), dBz_dx(elem,pt),
                                                       dBz_dy(elem,pt), dBz_dz(elem,pt),
                                                       Bx(elem,pt), By(elem,pt),
                                                       dBx_dx(elem,pt), dBy_dy(elem,pt), ux(elem,pt), uy(elem,pt), uz(elem,pt),
                                                       dux_dx(elem,pt), duy_dy(elem,pt), duz_dx(elem,pt), duz_dy(elem,pt), duz_dz(elem,pt),
                                                       1.0, dpsi_dz(elem,pt));
          EvalT spsires = this->computeStrongResidualPsi(dBx_dx(elem,pt), dBy_dy(elem,pt), dBz_dz(elem,pt));
          
          EvalT Bx_prime = tauB*sBxres*wts(elem,pt);
          EvalT By_prime = tauB*sByres*wts(elem,pt);
          EvalT Bz_prime = tauB*sBzres*wts(elem,pt);
          EvalT psi_prime = taupsi*spsires*wts(elem,pt);
          {
            //EvalT s_x = ux(elem,pt)*Bx_prime - Bx_prime*ux(elem,pt) + psi_prime;
            //EvalT s_y = ux(elem,pt)*By_prime - Bx_prime*uy(elem,pt);
            //EvalT s_z = ux(elem,pt)*Bz_prime - Bx_prime*uz(elem,pt);
            //for (size_type dof=0; dof<basis.extent(1); dof++) {
              //res(elem,off_Bx(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
            //}
          }
          {
            //EvalT s_x = ux(elem,pt)*Bx_prime - Bx_prime*ux(elem,pt);
            //EvalT s_y = ux(elem,pt)*By_prime - Bx_prime*uy(elem,pt) + psi_prime;
            //EvalT s_z = ux(elem,pt)*Bz_prime - Bx_prime*uz(elem,pt);
            //for (size_type dof=0; dof<basis.extent(1); dof++) {
              //res(elem,off_By(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
            //}
          }
          {
            //EvalT s_x = ux(elem,pt)*Bx_prime - Bx_prime*ux(elem,pt);
            //EvalT s_y = ux(elem,pt)*By_prime - Bx_prime*uy(elem,pt);
            //EvalT s_z = ux(elem,pt)*Bz_prime - Bx_prime*uz(elem,pt) + psi_prime;
            //for (size_type dof=0; dof<basis.extent(1); dof++) {
              //res(elem,off_Bz(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
            //}
          }
          */
          {
            EvalT s_x = 10.0*hsize(elem)*dpsi_dx(elem,pt);//Bx_prime;
            EvalT s_y = 10.0*hsize(elem)*dpsi_dy(elem,pt);//By_prime;
            EvalT s_z = 10.0*hsize(elem)*dpsi_dz(elem,pt);//Bz_prime;
            for (size_type dof=0; dof<basis.extent(1); dof++) {
              res(elem,off_psi(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
            }
          }
        }
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void induction<EvalT>::boundaryResidual() {
  
  /*
   int spaceDim = wkset->dimension;
   auto bcs = wkset->var_bcs;
   
   int cside = wkset->currentside;
   
   string ux_sidetype = bcs(ux_num,cside);
   string uy_sidetype = "Dirichlet";
   string uz_sidetype = "Dirichlet";
   if (spaceDim > 1) {
   uy_sidetype = bcs(uy_num,cside);
   }
   if (spaceDim > 2) {
   uz_sidetype = bcs(uz_num,cside);
   }
   
   Vista<EvalT> source_ux, source_uy, source_uz;
   
   if (ux_sidetype != "Dirichlet" || uy_sidetype != "Dirichlet" || uz_sidetype != "Dirichlet") {
   
   {
   //Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
   if (ux_sidetype == "Neumann") {
   source_ux = functionManager->evaluate("Neumann ux " + wkset->sidename,"side ip");
   }
   if (uy_sidetype == "Neumann") {
   source_uy = functionManager->evaluate("Neumann uy " + wkset->sidename,"side ip");
   }
   if (uz_sidetype == "Neumann") {
   source_uz = functionManager->evaluate("Neumann uz " + wkset->sidename,"side ip");
   }
   }
   
   // Since normals get recomputed often, this needs to be reset
   auto wts = wkset->wts_side;
   auto h = wkset->getSideElementSize();
   auto res = wkset->res;
   
   //Teuchos::TimeMonitor localtime(*boundaryResidualFill);
   
   if (spaceDim == 1) {
   int ux_basis = wkset->usebasis[ux_num];
   auto basis = wkset->basis_side[ux_basis];
   auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
   if (ux_sidetype == "Neumann") { // Neumann
   parallel_for("NS ux bndry resid 1D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   else if (spaceDim == 2) {
   
   // ux equation boundary residual
   {
   int ux_basis = wkset->usebasis[ux_num];
   auto basis = wkset->basis_side[ux_basis];
   auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
   
   if (ux_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS ux bndry resid 2D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   
   // uy equation boundary residual
   {
   int uy_basis = wkset->usebasis[uy_num];
   auto basis = wkset->basis_side[uy_basis];
   auto off = Kokkos::subview( wkset->offsets, uy_num, Kokkos::ALL());
   if (uy_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS uy bndry resid 2D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_uy(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   }
   
   else if (spaceDim == 3) {
   
   // ux equation boundary residual
   {
   int ux_basis = wkset->usebasis[ux_num];
   auto basis = wkset->basis_side[ux_basis];
   auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
   if (ux_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS ux bndry resid 3D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   
   // uy equation boundary residual
   {
   int uy_basis = wkset->usebasis[uy_num];
   auto basis = wkset->basis_side[uy_basis];
   auto off = Kokkos::subview( wkset->offsets, uy_num, Kokkos::ALL());
   if (uy_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS uy bndry resid 3D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_uy(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   
   // uz equation boundary residual
   {
   int uz_basis = wkset->usebasis[uz_num];
   auto basis = wkset->basis_side[uz_basis];
   auto off = Kokkos::subview( wkset->offsets, uz_num, Kokkos::ALL());
   if (uz_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS uz bndry resid 3D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_uz(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   }
   }
   */
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void induction<EvalT>::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void induction<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  
  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  psi_num = -1;
  Bx_num = -1;
  By_num = -1;
  Bz_num = -1;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "psi")
      psi_num = i;
    if (varlist[i] == "Bx")
      Bx_num = i;
    if (varlist[i] == "By")
      By_num = i;
    if (varlist[i] == "Bz")
      Bz_num = i;
    
  }
  
}



template<class EvalT>
KOKKOS_FUNCTION EvalT induction<EvalT>::computeTauB(const EvalT & xvl, const EvalT & yvl, const EvalT & zvl,
                                              const EvalT & Bx, const EvalT & By, const EvalT & Bz,
                                              const EvalT & mu0, const EvalT & ndens,
                                              const ScalarT & h, const ScalarT & dt) const {
  ScalarT C1 = 4.0;
  ScalarT C2 = 2.0;
  ScalarT C3 = 0.0;
  ScalarT C4 = 0.0;
  
  EvalT nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  EvalT nB = Bx*Bx + By*By + Bz*Bz;
  if (nvel > 1E-12) {
    nvel = sqrt(nvel);
  }
  if (nB > 1E-12) {
    nB = sqrt(nB);
  }
  
  EvalT tau = C1*(2.0/dt)*(2.0/dt) + C2*(nvel/h)*(nvel/h) + C3*(nB/h)*(nB/h) + C4*ndens*ndens/mu0/mu0*h*h;
  tau = 1./sqrt(tau);
  
  return tau;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT induction<EvalT>::computeTauPsi(const EvalT & tauB, const ScalarT & h) const {
  return 1.0/tauB;
}


template<class EvalT>
KOKKOS_FUNCTION EvalT induction<EvalT>::computeStrongResidualBx(const EvalT & dBx_dt, const EvalT & Bx, const EvalT & dBx_dx,
                                                          const EvalT & dBx_dy, const EvalT & dBx_dz,
                                                          const EvalT & By, const EvalT & Bz,
                                                          const EvalT & dBy_dy, const EvalT & dBz_dz,
                                                          const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                          const EvalT & dux_dx, const EvalT & dux_dy, const EvalT & dux_dz,
                                                          const EvalT & duy_dy, const EvalT & duz_dz,
                                                          const EvalT & S, const EvalT & dpsi_dx) {
  EvalT sres = dBx_dt + (duy_dy*Bx + uy*dBy_dy) - (dux_dy*By + ux*dBy_dy) + (duz_dz*Bx + uz*dBx_dz) - (dux_dz*Bz + ux*dBz_dz) + dpsi_dx;
  
  if (include_resistive) {
    // do nothing - h.o.t.
  }
  if (include_Hall) {
    // probably need to add something
  }
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT induction<EvalT>::computeStrongResidualBy(const EvalT & dBy_dt, const EvalT & By, const EvalT & dBy_dx,
                                                          const EvalT & dBy_dy, const EvalT & dBy_dz,
                                                          const EvalT & Bx, const EvalT & Bz,
                                                          const EvalT & dBx_dx, const EvalT & dBz_dz,
                                                          const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                          const EvalT & dux_dx,
                                                          const EvalT & duy_dx, const EvalT & duy_dy, const EvalT & duy_dz,
                                                          const EvalT & duz_dz,
                                                          const EvalT & S, const EvalT & dpsi_dy) {
  EvalT sres = dBy_dt + (dux_dx*By + ux*dBy_dx) - (duy_dx*Bx + uy*dBx_dx) + (duz_dz*By + uz*dBy_dz) - (duy_dz*Bz + uy*dBz_dz) + dpsi_dy;
  
  if (include_resistive) {
    // do nothing - h.o.t.
  }
  if (include_Hall) {
    // probably need to add something
  }
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT induction<EvalT>::computeStrongResidualBz(const EvalT & dBz_dt, const EvalT & Bz, const EvalT & dBz_dx,
                                                          const EvalT & dBz_dy, const EvalT & dBz_dz,
                                                          const EvalT & Bx, const EvalT & By,
                                                          const EvalT & dBx_dx, const EvalT & dBy_dy,
                                                          const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                          const EvalT & dux_dx, const EvalT & duy_dy,
                                                          const EvalT & duz_dx, const EvalT & duz_dy, const EvalT & duz_dz,
                                                          const EvalT & S, const EvalT & dpsi_dz) {
  EvalT sres = dBz_dt + (dux_dx*Bz + ux*dBz_dx) - (duz_dx*Bx + uz*dBx_dx) + (duy_dy*Bz + uy*dBz_dy) - (duz_dy*By + uz*dBy_dy) + dpsi_dz;
  
  if (include_resistive) {
    // do nothing - h.o.t.
  }
  if (include_Hall) {
    // probably need to add something
  }
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT induction<EvalT>::computeStrongResidualPsi(const EvalT & dBx_dx, const EvalT & dBy_dy, const EvalT & dBz_dz) {
  EvalT sres = dBx_dx + dBy_dy + dBz_dz;
  return sres;
}
//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::induction<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::induction<AD>;

// Standard built-in types
template class MrHyDE::induction<AD2>;
template class MrHyDE::induction<AD4>;
template class MrHyDE::induction<AD8>;
template class MrHyDE::induction<AD16>;
template class MrHyDE::induction<AD18>;
template class MrHyDE::induction<AD24>;
template class MrHyDE::induction<AD32>;
#endif
