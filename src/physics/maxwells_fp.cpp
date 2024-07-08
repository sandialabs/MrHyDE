/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "maxwells_fp.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
maxwells_fp<EvalT>::maxwells_fp(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  //potential approach to frequency-domain Maxwell's (see Boyse et al (1992)); uses -iwt convention
  int spaceDim = dimension_;
  if (spaceDim < 2)
    cout << "Not all aspects may be well-defined in 1D..." << endl;
  
  myvars.push_back("Arx");
  myvars.push_back("Aix");
  myvars.push_back("phir");
  myvars.push_back("phii");
  if (spaceDim > 1) {
    myvars.push_back("Ary");
    myvars.push_back("Aiy");
  }
  if (spaceDim > 2) {
    myvars.push_back("Arz");
    myvars.push_back("Aiz");
  }
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  if (spaceDim > 1) {
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
  }
  if (spaceDim > 2) {
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
  }
  
  essScale = settings.get<ScalarT>("weak ess BC scaling",100.0);
  calcE = settings.get<bool>("Calculate electric field",false);
  
  test = settings.get<int>("test",0);
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwells_fp<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                  Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwells_fp<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int resindex;
  int phir_basis_num = wkset->usebasis[phir_num];
  int phii_basis_num = wkset->usebasis[phii_num];
  
  ScalarT x = 0.0;
  ScalarT y = 0.0;
  ScalarT z = 0.0;
  
  //test functions
  ScalarT vr = 0.0, dvrdx = 0.0, dvrdy = 0.0, dvrdz = 0.0,
  vi = 0.0, dvidx = 0.0, dvidy = 0.0, dvidz = 0.0;
  
  //states and their gradients
  EvalT Axr = 0.0, dAxrdx = 0.0, dAxrdy = 0.0, dAxrdz = 0.0,
  Axi = 0.0, dAxidx = 0.0, dAxidy = 0.0, dAxidz = 0.0;
  EvalT Ayr = 0.0, dAyrdx = 0.0, dAyrdy = 0.0, dAyrdz = 0.0,
  Ayi = 0.0, dAyidx = 0.0, dAyidy = 0.0, dAyidz = 0.0;
  EvalT Azr = 0.0, dAzrdx = 0.0, dAzrdy = 0.0, dAzrdz = 0.0,
  Azi = 0.0, dAzidx = 0.0, dAzidy = 0.0, dAzidz = 0.0;
  EvalT phir = 0.0, dphirdx = 0.0, dphirdy = 0.0, dphirdz = 0.0,
  phii = 0.0, dphiidx = 0.0, dphiidy = 0.0, dphiidz = 0.0;
  EvalT Axrdot = 0.0, Axidot = 0.0, Ayrdot = 0.0, Ayidot = 0.0,
  Azrdot = 0.0, Azidot = 0.0, phirdot = 0.0, phiidot = 0.0;
  
  //parameters
  EvalT omega = 0.0;
  EvalT Jxr = 0.0, Jyr = 0.0, Jzr = 0.0,
  Jxi = 0.0, Jyi = 0.0, Jzi = 0.0;
  EvalT rhor = 0.0, mur = 0.0, invmur = 0.0, epsr = 0.0,
  rhoi = 0.0, mui = 0.0, invmui = 0.0, epsi = 0.0;
  
  //    for( size_t e=0; e<numCC; e++ ) {
  //      for( int i=0; i<numBasis; i++ ) {
  
  ScalarT current_time = wkset->time;
  
  auto phir_basis = wkset->basis[phir_basis_num];
  auto phir_basis_grad = wkset->basis_grad[phir_basis_num];
  auto phii_basis = wkset->basis[phii_basis_num];
  auto phii_basis_grad = wkset->basis_grad[phii_basis_num];
  
  auto res = wkset->res;
  auto offsets = wkset->offsets;
  
  View_EvalT2 Ax_r, Ax_i, phi_r, phi_i, Ay_r, Ay_i, Az_r, Az_i;
  Ax_r = wkset->getSolutionField("Arx");
  Ax_i = wkset->getSolutionField("Aix");
  phi_r = wkset->getSolutionField("phir");
  phi_i = wkset->getSolutionField("phii");
  
  View_EvalT2 dAxr_dt, dAxi_dt, dphir_dt, dphii_dt, dAyr_dt, dAyi_dt, dAzr_dt, dAzi_dt;
  dAxr_dt = wkset->getSolutionField("Arx_t");
  dAxi_dt = wkset->getSolutionField("Aix_t");
  dphir_dt = wkset->getSolutionField("phir_t");
  dphii_dt = wkset->getSolutionField("phii_t");
  
  View_EvalT2 dAxr_dx, dAxi_dx, dphir_dx, dphii_dx, dAyr_dx, dAyi_dx, dAzr_dx, dAzi_dx;
  dAxr_dx = wkset->getSolutionField("grad(Arx)[x]");
  dAxi_dx = wkset->getSolutionField("grad(Aix)[x]");
  dphir_dx = wkset->getSolutionField("grad(phir)[x]");
  dphii_dx = wkset->getSolutionField("grad(phii)[x]");
  
  View_EvalT2 dAxr_dy, dAxi_dy, dphir_dy, dphii_dy, dAyr_dy, dAyi_dy, dAzr_dy, dAzi_dy;
  View_EvalT2 dAxr_dz, dAxi_dz, dphir_dz, dphii_dz, dAyr_dz, dAyi_dz, dAzr_dz, dAzi_dz;
  
  if (spaceDim > 1) {
    Ay_r = wkset->getSolutionField("Ary");
    Ay_i = wkset->getSolutionField("Aiy");
    dAyr_dt = wkset->getSolutionField("Ary_t");
    dAyi_dt = wkset->getSolutionField("Aiy_t");
    dAxr_dy = wkset->getSolutionField("grad(Arx)[y]");
    dAxi_dy = wkset->getSolutionField("grad(Aix)[y]");
    dphir_dy = wkset->getSolutionField("grad(phir)[y]");
    dphii_dy = wkset->getSolutionField("grad(phii)[y]");
    dAyr_dx = wkset->getSolutionField("grad(Ary)[x]");
    dAyi_dx = wkset->getSolutionField("grad(Aiy)[x]");
    dAyr_dy = wkset->getSolutionField("grad(Ary)[y]");
    dAyi_dy = wkset->getSolutionField("grad(Aiy)[y]");
  }
  if (spaceDim > 2) {
    Az_r = wkset->getSolutionField("Arz");
    Az_i = wkset->getSolutionField("Aiz");
    dAzr_dt = wkset->getSolutionField("Arz_t");
    dAzi_dt = wkset->getSolutionField("Aiz_t");
    dAxr_dz = wkset->getSolutionField("grad(Arx)[z]");
    dAxi_dz = wkset->getSolutionField("grad(Aix)[z]");
    dAyr_dz = wkset->getSolutionField("grad(Ary)[z]");
    dAyi_dz = wkset->getSolutionField("grad(Aiy)[z]");
  
    dphir_dz = wkset->getSolutionField("grad(phir)[z]");
    dphii_dz = wkset->getSolutionField("grad(phii)[z]");
    dAzr_dx = wkset->getSolutionField("grad(Arz)[x]");
    dAzi_dx = wkset->getSolutionField("grad(Aiz)[x]");
    dAzr_dy = wkset->getSolutionField("grad(Arz)[y]");
    dAzi_dy = wkset->getSolutionField("grad(Aiz)[y]");
    dAzr_dz = wkset->getSolutionField("grad(Arz)[z]");
    dAzi_dz = wkset->getSolutionField("grad(Aiz)[z]");
  
  }
  
  View_Sc2 ip_x, ip_y, ip_z;
  ip_x = wkset->getScalarField("x");
  if (spaceDim > 1) {
    ip_y = wkset->getScalarField("y");
  }
  if (spaceDim > 2) {
    ip_z = wkset->getScalarField("z");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  for (size_type e=0; e<res.extent(0); e++) {
    for( size_type k=0; k<ip_x.extent(1); k++ ) {
      
      // gather up all the information at the integration point
      x = ip_x(e,k);
      
      Axr = Ax_r(e,k);
      Axrdot = dAxr_dt(e,k);
      dAxrdx = dAxr_dx(e,k);
      Axi = Ax_i(e,k);
      Axidot = dAxi_dt(e,k);
      dAxidx = dAxi_dx(e,k);
      
      phir = phi_r(e,k);
      phii = phi_i(e,k);
      
      phirdot = dphir_dt(e,k);
      phiidot = dphii_dt(e,k);
      dphirdx = dphir_dx(e,k);
      dphiidx = dphii_dx(e,k);
      
      if(spaceDim > 1){
        y = ip_y(e,k);
        dAxrdy = dAxr_dy(e,k);
        dAxidy = dAxi_dy(e,k);
        
        Ayr = Ay_r(e,k);
        Ayrdot = dAyr_dt(e,k);
        dAyrdx = dAyr_dx(e,k);
        dAyrdy = dAyr_dy(e,k);
        
        Ayi = Ay_i(e,k);
        Ayidot = dAyi_dt(e,k);
        dAyidx = dAyi_dx(e,k);
        dAyidy = dAyi_dy(e,k);
        
        dphirdy = dphir_dy(e,k);
        dphiidy = dphii_dy(e,k);
      }
      if(spaceDim > 2){
        z = ip_z(e,k);
        
        dAxrdz = dAxr_dz(e,k);
        dAxidz = dAxi_dz(e,k);
        
        dAyrdz = dAyr_dz(e,k);
        dAyidz = dAyi_dz(e,k);
        
        Azr = Az_r(e,k);
        Azrdot = dAzr_dt(e,k);
        dAzrdx = dAzr_dx(e,k);
        dAzrdy = dAzr_dy(e,k);
        dAzrdz = dAzr_dz(e,k);
        
        Azi = Az_i(e,k);
        Azidot = dAzi_dt(e,k);
        dAzidx = dAzi_dx(e,k);
        dAzidy = dAzi_dy(e,k);
        dAzidz = dAzi_dz(e,k);
        dphirdz = dphir_dz(e,k);
        dphiidz = dphii_dz(e,k);
      }
      
      for (size_type i=0; i<phir_basis.extent(1); i++ ) { // TMW: this will fail if using different basis for phir and phii
        //	v = wkset->basis[e_basis](0,i,k);
        vr = phir_basis(e,i,k,0);
        vi = phii_basis(e,i,k,0);
        
        //vr = basis(e,phir_num,i,k);
        //          vi = basis(e,phii_num,i,k);
        dvrdx = phir_basis_grad(e,i,k,0);
        dvidx = phii_basis_grad(e,i,k,0);
        
        omega = getFreq(x, y, z, current_time);
        vector<EvalT> permit = getPermittivity(x, y, z, current_time);
        epsr = permit[0]; epsi = permit[1];
        vector<EvalT> permea = getPermeability(x, y, z, current_time);
        mur = permea[0]; mui = permea[1];
        vector<EvalT> invperm = getInvPermeability(x, y, z, current_time);
        invmur = invperm[0]; invmui = invperm[1];
        vector<vector<EvalT> > source_current = getInteriorCurrent(x, y, z, current_time);
        Jxr = source_current[0][0];
        Jxi = source_current[1][0];
        if(spaceDim > 1){
          Jyr = source_current[0][1];
          Jyi = source_current[1][1];
          dvrdy = phir_basis_grad(e,i,k,1);
          dvidy = phii_basis_grad(e,i,k,1);
        }
        if(spaceDim > 2){
          Jzr = source_current[0][2];
          Jzi = source_current[1][2];
          dvrdz = phir_basis_grad(e,i,k,2);
          dvidz = phii_basis_grad(e,i,k,2);
        }
        
        vector<EvalT> source_charge = getInteriorCharge(x, y, z, current_time);
        rhor = source_charge[0]; rhoi = source_charge[1];
        
        // TMW: this will fail if running with other physics enabled
        res(e,0) += Axrdot*vr - Axidot*vi;
        res(e,1) += Axrdot*vi + Axidot*vr;
        res(e,2) += phirdot*vr - phiidot*vi;
        res(e,3) += phirdot*vi + phiidot*vr;
        if(spaceDim > 1){
          res(e,4) += Ayrdot*vr - Ayidot*vi;
          res(e,5) += Ayrdot*vi + Ayidot*vr;
        }
        if(spaceDim > 2){
          res(e,6) += Azrdot*vr - Azidot*vi;
          res(e,7) += Azrdot*vi + Azidot*vr;
        }
        
        resindex = offsets(Axr_num,i);
        res(e,resindex) += ( (  dvrdz*(dAxrdz - dAzrdx) - dvrdy*(dAyrdx - dAxrdy)
                              -(dvidz*(dAxidz - dAzidx) - dvidy*(dAyidx - dAxidy)))*invmur
                            - (  dvrdz*(dAxidz - dAzidx) - dvrdy*(dAyidx - dAxidy)
                               + dvidz*(dAxrdz - dAzrdx) - dvidy*(dAyrdx - dAxrdy) )*invmui)
        + ( (  dvrdx*(dAxrdx + dAyrdy + dAzrdz)
             - dvidx*(dAxidx + dAyidy + dAzidz) )*invmur
           - (  dvrdx*(dAxidx + dAyidy + dAzidz)
              + dvidx*(dAxrdx + dAyrdy + dAzrdz) )*invmui)
        - omega*omega*(epsr*vr*Axr - epsr*vi*Axi - epsi*vr*Axi - epsi*vi*Axr)
        + omega*(  epsi*(dphirdx*vr + phir*dvrdx)
                 + epsr*(dphiidx*vr + phii*dvrdx)
                 + epsr*(dphirdx*vi + phir*dvidx)
                 - epsi*(dphiidx*vi + phii*dvidx))
        - (vr*Jxr - vi*Jxi); //real
        resindex = offsets(Axi_num,i);
        res(e,resindex) += ( (  dvrdz*(dAxidz - dAzidx) - dvrdy*(dAyidx - dAxidy)
                              + dvidz*(dAxrdz - dAzrdx) - dvidy*(dAyrdx - dAxrdy))*invmur
                            + (  dvrdz*(dAxrdz - dAzrdx) - dvrdy*(dAyrdx - dAxrdy)
                               -(dvidz*(dAxidz - dAzidx) - dvidy*(dAyidx - dAxidy)))*invmui)
        + ( (  dvrdx*(dAxidx + dAyidy + dAzidz)
             + dvidx*(dAxrdx + dAyrdy + dAzrdz) )*invmur
           + (  dvrdx*(dAxrdx + dAyrdy + dAzrdz)
              - dvidx*(dAxidx + dAyidy + dAzidz) )*invmui)
        - omega*omega*(-epsi*vi*Axi + epsi*vr*Axr + epsr*vi*Axr + epsr*vr*Axi)
        - omega*(- epsr*(dphiidx*vi + phii*dvidx)
                 - epsi*(dphirdx*vi + phir*dvidx)
                 - epsi*(dphiidx*vr + phii*dvrdx)
                 + epsr*(dphirdx*vr + phir*dvrdx))
        - (vr*Jxi + vi*Jxr); //imaginary
        
        resindex = offsets(phir_num,i);
        res(e,resindex) += (  epsr*(dvrdx*dphirdx + dvrdy*dphirdy + dvrdz*dphirdz)
                            - epsr*(dvidx*dphiidx + dvidy*dphiidy + dvidz*dphiidz)
                            - epsi*(dvrdx*dphiidx + dvrdy*dphiidy + dvrdz*dphiidz)
                            - epsi*(dvidx*dphirdx + dvidy*dphirdy + dvidz*dphirdz))
        - omega*omega*( (epsr*epsr-epsi*epsi)*mur*vr*phir
                       - (2*epsr*epsi)*mui*vr*phir
                       - (2*epsr*epsi)*mur*vi*phir
                       - (2*epsr*epsi)*mur*vr*phii
                       - (epsr*epsr-epsi*epsi)*mui*vi*phir
                       - (epsr*epsr-epsi*epsi)*mui*vr*phii
                       - (epsr*epsr-epsi*epsi)*mur*vi*phii
                       + (2*epsr*epsi)*mui*vi*phii)
        + omega*(  epsi*(dvrdx*Axr + vr*dAxrdx + dvrdy*Ayr + vr*dAyrdy + dvrdz*Azr + vr*dAzrdz)
                 + epsr*(dvidx*Axr + vi*dAxrdx + dvidy*Ayr + vi*dAyrdy + dvidz*Azr + vi*dAzrdz)
                 + epsr*(dvrdx*Axi + vr*dAxidx + dvrdy*Ayi + vr*dAyidy + dvrdz*Azi + vr*dAzidz)
                 - epsi*(dvidx*Axi + vi*dAxidx + dvidy*Ayi + vi*dAyidy + dvidz*Azi + vi*dAzidz))
        - (vr*rhor - vi*rhoi); //real
        
        resindex = offsets(phii_num,i);
        res(e,resindex) += (- epsi*(dvidx*dphiidx + dvidy*dphiidy + dvidz*dphiidz)
                            + epsi*(dvrdx*dphirdx + dvrdy*dphirdy + dvrdz*dphirdz)
                            + epsr*(dvidx*dphirdx + dvidy*dphirdy + dvidz*dphirdz)
                            + epsr*(dvrdx*dphiidx + dvrdy*dphiidy + dvrdz*dphiidz))
        - omega*omega*( (2*epsr*epsi)*mur*vr*phir
                       + (epsr*epsr-epsi*epsi)*mui*vr*phir
                       + (epsr*epsr-epsi*epsi)*mur*vi*phir
                       + (epsr*epsr-epsi*epsi)*mur*vr*phii
                       - (epsr*epsr-epsi*epsi)*mui*vi*phii
                       - (2*epsr*epsi)*mur*vi*phii
                       - (2*epsr*epsi)*mui*vr*phii
                       - (2*epsr*epsi)*mui*vi*phir)
        - omega*(- epsr*(dvidx*Axi + vi*dAxidx + dvidy*Ayi + vi*dAyidy + dvidz*Azi + vi*dAzidz)
                 - epsi*(dvrdx*Axi + vr*dAxidx + dvrdy*Ayi + vr*dAyidy + dvrdz*Azi + vr*dAzidz)
                 - epsi*(dvidx*Axr + vi*dAxrdx + dvidy*Ayr + vi*dAyrdy + dvidz*Azr + vi*dAzrdz)
                 + epsr*(dvrdx*Axr + vr*dAxrdx + dvrdy*Ayr + vr*dAyrdy + dvrdz*Azr + vr*dAzrdz))
        - (vr*rhoi + vi*rhor); //imaginary
        if(spaceDim > 1){
          resindex = offsets(Ayr_num,i);
          res(e,resindex) += ( (  -dvrdz*(dAzrdy - dAyrdz) + dvrdx*(dAyrdx - dAxrdy)
                                -(-dvidz*(dAzidy - dAyidz) + dvidx*(dAyidx - dAxidy)))*invmur
                              - (  -dvrdz*(dAzidy - dAyidz) + dvrdx*(dAyidx - dAxidy)
                                 +(-dvidz*(dAzrdy - dAyrdz) + dvidx*(dAyrdx - dAxrdy)))*invmui)
          + ( ( dvrdy*(dAxrdx + dAyrdy + dAzrdz)
               - dvidy*(dAxidx + dAyidy + dAzidz) )*invmur
             - ( dvrdy*(dAxidx + dAyidy + dAzidz)
                + dvidy*(dAxrdx + dAyrdy + dAzrdz) )*invmui)
          - omega*omega*(epsr*vr*Ayr - epsr*vi*Ayi - epsi*vr*Ayi - epsi*vi*Ayr)
          + omega*(  epsi*(dphirdy*vr + phir*dvrdy)
                   + epsr*(dphiidy*vr + phii*dvrdy)
                   + epsr*(dphirdy*vi + phir*dvidy)
                   - epsi*(dphiidy*vi + phii*dvidy))
          - (vr*Jyr - vi*Jyi); //real
          resindex = offsets(Ayi_num,i);
          res(e,resindex) += ( (  -dvrdz*(dAzidy - dAyidz) + dvrdx*(dAyidx - dAxidy)
                                +(-dvidz*(dAzrdy - dAyrdz) + dvidx*(dAyrdx - dAxrdy)))*invmur
                              + (  -dvrdz*(dAzrdy - dAyrdz) + dvrdx*(dAyrdx - dAxrdy)
                                 -(-dvidz*(dAzidy - dAyidz) + dvidx*(dAyidx - dAxidy)))*invmui)
          + ( ( dvrdy*(dAxidx + dAyidy + dAzidz)
               + dvidy*(dAxrdx + dAyrdy + dAzrdz) )*invmur
             + ( dvrdy*(dAxrdx + dAyrdy + dAzrdz)
                - dvidy*(dAxidx + dAyidy + dAzidz) )*invmui)
          - omega*omega*(-epsi*vi*Ayi + epsi*vr*Ayr + epsr*vi*Ayr + epsr*vr*Ayi)
          - omega*(- epsr*(dphiidy*vi + phii*dvidy)
                   - epsi*(dphirdy*vi + phir*dvidy)
                   - epsi*(dphiidy*vr + phii*dvrdy)
                   + epsr*(dphirdy*vr + phir*dvrdy))
          - (vr*Jyi + vi*Jyr); //imaginary
        }
        if(spaceDim > 2){
          resindex = offsets(Azr_num,i);
          res(e,resindex) += ( (  dvrdy*(dAzrdy - dAyrdz) - dvrdx*(dAxrdz - dAzrdx)
                                -(dvidy*(dAzidy - dAyidz) - dvidx*(dAxidz - dAzidx)))*invmur
                              - ( dvrdy*(dAzidy - dAyidz) - dvrdx*(dAxidz - dAzidx)
                                 + dvidy*(dAzrdy - dAyrdz) - dvidx*(dAxrdz - dAzrdx))*invmui)
          + ( ( dvrdz*(dAxrdx + dAyrdy + dAzrdz)
               - dvidz*(dAxidx + dAyidy + dAzidz) )*invmur
             - ( dvrdz*(dAxidx + dAyidy + dAzidz)
                + dvidz*(dAxrdx + dAyrdy + dAzrdz) )*invmui)
          - omega*omega*(epsr*vr*Azr - epsr*vi*Azi - epsi*vr*Azi - epsi*vi*Azr)
          + omega*(  epsi*(dphirdz*vr + phir*dvrdz)
                   + epsr*(dphiidz*vr + phii*dvrdz)
                   + epsr*(dphirdz*vi + phir*dvidz)
                   - epsi*(dphiidz*vi + phii*dvidz))
          - (vr*Jzr - vi*Jzi); //real
          resindex = offsets(Azi_num,i);
          res(e,resindex) += ( (  dvrdy*(dAzidy - dAyidz) - dvrdx*(dAxidz - dAzidx)
                                + dvidy*(dAzrdy - dAyrdz) - dvidx*(dAxrdz - dAzrdx))*invmur
                              + (  dvrdy*(dAzrdy - dAyrdz) - dvrdx*(dAxrdz - dAzrdx)
                                 -(dvidy*(dAzidy - dAyidz) - dvidx*(dAxidz - dAzidx)))*invmui)
          + ( ( dvrdz*(dAxidx + dAyidy + dAzidz)
               + dvidz*(dAxrdx + dAyrdy + dAzrdz) )*invmur
             + ( dvrdz*(dAxrdx + dAyrdy + dAzrdz)
                - dvidz*(dAxidx + dAyidy + dAzidz) )*invmui)
          - omega*omega*(-epsi*vi*Azi + epsi*vr*Azr + epsr*vi*Azr + epsr*vr*Azi)
          - omega*(- epsr*(dphiidz*vi + phii*dvidz)
                   - epsi*(dphirdz*vi + phir*dvidz)
                   - epsi*(dphiidz*vr + phii*dvrdz)
                   + epsr*(dphirdz*vr + phir*dvrdz))
          - (vr*Jzi + vi*Jzr); //imaginary
        }
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwells_fp<EvalT>::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  int resindex;
  //int Axr_basis = wkset->usebasis[Axr_num];
  //int Axi_basis = wkset->usebasis[Axi_num];
  //int Ayr_basis = wkset->usebasis[Ayr_num];
  //int Ayi_basis = wkset->usebasis[Ayi_num];
  //int Azr_basis = wkset->usebasis[Azr_num];
  //int Azi_basis = wkset->usebasis[Azi_num];
  
  
  int phir_basis_num = wkset->usebasis[phir_num];
  int phii_basis_num = wkset->usebasis[phii_num];
  
  //int numBasis = wkset->basis[Axr_basis].extent(2);
  //int numSideCubPoints = wkset->ip_side.extent(1);
  
  //    FCAD local_resid(numCC, 2*(spaceDim+1), numBasis);
  
  ScalarT x = 0.0;
  ScalarT y = 0.0;
  ScalarT z = 0.0;
  
  //test functions
  ScalarT vr = 0.0, vi = 0.0;
  
  //boundary sources
  EvalT Jsxr = 0.0, Jsyr = 0.0, Jszr = 0.0,
  Jsxi = 0.0, Jsyi = 0.0, Jszi = 0.0; //electric current J_s
  EvalT Msxr = 0.0, Msyr = 0.0, Mszr = 0.0,
  Msxi = 0.0, Msyi = 0.0, Mszi = 0.0; //magnetic current M_s
  EvalT rhosr = 0.0, rhosi = 0.0; //electric charge (i*omega*rho_s = surface divergence of J_s
  
  EvalT omega = 0.0; //frequency
  //EvalT invmur = 0.0, invmui = 0.0; //inverse permeability
  EvalT epsr = 0.0, epsi = 0.0; //permittivity
  
  //states and their gradients
  EvalT Axr = 0.0;//, dAxrdx = 0.0, dAxrdy = 0.0, dAxrdz = 0.0;
  EvalT Axi = 0.0;//, dAxidx = 0.0, dAxidy = 0.0, dAxidz = 0.0;
  EvalT Ayr = 0.0;//, dAyrdx = 0.0, dAyrdy = 0.0, dAyrdz = 0.0;
  EvalT Ayi = 0.0;//, dAyidx = 0.0, dAyidy = 0.0, dAyidz = 0.0;
  EvalT Azr = 0.0;//, dAzrdx = 0.0, dAzrdy = 0.0, dAzrdz = 0.0;
  EvalT Azi = 0.0;//, dAzidx = 0.0, dAzidy = 0.0, dAzidz = 0.0;
  //EvalT phir = 0.0, dphirdx = 0.0, dphirdy = 0.0, dphirdz = 0.0;
  //EvalT phii = 0.0, dphiidx = 0.0, dphiidy = 0.0, dphiidz = 0.0;
  
  ScalarT nx = 0.0, ny = 0.0, nz = 0.0; //components of normal
  
  // TMW: needs to use bcs from wkset
  int boundary_type = 0;//getBoundaryType(wkset->sidename);
  
  
  ScalarT weakEssScale;
  
  ScalarT current_time = wkset->time;
  
  auto res = wkset->res;
  auto offsets = wkset->offsets;
  
  auto phir_basis = wkset->basis_side[phir_basis_num];
  auto phir_basis_grad = wkset->basis_grad_side[phir_basis_num];
  auto phii_basis = wkset->basis_side[phii_basis_num];
  auto phii_basis_grad = wkset->basis_grad_side[phii_basis_num];
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  //    for (size_t e=0; e<numCC; e++) {
  //weakEssScale = 100.0/h[e];
  //bvbw      weakEssScale = essScale/h[e];
  //bvbw       need to figure out how to extract mesh h
  weakEssScale = essScale/1.0;  //bvbw replace
  //    for( int i=0; i<numBasis; i++ ) {
  
  View_EvalT2 Ax_r, Ax_i, phi_r, phi_i, Ay_r, Ay_i, Az_r, Az_i;
  Ax_r = wkset->getSolutionField("Arx");
  Ax_i = wkset->getSolutionField("Aix");
  phi_r = wkset->getSolutionField("phir");
  phi_i = wkset->getSolutionField("phii");
  
  View_EvalT2 dAxr_dx, dAxi_dx, dphir_dx, dphii_dx, dAyr_dx, dAyi_dx, dAzr_dx, dAzi_dx;
  dAxr_dx = wkset->getSolutionField("grad(Arx)[x]");
  dAxi_dx = wkset->getSolutionField("grad(Aix)[x]");
  dphir_dx = wkset->getSolutionField("grad(phir)[x]");
  dphii_dx = wkset->getSolutionField("grad(phii)[x]");
  
  View_EvalT2 dAxr_dy, dAxi_dy, dphir_dy, dphii_dy, dAyr_dy, dAyi_dy, dAzr_dy, dAzi_dy;
  View_EvalT2 dAxr_dz, dAxi_dz, dphir_dz, dphii_dz, dAyr_dz, dAyi_dz, dAzr_dz, dAzi_dz;
  
  if (spaceDim > 1) {
    Ay_r = wkset->getSolutionField("Ary");
    Ay_i = wkset->getSolutionField("Aiy");
    dAxr_dy = wkset->getSolutionField("grad(Arx)[y]");
    dAxi_dy = wkset->getSolutionField("grad(Aix)[y]");
    dphir_dy = wkset->getSolutionField("grad(phir)[y]");
    dphii_dy = wkset->getSolutionField("grad(phii)[y]");
    dAyr_dx = wkset->getSolutionField("grad(Ary)[x]");
    dAyi_dx = wkset->getSolutionField("grad(Aiy)[x]");
    dAyr_dy = wkset->getSolutionField("grad(Ary)[y]");
    dAyi_dy = wkset->getSolutionField("grad(Aiy)[y]");
  }
  if (spaceDim > 2) {
    Az_r = wkset->getSolutionField("Arz");
    Az_i = wkset->getSolutionField("Aiz");
    dAxr_dz = wkset->getSolutionField("grad(Arx)[z]");
    dAxi_dz = wkset->getSolutionField("grad(Aix)[z]");
    dAyr_dz = wkset->getSolutionField("grad(Ary)[z]");
    dAyi_dz = wkset->getSolutionField("grad(Aiy)[z]");
  
    dphir_dz = wkset->getSolutionField("grad(phir)[z]");
    dphii_dz = wkset->getSolutionField("grad(phii)[z]");
    dAzr_dx = wkset->getSolutionField("grad(Arz)[x]");
    dAzi_dx = wkset->getSolutionField("grad(Aiz)[x]");
    dAzr_dy = wkset->getSolutionField("grad(Arz)[y]");
    dAzi_dy = wkset->getSolutionField("grad(Aiz)[y]");
    dAzr_dz = wkset->getSolutionField("grad(Arz)[z]");
    dAzi_dz = wkset->getSolutionField("grad(Aiz)[z]");
  
  }
  
  View_Sc2 ip_x, ip_y, ip_z, n_x, n_y, n_z;
  ip_x = wkset->getScalarField("x");
  n_x = wkset->getScalarField("n[x]");
  if (spaceDim > 1) {
    ip_y = wkset->getScalarField("y");
    n_y = wkset->getScalarField("n[y]");
  }
  if (spaceDim > 2) {
    ip_z = wkset->getScalarField("z");
    n_z = wkset->getScalarField("n[z]");
  }
  
  
  for (size_type e=0; e<res.extent(0); e++) { // elements in workset
    
    for( size_type k=0; k<ip_x.extent(1); k++) {
      
      x = ip_x(e,k);
      nx = n_x(e,k);
      
      Axr = Ax_r(e,k);
      //dAxrdx = dAxr_dx(e,k);
      Axi = Ax_i(e,k);
      //dAxidx = dAxi_dx(e,k);
      
      //phir = phi_r(e,k);
      //phii = phi_i(e,k);
      
      //dphirdx = dphir_dx(e,k);
      //dphiidx = dphii_dx(e,k);
      
      if(spaceDim > 1){
        y = ip_y(e,k);
        ny = n_y(e,k);
        
        //dAxrdy = dAxr_dy(e,k);
        //dAxidy = dAxi_dy(e,k);
        
        Ayr = Ay_r(e,k);
        //dAyrdx = dAyr_dx(e,k);
        //dAyrdy = dAyr_dy(e,k);
        
        Ayi = Ay_i(e,k);
        //dAyidx = dAyi_dx(e,k);
        //dAyidy = dAyi_dy(e,k);
        
        //dphirdy = dphir_dy(e,k);
        //dphiidy = dphii_dy(e,k);
      }
      if(spaceDim > 2){
        z = ip_z(e,k);
        nz = n_z(e,k);
        
        //dAxrdz = dAxr_dz(e,k);
        //dAxidz = dAxi_dz(e,k);
        
        //dAyrdz = dAyr_dz(e,k);
        //dAyidz = dAyi_dz(e,k);
        
        Azr = Az_r(e,k);
        //dAzrdx = dAzr_dx(e,k);
        //dAzrdy = dAzr_dy(e,k);
        //dAzrdz = dAzr_dz(e,k);
        
        Azi = Az_i(e,k);
        //dAzidx = dAzi_dx(e,k);
        //dAzidy = dAzi_dy(e,k);
        //dAzidz = dAzi_dz(e,k);
        //dphirdz = dphir_dz(e,k);
        //dphiidz = dphii_dz(e,k);
      }
      
      
      for (size_type i=0; i<phir_basis.extent(1); i++ ) {
        vr = phir_basis(e,i,k,0);
        vi = phii_basis(e,i,k,0);  //bvbw check to make sure first index  = 0
        
        vector<vector<EvalT> > bound_current = getBoundaryCurrent(x, y, z, current_time, wkset->sidename, boundary_type);
        vector<EvalT> bound_charge = getBoundaryCharge(x, y, z, current_time);
        rhosr = bound_charge[0]; rhosi = bound_charge[1];
        
        omega = getFreq(x, y, z, current_time);
        vector<EvalT> permit = getPermittivity(x, y, z, current_time);
        epsr = permit[0]; epsi = permit[1];
        vector<EvalT> invperm = getInvPermeability(x, y, z, current_time);
        //invmur = invperm[0]; invmui = invperm[1];
        
        if(boundary_type == 1){
          Msxr = bound_current[0][0]; Msxi = bound_current[1][0];
          Msyr = bound_current[0][1]; Msyi = bound_current[1][1];
          Mszr = bound_current[0][2]; Mszi = bound_current[1][2];
          
          //weak enforcement of essential boundary conditions that are not Dirichlet boundary conditions...
          if(spaceDim == 2){
            resindex = offsets(Axr_num,i);
            res(e,resindex) += weakEssScale*( vr*(nx*Ayr-ny*Axr + (1.0/omega)*Mszi)
                                             - vi*(nx*Ayi-ny*Axi - (1.0/omega)*Mszr)); //real
            resindex = offsets(Axi_num,i);
            res(e,resindex) += weakEssScale*( vr*(nx*Ayi-ny*Axi - (1.0/omega)*Mszr)
                                             + vi*(nx*Ayr-ny*Axr + (1.0/omega)*Mszi)); //imaginary
          }
          if(spaceDim == 3){
            resindex = offsets(Axr_num,i);
            res(e,resindex) += weakEssScale*( vr*(ny*Azr-nz*Ayr + (1.0/omega)*Msxi)
                                             - vi*(ny*Azi-nz*Ayi - (1.0/omega)*Msxr)); //real
            resindex = offsets(Axi_num,i);
            res(e,resindex) += weakEssScale*( vr*(ny*Azi-nz*Ayi - (1.0/omega)*Msxr)
                                             + vi*(ny*Azr-nz*Ayr + (1.0/omega)*Msxi)); //imaginary
            
            resindex = offsets(Ayr_num,i);
            res(e,resindex) += weakEssScale*( vr*(nz*Axr-nx*Azr + (1.0/omega)*Msyi)
                                             - vi*(nz*Axi-nx*Azi - (1.0/omega)*Msyr)); //real
            resindex = offsets(Ayi_num,i);
            res(e,resindex) += weakEssScale*( vr*(nz*Axi-nx*Azi - (1.0/omega)*Msyr)
                                             + vi*(nz*Axr-nx*Azr + (1.0/omega)*Msyi)); //imaginary
            resindex = offsets(Azr_num,i);
            res(e,resindex) += weakEssScale*( vr*(nx*Ayr-ny*Axr + (1.0/omega)*Mszi)
                                             - vi*(nx*Ayi-ny*Axi - (1.0/omega)*Mszr)); //real
            resindex = offsets(Azi_num,i);
            res(e,resindex) += weakEssScale*( vr*(nx*Ayi-ny*Axi - (1.0/omega)*Mszr)
                                             + vi*(nx*Ayr-ny*Axr + (1.0/omega)*Mszi)); //imaginary
          }
          // from applying divergence theorem and such to weak form
          //local_resid(e,0,i) += ( invmur*(- vr*nz*(dAxrdz-dAzrdx) + vr*ny*(dAyrdx-dAxrdy)
          // + vi*nz*(dAxidz-dAzidx) - vi*ny*(dAyidx-dAxidy))
          // - invmui*(- vr*nz*(dAxidz-dAzidx) + vr*ny*(dAyidx-dAxidy)
          // - vi*nz*(dAxrdz-dAzrdx) + vi*ny*(dAyrdx-dAxrdy))); //real
          // local_resid(e,1,i) += ( invmur*(- vr*nz*(dAxidz-dAzidx) + vr*ny*(dAyidx-dAxidy)
          // - vi*nz*(dAxrdz-dAzrdx) + vi*ny*(dAyrdx-dAxrdy))
          // + invmui*(- vr*nz*(dAxrdz-dAzrdx) + vr*ny*(dAyrdx-dAxrdy)
          // + vi*nz*(dAxidz-dAzidx) - vi*ny*(dAyidx-dAxidy))); //imaginary
          
          // local_resid(e,4,i) += ( invmur*(+ vr*nz*(dAzrdy-dAyrdz) - vr*nx*(dAyrdx-dAxrdy)
          // - vi*nz*(dAzidy-dAyidz) + vi*nx*(dAyidx-dAxidy))
          // - invmui*(+ vr*nz*(dAzidy-dAyidz) - vr*nx*(dAyidx-dAxidy)
          // + vi*nz*(dAzrdy-dAyrdz) - vi*nx*(dAyrdx-dAxrdy))); //real
          // local_resid(e,5,i) += ( invmur*(+ vr*nz*(dAzidy-dAyidz) - vr*nx*(dAyidx-dAxidy)
          // + vi*nz*(dAzrdy-dAyrdz) - vi*nx*(dAyrdx-dAxrdy))
          // + invmui*(+ vr*nz*(dAzrdy-dAyrdz) - vr*nx*(dAyrdx-dAxrdy)
          // - vi*nz*(dAzidy-dAyidz) + vi*nx*(dAyidx-dAxidy))); //imaginary
          
          // local_resid(e,6,i) += ( invmur*(- vr*ny*(dAzrdy-dAyrdz) + vr*nx*(dAxrdz-dAzrdx)
          // + vi*ny*(dAzidy-dAyidz) - vi*nx*(dAxidz-dAzidx))
          // - invmui*(- vr*ny*(dAzidy-dAyidz) + vr*nx*(dAxidz-dAzidx)
          // - vi*ny*(dAzrdy-dAyrdz) + vi*nx*(dAxrdz-dAzrdx))); //real
          // local_resid(e,7,i) += ( invmur*(- vr*ny*(dAzidy-dAyidz) + vr*nx*(dAxidz-dAzidx)
          // - vi*ny*(dAzrdy-dAyrdz) + vi*nx*(dAxrdz-dAzrdx))
          // + invmui*(- vr*ny*(dAzrdy-dAyrdz) + vr*nx*(dAxrdz-dAzrdx)
          // + vi*ny*(dAzidy-dAyidz) - vi*nx*(dAxidz-dAzidx))); //imaginary
          //
        }else if(boundary_type == 2){
          Jsxr = bound_current[0][0]; Jsxi = bound_current[1][0];
          
          //weak enforcement of essential boundary conditions that are not Dirichlet boundary conditions...
          resindex = offsets(phir_num,i);
          res(e,resindex) += weakEssScale*
          ( vr*(epsr*(nx*Axr+ny*Ayr+nz*Azr)-epsi*(nx*Axi+ny*Ayi+nz*Azi)-(1.0/omega)*rhosi)
           - vi*(epsr*(nx*Axi+ny*Ayi+nz*Azi)+epsi*(nx*Axr+ny*Ayr+nz*Azr)+(1.0/omega)*rhosr)); //real
          
          resindex = offsets(phii_num,i);
          res(e,resindex) += weakEssScale*
          ( vr*(epsr*(nx*Axi+ny*Ayi+nz*Azi)+epsi*(nx*Axr+ny*Ayr+nz*Azr)+(1.0/omega)*rhosr)
           + vi*(epsr*(nx*Axr+ny*Ayr+nz*Azr)-epsi*(nx*Axi+ny*Ayi+nz*Azi)-(1.0/omega)*rhosi)); //imaginary
          
          //from applying divergence theorem and such to weak form
          resindex = offsets(Axr_num,i);
          res(e,resindex) += (vr*Jsxr - vi*Jsxi);
          //- ( invmur*(vr*nx*(dAxrdx+dAyrdy+dAzrdz) - vi*nx*(dAxidx+dAyidy+dAzidz))
          //  - invmui*(vr*nx*(dAxidx+dAyidy+dAzidz) + vi*nx*(dAxrdx+dAyrdy+dAzrdz)))
          //- (omega*((epsr*phir-epsi*phii)*(vi*nx) + (epsr*phii+epsi*phir)*(vr*nx))); //real
          resindex = offsets(Axi_num,i);
          res(e,resindex) += (vr*Jsxi + vi*Jsxr);
          //- ( invmur*(vr*nx*(dAxidx+dAyidy+dAzidz) + vi*nx*(dAxrdx+dAyrdy+dAzrdz))
          //  + invmui*(vr*nx*(dAxrdx+dAyrdy+dAzrdz) - vi*nx*(dAxidx+dAyidy+dAzidz)))
          //+ (omega*((epsr*phir-epsi*phii)*(vr*nx) - (epsr*phii+epsi*phir)*(vi*nx))); //imaginary
          
          //local_resid(e,2,i) += -omega*( epsr*(vr*(Axi*nx+Ayi*ny+Azi*nz) + vi*(Axr*nx+Ayr*ny+Azr*nz))
          //                             + epsi*(vr*(Axr*nx+Ayr*ny+Azr*nz) - vi*(Axi*nx+Ayi*ny+Azi*nz))); //real
          //local_resid(e,3,i) +=  omega*(  epsr*(vr*(Axr*nx+Ayr*ny+Azr*nz) - vi*(Axi*nx+Ayi*ny+Azi*nz))
          //                              - epsi*(vr*(Axi*nx+Ayi*ny+Azi*nz) + vi*(Axr*nx+Ayr*ny+Azr*nz))); //imaginary
          resindex = offsets(phir_num,i);
          res(e,resindex) += vr*rhosr - vi*rhosi; //real
          resindex = offsets(phii_num,i);
          res(e,resindex) += vr*rhosi + vi*rhosr; //imaginary
          if(spaceDim > 1){
            Jsyr = bound_current[0][1]; Jsyi = bound_current[1][1];
            resindex = offsets(Ayr_num,i);
            res(e,resindex) += (vr*Jsyr - vi*Jsyi);
            //- ( invmur*(vr*ny*(dAxrdx+dAyrdy+dAzrdz) - vi*ny*(dAxidx+dAyidy+dAzidz))
            //  - invmui*(vr*ny*(dAxidx+dAyidy+dAzidz) + vi*ny*(dAxrdx+dAyrdy+dAzrdz)))
            //- (omega*((epsr*phir-epsi*phii)*(vi*ny) + (epsr*phii+epsi*phir)*(vr*ny))); //real
            resindex = offsets(Ayi_num,i);
            res(e,resindex) += (vr*Jsyi + vi*Jsyr);
            //- ( invmur*(vr*ny*(dAxidx+dAyidy+dAzidz) + vi*ny*(dAxrdx+dAyrdy+dAzrdz))
            //  + invmui*(vr*ny*(dAxrdx+dAyrdy+dAzrdz) - vi*ny*(dAxidx+dAyidy+dAzidz)))
            //+ (omega*((epsr*phir-epsi*phii)*(vr*ny) - (epsr*phii+epsi*phir)*(vi*ny))); //imaginary
          }
          if(spaceDim > 2){
            Jszr = bound_current[0][2]; Jszi = bound_current[1][2];
            resindex = offsets(Azr_num,i);
            res(e,resindex) += (vr*Jszr - vi*Jszi);
            //- ( invmur*(vr*nz*(dAxrdx+dAyrdy+dAzrdz) - vi*nz*(dAxidx+dAyidy+dAzidz))
            //  - invmui*(vr*nz*(dAxidx+dAyidy+dAzidz) + vi*nz*(dAxrdx+dAyrdy+dAzrdz)))
            //- (omega*((epsr*phir-epsi*phii)*(vi*nz) + (epsr*phii+epsi*phir)*(vr*nz))); //real
            resindex = offsets(Azi_num,i);
            res(e,resindex) += (vr*Jszi + vi*Jszr);
            //- ( invmur*(vr*nz*(dAxidx+dAyidy+dAzidz) + vi*nz*(dAxrdx+dAyrdy+dAzrdz))
            //  + invmui*(vr*nz*(dAxrdx+dAyrdy+dAzrdz) - vi*nz*(dAxidx+dAyidy+dAzidz)))
            //+ (omega*((epsr*phir-epsi*phii)*(vr*nz) - (epsr*phii+epsi*phir)*(vi*nz))); //imaginary
          }
        }
      }
    }
  }
  
}

// ========================================================================================
// true solution for error calculation
// ========================================================================================

template<class EvalT>
void maxwells_fp<EvalT>::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void maxwells_fp<EvalT>::computeFlux() {
  
}

// =======================================================================================
// return frequency
// ======================================================================================

template<class EvalT>
EvalT maxwells_fp<EvalT>::getFreq(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const{
  bool fndom = false;
  auto omvals = wkset->getParameter("maxwells_fp_freq", fndom);
  EvalT omega = 0.0;
  if (fndom) {
    omega = omvals(0);//freq_params[0];
  }
  
  return omega;
}

// ========================================================================================
// return magnetic permeability
// ========================================================================================

template<class EvalT>
vector<EvalT> maxwells_fp<EvalT>::getPermeability(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const{
  
  vector<EvalT> mu;
  if(test == 1){
    mu.push_back(2.0);
    mu.push_back(1.0);
  }else if(test == 2){
    mu.push_back(2.0/(x*x+1.0));
    mu.push_back(1.0/(x*x+1.0));
  }else if(test == 3){
    mu.push_back(1.0/(x*x+1.0));
    mu.push_back(0.0);
  }else{
    mu.push_back(1.0);
    mu.push_back(0.0);
  }
  
  return mu;
  
}

// ========================================================================================
// return inverse of magnetic permeability
// ========================================================================================

template<class EvalT>
vector<EvalT> maxwells_fp<EvalT>::getInvPermeability(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const {
  
  vector<EvalT> invmu;
  if(test == 1){
    invmu.push_back(0.4);
    invmu.push_back(-0.2);
  }else if(test == 2){
    invmu.push_back(0.4*(x*x+1.0));
    invmu.push_back(-0.2*(x*x+1.0));
  }else if(test == 3){
    invmu.push_back(x*x+1.0);
    invmu.push_back(0.0);
  }else{
    invmu.push_back(1.0);
    invmu.push_back(0.0);
  }
  
  return invmu;
  
}

// ========================================================================================
// return electric permittivity
// ========================================================================================

template<class EvalT>
vector<EvalT> maxwells_fp<EvalT>::getPermittivity(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const{
  
  vector<EvalT> permit;
  if(test == 1){
    permit.push_back(1.0);
    permit.push_back(1.0);
  }else if(test == 2){
    permit.push_back(x*x+1.0);
    permit.push_back(x*x+1.0);
  }else if(test == 3){
    permit.push_back(2.0*(x*x+1.0));;
    permit.push_back(0.0);
  }else if(test == 4){
    if((x-10.0)*(x-10.0)+y*y <= 100.0){
      permit.push_back(0.2);
      permit.push_back(0.0);
    }else{
      permit.push_back(0.01961);
      permit.push_back(0.003922);
    }
  }else{
    permit.push_back(1.0);
    permit.push_back(0.0);
  }
  
  return permit;
  
}

// ========================================================================================
// return current density in interior of domain
// ========================================================================================

template<class EvalT>
vector<vector<EvalT> > maxwells_fp<EvalT>::getInteriorCurrent(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const{
  
  vector<vector<EvalT> > J(2,vector<EvalT>(3,0.0));
  
  if(test == 1){
    J[0][0] = (1.8*PI*PI)*sin(PI*x)*sin(PI*y)*sin(PI*z);
    J[0][1] = (-1.8*PI*PI)*sin(PI*x)*sin(PI*y)*sin(PI*z);
    J[0][2] = (3.6*PI*PI)*sin(PI*x)*sin(PI*y)*sin(PI*z);
    J[1][0] = (0.6*PI*PI-2.0)*sin(PI*x)*sin(PI*y)*sin(PI*z);
    J[1][1] = (-0.6*PI*PI+2.0)*sin(PI*x)*sin(PI*y)*sin(PI*z);
    J[1][2] = (1.2*PI*PI-4.0)*sin(PI*x)*sin(PI*y)*sin(PI*z);
  }else if(test == 2){
    J[0][0] = (9.*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z))/5.
    - 4.*x*sin(PI*x)*sin(PI*y)*sin(PI*z)
    + (9.*x*x*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z))/5.
    - (6.*x*PI*cos(PI*x)*sin(PI*y)*sin(PI*z))/5.
    + (6.*x*PI*cos(PI*y)*sin(PI*x)*sin(PI*z))/5.
    - (12.*x*PI*cos(PI*z)*sin(PI*x)*sin(PI*y))/5.;
    J[0][1] = -(3.*PI*sin(PI*z)*(3.*PI*sin(PI*x)*sin(PI*y) - 2.*x*cos(PI*x)*sin(PI*y)
                                 - 2.*x*cos(PI*y)*sin(PI*x) + 3.*x*x*PI*sin(PI*x)*sin(PI*y)))/5.;
    J[0][2] = (6.*PI*sin(PI*y)*(3.*PI*sin(PI*x)*sin(PI*z) - 2.*x*cos(PI*x)*sin(PI*z)
                                + x*cos(PI*z)*sin(PI*x) + 3.*x*x*PI*sin(PI*x)*sin(PI*z)))/5.;
    J[1][0] = (3.*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z))/5.
    - 2.*x*x*sin(PI*x)*sin(PI*y)*sin(PI*z)
    - 2.*sin(PI*x)*sin(PI*y)*sin(PI*z)
    + (3.*x*x*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z))/5.
    - (2.*x*PI*cos(PI*x)*sin(PI*y)*sin(PI*z))/5.
    + (2.*x*PI*cos(PI*y)*sin(PI*x)*sin(PI*z))/5.
    - (4.*x*PI*cos(PI*z)*sin(PI*x)*sin(PI*y))/5.;
    J[1][1] = (3.*sin(PI*z)*((10.*sin(PI*x)*sin(PI*y))/3. - PI*PI*sin(PI*x)*sin(PI*y) + (10.*x*x*sin(PI*x)*sin(PI*y))/3.
                             - x*x*PI*PI*sin(PI*x)*sin(PI*y) + (2.*x*PI*cos(PI*x)*sin(PI*y))/3. + (2.*x*PI*cos(PI*y)*sin(PI*x))/3.))/5.;
    J[1][2] = -(6.*sin(PI*y)*((10.*sin(PI*x)*sin(PI*z))/3. - PI*PI*sin(PI*x)*sin(PI*z) + (10.*x*x*sin(PI*x)*sin(PI*z))/3.
                              - x*x*PI*PI*sin(PI*x)*sin(PI*z) + (2.*x*PI*cos(PI*x)*sin(PI*z))/3. - (x*PI*cos(PI*z)*sin(PI*x))/3.))/5.;
  }else if(test == 3){
    J[0][0] = -PI*sin(PI*x)*sin(PI*(y - z))*(x*x + 1.)*(3.*PI*PI - 2.);
    J[0][1] = (PI*cos(PI*x)*sin(PI*y)*sin(PI*z) - PI*cos(PI*z)*sin(PI*x)*sin(PI*y))*(2.*x*x + 2.)
    + (x*x + 1.)*(PI*PI*PI*cos(PI*x)*cos(PI*y)*cos(PI*z) - PI*PI*PI*cos(PI*x)*sin(PI*y)*sin(PI*z) + 2.*PI*PI*PI*cos(PI*z)*sin(PI*x)*sin(PI*y))
    - (x*x + 1.)*(PI*PI*PI*cos(PI*x)*cos(PI*y)*cos(PI*z) + 2.*PI*PI*PI*cos(PI*x)*sin(PI*y)*sin(PI*z) - PI*PI*PI*cos(PI*z)*sin(PI*x)*sin(PI*y))
    - 2.*x*(PI*PI*cos(PI*x)*cos(PI*z)*sin(PI*y) + PI*PI*cos(PI*y)*cos(PI*z)*sin(PI*x) + 2.*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z));
    J[0][2] = (x*x + 1.)*(PI*PI*PI*cos(PI*x)*cos(PI*y)*cos(PI*z) + 2.*PI*PI*PI*cos(PI*x)*sin(PI*y)*sin(PI*z) - PI*PI*PI*cos(PI*y)*sin(PI*x)*sin(PI*z))
    - (x*x + 1.)*(PI*PI*PI*cos(PI*x)*cos(PI*y)*cos(PI*z) - PI*PI*PI*cos(PI*x)*sin(PI*y)*sin(PI*z) + 2.*PI*PI*PI*cos(PI*y)*sin(PI*x)*sin(PI*z))
    - (PI*cos(PI*x)*sin(PI*y)*sin(PI*z) - PI*cos(PI*y)*sin(PI*x)*sin(PI*z))*(2.*x*x + 2.)
    + 2.*x*(PI*PI*cos(PI*x)*cos(PI*y)*sin(PI*z) + PI*PI*cos(PI*y)*cos(PI*z)*sin(PI*x) + 2.*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z));
    J[1][0] = 4.*x*sin(PI*x)*sin(PI*y)*sin(PI*z);
    J[1][1] = 0.0;
    J[1][2] = 0.0;
  }
  
  return J;
}

// ========================================================================================
// return charge density in interior of domain
// ========================================================================================

template<class EvalT>
vector<EvalT> maxwells_fp<EvalT>::getInteriorCharge(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const{
  
  vector<EvalT> rho(2,0.0);
  
  if(test == 1){
    rho[0] = 6.0*sin(PI*x)*sin(PI*y)*sin(PI*z);
    rho[1] = (6.0*PI*PI-2.0)*sin(PI*x)*sin(PI*y)*sin(PI*z);
  }else if(test == 2){
    rho[0] = 2.*sin(PI*x)*sin(PI*y)*sin(PI*z)*(3.*x*x - 2.*x + 3.);
    rho[1] = -2.*sin(PI*y)*sin(PI*z)*(sin(PI*x) - 3.*PI*PI*sin(PI*x) + x*x*sin(PI*x)
                                      - 3.*x*x*PI*PI*sin(PI*x) + 2.*x*PI*cos(PI*x));
  }else if(test == 3){
    rho[0] = 3.*PI*PI*sin(PI*x)*sin(PI*y)*sin(PI*z)*(2.*x*x + 2.)
    - sin(PI*x)*sin(PI*y)*sin(PI*z)*(4.*x*x + 4.)
    - 4.*x*PI*cos(PI*x)*sin(PI*y)*sin(PI*z);
    rho[1] = -4.*x*PI*sin(PI*x)*sin(PI*(y - z));
  }
  
  return rho;
  
}

// =======================================================================================
// return electric current on boundary of domain
// =======================================================================================

template<class EvalT>
vector<vector<EvalT> > maxwells_fp<EvalT>::getBoundaryCurrent(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time,
                                                    const string & side_name, const int & boundary_type) const{
  
  vector<vector<EvalT> > Js(2,vector<EvalT>(3,0.0));
  if(test == 3 && boundary_type == 1){
    if(side_name == "right"){
      Js[1][0] = 0.0;
      Js[1][1] = -PI*sin(PI*z)*sin(PI*(x - y));
      Js[1][2] = -PI*sin(PI*y)*sin(PI*(x - z));
    }else if(side_name == "left"){
      Js[1][0] = 0.0;
      Js[1][1] = PI*sin(PI*z)*sin(PI*(x - y));
      Js[1][2] = PI*sin(PI*y)*sin(PI*(x - z));
    }else if(side_name == "top"){
      Js[1][0] = PI*sin(PI*z)*sin(PI*(x - y));
      Js[1][1] = 0.0;
      Js[1][2] = -PI*sin(PI*x)*sin(PI*(y - z));
    }else if(side_name == "bottom"){
      Js[1][0] = -PI*sin(PI*z)*sin(PI*(x - y));
      Js[1][1] = 0.0;
      Js[1][2] = PI*sin(PI*x)*sin(PI*(y - z));
    }else if(side_name == "front"){
      Js[1][0] = PI*sin(PI*y)*sin(PI*(x - z));
      Js[1][1] = PI*sin(PI*x)*sin(PI*(y - z));
      Js[1][2] = 0.0;
    }else if(side_name == "back"){
      Js[1][0] = -PI*sin(PI*y)*sin(PI*(x - z));
      Js[1][1] = -PI*sin(PI*x)*sin(PI*(y - z));
      Js[1][2] = 0.0;
    }
  }else if(test == 4){ //J_s = nhat x H, H = <0,0,1>
    ScalarT nx = x/sqrt(x*x+y*y);
    ScalarT ny = y/sqrt(x*x+y*y);
    Js[0][0] = ny;
    Js[0][1] = -nx                                            ;
  }
  
  return Js;
  
}

// ========================================================================================
// return charge density on boundary of domain (should be surface divergence of boundary current divided by i*omega
// ========================================================================================

template<class EvalT>
vector<EvalT> maxwells_fp<EvalT>::getBoundaryCharge(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const{
  vector<EvalT> rhos(2,0.0);
  return rhos;
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwells_fp<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;
  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "Arx")
      Axr_num = i;
    if (varlist[i] == "Aix")
      Axi_num = i;
    if (varlist[i] == "Ary")
      Ayr_num = i;
    if (varlist[i] == "Aiy")
      Ayi_num = i;
    if (varlist[i] == "Arz")
      Azr_num = i;
    if (varlist[i] == "Aiz")
      Azi_num = i;
    if (varlist[i] == "phir")
      phir_num = i;
    if (varlist[i] == "phii")
      phii_num = i;
  }
}

// ========================================================================================
// TMW: this needs to be deprecated
// ========================================================================================

/*
template<class EvalT>
void maxwells_fp<EvalT>::updateParameters(const vector<Teuchos::RCP<vector<EvalT> > > & params, const std::vector<string> & paramnames) {
  for (size_t p=0; p<paramnames.size(); p++) {
    if (paramnames[p] == "maxwells_fp_mu")
      mu_params = *(params[p]);
    else if (paramnames[p] == "maxwells_fp_epsilon")
      eps_params = *(params[p]);
    else if (paramnames[p] == "maxwells_fp_freq")
      freq_params = *(params[p]);
    else if (paramnames[p] == "maxwells_fp_source")
      source_params = *(params[p]);
    else if (paramnames[p] == "maxwells_fp_boundary")
      boundary_params = *(params[p]);
  }
}
*/

//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::maxwells_fp<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::maxwells_fp<AD>;

// Standard built-in types
template class MrHyDE::maxwells_fp<AD2>;
template class MrHyDE::maxwells_fp<AD4>;
template class MrHyDE::maxwells_fp<AD8>;
template class MrHyDE::maxwells_fp<AD16>;
template class MrHyDE::maxwells_fp<AD18>;
template class MrHyDE::maxwells_fp<AD24>;
template class MrHyDE::maxwells_fp<AD32>;
#endif
