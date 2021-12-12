
#include "trilinos.hpp"
#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_DOFManager.hpp"

#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HCURL_HEX_I1_FEM.hpp"
#include "Intrepid2_HDIV_HEX_I1_FEM.hpp"
#include "Intrepid2_HVOL_C0_FEM.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<MpiComm> Comm = Teuchos::rcp( new MpiComm(MPI_COMM_WORLD) );
  
  Kokkos::initialize();

  {
    
    // Create a mesh from the file defined by the user
    
    std::string input_file_name = argv[1];
    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    
    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
    pl->set("File Name",input_file_name);
    
    mesh_factory->setParameterList(pl);
    Teuchos::RCP<panzer_stk::STK_Interface> mesh = mesh_factory->buildUncommitedMesh(*(Comm->getRawMpiComm()));
    
    mesh_factory->completeMeshConstruction(*mesh,*(Comm->getRawMpiComm()));
    
    if (Comm->getRank() == 0) {
      mesh->printMetaData(std::cout);
    }
    vector<string> blocknames;
    mesh->getElementBlockNames(blocknames);
    int spaceDim = mesh->getDimension();
    
    // Defined the connectivity manager
    Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
    
    // Test out a few DOF managers
    {
      if (Comm->getRank() == 0) {
        std::cout << "================================================" << std::endl;
        std::cout << " *** Testing one HGRAD variable *** " << std::endl;
      }
      Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
      DOF->setConnManager(conn,*(Comm->getRawMpiComm()));
      DOF->setOrientationsRequired(true);
      
      for (size_t b=0; b<blocknames.size(); b++) {
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
        basis_RCP basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::Device::execution_space,double,double>() );
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
        DOF->addField(blocknames[b], "T", Pattern, panzer::FieldType::CG);
      }
      
      DOF->buildGlobalUnknowns();
      if (Comm->getRank() == 0) {
        DOF->printFieldInformation(std::cout);
        std::cout << "================================================" << std::endl << std::endl;
      }
    }
    
    {
      if (Comm->getRank() == 0) {
        std::cout << "================================================" << std::endl;
        std::cout << " *** Testing one HCURL variable *** " << std::endl;
      }
      Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
      DOF->setConnManager(conn,*(Comm->getRawMpiComm()));
      DOF->setOrientationsRequired(true);
      
      for (size_t b=0; b<blocknames.size(); b++) {
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
        basis_RCP basis = Teuchos::rcp(new Intrepid2::Basis_HCURL_HEX_I1_FEM<PHX::Device::execution_space,double,double>() );
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
        DOF->addField(blocknames[b], "E", Pattern, panzer::FieldType::CG);
      }
      
      DOF->buildGlobalUnknowns();
      if (Comm->getRank() == 0) {
        DOF->printFieldInformation(std::cout);
        std::cout << "================================================" << std::endl << std::endl;
      }
    }
    
    {
      if (Comm->getRank() == 0) {
        std::cout << "================================================" << std::endl;
        std::cout << " *** Testing one HDIV variable *** " << std::endl;
      }
      Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
      DOF->setConnManager(conn,*(Comm->getRawMpiComm()));
      DOF->setOrientationsRequired(true);
      
      for (size_t b=0; b<blocknames.size(); b++) {
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
        basis_RCP basis = Teuchos::rcp(new Intrepid2::Basis_HDIV_HEX_I1_FEM<PHX::Device::execution_space,double,double>() );
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
        DOF->addField(blocknames[b], "B", Pattern, panzer::FieldType::CG);
      }
      
      DOF->buildGlobalUnknowns();
      if (Comm->getRank() == 0) {
        DOF->printFieldInformation(std::cout);
        std::cout << "================================================" << std::endl << std::endl;
      }
    }
    
    {
      if (Comm->getRank() == 0) {
        std::cout << "================================================" << std::endl;
        std::cout << " *** Testing one HVOL variable *** " << std::endl;
      }
      Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
      DOF->setConnManager(conn,*(Comm->getRawMpiComm()));
      DOF->setOrientationsRequired(true);
      
      for (size_t b=0; b<blocknames.size(); b++) {
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
        basis_RCP basis = Teuchos::rcp(new Intrepid2::Basis_HVOL_C0_FEM<PHX::Device::execution_space,double,double>(*cellTopo) );
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
        DOF->addField(blocknames[b], "p", Pattern, panzer::FieldType::CG);
      }
      
      DOF->buildGlobalUnknowns();
      if (Comm->getRank() == 0) {
        DOF->printFieldInformation(std::cout);
        std::cout << "================================================" << std::endl << std::endl;
      }
    }
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


