
#include "trilinos.hpp"
#include "preferences.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Kokkos_Core.hpp"

//#include "discretizationInterface.hpp"
#include "Panzer_OrientationsInterface.hpp"
#include "Panzer_DOFManager.hpp"
#include "Panzer_BlockedDOFManager.hpp"
#include "Panzer_ConnManager.hpp"
#include "Panzer_STK_Interface.hpp"

#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"

#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"

#include "Panzer_STKConnManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_STK_SetupUtilities.hpp"

#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"

typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;
typedef Intrepid2::FunctionSpaceTools<PHX::Device::execution_space> FuncTools;
typedef Intrepid2::OrientationTools<PHX::Device::execution_space> OrientTools;
typedef Intrepid2::RealSpaceTools<PHX::Device::execution_space> RealTools;
typedef Intrepid2::ArrayTools<PHX::Device::execution_space> ArrayTools;
  
class Compressor {
    
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    Compressor() {};
    
    ~Compressor() {};
    
    Compressor(const Teuchos::RCP<MpiComm> & Comm_,
               Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
               Teuchos::RCP<panzer::DOFManager> DOF_ ) :
      Comm(Comm_), mesh(mesh_), DOF(DOF_) {

      mesh->getElementBlockNames(blocknames);
      dimension = mesh->getDimension();
      
      int quadorder = 2;
      Intrepid2::DefaultCubatureFactory cubFactory;
      
      for (size_t block=0; block<blocknames.size(); ++block) {

        // =====================================================
        // Get the integration pts/wts on reference element
      
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[block]);
        Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double> > basisCub  = cubFactory.create<PHX::Device::execution_space, double, double>(*cellTopo, quadorder);
        int cubDim = basisCub->getDimension();
        int numCubPoints = basisCub->getNumPoints();
        ref_ip.push_back(DRV("ip", numCubPoints, cubDim));
        ref_wts.push_back(DRV("wts", numCubPoints));
        basisCub->getCubature(ref_ip[block], ref_wts[block]);
        numip.push_back(numCubPoints);
        
        // =====================================================
        // Get the nodes on this block

        vector<stk::mesh::Entity> stk_meshElems;
        mesh->getMyElements(blocknames[block], stk_meshElems);
        
        numElem.push_back(stk_meshElems.size());

        // list of all elements on this processor
        vector<size_t> blockmyElements = vector<size_t>(stk_meshElems.size());
        for( size_t e=0; e<stk_meshElems.size(); e++ ) {
          blockmyElements[e] = mesh->elementLocalId(stk_meshElems[e]);
        }
        myElements.push_back(blockmyElements);
        
        int numNodesPerElem = cellTopo->getNodeCount();

        DRV blocknodes("currnodes", numElem[block], numNodesPerElem, dimension);
        
        vector<size_t> local_grp(numElem[block]);
        for (size_t e=0; e<numElem[block]; ++e) {
          local_grp[e] = myElements[block][e];
        }
        
        mesh->getElementVertices(local_grp, blocknames[block], blocknodes);
      
        nodes.push_back(blocknodes);
        // =====================================================

      }

      // =====================================================
      // Get the orientations

      auto pOInt = panzer::OrientationsInterface(DOF);
      auto pO_orients = pOInt.getOrientations();
      orientations = *pO_orients;

      // =====================================================
      
    }

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
  
    void buildDatabase(const size_t & block) {

      // first_users just keeps track of a reference element for each entry in the database
      vector< size_t > first_users;
  
      /////////////////////////////////////////////////////////////////////////////
      // Step 1: identify the duplicate information
      /////////////////////////////////////////////////////////////////////////////
  
      this->identifyVolumetricDatabase(block, first_users);
  
      /////////////////////////////////////////////////////////////////////////////
      // Step 2: inform the user about the savings
      /////////////////////////////////////////////////////////////////////////////
    
      cout << " - Processor " << Comm->getRank() << ": Number of elements on block " << blocknames[block] << ": " << numElem[block] << endl;
      cout << " - Processor " << Comm->getRank() << ": Number of unique elements on block " << blocknames[block] << ": " << first_users.size() << endl;
      cout << " - Processor " << Comm->getRank() << ": Database memory savings on " << blocknames[block] << ": "
      << (100.0 - 100.0*((double)first_users.size()/(double)numElem[block])) << "%" << endl;
  
      /////////////////////////////////////////////////////////////////////////////
      // Step 3: build the database
      /////////////////////////////////////////////////////////////////////////////
  
      //this->buildVolumetricDatabase(block, first_users);
  
    }

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
  
    void identifyVolumetricDatabase(const size_t & block, vector<size_t> & first_users) {
  
      double database_TOL = 1.0e-12;
      bool ignore_orientations = false;
  
      vector<Kokkos::View<ScalarT***,HostDevice>> db_jacobians;
      vector<ScalarT> db_measures, db_jacobian_norms;      
      topo_RCP cellTopo = mesh->getCellTopology(blocknames[block]);

      // There are only so many unique orientation
      // Creating a short list of the unique ones and the index for each element
  
      vector<string> unique_orients;
      vector<size_t> all_orients(numElem[block]);
      for (size_t e=0; e<numElem[block]; ++e) {
        size_t elemID = myElements[block][e];
        string orient = orientations[elemID].to_string();
        bool found = false;
        size_t oprog = 0;
        while (!found && oprog<unique_orients.size()) {
          if (orient == unique_orients[oprog]) {
            found = true;
          }
          else {
            ++oprog;
          }
        }
        if (found) {
          all_orients[e] = oprog;
        }
        else {
          unique_orients.push_back(orient);
          all_orients[e] = unique_orients.size()-1;
        }
      }
      
      ///////////////////////////////////////////////////////////////
      // Now we actually determine the uniue elements
      ///////////////////////////////////////////////////////////////
      
      for (size_t e=0; e<numElem[block]; ++e) {

        // Get the Jacobian for this element
        DRV jacobian("jacobian", 1, numip[block], dimension, dimension);
        DRV currnodes("current nodes", 1, nodes[block].extent(1), nodes[block].extent(2));
        for (size_type n=0; n<nodes[block].extent(1); ++n) {
          for (size_type d=0; d<nodes[block].extent(2); ++d) {
            currnodes(0,n,d) = nodes[block](e,n,d);
          }
        }
        CellTools::setJacobian(jacobian, ref_ip[block], currnodes, *cellTopo);
         
        // Get the measures for this element
        ScalarT measure = 0.0;
        {
          DRV jacobianDet("determinant of jacobian", 1, numip[block]);
          CellTools::setJacobianDet(jacobianDet, jacobian);
          DRV wts("jacobian", 1, numip[block]);
          FuncTools::computeCellMeasure(wts, jacobianDet, ref_wts[block]);
      
          for (size_type pt=0; pt<wts.extent(1); ++pt) {
            measure += wts(0,pt);
          }
          
        }

        bool found = false;
        size_t prog = 0;
  
        while (!found && prog<first_users.size()) {
          size_t refelem = first_users[prog];
        
          // Check #1: element orientations
          size_t orient = all_orients[e];
          size_t reforient = all_orients[refelem];
          if (ignore_orientations || orient == reforient) {
          
            // Check #2: element measures
            ScalarT diff = std::abs(measure-db_measures[prog]);
        
            if (std::abs(diff/db_measures[prog])<database_TOL) { // abs(measure) is probably unnecessary here
            
              ScalarT refnorm = db_jacobian_norms[prog];
              // Check #3: element Jacobians
              ScalarT diff2 = 0.0;
              size_type pt=0;
              while (pt<numip[block] && diff2<database_TOL) {
                size_type d0=0;
                while (d0<dimension && diff2<database_TOL) {
                  size_type d1=0;
                  while (d1<dimension && diff2<database_TOL) { 
                    diff2 += std::abs(jacobian(0,pt,d0,d1) - db_jacobians[prog](pt,d0,d1));
                    d1++;
                  }
                  d0++;
                }
                pt++;
              }
            
              if (diff2/refnorm<database_TOL) {
                found = true;
              }
              else {
                ++prog;
              }
            
            }
            else {
              ++prog;
            }
          }
          else {
            ++prog;
          }
        }
        if (!found) {
          first_users.push_back(e);
        
          Kokkos::View<ScalarT***,HostDevice> new_jac("new db jac",numip[block], dimension, dimension);
          ScalarT jnorm = 0.0;
          for (size_type pt=0; pt<new_jac.extent(0); ++pt) {
            for (size_type d0=0; d0<new_jac.extent(1); ++d0) {
              for (size_type d1=0; d1<new_jac.extent(2); ++d1) {
                new_jac(pt,d0,d1) = jacobian(0,pt,d0,d1);
                jnorm += std::abs(jacobian(0,pt,d0,d1));
              }
            }
          }
          db_jacobians.push_back(new_jac);
          db_measures.push_back(measure);
          db_jacobian_norms.push_back(jnorm);
        
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    void buildVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users) {
      // this is used in MrHyDE, but it isn't clear if we need it here
    }
  
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
    Teuchos::RCP<panzer::DOFManager> DOF;
    std::vector<string> blocknames;
    vector<vector<size_t>> myElements;
    vector<DRV> nodes, ref_ip, ref_wts;
    std::vector<Intrepid2::Orientation> orientations;
    vector<size_t> numElem, numip;
    int dimension;
};


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
        
int main(int argc, char * argv[]) {
  
  TEUCHOS_TEST_FOR_EXCEPTION(argc==1,std::runtime_error,"Error: this test requires a mesh file");
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<Teuchos::MpiComm<int>> Comm = Teuchos::rcp( new Teuchos::MpiComm<int>(MPI_COMM_WORLD) );

  Kokkos::initialize();
  
  
  { // TMW: I think this does need to be scoped for Kokkos reasons
    
    // ==========================================================
    // Create a mesh from the file defined by the user
    // ==========================================================
    
    using namespace Intrepid2;
    int degree = 1;

    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
    {
      std::string input_file_name = argv[1];
      RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    
      Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
      pl->set("File Name",input_file_name);
    
      mesh_factory->setParameterList(pl);
      mesh = mesh_factory->buildUncommitedMesh(*(Comm->getRawMpiComm()));
    
      mesh_factory->completeMeshConstruction(*mesh,*(Comm->getRawMpiComm()));
      if (Comm->getRank() == 0) {
        mesh->printMetaData(std::cout);
      }
  
    }
    
    std::vector<string> blocknames;
    mesh->getElementBlockNames(blocknames);
    int dimension = mesh->getDimension();

    // ==========================================================
    // Create a DOF manager
    // ==========================================================
    
    Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
    Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
    DOF->setConnManager(conn,*(Comm->getRawMpiComm()));
    DOF->setOrientationsRequired(true);
      
    for (size_t b=0; b<blocknames.size(); b++) {
      topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
      string shape = cellTopo->getName();
      { // HGRAD basis
        basis_RCP basis;
        if (dimension == 2) {
          if (shape == "Quadrilateral_4") {
            basis = Teuchos::rcp(new Basis_HGRAD_QUAD_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
          }
          if (shape == "Triangle_3") {
            basis = Teuchos::rcp(new Basis_HGRAD_TRI_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_WARPBLEND) );
          }
        }
        else if (dimension == 3) {
          if (shape == "Hexahedron_8") {
            basis = Teuchos::rcp(new Basis_HGRAD_HEX_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
          }
          if (shape == "Tetrahedron_4") {
            basis = Teuchos::rcp(new Basis_HGRAD_TET_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
          }
        }
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
        DOF->addField(blocknames[b], "T", Pattern, panzer::FieldType::CG);
      }
    }
    DOF->buildGlobalUnknowns();
    if (Comm->getRank() == 0) {
      DOF->printFieldInformation(std::cout);
      std::cout << "================================================" << std::endl << std::endl;
    }
    
    // ==========================================================
    // Create a compressor object
    // ==========================================================
    
    Teuchos::RCP<Compressor> compressor = Teuchos::rcp(new Compressor(Comm, mesh, DOF));
    
    // ==========================================================
    // Compress and report results
    // ==========================================================
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      compressor->buildDatabase(block);
    }
    
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


