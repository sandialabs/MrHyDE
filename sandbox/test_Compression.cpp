
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
      Comm(Comm_), mesh(mesh_), DOF(DOF_),
      ignore_orientations(false), compute_scaling(false), compute_rotation(false),
      database_TOL(1.0e-12) {

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
  
      vector<Kokkos::View<ScalarT***,HostDevice>> db_jacobians;
      vector<ScalarT> db_rotations;      

      // scaling factors for x,y,z axes
      DRV scale;
      if(compute_scaling) {
        scale = DRV("scale", numElem[block], dimension);
      }

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
      // Now we actually determine the unique elements
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

        // GH: Okay, I'm going to duplicate some loops in if statements for performance, but here's the big picture.
        // We're skipping orientations for now because we may be able to cut down duplication by hitting orientations with rotations.
        // Check #1. Is ||J_A| - |J_B|| < database_TOL?
        //   If yes, go to check #2, leaving scaling constants as the identity.
        //   If no, rescale J_A by diagonal D so that diag(D*J_A) == diag(J_B), and check again.
        //     If ||D*J_A| - |J_B|| < database_TOL after the scaling, continue to check #2 but replace J_A in subsequent steps by D*J_A
        // Check #2. Is |J_A-J_B|_{l1} < database_TOL?
        //   If yes, set found=true, leaving theta=0.
        //   If no, go to check #3.
        // Check #3. Is there a rotation matrix R such that |R*J_A - J_B|_{l1} < database_TOL?
        // This can be done by checking is there a rotation matrix R such that |R - J_B inv(J_A)|_{l1} < database_TOL (this may bite us if |J_A|!=1?)
        // As a surrogate, we find theta so |R - M|_F is minimized (M takes the place of J_B*inv(J_A)).
        // If M has entries [a,b;c,d], theta=atan((c-b)/(a+d)) minimizes that objective.
        //   If yes, set found=true, theta=atan((c-b)/(a+d))
        //   If no, we have not found a match. Move on to the next database entry.

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
              ScalarT diff2 = compute_jacobian_diff(jacobian,db_jacobians[prog],block);
              
              if (diff2/refnorm<database_TOL) {
                found = true;
              }
              else { // Check #3 failed
                if(compute_scaling) {
                  ScalarT measure_scale = 1;
                  for(size_type d0=0; d0<dimension; ++d0) {
                    ScalarT rowsum = 0;
                    ScalarT dbrowsum = 0;
                    for(size_type d1=0; d1<dimension; ++d1) {
                      rowsum += jacobian(0,0,d0,d1);
                      dbrowsum += db_jacobians[prog](0,d0,d1); 
                    }
                    scale(e,d0) = dbrowsum/rowsum; // scale the diagonal entries to match at ip 0
                    measure_scale *= scale(e,d0);
                  }
                  
                  // Check #2 scaled: element measures
                  ScalarT scaled_diff = std::abs(measure_scale*measure-db_measures[prog]); // compute the diff again after scaling the Jacobian
                  if (std::abs(scaled_diff/db_measures[prog])<database_TOL) { // abs(measure) is probably unnecessary here
                    ScalarT refnorm = db_jacobian_norms[prog];

                    // Check #3 scaled: element Jacobians
                    ScalarT scaled_diff = std::abs(measure_scale*measure-db_measures[prog]); // compute the diff again after scaling the Jacobian
                    
                    if (diff2/refnorm<database_TOL) {
                      found = true;
                    } else { // Check scaled #3 failed
                      ++prog;
                    }
                  } else { // Check scaled #2 failed
                    ++prog;
                  }
                } else { // Check #3 failed above and no scaling
                  ++prog;
                }
              }
            } else { // Check #2 failed
              // if Check #2 failed and we want diagonal scaling, check if diagonal scaling is possible
              if(compute_scaling) {
                ScalarT measure_scale = 1;
                for(size_type d0=0; d0<dimension; ++d0) {
                  ScalarT rowsum = 0;
                  ScalarT dbrowsum = 0;
                  for(size_type d1=0; d1<dimension; ++d1) {
                    rowsum += jacobian(0,0,d0,d1);
                    dbrowsum += db_jacobians[prog](0,d0,d1); 
                  }
                  scale(e,d0) = dbrowsum/rowsum; // scale the diagonal entries to match at ip 0
                  measure_scale *= scale(e,d0);
                }

                // Check #2 scaled: element measures
                ScalarT scaled_diff = std::abs(measure_scale*measure-db_measures[prog]); // compute the diff again after scaling the Jacobian
                if (std::abs(scaled_diff/db_measures[prog])<database_TOL) { // abs(measure) is probably unnecessary here
                  ScalarT refnorm = db_jacobian_norms[prog];

                  // Check #3 scaled: element Jacobians
                  ScalarT diff2 = compute_scaled_jacobian_diff(jacobian, db_jacobians[prog], scale, e, block);
                
                  if (diff2/refnorm<database_TOL) {
                    found = true;
                  } else { // Check scaled #3 failed
                    ++prog;
                  }
                } else { // Check scaled #2 failed
                  ++prog;
                }
              } else { // Check #2 failed above and no scaling
                ++prog;
              }
            }
          } else { // Check #1 failed
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

      for(size_type idb=0; idb<db_jacobians.size(); ++idb) {
        std::cout << "J = [";
        for(size_type d0=0; d0<dimension; ++d0) {
          for(size_type d1=0; d1<dimension-1; ++d1) {
            std::cout << db_jacobians[idb](0,d0,d1) << ", ";
          }
          std::cout << db_jacobians[idb](0,d0,dimension-1) << "\n     ";
        }
        std::cout << "]" << std::endl;
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    void buildVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users) {
      // this is used in MrHyDE, but it isn't clear if we need it here
    }
  
    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    void setIgnoreOrientations(const bool ignoreOrientations) { ignore_orientations = ignoreOrientations; }
    void setComputeScaling(const bool computeScaling) { compute_scaling = computeScaling; }
    void setComputeRotation(const bool computeRotation) { compute_rotation = computeRotation; }
    void setTolerance(const double tolerance) { database_TOL = tolerance; }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    //! The database tolerance
    double database_TOL;
    //! Boolean flag whether to ignore Intrepid2's orientations or not
    bool ignore_orientations;
    //! Boolean flag whether to compute diagonal scaling factors or not
    bool compute_scaling;
    //! Scaling factors when scaling compression is considered (size: num_elems-by-dim)
    vector<DRV> scaling;
    //! Boolean flag whether to compute rotation angles or not
    bool compute_rotation;
    //! Rotation angles when rotational compression is considered (size: num_elems)
    vector<DRV> rotation;

    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
    Teuchos::RCP<panzer::DOFManager> DOF;
    std::vector<string> blocknames;
    vector<vector<size_t>> myElements;
    vector<DRV> nodes, ref_ip, ref_wts;
    std::vector<Intrepid2::Orientation> orientations;
    vector<size_t> numElem, numip;
    int dimension;

  private:
    //! Compute the l1 difference of two Jacobians across all integration points
    inline ScalarT compute_jacobian_diff(const DRV &jac, const Kokkos::View<ScalarT***,HostDevice> &database_jac, const size_t &block) {
      ScalarT diff2 = 0;
      size_type pt=0;
      while (pt<numip[block] && diff2<database_TOL) {
        size_type d0=0;
        while (d0<dimension && diff2<database_TOL) {
          size_type d1=0;
          while (d1<dimension && diff2<database_TOL) { 
            diff2 += std::abs(jac(0,pt,d0,d1) - database_jac(pt,d0,d1));
            d1++;
          }
          d0++;
        }
        pt++;
      }
      return diff2;
    }
    
    //! Compute the l1 difference of two Jacobians across all integration points using a diagonal scaling
    inline ScalarT compute_scaled_jacobian_diff(const DRV &jac, const Kokkos::View<ScalarT***,HostDevice> &database_jac, const DRV &scale, const size_t &e, const size_t &block) {
      ScalarT diff2 = 0;
      size_type pt=0;
      while (pt<numip[block] && diff2<database_TOL) {
        size_type d0=0;
        while (d0<dimension && diff2<database_TOL) {
          size_type d1=0;
          while (d1<dimension && diff2<database_TOL) { 
            diff2 += std::abs(scale(e,d0)*jac(0,pt,d0,d1) - database_jac(pt,d0,d1));
            d1++;
          }
          d0++;
        }
        pt++;
      }
      return diff2;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
        
int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<Teuchos::MpiComm<int>> comm = Teuchos::rcp( new Teuchos::MpiComm<int>(MPI_COMM_WORLD) );

  Kokkos::initialize();
  
  { // TMW: I think this does need to be scoped for Kokkos reasons
    
    // GH: using Teuchos CommandLineProcessor for more robust input customization
    // begin command line parsing
    Teuchos::CommandLineProcessor clp(false);
    std::string input_file_name = "hex-reference.exo";
    bool compute_scaling = false;
    bool compute_rotation = false; // TODO: only works in 2D at the moment
    bool ignore_orientations = false;
    double tol = 1e-12;
    bool verbose = false;

    // add options and reference 
    clp.setOption("mesh", &input_file_name,                                      "Name of the mesh (default: hex-reference.exo)");
    clp.setOption("compute-scaling", "no-compute-scaling", &compute_scaling,     "Whether to compute compression based on a diagonal scaling (default: false)");
    clp.setOption("compute-rotation", "no-compute-rotation", &compute_rotation,  "Whether to compute compression based on a rotation (default: false)");
    clp.setOption("ignore-orientations", "no-ignore-orientations", &ignore_orientations,  "Whether to ignore orientations when compressing (default: false)");
    clp.setOption("tol", &tol,                                                   "Compression tolerance (default: 1e-12)");
    clp.setOption("verbose", "no-verbose", &verbose,                             "Verbose output (default: false)");
    
    clp.recogniseAllOptions(true);
    switch (clp.parse(argc, argv)) {
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED: 
        return EXIT_SUCCESS;
      case Teuchos::CommandLineProcessor::PARSE_ERROR:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: 
        return EXIT_FAILURE;
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
        break;
    }
    // end of command line parsing

    if(comm->getRank() == 0) {
      std::cout << "Running compression sandbox...\n"
                << "Input settings:" << "\n"
                << "    mesh = " << input_file_name << "\n"
                << "    compute-scaling = "<< compute_scaling << "\n"
                << "    compute-rotation = " << compute_rotation << "\n"
                << "    ignore-orientations = " << ignore_orientations << "\n"
                << "    tol = " << tol << "\n"
                << "    verbose = " << verbose << std::endl; 
    }

    // ==========================================================
    // Create a mesh from the file defined by the user
    // ==========================================================
    
    using namespace Intrepid2;
    int degree = 1;

    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
    {
      RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    
      Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
      pl->set("File Name",input_file_name);
    
      mesh_factory->setParameterList(pl);
      mesh = mesh_factory->buildUncommitedMesh(*(comm->getRawMpiComm()));
    
      mesh_factory->completeMeshConstruction(*mesh,*(comm->getRawMpiComm()));
      if (comm->getRank() == 0) {
        mesh->printMetaData(std::cout);
        std::cout << "   dimension = " << mesh->getDimension() << std::endl;
      }
    }
    
    std::vector<string> blocknames;
    mesh->getElementBlockNames(blocknames);
    int dimension = mesh->getDimension();

    TEUCHOS_TEST_FOR_EXCEPTION(compute_rotation && (dimension != 2), std::runtime_error, "Error: compute_rotation=true but mesh dimension=3. This is not yet implemented!");

    // ==========================================================
    // Create a DOF manager
    // ==========================================================
    
    Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
    Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
    DOF->setConnManager(conn,*(comm->getRawMpiComm()));
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
    if (comm->getRank() == 0) {
      DOF->printFieldInformation(std::cout);
      std::cout << "================================================" << std::endl << std::endl;
    }
    
    // ==========================================================
    // Create a compressor object
    // ==========================================================
    
    Teuchos::RCP<Compressor> compressor = Teuchos::rcp(new Compressor(comm, mesh, DOF));
    compressor->setComputeRotation(compute_rotation);
    compressor->setComputeScaling(compute_scaling);
    compressor->setIgnoreOrientations(ignore_orientations);
    compressor->setTolerance(tol);
    
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


