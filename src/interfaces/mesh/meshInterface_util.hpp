/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "meshInterface.hpp"
#include "exodusII.h"

using namespace MrHyDE;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV MeshInterface::perturbMesh(const int & b, DRV & blocknodes) {
  
  ////////////////////////////////////////////////////////////////////////////////
  // Perturb the mesh (if requested)
  ////////////////////////////////////////////////////////////////////////////////
  
  //for (size_t b=0; b<block_names.size(); b++) {
    //vector<size_t> localIds;
    //DRV blocknodes;
    //panzer_stk::workset_utils::getIdsAndVertices(*mesh, block_names[b], localIds, blocknodes);
    int numNodesPerElem = blocknodes.extent(1);
    DRV blocknodePert("blocknodePert",blocknodes.extent(0),numNodesPerElem,dimension);
    
    if (settings->sublist("Mesh").get("modify mesh height",false)) {
      vector<vector<ScalarT> > values;
      
      string ptsfile = settings->sublist("Mesh").get("mesh pert file","meshpert.dat");
      std::ifstream fin(ptsfile.c_str());
      
      for (string line; getline(fin, line); )
      {
        replace(line.begin(), line.end(), ',', ' ');
        std::istringstream in(line);
        values.push_back(vector<ScalarT>(std::istream_iterator<ScalarT>(in),
                                         std::istream_iterator<ScalarT>()));
      }
      
      DRV pertdata("pertdata",values.size(),3);
      for (size_t i=0; i<values.size(); i++) {
        for (size_t j=0; j<3; j++) {
          pertdata(i,j) = values[i][j];
        }
      }
      //int Nz = settings->sublist("Mesh").get<int>("NZ",1);
      ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("zmin",0.0);
      ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("zmax",1.0);
      for (size_type k=0; k<blocknodes.extent(0); k++) {
        for (int i=0; i<numNodesPerElem; i++){
          ScalarT x = blocknodes(k,i,0);
          ScalarT y = blocknodes(k,i,1);
          ScalarT z = blocknodes(k,i,2);
          int node = -1;
          ScalarT dist = (ScalarT)RAND_MAX;
          for( size_type j=0; j<pertdata.extent(0); j++ ) {
            ScalarT xhat = pertdata(j,0);
            ScalarT yhat = pertdata(j,1);
            ScalarT d = std::sqrt((x-xhat)*(x-xhat) + (y-yhat)*(y-yhat));
            if( d<dist ) {
              node = j;
              dist = d;
            }
          }
          if (node > 0) {
            ScalarT ch = pertdata(node,2);
            blocknodePert(k,i,0) = 0.0;
            blocknodePert(k,i,1) = 0.0;
            blocknodePert(k,i,2) = (ch)*(z-zmin)/(zmax-zmin);
          }
        }
        //for (int k=0; k<blocknodeVert.extent(0); k++) {
        //  for (int i=0; i<numNodesPerElem; i++){
        //    for (int s=0; s<dimension; s++) {
        //      blocknodeVert(k,i,s) += blocknodePert(k,i,s);
        //    }
        //  }
        //}
      }
    }
    
    if (settings->sublist("Mesh").get("modify mesh",false)) {
      for (size_type k=0; k<blocknodes.extent(0); k++) {
        for (int i=0; i<numNodesPerElem; i++){
          blocknodePert(k,i,0) = 0.0;
          blocknodePert(k,i,1) = 0.0;
          blocknodePert(k,i,2) = 0.0 + 0.2*sin(2*3.14159*blocknodes(k,i,0))*sin(2*3.14159*blocknodes(k,i,1));
        }
      }
      //for (int k=0; k<blocknodeVert.extent(0); k++) {
      //  for (int i=0; i<numNodesPerElem; i++){
      //    for (int s=0; s<dimension; s++) {
      //      blocknodeVert(k,i,s) += blocknodePert(k,i,s);
      //    }
      //  }
      //}
    }
    //nodepert.push_back(blocknodePert);
  //}
  return blocknodePert;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

View_Sc2 MeshInterface::getElementCenters(DRV nodes, topo_RCP & reftopo) {
  
  typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;

  DRV tmp_refCenter("cell center", dimension);
  CellTools::getReferenceCellCenter(tmp_refCenter, *reftopo);
  DRV refCenter("cell center", 1, dimension);
  auto cent_sv = subview(refCenter,0, ALL());
  deep_copy(cent_sv, tmp_refCenter);
  DRV tmp_centers("tmp physical cell centers", nodes.extent(0), 1, dimension);
  CellTools::mapToPhysicalFrame(tmp_centers, refCenter, nodes, *reftopo);
  View_Sc2 centers("physics cell centers", nodes.extent(0), dimension);
  auto tmp_centers_sv = subview(tmp_centers, ALL(), 0, ALL());
  deep_copy(centers, tmp_centers_sv);
  
  return centers;
  
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

View_Sc2 MeshInterface::generateNewMicrostructure(int & randSeed) {
  
  debugger->print("**** Starting mesh::generateNewMicrostructure ...");
  
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  have_rotations = true;
  have_rotation_phi = false;
  
  View_Sc2 seeds;
  random_seeds.push_back(randSeed);
  std::default_random_engine generator(randSeed);
  num_seeds = 0;
  
  ////////////////////////////////////////////////////////////////////////////////
  // Generate the micro-structure using seeds and nearest neighbors
  ////////////////////////////////////////////////////////////////////////////////
  
  bool fast_and_crude = settings->sublist("Mesh").get<bool>("fast and crude microstructure",false);
  
  if (fast_and_crude) {
    int numxSeeds = settings->sublist("Mesh").get<int>("number of xseeds",10);
    int numySeeds = settings->sublist("Mesh").get<int>("number of yseeds",10);
    int numzSeeds = settings->sublist("Mesh").get<int>("number of zseeds",10);
    
    ScalarT xmin = settings->sublist("Mesh").get<ScalarT>("x min",0.0);
    ScalarT ymin = settings->sublist("Mesh").get<ScalarT>("y min",0.0);
    ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("z min",0.0);
    ScalarT xmax = settings->sublist("Mesh").get<ScalarT>("x max",1.0);
    ScalarT ymax = settings->sublist("Mesh").get<ScalarT>("y max",1.0);
    ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("z max",1.0);
    
    ScalarT dx = (xmax-xmin)/(ScalarT)(numxSeeds+1);
    ScalarT dy = (ymax-ymin)/(ScalarT)(numySeeds+1);
    ScalarT dz = (zmax-zmin)/(ScalarT)(numzSeeds+1);
    
    ScalarT maxpert = 0.25;
    
    Kokkos::View<ScalarT*,HostDevice> xseeds("xseeds",numxSeeds);
    Kokkos::View<ScalarT*,HostDevice> yseeds("yseeds",numySeeds);
    Kokkos::View<ScalarT*,HostDevice> zseeds("zseeds",numzSeeds);
    
    for (int k=0; k<numxSeeds; k++) {
      xseeds(k) = xmin + (k+1)*dx;
    }
    for (int k=0; k<numySeeds; k++) {
      yseeds(k) = ymin + (k+1)*dy;
    }
    for (int k=0; k<numzSeeds; k++) {
      zseeds(k) = zmin + (k+1)*dz;
    }
    
    std::uniform_real_distribution<ScalarT> pdistribution(-maxpert,maxpert);
    num_seeds = numxSeeds*numySeeds*numzSeeds;
    seeds = View_Sc2("seeds",num_seeds,3);
    auto seeds_host = create_mirror_view(seeds);
    
    int prog = 0;
    for (int i=0; i<numxSeeds; i++) {
      for (int j=0; j<numySeeds; j++) {
        for (int k=0; k<numzSeeds; k++) {
          ScalarT xp = pdistribution(generator);
          ScalarT yp = pdistribution(generator);
          ScalarT zp = pdistribution(generator);
          seeds_host(prog,0) = xseeds(i) + xp*dx;
          seeds_host(prog,1) = yseeds(j) + yp*dy;
          seeds_host(prog,2) = zseeds(k) + zp*dz;
          prog += 1;
        }
      }
    }
    deep_copy(seeds,seeds_host);
    
  }
  else {
    num_seeds = settings->sublist("Mesh").get<int>("number of seeds",10);
    seeds = View_Sc2("seeds",num_seeds,3);
    auto seeds_host = create_mirror_view(seeds);
    
    ScalarT xwt = settings->sublist("Mesh").get<ScalarT>("x weight",1.0);
    ScalarT ywt = settings->sublist("Mesh").get<ScalarT>("y weight",1.0);
    ScalarT zwt = settings->sublist("Mesh").get<ScalarT>("z weight",1.0);
    ScalarT nwt = sqrt(xwt*xwt+ywt*ywt+zwt*zwt);
    xwt *= 3.0/nwt;
    ywt *= 3.0/nwt;
    zwt *= 3.0/nwt;
    
    ScalarT xmin = settings->sublist("Mesh").get<ScalarT>("x min",0.0);
    ScalarT ymin = settings->sublist("Mesh").get<ScalarT>("y min",0.0);
    ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("z min",0.0);
    ScalarT xmax = settings->sublist("Mesh").get<ScalarT>("x max",1.0);
    ScalarT ymax = settings->sublist("Mesh").get<ScalarT>("y max",1.0);
    ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("z max",1.0);
    
    std::uniform_real_distribution<ScalarT> xdistribution(xmin,xmax);
    std::uniform_real_distribution<ScalarT> ydistribution(ymin,ymax);
    std::uniform_real_distribution<ScalarT> zdistribution(zmin,zmax);
    
    bool wellspaced = settings->sublist("Mesh").get<bool>("well spaced seeds",true);
    if (wellspaced) {
      // we use a relatively crude algorithm to obtain well-spaced points
      int batch_size = 10;
      int prog = 0;
      Kokkos::View<ScalarT**,HostDevice> cseeds("cand seeds",batch_size,3);
      
      while (prog<num_seeds) {
        // fill in the candidate seeds
        for (int k=0; k<batch_size; k++) {
          ScalarT x = xdistribution(generator);
          cseeds(k,0) = x;
          ScalarT y = ydistribution(generator);
          cseeds(k,1) = y;
          ScalarT z = zdistribution(generator);
          cseeds(k,2) = z;
        }
        int bestpt = 0;
        if (prog > 0) { // for prog = 0, just take the first one
          ScalarT maxdist = 0.0;
          for (int k=0; k<batch_size; k++) {
            ScalarT cmindist = 1.0e200;
            for (int j=0; j<prog; j++) {
              ScalarT dx = cseeds(k,0)-seeds(j,0);
              ScalarT dy = cseeds(k,1)-seeds(j,1);
              ScalarT dz = cseeds(k,2)-seeds(j,2);
              ScalarT cval = xwt*dx*dx + ywt*dy*dy + zwt*dz*dz;
              if (cval < cmindist) {
                cmindist = cval;
              }
            }
            if (cmindist > maxdist) {
              maxdist = cmindist;
              bestpt = k;
            }
          }
        }
        for (int j=0; j<3; j++) {
          seeds_host(prog,j) = cseeds(bestpt,j);
        }
        prog += 1;
      }
    }
    else {
      for (int k=0; k<num_seeds; k++) {
        ScalarT x = xdistribution(generator);
        seeds_host(k,0) = x;
        ScalarT y = ydistribution(generator);
        seeds_host(k,1) = y;
        ScalarT z = zdistribution(generator);
        seeds_host(k,2) = z;
      }
    }
    deep_copy(seeds, seeds_host);
    
  }
  //KokkosTools::print(seeds);
  
  meshimporttimer.stop();
  if (verbosity>5 && comm->getRank() == 0) {
    cout << "microstructure regeneration time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  debugger->print("**** Finished mesh::generateNewMicrostructure ...");
  
  return seeds;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV MeshInterface::getElemNodes(const int & block, const int & elemID) {
  vector<size_t> localIds;
  DRV blocknodes;
  int nnodes = 0;
  if (use_stk_mesh) {
    panzer_stk::workset_utils::getIdsAndVertices(*stk_mesh, block_names[block], localIds, blocknodes);
    nnodes = blocknodes.extent(1);
  }
  else if (use_simple_mesh) {
    //nnodes = simple_mesh->getNumNodes();
    //blocknodes = simple_mesh->getCellNodes({elemID});
  }
  
  DRV cnodes("element nodes",1,nnodes,dimension);
  for (int i=0; i<nnodes; i++) {
    for (int j=0; j<dimension; j++) {
      cnodes(0,i,j) = blocknodes(elemID,i,j);
    }
  }
  return cnodes;
}

DRV MeshInterface::getMyNodes(const size_t & block, vector<size_t> & elemIDs) {
  
  DRV currnodes("current nodes", elemIDs.size(), num_nodes_per_elem, dimension);
  this->getSTKElementVertices(elemIDs, block_names[block], currnodes);
  return currnodes;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<string> MeshInterface::breakupList(const string & list, const string & delimiter) {
  // Script to break delimited list into pieces
  string tmplist = list;
  vector<string> terms;
  size_t pos = 0;
  if (tmplist.find(delimiter) == string::npos) {
    terms.push_back(tmplist);
  }
  else {
    string token;
    while ((pos = tmplist.find(delimiter)) != string::npos) {
      token = tmplist.substr(0, pos);
      terms.push_back(token);
      tmplist.erase(0, pos + delimiter.length());
    }
    terms.push_back(tmplist);
  }
  return terms;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we might be able to get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

void MeshInterface::purgeMesh() {
  stk_mesh = Teuchos::null;
  mesh_factory = Teuchos::null;
  simple_mesh = Teuchos::null;
}

// ============================================================
// ============================================================

void MeshInterface::purgeMemory() {
  nfield_vals.clear();
  efield_vals.clear();
  meas = Teuchos::null;
}

// ============================================================
// ============================================================

vector<string> MeshInterface::getBlockNames() {
  return block_names;
}

// ============================================================
// ============================================================

vector<string> MeshInterface::getSideNames() {
  return side_names;
}

// ============================================================
// ============================================================

vector<string> MeshInterface::getNodeNames() {
  return node_names;
}

// ============================================================
// ============================================================

int MeshInterface::getDimension() {
  return dimension;
}

// ============================================================
// ============================================================

topo_RCP MeshInterface::getCellTopology(string & blockID) {
  topo_RCP currtopo;
  for (size_t blk=0; blk<block_names.size(); ++blk) {
    if (block_names[blk] == blockID) {
      currtopo = cell_topo[blk];
    }
  }
  return currtopo;
}
  
// ============================================================
// ============================================================

void MeshInterface::allocateMeshDataStructures() {
  if (use_simple_mesh) {
       simple_mesh->allocateDataStructures();
  }
}

// ============================================================
// ============================================================

void MeshInterface::purgeMaps(){
  if (use_simple_mesh) {
       simple_mesh->deallocateMaps();
  }
}
