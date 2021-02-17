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

#include "sensorManager.hpp"

using namespace MrHyDE;

// ========================================================================================
// Constructor
// ========================================================================================

template<class Node>
SensorManager<Node>::SensorManager(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                   Teuchos::RCP<meshInterface> & mesh_,
                                   Teuchos::RCP<AssemblyManager<Node> > & assembler_) :
settings(settings_), mesh(mesh_), assembler(assembler_) {
  
  debug_level = settings->get<int>("debug level",0);
  verbosity = settings->get<int>("verbosity",0);
  
  if (debug_level > 0) {
    if (assembler->Comm->getRank() == 0) {
      cout << "**** Starting SensorManager::constructor ..." << endl;
    }
  }
  
  spaceDim = mesh->mesh->getDimension();
  have_sensor_data = false;
  have_sensor_points = false;
  numSensors = 0;
  
  if (settings->sublist("Mesh").get<bool>("have element data", false)) {
    this->importSensorsFromExodus();
  }
  else {
    this->importSensorsFromFiles();
    
    
  }
  
  if (debug_level > 0) {
    if (assembler->Comm->getRank() == 0) {
      cout << "**** Finished SensorManager::constructor ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SensorManager<Node>::importSensorsFromExodus() {
  for (size_t i=0; i<assembler->cells[0].size(); i++) {
    vector<Kokkos::View<ScalarT**,HostDevice> > sensorLocations;
    vector<Kokkos::View<ScalarT**,HostDevice> > sensorData;
    int numSensorsInCell = mesh->efield_vals[0][i];
    
    if (numSensorsInCell > 0) {
      assembler->cells[0][i]->mySensorIDs.push_back(numSensors); // hack for dakota
      for (int j=0; j<numSensorsInCell; j++) {
        // sensorLocation
        Kokkos::View<ScalarT**,HostDevice> sensor_loc("sensor location",1,spaceDim);
        std::stringstream ssSensorNum;
        ssSensorNum << j+1;
        string sensorNum = ssSensorNum.str();
        string fieldLocx = "sensor_" + sensorNum + "_Loc_x";
        ptrdiff_t ind_Locx = std::distance(mesh->efield_names.begin(),
                                           std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocx));
        string fieldLocy = "sensor_" + sensorNum + "_Loc_y";
        ptrdiff_t ind_Locy = std::distance(mesh->efield_names.begin(),
                                           std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocy));
        sensor_loc(0,0) = mesh->efield_vals[ind_Locx][i];
        sensor_loc(0,1) = mesh->efield_vals[ind_Locy][i];
        if (spaceDim > 2) {
          string fieldLocz = "sensor_" + sensorNum + "_Loc_z";
          ptrdiff_t ind_Locz = std::distance(mesh->efield_names.begin(),
                                             std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocz));
          sensor_loc(0,2) = mesh->efield_vals[ind_Locz][i];
        }
        // sensorData
        Kokkos::View<ScalarT**,HostDevice> sensor_data("sensor data",1,mesh->numResponses+1);
        sensor_data(0,0) = 0.0; // time index
        for (int k=1; k<mesh->numResponses+1; k++) {
          std::stringstream ssRespNum;
          ssRespNum << k;
          string respNum = ssRespNum.str();
          string fieldResp = "sensor_" + sensorNum + "_Val_" + respNum;
          ptrdiff_t ind_Resp = std::distance(mesh->efield_names.begin(),
                                             std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldResp));
          sensor_data(0,k) = mesh->efield_vals[ind_Resp][i];
        }
        sensorLocations.push_back(sensor_loc);
        sensorData.push_back(sensor_data);
        numSensors += 1; // solver variable (total number of sensors)
      }
    }
    assembler->cells[0][i]->cellData->exodus_sensors = true;
    assembler->cells[0][i]->numSensors = numSensorsInCell;
    assembler->cells[0][i]->sensorLocations = sensorLocations;
    assembler->cells[0][i]->sensorData = sensorData;
  }
  
  Kokkos::View<ScalarT**,HostDevice> tmp_sensor_points;
  vector<Kokkos::View<ScalarT**,HostDevice> > tmp_sensor_data;
  //bool have_sensor_data = true;
  ScalarT sensor_loc_tol = 1.0;
  // only needed for passing of basis pointers
  
  assembler->cellData[0]->response_type = "pointwise";
  //bool useFineScale = true;
  //if (!(assembler->cellData[0]->multiscale) || assembler->cellData[0]->mortar_objective) {
  //  useFineScale = false;
  //}
  for (size_t j=0; j<assembler->cells[0].size(); j++) {
    //assembler->cells[0][j]->addSensors(sensor_points, sensor_loc_tol, sensor_data, have_sensor_data, disc, disc->basis_pointers[0], params->discretized_param_basis);
    assembler->cells[0][j]->useSensors = true;
    // don't use sensor_points
    // set sensorData and sensorLocations from exodus file
    if (assembler->cells[0][j]->sensorLocations.size() > 0) {
      DRV sensorPoints_drv("sensorPoints",1,assembler->cells[0][j]->sensorLocations.size(),spaceDim);
      auto sp_host = Kokkos::create_mirror_view(sensorPoints_drv);
      for (size_t i=0; i<assembler->cells[0][j]->sensorLocations.size(); i++) {
        for (int k=0; k<spaceDim; k++) {
          sp_host(0,i,k) = assembler->cells[0][j]->sensorLocations[i](0,k);
        }
        assembler->cells[0][j]->sensorElem.push_back(0);
      }
      Kokkos::deep_copy(sensorPoints_drv, sp_host);
      assembler->cells[0][j]->sensorPoints = View_Sc3("sensorPoints",1,assembler->cells[0][j]->sensorLocations.size(),spaceDim);
      Kokkos::deep_copy(assembler->cells[0][j]->sensorPoints, sensorPoints_drv);
      DRV refsenspts_buffer = assembler->disc->mapPointsToReference(sensorPoints_drv,assembler->cells[0][j]->nodes,assembler->cellData[0]->cellTopo);
      DRV refsenspts("refsenspts",assembler->cells[0][j]->sensorLocations.size(),spaceDim);
      Kokkos::deep_copy(refsenspts,Kokkos::subdynrankview(refsenspts_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
      
      vector<DRV> csensorBasis;
      vector<DRV> csensorBasisGrad;
      
      for (size_t b=0; b<assembler->disc->basis_pointers[0].size(); b++) {
        csensorBasis.push_back(assembler->disc->evaluateBasis(assembler->disc->basis_pointers[0][b], refsenspts, assembler->cells[0][j]->orientation));
        csensorBasisGrad.push_back(assembler->disc->evaluateBasisGrads(assembler->disc->basis_pointers[0][b], assembler->cells[0][j]->nodes, refsenspts,
                                                                       assembler->cellData[0]->cellTopo, assembler->cells[0][j]->orientation));
      }
      
      assembler->cells[0][j]->sensorBasis.push_back(csensorBasis);
      assembler->cells[0][j]->sensorBasisGrad.push_back(csensorBasisGrad);
      
      
      vector<DRV> cpsensorBasis;
      vector<DRV> cpsensorBasisGrad;
      
      for (size_t b=0; b<assembler->params->discretized_param_basis.size(); b++) {
        cpsensorBasis.push_back(assembler->disc->evaluateBasis(assembler->params->discretized_param_basis[b], refsenspts, assembler->cells[0][j]->orientation));
        cpsensorBasisGrad.push_back(assembler->disc->evaluateBasisGrads(assembler->params->discretized_param_basis[b], assembler->cells[0][j]->nodes,
                                                                        refsenspts, assembler->cellData[0]->cellTopo, assembler->cells[0][j]->orientation));
      }
      
      assembler->cells[0][j]->param_sensorBasis.push_back(cpsensorBasis);
      assembler->cells[0][j]->param_sensorBasisGrad.push_back(cpsensorBasisGrad);
    }
    
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SensorManager<Node>::importSensorsFromFiles() {
  if (settings->sublist("Analysis").get("have sensor data",false)) {
    data sdata("Sensor Measurements", spaceDim,
               settings->sublist("Analysis").get("sensor location file","sensor_points.dat"),
               settings->sublist("Analysis").get("sensor prefix","sensor"));
    sensor_data = sdata.getdata();
    sensor_points = sdata.getpoints();
    numSensors = sensor_points.extent(0);
    have_sensor_data = true;
    have_sensor_points = true;
  }
  else if (settings->sublist("Analysis").get("have sensor points",false)) {
    data sdata("Sensor Points", spaceDim,
               settings->sublist("Analysis").get("sensor location file","sensor_points.dat"));
    sensor_points = sdata.getpoints();
    numSensors = sensor_points.extent(0);
    have_sensor_data = false;
    have_sensor_points = true;
  }
  
  if (settings->sublist("Analysis").get("have sensor points",false)) {
    
    ScalarT sensor_loc_tol = settings->sublist("Analysis").get("sensor location TOL",1.0E-6);
    for (size_t b=0; b<assembler->cells.size(); b++) {
      // If we have sensors, then we set the response type to pointwise
      assembler->cellData[b]->response_type = "pointwise";
      bool useFineScale = true;
      if (!(assembler->cellData[b]->multiscale) || assembler->cellData[b]->mortar_objective) {
        useFineScale = false;
      }
      
      
      for (size_t j=0; j<assembler->cells[b].size(); j++) {
        //assembler->cells[b][j]->addSensors(sensor_points, sensor_loc_tol, sensor_data, have_sensor_data, disc, disc->basis_pointers[b], params->discretized_param_basis);
        
        assembler->cells[b][j]->useSensors = true;
                
        if (useFineScale) {
          
          for (size_t i=0; i<assembler->cells[b][j]->subgridModels.size(); i++) {
            //if (subgrid_model_index[0] == i) {
            assembler->cells[b][j]->subgridModels[i]->addSensors(sensor_points,sensor_loc_tol,sensor_data,have_sensor_data,
                                                                 assembler->disc->basis_pointers[b], assembler->cells[b][j]->subgrid_usernum);
            //}
          }
          
        }
        else {
          DRV phys_points("phys_points",1,sensor_points.extent(0),sensor_points.extent(1));
          auto pp_sub = subview(phys_points,0,ALL(),ALL());
          Kokkos::deep_copy(pp_sub,sensor_points);
          
          if (!(assembler->cellData[b]->loadSensorFiles)) {
            for (size_t e=0; e<assembler->cells[b][j]->numElem; e++) {
              auto nodes = assembler->cells[b][j]->nodes;
              DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
              auto n_sub = subview(nodes,e,ALL(),ALL());
              auto cn_sub = subview(cnodes,0,ALL(),ALL());
              Kokkos::deep_copy(cn_sub,n_sub);
              
              auto inRefCell = assembler->disc->checkInclusionPhysicalData(phys_points,cnodes,
                                                                           assembler->cellData[b]->cellTopo,
                                                                           sensor_loc_tol);
              
              //DRV refpts("refpts", 1, sensor_points.extent(0), sensor_points.extent(1));
              //Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell", 1, sensor_points.extent(0));
              
              //CellTools::mapToReferenceFrame(refpts, phys_points, cnodes, *(cellData->cellTopo));
              //CellTools::checkPointwiseInclusion(inRefCell, refpts, *(cellData->cellTopo), sensor_loc_tol);
              
              for (size_type i=0; i<sensor_points.extent(0); i++) {
                if (inRefCell(0,i) == 1) {
                  
                  Kokkos::View<ScalarT**,HostDevice> newsenspt("new sensor point",1,assembler->cellData[b]->dimension);
                  for (size_t j=0; j<assembler->cellData[b]->dimension; j++) {
                    newsenspt(0,j) = sensor_points(i,j);
                  }
                  assembler->cells[b][j]->sensorLocations.push_back(newsenspt);
                  assembler->cells[b][j]->mySensorIDs.push_back(i);
                  assembler->cells[b][j]->sensorElem.push_back(e);
                  if (have_sensor_data) {
                    assembler->cells[b][j]->sensorData.push_back(sensor_data[i]);
                  }
                  if (assembler->cellData[b]->writeSensorFiles) {
                    std::stringstream ss;
                    ss << assembler->cells[b][j]->localElemID(e);
                    string str = ss.str();
                    string fname = "sdat." + str + ".dat";
                    std::ofstream outfile(fname.c_str());
                    outfile.precision(8);
                    outfile << i << "  ";
                    outfile << sensor_points(i,0) << "  " << sensor_points(i,1) << "  ";
                    //outfile << sensor_data[i](0,0) << "  " << sensor_data[i](0,1) << "  " << sensor_data[i](0,2) << "  " ;
                    outfile << endl;
                    outfile.close();
                  }
                }
              }
            }
          }
          
          if (assembler->cellData[b]->loadSensorFiles) {
            for (size_t e=0; e<assembler->cells[b][j]->numElem; e++) {
              std::stringstream ss;
              ss << assembler->cells[b][j]->localElemID(e);
              string str = ss.str();
              std::ifstream sfile;
              sfile.open("sensorLocations/sdat." + str + ".dat");
              int cID;
              //ScalarT l1, l2, t1, d1, d2;
              ScalarT l1, l2;
              sfile >> cID;
              sfile >> l1;
              sfile >> l2;
              
              sfile.close();
              
              Kokkos::View<ScalarT**,HostDevice> newsenspt("sensor point",1,assembler->cellData[b]->dimension);
              //FC newsensdat(1,3);
              newsenspt(0,0) = l1;
              newsenspt(0,1) = l2;
              assembler->cells[b][j]->sensorLocations.push_back(newsenspt);
              assembler->cells[b][j]->mySensorIDs.push_back(cID);
              assembler->cells[b][j]->sensorElem.push_back(e);
            }
          }
          
          assembler->cells[b][j]->numSensors = assembler->cells[b][j]->sensorLocations.size();
          
          // Evaluate the basis functions and derivatives at sensor points
          if (assembler->cells[b][j]->numSensors > 0) {
            assembler->cells[b][j]->sensorPoints = View_Sc3("sensorPoints",assembler->cells[b][j]->numElem,
                                                            assembler->cells[b][j]->numSensors,
                                                            assembler->cellData[b]->dimension);
            auto sp_host = Kokkos::create_mirror_view(assembler->cells[b][j]->sensorPoints);
            for (size_t i=0; i<assembler->cells[b][j]->numSensors; i++) {
              
              DRV csensorPoints("sensorPoints",1,1,assembler->cellData[b]->dimension);
              DRV cnodes("current nodes",1,
                         assembler->cells[b][j]->nodes.extent(1),
                         assembler->cells[b][j]->nodes.extent(2));
              for (size_t j=0; j<assembler->cellData[b]->dimension; j++) {
                csensorPoints(0,0,j) = assembler->cells[b][j]->sensorLocations[i](0,j);
                sp_host(0,i,j) = assembler->cells[b][j]->sensorLocations[i](0,j);
                for (size_type k=0; k<assembler->cells[b][j]->nodes.extent(1); k++) {
                  cnodes(0,k,j) = assembler->cells[b][j]->nodes(assembler->cells[b][j]->sensorElem[i],k,j);
                }
              }
              
              
              //DRV refsenspts_buffer("refsenspts_buffer",1,1,assembler->cellData[b]->dimension);
              DRV refsenspts("refsenspts",1,assembler->cellData[b]->dimension);
              
              DRV refsenspts_buffer = assembler->disc->mapPointsToReference(csensorPoints,cnodes,
                                                                            assembler->cellData[b]->cellTopo);
              
              //CellTools::mapToReferenceFrame(refsenspts_buffer, csensorPoints, cnodes, *(assembler->cellData[b]->cellTopo));
              //CellTools<AssemblyDevice>::mapToReferenceFrame(refsenspts, csensorPoints, cnodes, *cellTopo);
              Kokkos::deep_copy(refsenspts,Kokkos::subdynrankview(refsenspts_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
              
              vector<DRV> csensorBasis;
              vector<DRV> csensorBasisGrad;
              
              for (size_t k=0; k<assembler->disc->basis_pointers[b].size(); k++) {
                csensorBasis.push_back(assembler->disc->evaluateBasis(assembler->disc->basis_pointers[b][k], refsenspts, assembler->cells[b][j]->orientation));
                csensorBasisGrad.push_back(assembler->disc->evaluateBasisGrads(assembler->disc->basis_pointers[b][k], cnodes,
                                                                               refsenspts, assembler->cellData[b]->cellTopo, assembler->cells[b][j]->orientation));
              }
              assembler->cells[b][j]->sensorBasis.push_back(csensorBasis);
              assembler->cells[b][j]->sensorBasisGrad.push_back(csensorBasisGrad);
              
              
              vector<DRV> cpsensorBasis;
              vector<DRV> cpsensorBasisGrad;
              
              for (size_t b=0; b<assembler->params->discretized_param_basis.size(); b++) {
                cpsensorBasis.push_back(assembler->disc->evaluateBasis(assembler->params->discretized_param_basis[b], refsenspts,
                                                            assembler->cells[b][j]->orientation));
                cpsensorBasisGrad.push_back(assembler->disc->evaluateBasisGrads(assembler->params->discretized_param_basis[b],
                                                                                assembler->cells[b][j]->nodes, refsenspts,
                                                                                assembler->cellData[b]->cellTopo,
                                                                                assembler->cells[b][j]->orientation));
              }
              
              assembler->cells[b][j]->param_sensorBasis.push_back(cpsensorBasis);
              assembler->cells[b][j]->param_sensorBasisGrad.push_back(cpsensorBasisGrad);
            }
            Kokkos::deep_copy(assembler->cells[b][j]->sensorPoints,sp_host);
            
          }
        }
        
      }
    }
  }
}

