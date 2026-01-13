/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::addSensors() {

  debugger->print("**** Starting PostprocessManager::addSensors ...");

  // Reading in sensors from a mesh file only works on a single element block (for now)
  // There isn't any problem with multiple blocks, it just hasn't been generalized for sensors yet
  for (size_t r = 0; r < objectives.size(); ++r) {
    if (objectives[r].type == "sensors") {
      if (objectives[r].sensor_points_file == "mesh") {
        // Teuchos::TimeMonitor localtimer(*importexodustimer);
        this->importSensorsFromExodus(r);
      }
      else if (objectives[r].use_sensor_grid) {
        // Teuchos::TimeMonitor localtimer(*importexodustimer);
        this->importSensorsOnGrid(r);
      }
      else if (objectives[r].use_quadrature_pts) {
        // Teuchos::TimeMonitor localtimer(*importexodustimer);
        this->importSensorsOnQuadrature(r);
      }
      else if (objectives[r].use_bndry_quadrature_pts) {
        // Teuchos::TimeMonitor localtimer(*importexodustimer);
        this->importSensorsOnBndryQuadrature(r);
      }
      else {
        // Teuchos::TimeMonitor localtimer(*importfiletimer);
        this->importSensorsFromFiles(r);
      }
    }
  }

  debugger->print("**** Finished PostprocessManager::addSensors ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::importSensorsFromExodus(const int &objID)
{

  debugger->print("**** Starting PostprocessManager::importSensorsFromExodus() ...");

  // vector<string> mesh_response_names = mesh->efield_names;
  string cresp = objectives[objID].sensor_data_file;

  size_t block = objectives[objID].block;

  int numFound = 0;
  for (size_t i = 0; i < assembler->groups[block].size(); i++)
  {
    int numSensorsIngrp = mesh->efield_vals[block][i];
    numFound += numSensorsIngrp;
  }

  objectives[objID].numSensors = numFound;

  if (numFound > 0)
  {

    Kokkos::View<ScalarT **, HostDevice> spts_host("exodus sensors on host", numFound, dimension);
    Kokkos::View<int *[2], HostDevice> spts_owners("exodus sensor owners", numFound);

    // TMW: as far as I can tell, this is limited to steady-state data
    Kokkos::View<ScalarT *, HostDevice> stime_host("sensor times", 1);
    stime_host(0) = 0.0;
    Kokkos::View<ScalarT **, HostDevice> sdat_host("sensor data", numFound, 1);

    size_t sprog = 0;
    for (size_t i = 0; i < assembler->groups[block].size(); i++)
    {
      int numSensorsIngrp = mesh->efield_vals[block][i];

      if (numSensorsIngrp > 0)
      {
        for (int j = 0; j < numSensorsIngrp; j++)
        {
          // sensorLocation
          std::stringstream ssSensorNum;
          ssSensorNum << j + 1;
          string sensorNum = ssSensorNum.str();
          string fieldLocx = "sensor_" + sensorNum + "_Loc_x";
          ptrdiff_t ind_Locx = std::distance(mesh->efield_names.begin(),
                                             std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocx));
          spts_host(sprog, 0) = mesh->efield_vals[ind_Locx][i];

          if (dimension > 1)
          {
            string fieldLocy = "sensor_" + sensorNum + "_Loc_y";
            ptrdiff_t ind_Locy = std::distance(mesh->efield_names.begin(),
                                               std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocy));
            spts_host(sprog, 1) = mesh->efield_vals[ind_Locy][i];
          }
          if (dimension > 2)
          {
            string fieldLocz = "sensor_" + sensorNum + "_Loc_z";
            ptrdiff_t ind_Locz = std::distance(mesh->efield_names.begin(),
                                               std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocz));
            spts_host(sprog, 2) = mesh->efield_vals[ind_Locz][i];
          }
          // sensorData
          ptrdiff_t ind_Resp = std::distance(mesh->efield_names.begin(),
                                             std::find(mesh->efield_names.begin(), mesh->efield_names.end(), cresp));
          sdat_host(sprog, 0) = mesh->efield_vals[ind_Resp][i];
          spts_owners(sprog, 0) = i;
          spts_owners(sprog, 1) = 0;

          sprog++;
        }
      }
    }

    // ========================================
    // Create and store more compact Views based on number of sensors on this proc
    // ========================================

    Kokkos::View<ScalarT **, AssemblyDevice> spts("sensor point", numFound, dimension);
    Kokkos::View<ScalarT *, AssemblyDevice> stime("sensor times", stime_host.extent(0));
    Kokkos::View<ScalarT **, AssemblyDevice> sdat("sensor data", numFound, stime_host.extent(0));
    Kokkos::View<int *[2], HostDevice> sowners("sensor owners", numFound);

    auto stime_tmp = create_mirror_view(stime);
    deep_copy(stime_tmp, stime_host);
    deep_copy(stime, stime_tmp);

    auto spts_tmp = create_mirror_view(spts);
    auto sdat_tmp = create_mirror_view(sdat);

    size_t prog = 0;

    for (size_type pt = 0; pt < spts_host.extent(0); ++pt)
    {
      for (size_type j = 0; j < sowners.extent(1); ++j)
      {
        sowners(prog, j) = spts_owners(pt, j);
      }

      for (size_type j = 0; j < spts.extent(1); ++j)
      {
        spts_tmp(prog, j) = spts_host(pt, j);
      }

      for (size_type j = 0; j < sdat.extent(1); ++j)
      {
        sdat_tmp(prog, j) = sdat_host(pt, j);
      }
      prog++;
    }
    deep_copy(spts, spts_tmp);
    deep_copy(sdat, sdat_tmp);

    objectives[objID].sensor_points = spts;
    objectives[objID].sensor_times = stime;
    objectives[objID].sensor_data = sdat;
    objectives[objID].sensor_owners = sowners;

    // ========================================
    // Evaluate the basis functions and grads for each sensor point
    // ========================================

    vector<Kokkos::View<ScalarT ****, AssemblyDevice>> csensorBasis;
    vector<Kokkos::View<ScalarT ****, AssemblyDevice>> csensorBasisGrad;
    for (size_t k = 0; k < assembler->disc->basis_pointers[block].size(); k++)
    {
      auto basis_ptr = assembler->disc->basis_pointers[block][k];
      string basis_type = assembler->disc->basis_types[block][k];
      int bnum = basis_ptr->getCardinality();

      if (basis_type == "HGRAD")
      {
        Kokkos::View<ScalarT ****, AssemblyDevice> cbasis("sensor basis", spts.extent(0), bnum, 1, 1);
        csensorBasis.push_back(cbasis);
        Kokkos::View<ScalarT ****, AssemblyDevice> cbasisgrad("sensor basis grad", spts.extent(0), bnum, 1, dimension);
        csensorBasisGrad.push_back(cbasisgrad);
      }
      else if (basis_type == "HVOL")
      {
        Kokkos::View<ScalarT ****, AssemblyDevice> cbasis("sensor basis", spts.extent(0), bnum, 1, 1);
        csensorBasis.push_back(cbasis);
      }
      else if (basis_type == "HDIV")
      {
        Kokkos::View<ScalarT ****, AssemblyDevice> cbasis("sensor basis", spts.extent(0), bnum, 1, dimension);
        csensorBasis.push_back(cbasis);
      }
      else if (basis_type == "HCURL")
      {
        Kokkos::View<ScalarT ****, AssemblyDevice> cbasis("sensor basis", spts.extent(0), bnum, 1, dimension);
        csensorBasis.push_back(cbasis);
      }
    }

    for (size_type pt = 0; pt < spts.extent(0); ++pt)
    {

      DRV cpt("point", 1, 1, dimension);
      auto cpt_sub = subview(cpt, 0, 0, ALL());
      auto pp_sub = subview(spts, pt, ALL());
      Kokkos::deep_copy(cpt_sub, pp_sub);

      Kokkos::View<LO *, AssemblyDevice> cids("current local elem ids", 1);
      cids(0) = assembler->groups[block][sowners(pt, 0)]->localElemID(sowners(pt, 1));
      // auto nodes = mesh->getMyNodes(block, assembler->groups[block][sowners(pt,0)]->localElemID);
      // auto nodes_sv = subview(nodes,sowners(pt,1),ALL(),ALL());
      // DRV cnodes("subnodes",1,nodes.extent(1),nodes.extent(2));
      // auto cnodes_sv = subview(cnodes,0,ALL(),ALL());
      // deep_copy(cnodes_sv,nodes_sv);

      DRV refpt("refsenspts", 1, dimension);
      Kokkos::DynRankView<Intrepid2::Orientation, PHX::Device> corientation("curr orient", 1);

      DRV refpt_tmp = assembler->disc->mapPointsToReference(cpt, cids, block, assembler->groupData[block]->cell_topo);
      for (size_type d = 0; d < refpt_tmp.extent(2); ++d)
      {
        refpt(0, d) = refpt_tmp(0, 0, d);
      }

      auto orient = assembler->groups[block][sowners(pt, 0)]->orientation;
      corientation(0) = orient(sowners(pt, 1));
      for (size_t k = 0; k < assembler->disc->basis_pointers[block].size(); k++)
      {
        auto basis_ptr = assembler->disc->basis_pointers[block][k];
        string basis_type = assembler->disc->basis_types[block][k];
        auto cellTopo = assembler->groupData[block]->cell_topo;

        Kokkos::View<ScalarT ****, AssemblyDevice> bvals2, bgradvals2;
        DRV bvals = disc->evaluateBasis(assembler->groupData[block], block, k, cids, refpt, cellTopo);

        if (basis_type == "HGRAD")
        {

          auto bvals_sv = subview(bvals, 0, ALL(), ALL());
          auto bvals2_sv = subview(csensorBasis[k], pt, ALL(), ALL(), 0);
          deep_copy(bvals2_sv, bvals_sv);

          DRV bgradvals = assembler->disc->evaluateBasisGrads2(assembler->groupData[block], block, basis_ptr, cids, refpt, cellTopo);
          auto bgradvals_sv = subview(bgradvals, 0, ALL(), ALL(), ALL());
          auto bgrad_sv = subview(csensorBasisGrad[k], pt, ALL(), ALL(), ALL());
          deep_copy(bgrad_sv, bgradvals_sv);
        }
        else if (basis_type == "HVOL")
        {
          auto bvals_sv = subview(bvals, 0, ALL(), ALL());
          auto bvals2_sv = subview(csensorBasis[k], pt, ALL(), ALL(), 0);
          deep_copy(bvals2_sv, bvals_sv);
        }
        else if (basis_type == "HDIV")
        {
          auto bvals_sv = subview(bvals, 0, ALL(), ALL(), ALL());
          auto bvals2_sv = subview(csensorBasis[k], pt, ALL(), ALL(), ALL());
          deep_copy(bvals2_sv, bvals_sv);
        }
        else if (basis_type == "HCURL")
        {
          auto bvals_sv = subview(bvals, 0, ALL(), ALL(), ALL());
          auto bvals2_sv = subview(csensorBasis[k], pt, ALL(), ALL(), ALL());
          deep_copy(bvals2_sv, bvals_sv);
        }
      }
    }
    objectives[objID].sensor_basis = csensorBasis;
    objectives[objID].sensor_basis_grad = csensorBasisGrad;
  }

  debugger->print("**** Finished SensorManager::importSensorsFromExodus() ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::importSensorsFromFiles(const int &objID)
{

  debugger->print("**** Starting PostprocessManager::importSensorsFromFiles() ...");

  size_t block = objectives[objID].block;

  // ========================================
  // Import the data from the files
  // ========================================

  Data sdata;
  bool have_data = false;

  bool addproc = settings->sublist("Postprocess").get<bool>("add processor rank to sensor files",false);
  if (addproc) {
    if (objectives[objID].sensor_points_file.ends_with(".dat")) {
      objectives[objID].sensor_points_file.resize(objectives[objID].sensor_points_file.size() - 4);
    }
    std::stringstream sfile;
    sfile << objectives[objID].sensor_points_file << "." << Comm->getRank() << ".dat";
    objectives[objID].sensor_points_file = sfile.str();
  
    if (objectives[objID].sensor_data_file != "") {
      if (objectives[objID].sensor_data_file.ends_with(".dat")) {
        objectives[objID].sensor_data_file.resize(objectives[objID].sensor_points_file.size() - 4);
      }
      
      std::stringstream sfile2;
      sfile2 << objectives[objID].sensor_data_file << "." << Comm->getRank() << ".dat";
      objectives[objID].sensor_data_file = sfile2.str();
    }
  }
  
  if (objectives[objID].sensor_data_file == "")
  {
    sdata = Data("Sensor Measurements", dimension,
                 objectives[objID].sensor_points_file);
  }
  else
  {
    sdata = Data("Sensor Measurements", dimension,
                 objectives[objID].sensor_points_file,
                 objectives[objID].sensor_data_file, false);
    have_data = true;
  }

  // ========================================
  // Save the locations in the appropriate view
  // ========================================

  Kokkos::View<ScalarT **, HostDevice> spts_host = sdata.getPoints();
  std::vector<Kokkos::View<ScalarT **, HostDevice>> sensor_data_host;
  if (have_data)
  {
    sensor_data_host = sdata.getData();
  }

  // Check that the data matches the expected format
  if (spts_host.extent(1) != static_cast<size_type>(dimension))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "Error: sensor points dimension does not match simulation dimension");
  }
  if (have_data)
  {
    if (spts_host.extent(0) + 1 != sensor_data_host.size())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                 "Error: number of sensors does not match data");
    }
  }

  // ========================================
  // Import the data from the files
  // ========================================

  Kokkos::View<ScalarT *, HostDevice> stime_host;

  Kokkos::View<ScalarT **, HostDevice> sdat_host;

  if (have_data)
  {
    stime_host = Kokkos::View<ScalarT *, HostDevice>("sensor times", sensor_data_host[0].extent(1));

    for (size_type d = 0; d < sensor_data_host[0].extent(1); ++d)
    {
      stime_host(d) = sensor_data_host[0](0, d);
    }

    sdat_host = Kokkos::View<ScalarT **, HostDevice>("sensor data", sensor_data_host.size() - 1,
                                                     sensor_data_host[0].extent(1));

    for (size_type pt = 1; pt < sensor_data_host.size(); ++pt)
    {
      for (size_type d = 0; d < sensor_data_host[pt].extent(1); ++d)
      {
        sdat_host(pt - 1, d) = sensor_data_host[pt](0, d);
      }
    }
  }

  // ========================================
  // Determine which element contains each sensor point
  // Note: a given processor might not find any
  // ========================================

  Kokkos::View<int *[2], HostDevice> spts_owners("sensor owners", spts_host.extent(0));
  Kokkos::View<bool *, HostDevice> spts_found("sensors found", spts_host.extent(0));

  this->locateSensorPoints(block, true, spts_host, spts_owners, spts_found);

  // ========================================
  // Determine the number of sensors on this proc
  // ========================================

  size_t numFound = 0;
  for (size_type pt = 0; pt < spts_found.extent(0); ++pt)
  {
    if (spts_found(pt))
    {
      numFound++;
    }
  }
  
  objectives[objID].numSensors = numFound;
  objectives[objID].sensor_found = spts_found;

  if (numFound > 0)
  {

    // ========================================
    // Create and store more compact Views based on number of sensors on this proc
    // ========================================

    Kokkos::View<ScalarT **, AssemblyDevice> spts("sensor point", numFound, dimension);
    Kokkos::View<ScalarT *, AssemblyDevice> stime;
    Kokkos::View<ScalarT **, AssemblyDevice> sdat;
    Kokkos::View<int *[2], HostDevice> sowners("sensor owners", numFound);

    auto spts_tmp = create_mirror_view(spts);

    if (have_data)
    {
      stime = Kokkos::View<ScalarT *, AssemblyDevice>("sensor times", stime_host.extent(0));
      auto stime_tmp = create_mirror_view(stime);
      deep_copy(stime_tmp, stime_host);
      deep_copy(stime, stime_tmp);

      sdat = Kokkos::View<ScalarT **, AssemblyDevice>("sensor data", numFound, stime_host.extent(0));
      auto sdat_tmp = create_mirror_view(sdat);
      size_t prog = 0;

      for (size_type pt = 0; pt < spts_host.extent(0); ++pt)
      {
        if (spts_found(pt))
        {
          if (have_data)
          {
            for (size_type j = 0; j < sdat.extent(1); ++j)
            {
              sdat_tmp(prog, j) = sdat_host(pt, j);
            }
          }
          prog++;
        }
      }
      deep_copy(sdat, sdat_tmp);
    }

    size_t prog = 0;

    for (size_type pt = 0; pt < spts_host.extent(0); ++pt)
    {
      if (spts_found(pt))
      {
        for (size_type j = 0; j < sowners.extent(1); ++j)
        {
          sowners(prog, j) = spts_owners(pt, j);
        }
        for (size_type j = 0; j < spts.extent(1); ++j)
        {
          spts_tmp(prog, j) = spts_host(pt, j);
        }
        prog++;
      }
    }
    deep_copy(spts, spts_tmp);

    objectives[objID].sensor_points = spts;
    objectives[objID].sensor_times = stime;
    objectives[objID].sensor_data = sdat;
    objectives[objID].sensor_owners = sowners;

    // ========================================
    // Evaluate the basis functions and grads for each sensor point
    // ========================================

    this->computeSensorBasis(objID);
  }

  debugger->print("**** Finished SensorManager::importSensorsFromFiles() ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::importSensorsOnGrid(const int &objID)
{

  debugger->print("**** Starting PostprocessManager::importSensorsOnGrid() ...");

  // Check that the data matches the expected format
  if (dimension != 3)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "Error: defining a grid of sensor points is only implemented in 3-dimensions");
  }

  size_t block = objectives[objID].block;

  // ========================================
  // Save the locations in the appropriate view
  // ========================================

  double xmin = objectives[objID].sensor_grid_xmin;
  double xmax = objectives[objID].sensor_grid_xmax;
  double ymin = objectives[objID].sensor_grid_ymin;
  double ymax = objectives[objID].sensor_grid_ymax;
  double zmin = objectives[objID].sensor_grid_zmin;
  double zmax = objectives[objID].sensor_grid_zmax;

  int Nx = objectives[objID].sensor_grid_Nx;
  int Ny = objectives[objID].sensor_grid_Ny;
  int Nz = objectives[objID].sensor_grid_Nz;

  double dx = (Nx > 1) ? (xmax - xmin) / (Nx - 1) : 0.0;
  double dy = (Ny > 1) ? (ymax - ymin) / (Ny - 1) : 0.0;
  double dz = (Nz > 1) ? (zmax - zmin) / (Nz - 1) : 0.0;

  std::vector<double> xgrid(Nx);
  std::vector<double> ygrid(Ny);
  std::vector<double> zgrid(Nz);

  double xval = xmin;
  for (int i = 0; i < Nx; ++i)
  {
    xgrid[i] = xval;
    xval += dx;
  }

  double yval = ymin;
  for (int i = 0; i < Ny; ++i)
  {
    ygrid[i] = yval;
    yval += dy;
  }

  double zval = zmin;
  for (int i = 0; i < Nz; ++i)
  {
    zgrid[i] = zval;
    zval += dz;
  }

  Kokkos::View<ScalarT **, HostDevice> spts_host("sensor locations", Nx * Ny * Nz, dimension);

  size_t prog = 0;
  for (int i = 0; i < Nx; i++)
  {
    for (int j = 0; j < Ny; j++)
    {
      for (int k = 0; k < Nz; k++)
      {
        spts_host(prog, 0) = xgrid[i];
        spts_host(prog, 1) = ygrid[j];
        spts_host(prog, 2) = zgrid[k];
        prog++;
      }
    }
  }

  // ========================================
  // Determine which element contains each sensor point
  // Note: a given processor might not find any
  // ========================================

  Kokkos::View<int *[2], HostDevice> spts_owners("sensor owners", spts_host.extent(0));
  Kokkos::View<bool *, HostDevice> spts_found("sensors found", spts_host.extent(0));

  this->locateSensorPoints(block, true, spts_host, spts_owners, spts_found);

  // ========================================
  // Determine the number of sensors on this proc
  // ========================================

  size_t numFound = 0;
  for (size_type pt = 0; pt < spts_found.extent(0); ++pt)
  {
    if (spts_found(pt))
    {
      numFound++;
    }
  }

  objectives[objID].numSensors = numFound;
  objectives[objID].sensor_found = spts_found;

  if (numFound > 0)
  {

    // ========================================
    // Create and store more compact Views based on number of sensors on this proc
    // ========================================

    Kokkos::View<ScalarT **, AssemblyDevice> spts("sensor point", numFound, dimension);
    Kokkos::View<ScalarT *, AssemblyDevice> stime;
    Kokkos::View<ScalarT **, AssemblyDevice> sdat;
    Kokkos::View<int *[2], HostDevice> sowners("sensor owners", numFound);

    auto spts_tmp = create_mirror_view(spts);

    size_t prog = 0;

    for (size_type pt = 0; pt < spts_host.extent(0); ++pt)
    {
      if (spts_found(pt))
      {
        for (size_type j = 0; j < sowners.extent(1); ++j)
        {
          sowners(prog, j) = spts_owners(pt, j);
        }
        for (size_type j = 0; j < spts.extent(1); ++j)
        {
          spts_tmp(prog, j) = spts_host(pt, j);
        }
        prog++;
      }
    }
    deep_copy(spts, spts_tmp);

    objectives[objID].sensor_points = spts;
    objectives[objID].sensor_times = stime;
    objectives[objID].sensor_data = sdat;
    objectives[objID].sensor_owners = sowners;

    // ========================================
    // Evaluate the basis functions and grads for each sensor point
    // ========================================

    this->computeSensorBasis(objID);
  }

  debugger->print("**** Finished SensorManager::importSensorsOnGrid() ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::importSensorsOnQuadrature(const int &objID)
{

  debugger->print("**** Starting PostprocessManager::importSensorsOnQuadrature() ...");

  size_t block = objectives[objID].block;

  // ========================================
  // Import the data from the files
  // ========================================

  
  // ========================================
  // Save the locations in the appropriate view
  // ========================================

  View_Sc2 qdata = assembler->getQuadratureData(blocknames[block]); // [pts wts]
  
  Kokkos::View<double **, HostDevice> spts_host("sensor points", qdata.extent(0), qdata.extent(1)-1); // strip off wts
  for (size_t i=0; i<qdata.extent(0); ++i) {
    for (size_t j=0; j<qdata.extent(1)-1; ++j) {
      spts_host(i,j) = qdata(i,j);
    }
  }
  // ========================================
  // For now assuming we don't have data to match
  // ========================================

  // ========================================
  // Determine which element contains each sensor point
  // This is not needed since we already know the owners
  // ========================================

  Kokkos::View<int *[2], HostDevice> spts_owners("sensor owners", spts_host.extent(0));
  Kokkos::View<bool *, HostDevice> spts_found("sensors found", spts_host.extent(0));

  this->locateSensorPoints(block, false, spts_host, spts_owners, spts_found);

  // ========================================
  // Determine the number of sensors on this proc
  // ========================================

  size_t numFound = 0;
  for (size_type pt = 0; pt < spts_found.extent(0); ++pt) {
    if (spts_found(pt)) {
      numFound++;
    }
    else {
      //cout << spts_host(pt,0) << " " << spts_host(pt,1) << endl;
    }
  }
  if (numFound != qdata.extent(0)) {
    // throw an error
  }
  objectives[objID].numSensors = numFound;
  objectives[objID].sensor_found = spts_found;
  
  if (numFound > 0) {

    // ========================================
    // Create and store more compact Views based on number of sensors on this proc
    // ========================================

    Kokkos::View<ScalarT **, AssemblyDevice> spts("sensor point", numFound, dimension);
    Kokkos::View<ScalarT *, AssemblyDevice> stime;
    Kokkos::View<ScalarT **, AssemblyDevice> sdat;
    Kokkos::View<int *[2], HostDevice> sowners("sensor owners", numFound);

    auto spts_tmp = create_mirror_view(spts);
    
    size_t prog = 0;

    for (size_type pt = 0; pt < spts_host.extent(0); ++pt) {
      if (spts_found(pt)) {
        for (size_type j = 0; j < sowners.extent(1); ++j) {
          sowners(prog, j) = spts_owners(pt, j);
        }
        for (size_type j = 0; j < spts.extent(1); ++j) {
          spts_tmp(prog, j) = spts_host(pt, j);
        }
        prog++;
      }
    }
    deep_copy(spts, spts_tmp);

    objectives[objID].sensor_points = spts;
    objectives[objID].sensor_times = stime;
    //objectives[objID].sensor_data = sdat;
    objectives[objID].sensor_owners = sowners;

    // ========================================
    // Evaluate the basis functions and grads for each sensor point
    // ========================================

    this->computeSensorBasis(objID);
  }

  debugger->print("**** Finished SensorManager::importSensorsOnQuadrature() ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::importSensorsOnBndryQuadrature(const int &objID)
{

  debugger->print("**** Starting PostprocessManager::importSensorsOnBndryQuadrature() ...");

  size_t block = objectives[objID].block;
  string sidename = objectives[objID].sideset;
  
  int sidenum = -1;
  for (size_t side=0; side<sideSets.size(); ++side) {
    if (sideSets[side] == sidename) {
      sidenum = side;
    }
  }
  View_Sc2 qdata = assembler->getboundaryQuadratureData(blocknames[block], sideSets[sidenum]); // [pts wts normals]
  
  Kokkos::View<double **, HostDevice> spts_host("sensor points", qdata.extent(0), qdata.extent(1)-dimension-1); // strip off wts
  for (size_t i=0; i<qdata.extent(0); ++i) {
    for (size_t j=0; j<qdata.extent(1)-dimension-1; ++j) {
      spts_host(i,j) = qdata(i,j);
    }
  }
  
  // ========================================
  // For now assuming we don't have data to match
  // ========================================

  // ========================================
  // Determine which element contains each sensor point
  // This is not needed since we already know the owners
  // ========================================

  Kokkos::View<int *[2], HostDevice> spts_owners("sensor owners", spts_host.extent(0));
  Kokkos::View<bool *, HostDevice> spts_found("sensors found", spts_host.extent(0));

  this->locateSensorPoints(block, false, spts_host, spts_owners, spts_found);

  // ========================================
  // Determine the number of sensors on this proc
  // ========================================

  size_t numFound = 0;
  for (size_type pt = 0; pt < spts_found.extent(0); ++pt) {
    if (spts_found(pt)) {
      numFound++;
    }
  }
  if (numFound != qdata.extent(0)) {
    // throw an error
  }
  objectives[objID].numSensors = numFound;
  objectives[objID].sensor_found = spts_found;

  if (numFound > 0) {

    // ========================================
    // Create and store more compact Views based on number of sensors on this proc
    // ========================================

    Kokkos::View<ScalarT **, AssemblyDevice> spts("sensor point", numFound, dimension);
    Kokkos::View<ScalarT *, AssemblyDevice> stime;
    Kokkos::View<ScalarT **, AssemblyDevice> sdat;
    Kokkos::View<int *[2], HostDevice> sowners("sensor owners", numFound);

    auto spts_tmp = create_mirror_view(spts);
    
    size_t prog = 0;

    for (size_type pt = 0; pt < spts_host.extent(0); ++pt) {
      if (spts_found(pt)) {
        for (size_type j = 0; j < sowners.extent(1); ++j) {
          sowners(prog, j) = spts_owners(pt, j);
        }
        for (size_type j = 0; j < spts.extent(1); ++j) {
          spts_tmp(prog, j) = spts_host(pt, j);
        }
        prog++;
      }
    }
    deep_copy(spts, spts_tmp);

    objectives[objID].sensor_points = spts;
    objectives[objID].sensor_times = stime;
    //objectives[objID].sensor_data = sdat;
    objectives[objID].sensor_owners = sowners;

    // ========================================
    // Evaluate the basis functions and grads for each sensor point
    // ========================================

    this->computeSensorBasis(objID);
  }

  debugger->print("**** Finished SensorManager::importSensorsOnBndryQuadrature() ...");
}


// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeSensorBasis(const int &objID)
{

  size_t block = objectives[objID].block;
  auto spts = objectives[objID].sensor_points;
  auto sowners = objectives[objID].sensor_owners;

  vector<Kokkos::View<ScalarT ****, AssemblyDevice>> csensorBasis;
  vector<Kokkos::View<ScalarT ****, AssemblyDevice>> csensorBasisGrad;
  for (size_t k = 0; k < assembler->disc->basis_pointers[block].size(); k++)
  {
    auto basis_ptr = assembler->disc->basis_pointers[block][k];
    string basis_type = assembler->disc->basis_types[block][k];
    int bnum = basis_ptr->getCardinality();

    if (basis_type == "HGRAD")
    {
      Kokkos::View<ScalarT ****, AssemblyDevice> cbasis("sensor basis", spts.extent(0), bnum, 1, 1);
      csensorBasis.push_back(cbasis);
      Kokkos::View<ScalarT ****, AssemblyDevice> cbasisgrad("sensor basis grad", spts.extent(0), bnum, 1, dimension);
      csensorBasisGrad.push_back(cbasisgrad);
    }
    else if (basis_type == "HVOL")
    {
      Kokkos::View<ScalarT ****, AssemblyDevice> cbasis("sensor basis", spts.extent(0), bnum, 1, 1);
      csensorBasis.push_back(cbasis);
    }
    else if (basis_type == "HDIV")
    {
      Kokkos::View<ScalarT ****, AssemblyDevice> cbasis("sensor basis", spts.extent(0), bnum, 1, dimension);
      csensorBasis.push_back(cbasis);
    }
    else if (basis_type == "HCURL")
    {
      Kokkos::View<ScalarT ****, AssemblyDevice> cbasis("sensor basis", spts.extent(0), bnum, 1, dimension);
      csensorBasis.push_back(cbasis);
    }
  }

  for (size_type pt = 0; pt < spts.extent(0); ++pt)
  {

    DRV cpt("point", 1, 1, dimension);
    auto cpt_sub = subview(cpt, 0, 0, ALL());
    auto pp_sub = subview(spts, pt, ALL());
    Kokkos::deep_copy(cpt_sub, pp_sub);

    Kokkos::View<LO *, AssemblyDevice> cids("current local elemids", 1);
    cids(0) = assembler->groups[block][sowners(pt, 0)]->localElemID(sowners(pt, 1));
    // auto nodes = mesh->getMyNodes(block, assembler->groups[block][sowners(pt,0)]->localElemID);
    // auto nodes_sv = subview(nodes,sowners(pt,1),ALL(),ALL());
    // DRV cnodes("subnodes",1,nodes.extent(1),nodes.extent(2));
    // auto cnodes_sv = subview(cnodes,0,ALL(),ALL());
    // deep_copy(cnodes_sv,nodes_sv);

    DRV refpt("refsenspts", 1, dimension);
    // Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> corientation("curr orient",1);

    DRV refpt_tmp = assembler->disc->mapPointsToReference(cpt, cids, block, assembler->groupData[block]->cell_topo);

    for (size_type d = 0; d < refpt_tmp.extent(2); ++d)
    {
      refpt(0, d) = refpt_tmp(0, 0, d);
    }

    // auto orient = assembler->groups[block][sowners(pt,0)]->orientation;
    // corientation(0) = orient(sowners(pt,1));

    for (size_t k = 0; k < assembler->disc->basis_pointers[block].size(); k++)
    {
      auto basis_ptr = assembler->disc->basis_pointers[block][k];
      string basis_type = assembler->disc->basis_types[block][k];
      auto cellTopo = assembler->groupData[block]->cell_topo;

      Kokkos::View<ScalarT ****, AssemblyDevice> bvals2, bgradvals2;
      DRV bvals = disc->evaluateBasis(assembler->groupData[block], block, k, cids, refpt, cellTopo);

      if (basis_type == "HGRAD")
      {
        auto bvals_sv = subview(bvals, 0, ALL(), ALL());
        auto bvals2_sv = subview(csensorBasis[k], pt, ALL(), ALL(), 0);
        deep_copy(bvals2_sv, bvals_sv);

        DRV bgradvals = assembler->disc->evaluateBasisGrads2(assembler->groupData[block], block, basis_ptr, cids, refpt, cellTopo);
        auto bgradvals_sv = subview(bgradvals, 0, ALL(), ALL(), ALL());
        auto bgrad_sv = subview(csensorBasisGrad[k], pt, ALL(), ALL(), ALL());
        deep_copy(bgrad_sv, bgradvals_sv);
      }
      else if (basis_type == "HVOL")
      {
        auto bvals_sv = subview(bvals, 0, ALL(), ALL());
        auto bvals2_sv = subview(csensorBasis[k], pt, ALL(), ALL(), 0);
        deep_copy(bvals2_sv, bvals_sv);
      }
      else if (basis_type == "HDIV")
      {
        auto bvals_sv = subview(bvals, 0, ALL(), ALL(), ALL());
        auto bvals2_sv = subview(csensorBasis[k], pt, ALL(), ALL(), ALL());
        deep_copy(bvals2_sv, bvals_sv);
      }
      else if (basis_type == "HCURL")
      {
        auto bvals_sv = subview(bvals, 0, ALL(), ALL(), ALL());
        auto bvals2_sv = subview(csensorBasis[k], pt, ALL(), ALL(), ALL());
        deep_copy(bvals2_sv, bvals_sv);
      }
    }
  }
  objectives[objID].sensor_basis = csensorBasis;
  objectives[objID].sensor_basis_grad = csensorBasisGrad;
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::locateSensorPoints(const int & block,
                                                  const bool & checkprocs,
                                                  Kokkos::View<ScalarT **, HostDevice> spts_host,
                                                  Kokkos::View<int *[2], HostDevice> spts_owners,
                                                  Kokkos::View<bool *, HostDevice> spts_found)
{

  global_num_sensors = spts_host.extent(0);
  size_t checksPerformed = 0;

  for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
  {

    auto nodes = disc->getMyNodes(block, assembler->groups[block][grp]->localElemID);
    auto nodes_host = create_mirror_view(nodes);
    deep_copy(nodes_host, nodes);

    // Create a bounding box for the element
    // This serves as a preprocessing check to avoid unnecessary inclusion checks
    // If a sensor point is not in the box, then it is not in the element
    Kokkos::View<double **[2], HostDevice> nodebox("bounding box", nodes_host.extent(0), dimension);
    for (size_type p = 0; p < nodes_host.extent(0); ++p)
    {
      for (size_type dim = 0; dim < nodes_host.extent(2); ++dim)
      {
        double dmin = 1.0e300;
        double dmax = -1.0e300;
        for (size_type k = 0; k < nodes_host.extent(1); ++k)
        {
          dmin = std::min(dmin, nodes_host(p, k, dim));
          dmax = std::max(dmax, nodes_host(p, k, dim));
        }
        nodebox(p, dim, 0) = dmin;
        nodebox(p, dim, 1) = dmax;
      }
    }

    for (size_type pt = 0; pt < spts_host.extent(0); ++pt)
    {
      if (!spts_found(pt))
      {
        for (size_type p = 0; p < nodebox.extent(0); ++p)
        {
          double xbuff = 0.1 * (nodebox(p, 0, 1) - nodebox(p, 0, 0));
          double ybuff = 0.1 * (nodebox(p, 1, 1) - nodebox(p, 1, 0));
          double zbuff = 0.1 * (nodebox(p, 2, 1) - nodebox(p, 2, 0));
          bool proceed = true;
          if (spts_host(pt, 0) < nodebox(p, 0, 0) - xbuff || spts_host(pt, 0) > nodebox(p, 0, 1) + xbuff)
          {
            proceed = false;
          }
          if (proceed && dimension > 1)
          {
            if (spts_host(pt, 1) < nodebox(p, 1, 0) - ybuff || spts_host(pt, 1) > nodebox(p, 1, 1) + ybuff)
            {
              proceed = false;
            }
          }
          if (proceed && dimension > 2)
          {
            if (spts_host(pt, 2) < nodebox(p, 2, 0) - zbuff || spts_host(pt, 2) > nodebox(p, 2, 1) + zbuff)
            {
              proceed = false;
            }
          }

          if (proceed)
          {
            checksPerformed++;
            // Need to use DRV, which are on AssemblyDevice
            // We have less control here
            DRV phys_pt("phys_pt", 1, 1, dimension);
            auto phys_pt_host = create_mirror_view(phys_pt);
            for (size_type d = 0; d < spts_host.extent(1); ++d)
            {
              phys_pt_host(0, 0, d) = spts_host(pt, d);
            }
            deep_copy(phys_pt, phys_pt_host);
            Kokkos::View<LO *, AssemblyDevice> cids("current local elem ids", 1);
            cids(0) = assembler->groups[block][grp]->localElemID(p);
            // DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
            // auto n_sub = subview(nodes,p,ALL(),ALL());
            // auto cn_sub = subview(cnodes,0,ALL(),ALL());
            // Kokkos::deep_copy(cn_sub,n_sub);

            auto inRefgrp = assembler->disc->checkInclusionPhysicalData(phys_pt, cids,
                                                                        assembler->groupData[block]->cell_topo,
                                                                        block, 1.0e-14);
            auto inRef_host = create_mirror_view(inRefgrp);
            deep_copy(inRef_host, inRefgrp);
            if (inRef_host(0, 0))
            {
              spts_found(pt) = true;
              spts_owners(pt, 0) = grp;
              spts_owners(pt, 1) = p;
            }
            else
            {
              //cout << "Sensor was in bounding box, but not in element: " << endl;
              //KokkosTools::print(phys_pt);
              //KokkosTools::print(nodes);
            }
          }
        }
      } // found
    } // pt
  } // elem

  if (checkprocs) {
    for (size_type pt = 0; pt < spts_found.extent(0); ++pt) {
      size_t fnd_flag = Comm->getSize()+1;
      if (spts_found(pt))
      {
        fnd_flag = Comm->getRank();
      }
      size_t globalFound = 0;
      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MIN, 1, &fnd_flag, &globalFound);
      if (Comm->getRank() != globalFound) {
        spts_found(pt) = false;
      }
    }
  }
  
  if (verbosity >= 10)
  {
    size_t numFound = 0;
    for (size_type pt = 0; pt < spts_found.extent(0); ++pt)
    {
      if (spts_found(pt))
      {
        numFound++;
      }
    }
    cout << "Total number of Intrepid inclusion checks performed on processor " << Comm->getRank() << ": " << checksPerformed << endl;
    cout << " - Processor " << Comm->getRank() << " has " << numFound << " sensors" << endl;

    size_t globalFound = 0;
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &numFound, &globalFound);
    if (Comm->getRank() == 0)
    {
      cout << " - Total Number of Sensors: " << spts_found.extent(0) << endl;
      cout << " - Total Number of Sensors Located: " << globalFound << endl;
    }
  }
}


// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeSensorSolution(vector<vector_RCP> &current_soln,
                                                     const ScalarT &current_time)
{

  Teuchos::TimeMonitor localtimer(*sensorSolutionTimer);

  debugger->print(1, "******** Starting PostprocessManager::computeSensorSolution ...");

  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type LA_device;

  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> sol_kv;
  for (size_t s = 0; s < current_soln.size(); ++s) {
    auto vec_kv = current_soln[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
    if (data_avail) {
      sol_kv.push_back(vec_slice);
    }
    else {
      auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), vec_slice);
      Kokkos::deep_copy(vec_dev, vec_slice);
      sol_kv.push_back(vec_dev);
    }
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> params_kv;
  auto Psol = params->getDiscretizedParams();
  auto p_kv = Psol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto pslice = Kokkos::subview(p_kv, Kokkos::ALL(), 0);

  if (data_avail) {
    params_kv.push_back(pslice);
  }
  else {
    auto p_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), pslice);
    Kokkos::deep_copy(p_dev, pslice);
    params_kv.push_back(p_dev);
  }

  for (size_t r = 0; r < objectives.size(); ++r) {

    if (objectives[r].type == "sensors") {
      if (objectives[r].compute_sensor_soln || objectives[r].compute_sensor_average_soln) {

        size_t block = objectives[r].block;

        int numSols = 0;
        for (size_t set = 0; set < varlist.size(); ++set) {
          numSols += varlist[set][block].size();
        }
        Kokkos::View<ScalarT ***, HostDevice> sensordat("sensor solution", objectives[r].numSensors, numSols, dimension);
        objectives[r].response_times.push_back(current_time); // might store this somewhere else

        for (size_t pt = 0; pt < objectives[r].numSensors; ++pt) {

          size_t solprog = 0;
          int grp_owner = objectives[r].sensor_owners(pt, 0);
          int elem_owner = objectives[r].sensor_owners(pt, 1);
          for (size_t set = 0; set < varlist.size(); ++set) {
            auto numDOF = assembler->groupData[block]->set_num_dof_host[set];
            if (!assembler->groups[block][grp_owner]->have_sols) {
              assembler->performGather(set, block, grp_owner, sol_kv[set], 0, 0);
            }
            auto cu = subview(assembler->groupData[block]->sol[set], elem_owner, ALL(), ALL());
            auto cu_host = create_mirror_view(cu);
            // KokkosTools::print(assembler->groups[block][grp_owner]->u[set]);
            deep_copy(cu_host, cu);
            for (size_type var = 0; var < numDOF.extent(0); var++) {
              auto cbasis = objectives[r].sensor_basis[assembler->wkset[block]->set_usebasis[set][var]];
              for (size_type dof = 0; dof < cbasis.extent(1); ++dof) {
                for (size_type dim = 0; dim < cbasis.extent(3); ++dim) {
                  // sensordat(pt,solprog,dim) += cu_host(solprog,dof)*cbasis(pt,dof,0,dim);
                  sensordat(pt, solprog, dim) += cu_host(var, dof) * cbasis(pt, dof, 0, dim);
                }
              }
              solprog++;
            }
            // KokkosTools::print(sensordat);
          }

        } // sensor points
        // KokkosTools::print(sensordat);

        if (objectives[r].output_type == "dft") {
          std::complex<double> imagi(0.0, 1.0);
          int N = objectives[r].dft_num_freqs;
          Kokkos::View<std::complex<double> ****, HostDevice> newdft;
          if (objectives[r].sensor_solution_dft.extent(0) == 0) {
            newdft = Kokkos::View<std::complex<double> ****, HostDevice>("KV of complex DFT", sensordat.extent(0),
                                                                         sensordat.extent(1), sensordat.extent(2), N);
            objectives[r].sensor_solution_dft = newdft;
          }
          else {
            newdft = objectives[r].sensor_solution_dft;
          }
          for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
              double freq = static_cast<double>(k * j / N);
              freq *= -2.0 * PI;
              for (size_type n = 0; n < newdft.extent(0); ++n) {
                for (size_type m = 0; m < newdft.extent(1); ++m) {
                  for (size_type p = 0; p < newdft.extent(2); ++p) {
                    newdft(n, m, p, k) += sensordat(n, m, p) * std::exp(imagi * freq);
                  }
                }
              }
            }
          }
        }
        else {
          objectives[r].sensor_solution_data.push_back(sensordat);
        }

      } // objectives
    }
  }

  debugger->print(1, "******** Finished PostprocessManager::computeSensorSolutions ...");
}

