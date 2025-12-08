/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// ========================================================================================

void AnalysisManager::recoverSolution(vector_RCP &solution, string &data_type,
                                      string &plist_filename, string &filename)
{

  string extension = filename.substr(filename.size() - 4, filename.size() - 1);
  filename.erase(filename.size() - 4, 4);

  cout << extension << "  " << filename << endl;
  if (data_type == "text")
  {
    std::stringstream sfile;
    sfile << filename << "." << comm_->getRank() << extension;
    std::ifstream fnmast(sfile.str());
    if (!fnmast.good())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(), std::runtime_error, "Error: could not find the data file: " + sfile.str());
    }

    std::vector<std::vector<ScalarT>> values;
    std::ifstream fin(sfile.str());

    for (std::string line; std::getline(fin, line);)
    {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream in(line);
      values.push_back(std::vector<ScalarT>(std::istream_iterator<ScalarT>(in),
                                            std::istream_iterator<ScalarT>()));
    }

    typedef typename SolverNode::device_type LA_device;
    auto sol_view = solution->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    for (size_type i = 0; i < values.size(); ++i)
    {
      sol_view(i, 0) = values[i][0];
    }
  }
  else if (data_type == "exodus")
  {
  }
  else if (data_type == "hdf5")
  {
  }
  else if (data_type == "binary")
  {
  }
  else
  {
    std::cout << "Unknown file type: " << data_type << std::endl;
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::updateRotationData(const int &newrandseed)
{

  // Determine how many seeds there are
  size_t localnumSeeds = 0;
  size_t numSeeds = 0;
  for (size_t block = 0; block < solver_->assembler->groups.size(); ++block)
  {
    for (size_t grp = 0; grp < solver_->assembler->groups[block].size(); ++grp)
    {
      for (size_t e = 0; e < solver_->assembler->groups[block][grp]->numElem; ++e)
      {
        if (solver_->assembler->groups[block][grp]->data_seed[e] > localnumSeeds)
        {
          localnumSeeds = solver_->assembler->groups[block][grp]->data_seed[e];
        }
      }
    }
  }
  // comm_->MaxAll(&localnumSeeds, &numSeeds, 1);
  Teuchos::reduceAll<int, size_t>(*comm_, Teuchos::REDUCE_MAX, 1, &localnumSeeds, &numSeeds);
  numSeeds += 1; // To properly allocate and iterate

  // Create a random number generator
  std::default_random_engine generator(newrandseed);

  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////

  int numdata = 9;

  // cout << "solver_r numSeeds = " << numSeeds << endl;

  std::normal_distribution<ScalarT> ndistribution(0.0, 1.0);
  Kokkos::View<ScalarT **, HostDevice> rotation_data("cell_data", numSeeds, numdata);
  for (size_t k = 0; k < numSeeds; k++)
  {
    ScalarT x = ndistribution(generator);
    ScalarT y = ndistribution(generator);
    ScalarT z = ndistribution(generator);
    ScalarT w = ndistribution(generator);

    ScalarT r = sqrt(x * x + y * y + z * z + w * w);
    x *= 1.0 / r;
    y *= 1.0 / r;
    z *= 1.0 / r;
    w *= 1.0 / r;

    rotation_data(k, 0) = w * w + x * x - y * y - z * z;
    rotation_data(k, 1) = 2.0 * (x * y - w * z);
    rotation_data(k, 2) = 2.0 * (x * z + w * y);

    rotation_data(k, 3) = 2.0 * (x * y + w * z);
    rotation_data(k, 4) = w * w - x * x + y * y - z * z;
    rotation_data(k, 5) = 2.0 * (y * z - w * x);

    rotation_data(k, 6) = 2.0 * (x * z - w * y);
    rotation_data(k, 7) = 2.0 * (y * z + w * x);
    rotation_data(k, 8) = w * w - x * x - y * y + z * z;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Set cell data
  ////////////////////////////////////////////////////////////////////////////////

  for (size_t block = 0; block < solver_->assembler->groups.size(); ++block)
  {
    for (size_t grp = 0; grp < solver_->assembler->groups[block].size(); ++grp)
    {
      int numElem = solver_->assembler->groups[block][grp]->numElem;
      for (int c = 0; c < numElem; c++)
      {
        int cnode = solver_->assembler->groups[block][grp]->data_seed[c];
        for (int i = 0; i < 9; i++)
        {
          solver_->assembler->groups[block][grp]->data(c, i) = rotation_data(cnode, i);
        }
      }
    }
  }
  for (size_t block = 0; block < solver_->assembler->boundary_groups.size(); ++block)
  {
    for (size_t grp = 0; grp < solver_->assembler->boundary_groups[block].size(); ++grp)
    {
      int numElem = solver_->assembler->boundary_groups[block][grp]->numElem;
      for (int e = 0; e < numElem; ++e)
      {
        int cnode = solver_->assembler->boundary_groups[block][grp]->data_seed[e];
        for (int i = 0; i < 9; i++)
        {
          solver_->assembler->boundary_groups[block][grp]->data(e, i) = rotation_data(cnode, i);
        }
      }
    }
  }
  solver_->multiscale_manager->updateMeshData(rotation_data);
}

// ========================================================================================
// ========================================================================================

#if defined(MrHyDE_ENABLE_HDSA)
void AnalysisManager::readExoForwardSolve()
{
  Teuchos::ParameterList read_exo_settings;

  if (settings_->sublist("Analysis").sublist("readExo+forward").isSublist("DataLoadParameters"))
    read_exo_settings = settings_->sublist("Analysis").sublist("readExo+forward").sublist("DataLoadParameters");
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the readExo+forward sublist in the input file!  Abort!");

  std::string exo_file = read_exo_settings.get<std::string>("ExoFile", "error");
  std::string txt_file = read_exo_settings.get<std::string>("TxtFile", "error");

  if (exo_file != "error")
  {
    HDSA::Ptr<HDSA::Random_Number_Generator<ScalarT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>();
    HDSA::Ptr<Data_Loader_MrHyDE<ScalarT>> data_loader = HDSA::makePtr<Data_Loader_MrHyDE<ScalarT>>(comm_, solver_, params_, random_number_generator);
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> tpetra_vec = data_loader->Read_Exodus_Data(exo_file, false);
    params_->updateParams(tpetra_vec);
  }
  else if (txt_file != "error")
  {
    if (params_->have_dynamic_scalar)
    {
      int num_params = params_->getNumParams("active");
      int num_time_steps = params_->getCurrentVector().dimension() / num_params; // This assumes that all active parameters are dynamic
      ScalarT val = 0.0;
      std::ifstream in(txt_file);
      std::vector<std::vector<ScalarT>> vec;
      vec.resize(num_time_steps);
      if (in)
      {
        for (int i = 0; i < num_time_steps; i++)
        {
          vec[i].resize(num_params);
          // read the elements in the file into a vector
          for (int j = 0; j < num_params; j++)
          {
            in >> val;
            vec[i][j] = val;
          }
          params_->dynamic_timeindex = i;
          params_->updateParams(vec[i], "active");
        }
        params_->dynamic_timeindex = 0;
      }
      else
      {
        std::cout << "Error loading the data from " << txt_file << std::endl;
      }
    }
    else
    {
      ScalarT val = 0.0;
      int dim = params_->getCurrentVector().dimension();
      // read in data
      std::ifstream in(txt_file);
      std::vector<ScalarT> vec = std::vector<ScalarT>(dim, 0.0);
      // read the elements in the file into a vector
      if (in)
      {
        for (int i = 0; i < dim; i++)
        {
          in >> val;
          vec[i] = val;
        }
      }
      else
      {
        std::cout << "Error loading the data from " << txt_file << std::endl;
      }

      params_->updateParams(vec, "active");
    }
  }

  if (read_exo_settings.get("Sample Set File", "error") != "error")
  {
    int nsamp = read_exo_settings.get("Number of Samples", 100);
    int dim = params_->getNumParams("stochastic");
    ROL::Ptr<ROL::BatchManager<ScalarT>> bman = ROL::makePtr<ROL::MrHyDETeuchosBatchManager<ScalarT, int>>(comm_);
    std::string sample_pt_file = read_exo_settings.get("Sample Set File", "error");
    std::string sample_wt_file = read_exo_settings.get("Sample Weight File", "error");
    ROL::Ptr<ROL::SampleGenerator<ScalarT>> sampler = ROL::makePtr<ROL::Sample_Set_Reader<ScalarT>>(nsamp, dim, bman, sample_pt_file, sample_wt_file);

    for (int i = 0; i < sampler->numMySamples(); i++)
    {
      std::vector<ScalarT> pt_i = sampler->getMyPoint(i);
      params_->updateParams(pt_i, "stochastic");
      std::string outfile = "output_sample_" + std::to_string(i) + ".exo";
      postproc_->setNewExodusFile(outfile);
      ScalarT objfun = 0.0;
      solver_->forwardModel(objfun);
      postproc_->report();
      std::string name = "obj_val_sample_" + std::to_string(i) + ".txt";
      std::ofstream fout;
      fout.open(name);
      fout << std::setprecision(8) << objfun;
      fout.close();
    }
  }
  else
  {
    ScalarT objfun = 0.0;
    solver_->forwardModel(objfun);
    postproc_->report();
    std::string name = "obj_val.txt";
    std::ofstream fout;
    fout.open(name);
    fout << std::setprecision(8) << objfun;
    fout.close();
  }
}
#endif

// ========================================================================================
// ========================================================================================

void AnalysisManager::writeSolutionToText(string &filename, vector<vector<vector_RCP>> &soln,
                                          const bool &only_write_final)
{
  typedef typename SolverNode::device_type LA_device;
  // vector<vector<vector_RCP> > soln = postproc_->soln[0]->extractAllData();
  int index = 0; // forget what this is for
  size_type numVecs = soln[index].size();
  auto v0_view = soln[index][0]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  size_type numEnt = v0_view.extent(0);
  View_Sc2 all_data("data for writing", numEnt, numVecs);
  for (size_type v = 0; v < numVecs; ++v)
  {
    auto vec_view = soln[index][v]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    for (size_type i = 0; i < numEnt; ++i)
    {
      all_data(i, v) = vec_view(i, 0);
    }
  }
  std::ofstream solnOUT(filename.c_str());
  solnOUT.precision(12);
  for (size_type i = 0; i < numEnt; ++i)
  {
    if (only_write_final)
    {
      solnOUT << all_data(i, numVecs - 1) << "  ";
    }
    else
    {
      for (size_type v = 0; v < numVecs; ++v)
      {
        solnOUT << all_data(i, v) << "  ";
      }
    }
    solnOUT << endl;
  }
  solnOUT.close();
}
