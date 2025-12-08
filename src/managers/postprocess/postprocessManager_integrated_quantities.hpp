/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// Create a vector of integrated quantities objects on each block
// This version uses a parameter list
// ========================================================================================

template <class Node>
vector<integratedQuantity> PostprocessManager<Node>::addIntegratedQuantities(Teuchos::ParameterList &iqs,
                                                                             const size_t &block)
{
  vector<integratedQuantity> IQs;
  Teuchos::ParameterList::ConstIterator iqs_itr = iqs.begin();
  while (iqs_itr != iqs.end())
  {
    Teuchos::ParameterList iqsettings = iqs.sublist(iqs_itr->first);
    integratedQuantity newIQ(iqsettings, iqs_itr->first, block);
    IQs.push_back(newIQ);
    if (newIQ.location == "volume")
    {
      assembler->addFunction(block, newIQ.name + " integrand", newIQ.integrand, "ip");
    }
    else if (newIQ.location == "boundary")
    {
      assembler->addFunction(block, newIQ.name + " integrand", newIQ.integrand, "side ip");
    }

    iqs_itr++;
  }

  return IQs;
}


// ========================================================================================
// Create a vector of integrated quantities objects on each block
// This version uses a vector of vector of strings
// ========================================================================================

template <class Node>
vector<integratedQuantity>
PostprocessManager<Node>::addIntegratedQuantities(vector<vector<string>> &integrandsNamesAndTypes,
                                                  const size_t &block)
{

  vector<integratedQuantity> IQs;

  // first index is QoI, second index is 0 for integrand, 1 for name, 2 for type
  for (size_t iIQ = 0; iIQ < integrandsNamesAndTypes.size(); ++iIQ)
  {
    integratedQuantity newIQ(integrandsNamesAndTypes[iIQ][0],
                             integrandsNamesAndTypes[iIQ][1],
                             integrandsNamesAndTypes[iIQ][2],
                             block);
    IQs.push_back(newIQ);
    if (newIQ.location == "volume")
    {
      assembler->addFunction(block, newIQ.name + " integrand", newIQ.integrand, "ip");
    }
    else if (newIQ.location == "boundary")
    {
      assembler->addFunction(block, newIQ.name + " integrand", newIQ.integrand, "side ip");
    }
  }

  return IQs;
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeIntegratedQuantities(vector<vector_RCP> &current_soln, const ScalarT &currenttime)
{

  debugger->print(1, "******** Starting PostprocessManager::computeIntegratedQuantities ...");

  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type LA_device;

  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible)
  {
    data_avail = false;
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> sol_kv;
  for (size_t s = 0; s < current_soln.size(); ++s)
  {
    auto vec_kv = current_soln[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
    if (data_avail)
    {
      sol_kv.push_back(vec_slice);
    }
    else
    {
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

  if (data_avail)
  {
    params_kv.push_back(pslice);
  }
  else
  {
    auto p_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), pslice);
    Kokkos::deep_copy(p_dev, pslice);
    params_kv.push_back(p_dev);
  }

  // TODO :: BWR -- currently, I am proceeding like quantities are requested over
  // a subvolume (or subboundary, etc.) which is defined by the block
  // Hence, if a user wanted an integral over the ENTIRE volume, they would need to
  // sum up the individual contributions (in a multiblock case)

  for (size_t iLocal = 0; iLocal < integratedQuantities.size(); iLocal++)
  {

    // iLocal indexes over the number of blocks where IQs are defined and
    // does not necessarily match the global block ID

    size_t globalBlock = integratedQuantities[iLocal][0].block; // all IQs with same first index share a block

    vector<ScalarT> allsums; // For the final results after summing over MPI processes

    // the first n IQs are needed by the workset for residual calculations
    size_t nIQsForResidual = assembler->wkset[globalBlock]->integrated_quantities.extent(0);

    // MPI sums happen on the host and later we pass to the device (where residual is formed)
    auto hostsums = Kokkos::View<ScalarT *, HostDevice>("host IQs", nIQsForResidual);

    for (size_t iIQ = 0; iIQ < integratedQuantities[iLocal].size(); ++iIQ)
    {

      ScalarT integral = 0.;
      ScalarT localContribution;

      if (integratedQuantities[iLocal][iIQ].location == "volume")
      {

        for (size_t grp = 0; grp < assembler->groups[globalBlock].size(); ++grp)
        {

          localContribution = 0.; // zero out this grp's contribution JIC here but needed below

          // setup the workset for this grp
          if (!assembler->groups[globalBlock][grp]->have_sols)
          {
            for (size_t set = 0; set < sol_kv.size(); ++set)
            {
              assembler->performGather(set, globalBlock, grp, sol_kv[set], 0, 0);
            }
            assembler->performGather(0, globalBlock, grp, params_kv[0], 4, 0);
          }
          assembler->updateWorkset(globalBlock, grp, 0, 0, true);
          // get integration weights
          auto wts = assembler->wkset[globalBlock]->wts;
          // evaluate the integrand at integration points
          // auto integrand = functionManagers[globalBlock]->evaluate(integratedQuantities[iLocal][iIQ].name+" integrand","ip");
          View_Sc2 integrand = assembler->evaluateFunction(globalBlock, integratedQuantities[iLocal][iIQ].name + " integrand", "ip");
          // expand this for integral integrands, etc.?

          parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
            for (size_t pt=0; pt<wts.extent(1); pt++) {
              ScalarT Idx = wts(elem,pt)*integrand(elem,pt);
              update += Idx;
            } }, localContribution); //// TODO :: may be illegal

          // add this grp's contribution to running total

          integral += localContribution;

        } // end loop over groups
      }
      else if (integratedQuantities[iLocal][iIQ].location == "boundary")
      {

        assembler->wkset[globalBlock]->isOnSide = true;

        for (size_t grp = 0; grp < assembler->boundary_groups[globalBlock].size(); ++grp)
        {

          localContribution = 0.; // zero out this grp's contribution

          // check if we are on one of the requested sides
          string sidename = assembler->boundary_groups[globalBlock][grp]->sidename;
          size_t found = integratedQuantities[iLocal][iIQ].boundarynames.find(sidename);

          if ((found != std::string::npos) ||
              (integratedQuantities[iLocal][iIQ].boundarynames == "all"))
          {

            // setup the workset for this grp
            for (size_t set = 0; set < sol_kv.size(); ++set)
            {
              assembler->performBoundaryGather(set, globalBlock, grp, sol_kv[set], 0, 0);
            }
            assembler->performBoundaryGather(0, globalBlock, grp, params_kv[0], 4, 0);
            assembler->updateWorksetBoundary(globalBlock, grp, 0, 0, true);
            // get integration weights
            auto wts = assembler->wkset[globalBlock]->wts_side;
            // evaluate the integrand at integration points
            // auto integrand = functionManagers[globalBlock]->evaluate(integratedQuantities[iLocal][iIQ].name+" integrand","side ip");
            View_Sc2 integrand = assembler->evaluateFunction(globalBlock, integratedQuantities[iLocal][iIQ].name + " integrand", "side ip");
            parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
              for (size_t pt=0; pt<wts.extent(1); pt++) {
                ScalarT Idx = wts(elem,pt)*integrand(elem,pt);
                update += Idx;
              } }, localContribution); //// TODO :: may be illegal, problematic ABOVE TOO

          } // end if requested side
          // add in this grp's contribution to running total
          integral += localContribution;
        } // end loop over boundary groups

        assembler->wkset[globalBlock]->isOnSide = false;

      } // end if volume or boundary
      // finalize the integral
      integratedQuantities[iLocal][iIQ].val(0) = integral;
      // reduce
      ScalarT gval = 0.0;
      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &integral, &gval);
      if (iIQ < nIQsForResidual)
      {
        hostsums(iIQ) = gval;
      }
      allsums.push_back(gval);
      // save global result back in IQ storage
      integratedQuantities[iLocal][iIQ].val(0) = allsums[iIQ];
    } // end loop over integrated quantities

    // need to put in the right place now (accessible to the residual) and
    // update any parameters which depend on the IQs
    // TODO :: BWR this ultimately is an "explicit" idea but doing things implicitly
    // would be super costly in general.

    // TODO CHECK THIS WITH TIM... am I dev/loc correctly?
    if (nIQsForResidual > 0)
    {
      Kokkos::deep_copy(assembler->wkset[globalBlock]->integrated_quantities, hostsums);
      for (size_t set = 0; set < physics->modules.size(); ++set)
      {
        for (size_t m = 0; m < physics->modules[set][globalBlock].size(); ++m)
        {
          // BWR -- called for all physics defined on the block regards of if they need IQs
          physics->modules[set][globalBlock][m]->updateIntegratedQuantitiesDependents();
        }
      }
    } // end if physics module needs IQs

  } // end loop over blocks (with IQs requested)
}
