# Teuchos::TimeMonitor report
---
Output mode: spacious
Number of processes: 4
Time unit: s
Statistics collected: 
  - MinOverProcs
  - MeanOverProcs
  - MaxOverProcs
  - MeanOverCallCounts
Timer names: 
  - "Belos: BlockGmresSolMgr total solve time"
  - "Belos: DGKS[2]: Ortho (Inner Product)"
  - "Belos: DGKS[2]: Ortho (Norm)"
  - "Belos: DGKS[2]: Ortho (Update)"
  - "Belos: DGKS[2]: Orthogonalization"
  - "Belos: Operation Op*x"
  - "Belos: Operation Prec*x"
  - "Ifpack2::Chebyshev::apply"
  - "Ifpack2::Chebyshev::compute"
  - "MILO::assembly::computeJacRes() - gather"
  - "MILO::assembly::computeJacRes() - insert"
  - "MILO::assembly::computeJacRes() - physics evaluation"
  - "MILO::assembly::computeJacRes() - total assembly"
  - "MILO::assembly::createCells()"
  - "MILO::assembly::createWorkset()"
  - "MILO::assembly::dofConstraints()"
  - "MILO::assembly::setDirichlet()"
  - "MILO::assembly::setInitial()"
  - "MILO::boundaryCell - build basis"
  - "MILO::cell::computeJacRes() - fill local Jacobian"
  - "MILO::cell::computeJacRes() - fill local residual"
  - "MILO::cell::computeJacRes() - volume residual"
  - "MILO::cell::computeSolAvg()"
  - "MILO::cell::computeSolnFaceIP()"
  - "MILO::cell::computeSolnVolIP()"
  - "MILO::cell::constructor - build basis"
  - "MILO::cell::constructor - build face basis"
  - "MILO::cellMetaData::constructor()"
  - "MILO::driver::total run time"
  - "MILO::driver::total setup and execution time"
  - "MILO::function::decompose"
  - "MILO::function::evaluate"
  - "MILO::physics::getSideInfo()"
  - "MILO::physics::setBCData()"
  - "MILO::physics::setDirichletData()"
  - "MILO::postprocess::computeError"
  - "MILO::postprocess::writeSolution"
  - "MILO::solver::linearSolver()"
  - "MILO::solver::projectDirichlet()"
  - "MILO::solver::setDirichlet()"
  - "MILO::solver::setInitial()"
  - "MILO::solver::setupFixedDOFs()"
  - "MILO::solver::setupLinearAlgebra()"
  - "MILO::thermal::volumeResidual() - evaluation of residual"
  - "MILO::thermal::volumeResidual() - function evaluation"
  - "MILO::workset::computeSolnVolIP - allocate/compute seeded"
  - "MILO::workset::computeSolnVolIP - compute seeded sol at ip"
  - "MILO::workset::reset*"
  - "MueLu: ParameterListInterpreter (ParameterList)"
  - "STK_Interface::setupExodusFile(filename)"
  - "STK_Interface::writeToExodus(timestep)"
  - "UtilitiesBase::GetMatrixDiagonalInverse"
  - "panzer::DOFManager::buildGlobalUnknowns"
  - "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array"
  - "panzer::DOFManager::buildGlobalUnknowns::build_local_ids"
  - "panzer::DOFManager::buildGlobalUnknowns::build_orientation"
  - "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment"
  - "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import"
  - "panzer::DOFManager::buildTaggedMultiVector"
  - "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector"
  - "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector"
  - "panzer::DOFManager::builderOverlapMapFromElements"
  - "panzer::SquareQuadMeshFactory::buildUncomittedMesh()"
  - "panzer::SquareQuadMeshFactory::completeMeshConstruction()"
Total times: 
  "Belos: BlockGmresSolMgr total solve time": 
    MinOverProcs: 0.00894976
    MeanOverProcs: 0.00908625
    MaxOverProcs: 0.00917315
    MeanOverCallCounts: 0.00302875
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000379324
    MeanOverProcs: 0.000388682
    MaxOverProcs: 0.00039959
    MeanOverCallCounts: 1.76674e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000393391
    MeanOverProcs: 0.000654399
    MaxOverProcs: 0.000797033
    MeanOverCallCounts: 1.81778e-05
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000170231
    MeanOverProcs: 0.000191629
    MaxOverProcs: 0.000229359
    MeanOverCallCounts: 8.71041e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.00116229
    MeanOverProcs: 0.00136745
    MaxOverProcs: 0.00149941
    MeanOverCallCounts: 9.7675e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000566244
    MeanOverProcs: 0.000796795
    MaxOverProcs: 0.00140786
    MeanOverCallCounts: 4.42664e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.011641
    MeanOverProcs: 0.0119233
    MaxOverProcs: 0.012146
    MeanOverCallCounts: 0.000851665
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00218296
    MeanOverProcs: 0.00228673
    MaxOverProcs: 0.00242686
    MeanOverCallCounts: 8.1669e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.0017314
    MeanOverProcs: 0.00258005
    MaxOverProcs: 0.0037303
    MeanOverCallCounts: 0.000860016
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 6.79493e-05
    MeanOverProcs: 7.18236e-05
    MaxOverProcs: 7.77245e-05
    MeanOverCallCounts: 3.59118e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000201941
    MeanOverProcs: 0.000219762
    MaxOverProcs: 0.000233889
    MeanOverCallCounts: 1.09881e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00652099
    MeanOverProcs: 0.00658226
    MaxOverProcs: 0.00665998
    MeanOverCallCounts: 0.000329113
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00684118
    MeanOverProcs: 0.00692296
    MaxOverProcs: 0.00702429
    MeanOverCallCounts: 0.00346148
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00664902
    MeanOverProcs: 0.00708401
    MaxOverProcs: 0.00761294
    MeanOverCallCounts: 0.00708401
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000176907
    MeanOverProcs: 0.000179708
    MaxOverProcs: 0.000181913
    MeanOverCallCounts: 0.000179708
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.28746e-05
    MeanOverProcs: 1.42455e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 7.12276e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000138044
    MeanOverProcs: 0.000155032
    MaxOverProcs: 0.000170946
    MeanOverCallCounts: 0.000155032
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000398874
    MeanOverProcs: 0.000509739
    MaxOverProcs: 0.000631094
    MeanOverCallCounts: 0.000509739
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 8.17776e-05
    MeanOverProcs: 0.000140131
    MaxOverProcs: 0.000193834
    MeanOverCallCounts: 5.60522e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 7.34329e-05
    MeanOverProcs: 7.90358e-05
    MaxOverProcs: 8.86917e-05
    MeanOverCallCounts: 3.95179e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.95775e-05
    MeanOverProcs: 4.35114e-05
    MaxOverProcs: 4.95911e-05
    MeanOverCallCounts: 2.17557e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00278783
    MeanOverProcs: 0.0028187
    MaxOverProcs: 0.00285339
    MeanOverCallCounts: 0.000140935
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000109434
    MeanOverProcs: 0.000115037
    MaxOverProcs: 0.000125647
    MeanOverCallCounts: 3.83457e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00362325
    MeanOverProcs: 0.00377518
    MaxOverProcs: 0.00394297
    MeanOverCallCounts: 9.43795e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00524592
    MeanOverProcs: 0.00533628
    MaxOverProcs: 0.00543046
    MeanOverCallCounts: 0.000177876
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00124574
    MeanOverProcs: 0.0013361
    MaxOverProcs: 0.00143313
    MeanOverCallCounts: 0.00013361
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00361609
    MeanOverProcs: 0.00391781
    MaxOverProcs: 0.00420809
    MeanOverCallCounts: 0.000391781
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.00067997
    MeanOverProcs: 0.000689209
    MaxOverProcs: 0.000702858
    MeanOverCallCounts: 0.000689209
  "MILO::driver::total run time": 
    MinOverProcs: 0.112882
    MeanOverProcs: 0.112903
    MaxOverProcs: 0.112926
    MeanOverCallCounts: 0.112903
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.195345
    MeanOverProcs: 0.195383
    MaxOverProcs: 0.195426
    MeanOverCallCounts: 0.195383
  "MILO::function::decompose": 
    MinOverProcs: 0.000715017
    MeanOverProcs: 0.00073278
    MaxOverProcs: 0.00075388
    MeanOverCallCounts: 0.00073278
  "MILO::function::evaluate": 
    MinOverProcs: 0.00149131
    MeanOverProcs: 0.00158477
    MaxOverProcs: 0.00164247
    MeanOverCallCounts: 9.75242e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 3.31402e-05
    MeanOverProcs: 4.05312e-05
    MaxOverProcs: 4.64916e-05
    MeanOverCallCounts: 3.24249e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.00019002
    MeanOverProcs: 0.00075376
    MaxOverProcs: 0.00111485
    MeanOverCallCounts: 0.00075376
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.38826e-05
    MeanOverProcs: 9.56655e-05
    MaxOverProcs: 0.000154972
    MeanOverCallCounts: 9.56655e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00625014
    MeanOverProcs: 0.00654131
    MaxOverProcs: 0.00677013
    MeanOverCallCounts: 0.00654131
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.0131021
    MeanOverProcs: 0.0133075
    MaxOverProcs: 0.0136139
    MeanOverCallCounts: 0.0133075
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.0484419
    MeanOverProcs: 0.0484516
    MaxOverProcs: 0.0484769
    MeanOverCallCounts: 0.0242258
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0326529
    MeanOverProcs: 0.0327289
    MaxOverProcs: 0.0327909
    MeanOverCallCounts: 0.0327289
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.19209e-05
    MeanOverProcs: 1.22786e-05
    MaxOverProcs: 1.3113e-05
    MeanOverCallCounts: 1.22786e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.021678
    MeanOverProcs: 0.0217565
    MaxOverProcs: 0.0218441
    MeanOverCallCounts: 0.0217565
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.59876e-05
    MeanOverProcs: 2.67029e-05
    MaxOverProcs: 2.69413e-05
    MeanOverCallCounts: 2.67029e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00154305
    MeanOverProcs: 0.00154489
    MaxOverProcs: 0.00154614
    MeanOverCallCounts: 0.00154489
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.0020287
    MeanOverProcs: 0.00204188
    MaxOverProcs: 0.00205564
    MeanOverCallCounts: 0.000102094
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000720978
    MeanOverProcs: 0.000743926
    MaxOverProcs: 0.000766754
    MeanOverCallCounts: 3.71963e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000269175
    MeanOverProcs: 0.00028193
    MaxOverProcs: 0.000299931
    MeanOverCallCounts: 9.39767e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00476575
    MeanOverProcs: 0.00485647
    MaxOverProcs: 0.00492382
    MeanOverCallCounts: 0.000161882
  "MILO::workset::reset*": 
    MinOverProcs: 0.000109196
    MeanOverProcs: 0.000115514
    MaxOverProcs: 0.000119209
    MeanOverCallCounts: 5.77569e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0348449
    MeanOverProcs: 0.0358977
    MaxOverProcs: 0.0365341
    MeanOverCallCounts: 0.0119659
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.000761032
    MeanOverProcs: 0.000979006
    MaxOverProcs: 0.00127506
    MeanOverCallCounts: 0.000979006
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.0121531
    MeanOverProcs: 0.0121922
    MaxOverProcs: 0.0122099
    MeanOverCallCounts: 0.0121922
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.00046587
    MeanOverProcs: 0.000542581
    MaxOverProcs: 0.000627041
    MeanOverCallCounts: 0.00018086
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.003093
    MeanOverProcs: 0.00337899
    MaxOverProcs: 0.00399613
    MeanOverCallCounts: 0.00337899
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 8.60691e-05
    MeanOverProcs: 0.000120342
    MaxOverProcs: 0.000161171
    MeanOverCallCounts: 0.000120342
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000158072
    MeanOverProcs: 0.000199497
    MaxOverProcs: 0.000306845
    MeanOverCallCounts: 0.000199497
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000178099
    MeanOverProcs: 0.000224829
    MaxOverProcs: 0.000341177
    MeanOverCallCounts: 0.000224829
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000209093
    MeanOverProcs: 0.000291049
    MaxOverProcs: 0.000372171
    MeanOverCallCounts: 0.000291049
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.00122595
    MeanOverProcs: 0.00125968
    MaxOverProcs: 0.00129294
    MeanOverCallCounts: 0.00125968
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000532866
    MeanOverProcs: 0.000550985
    MaxOverProcs: 0.000564098
    MeanOverCallCounts: 0.000550985
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 9.05991e-06
    MeanOverProcs: 1.20401e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 1.20401e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000431061
    MeanOverProcs: 0.000468493
    MaxOverProcs: 0.000492811
    MeanOverCallCounts: 0.000468493
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 2.86102e-06
    MeanOverProcs: 3.51667e-06
    MaxOverProcs: 4.05312e-06
    MeanOverCallCounts: 3.51667e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 9.799e-05
    MeanOverProcs: 0.000123024
    MaxOverProcs: 0.000194073
    MeanOverCallCounts: 0.000123024
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.09673e-05
    MeanOverProcs: 1.32322e-05
    MaxOverProcs: 1.5974e-05
    MeanOverCallCounts: 1.32322e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 3.79086e-05
    MeanOverProcs: 4.39882e-05
    MaxOverProcs: 4.69685e-05
    MeanOverCallCounts: 4.39882e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000519991
    MeanOverProcs: 0.000730455
    MaxOverProcs: 0.000814915
    MeanOverCallCounts: 0.000730455
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 6.38962e-05
    MeanOverProcs: 6.69956e-05
    MaxOverProcs: 7.10487e-05
    MeanOverCallCounts: 6.69956e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 7.20024e-05
    MeanOverProcs: 7.4327e-05
    MaxOverProcs: 7.60555e-05
    MeanOverCallCounts: 7.4327e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000347853
    MeanOverProcs: 0.000558972
    MaxOverProcs: 0.000643969
    MeanOverCallCounts: 0.000558972
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000163794
    MeanOverProcs: 0.000166178
    MaxOverProcs: 0.000169992
    MeanOverCallCounts: 0.000166178
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.044112
    MeanOverProcs: 0.0441381
    MaxOverProcs: 0.0441642
    MeanOverCallCounts: 0.0441381
Call counts:
  "Belos: BlockGmresSolMgr total solve time": 
    MinOverProcs: 3
    MeanOverProcs: 3
    MaxOverProcs: 3
    MeanOverCallCounts: 3
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 22
    MeanOverProcs: 22
    MaxOverProcs: 22
    MeanOverCallCounts: 22
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 36
    MeanOverProcs: 36
    MaxOverProcs: 36
    MeanOverCallCounts: 36
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 22
    MeanOverProcs: 22
    MaxOverProcs: 22
    MeanOverCallCounts: 22
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 14
    MeanOverProcs: 14
    MaxOverProcs: 14
    MeanOverCallCounts: 14
  "Belos: Operation Op*x": 
    MinOverProcs: 18
    MeanOverProcs: 18
    MaxOverProcs: 18
    MeanOverCallCounts: 18
  "Belos: Operation Prec*x": 
    MinOverProcs: 14
    MeanOverProcs: 14
    MaxOverProcs: 14
    MeanOverCallCounts: 14
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 28
    MeanOverProcs: 28
    MaxOverProcs: 28
    MeanOverCallCounts: 28
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 3
    MeanOverProcs: 3
    MaxOverProcs: 3
    MeanOverCallCounts: 3
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
  "MILO::assembly::createCells()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::assembly::setInitial()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 2
    MeanOverProcs: 2.5
    MaxOverProcs: 3
    MeanOverCallCounts: 2.5
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 30
    MeanOverProcs: 30
    MaxOverProcs: 30
    MeanOverCallCounts: 30
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 40
    MeanOverProcs: 40
    MaxOverProcs: 40
    MeanOverCallCounts: 40
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 30
    MeanOverProcs: 30
    MaxOverProcs: 30
    MeanOverCallCounts: 30
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 10
    MeanOverProcs: 10
    MaxOverProcs: 10
    MeanOverCallCounts: 10
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 10
    MeanOverProcs: 10
    MaxOverProcs: 10
    MeanOverCallCounts: 10
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::driver::total run time": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::function::decompose": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::function::evaluate": 
    MinOverProcs: 162
    MeanOverProcs: 162.5
    MaxOverProcs: 163
    MeanOverCallCounts: 162.5
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 12
    MeanOverProcs: 12.5
    MaxOverProcs: 12
    MeanOverCallCounts: 12.5
  "MILO::physics::setBCData()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::postprocess::computeError": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::linearSolver()": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::setInitial()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 30
    MeanOverProcs: 30
    MaxOverProcs: 30
    MeanOverCallCounts: 30
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 30
    MeanOverProcs: 30
    MaxOverProcs: 30
    MeanOverCallCounts: 30
  "MILO::workset::reset*": 
    MinOverProcs: 20
    MeanOverProcs: 20
    MaxOverProcs: 20
    MeanOverCallCounts: 20
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 3
    MeanOverProcs: 3
    MaxOverProcs: 3
    MeanOverCallCounts: 3
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 3
    MeanOverProcs: 3
    MaxOverProcs: 3
    MeanOverCallCounts: 3
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
