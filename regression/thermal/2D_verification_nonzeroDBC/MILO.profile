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
    MinOverProcs: 0.00658107
    MeanOverProcs: 0.00658172
    MaxOverProcs: 0.00658202
    MeanOverCallCounts: 0.00219391
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000294209
    MeanOverProcs: 0.000299454
    MaxOverProcs: 0.000303984
    MeanOverCallCounts: 1.36115e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000152111
    MeanOverProcs: 0.000163853
    MaxOverProcs: 0.000181913
    MeanOverCallCounts: 4.55148e-06
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000173092
    MeanOverProcs: 0.00017482
    MaxOverProcs: 0.00017643
    MeanOverCallCounts: 7.94638e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.000741482
    MeanOverProcs: 0.000752866
    MaxOverProcs: 0.000763178
    MeanOverCallCounts: 5.37762e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000464678
    MeanOverProcs: 0.000490725
    MaxOverProcs: 0.000517845
    MeanOverCallCounts: 2.72625e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.00902557
    MeanOverProcs: 0.00904453
    MaxOverProcs: 0.0090785
    MeanOverCallCounts: 0.000646038
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00155711
    MeanOverProcs: 0.00158256
    MaxOverProcs: 0.00160146
    MeanOverCallCounts: 5.65201e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.0013082
    MeanOverProcs: 0.00198179
    MaxOverProcs: 0.00269914
    MeanOverCallCounts: 0.000660598
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 7.39098e-05
    MeanOverProcs: 7.73072e-05
    MaxOverProcs: 8.41618e-05
    MeanOverCallCounts: 3.86536e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000257969
    MeanOverProcs: 0.000274479
    MaxOverProcs: 0.000309944
    MeanOverCallCounts: 1.3724e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00733614
    MeanOverProcs: 0.00780457
    MaxOverProcs: 0.00816488
    MeanOverCallCounts: 0.000390229
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00774789
    MeanOverProcs: 0.00821918
    MaxOverProcs: 0.0085628
    MeanOverCallCounts: 0.00410959
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00641394
    MeanOverProcs: 0.00671774
    MaxOverProcs: 0.00735497
    MeanOverCallCounts: 0.00671774
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000197887
    MeanOverProcs: 0.000211656
    MaxOverProcs: 0.000226974
    MeanOverCallCounts: 0.000211656
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.40667e-05
    MeanOverProcs: 1.5974e-05
    MaxOverProcs: 1.78814e-05
    MeanOverCallCounts: 7.98702e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.00013113
    MeanOverProcs: 0.000192583
    MaxOverProcs: 0.000251055
    MeanOverCallCounts: 0.000192583
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000314951
    MeanOverProcs: 0.00032264
    MaxOverProcs: 0.000326872
    MeanOverCallCounts: 0.00032264
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 9.08375e-05
    MeanOverProcs: 0.000141501
    MaxOverProcs: 0.000201941
    MeanOverCallCounts: 5.66006e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 9.01222e-05
    MeanOverProcs: 0.000105381
    MaxOverProcs: 0.000121355
    MeanOverCallCounts: 5.26905e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 4.29153e-05
    MeanOverProcs: 4.57168e-05
    MaxOverProcs: 4.8399e-05
    MeanOverCallCounts: 2.28584e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00332808
    MeanOverProcs: 0.00351083
    MaxOverProcs: 0.00363302
    MeanOverCallCounts: 0.000175542
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000111818
    MeanOverProcs: 0.000125647
    MaxOverProcs: 0.00014019
    MeanOverCallCounts: 4.18822e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00376272
    MeanOverProcs: 0.00392944
    MaxOverProcs: 0.00404143
    MeanOverCallCounts: 9.82359e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00523782
    MeanOverProcs: 0.00559705
    MaxOverProcs: 0.00587821
    MeanOverCallCounts: 0.000186568
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00127172
    MeanOverProcs: 0.00133413
    MaxOverProcs: 0.0014627
    MeanOverCallCounts: 0.000133413
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00342631
    MeanOverProcs: 0.00356668
    MaxOverProcs: 0.00393677
    MeanOverCallCounts: 0.000356668
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.000731945
    MeanOverProcs: 0.000757515
    MaxOverProcs: 0.000780106
    MeanOverCallCounts: 0.000757515
  "MILO::driver::total run time": 
    MinOverProcs: 0.102986
    MeanOverProcs: 0.103048
    MaxOverProcs: 0.103101
    MeanOverCallCounts: 0.103048
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.192345
    MeanOverProcs: 0.192752
    MaxOverProcs: 0.193591
    MeanOverCallCounts: 0.192752
  "MILO::function::decompose": 
    MinOverProcs: 0.000854015
    MeanOverProcs: 0.000896215
    MaxOverProcs: 0.00094986
    MeanOverCallCounts: 0.000896215
  "MILO::function::evaluate": 
    MinOverProcs: 0.00168633
    MeanOverProcs: 0.00174713
    MaxOverProcs: 0.00179696
    MeanOverCallCounts: 1.07516e-05
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 3.14713e-05
    MeanOverProcs: 3.42131e-05
    MaxOverProcs: 4.07696e-05
    MeanOverCallCounts: 2.73705e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000234842
    MeanOverProcs: 0.000348151
    MaxOverProcs: 0.000427961
    MeanOverCallCounts: 0.000348151
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.6982e-05
    MeanOverProcs: 8.07643e-05
    MaxOverProcs: 0.000106096
    MeanOverCallCounts: 8.07643e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00676394
    MeanOverProcs: 0.00705892
    MaxOverProcs: 0.00728083
    MeanOverCallCounts: 0.00705892
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.0087359
    MeanOverProcs: 0.008964
    MaxOverProcs: 0.00926995
    MeanOverCallCounts: 0.008964
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.044548
    MeanOverProcs: 0.0445485
    MaxOverProcs: 0.044549
    MeanOverCallCounts: 0.0222743
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0291779
    MeanOverProcs: 0.0292347
    MaxOverProcs: 0.029285
    MeanOverCallCounts: 0.0292347
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.19209e-05
    MeanOverProcs: 1.24574e-05
    MaxOverProcs: 1.3113e-05
    MeanOverCallCounts: 1.24574e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.0222301
    MeanOverProcs: 0.0222372
    MaxOverProcs: 0.0222418
    MeanOverCallCounts: 0.0222372
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 3.19481e-05
    MeanOverProcs: 3.8445e-05
    MaxOverProcs: 4.48227e-05
    MeanOverCallCounts: 3.8445e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00177288
    MeanOverProcs: 0.00177383
    MaxOverProcs: 0.00177479
    MeanOverCallCounts: 0.00177383
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00242114
    MeanOverProcs: 0.00255954
    MaxOverProcs: 0.00267076
    MeanOverCallCounts: 0.000127977
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000874758
    MeanOverProcs: 0.000918567
    MaxOverProcs: 0.000946522
    MeanOverCallCounts: 4.59284e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000541925
    MeanOverProcs: 0.000568748
    MaxOverProcs: 0.000597
    MeanOverCallCounts: 8.12496e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00891757
    MeanOverProcs: 0.00925666
    MaxOverProcs: 0.00963306
    MeanOverCallCounts: 0.000132238
  "MILO::workset::reset*": 
    MinOverProcs: 0.00010705
    MeanOverProcs: 0.000109851
    MaxOverProcs: 0.000115871
    MeanOverCallCounts: 5.49257e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0366721
    MeanOverProcs: 0.0374373
    MaxOverProcs: 0.0381069
    MeanOverCallCounts: 0.0124791
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.000978947
    MeanOverProcs: 0.00119901
    MaxOverProcs: 0.00149202
    MeanOverCallCounts: 0.00119901
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.00759602
    MeanOverProcs: 0.0076068
    MaxOverProcs: 0.007617
    MeanOverCallCounts: 0.0076068
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000494003
    MeanOverProcs: 0.000505924
    MaxOverProcs: 0.000522852
    MeanOverCallCounts: 0.000168641
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00291395
    MeanOverProcs: 0.00301099
    MaxOverProcs: 0.00312901
    MeanOverCallCounts: 0.00301099
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 0.000123978
    MeanOverProcs: 0.000154555
    MaxOverProcs: 0.000197172
    MeanOverCallCounts: 0.000154555
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.0001719
    MeanOverProcs: 0.000217021
    MaxOverProcs: 0.000252962
    MeanOverCallCounts: 0.000217021
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000195026
    MeanOverProcs: 0.000208497
    MaxOverProcs: 0.000246048
    MeanOverCallCounts: 0.000208497
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000246048
    MeanOverProcs: 0.000340283
    MaxOverProcs: 0.00039506
    MeanOverCallCounts: 0.000340283
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.00112891
    MeanOverProcs: 0.00117642
    MaxOverProcs: 0.00124002
    MeanOverCallCounts: 0.00117642
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000446081
    MeanOverProcs: 0.000476241
    MaxOverProcs: 0.00050211
    MeanOverCallCounts: 0.000476241
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 1.00136e-05
    MeanOverProcs: 1.12057e-05
    MaxOverProcs: 1.28746e-05
    MeanOverCallCounts: 1.12057e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000484943
    MeanOverProcs: 0.00049156
    MaxOverProcs: 0.000499964
    MeanOverCallCounts: 0.00049156
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 2.86102e-06
    MeanOverProcs: 3.15905e-06
    MaxOverProcs: 3.8147e-06
    MeanOverCallCounts: 3.15905e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 7.98702e-05
    MeanOverProcs: 9.75132e-05
    MaxOverProcs: 0.000120163
    MeanOverCallCounts: 9.75132e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.09673e-05
    MeanOverProcs: 1.2517e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 1.2517e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.7895e-05
    MeanOverProcs: 3.89814e-05
    MaxOverProcs: 5.91278e-05
    MeanOverCallCounts: 3.89814e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000397921
    MeanOverProcs: 0.00042057
    MaxOverProcs: 0.000441074
    MeanOverCallCounts: 0.00042057
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 5.29289e-05
    MeanOverProcs: 5.62668e-05
    MaxOverProcs: 6.10352e-05
    MeanOverCallCounts: 5.62668e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 5.60284e-05
    MeanOverProcs: 7.23004e-05
    MaxOverProcs: 8.58307e-05
    MeanOverCallCounts: 7.23004e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000243902
    MeanOverProcs: 0.000263751
    MaxOverProcs: 0.000282049
    MeanOverCallCounts: 0.000263751
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.00018692
    MeanOverProcs: 0.000189722
    MaxOverProcs: 0.000192881
    MeanOverCallCounts: 0.000189722
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.0492342
    MeanOverProcs: 0.0492433
    MaxOverProcs: 0.049257
    MeanOverCallCounts: 0.0492433
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
    MaxOverProcs: 13
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
    MinOverProcs: 70
    MeanOverProcs: 70
    MaxOverProcs: 70
    MeanOverCallCounts: 70
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 70
    MeanOverProcs: 70
    MaxOverProcs: 70
    MeanOverCallCounts: 70
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
