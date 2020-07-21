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
    MinOverProcs: 0.00681281
    MeanOverProcs: 0.00685662
    MaxOverProcs: 0.00688505
    MeanOverCallCounts: 0.00228554
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000286341
    MeanOverProcs: 0.00029403
    MaxOverProcs: 0.000299692
    MeanOverCallCounts: 1.3365e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000185728
    MeanOverProcs: 0.000265837
    MaxOverProcs: 0.000315666
    MeanOverCallCounts: 7.38435e-06
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000164032
    MeanOverProcs: 0.000175297
    MaxOverProcs: 0.000186682
    MeanOverCallCounts: 7.96806e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.000787258
    MeanOverProcs: 0.000855803
    MaxOverProcs: 0.000887156
    MeanOverCallCounts: 6.11288e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000452518
    MeanOverProcs: 0.000494599
    MaxOverProcs: 0.000530243
    MeanOverCallCounts: 2.74777e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.00910163
    MeanOverProcs: 0.00914311
    MaxOverProcs: 0.00917888
    MeanOverCallCounts: 0.00065308
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00160527
    MeanOverProcs: 0.00162876
    MaxOverProcs: 0.00165009
    MeanOverCallCounts: 5.81699e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00177693
    MeanOverProcs: 0.00595152
    MaxOverProcs: 0.0100551
    MeanOverCallCounts: 0.00198384
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 7.10487e-05
    MeanOverProcs: 8.60691e-05
    MaxOverProcs: 9.89437e-05
    MeanOverCallCounts: 4.30346e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000239611
    MeanOverProcs: 0.000261307
    MaxOverProcs: 0.000306606
    MeanOverCallCounts: 1.30653e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00711751
    MeanOverProcs: 0.00780958
    MaxOverProcs: 0.0088644
    MeanOverCallCounts: 0.000390479
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00752401
    MeanOverProcs: 0.00822651
    MaxOverProcs: 0.00932097
    MeanOverCallCounts: 0.00411326
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00632691
    MeanOverProcs: 0.00637144
    MaxOverProcs: 0.00647092
    MeanOverCallCounts: 0.00637144
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000170946
    MeanOverProcs: 0.000182688
    MaxOverProcs: 0.000192881
    MeanOverCallCounts: 0.000182688
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.38283e-05
    MeanOverProcs: 1.66893e-05
    MaxOverProcs: 2.09808e-05
    MeanOverCallCounts: 8.34465e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000124931
    MeanOverProcs: 0.000160515
    MaxOverProcs: 0.00020504
    MeanOverCallCounts: 0.000160515
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.00028491
    MeanOverProcs: 0.000297725
    MaxOverProcs: 0.00030899
    MeanOverCallCounts: 0.000297725
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 9.36985e-05
    MeanOverProcs: 0.000146508
    MaxOverProcs: 0.00019908
    MeanOverCallCounts: 5.86033e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 8.70228e-05
    MeanOverProcs: 9.5129e-05
    MaxOverProcs: 0.000105143
    MeanOverCallCounts: 4.75645e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.76701e-05
    MeanOverProcs: 4.20213e-05
    MaxOverProcs: 4.93526e-05
    MeanOverCallCounts: 2.10106e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.0031302
    MeanOverProcs: 0.00340098
    MaxOverProcs: 0.00386143
    MeanOverCallCounts: 0.000170049
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000118017
    MeanOverProcs: 0.000126898
    MaxOverProcs: 0.000138521
    MeanOverCallCounts: 4.22994e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00339174
    MeanOverProcs: 0.00342268
    MaxOverProcs: 0.0034678
    MeanOverCallCounts: 8.55669e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00516868
    MeanOverProcs: 0.00556695
    MaxOverProcs: 0.0061152
    MeanOverCallCounts: 0.000185565
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00125766
    MeanOverProcs: 0.00127572
    MaxOverProcs: 0.00128293
    MeanOverCallCounts: 0.000127572
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00332141
    MeanOverProcs: 0.00338906
    MaxOverProcs: 0.0034256
    MeanOverCallCounts: 0.000338906
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.000689983
    MeanOverProcs: 0.000697553
    MaxOverProcs: 0.000705004
    MeanOverCallCounts: 0.000697553
  "MILO::driver::total run time": 
    MinOverProcs: 0.116914
    MeanOverProcs: 0.116949
    MaxOverProcs: 0.11699
    MeanOverCallCounts: 0.116949
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.19374
    MeanOverProcs: 0.194275
    MaxOverProcs: 0.194963
    MeanOverCallCounts: 0.194275
  "MILO::function::decompose": 
    MinOverProcs: 0.000762939
    MeanOverProcs: 0.000778317
    MaxOverProcs: 0.000808954
    MeanOverCallCounts: 0.000778317
  "MILO::function::evaluate": 
    MinOverProcs: 0.00152707
    MeanOverProcs: 0.00157881
    MaxOverProcs: 0.00160527
    MeanOverCallCounts: 9.71574e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 2.81334e-05
    MeanOverProcs: 3.49283e-05
    MaxOverProcs: 4.14848e-05
    MeanOverCallCounts: 2.79427e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000191927
    MeanOverProcs: 0.00028348
    MaxOverProcs: 0.000355959
    MeanOverCallCounts: 0.00028348
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.60284e-05
    MeanOverProcs: 7.37309e-05
    MaxOverProcs: 9.20296e-05
    MeanOverCallCounts: 7.37309e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00617385
    MeanOverProcs: 0.00621039
    MaxOverProcs: 0.00626683
    MeanOverCallCounts: 0.00621039
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.015929
    MeanOverProcs: 0.0159931
    MaxOverProcs: 0.0160542
    MeanOverCallCounts: 0.0159931
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.0462821
    MeanOverProcs: 0.0462971
    MaxOverProcs: 0.0463202
    MeanOverCallCounts: 0.0231485
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.034116
    MeanOverProcs: 0.0341491
    MaxOverProcs: 0.0341871
    MeanOverCallCounts: 0.0341491
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 2.00272e-05
    MeanOverProcs: 2.27094e-05
    MaxOverProcs: 2.59876e-05
    MeanOverCallCounts: 2.27094e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.0211651
    MeanOverProcs: 0.0211691
    MaxOverProcs: 0.021173
    MeanOverCallCounts: 0.0211691
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.7895e-05
    MeanOverProcs: 3.02196e-05
    MaxOverProcs: 3.31402e-05
    MeanOverCallCounts: 3.02196e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00184608
    MeanOverProcs: 0.00184685
    MaxOverProcs: 0.00184822
    MeanOverCallCounts: 0.00184685
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00227356
    MeanOverProcs: 0.00248969
    MaxOverProcs: 0.00288367
    MeanOverCallCounts: 0.000124484
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000811815
    MeanOverProcs: 0.000872254
    MaxOverProcs: 0.000937939
    MeanOverCallCounts: 4.36127e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000481129
    MeanOverProcs: 0.000500917
    MaxOverProcs: 0.000522375
    MeanOverCallCounts: 7.15596e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00830626
    MeanOverProcs: 0.00872177
    MaxOverProcs: 0.00923228
    MeanOverCallCounts: 0.000124597
  "MILO::workset::reset*": 
    MinOverProcs: 0.000118256
    MeanOverProcs: 0.000127435
    MaxOverProcs: 0.000147343
    MeanOverCallCounts: 6.37174e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.036068
    MeanOverProcs: 0.0400455
    MaxOverProcs: 0.044414
    MeanOverCallCounts: 0.0133485
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.000878811
    MeanOverProcs: 0.000946939
    MaxOverProcs: 0.00099802
    MeanOverCallCounts: 0.000946939
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.0148702
    MeanOverProcs: 0.0148956
    MaxOverProcs: 0.0149062
    MeanOverCallCounts: 0.0148956
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000471115
    MeanOverProcs: 0.000489354
    MaxOverProcs: 0.000516176
    MeanOverCallCounts: 0.000163118
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00240421
    MeanOverProcs: 0.00246978
    MaxOverProcs: 0.00254703
    MeanOverCallCounts: 0.00246978
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 9.29832e-05
    MeanOverProcs: 0.00010407
    MaxOverProcs: 0.000136137
    MeanOverCallCounts: 0.00010407
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000164986
    MeanOverProcs: 0.000169218
    MaxOverProcs: 0.000172853
    MeanOverCallCounts: 0.000169218
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000185966
    MeanOverProcs: 0.000187516
    MaxOverProcs: 0.000190973
    MeanOverCallCounts: 0.000187516
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000210047
    MeanOverProcs: 0.000248015
    MaxOverProcs: 0.00029397
    MeanOverCallCounts: 0.000248015
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.000938892
    MeanOverProcs: 0.000961721
    MaxOverProcs: 0.000987053
    MeanOverCallCounts: 0.000961721
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000379801
    MeanOverProcs: 0.000382423
    MaxOverProcs: 0.000384808
    MeanOverCallCounts: 0.000382423
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 8.82149e-06
    MeanOverProcs: 9.0003e-06
    MaxOverProcs: 9.05991e-06
    MeanOverCallCounts: 9.0003e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000360966
    MeanOverProcs: 0.000386178
    MaxOverProcs: 0.000396013
    MeanOverCallCounts: 0.000386178
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 2.86102e-06
    MeanOverProcs: 3.03984e-06
    MaxOverProcs: 3.09944e-06
    MeanOverCallCounts: 3.03984e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 7.60555e-05
    MeanOverProcs: 0.000103295
    MaxOverProcs: 0.000145912
    MeanOverCallCounts: 0.000103295
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.19209e-05
    MeanOverProcs: 1.24574e-05
    MaxOverProcs: 1.28746e-05
    MeanOverCallCounts: 1.24574e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.5034e-05
    MeanOverProcs: 3.3021e-05
    MaxOverProcs: 4.00543e-05
    MeanOverCallCounts: 3.3021e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000335932
    MeanOverProcs: 0.000351727
    MaxOverProcs: 0.000366926
    MeanOverCallCounts: 0.000351727
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 4.91142e-05
    MeanOverProcs: 5.06043e-05
    MaxOverProcs: 5.31673e-05
    MeanOverCallCounts: 5.06043e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.81606e-05
    MeanOverProcs: 4.87566e-05
    MaxOverProcs: 4.91142e-05
    MeanOverCallCounts: 4.87566e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000209808
    MeanOverProcs: 0.000226736
    MaxOverProcs: 0.000243187
    MeanOverCallCounts: 0.000226736
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000169039
    MeanOverProcs: 0.000174522
    MaxOverProcs: 0.000185013
    MeanOverCallCounts: 0.000174522
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.0432181
    MeanOverProcs: 0.0432225
    MaxOverProcs: 0.0432281
    MeanOverCallCounts: 0.0432225
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
    MinOverProcs: 163
    MeanOverProcs: 162.5
    MaxOverProcs: 162
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
