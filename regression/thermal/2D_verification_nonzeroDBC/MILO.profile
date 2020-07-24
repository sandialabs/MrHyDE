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
    MinOverProcs: 0.00906634
    MeanOverProcs: 0.00916779
    MaxOverProcs: 0.009233
    MeanOverCallCounts: 0.00305593
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000394821
    MeanOverProcs: 0.000407934
    MaxOverProcs: 0.000428677
    MeanOverCallCounts: 1.85425e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000661612
    MeanOverProcs: 0.000800788
    MaxOverProcs: 0.000890493
    MeanOverCallCounts: 2.22441e-05
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000170231
    MeanOverProcs: 0.000199318
    MaxOverProcs: 0.000252008
    MeanOverCallCounts: 9.05991e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.00146174
    MeanOverProcs: 0.00152022
    MaxOverProcs: 0.00156283
    MeanOverCallCounts: 0.000108587
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000499249
    MeanOverProcs: 0.00089097
    MaxOverProcs: 0.00119185
    MeanOverCallCounts: 4.94983e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.0116136
    MeanOverProcs: 0.0119811
    MaxOverProcs: 0.012296
    MeanOverCallCounts: 0.000855795
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00222564
    MeanOverProcs: 0.00232941
    MaxOverProcs: 0.00243449
    MeanOverCallCounts: 8.31932e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00537515
    MeanOverProcs: 0.00806677
    MaxOverProcs: 0.0100269
    MeanOverCallCounts: 0.00268892
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 6.8903e-05
    MeanOverProcs: 7.67708e-05
    MaxOverProcs: 8.41618e-05
    MeanOverCallCounts: 3.83854e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000188112
    MeanOverProcs: 0.000218093
    MaxOverProcs: 0.000247717
    MeanOverCallCounts: 1.09047e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00624657
    MeanOverProcs: 0.00714284
    MaxOverProcs: 0.00821114
    MeanOverCallCounts: 0.000357142
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.0065608
    MeanOverProcs: 0.0074895
    MaxOverProcs: 0.00858998
    MeanOverCallCounts: 0.00374475
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00641108
    MeanOverProcs: 0.00823784
    MaxOverProcs: 0.0105951
    MeanOverCallCounts: 0.00823784
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000174999
    MeanOverProcs: 0.000198305
    MaxOverProcs: 0.000240088
    MeanOverCallCounts: 0.000198305
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.09673e-05
    MeanOverProcs: 1.32322e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 6.61612e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000188828
    MeanOverProcs: 0.000616491
    MaxOverProcs: 0.000777006
    MeanOverCallCounts: 0.000616491
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000454903
    MeanOverProcs: 0.000547171
    MaxOverProcs: 0.000699997
    MeanOverCallCounts: 0.000547171
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 9.29832e-05
    MeanOverProcs: 0.00016737
    MaxOverProcs: 0.000296116
    MeanOverCallCounts: 6.69479e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 7.03335e-05
    MeanOverProcs: 8.52942e-05
    MaxOverProcs: 0.000105143
    MeanOverCallCounts: 4.26471e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.52859e-05
    MeanOverProcs: 4.20809e-05
    MaxOverProcs: 5.10216e-05
    MeanOverCallCounts: 2.10404e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00267529
    MeanOverProcs: 0.00305998
    MaxOverProcs: 0.00350595
    MeanOverCallCounts: 0.000152999
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000106096
    MeanOverProcs: 0.000118613
    MaxOverProcs: 0.000128031
    MeanOverCallCounts: 3.95377e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00361562
    MeanOverProcs: 0.00399965
    MaxOverProcs: 0.00453377
    MeanOverCallCounts: 9.99913e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00484824
    MeanOverProcs: 0.00556278
    MaxOverProcs: 0.00638843
    MeanOverCallCounts: 0.000185426
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.001266
    MeanOverProcs: 0.00157112
    MaxOverProcs: 0.00196719
    MeanOverCallCounts: 0.000157112
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00345016
    MeanOverProcs: 0.00434899
    MaxOverProcs: 0.00549865
    MeanOverCallCounts: 0.000434899
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.000688791
    MeanOverProcs: 0.000995159
    MaxOverProcs: 0.00129795
    MeanOverCallCounts: 0.000995159
  "MILO::driver::total run time": 
    MinOverProcs: 0.135021
    MeanOverProcs: 0.135414
    MaxOverProcs: 0.135563
    MeanOverCallCounts: 0.135414
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.241507
    MeanOverProcs: 0.242029
    MaxOverProcs: 0.242563
    MeanOverCallCounts: 0.242029
  "MILO::function::decompose": 
    MinOverProcs: 0.000742912
    MeanOverProcs: 0.000823259
    MaxOverProcs: 0.00101209
    MeanOverCallCounts: 0.000823259
  "MILO::function::evaluate": 
    MinOverProcs: 0.0031538
    MeanOverProcs: 0.00358015
    MaxOverProcs: 0.00464773
    MeanOverCallCounts: 2.03129e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 3.12328e-05
    MeanOverProcs: 4.41074e-05
    MaxOverProcs: 5.72205e-05
    MeanOverCallCounts: 3.52859e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000333071
    MeanOverProcs: 0.000803053
    MaxOverProcs: 0.00114512
    MeanOverCallCounts: 0.000803053
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.6982e-05
    MeanOverProcs: 0.000110984
    MaxOverProcs: 0.000180006
    MeanOverCallCounts: 0.000110984
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00658512
    MeanOverProcs: 0.00726151
    MaxOverProcs: 0.00814891
    MeanOverCallCounts: 0.00726151
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.0127912
    MeanOverProcs: 0.0136731
    MaxOverProcs: 0.014364
    MeanOverCallCounts: 0.0136731
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.061023
    MeanOverProcs: 0.0610375
    MaxOverProcs: 0.0610549
    MeanOverCallCounts: 0.0305187
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.036607
    MeanOverProcs: 0.0369688
    MaxOverProcs: 0.0371771
    MeanOverCallCounts: 0.0369688
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.3113e-05
    MeanOverProcs: 1.57952e-05
    MaxOverProcs: 2.09808e-05
    MeanOverCallCounts: 1.57952e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.029496
    MeanOverProcs: 0.029592
    MaxOverProcs: 0.029722
    MeanOverCallCounts: 0.029592
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.40803e-05
    MeanOverProcs: 2.8789e-05
    MaxOverProcs: 4.19617e-05
    MeanOverCallCounts: 2.8789e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00203991
    MeanOverProcs: 0.002042
    MaxOverProcs: 0.00204301
    MeanOverCallCounts: 0.002042
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00192642
    MeanOverProcs: 0.00224853
    MaxOverProcs: 0.0026319
    MeanOverCallCounts: 0.000112426
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000718594
    MeanOverProcs: 0.000782132
    MaxOverProcs: 0.000840902
    MeanOverCallCounts: 3.91066e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000516176
    MeanOverProcs: 0.000541389
    MaxOverProcs: 0.000592947
    MeanOverCallCounts: 7.73413e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00828171
    MeanOverProcs: 0.00930738
    MaxOverProcs: 0.0106401
    MeanOverCallCounts: 0.000132963
  "MILO::workset::reset*": 
    MinOverProcs: 9.03606e-05
    MeanOverProcs: 9.80496e-05
    MaxOverProcs: 0.000101328
    MeanOverCallCounts: 4.90248e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0409079
    MeanOverProcs: 0.0424783
    MaxOverProcs: 0.044935
    MeanOverCallCounts: 0.0141594
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.00114417
    MeanOverProcs: 0.00339502
    MaxOverProcs: 0.00460815
    MeanOverCallCounts: 0.00339502
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.00747395
    MeanOverProcs: 0.00748175
    MaxOverProcs: 0.00749803
    MeanOverCallCounts: 0.00748175
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000458956
    MeanOverProcs: 0.000608921
    MaxOverProcs: 0.000912905
    MeanOverCallCounts: 0.000202974
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00317001
    MeanOverProcs: 0.00363773
    MaxOverProcs: 0.00405502
    MeanOverCallCounts: 0.00363773
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 9.29832e-05
    MeanOverProcs: 0.000134468
    MaxOverProcs: 0.000181913
    MeanOverCallCounts: 0.000134468
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000181198
    MeanOverProcs: 0.000247002
    MaxOverProcs: 0.000316858
    MeanOverCallCounts: 0.000247002
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000195026
    MeanOverProcs: 0.000269711
    MaxOverProcs: 0.000340939
    MeanOverCallCounts: 0.000269711
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000217915
    MeanOverProcs: 0.000309467
    MaxOverProcs: 0.000398874
    MeanOverCallCounts: 0.000309467
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.0013721
    MeanOverProcs: 0.00140154
    MaxOverProcs: 0.0014472
    MeanOverCallCounts: 0.00140154
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.00058794
    MeanOverProcs: 0.00062871
    MaxOverProcs: 0.00067091
    MeanOverCallCounts: 0.00062871
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 9.05991e-06
    MeanOverProcs: 1.22786e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 1.22786e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000506878
    MeanOverProcs: 0.000530005
    MaxOverProcs: 0.000544071
    MeanOverCallCounts: 0.000530005
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 3.09944e-06
    MeanOverProcs: 3.8147e-06
    MaxOverProcs: 5.00679e-06
    MeanOverCallCounts: 3.8147e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 7.51019e-05
    MeanOverProcs: 0.000121891
    MaxOverProcs: 0.000162125
    MeanOverCallCounts: 0.000121891
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.12057e-05
    MeanOverProcs: 1.35303e-05
    MaxOverProcs: 1.5974e-05
    MeanOverCallCounts: 1.35303e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.7895e-05
    MeanOverProcs: 4.66704e-05
    MaxOverProcs: 6.29425e-05
    MeanOverCallCounts: 4.66704e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.00049901
    MeanOverProcs: 0.000607967
    MaxOverProcs: 0.00069499
    MeanOverCallCounts: 0.000607967
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 4.48227e-05
    MeanOverProcs: 5.16772e-05
    MaxOverProcs: 5.79357e-05
    MeanOverCallCounts: 5.16772e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 5.10216e-05
    MeanOverProcs: 6.49691e-05
    MaxOverProcs: 7.79629e-05
    MeanOverCallCounts: 6.49691e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.00032711
    MeanOverProcs: 0.000459075
    MaxOverProcs: 0.000571012
    MeanOverCallCounts: 0.000459075
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000166893
    MeanOverProcs: 0.000184238
    MaxOverProcs: 0.000225067
    MeanOverCallCounts: 0.000184238
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.0596011
    MeanOverProcs: 0.0597085
    MaxOverProcs: 0.059746
    MeanOverCallCounts: 0.0597085
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
    MinOverProcs: 1762
    MeanOverProcs: 1762.5
    MaxOverProcs: 1763
    MeanOverCallCounts: 1762.5
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
