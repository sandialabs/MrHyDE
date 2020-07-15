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
    MinOverProcs: 0.00756693
    MeanOverProcs: 0.00760663
    MaxOverProcs: 0.00767708
    MeanOverCallCounts: 0.00253554
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000327826
    MeanOverProcs: 0.000367761
    MaxOverProcs: 0.000406504
    MeanOverCallCounts: 1.67164e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000223875
    MeanOverProcs: 0.000306726
    MaxOverProcs: 0.000455618
    MeanOverCallCounts: 8.52015e-06
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000172853
    MeanOverProcs: 0.000182748
    MaxOverProcs: 0.000188589
    MeanOverCallCounts: 8.30672e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.000889778
    MeanOverProcs: 0.000972211
    MaxOverProcs: 0.00108147
    MeanOverCallCounts: 6.94437e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.00053215
    MeanOverProcs: 0.000701308
    MaxOverProcs: 0.000856876
    MeanOverCallCounts: 3.89616e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.00976253
    MeanOverProcs: 0.00989133
    MaxOverProcs: 0.0100152
    MeanOverCallCounts: 0.000706524
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00170898
    MeanOverProcs: 0.00175375
    MaxOverProcs: 0.00181246
    MeanOverCallCounts: 6.26338e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00128102
    MeanOverProcs: 0.00158828
    MaxOverProcs: 0.00173807
    MeanOverCallCounts: 0.000529428
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 6.79493e-05
    MeanOverProcs: 7.05123e-05
    MaxOverProcs: 7.29561e-05
    MeanOverCallCounts: 3.52561e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.00017643
    MeanOverProcs: 0.00018841
    MaxOverProcs: 0.000210762
    MeanOverCallCounts: 9.42051e-06
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00606823
    MeanOverProcs: 0.00622457
    MaxOverProcs: 0.0063324
    MeanOverCallCounts: 0.000311229
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00635791
    MeanOverProcs: 0.00653005
    MaxOverProcs: 0.00665903
    MeanOverCallCounts: 0.00326502
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00586009
    MeanOverProcs: 0.00601178
    MaxOverProcs: 0.00619698
    MeanOverCallCounts: 0.00601178
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000172138
    MeanOverProcs: 0.000178874
    MaxOverProcs: 0.000182152
    MeanOverCallCounts: 0.000178874
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.21593e-05
    MeanOverProcs: 1.49608e-05
    MaxOverProcs: 1.81198e-05
    MeanOverCallCounts: 7.48038e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000138998
    MeanOverProcs: 0.000174761
    MaxOverProcs: 0.000231028
    MeanOverCallCounts: 0.000174761
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000268936
    MeanOverProcs: 0.000423491
    MaxOverProcs: 0.000487089
    MeanOverCallCounts: 0.000423491
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 9.29832e-05
    MeanOverProcs: 0.000142694
    MaxOverProcs: 0.000192165
    MeanOverCallCounts: 5.70774e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 6.48499e-05
    MeanOverProcs: 7.24196e-05
    MaxOverProcs: 8.03471e-05
    MeanOverCallCounts: 3.62098e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.48091e-05
    MeanOverProcs: 3.61204e-05
    MaxOverProcs: 3.83854e-05
    MeanOverCallCounts: 1.80602e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00266957
    MeanOverProcs: 0.0027985
    MaxOverProcs: 0.00288296
    MeanOverCallCounts: 0.000139925
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000104666
    MeanOverProcs: 0.000113904
    MaxOverProcs: 0.000119925
    MeanOverCallCounts: 3.79682e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00320244
    MeanOverProcs: 0.00325471
    MaxOverProcs: 0.00337243
    MeanOverCallCounts: 8.13678e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00459051
    MeanOverProcs: 0.00463641
    MaxOverProcs: 0.00468183
    MeanOverCallCounts: 0.000154547
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00117922
    MeanOverProcs: 0.00119138
    MaxOverProcs: 0.00121307
    MeanOverCallCounts: 0.000119138
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00312281
    MeanOverProcs: 0.00318807
    MaxOverProcs: 0.0032413
    MeanOverCallCounts: 0.000318807
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.000659943
    MeanOverProcs: 0.000670493
    MaxOverProcs: 0.000694036
    MeanOverCallCounts: 0.000670493
  "MILO::driver::total run time": 
    MinOverProcs: 0.093498
    MeanOverProcs: 0.0935355
    MaxOverProcs: 0.093591
    MeanOverCallCounts: 0.0935355
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.160777
    MeanOverProcs: 0.1611
    MaxOverProcs: 0.16139
    MeanOverCallCounts: 0.1611
  "MILO::function::decompose": 
    MinOverProcs: 0.000731945
    MeanOverProcs: 0.000751257
    MaxOverProcs: 0.000770092
    MeanOverCallCounts: 0.000751257
  "MILO::function::evaluate": 
    MinOverProcs: 0.00141454
    MeanOverProcs: 0.00145996
    MaxOverProcs: 0.00152707
    MeanOverCallCounts: 8.98435e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 2.69413e-05
    MeanOverProcs: 2.98619e-05
    MaxOverProcs: 3.24249e-05
    MeanOverCallCounts: 2.38895e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.00020504
    MeanOverProcs: 0.000308514
    MaxOverProcs: 0.000380993
    MeanOverCallCounts: 0.000308514
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.4121e-05
    MeanOverProcs: 7.12872e-05
    MaxOverProcs: 8.89301e-05
    MeanOverCallCounts: 7.12872e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00535583
    MeanOverProcs: 0.00548393
    MaxOverProcs: 0.00562215
    MeanOverCallCounts: 0.00548393
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.00690007
    MeanOverProcs: 0.00702858
    MaxOverProcs: 0.00715113
    MeanOverCallCounts: 0.00702858
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.0397041
    MeanOverProcs: 0.0397046
    MaxOverProcs: 0.039706
    MeanOverCallCounts: 0.0198523
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0302849
    MeanOverProcs: 0.0303792
    MaxOverProcs: 0.0304999
    MeanOverCallCounts: 0.0303792
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.09673e-05
    MeanOverProcs: 1.12653e-05
    MaxOverProcs: 1.21593e-05
    MeanOverCallCounts: 1.12653e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.0194809
    MeanOverProcs: 0.0196378
    MaxOverProcs: 0.019701
    MeanOverCallCounts: 0.0196378
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.40803e-05
    MeanOverProcs: 2.47955e-05
    MaxOverProcs: 2.59876e-05
    MeanOverCallCounts: 2.47955e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00159383
    MeanOverProcs: 0.00159997
    MaxOverProcs: 0.00160503
    MeanOverCallCounts: 0.00159997
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00188231
    MeanOverProcs: 0.00198126
    MaxOverProcs: 0.00205302
    MeanOverCallCounts: 9.90629e-05
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000756741
    MeanOverProcs: 0.000788629
    MaxOverProcs: 0.000820637
    MeanOverCallCounts: 3.94315e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.00026083
    MeanOverProcs: 0.000264943
    MaxOverProcs: 0.000267744
    MeanOverCallCounts: 8.83142e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00414205
    MeanOverProcs: 0.00417912
    MaxOverProcs: 0.00421286
    MeanOverCallCounts: 0.000139304
  "MILO::workset::reset*": 
    MinOverProcs: 0.00012207
    MeanOverProcs: 0.000128865
    MaxOverProcs: 0.000134706
    MeanOverCallCounts: 6.44326e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0334682
    MeanOverProcs: 0.0336799
    MaxOverProcs: 0.0340691
    MeanOverCallCounts: 0.0112266
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.000805855
    MeanOverProcs: 0.000941217
    MaxOverProcs: 0.00106692
    MeanOverCallCounts: 0.000941217
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.00593495
    MeanOverProcs: 0.00594944
    MaxOverProcs: 0.00595903
    MeanOverCallCounts: 0.00594944
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000446081
    MeanOverProcs: 0.00046587
    MaxOverProcs: 0.000503063
    MeanOverCallCounts: 0.00015529
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00243497
    MeanOverProcs: 0.00248218
    MaxOverProcs: 0.00258899
    MeanOverCallCounts: 0.00248218
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 8.29697e-05
    MeanOverProcs: 8.54731e-05
    MaxOverProcs: 8.79765e-05
    MeanOverCallCounts: 8.54731e-05
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000159979
    MeanOverProcs: 0.000164032
    MaxOverProcs: 0.000167131
    MeanOverCallCounts: 0.000164032
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000173807
    MeanOverProcs: 0.000174224
    MaxOverProcs: 0.000174999
    MeanOverCallCounts: 0.000174224
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000190973
    MeanOverProcs: 0.000203729
    MaxOverProcs: 0.000211
    MeanOverCallCounts: 0.000203729
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.000977039
    MeanOverProcs: 0.00100023
    MaxOverProcs: 0.00102901
    MeanOverCallCounts: 0.00100023
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000448942
    MeanOverProcs: 0.000451446
    MaxOverProcs: 0.000452995
    MeanOverCallCounts: 0.000451446
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 9.05991e-06
    MeanOverProcs: 9.29832e-06
    MaxOverProcs: 1.00136e-05
    MeanOverCallCounts: 9.29832e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000344992
    MeanOverProcs: 0.000365257
    MaxOverProcs: 0.000375032
    MeanOverCallCounts: 0.000365257
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 2.14577e-06
    MeanOverProcs: 2.74181e-06
    MaxOverProcs: 3.09944e-06
    MeanOverCallCounts: 2.74181e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 6.8903e-05
    MeanOverProcs: 9.44734e-05
    MaxOverProcs: 0.000132084
    MeanOverCallCounts: 9.44734e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 9.77516e-06
    MeanOverProcs: 1.07288e-05
    MaxOverProcs: 1.12057e-05
    MeanOverCallCounts: 1.07288e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.59876e-05
    MeanOverProcs: 3.39746e-05
    MaxOverProcs: 4.3869e-05
    MeanOverCallCounts: 3.39746e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000350952
    MeanOverProcs: 0.000385523
    MaxOverProcs: 0.00041914
    MeanOverCallCounts: 0.000385523
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 5.00679e-05
    MeanOverProcs: 5.05447e-05
    MaxOverProcs: 5.10216e-05
    MeanOverCallCounts: 5.05447e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.48227e-05
    MeanOverProcs: 4.6432e-05
    MaxOverProcs: 4.88758e-05
    MeanOverCallCounts: 4.6432e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000203133
    MeanOverProcs: 0.000256002
    MaxOverProcs: 0.000298977
    MeanOverCallCounts: 0.000256002
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000163078
    MeanOverProcs: 0.000163555
    MaxOverProcs: 0.000164032
    MeanOverCallCounts: 0.000163555
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.035939
    MeanOverProcs: 0.0359402
    MaxOverProcs: 0.035943
    MeanOverCallCounts: 0.0359402
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
