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
  - "MILO::driver::total run time"
  - "MILO::driver::total setup and execution time"
  - "MILO::function::decompose"
  - "MILO::function::evaluate"
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
    MinOverProcs: 0.00642705
    MeanOverProcs: 0.00649738
    MaxOverProcs: 0.0065403
    MeanOverCallCounts: 0.00216579
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000263929
    MeanOverProcs: 0.000271082
    MaxOverProcs: 0.000278234
    MeanOverCallCounts: 1.23219e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000172853
    MeanOverProcs: 0.000224233
    MaxOverProcs: 0.000278234
    MeanOverCallCounts: 6.22869e-06
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000161886
    MeanOverProcs: 0.000162482
    MaxOverProcs: 0.000163317
    MeanOverCallCounts: 7.38556e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.000732422
    MeanOverProcs: 0.000787616
    MaxOverProcs: 0.000843048
    MeanOverCallCounts: 5.62583e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000499964
    MeanOverProcs: 0.000663996
    MaxOverProcs: 0.000735521
    MeanOverCallCounts: 3.68887e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.0092833
    MeanOverProcs: 0.00939125
    MaxOverProcs: 0.0094893
    MeanOverCallCounts: 0.000670803
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00150251
    MeanOverProcs: 0.00156087
    MaxOverProcs: 0.00159764
    MeanOverCallCounts: 5.57452e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00200391
    MeanOverProcs: 0.00334585
    MaxOverProcs: 0.00418401
    MeanOverCallCounts: 0.00111528
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 3.79086e-05
    MeanOverProcs: 4.41074e-05
    MaxOverProcs: 4.79221e-05
    MeanOverCallCounts: 2.20537e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000154257
    MeanOverProcs: 0.000160336
    MaxOverProcs: 0.000169754
    MeanOverCallCounts: 8.01682e-06
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00590086
    MeanOverProcs: 0.00612575
    MaxOverProcs: 0.00640321
    MeanOverCallCounts: 0.000306287
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00614905
    MeanOverProcs: 0.00637579
    MaxOverProcs: 0.00666499
    MeanOverCallCounts: 0.00318789
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.07288e-05
    MeanOverProcs: 1.29342e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 6.4671e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000138044
    MeanOverProcs: 0.000534832
    MaxOverProcs: 0.000676155
    MeanOverCallCounts: 0.000534832
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000722885
    MeanOverProcs: 0.000728667
    MaxOverProcs: 0.000738859
    MeanOverCallCounts: 0.000728667
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 8.2016e-05
    MeanOverProcs: 0.000181079
    MaxOverProcs: 0.000280142
    MeanOverCallCounts: 7.24316e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 7.12872e-05
    MeanOverProcs: 7.87377e-05
    MaxOverProcs: 8.36849e-05
    MeanOverCallCounts: 3.93689e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.8147e-05
    MeanOverProcs: 4.04716e-05
    MaxOverProcs: 4.24385e-05
    MeanOverCallCounts: 2.02358e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00248623
    MeanOverProcs: 0.0025723
    MaxOverProcs: 0.00261164
    MeanOverCallCounts: 0.000128615
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 9.67979e-05
    MeanOverProcs: 0.000106871
    MaxOverProcs: 0.000115156
    MeanOverCallCounts: 3.56237e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00293851
    MeanOverProcs: 0.00305218
    MaxOverProcs: 0.0033803
    MeanOverCallCounts: 7.63044e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.0045073
    MeanOverProcs: 0.00465381
    MaxOverProcs: 0.00483751
    MeanOverCallCounts: 0.000155127
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00130224
    MeanOverProcs: 0.00140965
    MaxOverProcs: 0.00159979
    MeanOverCallCounts: 0.000140965
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00317097
    MeanOverProcs: 0.00373852
    MaxOverProcs: 0.00441813
    MeanOverCallCounts: 0.000373852
  "MILO::driver::total run time": 
    MinOverProcs: 0.102852
    MeanOverProcs: 0.103204
    MaxOverProcs: 0.103332
    MeanOverCallCounts: 0.103204
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.179554
    MeanOverProcs: 0.179677
    MaxOverProcs: 0.179764
    MeanOverCallCounts: 0.179677
  "MILO::function::decompose": 
    MinOverProcs: 0.000699043
    MeanOverProcs: 0.000772297
    MaxOverProcs: 0.000959158
    MeanOverCallCounts: 0.000772297
  "MILO::function::evaluate": 
    MinOverProcs: 0.00127292
    MeanOverProcs: 0.00130886
    MaxOverProcs: 0.00134683
    MeanOverCallCounts: 8.05451e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000193119
    MeanOverProcs: 0.000203967
    MaxOverProcs: 0.000219822
    MeanOverCallCounts: 0.000203967
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 4.88758e-05
    MeanOverProcs: 6.53863e-05
    MaxOverProcs: 8.2016e-05
    MeanOverCallCounts: 6.53863e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00495291
    MeanOverProcs: 0.00514179
    MaxOverProcs: 0.00565004
    MeanOverCallCounts: 0.00514179
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.012948
    MeanOverProcs: 0.0134844
    MaxOverProcs: 0.0136778
    MeanOverCallCounts: 0.0134844
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.043911
    MeanOverProcs: 0.0439202
    MaxOverProcs: 0.043941
    MeanOverCallCounts: 0.0219601
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0286899
    MeanOverProcs: 0.0290414
    MaxOverProcs: 0.0291748
    MeanOverCallCounts: 0.0290414
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.19209e-05
    MeanOverProcs: 1.44839e-05
    MaxOverProcs: 2.00272e-05
    MeanOverCallCounts: 1.44839e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.024503
    MeanOverProcs: 0.0245233
    MaxOverProcs: 0.02455
    MeanOverCallCounts: 0.0245233
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 1.69277e-05
    MeanOverProcs: 1.92523e-05
    MaxOverProcs: 2.38419e-05
    MeanOverCallCounts: 1.92523e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.002033
    MeanOverProcs: 0.00203371
    MaxOverProcs: 0.00203395
    MeanOverCallCounts: 0.00203371
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00180221
    MeanOverProcs: 0.00187564
    MaxOverProcs: 0.00191975
    MeanOverCallCounts: 9.37819e-05
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000655651
    MeanOverProcs: 0.000669062
    MaxOverProcs: 0.000695229
    MeanOverCallCounts: 3.34531e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000379801
    MeanOverProcs: 0.000431418
    MaxOverProcs: 0.000460148
    MeanOverCallCounts: 1.43806e-05
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00393081
    MeanOverProcs: 0.00403214
    MaxOverProcs: 0.00418639
    MeanOverCallCounts: 0.000134405
  "MILO::workset::reset*": 
    MinOverProcs: 0.000103712
    MeanOverProcs: 0.000105023
    MaxOverProcs: 0.000105619
    MeanOverCallCounts: 5.25117e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.03233
    MeanOverProcs: 0.0330493
    MaxOverProcs: 0.0344627
    MeanOverCallCounts: 0.0110164
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.00110912
    MeanOverProcs: 0.00167829
    MaxOverProcs: 0.00189114
    MeanOverCallCounts: 0.00167829
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.011641
    MeanOverProcs: 0.0116574
    MaxOverProcs: 0.0116909
    MeanOverCallCounts: 0.0116574
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000431061
    MeanOverProcs: 0.000470817
    MaxOverProcs: 0.0005548
    MeanOverCallCounts: 0.000156939
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00218391
    MeanOverProcs: 0.00220346
    MaxOverProcs: 0.00221992
    MeanOverCallCounts: 0.00220346
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 7.98702e-05
    MeanOverProcs: 8.31485e-05
    MaxOverProcs: 8.4877e-05
    MeanOverCallCounts: 8.31485e-05
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000148773
    MeanOverProcs: 0.000151634
    MaxOverProcs: 0.000156879
    MeanOverCallCounts: 0.000151634
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000176907
    MeanOverProcs: 0.00017935
    MaxOverProcs: 0.000183821
    MeanOverCallCounts: 0.00017935
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000179052
    MeanOverProcs: 0.00018996
    MaxOverProcs: 0.000206947
    MeanOverCallCounts: 0.00018996
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.000883818
    MeanOverProcs: 0.000905216
    MaxOverProcs: 0.000926018
    MeanOverCallCounts: 0.000905216
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000355005
    MeanOverProcs: 0.00035876
    MaxOverProcs: 0.000360966
    MeanOverCallCounts: 0.00035876
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 8.10623e-06
    MeanOverProcs: 8.58307e-06
    MaxOverProcs: 9.05991e-06
    MeanOverCallCounts: 8.58307e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000344992
    MeanOverProcs: 0.000368237
    MaxOverProcs: 0.00037694
    MeanOverCallCounts: 0.000368237
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 1.90735e-06
    MeanOverProcs: 2.5034e-06
    MaxOverProcs: 3.09944e-06
    MeanOverCallCounts: 2.5034e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 6.69956e-05
    MeanOverProcs: 9.23872e-05
    MaxOverProcs: 0.000131845
    MeanOverCallCounts: 9.23872e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.09673e-05
    MeanOverProcs: 1.12057e-05
    MaxOverProcs: 1.19209e-05
    MeanOverCallCounts: 1.12057e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.09808e-05
    MeanOverProcs: 2.85506e-05
    MaxOverProcs: 3.91006e-05
    MeanOverCallCounts: 2.85506e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000307083
    MeanOverProcs: 0.000326574
    MaxOverProcs: 0.000347137
    MeanOverCallCounts: 0.000326574
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 4.31538e-05
    MeanOverProcs: 4.43459e-05
    MaxOverProcs: 4.50611e-05
    MeanOverCallCounts: 4.43459e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.41074e-05
    MeanOverProcs: 4.43459e-05
    MaxOverProcs: 4.50611e-05
    MeanOverCallCounts: 4.43459e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000199795
    MeanOverProcs: 0.000217438
    MaxOverProcs: 0.000236988
    MeanOverCallCounts: 0.000217438
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000165939
    MeanOverProcs: 0.000179172
    MaxOverProcs: 0.000214815
    MeanOverCallCounts: 0.000179172
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.0395639
    MeanOverProcs: 0.0397024
    MaxOverProcs: 0.0397558
    MeanOverCallCounts: 0.0397024
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
    MaxOverProcs: 163
    MeanOverCallCounts: 162.5
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
