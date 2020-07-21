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
    MinOverProcs: 0.00741506
    MeanOverProcs: 0.00747967
    MaxOverProcs: 0.00751424
    MeanOverCallCounts: 0.00249322
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000311852
    MeanOverProcs: 0.000318289
    MaxOverProcs: 0.000325918
    MeanOverCallCounts: 1.44677e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000239134
    MeanOverProcs: 0.000324309
    MaxOverProcs: 0.000398397
    MeanOverCallCounts: 9.00858e-06
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000174761
    MeanOverProcs: 0.000181019
    MaxOverProcs: 0.000188112
    MeanOverCallCounts: 8.22815e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.00083828
    MeanOverProcs: 0.000948548
    MaxOverProcs: 0.00100088
    MeanOverCallCounts: 6.77535e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000486851
    MeanOverProcs: 0.000523984
    MaxOverProcs: 0.00057435
    MeanOverCallCounts: 2.91102e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.00951815
    MeanOverProcs: 0.00953841
    MaxOverProcs: 0.00957513
    MeanOverCallCounts: 0.000681315
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.0017159
    MeanOverProcs: 0.00178415
    MaxOverProcs: 0.00182891
    MeanOverCallCounts: 6.37195e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00176477
    MeanOverProcs: 0.00413692
    MaxOverProcs: 0.00604773
    MeanOverCallCounts: 0.00137897
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 7.39098e-05
    MeanOverProcs: 7.4029e-05
    MaxOverProcs: 7.41482e-05
    MeanOverCallCounts: 3.70145e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000208855
    MeanOverProcs: 0.000224769
    MaxOverProcs: 0.000238657
    MeanOverCallCounts: 1.12385e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00632572
    MeanOverProcs: 0.00673229
    MaxOverProcs: 0.00693893
    MeanOverCallCounts: 0.000336614
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00666308
    MeanOverProcs: 0.00708276
    MaxOverProcs: 0.00729299
    MeanOverCallCounts: 0.00354138
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00668406
    MeanOverProcs: 0.00736082
    MaxOverProcs: 0.0086441
    MeanOverCallCounts: 0.00736082
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000234127
    MeanOverProcs: 0.000243068
    MaxOverProcs: 0.000247002
    MeanOverCallCounts: 0.000243068
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.28746e-05
    MeanOverProcs: 1.49608e-05
    MaxOverProcs: 1.69277e-05
    MeanOverCallCounts: 7.48038e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000158072
    MeanOverProcs: 0.000214458
    MaxOverProcs: 0.000268936
    MeanOverCallCounts: 0.000214458
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000437021
    MeanOverProcs: 0.000586212
    MaxOverProcs: 0.000724077
    MeanOverCallCounts: 0.000586212
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 8.67844e-05
    MeanOverProcs: 0.000150144
    MaxOverProcs: 0.000198603
    MeanOverCallCounts: 6.00576e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 7.4625e-05
    MeanOverProcs: 8.2612e-05
    MaxOverProcs: 8.63075e-05
    MeanOverCallCounts: 4.1306e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.26633e-05
    MeanOverProcs: 3.68357e-05
    MaxOverProcs: 4.26769e-05
    MeanOverCallCounts: 1.84178e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00278044
    MeanOverProcs: 0.00296444
    MaxOverProcs: 0.00305557
    MeanOverCallCounts: 0.000148222
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000100374
    MeanOverProcs: 0.000107169
    MaxOverProcs: 0.000112057
    MeanOverCallCounts: 3.57231e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00314379
    MeanOverProcs: 0.00315177
    MaxOverProcs: 0.00316167
    MeanOverCallCounts: 7.87944e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00456119
    MeanOverProcs: 0.00476408
    MaxOverProcs: 0.00488281
    MeanOverCallCounts: 0.000158803
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00131607
    MeanOverProcs: 0.00143838
    MaxOverProcs: 0.00163198
    MeanOverCallCounts: 0.000143838
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00365543
    MeanOverProcs: 0.00394458
    MaxOverProcs: 0.00451756
    MeanOverCallCounts: 0.000394458
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.000694036
    MeanOverProcs: 0.000828266
    MaxOverProcs: 0.00119901
    MeanOverCallCounts: 0.000828266
  "MILO::driver::total run time": 
    MinOverProcs: 0.111524
    MeanOverProcs: 0.111576
    MaxOverProcs: 0.111629
    MeanOverCallCounts: 0.111576
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.204226
    MeanOverProcs: 0.204458
    MaxOverProcs: 0.204593
    MeanOverCallCounts: 0.204458
  "MILO::function::decompose": 
    MinOverProcs: 0.000715971
    MeanOverProcs: 0.000766039
    MaxOverProcs: 0.0008111
    MeanOverCallCounts: 0.000766039
  "MILO::function::evaluate": 
    MinOverProcs: 0.00147724
    MeanOverProcs: 0.00151712
    MaxOverProcs: 0.00155473
    MeanOverCallCounts: 9.3361e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 3.43323e-05
    MeanOverProcs: 3.91006e-05
    MaxOverProcs: 4.52995e-05
    MeanOverCallCounts: 3.12805e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000319958
    MeanOverProcs: 0.00089699
    MaxOverProcs: 0.00114703
    MeanOverCallCounts: 0.00089699
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.91278e-05
    MeanOverProcs: 8.27909e-05
    MaxOverProcs: 9.29832e-05
    MeanOverCallCounts: 8.27909e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00565004
    MeanOverProcs: 0.00566709
    MaxOverProcs: 0.00570202
    MeanOverCallCounts: 0.00566709
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.0137761
    MeanOverProcs: 0.0138055
    MaxOverProcs: 0.013824
    MeanOverCallCounts: 0.0138055
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.0472121
    MeanOverProcs: 0.0472156
    MaxOverProcs: 0.0472171
    MeanOverCallCounts: 0.0236078
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0326061
    MeanOverProcs: 0.032792
    MaxOverProcs: 0.032974
    MeanOverCallCounts: 0.032792
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.19209e-05
    MeanOverProcs: 1.24574e-05
    MaxOverProcs: 1.3113e-05
    MeanOverCallCounts: 1.24574e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.022037
    MeanOverProcs: 0.0221632
    MaxOverProcs: 0.0222871
    MeanOverCallCounts: 0.0221632
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.81334e-05
    MeanOverProcs: 2.90871e-05
    MaxOverProcs: 3.09944e-05
    MeanOverCallCounts: 2.90871e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00166297
    MeanOverProcs: 0.00166756
    MaxOverProcs: 0.00167203
    MeanOverCallCounts: 0.00166756
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00200963
    MeanOverProcs: 0.00211459
    MaxOverProcs: 0.00217581
    MeanOverCallCounts: 0.00010573
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000740528
    MeanOverProcs: 0.000817597
    MaxOverProcs: 0.000858784
    MeanOverCallCounts: 4.08798e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000457525
    MeanOverProcs: 0.000461459
    MaxOverProcs: 0.000467777
    MeanOverCallCounts: 6.59227e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.0074985
    MeanOverProcs: 0.00769097
    MaxOverProcs: 0.00780559
    MeanOverCallCounts: 0.000109871
  "MILO::workset::reset*": 
    MinOverProcs: 0.000116348
    MeanOverProcs: 0.000120282
    MaxOverProcs: 0.00012517
    MeanOverCallCounts: 6.01411e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0364132
    MeanOverProcs: 0.0381085
    MaxOverProcs: 0.040283
    MeanOverCallCounts: 0.0127028
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.000802994
    MeanOverProcs: 0.000834525
    MaxOverProcs: 0.000849009
    MeanOverCallCounts: 0.000834525
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.0128231
    MeanOverProcs: 0.012835
    MaxOverProcs: 0.0128431
    MeanOverCallCounts: 0.012835
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000487089
    MeanOverProcs: 0.000549674
    MaxOverProcs: 0.000603914
    MeanOverCallCounts: 0.000183225
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.002846
    MeanOverProcs: 0.00317043
    MaxOverProcs: 0.00381684
    MeanOverCallCounts: 0.00317043
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 8.51154e-05
    MeanOverProcs: 0.000106275
    MaxOverProcs: 0.000163078
    MeanOverCallCounts: 0.000106275
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000150919
    MeanOverProcs: 0.000188231
    MaxOverProcs: 0.00027585
    MeanOverCallCounts: 0.000188231
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000178814
    MeanOverProcs: 0.000210285
    MaxOverProcs: 0.000299215
    MeanOverCallCounts: 0.000210285
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000200033
    MeanOverProcs: 0.000241458
    MaxOverProcs: 0.000352859
    MeanOverCallCounts: 0.000241458
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.00125599
    MeanOverProcs: 0.001302
    MaxOverProcs: 0.00134611
    MeanOverCallCounts: 0.001302
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000531912
    MeanOverProcs: 0.000619411
    MaxOverProcs: 0.000653028
    MeanOverCallCounts: 0.000619411
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 9.05991e-06
    MeanOverProcs: 1.07288e-05
    MaxOverProcs: 1.38283e-05
    MeanOverCallCounts: 1.07288e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000413895
    MeanOverProcs: 0.000453472
    MaxOverProcs: 0.000476837
    MeanOverCallCounts: 0.000453472
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 2.86102e-06
    MeanOverProcs: 3.51667e-06
    MaxOverProcs: 5.00679e-06
    MeanOverCallCounts: 3.51667e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 6.60419e-05
    MeanOverProcs: 0.000130057
    MaxOverProcs: 0.000200033
    MeanOverCallCounts: 0.000130057
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.00136e-05
    MeanOverProcs: 1.18017e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 1.18017e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.40803e-05
    MeanOverProcs: 3.89218e-05
    MaxOverProcs: 5.48363e-05
    MeanOverCallCounts: 3.89218e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000559092
    MeanOverProcs: 0.000656068
    MaxOverProcs: 0.000710964
    MeanOverCallCounts: 0.000656068
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 4.69685e-05
    MeanOverProcs: 5.32866e-05
    MaxOverProcs: 6.79493e-05
    MeanOverCallCounts: 5.32866e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.81606e-05
    MeanOverProcs: 6.13332e-05
    MaxOverProcs: 9.89437e-05
    MeanOverCallCounts: 6.13332e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000353813
    MeanOverProcs: 0.000513196
    MaxOverProcs: 0.000590086
    MeanOverCallCounts: 0.000513196
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000167847
    MeanOverProcs: 0.000220239
    MaxOverProcs: 0.000251055
    MeanOverCallCounts: 0.000220239
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.0474401
    MeanOverProcs: 0.047499
    MaxOverProcs: 0.047637
    MeanOverCallCounts: 0.047499
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
