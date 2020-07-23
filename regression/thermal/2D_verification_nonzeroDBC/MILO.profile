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
    MinOverProcs: 0.00979495
    MeanOverProcs: 0.00986773
    MaxOverProcs: 0.00991893
    MeanOverCallCounts: 0.00328924
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.00041461
    MeanOverProcs: 0.000424206
    MaxOverProcs: 0.000431061
    MeanOverCallCounts: 1.92821e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000563383
    MeanOverProcs: 0.000613332
    MaxOverProcs: 0.000720024
    MeanOverCallCounts: 1.7037e-05
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000210524
    MeanOverProcs: 0.000239909
    MaxOverProcs: 0.000267029
    MeanOverCallCounts: 1.09049e-05
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.00132132
    MeanOverProcs: 0.00141358
    MaxOverProcs: 0.00151491
    MeanOverCallCounts: 0.00010097
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000927925
    MeanOverProcs: 0.00109273
    MaxOverProcs: 0.00125074
    MeanOverCallCounts: 6.07073e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.0128055
    MeanOverProcs: 0.0128209
    MaxOverProcs: 0.0128379
    MeanOverCallCounts: 0.000915779
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00249028
    MeanOverProcs: 0.00251269
    MaxOverProcs: 0.00253868
    MeanOverCallCounts: 8.97391e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00563216
    MeanOverProcs: 0.00649571
    MaxOverProcs: 0.00811291
    MeanOverCallCounts: 0.00216524
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 6.91414e-05
    MeanOverProcs: 8.13603e-05
    MaxOverProcs: 9.29832e-05
    MeanOverCallCounts: 4.06802e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000225306
    MeanOverProcs: 0.00025177
    MaxOverProcs: 0.000313997
    MeanOverCallCounts: 1.25885e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00666714
    MeanOverProcs: 0.00730222
    MaxOverProcs: 0.00897074
    MeanOverCallCounts: 0.000365111
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00702405
    MeanOverProcs: 0.007689
    MaxOverProcs: 0.00943708
    MeanOverCallCounts: 0.0038445
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00593901
    MeanOverProcs: 0.00765371
    MaxOverProcs: 0.0094409
    MeanOverCallCounts: 0.00765371
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000175953
    MeanOverProcs: 0.000195026
    MaxOverProcs: 0.000230074
    MeanOverCallCounts: 0.000195026
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.3113e-05
    MeanOverProcs: 1.45435e-05
    MaxOverProcs: 1.69277e-05
    MeanOverCallCounts: 7.27177e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000154972
    MeanOverProcs: 0.000478387
    MaxOverProcs: 0.000604868
    MeanOverCallCounts: 0.000478387
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000486851
    MeanOverProcs: 0.000576794
    MaxOverProcs: 0.000758171
    MeanOverCallCounts: 0.000576794
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 7.79629e-05
    MeanOverProcs: 0.000156462
    MaxOverProcs: 0.000276804
    MeanOverCallCounts: 6.25849e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 7.53403e-05
    MeanOverProcs: 9.05991e-05
    MaxOverProcs: 0.000110149
    MeanOverCallCounts: 4.52995e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.57628e-05
    MeanOverProcs: 4.56572e-05
    MaxOverProcs: 5.79357e-05
    MeanOverCallCounts: 2.28286e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00282001
    MeanOverProcs: 0.00310183
    MaxOverProcs: 0.00378156
    MeanOverCallCounts: 0.000155091
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000120163
    MeanOverProcs: 0.000130117
    MaxOverProcs: 0.000157118
    MeanOverCallCounts: 4.33723e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00413394
    MeanOverProcs: 0.0043996
    MaxOverProcs: 0.00475883
    MeanOverCallCounts: 0.00010999
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00538063
    MeanOverProcs: 0.00582272
    MaxOverProcs: 0.00688291
    MeanOverCallCounts: 0.000194091
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00116515
    MeanOverProcs: 0.00144708
    MaxOverProcs: 0.00177574
    MeanOverCallCounts: 0.000144708
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00310874
    MeanOverProcs: 0.00404572
    MaxOverProcs: 0.00513554
    MeanOverCallCounts: 0.000404572
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.000665903
    MeanOverProcs: 0.000940979
    MaxOverProcs: 0.00120997
    MeanOverCallCounts: 0.000940979
  "MILO::driver::total run time": 
    MinOverProcs: 0.140401
    MeanOverProcs: 0.140703
    MaxOverProcs: 0.140824
    MeanOverCallCounts: 0.140703
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.241825
    MeanOverProcs: 0.242176
    MaxOverProcs: 0.242641
    MeanOverCallCounts: 0.242176
  "MILO::function::decompose": 
    MinOverProcs: 0.000720024
    MeanOverProcs: 0.000789523
    MaxOverProcs: 0.000938892
    MeanOverCallCounts: 0.000789523
  "MILO::function::evaluate": 
    MinOverProcs: 0.00163269
    MeanOverProcs: 0.00178409
    MaxOverProcs: 0.0019784
    MeanOverCallCounts: 1.0979e-05
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 2.90871e-05
    MeanOverProcs: 3.85642e-05
    MaxOverProcs: 4.98295e-05
    MeanOverCallCounts: 3.08514e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000288963
    MeanOverProcs: 0.000721753
    MaxOverProcs: 0.00115204
    MeanOverCallCounts: 0.000721753
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.50747e-05
    MeanOverProcs: 9.35197e-05
    MaxOverProcs: 0.000147104
    MeanOverCallCounts: 9.35197e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00751495
    MeanOverProcs: 0.00800544
    MaxOverProcs: 0.00864005
    MeanOverCallCounts: 0.00800544
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.014982
    MeanOverProcs: 0.0156038
    MaxOverProcs: 0.016067
    MeanOverCallCounts: 0.0156038
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.0637391
    MeanOverProcs: 0.0637686
    MaxOverProcs: 0.063796
    MeanOverCallCounts: 0.0318843
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0373139
    MeanOverProcs: 0.0375533
    MaxOverProcs: 0.037734
    MeanOverCallCounts: 0.0375533
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.21593e-05
    MeanOverProcs: 1.50204e-05
    MaxOverProcs: 2.09808e-05
    MeanOverCallCounts: 1.50204e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.03197
    MeanOverProcs: 0.0320507
    MaxOverProcs: 0.032191
    MeanOverCallCounts: 0.0320507
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.21729e-05
    MeanOverProcs: 2.65241e-05
    MaxOverProcs: 3.48091e-05
    MeanOverCallCounts: 2.65241e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00214601
    MeanOverProcs: 0.00215334
    MaxOverProcs: 0.00216007
    MeanOverCallCounts: 0.00215334
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00205088
    MeanOverProcs: 0.00229472
    MaxOverProcs: 0.00285935
    MeanOverCallCounts: 0.000114736
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000732422
    MeanOverProcs: 0.000779033
    MaxOverProcs: 0.000893831
    MeanOverCallCounts: 3.89516e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.00053525
    MeanOverProcs: 0.000572979
    MaxOverProcs: 0.000644445
    MeanOverCallCounts: 8.18542e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00926471
    MeanOverProcs: 0.00992763
    MaxOverProcs: 0.0113068
    MeanOverCallCounts: 0.000141823
  "MILO::workset::reset*": 
    MinOverProcs: 8.08239e-05
    MeanOverProcs: 9.10759e-05
    MaxOverProcs: 0.000104189
    MeanOverCallCounts: 4.55379e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0439627
    MeanOverProcs: 0.0457484
    MaxOverProcs: 0.0469649
    MeanOverCallCounts: 0.0152495
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.00087285
    MeanOverProcs: 0.00144553
    MaxOverProcs: 0.00191212
    MeanOverCallCounts: 0.00144553
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.0139251
    MeanOverProcs: 0.0139522
    MaxOverProcs: 0.0139849
    MeanOverCallCounts: 0.0139522
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000473976
    MeanOverProcs: 0.000618339
    MaxOverProcs: 0.000798941
    MeanOverCallCounts: 0.000206113
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.0028131
    MeanOverProcs: 0.00318104
    MaxOverProcs: 0.00355005
    MeanOverCallCounts: 0.00318104
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 8.79765e-05
    MeanOverProcs: 0.000119328
    MaxOverProcs: 0.000150919
    MeanOverCallCounts: 0.000119328
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000154972
    MeanOverProcs: 0.000226259
    MaxOverProcs: 0.000296116
    MeanOverCallCounts: 0.000226259
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.00018096
    MeanOverProcs: 0.000250578
    MaxOverProcs: 0.000323057
    MeanOverCallCounts: 0.000250578
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000196934
    MeanOverProcs: 0.00028199
    MaxOverProcs: 0.000386953
    MeanOverCallCounts: 0.00028199
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.00114608
    MeanOverProcs: 0.0012055
    MaxOverProcs: 0.0012641
    MeanOverCallCounts: 0.0012055
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000542164
    MeanOverProcs: 0.000583291
    MaxOverProcs: 0.000625849
    MeanOverCallCounts: 0.000583291
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 9.05991e-06
    MeanOverProcs: 1.20401e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 1.20401e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000337124
    MeanOverProcs: 0.000373781
    MaxOverProcs: 0.000396013
    MeanOverCallCounts: 0.000373781
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 1.90735e-06
    MeanOverProcs: 3.69549e-06
    MaxOverProcs: 5.00679e-06
    MeanOverCallCounts: 3.69549e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 8.58307e-05
    MeanOverProcs: 0.00014025
    MaxOverProcs: 0.000206947
    MeanOverCallCounts: 0.00014025
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.00136e-05
    MeanOverProcs: 1.2517e-05
    MaxOverProcs: 1.5974e-05
    MeanOverCallCounts: 1.2517e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 3.48091e-05
    MeanOverProcs: 3.79086e-05
    MaxOverProcs: 4.19617e-05
    MeanOverCallCounts: 3.79086e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000484943
    MeanOverProcs: 0.000539541
    MaxOverProcs: 0.000597
    MeanOverCallCounts: 0.000539541
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 4.98295e-05
    MeanOverProcs: 5.61476e-05
    MaxOverProcs: 6.29425e-05
    MeanOverCallCounts: 5.61476e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.69685e-05
    MeanOverProcs: 6.17504e-05
    MaxOverProcs: 7.60555e-05
    MeanOverCallCounts: 6.17504e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000310898
    MeanOverProcs: 0.000390708
    MaxOverProcs: 0.000473976
    MeanOverCallCounts: 0.000390708
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000174046
    MeanOverProcs: 0.000221729
    MaxOverProcs: 0.000240088
    MeanOverCallCounts: 0.000221729
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.0573869
    MeanOverProcs: 0.0574477
    MaxOverProcs: 0.057611
    MeanOverCallCounts: 0.0574477
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
