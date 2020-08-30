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
  - "MILO::multiscale::reset()"
  - "MILO::physics::getSideInfo()"
  - "MILO::physics::setBCData()"
  - "MILO::physics::setDirichletData()"
  - "MILO::postprocess::computeError"
  - "MILO::postprocess::writeSolution"
  - "MILO::solver::linearSolver()"
  - "MILO::solver::nonlinearSolver()"
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
    MinOverProcs: 0.00994062
    MeanOverProcs: 0.00999165
    MaxOverProcs: 0.0100257
    MeanOverCallCounts: 0.00333055
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000415087
    MeanOverProcs: 0.000424743
    MaxOverProcs: 0.000436544
    MeanOverCallCounts: 1.93065e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000452995
    MeanOverProcs: 0.000598073
    MaxOverProcs: 0.00074482
    MeanOverCallCounts: 1.66131e-05
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000226021
    MeanOverProcs: 0.000258684
    MaxOverProcs: 0.000305891
    MeanOverCallCounts: 1.17584e-05
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.00132871
    MeanOverProcs: 0.00140768
    MaxOverProcs: 0.0015161
    MeanOverCallCounts: 0.000100549
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000696898
    MeanOverProcs: 0.00113159
    MaxOverProcs: 0.00151873
    MeanOverCallCounts: 6.28663e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.0116467
    MeanOverProcs: 0.0118858
    MaxOverProcs: 0.0120575
    MeanOverCallCounts: 0.000848983
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00212169
    MeanOverProcs: 0.00233173
    MaxOverProcs: 0.00244522
    MeanOverCallCounts: 8.32762e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00808811
    MeanOverProcs: 0.0169786
    MaxOverProcs: 0.0221131
    MeanOverCallCounts: 0.00565952
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 7.10487e-05
    MeanOverProcs: 8.39829e-05
    MaxOverProcs: 9.60827e-05
    MeanOverCallCounts: 4.19915e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000241756
    MeanOverProcs: 0.000263035
    MaxOverProcs: 0.000288963
    MeanOverCallCounts: 1.31518e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00369811
    MeanOverProcs: 0.00391996
    MaxOverProcs: 0.00438929
    MeanOverCallCounts: 0.000195998
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00407386
    MeanOverProcs: 0.00431597
    MaxOverProcs: 0.00482106
    MeanOverCallCounts: 0.00215799
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00613093
    MeanOverProcs: 0.00848216
    MaxOverProcs: 0.00994802
    MeanOverCallCounts: 0.00848216
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 7.08103e-05
    MeanOverProcs: 9.5129e-05
    MaxOverProcs: 0.000125885
    MeanOverCallCounts: 9.5129e-05
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.09673e-05
    MeanOverProcs: 1.29938e-05
    MaxOverProcs: 1.52588e-05
    MeanOverCallCounts: 6.49691e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000175953
    MeanOverProcs: 0.000405967
    MaxOverProcs: 0.000509024
    MeanOverCallCounts: 0.000405967
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000409126
    MeanOverProcs: 0.000468493
    MaxOverProcs: 0.000522852
    MeanOverCallCounts: 0.000468493
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 8.89301e-05
    MeanOverProcs: 0.000187755
    MaxOverProcs: 0.000306845
    MeanOverCallCounts: 7.51019e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 8.74996e-05
    MeanOverProcs: 9.53674e-05
    MaxOverProcs: 0.000110149
    MeanOverCallCounts: 4.76837e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 4.19617e-05
    MeanOverProcs: 4.74453e-05
    MaxOverProcs: 5.4121e-05
    MeanOverCallCounts: 2.37226e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00162697
    MeanOverProcs: 0.00172967
    MaxOverProcs: 0.00192571
    MeanOverCallCounts: 8.64834e-05
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 9.53674e-05
    MeanOverProcs: 0.000106573
    MaxOverProcs: 0.000118971
    MeanOverCallCounts: 3.55244e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00123715
    MeanOverProcs: 0.00161314
    MaxOverProcs: 0.00215697
    MeanOverCallCounts: 4.03285e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00244594
    MeanOverProcs: 0.0025658
    MaxOverProcs: 0.00272918
    MeanOverCallCounts: 8.55267e-05
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00124073
    MeanOverProcs: 0.00162548
    MaxOverProcs: 0.0018661
    MeanOverCallCounts: 0.000162548
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00330687
    MeanOverProcs: 0.00442344
    MaxOverProcs: 0.00516224
    MeanOverCallCounts: 0.000442344
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.00057292
    MeanOverProcs: 0.000953436
    MaxOverProcs: 0.00112104
    MeanOverCallCounts: 0.000953436
  "MILO::driver::total run time": 
    MinOverProcs: 0.134428
    MeanOverProcs: 0.134629
    MaxOverProcs: 0.134723
    MeanOverCallCounts: 0.134629
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.237862
    MeanOverProcs: 0.238244
    MaxOverProcs: 0.23881
    MeanOverCallCounts: 0.238244
  "MILO::function::decompose": 
    MinOverProcs: 0.000385046
    MeanOverProcs: 0.000414252
    MaxOverProcs: 0.000483036
    MeanOverCallCounts: 0.000414252
  "MILO::function::evaluate": 
    MinOverProcs: 0.00231314
    MeanOverProcs: 0.00254869
    MaxOverProcs: 0.00308847
    MeanOverCallCounts: 1.44607e-06
  "MILO::multiscale::reset()": 
    MinOverProcs: 1.90735e-06
    MeanOverProcs: 2.563e-06
    MaxOverProcs: 3.09944e-06
    MeanOverCallCounts: 1.2815e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 2.88486e-05
    MeanOverProcs: 4.36902e-05
    MaxOverProcs: 5.17368e-05
    MeanOverCallCounts: 3.49522e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000288963
    MeanOverProcs: 0.000695288
    MaxOverProcs: 0.00115108
    MeanOverCallCounts: 0.000695288
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.00679e-05
    MeanOverProcs: 9.04799e-05
    MaxOverProcs: 0.000141859
    MeanOverCallCounts: 9.04799e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00253177
    MeanOverProcs: 0.00325018
    MaxOverProcs: 0.00426507
    MeanOverCallCounts: 0.00325018
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.0137901
    MeanOverProcs: 0.0147888
    MaxOverProcs: 0.015511
    MeanOverCallCounts: 0.0147888
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.0614262
    MeanOverProcs: 0.0614353
    MaxOverProcs: 0.061456
    MeanOverCallCounts: 0.0307177
  "MILO::solver::nonlinearSolver()": 
    MinOverProcs: 0.0435829
    MeanOverProcs: 0.0436169
    MaxOverProcs: 0.0436339
    MeanOverCallCounts: 0.0436169
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0422032
    MeanOverProcs: 0.0423855
    MaxOverProcs: 0.0424778
    MeanOverCallCounts: 0.0423855
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.90735e-06
    MeanOverProcs: 2.92063e-06
    MaxOverProcs: 3.8147e-06
    MeanOverCallCounts: 2.92063e-06
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.0304658
    MeanOverProcs: 0.0304962
    MaxOverProcs: 0.0305159
    MeanOverCallCounts: 0.0304962
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.19345e-05
    MeanOverProcs: 3.22461e-05
    MaxOverProcs: 4.29153e-05
    MeanOverCallCounts: 3.22461e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00146294
    MeanOverProcs: 0.00146532
    MaxOverProcs: 0.00146818
    MeanOverCallCounts: 0.00146532
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00108433
    MeanOverProcs: 0.00117844
    MaxOverProcs: 0.00134468
    MeanOverCallCounts: 5.89222e-05
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000494719
    MeanOverProcs: 0.000517845
    MaxOverProcs: 0.000545025
    MeanOverCallCounts: 2.58923e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000282288
    MeanOverProcs: 0.000311553
    MaxOverProcs: 0.000362873
    MeanOverCallCounts: 4.45076e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00363851
    MeanOverProcs: 0.00394124
    MaxOverProcs: 0.00461674
    MeanOverCallCounts: 5.63034e-05
  "MILO::workset::reset*": 
    MinOverProcs: 4.8399e-05
    MeanOverProcs: 5.17368e-05
    MaxOverProcs: 5.45979e-05
    MeanOverCallCounts: 2.58684e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0368977
    MeanOverProcs: 0.0413954
    MaxOverProcs: 0.049392
    MeanOverCallCounts: 0.0137985
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.000800848
    MeanOverProcs: 0.0024327
    MaxOverProcs: 0.00337386
    MeanOverCallCounts: 0.0024327
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.0101049
    MeanOverProcs: 0.0101169
    MaxOverProcs: 0.0101418
    MeanOverCallCounts: 0.0101169
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000444174
    MeanOverProcs: 0.000632107
    MaxOverProcs: 0.00078702
    MeanOverCallCounts: 0.000210702
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00268197
    MeanOverProcs: 0.00317222
    MaxOverProcs: 0.003649
    MeanOverCallCounts: 0.00317222
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 7.89165e-05
    MeanOverProcs: 0.000115752
    MaxOverProcs: 0.000154972
    MeanOverCallCounts: 0.000115752
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000144958
    MeanOverProcs: 0.000217497
    MaxOverProcs: 0.000290155
    MeanOverCallCounts: 0.000217497
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000161886
    MeanOverProcs: 0.000238776
    MaxOverProcs: 0.000317097
    MeanOverCallCounts: 0.000238776
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000180006
    MeanOverProcs: 0.000261784
    MaxOverProcs: 0.000334978
    MeanOverCallCounts: 0.000261784
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.00118709
    MeanOverProcs: 0.00127679
    MaxOverProcs: 0.001333
    MeanOverCallCounts: 0.00127679
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000500917
    MeanOverProcs: 0.000560939
    MaxOverProcs: 0.000585794
    MeanOverCallCounts: 0.000560939
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 8.82149e-06
    MeanOverProcs: 1.24574e-05
    MaxOverProcs: 1.40667e-05
    MeanOverCallCounts: 1.24574e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000439882
    MeanOverProcs: 0.000462174
    MaxOverProcs: 0.000478029
    MeanOverCallCounts: 0.000462174
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 3.09944e-06
    MeanOverProcs: 3.63588e-06
    MaxOverProcs: 3.8147e-06
    MeanOverCallCounts: 3.63588e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 8.89301e-05
    MeanOverProcs: 0.000126004
    MaxOverProcs: 0.000159025
    MeanOverCallCounts: 0.000126004
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.00136e-05
    MeanOverProcs: 1.34706e-05
    MaxOverProcs: 1.50204e-05
    MeanOverCallCounts: 1.34706e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 3.69549e-05
    MeanOverProcs: 4.8995e-05
    MaxOverProcs: 5.60284e-05
    MeanOverCallCounts: 4.8995e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000437021
    MeanOverProcs: 0.000510693
    MaxOverProcs: 0.000550032
    MeanOverCallCounts: 0.000510693
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 4.19617e-05
    MeanOverProcs: 4.54783e-05
    MaxOverProcs: 5.4121e-05
    MeanOverCallCounts: 4.54783e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.31538e-05
    MeanOverProcs: 4.97699e-05
    MaxOverProcs: 6.81877e-05
    MeanOverCallCounts: 4.97699e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000283957
    MeanOverProcs: 0.000391483
    MaxOverProcs: 0.000442028
    MeanOverCallCounts: 0.000391483
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000165939
    MeanOverProcs: 0.000205815
    MaxOverProcs: 0.000252008
    MeanOverCallCounts: 0.000205815
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.0533142
    MeanOverProcs: 0.0534259
    MaxOverProcs: 0.0535202
    MeanOverCallCounts: 0.0534259
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
    MinOverProcs: 1763
    MeanOverProcs: 1762.5
    MaxOverProcs: 1762
    MeanOverCallCounts: 1762.5
  "MILO::multiscale::reset()": 
    MinOverProcs: 2
    MeanOverProcs: 2
    MaxOverProcs: 2
    MeanOverCallCounts: 2
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
  "MILO::solver::nonlinearSolver()": 
    MinOverProcs: 1
    MeanOverProcs: 1
    MaxOverProcs: 1
    MeanOverCallCounts: 1
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
