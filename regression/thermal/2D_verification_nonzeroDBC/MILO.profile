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
    MinOverProcs: 0.0434089
    MeanOverProcs: 0.0438089
    MaxOverProcs: 0.0439723
    MeanOverCallCounts: 0.014603
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.00198174
    MeanOverProcs: 0.00215822
    MaxOverProcs: 0.00228143
    MeanOverCallCounts: 9.81011e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.00450397
    MeanOverProcs: 0.00500339
    MaxOverProcs: 0.00537729
    MeanOverCallCounts: 0.000138983
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000492096
    MeanOverProcs: 0.000747979
    MaxOverProcs: 0.000865221
    MeanOverCallCounts: 3.3999e-05
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.00796676
    MeanOverProcs: 0.00825542
    MaxOverProcs: 0.00865412
    MeanOverCallCounts: 0.000589673
  "Belos: Operation Op*x": 
    MinOverProcs: 0.00178957
    MeanOverProcs: 0.00338084
    MaxOverProcs: 0.00589728
    MeanOverCallCounts: 0.000187824
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.0466275
    MeanOverProcs: 0.0475622
    MaxOverProcs: 0.0485771
    MeanOverCallCounts: 0.0033973
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00932145
    MeanOverProcs: 0.0104428
    MaxOverProcs: 0.0125499
    MeanOverCallCounts: 0.000372957
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.0236659
    MeanOverProcs: 0.0446219
    MaxOverProcs: 0.0657997
    MeanOverCallCounts: 0.014874
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 0.000119925
    MeanOverProcs: 0.000182927
    MaxOverProcs: 0.000319004
    MeanOverCallCounts: 9.14633e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000592232
    MeanOverProcs: 0.000629485
    MaxOverProcs: 0.000653744
    MeanOverCallCounts: 3.14742e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.0150895
    MeanOverProcs: 0.0160637
    MaxOverProcs: 0.0183249
    MeanOverCallCounts: 0.000803187
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.0161049
    MeanOverProcs: 0.0170541
    MaxOverProcs: 0.0191813
    MeanOverCallCounts: 0.00852704
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.0136459
    MeanOverProcs: 0.0169517
    MaxOverProcs: 0.0201461
    MeanOverCallCounts: 0.0169517
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.00030303
    MeanOverProcs: 0.000319481
    MaxOverProcs: 0.000351906
    MeanOverCallCounts: 0.000319481
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.78814e-05
    MeanOverProcs: 3.8743e-05
    MaxOverProcs: 6.69956e-05
    MeanOverCallCounts: 1.93715e-05
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000211954
    MeanOverProcs: 0.00145066
    MaxOverProcs: 0.00190806
    MeanOverCallCounts: 0.00145066
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000744104
    MeanOverProcs: 0.00122529
    MaxOverProcs: 0.00142789
    MeanOverCallCounts: 0.00122529
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 0.000163078
    MeanOverProcs: 0.000554502
    MaxOverProcs: 0.000993252
    MeanOverCallCounts: 0.000221801
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 0.000170231
    MeanOverProcs: 0.00023073
    MaxOverProcs: 0.000271082
    MeanOverCallCounts: 1.15365e-05
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 7.10487e-05
    MeanOverProcs: 0.000117719
    MaxOverProcs: 0.000151396
    MeanOverCallCounts: 5.88596e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00660706
    MeanOverProcs: 0.00738448
    MaxOverProcs: 0.00861073
    MeanOverCallCounts: 0.000369224
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000204802
    MeanOverProcs: 0.000261843
    MaxOverProcs: 0.000315428
    MeanOverCallCounts: 8.72811e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00665092
    MeanOverProcs: 0.0082317
    MaxOverProcs: 0.0107107
    MeanOverCallCounts: 0.000205792
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.0103235
    MeanOverProcs: 0.0113443
    MaxOverProcs: 0.0123897
    MeanOverCallCounts: 0.000378142
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00246859
    MeanOverProcs: 0.00298011
    MaxOverProcs: 0.00405431
    MeanOverCallCounts: 0.000298011
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00679946
    MeanOverProcs: 0.00897139
    MaxOverProcs: 0.0121455
    MeanOverCallCounts: 0.000897139
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.00152802
    MeanOverProcs: 0.00193805
    MaxOverProcs: 0.00228715
    MeanOverCallCounts: 0.00193805
  "MILO::driver::total run time": 
    MinOverProcs: 0.405093
    MeanOverProcs: 0.406317
    MaxOverProcs: 0.406809
    MeanOverCallCounts: 0.406317
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.683727
    MeanOverProcs: 0.684013
    MaxOverProcs: 0.684336
    MeanOverCallCounts: 0.684013
  "MILO::function::decompose": 
    MinOverProcs: 0.00128102
    MeanOverProcs: 0.00142032
    MaxOverProcs: 0.00162315
    MeanOverCallCounts: 0.00142032
  "MILO::function::evaluate": 
    MinOverProcs: 0.00599623
    MeanOverProcs: 0.00678784
    MaxOverProcs: 0.00763059
    MeanOverCallCounts: 3.85125e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 6.4373e-05
    MeanOverProcs: 7.56979e-05
    MaxOverProcs: 8.44002e-05
    MeanOverCallCounts: 6.05583e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000640154
    MeanOverProcs: 0.000976562
    MaxOverProcs: 0.00188494
    MeanOverCallCounts: 0.000976562
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 0.000114202
    MeanOverProcs: 0.000438809
    MaxOverProcs: 0.00135303
    MeanOverCallCounts: 0.000438809
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.0116169
    MeanOverProcs: 0.0142068
    MaxOverProcs: 0.0176802
    MeanOverCallCounts: 0.0142068
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.0229719
    MeanOverProcs: 0.0264419
    MaxOverProcs: 0.0288241
    MeanOverCallCounts: 0.0264419
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.228406
    MeanOverProcs: 0.228537
    MaxOverProcs: 0.228694
    MeanOverCallCounts: 0.114269
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0910101
    MeanOverProcs: 0.0923665
    MaxOverProcs: 0.0931079
    MeanOverCallCounts: 0.0923665
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 2.28882e-05
    MeanOverProcs: 2.40207e-05
    MaxOverProcs: 2.59876e-05
    MeanOverCallCounts: 2.40207e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.063251
    MeanOverProcs: 0.0634635
    MaxOverProcs: 0.063828
    MeanOverCallCounts: 0.0634635
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 4.50611e-05
    MeanOverProcs: 7.25389e-05
    MaxOverProcs: 0.000138998
    MeanOverCallCounts: 7.25389e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00454712
    MeanOverProcs: 0.00456154
    MaxOverProcs: 0.00457001
    MeanOverCallCounts: 0.00456154
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.0045774
    MeanOverProcs: 0.00562173
    MaxOverProcs: 0.00665617
    MeanOverCallCounts: 0.000281087
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.00144911
    MeanOverProcs: 0.00168866
    MaxOverProcs: 0.0019486
    MeanOverCallCounts: 8.4433e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000806332
    MeanOverProcs: 0.000937283
    MaxOverProcs: 0.00119901
    MeanOverCallCounts: 1.33898e-05
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.0164964
    MeanOverProcs: 0.0190341
    MaxOverProcs: 0.0219128
    MeanOverCallCounts: 0.000271916
  "MILO::workset::reset*": 
    MinOverProcs: 0.000121832
    MeanOverProcs: 0.000128329
    MaxOverProcs: 0.00013423
    MeanOverCallCounts: 6.41644e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.106999
    MeanOverProcs: 0.131407
    MaxOverProcs: 0.151268
    MeanOverCallCounts: 0.0438022
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.00141692
    MeanOverProcs: 0.00518912
    MaxOverProcs: 0.00754285
    MeanOverCallCounts: 0.00518912
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.0159409
    MeanOverProcs: 0.0159903
    MaxOverProcs: 0.0160182
    MeanOverCallCounts: 0.0159903
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000976801
    MeanOverProcs: 0.00104898
    MaxOverProcs: 0.0012362
    MeanOverCallCounts: 0.000349661
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00742793
    MeanOverProcs: 0.00808299
    MaxOverProcs: 0.00843191
    MeanOverCallCounts: 0.00808299
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 0.000179052
    MeanOverProcs: 0.000264466
    MaxOverProcs: 0.00037694
    MeanOverCallCounts: 0.000264466
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000324965
    MeanOverProcs: 0.000421226
    MaxOverProcs: 0.000684977
    MeanOverCallCounts: 0.000421226
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000352144
    MeanOverProcs: 0.000421345
    MaxOverProcs: 0.000492096
    MeanOverCallCounts: 0.000421345
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000400066
    MeanOverProcs: 0.000680029
    MaxOverProcs: 0.001019
    MeanOverCallCounts: 0.000680029
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.00350809
    MeanOverProcs: 0.00414073
    MaxOverProcs: 0.00474596
    MeanOverCallCounts: 0.00414073
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.00188398
    MeanOverProcs: 0.00204414
    MaxOverProcs: 0.00223494
    MeanOverCallCounts: 0.00204414
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 1.81198e-05
    MeanOverProcs: 4.90546e-05
    MaxOverProcs: 0.000141859
    MeanOverCallCounts: 4.90546e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.00128913
    MeanOverProcs: 0.00148052
    MaxOverProcs: 0.00186396
    MeanOverCallCounts: 0.00148052
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 4.76837e-06
    MeanOverProcs: 5.24521e-06
    MaxOverProcs: 6.19888e-06
    MeanOverCallCounts: 5.24521e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 0.000117064
    MeanOverProcs: 0.000280559
    MaxOverProcs: 0.000735044
    MeanOverCallCounts: 0.000280559
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.78814e-05
    MeanOverProcs: 4.82798e-05
    MaxOverProcs: 0.000137091
    MeanOverCallCounts: 4.82798e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 4.91142e-05
    MeanOverProcs: 0.000128508
    MaxOverProcs: 0.000271082
    MeanOverCallCounts: 0.000128508
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.00088501
    MeanOverProcs: 0.00117576
    MaxOverProcs: 0.0014441
    MeanOverCallCounts: 0.00117576
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 7.29561e-05
    MeanOverProcs: 9.87649e-05
    MaxOverProcs: 0.000174046
    MeanOverCallCounts: 9.87649e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 8.51154e-05
    MeanOverProcs: 0.000127733
    MaxOverProcs: 0.000250816
    MeanOverCallCounts: 0.000127733
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000655174
    MeanOverProcs: 0.000899494
    MaxOverProcs: 0.00113487
    MeanOverCallCounts: 0.000899494
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000263929
    MeanOverProcs: 0.000712276
    MaxOverProcs: 0.00158501
    MeanOverCallCounts: 0.000712276
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.192012
    MeanOverProcs: 0.192726
    MaxOverProcs: 0.19331
    MeanOverCallCounts: 0.192726
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
