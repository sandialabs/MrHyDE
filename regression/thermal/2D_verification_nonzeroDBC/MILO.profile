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
    MinOverProcs: 0.00610375
    MeanOverProcs: 0.00610638
    MaxOverProcs: 0.00610781
    MeanOverCallCounts: 0.00203546
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000268459
    MeanOverProcs: 0.00028038
    MaxOverProcs: 0.000288725
    MeanOverCallCounts: 1.27446e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.000145674
    MeanOverProcs: 0.000160694
    MaxOverProcs: 0.00019002
    MeanOverCallCounts: 4.46373e-06
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000163555
    MeanOverProcs: 0.000165641
    MaxOverProcs: 0.00016737
    MeanOverCallCounts: 7.52915e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.000712872
    MeanOverProcs: 0.000725389
    MaxOverProcs: 0.000736237
    MeanOverCallCounts: 5.18135e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000406027
    MeanOverProcs: 0.000581503
    MaxOverProcs: 0.000772953
    MeanOverCallCounts: 3.23057e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.00827646
    MeanOverProcs: 0.00835621
    MaxOverProcs: 0.00858688
    MeanOverCallCounts: 0.000596872
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00140738
    MeanOverProcs: 0.00145215
    MaxOverProcs: 0.00151753
    MeanOverCallCounts: 5.18624e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.0012958
    MeanOverProcs: 0.00200266
    MaxOverProcs: 0.00266385
    MeanOverCallCounts: 0.000667552
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 6.91414e-05
    MeanOverProcs: 7.08699e-05
    MaxOverProcs: 7.51019e-05
    MeanOverCallCounts: 3.5435e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000189066
    MeanOverProcs: 0.00020349
    MaxOverProcs: 0.000215292
    MeanOverCallCounts: 1.01745e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00642419
    MeanOverProcs: 0.00656158
    MaxOverProcs: 0.00669909
    MeanOverCallCounts: 0.000328079
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00674486
    MeanOverProcs: 0.00689512
    MaxOverProcs: 0.00705171
    MeanOverCallCounts: 0.00344756
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00617194
    MeanOverProcs: 0.00636399
    MaxOverProcs: 0.00648403
    MeanOverCallCounts: 0.00636399
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000169039
    MeanOverProcs: 0.000207067
    MaxOverProcs: 0.000250101
    MeanOverCallCounts: 0.000207067
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.40667e-05
    MeanOverProcs: 1.68681e-05
    MaxOverProcs: 2.21729e-05
    MeanOverCallCounts: 8.43406e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000159025
    MeanOverProcs: 0.000484049
    MaxOverProcs: 0.000814915
    MeanOverCallCounts: 0.000484049
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.00027895
    MeanOverProcs: 0.000285268
    MaxOverProcs: 0.000298023
    MeanOverCallCounts: 0.000285268
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 9.70364e-05
    MeanOverProcs: 0.00014478
    MaxOverProcs: 0.000192881
    MeanOverCallCounts: 5.79119e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 6.86646e-05
    MeanOverProcs: 7.36713e-05
    MaxOverProcs: 7.84397e-05
    MeanOverCallCounts: 3.68357e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 3.14713e-05
    MeanOverProcs: 3.44515e-05
    MaxOverProcs: 3.86238e-05
    MeanOverCallCounts: 1.72257e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00276899
    MeanOverProcs: 0.00286919
    MaxOverProcs: 0.00298238
    MeanOverCallCounts: 0.000143459
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 9.94205e-05
    MeanOverProcs: 0.000102758
    MaxOverProcs: 0.000108242
    MeanOverCallCounts: 3.42528e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00318456
    MeanOverProcs: 0.00322592
    MaxOverProcs: 0.00327969
    MeanOverCallCounts: 8.06481e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.0047102
    MeanOverProcs: 0.00475687
    MaxOverProcs: 0.00482607
    MeanOverCallCounts: 0.000158562
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00119328
    MeanOverProcs: 0.00123125
    MaxOverProcs: 0.00127697
    MeanOverCallCounts: 0.000123125
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00329018
    MeanOverProcs: 0.00339055
    MaxOverProcs: 0.00357628
    MeanOverCallCounts: 0.000339055
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.00067997
    MeanOverProcs: 0.000686228
    MaxOverProcs: 0.000699043
    MeanOverCallCounts: 0.000686228
  "MILO::driver::total run time": 
    MinOverProcs: 0.0927901
    MeanOverProcs: 0.0930958
    MaxOverProcs: 0.09341
    MeanOverCallCounts: 0.0930958
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.175118
    MeanOverProcs: 0.17525
    MaxOverProcs: 0.175386
    MeanOverCallCounts: 0.17525
  "MILO::function::decompose": 
    MinOverProcs: 0.000720024
    MeanOverProcs: 0.000875771
    MaxOverProcs: 0.00102115
    MeanOverCallCounts: 0.000875771
  "MILO::function::evaluate": 
    MinOverProcs: 0.00142765
    MeanOverProcs: 0.00148565
    MaxOverProcs: 0.00155878
    MeanOverCallCounts: 9.14244e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 3.19481e-05
    MeanOverProcs: 3.78489e-05
    MaxOverProcs: 5.05447e-05
    MeanOverCallCounts: 3.02792e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000341892
    MeanOverProcs: 0.000534236
    MaxOverProcs: 0.00069809
    MeanOverCallCounts: 0.000534236
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 5.4121e-05
    MeanOverProcs: 9.799e-05
    MaxOverProcs: 0.000154018
    MeanOverCallCounts: 9.799e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00571489
    MeanOverProcs: 0.00580299
    MaxOverProcs: 0.00590301
    MeanOverCallCounts: 0.00580299
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.00655699
    MeanOverProcs: 0.00666529
    MaxOverProcs: 0.006778
    MeanOverCallCounts: 0.00666529
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.0397091
    MeanOverProcs: 0.0397093
    MaxOverProcs: 0.03971
    MeanOverCallCounts: 0.0198547
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.029238
    MeanOverProcs: 0.0295405
    MaxOverProcs: 0.0298369
    MeanOverCallCounts: 0.0295405
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.19209e-05
    MeanOverProcs: 1.29938e-05
    MaxOverProcs: 1.40667e-05
    MeanOverCallCounts: 1.29938e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.0193841
    MeanOverProcs: 0.0193883
    MaxOverProcs: 0.0193961
    MeanOverCallCounts: 0.0193883
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.59876e-05
    MeanOverProcs: 3.41535e-05
    MaxOverProcs: 3.88622e-05
    MeanOverCallCounts: 3.41535e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00226998
    MeanOverProcs: 0.00227094
    MaxOverProcs: 0.00227284
    MeanOverCallCounts: 0.00227094
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.00197124
    MeanOverProcs: 0.00204557
    MaxOverProcs: 0.00210547
    MeanOverCallCounts: 0.000102279
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000755072
    MeanOverProcs: 0.000795066
    MaxOverProcs: 0.000851631
    MeanOverCallCounts: 3.97533e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000455618
    MeanOverProcs: 0.00046593
    MaxOverProcs: 0.000481129
    MeanOverCallCounts: 6.65614e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.00768876
    MeanOverProcs: 0.00776005
    MaxOverProcs: 0.00787377
    MeanOverCallCounts: 0.000110858
  "MILO::workset::reset*": 
    MinOverProcs: 0.00011611
    MeanOverProcs: 0.000118673
    MaxOverProcs: 0.00012517
    MeanOverCallCounts: 5.93364e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0328088
    MeanOverProcs: 0.0332407
    MaxOverProcs: 0.0341766
    MeanOverCallCounts: 0.0110802
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.000822067
    MeanOverProcs: 0.00093931
    MaxOverProcs: 0.00103807
    MeanOverCallCounts: 0.00093931
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.00555587
    MeanOverProcs: 0.00558919
    MaxOverProcs: 0.00560594
    MeanOverCallCounts: 0.00558919
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000450611
    MeanOverProcs: 0.00046885
    MaxOverProcs: 0.000494957
    MeanOverCallCounts: 0.000156283
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00227618
    MeanOverProcs: 0.00245953
    MaxOverProcs: 0.00267315
    MeanOverCallCounts: 0.00245953
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 8.98838e-05
    MeanOverProcs: 0.000124514
    MaxOverProcs: 0.000189066
    MeanOverCallCounts: 0.000124514
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000161171
    MeanOverProcs: 0.000209332
    MaxOverProcs: 0.000262976
    MeanOverCallCounts: 0.000209332
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.00018096
    MeanOverProcs: 0.000240743
    MaxOverProcs: 0.000305891
    MeanOverCallCounts: 0.000240743
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000197887
    MeanOverProcs: 0.000203967
    MaxOverProcs: 0.000217915
    MeanOverCallCounts: 0.000203967
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.000889778
    MeanOverProcs: 0.000913262
    MaxOverProcs: 0.000937939
    MeanOverCallCounts: 0.000913262
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000374079
    MeanOverProcs: 0.000375509
    MaxOverProcs: 0.00037694
    MeanOverCallCounts: 0.000375509
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 8.10623e-06
    MeanOverProcs: 8.82149e-06
    MaxOverProcs: 9.05991e-06
    MeanOverCallCounts: 8.82149e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000337124
    MeanOverProcs: 0.000356495
    MaxOverProcs: 0.000365019
    MeanOverCallCounts: 0.000356495
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 3.09944e-06
    MeanOverProcs: 3.09944e-06
    MaxOverProcs: 3.09944e-06
    MeanOverCallCounts: 3.09944e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 7.10487e-05
    MeanOverProcs: 9.40561e-05
    MaxOverProcs: 0.00013113
    MeanOverCallCounts: 9.40561e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.00136e-05
    MeanOverProcs: 1.10269e-05
    MaxOverProcs: 1.19209e-05
    MeanOverCallCounts: 1.10269e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.71797e-05
    MeanOverProcs: 3.32594e-05
    MaxOverProcs: 4.29153e-05
    MeanOverCallCounts: 3.32594e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000344992
    MeanOverProcs: 0.000358045
    MaxOverProcs: 0.000374079
    MeanOverCallCounts: 0.000358045
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 5.19753e-05
    MeanOverProcs: 5.26905e-05
    MaxOverProcs: 5.38826e-05
    MeanOverCallCounts: 5.26905e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.69685e-05
    MeanOverProcs: 4.72069e-05
    MaxOverProcs: 4.79221e-05
    MeanOverCallCounts: 4.72069e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000226021
    MeanOverProcs: 0.000236511
    MaxOverProcs: 0.000250101
    MeanOverCallCounts: 0.000236511
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000172853
    MeanOverProcs: 0.000210762
    MaxOverProcs: 0.000251055
    MeanOverCallCounts: 0.000210762
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.0420179
    MeanOverProcs: 0.0420927
    MaxOverProcs: 0.042177
    MeanOverCallCounts: 0.0420927
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
