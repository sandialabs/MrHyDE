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
    MinOverProcs: 0.00806618
    MeanOverProcs: 0.0081867
    MaxOverProcs: 0.00825334
    MeanOverCallCounts: 0.0027289
  "Belos: DGKS[2]: Ortho (Inner Product)": 
    MinOverProcs: 0.000345945
    MeanOverProcs: 0.000352442
    MaxOverProcs: 0.00035882
    MeanOverCallCounts: 1.60201e-05
  "Belos: DGKS[2]: Ortho (Norm)": 
    MinOverProcs: 0.00034976
    MeanOverProcs: 0.000445485
    MaxOverProcs: 0.000548124
    MeanOverCallCounts: 1.23746e-05
  "Belos: DGKS[2]: Ortho (Update)": 
    MinOverProcs: 0.000187397
    MeanOverProcs: 0.00020963
    MaxOverProcs: 0.000252962
    MeanOverCallCounts: 9.52862e-06
  "Belos: DGKS[2]: Orthogonalization": 
    MinOverProcs: 0.00103498
    MeanOverProcs: 0.00115615
    MaxOverProcs: 0.00124168
    MeanOverCallCounts: 8.25822e-05
  "Belos: Operation Op*x": 
    MinOverProcs: 0.000538588
    MeanOverProcs: 0.000701606
    MaxOverProcs: 0.000940084
    MeanOverCallCounts: 3.89781e-05
  "Belos: Operation Prec*x": 
    MinOverProcs: 0.0106661
    MeanOverProcs: 0.0106953
    MaxOverProcs: 0.010752
    MeanOverCallCounts: 0.000763948
  "Ifpack2::Chebyshev::apply": 
    MinOverProcs: 0.00192642
    MeanOverProcs: 0.00197047
    MaxOverProcs: 0.00201821
    MeanOverCallCounts: 7.03739e-05
  "Ifpack2::Chebyshev::compute": 
    MinOverProcs: 0.00318885
    MeanOverProcs: 0.00439072
    MaxOverProcs: 0.00672698
    MeanOverCallCounts: 0.00146357
  "MILO::assembly::computeJacRes() - gather": 
    MinOverProcs: 8.4877e-05
    MeanOverProcs: 9.31025e-05
    MaxOverProcs: 0.000107765
    MeanOverCallCounts: 4.65512e-05
  "MILO::assembly::computeJacRes() - insert": 
    MinOverProcs: 0.000242472
    MeanOverProcs: 0.000252724
    MaxOverProcs: 0.000276566
    MeanOverCallCounts: 1.26362e-05
  "MILO::assembly::computeJacRes() - physics evaluation": 
    MinOverProcs: 0.00724626
    MeanOverProcs: 0.00760674
    MaxOverProcs: 0.00784588
    MeanOverCallCounts: 0.000380337
  "MILO::assembly::computeJacRes() - total assembly": 
    MinOverProcs: 0.00762892
    MeanOverProcs: 0.00801861
    MaxOverProcs: 0.00824594
    MeanOverCallCounts: 0.00400931
  "MILO::assembly::createCells()": 
    MinOverProcs: 0.00553799
    MeanOverProcs: 0.00567544
    MaxOverProcs: 0.00586796
    MeanOverCallCounts: 0.00567544
  "MILO::assembly::createWorkset()": 
    MinOverProcs: 0.000183105
    MeanOverProcs: 0.000216305
    MaxOverProcs: 0.000252962
    MeanOverCallCounts: 0.000216305
  "MILO::assembly::dofConstraints()": 
    MinOverProcs: 1.3113e-05
    MeanOverProcs: 1.48416e-05
    MaxOverProcs: 1.71661e-05
    MeanOverCallCounts: 7.42078e-06
  "MILO::assembly::setDirichlet()": 
    MinOverProcs: 0.000132084
    MeanOverProcs: 0.00023675
    MaxOverProcs: 0.000370979
    MeanOverCallCounts: 0.00023675
  "MILO::assembly::setInitial()": 
    MinOverProcs: 0.000413895
    MeanOverProcs: 0.00043118
    MaxOverProcs: 0.000452995
    MeanOverCallCounts: 0.00043118
  "MILO::boundaryCell - build basis": 
    MinOverProcs: 8.98838e-05
    MeanOverProcs: 0.000130713
    MaxOverProcs: 0.0001719
    MeanOverCallCounts: 5.22852e-05
  "MILO::cell::computeJacRes() - fill local Jacobian": 
    MinOverProcs: 9.03606e-05
    MeanOverProcs: 9.93609e-05
    MaxOverProcs: 0.000110388
    MeanOverCallCounts: 4.96805e-06
  "MILO::cell::computeJacRes() - fill local residual": 
    MinOverProcs: 4.79221e-05
    MeanOverProcs: 4.8697e-05
    MaxOverProcs: 5.00679e-05
    MeanOverCallCounts: 2.43485e-06
  "MILO::cell::computeJacRes() - volume residual": 
    MinOverProcs: 0.00327277
    MeanOverProcs: 0.00344008
    MaxOverProcs: 0.00373149
    MeanOverCallCounts: 0.000172004
  "MILO::cell::computeSolAvg()": 
    MinOverProcs: 0.000118494
    MeanOverProcs: 0.000125647
    MaxOverProcs: 0.000133514
    MeanOverCallCounts: 4.18822e-06
  "MILO::cell::computeSolnFaceIP()": 
    MinOverProcs: 0.00304914
    MeanOverProcs: 0.00337726
    MaxOverProcs: 0.00379419
    MeanOverCallCounts: 8.44315e-05
  "MILO::cell::computeSolnVolIP()": 
    MinOverProcs: 0.00508547
    MeanOverProcs: 0.00531107
    MaxOverProcs: 0.00555491
    MeanOverCallCounts: 0.000177036
  "MILO::cell::constructor - build basis": 
    MinOverProcs: 0.00109601
    MeanOverProcs: 0.00111538
    MaxOverProcs: 0.00113249
    MeanOverCallCounts: 0.000111538
  "MILO::cell::constructor - build face basis": 
    MinOverProcs: 0.00294542
    MeanOverProcs: 0.00301284
    MaxOverProcs: 0.00311208
    MeanOverCallCounts: 0.000301284
  "MILO::cellMetaData::constructor()": 
    MinOverProcs: 0.000619173
    MeanOverProcs: 0.000633299
    MaxOverProcs: 0.000642061
    MeanOverCallCounts: 0.000633299
  "MILO::driver::total run time": 
    MinOverProcs: 0.123869
    MeanOverProcs: 0.123971
    MaxOverProcs: 0.124104
    MeanOverCallCounts: 0.123971
  "MILO::driver::total setup and execution time": 
    MinOverProcs: 0.198755
    MeanOverProcs: 0.198957
    MaxOverProcs: 0.199111
    MeanOverCallCounts: 0.198957
  "MILO::function::decompose": 
    MinOverProcs: 0.000695944
    MeanOverProcs: 0.000711441
    MaxOverProcs: 0.00072813
    MeanOverCallCounts: 0.000711441
  "MILO::function::evaluate": 
    MinOverProcs: 0.00144434
    MeanOverProcs: 0.00151181
    MaxOverProcs: 0.00158191
    MeanOverCallCounts: 9.30346e-06
  "MILO::physics::getSideInfo()": 
    MinOverProcs: 2.67029e-05
    MeanOverProcs: 2.94447e-05
    MaxOverProcs: 3.17097e-05
    MeanOverCallCounts: 2.35558e-06
  "MILO::physics::setBCData()": 
    MinOverProcs: 0.000191927
    MeanOverProcs: 0.000220716
    MaxOverProcs: 0.000238895
    MeanOverCallCounts: 0.000220716
  "MILO::physics::setDirichletData()": 
    MinOverProcs: 4.79221e-05
    MeanOverProcs: 6.34193e-05
    MaxOverProcs: 7.89165e-05
    MeanOverCallCounts: 6.34193e-05
  "MILO::postprocess::computeError": 
    MinOverProcs: 0.00508404
    MeanOverProcs: 0.00560683
    MaxOverProcs: 0.00626302
    MeanOverCallCounts: 0.00560683
  "MILO::postprocess::writeSolution": 
    MinOverProcs: 0.0128732
    MeanOverProcs: 0.0135419
    MaxOverProcs: 0.0140612
    MeanOverCallCounts: 0.0135419
  "MILO::solver::linearSolver()": 
    MinOverProcs: 0.0552998
    MeanOverProcs: 0.0553116
    MaxOverProcs: 0.0553257
    MeanOverCallCounts: 0.0276558
  "MILO::solver::projectDirichlet()": 
    MinOverProcs: 0.0350931
    MeanOverProcs: 0.0351928
    MaxOverProcs: 0.0353351
    MeanOverCallCounts: 0.0351928
  "MILO::solver::setDirichlet()": 
    MinOverProcs: 1.81198e-05
    MeanOverProcs: 1.90139e-05
    MaxOverProcs: 2.00272e-05
    MeanOverCallCounts: 1.90139e-05
  "MILO::solver::setInitial()": 
    MinOverProcs: 0.029382
    MeanOverProcs: 0.029387
    MaxOverProcs: 0.029393
    MeanOverCallCounts: 0.029387
  "MILO::solver::setupFixedDOFs()": 
    MinOverProcs: 2.31266e-05
    MeanOverProcs: 2.74181e-05
    MaxOverProcs: 2.98023e-05
    MeanOverCallCounts: 2.74181e-05
  "MILO::solver::setupLinearAlgebra()": 
    MinOverProcs: 0.00147796
    MeanOverProcs: 0.00147867
    MaxOverProcs: 0.00147891
    MeanOverCallCounts: 0.00147867
  "MILO::thermal::volumeResidual() - evaluation of residual": 
    MinOverProcs: 0.0024569
    MeanOverProcs: 0.00257784
    MaxOverProcs: 0.00285697
    MeanOverCallCounts: 0.000128892
  "MILO::thermal::volumeResidual() - function evaluation": 
    MinOverProcs: 0.000782251
    MeanOverProcs: 0.000822484
    MaxOverProcs: 0.00083971
    MeanOverCallCounts: 4.11242e-05
  "MILO::workset::computeSolnVolIP - allocate/compute seeded": 
    MinOverProcs: 0.000255823
    MeanOverProcs: 0.000275731
    MaxOverProcs: 0.000291824
    MeanOverCallCounts: 9.19104e-06
  "MILO::workset::computeSolnVolIP - compute seeded sol at ip": 
    MinOverProcs: 0.0045898
    MeanOverProcs: 0.00481707
    MaxOverProcs: 0.0050447
    MeanOverCallCounts: 0.000160569
  "MILO::workset::reset*": 
    MinOverProcs: 0.000119448
    MeanOverProcs: 0.000128686
    MaxOverProcs: 0.000137091
    MeanOverCallCounts: 6.43432e-06
  "MueLu: ParameterListInterpreter (ParameterList)": 
    MinOverProcs: 0.0423322
    MeanOverProcs: 0.0448775
    MaxOverProcs: 0.0461781
    MeanOverCallCounts: 0.0149592
  "STK_Interface::setupExodusFile(filename)": 
    MinOverProcs: 0.00100207
    MeanOverProcs: 0.00188178
    MaxOverProcs: 0.00269794
    MeanOverCallCounts: 0.00188178
  "STK_Interface::writeToExodus(timestep)": 
    MinOverProcs: 0.0112071
    MeanOverProcs: 0.0112283
    MaxOverProcs: 0.01125
    MeanOverCallCounts: 0.0112283
  "UtilitiesBase::GetMatrixDiagonalInverse": 
    MinOverProcs: 0.000541925
    MeanOverProcs: 0.000608563
    MaxOverProcs: 0.000709057
    MeanOverCallCounts: 0.000202854
  "panzer::DOFManager::buildGlobalUnknowns": 
    MinOverProcs: 0.00212717
    MeanOverProcs: 0.00214624
    MaxOverProcs: 0.00218987
    MeanOverCallCounts: 0.00214624
  "panzer::DOFManager::buildGlobalUnknowns::build_ghosted_array": 
    MinOverProcs: 7.89165e-05
    MeanOverProcs: 8.1718e-05
    MaxOverProcs: 8.29697e-05
    MeanOverCallCounts: 8.1718e-05
  "panzer::DOFManager::buildGlobalUnknowns::build_local_ids": 
    MinOverProcs: 0.000141859
    MeanOverProcs: 0.000144184
    MaxOverProcs: 0.000146866
    MeanOverCallCounts: 0.000144184
  "panzer::DOFManager::buildGlobalUnknowns::build_orientation": 
    MinOverProcs: 0.000163078
    MeanOverProcs: 0.000164986
    MaxOverProcs: 0.000166893
    MeanOverCallCounts: 0.000164986
  "panzer::DOFManager::buildGlobalUnknowns::build_owned_vector": 
    MinOverProcs: 0.000179052
    MeanOverProcs: 0.000182986
    MaxOverProcs: 0.000191927
    MeanOverCallCounts: 0.000182986
  "panzer::DOFManager::buildGlobalUnknowns_GUN": 
    MinOverProcs: 0.000849962
    MeanOverProcs: 0.000871718
    MaxOverProcs: 0.000891924
    MeanOverCallCounts: 0.000871718
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_04 createOneToOne": 
    MinOverProcs: 0.000345945
    MeanOverProcs: 0.000348687
    MaxOverProcs: 0.000351906
    MeanOverCallCounts: 0.000348687
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_05 alloc_unique_mv": 
    MinOverProcs: 8.10623e-06
    MeanOverProcs: 8.82149e-06
    MaxOverProcs: 9.05991e-06
    MeanOverCallCounts: 8.82149e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_06 export": 
    MinOverProcs: 0.000337124
    MeanOverProcs: 0.000356376
    MaxOverProcs: 0.000364065
    MeanOverCallCounts: 0.000356376
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_07-09 local_count": 
    MinOverProcs: 2.86102e-06
    MeanOverProcs: 2.98023e-06
    MaxOverProcs: 3.09944e-06
    MeanOverCallCounts: 2.98023e-06
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_10 prefix_sum": 
    MinOverProcs: 6.38962e-05
    MeanOverProcs: 8.72016e-05
    MaxOverProcs: 0.000123024
    MeanOverCallCounts: 8.72016e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_13-21 gid_assignment": 
    MinOverProcs: 1.00136e-05
    MeanOverProcs: 1.0252e-05
    MaxOverProcs: 1.09673e-05
    MeanOverCallCounts: 1.0252e-05
  "panzer::DOFManager::buildGlobalUnknowns_GUN::line_23 final_import": 
    MinOverProcs: 2.31266e-05
    MeanOverProcs: 2.77758e-05
    MaxOverProcs: 3.60012e-05
    MeanOverCallCounts: 2.77758e-05
  "panzer::DOFManager::buildTaggedMultiVector": 
    MinOverProcs: 0.000320911
    MeanOverProcs: 0.000335038
    MaxOverProcs: 0.000349045
    MeanOverCallCounts: 0.000335038
  "panzer::DOFManager::buildTaggedMultiVector::allocate_tagged_multivector": 
    MinOverProcs: 4.79221e-05
    MeanOverProcs: 4.79221e-05
    MaxOverProcs: 4.79221e-05
    MeanOverCallCounts: 4.79221e-05
  "panzer::DOFManager::buildTaggedMultiVector::fill_tagged_multivector": 
    MinOverProcs: 4.1008e-05
    MeanOverProcs: 4.18425e-05
    MaxOverProcs: 4.22001e-05
    MeanOverCallCounts: 4.18425e-05
  "panzer::DOFManager::builderOverlapMapFromElements": 
    MinOverProcs: 0.000211
    MeanOverProcs: 0.000225067
    MaxOverProcs: 0.000239134
    MeanOverCallCounts: 0.000225067
  "panzer::SquareQuadMeshFactory::buildUncomittedMesh()": 
    MinOverProcs: 0.000165939
    MeanOverProcs: 0.000168443
    MaxOverProcs: 0.000170946
    MeanOverCallCounts: 0.000168443
  "panzer::SquareQuadMeshFactory::completeMeshConstruction()": 
    MinOverProcs: 0.040746
    MeanOverProcs: 0.0407517
    MaxOverProcs: 0.040766
    MeanOverCallCounts: 0.0407517
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
