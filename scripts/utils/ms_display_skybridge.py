from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()
# To run:
#   1. Run write_list.py (on Skybridge!)
#   2. Scp the fnames.txt file to your computer and update the fnames_file variable
#       in this script
#   3. Set the Skybridge path for exofiles_dir
#   4. Set dim, t, and camera variables in this script
#   5. Open Paraview and connect to Skybridge
#   6. Click Tools > Python Shell
#   7. Click Run Script and select this script
#   8. In the pipeline browser, scroll down to the bottom and click
#      on the eye next to the GroupDatasets1 filter

# User defined quantities:
# Folder containing subgrid output (on Skybridge!)
exofiles_dir = "/gscratch/dtseidl/jobs/2017/5_24/ms_thermal_only/second_round/subgrid_data"
# File that contains names of subgrid_output* files in exofiles_dir (must be on local machine!)
fnames_file = "/Users/dtseidl/Desktop/paraview_scripting/fnames2.txt"
# spatial dimension (as string)
dim = "3D"
# time step to display
t = 8 
# camera variables
# 2D
#camera_position = [0.005, 0.005, 10000.0]
#camera_focal_point = [0.005, 0.005, 0.0]
#camera_parallel_scale = 0.01
# 3D
camera_position = [0.017689521496123933, 0.010350820509929916, 0.027932216343520083]
camera_focal_point = [0.004999999888241282, 0.0012499999720603225, 0.004999999888241285]
camera_view_up = [-0.14986320289908484, 0.9446188873455699, -0.2919523558508227]
camera_parallel_scale = 0.007180703147671309

with open(fnames_file) as f:
  files = f.read().splitlines()
samp = "subgrid_output" + str(t) + ".exo."
nelem = []
ncores = 1 + max([ int(files[i].split(".")[2]) for i in range(len(files))])

for core in range(ncores):
  sample = samp + str(core) + "."
  nelem.append(len([int(files[i].replace(sample,"")) for i in  range(len(files)) if sample in files[i]]))

readers = {}
readers_list =[]

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.ResetCamera()
renderView1.InteractionMode = dim
renderView1.CameraPosition = camera_position
renderView1.CameraFocalPoint = camera_focal_point
if dim == "3D":
  renderView1.CameraViewUp = camera_view_up
renderView1.CameraParallelScale = camera_parallel_scale

for core in range(ncores):
  base = exofiles_dir + "/subgrid_time_" + str(t) + ".exo." + str(core) + "."
  name = [base + str(i) for i in range(nelem[core])]
  readers[core] = ExodusIIReader(FileName=name,FaceVariables=None,GlobalVariables=None)
  readers_list.append(readers[core])

groupDatasets1 = GroupDatasets(Input=readers_list)


