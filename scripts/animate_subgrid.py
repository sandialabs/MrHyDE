from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()
from os import listdir

# To run:
#   1. Adjust user defined quantities
#   2. Open Paraview
#   3. Click Tools > Python Shell
#   4. Click Run Script and select this script
#   5. Press the play button to view the animation
#   6. Save with File > Save Animation

# User defined quantities:
# Folder containing subgrid output
exofiles_dir = "/Users/dtseidl/src/milo-svn/regression/thermal/2D_verification_multiscale_dynamicmultimodel/subgrid_data"
# Field to be animated
field = 'e'
# If a vector, true will plot a component and false will plot magnitude
plotComponent = False
# Which component to plot if a vector
component = 0
# Camera x and y positions (z needed if 3D)
camera_xpos = 0.5
camera_ypos = 0.5

def create_display(readers,displays,t,core):
  base = exofiles_dir + "/subgrid_time_" + str(t) + ".exo." + str(core) + "."
  name = [base + str(i) for i in range(nelem[core])]
  readers[t] = ExodusIIReader(FileName=name)

def create_animation(readers,displays,tracks,keyframe1s,keyframe2s,keyframe3s,t):
  ColorBy(displays[t], ('POINTS', field))
  if plotComponent:
    fLUT = GetColorTransferFunction(field)
    fLUT.VectorMode = 'Component'
    fLUT.VectorComponent = component
  displays[t].RescaleTransferFunctionToDataRange(True)
  if (t != 0):
    displays[t].SetRepresentationType('Surface With Edges')
  tracks[t] = GetAnimationTrack('Visibility',index=0,proxy=readers[t])

  if (t == 0):
    keyframe1s[t] = CompositeKeyFrame()
    keyframe1s[t].KeyTime = 0.0
    keyframe1s[t].KeyValues = [1.0]
    keyframe1s[t].Interpolation = 'Boolean'
    keyframe2s[t] = CompositeKeyFrame()
    keyframe2s[t].KeyTime = (float(t)+1.0)/N
    keyframe2s[t].KeyValues = [0.0]
    tracks[t].KeyFrames = [keyframe1s[t], keyframe2s[t]]
  else:
    keyframe1s[t] = CompositeKeyFrame()
    keyframe1s[t].KeyTime = 0
    keyframe1s[t].KeyValues = [0.0]
    keyframe1s[t].Interpolation = 'Boolean'
    keyframe2s[t] = CompositeKeyFrame()
    keyframe2s[t].KeyTime = float(t)/N
    keyframe2s[t].KeyValues = [1.0]
    keyframe2s[t].Interpolation = 'Boolean'
    keyframe3s[t] = CompositeKeyFrame()
    keyframe3s[t].KeyTime = (float(t)+1.0)/N
    if t == nsteps: 
      keyframe3s[t].KeyValues = [1.0]
    else:
      keyframe3s[t].KeyValues = [0.0]
    tracks[t].KeyFrames = [keyframe1s[t], keyframe2s[t], keyframe3s[t]]

  tracks[t].TimeMode = "Normalized"
  tracks[t].StartTime = 0.0
  tracks[t].EndTime = 1.0
  tracks[t].Enabled = 1

files = listdir(exofiles_dir)
nsteps = max( [int(files[i].split("_")[2].split(".")[0]) for i in range(len(files)) ])
ncores = 1 + max([ int(files[i].split(".")[2]) for i in range(len(files))])
samp = "subgrid_time_0.exo."
nelem = []

for j in range(ncores):
  sample = samp + str(j) + "."
  nelem.append(len([int(files[i].replace(sample,"")) for i in  range(len(files)) if sample in files[i]]))

readers = [dict() for core in range(ncores)]
displays = [dict() for cores in range(ncores)]
tracks = [dict() for cores in range(ncores)]
keyframe1s = [dict() for cores in range(ncores)]
keyframe2s = [dict() for cores in range(ncores)]
keyframe3s = [dict() for cores in range(ncores)]

N = 1 + nsteps

for t in range(N):
  for core in range(ncores):

    create_display(readers[core],displays[core],t,core)
    if t == 0:
      renderView1 = GetActiveViewOrCreate('RenderView')

    displays[core][t] = Show(readers[core][t],renderView1)
    if t == 0:
      renderView1.ResetCamera()
      renderView1.InteractionMode = '2D'
      renderView1.CameraPosition = [camera_xpos, camera_ypos, 10000.0]
      renderView1.CameraFocalPoint = [camera_xpos, camera_ypos, 0.0]
      animationScene1 = GetAnimationScene()
      animationScene1.EndTime = N
      animationScene1.PlayMode = 'Real Time'

    create_animation(readers[core],displays[core],tracks[core],keyframe1s[core],keyframe2s[core],keyframe3s[core],t)
