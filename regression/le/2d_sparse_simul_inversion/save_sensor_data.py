# run with pvpython

from paraview.simple import *
import numpy as np
import pickle

path = "."
efile = "clean_data.exo"
fname = path + "/" + efile

Lx = 1.0
Ly = 1.0
Nx = 20
Ny = 20
Ns = Nx*Ny # number of sensors

sx = np.linspace(0,Lx,Nx+1) + Lx/(2*Nx);
sy = np.linspace(0,Ly,Ny+1) + Ly/(2*Ny);
sx = sx[0:-1]
sy = sy[0:-1]

xCoor = np.zeros(Ns)
yCoor = np.zeros(Ns)
zCoor = np.zeros(Ns)

field_names = ["d", "d", "sig11", "sig12", "sig22"]

field_vals = np.empty((len(field_names),Ns))

reader = ExodusIIReader(FileName=fname)
probe = ProbeLocation(reader, ProbeType = "Fixed Radius Point Source")

k = 0
zc = 0.0

for ey in range(Ny):
  for ex in range(Nx):
    xc = sx[ex]
    yc = sy[ey]
    xCoor[k] = xc
    yCoor[k] = yc
    zCoor[k] = zc
    probe.ProbeType.Center = [xc, yc, zc]
    data = servermanager.Fetch(probe)
    for f in range(len(field_names)):
      values = data.GetPointData().GetArray(field_names[f])
      if f < 2:
        val = values.GetValue(f)
      else:
        val = values.GetValue(0)
      field_vals[f,k] = val
    k += 1

sensor_data = { "xCoor" : xCoor, "yCoor" : yCoor, "zCoor" : zCoor}

for f in range(len(field_names)):
  if f == 0:
    sensor_data.update({field_names[f] + "_x" : field_vals[f,:]})
  elif f == 1:
    sensor_data.update({field_names[f] + "_y" : field_vals[f,:]})
  else:
    sensor_data.update({field_names[f] : field_vals[f,:]})

pickle.dump( sensor_data, open( "sensor_data.p", "wb"))
