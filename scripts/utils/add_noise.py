#!/scratch/dtseidl/software/anaconda2/bin/python
"""
This adds gaussian noise to a set of measured data points
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess

stotal = 16
snum = range(0,stotal)

np.random.seed(200)

for s in snum:
  file = open("sensor." + str(s) + ".dat")
  lines_list = file.readlines()
  sdat = [[float(val) for val in line.split()] for line in lines_list]
  if s > 0:
    sensors = np.vstack((sensors,sdat))
  else:
    sensors = sdat
  
std_ux = np.std(sensors.T[1])
std_uy = np.std(sensors.T[2])

percent_noise = 5.0

noise_ux = np.random.normal(0.0,percent_noise/100.0*std_ux,stotal)
noise_uy = np.random.normal(0.0,percent_noise/100.0*std_uy,stotal)

noisy_sensors = np.array(sensors,copy=True)
noisy_sensors.T[1] = noisy_sensors.T[1] + noise_ux
noisy_sensors.T[2] = noisy_sensors.T[2] + noise_uy

'''
plt.scatter(np.array(snum),sensors.T[1],c='b')
plt.scatter(np.array(snum),noisy_sensors.T[1],c='r')
plt.title('u_x')
plt.show()

plt.figure()
plt.scatter(np.array(snum),sensors.T[2],c='b')
plt.scatter(np.array(snum),noisy_sensors.T[2],c='r')
plt.title('u_y')
plt.show()
'''

for s in snum:
  fid = open("noisy_sensor." + str(s) + ".dat","w")
  sdat = noisy_sensors[s]
  fid.write('%2.1f  ' % sdat[0])
  fid.write('%.8e  ' % sdat[1])
  fid.write('%.8e' % sdat[2])
  fid.close()

subprocess.call("./mover")

