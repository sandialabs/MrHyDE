#!/Users/dtseidl/anaconda/bin/python

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

dat = pd.read_csv("ROL_out.txt", delim_whitespace=True, header=1)

iters = dat["iter"]
obj = dat["value"]

sns.set(style="white")
sns.set_palette("hls",9)

fig1, ax1 = plt.subplots(figsize=(11.75,8.75))
ax1.set_xlabel("iter", fontsize=14)
ax1.set_ylabel("Log10 Objective Function Value", fontsize=14)
ax1.set_title("Convergence Data Rich Simulation", fontsize=14)
#ax1.semilogy(iters,obj,label="cvg",marker="o",linestyle="None")
ax1.loglog(iters,obj,label="cvg",marker="o",linestyle="None")
