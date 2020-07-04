#!/usr/bin/python

# This reads in the profile file (MILO.profile) as YAML and creates graphics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

prof = open("MILO.profile", "r")

lines = prof.readlines()
numlines = len(lines)


# Important settings/options
nargs = len(sys.argv)
profile_time = False
profile_timepereval = True
profile_counts = True
filters = ["MILO","panzer"]
chart_type = "bar"
chart_orient = "vertical"

memotol = 0.1 # only use labels for the functions that take over a certain proportion of the total time
if nargs > 1:
  for k in range(1,nargs) :
    if sys.argv[k] == "time" :
      profile_time = True
      profile_timepereval = False
      profile_counts = False
    elif sys.argv[k] == "timepereval" :
      profile_time = False
      profile_timepereval = True
      profile_counts = False
    elif sys.argv[k] == "counts" :
      profile_time = False
      profile_timepereval = False
      profile_counts = True
    elif sys.argv[k] == "bar" :
      chart_type = "bar"
    elif sys.argv[k] == "donut" :
      chart_type = "donut"
    elif sys.argv[k] == "vertical" :
      chart_orient = "vertical"
    elif sys.argv[k] == "horizontal" :
      chart_orient = "horizontal"



scale_timers = False
if chart_type == "donut" :
  scale_timers = True

timer_tags = []
timer_mins = []
timer_maxs = []
timer_means = []
timer_meanocs = []
timer_memos = []

intimers = False
havetimer = False
  
ll = 0
while (ll < numlines):
  line = lines[ll]
  ll += 1
  tstart = False
  inlabel = False
  if (profile_time) :
    if (line[0:11] == "Total times"):
      tstart = True
      intimers = True
  elif (profile_counts) :
    if (line[0:11] == "Call counts"):
      tstart = True
      intimers = True
    
  if intimers and not tstart :
    if (line[0:2] == "  ") :
      for filter in filters :

        numfilt = len(filter)

        if (line[3:numfilt+3] == filter) :
          tag = line[numfilt+5]
          intag = True
          for i in range(numfilt+6,len(line)) :
            if intag :
              if line[i] == ":":
                intag = False
                memo = line[i+2:len(line)-4]
              elif line[i] == "-" :
                intag = False
                memo = line[i+2:len(line)-4]
              else :
                tag += line[i]

          timer_tags.append(tag)
          timer_memos.append(memo)
          
          line = lines[ll]
          timer_mins.append(float(line[17:len(line)-1]))
          ll += 1
          
          line = lines[ll]
          timer_means.append(float(line[18:len(line)-1]))
          ll += 1
          
          line = lines[ll]
          timer_maxs.append(float(line[17:len(line)-1]))
          ll += 1
          
          line = lines[ll]
          timer_meanocs.append(float(line[23:len(line)-1]))
          ll += 1
          
          havetimer = True
          inlabel = True
        else :
          inlabel = False
          havetimer = True
    else :
      intimers = False


timer_groups = [timer_tags[0]]
timer_group_sizes = [timer_means[0]]
num_groups = [1]

for k in range(1,len(timer_tags)):
  found = False
  index = 0
  for j in range(len(timer_groups)) :
    if timer_tags[k] == timer_groups[j] :
      found = True
      index = j

  if found :
    timer_group_sizes[index] += timer_means[k]
    num_groups[index] += 1
  else :
    timer_groups.append(timer_tags[k])
    timer_group_sizes.append(timer_means[k])
    num_groups.append(1)


if scale_timers :
  totaltime = np.sum(timer_means)

  for k in range(len(timer_memos)):
    if timer_means[k]/totaltime < memotol :
      timer_memos[k] = ""

  for k in range(len(timer_group_sizes)) :
    timer_group_sizes[k] = timer_group_sizes[k]/totaltime

  for k in range(len(timer_means)) :
    timer_means[k] = timer_means[k]/totaltime



if chart_type == "donut" :
  piecolors=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]

  outercolors=[]
  for k in range(len(timer_group_sizes)) :
    ii = k%4
    outercolors.append(piecolors[ii](0.6))

  fig, ax = plt.subplots()
  ax.axis('equal')
  mypie, _ = ax.pie(timer_group_sizes, radius=1.0, labels=timer_groups, colors=outercolors)
  plt.setp( mypie, width=0.3, edgecolor='white')


  innercolors=[]
  for k in range(len(timer_group_sizes)) :
    ii = k%4
    shades = np.linspace(0.0,0.6,num_groups[k])
    for j in range(len(shades)) :
      innercolors.append(piecolors[ii](shades[j]))

  mypie2, _ = ax.pie(timer_means, radius=1.0-0.3, labels=timer_memos, labeldistance=0.5, colors=innercolors)
  plt.setp( mypie2, width=0.4, edgecolor='white')
  plt.margins(0,0)
  plt.show()

elif chart_type == "bar" :
  barWidth = 0.7

  #barcolors=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]
  cmap = plt.get_cmap('gist_rainbow')
  barcolors = cmap(np.linspace(0, 1.0, len(timer_groups)))
  colors=[]
  for k in range(len(timer_groups)) :
    #ii = k%4
    #colors.append(barcolors[ii](0.6))
    colors.append(barcolors[k])
  
  prog = 0

  for k in range(len(timer_groups)) :
    bars = []
    labs = []
    errs = []
    x = []

    for j in range(len(timer_means)) :
      if timer_tags[j] == timer_groups[k] :
        labs.append(timer_memos[j])
        bars.append(timer_means[j])
        diff1 = timer_maxs[j] - timer_means[j]
        diff2 = timer_means[j] - timer_mins[j]
        errs.append(max(diff1,diff2))
        x.append(prog+1)
        prog += 1

    if chart_orient == "horizontal" :
      plt.barh(x, bars, height=barWidth, color = colors[k], edgecolor = 'black', capsize=7, label=timer_groups[k])
    else :
      plt.bar(x, bars, width = barWidth, color = colors[k], edgecolor = 'black', yerr=errs, capsize=7, label=timer_groups[k])

  if chart_orient == "horizontal" :
    plt.yticks([r + 2*barWidth for r in range(prog)], timer_memos)
  else :
    plt.xticks([r + 1 + barWidth/2 for r in range(prog)], timer_memos, rotation=90)
    plt.subplots_adjust(bottom= 0.5, top = 0.98)

  plt.ylabel('height')
  plt.legend(fontsize=10)

  plt.rcParams.update({'font.size': 10})
  # Show graphic
  plt.show()

