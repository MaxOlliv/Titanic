# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:11:15 2016

@author: ollivima
"""
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

## the data
N = 5
pourcentage = [75.120,76.077,74.163,74.163,77.990]

## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.50                      # the width of the bars

## the bars


rects= ax.bar(ind, pourcentage, width,
                    color='red')

# axes and labels
ax.set_xlim(-width,len(ind))
ax.set_ylim(50,100)
ax.set_ylabel(ur"$Scores$")
ax.set_title(ur"$Qualité des prévisions par algorithme$")
xTickMarks = [ur"$Régression linéaire$",ur"$Régression logistique$",ur"$Random forest$",ur"$svm$",ur"$K nearest neighbours$"]
ax.set_xticks(ind+(width/2))
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)

plt.text(0,80,'75.120')
plt.text(1,80,'76.077')
plt.text(2,80,'74.163')
plt.text(3,80,'74.163')
plt.text(4,80,'77.990')

plt.show()