#!/usr/bin/env python3

import rospy
import numpy as np

from geomdl import fitting
from geomdl.visualization import VisMPL as vis 
import matplotlib.pyplot as plt


# waypoints_3d = [(-4, -3, -1), (-2, 3, 1), (2, 1, 1), (5, 0, 2), (7, 6, 2), (9, 7, 3)]
# waypoints_3d = [(-4, -3, -1), (-2, 3, 1), (2, 3, 1), (5, 2, 1), (7, 2, 2), (9, 2, 2)]
waypoints_3d = [(-4, -3, 0), (-2, 3, 0), (2, 1, 0), (5, 0, 0), (7, 6, 0), (9, 7, 0)]   # z=0
degree = 3

# Fitting Curve 
curve_fit = fitting.interpolate_curve(waypoints_3d, degree)


# Definition of sampling instants
n_samples = 40 
final_time = 10
sample_times = np.linspace(0, 1, n_samples)
# delta = 0.1


# sampling fitting curve
# sample_points = curve_fit.evalpts
sample_points = [curve_fit.evaluate_single(t) for t in sample_times]  
fitted_points = np.transpose(np.array(sample_points))

print("Punti campionati sulla curva fittata:")
print(fitted_points)
# print(np.shape(fitted_points))


# Plot Curve_fit
vis_config = vis.VisConfig(ctrlpts=True, legend=True)
curve_comp = vis.VisCurve2D(vis_config)

curve_fit.vis = curve_comp # aggiugo il componente di visualizzazione della curva
curve_fit.render() # rende visibile la curva

# plt.plot(fitted_points[:,0],fitted_points[:,1], marker='o', color='r')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(fitted_points[0, :], fitted_points[1, :], fitted_points[2, :], marker='o', color='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

