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
n_samples = rospy.get_param('/N', 40)  # take the same value of N in ocp_planner.py


fitted_points = None
curve_fit = None

def get_fitted_points():
    global fitted_points, curve_fit

    ## Fitting Curve 
    curve_fit = fitting.interpolate_curve(waypoints_3d, degree)

    ## Definition of sampling instants
    sample_times = np.linspace(0, 1, n_samples)
    # delta = 0.1

    ## Sampling fitting curve
    # sample_points = curve_fit.evalpts
    sample_points = [curve_fit.evaluate_single(t) for t in sample_times]  
    fitted_points = np.transpose(np.array(sample_points))

    # rospy.loginfo("Points sampled on the fitting curve:")
    # rospy.loginfo(fitted_points)

    return fitted_points


## Plot curve fitting in matplotlibs
def plot_curve_fit(fitted_points, curve_fit):

    # Plot Curve_fit
    vis_config = vis.VisConfig(ctrlpts=True, legend=True)
    curve_comp = vis.VisCurve2D(vis_config)

    curve_fit.vis = curve_comp # add curve display component
    curve_fit.render()         # makes the curve visible 

    # plt.plot(fitted_points[:,0],fitted_points[:,1], marker='o', color='r')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fitted_points[0, :], fitted_points[1, :], fitted_points[2, :], marker='o', color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == "__main__":
    rospy.init_node('fitting_node')

    fitted_points = get_fitted_points()

    plot_fit = plot_curve_fit(fitted_points, curve_fit)

    rospy.spin()


