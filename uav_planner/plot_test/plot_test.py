#!/usr/bin/env python3

import rospy
from rockit import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import * 
from geomdl import fitting
from geomdl.visualization import VisMPL as vis 
import time



# from interpolation_method.scripts.wayPoint_fitting import get_fitted_points

# import sys
# import os
# import rospkg 
# rospack = rospkg.RosPack()
# package_path = rospack.get_path('uav_path_planning')
# sys.path.append(package_path)
# from interpolation_method.scripts.wayPoint_fitting import get_fitted_points

# current_dir = os.path.dirname(__file__)
# package_path = os.path.abspath(os.path.join(current_dir, '../../interpolation_method/scripts'))
# sys.path.append(package_path)
# from interpolation_method.scripts.wayPoint_fitting import get_fitted_points


# interpolation_method_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../interpolation_method/scripts'))
# if interpolation_method_path not in sys.path:
#     sys.path.append(interpolation_method_path)
# import wayPoint_fitting

# from uav_path_planning.interpolation_method.scripts.wayPoint_fitting import get_fitted_points
# export PYTHONPATH=/home/sara/alessio/rosbot_ws/src:$PYTHONPATH

## Parameters
x_min = -50
x_max = 50
y_min =  -50
y_max =  50
z_min =  0.5
z_max =  5
r_uav =  1
theta_min = -0.5
theta_max =  0.5
v_min =  0
v_max =  2
w_psi_min = -1
w_psi_max = 1
w_theta_min =  -2
w_theta_max =  2
N = 40
K_time = 1
K_dist = 10
degree = 3
n_samples = N  # take the same value of N 


## Posizione Ostacoli (per visualizzazione plot)
p0_value = np.array([0, 0, 2])
r0_val = 1
p1_value = np.array([5, 0, 1])
r1_val = 0.5
pos_obs_val = np.array([p0_value, p1_value])
r_obs_val = np.array([r0_val, r1_val])


# goal_position = np.array([9.5, 7, 2])
goal_position = np.array([8, 1, 2])


waypoints_3d = [(-4, -3, 0), (-2, 3, 0), (2, 1, 0), (5, 0, 0), (7, 6, 0), (9, 7, 0)]   # z=0



def uav_planner(fitted_points, goal_position):

    t1= time.time()

    # Initialization OCP
    ocp = Ocp(T=FreeTime(10.0))  

    ## State Variables
    x = ocp.state()
    y = ocp.state()
    z = ocp.state()
    psi = ocp.state()     # yaw angle
    theta = ocp.state()   # pitch angle

    w_psi = ocp.control()   # yaw_dot
    w_theta = ocp.control() # pitch_dot
    v = ocp.control()

    ## Space-State form
    ocp.set_der(x, v*sin(psi)*cos(theta))
    ocp.set_der(y, v*cos(psi)*cos(theta))
    ocp.set_der(z, v*sin(theta))                 
    ocp.set_der(psi, w_psi)
    ocp.set_der(theta, w_theta)


    ## Problem Parameters:
    # p0 = ocp.parameter(3)
    # r0 = ocp.parameter()
    p = vertcat(x,y,z)                         # robot position
    fitPts = ocp.parameter(3, grid='control')  # fitted points
    pos_obs = ocp.parameter(3,len(r_obs_val))  # obstacles position
    r_obs = ocp.parameter(1,len(r_obs_val))    # radius obstacles

    ## Hard Constraints:
    # Initial constraints
    ocp.subject_to(ocp.at_t0(x)== 1) #-5 
    ocp.subject_to(ocp.at_t0(y)== -8) #-8
    ocp.subject_to(ocp.at_t0(z)== 1)
    # Final constraint
    ocp.subject_to(ocp.at_tf(x)== goal_position[0]) 
    ocp.subject_to(ocp.at_tf(y)== goal_position[1]) 
    ocp.subject_to(ocp.at_tf(z)== goal_position[2])
    # Model constraint
    ocp.subject_to(theta_min <= (theta <= theta_max))  # pitch constraint
    # State Limits
    ocp.subject_to(x_min <= (x <= x_max))
    ocp.subject_to(y_min <= (y <= y_max))
    ocp.subject_to(z_min <= (z <= z_max))
    # Input Limits
    ocp.subject_to(v_min <= (v <= v_max))
    ocp.subject_to(w_psi_min <= (w_psi <= w_psi_max))        # w_yaw
    ocp.subject_to(w_theta_min <= (w_theta <= w_theta_max))  # w_pitch
    # Obstacle Avoidance
    for idx_obj in range(len(r_obs_val)):
        # ocp.subject_to(sumsqr(p-p0)>=((r0+r_uav)**2))  # per evitare l'ostacolo
        ocp.subject_to(sumsqr(p-pos_obs[:,idx_obj])>=((r_obs[:,idx_obj]+r_uav)**2))  


    ## Soft Consraints: Objective Function:
    # Minimal time
    ocp.add_objective(K_time*ocp.T) # minimizza tempo finale (T)
    # Minimal distance to the other curve fitted: 
    ocp.add_objective(K_dist*ocp.integral(sumsqr(p[:2,:]-fitPts[:2,:]), grid='control')/ocp.T)  

    ## Set_Initial
    ocp.set_initial(x,0)
    ocp.set_initial(y,ocp.t) # gli passo come valore iniziale il tempo corrente (t)
    ocp.set_initial(z, 0)


    ## Pick a solution method
    ocp.solver('ipopt')
    ## Make it concrete for this ocp
    ocp.method(MultipleShooting(N=N,M=4,intg='rk'))  # N--> deve essere lo stesso numero con cui si è campionata la curva fittata

    ## Give concrete numerical value at parameters
    # ocp.set_value(p0, p0_value)
    # ocp.set_value(r0, r0_val)
    ocp.set_value(fitPts, fitted_points)
    ocp.set_value(pos_obs, pos_obs_val.T)
    ocp.set_value(r_obs, r_obs_val.T)

    ## Solve
    sol = ocp.solve()

    t2= time.time()
    tempoEsec = t2-t1
    rospy.loginfo("Tempo di esecuzione: %f secondi" % tempoEsec)
    return sol, x, y, z



## OCP con modello controllato in ACCELERAZIONE
def uav_planner_pointMass(fitted_points, goal_position):

    t1= time.time()

    # Initialization OCP
    ocp = Ocp(T=FreeTime(10.0))  

    ## State Variables
    x1 = ocp.state()
    y1 = ocp.state()
    z1 = ocp.state()
    x1_dot = ocp.state()
    y1_dot = ocp.state()
    z1_dot = ocp.state()
    
    x1_ddot = ocp.control()  
    y1_ddot = ocp.control() 
    z1_ddot = ocp.control()

    ## Space-State form
    ocp.set_der(x1, x1_dot)
    ocp.set_der(y1, y1_dot)
    ocp.set_der(z1, z1_dot)                 
    ocp.set_der(x1_dot, x1_ddot)
    ocp.set_der(y1_dot, y1_ddot)
    ocp.set_der(z1_dot, z1_ddot)



    ## Problem Parameters:
    # p0 = ocp.parameter(3)
    # r0 = ocp.parameter()
    p1 = vertcat(x1,y1,z1)                         # robot position
    fitPts = ocp.parameter(3, grid='control')  # fitted points
    pos_obs = ocp.parameter(3,len(r_obs_val))  # obstacles position
    r_obs = ocp.parameter(1,len(r_obs_val))    # radius obstacles

    ## Hard Constraints:
    # Initial constraints
    ocp.subject_to(ocp.at_t0(x1)== 1) #-5 
    ocp.subject_to(ocp.at_t0(y1)== -8) #-8
    ocp.subject_to(ocp.at_t0(z1)== 1)
    # Final constraint
    ocp.subject_to(ocp.at_tf(x1)== goal_position[0]) 
    ocp.subject_to(ocp.at_tf(y1)== goal_position[1]) 
    ocp.subject_to(ocp.at_tf(z1)== goal_position[2])
    # State Limits
    ocp.subject_to(x_min <= (x1 <= x_max))
    ocp.subject_to(y_min <= (y1 <= y_max))
    ocp.subject_to(z_min <= (z1 <= z_max))
    ocp.subject_to(v_min <= (x1_dot <= v_max))
    ocp.subject_to(v_min <= (y1_dot <= v_max))
    ocp.subject_to(v_min <= (z1_dot <= v_max))
    # Input Limits
    ocp.subject_to(v_min <= (x1_ddot <= v_max))
    ocp.subject_to(v_min <= (y1_ddot <= v_max))        
    ocp.subject_to(v_min <= (z1_ddot <= v_max))  
    # Obstacle Avoidance
    for idx_obj in range(len(r_obs_val)):
        # ocp.subject_to(sumsqr(p-p0)>=((r0+r_uav)**2))  # per evitare l'ostacolo
        ocp.subject_to(sumsqr(p1-pos_obs[:,idx_obj])>=((r_obs[:,idx_obj]+r_uav)**2))  


    ## Soft Consraints: Objective Function:
    # Minimal time
    ocp.add_objective(K_time*ocp.T) # minimizza tempo finale (T)
    # Minimal distance to the other curve fitted: 
    ocp.add_objective(K_dist*ocp.integral(sumsqr(p1[:2,:]-fitPts[:2,:]), grid='control')/ocp.T)  

    ## Set_Initial
    ocp.set_initial(x1,0)
    ocp.set_initial(y1,ocp.t) # gli passo come valore iniziale il tempo corrente (t)
    ocp.set_initial(z1, 0)


    ## Pick a solution method
    ocp.solver('ipopt')
    ## Make it concrete for this ocp
    ocp.method(MultipleShooting(N=N,M=4,intg='rk'))  # N--> deve essere lo stesso numero con cui si è campionata la curva fittata

    ## Give concrete numerical value at parameters
    # ocp.set_value(p0, p0_value)
    # ocp.set_value(r0, r0_val)
    ocp.set_value(fitPts, fitted_points)
    ocp.set_value(pos_obs, pos_obs_val.T)
    ocp.set_value(r_obs, r_obs_val.T)

    ## Solve
    sol1 = ocp.solve()

    t2= time.time()
    tempoEsec_modelMass = t2-t1
    rospy.loginfo("Tempo di esecuzione: %f secondi" % tempoEsec_modelMass)
    return sol1, x1, y1, z1


## Plot OCP_Path in matplotlibs
def plot_planning(sol,x ,y, z, sol1,x1,y1,z1, fitted_points):

    fig = plt.figure()
    plot = fig.add_subplot(projection='3d',aspect='equal')
    plot.set_xlabel('X-Position (m)')
    plot.set_ylabel('Y-Position (m)')
    plot.set_zlabel('Z-Position (m)')

    plot.set_zlim(-2, 6)
    plot.set_zticks(range(-2, 7, 2))

    # Sample data from sol and variables x, y, z
    ts, xs = sol.sample(x, grid='control')
    ts, ys = sol.sample(y, grid='control')
    ts, zs = sol.sample(z, grid='control')  
    plot.scatter(xs, ys, zs, c='b', marker='o')

    ts, xs = sol.sample(x, grid='integrator', refine=10)
    ts, ys = sol.sample(y, grid='integrator', refine=10)
    ts, zs = sol.sample(z, grid='integrator', refine=10)  
    plot.plot(xs, ys, zs, '-',label="UAV Trajectory")

    ## Plot a CIRCLE in 3D space
    ts = np.linspace(0, 2 * np.pi, 1000)
    # xs = p0_value[0] + r0_val * np.cos(ts)
    # ys = p0_value[1] + r0_val * np.sin(ts)
    # zs = np.zeros_like(ts) 
    # plot.plot(xs, ys, zs, 'r-')

    ## Plot a SPHERE
    theta = np.linspace(0, np.pi, 100) # angolo polar -> varia da 0 a pi, corrisponde all'angolo verticale dal polo nord al polo sud
    phi = np.linspace(0, 2*np.pi, 100) # angolo azimut -> varia da 0 a 2pi, corrisponde all'angolo orizzontale intorno all'asse z
    theta, phi = np. meshgrid(theta, phi)
    # x_sphere = p0_value[0] + r0_val * np.cos(theta) * np.sin(phi)
    # y_sphere = p0_value[1] + r0_val * np.sin(theta) * np.sin(phi)
    # z_sphere = p0_value[2] + r0_val * np.cos(phi)
    for idx_plt in range(len(r_obs_val)):
        x_sphere = pos_obs_val[idx_plt,0] + r_obs_val[idx_plt] * np.cos(theta) * np.sin(phi)
        y_sphere = pos_obs_val[idx_plt,1] + r_obs_val[idx_plt] * np.sin(theta) * np.sin(phi)
        z_sphere = pos_obs_val[idx_plt,2] + r_obs_val[idx_plt] * np.cos(phi)

        plot.plot_surface(x_sphere, y_sphere, z_sphere, color='r', alpha=1)

    plot.plot(fitted_points[0,:],fitted_points[1,:],fitted_points[2,:], marker='^', color='g', label="UGV trajectory")

    # Set axis to be equal
    # plot.set_box_aspect([1,1,1]) 
    plot.set_box_aspect([ub - lb for lb, ub in (getattr(plot, f'get_{a}lim')() for a in 'xyz')])

    plot.set_title('Dubins Model', fontsize=16, fontweight='bold')    
    plot.legend()
    # Show the plot
    plt.show(block=False)  # block=True --> mostra il plot e il processo principale del programma viene bloccato finchè non si chiude la finestra plot (il programma è in attesa)

    ##### Plot Traj modelPointMass
    fig1 = plt.figure()
    plot1 = fig1.add_subplot(projection='3d',aspect='equal')
    plot1.set_xlabel('X-Position (m)')
    plot1.set_ylabel('Y-Position (m)')
    plot1.set_zlabel('Z-Position (m)')

    plot1.set_zlim(-2, 6)
    plot1.set_zticks(range(-2, 7, 2))

    # Sample data from sol and variables x, y, z
    ts1, xs1 = sol1.sample(x1, grid='control')
    ts1, ys1 = sol1.sample(y1, grid='control')
    ts1, zs1 = sol1.sample(z1, grid='control')  
    plot1.scatter(xs1, ys1, zs1, c='b', marker='o')

    ts1, xs1 = sol1.sample(x1, grid='integrator', refine=10)
    ts1, ys1 = sol1.sample(y1, grid='integrator', refine=10)
    ts1, zs1 = sol1.sample(z1, grid='integrator', refine=10)  
    plot1.plot(xs1, ys1, zs1, '-',label="UAV Trajectory")

    ## Plot a CIRCLE in 3D space
    ts1 = np.linspace(0, 2 * np.pi, 1000)

    ## Plot a SPHERE
    theta = np.linspace(0, np.pi, 100) # angolo polar -> varia da 0 a pi, corrisponde all'angolo verticale dal polo nord al polo sud
    phi = np.linspace(0, 2*np.pi, 100) # angolo azimut -> varia da 0 a 2pi, corrisponde all'angolo orizzontale intorno all'asse z
    theta, phi = np. meshgrid(theta, phi)
    for idx_plt in range(len(r_obs_val)):
        x_sphere = pos_obs_val[idx_plt,0] + r_obs_val[idx_plt] * np.cos(theta) * np.sin(phi)
        y_sphere = pos_obs_val[idx_plt,1] + r_obs_val[idx_plt] * np.sin(theta) * np.sin(phi)
        z_sphere = pos_obs_val[idx_plt,2] + r_obs_val[idx_plt] * np.cos(phi)

        plot1.plot_surface(x_sphere, y_sphere, z_sphere, color='r', alpha=1)

    plot1.plot(fitted_points[0,:],fitted_points[1,:],fitted_points[2,:], marker='^', color='g', label="UGV trajectory")

    # Set axis to be equal
    plot1.set_box_aspect([ub - lb for lb, ub in (getattr(plot1, f'get_{a}lim')() for a in 'xyz')])

    plot1.set_title('Point-Mass Model', fontsize=16, fontweight='bold')    
    plot1.legend()
    # Show the plot
    plt.show(block=False)



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
    rospy.init_node('uav_path_planning')

    fitted_points = get_fitted_points()
    sol, x, y, z = uav_planner(fitted_points, goal_position)
    sol1, x1, y1, z1 = uav_planner_pointMass(fitted_points, goal_position)
    plot_planning(sol, x, y, z, sol1, x1, y1, z1, fitted_points)
    plot_curve_fit(fitted_points, curve_fit)

