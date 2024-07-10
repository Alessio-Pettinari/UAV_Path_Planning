#!/usr/bin/env python3

import rospy
from rockit import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import * 
from uav_frontier_exploration_3d.msg import ClusterInfo
from geometry_msgs.msg import Point

import sys
import os
interpolation_method_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../interpolation_method/scripts'))
if interpolation_method_path not in sys.path:
    sys.path.append(interpolation_method_path)
import wayPoint_fitting




# Initialization OCP
ocp = Ocp(T=FreeTime(10.0))  # FreeTime()--> dichiara il tempo finale (pari a 10) inizialmente, ma poi sarà soggetto all'ottimizzazione stessa (cioè il tempo finale verrà det come parte della soluzione dell'ottimizzazione)

# Parameters
N = 40 # num of sample
x_min = -50
x_max = 50
y_min = -50
y_max = 50
z_min = 0.5
z_max = 5

theta_min = -0.5
theta_max = 0.5
v_min = 0
v_max = 2
w_psi_min = -1
w_psi_max = 1
w_theta_min = -2
w_theta_max = 2

# Cost Function Parameters
K_time = 1
K_dist = 10

#Obstacle
r_uav = 1
p0_value = np.array([0, 0, 2])
r0_val = 1
p1_value = np.array([5, 0, 1])
r1_val = 0.5


pos_obs_val = np.array([p0_value, p1_value])
print(pos_obs_val)
r_obs_val = np.array([r0_val, r1_val])


# State Variables
x = ocp.state()
y = ocp.state()
z = ocp.state()
psi = ocp.state()     # yaw angle
theta = ocp.state()   # pitch angle

w_psi = ocp.control()   # yaw_dot
w_theta = ocp.control() # pitch_dot
v = ocp.control()

# Space-State form
ocp.set_der(x, v*cos(psi)*cos(theta))
ocp.set_der(y, v*sin(psi)*cos(theta))
ocp.set_der(z, v*sin(theta))                 
ocp.set_der(psi, w_psi)
ocp.set_der(theta, w_theta)


# Problem Parameters
# p0 = ocp.parameter(3)
# r0 = ocp.parameter()
p = vertcat(x,y,z)
fitPts = ocp.parameter(3, grid='control')

pos_obs = ocp.parameter(3,len(r_obs_val)) 
r_obs = ocp.parameter(1,len(r_obs_val))


# Initial constraints
ocp.subject_to(ocp.at_t0(x)== -5) #-2 
ocp.subject_to(ocp.at_t0(y)== -8) #-2
ocp.subject_to(ocp.at_t0(z)== 1)

# Final constraint
ocp.subject_to(ocp.at_tf(x)== 8) #5
ocp.subject_to(ocp.at_tf(y)== -3) #8
ocp.subject_to(ocp.at_tf(z)== 2)

# Model constraint
ocp.subject_to(theta_min <= (theta <= theta_max))  # pitch constraint


## set_initial--> viene utilizzato per fornire un valore iniziale suggerito per le variabili di stato e di controllo.
##                Questo non è un vincolo rigido come subject_to, ma piuttosto un punto di partenza per l'ottimizzatore.
##                L'ottimizzatore può decidere di partire da questo valore iniziale per cercare la soluzione ottimale,
##                ma non è obbligato a rispettarlo se trova un percorso migliore.
ocp.set_initial(x,0)
ocp.set_initial(y,ocp.t) # gli passo come valore iniziale il tempo corrente (t)
ocp.set_initial(z, 0)


## subject_to        --> aggiungo vincoli al prob di ottimizzazione, cioè cond che devono essere soddisfatte per considerare la soluz valida. Possono includere limiti su var di stato e di controllo, eq diff, ecc..
## (hard constraints)    
ocp.subject_to(x_min <= (x <= x_max))
ocp.subject_to(y_min <= (y <= y_max))
ocp.subject_to(z_min <= (z <= z_max))

ocp.subject_to(v_min <= (v <= v_max))
ocp.subject_to(w_psi_min <= (w_psi <= w_psi_max))        # w_yaw
ocp.subject_to(w_theta_min <= (w_theta <= w_theta_max))  # w_pitch


for idx_obj in range(len(r_obs_val)):
    # ocp.subject_to(sumsqr(p-p0)>=((r0+r_uav)**2))  # per evitare l'ostacolo
    ocp.subject_to(sumsqr(p-pos_obs[:,idx_obj])>=((r_obs[:,idx_obj]+r_uav)**2))  # per evitare l'ostacolo




## add_objective     --> utilizzato per aggiungere funzioni obiettivo al prob di ottimizzazione, cioè quelle che si vuole 
## (soft constraints)    ottimizzare (minimizzare o massimizzare). L'ottimizzatore cercherà la soluzione che ottimizzi queste funzione.
##                      "ATTENZIONE"--> di default si MINIMIZZA l'obiettivo. Se si vuole MASSIMIZZARE si deve trasformare in un prob equivalente

# Objective Function:
# Minimal time
ocp.add_objective(K_time*ocp.T) # minimizza tempo finale (T)

# Minimal distance to the other curve fitted: 
ocp.add_objective(K_dist*ocp.integral(sumsqr(p[:2,:]-fitPts[:2,:]), grid='control')/ocp.T)  # minimizzo la distanza nel piano x-y (altrimenti il drone cercherebbe di avvicinarsi il più possibile al terreno)


# Pick a solution method
# options = {"max_iter": 5000}
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N,M=4,intg='rk'))  # N--> deve essere lo stesso numero con cui si è campionata la curva fittata


# Give concrete numerical value at parameters

# ocp.set_value(p0, p0_value)
# ocp.set_value(r0, r0_val)
ocp.set_value(fitPts, wayPoint_fitting.fitted_points)

ocp.set_value(pos_obs, pos_obs_val.T)
ocp.set_value(r_obs, r_obs_val.T)


# Solve
sol = ocp.solve()


# PLOT
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

# Sample integrator data
# ts, xs = sol.sample(x, grid='integrator')
# ts, ys = sol.sample(y, grid='integrator')
# ts, zs = sol.sample(z, grid='integrator')  
# plot.scatter(xs, ys, zs, c='b', marker='.')

ts, xs = sol.sample(x, grid='integrator', refine=10)
ts, ys = sol.sample(y, grid='integrator', refine=10)
ts, zs = sol.sample(z, grid='integrator', refine=10)  
plot.plot(xs, ys, zs, '-')

# Plot a CIRCLE in 3D space
ts = np.linspace(0, 2 * np.pi, 1000)
# xs = p0_value[0] + r0_val * np.cos(ts)
# ys = p0_value[1] + r0_val * np.sin(ts)
# zs = np.zeros_like(ts) 
# plot.plot(xs, ys, zs, 'r-')

# Plot a SPHERE
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

plot.plot(wayPoint_fitting.fitted_points[0,:],wayPoint_fitting.fitted_points[1,:],wayPoint_fitting.fitted_points[2,:], marker='^', color='g')


# Set axis to be equal
# plot.set_box_aspect([1,1,1]) 
plot.set_box_aspect([ub - lb for lb, ub in (getattr(plot, f'get_{a}lim')() for a in 'xyz')])


# Show the plot
plt.show(block=True)  # block=True --> mostra il plot e il processo principale del programma viene bloccato finchè non si chiude la finestra plot (il programma è in attesa)




