#!/usr/bin/env python3

import rospy
import threading
from rockit import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import * 
from uav_frontier_exploration_3d.msg import ClusterInfo
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path, Odometry
from core.msg import PosVel

from uav_path_planning.interpolation_method.scripts.wayPoint_fitting import get_fitted_points
# export PYTHONPATH=/home/sara/alessio/rosbot_ws/src:$PYTHONPATH


# goal_position = np.array([3, 5, 0])

## Initializing global variables
pos_obs_val = np.zeros((1, 3))
r_obs_val = np.zeros(1)
robot_position = np.array([np.nan] * 3)
rospy.loginfo(robot_position)

goal_reached = True  
new_goal_compute = True
waypoints_ready = False
position_ready = False
first_goal = True

waypoints_3d = []
goal_position = []


## Callback waypoints from Rosbot Path
def waypoints_callback(msg):
    global waypoints_3d, waypoints_ready
    waypoints_3d = []
    for pose in msg.poses:
        waypoints_3d.append((pose.pose.position.x, pose.pose.position.y, pose.pose.position.z))

    if waypoints_3d:
        waypoints_ready = True
    # rospy.loginfo("waypoints ok")


## Callback frontier goal
def goal_callback(data):
    global goal_position
    goal_position = []
    goal_position.append((data.pose.position.x, data.pose.position.y, data.pose.position.z))
    rospy.loginfo("goal ok")



## Callback function obstacle info
def obstacle_info(info):
    global pos_obs_val, r_obs_val
    pos_obs_val = np.array([[p.x, p.y, p.z] for p in info.clst_centers])
    r_obs_val = np.array(info.clst_radius)
    # rospy.loginfo("posizione ostacoli ok")

## Callback function robot position info
def robot_position_info(odom):
    global robot_position, position_ready
    robot_x = odom.pose.pose.position.x
    robot_y = odom.pose.pose.position.y
    robot_z = odom.pose.pose.position.z
    robot_position = np.array([robot_x, robot_y, robot_z])
    if not np.all(np.isnan(robot_position)):
        position_ready = True
        # rospy.loginfo("robot ok")
   


## OCP Planner    
def uav_planner(fitted_points, robot_position, goal_position):
    global first_goal

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
    ocp.set_der(x, v*cos(psi)*cos(theta))
    ocp.set_der(y, v*sin(psi)*cos(theta))
    ocp.set_der(z, v*sin(theta))                 
    ocp.set_der(psi, w_psi)
    ocp.set_der(theta, w_theta)


    ## Problem Parameters:
    p = vertcat(x,y,z)                         # robot position
    fitPts = ocp.parameter(3, grid='control')  # fitted points
    pos_obs = ocp.parameter(3,len(r_obs_val))  # obstacles position
    r_obs = ocp.parameter(1,len(r_obs_val))    # radius obstacles


    ## Hard Constraints:
    # Initial constraints
    ocp.subject_to(ocp.at_t0(x)== robot_position[0]) 
    ocp.subject_to(ocp.at_t0(y)== robot_position[1]) 
    ocp.subject_to(ocp.at_t0(z)== robot_position[2])
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
    ocp.add_objective(K_time*ocp.T) 
    # Minimal distance to the other curve fitted: 
    if not no_UGV:
        ocp.add_objective(K_dist*ocp.integral(sumsqr(p[:2,:]-fitPts[:2,:]), grid='control')/ocp.T)  


    ## Pick a solution method
    ocp.solver('ipopt')
    ## Make it concrete for this ocp
    ocp.method(MultipleShooting(N=N,M=4,intg='rk'))  


    ## Set_Initial
    ocp.set_initial(x,0)
    ocp.set_initial(y,ocp.t) # Initial value of the solver equal to the current time
    ocp.set_initial(z, 0)

    ## Give concrete numerical value at parameters
    ocp.set_value(fitPts, fitted_points)
    ocp.set_value(pos_obs, pos_obs_val.T)
    ocp.set_value(r_obs, r_obs_val)

    ## Solve
    sol = ocp.solve()

    first_goal = False
    return sol, x, y, z, psi, w_psi


##Publisher Path (Rviz)
def publish_ocp_path(sol, x, y, z, psi, w_psi):

    #Create Path message
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = frame

    # OCP trajectory points
    ts, xs = sol.sample(x, grid='control')
    ts, ys = sol.sample(y, grid='control')
    ts, zs = sol.sample(z, grid='control')  

    # Velocity points
    vxs = np.diff(xs) / np.diff(ts)
    vys = np.diff(ys) / np.diff(ts)
    vzs = np.diff(zs) / np.diff(ts)


    for i in range(len(ts)):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame
        pose.pose.position.x = xs[i]
        pose.pose.position.y = ys[i]
        pose.pose.position.z = zs[i]
        path_msg.poses.append(pose)

    for i in range(len(vxs)):
        posVel = PosVel()
        posVel.pose_frame_id = frame
        posVel.position.x = xs[i]
        posVel.position.y = ys[i]
        posVel.position.z = zs[i]
        posVel.yaw = 0              # ????
        posVel.velocity.x = vxs[i]
        posVel.velocity.y = vys[i]
        posVel.velocity.z = vzs[i]
        posVel.yaw_rate = 0         # ????

    posVel_pub.publish(PosVel)
    path_ocp_pub.publish(path_msg)
   


if __name__ == "__main__":
    rospy.init_node('uav_path_planning')
    rospy.loginfo("Node uav_path_planning started")

    ## Parameters
    frame = rospy.get_param('global_frame', 'map')
    x_min = rospy.get_param('x_min', -50)
    x_max = rospy.get_param('x_max', 50)
    y_min = rospy.get_param('y_min', -50)
    y_max = rospy.get_param('y_max', 50)
    z_min = rospy.get_param('z_min', 0.5)
    z_max = rospy.get_param('z_max', 5)
    r_uav = rospy.get_param('r_uav', 1)
    theta_min = rospy.get_param('theta_min', -0.5)
    theta_max = rospy.get_param('theta_max', 0.5)
    v_min = rospy.get_param('v_min', 0)
    v_max = rospy.get_param('v_max', 2)
    w_psi_min = rospy.get_param('w_psi_min', -1)
    w_psi_max = rospy.get_param('w_psi_max', 1)
    w_theta_min = rospy.get_param('w_theta_min', -2)
    w_theta_max = rospy.get_param('w_theta_max', 2)
    N = rospy.get_param('N', 40)
    K_time = rospy.get_param('K_time', 1)
    K_dist = rospy.get_param('K_dist', 1)
    dist_to_goal = rospy.get_param('dist_to_goal', 1)
    no_UGV = rospy.get_param('no_pathUGV', False)


    
    obs_info = rospy.Subscriber('obstacle_info', ClusterInfo, obstacle_info)
    position_robot = rospy.Subscriber('odometry_drone', Odometry, robot_position_info)
    wp_sub = rospy.Subscriber('wp_rosbot', Path, waypoints_callback)
    goal_sub = rospy.Subscriber('frontier_goal', PoseStamped, goal_callback)

    path_ocp_pub = rospy.Publisher("OCP_path_planning", Path, queue_size=10)
    posVel_pub = rospy.Publisher("setpoint_raw", PosVel, queue_size=10)


    rospy.loginfo(no_UGV)
    rate = rospy.Rate(10)
    # while np.array_equal(robot_position, np.array([np.nan] * 3)):
    #     rate.sleep()
    while  not position_ready:
        rate.sleep()

    # while not rospy.is_shutdown():                          
    if not no_UGV:
        while not waypoints_ready:
            rate.sleep()


    while not rospy.is_shutdown():
        if first_goal:
            goal_position = robot_position

        if np.linalg.norm(robot_position - goal_position) < dist_to_goal:  
            goal_reached = True
            new_goal_compute = True
            rospy.loginfo("condizione goal")

        if not goal_reached:
            rate.sleep()
            continue

        if new_goal_compute:
            new_goal_compute = False

            if not np.all(np.isnan(robot_position)):
                rospy.loginfo("Computing new goal")
                if not no_UGV:
                    fitted_points = get_fitted_points(waypoints_3d)
                sol, x, y, z, psi, w_psi = uav_planner(fitted_points, robot_position, goal_position)
                publish_ocp_path(sol, x, y, z, psi, w_psi)
                rospy.loginfo("Processed new robot position")
                goal_reached = False

        rate.sleep()
