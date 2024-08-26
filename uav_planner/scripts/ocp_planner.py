#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs
from rockit import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import * 
from uav_frontier_exploration_3d.msg import ClusterInfo, FreeMeshInfo
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

goal_reached = True  
new_goal_compute = True
waypoints_ready = False
position_ready = False
first_goal = True
goal_update = False

waypoints_3d = []
goal_position = []
coeff_matrix = []
const_matrix = []


## Transform function
def transform_point(point, from_frame, to_frame):
    point_stamped = tf2_geometry_msgs.PointStamped()
    point_stamped.header.stamp = rospy.Time.now()
    point_stamped.header.frame_id = from_frame
    point_stamped.point = point

    try:
        trasformed_point = tf_buffer.transform(point_stamped, to_frame, timeout=rospy.Duration(1.0))
        return trasformed_point.point
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Transformation failed!: {e}")
        return None



## Callback waypoints from Rosbot Path
def waypoints_callback(msg):
    global waypoints_3d, waypoints_ready
    waypoints_3d = []
    for pose in msg.poses:
        waypoints_3d.append((pose.pose.position.x, pose.pose.position.y, pose.pose.position.z))
    if waypoints_3d:
        waypoints_ready = True


## Callback frontier goal
def goal_callback(data):
    global goal_position, goal_update
    # goal_position = []
    # goal_position = (data.pose.position.x, data.pose.position.y, data.pose.position.z)

    # Odom2Map
    goal_point = Point(x=data.pose.position.x, y=data.pose.position.y, z=data.pose.position.z)
    goal_point_map = transform_point(goal_point, "odom", "map")
    if goal_point_map is not None:
        goal_mapFrame = [goal_point_map.x, goal_point_map.y, goal_point_map.z]
    else:
        rospy.logwarn("Failed to transform goal_point in map frame!")
    goal_position = goal_mapFrame
    goal_update = True
    rospy.loginfo("Goal Position obtained!")



## Callback function obstacle info
def obstacle_info(info):
    global pos_obs_val, r_obs_val
    pos_obs_val = np.array([[p.x, p.y, p.z] for p in info.clst_centers])
    r_obs_val = np.array(info.clst_radius)

    # Odom2Map
    obst_mapFrame = []
    for pos in pos_obs_val:
        obst_point = Point(x=pos[0], y=pos[1], z=pos[2])
        obst_point_map = transform_point(obst_point, "odom", "map") # Transform obstacles from "odom" to "map"
        if obst_point_map is not None:
            obst_mapFrame.append([obst_point_map.x, obst_point_map.y, obst_point_map.z])
        else:
            rospy.logwarn("Failed to transform obstacle_point in map frame!")
    
    pos_obs_val = np.array(obst_mapFrame)
    # rospy.loginfo("Obstacle Position obtained!")


## Callback function free space
def freeSpace_callback(free):
    global coeff_matrix, const_matrix

    coeff_msg = np.array(free.coeff)
    const_msg = np.array(free.const)

    # Num hyperplane
    num_planes = len(const_msg)
    num_coeff_per_plane = len(coeff_msg) // num_planes

    coeff_matrix = coeff_msg.reshape((num_planes, num_coeff_per_plane)) # [nx3]
    const_matrix = const_msg

    # Odom2Map
    coeff_matrix_map = np.zeros_like(coeff_matrix)
    for i, coeff in enumerate(coeff_matrix):
        coeff_point = Point(x=coeff[0], y=coeff[1], z=coeff[2])
        coeff_point_map = transform_point(coeff_point, "odom", "map")
        if coeff_point_map is not None:
            coeff_matrix_map[i] = [coeff_point_map.x, coeff_point_map.y, coeff_point_map.z]
        else:
            rospy.logwarn("Failed to transform coeff_freeEquation in map frame!")
    coeff_matrix = coeff_matrix_map


    # const_matrix_map = []
    # for const in const_matrix:
    #     const_point = Point(x=const, y=0, z=0)
    #     const_point_map = transform_point(const_point, "odom", "map")
    #     if const_point_map is not None:
    #         const_matrix_map.append(const_point_map)
    #     else:
    #         rospy.logwarn("Failed to transform coeff_freeEquation in map frame!")
    # const_matrix = const_matrix_map
    const_matrix_map = np.zeros(len(const_matrix))
    for i, const in enumerate(const_matrix):
        const_point = Point(x=const, y=0, z=0)
        const_point_map = transform_point(const_point, "odom", "map")
        if const_point_map is not None:
            const_matrix_map[i] = const_point_map.x
        else:
            rospy.logwarn("Failed to transform coeff_freeEquation in map frame!")
    const_matrix = const_matrix_map
  
  

## Callback function robot position info
def robot_position_info(odom):
    global robot_position, position_ready
    robot_x = odom.pose.pose.position.x
    robot_y = odom.pose.pose.position.y
    robot_z = odom.pose.pose.position.z
    robot_yaw = odom.twist.twist.angular.x
    robot_pitch = odom.twist.twist.angular.y
    # robot_position = np.array([robot_x, robot_y, robot_z])

    # Odom2Map
    robot_point = Point(x=robot_x, y=robot_y, z=robot_z)
    robot_orient_point = Point(yaw=robot_yaw, pitch=robot_pitch)
    robot_point_map = transform_point(robot_point, "odom", "map")  # Transform robot_position from "odom" to "map"
    if robot_point_map is not None:
        robot_mapFrame = np.array([robot_point_map.x, robot_point_map.y, robot_point_map.z])
    else:
        rospy.logwarn("Failed to transform robot_position in map frame!")
    robot_position = robot_mapFrame

    # rospy.loginfo(f"Updated robot position: {robot_position}")
    if not np.all(np.isnan(robot_position)):
        position_ready = True
        # rospy.loginfo("Position obtained!")

   
 

## OCP Planner    
def uav_planner(robot_position, goal_position, fitted_points='none'):
    global first_goal, pos_obs_val, r_obs_val, coeff_matrix, const_matrix

    rospy.loginfo(f"goal: {goal_position}")
    rospy.loginfo(f"robot: {robot_position}")
    t0 = rospy.Time.now()

    initial_success = False
    max_attempts = 5
    attempt = 0
    while not initial_success and attempt < max_attempts:
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
        p = vertcat(x,y,z)                             # robot position
        pos_obs = ocp.parameter(3,len(r_obs_val))      # obstacles position
        r_obs = ocp.parameter(1,len(r_obs_val))        # radius obstacles
        if not no_UGV:
            fitPts = ocp.parameter(3, grid='control')  # fitted points


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
            ocp.subject_to(sumsqr(p-pos_obs[:,idx_obj])>=((r_obs[:,idx_obj]+r_uav)**2))  
        
        # constraint_free = []
        # Solution inside Free_Space Octomap:
        # for idx_plane in range(coeff_matrix.shape[0]):
        #     hyperplane_funct = mtimes(coeff_matrix[idx_plane,:].reshape(1, -1), p) + const_matrix[idx_plane] # Hyperplane function [ a*x + b <= 0]
        #     # constraint_free.append(hyperplane_funct <= 0)     
        #     ocp.subject_to(hyperplane_funct <= 0)


        ## Soft Consraints: Objective Function:
        # Minimal time
        ocp.add_objective(K_time*ocp.T) 
        # Minimal distance to the other curve fitted: 
        if not no_UGV:
            ocp.add_objective(K_dist*ocp.integral(sumsqr(p[:2,:]-fitPts[:2,:])/ocp.T, grid='control'))  


        ## Pick a solution method
        ocp.solver(
                "ipopt",
                {
                    "expand": True,
                    "verbose": False,
                    "print_time": True,
                    # "print_in": False,
                    # "print_out": False,
                    "error_on_fail": False,
                    "ipopt": {
                        #"linear_solver": "ma27",
                        "max_iter": 5000,
                        "sb": "yes",  # Suppress IPOPT banner
                        "tol": 1e-2,
                        "print_level": 5,
                        "hessian_approximation": "limited-memory"
                    },
                },
            )
        ## Make it concrete for this ocp
        ocp.method(MultipleShooting(N=N,M=4,intg='rk'))  

        t1 = rospy.Time.now()

        t2 = rospy.Time.now()
        

        ## Set_Initial
        
        if attempt == 0:
            ocp.set_initial(x,(robot_position[0]+((goal_position[0]-robot_position[0])*(ocp.t/ocp.T))))
            ocp.set_initial(y,(robot_position[1]+((goal_position[1]-robot_position[1])*(ocp.t/ocp.T)))) # Initial value of the solver equal to the current time
            ocp.set_initial(z,(robot_position[2]+((goal_position[2]-robot_position[2])*(ocp.t/ocp.T))))
        else:
            perturbation = 0.1*attempt
            # x_init = np.linspace(robot_position[0], goal_position[0], N + 1) + np.random.uniform(-perturbation, perturbation, size=N + 1)
            # y_init = np.linspace(robot_position[1], goal_position[1], N + 1) + np.random.uniform(-perturbation, perturbation, size=N + 1)
            # z_init = np.linspace(robot_position[2], goal_position[2], N + 1) + np.random.uniform(-perturbation, perturbation, size=N + 1)
            x_init = (robot_position[0]+((goal_position[0]-robot_position[0])*(ocp.t/ocp.T))) + np.random.uniform(-perturbation, perturbation, size=N + 1)
            y_init = (robot_position[1]+((goal_position[1]-robot_position[1])*(ocp.t/ocp.T))) + np.random.uniform(-perturbation, perturbation, size=N + 1)
            z_init = (robot_position[2]+((goal_position[2]-robot_position[2])*(ocp.t/ocp.T))) + np.random.uniform(-perturbation, perturbation, size=N + 1)
            ocp.set_initial(x, x_init)
            ocp.set_initial(y, y_init)
            ocp.set_initial(z, z_init)
    # ocp.set_initial(x, ocp.t)
    # ocp.set_initial(y, ocp.t) 
    # ocp.set_initial(z, ocp.t)


        ## Give concrete numerical value at parameters
        if not no_UGV:
            ocp.set_value(fitPts, fitted_points)
        ocp.set_value(pos_obs, pos_obs_val.T)
        ocp.set_value(r_obs, r_obs_val)

        try:
            ## Solve
            sol = ocp.solve()
            correct = True
        except Exception as e:
            rospy.logwarn(f"Solver failed on attempt {attempt + 1}")
            correct = False
            continue   

            # Check if solver found a solution
        if correct and sol.stats["return_status"] == "Solve_Succeeded":
            initial_success = True
            rospy.loginfo(f"Solver found a solution on attempt {attempt + 1}")
        else:
            attempt += 1

        t3 = rospy.Time.now()

    diff1 = t1-t0
    diff2 = t3-t2
    rospy.loginfo(f"diff1: {diff1}")
    rospy.loginfo(f"diff2: {diff2}")
    rospy.loginfo("Tempo di esecuzione: %f secondi" % diff1.to_sec())
    rospy.loginfo("Tempo di esecuzione: %f secondi" % diff2.to_sec())

    first_goal = False

    return sol, x, y, z, psi, w_psi


##Publisher Path (Rviz)
def publish_ocp_path(sol, x, y, z):

    #Create Path message
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = frame

    # OCP trajectory points
    ts, xs = sol.sample(x, grid='control')
    ts, ys = sol.sample(y, grid='control')
    ts, zs = sol.sample(z, grid='control')

    for i in range(len(ts)):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame
        pose.pose.position.x = xs[i]
        pose.pose.position.y = ys[i]
        pose.pose.position.z = zs[i]
        path_msg.poses.append(pose)

    if len(ts) > 0:
        path_ocp_pub.publish(path_msg)
        rospy.loginfo("Path Published!")

def distance(point1, point2):
    return np.linalg.norm(np.array(point1)-np.array(point2))
    

## Follow Path: position and velocity of each sampled point of the curve to be given to the low-level control
def follow_path(sol, x, y, z, psi, w_psi):
    global robot_position
    #Points
    ts, xs = sol.sample(x, grid='control')
    ts, ys = sol.sample(y, grid='control')
    ts, zs = sol.sample(z, grid='control')
    ts, psi_val = sol.sample(psi, grid='control')
    ts, w_psi_val = sol.sample(w_psi, grid='control')  

    # Velocity points
    vxs = np.diff(xs) / np.diff(ts)
    vys = np.diff(ys) / np.diff(ts)
    vzs = np.diff(zs) / np.diff(ts)

    control_threshold = 0.5
    # rospy.loginfo(f"d_xs: {xs}")
    # rospy.loginfo(f"d_vxs: {len(vxs)}")
    # rospy.loginfo(f"d_ws: {len(w_psi_val)}")

    init_control = True  
    i=1
    while i<(len(xs)):
        if init_control:
            # robot_position = [xs[i], ys[i], zs[i]]
            # rospy.loginfo(f"init_rob pos: {robot_position}")
            # rospy.loginfo(f"init_point: {[xs[0], ys[0], zs[0]]}")

            posVel = PosVel()
            posVel.pose_frame_id = frame
            posVel.position = Point()
            posVel.position.x = xs[0]
            posVel.position.y = ys[0]
            posVel.position.z = zs[0]
            # rospy.loginfo(f"control_pos: {[xs[0], ys[0], zs[0]]}")
            posVel.yaw = psi_val[0]
            # rospy.loginfo(f"control_yaw: {psi_val[0]}")
            posVel.velocity = Point()         
            posVel.velocity.x = 0 #vxs[i]
            posVel.velocity.y = 0 #vys[i]
            posVel.velocity.z = 0 #vzs[i]
            # rospy.loginfo(f"control_vel: {[vxs[0], vys[0], vzs[0]]}")
            posVel.yaw_rate = w_psi_val[0] 
            # rospy.loginfo(f"control_wpsi: {w_psi_val[0]}")
            posVel_pub.publish(posVel)
            init_control = False

        rate_control = rospy.Rate(200)
        # rospy.loginfo(f"dist: {distance(robot_position, [xs[i-1], ys[i-1], zs[i-1]])}") 
        while not distance(robot_position, [xs[i-1], ys[i-1], zs[i-1]]) < control_threshold:
            rate_control.sleep()

        posVel = PosVel()
        posVel.pose_frame_id = frame
        posVel.position = Point()
        posVel.position.x = xs[i]
        posVel.position.y = ys[i]
        posVel.position.z = zs[i]
        # rospy.loginfo(f"control_pos: {[xs[i], ys[i], zs[i]]}")
        posVel.yaw = psi_val[i]
        # rospy.loginfo(f"control_yaw: {psi_val[i]}")
        posVel.velocity = Point()
        if i < len(vxs):          
            posVel.velocity.x = 0 #vxs[i]
            posVel.velocity.y = 0 #vys[i]
            posVel.velocity.z = 0 #vzs[i]
        else:
            posVel.velocity.x = 0
            posVel.velocity.y = 0
            posVel.velocity.z = 0
        # rospy.loginfo(f"control_vel: {[vxs[i], vys[i], vzs[i]]}")
        posVel.yaw_rate = w_psi_val[i] 
        # rospy.loginfo(f"control_wpsi: {w_psi_val[i]}")
        posVel_pub.publish(posVel)
        i += 1

    rospy.loginfo("Control published!")

   

if __name__ == "__main__":
    rospy.init_node('uav_path_planning')
    rospy.loginfo("Node uav_path_planning started")

    ## Parameters
    frame = rospy.get_param('/global_frame', 'map')
    x_min = rospy.get_param('/x_min', -50)
    x_max = rospy.get_param('/x_max', 50)
    y_min = rospy.get_param('/y_min', -50)
    y_max = rospy.get_param('/y_max', 50)
    z_min = rospy.get_param('/z_min', 0.5)
    z_max = rospy.get_param('/z_max', 5)
    r_uav = rospy.get_param('/r_uav', 1)
    theta_min = rospy.get_param('/theta_min', -0.5)
    theta_max = rospy.get_param('/theta_max', 0.5)
    v_min = rospy.get_param('/v_min', 0)
    v_max = rospy.get_param('/v_max', 2)
    w_psi_min = rospy.get_param('/w_psi_min', -1)
    w_psi_max = rospy.get_param('/w_psi_max', 1)
    w_theta_min = rospy.get_param('/w_theta_min', -2)
    w_theta_max = rospy.get_param('/w_theta_max', 2)
    N = rospy.get_param('/N', 40)
    K_time = rospy.get_param('/K_time', 1)
    K_dist = rospy.get_param('/K_dist', 1)
    dist_to_goal = rospy.get_param('/dist_to_goal', 1)
    no_UGV = rospy.get_param('/no_pathUGV', False)



    tf_buffer = tf2_ros.Buffer() # Initializing TF2  listener
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    
    obs_info = rospy.Subscriber('obstacle_info', ClusterInfo, obstacle_info)
    position_robot = rospy.Subscriber('odometry_drone', Odometry, robot_position_info)
    free_space = rospy.Subscriber('free_space', FreeMeshInfo, freeSpace_callback)
    if not no_UGV:
        wp_sub = rospy.Subscriber('wp_rosbot', Path, waypoints_callback)
    goal_sub = rospy.Subscriber('frontier_goal', PoseStamped, goal_callback)

    path_ocp_pub = rospy.Publisher("OCP_path_planning", Path, queue_size=10)
    posVel_pub = rospy.Publisher("/drone_interface/setpoint_raw", PosVel, queue_size=10)


    rate = rospy.Rate(10)

    while not position_ready:
        rospy.loginfo_throttle(5,"Waiting for UAV position..")
        rate.sleep()
    rospy.loginfo("Position obtained!")

    if not no_UGV:
        while not waypoints_ready:
            rospy.loginfo_throttle(5,"Waiting for rosbot waypoints..")
            rate.sleep()
        rospy.loginfo("Rosbot waypoints obtained!")

    if first_goal:
        while not goal_update:
            rospy.loginfo_throttle(5,"Waiting for  frontier goal..")
            rate.sleep()   

    while not rospy.is_shutdown():
        if goal_update and np.linalg.norm(robot_position - goal_position) < dist_to_goal :  
            goal_reached = True
            new_goal_compute = True
            goal_update = False
            rospy.loginfo("Goal reached!")

        if not goal_reached:
            rate.sleep()
            continue

        if goal_update and new_goal_compute:
            new_goal_compute = False

            if not np.all(np.isnan(robot_position)):
                rospy.loginfo("OCP planner starting")
                if not no_UGV:
                    fitted_points = get_fitted_points(waypoints_3d)
                    sol, x, y, z, psi, w_psi = uav_planner(robot_position, goal_position, fitted_points)
                else:
                    sol, x, y, z, psi, w_psi = uav_planner(robot_position, goal_position)
                rospy.loginfo("Follow OCP Path!")
                publish_ocp_path(sol, x, y, z)
                follow_path(sol, x, y, z, psi, w_psi)
                goal_reached = False

        rate.sleep()
