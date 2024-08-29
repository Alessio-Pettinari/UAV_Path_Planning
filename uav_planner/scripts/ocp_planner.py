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
from geometry_msgs.msg import Point, PoseStamped, PoseArray
from nav_msgs.msg import Path, Odometry
from core.msg import PosVel
import time
import casadi as cs
from scipy.spatial import KDTree


from uav_path_planning.interpolation_method.scripts.wayPoint_fitting import get_fitted_points
# export PYTHONPATH=/home/sara/alessio/rosbot_ws/src:$PYTHONPATH


# goal_position = np.array([3, 5, 0])

## Initializing global variables
pos_obs_val = np.zeros((1, 3))
r_obs_val = np.zeros(1)
robot_position = np.array([np.nan] * 3)
robot_orientation = np.array([np.nan]*2)
kdtree_obs = None
kdtree_free = None
freeVoxel_points = np.array([np.nan] * 3)

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


## Obstacle Map
def generate_obstacle_lookup():
    global kdtree_obs, freeVoxel_points, val_lim
    x_grid = np.linspace(x_min, x_max, N+1)
    y_grid = np.linspace(y_min, y_max, N+1)
    z_grid = np.linspace(z_min, z_max, N+1)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    data_flat = []
    idx = []
    for k in range(N+1):
        for j in range(N+1):
            for i in range(N+1):
                # d_free, idx_free = kdtree_free.query([X[i,j,k], Y[i,j,k], Z[i,j,k]], k=1)
                d_obs, idx_obs = kdtree_obs.query([X[i,j,k], Y[i,j,k], Z[i,j,k]], k=1)
                data_flat.append(d_obs)
                idx.append((i,j,k))
    data_flat = np.array(data_flat)
    data = data_flat.reshape((N+1, N+1, N+1))
    min_d = np.min(data_flat)
    alpha = 1
    eps = 1e-10
    data_flat = alpha / (data_flat - min_d + eps)
    ##
    # d_obs_target = 0.8
    # indices_closest = np.abs(data_flat - d_obs_target).argmin()
    # d_obs_value = data_flat[indices_closest]
    # val_lim = alpha / (d_obs_value - min_d + eps)
    # rospy.loginfo(f"Potential for d_obs = {d_obs_target} is {val_lim}") 

    ##
    # # rospy.loginfo(f"U: {data_flat}")
    # min_index = np.argmin(data_flat)
    # min_i, min_j, min_k = idx[min_index]  
    # min_x = x_grid[min_i]
    # min_y = y_grid[min_j]
    # min_z = z_grid[min_k]
    # rospy.loginfo(f"Min_U: {np.min(data_flat)}")
    # rospy.loginfo(f"Coordinates of Min_U: X={min_x}, Y={min_y}, Z={min_z}")
    # max_index = np.argmax(data_flat)
    # max_i, max_j, max_k = idx[max_index]     
    # max_x = x_grid[max_i]
    # max_y = y_grid[max_j]
    # max_z = z_grid[max_k]
    # rospy.loginfo(f"Max_U: {np.max(data_flat)}")
    # rospy.loginfo(f"Coordinates of Max_U: X={max_x}, Y={max_y}, Z={max_z}")
    
    # point = [max_x, max_y, max_z]
    # # point = [min_x, min_y, min_z]

    # interpolant = cs.interpolant('pot_field', 'linear', [x_grid, y_grid, z_grid], data_flat)
    # potential = interpolant(cs.vertcat(*point)).full().flatten()[0]
    # rospy.loginfo(f'Point: {point}, Potential: {potential}')

    
    # from scipy.interpolate import RegularGridInterpolator
    # interp_func = RegularGridInterpolator((x_grid, y_grid, z_grid), data, method='linear')
    # pot = interp_func(np.array(point))
    # rospy.loginfo(f'Point: {point}, Potential2: {pot}')


    ##PLOT
    #Flatten the meshgrid arrays
    # X_flat = X.flatten()
    # Y_flat = Y.flatten()
    # Z_flat = Z.flatten()
    # # 3D slices of contour plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    num_slices = min(len(z_grid) // 10, len(ax))  # Ensure we do not exceed the number of subplots
    for i in range(num_slices):
        data_slice = data_flat.reshape(41, 41, 41, order='F')[:, :, i*10]
        ax[i].contourf(X[:, :, i*10], Y[:, :, i*10], data_slice, cmap='viridis')
        ax[i].set_title('z = {}'.format(z_grid[i*10]))
        ax[i].axis('equal')
        ax[i].grid(True)
    # Add a colorbar
    fig.colorbar(ax[0].contourf(X[:, :, 0], Y[:, :, 0], data_slice, cmap='viridis'), ax=ax, orientation='horizontal')
    fig.suptitle('Potential Map at Different Z-slices', fontsize=16, fontweight='bold')
    plt.show()
    return

    
    ## Plot Potential Field
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import Normalize
    # data = np.array(data_flat).reshape((N+1,N+1,N+1))
    # z_index=20
    # data_slice = data[:, :, z_index]
    # plt.figure(figsize=(8, 6))
    # #plt.contourf(x_grid, y_grid, data_slice, cmap='viridis', norm=Normalize(vmin=data.min(), vmax=data.max()))
    # plt.imshow(data_slice, extent=(x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
    # plt.colorbar(label='Potential')
    # plt.title(f'Potential Map Slice at Z-index {z_index}')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.axis('equal')
    # plt.show()
    # return

    return data_flat, cs.interpolant('pot_field', 'linear', [x_grid, y_grid, z_grid], data_flat)

## Obstacle Map
# def generate_obstacle_lookup():
#     global kdtree_obs
#     data_flat = []
#     x_list = np.linspace(0, 10, 41)
#     y_list = np.linspace(0, 20, 41)
#     z_list = np.linspace(z_min, z_max, 41)
#     for y in y_list:
#         for x in x_list:
#             for z in z_list:
#                 # d_free, idx_free = kdtree_free.query([X[i,j,k], Y[i,j,k], Z[i,j,k]], k=1)
#                 d_obs, idx_obs = kdtree_obs.query([x, y, z], k=1)
#                 # if d_free>0.1:
#                 #     d_obs=0.1
#                 data_flat.append(d_obs)
#     # rospy.loginfo(f"AAAA: {data_flat}")

#     return data_flat, cs.interpolant('pot_field', 'linear', [x_list, y_list, z_list], np.array(data_flat))


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
# def obstacle_info(info):
#     global pos_obs_val, r_obs_val
#     pos_obs_val = np.array([[p.x, p.y, p.z] for p in info.clst_centers])
#     r_obs_val = np.array(info.clst_radius)

#     # Odom2Map
#     obst_mapFrame = []
#     for pos in pos_obs_val:
#         obst_point = Point(x=pos[0], y=pos[1], z=pos[2])
#         obst_point_map = transform_point(obst_point, "odom", "map") # Transform obstacles from "odom" to "map"
#         if obst_point_map is not None:
#             obst_mapFrame.append([obst_point_map.x, obst_point_map.y, obst_point_map.z])
#         else:
#             rospy.logwarn("Failed to transform obstacle_point in map frame!")
    
#     pos_obs_val = np.array(obst_mapFrame)
#     # rospy.loginfo("Obstacle Position obtained!")


## Callback function obstacle voxel from Octomap
def obstacle_voxel(pose_array):
    global kdtree_obs
    points = []
    for pose in pose_array.poses:
        points.append([pose.position.x, pose.position.y, pose.position.z]) 
    points = np.array(points)

    if len(points) == 0:
        rospy.loginfo("Cluster points array empty!")

    obsVoxel_mapFrame = []
    for pos in points:
        obsVoxel_point = Point(x=pos[0], y=pos[1], z=pos[2])
        obsVoxel_point_map = transform_point(obsVoxel_point, "odom", "map") # Transform obstacles from "odom" to "map"
        if obsVoxel_point_map is not None:
            obsVoxel_mapFrame.append([obsVoxel_point_map.x, obsVoxel_point_map.y, obsVoxel_point_map.z])
        else:
            rospy.logwarn("Failed to transform obstacle_point in map frame!")
    
    points = np.array(obsVoxel_mapFrame)

    kdtree_obs = KDTree(points)


def freeVoxel_Callback(data):
    global freeVoxel_points
    freeVoxel_points = []

    for pose in data.poses:
        freeVoxel_points.append([pose.position.x, pose.position.y, pose.position.z])
    
    # Converto in un array numPy
    freeVoxel_points = np.array(freeVoxel_points)
    if len(freeVoxel_points) == 0:
        rospy.loginfo("FreeVoxel points array empty!")

    freeVoxel_mapFrame = []
    for freepos in freeVoxel_points:
        freeVoxel_point = Point(x=freepos[0], y=freepos[1], z=freepos[2])
        freeVoxel_point_map = transform_point(freeVoxel_point, "odom", "map") # Transform obstacles from "odom" to "map"
        if freeVoxel_point_map is not None:
            freeVoxel_mapFrame.append([freeVoxel_point_map.x, freeVoxel_point_map.y, freeVoxel_point_map.z])
        else:
            rospy.logwarn("Failed to transform obstacle_point in map frame!")
    
    freeVoxel_points = np.array(freeVoxel_mapFrame)
    # kdtree_free = KDTree(freeVoxel_points)



## Callback function robot position info
def robot_position_info(odom):
    global robot_position, position_ready, robot_orientation
    robot_x = odom.pose.pose.position.x
    robot_y = odom.pose.pose.position.y
    robot_z = odom.pose.pose.position.z
    robot_yaw = odom.twist.twist.angular.x
    robot_pitch = odom.twist.twist.angular.y
    robot_orientation = np.array([robot_yaw, robot_pitch])
    # robot_position = np.array([robot_x, robot_y, robot_z])

    # Odom2Map
    robot_point = Point(x=robot_x, y=robot_y, z=robot_z)
    # robot_orient_point = Point(yaw=robot_yaw, pitch=robot_pitch)
    robot_point_map = transform_point(robot_point, "odom", "map")  # Transform robot_position from "odom" to "map"
    if robot_point_map is not None:
        robot_mapFrame = np.array([robot_point_map.x, robot_point_map.y, robot_point_map.z])
    else:
        rospy.logwarn("Failed to transform robot_position in map frame!")
    robot_position = robot_mapFrame

    # rospy.loginfo(f"Updated robot position: {robot_position}")
    if not np.all(np.isnan(robot_position)) & np.all(np.isnan(robot_orientation)):
        position_ready = True
        # rospy.loginfo("Position obtained!")

   
 

## OCP Planner    
def uav_planner(robot_position,robot_orientation, goal_position, fitted_points='none'):
    global first_goal, pos_obs_val, r_obs_val, coeff_matrix, const_matrix, val_lim

    rospy.loginfo(f"goal: {goal_position}")
    rospy.loginfo(f"robot: {robot_position}")
    t0 = time.time()

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
        # pos_obs = ocp.parameter(3,len(r_obs_val))      # obstacles position
        # r_obs = ocp.parameter(1,len(r_obs_val))        # radius obstacles
        if not no_UGV:
            fitPts = ocp.parameter(3, grid='control')  # fitted points

        
        ## Hard Constraints:
        # Initial constraints
        ocp.subject_to(ocp.at_t0(x)== robot_position[0]) 
        ocp.subject_to(ocp.at_t0(y)== robot_position[1]) 
        ocp.subject_to(ocp.at_t0(z)== robot_position[2])
        ocp.subject_to(ocp.at_t0(psi)== robot_orientation[0])
        ocp.subject_to(ocp.at_t0(theta)== robot_orientation[1])
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
        # for idx_obj in range(len(r_obs_val)):
        #     ocp.subject_to(sumsqr(p-pos_obs[:,idx_obj])>=((r_obs[:,idx_obj]+r_uav)**2)) 
        #  
        data_flat, obstacle_lookup = generate_obstacle_lookup()

        # test_points = np.array([[3, 11, 1.4], [4, 6, 0.65], [2.5, 2.5, 1.4]])  
        # # Stampa il valore di potenziale per ciascun punto di test
        # for point in test_points:
        #     potential = obstacle_lookup(cs.vertcat(*point)).full().flatten()[0]
        #     rospy.loginfo(f'Point: {point}, Potential: {potential}')
        # return

        # ocp.subject_to(obstacle_lookup(cs.vertcat(x, y, z)) >= r_uav)
        # ocp.subject_to(obstacle_lookup(cs.vertcat(x, y, z)) <= val_lim)
    
        # import matplotlib.pyplot as plt
        # from matplotlib.colors import Normalize
        # x_grid = np.linspace(0, 10, 41)
        # y_grid = np.linspace(0, 20, 41)
        # z_grid = np.linspace(z_min, z_max, 41)
        # z_index=20
        # data_slice = data[:, :, z_index]
        # plt.contourf(x_grid, y_grid, data_slice, cmap='gray')
        # # plt.imshow(data_slice, extent=(x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
        # plt.colorbar(label='Distance')
        # plt.title(f'Grayscale Map with Contours (Slice at Z-index {z_index})')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.axis("equal")
        # plt.show()
        # return

        # with open('/home/sara/Downloads/data.npy', 'wb') as f:
        #     np.save(f, data_flat)
        # return

        


        ## Soft Consraints: Objective Function:
        # Minimal time
        ocp.add_objective(K_time*ocp.T) 
        # Minimal distance to the other curve fitted: 
        if not no_UGV:
            ocp.add_objective(K_dist*ocp.integral(sumsqr(p[:2,:]-fitPts[:2,:])/ocp.T, grid='control'))
        # Obstacle Avoidance
        ocp.add_objective(K_obs*ocp.integral(obstacle_lookup(p)))  


        ## Pick a solution method
        # ocp.solver(
        #         "ipopt",
        #         {
        #             "expand": True,
        #             "verbose": False,
        #             "print_time": True,
        #             # "print_in": False,
        #             # "print_out": False,
        #             "error_on_fail": False,
        #             "ipopt": {
        #                 #"linear_solver": "ma27",
        #                 "max_iter": 5000,
        #                 "sb": "yes",  # Suppress IPOPT banner
        #                 "tol": 1e-2,
        #                 "print_level": 5,
        #                 "hessian_approximation": "limited-memory"
        #             },
        #         },
        #     )
        
        ocp.solver(
                "ipopt",
                {
                    # "expand": True,
                    "verbose": False,
                    "print_time": True,
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

        t1 = time.time()

        t2 = time.time()
        

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
        # ocp.set_value(pos_obs, pos_obs_val.T)
        # ocp.set_value(r_obs, r_obs_val)

        try:
            ## Solve
            sol = ocp.solve()
            correct = True
        except Exception as e:
            rospy.logwarn(f"Solver failed on attempt {attempt + 1} with error {e}")
            correct = False
            continue   

        rospy.loginfo(f"Solver status {sol.stats['return_status']}")

        # Check if solver found a solution
        # if correct and sol.stats["return_status"] == "Solve_Succeeded":
        if correct and sol.stats["return_status"] in ["Solve_Succeeded","Solved To Acceptable Level"]:
            initial_success = True
            rospy.loginfo(f"Solver found a solution on attempt {attempt + 1}")
        else:
            attempt += 1

        t3 = time.time()

    diff1 = t1-t0
    diff2 = t3-t2
    rospy.loginfo("Tempo di esecuzione: %f secondi" % diff1)
    rospy.loginfo("Tempo di esecuzione: %f secondi" % diff2)

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


    ## Plot traj point with potential
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # ts, xs = sol.sample(x, grid='control')
    # ts, ys = sol.sample(y, grid='control')
    # ts, zs = sol.sample(z, grid='control')
    # trajectory_potentials = calculate_potential_for_trajectory_points(xs, ys, zs)
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(xs, ys, zs, c=trajectory_potentials, cmap='viridis', marker='o', s=50)
    # plt.colorbar(sc, label='Potential')

    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Z Coordinate')
    # ax.set_title('Trajectory Points with Associated Potential Values')

    # plt.show()
    ##########
    # data_flat, obstacle_lookup = generate_obstacle_lookup()
    # potentials = []
    # for i in range(len(xs)):
    #     point = np.array([xs[i], ys[i], zs[i]])
    #     potential = obstacle_lookup(point)
    #     potentials.append(potential)
    # potentials = np.array(potentials)
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(xs, ys, zs, c=potentials, cmap='viridis')
    # ax.plot(xs, ys, zs, 'k--', alpha=0.5)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # cb = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    # cb.set_label('Potential')
    # plt.title('Trajectory and Corresponding Potential')
    # plt.show()
    # return
    return

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
    K_obs = rospy.get_param('/K_obs', 1)
    dist_to_goal = rospy.get_param('/dist_to_goal', 1)
    no_UGV = rospy.get_param('/no_pathUGV', False)



    tf_buffer = tf2_ros.Buffer() # Initializing TF2  listener
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    ## obstacle sphere clustering
    # obs_info = rospy.Subscriber('obstacle_info', ClusterInfo, obstacle_info)
    obs_voxel = rospy.Subscriber('obstacle_voxel', PoseArray, obstacle_voxel)

    free_voxel = rospy.Subscriber('free_voxel', PoseArray, freeVoxel_Callback)


    position_robot = rospy.Subscriber('odometry_drone', Odometry, robot_position_info)
    # free_space = rospy.Subscriber('free_space', FreeMeshInfo, freeSpace_callback)
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
                    sol, x, y, z, psi, w_psi = uav_planner(robot_position, robot_orientation, goal_position, fitted_points)
                else:
                    sol, x, y, z, psi, w_psi = uav_planner(robot_position,robot_orientation, goal_position)
                rospy.loginfo("Follow OCP Path!")
                publish_ocp_path(sol, x, y, z)
                follow_path(sol, x, y, z, psi, w_psi)
                goal_reached = False

        rate.sleep()
