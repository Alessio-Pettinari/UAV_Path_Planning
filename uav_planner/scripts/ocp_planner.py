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
from std_msgs.msg import Bool
from core.msg import PosVel
import time
import casadi as cs
from scipy.spatial import cKDTree
import pickle


from uav_path_planning.interpolation_method.scripts.wayPoint_fitting import get_fitted_points
# export PYTHONPATH=/home/sara/alessio/rosbot_ws/src:$PYTHONPATH


# goal_position = np.array([3, 5, 0])

## Initializing global variables
pos_obs_val = np.zeros((1, 3))
r_obs_val = np.zeros(1)
robot_position = np.array([np.nan] * 3)
robot_orientation = np.array([np.nan]*2)
UGV_position = np.array([np.nan] * 3)
kdtree_obs = None
kdtree_free = None
freeVoxel_points = np.array([np.nan] * 3)
flag_UGVgoalReach = False
flag_UAVupdateGoal = False


goal_reached = True  
new_goal_compute = True
waypoints_ready = False
position_ready = False
first_goal = True
goal_update = False
start_ocp = False

waypoints_3d = []
goal_position = []
coeff_matrix = []
const_matrix = []

l=0

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
def generate_obstacle_lookup(kdtree_free, kdtree_obs):
    x_grid = np.linspace(x_min, x_max, N+1)
    y_grid = np.linspace(y_min, y_max, N+1)
    z_grid = np.linspace(z_min, z_max, N+1)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    data_flat = []
    idx = []
    voxel_res=0.4
    for k in range(N+1):
        for j in range(N+1):
            for i in range(N+1):
                # Query per calcolare la distanza da ogni punto campionato all'ostacolo più vicino
                d_obs, idx_obs = kdtree_obs.query([X[i,j,k], Y[i,j,k], Z[i,j,k]], k=1)
                # Query per calcolare la distanza dai voxel liberi
                d_free, idx_free = kdtree_free.query([X[i,j,k], Y[i,j,k], Z[i,j,k]], k=1)
                # Verifico se sono dentro la parte libera, altrimenti lo considero come ostacolo
                if d_free > voxel_res:
                    point = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    coord_free_point = kdtree_free.data[idx_free]
                    delta = np.abs(point - coord_free_point)

                    if np.all(delta <= voxel_res):
                        data_flat.append(d_obs)
                    else:
                        d_obs=0.05
                        data_flat.append(d_obs)
                else:
                    data_flat.append(d_obs)
                idx.append((i,j,k))
    data_flat = np.array(data_flat)
    data = data_flat.reshape((N+1, N+1, N+1))
    min_d = np.min(data_flat)
    alpha = 1
    eps = 1e-10
    data_flat = alpha / (data_flat - (min_d/2) + eps)

    # data_flat_save={'data_flat':data_flat}
    # with open('data_flat.pkl', 'wb') as file:
    #     pickle.dump(data_flat_save, file)


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
    ##3D slices of contour plots
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # num_slices = min(len(z_grid) // 10, len(ax))  # Ensure we do not exceed the number of subplots
    # for i in range(num_slices):
    #     data_slice = data_flat.reshape(41, 41, 41, order='F')[:, :, i*10]
    #     ax[i].contourf(X[:, :, i*10], Y[:, :, i*10], data_slice, cmap='viridis')
    #     ax[i].set_title('z = {}'.format(z_grid[i*10]))
    #     ax[i].axis('equal')
    #     ax[i].grid(True)
    # # Add a colorbar
    # fig.colorbar(ax[0].contourf(X[:, :, 0], Y[:, :, 0], data_slice, cmap='viridis'), ax=ax, orientation='horizontal')
    # fig.suptitle('Potential Map at Different Z-slices', fontsize=16, fontweight='bold')
    # plt.show()


    ##Potential distances 3d
    ## Flatten the meshgrid arrays
    # X_flat = X.flatten()
    # Y_flat = Y.flatten()
    # Z_flat = Z.flatten()

    # # 3D scatter plot of values above a threshold
    # threshold = 30
    # above_threshold_indices = np.where(data_flat > threshold)
    # above_threshold_values = data_flat[above_threshold_indices]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_flat[above_threshold_indices], Y_flat[above_threshold_indices], Z_flat[above_threshold_indices], c=above_threshold_values, cmap='viridis')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()
        

    ##PLOT Potential ERROR
    # plane_dist_obst = 0.75
    # d_fix = np.array(plane_dist_obst*np.ones((N+1, N+1)))
    # pot_fix = alpha / (d_fix - min_d + eps)
    # x_target = 2
    # x_idx = np.argmin(np.abs(X[:,0,0]-x_target))
    # yz_plane_data = data[x_idx, :, :]
    # yz_plane_X = Y[x_idx, :, :]
    # yz_plane_Y = Z[x_idx, :, :]
    # mask = (yz_plane_X >= 8) & (yz_plane_X <= 10.5) & (yz_plane_Y >= 0.7) & (yz_plane_Y <= 2)
    # slice_data = yz_plane_data[mask]
    # slice_X = yz_plane_X[mask]
    # slice_Y = yz_plane_Y[mask]
    # potential_true_slice = pot_fix[mask]
    # error = slice_data - potential_true_slice
    # unique_y = np.unique(slice_Y)
    # unique_z = np.unique(slice_X)
    # mesh_y, mesh_z = np.meshgrid(unique_y, unique_z, indexing='ij')
    # error_matrix = np.zeros_like(mesh_y)
    # for i, y in enumerate(unique_y):
    #     for j, z in enumerate(unique_z):
    #         idx = (slice_Y == y) & (slice_X == z)
    #         if np.any(idx):
    #             error_matrix[i, j] = error[idx].mean()
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(mesh_y, mesh_z, error_matrix, cmap='viridis')
    # fig.colorbar(surf, ax=ax, orientation='vertical')
    # ax.set_xlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Potential Error')
    # ax.set_title(f'Potential Error for the plane at coordinate X = {x_target}', fontsize=16, fontweight='bold')
    # plt.show()
    # return
    



    
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
        # waypoints_3d.append((pose.pose.position.x, pose.pose.position.y, pose.pose.position.z))
        waypoints_3d.append((pose.position.x, pose.position.y, pose.position.z))
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


## Callback UGV frontier goal
def UGV_goal_callback(data):
    global UGV_goal
    UGV_goal = []
    UGV_goal = (data.pose.position.x, data.pose.position.y, data.pose.position.z)



## Callback UGV goal reached
def UGV_goal_reach_callback(data):
    global flag_UGVgoalReach
    flag_UGVgoalReach = data.data
    rospy.loginfo(f"flagUGVGOAL: {flag_UGVgoalReach}")


## Calback flag UAV update oal
def flag_uav_startOCP_callback(data):
    global flag_UAVupdateGoal
    flag_UAVupdateGoal = data.data


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
    global points, size_occVoxel
    points_odom = []
    for pose in pose_array.poses:
        points_odom.append([pose.position.x, pose.position.y, pose.position.z]) 
    points_odom = np.array(points_odom)

    if len(points_odom) == 0:
        rospy.loginfo("Cluster points array empty!")

    obsVoxel_mapFrame = []
    for pos in points_odom:
        obsVoxel_point = Point(x=pos[0], y=pos[1], z=pos[2])
        obsVoxel_point_map = transform_point(obsVoxel_point, "odom", "map") # Transform obstacles from "odom" to "map"
        if obsVoxel_point_map is not None:
            obsVoxel_mapFrame.append([obsVoxel_point_map.x, obsVoxel_point_map.y, obsVoxel_point_map.z])
        else:
            rospy.logwarn("Failed to transform obstacle_point in map frame!")
    
    points = np.array(obsVoxel_mapFrame)
    size_occVoxel = np.size(points)

    data_obstacle={'points':points}
    # with open('data_obs.pkl', 'wb') as f:
    #     pickle.dump(data_obstacle, f)
    # rospy.loginfo("Obstacle saved")
    # kdtree_obs = cKDTree(points)


def freeVoxel_Callback(data):
    global freeVoxel_points, size_freeVoxel
    freeVoxel_points_odom = []

    for pose in data.poses:
        freeVoxel_points_odom.append([pose.position.x, pose.position.y, pose.position.z])
    
    # Converto in un array numPy
    freeVoxel_points_odom = np.array(freeVoxel_points_odom)
    if len(freeVoxel_points_odom) == 0:
        rospy.loginfo("FreeVoxel points array empty!")

    freeVoxel_mapFrame = []
    for freepos in freeVoxel_points_odom:
        freeVoxel_point_odom = Point(x=freepos[0], y=freepos[1], z=freepos[2])
        freeVoxel_point_map = transform_point(freeVoxel_point_odom, "odom", "map") # Transform obstacles from "odom" to "map"
        if freeVoxel_point_map is not None:
            freeVoxel_mapFrame.append([freeVoxel_point_map.x, freeVoxel_point_map.y, freeVoxel_point_map.z])
        else:
            rospy.logwarn("Failed to transform obstacle_point in map frame!")
    
    freeVoxel_points = np.array(freeVoxel_mapFrame)
    size_freeVoxel = np.size(freeVoxel_points)
    

  




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
        # rospy.loginfo("UAV Position obtained!")



## Callback function UGV position info
def UGV_position_info(map):
    global UGV_position
    ugv_x = map.pose.pose.position.x
    ugv_y = map.pose.pose.position.y
    ugv_z = map.pose.pose.position.z
    UGV_position = np.array([ugv_x, ugv_y, ugv_z])


   
 

## OCP Planner    
def uav_planner(robot_position, robot_orientation, goal_position, fitted_points='none'):
    global first_goal, pos_obs_val, r_obs_val, coeff_matrix, const_matrix, freeVoxel_points, points, kdtree_free, kdtree_obs

    rospy.loginfo(f"Goal_UAV: {goal_position}")
    rospy.loginfo(f"Robot Position: {robot_position}")
    t0 = time.time()

    initial_success = False
    max_attempts = 5
    attempt = 0
    kdtree_free = cKDTree(freeVoxel_points) 
    kdtree_obs = cKDTree(points)
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
        #ocp.subject_to(ocp.at_t0(psi)== robot_orientation[0])
        ocp.subject_to(ocp.at_t0(psi)== np.arctan2((UGV_position[1]-robot_position[1]),(UGV_position[0]-robot_position[0])))
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
        data_flat, obstacle_lookup = generate_obstacle_lookup(kdtree_free, kdtree_obs)

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
        #ocp.add_objective(1*ocp.T) 

        # Minimal distance to the other curve fitted: 
        if not no_UGV:
           dist_front = np.linalg.norm(goal_position - fitted_points[:,-1])
           rospy.loginfo(f"dist_frontiera: {dist_front}")
           if (dist_front > 5) or (attempt != 0):
               ocp.add_objective(1*ocp.integral(sumsqr(p[:2,:]-fitPts[:2,:]), grid='control'))
           else:
               ocp.add_objective(K_dist*ocp.integral(sumsqr(p[:2,:]-fitPts[:2,:]), grid='control'))
        #    ##ocp.add_objective(0*ocp.integral(sumsqr(p[:2,:]-fitPts[:2,:])/ocp.T, grid='control'))

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
                        "tol": 5e-1,
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
        if correct and sol.stats["return_status"] in ["Solve_Succeeded","Solved_To_Acceptable_Level"]:
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
    global kdtree_obs,  kdtree_free
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
    # data_flat, obstacle_lookup = generate_obstacle_lookup(kdtree_free,kdtree_obs)
    # potentials = []
    # for i in range(len(xs)):
    #     point = np.array([xs[i], ys[i], zs[i]])
    #     potential = obstacle_lookup(point)
    #     potentials.append(potential)
    # potentials = np.array(potentials)
    # # fig = plt.figure(figsize=(10, 8))
    # # ax = fig.add_subplot(111, projection='3d')
    # # sc = ax.scatter(xs, ys, zs, c=potentials, cmap='viridis')
    # # ax.plot(xs, ys, zs, 'k--', alpha=0.5)
    # # ax.set_xlabel('X')
    # # ax.set_ylabel('Y')
    # # ax.set_zlabel('Z')
    # # cb = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    # # cb.set_label('Potential')
    # # plt.title('Trajectory and Corresponding Potential')
    # # plt.show()
    # fig, ax = plt.subplots(figsize=(10, 6))
    # potentials = np.squeeze(potentials)
    # ax.plot(ts, potentials, linestyle='-', color='blue', marker='o', markerfacecolor='red', markeredgecolor='black')  # Linea blu, marker rosso con bordo nero 
    # ax.set_xlabel('Time (ts)')
    # ax.set_ylabel('Potential')
    # plt.title('Trajectory and Corresponding Potential',fontsize=16, fontweight='bold')
    # plt.grid(True)
    # plt.show()
    # return

def distance(point1, point2):
    return np.linalg.norm(np.array(point1)-np.array(point2))
    

## Follow Path: position and velocity of each sampled point of the curve to be given to the low-level control
def follow_path(sol, x, y, z, psi, w_psi):
    global robot_position, UGV_goal
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
        #posVel.yaw = psi_val[i]
        posVel.yaw = np.arctan2((UGV_goal[1]-ys[i]),(UGV_goal[0]-xs[i]))
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


## Plot Trajectory
def plot_traj(sol, x, y, z, fitted_points):
    global UGV_goal, goal_position

    fig = plt.figure()
    plot = fig.add_subplot(projection='3d',aspect='equal')
    plot.set_xlabel('X-Position (m)')
    plot.set_ylabel('Y-Position (m)')
    plot.set_zlabel('Z-Position (m)')
    ts, xs = sol.sample(x, grid='integrator', refine=10)
    ts, ys = sol.sample(y, grid='integrator', refine=10)
    ts, zs = sol.sample(z, grid='integrator', refine=10)  
    plot.plot(xs, ys, zs, '-',label="UAV Trajectory", color="r")
    plot.plot(fitted_points[0,:],fitted_points[1,:],fitted_points[2,:], linestyle='-', color='g', label="UGV Trajectory")
    # Add the UGV_goal point
    plot.scatter(fitted_points[0,-1], fitted_points[1,-1], fitted_points[2,-1], 
                 color='b', s=80, label="UGV Goal", marker='o')
    # Add the goal_position point
    plot.scatter(goal_position[0], goal_position[1], goal_position[2], 
                 color='m', s=80, label="UAV Goal", marker='o')
    # Set axis to be equal
    plot.set_box_aspect([ub - lb for lb, ub in (getattr(plot, f'get_{a}lim')() for a in 'xyz')])
    plot.set_title('Model Trajectory', fontsize=16, fontweight='bold')    
    plot.legend()
    # Show the plot
    plt.show(block=False)

# def plot_traj(sol, x, y, z, fitted_points, sol1, x1, y1, z1, sol2, x2, y2, z2, sol3, x3, y3, z3, points):
#     global UGV_goal, goal_position

#     fig = plt.figure()
#     plot = fig.add_subplot(projection='3d',aspect='equal')
#     plot.set_xlabel('X-Position (m)')
#     plot.set_ylabel('Y-Position (m)')
#     plot.set_zlabel('Z-Position (m)')
#     # ts, xs = sol.sample(x, grid='integrator', refine=10)
#     # ts, ys = sol.sample(y, grid='integrator', refine=10)
#     # ts, zs = sol.sample(z, grid='integrator', refine=10)  
#     # ts1, xs1 = sol1.sample(x1, grid='integrator', refine=10)
#     # ts1, ys1 = sol1.sample(y1, grid='integrator', refine=10)
#     # ts1, zs1 = sol1.sample(z1, grid='integrator', refine=10)  
#     # ts2, xs2 = sol2.sample(x2, grid='integrator', refine=10)
#     # ts2, ys2 = sol2.sample(y2, grid='integrator', refine=10)
#     # ts2, zs2 = sol2.sample(z2, grid='integrator', refine=10)  
#     # ts3, xs3 = sol3.sample(x3, grid='integrator', refine=10)
#     # ts3, ys3 = sol3.sample(y3, grid='integrator', refine=10)
#     # ts3, zs3 = sol3.sample(z3, grid='integrator', refine=10)  
#     plot.plot(xs, ys, zs, '-',label="UAV Trajectory", color="r")
#     plot.plot(xs1, ys1, zs1, '-',label="UAV Trajectory 1", color="m")
#     plot.plot(xs2, ys2, zs2, '-',label="UAV Trajectory 2", color="y")
#     plot.plot(xs3, ys3, zs3, '-',label="UAV Trajectory 3", color="c")
#     plot.plot(fitted_points[0,:],fitted_points[1,:],fitted_points[2,:], linestyle='-', color='g', label="UGV Trajectory")
#     plot.scatter(fitted_points[0,-1], fitted_points[1,-1], fitted_points[2,-1], 
#                  color='b', s=80, label="UGV Goal", marker='o')
#     # plot.scatter(goal1[0], goal1[1], goal1[2], color='m', s=80, marker='o')
#     # plot.scatter(goal2[0], goal2[1], goal2[2], color='m', s=80, marker='o')
#     # plot.scatter(goal3[0], goal3[1], goal3[2], color='m', s=80, marker='o')
#     plot.scatter(goal_position[0], goal_position[1], goal_position[2], color='m', s=80, label="UAV Goal", marker='o')
#     step = 20
#     points2 = points[::step] 
#     plot.scatter(points2[:,0],points2[:,1],points2[:,2], c='k',marker=',')
#     plot.set_box_aspect([ub - lb for lb, ub in (getattr(plot, f'get_{a}lim')() for a in 'xyz')])
#     plot.set_title('OCP Trajectory to Small Variation Goal', fontsize=16, fontweight='bold')    
#     plot.legend()
    
#     plt.savefig('prova1')
#     rospy.loginfo("immagine salvata")
#     plt.show(block=False)


## Reduced waypoints UGV path
def downsample_waypoints(waypoints_3d, num_points=100):
    # Se ci sono meno o uguali punti di quanti ne vogliamo, restituisci direttamente
    if len(waypoints_3d) <= num_points:
        return waypoints_3d
    
    # Mantieni il punto iniziale
    start_point = [waypoints_3d[0]]
    
    # Mantieni il punto finale
    end_point = [waypoints_3d[-1]]
    
    # Campiona in modo uniforme i punti intermedi
    num_internal_points = num_points - 2  # -2 per il punto iniziale e finale
    indices = np.linspace(1, len(waypoints_3d) - 2, num_internal_points, dtype=int)
    internal_points = [waypoints_3d[i] for i in indices]
    
    # Combina tutti i punti
    reduced_waypoints = start_point + internal_points + end_point
    
    return reduced_waypoints





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
        # wp_sub = rospy.Subscriber('wp_rosbot', Path, waypoints_callback)
        wp_sub = rospy.Subscriber('wp_rosbot', PoseArray, waypoints_callback)

    goal_sub = rospy.Subscriber('frontier_goal', PoseStamped, goal_callback)

    UGV_pos_sub = rospy.Subscriber('odometry_UGV', Odometry, UGV_position_info)
    UGV_goal_sub = rospy.Subscriber('frontier_goal_UGV', PoseStamped, UGV_goal_callback)

    UGV_goalReached_sub = rospy.Subscriber('ugv_goal_reach', Bool, UGV_goal_reach_callback)

    UAV_updateGoal_sub = rospy.Subscriber('uav_updateGoal', Bool, flag_uav_startOCP_callback)


    path_ocp_pub = rospy.Publisher("OCP_path_planning", Path, queue_size=10)
    posVel_pub = rospy.Publisher("/drone_interface/setpoint_raw", PosVel, queue_size=10)
    go_UGV = rospy.Publisher("/first/navigation_enabler", Bool, queue_size=10)



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
            rospy.loginfo_throttle(5,"Waiting for frontier goal..")
            start_ocp = True
            rate.sleep()  



    while not rospy.is_shutdown():
        if goal_update and np.linalg.norm(robot_position - goal_position) < dist_to_goal :  
            rospy.loginfo_throttle(15,"Goal reached!")
            goal_reached = True
            new_goal_compute = True
            # if not enable_navigation:
            # while enable_navigation:
            #     rate.sleep()
            while not flag_UAVupdateGoal:
                rospy.loginfo_throttle(10,"Waiting for UAV goal update..")
                rate.sleep()
            goal_update = False
            # while not goal_update:
            #     rate.sleep()           
            start_ocp = True
            
            

        if not goal_reached:
            rate.sleep()
            continue

        if start_ocp and new_goal_compute:
            rospy.loginfo("Sono dentro al ciclo per iniziare l'OCP!")
            while not goal_update:
                rospy.loginfo_throttle(10, "Waiting for update UAV goal..")
                rate.sleep()
            new_goal_compute = False
            tot_voxel = size_freeVoxel+size_occVoxel
            voxel_esplorabili = 600/(0.1**3)
            vol_expl = (tot_voxel/voxel_esplorabili)*100
            rospy.loginfo(f"VOLUME ESPLORATO: {vol_expl}")
            # if (not np.all(np.isnan(robot_position))) & (goal_update):
            if(goal_update):
                rospy.loginfo("OCP planner starting")
                if not no_UGV:
                    while not waypoints_ready:
                        rospy.loginfo_throttle(5,"Waiting for rosbot waypoints..")
                        rate.sleep()
                    reduced_wp = downsample_waypoints(waypoints_3d, num_points=100)
                    fitted_points = get_fitted_points(reduced_wp)
                    # fitted_points = get_fitted_points(waypoints_3d)
                    sol, x, y, z, psi, w_psi = uav_planner(robot_position, robot_orientation, goal_position, fitted_points)
                else:
                    sol, x, y, z, psi, w_psi = uav_planner(robot_position,robot_orientation, goal_position)
                rospy.loginfo("Follow OCP Path!")

                # ts, xs = sol.sample(x, grid='integrator')
                # ts, ys = sol.sample(y, grid='integrator')
                # ts, zs = sol.sample(z, grid='integrator')
                ts, xs = sol.sample(x, grid='control')
                ts, ys = sol.sample(y, grid='control')
                ts, zs = sol.sample(z, grid='control')
                data_flat, obstacle_lookup = generate_obstacle_lookup(kdtree_free,kdtree_obs)
                potentials = []
                for i in range(len(xs)):
                    point = np.array([xs[i], ys[i], zs[i]])
                    potential = obstacle_lookup(point)
                    potentials.append(potential)
                potentials = np.array(potentials)
                data_plot = {'xs':xs,'ys':ys,'zs':zs,'ts':ts,'fitted_points':fitted_points,'goal_position':goal_position,'UGV_goal':UGV_goal,'points':points,'potentials':potentials}  
                with open('data_plot_Prov_quattro.pkl', 'wb') as f:
                    pickle.dump(data_plot, f)
                print("Dati salvati con successo in 'data_plot.pkl'.")

                # data_plot = {'xs':xs,'ys':ys,'zs':zs,'ts':ts,'fitted_points':fitted_points,'goal_position':goal_position,'UGV_goal':UGV_goal,'points':points}  
                # with open(f'PROVA_OSTACOLO{l}.pkl', 'wb') as f:
                #     pickle.dump(data_plot, f)
                # print(f"Dati salvati con successo in 'PROVA_RIF{l}.pkl'.")

                #l=l+1

                # # Start UGV Navigation
                enable_navigation = True
                go_UGV.publish(enable_navigation)
                # Start UAV Navigation
                publish_ocp_path(sol, x, y, z)
                follow_path(sol, x, y, z, psi, w_psi)
                goal_reached = False

                if not no_UGV:
                    waypoints_ready = False
                    start_ocp = False

                    if flag_UGVgoalReach:
                        enable_navigation = False
                        go_UGV.publish(enable_navigation)
                    else:
                        # Attendo l'arrivo UGV al goal
                        while not flag_UGVgoalReach: 
                            rospy.loginfo_throttle(15,"attendo che UGV raggiunge Goal!!!!!!!!")      
                            rate.sleep()

                        enable_navigation = False
                        rospy.loginfo(f"enable: {enable_navigation}")
                        go_UGV.publish(enable_navigation)
                

                #plot_traj(sol, x, y, z, fitted_points)

                # goal1 = [goal_position[0]-0.5, goal_position[1], goal_position[2]]
                # robot1 = [robot_position[0], robot_position[1]-0.4, robot_position[2]]
                # sol1, x1, y1, z1, psi1, w_psi1 = uav_planner(robot1, robot_orientation, goal_position, fitted_points)
                # sol1, x1, y1, z1, psi1, w_psi1 = uav_planner(robot_position, robot_orientation, goal1, fitted_points)
                # sol1, x1, y1, z1, psi1, w_psi1 = uav_planner(1,robot_position, robot_orientation, goal_position, fitted_points)


                # goal2 = [goal_position[0], goal_position[1]+0.5, goal_position[2]]
                # robot2 = [robot_position[0], robot_position[1]+0.4, robot_position[2]]
                # sol2, x2, y2, z2, psi2, w_psi2 = uav_planner(robot2, robot_orientation, goal_position, fitted_points)
                # sol2, x2, y2, z2, psi2, w_psi2 = uav_planner(robot_position, robot_orientation, goal2, fitted_points)
                # sol2, x2, y2, z2, psi2, w_psi2 = uav_planner(3,robot_position, robot_orientation, goal_position, fitted_points)

                # robot3 = [robot_position[0]-0.5, robot_position[1]+0.4, robot_position[2]]
                # goal3 = [goal_position[0]-0.4, goal_position[1]+0.4, goal_position[2]]
                # sol3, x3, y3, z3, psi3, w_psi3 = uav_planner(robot3, robot_orientation, goal_position, fitted_points)
                # sol3, x3, y3, z3, psi3, w_psi3 = uav_planner(robot_position, robot_orientation, goal3, fitted_points)
                # sol3, x3, y3, z3, psi3, w_psi3 = uav_planner(8,robot_position, robot_orientation, goal_position, fitted_points)

                # robot4 = [robot_position[0]-0.2, robot_position[1]+0.3, robot_position[2]]
                # sol4, x4, y4, z4, psi4, w_psi4 = uav_planner(robot4, robot_orientation, goal_position, fitted_points)
                # goal4 = [goal_position[0], goal_position[1], goal_position[2]+0.1]
                # sol4, x4, y4, z4, psi4, w_psi4 = uav_planner(robot_position, robot_orientation, goal4, fitted_points)
                # sol4, x4, y4, z4, psi4, w_psi4 = uav_planner(20,robot_position, robot_orientation, goal_position, fitted_points)


                # robot5 = [robot_position[0]-0.3, robot_position[1], robot_position[2]+0.2]
                # sol5, x5, y5, z5, psi5, w_psi5 = uav_planner(robot5, robot_orientation, goal_position, fitted_points)
                # goal5 = [goal_position[0]-0.3, goal_position[1]-0.3, goal_position[2]]
                # sol5, x5, y5, z5, psi5, w_psi5 = uav_planner(robot_position, robot_orientation, goal5, fitted_points)

                # robot6 = [robot_position[0]-0.2, robot_position[1]-0.3, robot_position[2]-0.1]
                # sol6, x6, y6, z6, psi6, w_psi6 = uav_planner(robot6, robot_orientation, goal_position, fitted_points)
                # goal6 = [goal_position[0]+0.3, goal_position[1]+0.3, goal_position[2]-0.2]
                # sol6, x6, y6, z6, psi6, w_psi6 = uav_planner(robot_position, robot_orientation, goal6, fitted_points)


                # # #plot_traj(sol, x, y, z, fitted_points, sol1, x1, y1, z1, sol2, x2, y2, z2, sol3, x3, y3, z3, points, goal1, goal2, goal3)  

                # #ts, xs = sol.sample(x, grid='integrator', refine=10)
                # ts, xs = sol.sample(x, grid='integrator')
                # ts, ys = sol.sample(y, grid='integrator')
                # ts, zs = sol.sample(z, grid='integrator') 
                # ts1, xs1 = sol1.sample(x1, grid='integrator')
                # ts1, ys1 = sol1.sample(y1, grid='integrator')
                # ts1, zs1 = sol1.sample(z1, grid='integrator')
                # ts2, xs2 = sol2.sample(x2, grid='integrator')
                # ts2, ys2 = sol2.sample(y2, grid='integrator')
                # ts2, zs2 = sol2.sample(z2, grid='integrator')
                # ts3, xs3 = sol3.sample(x3, grid='integrator')
                # ts3, ys3 = sol3.sample(y3, grid='integrator')
                # ts3, zs3 = sol3.sample(z3, grid='integrator')
                # ts4, xs4 = sol4.sample(x4, grid='integrator')
                # ts4, ys4 = sol4.sample(y4, grid='integrator')
                # ts4, zs4 = sol4.sample(z4, grid='integrator')
                # ts5, xs5 = sol5.sample(x5, grid='control')
                # ts5, ys5 = sol5.sample(y5, grid='control')
                # ts5, zs5 = sol5.sample(z5, grid='control')
                # ts6, xs6 = sol6.sample(x6, grid='control')
                # ts6, ys6 = sol6.sample(y6, grid='control')
                # ts6, zs6 = sol6.sample(z6, grid='control')
                

                # import matplotlib.pyplot as plt
                # data_flat, obstacle_lookup = generate_obstacle_lookup(kdtree_free,kdtree_obs)
                # potentials = []
                # potentials1 = []
                # potentials2 = []
                # potentials3 = []
                # potentials4 = []
                # potentials5 = []
                # potentials6 = []
                # for i in range(len(xs)):
                #     point = np.array([xs[i], ys[i], zs[i]])
                #     point1 = np.array([xs1[i], ys1[i], zs1[i]])
                #     point2 = np.array([xs2[i], ys2[i], zs2[i]])
                #     point3 = np.array([xs3[i], ys3[i], zs3[i]])
                #     point4 = np.array([xs4[i], ys4[i], zs4[i]])
                #     point5 = np.array([xs5[i], ys5[i], zs5[i]])
                #     point6 = np.array([xs6[i], ys6[i], zs6[i]])

                #     potential = obstacle_lookup(point)
                #     potential1 = obstacle_lookup(point1)
                #     potential2 = obstacle_lookup(point2)
                #     potential3 = obstacle_lookup(point3)
                #     potential4 = obstacle_lookup(point4)
                #     potential5 = obstacle_lookup(point5)
                #     potential6 = obstacle_lookup(point6)
                #     potentials.append(potential)
                #     potentials1.append(potential1)
                #     potentials2.append(potential2)
                #     potentials3.append(potential3)
                #     potentials4.append(potential4)
                #     potentials5.append(potential5)
                #     potentials6.append(potential6)
                # potentials = np.array(potentials)
                # potentials1 = np.array(potentials1)
                # potentials2 = np.array(potentials2)
                # potentials3 = np.array(potentials3)
                # potentials4 = np.array(potentials4)
                # potentials5 = np.array(potentials5)
                # potentials6 = np.array(potentials6)
                # # fig, ax = plt.subplots(figsize=(10, 6))
                # # potentials = np.squeeze(potentials)
                # # potentials1 = np.squeeze(potentials1)
                # # potentials2 = np.squeeze(potentials2)
                # # potentials3 = np.squeeze(potentials3)
                # # ax.plot(ts, potentials, linestyle='-', color='r', marker='o', label="Trajectory ")  
                # # ax.plot(ts1, potentials1, linestyle='-', color='m', marker='o', label="Trajectory 1")  
                # # ax.plot(ts2, potentials2, linestyle='-', color='y', marker='o',label="Trajectory 2")  
                # # ax.plot(ts3, potentials3, linestyle='-', color='c', marker='o', label="Trajectory 3")  

                # # ax.set_xlabel('Time (ts)')
                # # ax.set_ylabel('Potential')
                # # plt.title('Trajectory and Corresponding Potential',fontsize=16, fontweight='bold')
                # # plt.grid(True)
                # # plt.show()

                # data_diffPosInit={'xs':xs,'ys':ys,'zs':zs,'xs1':xs1,'ys1':ys1,'zs1':zs1,'xs2':xs2,'ys2':ys2,'zs2':zs2,'xs3':xs3,'ys3':ys3,'zs3':zs3,'fitted_points': fitted_points, 'points':points,
                #       'goal_position':goal_position,'xs4':xs4,'ys4':ys4,'zs4':zs4,'xs5':xs5,'ys5':ys5,'zs5':zs5,'xs6':xs6,'ys6':ys6,'zs6':zs6, 'ts':ts, 'ts1':ts1,'ts2':ts2,
                #       'ts3':ts3, 'ts4':ts4, 'ts5':ts5, 'ts6':ts6,'potentials':potentials,'potentials1':potentials1,'potentials2':potentials2,'potentials3':potentials3,
                #       'potentials4':potentials4,'potentials5':potentials5,'potentials6':potentials6}

                # data_diffPosInit={'xs':xs,'ys':ys,'zs':zs,'xs1':xs1,'ys1':ys1,'zs1':zs1,'xs2':xs2,'ys2':ys2,'zs2':zs2,'xs3':xs3,'ys3':ys3,'zs3':zs3,'fitted_points': fitted_points, 'points':points,
                #       'goal_position':goal_position,'xs4':xs4,'ys4':ys4,'zs4':zs4,'xs5':xs5,'ys5':ys5,'zs5':zs5,'xs6':xs6,'ys6':ys6,'zs6':zs6, 'ts':ts, 'ts1':ts1,'ts2':ts2,
                #       'ts3':ts3, 'ts4':ts4, 'ts5':ts5, 'ts6':ts6,'UGV_goal':UGV_goal}
                
                # with open('data_diffGoal_GIUSTO.pkl', 'wb') as f:
                #     pickle.dump(data_diffPosInit, f)
                # print("Dati salvati con successo in 'data.pkl'.")



                # data_prov={'xs':xs,'ys':ys,'zs':zs,'xs1':xs1,'ys1':ys1,'zs1':zs1,'xs2':xs2,'ys2':ys2,'zs2':zs2,'xs3':xs3,'ys3':ys3,'zs3':zs3,'xs4':xs4,'ys4':ys4,'zs4':zs4,'fitted_points': fitted_points, 'points':points,
                #       'goal_position':goal_position,'ts':ts, 'ts1':ts1,'ts2':ts2,'ts3':ts3,'goal_position':goal_position,'UGV_goal':UGV_goal}
                # with open('PROVA_diffPesiCOMPLETA.pkl', 'wb') as f:
                #     pickle.dump(data_prov, f)
                # print("Dati salvati con successo in 'data.pkl'.")

                # # with open('data_diffPosInit.pkl', 'wb') as f:
                # #     pickle.dump(data_diffPosInit, f)

                # print("Dati salvati con successo in 'data.pkl'.")
                
                    
                    


        rate.sleep()
