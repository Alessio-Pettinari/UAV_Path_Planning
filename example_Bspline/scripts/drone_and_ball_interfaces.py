#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Float32
from geometry_msgs.msg import Transform, Twist, PointStamped
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

import symbolic_trajectory


class DroneInterface:
    def __init__(self):
        self.current_xy = [0., 0.]
        self.current_vel = [0., 0.]

        self._prev_vel = [0., 0.]
        self.current_acc = [0., 0.]

        self.current_z = -1
        self.traj_pub = rospy.Publisher("/red/position_hold/trajectory", MultiDOFJointTrajectoryPoint, queue_size=1)
        self.visualize_wp_pub = rospy.Publisher("/visualization/phase23/wp", MultiDOFJointTrajectoryPoint, queue_size=1)
        self.pose_sub = rospy.Subscriber("/red/odometry", Odometry, self.pose_callback)

        self.waypoint_msg = MultiDOFJointTrajectoryPoint()
        self.waypoint_msg.transforms = [Transform()]
        self.waypoint_msg.velocities = [Twist()]
        self.waypoint_msg.accelerations = [Twist()]
        self.waypoint_msg.time_from_start = rospy.Duration()

    def wait_for_first_pose(self):
        r = rospy.Rate(100)  # Hz
        while self.current_z == -1:
            r.sleep()

    def set_waypoint_x(self, x, dx=0., ddx=0.):
        self.waypoint_msg.transforms[0].translation.x = x
        self.waypoint_msg.velocities[0].linear.x = dx
        self.waypoint_msg.accelerations[0].linear.x = ddx

    def set_waypoint_y(self, y, dy=0., ddy=0.):
        self.waypoint_msg.transforms[0].translation.y = y
        self.waypoint_msg.velocities[0].linear.y = dy
        self.waypoint_msg.accelerations[0].linear.y = ddy

    def set_waypoint_z(self, z, dz=0., ddz=0.):
        self.waypoint_msg.transforms[0].translation.z = z
        self.waypoint_msg.velocities[0].linear.z = dz
        self.waypoint_msg.accelerations[0].linear.z = ddz

    def set_waypoint_yaw(self, psi):  # , d_psi=0., dd_psi=0.):
        # self.waypoint_msg.transforms[0].rotation.x = 0.
        # self.waypoint_msg.transforms[0].rotation.y = 0.
        self.waypoint_msg.transforms[0].rotation.z = np.sin(psi / 2)
        self.waypoint_msg.transforms[0].rotation.w = np.cos(psi / 2)

        # self.waypoint_msg.velocities[0].angular
        # self.waypoint_msg.accelerations[0].angular

    def publish_waypoint(self, time_from_start=None):
        if time_from_start is not None:
            self.waypoint_msg.time_from_start = time_from_start
        self.traj_pub.publish(self.waypoint_msg)
        self.visualize_wp_pub.publish(self.waypoint_msg)

    def pose_callback(self, msg: Odometry):
        current_position = msg.pose.pose.position
        self.current_xy = [current_position.x, current_position.y]
        self.current_z = current_position.z

        current_velocity = msg.twist.twist.linear
        self._prev_vel = self.current_vel
        self.current_vel = [current_velocity.x, current_velocity.y]

        # TODO: maybe acceleration values should be filtered
        self.current_acc = [self.current_vel[i] - self._prev_vel[i] for i in range(2)]


class BallInterface:
    def __init__(self):
        self.current_xyz = np.array([0., 0., 0.])
        self.position_history = []

        self.pose_sub = rospy.Subscriber("/red/ball/position", PointStamped, self.pose_callback)
        self.magnet_pub = rospy.Publisher("/red/uav_magnet/gain", Float32, queue_size=1, latch=True)

    def pose_callback(self, msg: PointStamped):
        self.current_xyz = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.position_history.append(self.current_xyz)

    def get_min_distance(self, marker_position):
        if isinstance(marker_position, list):
            marker_position = np.array(marker_position)
        return np.min([np.linalg.norm(p - marker_position) for p in self.position_history])

    def release_magnet(self):
        ball_msg = Float32(0.)
        self.magnet_pub.publish(ball_msg)


def execute_trajectory(drone: DroneInterface, trajectory: symbolic_trajectory.ActualTrajectory,
                       timescale_factor=1., early_quit_seconds=0., log=False):
    if timescale_factor != 1.:
        print(f"-> Corrected trajectory time: {trajectory.numerical_p['final_time'] / timescale_factor} seconds")

    des_positions, des_velocities = [], []
    assumed_positions, assumed_velocities = [], []

    r = rospy.Rate(100)  # Hz
    start_t, t = rospy.Time.now(), 0.
    while not rospy.is_shutdown() and t < trajectory.numerical_p['final_time'] - early_quit_seconds:
        next_waypoint_p = trajectory.get_position(t)
        next_waypoint_v = trajectory.get_velocity(t)
        next_waypoint_a = trajectory.get_acceleration(t)

        drone.set_waypoint_x(next_waypoint_p[0], next_waypoint_v[0], next_waypoint_a[0])
        drone.set_waypoint_y(next_waypoint_p[1], next_waypoint_v[1], next_waypoint_a[1])

        time_from_start = rospy.Time.now() - start_t
        drone.publish_waypoint(time_from_start)

        t = (time_from_start.secs + 1e-9 * time_from_start.nsecs) * timescale_factor

        # log current and desired positions
        if log:
            assumed_positions.append(drone.current_xy)
            assumed_velocities.append(drone.current_vel)
            des_positions.append(next_waypoint_p)
            des_velocities.append(next_waypoint_v)

        r.sleep()

    return [assumed_positions, assumed_velocities, des_positions, des_velocities]
