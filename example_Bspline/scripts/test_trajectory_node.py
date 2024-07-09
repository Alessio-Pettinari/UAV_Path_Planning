#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt

import symbolic_trajectory
from drone_and_ball_interfaces import DroneInterface


if __name__ == '__main__':
    rospy.init_node('test_trajectory_node')

    drone = DroneInterface()

    # TODO: wait for '/red/carrot/status' == 'HOLD'

    drone.wait_for_first_pose()

    des_positions, des_velocities = [], []
    assumed_positions, assumed_velocities = [], []
    drone.set_waypoint_z(drone.current_z)

    t_f = 10.  # seconds
    actual_trajectory = symbolic_trajectory.ActualTrajectory(n_control_points=7, spline_degree=5)
    actual_trajectory.fit_points([
        drone.current_xy,
        [-9.5, .3],
        [-8.5, -.3],
        [-8., 0.],
        [-7.5, .3],
        [-6.5, 0.],
        [-6, 0.]
    ], t_f)

    r = rospy.Rate(100)  # Hz

    print("Start moving...")
    start_t, t = rospy.Time.now(), 0.
    while not rospy.is_shutdown() and t < t_f:
        next_waypoint_p = actual_trajectory.get_position(t)
        next_waypoint_v = actual_trajectory.get_velocity(t)
        next_waypoint_a = actual_trajectory.get_acceleration(t)

        drone.set_waypoint_x(next_waypoint_p[0], next_waypoint_v[0], next_waypoint_a[0])
        drone.set_waypoint_y(next_waypoint_p[1], next_waypoint_v[1], next_waypoint_a[1])

        time_from_start = rospy.Time.now() - start_t
        drone.publish_waypoint(time_from_start)

        t = time_from_start.secs + 1e-9 * time_from_start.nsecs

        # log current and desired positions
        assumed_positions.append(drone.current_xy)
        assumed_velocities.append(drone.current_vel)
        des_positions.append(next_waypoint_p)
        des_velocities.append(next_waypoint_v)

        r.sleep()
    print("Trajectory sequence finished.")

    assumed_positions = np.array(assumed_positions).T
    assumed_velocities = np.array(assumed_velocities).T
    des_positions = np.array(des_positions).T
    des_velocities = np.array(des_velocities).T

    " plot trajectory tracking "
    _, ax = plt.subplots()
    ax.plot(assumed_positions[0, :], assumed_positions[1, :])
    ax.plot(des_positions[0, :], des_positions[1, :])
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Position tracking result")
    ax.set_aspect(1.)
    plt.grid(True)
    plt.show()

    " plot velocity tracking "
    t = list(range(assumed_velocities.shape[1]))
    _, ax = plt.subplots()
    ax.plot(t, assumed_velocities[0, :])
    ax.plot(t, des_velocities[0, :])
    ax.set(xlabel="index", ylabel="v_x [m/s]", title="Velocity tracking result")
    plt.grid(True)
    plt.show()
    _, ax = plt.subplots()
    ax.plot(t, assumed_velocities[1, :])
    ax.plot(t, des_velocities[1, :])
    ax.set(xlabel="index", ylabel="v_y [m/s]", title="Velocity tracking result")
    plt.grid(True)
    plt.show()
