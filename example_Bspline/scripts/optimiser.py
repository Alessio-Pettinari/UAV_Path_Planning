#!/usr/bin/env python3

import rospy
import numpy as np
import casadi as cs
from casadi.tools import struct_SX, entry

from symbolic_trajectory import SymbolicTrajectory, ActualTrajectory


class Optimiser:
    def __init__(self, n_control_points=10, spline_degree=4, constrained_samples=50, minimise_acceleration=False):
        assert n_control_points - spline_degree > 1

        zone_3_bounds = rospy.get_param("/zone3")
        x_min, x_max = zone_3_bounds['x']
        y_min, y_max = zone_3_bounds['y']
        v_min, v_max = rospy.get_param("/phisics_param/vel_limits")
        a_min, a_max = rospy.get_param("/phisics_param/acc_limits")

        traj_sx = SymbolicTrajectory(n_control_points, spline_degree)

        # set knot values equispaced in [0, 1]
        knot_vector = [0. for _ in range(spline_degree + 1)] + \
                      np.linspace(0., 1., n_control_points - spline_degree, endpoint=False).tolist()[1:] + \
                      [1. for _ in range(spline_degree + 1)]

        # initialise numerical spline
        self.optimal_trajectory = ActualTrajectory(n_control_points, spline_degree)
        self.optimal_trajectory.numerical_knots = knot_vector

        # set constraint functions
        final_time = traj_sx.p['final_time']
        positions, velocities, accelerations = [], [], []
        for u_i in np.linspace(0., 1., constrained_samples, endpoint=False):
            position_i = traj_sx.trajectory_in_u(cs.DM(u_i), traj_sx.p, knot_vector)
            velocity_i = traj_sx.d_trajectory_in_u(cs.DM(u_i), traj_sx.p, knot_vector) / final_time
            acceleration_i = traj_sx.dd_trajectory_in_u(cs.DM(u_i), traj_sx.p, knot_vector) / final_time

            positions.append(position_i)
            velocities.append(velocity_i)
            accelerations.append(acceleration_i)

        g = struct_SX([
            entry('positions', expr=positions),
            entry('velocities', expr=velocities),
            entry('accelerations', expr=accelerations),
        ])

        # set cost
        cost = final_time ** 2
        if minimise_acceleration:
            cost += .1 * sum([a[0] ** 2 for a in accelerations]) + .1 * sum([a[1] ** 2 for a in accelerations])

        # full list of ipopt options: https://coin-or.github.io/Ipopt/OPTIONS.html
        solver_options = {
            # 'calc_lam_p': False,
            # 'monitor': 'nlp_f',  # this prints cost output at each iteration
            # 'ipopt': {'max_iter': 100,
            #           # 'tol': 1e-4,
            #           # 'constr_viol_tol': 1e-4,
            #           # 'acceptable_tol': 1e-2,
            #           'max_cpu_time': .2,
            #           'linear_solver': 'ma57',
            #           'warm_start_init_point': 'yes',
            #           'bound_mult_init_method': 'mu-based',
            #           'mu_strategy': 'adaptive',
            #           'nlp_scaling_method': 'none',
            #           }
        }

        # solver setup
        self.optimization_vars = traj_sx.p
        nlp = {"x": self.optimization_vars, "f": cost, "g": g}
        self.solver = cs.nlpsol("solver", "ipopt", nlp, solver_options)

        # inequality constraints
        self.optimization_vars_lower_bound = self.optimization_vars(-cs.inf)
        self.optimization_vars_lower_bound['final_time'] = 0. + 0.1  # can only be positive
        self.optimization_vars_upper_bound = self.optimization_vars(cs.inf)
        # self.optimization_vars_lower_bound['trajectory_ctrl_pts'] = -100.
        # self.optimization_vars_upper_bound['trajectory_ctrl_pts'] = 100.

        self.g_upper_bound, self.g_lower_bound = g(), g()
        # Zone 3 bounds TODO: consider drone bounding box
        for i in range(constrained_samples):
            self.g_lower_bound['positions', i] = [x_min, y_min]
            self.g_upper_bound['positions', i] = [x_max, y_max]

        # velocity and acceleration constraints as imposed in Toppra
        self.g_lower_bound['velocities'] = v_min
        self.g_upper_bound['velocities'] = v_max
        self.g_lower_bound['accelerations'] = a_min
        self.g_upper_bound['accelerations'] = a_max

    def compute_optimal_trajectory(self, initial_position, initial_velocity, initial_acceleration,
                                   launch_position, launch_velocity):
        # initial values
        self.g_lower_bound['positions', 0] = initial_position
        self.g_upper_bound['positions', 0] = initial_position
        self.g_lower_bound['velocities', 0] = initial_velocity
        self.g_upper_bound['velocities', 0] = initial_velocity
        self.g_lower_bound['accelerations', 0] = initial_acceleration
        self.g_upper_bound['accelerations', 0] = initial_acceleration

        # launch values
        self.g_lower_bound['positions', -1] = launch_position
        self.g_upper_bound['positions', -1] = launch_position
        self.g_lower_bound['velocities', -1] = launch_velocity
        self.g_upper_bound['velocities', -1] = launch_velocity
        self.g_lower_bound['accelerations', -1] = [0., 0.]
        self.g_upper_bound['accelerations', -1] = [0., 0.]

        initial_guess = self.optimization_vars(0.)

        # solve the problem
        res = self.solver(x0=initial_guess,
                          lbx=self.optimization_vars_lower_bound, ubx=self.optimization_vars_upper_bound,
                          lbg=self.g_lower_bound, ubg=self.g_upper_bound)

        opt_solution = self.optimization_vars(res["x"])
        self.optimal_trajectory.numerical_p['final_time'] = opt_solution['final_time']
        self.optimal_trajectory.numerical_p['trajectory_ctrl_pts'] = opt_solution['trajectory_ctrl_pts']

        print(f"Total trajectory time: {opt_solution['final_time']} seconds")

        return self.optimal_trajectory


if __name__ == '__main__':
    optimiser = Optimiser(n_control_points=10, spline_degree=4,
                          constrained_samples=50,
                          minimise_acceleration=False)

    computed_trajectory = optimiser.compute_optimal_trajectory([1., 3.5], [1., 0.], [0., 0.],
                                                               [8.5, 3.5], [0., 0.])

    # evaluate results
    computed_trajectory.plot_trajectory(show_ctrl_points=False)
    computed_trajectory.plot_velocities()
    computed_trajectory.plot_accelerations()
