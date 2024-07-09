#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import casadi as cs
from casadi.tools import struct_symSX, entry

from geomdl import fitting as geomdl_fitting
from geomdl.operations import length_curve as geomdl_length_curve
from geomdl.visualization.VisMPL import VisCurve2D


class SymbolicTrajectory:
    """ Symbolic spline function parametrized for generic control points and knot vector lists.
        Fixed elements are only the number of control points (i.e. number of points to interpolate) and the spline degree. """

    def __init__(self, n_control_points=5, spline_degree=4):
        self.n_control_points = n_control_points
        self.spline_degree = spline_degree

        u = cs.SX.sym('u')  # curve parameter
        n_knot_spans = self.n_control_points + self.spline_degree

        # TODO: test a fixed assignment of the knot_vector, e.g. with equidistant values
        knots = cs.SX.sym("knot_vector", n_knot_spans + 1)

        p = struct_symSX([(
            entry("trajectory_ctrl_pts", struct=struct_symSX(["x", "y"]), repeat=self.n_control_points),
            entry("final_time")
        )])

        " Basis functions "
        N_0 = []  # zero-order basis functions (step functions)
        for i in range(n_knot_spans):
            knot_i = knots[i]
            knot_i_next = knots[i + 1]
            N_0.append(cs.if_else(cs.logic_and(u >= knot_i, u < knot_i_next), 1, 0))
        N = [N_0]  # greater orders are linear interpolation of the previous ones
        for current_deg in range(1, self.spline_degree + 1):
            N_current, N_prev = [], N[-1]
            for i in range(n_knot_spans - current_deg):  # each p-th order has one function less than the (p-1) one
                knot_i = knots[i]
                knot_i_next = knots[i + 1]
                knot_i_deg = knots[i + current_deg]
                knot_i_deg_next = knots[i + current_deg + 1]
                c_1 = cs.if_else(knot_i_deg == knot_i, 0,
                                 (u - knot_i) / (knot_i_deg - knot_i))
                c_2 = cs.if_else(knot_i_deg_next == knot_i_next, 0,
                                 (knot_i_deg_next - u) / (knot_i_deg_next - knot_i_next))
                N_current.append(c_1 * N_prev[i] + c_2 * N_prev[i + 1])
            N.append(N_current)
        Np = N[-1]

        " Basis function first derivatives "
        current_deg = self.spline_degree
        Np_first, N_prev = [], N[self.spline_degree - 1]
        for i in range(n_knot_spans - current_deg):
            knot_i = knots[i]
            knot_i_next = knots[i + 1]
            knot_i_deg = knots[i + current_deg]
            knot_i_deg_next = knots[i + current_deg + 1]
            c_1 = cs.if_else(knot_i_deg == knot_i, 0,
                             current_deg / (knot_i_deg - knot_i))
            c_2 = cs.if_else(knot_i_deg_next == knot_i_next, 0,
                             current_deg / (knot_i_deg_next - knot_i_next))
            Np_first.append(c_1 * N_prev[i] - c_2 * N_prev[i + 1])

        " Trajectory spline function "
        t = cs.SX.sym('t')  # elapsed time (u = t / p['final_time'])

        # cs.sum2: sum columns (and cs.sum1 should sum rows)
        trajectory_curve_expr = cs.sum2(cs.horzcat(*[Np[i] * p['trajectory_ctrl_pts', i]
                                                     for i in range(self.n_control_points)]))
        self.trajectory_in_u = cs.Function('C_u', [u, p, knots], [trajectory_curve_expr])
        self._trajectory_in_t = cs.Function('C_t', [t, p, knots], [self.trajectory_in_u(t / p['final_time'], p, knots)])

        " Time derivatives "
        trajectory_first_derivative_expr = cs.sum2(cs.horzcat(*[Np_first[i] * p['trajectory_ctrl_pts', i]
                                                                for i in range(self.n_control_points)]))
        self.d_trajectory_in_u = cs.Function('dC_u', [u, p, knots], [trajectory_first_derivative_expr])
        self._d_trajectory_in_t = \
            cs.Function('dC_t', [t, p, knots], [self.d_trajectory_in_u(t / p['final_time'], p, knots) / p['final_time']])

        # TODO: formulate derivatives in closed form instead of computing jacobians
        self.dd_trajectory_in_u = cs.Function('ddC_u', [u, p, knots], [cs.jacobian(self.d_trajectory_in_u(u, p, knots), u)])
        self._dd_trajectory_in_t = \
            cs.Function('ddC_t', [t, p, knots],
                        [cs.jacobian(self.d_trajectory_in_u(t / p['final_time'], p, knots), t) / p['final_time']])

        " other necessary elements "
        self.knots = knots
        self.p = p

    def dx_du_norm(self, u, p, knots):
        if isinstance(u, int) or isinstance(u, float):
            u = u if u < 1. else 1. - 1.e-6  # in general, u has to stay in [0,1)
        return cs.norm_2(self.d_trajectory_in_u(u, p, knots))

    def get_position(self, t, p, knots):
        if isinstance(t, int) or isinstance(t, float):
            if t >= p['final_time']:
                print(f"[get_position] Parameter 't' (= {t}) should not exceed final time {p['final_time']}. Reducing it.")
                t = p['final_time'] - 1.e-6
        return self._trajectory_in_t(t, p, knots)

    def get_velocity(self, t, p, knots):
        if isinstance(t, int) or isinstance(t, float):
            if t >= p['final_time']:
                print(f"[get_velocity] Parameter 't' (= {t}) should not exceed final time {p['final_time']}. Reducing it.")
                t = p['final_time'] - 1.e-6
        return self._d_trajectory_in_t(t, p, knots)

    def get_acceleration(self, t, p, knots):
        if isinstance(t, int) or isinstance(t, float):
            if t >= p['final_time']:
                print(f"[get_acceleration] Parameter 't' (= {t}) should not exceed final time {p['final_time']}. Reducing it.")
                t = p['final_time'] - 1.e-6
        return self._dd_trajectory_in_t(t, p, knots)

    def tangent(self, u, p, knots):
        if isinstance(u, int) or isinstance(u, float):
            u = u if u < 1. else 1. - 1.e-6
        return self.d_trajectory_in_u(u, p, knots) / self.dx_du_norm(u, p, knots)

    def normal(self, u, p, knots):
        if isinstance(u, int) or isinstance(u, float):
            u = u if u < 1. else 1. - 1.e-6
        rot_90_deg = cs.blockcat([[0, -1], [1, 0]])
        return rot_90_deg @ self.tangent(u, p, knots)

    def curvature(self, u, p, knots):
        if isinstance(u, int) or isinstance(u, float):
            u = u if u < 1. else 1. - 1.e-6
        return self.dd_trajectory_in_u(u, p, knots).T @ self.normal(u, p, knots) / self.dx_du_norm(u, p, knots) ** 2


class NumericalFitTrajectory:
    """ Interpolate any given trajectory points with the desired spline degree.
        The fit_data method can be used iteratively, without worrying about repetitions: the solution of the previous
        call is cached, so the same spline is not recomputed twice. """

    def __init__(self, spline_degree=3):
        self.spline_degree = spline_degree
        self.trajectory_spline, self.estimated_length, self.fitted_points = [None] * 3

    def fit_data(self, points_to_fit: list):
        """ Fit the given dataset.
            Do not recompute anything if the data to fit is the same as the previous call.
            Return whether if data has changed or not. """
        if points_to_fit == self.fitted_points:
            return False

        " trajectory curve computation "
        self.trajectory_spline = geomdl_fitting.interpolate_curve(points_to_fit, self.spline_degree)

        " values for plotting and sampling used by 'evaluate' function "
        self.trajectory_spline.delta = 0.001
        self.trajectory_spline.vis = VisCurve2D()

        " length of the computed curve, approximated with the sum of distances from each evaluated point and its successor "
        # reference: https://nurbs-python.readthedocs.io/en/latest/module_operations.html#geomdl.operations.length_curve
        self.estimated_length = geomdl_length_curve(self.trajectory_spline)

        self.fitted_points = np.array(points_to_fit).T
        return True

    # def get_nearest_u(self, point: np.ndarray):
    #     if point.shape == (2, 1):
    #         point = point.T
    #     distances = [np.linalg.norm(point - p_i) for p_i in self.trajectory_spline.evalpts]
    #     i = np.argmin(distances)
    #     return i * self.trajectory_spline.delta


class ActualTrajectory(SymbolicTrajectory):
    """ Numerical instance of SymbolicTrajectory, i.e. an actual curve that fits the given data. """

    def __init__(self, n_control_points=5, spline_degree=3):
        super().__init__(n_control_points=n_control_points, spline_degree=spline_degree)
        self.numerical_p, self.numerical_knots = self.p(), []
        self.estimated_length = None
        self.numerical_fit_spline = NumericalFitTrajectory(spline_degree)  # instantiate the numerical fitter

    def fit_points(self, trajectory_points: list, final_time):
        assert len(trajectory_points) == self.n_control_points, \
            f"Number of point ({len(trajectory_points)}) to interpolate must be equal to control points ({self.n_control_points})"

        self.numerical_p['final_time'] = final_time
        something_changed = self.numerical_fit_spline.fit_data(trajectory_points)

        if something_changed:
            self.numerical_knots = self.numerical_fit_spline.trajectory_spline.knotvector
            self.numerical_p['trajectory_ctrl_pts'] = self.numerical_fit_spline.trajectory_spline.ctrlpts

            self.estimated_length = self.numerical_fit_spline.estimated_length

    def get_position(self, t, p=None, knots=None):
        if p is None:
            p = self.numerical_p
        if knots is None:
            knots = self.numerical_knots
        return super(ActualTrajectory, self).get_position(t, p, knots).toarray().flatten()

    def get_velocity(self, t, p=None, knots=None):
        if p is None:
            p = self.numerical_p
        if knots is None:
            knots = self.numerical_knots
        return super(ActualTrajectory, self).get_velocity(t, p, knots).toarray().flatten()

    def get_acceleration(self, t, p=None, knots=None):
        if p is None:
            p = self.numerical_p
        if knots is None:
            knots = self.numerical_knots
        return super(ActualTrajectory, self).get_acceleration(t, p, knots).toarray().flatten()

    def plot_trajectory(self, t_0=0, t_f=None, show_ctrl_points=False, n_samples=100):
        trajectory_points = []

        if t_f is None or t_f > self.numerical_p['final_time']:
            t_f = float(self.numerical_p['final_time'])

        t_interval = np.linspace(t_0, t_f, n_samples, endpoint=False)
        for t_k in t_interval:
            trajectory_point_k = self.get_position(t_k).reshape(2, 1)
            trajectory_points = \
                np.append(trajectory_points, trajectory_point_k, axis=1) if len(trajectory_points) != 0 else trajectory_point_k

        _, ax = plt.subplots()

        if show_ctrl_points:
            ctrl_points = np.array(self.numerical_p['trajectory_ctrl_pts']).reshape(-1, 2).T
            ax.plot(ctrl_points[0, :], ctrl_points[1, :], color='blue')
            ax.scatter(ctrl_points[0, :], ctrl_points[1, :], color='blue')

        ax.plot(trajectory_points[0, :], trajectory_points[1, :], color='black', ls='--')

        if self.numerical_fit_spline.fitted_points is not None:
            ax.scatter(self.numerical_fit_spline.fitted_points[0, :],
                       self.numerical_fit_spline.fitted_points[1, :], color='red', s=15)

        ax.set(xlabel="x [m]", ylabel="y [m]", title="Interpolated trajectory")
        ax.set_aspect(1.)
        plt.grid(True)
        plt.show()

    def plot_velocities(self, t_0=0, t_f=None, n_samples=100):
        vel_x, vel_y = [], []

        if t_f is None or t_f > self.numerical_p['final_time']:
            t_f = float(self.numerical_p['final_time'])

        t_interval = np.linspace(t_0, t_f, n_samples, endpoint=False)
        for t_k in t_interval:
            velocity_k = self.get_velocity(t_k)
            vel_x.append(velocity_k[0])
            vel_y.append(velocity_k[1])

        _, ax = plt.subplots()
        ax.plot(t_interval, vel_x)
        ax.set(xlabel="t [s]", ylabel="v_x [m/s]", title="Velocity along global x")
        plt.grid(True)
        plt.show()

        _, ax = plt.subplots()
        ax.plot(t_interval, vel_y)
        ax.set(xlabel="t [s]", ylabel="v_y [m/s]", title="Velocity along global y")
        plt.grid(True)
        plt.show()

    def plot_accelerations(self, t_0=0, t_f=None, n_samples=100):
        acc_x, acc_y = [], []

        if t_f is None or t_f > self.numerical_p['final_time']:
            t_f = float(self.numerical_p['final_time'])

        t_interval = np.linspace(t_0, t_f, n_samples, endpoint=False)
        for t_k in t_interval:
            acceleration_k = self.get_acceleration(t_k)
            acc_x.append(acceleration_k[0])
            acc_y.append(acceleration_k[1])

        _, ax = plt.subplots()
        ax.plot(t_interval, acc_x)
        ax.set(xlabel="t [s]", ylabel="a_x [m/s^2]", title="Acceleration along global x")
        plt.grid(True)
        plt.show()

        _, ax = plt.subplots()
        ax.plot(t_interval, acc_y)
        ax.set(xlabel="t [s]", ylabel="a_y [m/s^2]", title="Acceleration along global y")
        plt.grid(True)
        plt.show()

    def plot_curvature(self, u_0=0, u_f=1, n_samples=None):
        curvature_points = []

        if n_samples is None:
            n_samples = int(np.ceil(100 * (u_f - u_0)))

        u_interval = np.linspace(u_0, u_f, n_samples, endpoint=False)
        for u_k in u_interval:
            curvature_points.append([u_k, float(self.curvature(u_k, self.numerical_p, self.numerical_knots))])
        curvature_points = np.array(curvature_points).T

        _, ax = plt.subplots()
        ax.scatter(curvature_points[0, :], curvature_points[1, :], color='black', s=5)
        ax.set(xlabel="Spline parameter u", ylabel="curvature", title="Trajectory curvature")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    actual_trajectory = ActualTrajectory(n_control_points=7, spline_degree=5)
    # a NumericalFitSpline object is used internally to obtain numerical values for the symbolic parametrized splines

    actual_trajectory.fit_points([
        [-10., 0.],
        [-9.5, .3],
        [-8.5, -.3],
        [-8., 0.],
        [-7.5, .3],
        [-6.5, 0.],
        [-6, 0.]
    ], 10.)

    actual_trajectory.plot_trajectory(show_ctrl_points=True)
    actual_trajectory.plot_velocities()
    actual_trajectory.plot_accelerations()
    # actual_trajectory.plot_curvature()

    print(f"Estimated trajectory length: {actual_trajectory.estimated_length} meters")
