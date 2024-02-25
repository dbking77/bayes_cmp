#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Derek King

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard libraries
import argparse
from math import cos, sin, radians, isnan
import time
from typing import List

# Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Local
from ekf import EKF
from pf import PF
from ukf import UKF
from util import Point, plot_cov_ellipse, normalize_angles
from vehicle import Vehicle


class FilterLog:
    def __init__(self, name, filter, color, linestyle):
        self.filter = filter
        self.name = name
        self.states = []
        self.covs = []
        self.color = color
        self.linestyle = linestyle

    def predict(self, v, phi):
        state, cov = self.filter.predict(v, phi)
        self.states.append(state)
        self.covs.append(cov)

    def measure(self, meas):
        state, cov = self.filter.measure(meas)
        self.states.append(state)
        self.covs.append(cov)


class Sim:
    def __init__(self, *, vehicle: Vehicle, features: List[Point],
                 start_cov, Q, R, figsize, viz_rate, runtime, save_base_fn):
        self.vehicle = vehicle
        self.features = features
        self.start_cov = start_cov
        self.R = R
        self.Q = Q
        self.figsize = figsize
        self.runtime = runtime
        self.save_base_fn = save_base_fn

        # true state always starts at zero, but filters are given noise starting location
        self.start_state = np.zeros(3)
        self.filter_start_state = np.random.multivariate_normal(
            self.start_state, start_cov)

        self.filters = {}
        filter_classes = [
            ('ekf', EKF, (1, 0, 0), '-'),
            ('ukf', UKF, (0, 1, 0), '-.'),
            ('pf', PF, (0, 0, 1), '--'),
        ]
        for name, filter_class, color, linestyle in filter_classes:
            filter = filter_class(
                self.vehicle, self.filter_start_state, start_cov, self.R, self.Q)
            self.filters[name] = FilterLog(name, filter, color, linestyle)

        self.true_states = []
        self.phis = []
        self.true_meas = []
        self.noisy_meas = []
        self.ts = []
        self.stage = "NONE"
        self.viz_period = 1.0 / viz_rate
        self.next_viz_t = time.time()
        self.plot_filter_iteration = 0

    def viz_pause(self):
        # have phi = 0.1*sin(t)
        plt.show(block=False)
        self.next_viz_t += self.viz_period
        dt = max(0.001, self.next_viz_t - time.time())
        plt.pause(dt)

    def run(self):
        self.next_viz_t = time.time()

        v = 1.0

        def calc_phi(t):
            return radians(30)
        t = 0.0
        state = self.start_state
        while t < self.runtime:
            # motion update (predict)
            self.stage = "PREDICT"
            phi = calc_phi(t)
            phi = self.vehicle.limit_phi(phi)
            e = np.random.multivariate_normal(np.zeros(3), self.R)
            state = self.vehicle.update(state, v, phi) + e
            self.true_states.append(state)
            for filter in self.filters.values():
                filter.predict(v, phi)
            self.phis.append(phi)
            self.plot_filters()
            self.viz_pause()

            # measurement update
            self.stage = "MEASURE"
            true_meas = self.vehicle.measure(state)
            meas_noise = np.random.normal(
                0, float(self.Q), size=len(self.features))
            noisy_meas = true_meas + meas_noise
            self.true_meas.append(true_meas)
            self.noisy_meas.append(noisy_meas)
            for filter in self.filters.values():
                filter.measure(noisy_meas)
            t += self.vehicle.dt
            self.ts.append(t)
            self.true_states.append(state)
            self.plot_filters()
            self.viz_pause()

    def plot_vehicle(self, ax):
        ln = self.vehicle.length
        hw = ln * 0.7 / 2
        ht = ln * 0.2 / 2
        phi = self.phis[-1]
        tc, ts = (cos(phi), sin(phi))
        cam_focal, cam_width = (self.vehicle.cam_focal, self.vehicle.cam_width)
        lines = [
            # Frame
            (0, 0, ln, 0),
            (0, hw, 0, -hw),
            (ln, hw, ln, -hw),
            # Back tires
            (ht, hw, -ht, hw),
            (ht, -hw, -ht, -hw),
            # Front tires
            (ln+ht*tc, hw+ht*ts, ln-ht*tc, hw-ht*ts),
            (ln+ht*tc, -hw+ht*ts, ln-ht*tc, -hw-ht*ts),
            # camera
            (ln, 0, ln+cam_focal, 0.5*cam_width),
            (ln, 0, ln+cam_focal, -0.5*cam_width),
        ]
        x, y, th = self.true_states[-1]
        c, s = (cos(th), sin(th))
        tx_lines = []
        for x1, y1, x2, y2 in lines:
            tx_lines.append((
                x + c*x1 - s*y1, y + s*x1 + c*y1,
                x + c*x2 - s*y2, y + s*x2 + c*y2
            ))
        for x1, y1, x2, y2 in tx_lines:
            ax.plot([x1, x2], [y1, y2], color=(0.5, 0.5, 0.5, 0.5), lw=3.0)

    def draw_grid(self, ax):
        sz = 10.0
        for d in np.arange(-sz, sz+1e-6, 2.0):
            ax.plot([-sz, sz], [d]*2, color=(0.5, 0.5, 0.5, 0.5), linestyle=':')
            ax.plot([d]*2, [-sz, sz], color=(0.5, 0.5, 0.5, 0.5), linestyle=':')

    def plot_filters(self):
        self.plot_filter_iteration += 1
        fig = plt.figure("filter xy", figsize=self.figsize)
        fig.clf()
        ax = fig.subplots()
        self.draw_grid(ax)
        self.plot_vehicle(ax)

        ax.plot(*zip(*self.features), marker='v',
                linestyle='', color="tab:grey", markersize=5)

        true_x, true_y, true_th = self.true_states[-1]
        c, s = (cos(true_th), sin(true_th))
        cam_x = true_x + c*self.vehicle.length
        cam_y = true_y + s*self.vehicle.length
        measurements = self.vehicle.measure(self.true_states[-1])
        for (x, y), m in zip(self.features, measurements):
            linestyle = ':' if isnan(m) else '--'
            ax.plot([cam_x, x], [cam_y, y],
                    linestyle=linestyle, color='tab:grey')

        def plot(states, cov, color, label, size=0.2):
            if len(states.shape) == 1:
                states = states.reshape(1, states.shape[0])
            xs, ys, ths = states.T
            for x, y, th in zip(xs, ys, ths):
                dx, dy = (size*cos(th), size*sin(th))
                ax.arrow(x, y, dx, dy, color=color, width=size/10, label=label)
            if cov is not None:
                plot_cov_ellipse(ax, [x, y], cov[:2, :2], color)
        pf = self.filters['pf'].filter
        plot(pf.samples, None, (0, 0, 1, 0.5), None, size=0.02)
        plot(self.true_states[-1], None, 'k', 'true')
        for name, filter in self.filters.items():
            color = list(filter.color) + [0.8]
            plot(filter.states[-1], filter.covs[-1], color, name)

        size = 3.0
        ax.axis('equal')
        ax.set_xlim((true_x-0.5*size, true_x+0.5*size))
        ax.set_ylim((true_y-0.5*size, true_y+0.5*size))
        ax.legend(loc=1)
        plt.text(true_x-0.5*size + 0.1, true_y-0.5*size + 0.1, self.stage)
        if self.save_base_fn:
            fn = f"{self.save_base_fn}_{self.plot_filter_iteration:04d}.png"
            fig.savefig(fn)

    def plot_path(self):
        fig = plt.figure("path")
        ax = fig.subplots()

        def plot(states, color, style, label):
            x, y, th = zip(*states)
            ax.plot(x, y, color=color, linestyle=style, marker='.', label=label)
        plot(self.true_states, 'k', ':', 'true')
        for name, filter in self.filters.items():
            plot(filter.states, filter.color, filter.linestyle, name)
        ax.legend(loc='best')

    def plot_heading(self):
        fig = plt.figure("heading err")
        ax = fig.subplots()
        t2 = []
        for t in np.array(self.ts):
            t2 += [t, t]
        true_th = np.array([th for x, y, th in self.true_states])
        for name, filter in self.filters.items():
            x, y, th = zip(*filter.states)
            th_err = normalize_angles(np.array(th) - true_th)
            ax.plot(t2, th_err, color=filter.color, linestyle=filter.linestyle, marker='.', label=name)

    def plot_cov(self):
        fig = plt.figure("cov")
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        t2 = []
        for t in np.array(self.ts):
            t2 += [t, t]

        def plot(covs, color, linestyle, name):
            cov_xx = [c[0, 0] for c in covs]
            cov_yy = [c[1, 1] for c in covs]
            ax1.plot(t2, cov_xx, color=color,
                     linestyle=linestyle, label=name + "_cov_xx")
            ax2.plot(t2, cov_yy, color=color,
                     linestyle=linestyle, label=name + "_cov_yy")
        for name, filter in self.filters.items():
            plot(filter.covs, filter.color, filter.linestyle, name)
        ax2.legend(loc='best')

    def plot_meas(self):
        fig = plt.figure("meas")
        ax = fig.subplots()
        true_meas = list(zip(*self.true_meas))
        noisy_meas = list(zip(*self.noisy_meas))
        assert len(true_meas) == len(noisy_meas)
        colors = mcolors.TABLEAU_COLORS
        for color, tm, nm in zip(colors, true_meas, noisy_meas):
            ax.plot(self.ts, tm, color=color, marker='*', linestyle='')
            ax.plot(self.ts, nm, color=color, marker='.', linestyle='')

    def plot(self):
        self.plot_path()
        self.plot_heading()
        self.plot_meas()
        self.plot_cov()


def main():
    parser = argparse.ArgumentParser(
        "simulate different bayes filters on car-like vehicle with 1-D camera")
    parser.add_argument("--motion-cov-xy", "--Rxy", type=float, default=0.001,
                        help="Motion state noise covariance in for x and y states")
    parser.add_argument("--motion-cov-th", "--Rth", type=float, default=0.002,
                        help="Motion state noise covariance in for theta state")
    parser.add_argument("--meas-cov", "-Q", type=float, default=0.001,
                        help="1-D camera measurement covariance")
    parser.add_argument("--start-cov-x", "--Sx", type=float, default=0.05,
                        help="Starting x-state covariance used for filters and start state")
    parser.add_argument("--start-cov-y", "--Sy", type=float, default=0.05,
                        help="Starting y-state covariance used for filters and start state")
    parser.add_argument("--start-cov-th", "--Sth", type=float, default=0.2,
                        help="Starting theta-state covariance used for filters and start state")
    parser.add_argument("--figsize", type=float, default=6.0,
                        help="Figure size for animation")
    parser.add_argument("--seed", type=int, default=4,
                        help="Random seed")
    parser.add_argument("--num-features", type=int, default=8,
                        help="Mumber of features to randomly place in map")
    parser.add_argument("--viz-rate", type=float, default=5.0,
                        help="Rate to visualize bayes filters at (in Hz).")
    parser.add_argument("--run-time", type=float, default=5.0,
                        help="Amount of time to run simulation")
    parser.add_argument("--save", default=None,
                        help="Base filenmae to save animation as")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # state transition noise coveraiance
    #  x' = A*x + B*u + noise(R)
    xy_cov = args.motion_cov_xy
    th_cov = args.motion_cov_th
    R = np.diag((xy_cov, xy_cov, th_cov))

    # noise on measurement position in camera space
    Q = np.diag([args.meas_cov])

    # figure is square
    figsize = [args.figsize]*2

    # start covariance for different filters
    start_cov = np.diag(
        [args.start_cov_x, args.start_cov_y, args.start_cov_th])

    # Create a map features that can be detected with a 1-D camera
    map_size = 5.0
    features = []
    for _ in range(args.num_features):
        features.append(Point(np.random.uniform(-0.5*map_size, 0.5*map_size),
                              np.random.uniform(-0.5*map_size, 0.5*map_size)))

    # camera u = y*f/z
    # TODO argparse for vehicle params
    cam_focal = 0.075
    cam_width = 0.25
    cam_max_dist = 3.0
    dt = 0.1
    vehicle_length = 0.5
    max_phi = radians(30)
    vehicle = Vehicle(vehicle_length, max_phi, dt, cam_focal,
                      cam_width, cam_max_dist, features)

    sim = Sim(vehicle=vehicle, features=features, start_cov=start_cov, Q=Q, R=R,
              figsize=figsize, viz_rate=args.viz_rate, runtime=args.run_time, save_base_fn=args.save)
    sim.run()
    sim.plot()
    plt.show()


if __name__ == "__main__":
    main()
