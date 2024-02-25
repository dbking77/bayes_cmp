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

import numpy as np
from util import State
from vehicle import Vehicle


class EKF:
    def __init__(self, vehicle: Vehicle, start_state: State, start_cov, R, Q):
        self.vehicle = vehicle
        self.mu = np.array(start_state)
        self.cov = start_cov
        assert start_cov.shape == (3, 3)
        self.R = R
        self.Q = Q

    def update(self, v, phi, meas) -> State:
        self.predict(v, phi)
        return self.measure(meas)

    def predict(self, v, phi):
        eps = 1e-6
        # G = linearized state transition around previous state
        G = np.ones((3, 3))
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps
            G[:, i] = (self.vehicle.update(self.mu + delta, v, phi) -
                       self.vehicle.update(self.mu - delta, v, phi)) / (2*eps)
        # prediction
        self.mu = self.vehicle.update(self.mu, v, phi)
        self.cov = G @ self.cov @ G.T + self.R
        return (self.mu, self.cov)

    def measure(self, actual_meas):
        eps = 1e-6
        # H = sense (just, x, and y, not th)
        # dz/dx
        H = np.zeros((len(actual_meas), 3))
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps
            m1 = self.vehicle.measure(self.mu + delta, outside_is_nan=False)
            m2 = self.vehicle.measure(self.mu - delta, outside_is_nan=False)
            H[:, i] = (m1 - m2) / (2*eps)
        predict_meas = self.vehicle.measure(self.mu, outside_is_nan=False)
        # throw away any invalid measurement
        keep = np.logical_not(np.logical_or(
            np.isnan(actual_meas), np.isnan(predict_meas)))
        actual_meas = np.compress(keep, actual_meas)
        predict_meas = np.compress(keep, predict_meas)
        H = np.compress(keep, H, axis=0)
        # make Q a diagonal matrix, that matches size of valid measurements
        Q = np.eye(len(actual_meas)) * float(self.Q)
        K = self.cov @ H.T @ np.linalg.inv(H @ self.cov @ H.T + Q)
        self.mu = self.mu + K@(actual_meas - predict_meas)
        self.cov = (np.eye(3) - K @ H) @ self.cov
        return (self.mu, self.cov)
