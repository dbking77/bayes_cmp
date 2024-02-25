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
from math import sqrt
from util import State
from vehicle import Vehicle


class UKF:
    def __init__(self, vehicle: Vehicle, start_state: State, start_cov, R, Q):
        self.vehicle = vehicle
        self.mu = np.array(start_state)
        self.cov = start_cov
        assert start_cov.shape == (3, 3)
        self.R = R  # Processes noise
        self.Q = Q  # Measurement noise

    def update(self, v, phi, meas) -> State:
        # https://www.youtube.com/watch?v=c_6WDC66aVk&t=303s
        # https://www.mathworks.com/help/control/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html
        self.predict(v, phi)
        return self.measure(meas)

    def sample(self):
        n = 3
        # https://math.stackexchange.com/questions/796331/scaling-factor-and-weights-in-unscented-transform-ukf
        weights = np.zeros(2*n+1)
        w0 = 0.01
        weights[0] = w0
        samples = np.zeros((3, 2*n+1))
        samples[:, 0] = self.mu

        wn = (1.0 - w0) / (2*n)
        L = np.linalg.cholesky(self.cov)
        for i in range(n):
            dx = sqrt(n / (1.0 - w0)) * L[:, i]
            samples[:, i + 1] = self.mu + dx
            samples[:, i + 1 + n] = self.mu - dx
            weights[i + 1] = wn
            weights[i + 1 + n] = wn
        return (samples, weights)

    def predict(self, v, phi):
        samples, weights = self.sample()

        for i, sample in enumerate(samples.T):
            samples[:, i] = self.vehicle.update(sample, v, phi)

        self.mu = np.average(samples, 1, weights=weights)
        self.cov = np.copy(self.R)
        for i, w in enumerate(weights):
            dx = np.asmatrix(samples[:, i] - self.mu)
            self.cov += w * (dx.T @ dx)
        return (self.mu, self.cov)

    def measure(self, actual_meas):
        samples, weights = self.sample()
        # 2 measurements per update
        n = 3
        predict_meas = np.zeros((len(actual_meas), 2*n+1))
        for i, sample in enumerate(samples.T):
            # assume x and y can be measuremed
            predict_meas[:, i] = self.vehicle.measure(
                sample, outside_is_nan=False)

        predict_meas_avg = np.average(predict_meas, 1, weights=weights)
        assert len(predict_meas_avg) == len(actual_meas)

        keep = np.logical_not(np.logical_or(
            np.isnan(actual_meas), np.isnan(predict_meas_avg)))
        actual_meas = np.compress(keep, actual_meas)
        predict_meas = np.compress(keep, predict_meas, axis=0)
        predict_meas_avg = np.compress(keep, predict_meas_avg)

        predict_meas_cov = np.eye(len(actual_meas)) * float(self.Q)
        for i, w in enumerate(weights):
            dz = np.asmatrix(predict_meas[:, i] - predict_meas_avg)
            predict_meas_cov += w * (dz.T @ dz)

        # cross covariance of measurement and state
        SH = np.zeros((3, len(actual_meas)))
        for s, m, w in zip(samples.T, predict_meas.T, weights):
            dx = np.asmatrix(s - self.mu).T
            dz = np.asmatrix(m - predict_meas_avg).T
            SH += w * dx @ dz.T

        K = SH @ np.linalg.inv(predict_meas_cov)
        self.mu += K @ (actual_meas - predict_meas_avg)
        self.cov -= K @ predict_meas_cov @ K.T
        return (self.mu, self.cov)
