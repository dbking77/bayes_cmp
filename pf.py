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
from util import State, single_gaussian_pdf
from vehicle import Vehicle
from math import isnan


class PF:
    def __init__(self, vehicle: Vehicle, start_state: State, start_cov, R, Q):
        self.vehicle = vehicle
        self.R = R  # Processes noise
        self.Q = Q  # Measurement noise
        assert start_cov.shape == (3, 3)
        self.n = 300
        self.samples = np.random.multivariate_normal(
            np.array(start_state), start_cov, size=self.n)
        self.weights = np.ones(self.n)
        self.iteration = 0

    def update(self, v, phi, meas):
        self.predict(v, phi)
        return self.measure(meas)

    def predict(self, v, phi):
        for i, sample in enumerate(self.samples):
            self.samples[i, :] = self.vehicle.update(sample, v, phi)
        self.samples += np.random.multivariate_normal(
            np.zeros(3), self.R, size=self.n)
        return self.get_state()

    def measure(self, actual_meas):
        self.iteration += 1
        sample_meas = np.zeros((self.n, 1))
        for i, sample in enumerate(self.samples):
            # predict what sample would measure
            sample_meas = self.vehicle.measure(sample, outside_is_nan=False)
            for m1, m2 in zip(actual_meas, sample_meas):
                prob = 1.0
                if isnan(m1):
                    prob = 1.0
                elif isnan(m2):
                    prob = 0.0001
                else:
                    prob = single_gaussian_pdf(m1, self.Q[0, 0], m2)
                assert not isnan(prob), f"{m1} {m2}"
                self.weights[i] *= prob
        self.weights /= np.sum(self.weights)
        # resample based on effective sample size
        # https://arxiv.org/pdf/1602.03572.pdf
        ess = 1.0 / np.sum(self.weights * self.weights)
        # resample when effective sample size is 70% of original
        if ess < self.n * 0.7:
            # print(f"resample on iteration {self.iteration} : ess {ess} of {self.n}")
            self.samples = self.resample(self.samples, self.weights)
            self.weights = np.ones(self.weights.shape)
        return self.get_state()

    def get_state(self):
        mu = np.average(self.samples, 0, weights=self.weights)
        cov = np.cov(self.samples.T, aweights=self.weights)
        return (mu, cov)

    @staticmethod
    def resample(samples, weights, wstart=None):
        """
        First dimention is for each individual sample
        """
        wsum = np.sum(weights)
        wcsum = np.cumsum(weights)
        assert wsum > 0.0
        n = len(weights)
        dw = wsum / n
        if wstart is None:
            wstart = np.random.uniform(0.0, 1.0)
        w = wstart * wsum
        i = 0
        resamples = np.zeros(samples.shape)
        for c in range(n):
            while w >= wcsum[-1]:
                i = 0
                w -= wcsum[-1]
            while wcsum[i] < w:
                i = (i+1) % n
            resamples[c, :] = samples[i, :]
            w += dw
        return resamples
