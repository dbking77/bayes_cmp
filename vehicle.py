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
from math import cos, sin, tan
from typing import List, Optional
from util import State, Point


class Vehicle:
    def __init__(self, length, max_phi, dt, cam_focal, cam_width, cam_max_dist, features: List[Point]):
        self.length = length
        self.max_phi = max_phi
        self.cam_focal = cam_focal
        self.cam_width = cam_width
        self.cam_max_dist = cam_max_dist
        self.dt = dt
        self.features = features

    def limit_phi(self, phi) -> float:
        return max(-self.max_phi, min(self.max_phi, phi))

    # https://www.shuffleai.blog/blog/Simple_Understanding_of_Kinematic_Bicycle_Model.html
    # Simulate bicycle model with fixed L = 0.5 and a small motion
    def update(self, state, v: float, phi: float) -> State:
        phi = self.limit_phi(phi)
        x, y, th = state
        dist = v*self.dt
        max_delta = 0.01
        L = 0.5
        while dist > 0:
            delta = min(dist, max_delta)
            dist -= delta
            x += cos(th)*delta
            y += sin(th)*delta
            th += tan(phi)/L*delta
        return np.array((x, y, th))

    def measure(self, state, outside_is_nan=True) -> List[Optional[int]]:
        # return measurements (cam_u) to expect from a certain state
        # if no_nan is true return expected cordinate for measurement even if measurement
        # shouldn't be possible (because it would be outside of camera width or behind camera)
        nan = float("nan")
        measurements = np.zeros(len(self.features))
        x, y, th = state
        c, s = (cos(th), sin(th))
        cam_x = x + c*self.length
        cam_y = y + s*self.length
        for i, (feature_x, feature_y) in enumerate(self.features):
            dx, dy = (feature_x - cam_x), (feature_y - cam_y)
            x_wrt_cam = -s*dx + c*dy
            z_wrt_cam = c*dx + s*dy
            u_cam = self.cam_focal*x_wrt_cam/z_wrt_cam
            if outside_is_nan:
                if abs(u_cam) > 0.5*self.cam_width:
                    # feature is too far to sides of camera
                    u_cam = nan
                elif not (0 <= z_wrt_cam <= self.cam_max_dist):
                    # feature is behind or too far in front of camera
                    u_cam = nan
            measurements[i] = u_cam
        return measurements
