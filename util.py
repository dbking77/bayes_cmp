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
import matplotlib.patches as patches
from collections import namedtuple
from math import cos, sin, pi, sqrt, exp, pow, atan2, degrees

State = namedtuple("State", ['x', 'y', 'th'])
Point = namedtuple("Point", ['x', 'y'])


def normalize_angles(thetas):
    return np.mod(3*pi + np.mod(thetas, 2*pi), 2*pi)-pi


def single_gaussian_pdf(mu, cov, x):
    """
    this calculates a gaussian of a single variable
    note cov is covariance (not standard deviation)
    """
    dx = (x - mu)
    return exp(-0.5 * dx**2 / cov) / sqrt(2*pi*cov)


def mutlivariate_gaussian_pdf(mu, cov, x):
    """
    computes multivariate gaussian.
    if cov is diagnonal matrix, the multivariate gausian is same it computing
    single variable gausians and multiplying the result together

    This function can compute probably of multiple samples (x) at once
    if COV is an NxN matrix then X's shape should be NxL
    and output will be a array of probabilities with shape (L,)
    """
    if len(x.shape) != 2:
        x = x.reshape(-1, 1)
    if len(mu.shape) != 2:
        mu = mu.reshape(-1, 1)
    dx = (x - mu)
    # https://www.youtube.com/watch?v=jAyTgkiaBbY
    # https://jamesmccaffrey.wordpress.com/2021/11/29/multivariate-gaussian-probability-density-function-from-scratch-almost/multivariate_log_pdf_scratch_demo_run/
    # https://math.stackexchange.com/questions/3157810/how-to-vectorize-matricize-multivariate-gaussian-pdf-for-more-efficient-computat
    term = np.sum(np.multiply(dx, np.linalg.inv(cov) @ dx), 0)
    prob = np.exp(-0.5 * term) / sqrt(np.linalg.det(cov)
                                      * pow(2*pi, cov.shape[0]))
    return np.array(prob)


def gen_cov_xy(x_std, y_std, th=0.0):
    L = np.diag([x_std, y_std])
    rot = np.array([[cos(th), -sin(th)],
                    [sin(th), cos(th)]])
    L = rot @ L
    return L @ L.T


def plot_cov_ellipse(ax, xy, cov_xy, color, n_std=3.0):
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    # with n_std of 3.0 -> 98.9% of gaussian samples should fall within 3-STDs
    eigval, eigvec = np.linalg.eig(cov_xy)
    w, h = n_std * 2 * np.sqrt(eigval)
    angle = atan2(eigvec[1, 0], eigvec[0, 0])
    ellipse = patches.Ellipse(
        xy, w, h, degrees(angle), fill=False, color=color)
    ax.add_patch(ellipse)
