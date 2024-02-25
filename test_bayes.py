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
from util import single_gaussian_pdf, mutlivariate_gaussian_pdf
from pf import PF


def test_single_gaussian_pdf():
    assert abs(single_gaussian_pdf(0.0, 1.0, 0.0) - 0.39894228) < 1e-6
    assert abs(single_gaussian_pdf(0.1, 1.0, 0.1) - 0.39894228) < 1e-6
    assert abs(single_gaussian_pdf(0.0, 1.0, 1.0) - 0.24197072) < 1e-6
    assert abs(single_gaussian_pdf(0.0, 1.0, 1.0) - 0.24197072) < 1e-6
    assert abs(single_gaussian_pdf(1.0, 9.0, 0.0) - 0.12579441) < 1e-6


def test_1x1_multivariate_gaussian_pdf():
    one = np.ones((1, 1))
    zero = np.zeros((1, 1))
    assert abs(single_gaussian_pdf(zero, one, zero) - 0.39894228) < 1e-6
    assert abs(single_gaussian_pdf(0.1*one, one, 0.1*one) - 0.39894228) < 1e-6
    assert abs(single_gaussian_pdf(zero, one, one) - 0.24197072) < 1e-6
    assert abs(single_gaussian_pdf(zero, one, one) - 0.24197072) < 1e-6
    assert abs(single_gaussian_pdf(one, 9*one, zero) - 0.12579441) < 1e-6


def test_2x2_mutlivariate_gaussian_pdf():
    # with this test there is no-correlation between different variables
    # so covariance matrix is diagonal
    covs = (1, 4)
    covsM = np.diag(covs)
    mus = np.array((0, 1))
    xs = np.array((1, 3))
    prob = mutlivariate_gaussian_pdf(mus, covsM, xs)
    print("prob.shape", prob.shape)

    print("det(cov)", np.linalg.det(covsM))

    # can compute expected probability from
    expected_prob = 1.0
    for mu, cov, x in zip(mus, covs, xs):
        p = single_gaussian_pdf(mu, cov, x)
        print("p", p)
        expected_prob *= p
    assert abs(prob[0] - expected_prob) < 1e-6


def test_3x3_mutlivariate_gaussian_pdf():
    # with this test there is no-correlation between different variables
    # so covariance matrix is diagonal
    covs = (1, 2, 3)
    covsM = np.diag(covs)
    mus = np.array((4, 5, 6))
    xs = np.array((3, 2, 1))
    prob = mutlivariate_gaussian_pdf(mus, covsM, xs)
    print("prob.shape", prob.shape)

    print("det(cov)", np.linalg.det(covsM))

    # can compute expected probability from
    expected_prob = 1.0
    for mu, cov, x in zip(mus, covs, xs):
        p = single_gaussian_pdf(mu, cov, x)
        print("p", p)
        expected_prob *= p
    assert abs(prob[0] - expected_prob) < 1e-6


def test_multisample_mutlivariate_gaussian_pdf():
    covs = (1, 2, 3)
    mus = np.array((4, 5, 6))
    offset = np.array((3, 2, 1))*0.01
    n = 5
    samples = np.zeros((3, n))

    for i in range(5):
        samples[:, i] = mus + offset * i

    assert samples.shape == (3, n)
    probs = mutlivariate_gaussian_pdf(mus, np.diag(covs), samples)
    assert len(probs) == n

    expected_probs = np.ones(n)
    for i, xs in enumerate(samples.T):
        for mu, cov, x in zip(mus, covs, xs):
            expected_probs[i] *= single_gaussian_pdf(mu, cov, x)
    print("probs", probs)
    print("expected probs", expected_probs)
    for i in range(n):
        assert abs(expected_probs[i] - probs[i]) < 1e-6


def test_pf_resample():
    samples = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    print('samples.shape', samples.shape)
    weights = np.array([1.0, 2.1, 2.6, 4.0, 0.5, 0.2])
    n = len(weights)
    # expected count is a fractional count of how often certain sample should appear
    # in the resampled output.
    expected_counts = {tuple(s): c for s, c in zip(
        samples, n * weights / np.sum(weights))}

    total_counts = {tuple(sample): 0 for sample in samples}

    print("expected counts", expected_counts)
    wstarts = np.linspace(0.0, 0.9999, 100)
    for wstart in wstarts:
        resamples = PF.resample(samples, weights, wstart)
        counts = {tuple(sample): 0 for sample in samples}
        for sample in resamples:
            counts[tuple(sample)] += 1
            total_counts[tuple(sample)] += 1

        # with low variance resampling a particle should show up a specific
        # number of times based on it weight relative to other samples
        for sample, count in counts.items():
            expected = expected_counts[sample]
            assert np.floor(expected) <= count <= np.ceil(
                expected), f"wstart {wstart}"

    # after multiple runs with different seeds, the count of samples should
    # start to average out to exepcted value
    avg_counts = {s: c/len(wstarts) for s, c in total_counts.items()}
    for sample, expected in expected_counts.items():
        avg_count = avg_counts[sample]
        assert abs(expected - avg_count) <= 1.5/len(wstarts)

    print("avg_counts", avg_counts)
