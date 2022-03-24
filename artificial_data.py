# Copyright 2022 Joscha Diehl, Nikolas Tapia
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import numpy as np
import itertools


def _permuted_pattern(n_in, n_samples, pattern=[3, -4, 10], sigma=1.0):
    tmp = list(itertools.permutations(pattern))
    assert list(tmp[0]) == list(pattern)
    all_other_orderings = tmp[1:]

    xs0 = []
    xs1 = []
    for m in range(n_samples // 2):
        places = sorted(np.random.choice(range(n_in), len(pattern), replace=False))
        x = sigma * np.random.normal(size=n_in)
        for i in range(len(pattern)):
            x[places[i]] += pattern[i]
        xs0.append(x)

        places = sorted(np.random.choice(range(n_in), len(pattern), replace=False))
        x = sigma * np.random.normal(size=n_in)
        which_order = np.random.randint(len(all_other_orderings))  # Some _other_ order.
        for i in range(len(pattern)):
            x[places[i]] += all_other_orderings[which_order][i]
        xs1.append(x)
    return xs0, xs1


def _permuted_pattern_consecutive(n_in, n_samples, pattern=[3, -4, 10], sigma=1.0):
    tmp = list(itertools.permutations(pattern))
    assert list(tmp[0]) == list(pattern)
    all_other_orderings = tmp[1:]

    xs0 = []
    xs1 = []
    for m in range(n_samples // 2):
        pos = np.random.randint(0, n_in - len(pattern))
        x = sigma * np.random.normal(size=n_in)
        for i in range(len(pattern)):
            x[pos + i] += pattern[i]
        xs0.append(x)

        pos = np.random.randint(0, n_in - len(pattern))
        x = sigma * np.random.normal(size=n_in)
        which_order = np.random.randint(len(all_other_orderings))  # Some _other_ order.
        for i in range(len(pattern)):
            x[pos + i] += all_other_orderings[which_order][i]
        xs1.append(x)

    return xs0, xs1


def permuted_pattern(*args, **kwargs):
    xs0, xs1 = _permuted_pattern(*args, **kwargs)
    X = np.vstack(xs0 + xs1)[:, :, None]
    Y = np.hstack((np.zeros(len(xs0)), np.ones(len(xs1))))
    return X, Y


def permuted_pattern_consecutive(*args, **kwargs):
    xs0, xs1 = _permuted_pattern_consecutive(*args, **kwargs)
    X = np.vstack(xs0 + xs1)[:, :, None]
    Y = np.hstack((np.zeros(len(xs0)), np.ones(len(xs1))))
    return X, Y


def look_at_permuted_pattern():
    np.random.seed(12345)

    n_in = 100
    n_MC = 20
    n_classes = 2
    # xs0, xs1 = permuted_pattern(n_in, n_MC,sigma=0.1) #,abc=[-5,10])
    xs0, xs1 = _permuted_pattern_consecutive(n_in, n_MC, sigma=0.1)  # ,abc=[-5,10])
    import seaborn as sns

    sns.set()
    import html_plotter
    from matplotlib import pyplot as plt

    ts = list(range(n_in))
    # for _x, _label in [[xs0[0],'xs0'], [xs0[1],'xs0'],[xs1[0],'xs1'],[xs1[1],'xs1']]:
    for _x, _label in [[xs1[i], "xs1"] for i in range(10)]:
        # plt.plot(ts, _x, label=_label)
        plt.bar(ts, _x, label=_label)
        plt.legend()
        html_plotter.show()


if __name__ == "__main__":
    look_at_permuted_pattern()
