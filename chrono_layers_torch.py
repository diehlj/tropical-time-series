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

import sys
import numpy as np
import torch
import torch.nn as nn
import operator


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def cummax(_x):
    cs, _ = torch.cummax(_x, axis=-2)
    return cs


def cumsum(_x):
    cs = torch.cumsum(_x, axis=-2)
    return cs


class LayerSumSimple(nn.Module):
    def __init__(
        self,
        functions,
        cumoplus=cummax,
        odot=operator.add,
        pad_with=None,
        return_all=False,
        take_diff=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.functions = functions
        self.kernel_length = len(self.functions)
        self.pad_with = pad_with
        self.return_all = return_all
        self.take_diff = take_diff
        self.cumoplus = cumoplus
        self.odot = odot
        if self.return_all:
            assert exists(self.pad_with), "Must specify pad_with if return_all=True."

    def forward(self, X):
        """X : (b, d_in, in_length)"""

        def apply_fn_at_fixed_timepoint(_f, _x):
            tmp = _f(_x)
            return tmp

        if self.take_diff:
            X = torch.diff(X, axis=-2)

        p = apply_fn_at_fixed_timepoint(self.functions[0], X)
        cs = self.cumoplus(p)

        allz = [cs]
        for k in range(1, self.kernel_length):
            A = apply_fn_at_fixed_timepoint(self.functions[k], X)[:, k:]
            tmp = self.odot(cs[:, :-1, :], A)
            cs = self.cumoplus(tmp)
            if self.pad_with:
                cs_pad = nn.ConstantPadding1D((k, 0), self.pad_with)(cs)
                allz.append(cs_pad)
            else:
                allz.append(cs)

        if self.return_all:
            return torch.concat(allz, axis=-2)
        else:
            return allz[-1]


def test_layer_sum_simple():
    import iterated_sums_signature as iss
    from linear_combination import linear_combination as lc
    from iterated_sums_signature.divided_powers import M_qs, M_concat, M_sh

    def flip(_t):
        return torch.transpose(_t, -1, -2)

    def minus_id(_x):
        return -_x

    def id(_x):
        return _x

    def sq(_x):
        return _x * _x

    def cb(_x):
        return _x * _x * _x

    X = flip(torch.reshape(torch.arange(0, 20, dtype=torch.float32), (1, 4, 5)))
    # print('X=', X, X.shape)
    ls = LayerSumSimple([id], cumoplus=cumsum, odot=torch.mul)

    x = np.array(flip(X[0, :, :]))
    x = np.c_[np.zeros(4), x]
    x = np.cumsum(x, axis=-1)
    # print('x=',x,x.shape)
    ISS = iss.signature(x, upto_level=2)

    def get(z, _DS):
        return lc.LinearCombination.inner_product(z, _DS)

    t = ls(X)
    # print('t=',t)

    for i in range(4):
        _g = get(M_qs({i: 1}), ISS)
        # print(_g)
        np.testing.assert_allclose(t[0, :, i], _g[1:])

    ls = LayerSumSimple([sq, cb], cumoplus=cumsum, odot=torch.mul)
    t = ls(X)
    x = np.array(X[0, :, :].T)
    x = np.c_[np.zeros(4), x]
    xx_xxx = np.vstack((x * x, x * x * x))
    xx_xxx = np.cumsum(xx_xxx, axis=-1)
    ISS = iss.signature(xx_xxx, upto_level=2)
    for i in range(4):
        _g = get(M_qs({i: 1}, {i + 4: 1}), ISS)
        np.testing.assert_allclose(t[0, :, i], _g[2:])

    x = flip(torch.tensor([[[0, 2, 5, 3, 8, 1, 7], [0, 3, 2, 1, 8, 1, 7]]]))
    # print('x=',x,x.shape)
    ls = LayerSumSimple([id], cumoplus=cummax, odot=operator.add)
    t = flip(ls(x))
    # print( 't=',t )
    assert torch.allclose(
        t, torch.tensor([[[0, 2, 5, 5, 8, 8, 8], [0, 3, 3, 3, 8, 8, 8]]])
    )

    ls = LayerSumSimple([id, minus_id], cumoplus=cummax, odot=operator.add)
    t = flip(ls(x))
    # print('t=', t)
    assert torch.allclose(
        t, torch.tensor([[[-2, -2, 2, 2, 7, 7], [-3, 1, 2, 2, 7, 7]]])
    )


if __name__ == "__main__":
    test_layer_sum_simple()
