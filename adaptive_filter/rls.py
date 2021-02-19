# Copyright 2020 ewan xu<ewan_xu@outlook.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

""" recursive least squares filter """

import numpy as np

def rls(x, d, N = 4, lmbd = 0.999, delta = 0.0002):
    L = min(len(x),len(d))
    lmbd_inv = 1/lmbd
    h = np.zeros((N, 1))
    P = np.eye(N)/delta
    e = np.zeros(L-N)
    for n in range(L-N):
        x_n = np.array(x[n:n+N][::-1]).reshape(N, 1)
        d_n = d[n] 
        y_n = np.dot(x_n.T, h)
        e_n = d_n - y_n
        g = np.dot(P, x_n)
        g = g / (lmbd + np.dot(x_n.T, g))
        h = h + e_n * g
        P = lmbd_inv*(P - np.dot(g, np.dot(x_n.T, P)))
        e[n] = e_n
    return e