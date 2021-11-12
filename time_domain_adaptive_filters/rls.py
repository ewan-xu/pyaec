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

""" Recursive Least Squares Filter """

import numpy as np

def rls(x, d, N = 4, lmbd = 0.999, delta = 0.01):
  nIters = min(len(x),len(d)) - N
  lmbd_inv = 1/lmbd
  u = np.zeros(N)
  w = np.zeros(N)
  P = np.eye(N)*delta
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    e_n = d[n] - np.dot(u, w)
    r = np.dot(P, u)
    g = r / (lmbd + np.dot(u, r))
    w = w + e_n * g
    P = lmbd_inv*(P - np.outer(g, np.dot(u, P)))
    e[n] = e_n
  return e