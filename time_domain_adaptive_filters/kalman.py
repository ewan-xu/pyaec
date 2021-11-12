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

""" Time Domain Kalman Filter """

import numpy as np


def kalman(x, d, N = 64, sgm2v=1e-4):
  nIters = min(len(x),len(d)) - N
  u = np.zeros(N)
  w = np.zeros(N)
  Q = np.eye(N)*sgm2v
  P = np.eye(N)*sgm2v
  I = np.eye(N)
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    e_n =  d[n] - np.dot(u, w)
    R = e_n**2+1e-10
    Pn = P + Q
    r = np.dot(Pn,u)
    K = r / (np.dot(u, r) + R + 1e-10)
    w = w + np.dot(K, e_n)
    P = np.dot(I - np.outer(K, u), Pn)
    e[n] = e_n

  return e