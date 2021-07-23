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

""" time domain kalman filter """

import numpy as np

def kalman(x, d, N = 64, sgm2v=1e-4):
  L = min(len(x),len(d))
  Q = np.eye(N)*sgm2v
  H = np.zeros((N, 1))
  P = np.eye(N)*sgm2v
  I = np.eye(N)

  e = np.zeros(L-N)
  for n in range(L-N):
    x_n = np.array(x[n:n+N][::-1]).reshape(1, N)
    d_n = d[n] 
    y_n = np.dot(x_n, H)
    e_n = d_n - y_n
    R = e_n**2+1e-10
    Pn = P + Q
    K = np.dot(Pn, x_n.T) / (np.dot(x_n, np.dot(Pn, x_n.T)) + R)
    H = H + np.dot(K, e_n)
    P = np.dot(I - np.dot(K, x_n), Pn)
    e[n] = e_n
  return e
