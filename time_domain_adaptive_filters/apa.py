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

""" Affine Projection Algorithm """

import numpy as np

def apa(x, d, N = 4, P = 4, mu = 0.1):
  nIters = min(len(x),len(d)) - N
  u = np.zeros(N)
  A = np.zeros((N,P))
  D = np.zeros(P)
  w = np.zeros(N)
  e = np.zeros(nIters)
  alpha = np.eye(P)*1e-2
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    A[:,1:] = A[:,:-1]
    A[:,0] = u
    D[1:] = D[:-1]
    D[0] = d[n] 
    e_n = D - np.dot(A.T, w)
    delta = np.dot(np.linalg.inv(np.dot(A.T,A)+alpha),e_n)
    w = w + mu * np.dot(A ,delta)
    e[n] = e_n[0]
  return e