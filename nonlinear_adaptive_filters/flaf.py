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

""" Trigonometric Functional Link Adaptive Filter """

import numpy as np

def flaf(x, d, M=128, P=5, mu=0.2):
  nIters = min(len(x),len(d)) - M
  Q = P*2
  u = np.zeros(M)
  w = np.zeros((Q+1)*M)
  e = np.zeros(nIters)
  sk = np.zeros(P*M,dtype=np.int32)
  ck = np.zeros(P*M,dtype=np.int32)
  pk = np.tile(np.arange(P),M)
  for k in range(M):
    sk[k*P:(k+1)*P] = np.arange(1,Q,2) + k*(Q+1)
    ck[k*P:(k+1)*P] = np.arange(2,Q+1,2) + k*(Q+1)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    g = np.repeat(u,Q+1)
    g[sk] = np.sin(np.pi*pk*g[sk])
    g[ck] = np.cos(np.pi*pk*g[ck])
    y = np.dot(w, g.T)
    e_n = d[n] - y
    w = w + 2*mu*e_n*g/(np.dot(g,g)+1e-3)
    e[n] = e_n
  return e