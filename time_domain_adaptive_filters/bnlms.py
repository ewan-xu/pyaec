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

""" least mean squares filter """

import numpy as np
from scipy.linalg import hankel

def bnlms(x, d, N = 4, L=4, mu = 0.1):
  beta = 0.9
  nIters = min(len(x),len(d))//L
  u = np.zeros(L+N-1)
  h = np.zeros(N)
  e = np.zeros(nIters*L)
  norm = np.full(L,1e-3)
  for n in range(nIters):
    u[:-L] = u[L:]
    u[-L:] = x[n*L:(n+1)*L]
    d_n = d[n*L:(n+1)*L]
    A = hankel(u[:L],u[-N:])
    e_n = d_n - np.dot(A,h)
    norm = beta*norm + (1-beta)*(np.sum(A**2,axis=1))
    h = h + mu*np.dot(A.T/(norm+1e-3),e_n)/L
    e[n*L:(n+1)*L] = e_n
  return e
