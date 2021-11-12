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

""" Second Order Volterra Filter """

import numpy as np

def svf(x, d, M=128, L=10, mu1=0.2, mu2=0.2):
  nIters = min(len(x),len(d)) - M
  L2=int(L*(L+1)/2)
  u = np.zeros(M)
  u2 = np.zeros((M,L2))
  w = np.zeros(M)
  h2 = np.zeros(L2)
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    u2_n = np.outer(u[:L],u[:L])
    u2_n = u2_n[np.triu_indices_from(u2_n)]
    u2[1:] = u2[:-1]
    u2[0] = u2_n
    x2 = np.dot(u2,h2)
    g = u + x2
    y = np.dot(w, g.T)
    e_n = d[n] - y
    w = w + mu1*e_n*g/(np.dot(g,g)+1e-3)
    grad_2 = np.dot(u2.T,w)
    h2 = h2 + mu2*e_n*grad_2/(np.dot(grad_2,grad_2)+1e-3)
    e[n] = e_n
  return e