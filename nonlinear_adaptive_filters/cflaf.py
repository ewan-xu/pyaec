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

""" Collaborative Functional Link Adaptive Filter """

import numpy as np

def cflaf(x, d, M=128, P=5, mu_L=0.2, mu_FL=0.5, mu_a=0.5):
  nIters = min(len(x),len(d)) - M
  Q = P*2
  beta = 0.9
  sk = np.arange(0,Q*M,2)
  ck = np.arange(1,Q*M,2)
  pk = np.tile(np.arange(P),M)
  u = np.zeros(M)
  w_L = np.zeros(M)
  w_FL = np.zeros(Q*M)
  alpha = 0
  gamma = 1
  e = np.zeros(nIters)    
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    g = np.repeat(u,Q)
    g[sk] = np.sin(pk*np.pi*g[sk])
    g[ck] = np.cos(pk*np.pi*g[ck])
    y_L = np.dot(w_L, u.T)
    y_FL = np.dot(w_FL,g.T)
    e_FL = d[n] - (y_L+y_FL)
    w_FL = w_FL + mu_FL * e_FL * g / (np.dot(g,g)+1e-3)
    lambda_n = 1 / (1 + np.exp(-alpha))
    y_N = y_L + lambda_n*y_FL
    e_n = d[n] - y_N
    gamma = beta*gamma + (1-beta)*(y_FL**2)
    alpha = alpha + (mu_a*e_n*y_FL*lambda_n*(1-lambda_n) /  gamma)
    alpha = np.clip(alpha,-4,4)
    w_L = w_L + mu_L*e_n*u/(np.dot(u,u)+1e-3)
    e[n] = e_n
  return e