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

""" Normalized Least Mean Squares Filter """

import numpy as np

def nlms(x, d, N=4, mu=0.1):
  nIters = min(len(x),len(d)) - N
  u = np.zeros(N)
  w = np.zeros(N)
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    e_n = d[n] - np.dot(u, w)
    w = w + mu * e_n * u / (np.dot(u,u)+1e-3)
    e[n] = e_n
  return e






