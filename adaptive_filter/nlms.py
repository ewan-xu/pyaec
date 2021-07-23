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

""" normalized least mean squares filter """

import numpy as np

def nlms(x, d, N = 4, mu = 0.1):
  L = min(len(x),len(d))
  h = np.zeros(N)
  e = np.zeros(L-N)
  for n in range(L-N):
    x_n = x[n:n+N][::-1]
    d_n = d[n] 
    y_n = np.dot(h, x_n.T)
    e_n = d_n - y_n
    h = h + mu * e_n * x_n / (np.dot(x_n,x_n)+1e-5)
    e[n] = e_n
  return e