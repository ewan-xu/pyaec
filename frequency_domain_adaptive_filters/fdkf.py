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

""" Frequency Domain Kalman Filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

def fdkf(x, d, M, beta=0.95, sgm2u=1e-2, sgm2v=1e-6):
  Q = sgm2u
  R = np.full(M+1,sgm2v)
  H = np.zeros(M+1,dtype=np.complex)
  P = np.full(M+1,sgm2u)

  window =  np.hanning(M)
  x_old = np.zeros(M)

  num_block = min(len(x),len(d)) // M
  e = np.zeros(num_block*M)

  for n in range(num_block):
    x_n = np.concatenate([x_old,x[n*M:(n+1)*M]])
    d_n = d[n*M:(n+1)*M]
    x_old = x[n*M:(n+1)*M]

    X_n = np.fft.rfft(x_n)

    y_n = ifft(H*X_n)[M:]
    e_n = d_n-y_n

    e_fft = np.concatenate([np.zeros(M),e_n*window])
    E_n = fft(e_fft)

    R = beta*R + (1.0 - beta)*(np.abs(E_n)**2)
    P_n = P + Q*(np.abs(H))
    K = P_n*X_n.conj()/(X_n*P_n*X_n.conj()+R)
    P = (1.0 - K*X_n)*P_n 

    H = H + K*E_n
    h = ifft(H)
    h[M:] = 0
    H = fft(h)

    e[n*M:(n+1)*M] = e_n
  
  return e