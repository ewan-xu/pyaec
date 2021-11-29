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

""" Partitioned-Block-Based Frequency Domain Kalman Filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

class PFDKF:
  def __init__(self,N,M,A=0.999,P_initial=1e+2, partial_constrain=True):
    self.N = N
    self.M = M
    self.N_freq = 1+M
    self.N_fft = 2*M
    self.A2 = A**2
    self.partial_constrain = partial_constrain
    self.p = 0

    self.x = np.zeros(shape=(2*self.M),dtype=np.float32)
    self.P = np.full((self.N,self.N_freq),P_initial)
    self.X = np.zeros((N,self.N_freq),dtype=np.complex)
    self.window = np.hanning(self.M)
    self.H = np.zeros((self.N,self.N_freq),dtype=np.complex)

  def filt(self, x, d):
    assert(len(x) == self.M)
    self.x[self.M:] = x
    X = fft(self.x)
    self.X[1:] = self.X[:-1]
    self.X[0] = X
    self.x[:self.M] = self.x[self.M:]
    Y = np.sum(self.H*self.X,axis=0)
    y = ifft(Y)[self.M:]
    e = d-y
    return e

  def update(self, e):
    e_fft = np.zeros(shape=(self.N_fft,),dtype=np.float32)
    e_fft[self.M:] = e*self.window
    E = fft(e_fft)
    X2 = np.sum(np.abs(self.X)**2,axis=0)
    Pe = 0.5*self.P*X2 + np.abs(E)**2/self.N
    mu = self.P / (Pe + 1e-10)
    self.P = self.A2*(1 - 0.5*mu*X2)*self.P + (1-self.A2)*np.abs(self.H)**2
    G = mu*self.X.conj()
    self.H += E*G

    if self.partial_constrain:
      h = ifft(self.H[self.p])
      h[self.M:] = 0
      self.H[self.p] = fft(h)
      self.p = (self.p + 1) % self.N
    else:
      for p in range(self.N):
        h = ifft(self.H[p])
        h[self.M:] = 0
        self.H[p] = fft(h)

def pfdkf(x, d, N=4, M=64, A=0.999,P_initial=1e+2, partial_constrain=True):
  ft = PFDKF(N, M, A, P_initial, partial_constrain)
  num_block = min(len(x),len(d)) // M

  e = np.zeros(num_block*M)
  for n in range(num_block):
    x_n = x[n*M:(n+1)*M]
    d_n = d[n*M:(n+1)*M]
    e_n = ft.filt(x_n,d_n)
    ft.update(e_n)
    e[n*M:(n+1)*M] = e_n
  return e
