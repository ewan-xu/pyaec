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

""" Partitioned-Block-Based Frequency Domain Adaptive Filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

class PFDAF:
  def __init__(self, N, M, mu, partial_constrain):
    self.N = N
    self.M = M
    self.N_freq = 1+M
    self.N_fft = 2*M
    self.mu = mu
    self.partial_constrain = partial_constrain
    self.p = 0
    self.x_old = np.zeros(self.M,dtype=np.float32)
    self.X = np.zeros((N,self.N_freq),dtype=np.complex)
    self.H = np.zeros((self.N,self.N_freq),dtype=np.complex)
    self.window = np.hanning(self.M)

  def filt(self, x, d):
    assert(len(x) == self.M)
    x_now = np.concatenate([self.x_old,x])
    X = fft(x_now)
    self.X[1:] = self.X[:-1]
    self.X[0] = X
    self.x_old = x
    Y = np.sum(self.H*self.X,axis=0)
    y = ifft(Y)[self.M:]
    e = d-y
    return e

  def update(self,e):
    X2 = np.sum(np.abs(self.X)**2,axis=0)
    e_fft = np.zeros(shape=(self.N_fft,),dtype=np.float32)
    e_fft[self.M:] = e*self.window
    E = fft(e_fft)
    
    G = self.mu*E/(X2+1e-10)
    self.H += self.X.conj()*G

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

def pfdaf(x, d, N=4, M=64, mu=0.2, partial_constrain=True):
  ft = PFDAF(N, M, mu, partial_constrain)
  num_block = min(len(x),len(d)) // M

  e = np.zeros(num_block*M)
  for n in range(num_block):
    x_n = x[n*M:(n+1)*M]
    d_n = d[n*M:(n+1)*M]
    e_n = ft.filt(x_n,d_n)
    ft.update(e_n)
    e[n*M:(n+1)*M] = e_n
    
  return e