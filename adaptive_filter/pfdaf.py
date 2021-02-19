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

""" partitioned-block-based frequency domain adaptive filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

class X:
    def __init__(self,N,M,dtype=np.complex):
        self.N = N
        self.M = M
        self.buf = np.zeros(shape=(N,M),dtype=dtype)
        self.p = 0
    
    def append(self,buf):
        assert(len(buf) == self.M)
        self.p = self.p - 1 if self.p > 0 else self.N - 1 
        self.buf[self.p] = buf
    
    def __call__(self):
        return np.concatenate((self.buf[self.p:],self.buf[:self.p]))
    
    def __array__(self):
        return np.concatenate((self.buf[self.p:],self.buf[:self.p]))
    
    def __getitem__(self, i):
        assert(i < self.N)
        index = (self.p + i) % self.N
        return self.buf[index]

class H:
    def __init__(self,N,M,dtype=np.complex):
        self.N = N
        self.M = M
        self.w = np.zeros(shape=(N,M),dtype=dtype)
        self.p = 0
    
    def partition(self):
        return self.w[self.p]

    def constrain(self,w):
        self.w[self.p] = w
        self.p = (self.p + 1) % self.N
    
    def __call__(self):
        return self.w

    def __array__(self):
        return self.w
    
    def __getitem__(self, i):
        return self.w[i]

    def __setitem__(self, i ,v):
        self.w[i] = v
    
    def __mul__(self, v):
        return self.w * v

    def __iadd__(self, v):
        self.w += v
        return self

class PFDAF:
    def __init__(self, N, M, mu=0.2):
        self.N = N
        self.M = M
        self.N_freq = 1+M
        self.N_fft = 2*M
        self.mu = mu
        self.x_old = np.zeros(self.M,dtype=np.float32)
        self.X = X(N,self.N_freq,dtype=np.complex)
        self.H = H(self.N,self.N_freq,dtype=np.complex)
        self.window = np.hanning(self.M)

    def filt(self, x, d):
        assert(len(x) == self.M)
        x_now = np.concatenate([self.x_old,x])
        X = fft(x_now)
        self.X.append(X)
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
        self.H += self.X().conj()*G
        h1 = ifft(self.H.partition())
        h1[self.M:] = 0
        self.H.constrain(fft(h1))

def pfdaf(x, d, N=4, M=64, mu=0.2):
    ft = PFDAF(N, M, mu)
    num_block = len(x) // M

    e = np.zeros(num_block*M)
    for n in range(num_block):
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        e_n = ft.filt(x_n,d_n)
        ft.update(e_n)
        e[n*M:(n+1)*M] = e_n
    
    return e