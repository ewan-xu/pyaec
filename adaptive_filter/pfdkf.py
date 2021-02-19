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

"""
implemention of partitioned-block-based frequency domain kalman filter
according to the paper:
F. Kuech, E. Mabande, and G. Enzner, "State-space architecture of the partitioned-block-based acoustic echo controller,"
in 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2014, pp. 1295-1299: IEEE
"""

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

    def conj(self):
        return np.concatenate((self.buf[self.p:],self.buf[:self.p])).conj()
    
    def __array__(self):
        return np.concatenate((self.buf[self.p:],self.buf[:self.p]))

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

    def __array__(self):
        return self.w
    
    def __mul__(self, v):
        return self.w * v

    def __iadd__(self, v):
        self.w += v
        return self


class PFDKF:
    def __init__(self,N,M,A=0.999,P_initial=10e+5):
        self.N = N
        self.M = M
        self.N_freq = 1+M
        self.N_fft = 2*M
        self.A2 = A**2

        self.x = np.zeros(shape=(2*self.M),dtype=np.float32)
        self.P = np.full((self.N,self.N_freq),P_initial)
        self.X = X(N,self.N_freq,dtype=np.complex)
        self.window = np.hanning(self.M)
        self.H = H(self.N,self.N_freq,dtype=np.complex)

    def filt(self, x, d):
        assert(len(x) == self.M)
        self.x[self.M:] = x
        X = fft(self.x)
        self.X.append(X)
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
        self.P = self.A2*(1 - 0.5*mu*X2)*self.P + (1-self.A2)*np.sum(np.abs(self.H)**2,axis=0)
        G = mu*E
        self.H += self.X.conj()*G
        h = ifft(self.H.partition())
        h[self.M:] = 0
        self.H.constrain(fft(h))

def pfdkf(x, d, N=4, M=64, A=0.999,P_initial=10e+5):
    ft = PFDKF(N, M, A, P_initial)
    num_block = len(x) // M

    e = np.zeros(num_block*M)
    for n in range(num_block):
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        e_n = ft.filt(x_n,d_n)
        ft.update(e_n)
        e[n*M:(n+1)*M] = e_n

    return e