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

import numpy as np
import librosa
import soundfile as sf
import pyroomacoustics as pra

from time_domain_adaptive_filters.lms import lms
from time_domain_adaptive_filters.nlms import nlms
from time_domain_adaptive_filters.blms import blms
from time_domain_adaptive_filters.bnlms import bnlms
from time_domain_adaptive_filters.rls import rls
from time_domain_adaptive_filters.apa import apa
from time_domain_adaptive_filters.kalman import kalman
from frequency_domain_adaptive_filters.pfdaf import pfdaf
from frequency_domain_adaptive_filters.fdaf import fdaf
from frequency_domain_adaptive_filters.fdkf import fdkf
from frequency_domain_adaptive_filters.pfdkf import pfdkf
from nonlinear_adaptive_filters.volterra import svf
from nonlinear_adaptive_filters.flaf import flaf
from nonlinear_adaptive_filters.aeflaf import aeflaf
from nonlinear_adaptive_filters.sflaf import sflaf
from nonlinear_adaptive_filters.cflaf import cflaf


def main():
  x, sr  = librosa.load('samples/female.wav',sr=8000)
  d, sr  = librosa.load('samples/male.wav',sr=8000)

  rt60_tgt = 0.08
  room_dim = [2, 2, 2]

  e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
  room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
  room.add_source([1.5, 1.5, 1.5])
  room.add_microphone([0.1, 0.5, 0.1])
  room.compute_rir()
  rir = room.rir[0][0]
  rir = rir[np.argmax(rir):]

  y = np.convolve(x,rir)
  scale = np.sqrt(np.mean(x**2)) /  np.sqrt(np.mean(y**2))
  y = y*scale

  L = max(len(y),len(d))
  y = np.pad(y,[0,L-len(y)])
  d = np.pad(d,[L-len(d),0])
  x = np.pad(x,[0,L-len(x)])
  d = d + y

  sf.write('samples/x.wav', x, sr, subtype='PCM_16')
  sf.write('samples/d.wav', d, sr, subtype='PCM_16')

  print("processing time domain adaptive filters.")

  e = lms(x, d, N=256, mu=0.1)
  e = np.clip(e,-1,1)
  sf.write('samples/lms.wav', e, sr, subtype='PCM_16')

  e = blms(x, d, N=256, L=4, mu=0.1)
  e = np.clip(e,-1,1)
  sf.write('samples/blms.wav', e, sr, subtype='PCM_16')
  
  e = nlms(x, d, N=256, mu=0.1)
  e = np.clip(e,-1,1)
  sf.write('samples/nlms.wav', e, sr, subtype='PCM_16')

  e = bnlms(x, d, N=256, L=4, mu=0.1)
  e = np.clip(e,-1,1)
  sf.write('samples/bnlms.wav', e, sr, subtype='PCM_16')

  e = rls(x, d, N=256)
  e = np.clip(e,-1,1)
  sf.write('samples/rls.wav', e, sr, subtype='PCM_16')

  e = apa(x, d, N=256, P=5, mu=0.1)
  e = np.clip(e,-1,1)
  sf.write('samples/apa.wav', e, sr, subtype='PCM_16')

  e = kalman(x, d, N=256)
  e = np.clip(e,-1,1)
  sf.write('samples/kalman.wav', e, sr, subtype='PCM_16')

  print("processing nonlinear adaptive filters.")

  e = svf(x, d, M=256, mu1=0.1, mu2=0.1)
  e = np.clip(e,-1,1)
  sf.write('samples/volterra.wav', e, sr, subtype='PCM_16')
  
  e = flaf(x, d, M=256, P=5, mu=0.2)
  e = np.clip(e,-1,1)
  sf.write('samples/flaf.wav', e, sr, subtype='PCM_16')

  e = aeflaf(x, d, M=256, P=5, mu=0.05, mu_a=0.1)
  e = np.clip(e,-1,1)
  sf.write('samples/aeflaf.wav', e, sr, subtype='PCM_16')
  
  e = sflaf(x, d, M=256, P=5, mu_L=0.2, mu_FL=0.5)
  e = np.clip(e,-1,1)
  sf.write('samples/sflaf.wav', e, sr, subtype='PCM_16')

  e = cflaf(x, d, M=256, P=5, mu_L=0.2, mu_FL=0.5, mu_a=0.5)
  e = np.clip(e,-1,1)
  sf.write('samples/cflaf.wav', e, sr, subtype='PCM_16')

  print("processing frequency domain adaptive filters.")

  e = fdaf(x, d, M=256, mu=0.1)
  e = np.clip(e,-1,1)
  sf.write('samples/fdaf.wav', e, sr, subtype='PCM_16')

  e = fdkf(x, d, M=256)
  e = np.clip(e,-1,1)
  sf.write('samples/fdkf.wav', e, sr, subtype='PCM_16')

  e = pfdaf(x, d, N=8, M=64, mu=0.1, partial_constrain=True)
  e = np.clip(e,-1,1)
  sf.write('samples/pfdaf.wav', e, sr, subtype='PCM_16')

  e = pfdkf(x, d, N=8, M=64, partial_constrain=True)
  e = np.clip(e,-1,1)
  sf.write('samples/pfdkf.wav', e, sr, subtype='PCM_16')


if __name__ == '__main__':
  main()
  