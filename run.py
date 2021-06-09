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

import librosa
import numpy as np

from adaptive_filter.lms import lms
from adaptive_filter.nlms import nlms
from adaptive_filter.rls import rls
from adaptive_filter.kalman import kalman
from adaptive_filter.pfdaf import pfdaf
from adaptive_filter.fdaf import fdaf
from adaptive_filter.fdkf import fdkf
from adaptive_filter.pfdkf import pfdkf
import soundfile as sf
def main():
    x, sr  = librosa.load('samples/ref.wav',sr=None)
    h = [0.3,0.2,0.1,0.5,0.3]
    d = np.convolve(x,h)

    e = lms(x, d, N=10)
    e = np.clip(e,-1,1)
    sf.write('samples/lms.wav', e, sr,  subtype='PCM_16')
    
    e = nlms(x, d, N=10)
    e = np.clip(e,-1,1)
    sf.write('samples/nlms.wav', e, sr,  subtype='PCM_16')

    e = rls(x, d, N=10)
    e = np.clip(e,-1,1)
    sf.write('samples/rls.wav', e, sr,  subtype='PCM_16')

    e = kalman(x, d, N=10)
    e = np.clip(e,-1,1)
    sf.write('samples/kalman.wav', e, sr,  subtype='PCM_16')

    e = pfdaf(x, d, N=4, M=64)
    e = np.clip(e,-1,1)
    sf.write('samples/pfdaf.wav', e, sr,  subtype='PCM_16')

    e = fdaf(x, d, M=64)
    e = np.clip(e,-1,1)
    sf.write('samples/fdaf.wav', e, sr,  subtype='PCM_16')

    e = fdkf(x, d, M=64)
    e = np.clip(e,-1,1)
    sf.write('samples/fdkf.wav', e, sr,  subtype='PCM_16')

    e = pfdkf(x, d, N=4, M=64)
    e = np.clip(e,-1,1)
    sf.write('samples/pfdkf.wav', e, sr,  subtype='PCM_16')
if __name__ == '__main__':
    main()