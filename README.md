# pyaec

pyaec is a simple and efficient python implemention of a series of adaptive filters for acoustic echo cancellation.

# About
This project aims to use the simplest lines of python code to implement these adaptive filters, making it easier to learn these algorithms.

# List of Implementioned Adaptive Filters
### Time Domain Adaptive Filters
- Least Mean Squares Filter (LMS)
- Block Least Mean Squares Filter (BLMS)
- Normalized Least Mean Squares Filter (NLMS)
- Block Normalized Least Mean Squares Filter (BNLMS)
- Recursive Least Squares Filter (RLS)
- Affine Projection Algorithm (APA)
- Kalman Filter (KALMAN)

### Frequency Domain Adaptive Filters
- Frequency Domain Adaptive Filter (FDAF)
- Partitioned-Block-Based Frequency Domain Adaptive Filter (PFDAF)
- Frequency Domain Kalman Filter (FDKF)
- Partitioned-Block-Based Frequency Domain Kalman Filter (PFDKF)

### Nonlinear Adaptive Filters
- Second Order Volterra Filter (SVF)
- Trigonometric Functional Link Adaptive Filter (FLAF)
- Adaptive Exponential Functional Link Adaptive Filter (AEFLAF)
- Split Funcional Link Adaptive Filter (SFLAF)
- Collaborative Functional Link Adaptive Filter (CFLAF)

# Requirements
- Python 3.6+
- librosa

# Usage
```
python run.py
```

# Author
ewan xu <ewan_xu@outlook.com>

# Some Reference Books And Papers
- Kong-Aik Lee, Woon-Seng Gan, Sen M. Kuo - Subband Adaptive Filtering Theory and Implementation

- Simon Haykin - Adaptive Filter Theory 

- F.Kuech, E.Mabande, and G.Enzner, "State-space architecture of the partitioned-block-based acoustic echo controller,"in 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2014, pp. 1295-1299: IEEE

- A.Gurin, G.Faucon and R.Le Bouquin-Jeanns, "Nonlinear acoustic echo cancellation based on Volterra filters", IEEE Trans.Speech Audio Process., vol. 11, no. 6, pp. 672-683, Nov. 2003

- V.Patel, V.Gandhi, S.Heda, and N.V.George, “Design of Adaptive Exponential Functional Link Network-Based Nonlinear Filters,” IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 63, no. 9, pp. 1434–1442, 2016.

- D.Comminiello, M.Scarpiniti, L.A.Azpicueta-Ruiz, J. Arenas-Garcia, and A. Uncini, “Functional link adaptive filters for non-linear acoustic echo cancellation,”IEEE Transactions on Audio,Speech, and Language Processing, vol. 21, no. 7, pp. 1502–1512,2013
