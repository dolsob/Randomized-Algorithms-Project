import numpy as np
import scipy as sp
import random as rn
import math
from scipy import random, linalg, dot, diag, all, allclose
from scipy.fftpack import fft, dct
from scipy.sparse.linalg import lsqr
from numpy import matrix
from random import uniform, randrange

m = 1234
n = 20
gamma = 4
p = gamma * n / m

A = random.randn(m, n)
b = random.randn(n)

_m = math.ceil(m / 1000) * 1000
M = np.vstack( (A, np.zeros((_m - m , n))) )

D = np.zeros((n,n), int)
for i in range(0, n) :
    D[i, i] = randrange(-1, 2, 2)

M = sp.fftpack.dct(diag(D) * M)


S = np.zeros((_m,_m), int)
for i in range(0, _m) :
    if (uniform(0,1) < p) :
        S[i, i] = 1

Q, R = linalg.qr(np.matmul(S, M))

k = np.linalg.cond(R)

if (1/k > 5 * np.finfo(float).eps) :
    x = sp.sparse.linalg.lsqr(A, b) # need to solve A*inv(M)*y = b for y, where y = M*x
print(np.shape(x))