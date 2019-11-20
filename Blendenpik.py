import numpy as np
import scipy as sp
import random as rn
import math
from scipy import random, linalg, diag
from scipy.fftpack import dct
from scipy.sparse.linalg import lsqr
from numpy import matrix
from random import uniform, randrange

m = 1234
n = 20
gamma = 4
p = gamma * n / m

A = random.randn(m, n)
b = random.randn(m)

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
    y = sp.sparse.linalg.lsqr( np.matmul(A, R.T), b)[0] # need to solve "(A * R.T) * y = b" then solve "R * x = y" to get x
    x = sp.sparse.linalg.lsqr(R, y)[0]

x_ = sp.sparse.linalg.lsqr(A, b)[0] # Directly solves the system

n1 = np.linalg.norm((np.matmul(A, x) - b))
n2 = np.linalg.norm((np.matmul(A, x_) - b))

print("Norm of Blendenpik estimate", n1)
print("Norm from numpy lsqr", n2)
print("Ratio:", n1/n2)