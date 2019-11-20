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

#_m (~m) is the next multiple of 1000
_m = math.ceil(m / 1000) * 1000
# M is the matrix A with ~m-m rows padded on
M = np.vstack( (A, np.zeros((_m - m , n))) )

# D is a diagonal matrix of -1 or 1s with probability 1/2
D = np.zeros((_m,_m), int)
for i in range(0, _m) :
    D[i, i] = randrange(-1, 2, 2)

# Applys the discrete cosine transform to D * M
M = sp.fftpack.dct(np.matmul(D, M))

# S is the selector matrix, with probability gamma * n/m we select a row by putting 1 in the diagonal of S
S = np.zeros((_m,_m), int)
for i in range(0, _m) :
    if (uniform(0,1) < p) :
        S[i, i] = 1

# Q R factorization of S applied to M
Q, R = linalg.qr(np.matmul(S, M))

# Solves the related system
y = sp.sparse.linalg.lsqr( np.matmul(A, R.T), b)[0] # need to solve "(A * R.T) * y = b" then solve "R * x = y" to get x
# solves for the final answer
x = sp.sparse.linalg.lsqr(R, y)[0]


# Directly solves the system
x_ = sp.sparse.linalg.lsqr(A, b)[0]

n1 = np.linalg.norm((np.matmul(A, x) - b))
n2 = np.linalg.norm((np.matmul(A, x_) - b))

print("Norm of Blendenpik estimate", n1)
print("Norm from numpy lsqr", n2)
print("Ratio:", n1/n2)